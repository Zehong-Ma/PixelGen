"""
Token Merging (ToMe) for PixelGen JiT.

Reduces sequence length by merging similar tokens during inference.
Based on "Token Merging: Your ViT But Faster" (Bolya et al.)

Key insight: Many tokens in transformers are redundant.
By merging similar tokens, we can reduce computation while
maintaining quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
import math


@dataclass
class TokenMergingConfig:
    """Configuration for token merging."""
    merge_ratio: float = 0.5      # Fraction of tokens to merge (0.5 = half)
    merge_mode: str = 'mean'      # 'mean', 'weighted', or 'max'
    start_layer: int = 4          # Start merging after this layer
    end_layer: int = -1           # Stop merging at this layer (-1 = end)
    use_random_partition: bool = True  # Random bipartite matching
    merge_attn: bool = True       # Merge in attention
    merge_mlp: bool = False       # Also merge in MLP (more aggressive)
    sx: int = 2                   # Stride for source tokens (spatial)
    sy: int = 2                   # Stride for source tokens (spatial)


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Bipartite soft matching for token merging.

    Partitions tokens into source (to be merged) and destination (to keep).
    Finds optimal assignment that minimizes distance.

    Args:
        metric: Token features for similarity [B, N, C]
        r: Number of tokens to remove
        class_token: Whether first token is class token (don't merge)
        distill_token: Whether second token is distillation token

    Returns:
        merge: Function to merge tokens
        unmerge: Function to unmerge tokens
    """
    protected = int(class_token) + int(distill_token)

    # Cannot merge if not enough tokens
    t = metric.shape[1]
    if r >= t - protected:
        # Return identity functions
        return lambda x, mode='mean': x, lambda x: x

    with torch.no_grad():
        # Normalize for cosine similarity
        metric = metric / metric.norm(dim=-1, keepdim=True)

        # Partition into source (odd indices) and destination (even indices)
        # Skip protected tokens
        a, b = metric[:, protected::2], metric[:, protected + 1::2]

        # Compute similarity scores
        scores = torch.bmm(a, b.transpose(-1, -2))

        # For each source token, find most similar destination
        node_max, node_idx = scores.max(dim=-1)

        # Sort by similarity and take top-r for merging
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., :r]

        # Gather indices
        src_idx = edge_idx
        dst_idx = node_idx.gather(dim=-1, index=edge_idx)

    def merge(x: torch.Tensor, mode: str = 'mean') -> torch.Tensor:
        """Merge tokens based on computed matching."""
        B, N, C = x.shape

        # Separate protected tokens
        protected_tokens = x[:, :protected]
        x = x[:, protected:]

        # Split into source and destination
        src = x[:, ::2]
        dst = x[:, 1::2]

        n_src, n_dst = src.shape[1], dst.shape[1]

        # Merge source into destination
        # Scatter-add source tokens to their matched destinations
        dst = dst.scatter_reduce(
            dim=1,
            index=dst_idx.unsqueeze(-1).expand(-1, -1, C),
            src=src.gather(dim=1, index=src_idx.unsqueeze(-1).expand(-1, -1, C)),
            reduce='mean' if mode == 'mean' else 'sum',
            include_self=True,
        )

        # Remove merged source tokens
        # Create mask for unmerged source tokens
        unm_idx = torch.arange(n_src, device=x.device).unsqueeze(0).expand(B, -1)
        unm_mask = torch.ones(B, n_src, dtype=torch.bool, device=x.device)
        unm_mask.scatter_(1, src_idx, False)

        # Gather unmerged source tokens
        unm_src = src[unm_mask].view(B, n_src - r, C)

        # Concatenate: protected + destination + unmerged source
        out = torch.cat([protected_tokens, dst, unm_src], dim=1)

        return out

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """Unmerge tokens to original length (approximate)."""
        B, N, C = x.shape

        # Separate protected tokens
        protected_tokens = x[:, :protected]
        x = x[:, protected:]

        # Split into merged destination and unmerged source
        n_dst = (t - protected) // 2
        n_unm = (t - protected) - n_dst - r

        dst = x[:, :n_dst]
        unm_src = x[:, n_dst:n_dst + n_unm]

        # Reconstruct source tokens
        src = torch.zeros(B, (t - protected) // 2, C, device=x.device, dtype=x.dtype)

        # Place unmerged tokens
        unm_mask = torch.ones(B, src.shape[1], dtype=torch.bool, device=x.device)
        unm_mask.scatter_(1, src_idx, False)
        src[unm_mask] = unm_src.reshape(-1, C)

        # Copy merged tokens from destination
        src.scatter_(
            dim=1,
            index=src_idx.unsqueeze(-1).expand(-1, -1, C),
            src=dst.gather(dim=1, index=dst_idx.unsqueeze(-1).expand(-1, -1, C)),
        )

        # Interleave source and destination
        out = torch.zeros(B, t - protected, C, device=x.device, dtype=x.dtype)
        out[:, ::2] = src
        out[:, 1::2] = dst

        # Add protected tokens
        out = torch.cat([protected_tokens, out], dim=1)

        return out

    return merge, unmerge


class TokenMergingAttention(nn.Module):
    """
    Attention block with token merging.

    Wraps standard attention to merge tokens before computation
    and unmerge after (if needed for skip connections).
    """

    def __init__(
        self,
        original_attn: nn.Module,
        merge_ratio: float = 0.5,
    ):
        super().__init__()
        self.attn = original_attn
        self.merge_ratio = merge_ratio

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward with token merging."""
        B, N, C = x.shape

        # Number of tokens to remove
        r = int(N * self.merge_ratio)

        if r > 0:
            # Compute merge function based on input features
            merge, unmerge = bipartite_soft_matching(x, r)

            # Merge tokens
            x = merge(x)

            # Run attention on reduced sequence
            x = self.attn(x, *args, **kwargs)

            # Unmerge for skip connection
            x = unmerge(x)
        else:
            x = self.attn(x, *args, **kwargs)

        return x


def apply_tome_to_jit(
    model: nn.Module,
    config: TokenMergingConfig = None,
) -> nn.Module:
    """
    Apply Token Merging to JiT model.

    This patches the model's attention layers to use token merging
    during inference.

    Args:
        model: JiT model to modify
        config: TokenMergingConfig

    Returns:
        Modified model with token merging
    """
    config = config or TokenMergingConfig()

    # Find attention layers
    blocks = model.blocks if hasattr(model, 'blocks') else []
    num_blocks = len(blocks)

    end_layer = config.end_layer if config.end_layer >= 0 else num_blocks

    # Patch each block's attention
    for i, block in enumerate(blocks):
        if config.start_layer <= i < end_layer:
            if hasattr(block, 'attn'):
                # Wrap attention with token merging
                original_attn = block.attn
                block.attn = TokenMergingWrapper(
                    original_attn,
                    merge_ratio=config.merge_ratio,
                )
                print(f"  Applied ToMe to block {i} (merge_ratio={config.merge_ratio})")

    return model


class TokenMergingWrapper(nn.Module):
    """
    Wrapper that adds token merging to any attention module.

    More general than TokenMergingAttention - handles various
    attention signatures.
    """

    def __init__(
        self,
        attention: nn.Module,
        merge_ratio: float = 0.5,
    ):
        super().__init__()
        self.attention = attention
        self.merge_ratio = merge_ratio
        self._merge_fn = None
        self._unmerge_fn = None

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward with optional token merging."""
        B, N, C = x.shape

        # Skip if too few tokens
        r = int(N * self.merge_ratio)
        if r < 2 or N < 8:
            return self.attention(x, *args, **kwargs)

        # Compute merge/unmerge functions
        merge, unmerge = bipartite_soft_matching(x, r)

        # Store for potential use in cross-attention
        self._merge_fn = merge
        self._unmerge_fn = unmerge

        # Merge, attend, unmerge
        x_merged = merge(x)
        out_merged = self.attention(x_merged, *args, **kwargs)
        out = unmerge(out_merged)

        return out


class AdaptiveTokenMerging(nn.Module):
    """
    Adaptive token merging that adjusts merge ratio based on content.

    Key insight: Not all images need the same amount of merging.
    Complex images should merge less, simple images can merge more.
    """

    def __init__(
        self,
        base_merge_ratio: float = 0.5,
        min_merge_ratio: float = 0.1,
        max_merge_ratio: float = 0.7,
    ):
        super().__init__()
        self.base_merge_ratio = base_merge_ratio
        self.min_merge_ratio = min_merge_ratio
        self.max_merge_ratio = max_merge_ratio

    def compute_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate image complexity from token features.

        High variance = complex image = merge less
        Low variance = simple image = merge more
        """
        # Compute token variance as complexity proxy
        token_var = x.var(dim=1).mean(dim=-1)  # [B]

        # Normalize to [0, 1]
        complexity = torch.sigmoid(token_var - token_var.mean())

        return complexity

    def get_merge_ratio(self, x: torch.Tensor) -> float:
        """Get adaptive merge ratio for input."""
        complexity = self.compute_complexity(x).mean().item()

        # High complexity -> low merge ratio
        # Low complexity -> high merge ratio
        merge_ratio = self.max_merge_ratio - complexity * (
            self.max_merge_ratio - self.min_merge_ratio
        )

        return max(self.min_merge_ratio, min(self.max_merge_ratio, merge_ratio))


def benchmark_tome_speedup(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 256),
    merge_ratios: list = [0.0, 0.25, 0.5, 0.75],
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = 'cuda',
) -> dict:
    """
    Benchmark token merging speedup.

    Args:
        model: JiT model
        input_size: Input tensor size (B, C, H, W)
        merge_ratios: List of merge ratios to test
        num_warmup: Warmup iterations
        num_runs: Benchmark iterations

    Returns:
        Dict with timing results
    """
    import time
    import copy

    results = {}

    for ratio in merge_ratios:
        # Create model copy with this merge ratio
        test_model = copy.deepcopy(model)

        if ratio > 0:
            config = TokenMergingConfig(merge_ratio=ratio)
            test_model = apply_tome_to_jit(test_model, config)

        test_model = test_model.to(device).eval()

        # Create test inputs
        x = torch.randn(input_size, device=device)
        t = torch.rand(input_size[0], device=device)
        y = torch.randint(0, 1000, (input_size[0],), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = test_model(x, t, y)
                torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = test_model(x, t, y)
                torch.cuda.synchronize()
                times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        results[f'ratio_{ratio}'] = {
            'merge_ratio': ratio,
            'avg_time_ms': avg_time * 1000,
            'speedup': results.get('ratio_0.0', {}).get('avg_time_ms', avg_time * 1000) / (avg_time * 1000) if ratio > 0 else 1.0,
        }

        del test_model
        torch.cuda.empty_cache()

    return results
