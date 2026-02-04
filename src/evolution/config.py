"""
Configuration dataclasses for PixelGen evolutionary optimization.

This module defines the hyperparameters for:
- Evolution strategy (population size, noise schedule, voting)
- Fitness function weights (FM, LPIPS, DINO, SSIM)
- Selective layer targeting (which JiT parameters to evolve)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class FitnessConfig:
    """
    Configuration for the composite fitness function.

    PixelGen uses 4 main loss components:
    1. Flow Matching (FM) - velocity prediction accuracy
    2. LPIPS - local perceptual texture quality
    3. P-DINO - global semantic alignment
    4. SSIM - structural similarity (optional)

    These are converted to fitness scores (higher = better).
    """
    # Loss weights (normalized internally)
    w_flow_matching: float = 0.35      # Weight for FM loss → fitness
    w_lpips: float = 0.30              # Weight for LPIPS loss → fitness
    w_dino: float = 0.25               # Weight for DINO cosine similarity
    w_ssim: float = 0.10               # Weight for structural similarity

    # Noise gating threshold (only apply perceptual losses for t >= threshold)
    percept_t_threshold: float = 0.3

    # Loss scaling factors (convert raw losses to [0, 1] range)
    fm_loss_scale: float = 10.0        # exp(-fm_loss * scale)
    lpips_loss_scale: float = 5.0      # 1 / (1 + lpips * scale)

    # DINO encoder settings
    dino_layers: List[int] = field(default_factory=lambda: [11])
    dino_base_patch_size: int = 16

    # LPIPS settings
    lpips_net: str = 'vgg'

    # Timestep sampling parameters (for fitness evaluation)
    t_P_mean: float = -0.8             # Log-normal mean
    t_P_std: float = 0.8               # Log-normal std

    # Optional: Include feature alignment loss (REPA-style)
    use_feature_alignment: bool = True
    feature_align_layer: int = 8
    w_feature_align: float = 0.0       # Set >0 to include


@dataclass
class SelectiveLayerConfig:
    """
    Configuration for which JiT layers to evolve.

    Full JiT-XL has ~600M parameters. Evolving all is expensive.
    We selectively target high-impact layers:

    1. Final layer (patch reconstruction) - direct output influence
    2. In-context tokens - class-conditional generation
    3. Late attention layers - high-level feature composition
    4. AdaLN modulation - time/class conditioning gates
    """
    # Layer groups to evolve
    evolve_final_layer: bool = True           # FinalLayer (norm, linear, adaLN)
    evolve_in_context_tokens: bool = True     # in_context_posemb
    evolve_late_attention: bool = True        # Last N attention layers
    evolve_adaln_modulation: bool = True      # adaLN_modulation in blocks
    evolve_embedders: bool = False            # t_embedder, y_embedder
    evolve_patch_embed: bool = False          # x_embedder (rarely needed)

    # How many late layers to evolve (from end)
    num_late_layers: int = 4                  # Last 4 transformer blocks

    # Within attention, what to evolve
    evolve_qkv: bool = True                   # Q, K, V projections
    evolve_proj: bool = True                  # Output projection
    evolve_mlp: bool = False                  # SwiGLU FFN (expensive)

    def get_evolvable_param_names(self, depth: int) -> List[str]:
        """
        Generate list of parameter name patterns to evolve.

        Args:
            depth: Number of transformer blocks in JiT

        Returns:
            List of parameter name patterns (supports wildcards)
        """
        patterns = []

        if self.evolve_final_layer:
            patterns.extend([
                'final_layer.norm_final.',
                'final_layer.linear.',
                'final_layer.adaLN_modulation.',
            ])

        if self.evolve_in_context_tokens:
            patterns.append('in_context_posemb')

        if self.evolve_late_attention:
            start_layer = depth - self.num_late_layers
            for i in range(start_layer, depth):
                if self.evolve_qkv:
                    patterns.append(f'blocks.{i}.attn.qkv.')
                if self.evolve_proj:
                    patterns.append(f'blocks.{i}.attn.proj.')
                if self.evolve_mlp:
                    patterns.extend([
                        f'blocks.{i}.mlp.w12.',
                        f'blocks.{i}.mlp.w3.',
                    ])
                if self.evolve_adaln_modulation:
                    patterns.append(f'blocks.{i}.adaLN_modulation.')

        if self.evolve_embedders:
            patterns.extend([
                't_embedder.',
                'y_embedder.',
            ])

        if self.evolve_patch_embed:
            patterns.append('x_embedder.')

        return patterns


@dataclass
class EvolutionConfig:
    """
    Configuration for the evolutionary optimization strategy.

    Uses antithetic sampling (±ε pairs) with voting-based updates,
    inspired by Evolution Strategies (ES) and the EggRoll algorithm.
    """
    # Population settings
    population_size: int = 8              # Must be even (antithetic pairs)
    num_generations: int = 1000           # Total evolution generations

    # Noise schedule (perturbation magnitude)
    noise_scale: float = 0.01             # Initial perturbation scale
    noise_decay: float = 0.995            # Decay per generation
    noise_min: float = 1e-5               # Minimum noise scale

    # Voting threshold for weight updates
    vote_threshold: int = 3               # Min votes to update weight
    update_scale: float = 0.001           # Weight update magnitude

    # Timestep sampling for fitness evaluation
    t_sampling: str = 'lognormal'         # 'uniform' or 'lognormal'
    t_P_mean: float = -0.8                # LogNormal mean
    t_P_std: float = 0.8                  # LogNormal std
    t_timeshift: float = 1.0              # Time warping factor

    # Batch settings
    eval_batch_size: int = 4              # Samples per fitness evaluation
    num_eval_batches: int = 1             # Batches to average for fitness

    # Checkpointing
    checkpoint_every: int = 50            # Save checkpoint every N generations
    log_every: int = 10                   # Log metrics every N generations

    # Early stopping
    patience: int = 100                   # Generations without improvement
    min_improvement: float = 1e-4         # Minimum fitness delta

    # Device and precision
    device: str = 'cuda'
    dtype: torch.dtype = torch.bfloat16

    # Selective layer config
    layer_config: SelectiveLayerConfig = field(default_factory=SelectiveLayerConfig)

    # Fitness config
    fitness_config: FitnessConfig = field(default_factory=FitnessConfig)

    # Random seed (for reproducibility)
    seed: int = 42

    def validate(self):
        """Validate configuration parameters."""
        assert self.population_size % 2 == 0, "Population size must be even"
        assert self.noise_scale > 0, "Noise scale must be positive"
        assert 0 < self.noise_decay <= 1, "Noise decay must be in (0, 1]"
        assert self.vote_threshold >= 1, "Vote threshold must be >= 1"
        assert self.eval_batch_size >= 1, "Batch size must be >= 1"

    def __post_init__(self):
        self.validate()


def create_default_config(
    model_size: str = 'XL',
    quick_test: bool = False
) -> EvolutionConfig:
    """
    Create default evolution config for different model sizes.

    Args:
        model_size: 'B', 'L', 'H', or 'XL' (JiT model variants)
        quick_test: If True, use smaller population for testing

    Returns:
        EvolutionConfig with appropriate defaults
    """
    # Model-specific layer depths
    depths = {'B': 12, 'L': 24, 'H': 32, 'XL': 28}
    depth = depths.get(model_size, 28)

    # Adjust late layers based on depth
    num_late = max(2, depth // 7)  # ~14% of layers

    layer_config = SelectiveLayerConfig(
        num_late_layers=num_late,
        evolve_mlp=False,  # Too expensive for full evolution
    )

    if quick_test:
        return EvolutionConfig(
            population_size=4,
            num_generations=10,
            eval_batch_size=2,
            checkpoint_every=5,
            log_every=1,
            layer_config=layer_config,
        )

    return EvolutionConfig(
        population_size=8,
        num_generations=1000,
        layer_config=layer_config,
    )
