"""
Hash-based perturbation mechanism for evolutionary optimization.

This module implements memory-efficient weight perturbations using
deterministic hash functions (matching the EggRoll/egg.c pattern).

Key features:
- No storage of perturbation matrices (regenerate from seeds)
- GPU-native operations (torch.jit compiled)
- Antithetic sampling support (±ε pairs)
- Selective parameter targeting
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
import time


@torch.jit.script
def hash_rng(seed: int, idx: torch.Tensor) -> torch.Tensor:
    """
    Hash-based RNG matching egg.c implementation.

    Uses a variant of the MurmurHash3 finalizer for fast,
    high-quality pseudo-random numbers on GPU.

    Args:
        seed: Base seed value
        idx: Tensor of indices to hash

    Returns:
        Tensor of hashed int32 values
    """
    x = (seed + idx * 0x9e3779b9).to(torch.int64)
    x = x ^ (x >> 16)
    x = x * 0x85ebca6b
    x = x ^ (x >> 13)
    x = x * 0xc2b2ae35
    x = x ^ (x >> 16)
    return x.to(torch.int32)


@torch.jit.script
def noise_from_hash(seed: int, idx: torch.Tensor) -> torch.Tensor:
    """
    Generate normalized noise from hash values.

    Converts hash output to floating-point noise in [-1, 1] range.

    Args:
        seed: Base seed for hash
        idx: Tensor of indices

    Returns:
        Float tensor of noise values in [-1, 1]
    """
    r = hash_rng(seed, idx)
    # Extract sign from lowest bit
    sign = torch.where(r & 1 == 1,
                       torch.ones_like(r, dtype=torch.float32),
                       -torch.ones_like(r, dtype=torch.float32))
    # Use next 5 bits for magnitude [0, 31]
    magnitude = ((r >> 1) & 31).float() / 31.0
    return sign * magnitude


@torch.jit.script
def gaussian_from_hash(seed: int, idx: torch.Tensor) -> torch.Tensor:
    """
    Generate approximate Gaussian noise from hash using Box-Muller.

    More expensive but produces better exploration dynamics.

    Args:
        seed: Base seed for hash
        idx: Tensor of indices

    Returns:
        Float tensor of approximately Gaussian noise
    """
    # Use two hash values for Box-Muller transform
    r1 = hash_rng(seed, idx)
    r2 = hash_rng(seed + 1000000, idx)

    # Convert to uniform [0, 1]
    u1 = (r1.abs().float() / 2147483647.0).clamp(1e-7, 1.0)
    u2 = (r2.abs().float() / 2147483647.0)

    # Box-Muller transform
    pi = 3.14159265358979323846
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * pi * u2)


@dataclass
class HashPerturbation:
    """
    Stores perturbation metadata (seed, scale, direction).

    The actual perturbation values are computed on-the-fly from
    the seed, avoiding storage of full perturbation matrices.
    """
    seed: int
    scale: float
    direction: int  # +1 or -1 for antithetic pairs

    def __repr__(self):
        sign = '+' if self.direction > 0 else '-'
        return f"Perturbation(seed={self.seed}, scale={self.scale:.4f}, dir={sign})"


def generate_perturbation_seeds(
    population_size: int,
    base_seed: Optional[int] = None
) -> List[int]:
    """
    Generate unique seeds for a population of perturbations.

    Uses time-based seeding with large offsets to ensure
    independence between seeds.

    Args:
        population_size: Number of perturbation pairs
        base_seed: Optional base seed (uses time if None)

    Returns:
        List of seeds, one per population member
    """
    if base_seed is None:
        base_seed = int(time.time() * 1000) % (2**31)

    # Large prime offset to separate seeds
    return [base_seed + i * 999983 for i in range(population_size)]


def apply_perturbation(
    param: torch.Tensor,
    seed: int,
    scale: float,
    direction: int,
    use_gaussian: bool = False
) -> torch.Tensor:
    """
    Apply hash-based perturbation to a parameter tensor.

    This is the core operation: given a parameter tensor and a seed,
    generate a deterministic perturbation and apply it.

    Args:
        param: Parameter tensor to perturb
        seed: Hash seed for reproducibility
        scale: Perturbation magnitude
        direction: +1 or -1 (for antithetic sampling)
        use_gaussian: Use Gaussian noise instead of uniform

    Returns:
        Perturbed parameter tensor (new tensor, doesn't modify input)
    """
    # Flatten parameter for indexing
    flat_size = param.numel()
    idx = torch.arange(flat_size, device=param.device, dtype=torch.int64)

    # Generate noise from hash
    if use_gaussian:
        noise = gaussian_from_hash(seed, idx)
    else:
        noise = noise_from_hash(seed, idx)

    # Reshape noise to match parameter
    noise = noise.view(param.shape).to(param.dtype)

    # Apply scaled perturbation
    return param + (noise * scale * direction)


def create_antithetic_pair(
    param: torch.Tensor,
    seed: int,
    scale: float,
    use_gaussian: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create antithetic perturbation pair (±ε).

    Antithetic sampling reduces variance by evaluating both
    positive and negative directions for each random perturbation.

    Args:
        param: Base parameter tensor
        seed: Hash seed
        scale: Perturbation magnitude
        use_gaussian: Use Gaussian noise

    Returns:
        Tuple of (positive_perturbed, negative_perturbed)
    """
    pos = apply_perturbation(param, seed, scale, +1, use_gaussian)
    neg = apply_perturbation(param, seed, scale, -1, use_gaussian)
    return pos, neg


class SelectiveParameterPerturbation:
    """
    Manages perturbations for selected parameters in a model.

    This class handles:
    1. Identifying which parameters to evolve (by name pattern)
    2. Storing original parameter values
    3. Applying/reverting perturbations efficiently
    """

    def __init__(
        self,
        model: nn.Module,
        param_patterns: List[str],
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize selective perturbation handler.

        Args:
            model: The JiT model
            param_patterns: List of parameter name patterns to evolve
            device: Target device
            dtype: Parameter dtype
        """
        self.model = model
        self.device = device
        self.dtype = dtype

        # Find matching parameters
        self.evolvable_params: Dict[str, nn.Parameter] = {}
        self.original_values: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if self._matches_pattern(name, param_patterns):
                self.evolvable_params[name] = param
                # Store original values (detached clone)
                self.original_values[name] = param.data.clone().detach()

        self.num_params = sum(p.numel() for p in self.evolvable_params.values())
        print(f"[SelectiveParameterPerturbation] Evolving {len(self.evolvable_params)} "
              f"parameters ({self.num_params:,} values)")

    def _matches_pattern(self, name: str, patterns: List[str]) -> bool:
        """Check if parameter name matches any pattern."""
        for pattern in patterns:
            if pattern in name:
                return True
        return False

    def apply_perturbation(
        self,
        perturbation: HashPerturbation,
        use_gaussian: bool = False
    ) -> None:
        """
        Apply perturbation to all evolvable parameters in-place.

        Args:
            perturbation: HashPerturbation with seed, scale, direction
            use_gaussian: Use Gaussian noise
        """
        for i, (name, param) in enumerate(self.evolvable_params.items()):
            # Use parameter-specific seed (offset by index)
            param_seed = perturbation.seed + i * 10007
            perturbed = apply_perturbation(
                self.original_values[name],
                param_seed,
                perturbation.scale,
                perturbation.direction,
                use_gaussian
            )
            param.data.copy_(perturbed)

    def revert_to_original(self) -> None:
        """Revert all parameters to original values."""
        for name, param in self.evolvable_params.items():
            param.data.copy_(self.original_values[name])

    def update_from_votes(
        self,
        votes: Dict[int, int],
        seeds: List[int],
        threshold: int,
        update_scale: float,
        use_gaussian: bool = False
    ) -> int:
        """
        Update original weights based on accumulated votes.

        For each seed, if votes exceed threshold, update weights
        in the winning direction.

        Args:
            votes: Dict mapping seed to vote count (+/- indicates direction)
            seeds: List of seeds that were evaluated
            threshold: Minimum |votes| to trigger update
            update_scale: Magnitude of weight update
            use_gaussian: Use Gaussian noise

        Returns:
            Number of parameters updated
        """
        updates_applied = 0

        for seed in seeds:
            vote = votes.get(seed, 0)
            if abs(vote) >= threshold:
                direction = 1 if vote > 0 else -1
                # Apply small update in winning direction
                for i, (name, _) in enumerate(self.evolvable_params.items()):
                    param_seed = seed + i * 10007
                    flat_size = self.original_values[name].numel()
                    idx = torch.arange(flat_size, device=self.device, dtype=torch.int64)

                    if use_gaussian:
                        noise = gaussian_from_hash(param_seed, idx)
                    else:
                        noise = noise_from_hash(param_seed, idx)

                    noise = noise.view(self.original_values[name].shape)
                    noise = noise.to(self.original_values[name].dtype)

                    # Update original values
                    self.original_values[name] += noise * update_scale * direction
                    updates_applied += 1

        # Copy updated originals to model
        for name, param in self.evolvable_params.items():
            param.data.copy_(self.original_values[name])

        return updates_applied

    def get_param_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about evolvable parameters."""
        stats = {}
        for name, param in self.evolvable_params.items():
            stats[name] = {
                'mean': param.data.float().mean().item(),
                'std': param.data.float().std().item(),
                'min': param.data.float().min().item(),
                'max': param.data.float().max().item(),
                'numel': param.numel(),
            }
        return stats

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return state dict of original (evolved) parameter values."""
        return {name: val.clone() for name, val in self.original_values.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load evolved parameter values."""
        for name, val in state_dict.items():
            if name in self.original_values:
                self.original_values[name].copy_(val)
        # Copy to model
        for name, param in self.evolvable_params.items():
            param.data.copy_(self.original_values[name])
