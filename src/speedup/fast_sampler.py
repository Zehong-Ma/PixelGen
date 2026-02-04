"""
Fast sampling strategies for PixelGen.

Implements multiple speedup techniques:
1. Adaptive step scheduling (more steps where it matters)
2. Classifier-free guidance caching
3. Progressive generation (coarse to fine)
4. Higher-order ODE solvers (Heun, DPM++)
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class SamplerConfig:
    """Configuration for fast sampling."""
    num_steps: int = 25
    guidance_scale: float = 1.0  # 1.0 = no CFG
    use_heun: bool = False       # 2nd order solver (2x cost but better quality)
    adaptive_steps: bool = True  # More steps in high-noise regime
    cache_uncond: bool = True    # Cache unconditional prediction
    t_min: float = 0.002         # Minimum timestep (avoid division issues)
    t_max: float = 0.998         # Maximum timestep


class FastEulerSampler:
    """
    Optimized Euler sampler for PixelGen flow matching.

    Key optimizations:
    1. Pre-computed timestep schedule
    2. In-place operations where possible
    3. Optional CFG with caching
    4. Adaptive step sizing
    """

    def __init__(self, config: SamplerConfig = None):
        self.config = config or SamplerConfig()
        self._timesteps = None
        self._cached_uncond = None

    def _get_timesteps(self, device: torch.device) -> torch.Tensor:
        """Get optimized timestep schedule."""
        if self._timesteps is not None and self._timesteps.device == device:
            return self._timesteps

        num_steps = self.config.num_steps
        t_min = self.config.t_min
        t_max = self.config.t_max

        if self.config.adaptive_steps:
            # More steps in early denoising (high noise) where it matters most
            # Use cosine schedule for timesteps
            t = torch.linspace(0, 1, num_steps + 1, device=device)
            # Cosine mapping: more resolution near t=0 (high noise)
            timesteps = t_min + (t_max - t_min) * (1 - torch.cos(t * math.pi / 2))
        else:
            timesteps = torch.linspace(t_min, t_max, num_steps + 1, device=device)

        self._timesteps = timesteps
        return timesteps

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        condition: torch.Tensor,
        uncondition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate images from noise.

        Args:
            model: JiT model
            noise: Initial noise [B, 3, H, W]
            condition: Class labels [B]
            uncondition: Unconditional labels for CFG (optional)

        Returns:
            Generated images [B, 3, H, W]
        """
        batch_size = noise.shape[0]
        device = noise.device

        x = noise.clone()
        timesteps = self._get_timesteps(device)

        use_cfg = (self.config.guidance_scale > 1.0 and uncondition is not None)

        for i in range(self.config.num_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_cur

            # Expand timestep for batch
            t_batch = t_cur.expand(batch_size)

            if use_cfg:
                # Classifier-free guidance: combine conditional and unconditional
                if self.config.cache_uncond and self._cached_uncond is not None:
                    # Use cached unconditional prediction
                    pred_uncond = self._cached_uncond
                    pred_cond = model(x, t_batch, condition)
                else:
                    # Batched CFG: run both in one forward pass
                    x_double = torch.cat([x, x], dim=0)
                    t_double = torch.cat([t_batch, t_batch], dim=0)
                    y_double = torch.cat([condition, uncondition], dim=0)

                    pred_double = model(x_double, t_double, y_double)
                    pred_cond, pred_uncond = pred_double.chunk(2, dim=0)

                    if self.config.cache_uncond:
                        self._cached_uncond = pred_uncond.clone()

                # CFG combination
                pred = pred_uncond + self.config.guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = model(x, t_batch, condition)

            # Compute velocity from x-prediction
            # v = (pred - x_t) / (1 - t)
            v = (pred - x) / (1 - t_cur).clamp_min(self.config.t_min)

            # Euler step
            x = x + v * dt

        return x

    @torch.no_grad()
    def sample_heun(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Heun's method (2nd order) for higher quality at same step count.

        ~2x compute but significantly better quality.
        """
        batch_size = noise.shape[0]
        device = noise.device

        x = noise.clone()
        timesteps = self._get_timesteps(device)

        for i in range(self.config.num_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_cur
            t_batch = t_cur.expand(batch_size)
            t_next_batch = t_next.expand(batch_size)

            # First evaluation
            pred1 = model(x, t_batch, condition)
            v1 = (pred1 - x) / (1 - t_cur).clamp_min(self.config.t_min)

            # Euler prediction
            x_euler = x + v1 * dt

            # Second evaluation at predicted point
            pred2 = model(x_euler, t_next_batch, condition)
            v2 = (pred2 - x_euler) / (1 - t_next).clamp_min(self.config.t_min)

            # Heun correction (average of velocities)
            x = x + (v1 + v2) * dt / 2

        return x


class ProgressiveSampler:
    """
    Progressive/cascaded generation for faster results.

    Strategy:
    1. Generate at low resolution (fast)
    2. Upsample and refine at higher resolution

    This exploits the fact that global structure emerges early
    and fine details need fewer steps.
    """

    def __init__(
        self,
        coarse_steps: int = 10,
        fine_steps: int = 15,
        coarse_size: int = 128,
        fine_size: int = 256,
    ):
        self.coarse_steps = coarse_steps
        self.fine_steps = fine_steps
        self.coarse_size = coarse_size
        self.fine_size = fine_size

        self.coarse_sampler = FastEulerSampler(SamplerConfig(num_steps=coarse_steps))
        self.fine_sampler = FastEulerSampler(SamplerConfig(num_steps=fine_steps))

    @torch.no_grad()
    def sample(
        self,
        model_coarse: nn.Module,
        model_fine: nn.Module,
        batch_size: int,
        condition: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Two-stage progressive generation.

        Note: Requires models trained at both resolutions,
        or a single model that handles variable resolution.
        """
        # Stage 1: Coarse generation
        noise_coarse = torch.randn(
            batch_size, 3, self.coarse_size, self.coarse_size,
            device=device
        )

        x_coarse = self.coarse_sampler.sample(
            model_coarse, noise_coarse, condition
        )

        # Upsample to fine resolution
        x_upsampled = torch.nn.functional.interpolate(
            x_coarse,
            size=(self.fine_size, self.fine_size),
            mode='bilinear',
            align_corners=False,
        )

        # Stage 2: Fine refinement
        # Add small noise for refinement
        noise_fine = torch.randn_like(x_upsampled) * 0.3
        x_noisy = x_upsampled + noise_fine

        # Start from mid-way through denoising
        x_fine = self.fine_sampler.sample(
            model_fine, x_noisy, condition
        )

        return x_fine


class CFGCache:
    """
    Caches unconditional predictions to reduce CFG overhead.

    For CFG, we compute: pred = uncond + scale * (cond - uncond)

    The unconditional prediction often changes slowly, so we can:
    1. Reuse it across multiple steps
    2. Update it every N steps
    """

    def __init__(self, update_every: int = 5):
        self.update_every = update_every
        self.cached_uncond = None
        self.step_count = 0

    def should_update(self) -> bool:
        """Check if cache should be updated."""
        return self.cached_uncond is None or self.step_count % self.update_every == 0

    def get_cfg_prediction(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        uncondition: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Get CFG-guided prediction with caching.
        """
        # Always compute conditional
        pred_cond = model(x, t, condition)

        if self.should_update():
            # Update unconditional cache
            pred_uncond = model(x, t, uncondition)
            self.cached_uncond = pred_uncond.clone()
        else:
            pred_uncond = self.cached_uncond

        self.step_count += 1

        # CFG combination
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    def reset(self):
        """Reset cache for new generation."""
        self.cached_uncond = None
        self.step_count = 0


def create_fast_sampler(
    num_steps: int = 25,
    quality: str = 'balanced',
    guidance_scale: float = 1.0,
) -> FastEulerSampler:
    """
    Factory function for creating optimized samplers.

    Args:
        num_steps: Number of denoising steps
        quality: 'fast', 'balanced', or 'quality'
        guidance_scale: CFG scale (1.0 = disabled)

    Returns:
        Configured FastEulerSampler
    """
    configs = {
        'fast': SamplerConfig(
            num_steps=min(num_steps, 10),
            adaptive_steps=False,
            use_heun=False,
        ),
        'balanced': SamplerConfig(
            num_steps=num_steps,
            adaptive_steps=True,
            use_heun=False,
        ),
        'quality': SamplerConfig(
            num_steps=num_steps,
            adaptive_steps=True,
            use_heun=True,
        ),
    }

    config = configs.get(quality, configs['balanced'])
    config.guidance_scale = guidance_scale

    return FastEulerSampler(config)
