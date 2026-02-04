"""
Fitness function for PixelGen evolutionary optimization.

Adapts PixelGen's perceptual losses (LPIPS, P-DINO, Flow Matching)
as fitness metrics for gradient-free optimization.

Key insight: All these losses use frozen feature extractors,
so they don't require gradients - perfect for fitness evaluation!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import lpips

from .config import FitnessConfig


@dataclass
class FitnessResult:
    """Container for fitness evaluation results."""
    total_fitness: float
    fm_fitness: float
    lpips_fitness: float
    dino_fitness: float
    ssim_fitness: float
    raw_fm_loss: float
    raw_lpips_loss: float
    raw_dino_loss: float
    raw_ssim_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'fitness/total': self.total_fitness,
            'fitness/fm': self.fm_fitness,
            'fitness/lpips': self.lpips_fitness,
            'fitness/dino': self.dino_fitness,
            'fitness/ssim': self.ssim_fitness,
            'loss/fm': self.raw_fm_loss,
            'loss/lpips': self.raw_lpips_loss,
            'loss/dino': self.raw_dino_loss,
            'metric/ssim': self.raw_ssim_score,
        }


def gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create Gaussian window for SSIM computation."""
    coords = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, -1) * g.view(1, 1, -1).transpose(-1, -2)


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).

    Args:
        img1, img2: Images in [B, C, H, W] format, range [-1, 1]
        window_size: Size of Gaussian window
        size_average: Return mean SSIM across batch

    Returns:
        SSIM score in [0, 1], higher is better
    """
    # Convert to [0, 1] range
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    window = gaussian_window(window_size, 1.5, img1.device)
    window = window.expand(img1.shape[1], 1, window_size, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


class PixelGenFitness:
    """
    Fitness evaluator for PixelGen evolutionary optimization.

    Converts PixelGen's loss components into fitness scores:
    - FM loss → exp(-loss * scale) ∈ [0, 1]
    - LPIPS → 1 - lpips ∈ [0, 1]
    - DINO → cosine_similarity ∈ [0, 1]
    - SSIM → ssim ∈ [0, 1]

    Higher fitness = better generation quality.
    """

    def __init__(
        self,
        config: FitnessConfig,
        dino_encoder: Optional[nn.Module] = None,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize fitness evaluator.

        Args:
            config: FitnessConfig with weights and settings
            dino_encoder: Pre-initialized DINOv2 encoder (or None to create)
            device: Target device
            dtype: Computation dtype
        """
        self.config = config
        self.device = device
        self.dtype = dtype

        # Initialize LPIPS
        self.lpips_fn = lpips.LPIPS(net=config.lpips_net).to(device).eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

        # Initialize or use provided DINO encoder
        if dino_encoder is not None:
            self.dino_encoder = dino_encoder
        else:
            from src.models.encoder import DINOv2
            self.dino_encoder = DINOv2(base_patch_size=config.dino_base_patch_size)
            self.dino_encoder = self.dino_encoder.to(device).eval()
            for param in self.dino_encoder.parameters():
                param.requires_grad = False

        print(f"[PixelGenFitness] Initialized with weights: "
              f"FM={config.w_flow_matching:.2f}, LPIPS={config.w_lpips:.2f}, "
              f"DINO={config.w_dino:.2f}, SSIM={config.w_ssim:.2f}")

    @torch.no_grad()
    def compute_fitness(
        self,
        pred_img: torch.Tensor,
        target_img: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        raw_images: Optional[torch.Tensor] = None,
    ) -> FitnessResult:
        """
        Compute composite fitness from predicted and target images.

        Args:
            pred_img: Model prediction in [-1, 1] range
            target_img: Ground truth image in [-1, 1] range
            x_t: Noisy input at timestep t
            t: Timestep values [B]
            raw_images: Original images in [0, 1] range (for DINO)

        Returns:
            FitnessResult with all component scores
        """
        batch_size = pred_img.shape[0]

        # Ensure correct dtype
        pred_img = pred_img.to(self.dtype)
        target_img = target_img.to(self.dtype)
        x_t = x_t.to(self.dtype)

        # 1. Flow Matching Loss → Fitness
        t_expanded = t.view(-1, 1, 1, 1)
        t_eps = 0.05

        v_pred = (pred_img - x_t) / (1 - t_expanded).clamp_min(t_eps)
        v_target = (target_img - x_t) / (1 - t_expanded).clamp_min(t_eps)

        fm_loss = ((v_pred - v_target) ** 2).mean()
        fm_fitness = torch.exp(-fm_loss * self.config.fm_loss_scale)

        # 2. LPIPS Loss → Fitness
        # LPIPS expects float32 and [-1, 1] range
        lpips_loss = self.lpips_fn(
            pred_img.float().clamp(-1, 1),
            target_img.float().clamp(-1, 1)
        ).mean()
        lpips_fitness = 1.0 - lpips_loss.clamp(0, 1)

        # 3. DINO Loss → Fitness (cosine similarity)
        # DINO expects [0, 1] range
        if raw_images is None:
            raw_images = (target_img + 1) / 2
        raw_pred = (pred_img + 1) / 2

        pred_feats = self.dino_encoder.get_intermediate_feats(
            raw_pred.float(),
            n=self.config.dino_layers
        )
        target_feats = self.dino_encoder.get_intermediate_feats(
            raw_images.float(),
            n=self.config.dino_layers
        )

        # Average cosine similarity across layers and spatial positions
        dino_sim_total = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            cos_sim = F.cosine_similarity(pf, tf, dim=-1).mean()
            dino_sim_total += cos_sim
        dino_fitness = dino_sim_total / len(pred_feats)
        dino_loss = 1.0 - dino_fitness

        # 4. SSIM
        ssim_score = ssim(pred_img.float(), target_img.float())
        ssim_fitness = ssim_score.clamp(0, 1)

        # Apply noise gating: only count perceptual for t >= threshold
        # For low t (high noise), perceptual metrics are unreliable
        t_mean = t.mean().item()
        if t_mean < self.config.percept_t_threshold:
            # High noise regime: only use FM loss
            percept_weight = 0.0
        else:
            percept_weight = 1.0

        # Compute total fitness
        total_fitness = (
            self.config.w_flow_matching * fm_fitness +
            percept_weight * (
                self.config.w_lpips * lpips_fitness +
                self.config.w_dino * dino_fitness +
                self.config.w_ssim * ssim_fitness
            )
        )

        # Normalize if perceptual was zeroed
        if percept_weight == 0.0:
            total_fitness = fm_fitness  # Just FM fitness

        return FitnessResult(
            total_fitness=total_fitness.item(),
            fm_fitness=fm_fitness.item(),
            lpips_fitness=lpips_fitness.item(),
            dino_fitness=dino_fitness.item(),
            ssim_fitness=ssim_fitness.item(),
            raw_fm_loss=fm_loss.item(),
            raw_lpips_loss=lpips_loss.item(),
            raw_dino_loss=dino_loss.item(),
            raw_ssim_score=ssim_score.item(),
        )

    @torch.no_grad()
    def evaluate_batch(
        self,
        model: nn.Module,
        batch: Tuple[torch.Tensor, Any, Dict],
        scheduler,
        t: Optional[torch.Tensor] = None,
    ) -> FitnessResult:
        """
        Evaluate fitness on a batch of data.

        This is the main entry point for fitness evaluation during evolution.

        Args:
            model: JiT model to evaluate
            batch: Tuple of (x, y, metadata) from dataloader
            scheduler: Flow matching scheduler
            t: Optional fixed timestep (random if None)

        Returns:
            FitnessResult averaged across batch
        """
        x, y, metadata = batch
        raw_images = metadata.get("raw_image", (x + 1) / 2)

        batch_size = x.shape[0]
        device = x.device

        # Convert y to tensor if it's a tuple/list (class labels)
        if isinstance(y, (list, tuple)):
            y = torch.tensor(y, device=device, dtype=torch.long)
        elif not isinstance(y, torch.Tensor):
            y = torch.tensor([y] * batch_size, device=device, dtype=torch.long)

        # Sample timestep if not provided
        if t is None:
            # Log-normal sampling (matches PixelGen training)
            P_mean = getattr(self.config, 't_P_mean', -0.8)
            P_std = getattr(self.config, 't_P_std', 0.8)
            t = (torch.randn(batch_size, device=device) * P_std + P_mean).sigmoid()

        # Ensure t is float32 for model (timestep embeddings require float32)
        t = t.float()

        # Get alpha, sigma from scheduler
        alpha = scheduler.alpha(t)
        sigma = scheduler.sigma(t)

        # Create noisy input (keep x's dtype)
        noise = torch.randn_like(x)
        x_t = alpha * x + noise * sigma

        # Forward pass (no gradients needed!)
        # Model will handle internal dtype conversions
        pred_img = model(x_t.to(x.dtype), t, y)

        # Compute fitness
        return self.compute_fitness(
            pred_img=pred_img,
            target_img=x,
            x_t=x_t,
            t=t,
            raw_images=raw_images,
        )

    def compare_pair(
        self,
        fitness_pos: FitnessResult,
        fitness_neg: FitnessResult,
    ) -> int:
        """
        Compare antithetic pair and return vote.

        Args:
            fitness_pos: Fitness of positive perturbation
            fitness_neg: Fitness of negative perturbation

        Returns:
            +1 if positive wins, -1 if negative wins, 0 if tie
        """
        diff = fitness_pos.total_fitness - fitness_neg.total_fitness

        # Use small epsilon for tie-breaking
        eps = 1e-6
        if diff > eps:
            return +1
        elif diff < -eps:
            return -1
        else:
            return 0


class MultiTimestepFitness(PixelGenFitness):
    """
    Extended fitness evaluator that samples multiple timesteps.

    For more robust evaluation, evaluate at different noise levels
    and average the fitness scores.
    """

    def __init__(
        self,
        config: FitnessConfig,
        num_timesteps: int = 3,
        timestep_distribution: str = 'stratified',
        **kwargs
    ):
        """
        Args:
            config: FitnessConfig
            num_timesteps: Number of timesteps to evaluate at
            timestep_distribution: 'stratified', 'uniform', or 'lognormal'
            **kwargs: Passed to parent class
        """
        super().__init__(config, **kwargs)
        self.num_timesteps = num_timesteps
        self.timestep_distribution = timestep_distribution

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Sample multiple timesteps for evaluation."""
        timesteps = []

        if self.timestep_distribution == 'stratified':
            # Stratified sampling across t range
            for i in range(self.num_timesteps):
                t_low = i / self.num_timesteps
                t_high = (i + 1) / self.num_timesteps
                t = torch.rand(batch_size, device=device) * (t_high - t_low) + t_low
                timesteps.append(t)

        elif self.timestep_distribution == 'uniform':
            for _ in range(self.num_timesteps):
                t = torch.rand(batch_size, device=device)
                timesteps.append(t)

        elif self.timestep_distribution == 'lognormal':
            P_mean = getattr(self.config, 't_P_mean', -0.8)
            P_std = getattr(self.config, 't_P_std', 0.8)
            for _ in range(self.num_timesteps):
                t = (torch.randn(batch_size, device=device) * P_std + P_mean).sigmoid()
                timesteps.append(t)

        return timesteps

    @torch.no_grad()
    def evaluate_batch(
        self,
        model: nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor, Dict],
        scheduler,
        t: Optional[torch.Tensor] = None,
    ) -> FitnessResult:
        """
        Evaluate fitness across multiple timesteps.

        Returns averaged FitnessResult.
        """
        x, y, metadata = batch
        raw_images = metadata.get("raw_image", (x + 1) / 2)

        batch_size = x.shape[0]
        device = x.device

        # Sample multiple timesteps
        timesteps = self._sample_timesteps(batch_size, device)

        # Accumulate results
        results = []
        for t in timesteps:
            alpha = scheduler.alpha(t)
            sigma = scheduler.sigma(t)

            noise = torch.randn_like(x)
            x_t = alpha * x + noise * sigma

            pred_img = model(x_t, t, y)

            result = self.compute_fitness(
                pred_img=pred_img,
                target_img=x,
                x_t=x_t,
                t=t,
                raw_images=raw_images,
            )
            results.append(result)

        # Average results
        return FitnessResult(
            total_fitness=sum(r.total_fitness for r in results) / len(results),
            fm_fitness=sum(r.fm_fitness for r in results) / len(results),
            lpips_fitness=sum(r.lpips_fitness for r in results) / len(results),
            dino_fitness=sum(r.dino_fitness for r in results) / len(results),
            ssim_fitness=sum(r.ssim_fitness for r in results) / len(results),
            raw_fm_loss=sum(r.raw_fm_loss for r in results) / len(results),
            raw_lpips_loss=sum(r.raw_lpips_loss for r in results) / len(results),
            raw_dino_loss=sum(r.raw_dino_loss for r in results) / len(results),
            raw_ssim_score=sum(r.raw_ssim_score for r in results) / len(results),
        )
