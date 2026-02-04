"""
Main evolution engine for PixelGen gradient-free optimization.

Implements the core evolutionary loop:
1. Generate antithetic perturbation pairs
2. Evaluate fitness for each candidate
3. Vote on winning direction
4. Update weights based on accumulated votes
5. Decay noise scale
6. Repeat

Based on Evolution Strategies (ES) with EggRoll-style voting.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Iterator, Any
from dataclasses import dataclass, field
import time
import json
import os
from pathlib import Path
from collections import defaultdict

from .config import EvolutionConfig, FitnessConfig
from .perturbation import (
    HashPerturbation,
    SelectiveParameterPerturbation,
    generate_perturbation_seeds,
)
from .fitness import PixelGenFitness, FitnessResult, MultiTimestepFitness


@dataclass
class EvolutionState:
    """Tracks evolution progress and best solutions."""
    generation: int = 0
    best_fitness: float = float('-inf')
    best_generation: int = 0
    current_noise_scale: float = 0.01
    fitness_history: List[float] = field(default_factory=list)
    generations_without_improvement: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_generation': self.best_generation,
            'current_noise_scale': self.current_noise_scale,
            'fitness_history': self.fitness_history[-100:],  # Last 100
            'generations_without_improvement': self.generations_without_improvement,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EvolutionState':
        return cls(
            generation=d['generation'],
            best_fitness=d['best_fitness'],
            best_generation=d['best_generation'],
            current_noise_scale=d['current_noise_scale'],
            fitness_history=d['fitness_history'],
            generations_without_improvement=d['generations_without_improvement'],
        )


@dataclass
class GenerationResult:
    """Results from a single generation."""
    generation: int
    mean_fitness: float
    best_fitness: float
    worst_fitness: float
    num_updates: int
    positive_wins: int
    negative_wins: int
    ties: int
    noise_scale: float
    elapsed_time: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'gen': self.generation,
            'mean_fitness': self.mean_fitness,
            'best_fitness': self.best_fitness,
            'worst_fitness': self.worst_fitness,
            'num_updates': self.num_updates,
            'positive_wins': self.positive_wins,
            'negative_wins': self.negative_wins,
            'ties': self.ties,
            'noise_scale': self.noise_scale,
            'elapsed_time': self.elapsed_time,
        }


class PixelGenEvolution:
    """
    Main evolutionary optimization engine for PixelGen.

    This class orchestrates the entire evolution process:
    - Managing population of perturbations
    - Evaluating fitness
    - Voting and weight updates
    - Checkpointing and logging
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler,
        dataloader: Iterator,
        config: EvolutionConfig,
        dino_encoder: Optional[nn.Module] = None,
        output_dir: str = './evolution_output',
        wandb_run=None,
    ):
        """
        Initialize evolution engine.

        Args:
            model: JiT model to evolve
            scheduler: Flow matching scheduler
            dataloader: Data iterator for fitness evaluation
            config: EvolutionConfig with all hyperparameters
            dino_encoder: Optional pre-loaded DINOv2 encoder
            output_dir: Directory for checkpoints and logs
            wandb_run: Optional W&B run for logging
        """
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run

        # Store model and scheduler
        self.model = model.to(self.device)
        self.scheduler = scheduler
        self.dataloader = dataloader
        self._dataloader_iter = None  # Will be initialized on first _get_batch

        # Initialize selective parameter perturbation
        param_patterns = config.layer_config.get_evolvable_param_names(
            depth=len(model.blocks) if hasattr(model, 'blocks') else 24
        )
        self.perturbation_handler = SelectiveParameterPerturbation(
            model=model,
            param_patterns=param_patterns,
            device=self.device,
            dtype=self.dtype,
        )

        # Initialize fitness evaluator
        self.fitness_evaluator = MultiTimestepFitness(
            config=config.fitness_config,
            dino_encoder=dino_encoder,
            device=self.device,
            dtype=self.dtype,
            num_timesteps=3,
            timestep_distribution='stratified',
        )

        # Initialize state
        self.state = EvolutionState(
            current_noise_scale=config.noise_scale,
        )

        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        print(f"[PixelGenEvolution] Initialized:")
        print(f"  - Population size: {config.population_size}")
        print(f"  - Generations: {config.num_generations}")
        print(f"  - Initial noise scale: {config.noise_scale}")
        print(f"  - Vote threshold: {config.vote_threshold}")
        print(f"  - Evolvable parameters: {self.perturbation_handler.num_params:,}")
        print(f"  - Output directory: {self.output_dir}")

    def _get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get next batch from dataloader."""
        # Initialize iterator if needed
        if self._dataloader_iter is None:
            self._dataloader_iter = iter(self.dataloader)

        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            # Reset dataloader iterator
            self._dataloader_iter = iter(self.dataloader)
            batch = next(self._dataloader_iter)

        x, y, metadata = batch
        x = x.to(self.device, dtype=self.dtype)
        y = y.to(self.device)

        # Move metadata tensors to device
        if "raw_image" in metadata:
            metadata["raw_image"] = metadata["raw_image"].to(self.device, dtype=self.dtype)

        return x, y, metadata

    def _evaluate_candidate(
        self,
        perturbation: HashPerturbation,
    ) -> FitnessResult:
        """
        Evaluate fitness of a single candidate.

        Args:
            perturbation: HashPerturbation to apply

        Returns:
            FitnessResult
        """
        # Apply perturbation
        self.perturbation_handler.apply_perturbation(perturbation)

        # Evaluate on multiple batches for stability
        results = []
        for _ in range(self.config.num_eval_batches):
            batch = self._get_batch()
            result = self.fitness_evaluator.evaluate_batch(
                model=self.model,
                batch=batch,
                scheduler=self.scheduler,
            )
            results.append(result)

        # Revert to original weights
        self.perturbation_handler.revert_to_original()

        # Average results
        if len(results) == 1:
            return results[0]

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

    def run_generation(self) -> GenerationResult:
        """
        Run a single generation of evolution.

        Returns:
            GenerationResult with statistics
        """
        start_time = time.time()
        gen = self.state.generation

        # Generate seeds for antithetic pairs
        num_pairs = self.config.population_size // 2
        seeds = generate_perturbation_seeds(num_pairs)

        # Track votes and fitness scores
        votes: Dict[int, int] = defaultdict(int)
        all_fitness: List[float] = []
        positive_wins = 0
        negative_wins = 0
        ties = 0

        # Evaluate each antithetic pair
        eval_count = 0
        for seed in seeds:
            # Positive perturbation
            pert_pos = HashPerturbation(
                seed=seed,
                scale=self.state.current_noise_scale,
                direction=+1,
            )
            fitness_pos = self._evaluate_candidate(pert_pos)
            eval_count += 1

            # Memory optimization: clear cache periodically
            if self.config.empty_cache_freq > 0 and eval_count % self.config.empty_cache_freq == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Negative perturbation
            pert_neg = HashPerturbation(
                seed=seed,
                scale=self.state.current_noise_scale,
                direction=-1,
            )
            fitness_neg = self._evaluate_candidate(pert_neg)
            eval_count += 1

            # Memory optimization: clear cache periodically
            if self.config.empty_cache_freq > 0 and eval_count % self.config.empty_cache_freq == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Vote
            vote = self.fitness_evaluator.compare_pair(fitness_pos, fitness_neg)
            votes[seed] = vote

            if vote > 0:
                positive_wins += 1
            elif vote < 0:
                negative_wins += 1
            else:
                ties += 1

            all_fitness.extend([fitness_pos.total_fitness, fitness_neg.total_fitness])

        # Update weights based on votes
        num_updates = self.perturbation_handler.update_from_votes(
            votes=votes,
            seeds=seeds,
            threshold=self.config.vote_threshold,
            update_scale=self.config.update_scale,
        )

        # Compute statistics
        mean_fitness = sum(all_fitness) / len(all_fitness)
        best_fitness = max(all_fitness)
        worst_fitness = min(all_fitness)

        # Update state
        self.state.fitness_history.append(mean_fitness)

        if best_fitness > self.state.best_fitness + self.config.min_improvement:
            self.state.best_fitness = best_fitness
            self.state.best_generation = gen
            self.state.generations_without_improvement = 0
        else:
            self.state.generations_without_improvement += 1

        # Decay noise scale
        self.state.current_noise_scale = max(
            self.state.current_noise_scale * self.config.noise_decay,
            self.config.noise_min
        )

        self.state.generation += 1
        elapsed = time.time() - start_time

        return GenerationResult(
            generation=gen,
            mean_fitness=mean_fitness,
            best_fitness=best_fitness,
            worst_fitness=worst_fitness,
            num_updates=num_updates,
            positive_wins=positive_wins,
            negative_wins=negative_wins,
            ties=ties,
            noise_scale=self.state.current_noise_scale,
            elapsed_time=elapsed,
        )

    def run(
        self,
        num_generations: Optional[int] = None,
        callback=None,
    ) -> EvolutionState:
        """
        Run the full evolution loop.

        Args:
            num_generations: Override config num_generations
            callback: Optional callback function(generation_result)

        Returns:
            Final EvolutionState
        """
        if num_generations is None:
            num_generations = self.config.num_generations

        print(f"\n[PixelGenEvolution] Starting evolution for {num_generations} generations")
        print("=" * 60)

        total_start = time.time()

        for gen in range(num_generations):
            result = self.run_generation()

            # Logging
            if gen % self.config.log_every == 0 or gen == num_generations - 1:
                self._log_generation(result)

            # Checkpointing
            if gen % self.config.checkpoint_every == 0 or gen == num_generations - 1:
                self.save_checkpoint(f"gen_{gen:05d}")

            # W&B logging
            if self.wandb_run is not None:
                self.wandb_run.log(result.to_dict(), step=gen)

                # Log sample images periodically
                if gen % self.config.log_images_every == 0 or gen == num_generations - 1:
                    self._log_sample_images(gen)

            # Callback
            if callback is not None:
                callback(result)

            # Early stopping
            if self.state.generations_without_improvement >= self.config.patience:
                print(f"\n[Early stopping] No improvement for {self.config.patience} generations")
                break

        total_time = time.time() - total_start
        print("=" * 60)
        print(f"[PixelGenEvolution] Evolution complete!")
        print(f"  - Total time: {total_time:.1f}s")
        print(f"  - Best fitness: {self.state.best_fitness:.4f} (gen {self.state.best_generation})")
        print(f"  - Final noise scale: {self.state.current_noise_scale:.6f}")

        # Save final checkpoint
        self.save_checkpoint("final")

        return self.state

    def _log_generation(self, result: GenerationResult):
        """Log generation results to console."""
        print(f"Gen {result.generation:4d} | "
              f"Fitness: {result.mean_fitness:.4f} (best: {result.best_fitness:.4f}) | "
              f"Votes: +{result.positive_wins}/-{result.negative_wins}/={result.ties} | "
              f"Updates: {result.num_updates} | "
              f"σ: {result.noise_scale:.5f} | "
              f"Time: {result.elapsed_time:.1f}s")

    def save_checkpoint(self, name: str):
        """Save checkpoint with evolved weights and state."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save evolved parameters
        param_path = checkpoint_dir / f"{name}_params.pt"
        torch.save(self.perturbation_handler.state_dict(), param_path)

        # Save evolution state
        state_path = checkpoint_dir / f"{name}_state.json"
        with open(state_path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

        # Save full model checkpoint
        model_path = checkpoint_dir / f"{name}_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'evolution_state': self.state.to_dict(),
            'config': {
                'population_size': self.config.population_size,
                'noise_scale': self.config.noise_scale,
                'noise_decay': self.config.noise_decay,
                'vote_threshold': self.config.vote_threshold,
            },
        }, model_path)

    def load_checkpoint(self, path: str):
        """Load checkpoint and restore state."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load evolution state
        if 'evolution_state' in checkpoint:
            self.state = EvolutionState.from_dict(checkpoint['evolution_state'])

        # Update perturbation handler's original values
        for name, param in self.model.named_parameters():
            if name in self.perturbation_handler.evolvable_params:
                self.perturbation_handler.original_values[name] = param.data.clone()

        print(f"[PixelGenEvolution] Loaded checkpoint from {path}")
        print(f"  - Generation: {self.state.generation}")
        print(f"  - Best fitness: {self.state.best_fitness:.4f}")

    def evaluate_current(self, num_batches: int = 10) -> FitnessResult:
        """
        Evaluate current model (without perturbation).

        Args:
            num_batches: Number of batches to average

        Returns:
            FitnessResult
        """
        results = []
        for _ in range(num_batches):
            batch = self._get_batch()
            result = self.fitness_evaluator.evaluate_batch(
                model=self.model,
                batch=batch,
                scheduler=self.scheduler,
            )
            results.append(result)

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

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int = 4,
        num_steps: int = 25,
        seed: Optional[int] = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sample images using current model.

        Args:
            num_samples: Number of images to generate
            num_steps: Denoising steps
            seed: Random seed for reproducibility
            class_labels: Optional class labels (random if None)

        Returns:
            Tuple of (generated_images, class_labels)
        """
        import math

        if seed is not None:
            torch.manual_seed(seed)

        # Get image size from model
        img_size = self.model.input_size if hasattr(self.model, 'input_size') else 256

        # Create noise
        noise = torch.randn(
            num_samples, 3, img_size, img_size,
            device=self.device, dtype=self.dtype
        )

        # Class labels
        if class_labels is None:
            # Check if model has num_classes attribute
            num_classes = getattr(self.model, 'num_classes', 1000)
            if num_classes <= 1:
                # Unconditional: all zeros
                class_labels = torch.zeros(num_samples, device=self.device, dtype=torch.long)
            else:
                class_labels = torch.randint(
                    0, num_classes, (num_samples,), device=self.device
                )

        # Timestep schedule (adaptive cosine)
        t_min, t_max = 0.002, 0.998
        t = torch.linspace(0, 1, num_steps + 1, device=self.device)
        timesteps = t_min + (t_max - t_min) * (1 - torch.cos(t * math.pi / 2))

        x = noise

        for i in range(num_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_cur

            t_batch = t_cur.expand(num_samples)

            # Model prediction (x_0 prediction)
            pred = self.model(x, t_batch, class_labels)

            # Compute velocity
            v = (pred - x) / (1 - t_cur).clamp_min(0.002)

            # Euler step
            x = x + v * dt

        return x, class_labels

    @torch.no_grad()
    def reconstruct_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, Dict],
        t_values: List[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct images at different noise levels.

        Shows model's ability to denoise at various timesteps.

        Args:
            batch: (x, y, metadata) tuple
            t_values: Timesteps to visualize

        Returns:
            Dict with 'original', 'noisy_t', 'reconstructed_t' for each t
        """
        if t_values is None:
            t_values = [0.2, 0.5, 0.8]

        x, y, metadata = batch
        x = x.to(self.device, dtype=self.dtype)
        y = y.to(self.device)

        results = {'original': x.clone()}

        for t_val in t_values:
            # Create noisy version
            t = torch.full((x.shape[0],), t_val, device=self.device)
            noise = torch.randn_like(x)

            # Flow matching interpolation: x_t = t*x_0 + (1-t)*noise
            x_noisy = t_val * x + (1 - t_val) * noise

            # Model prediction
            x_pred = self.model(x_noisy, t, y)

            results[f'noisy_{t_val}'] = x_noisy.clone()
            results[f'reconstructed_{t_val}'] = x_pred.clone()

        return results

    def _log_sample_images(self, generation: int):
        """Log sample images to wandb."""
        try:
            import wandb
        except ImportError:
            return

        if self.wandb_run is None:
            return

        print(f"  [Logging images to wandb...]")

        # 1. Generate fresh samples
        generated, labels = self.generate_samples(
            num_samples=self.config.num_sample_images,
            num_steps=self.config.sample_steps,
            seed=self.config.fixed_noise_seed,
        )

        # 2. Get a batch for reconstruction
        batch = self._get_batch()
        reconstructions = self.reconstruct_batch(batch, t_values=[0.3, 0.6, 0.9])

        # Convert to displayable format [0, 1]
        def to_display(img):
            # Clamp and normalize from [-1, 1] to [0, 1]
            img = img.float().clamp(-1, 1)
            return (img + 1) / 2

        # Create image grids
        images_to_log = {}

        # Generated images
        gen_images = to_display(generated)
        images_to_log['generated'] = [
            wandb.Image(
                gen_images[i].permute(1, 2, 0).cpu().numpy(),
                caption=f"Class {labels[i].item()}"
            )
            for i in range(min(4, gen_images.shape[0]))
        ]

        # Reconstruction comparison
        original = to_display(reconstructions['original'])

        # Create comparison grid for t=0.6
        if 'noisy_0.6' in reconstructions:
            noisy = to_display(reconstructions['noisy_0.6'])
            recon = to_display(reconstructions['reconstructed_0.6'])

            comparison_images = []
            for i in range(min(4, original.shape[0])):
                comparison_images.append(
                    wandb.Image(
                        original[i].permute(1, 2, 0).cpu().numpy(),
                        caption="Original"
                    )
                )
                comparison_images.append(
                    wandb.Image(
                        noisy[i].permute(1, 2, 0).cpu().numpy(),
                        caption="Noisy (t=0.6)"
                    )
                )
                comparison_images.append(
                    wandb.Image(
                        recon[i].permute(1, 2, 0).cpu().numpy(),
                        caption="Reconstructed"
                    )
                )

            images_to_log['reconstruction'] = comparison_images

        # Log all images
        self.wandb_run.log({
            'images/generated': images_to_log.get('generated', []),
            'images/reconstruction': images_to_log.get('reconstruction', []),
            'generation': generation,
        }, step=generation)

        # Also save locally
        self._save_sample_images(generation, generated, labels, reconstructions)

    def _save_sample_images(
        self,
        generation: int,
        generated: torch.Tensor,
        labels: torch.Tensor,
        reconstructions: Dict[str, torch.Tensor],
    ):
        """Save sample images locally."""
        import torchvision.utils as vutils

        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        def to_display(img):
            img = img.float().clamp(-1, 1)
            return (img + 1) / 2

        # Save generated images grid
        gen_grid = vutils.make_grid(to_display(generated), nrow=2, padding=2)
        vutils.save_image(gen_grid, images_dir / f"gen_{generation:05d}_generated.png")

        # Save reconstruction comparison
        if 'original' in reconstructions and 'reconstructed_0.6' in reconstructions:
            orig = to_display(reconstructions['original'][:4])
            recon = to_display(reconstructions['reconstructed_0.6'][:4])

            # Interleave original and reconstructed
            comparison = torch.stack([orig, recon], dim=1).view(-1, *orig.shape[1:])
            comp_grid = vutils.make_grid(comparison, nrow=4, padding=2)
            vutils.save_image(comp_grid, images_dir / f"gen_{generation:05d}_reconstruction.png")
