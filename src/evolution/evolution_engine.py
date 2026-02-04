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
        for seed in seeds:
            # Positive perturbation
            pert_pos = HashPerturbation(
                seed=seed,
                scale=self.state.current_noise_scale,
                direction=+1,
            )
            fitness_pos = self._evaluate_candidate(pert_pos)

            # Negative perturbation
            pert_neg = HashPerturbation(
                seed=seed,
                scale=self.state.current_noise_scale,
                direction=-1,
            )
            fitness_neg = self._evaluate_candidate(pert_neg)

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
