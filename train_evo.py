#!/usr/bin/env python3
"""
PixelGen Evolutionary Training Script

Trains PixelGen using gradient-free evolutionary optimization instead of backpropagation.

Usage:
    # Quick test run
    python train_evo.py --config configs_evo/PixelGen_XL_evo.yaml --quick-test

    # Full training
    python train_evo.py --config configs_evo/PixelGen_XL_evo.yaml

    # Resume from checkpoint
    python train_evo.py --config configs_evo/PixelGen_XL_evo.yaml --resume path/to/checkpoint.pt

Key differences from standard training:
- No gradients computed (pure fitness evaluation)
- Uses antithetic sampling with voting-based weight updates
- Selectively evolves final layer, in-context tokens, late attention
- Memory efficient (no optimizer state, no gradient storage)
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='PixelGen Evolutionary Training')

    # Config
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')

    # Overrides
    parser.add_argument('--population', type=int, default=None,
                       help='Population size (override config)')
    parser.add_argument('--generations', type=int, default=None,
                       help='Number of generations (override config)')
    parser.add_argument('--noise-scale', type=float, default=None,
                       help='Initial noise scale (override config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Evaluation batch size (override config)')

    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained JiT weights')

    # Output
    parser.add_argument('--output-dir', type=str, default='./evolution_output',
                       help='Output directory')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name (default: timestamp)')

    # Logging
    parser.add_argument('--wandb', action='store_true',
                       help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='pixelgen-evo',
                       help='W&B project name')

    # Debug
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with small population')
    parser.add_argument('--dry-run', action='store_true',
                       help='Initialize but don\'t train')
    parser.add_argument('--analyze-model', action='store_true',
                       help='Just analyze model and exit')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Computation dtype')

    # Evolution strategy
    parser.add_argument('--strategy', type=str, default='default',
                       choices=['minimal', 'default', 'aggressive', 'full'],
                       help='Evolution strategy for layer selection')

    # Data options
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data (for testing without ImageNet)')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class SyntheticImageDataset(torch.utils.data.Dataset):
    """Synthetic dataset for testing when ImageNet is not available."""

    def __init__(self, size=1000, resolution=256, num_classes=1000):
        self.size = size
        self.resolution = resolution
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random image in [0, 1]
        raw_image = torch.rand(3, self.resolution, self.resolution)

        # Normalize to [-1, 1]
        normalized_image = raw_image * 2 - 1

        # Random class
        target = idx % self.num_classes

        metadata = {
            "raw_image": raw_image,
            "class": target,
        }
        return normalized_image, target, metadata


def collate_fn(batch):
    """Collate function for PixelGen data format."""
    import copy
    batch = copy.deepcopy(batch)
    x, y, metadata = list(zip(*batch))
    stacked_metadata = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked_metadata[key] = torch.stack([m[key] for m in metadata], dim=0)
            else:
                stacked_metadata[key] = [m[key] for m in metadata]
        except:
            pass
    x = torch.stack(x, dim=0)
    y = torch.tensor(y) if not isinstance(y[0], torch.Tensor) else torch.stack(y)
    return x, y, stacked_metadata


def create_dataloader(config: dict, batch_size: int, use_synthetic: bool = False) -> DataLoader:
    """Create data loader for fitness evaluation."""
    from torch.utils.data import DataLoader

    data_config = config.get('data', {})
    data_dir = data_config.get('train_data_dir', '/data/imagenet/train')

    # Check if data exists
    if use_synthetic or not os.path.exists(data_dir):
        print(f"[INFO] Using synthetic dataset (data dir not found: {data_dir})")
        dataset = SyntheticImageDataset(
            size=1000,
            resolution=data_config.get('img_size', 256),
            num_classes=1000,
        )
    else:
        from src.data.dataset.imagenet import PixImageNet
        dataset = PixImageNet(
            root=data_dir,
            resolution=data_config.get('img_size', 256),
            random_crop=False,
            random_flip=True,
        )

    # Create dataloader (no distributed sampler for evolution)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get('train_num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader  # Return DataLoader, not iterator


def create_model(config: dict, device: str, dtype: torch.dtype) -> nn.Module:
    """Create JiT model from config."""
    from src.models.transformer.JiT import JiT, JiT_models

    denoiser_config = config.get('model', {}).get('denoiser', {})

    # Get model variant
    model_name = denoiser_config.get('model_name', 'JiT-L/16')
    if model_name in JiT_models:
        model = JiT_models[model_name](
            input_size=denoiser_config.get('input_size', 256),
            num_classes=denoiser_config.get('num_classes', 1000),
        )
    else:
        # Custom configuration
        model = JiT(
            input_size=denoiser_config.get('input_size', 256),
            patch_size=denoiser_config.get('patch_size', 16),
            hidden_size=denoiser_config.get('hidden_size', 1024),
            depth=denoiser_config.get('depth', 24),
            num_heads=denoiser_config.get('num_heads', 16),
            mlp_ratio=denoiser_config.get('mlp_ratio', 4.0),
            num_classes=denoiser_config.get('num_classes', 1000),
            in_context_len=denoiser_config.get('in_context_len', 32),
            in_context_start=denoiser_config.get('in_context_start', 8),
        )

    model = model.to(device=device, dtype=dtype)
    model.eval()  # Evolution doesn't use dropout

    return model


def create_scheduler(config: dict):
    """Create flow matching scheduler."""
    from src.diffusion.flow_matching.scheduling import LinearScheduler

    scheduler_config = config.get('model', {}).get('scheduler', {})
    return LinearScheduler()


def create_dino_encoder(device: str, dtype: torch.dtype):
    """Create DINOv2 encoder for fitness evaluation."""
    from src.models.encoder import DINOv2

    encoder = DINOv2(base_patch_size=16)
    encoder = encoder.to(device=device)
    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    return encoder


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device and dtype
    device = args.device
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 70)
    print("PIXELGEN EVOLUTIONARY TRAINING")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Strategy: {args.strategy}")

    # Create output directory
    exp_name = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create model
    print("\nLoading model...")
    model = create_model(config, device, dtype)

    # Load pretrained weights if provided
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)

    # Analyze model
    from src.evolution import EvolvableJiT, create_evolvable_jit
    evolvable = create_evolvable_jit(model, args.strategy)
    evolvable.print_summary()

    if args.analyze_model:
        # Just analyze and exit
        stats = evolvable.get_group_statistics()
        print("\nDetailed Statistics:")
        for group, s in stats.items():
            print(f"\n{group}:")
            for k, v in s.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
        return

    # Create scheduler
    scheduler = create_scheduler(config)

    # Create DINO encoder
    print("Loading DINO encoder...")
    dino_encoder = create_dino_encoder(device, dtype)

    # Create dataloader
    batch_size = args.batch_size or config.get('evolution', {}).get('eval_batch_size', 4)
    print(f"Creating dataloader (batch_size={batch_size})...")
    dataloader = create_dataloader(config, batch_size, use_synthetic=args.synthetic)

    # Create evolution config
    from src.evolution import EvolutionConfig, FitnessConfig

    evo_config_dict = config.get('evolution', {})
    fitness_config_dict = evo_config_dict.get('fitness', {})

    fitness_config = FitnessConfig(
        w_flow_matching=fitness_config_dict.get('w_flow_matching', 0.35),
        w_lpips=fitness_config_dict.get('w_lpips', 0.30),
        w_dino=fitness_config_dict.get('w_dino', 0.25),
        w_ssim=fitness_config_dict.get('w_ssim', 0.10),
        percept_t_threshold=fitness_config_dict.get('percept_t_threshold', 0.3),
    )

    # Apply overrides
    population_size = args.population or evo_config_dict.get('population_size', 8)
    num_generations = args.generations or evo_config_dict.get('num_generations', 1000)
    noise_scale = args.noise_scale or evo_config_dict.get('noise_scale', 0.01)

    # Quick test overrides
    if args.quick_test:
        population_size = 4
        num_generations = 10
        batch_size = 2
        print("\n[QUICK TEST MODE]")

    evo_config = EvolutionConfig(
        population_size=population_size,
        num_generations=num_generations,
        noise_scale=noise_scale,
        noise_decay=evo_config_dict.get('noise_decay', 0.995),
        vote_threshold=evo_config_dict.get('vote_threshold', 3),
        update_scale=evo_config_dict.get('update_scale', 0.001),
        eval_batch_size=batch_size,
        num_eval_batches=evo_config_dict.get('num_eval_batches', 1),
        checkpoint_every=evo_config_dict.get('checkpoint_every', 50),
        log_every=evo_config_dict.get('log_every', 10),
        patience=evo_config_dict.get('patience', 100),
        device=device,
        dtype=dtype,
        layer_config=evolvable.config,
        fitness_config=fitness_config,
    )

    print(f"\nEvolution Config:")
    print(f"  Population: {evo_config.population_size}")
    print(f"  Generations: {evo_config.num_generations}")
    print(f"  Noise scale: {evo_config.noise_scale}")
    print(f"  Vote threshold: {evo_config.vote_threshold}")

    # Setup W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config={
                'evolution': evo_config.__dict__,
                'model': config.get('model', {}),
                'strategy': args.strategy,
            },
        )

    # Create evolution engine
    from src.evolution import PixelGenEvolution

    evolution = PixelGenEvolution(
        model=model,
        scheduler=scheduler,
        dataloader=dataloader,
        config=evo_config,
        dino_encoder=dino_encoder,
        output_dir=str(output_dir),
        wandb_run=wandb_run,
    )

    # Resume if requested
    if args.resume:
        evolution.load_checkpoint(args.resume)

    if args.dry_run:
        print("\n[DRY RUN] Initialized successfully, exiting.")
        return

    # Run evolution
    print("\nStarting evolution...")
    final_state = evolution.run()

    # Save final results
    results = {
        'best_fitness': final_state.best_fitness,
        'best_generation': final_state.best_generation,
        'final_generation': final_state.generation,
        'fitness_history': final_state.fitness_history,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
