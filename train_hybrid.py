#!/usr/bin/env python3
"""
PixelGen Hybrid Training: Backprop Warmup + Evolution Fine-tuning

Strategy:
1. Phase 1 (Backprop): Train with gradient descent to get basic denoising
2. Phase 2 (Evolution): Fine-tune with gradient-free evolution

This addresses the key issue: evolution can fine-tune but can't train from scratch.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import math

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='PixelGen Hybrid Training')

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')

    # Phase 1: Backprop warmup
    parser.add_argument('--warmup-steps', type=int, default=500,
                       help='Number of backprop warmup steps')
    parser.add_argument('--warmup-lr', type=float, default=1e-4,
                       help='Learning rate for warmup')

    # Phase 2: Evolution
    parser.add_argument('--evo-generations', type=int, default=None,
                       help='Evolution generations (override config)')

    # Skip phases
    parser.add_argument('--skip-warmup', action='store_true',
                       help='Skip backprop warmup (use pretrained)')
    parser.add_argument('--skip-evolution', action='store_true',
                       help='Skip evolution (only backprop)')

    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained JiT weights')

    # Output
    parser.add_argument('--output-dir', type=str, default='./hybrid_output',
                       help='Output directory')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name')

    # Logging
    parser.add_argument('--wandb', action='store_true',
                       help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='pixelgen-hybrid',
                       help='W&B project name')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'])

    # Quick test
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with minimal steps')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: dict, device: str, dtype: torch.dtype):
    """Create JiT model from config."""
    from src.models.transformer.JiT import JiT, JiT_models

    denoiser_config = config.get('model', {}).get('denoiser', {})
    model_name = denoiser_config.get('model_name', 'JiT-L/16')

    if model_name in JiT_models:
        model = JiT_models[model_name](
            input_size=denoiser_config.get('input_size', 256),
            num_classes=denoiser_config.get('num_classes', 1000),
        )
    else:
        model = JiT(
            input_size=denoiser_config.get('input_size', 256),
            patch_size=denoiser_config.get('patch_size', 16),
            hidden_size=denoiser_config.get('hidden_size', 1024),
            depth=denoiser_config.get('depth', 24),
            num_heads=denoiser_config.get('num_heads', 16),
            mlp_ratio=denoiser_config.get('mlp_ratio', 4.0),
            num_classes=denoiser_config.get('num_classes', 1000),
        )

    model = model.to(device=device, dtype=dtype)
    return model


def create_scheduler(config: dict):
    """Create flow matching scheduler."""
    from src.diffusion.flow_matching.scheduling import LinearScheduler
    return LinearScheduler()


def create_dataloader(config: dict, batch_size: int):
    """Create data loader."""
    from train_evo import (
        SyntheticImageDataset, OverfitDataset, FolderImageDataset, collate_fn
    )
    from torch.utils.data import DataLoader

    data_config = config.get('data', {})
    data_dir = data_config.get('train_data_dir', '/data/imagenet/train')
    dataset_type = data_config.get('dataset_type', 'imagenet')
    img_size = data_config.get('img_size', 256)
    num_classes = config.get('model', {}).get('denoiser', {}).get('num_classes', 1000)

    if not os.path.exists(data_dir):
        print(f"[INFO] Using synthetic dataset (data dir not found: {data_dir})")
        dataset = SyntheticImageDataset(size=10000, resolution=img_size, num_classes=num_classes)
    elif dataset_type == 'overfit':
        num_images = data_config.get('num_images', 1)
        dataset = OverfitDataset(root=data_dir, resolution=img_size, num_images=num_images)
    elif dataset_type == 'folder':
        dataset = FolderImageDataset(root=data_dir, resolution=img_size, num_classes=num_classes)
    else:
        from src.data.dataset.imagenet import PixImageNet
        dataset = PixImageNet(root=data_dir, resolution=img_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get('train_num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader


def backprop_warmup(
    model: nn.Module,
    scheduler,
    dataloader: DataLoader,
    num_steps: int,
    lr: float,
    device: str,
    dtype: torch.dtype,
    wandb_run=None,
    log_every: int = 10,
):
    """
    Phase 1: Backprop warmup to get basic denoising working.

    Uses simple flow matching loss with gradient descent.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: BACKPROP WARMUP")
    print("=" * 60)
    print(f"Steps: {num_steps}")
    print(f"Learning rate: {lr}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine annealing
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, eta_min=lr * 0.1)

    data_iter = iter(dataloader)
    losses = []

    for step in range(num_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x, y, metadata = batch
        x = x.to(device, dtype=dtype)
        y = y.to(device)

        # Sample timestep (log-normal as in PixelGen)
        t = torch.sigmoid(torch.randn(x.shape[0], device=device, dtype=dtype) * 0.8 - 0.8)

        # Create noisy input
        alpha = scheduler.alpha(t)
        sigma = scheduler.sigma(t)
        noise = torch.randn_like(x)
        x_t = alpha * x + sigma * noise

        # Forward pass - use autocast for mixed precision training
        with torch.amp.autocast('cuda', dtype=dtype):
            pred = model(x_t, t, y)

        # Flow matching loss (predict clean image) - compute in float32 for stability
        loss = F.mse_loss(pred.float(), x.float())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler_lr.step()

        losses.append(loss.item())

        # Logging
        if step % log_every == 0 or step == num_steps - 1:
            avg_loss = sum(losses[-log_every:]) / len(losses[-log_every:])
            current_lr = scheduler_lr.get_last_lr()[0]
            print(f"Step {step:4d}/{num_steps} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

            if wandb_run:
                wandb_run.log({
                    'warmup/loss': avg_loss,
                    'warmup/lr': current_lr,
                    'warmup/step': step,
                }, step=step)

    model.eval()

    final_loss = sum(losses[-50:]) / min(50, len(losses))
    print(f"\nWarmup complete! Final avg loss: {final_loss:.4f}")

    return final_loss


def evolution_phase(
    model: nn.Module,
    scheduler,
    dataloader: DataLoader,
    config: dict,
    num_generations: int,
    device: str,
    dtype: torch.dtype,
    output_dir: Path,
    wandb_run=None,
    strategy: str = 'default',
):
    """
    Phase 2: Evolution fine-tuning.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: EVOLUTION FINE-TUNING")
    print("=" * 60)

    from src.evolution import (
        EvolutionConfig, FitnessConfig, PixelGenEvolution,
        create_evolvable_jit
    )
    from train_evo import create_dino_encoder

    # Create evolvable wrapper
    evolvable = create_evolvable_jit(model, strategy)
    evolvable.print_summary()

    # Create DINO encoder
    dino_encoder = create_dino_encoder(device, dtype)

    # Get evolution config from YAML
    evo_config_dict = config.get('evolution', {})
    fitness_config_dict = evo_config_dict.get('fitness', {})

    fitness_config = FitnessConfig(
        w_flow_matching=fitness_config_dict.get('w_flow_matching', 0.35),
        w_lpips=fitness_config_dict.get('w_lpips', 0.30),
        w_dino=fitness_config_dict.get('w_dino', 0.25),
        w_ssim=fitness_config_dict.get('w_ssim', 0.10),
    )

    batch_size = evo_config_dict.get('eval_batch_size', 4)

    evo_config = EvolutionConfig(
        population_size=evo_config_dict.get('population_size', 8),
        num_generations=num_generations,
        noise_scale=evo_config_dict.get('noise_scale', 0.01),
        noise_decay=evo_config_dict.get('noise_decay', 0.995),
        vote_threshold=evo_config_dict.get('vote_threshold', 1),
        update_scale=evo_config_dict.get('update_scale', 0.0001),
        eval_batch_size=batch_size,
        checkpoint_every=evo_config_dict.get('checkpoint_every', 50),
        log_every=evo_config_dict.get('log_every', 10),
        log_images_every=evo_config_dict.get('log_images_every', 50),
        patience=evo_config_dict.get('patience', 200),
        device=device,
        dtype=dtype,
        layer_config=evolvable.config,
        fitness_config=fitness_config,
    )

    # Create evolution engine
    evolution = PixelGenEvolution(
        model=model,
        scheduler=scheduler,
        dataloader=dataloader,
        config=evo_config,
        dino_encoder=dino_encoder,
        output_dir=str(output_dir),
        wandb_run=wandb_run,
    )

    # Run evolution
    final_state = evolution.run()

    return final_state


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Setup
    device = args.device
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Output directory
    exp_name = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PIXELGEN HYBRID TRAINING")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Skip warmup: {args.skip_warmup}")
    print(f"Skip evolution: {args.skip_evolution}")

    # Quick test overrides
    if args.quick_test:
        args.warmup_steps = 50
        args.evo_generations = 20
        print("\n[QUICK TEST MODE]")

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
        model.load_state_dict(state_dict, strict=False)
        args.skip_warmup = True  # Skip warmup if using pretrained

    # Create scheduler and dataloader
    scheduler = create_scheduler(config)
    batch_size = config.get('data', {}).get('train_batch_size', 4)
    dataloader = create_dataloader(config, batch_size)

    # Setup W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config={
                'warmup_steps': args.warmup_steps,
                'warmup_lr': args.warmup_lr,
                'model': config.get('model', {}),
                'evolution': config.get('evolution', {}),
            },
        )

    # Phase 1: Backprop Warmup
    if not args.skip_warmup:
        warmup_loss = backprop_warmup(
            model=model,
            scheduler=scheduler,
            dataloader=dataloader,
            num_steps=args.warmup_steps,
            lr=args.warmup_lr,
            device=device,
            dtype=dtype,
            wandb_run=wandb_run,
        )

        # Save warmup checkpoint
        warmup_path = output_dir / 'warmup_checkpoint.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'warmup_loss': warmup_loss,
        }, warmup_path)
        print(f"Saved warmup checkpoint to {warmup_path}")

    # Phase 2: Evolution
    if not args.skip_evolution:
        evo_generations = args.evo_generations or config.get('evolution', {}).get('num_generations', 500)

        final_state = evolution_phase(
            model=model,
            scheduler=scheduler,
            dataloader=dataloader,
            config=config,
            num_generations=evo_generations,
            device=device,
            dtype=dtype,
            output_dir=output_dir,
            wandb_run=wandb_run,
        )

        print(f"\nEvolution complete!")
        print(f"Best fitness: {final_state.best_fitness:.4f} (gen {final_state.best_generation})")

    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_path)
    print(f"\nSaved final model to {final_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
