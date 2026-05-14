# Vast.ai Cloud Training for PixelGen Evolution

This directory contains scripts for running PixelGen evolution training on Vast.ai cloud GPUs.

## Why Cloud Training?

- **A100 80GB**: Much faster than local consumer GPUs
- **No VRAM constraints**: Full population size, more generations
- **Cost-effective**: ~$1.50-2.50/hr for A100 80GB on-demand
- **Checkpoint protection**: Auto-sync to S3 every 15 minutes

## Prerequisites

1. **Vast.ai Account**: https://vast.ai/
2. **AWS S3 Bucket**: For training data and checkpoints
3. **W&B Account** (optional): For training visualization

## Quick Start

### 1. Rent a GPU Instance

```bash
# Install Vast.ai CLI
pip install vastai

# Set API key
vastai set api-key YOUR_API_KEY

# Search for A100 instances
vastai search offers "gpu_ram>=40 reliability>=0.95 num_gpus=1"

# Rent an instance (note the instance ID)
vastai create instance <offer_id> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
```

### 2. SSH and Setup

```bash
# Get SSH command
vastai ssh-url <instance_id>

# SSH into the instance
ssh -p <port> root@<host>

# Set credentials
export AWS_ACCESS_KEY_ID="your_aws_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret"
export WANDB_API_KEY="your_wandb_key"  # Optional

# Run setup script
curl -sSL https://raw.githubusercontent.com/johndpope/PixelGen/main/tools/vast-cloud-training/setup_vastai_pixelgen.sh | bash
```

### 3. Start Training

```bash
# Quick test (synthetic data, 20 generations)
cd /workspace/PixelGen
bash tools/vast-cloud-training/scripts/train_pixelgen_evo.sh --quick-test --synthetic

# Full training (ImageNet, 1000 generations)
bash tools/vast-cloud-training/scripts/train_pixelgen_evo.sh
```

### 4. Monitor Training

```bash
# Attach to training session
tmux attach -t training

# View logs
tail -f /workspace/PixelGen/evolution_output/training.log

# GPU usage
nvtop

# W&B dashboard
# Open https://wandb.ai/your-project
```

### 5. Download Checkpoints Locally

```bash
# On your local machine
cd /path/to/PixelGen
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
bash tools/vast-cloud-training/scripts/download_checkpoints.sh ./my_checkpoints
```

## Scripts

| Script | Description |
|--------|-------------|
| `setup_vastai_pixelgen.sh` | Full instance setup (dependencies, data, repo) |
| `scripts/train_pixelgen_evo.sh` | Start evolution training with auto S3 sync |
| `scripts/download_checkpoints.sh` | Download checkpoints from S3 locally |
| `scripts/upload_checkpoint.sh` | Upload local checkpoint to S3 |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | Yes | - | AWS access key for S3 |
| `AWS_SECRET_ACCESS_KEY` | Yes | - | AWS secret key |
| `S3_BUCKET` | No | `pixelgen-training` | S3 bucket name |
| `S3_REGION` | No | `us-east-1` | AWS region |
| `WANDB_API_KEY` | No | - | W&B API key for logging |
| `TRAINING_GENERATIONS` | No | `1000` | Number of generations |
| `POPULATION_SIZE` | No | `16` | Population size |
| `MAX_RUNTIME_HOURS` | No | `24` | Auto-shutdown timer |

## Cost Estimation

| Duration | A100 On-Demand | A100 Spot |
|----------|----------------|-----------|
| 100 generations (~2h) | ~$4-5 | ~$2-3 |
| 500 generations (~10h) | ~$20-25 | ~$10-15 |
| 1000 generations (~20h) | ~$40-50 | ~$20-30 |

*Note: Spot instances can be preempted. Use on-demand for critical training.*

## S3 Structure

```
s3://pixelgen-training/
├── imagenet/                    # Training data (cached)
│   └── train/
│       ├── n00000001/
│       └── ...
├── models/                      # Pre-trained weights
│   └── jit_pretrained.pt
└── checkpoints/
    └── pixelgen_evo/            # Evolution checkpoints
        ├── gen_00000_model.pt
        ├── gen_00050_model.pt
        ├── ...
        ├── final_model.pt
        └── images/              # Sample images
            └── gen_00050_generated.png
```

## Troubleshooting

### Instance won't start
```bash
# Check available offers with more relaxed requirements
vastai search offers "gpu_ram>=24 reliability>=0.90"
```

### Training OOM
- Reduce `POPULATION_SIZE` (try 8 instead of 16)
- Use `--synthetic` for testing

### Checkpoints not syncing
```bash
# Manual sync
aws s3 sync /workspace/PixelGen/evolution_output s3://pixelgen-training/checkpoints/pixelgen_evo/
```

### Can't SSH
```bash
# Get fresh SSH URL
vastai ssh-url <instance_id>

# Check instance status
vastai show instances
```
