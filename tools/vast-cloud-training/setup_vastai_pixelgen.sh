#!/bin/bash
# =============================================================================
# Full Vast.ai Setup Script for PixelGen Evolution Training
# =============================================================================
#
# Usage:
#   1. Rent a GPU instance on Vast.ai (A100 80GB recommended)
#   2. SSH into the instance
#   3. Set environment variables:
#      export AWS_ACCESS_KEY_ID="your_aws_key"
#      export AWS_SECRET_ACCESS_KEY="your_aws_secret"
#      export WANDB_API_KEY="your_wandb_key"
#   4. Run: curl -sSL https://raw.githubusercontent.com/johndpope/PixelGen/main/tools/vast-cloud-training/setup_vastai_pixelgen.sh | bash
#
# =============================================================================

set -e

# Configuration
WORKSPACE="/workspace"
DATA_DIR="/data"
MODEL_DIR="$DATA_DIR/models"
IMAGENET_DIR="$DATA_DIR/imagenet"
S3_BUCKET="${S3_BUCKET:-pixelgen-training}"
S3_REGION="${S3_REGION:-us-east-1}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║            PixelGen Evolution Training - Vast.ai Setup               ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# Check environment variables
# =============================================================================
check_env() {
    if [ -z "${!1}" ]; then
        echo "⚠️  WARNING: $1 is not set"
        return 1
    else
        echo "✓ $1 is set"
        return 0
    fi
}

echo "[0/8] Checking environment variables..."
check_env "AWS_ACCESS_KEY_ID" || AWS_MISSING=1
check_env "AWS_SECRET_ACCESS_KEY" || AWS_MISSING=1
check_env "WANDB_API_KEY" || WANDB_MISSING=1

if [ "$AWS_MISSING" = "1" ]; then
    echo ""
    echo "❌ AWS credentials are REQUIRED for S3 data sync."
    echo "   Set: export AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx"
    exit 1
fi

echo ""

# =============================================================================
# Step 1: Install system dependencies
# =============================================================================
echo "[1/8] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git curl unzip htop nvtop wget

# Install AWS CLI
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install
    rm -rf awscliv2.zip aws/
fi
echo "✓ System dependencies installed"

# =============================================================================
# Step 2: Install uv for Python package management
# =============================================================================
echo "[2/8] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
echo "✓ uv installed"

# =============================================================================
# Step 3: Configure AWS credentials
# =============================================================================
echo "[3/8] Configuring AWS credentials..."
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
cat > ~/.aws/config << EOF
[default]
region = $S3_REGION
EOF
echo "✓ AWS credentials configured"

# =============================================================================
# Step 4: Clone PixelGen repository
# =============================================================================
echo "[4/8] Setting up PixelGen repository..."
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

if [ ! -d "PixelGen" ]; then
    git clone https://github.com/johndpope/PixelGen.git
    cd PixelGen
    git checkout feature/evolution-training-and-speedup || git checkout main
    echo "✓ Repository cloned"
else
    cd PixelGen
    git pull origin main || true
    echo "✓ Repository updated"
fi

# =============================================================================
# Step 5: Setup Python environment
# =============================================================================
echo "[5/8] Setting up Python environment..."
cd "$WORKSPACE/PixelGen"

# Create requirements if not exists
if [ ! -f "requirements.txt" ]; then
    cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision
timm
einops
lpips
wandb
PyYAML
tqdm
pillow
numpy
scipy
EOF
fi

# Install with uv or pip
if command -v uv &> /dev/null; then
    uv pip install -r requirements.txt --system
else
    pip install -r requirements.txt
fi
echo "✓ Python environment ready"

# =============================================================================
# Step 6: Download ImageNet subset for training
# =============================================================================
echo "[6/8] Setting up training data..."
mkdir -p "$IMAGENET_DIR"

# Check if we have data on S3
echo "   Checking S3 for training data..."
if aws s3 ls "s3://$S3_BUCKET/imagenet/" --region "$S3_REGION" 2>/dev/null; then
    echo "   Downloading ImageNet from S3..."
    aws s3 sync "s3://$S3_BUCKET/imagenet/" "$IMAGENET_DIR/" --region "$S3_REGION"
else
    echo "   No ImageNet data on S3. Downloading ImageNet-1k subset..."

    # Option 1: Download ImageNet-1k validation set (smaller, good for testing)
    # This is a common subset used for quick experiments
    pip install datasets -q
    python << 'PYEOF'
import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

output_dir = "/data/imagenet/train"
os.makedirs(output_dir, exist_ok=True)

print("Downloading ImageNet-1k validation set (50k images)...")
# Use validation set as it's smaller and doesn't require special access
dataset = load_dataset("imagenet-1k", split="validation", trust_remote_code=True)

# Save images organized by class
for i, sample in enumerate(tqdm(dataset, desc="Saving images")):
    img = sample['image']
    label = sample['label']

    class_dir = os.path.join(output_dir, f"n{label:08d}")
    os.makedirs(class_dir, exist_ok=True)

    img_path = os.path.join(class_dir, f"{i:08d}.JPEG")
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(img_path)

print(f"Saved {len(dataset)} images to {output_dir}")
PYEOF

    # Upload to S3 for future use
    echo "   Uploading ImageNet to S3 for future instances..."
    aws s3 sync "$IMAGENET_DIR/" "s3://$S3_BUCKET/imagenet/" --region "$S3_REGION" || true
fi

# Count images
NUM_IMAGES=$(find "$IMAGENET_DIR" -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
echo "✓ Training data ready: $NUM_IMAGES images"

# =============================================================================
# Step 7: Download pre-trained JiT weights (optional)
# =============================================================================
echo "[7/8] Checking for pre-trained weights..."
mkdir -p "$MODEL_DIR"

# Check S3 for pre-trained weights
if aws s3 ls "s3://$S3_BUCKET/models/jit_pretrained.pt" --region "$S3_REGION" 2>/dev/null; then
    echo "   Downloading pre-trained JiT weights from S3..."
    aws s3 cp "s3://$S3_BUCKET/models/jit_pretrained.pt" "$MODEL_DIR/jit_pretrained.pt" --region "$S3_REGION"
    echo "✓ Pre-trained weights downloaded"
else
    echo "   No pre-trained weights found. Training from scratch."
    echo "   (You can upload weights to s3://$S3_BUCKET/models/jit_pretrained.pt)"
fi

# =============================================================================
# Step 8: Configure W&B
# =============================================================================
if [ -n "$WANDB_API_KEY" ]; then
    echo "[8/8] Configuring Weights & Biases..."
    pip install -q wandb
    wandb login "$WANDB_API_KEY"
    echo "✓ W&B configured"
else
    echo "[8/8] Skipping W&B (no API key)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                        Setup Complete!                                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Directories:"
echo "   Repository:     $WORKSPACE/PixelGen"
echo "   Training Data:  $IMAGENET_DIR"
echo "   Models:         $MODEL_DIR"
echo ""
echo "📊 Disk usage:"
du -sh "$IMAGENET_DIR" "$MODEL_DIR" 2>/dev/null || true
echo ""
echo "🚀 To start EVOLUTION training:"
echo "   cd $WORKSPACE/PixelGen"
echo "   python train_evo.py --config configs_evo/PixelGen_XL_evo.yaml \\"
echo "       --wandb --wandb-project pixelgen-evo"
echo ""
echo "🚀 Quick test (synthetic data):"
echo "   python train_evo.py --config configs_evo/PixelGen_XL_evo.yaml \\"
echo "       --quick-test --synthetic --wandb"
echo ""
echo "📈 Monitor:"
echo "   GPU: nvtop"
echo "   W&B: https://wandb.ai"
echo ""
