#!/bin/bash
# =============================================================================
# Download PixelGen Evolution Checkpoints from S3
# =============================================================================
# Run this LOCALLY to download checkpoints from cloud training
# Usage: ./download_checkpoints.sh [output_dir]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

# Load environment if exists
if [ -f "$ENV_FILE" ]; then
    echo "Loading credentials from: $ENV_FILE"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Check required variables
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "ERROR: Missing AWS credentials"
    echo ""
    echo "Either:"
    echo "  1. Create .env file with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    echo "  2. Export them: export AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx"
    exit 1
fi

# Configuration
S3_BUCKET="${S3_BUCKET:-pixelgen-training}"
S3_REGION="${S3_REGION:-us-east-1}"
S3_PREFIX="${S3_PREFIX:-checkpoints/pixelgen_evo}"
LOCAL_DIR="${1:-./checkpoints_from_cloud}"

# Configure AWS
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           Download PixelGen Evolution Checkpoints                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "S3:    s3://$S3_BUCKET/$S3_PREFIX/"
echo "Local: $LOCAL_DIR"
echo ""

# List available checkpoints
echo "=== Available Checkpoints on S3 ==="
aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" --region "$S3_REGION" --human-readable || {
    echo "No checkpoints found at s3://$S3_BUCKET/$S3_PREFIX/"
    echo ""
    echo "Make sure:"
    echo "  1. You have the correct S3_BUCKET ($S3_BUCKET)"
    echo "  2. Training has saved checkpoints"
    echo "  3. Your AWS credentials have read access"
    exit 1
}

echo ""

# Options
echo "=== Download Options ==="
echo "1. Download all checkpoints"
echo "2. Download latest only"
echo "3. Download specific checkpoint"
echo "4. Cancel"
echo ""
read -p "Select option [1-4]: " -n 1 -r OPTION
echo ""

case $OPTION in
    1)
        echo ""
        echo "Downloading all checkpoints..."
        mkdir -p "$LOCAL_DIR"
        aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/" "$LOCAL_DIR" --region "$S3_REGION"
        ;;
    2)
        echo ""
        echo "Finding latest checkpoint..."
        LATEST=$(aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" --region "$S3_REGION" | grep "_model.pt" | sort | tail -1 | awk '{print $4}')
        if [ -n "$LATEST" ]; then
            echo "Latest: $LATEST"
            mkdir -p "$LOCAL_DIR"
            aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/$LATEST" "$LOCAL_DIR/$LATEST" --region "$S3_REGION"

            # Also download corresponding state and config
            STATE_FILE="${LATEST/_model.pt/_state.json}"
            aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/$STATE_FILE" "$LOCAL_DIR/$STATE_FILE" --region "$S3_REGION" 2>/dev/null || true
        else
            echo "No checkpoint files found!"
            exit 1
        fi
        ;;
    3)
        echo ""
        echo "Enter checkpoint name (e.g., gen_00500_model.pt):"
        read CHECKPOINT_NAME
        if [ -n "$CHECKPOINT_NAME" ]; then
            mkdir -p "$LOCAL_DIR"
            aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/$CHECKPOINT_NAME" "$LOCAL_DIR/$CHECKPOINT_NAME" --region "$S3_REGION"
        fi
        ;;
    *)
        echo "Cancelled."
        exit 0
        ;;
esac

echo ""
echo "=== Download Complete ==="
echo ""
echo "Downloaded to: $LOCAL_DIR"
ls -la "$LOCAL_DIR"

echo ""
echo "To use the checkpoint:"
echo "  python train_evo.py --config configs_evo/PixelGen_XL_evo.yaml \\"
echo "      --resume $LOCAL_DIR/<checkpoint_name>"
