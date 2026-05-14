#!/bin/bash
# =============================================================================
# Upload PixelGen Checkpoint to S3
# =============================================================================
# Upload a trained checkpoint to S3 for sharing or cloud deployment
# Usage: ./upload_checkpoint.sh /path/to/checkpoint.pt
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

# Load environment if exists
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/checkpoint.pt [s3_prefix]"
    echo ""
    echo "Examples:"
    echo "  $0 ./evolution_output/checkpoints/final_model.pt"
    echo "  $0 ./evolution_output/checkpoints/gen_01000_model.pt trained/v1"
    exit 1
fi

CHECKPOINT_PATH="$1"
S3_PREFIX="${2:-checkpoints/pixelgen_evo}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check required variables
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "ERROR: Missing AWS credentials"
    echo "Set: export AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx"
    exit 1
fi

# Configuration
S3_BUCKET="${S3_BUCKET:-pixelgen-training}"
S3_REGION="${S3_REGION:-us-east-1}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║              Upload PixelGen Checkpoint to S3                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Local:  $CHECKPOINT_PATH"
echo "S3:     s3://$S3_BUCKET/$S3_PREFIX/"
echo ""

# Get file info
FILE_SIZE=$(du -h "$CHECKPOINT_PATH" | cut -f1)
FILE_NAME=$(basename "$CHECKPOINT_PATH")
echo "File:   $FILE_NAME ($FILE_SIZE)"
echo ""

# Confirm
read -p "Upload this checkpoint? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Upload
echo ""
echo "Uploading..."
aws s3 cp "$CHECKPOINT_PATH" "s3://$S3_BUCKET/$S3_PREFIX/$FILE_NAME" \
    --region "$S3_REGION"

# Also upload associated files if they exist
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
BASE_NAME="${FILE_NAME%_model.pt}"

# Upload state file
STATE_FILE="$CHECKPOINT_DIR/${BASE_NAME}_state.json"
if [ -f "$STATE_FILE" ]; then
    echo "Uploading state file..."
    aws s3 cp "$STATE_FILE" "s3://$S3_BUCKET/$S3_PREFIX/$(basename $STATE_FILE)" \
        --region "$S3_REGION"
fi

# Upload params file
PARAMS_FILE="$CHECKPOINT_DIR/${BASE_NAME}_params.pt"
if [ -f "$PARAMS_FILE" ]; then
    echo "Uploading params file..."
    aws s3 cp "$PARAMS_FILE" "s3://$S3_BUCKET/$S3_PREFIX/$(basename $PARAMS_FILE)" \
        --region "$S3_REGION"
fi

echo ""
echo "=== Upload Complete ==="
echo ""
echo "Checkpoint URL: s3://$S3_BUCKET/$S3_PREFIX/$FILE_NAME"
echo ""
echo "To download on another machine:"
echo "  aws s3 cp s3://$S3_BUCKET/$S3_PREFIX/$FILE_NAME ./"
