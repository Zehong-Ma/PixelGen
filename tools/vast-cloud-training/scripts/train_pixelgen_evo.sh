#!/bin/bash
# =============================================================================
# PixelGen Evolution Training on Vast.ai
# =============================================================================
# Run this on the Vast.ai instance after setup completes
# Usage: bash train_pixelgen_evo.sh [--resume /path/to/checkpoint]
# =============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║              PixelGen Evolution Training on Vast.ai                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Started at: $(date)"
echo ""

# Configuration from environment
S3_BUCKET=${S3_BUCKET:-"pixelgen-training"}
S3_REGION=${S3_REGION:-"us-east-1"}
TRAINING_GENERATIONS=${TRAINING_GENERATIONS:-1000}
POPULATION_SIZE=${POPULATION_SIZE:-16}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-24}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-50}

WORKSPACE="/workspace"
REPO_PATH="$WORKSPACE/PixelGen"
DATA_DIR="/data/imagenet"
MODEL_DIR="/data/models"
OUTPUT_DIR="$REPO_PATH/evolution_output"
CHECKPOINT_S3_PREFIX="checkpoints/pixelgen_evo"

# Parse arguments
RESUME_PATH=""
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        --quick-test)
            TRAINING_GENERATIONS=20
            POPULATION_SIZE=4
            EXTRA_ARGS="$EXTRA_ARGS --quick-test"
            shift
            ;;
        --synthetic)
            EXTRA_ARGS="$EXTRA_ARGS --synthetic"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# =============================================================================
# Setup auto-shutdown watchdog + checkpoint upload
# =============================================================================
if [ "$MAX_RUNTIME_HOURS" != "0" ]; then
    MAX_SECONDS=$((MAX_RUNTIME_HOURS * 3600))
    echo "⏰ Auto-shutdown in $MAX_RUNTIME_HOURS hours"
    (
        sleep $MAX_SECONDS
        echo ""
        echo "=== MAX RUNTIME REACHED ==="
        # Upload final checkpoints
        if [ -d "$OUTPUT_DIR" ]; then
            echo "Uploading final checkpoints to S3..."
            aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/$CHECKPOINT_S3_PREFIX/" \
                --region "$S3_REGION" \
                --exclude "*.log" || true
        fi
        echo "Shutting down..."
        poweroff || exit 0
    ) &
    WATCHDOG_PID=$!
    echo "   Watchdog PID: $WATCHDOG_PID"
fi

# =============================================================================
# Setup periodic checkpoint upload (every 15 minutes)
# =============================================================================
(
    while true; do
        sleep 900  # 15 minutes
        if [ -d "$OUTPUT_DIR/checkpoints" ]; then
            echo "[$(date)] 📤 Syncing checkpoints to S3..."
            aws s3 sync "$OUTPUT_DIR/checkpoints" "s3://$S3_BUCKET/$CHECKPOINT_S3_PREFIX/" \
                --region "$S3_REGION" \
                --exclude "*.log" \
                --include "*.pt" \
                --include "*.json" \
                --include "*.yaml" \
                || true
            echo "[$(date)] ✓ Checkpoint sync complete"
        fi

        # Also sync images if they exist
        if [ -d "$OUTPUT_DIR/images" ]; then
            aws s3 sync "$OUTPUT_DIR/images" "s3://$S3_BUCKET/$CHECKPOINT_S3_PREFIX/images/" \
                --region "$S3_REGION" \
                --include "*.png" \
                || true
        fi
    done
) &
CHECKPOINT_SYNC_PID=$!
echo "📤 Checkpoint sync process: $CHECKPOINT_SYNC_PID (every 15 min)"

# =============================================================================
# Configure AWS
# =============================================================================
echo ""
echo "=== Configuring AWS ==="
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "$S3_REGION"

# Test S3 access
aws s3 ls "s3://$S3_BUCKET/" --region "$S3_REGION" > /dev/null 2>&1 || {
    echo "Creating S3 bucket..."
    aws s3 mb "s3://$S3_BUCKET" --region "$S3_REGION" || true
}
echo "✓ S3 bucket ready: s3://$S3_BUCKET"

# =============================================================================
# Check for Resume
# =============================================================================
RESUME_FLAG=""
if [ -n "$RESUME_PATH" ]; then
    echo ""
    echo "=== Resuming from checkpoint ==="
    echo "Checkpoint: $RESUME_PATH"
    RESUME_FLAG="--resume $RESUME_PATH"
elif [ -d "$OUTPUT_DIR/checkpoints" ]; then
    # Check for latest checkpoint
    LATEST=$(ls -t "$OUTPUT_DIR/checkpoints"/*_model.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo ""
        echo "=== Found existing checkpoint ==="
        echo "Latest: $LATEST"
        read -p "Resume from this checkpoint? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            RESUME_FLAG="--resume $LATEST"
        fi
    fi
else
    # Try to download latest checkpoint from S3
    echo ""
    echo "=== Checking S3 for existing checkpoints ==="
    LATEST_S3=$(aws s3 ls "s3://$S3_BUCKET/$CHECKPOINT_S3_PREFIX/" --region "$S3_REGION" 2>/dev/null | grep "_model.pt" | sort | tail -1 | awk '{print $4}')
    if [ -n "$LATEST_S3" ]; then
        echo "Found checkpoint on S3: $LATEST_S3"
        read -p "Download and resume? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            mkdir -p "$OUTPUT_DIR/checkpoints"
            aws s3 cp "s3://$S3_BUCKET/$CHECKPOINT_S3_PREFIX/$LATEST_S3" "$OUTPUT_DIR/checkpoints/$LATEST_S3" --region "$S3_REGION"
            RESUME_FLAG="--resume $OUTPUT_DIR/checkpoints/$LATEST_S3"
        fi
    fi
fi

# =============================================================================
# Start Training
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    Starting Evolution Training                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Configuration:"
echo "   Generations:     $TRAINING_GENERATIONS"
echo "   Population:      $POPULATION_SIZE"
echo "   Output:          $OUTPUT_DIR"
echo "   S3 Checkpoints:  s3://$S3_BUCKET/$CHECKPOINT_S3_PREFIX/"
echo ""

cd "$REPO_PATH"
mkdir -p "$OUTPUT_DIR"

# Build the training command
TRAIN_CMD="python train_evo.py \
    --config configs_evo/PixelGen_XL_evo.yaml \
    --generations $TRAINING_GENERATIONS \
    --population $POPULATION_SIZE \
    --output-dir $OUTPUT_DIR \
    --wandb --wandb-project pixelgen-evo \
    $RESUME_FLAG \
    $EXTRA_ARGS"

echo "Command: $TRAIN_CMD"
echo ""

# Run training in tmux for persistence
tmux new-session -d -s training "cd $REPO_PATH && $TRAIN_CMD 2>&1 | tee $OUTPUT_DIR/training.log"

echo "✓ Training started in tmux session 'training'"
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                         Monitoring Commands                           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  📺 Attach to training:  tmux attach -t training"
echo "  📜 View logs:           tail -f $OUTPUT_DIR/training.log"
echo "  🖥️  GPU usage:           nvtop"
echo "  📁 Local checkpoints:   ls -la $OUTPUT_DIR/checkpoints/"
echo "  ☁️  S3 checkpoints:      aws s3 ls s3://$S3_BUCKET/$CHECKPOINT_S3_PREFIX/"
echo ""
echo "  To detach from tmux: Ctrl+B, then D"
echo ""

# Attach to the session
sleep 2
echo "Attaching to training session..."
tmux attach -t training
