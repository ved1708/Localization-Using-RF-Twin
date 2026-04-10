#!/bin/bash
# Quick visualization of fine-tuning results

EXPERIMENTS_DIR="/home/ved/Ved/Project_1/finetuning_experiments"
RESULTS_FILE="$EXPERIMENTS_DIR/finetuning_results.txt"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Results file not found: $RESULTS_FILE"
    echo "Please run the experiments first."
    exit 1
fi

echo ""
echo "========================================"
echo "Fine-tuning Results Summary"
echo "========================================"
echo ""

# Display full results
cat "$RESULTS_FILE"

echo ""
echo "========================================"
echo "Analysis"
echo "========================================"
echo ""

# Extract just the metrics
echo "N_imgs  | Train_PSNR | Test_PSNR | Generalization_Gap"
echo "--------|------------|-----------|-------------------"

for N in 5 10 20 50 100; do
    LOG_FILE="$EXPERIMENTS_DIR/log_${N}images.txt"
    if [ -f "$LOG_FILE" ]; then
        TRAIN_PSNR=$(grep "Evaluating train:" "$LOG_FILE" | tail -1 | sed -n 's/.*PSNR \([0-9.]*\).*/\1/p')
        TEST_PSNR=$(grep "Evaluating test:" "$LOG_FILE" | tail -1 | sed -n 's/.*PSNR \([0-9.]*\).*/\1/p')
        
        if [ -n "$TRAIN_PSNR" ] && [ -n "$TEST_PSNR" ]; then
            GAP=$(echo "$TRAIN_PSNR - $TEST_PSNR" | bc -l)
            printf "%7d | %10.4f | %9.4f | %17.4f\n" "$N" "$TRAIN_PSNR" "$TEST_PSNR" "$GAP"
        fi
    fi
done

echo ""
echo "Interpretation:"
echo "- Higher PSNR = Better reconstruction quality"
echo "- Smaller gap = Better generalization to unseen views"
echo "- Test PSNR improvement = Effective adaptation to dynamic scene"
