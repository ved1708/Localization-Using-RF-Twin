# Fine-tuning Experiments System

This system runs systematic fine-tuning experiments to evaluate how many dynamic scene images are needed for effective adaptation.

## Quick Start

```bash
# 1. Setup experiment datasets (creates subsets with 5, 10, 20, 50, 100 images)
python setup_finetuning_experiments.py

# 2. Run all experiments (trains 5 models, ~30-60 min total)
./run_finetuning_experiments.sh

# 3. View results summary
./visualize_results.sh
```

## What It Does

### Step 1: Setup (`setup_finetuning_experiments.sh`)
- Creates 5 dataset subsets from `dataset_dynamic_scene_ideal_mpc/`
- Each subset has N training images + all test images
- N = 5, 10, 20, 50, 100 images
- Preserves train/test split consistency

**Output Structure:**
```
finetuning_experiments/
├── dynamic_5images/
│   ├── spectrum/        # 5 train + 20 test images
│   ├── cameras.txt
│   ├── images.txt
│   ├── train_index.txt
│   └── test_index.txt
├── dynamic_10images/
├── dynamic_20images/
├── dynamic_50images/
└── dynamic_100images/
```

### Step 2: Training (`run_finetuning_experiments.sh`)
For each dataset size:
1. Starts from checkpoint: `output/rf_model/chkpnt40000.pth`
2. Trains for 5,000 more iterations (40000 → 45000)
3. Evaluates on test set at iteration 45000
4. Saves results and model

**Output:**
```
finetuning_experiments/
├── finetuning_results.txt     # Summary table
├── log_5images.txt             # Detailed training log
├── log_10images.txt
├── log_20images.txt
├── log_50images.txt
└── log_100images.txt

output/
├── finetune_5images/           # Trained model
│   └── point_cloud/iteration_45000/
├── finetune_10images/
├── finetune_20images/
├── finetune_50images/
└── finetune_100images/
```

### Step 3: Results (`visualize_results.sh`)
Generates summary table:

```
N_imgs  | Train_PSNR | Test_PSNR | Generalization_Gap
--------|------------|-----------|-------------------
      5 |    18.5432 |   14.2134 |            4.3298
     10 |    19.1245 |   15.0823 |            4.0422
     20 |    19.8567 |   15.9234 |            3.9333
     50 |    20.3421 |   16.5123 |            3.8298
    100 |    20.6234 |   16.8745 |            3.7489
```

## Expected Runtime

- **Setup**: ~1 minute (file copying)
- **Training per experiment**: ~5-10 minutes (5000 iterations)
- **Total runtime**: ~30-60 minutes (5 experiments)

## Experiment Details

**Training Configuration:**
- Base model: RF model trained on static scene (40K iterations)
- Fine-tuning: 5K additional iterations on dynamic scene
- Learning rate: Inherited from checkpoint (decayed)
- Frozen parameters: xyz positions, scaling, rotation (RF-specific)
- Trainable: Opacity, SH features (color/RF response)

**Evaluation Metrics:**
- **L1 Loss**: Mean absolute error (lower is better)
- **PSNR**: Peak Signal-to-Noise Ratio in dB (higher is better)
- **Gap**: Train PSNR - Test PSNR (measures overfitting)

## Research Questions

This experiment answers:
1. **Minimum data requirement**: How few images can still adapt effectively?
2. **Diminishing returns**: At what point do more images stop helping?
3. **Generalization**: Does more data reduce overfitting?
4. **Cost-benefit**: Optimal training size vs. quality tradeoff

## Manual Inspection

Check specific model:
```bash
# View training log
cat finetuning_experiments/log_20images.txt

# Render test views
cd output/finetune_20images
python ../../render.py -m . --images spectrum
```

## Troubleshooting

**If setup fails:**
```bash
# Check source dataset
ls -la /home/ved/Ved/Project_1/dataset_dynamic_scene_ideal_mpc/spectrum/
cat /home/ved/Ved/Project_1/dataset_dynamic_scene_ideal_mpc/train_index.txt
```

**If training fails:**
```bash
# Check checkpoint exists
ls -la output/rf_model/chkpnt40000.pth

# Run single experiment manually
python train.py \
  -s /home/ved/Ved/Project_1/finetuning_experiments/dynamic_5images \
  -m output/test_run \
  --images spectrum \
  --start_checkpoint output/rf_model/chkpnt40000.pth \
  --iterations 45000 \
  --test_iterations 45000
```

## Expected Outcomes

**Hypothesis:**
- 5-10 images: Underfitting, high test error
- 20-50 images: Good adaptation, reasonable generalization
- 100 images: Best performance, diminishing returns

**Success Criteria:**
- Test PSNR improves from baseline (14.16 dB)
- Generalization gap stays under 5 dB
- Training converges within 5K iterations
