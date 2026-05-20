# Domain Generalization Fault Diagnosis — Final Package

12-task cross-domain fault diagnosis benchmark with SpecAug + MMD/ERM on PU, PHM, and HUST datasets.

## Package Contents

```
final_package/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes large data/model files
├── code/                       # Source code
│   ├── ERM_2d_all_mmd.py      # Training script (PU/PHM/HUST)
│   ├── extract_stft_from_mat.py # STFT extraction from .mat
│   ├── generate_tsne_all.py   # t-SNE visualization
│   ├── resnet18_2d.py         # ResNet18 backbone
│   ├── data_loader_2d.py      # Data loader
│   ├── utils.py               # MMD / CORAL / losses
│   └── arcface_module.py      # ArcFace (optional)
├── data/                       # Original .mat data (not tracked by git)
│   ├── C-PUdata12.mat
│   ├── C-PHMdata6.mat
│   └── C-HUSTbearingdata9.mat
├── models/                     # Pre-trained best models (not tracked by git)
│   └── *.pth                   # 12 task models (~513 MB total)
└── visualization/              # t-SNE plots (48 images)
```

## Quick Start

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Extract STFT features from .mat files

All scripts run from the `code/` directory.

```bash
cd code/

# Extract all datasets at once (~10-15 min, generates ~21 GB)
python extract_stft_from_mat.py --dataset all

# Or extract one by one
python extract_stft_from_mat.py --dataset pu
python extract_stft_from_mat.py --dataset phm
python extract_stft_from_mat.py --dataset hust
```

Output directories created:
- `../data/pu_2d_stft_224/`
- `../data/phm_2d_stft_224/`
- `../data/hust_2d_stft_224/`

### Step 3: Train a model

```bash
# Example: PU Task 0 (4-class + SpecAug + MMD)
python ERM_2d_all_mmd.py --dataset pu --task_index 0 --repeat 0 \
    --iteration 5000 --lr 0.005 --batch_size 256 --mmd_weight 1.0

# Example: PHM Task 1 (use MMD lambda=0.01 to avoid negative transfer)
python ERM_2d_all_mmd.py --dataset phm --task_index 1 --repeat 0 \
    --iteration 5000 --lr 0.005 --batch_size 256 --mmd_weight 0.01

# Example: HUST Task 0 (9-class + SpecAug + MMD)
python ERM_2d_all_mmd.py --dataset hust --task_index 0 --repeat 0 \
    --iteration 5000 --lr 0.005 --batch_size 256 --mmd_weight 1.0
```

### Step 4: Generate t-SNE visualizations

```bash
# Generate all 12 tasks x 4 plots = 48 images
python generate_tsne_all.py
```

Output: `../visualization/{dataset}_task{index}/` (4 PNGs per task)

## Data & Models

Due to file size limits, **raw `.mat` data and trained model weights are not tracked by git** (see `.gitignore`).

### Option A: Use provided files (if available)
If you received this package with `data/*.mat` and `models/*.pth` included, place them in the corresponding directories.

### Option B: Download original datasets
- **PU**: Request from [C-PU bearing dataset](https://github.com/) or contact authors
- **PHM**: [PHM Society 2009 Competition](https://github.com/)
- **HUST**: [HUST-bearing dataset](https://github.com/)

Place `.mat` files in `data/` and run `extract_stft_from_mat.py`.

### Option C: Train from scratch
After extracting STFT features, run `ERM_2d_all_mmd.py` for each task. Training takes ~13 minutes per task on a single GPU.

## Results Summary

| Dataset | Task Avg | Best Config |
|---------|----------|-------------|
| PU (4-class) | 97.40% | SpecAug + MMD λ=1.0 |
| PHM (6-class) | 97.79% | SpecAug + MMD λ=1.0 (Task 1: λ=0.01) |
| HUST (9-class) | 99.42% | SpecAug + MMD λ=1.0 |
| **Overall 12-task avg** | **98.20%** | |

## Key Finding: PHM Task 1 MMD Sensitivity

PHM Task 1 (source=9,10,12 → target=11) is uniquely sensitive to MMD weight:
- λ=1.0 → **22.17%** (feature collapse)
- λ=0.5 → 71.58%
- λ=0.1 → 76.33%
- **λ=0.01 → 94.25%** ✅
- λ=0.0 (pure ERM) → 92.33%

**Recommendation**: For PHM Task 1 only, use MMD λ=0.01. All other tasks use λ=1.0.

## Citation

Chao Zhao, Enrico Zio, Weiming Shen. "Domain Generalization for Cross-Domain Fault Diagnosis: an Application-oriented Perspective and a Benchmark Study." Reliability Engineering and System Safety, 2024.
