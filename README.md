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

