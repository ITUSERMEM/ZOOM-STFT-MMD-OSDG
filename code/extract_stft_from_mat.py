"""
Extract 2D STFT 224x224 features from original .mat files.

Usage:
    python extract_stft_from_mat.py --dataset {pu,phm,hust}

Output directory structure:
    ../data/
        pu_2d_stft_224/     # for PU dataset
        phm_2d_stft_224/    # for PHM dataset
        hust_2d_stft_224/   # for HUST dataset
"""
import os
import argparse
import numpy as np
import scipy.io
from scipy.signal import stft
from scipy.ndimage import zoom
from scipy.signal.windows import hann


# Dataset configurations
DATASETS = {
    'pu': {
        'mat_path': '../data/C-PUdata12.mat',
        'output_dir': '../data/pu_2d_stft_224',
        'fs': 64000,
        'loads': [6, 7, 8, 9],
        'nperseg': 256,
        'noverlap': 128,
    },
    'phm': {
        'mat_path': '../data/C-PHMdata6.mat',
        'output_dir': '../data/phm_2d_stft_224',
        'fs': 66666.7,
        'loads': [9, 10, 11, 12],
        'nperseg': 256,
        'noverlap': 128,
    },
    'hust': {
        'mat_path': '../data/C-HUSTbearingdata9.mat',
        'output_dir': '../data/hust_2d_stft_224',
        'fs': 25600,
        'loads': [17, 18, 19, 20],
        'nperseg': 128,
        'noverlap': 78,
    },
}

TARGET_H = 224
TARGET_W = 224


def process_single_sample(sig, fs, nperseg, noverlap):
    f, t, Zxx = stft(sig, fs=fs, window=hann(nperseg),
                     nperseg=nperseg, noverlap=noverlap,
                     boundary='zeros')
    S = np.abs(Zxx)
    zoom_factors = (TARGET_H / S.shape[0], TARGET_W / S.shape[1])
    S_zoomed = zoom(S, zoom_factors, order=1)
    S_min = S_zoomed.min()
    S_max = S_zoomed.max()
    if S_max > S_min:
        S_norm = (S_zoomed - S_min) / (S_max - S_min)
    else:
        S_norm = np.zeros_like(S_zoomed)
    return S_norm.astype(np.float32)


def process_load(mat_data, load_num, split, fs, nperseg, noverlap):
    var_name = f'load{load_num}_{split}'
    signals = mat_data[var_name]
    n_samples = signals.shape[0]
    stft_stack = np.zeros((n_samples, TARGET_H, TARGET_W), dtype=np.float32)
    for i in range(n_samples):
        stft_stack[i] = process_single_sample(signals[i], fs, nperseg, noverlap)
        if (i + 1) % 500 == 0 or i == n_samples - 1:
            print(f'  {var_name}: processed {i+1}/{n_samples}')
    return stft_stack


def extract_dataset(dataset_name):
    cfg = DATASETS[dataset_name]
    mat_path = cfg['mat_path']
    output_dir = cfg['output_dir']
    fs = cfg['fs']
    loads = cfg['loads']
    nperseg = cfg['nperseg']
    noverlap = cfg['noverlap']

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f'MAT file not found: {mat_path}')

    os.makedirs(output_dir, exist_ok=True)
    print(f'Loading {mat_path} ...')
    mat_data = scipy.io.loadmat(mat_path)

    for load_num in loads:
        for split in ['train', 'test']:
            stft_stack = process_load(mat_data, load_num, split, fs, nperseg, noverlap)
            out_name = f'load{load_num}_{split}_stft.npy'
            out_path = os.path.join(output_dir, out_name)
            np.save(out_path, stft_stack)
            print(f'Saved {out_path}: shape={stft_stack.shape}\n')

    print(f'All {dataset_name.upper()} STFT features extracted!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['pu', 'phm', 'hust', 'all'],
                        help='Dataset to extract STFT from')
    args = parser.parse_args()

    if args.dataset == 'all':
        for name in ['pu', 'phm', 'hust']:
            print(f'\n{"="*50}')
            print(f'Extracting {name.upper()}')
            print(f'{"="*50}')
            extract_dataset(name)
    else:
        extract_dataset(args.dataset)


if __name__ == '__main__':
    main()
