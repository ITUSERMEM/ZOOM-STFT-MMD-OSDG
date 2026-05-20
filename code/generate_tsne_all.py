"""
Generate t-SNE visualizations for all 12 tasks (PU/PHM/HUST).
Outputs 4 plots per task to ../visualization/{dataset}_task{task_index}/
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from resnet18_2d import CNN_2D

# Config
DATASET_CONFIG = {
    'pu': {
        'class_num': 4,
        'class_names': ['EDM', 'Drilling', 'Engraver', 'Pitting'],
        'src_tar': np.array([[6, 7, 8, 9], [6, 7, 9, 8], [6, 8, 9, 7], [7, 8, 9, 6]]),
        'data_root': '../data/pu_2d_stft_224',
        'use_map': True,
        'domain_names': {6: 'Load6 (25Hz)', 7: 'Load7 (25Hz)', 8: 'Load8 (25Hz)', 9: 'Load9 (15Hz)'},
    },
    'phm': {
        'class_num': 6,
        'class_names': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'],
        'src_tar': np.array([[9, 10, 11, 12], [9, 10, 12, 11], [9, 11, 12, 10], [10, 11, 12, 9]]),
        'data_root': '../data/phm_2d_stft_224',
        'use_map': False,
        'domain_names': {9: 'Load9', 10: 'Load10', 11: 'Load11', 12: 'Load12'},
    },
    'hust': {
        'class_num': 9,
        'class_names': [f'C{i}' for i in range(9)],
        'src_tar': np.array([[17, 18, 19, 20], [17, 18, 20, 19], [17, 19, 20, 18], [18, 19, 20, 17]]),
        'data_root': '../data/hust_2d_stft_224',
        'use_map': False,
        'domain_names': {17: 'Load17', 18: 'Load18', 19: 'Load19', 20: 'Load20'},
    },
}

MAP_12_TO_4 = torch.tensor([0, 0, 1, 2, 2, 3, 3, 1, 2, 2, 3, 3], dtype=torch.long)


def map_labels(labels):
    if labels.dim() == 2:
        labels = labels[:, 0]
    labels = labels.long().cpu()
    return MAP_12_TO_4[labels]


def load_test_data(data_root, load_num, class_num, device):
    path = os.path.join(data_root, f'load{load_num}_test_stft.npy')
    data = np.load(path)
    samples_per_class = data.shape[0] // class_num
    labels = np.arange(data.shape[0]) // samples_per_class
    data = torch.from_numpy(data).float().unsqueeze(1).to(device)
    labels = torch.from_numpy(labels).long()
    return data, labels


def extract_features(model, data, batch_size=256):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            _, feat = model(data[i:i+batch_size])
            feats.append(feat.cpu())
    return torch.cat(feats, dim=0).numpy()


def plot_tsne(feat_2d, labels, colors, title, out_path, cmap='tab10', alpha=0.6, s=15):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=colors, cmap=cmap, alpha=alpha, s=s)
    plt.title(title, fontsize=14)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_tsne_categorical(feat_2d, values, value_names, title, out_path, colors_map=None):
    plt.figure(figsize=(10, 8))
    unique_vals = sorted(set(values))
    for val in unique_vals:
        mask = values == val
        color = colors_map[val] if colors_map else None
        plt.scatter(feat_2d[mask, 0], feat_2d[mask, 1],
                    label=value_names.get(val, str(val)), alpha=0.6, s=15, color=color)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_task_tsne(dataset, task_index, model_path, out_dir):
    cfg = DATASET_CONFIG[dataset]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_num = cfg['class_num']
    use_map = cfg['use_map']
    src_tar = cfg['src_tar']
    src_list = src_tar[task_index, :3]
    tgt_load = src_tar[task_index, 3]

    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = CNN_2D(num_classes=class_num, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load all data for this task (train + test from all 4 loads)
    all_data = []
    all_labels = []
    all_domains = []
    all_is_target = []

    for load in list(src_list) + [tgt_load]:
        data, labels = load_test_data(cfg['data_root'], load, cfg['class_num'] if not use_map else 12, device)
        if use_map:
            labels = map_labels(labels)
        # Use a subset to keep t-SNE fast (max 800 per load)
        n = min(len(data), 800)
        indices = torch.randperm(len(data))[:n]
        all_data.append(data[indices])
        all_labels.append(labels[indices])
        all_domains.extend([load] * n)
        all_is_target.extend([0 if load in src_list else 1] * n)

    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_domains = np.array(all_domains)
    all_is_target = np.array(all_is_target)

    # Extract features
    print(f'  Extracting features for {dataset} task {task_index}...')
    feats = extract_features(model, all_data)

    # t-SNE (use a subset if too large, but 3200 is fine)
    print(f'  Running t-SNE...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, max_iter=1000)
    feat_2d = tsne.fit_transform(feats)

    # Plot 1: by class (all data)
    class_names = cfg['class_names']
    plot_tsne_categorical(
        feat_2d, all_labels.numpy(),
        {i: class_names[i] for i in range(len(class_names))},
        f'{dataset.upper()} Task {task_index}: t-SNE by class',
        os.path.join(out_dir, f'{dataset}_task{task_index}_tsne_by_class.png')
    )

    # Plot 2: by domain (all data)
    domain_names = cfg['domain_names']
    plot_tsne_categorical(
        feat_2d, all_domains,
        domain_names,
        f'{dataset.upper()} Task {task_index}: t-SNE by domain',
        os.path.join(out_dir, f'{dataset}_task{task_index}_tsne_by_domain.png')
    )

    # Plot 3: target only (by class)
    target_mask = all_is_target == 1
    if target_mask.sum() > 0:
        plot_tsne_categorical(
            feat_2d[target_mask], all_labels[target_mask].numpy(),
            {i: class_names[i] for i in range(len(class_names))},
            f'{dataset.upper()} Task {task_index}: Target only (Load{tgt_load})',
            os.path.join(out_dir, f'{dataset}_task{task_index}_tsne_target.png')
        )

    # Plot 4: target correct vs incorrect
    with torch.no_grad():
        preds = []
        for i in range(0, len(all_data), 256):
            pred, _ = model(all_data[i:i+256])
            preds.append(pred.argmax(dim=1).cpu())
        preds = torch.cat(preds, dim=0).numpy()

    correct = (preds == all_labels.numpy()).astype(int)
    plot_tsne_categorical(
        feat_2d, correct,
        {1: 'Correct', 0: 'Incorrect'},
        f'{dataset.upper()} Task {task_index}: Correct vs Incorrect',
        os.path.join(out_dir, f'{dataset}_task{task_index}_tsne_errors.png'),
        colors_map={1: 'green', 0: 'red'}
    )

    print(f'  Saved 4 plots to {out_dir}')


def main():
    out_base = '../visualization'
    os.makedirs(out_base, exist_ok=True)

    tasks = [
        ('pu', 0), ('pu', 1), ('pu', 2), ('pu', 3),
        ('phm', 0), ('phm', 1), ('phm', 2), ('phm', 3),
        ('hust', 0), ('hust', 1), ('hust', 2), ('hust', 3),
    ]

    for dataset, task_index in tasks:
        print(f'\n[{dataset.upper()}] Task {task_index}')
        model_name = f'ERM_2d_{dataset.upper()}_task{task_index}_repeat0_best.pth'
        model_path = f'../models/{model_name}'
        out_dir = os.path.join(out_base, f'{dataset}_task{task_index}')

        if not os.path.exists(model_path):
            print(f'  WARNING: model not found: {model_path}')
            continue

        generate_task_tsne(dataset, task_index, model_path, out_dir)

    print('\nAll t-SNE visualizations completed!')


if __name__ == '__main__':
    main()
