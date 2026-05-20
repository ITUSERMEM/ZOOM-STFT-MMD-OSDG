"""
ERM 2D with Strong SpecAugment + MMD for all datasets: PU, PHM, HUST.

- PU: 4-class damage type mapping (from 12 original classes)
- PHM: 6-class (original)
- HUST: 9-class (original)
"""
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
from data_loader_2d import load_training, load_testing
from resnet18_2d import CNN_2D
from utils import mmd_rbf_noaccelerate

# PU 12-class -> 4-class damage type mapping
MAP_12_TO_4 = torch.tensor([0, 0, 1, 2, 2, 3, 3, 1, 2, 2, 3, 3], dtype=torch.long)


def map_labels(labels):
    if labels.dim() == 2:
        labels = labels[:, 0]
    labels = labels.long().cpu()
    return MAP_12_TO_4[labels]


def spec_augment(x, freq_mask_param=50, time_mask_param=50,
                 num_freq_masks=2, num_time_masks=2):
    B, C, H, W = x.shape
    y = x.clone()
    for b in range(B):
        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_param)
            if f > 0:
                f0 = random.randint(0, max(0, H - f))
                y[b, :, f0:f0 + f, :] = 0.0
        for _ in range(num_time_masks):
            t = random.randint(0, time_mask_param)
            if t > 0:
                t0 = random.randint(0, max(0, W - t))
                y[b, :, :, t0:t0 + t] = 0.0
    return y


DATASET_CONFIG = {
    'pu': {
        'class_num_orig': 12,
        'class_num_train': 4,
        'src_tar': np.array([[6, 7, 8, 9], [6, 7, 9, 8], [6, 8, 9, 7], [7, 8, 9, 6]]),
        'data_root': 'data/pu_2d_stft_224',
        'use_map': True,
    },
    'phm': {
        'class_num_orig': 6,
        'class_num_train': 6,
        'src_tar': np.array([[9, 10, 11, 12], [9, 10, 12, 11], [9, 11, 12, 10], [10, 11, 12, 9]]),
        'data_root': 'data/phm_2d_stft_224',
        'use_map': False,
    },
    'hust': {
        'class_num_orig': 9,
        'class_num_train': 9,
        'src_tar': np.array([[17, 18, 19, 20], [17, 18, 20, 19], [17, 19, 20, 18], [18, 19, 20, 17]]),
        'data_root': 'data/hust_2d_stft_224',
        'use_map': False,
    },
}


def train(args):
    setup_seed(args.seed)
    cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if cuda else 'cpu'
    kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}

    cfg = DATASET_CONFIG[args.dataset]
    src_tar = cfg['src_tar']
    class_num_orig = cfg['class_num_orig']
    class_num_train = cfg['class_num_train']
    use_map = cfg['use_map']

    data_root = os.path.join(os.path.dirname(__file__), '..', cfg['data_root'])

    src_list = src_tar[args.task_index, :3]
    tgt_list = src_tar[args.task_index, 3]
    print(f"[{args.dataset.upper()}] Task {args.task_index}: source={src_list}, target={tgt_list}")

    src1 = f'load{src_list[0]}_train'
    src2 = f'load{src_list[1]}_train'
    src3 = f'load{src_list[2]}_train'

    train_loader = load_training(data_root, src1, src2, src3, src_list, False,
                                 class_num_orig, args.batch_size, kwargs, device=device)
    tgt_test_loader = load_testing(data_root, f'load{tgt_list}_test', False,
                                   class_num_orig, args.batch_size, kwargs, device=device)
    tgt_train_loader_raw = load_testing(data_root, f'load{tgt_list}_train', False,
                                        class_num_orig, args.batch_size, kwargs, device=device)
    tgt_train_loader = torch.utils.data.DataLoader(
        tgt_train_loader_raw.dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    model = CNN_2D(num_classes=class_num_train, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_class = nn.CrossEntropyLoss().to(device)

    best_acc = 0.0
    for p in model.parameters():
        p.requires_grad = True

    src_iter = iter(train_loader)
    tgt_iter = iter(tgt_train_loader)

    for iter_num in range(args.iteration):
        model.train()
        try:
            data_src, label_src = next(src_iter)
        except StopIteration:
            src_iter = iter(train_loader)
            data_src, label_src = next(src_iter)

        try:
            data_tgt, _ = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_train_loader)
            data_tgt, _ = next(tgt_iter)

        data_src = data_src.to(device)
        data_src = spec_augment(data_src,
                                freq_mask_param=args.freq_mask_param,
                                time_mask_param=args.time_mask_param,
                                num_freq_masks=args.num_freq_masks,
                                num_time_masks=args.num_time_masks)
        if use_map:
            label_src = map_labels(label_src).to(device)
        else:
            if label_src.dim() == 2:
                label_src = label_src[:, 0].long().to(device)
            else:
                label_src = label_src.long().to(device)
        data_tgt = data_tgt.to(device)

        optimizer.zero_grad()

        pred_src, feat_src = model(data_src)
        loss_cls = loss_class(pred_src, label_src)

        _, feat_tgt = model(data_tgt)

        loss_mmd = mmd_rbf_noaccelerate(feat_src, feat_tgt,
                                        kernel_mul=args.mmd_kernel_mul,
                                        kernel_num=args.mmd_kernel_num)

        loss = loss_cls + args.mmd_weight * loss_mmd
        loss.backward()
        optimizer.step()

        if (iter_num + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                tgt_correct = 0
                tgt_total = 0
                for data_tgt_test, label_tgt_test in tgt_test_loader:
                    data_tgt_test = data_tgt_test.to(device)
                    if use_map:
                        label_tgt_test = map_labels(label_tgt_test).to(device)
                    else:
                        if label_tgt_test.dim() == 2:
                            label_tgt_test = label_tgt_test[:, 0].long().to(device)
                        else:
                            label_tgt_test = label_tgt_test.long().to(device)
                    pred_tgt, _ = model(data_tgt_test)
                    pred_cls = pred_tgt.argmax(dim=1)
                    tgt_correct += pred_cls.eq(label_tgt_test).sum().item()
                    tgt_total += label_tgt_test.size(0)
                tgt_acc = 100.0 * tgt_correct / tgt_total
                print(f'Iter {iter_num + 1}/{args.iteration} | Loss={loss.item():.4f} | '
                      f'Cls={loss_cls.item():.4f} | MMD={loss_mmd.item():.4f} | Target Acc={tgt_acc:.2f}%')
                if tgt_acc > best_acc:
                    best_acc = tgt_acc
                    os.makedirs(args.model_dir, exist_ok=True)
                    save_name = f'ERM_2d_{args.dataset.upper()}_task{args.task_index}_repeat{args.repeat}_best.pth'
                    torch.save(model.state_dict(), os.path.join(args.model_dir, save_name))
                    print(f'  -> New best saved: {best_acc:.2f}%')

    print(f'\n[{args.dataset.upper()}] Task {args.task_index} Repeat {args.repeat} | Best Target Acc: {best_acc:.2f}%')
    return best_acc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['pu', 'phm', 'hust'])
    parser.add_argument('--task_index', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--iteration', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freq_mask_param', type=int, default=50)
    parser.add_argument('--time_mask_param', type=int, default=50)
    parser.add_argument('--num_freq_masks', type=int, default=2)
    parser.add_argument('--num_time_masks', type=int, default=2)
    parser.add_argument('--mmd_weight', type=float, default=1.0)
    parser.add_argument('--mmd_kernel_mul', type=float, default=2.0)
    parser.add_argument('--mmd_kernel_num', type=int, default=5)
    parser.add_argument('--model_dir', type=str, default='../models')
    args = parser.parse_args()
    args.seed += args.repeat * 10
    train(args)
