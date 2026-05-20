import torch
import numpy as np
import os


def _maybe_cuda(tensor, device):
    if device == 'cuda' and torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def load_training_with_val(root_path, dir1, dir2, dir3, src_list, fft1, class_num, batch_size, kwargs, device='cpu', val_ratio=0.15):
    """Load 3 source domains and split a stratified validation set from them.

    Returns (train_loader, val_loader).  The validation set is drawn
    proportionally from each class so that class balance is preserved.
    """
    src_list = torch.from_numpy(src_list)

    # --- load & label 3 sources (same logic as load_training) ---
    path1 = os.path.join(root_path, f'{dir1}_stft.npy')
    train_fea_1 = np.load(path1)
    samples_per_class_1 = train_fea_1.shape[0] // class_num
    train_label_1 = torch.zeros((train_fea_1.shape[0], 2))
    for i in range(train_fea_1.shape[0]):
        train_label_1[i][0] = i // samples_per_class_1
        train_label_1[i][1] = 0

    path2 = os.path.join(root_path, f'{dir2}_stft.npy')
    train_fea_2 = np.load(path2)
    samples_per_class_2 = train_fea_2.shape[0] // class_num
    train_label_2 = torch.zeros((train_fea_2.shape[0], 2))
    for i in range(train_fea_2.shape[0]):
        train_label_2[i][0] = i // samples_per_class_2
        train_label_2[i][1] = 1

    path3 = os.path.join(root_path, f'{dir3}_stft.npy')
    train_fea_3 = np.load(path3)
    samples_per_class_3 = train_fea_3.shape[0] // class_num
    train_label_3 = torch.zeros((train_fea_3.shape[0], 2))
    for i in range(train_fea_3.shape[0]):
        train_label_3[i][0] = i // samples_per_class_3
        train_label_3[i][1] = 2

    train_fea = np.vstack((train_fea_1, train_fea_2, train_fea_3))
    train_label = torch.cat([train_label_1, train_label_2, train_label_3], dim=0)

    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea).float()
    train_fea = train_fea.unsqueeze(1)

    # --- stratified split by class (first column of label) ---
    n_total = train_fea.shape[0]
    indices = torch.randperm(n_total)

    # build class masks
    class_ids = train_label[:, 0]
    train_idx = []
    val_idx = []
    for c in range(class_num):
        c_mask = (class_ids == c)
        c_indices = indices[c_mask[indices]]  # indices that belong to class c
        n_val = int(len(c_indices) * val_ratio)
        val_idx.append(c_indices[:n_val])
        train_idx.append(c_indices[n_val:])

    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)

    train_fea_split = train_fea[train_idx]
    train_label_split = train_label[train_idx]
    val_fea = train_fea[val_idx]
    val_label = train_label[val_idx]

    if device == 'cuda' and torch.cuda.is_available():
        train_fea_split = train_fea_split.cuda()
        train_label_split = train_label_split.cuda()
        val_fea = val_fea.cuda()
        val_label = val_label.cuda()

    train_data = torch.utils.data.TensorDataset(train_fea_split, train_label_split)
    val_data = torch.utils.data.TensorDataset(val_fea, val_label)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader


def load_training(root_path, dir1, dir2, dir3, src_list, fft1, class_num, batch_size, kwargs, device='cpu'):
    src_list = torch.from_numpy(src_list)

    path1 = os.path.join(root_path, f'{dir1}_stft.npy')
    train_fea_1 = np.load(path1)  # (N, F, T)
    samples_per_class_1 = train_fea_1.shape[0] // class_num

    train_label_1 = torch.zeros((train_fea_1.shape[0], 2))
    for i in range(train_fea_1.shape[0]):
        train_label_1[i][0] = i // samples_per_class_1
        train_label_1[i][1] = 0

    path2 = os.path.join(root_path, f'{dir2}_stft.npy')
    train_fea_2 = np.load(path2)
    samples_per_class_2 = train_fea_2.shape[0] // class_num

    train_label_2 = torch.zeros((train_fea_2.shape[0], 2))
    for i in range(train_fea_2.shape[0]):
        train_label_2[i][0] = i // samples_per_class_2
        train_label_2[i][1] = 1

    path3 = os.path.join(root_path, f'{dir3}_stft.npy')
    train_fea_3 = np.load(path3)
    samples_per_class_3 = train_fea_3.shape[0] // class_num

    train_label_3 = torch.zeros((train_fea_3.shape[0], 2))
    for i in range(train_fea_3.shape[0]):
        train_label_3[i][0] = i // samples_per_class_3
        train_label_3[i][1] = 2

    train_fea = np.vstack((train_fea_1, train_fea_2, train_fea_3))
    train_label = torch.cat([train_label_1, train_label_2, train_label_3], dim=0)

    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea).float()
    train_fea = train_fea.unsqueeze(1)

    if device == 'cuda' and torch.cuda.is_available():
        train_fea = train_fea.cuda()
        train_label = train_label.cuda()

    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_testing(root_path, dir_name, fft1, class_num, batch_size, kwargs, device='cpu'):
    path = os.path.join(root_path, f'{dir_name}_stft.npy')
    train_fea = np.load(path)
    samples_per_class = train_fea.shape[0] // class_num

    train_label = torch.zeros((train_fea.shape[0]))
    for i in range(train_fea.shape[0]):
        train_label[i] = i // samples_per_class

    print(train_fea.shape)
    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea).float()
    train_fea = train_fea.unsqueeze(1)

    if device == 'cuda' and torch.cuda.is_available():
        train_fea = train_fea.cuda()
        train_label = train_label.cuda()

    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
    return train_loader


def load_train_si(root_path, dir_name, fft1, class_num, batch_size, kwargs, device='cpu'):
    path = os.path.join(root_path, f'{dir_name}_stft.npy')
    train_fea = np.load(path)
    samples_per_class = train_fea.shape[0] // class_num

    train_label = torch.zeros((train_fea.shape[0]))
    for i in range(train_fea.shape[0]):
        train_label[i] = i // samples_per_class

    print(train_fea.shape)
    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea).float()
    train_fea = train_fea.unsqueeze(1)

    # Keep data on CPU; GPU transfer handled by DataLoader with pin_memory
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_source_training(root_path, dir1, dir2, dir3, src_list, fft1, class_num, batch_size, kwargs, device='cpu'):
    src_list = torch.from_numpy(src_list)

    path1 = os.path.join(root_path, f'{dir1}_stft.npy')
    train_fea_1 = np.load(path1)
    samples_per_class_1 = train_fea_1.shape[0] // class_num

    train_label_1 = torch.zeros((train_fea_1.shape[0], 2))
    for i in range(train_fea_1.shape[0]):
        train_label_1[i][0] = i // samples_per_class_1
        train_label_1[i][1] = 0

    path2 = os.path.join(root_path, f'{dir2}_stft.npy')
    train_fea_2 = np.load(path2)
    samples_per_class_2 = train_fea_2.shape[0] // class_num

    train_label_2 = torch.zeros((train_fea_2.shape[0], 2))
    for i in range(train_fea_2.shape[0]):
        train_label_2[i][0] = i // samples_per_class_2
        train_label_2[i][1] = 0

    path3 = os.path.join(root_path, f'{dir3}_stft.npy')
    train_fea_3 = np.load(path3)
    samples_per_class_3 = train_fea_3.shape[0] // class_num

    train_label_3 = torch.zeros((train_fea_3.shape[0], 2))
    for i in range(train_fea_3.shape[0]):
        train_label_3[i][0] = i // samples_per_class_3
        train_label_3[i][1] = 0

    train_fea = np.vstack((train_fea_1, train_fea_2, train_fea_3))
    train_label = torch.cat([train_label_1, train_label_2, train_label_3], dim=0)

    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea).float()
    train_fea = train_fea.unsqueeze(1)

    # Keep data on CPU; GPU transfer handled by DataLoader with pin_memory
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_target_training(AP, SNR, root_path, dir1, dir2, dir3, src_list, fft1, class_num, batch_size, kwargs, device='cpu'):
    src_list = torch.from_numpy(src_list)

    path1 = os.path.join(root_path, f'{dir1}_stft.npy')
    train_fea_1 = np.load(path1)
    samples_per_class_1 = train_fea_1.shape[0] // class_num

    train_label_1 = torch.zeros((train_fea_1.shape[0], 2))
    for i in range(train_fea_1.shape[0]):
        train_label_1[i][0] = i // samples_per_class_1
        train_label_1[i][1] = 1

    path2 = os.path.join(root_path, f'{dir2}_stft.npy')
    train_fea_2 = np.load(path2)
    samples_per_class_2 = train_fea_2.shape[0] // class_num

    train_label_2 = torch.zeros((train_fea_2.shape[0], 2))
    for i in range(train_fea_2.shape[0]):
        train_label_2[i][0] = i // samples_per_class_2
        train_label_2[i][1] = 1

    path3 = os.path.join(root_path, f'{dir3}_stft.npy')
    train_fea_3 = np.load(path3)
    samples_per_class_3 = train_fea_3.shape[0] // class_num

    train_label_3 = torch.zeros((train_fea_3.shape[0], 2))
    for i in range(train_fea_3.shape[0]):
        train_label_3[i][0] = i // samples_per_class_3
        train_label_3[i][1] = 1

    train_fea = np.vstack((train_fea_1, train_fea_2, train_fea_3))
    train_label = torch.cat([train_label_1, train_label_2, train_label_3], dim=0)

    train_label = train_label.long()
    train_fea = torch.from_numpy(train_fea).float()
    train_fea = train_fea.unsqueeze(1)

    # Keep data on CPU; GPU transfer handled by DataLoader with pin_memory
    data = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader
