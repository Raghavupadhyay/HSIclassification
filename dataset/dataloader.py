import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os

# =========================================
# CONFIG
# =========================================
PATCH_SIZE = 7
PAD = PATCH_SIZE // 2

# =========================================
# LOAD ALL IMAGES
# =========================================
def load_all_images(image_dir, label_dir):
    images, labels = [], []

    for i in range(1, 11):
        img = tiff.imread(os.path.join(image_dir, f"201912_{i}.tif"))
        lbl = tiff.imread(os.path.join(label_dir, f"201912_{i}.tif"))

        images.append(img)
        labels.append(lbl)

    return images, labels

# =========================================
# GET ALL VALID PIXELS
# =========================================
def get_all_indices(images, labels):
    all_indices, image_ids, all_labels = [], [], []

    for img_id in range(len(images)):
        lbl = labels[img_id]
        H, W = lbl.shape

        for i in range(H):
            for j in range(W):
                if lbl[i, j] != 0:
                    all_indices.append((i, j))
                    image_ids.append(img_id)
                    all_labels.append(lbl[i, j] - 1)

    return np.array(all_indices), np.array(image_ids), np.array(all_labels)

# =========================================
# SAMPLE (500 PER CLASS)
# =========================================
def sample_per_class(all_labels, samples_per_class=500):
    class_indices = defaultdict(list)

    for idx, label in enumerate(all_labels):
        class_indices[label].append(idx)

    train_idx, test_idx = [], []

    for cls, indices in class_indices.items():
        np.random.shuffle(indices)
        train_idx.extend(indices[:samples_per_class])
        test_idx.extend(indices[samples_per_class:])

    return train_idx, test_idx

# =========================================
# NORMALIZATION (TRAIN ONLY)
# =========================================
def compute_mean_std(images, all_indices, image_ids, train_idx):
    pixels = []

    for idx in train_idx:
        i, j = all_indices[idx]
        img_id = image_ids[idx]

        img = images[img_id]
        padded = np.pad(img, ((PAD,PAD),(PAD,PAD),(0,0)), mode='reflect')

        patch = padded[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]
        pixels.append(patch.reshape(-1, patch.shape[-1]))

    pixels = np.concatenate(pixels, axis=0)

    mean = np.mean(pixels, axis=0)
    std  = np.std(pixels, axis=0)

    return mean, std

# =========================================
# DATASET
# =========================================
class HSIDataset(Dataset):
    def __init__(self, images, labels, all_indices, image_ids, selected_idx, mean, std):
        self.images = images
        self.labels = labels
        self.all_indices = all_indices
        self.image_ids = image_ids
        self.selected_idx = selected_idx
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.selected_idx)

    def __getitem__(self, idx):
        real_idx = self.selected_idx[idx]

        i, j = self.all_indices[real_idx]
        img_id = self.image_ids[real_idx]

        image = self.images[img_id]
        label_map = self.labels[img_id]

        padded = np.pad(image, ((PAD,PAD),(PAD,PAD),(0,0)), mode='reflect')
        patch = padded[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]

        # ✅ NORMALIZATION (ONLY HERE)
        patch = (patch - self.mean) / (self.std + 1e-8)

        label = label_map[i, j] - 1

        patch = torch.tensor(patch, dtype=torch.float32)
        patch = patch.permute(2, 0, 1)   # (32,7,7)
        patch = patch.unsqueeze(0)       # (1,32,7,7)

        return patch, torch.tensor(label, dtype=torch.long)

# =========================================
# DATALOADER FUNCTION
# =========================================
def get_dataloaders(image_dir, label_dir, batch_size=64):

    images, labels = load_all_images(image_dir, label_dir)

    all_indices, image_ids, all_labels = get_all_indices(images, labels)

    train_idx, test_idx = sample_per_class(all_labels)

    mean, std = compute_mean_std(images, all_indices, image_ids, train_idx)

    train_dataset = HSIDataset(images, labels, all_indices, image_ids, train_idx, mean, std)
    test_dataset  = HSIDataset(images, labels, all_indices, image_ids, test_idx, mean, std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\n📊 Total samples:", len(all_indices))
    print("Train samples:", len(train_idx))
    print("Test samples:", len(test_idx))

    return train_loader, test_loader