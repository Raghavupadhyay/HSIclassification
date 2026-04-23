import torch
import torch.nn as nn
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import cohen_kappa_score

from dataset.dataloader import get_dataloaders
from model.CNN_3D import HSI_3DCNN

# =========================================
# CONFIG
# =========================================
EPOCHS = 50
LR = 5e-5
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 7
PAD = PATCH_SIZE // 2

# =========================================
# LOAD DATA
# =========================================
train_loader, test_loader = get_dataloaders(
    "data/images", "data/labels", batch_size=64
)

# =========================================
# CLASS WEIGHTS
# =========================================
all_labels = []
for _, y in train_loader:
    all_labels.extend(y.numpy())

counter = Counter(all_labels)

weights = torch.tensor([1.0 / counter[i] for i in range(7)], dtype=torch.float32)
weights = weights / weights.sum()
weights = weights.to(DEVICE)

# =========================================
# MODEL
# =========================================
model = HSI_3DCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================================
# TRAINING
# =========================================
best_loss = float('inf')
early_counter = 0

for epoch in range(EPOCHS):
    model.train()

    epoch_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _, pred = torch.max(outputs, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    epoch_loss /= len(train_loader)
    train_acc = 100 * correct / total

    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        early_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        early_counter += 1

    if early_counter >= PATIENCE:
        print("⛔ Early stopping triggered")
        break

# =========================================
# LOAD BEST MODEL
# =========================================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# =========================================
# EVALUATION
# =========================================
all_preds, all_labels = [], []
correct, total = 0, 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        outputs = model(x)
        _, pred = torch.max(outputs, 1)

        total += y.size(0)
        correct += (pred == y).sum().item()

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

OA = 100 * correct / total

num_classes = 7
class_correct = [0]*num_classes
class_total = [0]*num_classes

for t, p in zip(all_labels, all_preds):
    class_total[t] += 1
    if t == p:
        class_correct[t] += 1

class_acc = []

print("\n📊 Class-wise Accuracy:")
for i in range(num_classes):
    acc = 100 * class_correct[i] / class_total[i] if class_total[i] else 0
    class_acc.append(acc)
    print(f"Class {i}: {acc:.2f}%")

AA = sum(class_acc) / num_classes
kappa = cohen_kappa_score(all_labels, all_preds)

print("\n🔥 FINAL RESULTS:")
print(f"OA: {OA:.2f}% | AA: {AA:.2f}% | Kappa: {kappa:.4f}")

# =========================================
# INFERENCE
# =========================================
print("\n🚀 Running inference...")

IMAGE_ID = 1

image = tiff.imread(f"data/images/201912_{IMAGE_ID}.tif")
gt = tiff.imread(f"data/labels/201912_{IMAGE_ID}.tif")

H, W, B = image.shape

mean = train_loader.dataset.mean
std = train_loader.dataset.std

image = (image - mean) / (std + 1e-8)

padded = np.pad(image, ((PAD,PAD),(PAD,PAD),(0,0)), mode='reflect')

pred_map = np.zeros((H, W), dtype=np.uint8)

with torch.no_grad():
    for i in range(H):
        if i % 50 == 0:
            print(f"Inference row {i}/{H}")

        for j in range(W):
            patch = padded[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]

            patch = torch.tensor(patch, dtype=torch.float32)
            patch = patch.permute(2,0,1).unsqueeze(0).unsqueeze(0).to(DEVICE)

            output = model(patch)
            pred_map[i,j] = torch.argmax(output,1).item()

# =========================================
# VISUALIZATION
# =========================================
def label_to_color(label):
    colors = np.array([
        [0,0,0],[255,165,0],[0,255,0],[255,0,0],
        [0,0,255],[139,69,19],[255,192,203],[128,128,128]
    ])
    img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for i in range(len(colors)):
        img[label == i] = colors[i]
    return img

gt = gt - 1
gt[gt < 0] = 0

rgb = image[:,:, [10,20,30]]
rgb = (rgb - rgb.min())/(rgb.max()-rgb.min())

plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(rgb); plt.title("RGB"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(label_to_color(gt)); plt.title("GT"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(label_to_color(pred_map)); plt.title("Pred"); plt.axis('off')

plt.tight_layout()
plt.savefig("result.png", dpi=300)
plt.show()

print("✅ Done!")