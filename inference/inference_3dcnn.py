import numpy as np
import torch
import tifffile as tiff
import matplotlib.pyplot as plt
from model.CNN_3D import HSI_3DCNN

# =========================================
# CONFIG
# =========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 7
PAD = PATCH_SIZE // 2

IMAGE_ID = 1  # change for other images

# =========================================
# LOAD MODEL
# =========================================
model = HSI_3DCNN().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# =========================================
# LOAD IMAGE & LABEL
# =========================================
image = tiff.imread(f"data/images/201912_{IMAGE_ID}.tif")   # (H,W,32)
gt = tiff.imread(f"data/labels/201912_{IMAGE_ID}.tif")      # (H,W)

H, W, B = image.shape

# =========================================
# LOAD NORMALIZATION (IMPORTANT)
# =========================================
# 👉 You MUST save these during training
try:
    mean = np.load("mean.npy")
    std = np.load("std.npy")
except:
    print("⚠️ Using fallback normalization (NOT recommended)")
    mean = np.mean(image, axis=(0,1))
    std = np.std(image, axis=(0,1))

image = (image - mean) / (std + 1e-8)

# =========================================
# PAD IMAGE
# =========================================
padded = np.pad(image, ((PAD,PAD),(PAD,PAD),(0,0)), mode='reflect')

# =========================================
# INFERENCE
# =========================================
pred_map = np.zeros((H, W), dtype=np.uint8)

with torch.no_grad():
    for i in range(H):
        if i % 50 == 0:
            print(f"Processing row {i}/{H}")

        for j in range(W):

            patch = padded[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]

            patch = torch.tensor(patch, dtype=torch.float32)
            patch = patch.permute(2, 0, 1)   # (32,7,7)
            patch = patch.unsqueeze(0)       # (1,32,7,7)
            patch = patch.unsqueeze(0)       # (1,1,32,7,7)

            patch = patch.to(DEVICE)

            output = model(patch)
            pred = torch.argmax(output, dim=1).item()

            pred_map[i, j] = pred

print("✅ Inference completed!")

# =========================================
# SAVE PREDICTION
# =========================================
np.save("prediction.npy", pred_map)

# =========================================
# COLOR MAP
# =========================================
def get_color_map():
    return np.array([
        [0, 0, 0],        # background
        [255, 165, 0],    # building
        [0, 255, 0],      # farmland
        [255, 0, 0],      # forest
        [0, 0, 255],      # road
        [139, 69, 19],    # water
        [255, 192, 203],  # bare land
        [128, 128, 128],  # fish pond
    ])

def label_to_color(label_map):
    color_map = get_color_map()
    H, W = label_map.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(len(color_map)):
        color_img[label_map == i] = color_map[i]

    return color_img

# =========================================
# PREPARE VISUALIZATION
# =========================================

# Fix GT labels (1–7 → 0–6)
gt = gt - 1
gt[gt < 0] = 0

# RGB (pick 3 bands)
rgb = image[:, :, [10, 20, 30]]
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

# Color maps
gt_color = label_to_color(gt)
pred_color = label_to_color(pred_map)

# =========================================
# PLOT
# =========================================
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(rgb)
plt.title("Pseudo RGB")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(gt_color)
plt.title("Ground Truth")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(pred_color)
plt.title("CNN Prediction")
plt.axis('off')

plt.tight_layout()
plt.savefig("result.png", dpi=300)
plt.show()

print("✅ Visualization saved as result.png")