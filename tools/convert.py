from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img_path = r"E:\dataset\VOS-Endovis17\VOS-Endovis17\train\JPEGImages\seq_1\00120.png"
mask_path = r"E:\dataset\VOS-Endovis17\VOS-Endovis17\train\Annotations\seq_1\00120.png"

# load image
img = Image.open(img_path).convert("RGB")
img_np = np.array(img)

# load mask
mask = Image.open(mask_path)
mask_np = np.array(mask)

print("original image shape:", img_np.shape)
print("mask shape:", mask_np.shape)

target_h, target_w = mask_np.shape[:2]

H, W = img_np.shape[:2]
start_y = (H - target_h) // 2
start_x = (W - target_w) // 2
end_y = start_y + target_h
end_x = start_x + target_w

cropped_img = img_np[start_y:end_y, start_x:end_x]

print("center crop box:", (start_x, start_y, end_x, end_y))
print("cropped image shape:", cropped_img.shape)

# overlay
overlay = cropped_img.copy()
alpha = 0.45
red = np.array([255, 0, 0], dtype=np.uint8)

fg = mask_np > 0
overlay[fg] = ((1 - alpha) * overlay[fg] + alpha * red).astype(np.uint8)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(cropped_img)
plt.title("Center-cropped image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_np, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Overlay")
plt.axis("off")

plt.show()