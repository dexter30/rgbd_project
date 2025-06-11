import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
input_dir = "frameData"               # Folder with input images
output_dir = "depth_maps"          # Where to save depth maps
model_type = "DPT_Large"           # or "DPT_Hybrid", "MiDaS_small"
resize_output = False              # Set to True to save the depth map in original image size
image_extensions = ['.jpg', '.png', '.jpeg']

# --- SETUP MODEL ---
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform

# --- CREATE OUTPUT DIR ---
os.makedirs(output_dir, exist_ok=True)

# --- PROCESS IMAGES ---
for filename in tqdm(os.listdir(input_dir)):
    if not any(filename.lower().endswith(ext) for ext in image_extensions):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_depth.png")

    # Read and transform image
    img = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_tensor)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2] if resize_output else prediction.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize to 8-bit image
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_vis = 255 * (depth_map - depth_min) / (depth_max - depth_min)
    depth_vis = depth_vis.astype(np.uint8)

    # Save as grayscale PNG
    cv2.imwrite(output_path, depth_vis)

print(f"✅ Depth maps saved to '{output_dir}'")
