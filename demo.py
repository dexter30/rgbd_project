import os
import cv2
import glob
import time
import numpy as np
import fusion
import pycolmap
from skimage import measure
import matplotlib.pyplot as plt


# === PATHS ===
root_dir = "."  # adjust if needed
rgb_dir = os.path.join(root_dir, "frameData")
depth_dir = os.path.join(root_dir, "depth_maps")
colmap_sparse = os.path.join(root_dir, "colmap_output", "sparse", "0")
intrinsics_path = os.path.join(root_dir, "camera-intrinsics.txt")

# === SETTINGS ===
voxel_size = 0.02  # 2cm
depth_scale = 1.0  # assume already in meters
max_depth = 3.0

# === LOAD INTRINSICS ===
fx, fy, cx, cy = np.loadtxt(intrinsics_path)
cam_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# === LOAD COLMAP RECONSTRUCTION ===
print("[*] Loading COLMAP reconstruction...")
recon = pycolmap.Reconstruction(colmap_sparse)
frame_names = sorted([img.name for img in recon.images.values()])

# === VOLUME BOUNDS ESTIMATION ===
print("[*] Estimating voxel volume bounds...")
vol_bnds = np.zeros((3, 2))
for img in recon.images.values():
    depth_path = os.path.join(depth_dir, f"{os.path.splitext(img.name)[0]}_depth.png")
    if not os.path.exists(depth_path):
        continue

    # Load MiDaS depth (usually in 0–255)
    # Load and normalize MiDaS depth
    raw_depth = cv2.imread(depth_path, -1).astype(np.float32)
    if raw_depth.max() > 1.0:
        raw_depth /= 255.0  # Normalize if needed
    
    # Optional: Invert if using MiDaS inverse depth
    raw_depth = 1.0 / (raw_depth + 1e-6)
    
    # 💡 Scale down to bring depth closer
    depth = raw_depth * 0.6  # Try 0.6, 0.7, 0.8
    
    depth = np.clip(depth, 0.2, 2.5)
    depth[np.isnan(depth)] = 0.0
    
    
    
    pose = np.eye(4)
    pose[:3, :3] = img.cam_from_world.rotation.matrix()
    pose[:3, 3] = img.cam_from_world.translation

    frustum = fusion.get_view_frustum(depth, cam_intr, pose)
    vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.nanmin(frustum, axis=1))
    vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.nanmax(frustum, axis=1))

# === INITIALIZE TSDF VOLUME ===
print("[*] Initializing TSDF volume...")
tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size)

# === INTEGRATE FRAMES ===
print("[*] Fusing frames into TSDF volume...")
start = time.time()
for img in recon.images.values():
    frame_name = img.name
    rgb_path = os.path.join(rgb_dir, frame_name)
    depth_path = os.path.join(depth_dir, f"{os.path.splitext(frame_name)[0]}_depth.png")

    if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
        print(f"[!] Skipping missing frame: {frame_name}")
        continue

    # --- Load RGB ---
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    # --- Load and preprocess MiDaS depth ---
    raw_depth = cv2.imread(depth_path, -1).astype(np.float32)
    if raw_depth.max() > 1.0:
        raw_depth /= 255.0  # Normalize if 0–255

    # Invert if MiDaS depth is inverse depth (very common!)
    depth = 1.0 / (raw_depth + 1e-6)

    # Scale to metric (experiment with 1.5–3.0)
    depth *= 2.0

    # Clean up depth
    depth = np.clip(depth, 0.2, max_depth)
    depth[np.isnan(depth)] = 0.0

    print(f"[Integrate] {frame_name} | depth min: {depth.min():.3f}, max: {depth.max():.3f}")
    print(f"{depth_path}")

    # --- Build pose ---
    pose = np.eye(4)
    pose[:3, :3] = img.cam_from_world.rotation.matrix()
    pose[:3, 3] = img.cam_from_world.translation

    # --- Integrate into TSDF ---
    tsdf_vol.integrate(rgb, depth, cam_intr, pose, obs_weight=1.0)

fps = len(recon.images) / (time.time() - start)
print(f"[*] Fusion complete at {fps:.2f} FPS")

# === SAVE MESH AND POINT CLOUD ===
print("[*] Saving mesh to mesh.ply...")
min_val = np.nanmin(tsdf_vol._tsdf_vol_cpu)
max_val = np.nanmax(tsdf_vol._tsdf_vol_cpu)


level = 0.5 * (min_val + max_val)
print(f"Using fallback surface level: {level}")
tsdf = tsdf_vol._tsdf_vol_cpu
print(f"TSDF min: {np.min(tsdf)}, max: {np.max(tsdf)}, mean: {np.mean(tsdf)}")
print(f"Non-1.0 voxels: {(tsdf != 1.0).sum()}")

plt.imshow(depth, cmap='plasma')
plt.colorbar()
plt.title("Metric Depth after scaling")
plt.show()


verts, faces, norms, vals = measure.marching_cubes(tsdf_vol._tsdf_vol_cpu, level=level)
fusion.meshwrite("mesh.ply", verts, faces, norms, None)

print("[*] Saving point cloud to pc.ply...")
pointcloud = tsdf_vol.get_point_cloud()
fusion.pcwrite("pc.ply", pointcloud)
