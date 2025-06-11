
import os
import cv2
import time
import numpy as np
import fusion
import pycolmap
from skimage import measure
import matplotlib.pyplot as plt

# === PATHS ===
root_dir = "."
rgb_dir = os.path.join(root_dir, "frameData")
depth_dir = os.path.join(root_dir, "depth_maps")
colmap_sparse = os.path.join(root_dir, "colmap_output", "sparse", "0")
intrinsics_path = os.path.join(root_dir, "camera-intrinsics.txt")

# === SETTINGS ===
voxel_size = 0.02
max_depth = 3.0
depth_scale_factor = 0.8  # Adjust this value to bring depth closer

# === LOAD INTRINSICS ===
fx, fy, cx, cy = np.loadtxt(intrinsics_path)
cam_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# === LOAD COLMAP RECONSTRUCTION ===
print("[*] Loading COLMAP reconstruction...")
recon = pycolmap.Reconstruction(colmap_sparse)

# === VOLUME BOUNDS ESTIMATION ===
print("[*] Estimating voxel volume bounds...")
vol_bnds = np.zeros((3, 2))
for img in recon.images.values():
    depth_path = os.path.join(depth_dir, f"{os.path.splitext(img.name)[0]}_depth.png")
    if not os.path.exists(depth_path):
        continue

    raw_depth = cv2.imread(depth_path, -1).astype(np.float32)
    if raw_depth.max() > 1.0:
        raw_depth /= 255.0

    depth = raw_depth * depth_scale_factor
    depth = np.clip(depth, 0.2, max_depth)
    depth[depth == 0] = np.nan
    

    # === Apply scale to camera translation ===
    SCALE = 0.1  # try 0.1 to 0.5 until fusion works
    
    pose = np.eye(4)
    pose[:3, :3] = img.cam_from_world.rotation.matrix()
    pose[:3, 3] = img.cam_from_world.translation * SCALE
    

    frustum = fusion.get_view_frustum(depth, cam_intr, pose)
    vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.nanmin(frustum, axis=1))
    vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.nanmax(frustum, axis=1))

# === INITIALIZE TSDF VOLUME ===
print("[*] Initializing TSDF volume...")
tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size)
print("[*] Voxel bounds (meters):")
print("  X:", vol_bnds[0])
print("  Y:", vol_bnds[1])
print("  Z:", vol_bnds[2])


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

    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    raw_depth = cv2.imread(depth_path, -1).astype(np.float32)
    if raw_depth.max() > 1.0:
        raw_depth /= 255.0

    depth = raw_depth * depth_scale_factor
    depth = np.clip(depth, 0.2, max_depth)
    depth[depth == 0] = np.nan

    print(f"[Integrate] {frame_name} | depth min: {np.nanmin(depth):.3f}, max: {np.nanmax(depth):.3f}")
    if frame_name.endswith("00001.png"):
        plt.imshow(depth, cmap='plasma')
        plt.colorbar()
        plt.title("Depth Preview (after scaling)")
        plt.savefig("depth_preview.png")

    pose = np.eye(4)
    pose[:3, :3] = img.cam_from_world.rotation.matrix()
    pose[:3, 3] = img.cam_from_world.translation

    tsdf_vol.integrate(rgb, depth, cam_intr, pose, obs_weight=1.0)

fps = len(recon.images) / (time.time() - start)
print(f"[*] Fusion complete at {fps:.2f} FPS")

# === SAVE MESH AND POINT CLOUD ===
print("[*] Saving mesh to mesh.ply...")
tsdf = tsdf_vol._tsdf_vol_cpu
print(f"TSDF min: {np.min(tsdf)}, max: {np.max(tsdf)}, mean: {np.mean(tsdf)}")
print(f"Non-1.0 voxels: {(tsdf != 1.0).sum()}")

level = 0.5 * (np.min(tsdf) + np.max(tsdf))
verts, faces, norms, vals = measure.marching_cubes(tsdf, level=level)
fusion.meshwrite("mesh.ply", verts, faces, norms, None)

print("[*] Saving point cloud to pc.ply...")
pointcloud = tsdf_vol.get_point_cloud()
fusion.pcwrite("pc.ply", pointcloud)
