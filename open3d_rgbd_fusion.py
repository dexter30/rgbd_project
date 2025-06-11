
import os
import cv2
import numpy as np
import open3d as o3d
import pycolmap

# === PATHS ===
root_dir = "."
rgb_dir = os.path.join(root_dir, "frameData")
depth_dir = os.path.join(root_dir, "depth_maps")
colmap_sparse = os.path.join(root_dir, "colmap_output", "sparse", "0")
intrinsics_path = os.path.join(root_dir, "camera-intrinsics.txt")

# === PARAMETERS ===
depth_scale_factor = 0.8
depth_trunc = 3.0
voxel_length = 0.005
sdf_trunc = 0.04
tsdf_color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8

# === LOAD INTRINSICS ===
fx, fy, cx, cy = np.loadtxt(intrinsics_path)
width = 480
height = 480
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# === LOAD COLMAP RECONSTRUCTION ===
print("[*] Loading COLMAP reconstruction...")
recon = pycolmap.Reconstruction(colmap_sparse)

# === INITIALIZE TSDF VOLUME ===
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_length,
    sdf_trunc=sdf_trunc,
    color_type=tsdf_color_type,
)

# === INTEGRATE FRAMES ===
for img in recon.images.values():
    name = img.name
    rgb_path = os.path.join(rgb_dir, name)
    depth_path = os.path.join(depth_dir, f"{os.path.splitext(name)[0]}_depth.png")

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"[!] Skipping missing frame: {name}")
        continue

    color = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    raw_depth = cv2.imread(depth_path, -1).astype(np.float32)

    if raw_depth.max() > 1.0:
        raw_depth /= 255.0

    depth = raw_depth * depth_scale_factor
    depth = np.clip(depth, 0.2, depth_trunc)
    depth[np.isnan(depth)] = 0.0

    if depth.shape != color.shape[:2]:
        depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_LINEAR)

    color_o3d = o3d.geometry.Image(color)
    depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))  # in mm

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
    )

    pose = np.eye(4)
    pose[:3, :3] = img.cam_from_world.rotation.matrix()
    pose[:3, 3] = img.cam_from_world.translation * 0.2  # adjust scale to align with depth

    volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

# === EXTRACT MESH ===
print("[*] Extracting mesh...")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("fused_mesh.ply", mesh)
print("[*] Mesh saved to 'fused_mesh.ply'")

# === EXTRACT POINT CLOUD ===
print("[*] Extracting point cloud...")
pcd = volume.extract_point_cloud()
o3d.io.write_point_cloud("fused_pc.ply", pcd)
print("[*] Point cloud saved to 'fused_pc.ply'")
