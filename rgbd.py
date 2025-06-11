import os
import numpy as np
import open3d as o3d
import cv2
import pycolmap

# --- CONFIG ---
image_dir = "frameData"
depth_dir = "depth_maps"
sparse_dir = "colmap_output/sparse/0"
depth_scale = 1.0           # Adjust if your depth maps are scaled
depth_trunc = 3.0           # Max depth value (in meters)
voxel_length = 0.005        # 5mm voxels
sdf_trunc = 0.04            # Truncation for TSDF
color_map = o3d.geometry.TriangleMesh.create_coordinate_frame()

# --- Load Reconstruction ---
recon = pycolmap.Reconstruction(sparse_dir)

# --- Initialize Volume ---
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_length,
    sdf_trunc=sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

# --- Loop Through Images ---
for image in recon.images.values():
    img_name = image.name
    print(f"[+] Processing {img_name}")

    # Load camera and image data
    cam = recon.cameras[image.camera_id]
    rgb = cv2.imread(os.path.join(image_dir, img_name))
    depth_path = os.path.join(depth_dir, f"{os.path.splitext(img_name)[0]}_depth.png")
    if not os.path.exists(depth_path):
        print(f"[!] Missing depth: {depth_path}")
        continue
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0  # normalize

    # Get camera intrinsics
    width, height = cam.width, cam.height
    if cam.model_name == "PINHOLE":
        fx, fy, cx, cy = cam.params
    elif cam.model_name == "SIMPLE_PINHOLE":
        fx = fy = cam.params[0]
        cx, cy = cam.params[1], cam.params[2]
    else:
        print(f"[!] Unsupported camera model: {cam.model_name}")
        continue

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Prepare images
    rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image((depth * depth_scale).astype(np.float32))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1.0 / depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
    )

    # Convert COLMAP pose to Open3D (extrinsic matrix)
    R = image.cam_from_world.rotation.matrix()
    t = image.cam_from_world.translation
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    # Open3D expects world_to_camera, so we invert cam_from_world
    extrinsic = np.linalg.inv(extrinsic)

    # Integrate RGB-D into TSDF volume
    volume.integrate(rgbd, intrinsic, extrinsic)

# --- Extract Mesh ---
print("[*] Extracting mesh...")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

# --- Save / Show Mesh ---
o3d.io.write_triangle_mesh("output_mesh.ply", mesh)
o3d.visualization.draw_geometries([mesh])
