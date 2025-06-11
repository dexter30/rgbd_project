import pycolmap
import numpy as np

# Load reconstruction
recon = pycolmap.Reconstruction('colmap_output/sparse/0')

first_image = next(iter(recon.images.values()))
camera = recon.cameras[first_image.camera_id]

# Iterate over images and extract camera pose
for image in recon.images.values():
    name = image.name

    # Rotation and translation (world to camera)
    R = image.cam_from_world.rotation.matrix()      # 3×3 rotation matrix
    t = image.cam_from_world.translation            # 3×1 translation vector

    # Compute camera center in world coordinates
    C = -R.T @ t

    print(f"Image: {name}")
    print(f" - Camera center (world coordinates): {C}")
    print(f" - Rotation matrix:\n{R}")

params = camera.params
model = camera.model.name if hasattr(camera.model, "name") else str(camera.model)

print("\nCamera intrinsics:")
print(f" - Model: {model}")
print(f" - Resolution: {camera.width} x {camera.height}")
print(f" - Parameters: {params}")

if model == "PINHOLE":
    fx, fy, cx, cy = params
elif model == "SIMPLE_PINHOLE":
    f, cx, cy = params
    fx = fy = f
elif model == "SIMPLE_RADIAL":
    f, cx, cy, k1 = params  # ignore k1 for now
    fx = fy = f
else:
    raise ValueError(f"Unsupported camera model: {model}")

# Build intrinsics matrix
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])
print("\nIntrinsic matrix K:\n", K)

# Save for TSDF fusion
np.savetxt("camera-intrinsics.txt", [fx, fy, cx, cy])
print("\nSaved intrinsics to camera-intrinsics.txt")
