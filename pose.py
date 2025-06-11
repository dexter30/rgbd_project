import pycolmap
from pycolmap._core import IncrementalPipelineOptions
import os
import shutil

# Paths
image_dir = 'frameData'
output_dir = 'colmap_output'
sparse_dir = os.path.join(output_dir, 'sparse')
database_path = os.path.join(output_dir, 'database.db')

# Clean previous outputs
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# 1. Feature extraction
pycolmap.extract_features(database_path, image_dir)

# 2. Feature matching
pycolmap.match_exhaustive(database_path)

# 3. SfM with options
options = IncrementalPipelineOptions()
options.num_threads = 8
options.ba_refine_focal_length = True
options.ba_refine_principal_point = False

reconstructions = pycolmap.incremental_mapping(
    database_path=database_path,
    image_path=image_dir,
    output_path=sparse_dir,
    options=options
)
reconstruction = list(reconstructions.values())[0]

# Now you can safely access .images
for image_id, image in reconstruction.images.items():
    print(f'Image ID: {image_id}')
    print(f' - Camera center: {image.cam_from_world.translation}')
    print(f' - Rotation (quaternion): {image.cam_from_world.rotation}')


# Get camera from any image (all frames usually share one camera)
first_image = next(iter(reconstruction.images.values()))
camera = reconstruction.cameras[first_image.camera_id]

print(f"\nCamera model: {camera.model_name}")
print(f"Resolution: {camera.width} x {camera.height}")
print(f"Parameters: {camera.params}")

# Parse intrinsics
if camera.model_name == "PINHOLE":
    fx, fy, cx, cy = camera.params
elif camera.model_name == "SIMPLE_PINHOLE":
    f, cx, cy = camera.params
    fx = fy = f
else:
    raise ValueError(f"Unsupported camera model: {camera.model_name}")

# Save to file
with open("camera-intrinsics.txt", "w") as f:
    f.write(f"{fx} {fy} {cx} {cy}\n")

print(f"\nSaved intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
