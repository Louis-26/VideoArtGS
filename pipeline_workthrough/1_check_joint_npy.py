import numpy as np

# Load the NPY file shown in your screenshot
npy_path = "../outputs/videoartgs/sapien/168/final/train/ours_20000/joint_value.npy"
data = np.load(npy_path)

# If the shape is something like [100, 3, 1], squeeze it into [100, 3]
if data.ndim == 3:
    data = data.squeeze(-1)

print("=" * 60)
print(" 📂 Inspecting the contents of joint_value.npy")
print("=" * 60)
print(
    f"└── Data shape: {data.shape}  "
    "(typically [number of frames, number of joints])"
)
print(f"└── Data type: {data.dtype}")
print("-" * 60)

# Print the joint values for the third frame
print(data[2])

# Calculate the motion range of each joint
num_joints = data.shape[1]

for j in range(num_joints):
    j_data = data[:, j]
    min_val, max_val = j_data.min(), j_data.max()

    # Convert radians to degrees for easier interpretation
    print(
        f"Joint {j} motion range: "
        f"{min_val:.4f} to {max_val:.4f} radians "
        f"(approximately {np.degrees(min_val):.1f}° "
        f"to {np.degrees(max_val):.1f}°)"
    )

print("=" * 60)