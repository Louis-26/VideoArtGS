import numpy as np
import os

# Path to your NPZ file
file_path = "../data/videoartgs/sapien/168/filtered.npz"

print("=" * 50)
print("🔍 Starting inspection of the 3D trajectory file (filtered.npz)")
print("=" * 50)

# 1. Check whether the file exists to avoid path-related issues
if not os.path.exists(file_path):
    print(f"❌ Error: File not found: '{file_path}'")
    print(f"💡 Current working directory: {os.getcwd()}")
    print(
        "👉 Please make sure you are running the script from the correct "
        "directory, or use an absolute file path."
    )
else:
    print(f"✅ File found successfully: {file_path}\n")

    # 2. Load the data
    data = np.load(file_path)
    keys = list(data.keys())
    print(f"📦 Arrays contained in the file: {keys}\n")

    # 3. Inspect the contents of each array
    for key in keys:
        arr = data[key]

        print(f"--- Array: [{key}] ---")
        print(f"📐 Shape: {arr.shape}")
        print(f"🔤 Data type: {arr.dtype}")

        # 4. Print only a small sample to avoid flooding the terminal
        if key == "coords":
            # Print the XYZ coordinates of the first two points in frame 0
            print(
                "👁️  Data sample "
                "(frame 0, XYZ coordinates of the first two points):"
            )
            print(arr[0, :2, :])

        elif key == "visibs":
            # Print the visibility values of the first five points in frame 0
            print(
                "👁️  Data sample "
                "(frame 0, visibility of the first five points):"
            )
            print(arr[0, :5])

        else:
            # Fallback for any additional keys that may be added in the future
            print(f"👁️  Data sample:\n{arr.flatten()[:5]}")

        print("")

print("✅ Inspection completed!")