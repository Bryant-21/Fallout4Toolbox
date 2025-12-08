import numpy as np
from PIL import Image
import subprocess
import os

# --- SETTINGS ---
SIZE = 4096
SAVE_AS_DDS = True  # <-- Toggle this (True = convert to DDS, False = PNG only)
OUTPUT_NAME = "debug_grey"
# ----------------

# Create blank array
img = np.zeros((SIZE, SIZE), dtype=np.uint8)

# Top half: vertical gradient (columns)
for x in range(SIZE):
    value = int((x / (SIZE - 1)) * 255)
    img[0:SIZE // 2, x] = value

# Bottom half: horizontal gradient (rows)
for y in range(SIZE // 2, SIZE):
    value = int(((y - SIZE // 2) / (SIZE // 2 - 1)) * 255)
    img[y, :] = value

# Save temporarily as PNG
png_path = f"{OUTPUT_NAME}.png"
Image.fromarray(img, mode="L").save(png_path)
print(f"Saved PNG: {png_path}")

# Convert to DDS (requires texconv.exe available in PATH)
if SAVE_AS_DDS:
    dds_path = f"{OUTPUT_NAME}.dds"
    print("Converting to DDS...")
    subprocess.run([
        "texconv.exe",
        "-f", "BC7_UNORM",
        "-o", ".",
        png_path
    ], check=True)

    # texconv outputs as OUTPUT_NAME.DDS inside working directory
    # Clean output name
    generated_dds = OUTPUT_NAME.upper() + ".DDS"
    if os.path.exists(generated_dds):
        os.rename(generated_dds, dds_path)
        print(f"Saved DDS: {dds_path}")
    else:
        print("texconv did not produce expected file — check output log.")
