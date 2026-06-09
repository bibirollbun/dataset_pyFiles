import os
from PIL import Image  # this will now use Pillow-SIMD
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Input and output directories
input_dir = "/kaggle/input/grand-xray-slam-division-a/train1"
output_dir = "/kaggle/working/train1_resized"

os.makedirs(output_dir, exist_ok=True)

# Collect image files
img_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

def resize_and_save(fname):
    try:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        # Open with Pillow-SIMD and resize
        img = Image.open(in_path).convert("RGB")
        img = img.resize((600, 600), Image.BILINEAR)

        # Save as JPEG (quality=95 = good tradeoff)
        img.save(out_path, "JPEG", quality=95, optimize=True)
    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

# Use all available CPU cores for speed
with Pool(processes=cpu_count()) as pool:
    list(tqdm(pool.imap_unordered(resize_and_save, img_files), total=len(img_files)))

print("✅ All images resized to 600×600 and saved in:", output_dir)


# Input and output directories
input_dir = "/kaggle/input/grand-xray-slam-division-a/test1"
output_dir = "/kaggle/working/test1_resized"

os.makedirs(output_dir, exist_ok=True)

# Collect image files
img_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

def resize_and_save(fname):
    try:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        # Open with Pillow-SIMD and resize
        img = Image.open(in_path).convert("RGB")
        img = img.resize((600, 600), Image.BILINEAR)

        # Save as JPEG (quality=95 = good tradeoff)
        img.save(out_path, "JPEG", quality=95, optimize=True)
    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

# Use all available CPU cores for speed
with Pool(processes=cpu_count()) as pool:
    list(tqdm(pool.imap_unordered(resize_and_save, img_files), total=len(img_files)))

print("✅ All images resized to 600×600 and saved in:", output_dir)

