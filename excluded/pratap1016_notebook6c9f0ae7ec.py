from PIL import Image
import os

# === Compression parameters for each app ===
compression_profiles = {
    "whatsapp": {"max_dim": 1280, "quality": 72},
    "telegram": {"max_dim": 1280, "quality": 80},
    "instagram": {"max_dim": 1080, "quality": 85},
    "facebook": {"max_dim": 2048, "quality": 70}
}

def compress_image(input_path, output_path, max_dim, quality):
    img = Image.open(input_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Resize while maintaining aspect ratio
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    # Save compressed version
    img.save(
        output_path,
        "JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling=2
    )

def process_dataset(input_root, output_root):
    for app, params in compression_profiles.items():
        print(f"\nðŸ”¹ Processing for {app}...")
        for cam_folder in os.listdir(input_root):
            cam_path = os.path.join(input_root, cam_folder)
            if not os.path.isdir(cam_path):
                continue

            save_dir = os.path.join(output_root, app, cam_folder)
            os.makedirs(save_dir, exist_ok=True)

            for img_file in os.listdir(cam_path):
                in_path = os.path.join(cam_path, img_file)
                out_path = os.path.join(save_dir, img_file)

                try:
                    compress_image(
                        in_path, out_path,
                        max_dim=params["max_dim"],
                        quality=params["quality"]
                    )
                except Exception as e:
                    print(f" Error with {img_file}: {e}")

# === Update these paths for Kaggle ===
input_root = "../input/sp-society-camera-model-identification/train/train"
output_root = "/kaggle/working/compressed"

process_dataset(input_root, output_root)

print("\nâœ… All compressions done! Check /kaggle/working/compressed/")



from PIL import Image
import os

# Compression parameters for each app
compression_profiles = {
    "whatsapp": {"max_dim": 1280, "quality": 72},
    "telegram": {"max_dim": 1280, "quality": 80},
    "instagram": {"max_dim": 1080, "quality": 85},
    "facebook": {"max_dim": 2048, "quality": 70}
}

def compress_image(input_path, output_path, max_dim, quality):
    img = Image.open(input_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    img.save(output_path, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=2)

def process_test_dataset(input_root, output_root):
    # Loop through each app compression
    for app, params in compression_profiles.items():
        print(f"\nðŸ”¹ Processing test images for {app}...")
        save_dir = os.path.join(output_root, app)
        os.makedirs(save_dir, exist_ok=True)

        for img_file in os.listdir(input_root):
            in_path = os.path.join(input_root, img_file)
            out_path = os.path.join(save_dir, img_file)
            try:
                compress_image(
                    in_path, out_path,
                    max_dim=params["max_dim"],
                    quality=params["quality"]
                )
            except Exception as e:
                print(f" Error with {img_file}: {e}")

# === Update these paths ===
test_input_root = "../input/sp-society-camera-model-identification/test/test"
test_output_root = "/kaggle/working/compressed_test"

process_test_dataset(test_input_root, test_output_root)

print("\nâœ… Test image compressions done! Check /kaggle/working/compressed_test/")


