!pip install faiss-cpu


%%writefile cluster_images.py
import os
import json
import random
import numpy as np
import faiss
import glob
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms


def parse_val_name(filename: str) -> str:
    """ Extracts "00000001" from "ILSVRC2012_val_00000001.jpg". """
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.replace("ILSVRC2012_val_", "")


def parse_train_name(filepath: str) -> str:
    """ Extracts "n01440764_10040" from "n01440764/n01440764_10040.jpg". """
    base = os.path.splitext(filepath)[0]
    return base.split("/")[-1]


def load_and_flatten(filepath: str) -> np.ndarray:
    """ Loads an image, resizes to 256x256, center-crops it to 224x224, and normalizes. """
    imagenet_transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(filepath).convert("RGB")
    img = imagenet_transform(img)
    return img.numpy().flatten()


def extract_class_from_xml(xml_path: str) -> str:
    """ Extracts the class ID from an ImageNet validation XML annotation file. """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find("object")
        if obj is not None:
            return obj.find("name").text  # e.g., "n01751748"
    except Exception as e:
        print(f"Warning: Could not parse {xml_path} - {e}")
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to the folder containing 1k subfolders of train images.")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to the folder containing 50k val images named ILSVRC2012_val_XXXXXX.jpg.")
    parser.add_argument("--val_xml_dir", type=str, required=True,
                        help="Path to the folder containing XML annotations for val images.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Where to save train_grouping_X.json and val_grouping.json.")
    parser.add_argument("--split_index", type=int, choices=range(5), required=True,
                        help="Which split of the training data to process (0 to 4).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for picking 10k centroid images.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_dir = args.train_dir
    val_dir = args.val_dir
    val_xml_dir = args.val_xml_dir
    output_dir = args.output_dir
    split_index = args.split_index
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) GATHER ALL VAL IMAGES
    # ---------------------------------------------------------------------
    val_paths = sorted(glob.glob(os.path.join(val_dir, "ILSVRC2012_val_*.JPEG")))
    print(f"Found {len(val_paths)} validation images.")

    # 2) RANDOMLY PICK 10k OF THEM AS CENTROIDS
    centroid_paths = random.sample(val_paths, 10000)
    centroid_set = set(centroid_paths)

    # 3) LOAD AND FLATTEN THE 10k CENTROID IMAGES
    centroid_vectors = []
    for i, cpath in tqdm(enumerate(centroid_paths), total=len(centroid_paths)):
        centroid_vectors.append(load_and_flatten(cpath))
        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1} centroid images.")
    centroid_vectors = np.stack(centroid_vectors, axis=0).astype(np.float32)

    # 4) BUILD A FAISS INDEX
    d = centroid_vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroid_vectors)

    # ---------------------------------------------------------------------
    # 5) PROCESS VALIDATION IMAGES (ALWAYS FULL)
    # ---------------------------------------------------------------------
    val_grouping = { str(i): [] for i in range(10000) }
    remaining_val_paths = [p for p in val_paths if p not in centroid_set]

    print(f"Remaining val images to cluster: {len(remaining_val_paths)}")

    batch_size = 64
    for start_idx in tqdm(range(0, len(remaining_val_paths), batch_size), total=(len(remaining_val_paths) // batch_size + 1)):
        batch_paths = remaining_val_paths[start_idx:start_idx + batch_size]
        batch_vecs = np.stack([load_and_flatten(p) for p in batch_paths], axis=0).astype(np.float32)
        distances, indices = index.search(batch_vecs, 1)

        for i, pth in enumerate(batch_paths):
            cluster_id = indices[i, 0]
            val_name = parse_val_name(pth)
            xml_path = os.path.join(val_xml_dir, f"ILSVRC2012_val_{val_name}.xml")
            class_id = extract_class_from_xml(xml_path)
            val_grouping[str(cluster_id)].append(f"{class_id}_{val_name}")

    # ---------------------------------------------------------------------
    # 6) PROCESS TRAINING IMAGES (ONLY 1/5 BASED ON SPLIT INDEX)
    # ---------------------------------------------------------------------
    train_grouping = { str(i): [] for i in range(10000) }
    train_paths = sorted(glob.glob(os.path.join(train_dir, "*/*.JPEG")))

    total_train = len(train_paths)
    chunk_size = total_train // 5
    start_idx = split_index * chunk_size
    end_idx = total_train if split_index == 4 else (split_index + 1) * chunk_size

    train_paths = train_paths[start_idx:end_idx]  # Assign 1/5th of data to this session
    print(f"Processing training images {start_idx} to {end_idx} ({len(train_paths)})")

    for start_idx in tqdm(range(0, len(train_paths), batch_size), total=(len(train_paths) // batch_size + 1)):
        batch_slice = train_paths[start_idx: start_idx + batch_size]
        batch_vecs = np.stack([load_and_flatten(p) for p in batch_slice], axis=0).astype(np.float32)
        distances, indices = index.search(batch_vecs, 1)

        for i, p in enumerate(batch_slice):
            cluster_id = indices[i, 0]
            train_name = parse_train_name(p)
            train_grouping[str(cluster_id)].append(train_name)

    # ---------------------------------------------------------------------
    # 7) SAVE JSON FILES
    # ---------------------------------------------------------------------
    val_json_path = os.path.join(output_dir, "val_grouping.json")
    train_json_path = os.path.join(output_dir, f"train_grouping_{split_index}.json")

    with open(val_json_path, "w") as f:
        json.dump(val_grouping, f)
    with open(train_json_path, "w") as f:
        json.dump(train_grouping, f)

    print(f"\nSaved val_grouping.json and train_grouping_{split_index}.json.")


if __name__ == "__main__":
    main()



!python3 cluster_images.py --train_dir "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train" \
                          --val_dir "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val" \
                          --val_xml_dir "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val" \
                          --output_dir "/kaggle/working" \
                          --split_index 0 \
                          --seed 2411

