import os
import shutil
import random

def extract_and_split_imagenet(
    src_root: str,
    mapping_txt: str,
    dest_root: str,
    num_classes: int,
    train_ratio: float = 0.75,
    seed: int = None,
) -> None:
    """
    Randomly pick `num_classes` sub-directories from `src_root`, rename them using
    the first label in `mapping_txt`, and split their images into train/val.

    If `seed` is provided, the sampling & shuffle will be repeatable; otherwise,
    each call will be different.
    """
    # 1) Load mapping: id → first label before any comma
    id2label = {}
    with open(mapping_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            synset_id = parts[0]
            labels = " ".join(parts[1:])
            primary = labels.split(',')[0].strip().replace(' ', '_')
            id2label[synset_id] = primary

    # 2) List all class-dirs in src_root
    all_ids = [d for d in os.listdir(src_root)
               if os.path.isdir(os.path.join(src_root, d))]
    if num_classes > len(all_ids):
        raise ValueError(f"Requested {num_classes} classes, but only found {len(all_ids)}")

    # 3) Seed global RNG if requested
    if seed is not None:
        random.seed(seed)

    # 4) Sample random IDs using the global random
    selected = random.sample(all_ids, num_classes)

    # 5) Prepare destination train/val roots
    train_root = os.path.join(dest_root, 'train')
    val_root   = os.path.join(dest_root, 'val')
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root,   exist_ok=True)

    # 6) Process each selected class
    for syn_id in selected:
        if syn_id not in id2label:
            raise KeyError(f"Synset ID '{syn_id}' not found in mapping file!")
        label = id2label[syn_id]

        # Create label-named subdirs
        trgt_train = os.path.join(train_root, label)
        trgt_val   = os.path.join(val_root,   label)
        os.makedirs(trgt_train, exist_ok=True)
        os.makedirs(trgt_val,   exist_ok=True)

        # Gather & shuffle images
        src_dir = os.path.join(src_root, syn_id)
        imgs = [f for f in os.listdir(src_dir)
                if os.path.isfile(os.path.join(src_dir, f))]
        random.shuffle(imgs)

        # Split indices
        cut = int(len(imgs) * train_ratio)
        train_imgs = imgs[:cut]
        val_imgs   = imgs[cut:]

        # Copy files
        for fn in train_imgs:
            shutil.copy2(os.path.join(src_dir, fn),
                         os.path.join(trgt_train, fn))
        for fn in val_imgs:
            shutil.copy2(os.path.join(src_dir, fn),
                         os.path.join(trgt_val, fn))

        print(f"  • {syn_id} → '{label}' ({len(train_imgs)} train, {len(val_imgs)} val)")

    print(f"\nDone! {num_classes} classes split into:\n"
          f"  {train_root}/\n  {val_root}/")



extract_and_split_imagenet(
    src_root='/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
    mapping_txt='/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt',
    dest_root='/kaggle/working/imagenet_sample',
    num_classes=10,
    seed=None,
    train_ratio = 0.8
)



import shutil
import os

def zip_train_val(dest_root: str, zip_root: str = None):
    """
    Create ZIP archives of the `train` and `val` folders under dest_root.

    Args:
        dest_root (str): path where 'train' and 'val' live.
        zip_root  (str): directory where .zip files should be placed.
                         Defaults to dest_root itself.
    """
    if zip_root is None:
        zip_root = dest_root

    for split in ('train', 'val'):
        folder = os.path.join(dest_root, split)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Expected directory not found: {folder}")

        # make_archive will append .zip for you
        archive_name = os.path.join(zip_root, split)
        print(f"Zipping {folder} → {archive_name}.zip …")
        shutil.make_archive(archive_name, 'zip', root_dir=dest_root, base_dir=split)

    print(f"\n✅ Zipped 'train' and 'val' into:\n  {zip_root}/train.zip\n  {zip_root}/val.zip")
zip_train_val(dest_root='/kaggle/working/imagenet_sample')




