import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. é…�ç½®å�‚æ•° =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MODEL_NAMES = ["alexnet", "vgg16", "resnet50"]


def parse_xml_annotation(xml_path):
    """è§£æ��XMLæ ‡æ³¨æ–‡ä»¶ï¼Œè�·å�–ç±»åˆ«ID"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        class_id = root.find("object").find("name").text
        return class_id
    except Exception as e:
        print(f"âš ï¸� è§£æ��XMLå¤±è´¥ï¼š{xml_path}ï¼Œé”™è¯¯ï¼š{e}")
        return None


def get_image_ids_from_imageset(imageset_path, split="train"):
    """ä»�txtæ–‡ä»¶è¯»å�–å›¾åƒ�ID - ä¿®å¤�Kaggleæ ¼å¼�è§£æ��ï¼ˆç¬¬äºŒåˆ—æ˜¯åº�å�·ï¼Œä¸�æ˜¯æ ‡è®°ï¼‰"""
    image_ids = []
    if not imageset_path.exists():
        print(f"â�Œ å›¾åƒ�IDæ–‡ä»¶ä¸�å­˜åœ¨ï¼š{imageset_path}")
        return image_ids
    
    print(f"ğŸ“– è¯»å�–{split}é›†å›¾åƒ�IDæ–‡ä»¶ï¼š{imageset_path}")
    with open(imageset_path, "r") as f:
        lines = f.readlines()
        print(f"  - å…±{len(lines)}è¡Œæ•°æ�®")
        
        for idx, line in enumerate(lines[:5]):  # æ‰“å�°å‰�5è¡Œç¤ºä¾‹
            print(f"  - ç¤ºä¾‹è¡Œ{idx+1}ï¼š{line.strip()}")
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            
            # å…³é”®ä¿®å¤�ï¼šKaggleæ ¼å¼�ä¸­ï¼Œç¬¬ä¸€åˆ—æ˜¯å›¾åƒ�IDï¼Œç¬¬äºŒåˆ—æ˜¯åº�å�·ï¼ˆå¿½ç•¥ï¼‰
            img_id_with_path = parts[0]
            image_ids.append(img_id_with_path)
    
    print(f"âœ… æˆ�åŠŸè¯»å�–{split}é›†å›¾åƒ�IDæ•°é‡�ï¼š{len(image_ids)}")
    return image_ids


# ====================== 2. æ•°æ�®é¢„å¤„ç�† =======================
def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


class ILSVRC_Kaggle_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_ids_with_path, data_root, anno_root, split="train", transform=None):
        self.data_root = Path(data_root)
        self.anno_root = Path(anno_root)
        self.split = split
        self.transform = transform
        self.valid_image_ids = []  # å­˜å‚¨(å›¾åƒ�è·¯å¾„, æ ‡æ³¨è·¯å¾„, ç±»åˆ«ID)
        self.class_to_idx = {}

        # éªŒè¯�æ ¹è·¯å¾„æ˜¯å�¦å­˜åœ¨
        print(f"\nğŸ“� {split}é›†è·¯å¾„éªŒè¯�ï¼š")
        print(f"  - å›¾åƒ�æ ¹è·¯å¾„ï¼š{self.data_root} {'âœ… å­˜åœ¨' if self.data_root.exists() else 'â�Œ ä¸�å­˜åœ¨'}")
        print(f"  - æ ‡æ³¨æ ¹è·¯å¾„ï¼š{self.anno_root} {'âœ… å­˜åœ¨' if self.anno_root.exists() else 'â�Œ ä¸�å­˜åœ¨'}")
        
        if not self.data_root.exists() or not self.anno_root.exists():
            raise FileNotFoundError(f"{split}é›†å›¾åƒ�æˆ–æ ‡æ³¨æ ¹è·¯å¾„ä¸�å­˜åœ¨")

        # æ”¶é›†æ‰€æœ‰ç±»åˆ«ï¼ˆä»�æ ‡æ³¨æ–‡ä»¶ä¸­æ��å�–ï¼‰
        print(f"\nğŸ”� æ­£åœ¨æ‰«æ��{split}é›†æ ‡æ³¨æ–‡ä»¶ï¼Œæ��å�–ç±»åˆ«ä¿¡æ�¯...")
        all_class_ids = set()
        xml_files = list(self.anno_root.glob("**/*.xml"))
        print(f"  - æ ‡æ³¨æ–‡ä»¶å¤¹ä¸‹å…±æœ‰{len(xml_files)}ä¸ªæ ‡æ³¨æ–‡ä»¶")
        
        for xml_file in xml_files[:10]:  # æ‰“å�°å‰�10ä¸ªæ ‡æ³¨æ–‡ä»¶ç¤ºä¾‹
            class_id = parse_xml_annotation(xml_file)
            if class_id:
                all_class_ids.add(class_id)
        print(f"  - å·²å�‘ç�°ç±»åˆ«æ•°ï¼š{len(all_class_ids)}ï¼ˆç¤ºä¾‹ï¼š{list(all_class_ids)[:5]}ï¼‰")

        # æ�„å»ºç±»åˆ«æ˜ å°„
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_class_ids))}
        self.num_classes = len(self.class_to_idx)
        print(f"  - ç±»åˆ«æ˜ å°„æ�„å»ºå®Œæˆ�ï¼š{self.num_classes}ä¸ªç±»åˆ«")

        # é��å�†æ‰€æœ‰å›¾åƒ�IDï¼ŒéªŒè¯�æ–‡ä»¶å­˜åœ¨æ€§
        print(f"\nğŸ”� æ­£åœ¨éªŒè¯�{split}é›†æ ·æœ¬ï¼ˆå‰�10ä¸ªï¼‰ï¼š")
        valid_count = 0
        for idx, img_id_with_path in enumerate(image_ids_with_path):
            try:
                if self.split == "train":
                    # è®­ç»ƒé›†å›¾åƒ�è·¯å¾„ï¼šdata_root / ç±»åˆ«æ–‡ä»¶å¤¹ / å›¾åƒ�ID.JPEG
                    # ä¾‹å¦‚ï¼šn01440764/n01440764_10026 â†’ data_root/n01440764/n01440764_10026.JPEG
                    img_path = self.data_root / img_id_with_path
                    
                    # å°�è¯•ä¸�å�Œçš„å›¾åƒ�å��ç¼€
                    img_suffixes = [".JPEG", ".jpg", ".png", ".jpeg"]
                    found_img_path = None
                    for suffix in img_suffixes:
                        test_path = img_path.with_suffix(suffix)
                        if test_path.exists():
                            found_img_path = test_path
                            break
                    
                    if not found_img_path:
                        if idx < 10:
                            print(f"  - â�Œ å›¾åƒ�ä¸�å­˜åœ¨ï¼š{img_path}ï¼ˆå°�è¯•äº†å¤šç§�å��ç¼€ï¼‰")
                        continue
                    
                    # è®­ç»ƒé›†æ ‡æ³¨è·¯å¾„ï¼šanno_root / å›¾åƒ�ID.xmlï¼ˆå…³é”®ä¿®å¤�ï¼šæ²¡æœ‰ç±»åˆ«å­�æ–‡ä»¶å¤¹ï¼�ï¼‰
                    xml_filename = f"{os.path.basename(img_id_with_path)}.xml"
                    xml_path = self.anno_root / xml_filename
                    
                    if not xml_path.exists():
                        # å¤‡ç”¨è·¯å¾„ï¼šå°�è¯•å¸¦ç±»åˆ«æ–‡ä»¶å¤¹çš„æ ‡æ³¨è·¯å¾„
                        xml_path = self.anno_root / img_id_with_path.replace("/", ".") + ".xml"
                        if not xml_path.exists():
                            if idx < 10:
                                print(f"  - â�Œ æ ‡æ³¨ä¸�å­˜åœ¨ï¼š{xml_path}")
                            continue

                else:  # val
                    # éªŒè¯�é›†å›¾åƒ�è·¯å¾„ï¼šdata_root / å›¾åƒ�ID.JPEG
                    img_id = img_id_with_path
                    img_path = self.data_root / img_id
                    
                    # å°�è¯•ä¸�å�Œçš„å›¾åƒ�å��ç¼€
                    img_suffixes = [".JPEG", ".jpg", ".png", ".jpeg"]
                    found_img_path = None
                    for suffix in img_suffixes:
                        test_path = img_path.with_suffix(suffix)
                        if test_path.exists():
                            found_img_path = test_path
                            break
                    
                    if not found_img_path:
                        if idx < 10:
                            print(f"  - â�Œ å›¾åƒ�ä¸�å­˜åœ¨ï¼š{img_path}ï¼ˆå°�è¯•äº†å¤šç§�å��ç¼€ï¼‰")
                        continue
                    
                    # éªŒè¯�é›†æ ‡æ³¨è·¯å¾„ï¼šanno_root / å›¾åƒ�ID.xml
                    xml_path = self.anno_root / f"{img_id}.xml"
                    if not xml_path.exists():
                        if idx < 10:
                            print(f"  - â�Œ æ ‡æ³¨ä¸�å­˜åœ¨ï¼š{xml_path}")
                        continue

                # è§£æ��ç±»åˆ«ID
                class_id = parse_xml_annotation(xml_path)
                if class_id is None or class_id not in self.class_to_idx:
                    if idx < 10:
                        print(f"  - âš ï¸� ç±»åˆ«æ— æ•ˆï¼š{class_id}")
                    continue

                # å­˜å‚¨æœ‰æ•ˆæ ·æœ¬ä¿¡æ�¯
                self.valid_image_ids.append((found_img_path, xml_path, class_id))
                valid_count += 1
                
                # æ‰“å�°å‰�10ä¸ªæœ‰æ•ˆæ ·æœ¬
                if idx < 10 and valid_count <= 10:
                    print(f"  - âœ… æœ‰æ•ˆæ ·æœ¬ï¼š{found_img_path.name} | ç±»åˆ«ï¼š{class_id}")

            except Exception as e:
                if idx < 10:
                    print(f"  - âš ï¸� å¤„ç�†å¤±è´¥ï¼š{img_id_with_path}ï¼Œé”™è¯¯ï¼š{str(e)[:50]}")
                continue

        # æ£€æŸ¥æœ‰æ•ˆæ ·æœ¬æ•°
        if not self.valid_image_ids:
            print(f"\nâ�Œ {split}é›†æ— æœ‰æ•ˆæ ·æœ¬ï¼�è¯¦ç»†å�Ÿå› ï¼š")
            print(f"  - è¾“å…¥å›¾åƒ�IDæ•°é‡�ï¼š{len(image_ids_with_path)}")
            print(f"  - å›¾åƒ�æ ¹è·¯å¾„ä¸‹çš„æ–‡ä»¶/æ–‡ä»¶å¤¹æ•°é‡�ï¼š{len(list(self.data_root.glob('*')))}")
            print(f"  - æ ‡æ³¨æ ¹è·¯å¾„ä¸‹çš„XMLæ–‡ä»¶æ•°é‡�ï¼š{len(list(self.anno_root.glob('*.xml')))}")
            raise ValueError(f"{split}é›†æ— æœ‰æ•ˆæ ·æœ¬ï¼Œè¯·æ£€æŸ¥è·¯å¾„å’Œæ•°æ�®é›†æ ¼å¼�")
        
        print(f"\nâœ… {split}é›†åŠ è½½å®Œæˆ�ï¼š")
        print(f"  - æœ‰æ•ˆæ ·æœ¬æ•°ï¼š{len(self.valid_image_ids)}")
        print(f"  - ç±»åˆ«æ•°ï¼š{self.num_classes}")
        print(f"  - ç±»åˆ«ç¤ºä¾‹ï¼š{list(self.class_to_idx.keys())[:5]}")

    def __len__(self):
        return len(self.valid_image_ids)

    def __getitem__(self, idx):
        img_path, xml_path, class_id = self.valid_image_ids[idx]

        # è¯»å�–å›¾åƒ�
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"æ— æ³•è¯»å�–å›¾åƒ�ï¼š{img_path}ï¼Œé”™è¯¯ï¼š{e}")

        # è�·å�–æ ‡ç­¾
        label = self.class_to_idx[class_id]

        if self.transform:
            image = self.transform(image)
        return image, label


def load_kaggle_ilsvrc_dataset(train_transform, val_transform):
    # Kaggle ImageNetæ•°æ�®é›†çš„æ ‡å‡†è·¯å¾„
    KAGGLE_DATA_ROOT = Path("/kaggle/input/imagenet-object-localization-challenge/ILSVRC")
    print(f"ğŸŒ� æ•°æ�®é›†æ ¹è·¯å¾„ï¼š{KAGGLE_DATA_ROOT} {'âœ… å­˜åœ¨' if KAGGLE_DATA_ROOT.exists() else 'â�Œ ä¸�å­˜åœ¨'}")
    
    if not KAGGLE_DATA_ROOT.exists():
        raise FileNotFoundError("Kaggleæ•°æ�®é›†æ ¹è·¯å¾„ä¸�å­˜åœ¨ï¼�è¯·æ£€æŸ¥æ•°æ�®é›†æ˜¯å�¦æ­£ç¡®æŒ‚è½½")

    # 1. å®šä¹‰æ‰€æœ‰å¿…è¦�è·¯å¾„
    train_imageset_path = KAGGLE_DATA_ROOT / "ImageSets" / "CLS-LOC" / "train_cls.txt"
    val_imageset_path = KAGGLE_DATA_ROOT / "ImageSets" / "CLS-LOC" / "val.txt"
    
    train_data_root = KAGGLE_DATA_ROOT / "Data" / "CLS-LOC" / "train"
    val_data_root = KAGGLE_DATA_ROOT / "Data" / "CLS-LOC" / "val"
    train_anno_root = KAGGLE_DATA_ROOT / "Annotations" / "CLS-LOC" / "train"
    val_anno_root = KAGGLE_DATA_ROOT / "Annotations" / "CLS-LOC" / "val"

    # æ‰“å�°æ‰€æœ‰è·¯å¾„ä¾›è°ƒè¯•
    print("\nğŸ“‹ æ‰€æœ‰å…³é”®è·¯å¾„ï¼š")
    paths_to_check = [
        ("train_cls.txt", train_imageset_path),
        ("val.txt", val_imageset_path),
        ("trainå›¾åƒ�æ ¹è·¯å¾„", train_data_root),
        ("valå›¾åƒ�æ ¹è·¯å¾„", val_data_root),
        ("trainæ ‡æ³¨æ ¹è·¯å¾„", train_anno_root),
        ("valæ ‡æ³¨æ ¹è·¯å¾„", val_anno_root)
    ]
    for name, path in paths_to_check:
        print(f"  - {name}ï¼š{path} {'âœ… å­˜åœ¨' if path.exists() else 'â�Œ ä¸�å­˜åœ¨'}")

    # 2. è¯»å�–å›¾åƒ�IDåˆ—è¡¨ï¼ˆä¿®å¤�äº†è§£æ��é€»è¾‘ï¼‰
    print("\n" + "="*50)
    print("ğŸ“¥ æ­£åœ¨è¯»å�–å›¾åƒ�IDåˆ—è¡¨...")
    print("="*50)
    train_image_ids = get_image_ids_from_imageset(train_imageset_path, split="train")
    val_image_ids = get_image_ids_from_imageset(val_imageset_path, split="val")

    # 3. åˆ�å§‹åŒ–Datasetï¼ˆä¿®å¤�äº†æ ‡æ³¨è·¯å¾„æ‹¼æ�¥ï¼‰
    print("\n" + "="*50)
    print("ğŸ“¦ æ­£åœ¨åŠ è½½è®­ç»ƒé›†...")
    print("="*50)
    train_dataset = ILSVRC_Kaggle_Dataset(
        image_ids_with_path=train_image_ids[:1000],  # æµ‹è¯•ç”¨ï¼šå�ªå�–å‰�1000ä¸ªæ ·æœ¬ï¼ˆå�¯é€‰ï¼‰
        data_root=train_data_root,
        anno_root=train_anno_root,
        split="train",
        transform=train_transform
    )

    print("\n" + "="*50)
    print("ğŸ“¦ æ­£åœ¨åŠ è½½éªŒè¯�é›†...")
    print("="*50)
    val_dataset = ILSVRC_Kaggle_Dataset(
        image_ids_with_path=val_image_ids[:100],  # æµ‹è¯•ç”¨ï¼šå�ªå�–å‰�100ä¸ªæ ·æœ¬ï¼ˆå�¯é€‰ï¼‰
        data_root=val_data_root,
        anno_root=val_anno_root,
        split="val",
        transform=val_transform
    )

    # 4. åˆ�å§‹åŒ–DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 if DEVICE.type == "cpu" else 2,  # CPUæ¨¡å¼�ä¸‹ç¦�ç”¨å¤šçº¿ç¨‹
        pin_memory=True if DEVICE.type != "cpu" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0 if DEVICE.type == "cpu" else 2,
        pin_memory=True if DEVICE.type != "cpu" else False
    )

    print(f"\nğŸš€ æ•°æ�®åŠ è½½å®Œæˆ�ï¼š")
    print(f"  - è®­ç»ƒé›†æ‰¹æ¬¡æ•°é‡�ï¼š{len(train_loader)}")
    print(f"  - éªŒè¯�é›†æ‰¹æ¬¡æ•°é‡�ï¼š{len(val_loader)}")
    print(f"  - æ€»ç±»åˆ«æ•°ï¼š{train_dataset.num_classes}")
    
    return train_loader, val_loader, train_dataset.num_classes


# ====================== 3. æ¨¡å�‹åŠ è½½/è®­ç»ƒ/è¯„ä¼°ï¼ˆä¿�æŒ�ä¸�å�˜ï¼‰======================
def load_pretrained_model(model_name, num_classes):
    print(f"\nğŸ“¦ åŠ è½½æ¨¡å�‹ï¼š{model_name}")
    if model_name == "alexnet":
        try:
            model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        except:
            model = models.alexnet(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg16":
        try:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        except:
            model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except:
            model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"â�Œ ä¸�æ”¯æŒ�çš„æ¨¡å�‹ï¼š{model_name}")

    for param in model.parameters():
        param.requires_grad = False
    if model_name in ["alexnet", "vgg16"]:
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
    elif model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
    model = model.to(DEVICE)
    print(f"âœ… æ¨¡å�‹åŠ è½½å®Œæˆ�ï¼ˆè®¾å¤‡ï¼š{DEVICE}ï¼‰")
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} | Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({"Loss": f"{running_loss/total:.4f}", "Top-1 Acc": f"{correct/total:.4f}"})
    return running_loss/total, correct/total


def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            _, top1_pred = outputs.max(1)
            top1_correct += top1_pred.eq(labels).sum().item()
            top5_pred = outputs.topk(5, dim=1)[1]
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            pbar.set_postfix({
                "Val Loss": f"{running_loss/total:.4f}",
                "Top-1 Acc": f"{top1_correct/total:.4f}",
                "Top-5 Acc": f"{top5_correct/total:.4f}"
            })
    return running_loss/total, top1_correct/total, top5_correct/total


def main():
    print("="*60)
    print("ğŸ�¯ ImageNetå›¾åƒ�åˆ†ç±»è®­ç»ƒå¼€å§‹")
    print("="*60)
    print(f"ğŸ“‹ è®­ç»ƒé…�ç½®ï¼š")
    print(f"  - è®¾å¤‡ï¼š{DEVICE}")
    print(f"  - æ‰¹æ¬¡å¤§å°�ï¼š{BATCH_SIZE}")
    print(f"  - è®­ç»ƒè½®æ•°ï¼š{EPOCHS}")
    print(f"  - å­¦ä¹ ç�‡ï¼š{LEARNING_RATE}")
    print(f"  - æ¨¡å�‹åˆ—è¡¨ï¼š{MODEL_NAMES}")
    print("="*60)

    # æ•°æ�®é¢„å¤„ç�†
    train_transform, val_transform = get_data_transforms()
    
    # åŠ è½½æ•°æ�®ï¼ˆå…³é”®ä¿®å¤�ï¼‰
    try:
        train_loader, val_loader, num_classes = load_kaggle_ilsvrc_dataset(train_transform, val_transform)
    except Exception as e:
        print(f"\nâ�Œ æ•°æ�®åŠ è½½å¤±è´¥ï¼š{e}")
        return

    results = {"model": [], "top1_acc": [], "top5_acc": [], "best_val_loss": []}

    for model_name in MODEL_NAMES:
        print(f"\n{'='*60}\nğŸ“Œ å¼€å§‹è®­ç»ƒæ¨¡å�‹ï¼š{model_name}\n{'='*60}")
        model = load_pretrained_model(model_name, num_classes)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        best_top1_acc = 0.0
        best_val_loss = float("inf")

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
            val_loss, top1_acc, top5_acc = evaluate_model(model, val_loader, criterion)
            scheduler.step()

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{model_name}_best_weights.pth")
                print(f"ğŸ’¾ ä¿�å­˜æœ€ä½³æ¨¡å�‹ -> Top-1 Acc: {best_top1_acc:.4f}")

            print(f"ğŸ“Š Epoch {epoch+1} æ€»ç»“ï¼š")
            print(f"  - è®­ç»ƒæ�Ÿå¤±ï¼š{train_loss:.4f} | è®­ç»ƒå‡†ç¡®ç�‡ï¼š{train_acc:.4f}")
            print(f"  - éªŒè¯�æ�Ÿå¤±ï¼š{val_loss:.4f} | Top-1å‡†ç¡®ç�‡ï¼š{top1_acc:.4f} | Top-5å‡†ç¡®ç�‡ï¼š{top5_acc:.4f}")

        results["model"].append(model_name)
        results["top1_acc"].append(best_top1_acc)
        results["top5_acc"].append(top5_acc)
        results["best_val_loss"].append(best_val_loss)

    print(f"\n{'='*60}\nğŸ�† æ‰€æœ‰æ¨¡å�‹è®­ç»ƒå®Œæˆ�ï¼Œç»“æ�œå¯¹æ¯”ï¼š\n{'='*60}")
    for i in range(len(results["model"])):
        print(f"{results['model'][i]}:")
        print(f"  - Top-1å‡†ç¡®ç�‡ï¼š{results['top1_acc'][i]:.4f}")
        print(f"  - Top-5å‡†ç¡®ç�‡ï¼š{results['top5_acc'][i]:.4f}")
        print(f"  - æœ€ä½³éªŒè¯�æ�Ÿå¤±ï¼š{results['best_val_loss'][i]:.4f}\n")

    # å�¯è§†åŒ–
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results["model"], results["top1_acc"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel("ç»�å…¸CNNæ¨¡å�‹", fontsize=12)
    plt.ylabel("Top-1å‡†ç¡®ç�‡", fontsize=12)
    plt.title("ImageNetå›¾åƒ�åˆ†ç±» - Top-1å‡†ç¡®ç�‡å¯¹æ¯”", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    for bar, acc in zip(bars, results["top1_acc"]):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f"{acc:.4f}", ha='center', va='bottom', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("top1_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(results["model"], results["top5_acc"], color=['#d62728', '#9467bd', '#8c564b'])
    plt.xlabel("ç»�å…¸CNNæ¨¡å�‹", fontsize=12)
    plt.ylabel("Top-5å‡†ç¡®ç�‡", fontsize=12)
    plt.title("ImageNetå›¾åƒ�åˆ†ç±» - Top-5å‡†ç¡®ç�‡å¯¹æ¯”", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    for bar, acc in zip(bars, results["top5_acc"]):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f"{acc:.4f}", ha='center', va='bottom', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("top5_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

