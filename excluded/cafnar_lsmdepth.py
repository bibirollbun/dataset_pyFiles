!git clone https://github.com/mmaaz60/EdgeNeXt.git


!pip install -r /kaggle/working/EdgeNeXt/requirements.txt
!pip install --upgrade --ignore-installed matplotlib==3.8.0


!sed -i 's/from torch._six import inf/from math import inf/g' /kaggle/working/EdgeNeXt/utils.py
!sed -i "s/torch.load(args.resume, map_location='cpu')/torch.load(args.resume, map_location='cpu', weights_only=False)/" /kaggle/working/EdgeNeXt/utils.py


!wget https://github.com/mmaaz60/EdgeNeXt/releases/download/v1.0/edgenext_small.pth


import os
import shutil

# Define the source paths from the read-only input directory
source_data_path = '/kaggle/input/imagenet1k-subset-100k-train-and-10k-val'
source_train_path = os.path.join(source_data_path, 'imagenet_subtrain')
source_val_path = os.path.join(source_data_path, 'imagenet_subval')

# Define the destination paths in the writable working directory
# We'll create a new directory to hold our data
working_data_path = '/kaggle/working/imagenet_data'
os.makedirs(working_data_path, exist_ok=True)

dest_train_path = os.path.join(working_data_path, 'train')
dest_val_path = os.path.join(working_data_path, 'val')

# Copy the data to the writable directory
print("Copying data to writable directory...")
shutil.copytree(source_train_path, dest_train_path)
shutil.copytree(source_val_path, dest_val_path)
print("Copying complete.")

# Now, your script should use the new data path
new_data_path_for_script = working_data_path
print(f"New data path to use: {new_data_path_for_script}")


!python3 /kaggle/working/EdgeNeXt/main.py \
--model edgenext_small \
--eval True \
--batch_size 16 \
--data_path /kaggle/working/imagenet_data \
--output_dir /kaggle/working/ \
--resume /kaggle/working/edgenext_small.pth \
--num_workers 4


!python /kaggle/working/EdgeNeXt/main.py \
--model edgenext_small --drop_path 0.1 \
--batch_size 256 --lr 6e-3 --update_freq 2 \
--model_ema true --model_ema_eval true \
--data_path /kaggle/working/imagenet_data \
--output_dir /kaggle/working/ \
--use_amp True --multi_scale_sampler


!git clone https://github.com/Akshathakrbhat/Manipal-UAV-Person-Dataset.git


import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

# Kích thước chuẩn cho mô hình
TARGET_DIM = 300

class UAVPersonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'Test Samples')
        self.label_dir = os.path.join(root_dir, 'Test Labels')
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])
        
        assert len(self.image_files) == len(self.label_files), "The number of image files and label files do not match."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = self.label_files[idx]
        
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return None, None
            
        img_w, img_h = image.size
        
        boxes = []
        labels = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                x_min = (x_center - w / 2) * img_w
                y_min = (y_center - h / 2) * img_h
                x_max = (x_center + w / 2) * img_w
                y_max = (y_center + h / 2) * img_h
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id + 1)
        except FileNotFoundError:
            pass
            
        if not boxes:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        
        scale = TARGET_DIM / max(img_w, img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        pad_w = (TARGET_DIM - new_w) // 2
        pad_h = (TARGET_DIM - new_h) // 2
        
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded_image = Image.new('RGB', (TARGET_DIM, TARGET_DIM), (0, 0, 0))
        padded_image.paste(resized_image, (pad_w, pad_h))

        if boxes.numel() > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_h
            
        image_tensor = F.to_tensor(padded_image)

        return image_tensor, target

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    if not batch:
        return torch.tensor([]), []

    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    images = torch.stack(images, 0)
    
    return images, targets


import torch
import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch.nn.functional as F
from torchvision.transforms import functional as T_F
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from EdgeNeXt.models.edgenext import EdgeNeXt

# --- 1. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH ---
def edgenext_small(**kwargs):
    kwargs.setdefault('classifier_dropout', 0.0)
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                     global_block=[0, 1, 1, 1],
                     global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                     use_pos_embd_xca=[False, True, False, False],
                     kernel_sizes=[3, 5, 7, 9],
                     d2_scales=[2, 2, 3, 4],
                     **kwargs)
    return model

class EdgeNeXtBackbone(nn.Module):
    def __init__(self, backbone_path):
        super(EdgeNeXtBackbone, self).__init__()
        self.original_model = edgenext_small(num_classes=1000)
        state_dict = torch.load(backbone_path, map_location='cpu', weights_only=False)['model']
        self.original_model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded EdgeNeXt pre-trained weights.")

    def forward(self, x):
        features = []
        x = self.original_model.downsample_layers[0](x)
        x = self.original_model.stages[0](x)
        x = self.original_model.downsample_layers[1](x)
        x = self.original_model.stages[1](x)
        features.append(x)
        x = self.original_model.downsample_layers[2](x)
        x = self.original_model.stages[2](x)
        features.append(x)
        x = self.original_model.downsample_layers[3](x)
        x = self.original_model.stages[3](x)
        features.append(x)
        return features

def _prediction_head(in_channels, num_anchors, num_classes):
    cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
    reg_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
    return cls_head, reg_head

# Thêm lớp hậu xử lý
class SSDLitePostProcessor(object):
    def __init__(self, anchor_generator):
        self.anchor_generator = anchor_generator
    
    def __call__(self, cls_logits, bbox_regression, image_sizes):
        # Đây là phần logic phức tạp, tôi chỉ đưa ra một ví dụ đơn giản.
        # Bạn sẽ cần một hàm để giải mã bbox và một hàm NMS.
        
        # Hàm decode_boxes và NMS_func cần được triển khai
        # Đây là ví dụ giả lập đầu ra
        detections = []
        for i in range(cls_logits.size(0)):
            # Giả định đầu ra đã được xử lý
            boxes = torch.randn(10, 4) * 100
            scores = torch.rand(10)
            labels = torch.randint(1, 2, (10,))
            
            # Giả lập NMS
            keep = torch.ones(10, dtype=torch.bool)
            
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            detections.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })
        return detections

class EdgeNeXtSSDLite(nn.Module):
    def __init__(self, num_classes, backbone_path):
        super(EdgeNeXtSSDLite, self).__init__()
        self.backbone = EdgeNeXtBackbone(backbone_path)
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(304, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
            ),
        ])
        
        anchor_sizes = ( (16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.num_anchors = [len(s) * len(r) for s, r in zip(anchor_sizes, aspect_ratios)]
        out_channels = [96, 160, 304, 256, 128]

        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for channels, num_anchor in zip(out_channels, self.num_anchors):
            cls_head, reg_head = _prediction_head(channels, num_anchor, num_classes)
            self.cls_heads.append(cls_head)
            self.reg_heads.append(reg_head)
        
        # Thêm hậu xử lý
        self.postprocessor = SSDLitePostProcessor(
            anchor_generator=AnchorGenerator(
                sizes=anchor_sizes, aspect_ratios=aspect_ratios
            )
        )

    def forward(self, images, targets=None):
        features = self.backbone(images)
        last_feature = features[-1]
        for layer in self.extra_layers:
            last_feature = layer(last_feature)
            features.append(last_feature)
        
        cls_logits = []
        bbox_regression = []

        for i, feature in enumerate(features):
            cls_logits.append(self.cls_heads[i](feature))
            bbox_regression.append(self.reg_heads[i](feature))
        
        cls_logits = [l.permute(0, 2, 3, 1).contiguous().view(l.size(0), -1) for l in cls_logits]
        bbox_regression = [r.permute(0, 2, 3, 1).contiguous().view(r.size(0), -1) for r in bbox_regression]
        
        cls_logits = torch.cat(cls_logits, dim=1)
        bbox_regression = torch.cat(bbox_regression, dim=1)
        
        if self.training:
            loss_dict = {}
            loss_dict['loss_classifier'] = F.cross_entropy(cls_logits, torch.zeros_like(cls_logits))
            loss_dict['loss_box_reg'] = F.l1_loss(bbox_regression, torch.zeros_like(bbox_regression))
            return loss_dict
        
        # Khi không ở chế độ training, thực hiện hậu xử lý
        return self.postprocessor(cls_logits, bbox_regression, images.shape[-2:])


# --- XỬ LÝ DATASET VÀ DATALOADER ---
TARGET_DIM = 300

class UAVPersonDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'Test Samples')
        self.label_dir = os.path.join(root_dir, 'Test Labels')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])
        
        assert len(self.image_files) == len(self.label_files), "The number of image files and label files do not match."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = self.label_files[idx]
        
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return None, None
            
        img_w, img_h = image.size
        
        boxes = []
        labels = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                x_min = (x_center - w / 2) * img_w
                y_min = (y_center - h / 2) * img_h
                x_max = (x_center + w / 2) * img_w
                y_max = (y_center + h / 2) * img_h
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id + 1)
        except FileNotFoundError:
            pass
            
        if not boxes:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        
        scale = TARGET_DIM / max(img_w, img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        pad_w = (TARGET_DIM - new_w) // 2
        pad_h = (TARGET_DIM - new_h) // 2
        
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded_image = Image.new('RGB', (TARGET_DIM, TARGET_DIM), (0, 0, 0))
        padded_image.paste(resized_image, (pad_w, pad_h))

        if boxes.numel() > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_h
            
        image_tensor = T_F.to_tensor(padded_image)

        return image_tensor, target

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    if not batch:
        return torch.tensor([]), []

    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    images = torch.stack(images, 0)
    
    return images, targets

# --- VÒNG LẶP HUẤN LUYỆN VÀ LƯU MÔ HÌNH ---
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        
        if not loss_dict:
            continue
            
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
    
    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch+1} hoàn thành. Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    num_classes = 2
    num_epochs = 10
    learning_rate = 0.001
    BACKBONE_PATH = '/kaggle/working/edgenext_small.pth'
    DATA_PATH = "/kaggle/working/Manipal-UAV-Person-Dataset"
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Sử dụng thiết bị: {device}")

    model = EdgeNeXtSSDLite(num_classes=num_classes, backbone_path=BACKBONE_PATH)
    model.to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    dataset = UAVPersonDataset(DATA_PATH)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)

    print(f"Số lượng mẫu trong dataset: {len(dataset)}")
    print(f"Số lượng batches trong dataloader: {len(data_loader)}")
    print("\n--- Bắt đầu huấn luyện ---")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
        
        torch.save(model.state_dict(), f'/kaggle/working/edgenext_ssdlite_epoch_{epoch+1}.pth')
        print(f"Mô hình đã được lưu tại: edgenext_ssdlite_epoch_{epoch+1}.pth")

    print("\n--- Huấn luyện hoàn tất! ---")


import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.data import DataLoader
import os

def evaluate_model(model, data_loader, device):
    model.eval()

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Đánh giá mô hình"):
            images = images.to(device)
            
            # --- Tải predictions từ mô hình ---
            predictions = model(images)
            
            # --- Chuyển targets sang định dạng phù hợp cho torchmetrics ---
            target_list = []
            for t in targets:
                target_list.append({
                    'boxes': t['boxes'],
                    'labels': t['labels'],
                })
            
            metric.update(predictions, target_list)
            
    try:
        results = metric.compute()
        print(f"Kết quả đánh giá: {results}")
    except RuntimeError as e:
        print(f"Lỗi khi tính mAP: {e}")
        print("Có thể có lỗi trong định dạng đầu ra hoặc không có bounding box được phát hiện.")

if __name__ == "__main__":
    num_classes = 2
    BACKBONE_PATH = '/kaggle/working/edgenext_small.pth'
    DATA_PATH = "/kaggle/working/Manipal-UAV-Person-Dataset"
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Khởi tạo lại mô hình
    model = EdgeNeXtSSDLite(num_classes=num_classes, backbone_path=BACKBONE_PATH)
    
    # Tải mô hình đã được huấn luyện. Thay đổi số epoch tùy ý.
    model_to_evaluate = 10
    model_path = f'/kaggle/working/edgenext_ssdlite_epoch_{model_to_evaluate}.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"\nĐã tải mô hình từ: {model_path}")
    else:
        print(f"\nKhông tìm thấy file mô hình tại {model_path}. Hãy đảm bảo bạn đã chạy cell huấn luyện trước.")

    model.to(device)

    # Khởi tạo DataLoader cho tập đánh giá
    eval_dataset = UAVPersonDataset(DATA_PATH)
    eval_data_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Chạy đánh giá
    evaluate_model(model, eval_data_loader, device)

