# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pydicom
import math
import cv2 as cv
import tensorflow as tf
from tqdm import tqdm

USE_PRETRAINED_MODEL = True


if not USE_PRETRAINED_MODEL:
    # Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
    model = fasterrcnn_resnet50_fpn(weights=True)
    
    # Number of classes (your dataset classes + 1 for background)
    num_classes = 2  # For example, 2 classes + background
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the head of the model with a new one (for the number of classes in your dataset)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



input_size = 244

def format_image(img, box):
    height, width = img.shape 
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((input_size, input_size), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized

    x, y, w, h = (box[0], box[1], box[2], box[3]) if box[0] else (0.0,0.0,0.0,0.0)
    new_box = [int((x)/ r), int((y)/ r), int(w/ r), int(h/ r)] if box[0] else [0.0,0.0,0.0,0.0]

    return new_image, new_box


import torch
from torchvision import transforms as T

# 定義轉換器
transform = T.Compose([
    T.ToTensor(),                     # 確保影像轉換為 PyTorch Tensor
    T.Normalize(mean=[0.5], std=[0.5]) # 正規化
])


# Define transformations (e.g., resizing, normalization)
transform = T.Compose([
    T.ToTensor(),
])
# Custom Dataset class or using an existing one
class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, transforms=None):
        dataframe = dataframe.reset_index(drop=True)
        # Initialize dataset paths and annotations here
        self.transforms = transforms
        # Your dataset logic (image paths, annotations, etc.)
        self.dataframe = dataframe

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['patientId']
        img_path = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/"+img_path+".dcm"
        # Load DICOM image
        temp_img = pydicom.dcmread(img_path).pixel_array
        # 確認標註框是否有效
        temp_box = [row['x'], row['y'], row['width'], row['height']] if not math.isnan(row['x']) else [0.0, 0.0, 0.0, 0.0]

        # 格式化影像與標註框
        img, box = format_image(temp_img, temp_box)
        
        # 修正邊界框的格式
        xmin, ymin, width, height = box
        # print('width = ', width)
        # print('height = ', height)
        xmax = xmin + width
        ymax = ymin + height

        # 處理有效框或空框
        if width <= 0.0 or height <= 0.0:
            # print('空框處理')
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # 空框
            labels = torch.zeros((0,), dtype=torch.int64)  # 空標籤
        else:
            if xmin > xmax or ymin > ymax:
                print('wrong coordination')
        
            # 構建有效框
            boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
            labels = torch.tensor([row['Target']], dtype=torch.int64)

        # 影像正規化
        # 影像正規化並轉換為 float32
        img = img.astype(np.float32) / 255.  # 明確指定 float32

       # 建立目標字典
        target = {}
        target["boxes"] = boxes.clone().detach().float()  # 確保格式正確
        target["labels"] = labels.clone().detach().long() # labels 保持為 int64
        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    def __len__(self):
        # Return the length of your dataset
        return len(self.dataframe)


data_labels = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
dataframe = data_labels


# 檢查 DataFrame 長度和索引
print(f"DataFrame 長度: {len(dataframe)}")
print(f"DataFrame 索引: {dataframe.index}")

# 確認是否存在索引不連續問題
print(f"索引是否連續: {dataframe.index.is_monotonic_increasing}")



data_labels[:1001]['Target'].value_counts()


# 取前6000個資料作為訓練集
train_dataset = PneumoniaDataset(data_labels[:6001],transform)

# 取第6001~6200個資料作為驗證集
valid_dataset = PneumoniaDataset(data_labels[6001:6801],transform)

# 建立DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, 
                                   collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, 
                                    collate_fn=lambda x: tuple(zip(*x)))


# 查看訓練集資料內容
for i, (img, target) in enumerate(train_dataset):
    print(f"Index {i}:")
    print("  Image shape:", img.shape)
    print("  Target:", target)
    
    # 示範只看前 5 筆
    if i == 4:
        break


if not USE_PRETRAINED_MODEL:
    # Move model to GPU if available
    if torch.cuda.is_availabel():
        device = torch.device('cuda')
        print("using cude as device.")
    else:
        torch.device('cpu')
        print("using cpu as device.")
    
    model.to(device)
    
    # Set up the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, 
                                                       weight_decay=0.0005)
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, 
                                                                   gamma=0.1)
    # Train the model
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
    
       # Training loop
        for images, targets in tqdm(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
            # Zero the gradients
            optimizer.zero_grad()
    
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
    
            # Backward pass
            losses.backward()
            optimizer.step()
            train_loss += losses.item()
    
        # Update the learning rate
        lr_scheduler.step()
        print(f'Epoch: {epoch + 1}, Loss: {train_loss / len(train_loader)}')
    print("Training complete!")


if not USE_PRETRAINED_MODEL:
    # 儲存完整模型
    torch.save(model, '/kaggle/working/fasterrcnn_model.pth')
    print("模型訓練完成並儲存至: /kaggle/working/fasterrcnn_model.pth")


# 1. 先建立跟訓練時相同結構的模型
model = fasterrcnn_resnet50_fpn(weights=False)  # 訓練好的模型就不需要再去下載預訓練權重，可以使用 weights=False

# 2. 根據先前訓練時的 class 數量，替換掉 roi_heads.box_predictor
num_classes = 2  # 要與當初訓練時設定的 class 數量相同
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 3. 載入模型權重
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load("/kaggle/input/trained-model/fasterrcnn_model.pth", map_location=device)
# 4. 將模型放到指定裝置上 (GPU or CPU)
model.to(device)

print("模型已成功載入！")


CONF = 0.65

def post_process_predictions(predictions, targets, conf_thresh):
    """
    將模型輸出的預測結果 (predictions)，根據 conf_thresh 做過濾並只取最高信心度的框。
    
    參數:
        predictions (list of dict): 模型對一個 batch 的推論結果，每個元素是一張影像的預測字典，
                                    其結構常見為 {'boxes': ..., 'labels': ..., 'scores': ...}
        targets (list of dict):     該 batch 各影像對應的標註 (ground truth)，用來對照。
        conf_thresh (float):        信心度閾值，預設 0.7
    
    回傳:
        results (list of dict):     後處理後的結果列表，每個元素包含:
            {
                'best_box': tensor([...]) 或 None,
                'best_label': int 或 None,
                'best_score': float 或 None,
                'all_filtered_boxes': tensor([...]),
                'all_filtered_labels': tensor([...]),
                'all_filtered_scores': tensor([...]),
                'no_high_conf': bool,  # 如果沒有任何框達到閾值，這裡為 True
            }
    """
    results = []

    for i in range(len(predictions)):
        pred_boxes = predictions[i]['boxes']
        pred_labels = predictions[i]['labels']
        pred_scores = predictions[i]['scores']

        # 先依照 conf_thresh 過濾
        high_conf_mask = pred_scores > conf_thresh
        filtered_boxes = pred_boxes[high_conf_mask]
        filtered_labels = pred_labels[high_conf_mask]
        filtered_scores = pred_scores[high_conf_mask]

        # 建立一個字典存放後處理結果
        result_dict = {
            'best_box': None,
            'best_label': None,
            'best_score': None,
            'all_filtered_boxes': filtered_boxes,
            'all_filtered_labels': filtered_labels,
            'all_filtered_scores': filtered_scores,
            'no_high_conf': False
        }

        if len(filtered_scores) == 0:
            # 沒有任何預測超過閾值
            result_dict['no_high_conf'] = True
        else:
            # 在篩選過的框中，找最高信心度
            max_score_idx = torch.argmax(filtered_scores)
            result_dict['best_box'] = filtered_boxes[max_score_idx]
            result_dict['best_label'] = filtered_labels[max_score_idx].item()
            result_dict['best_score'] = filtered_scores[max_score_idx].item()

        results.append(result_dict)

    return results



model.eval()

with torch.no_grad():
    batch_num = 1
    for images, targets in valid_loader:
        if(batch_num == 3):
            break
        print('===========================================')
        print(f"Validating batch {batch_num}")
        batch_num += 1
        images = list(img.to(device) for img in images)
        predictions = model(images)  # 一次對整個 batch 做推論
        
        # 呼叫我們的後處理函式
        processed_results = post_process_predictions(predictions, targets, CONF)
        
        # 在這裡，可以決定如何輸出或使用 processed_results
        # 例如，逐張影像查看結果:
        
        for i, result in enumerate(processed_results):
            print(f"image{i}: actual_labels:", targets[i]['labels'])
            if result['no_high_conf']:
                print(f"  No box over {CONF} confidence for image {i}.")
                print('----------------------------------------------')
                continue
            print(f"  Highest confidence box :")
            print("    Box:", result['best_box'])
            print("    Label:", result['best_label'])
            print("    Score:", result['best_score'])
            print('----------------------------------------------')


def evaluate_model(valid_loader, model, device):
    """
    使用簡化的『有無預測框』來判斷 TP, FP, TN, FN，
    並計算 Accuracy 與 F1-Score。
    
    參數:
      valid_loader: 驗證資料的 DataLoader
      model:        已訓練好的 PyTorch 目標偵測模型 (e.g. fasterrcnn_resnet50_fpn)
      device:       'cpu' or 'cuda'
      post_process_fn: 您的後處理函式 (例如 post_process_predictions)
      conf_thresh:  閾值（預設 0.7）
      
    回傳:
      None (直接在函式裡打印結果)
    """
    model.eval()
    
    # 建立混淆矩陣 (Confusion Matrix) 的四個計數器
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    with torch.no_grad():
        print(f'Total batch number: {len(valid_loader)}')
        num = 0
        for images, targets in valid_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)  # 模型對一個 batch 的推論
            
            # 後處理 -> 得到每張影像最終篩選/最高分框資訊
            processed_results = post_process_predictions(predictions, targets, CONF)

            # 針對此 batch 的每張影像，更新 TP/FP/TN/FN
            for i, result in enumerate(processed_results):
                # 1. 實際標註（是否有標籤）
                #    假設 labels 不為空 -> 視為「實際正 (actual positive)」
                #    若 labels 為空 -> 「實際負 (actual negative)」
                actual_positive = (len(targets[i]['labels']) > 0)
                
                # 2. 模型預測（是否有框 > conf_thresh）
                #    如果後處理後的 'no_high_conf' 為 False，代表有至少一個框超過閾值
                predicted_positive = (not result['no_high_conf'])
                
                # 3. 根據 actual_positive / predicted_positive 分類成 TP, FP, TN, FN
                if actual_positive and predicted_positive:
                    TP += 1
                elif not actual_positive and predicted_positive:
                    FP += 1
                elif actual_positive and not predicted_positive:
                    FN += 1
                else:
                    TN += 1
            num += 1
            print(f'batch {num} done')

    # 計算指標
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0

    # 避免分母為 0
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print("Confusion Matrix (簡化二元判斷，只判有無畫框):")
    print(f"  TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")


evaluate_model(valid_loader, model, device)




def compute_iou(box1, box2):
    """
    計算兩個框的 IoU (Intersection over Union)。
    box1, box2 皆是 [xmin, ymin, xmax, ymax] 的格式 (可以是 Tensor 或 ndarray)。
    回傳一個介於 [0, 1] 的浮點數。
    """
    # 先確保是 numpy array (若傳入的是 Tensor 則轉成 numpy)
    if torch.is_tensor(box1):
        box1 = box1.detach().cpu().numpy()
    if torch.is_tensor(box2):
        box2 = box2.detach().cpu().numpy()
    
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 交集部分
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    # 各自面積
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # 聯集面積 = area1 + area2 - inter_area
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou



import torch
import numpy as np
import matplotlib.patches as patches

def draw_image_and_boxes_on_ax(ax, img_tensor, target, predicted_box=None):
    """
    在給定的 Matplotlib Axes 上，繪製單張影像 + 真實標註(紅色) + 預測框(綠色)，
    並在圖上顯示 IoU (若 predicted_box 不為 None)。

    參數:
      ax:             Matplotlib Axes 物件 (例如由 plt.subplots() 建立)。
      img_tensor:     形狀 [C,H,W] (或 [H,W]) 的 PyTorch Tensor (灰階或 RGB)。
      target:         {'boxes': Tensor, 'labels': Tensor}，是真實標註。
      predicted_box:  shape = [4] (xmin, ymin, xmax, ymax)，若無可填 None。
    """
    # ---------- 1) Tensor -> NumPy，調整通道順序 ----------
    img_np = img_tensor.detach().cpu().numpy()
    if len(img_np.shape) == 3 and img_np.shape[0] in [1, 3]:
        # [C, H, W] -> [H, W, C]
        img_np = np.transpose(img_np, (1, 2, 0))  

    # 判斷灰階
    is_grayscale = False
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        is_grayscale = True
        img_np = img_np[..., 0]  # 變成 [H, W]

    # ---------- 2) 顯示影像 ----------
    if is_grayscale:
        ax.imshow(img_np, cmap='gray')
    else:
        ax.imshow(img_np)
    ax.axis('off')

    # ---------- 3) 繪製真實框 (紅色) ----------
    gt_boxes = target["boxes"]
    for box_tensor in gt_boxes:
        box = box_tensor.cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = box
        rect_w = xmax - xmin
        rect_h = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), rect_w, rect_h,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # ---------- 4) 繪製預測框 (綠色)，並計算 IoU ----------
    if predicted_box is not None:
        box_pred = predicted_box.detach().cpu().numpy().astype(int)
        xmin_p, ymin_p, xmax_p, ymax_p = box_pred
        rect_w = xmax_p - xmin_p
        rect_h = ymax_p - ymin_p
        rect = patches.Rectangle((xmin_p, ymin_p), rect_w, rect_h,
                                 linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

        # 針對每個真實框，都計算 IoU 並顯示
        for box_tensor in gt_boxes:
            iou_val = compute_iou(box_tensor, predicted_box)
            box_gt = box_tensor.cpu().numpy().astype(int)
            xmin_g, ymin_g, xmax_g, ymax_g = box_gt

            # 可以把 IoU 數字畫在真實框上方一點點
            text_x = xmin_g
            text_y = max(ymin_g - 5, 0)  # 避免座標<0

            ax.text(text_x, text_y, f"IoU={iou_val:.2f}", 
                    color='yellow', fontsize=10,
                    bbox=dict(facecolor='blue', alpha=0.3))




import matplotlib.pyplot as plt

model.eval()
all_ious =[]
with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(valid_loader):
        images = [img.to(device) for img in images]
        predictions = model(images)
        processed_results = post_process_predictions(predictions, targets, CONF)

        # 每個 batch 顯示 ncols = batch_size 張圖 (一行)
        batch_size = len(images)
        fig, axs = plt.subplots(nrows=1, ncols=batch_size, figsize=(5*batch_size, 5))

        # 若只有 1 張圖，axs 不是 list，需要手動包成 list
        if batch_size == 1:
            axs = [axs]

        for i, result in enumerate(processed_results):
            # 繪圖邏輯
            predicted_box = None
            if not result['no_high_conf']:
                predicted_box = result['best_box']  # shape=[4,]

            # 在 axs[i] 上繪圖
            draw_image_and_boxes_on_ax(
                ax=axs[i],
                img_tensor=images[i],
                target=targets[i],
                predicted_box=predicted_box
            )

            # IoU邏輯
            gt_boxes = targets[i]["boxes"]  # shape: [N, 4]
            has_gt = (len(gt_boxes) > 0)         # True 表示 GT 有框
            has_pred = (not result['no_high_conf'])  # True 表示預測有框
            
            # -- Case 1: 兩者都沒有框 -> skip
            if (not has_gt) and (not has_pred):
                # 什麼都不做，不加入 IoU
                continue

            # -- Case 2: 兩者都有框 -> 計算 IoU
            if has_gt and has_pred:
                predicted_box = result['best_box']  # shape=[4,]
                # 計算對所有 gt_box 的 IoU，取最大值
                max_iou_for_this_image = 0.0
                for gt_box in gt_boxes:
                    iou_val = compute_iou(gt_box, predicted_box)
                    max_iou_for_this_image = max(max_iou_for_this_image, iou_val)
                all_ious.append(max_iou_for_this_image)
            
            else:
                # -- Case 3: 僅一方有框 -> IoU=0
                #   (要嘛 GT 有框但模型沒框，
                #    或   GT 沒框但模型有框)
                all_ious.append(0.0)

        plt.suptitle(f"Batch {batch_idx} (showing {batch_size} images)", fontsize=16)
        plt.tight_layout()
        plt.show()


# 最後計算平均 IoU
if len(all_ious) > 0:
    mean_iou = sum(all_ious) / len(all_ious)
else:
    mean_iou = 0.0

print(f'length of all_ious : {len(all_ious)}')
print(f"Average IoU over validation set = {mean_iou:.4f}")


