!pip install pydicom albumentations
!pip install pydicom albumentations timm

import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection.rpn import AnchorGenerator
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import timm  # 用于加载 EfficientNet 模型
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



TRAIN_DIR = "../input/rsna-pneumonia-detection-challenge/stage_2_train_images/"
TEST_DIR = "../input/rsna-pneumonia-detection-challenge/stage_2_test_images/"
TRAIN_LABELS = "../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
DETAILED_CLASS_INFO = "../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv"
OUTPUT_DIR = "/kaggle/working/"


torch.backends.cudnn.benchmark = True
print("Imports and configuration completed.")

class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None, mode='train', class_info=None):
        """
        初始化数据集类
        :param dataframe: 数据集的 DataFrame，包含标注信息
        :param image_dir: 图像所在目录
        :param transforms: 数据增强函数
        :param mode: 'train' 或 'test'
        :param class_info: 类别信息 (从 stage_2_detailed_class_info.csv 读取)
        """
        self.class_info = class_info  # 将 class_info 赋值给实例属性
        self.dataframe = self._preprocess(dataframe, mode)
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode
        # 创建一个映射字典，确保标签从1开始（0通常是背景类）
        self.label_map = {"No Lung Opacity / Not Normal": 1, "Normal": 2, "Lung Opacity": 3}
        # 过滤掉没有有效边界框的患者ID
        self.valid_patient_ids = self.dataframe.patientId.unique()

    def _preprocess(self, df, mode):
        """处理重复标注和路径问题"""
        # 合并类别信息
        if self.class_info is not None:
            df = df.merge(self.class_info[['patientId', 'class']], on='patientId', how='left')
            # 只保留肺部异常的样本
            df = df[df['class'] != 'No Lung Opacity / Not Normal'].reset_index(drop=True)
        
        # 添加边界框的右下角坐标
        df['x_max'] = df['x'] + df['width']
        df['y_max'] = df['y'] + df['height']

        # 丢弃包含 NaN 的样本
        df = df.dropna(subset=['x', 'y', 'x_max', 'y_max'])
        
        # 过滤无效边界框 - 确保所有坐标都是有效的
        df = df[(df['x'] >= 0) & (df['y'] >= 0) & (df['x_max'] > df['x']) & (df['y_max'] > df['y'])]
        
        # 确保每个患者至少有一个有效边界框
        if mode == 'train':
            return df.groupby('patientId').filter(lambda x: len(x) > 0)
        return df

    def __getitem__(self, idx):
        patient_id = self.valid_patient_ids[idx].strip()  # 去除空格
        dicom_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM 文件不存在: {dicom_path}")
            
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array.astype(np.float32)
        # 归一化
        image = (image - image.min()) / (image.max() - image.min())
        image = np.stack([image] * 3, axis=-1)
        
        if self.mode == 'train':
            # 获取当前患者的所有边界框
            patient_df = self.dataframe[self.dataframe.patientId == patient_id]
            boxes = patient_df[['x', 'y', 'x_max', 'y_max']].values.astype(np.float32)
            
            # 确保边界框有效（非空且没有NaN）
            if len(boxes) == 0 or np.any(np.isnan(boxes)):
                # 创建一个虚拟的边界框和标签，避免返回None
                boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
                labels = np.array([1], dtype=np.int64)  # 使用1作为默认标签（非背景）
            else:
                # 获取标签并映射
                if 'class' in patient_df.columns:
                    labels = patient_df['class'].map(self.label_map).values
                else:
                    # 如果没有类别信息，默认使用标签1
                    labels = np.ones(len(boxes), dtype=np.int64)
            
            # 应用数据增强
            if self.transforms:
                transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
                image = transformed['image']
                boxes = np.array(transformed['bboxes'])
                
                # 确保变换后的边界框非空
                if len(boxes) == 0:
                    boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int64)
                else:
                    labels = np.array(transformed['labels'])
            
            # 创建目标字典
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx])
            }
            return image, target
        
        return image, patient_id

    def __len__(self):
        return len(self.valid_patient_ids)


# 数据增强操作配置
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(
        size=(512, 512),
        scale=(0.08, 1.0),
        ratio=(0.75, 1.33)
    ),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['labels'],
    min_visibility=0.3
))
print("Dataset and augmentations defined.")




class EfficientNetBackboneWithFPN(nn.Module):
    """
    利用 EfficientNet-B3 预测多个层次特征，再用 FPN 融合输出字典形式的特征，
    符合 Faster R-CNN 对 backbone 的要求。
    """
    def __init__(self, model_name='efficientnet_b3', out_channels=256, pretrained=True, out_indices=None):
        """
        :param model_name: EfficientNet 模型名称
        :param out_channels: FPN 输出的通道数
        :param pretrained: 是否加载预训练权重
        :param out_indices: 指定哪些特征层用于 FPN，默认为最后4个特征层
        """
        super().__init__()
        self.backbone = timm.create_model(model_name, features_only=True, pretrained=pretrained)
        if out_indices is None:
            out_indices = [1, 2, 3, 4]
        self.out_indices = out_indices
        in_channels_list = [self.backbone.feature_info[i]['num_chs'] for i in out_indices]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features = [features[i] for i in self.out_indices]
        features_dict = {str(i): feature for i, feature in enumerate(features)}
        x = self.fpn(features_dict)
        return x

def get_model(num_classes=4, score_thresh=0.5):  # 增加到4类，包括背景类
    # 构造自定义 backbone
    custom_backbone = EfficientNetBackboneWithFPN(model_name='efficientnet_b3', out_channels=256, pretrained=True)
    custom_backbone.out_channels = 256  # 必须设置

    # 定义 AnchorGenerator，为每个 FPN 层指定尺寸，注意这里 4 个元素对应 4 个特征图
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )

    # 构造 Faster R-CNN 模型时传入自定义的 rpn_anchor_generator
    model = FasterRCNN(
        custom_backbone, 
        num_classes=num_classes,  # 包括背景类
        rpn_anchor_generator=anchor_generator,
        min_size=512,  # 确保输入图像大小一致
        max_size=512
    )
    model.roi_heads.score_thresh = score_thresh  # 设置候选框过滤阈值
    return model

print("Custom model defined.")



def collate_fn(batch):
    filtered_batch = list(filter(lambda x: x is not None, batch))
    if len(filtered_batch) == 0:
        return [], []
    images = [item[0] for item in filtered_batch]
    targets = [item[1] for item in filtered_batch]
    return images, targets

def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    valid_batches = 0
    progress = tqdm(loader, desc='Training')
    
    for images, targets in progress:
        # 跳过空批次
        if len(images) == 0:
            continue
            
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 检查边界框和标签
            valid_batch = True
            for target in targets:
                # 确保每个目标至少有一个边界框
                if len(target['boxes']) == 0:
                    valid_batch = False
                    break
                # 确保标签值在有效范围内
                if torch.any(target['labels'] <= 0):
                    valid_batch = False
                    break
            
            if not valid_batch:
                continue
                
            outputs = model(images, targets)
            losses = sum(loss for loss in outputs.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            valid_batches += 1
            progress.set_postfix(loss=total_loss / (valid_batches))
            
        except Exception as e:
            print(f"Error in batch: {e}")
            continue
    
    return total_loss / max(1, valid_batches)


def predict(model, test_dir, device, threshold=0.3):
    model.eval()
    submission = []
    test_files = [f.replace(' ', '') for f in os.listdir(test_dir) if f.endswith('.dcm')]
    progress = tqdm(test_files, desc='Predicting')
    for filename in progress:
        patient_id = filename[:-4]
        dicom_path = os.path.join(test_dir, filename)
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = np.stack([image] * 3, axis=-1).transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(image)
        boxes = preds[0]['boxes'].cpu().numpy()
        scores = preds[0]['scores'].cpu().numpy()
        keep = scores > threshold
        boxes = boxes[keep]
        scores = scores[keep]
        
        if len(scores) > 0:
            order = np.argsort(-scores)
            boxes = boxes[order]
            scores = scores[order]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            pred_str = ' '.join([f"{s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}" 
                                 for s, (x, y, w, h) in zip(scores, boxes)])
        else:
            pred_str = ""
        submission.append({'patientId': patient_id, 'PredictionString': pred_str})
    return pd.DataFrame(submission)

print("Training and prediction functions defined.")

def visualize_samples(model, dataset, indices, device, save_path=None, show=True):
    """可视化几个样本的预测结果，并可选择显示和保存"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        image, target = dataset[idx]
        
        # 预测
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(image_tensor)
        
        # 转回numpy显示
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # 显示图像
        axes[i].imshow(image_np[:,:,0], cmap='gray')
        
        # 绘制真实边界框
        for box, label in zip(target['boxes'], target['labels']):
            x, y, x_max, y_max = box.cpu().numpy()
            width = x_max - x
            height = y_max - y
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='g', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(x, y-5, f'GT: {label.item()}', color='g', fontsize=8, backgroundcolor='w')
        
        # 绘制预测边界框
        for box, score, label in zip(preds[0]['boxes'], preds[0]['scores'], preds[0]['labels']):
            if score > 0.5:  # 只显示高置信度的预测
                x, y, x_max, y_max = box.cpu().numpy()
                width = x_max - x
                height = y_max - y
                rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect)
                axes[i].text(x, y+height+5, f'Pred: {label.item()} ({score:.2f})', color='r', fontsize=8, backgroundcolor='w')
        
        axes[i].set_title(f"Sample {idx}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 保存图片（如果提供了路径）
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存至: {save_path}")
    
    # 显示图片（如果需要）
    if show:
        plt.show()
    else:
        plt.close()

    return fig  # 返回图形对象，便于进一步处理

def visualize_confusion_matrix(model, dataset, device, save_path=None):
    """绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    # 使用DataLoader批量处理数据
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            # 获取预测结果
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                
                # 获取得分最高的预测
                if len(pred_scores) > 0:
                    best_idx = np.argmax(pred_scores)
                    all_preds.append(pred_labels[best_idx])
                else:
                    all_preds.append(0)  # 背景类
                
                # 获取真实标签
                true_boxes = targets[i]['boxes'].cpu().numpy()
                true_labels = targets[i]['labels'].cpu().numpy()
                
                if len(true_labels) > 0:
                    all_labels.append(true_labels[0])  # 取第一个标签
                else:
                    all_labels.append(0)  # 背景类
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Background', 'Class 1', 'Class 2', 'Class 3'],
                yticklabels=['Background', 'Class 1', 'Class 2', 'Class 3'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩阵已保存至: {save_path}")
    
    plt.show()

def visualize_precision_recall(model, dataset, device, save_path=None):
    """绘制精确率-召回率曲线"""
    model.eval()
    class_scores = {1: [], 2: [], 3: []}  # 每个类别的得分
    class_labels = {1: [], 2: [], 3: []}  # 每个类别的真实标签
    
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                
                true_boxes = targets[i]['boxes'].cpu().numpy()
                true_labels = targets[i]['labels'].cpu().numpy()
                
                # 为每个类别收集预测分数和真实标签
                for cls in [1, 2, 3]:
                    # 找到该类别的预测
                    cls_indices = np.where(pred_labels == cls)[0]
                    if len(cls_indices) > 0:
                        max_score_idx = cls_indices[np.argmax(pred_scores[cls_indices])]
                        class_scores[cls].append(pred_scores[max_score_idx])
                    else:
                        class_scores[cls].append(0.0)
                    
                    # 检查真实标签中是否有该类别
                    class_labels[cls].append(1 if cls in true_labels else 0)
    
    # 绘制每个类别的PR曲线
    plt.figure(figsize=(12, 8))
    
    for cls in [1, 2, 3]:
        precision, recall, _ = precision_recall_curve(class_labels[cls], class_scores[cls])
        ap = average_precision_score(class_labels[cls], class_scores[cls])
        
        plt.plot(recall, precision, lw=2, 
                 label=f'Class {cls} (AP = {ap:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"PR曲线已保存至: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # 适用于多进程环境（例如 Windows）
    torch.multiprocessing.freeze_support()
    # 初始化设备（Kaggle Notebook 上用 GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载类别数据信息
    class_info_df = pd.read_csv(DETAILED_CLASS_INFO)
    train_df = pd.read_csv(TRAIN_LABELS, encoding='utf-8')
    
    # 构造训练数据集（仅用于可视化）
    train_dataset = PneumoniaDataset(train_df, TRAIN_DIR, train_transform, 'train', class_info=class_info_df)
    
    # 初始化模型
    model = get_model(num_classes=4).to(device)  # 4类包括背景类
    
    # 加载训练好的模型
    checkpoint_path = "/kaggle/input/efficientnet-b3-fpn/pytorch/default/1/model_epoch_20.pth"  # 指定要加载的模型文件
    if os.path.exists(checkpoint_path):
        # 添加weights_only=True以避免安全警告
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Error: Model file {checkpoint_path} not found!")
        exit()
    
    # 设置模型为评估模式
    model.eval()
    
    # 可视化一些样本
    random_indices = np.random.choice(len(train_dataset), 6, replace=False)
    visualize_samples(model, train_dataset, random_indices, device, 
                 save_path=os.path.join(OUTPUT_DIR, "sample_predictions.png"),
                 show=True)
    
    #  混淆矩阵可视化
    visualize_confusion_matrix(model, train_dataset, device, 
                              os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    #  精确率-召回率曲线
    visualize_precision_recall(model, train_dataset, device, 
                              os.path.join(OUTPUT_DIR, "pr_curve.png"))
    
    # 生成提交文件
    submission_df = predict(model, TEST_DIR, device, threshold=0.3)
    submission_df.to_csv("submission3.csv", index=False)
    print("Submission file saved!")


