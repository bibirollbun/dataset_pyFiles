import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

dataset_root = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
train_dir = os.path.join(dataset_root, "train")
label_file = os.path.join(dataset_root, "train_labels.csv")
mask_root = "/kaggle/working/masks"  # 掩码图保存目录
index_save_path = "/kaggle/working/image_mask_index.csv"
sigma = 8  # 高斯核标准差

# 高斯掩码函数
def draw_gaussian_mask(shape, centers, sigma=6):
    heatmap = np.zeros(shape, dtype=np.float32)
    for y, x in centers:
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            tmp = np.zeros_like(heatmap)
            tmp[y, x] = 1
            tmp = cv2.GaussianBlur(tmp, (0, 0), sigma)
            tmp /= tmp.max()  # 归一化到 [0, 1]
            heatmap = np.maximum(heatmap, tmp)
    return (heatmap * 255).astype(np.uint8)

# 加载标签数据
labels_df = pd.read_csv(label_file)
labels_df = labels_df[labels_df["Motor axis 0"] != -1]
labels_df = labels_df.rename(columns={
    "Motor axis 0": "z",
    "Motor axis 1": "y",
    "Motor axis 2": "x"
})
labels_df[["z", "y", "x"]] = labels_df[["z", "y", "x"]].astype(int)

# 遍历 tomogram
records = []
for tomo_id in tqdm(sorted(os.listdir(train_dir)), desc="处理Tomogram"):
    tomo_path = os.path.join(train_dir, tomo_id)
    if not os.path.isdir(tomo_path):
        continue

    os.makedirs(os.path.join(mask_root, tomo_id), exist_ok=True)
    tomo_labels = labels_df[labels_df["tomo_id"] == tomo_id]

    for fname in sorted(os.listdir(tomo_path)):
        if not fname.endswith(".jpg"):
            continue

        slice_idx = int(fname.replace("slice_", "").replace(".jpg", ""))
        image_path = os.path.join(tomo_path, fname)

        with Image.open(image_path) as img:
            width, height = img.size

        slice_labels = tomo_labels[tomo_labels["z"] == slice_idx][["y", "x"]].values.tolist()
        is_positive = int(len(slice_labels) > 0)

        if is_positive:
            mask_np = draw_gaussian_mask((height, width), slice_labels, sigma=sigma)
            mask_path = os.path.join(mask_root, tomo_id, fname.replace(".jpg", ".png"))

            # 保存为灰度 PNG 图像
            cv2.imwrite(mask_path, mask_np)
        else:
            mask_path = None

        records.append({
            "image_path": image_path,
            "mask_path": mask_path,
            "tomo_id": tomo_id,
            "slice_idx": slice_idx,
            "width": width,
            "height": height,
            "is_positive": is_positive,
            "points": str(slice_labels)
        })

# 保存索引 CSV
df = pd.DataFrame(records)
df.to_csv(index_save_path, index=False)
print(f"掩码索引文件保存至: {index_save_path}，共记录 {len(df)} 条")
print(f"正样本数量: {df['is_positive'].sum()}，负样本数量: {(df['is_positive'] == 0).sum()}")


import pandas as pd

# 这里的路径变化是为了避免kaggle重启内核导致数据丢失,下载了掩码索引和掩码图作为数据集上传到了/kaggle/input中
csv_path = "/kaggle/input/mask-image/kaggle/working/image_mask_index.csv"
mask_root_prefix = "/kaggle/input/mask-image/kaggle/working/masks"

# 原始索引文件
df = pd.read_csv(csv_path)

# 修正掩码路径,原因是原掩码索引构建时掩码图位于/kaggle/working文件夹中
df["mask_path"] = df["mask_path"].apply(
    lambda p: p.replace("/kaggle/working/masks", mask_root_prefix) if isinstance(p, str) else None
)

#正负样本划分
positive_df = df[df["is_positive"] == 1]
negative_df = df[df["is_positive"] == 0]

# 随机采样等量负样本
negative_sample = negative_df.sample(n=len(positive_df), random_state=42)

# 合并平衡数据集
balanced_df = pd.concat([positive_df, negative_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# 保存平衡索引
balanced_df.to_csv("/kaggle/working/balanced_image_mask_index.csv", index=False)

print(f"平衡数据集已保存，共计样本数: {len(balanced_df)} (正样本: {len(positive_df)}, 负样本: {len(negative_sample)})")


import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

csv_path = "/kaggle/working/balanced_image_mask_index.csv"
mask_root_prefix = "/kaggle/input/mask-image/kaggle/working/masks" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(csv_path)

#数据集划分
df = df.reset_index(drop=True)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# dataset函数
class FlagellaDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        if pd.isna(row["mask_path"]):
            # 对阴性样本创建空白mask
            mask_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
        else:
            mask = Image.open(row["mask_path"]).convert("L")
            mask = np.array(mask).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        encoded = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "pixel_mask": encoded["pixel_mask"].squeeze(0),
            "mask_labels": mask_tensor,
        }

    def __len__(self):
        return len(self.df)

def custom_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    mask_labels = [item["mask_labels"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
    }
# 初始化
processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-panoptic")
train_dataset = FlagellaDataset(train_df, processor)
val_dataset = FlagellaDataset(val_df, processor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
# 用于fine-tunning的基模型
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-panoptic").to(device)

# 解冻非主干层
for name, param in model.named_parameters():
    param.requires_grad = not name.startswith("backbone")

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

#使用Mask2Former的循环逻辑会报错,这里改为标准的pytorch循环逻辑
best_val_loss = float("inf")
patience = 3
patience_counter = 0

print("\n 开始训练，设备：", device)
for epoch in range(20):
    model.train()
    print(f"\n Epoch {epoch + 1}")
    train_losses = []

    for step, batch in enumerate(tqdm(train_loader, desc="训练中")):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        mask_labels_list = batch["mask_labels"]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        mask_preds = outputs["masks_queries_logits"]

        losses = []
        for i in range(pixel_values.size(0)):
            mask_pred = mask_preds[i][0:1]
            mask_label = mask_labels_list[i].to(device).float()

            if mask_label.dim() == 2:
                mask_label = mask_label.unsqueeze(0).unsqueeze(0)
            elif mask_label.dim() == 3:
                mask_label = mask_label.unsqueeze(0)

            mask_label_resized = F.interpolate(mask_label, size=mask_pred.shape[-2:], mode="bilinear", align_corners=False)
            mask_label_resized = mask_label_resized.squeeze(1)

            # 采用BCEloss，添加pos_weight避免模型去向全0预测
            loss = F.binary_cross_entropy_with_logits(
                mask_pred, mask_label_resized, pos_weight=torch.tensor(3.0).to(device)
            )
            losses.append(loss)

            if i == 0:
                with torch.no_grad():
                    #可视化训练过程
                    print("sigmoid(mask_pred).mean():", torch.sigmoid(mask_pred).mean().item())
                    plt.subplot(1, 2, 1)
                    plt.imshow(mask_label_resized.squeeze().cpu(), cmap='gray')
                    plt.title("GT Mask")
                    plt.subplot(1, 2, 2)
                    plt.imshow(torch.sigmoid(mask_pred).squeeze().cpu(), cmap='gray')
                    plt.title("Pred Mask")
                    plt.show()

        loss_batch = torch.stack(losses).mean()
        loss_batch.backward()

        # 统计梯度
        with torch.no_grad():
            grads = [param.grad.abs().mean().item() for param in model.parameters() if param.grad is not None]
            if grads:
                print(f"平均梯度：{np.mean(grads):.6f}")
            else:
                print("没有可训练的参数或无梯度")

        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss_batch.item())

        if step % 10 == 0:
            print(f"🟢 Step {step}: Train Loss = {loss_batch.item():.4f}")

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        #验证集评估
        for val_batch in tqdm(val_loader, desc="验证中"):
            pixel_values = val_batch["pixel_values"].to(device)
            pixel_mask = val_batch["pixel_mask"].to(device)
            mask_labels_list = val_batch["mask_labels"]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            mask_preds = outputs["masks_queries_logits"]

            for i in range(pixel_values.size(0)):
                mask_pred = mask_preds[i][0:1]
                mask_label = mask_labels_list[i].to(device).float()

                if mask_label.dim() == 2:
                    mask_label = mask_label.unsqueeze(0).unsqueeze(0)
                elif mask_label.dim() == 3:
                    mask_label = mask_label.unsqueeze(0)

                mask_label_resized = F.interpolate(mask_label, size=mask_pred.shape[-2:], mode="bilinear", align_corners=False)
                mask_label_resized = mask_label_resized.squeeze(1)

                val_loss = F.binary_cross_entropy_with_logits(
                    mask_pred, mask_label_resized, pos_weight=torch.tensor(3.0).to(device)
                )
                total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"验证集平均 Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        print("新最佳模型，保存中...")
        best_val_loss = avg_val_loss
        patience_counter = 0
        model.save_pretrained("/kaggle/working/mask2former_bce_best")
    else:
        patience_counter += 1
        print(f"验证集无提升，Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("提前停止训练")
            break

print("训练完成！最佳验证 loss：", best_val_loss)


import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, AutoConfig
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model_path = "/kaggle/input/heatmap-mask2former/pytorch/default/3/mask2former_package/model"
processor_path = "/kaggle/input/heatmap-mask2former/pytorch/default/3/preprocessor_config.json"
csv_path = "/kaggle/working/balanced_image_mask_index.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Mask2FormerImageProcessor.from_pretrained(processor_path, local_files_only=True)
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path, config=config, local_files_only=True).to(device).eval()

# 推理dataset
class InferenceDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        encoded = self.processor(images=image, return_tensors="pt")
        label = 1 if pd.notna(row["mask_path"]) else 0
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "pixel_mask": encoded["pixel_mask"].squeeze(0),
            "label": label
        }

    def __len__(self):
        return len(self.df)

# 特征提取,用于逻辑回归
def extract_features(df):
    dataset = InferenceDataset(df, processor)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    results = []
    with torch.no_grad():
        for batch in tqdm(loader):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = batch["label"].cpu().numpy()

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            mask_preds = outputs["masks_queries_logits"]

            for i in range(pixel_values.size(0)):
                mask_pred = mask_preds[i][0:1]
                pred_mask = torch.sigmoid(mask_pred).squeeze().cpu().numpy()

                max_value = pred_mask.max()
                mean_value = pred_mask.mean()
                area_ratio = np.sum(pred_mask > 0.5) / pred_mask.size

                results.append({
                    "max": max_value,
                    "mean": mean_value,
                    "area": area_ratio,
                    "label": labels[i]
                })

    return pd.DataFrame(results)

# 相同的数据集划分
df = pd.read_csv(csv_path).reset_index(drop=True)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_features = extract_features(train_df)
train_features.to_csv("/kaggle/working/train_features.csv", index=False)

val_features = extract_features(val_df)
val_features.to_csv("/kaggle/working/val_features.csv", index=False)

# 逻辑回归
X_train = train_features[["max", "mean", "area"]].values
y_train = train_features["label"].values
X_val = val_features[["max", "mean", "area"]].values
y_val = val_features["label"].values

model_lr = LogisticRegression(max_iter=500)
model_lr.fit(X_train, y_train)

# 验证集评估 
val_preds = model_lr.predict(X_val)
val_probs = model_lr.predict_proba(X_val)[:,1]
val_f1 = f1_score(y_val, val_preds)
cm = confusion_matrix(y_val, val_preds)
print(f"\n 验证集逻辑回归 F1 = {val_f1:.4f}")
print("验证集混淆矩阵:\n", cm)

# 保存逻辑回归模型
joblib.dump(model_lr, "/kaggle/working/logistic_model.pkl")


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation, AutoConfig

model_path = "/kaggle/input/heatmap-mask2former/pytorch/default/4/mask2former_package/model"
processor_path = "/kaggle/input/heatmap-mask2former/pytorch/default/4"
logistic_path = "/kaggle/input/heatmap-mask2former/pytorch/default/4/logistic_model.pkl"  
val_csv = "/kaggle/working/balanced_image_mask_index.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path, config=config, local_files_only=True).to(device).eval()
processor = Mask2FormerImageProcessor.from_pretrained(processor_path, local_files_only=True)
logistic_model = joblib.load(logistic_path)

# 验证集数据加载
df = pd.read_csv(val_csv).reset_index(drop=True)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# 定义Dataset与Dataloader
class ValDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        encoded = self.processor(images=image, return_tensors="pt")
        label = 1 if pd.notna(row["mask_path"]) else 0

        # 加载GT mask
        if pd.notna(row["mask_path"]):
            mask = Image.open(row["mask_path"]).convert("L")
            mask = np.array(mask) / 255.0  # 归一化
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros((512, 512), dtype=torch.float32)  # 假设原始掩码大小为512×512

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "pixel_mask": encoded["pixel_mask"].squeeze(0),
            "mask_labels": mask_tensor,
            "image_path": row["image_path"],
            "label": label
        }

    def __len__(self):
        return len(self.df)

val_dataset = ValDataset(val_df, processor)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 分类逻辑回归判别函数
def classify_mask_logistic(pred_mask, logistic_model):
    max_value = pred_mask.max().item()
    mean_value = pred_mask.mean().item()
    area_ratio = np.sum(pred_mask.numpy() > 0.5) / pred_mask.numel()
    feature = np.array([[max_value, mean_value, area_ratio]])
    pred = logistic_model.predict(feature)[0]
    return int(pred)

# 提取mask中心
def get_center_from_mask(mask_tensor):
    if torch.all(mask_tensor == 0):
        return None
    coords = torch.nonzero(mask_tensor == mask_tensor.max())
    y, x = coords[0][-2:].tolist()
    return x, y

def evaluate_and_visualize(model, val_loader, val_df, num_visualize=10, radius=6):
    model.eval()
    count = 0

    # 评分核心参数
    angstrom_threshold = 1000
    pixel_size = 20  # 官方没有生命测试集的voxel spacing(Å/pixel),这里采用一个较为严格的指标评估

    TP, FP, FN = 0, 0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            mask_preds = outputs["masks_queries_logits"]

            image_path = batch["image_path"][0]
            image = Image.open(image_path).convert("RGB")
            w_img, h_img = image.size

            gt_mask_raw = batch["mask_labels"][0].squeeze().cpu()
            pred_mask = torch.sigmoid(mask_preds[0][0]).cpu()
            gt_mask = F.interpolate(gt_mask_raw.unsqueeze(0).unsqueeze(0), size=pred_mask.shape, mode="bilinear", align_corners=False).squeeze()

            h_pred, w_pred = pred_mask.shape

            gt_center = get_center_from_mask(gt_mask)
            pred_center = get_center_from_mask(pred_mask)

            # 计算缩放系数 (特征图分辨率 → 原图像素)
            scale_x = w_img / w_pred
            scale_y = h_img / h_pred

            pred_label = classify_mask_logistic(pred_mask, logistic_model)

            row = val_df.iloc[batch_idx]
            is_positive = int(row["is_positive"])

            if is_positive == 0 and pred_label == 0:
                pass  # 正确负样本
            elif is_positive == 0 and pred_label == 1:
                FP += 1
            elif is_positive == 1 and pred_label == 0:
                FN += 1
            else:  
                z_true = int(row["slice_idx"])
                y_true, x_true = eval(row["points"])[0]
                z_pred = z_true 

                if pred_center is not None:
                    # 预测中心先还原回原图分辨率
                    x_pred_rescaled = pred_center[0] * scale_x
                    y_pred_rescaled = pred_center[1] * scale_y

                    # 计算像素空间欧氏距离
                    dist_pixel = np.sqrt(
                        (z_true - z_pred) ** 2 + 
                        (y_true - y_pred_rescaled) ** 2 + 
                        (x_true - x_pred_rescaled) ** 2
                    )

                    # 转换为物理单位 Å
                    dist_angstrom = dist_pixel * pixel_size

                    if dist_angstrom <= angstrom_threshold:
                        TP += 1
                    else:
                        FN += 1
                else:
                    FN += 1

            # 可视化
            if count < num_visualize:
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                axs[0].imshow(gt_mask.numpy(), cmap='gray')
                axs[0].set_title("GT Mask")
                axs[0].axis('off')

                axs[1].imshow(pred_mask.numpy(), cmap='gray')
                axs[1].set_title("Pred Mask")
                axs[1].axis('off')

                axs[2].imshow(image)
                axs[2].set_title("Pred vs GT Center")
                axs[2].axis('off')

                if pred_label == 1 and pred_center is not None:
                    pred_x = int(pred_center[0] * scale_x)
                    pred_y = int(pred_center[1] * scale_y)
                    pred_x = min(max(pred_x, 0), w_img - 1)
                    pred_y = min(max(pred_y, 0), w_img - 1)
                    axs[2].add_patch(patches.Circle(
                        (pred_x, pred_y), radius=radius,
                        edgecolor='red', facecolor='none',
                        linewidth=2, alpha=0.7, linestyle='-'
                    ))
                    axs[2].text(pred_x + 2, pred_y + 10, "Pred", color="red", fontsize=10, weight='bold')
                else:
                    axs[2].text(w_img - 10, 30, "Pred: None", color="red", fontsize=14, weight='bold', ha='right')

                if gt_center is not None:
                    gt_x = int(gt_center[0] * scale_x)
                    gt_y = int(gt_center[1] * scale_y)
                    gt_x = min(max(gt_x, 0), w_img - 1)
                    gt_y = min(max(gt_y, 0), w_img - 1)
                    axs[2].add_patch(patches.Circle(
                        (gt_x, gt_y), radius=radius,
                        edgecolor='cyan', facecolor='none',
                        linewidth=2, alpha=1.0, linestyle='--'
                    ))
                    axs[2].text(gt_x + 2, gt_y - 2, "GT", color="cyan", fontsize=10, weight='bold')
                else:
                    axs[2].text(10, 30, "GT: None", color="cyan", fontsize=14, weight='bold')

                plt.tight_layout()
                plt.show()

                count += 1

    # 最终评分指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    beta = 2
    if precision + recall == 0:
        f_beta = 0
    else:
        f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    print(f"\n Final Evaluation Result (Å-based strict scoring):")
    print(f"TP={TP}, FP={FP}, FN={FN}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F2={f_beta:.4f}")


# 执行可视化
evaluate_and_visualize(model, val_loader, val_df, num_visualize=40)

