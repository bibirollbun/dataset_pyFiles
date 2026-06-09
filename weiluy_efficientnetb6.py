import cv2
import numpy as np
import os
import pandas as pd
import random
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations import Compose, Normalize, HorizontalFlip, Rotate, ColorJitter
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.optimize import minimize

# ==============================================================================
# 0. 全局设置
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ==============================================================================
# 1. 预处理函数
# ==============================================================================
def circle_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image

def apply_ben_graham_preprocessing(image, sigmaX=30):
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX)
    processed_image = cv2.addWeighted(image, 4, blurred_image, -4, 128)
    return processed_image

def preprocess_directory(input_dir, output_dir, image_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n开始处理目录: {input_dir}")
    for filename in tqdm(image_files, desc=f"处理中 {os.path.basename(input_dir)}"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        image = cv2.imread(input_path)
        if image is None:
            continue

        processed_image = circle_crop(image)
        processed_image = cv2.resize(processed_image, (image_size, image_size))
        processed_image = apply_ben_graham_preprocessing(processed_image)
        cv2.imwrite(output_path, processed_image)

    print(f"处理完成！处理后的图片已保存至: {output_dir}")

# ==============================================================================
# 2. PyTorch 模型与数据类
# ==============================================================================
def replace_batchnorm_with_groupnorm(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            if num_channels % num_groups == 0:
                setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            replace_batchnorm_with_groupnorm(child, num_groups)

class EfficientNetModel(nn.Module):
    def __init__(self, model_name, pretrained_model_path=None):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        replace_batchnorm_with_groupnorm(self.model)
        
        in_features = self.model.classifier.in_features
        # 简化分类头
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 1)
        )
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.model.load_state_dict(torch.load(pretrained_model_path), strict=False)
        elif pretrained_model_path:
            print(f"警告: 预训练模型路径未找到: {pretrained_model_path}")

    def forward(self, x):
        return self.model(x)

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx, 0]
        label = float(self.labels_df.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, img_id + '.png')
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, label
        except Exception as e:
            # 如果图片读取失败，则尝试加载下一张图片
            return self.__getitem__((idx + 1) % len(self))

def get_transforms(data_type='train'):
    if data_type == 'train':
        return Compose([
            Rotate(limit=40, p=0.5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
            HorizontalFlip(p=0.5),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else: # 'valid' or 'test'
        return Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

# ==============================================================================
# 3. 训练与预测函数
# ==============================================================================
def train(model, train_loader, valid_loader, device, num_epochs, lr, image_size):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_valid_kappa = 0.0
    epochs_no_improve = 0
    patience = 5
    save_path = os.path.join(os.getcwd(), f'best_model_kappa_{image_size}.pth')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc="训练中"):
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        
        model.eval()
        all_valid_labels = []
        all_valid_preds = []
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="验证中"):
                images = images.to(device)
                outputs = model(images).squeeze()
                all_valid_labels.extend(labels.cpu().numpy())
                preds_np = outputs.cpu().numpy()
                all_valid_preds.extend(np.atleast_1d(preds_np))
        
        rounded_preds = np.round(all_valid_preds).astype(int)
        valid_kappa = cohen_kappa_score(all_valid_labels, rounded_preds, weights='quadratic')

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证 Kappa (四舍五入): {valid_kappa:.4f}")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        if valid_kappa > best_valid_kappa:
            best_valid_kappa = valid_kappa
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> 已保存新最佳模型! 验证 Kappa: {best_valid_kappa:.4f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\n在 {patience} 个 epoch 没有提升后触发早停。")
            break

class OptimizedRounder:
    def __init__(self):
        self.coef_ = 0
        
    def _kappa_loss(self, coef, X, y):
        X_p = self.predict(X, coef)
        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll
    
    def fit(self, X, y):
        loss_partial = lambda coef: self._kappa_loss(coef, X, y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = minimize(loss_partial, initial_coef, method='nelder-mead')['x']
        print(f"优化后的阈值: {self.coef_}")
        
    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]: X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]: X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]: X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]: X_p[i] = 3
            else: X_p[i] = 4
        return X_p.astype(int)

def get_validation_predictions(loader, model, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="获取验证集预测结果"):
            images = images.to(device)
            outputs = model(images).squeeze()
            all_labels.extend(labels.cpu().numpy())
            preds_np = outputs.cpu().numpy()
            all_preds.extend(np.atleast_1d(preds_np))
    return np.array(all_preds), np.array(all_labels)

# ==============================================================================
# 预测函数 (已集成 TTA)
# ==============================================================================
def load_and_predict_test_data(test_csv_path, img_dir, model, device, coefficients):
    test_df = pd.read_csv(test_csv_path)
    predictions = []
    test_transform = get_transforms('test')
    rounder = OptimizedRounder()
    model.eval()

    for img_id in tqdm(test_df['id_code'], desc="在测试集上使用 TTA 预测"):
        img_path = os.path.join(img_dir, img_id + '.png')
        if not os.path.exists(img_path):
            predictions.append(-1) 
            continue
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- TTA 修改开始 ---
        
        # 1. 创建原始图片和水平翻转图片的 Tensor
        img_flipped = cv2.flip(img, 1) # 1 表示水平翻转
        
        tensor_original = test_transform(image=img)["image"]
        tensor_flipped = test_transform(image=img_flipped)["image"]
        
        # 2. 将两个 Tensor 堆叠成一个 mini-batch
        image_batch = torch.stack([tensor_original, tensor_flipped]).to(device)

        # 3. 使用模型进行预测，会得到两个结果
        with torch.no_grad():
            preds_batch = model(image_batch)
            
            # 4. 将两个预测结果求平均，得到最终的稳定预测值
            final_pred = preds_batch.mean().cpu().numpy()
            
        # --- TTA 修改结束 ---

        predictions.append(final_pred)

    if -1 in predictions:
        valid_preds = [p for p in predictions if p != -1]
        rounded_valid_preds = np.round(valid_preds).astype(int)
        if len(rounded_valid_preds) > 0: 
            mode_val = pd.Series(rounded_valid_preds).mode()[0]
        else: 
            mode_val = 0
        predictions = [mode_val if p == -1 else p for p in predictions]
        
    test_preds_rounded = rounder.predict(np.array(predictions), coefficients)
    test_df['diagnosis'] = test_preds_rounded
    submission_file = os.path.join(os.getcwd(), 'submission.csv')
    test_df.to_csv(submission_file, index=False)
    print(f"\n预测结果已保存至 {submission_file}")


# ==============================================================================
# 4. 主执行逻辑 (已修改)
# ==============================================================================
if __name__ == "__main__":
    # --- 步骤 1: 定义路径和参数 ---
    print("--- 步骤 1: 初始化路径和参数 ---")
    MODEL_NAME = 'efficientnet_b6'
    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 2e-4
    NUM_WORKERS = 2
    
    # --- 路径定义 (关键修改) ---
    # <--- 修改: 将路径从 ./input/... 改为 /kaggle/input/...
    # 原始竞赛数据路径 (主要用于读取 .csv 文件)
    COMPETITION_DATA_DIR = '/kaggle/input/aptos2019-blindness-detection'
    source_train_dir = os.path.join(COMPETITION_DATA_DIR, 'train_images')
    source_test_dir = os.path.join(COMPETITION_DATA_DIR, 'test_images')
    train_label_file = os.path.join(COMPETITION_DATA_DIR, 'train.csv')
    test_label_file = os.path.join(COMPETITION_DATA_DIR, 'test.csv')
    
    # <--- 修改: 这是您提供的预处理数据集的正确路径
    PREPROCESSED_DATA_INPUT_DIR = '/kaggle/input/images-preprocessed-512'
    
    # Notebook 运行时生成的预处理数据的保存路径 (作为备用)
    PREPROCESSED_DATA_WORKING_DIR = '/kaggle/working'

    # <--- 修改: 预训练模型权重路径
    PRETRAINED_MODEL_PATH = '/kaggle/input/efficientnet-b6/pytorch_model_effi_b6.bin'
    
    # 输出模型文件的路径
    model_path = os.path.join(os.getcwd(), f'best_model_kappa_{IMAGE_SIZE}.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # --- 步骤 2: 智能判断使用哪个图片数据集 ---
    print("\n--- 步骤 2: 检查预处理数据 ---")
    
    # 现在这个路径是正确的了
    final_train_image_dir = os.path.join(PREPROCESSED_DATA_INPUT_DIR, f'train_images_preprocessed_{IMAGE_SIZE}')
    final_test_image_dir = os.path.join(PREPROCESSED_DATA_INPUT_DIR, f'test_images_preprocessed_{IMAGE_SIZE}')

    # 这个 if 判断现在应该会成功
    if os.path.exists(final_train_image_dir) and os.path.exists(final_test_image_dir):
        print(f"成功找到已上传的预处理数据，将使用路径: {PREPROCESSED_DATA_INPUT_DIR}")
    else:
        print(f"未在 '{PREPROCESSED_DATA_INPUT_DIR}' 找到预处理数据。")
        print(f"将执行实时预处理，并将图片保存到 {PREPROCESSED_DATA_WORKING_DIR} 目录。")
        final_train_image_dir = os.path.join(PREPROCESSED_DATA_WORKING_DIR, f'train_images_preprocessed_{IMAGE_SIZE}')
        final_test_image_dir = os.path.join(PREPROCESSED_DATA_WORKING_DIR, f'test_images_preprocessed_{IMAGE_SIZE}')
        
        # 这一部分现在不会被执行，从而避免了错误
        preprocess_directory(source_train_dir, final_train_image_dir, IMAGE_SIZE)
        preprocess_directory(source_test_dir, final_test_image_dir, IMAGE_SIZE)

    # --- 步骤 3: 执行模型训练 (如果需要) ---
    print("\n--- 步骤 3: 检查现有模型 ---")
    run_training = True
    if os.path.exists(model_path):
        print(f"在 {model_path} 找到模型文件。跳过训练。")
        run_training = False

    if run_training:
        print("开始模型训练...")
        all_labels_df = pd.read_csv(train_label_file)
        train_df, valid_df = train_test_split(
            all_labels_df, test_size=0.2, random_state=42, stratify=all_labels_df['diagnosis']
        )
        train_dataset = CustomImageDataset(final_train_image_dir, train_df, transform=get_transforms('train'))
        valid_dataset = CustomImageDataset(final_train_image_dir, valid_df, transform=get_transforms('valid'))
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        model = EfficientNetModel(MODEL_NAME, pretrained_model_path=PRETRAINED_MODEL_PATH).to(device)
        train(model, train_loader, valid_loader, device, NUM_EPOCHS, LEARNING_RATE, IMAGE_SIZE)
    
    # --- 步骤 4: 优化舍入阈值 ---
    print("\n--- 步骤 4: 优化舍入阈值 ---")
    if not os.path.exists(model_path):
         print("错误: 未找到模型文件。无法进行阈值优化。")
    else:
        all_labels_df = pd.read_csv(train_label_file)
        _, valid_df_for_opt = train_test_split(
            all_labels_df, test_size=0.2, random_state=42, stratify=all_labels_df['diagnosis']
        )
        valid_dataset_for_opt = CustomImageDataset(final_train_image_dir, valid_df_for_opt, transform=get_transforms('valid'))
        valid_loader_for_opt = DataLoader(valid_dataset_for_opt, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        model_for_opt = EfficientNetModel(MODEL_NAME, pretrained_model_path=None).to(device)
        model_for_opt.load_state_dict(torch.load(model_path, map_location=device))
        
        val_preds, val_labels = get_validation_predictions(valid_loader_for_opt, model_for_opt, device)
        opt_rounder = OptimizedRounder()
        opt_rounder.fit(val_preds, val_labels)
        optimized_coefficients = opt_rounder.coef_
        
        optimized_preds = opt_rounder.predict(val_preds, optimized_coefficients)
        final_val_kappa = cohen_kappa_score(val_labels, optimized_preds, weights='quadratic')
        print(f"最终优化后的验证集 Kappa: {final_val_kappa:.4f}")
    
        # --- 步骤 5: 加载最佳模型并进行预测 ---
        print("\n--- 步骤 5: 加载最佳模型并在测试集上预测 ---")
        model_for_pred = EfficientNetModel(MODEL_NAME, pretrained_model_path=None).to(device)
        model_for_pred.load_state_dict(torch.load(model_path, map_location=device))
        load_and_predict_test_data(test_label_file, final_test_image_dir, model_for_pred, device, optimized_coefficients)

    print("\n--- 脚本执行结束 ---")

