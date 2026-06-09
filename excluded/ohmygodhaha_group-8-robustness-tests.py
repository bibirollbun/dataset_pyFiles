# Reproduction of the training setup and dataset split used in the training the model
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import timm 
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math 
import torchvision.transforms.functional as TF
import warnings
warnings.filterwarnings("ignore")


CONFIG = {
    "seed": 42,
    "img_size": 224,
    "batch_size": 32,                         
    "num_classes": 3,        
    "model_name": "swin_tiny_patch4_window7_224", 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "csv_path": "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "image_dir": "/kaggle/input/preprocessed-images-224"
}

LABEL_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
seed_everything(CONFIG['seed'])

class APTOSFastDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['id_code']
        img_path = os.path.join(self.img_dir, img_name + ".png")
        
        image = cv2.imread(img_path)
        if image is None: 
            
            image = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform: 
            image = self.transform(image)
            
        new_label = LABEL_MAP[row['diagnosis']]
        return image, torch.tensor(new_label, dtype=torch.long)


val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


test_tf = val_tf

df = pd.read_csv(CONFIG['csv_path'])
df['new_label'] = df['diagnosis'].map(LABEL_MAP)


train_val_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=CONFIG['seed'], 
    stratify=df['new_label']
)
test_loader = DataLoader(
    APTOSFastDataset(test_df, CONFIG['image_dir'], test_tf), 
    batch_size=CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=2
)

# Completed the reproduction of the training setup and dataset split used in the training the model

# Load trained model

MODEL_PATH = "/kaggle/input/final-best-sick-f1-model/pytorch/default/1/best_sick_f1_model.pth"
model = timm.create_model(CONFIG['model_name'], pretrained=False, num_classes=CONFIG['num_classes'])
model.load_state_dict(torch.load(MODEL_PATH, map_location=CONFIG['device']))
model = model.to(CONFIG['device'])
model.eval()

# ====================================================
# Define function to add Gaussian noise
# ====================================================

class AddGaussianNoise:
    
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # tensor is already normalized to [-1, 1] range
        noise = torch.randn_like(tensor) * self.std + self.mean
        noisy_tensor = tensor + noise
        return noisy_tensor
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# ====================================================
# Noise Test Function 
# ====================================================

def test_noisy(clean_result, noise_levels, model, test_df):
    model.eval()
    all_results = []

    print("\n" + "="*70)
    print("Starting Gaussian Noise Test")
    print("="*70)

    # ====================================================
    #Test model performance under different Gaussian noise levels
    # ====================================================
    
    for noise_std in noise_levels: 
        # 1. Create transform with noise
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            AddGaussianNoise(std=noise_std) # Add noise after normalization
        ])
        
        # 2. Create DataLoader with noisy transform
        test_dataset = APTOSFastDataset(test_df, CONFIG['image_dir'], transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # 3. Inference on noisy images
        test_preds, test_labels = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(test_loader, desc="Testing...", leave=True):
                imgs = imgs.to(CONFIG['device'])
                outputs = model(imgs)
                _, p = torch.max(outputs, 1)
                test_preds.extend(p.cpu().numpy())
                test_labels.extend(lbls.numpy())
        
        # 4. Calculate metrics
        test_metrics = precision_recall_fscore_support(test_labels, test_preds, 
                                                       labels=[0, 1, 2], zero_division=0)
        test_precision, test_recall, test_f1_scores = test_metrics[0], test_metrics[1], test_metrics[2]
        test_avg_sick_f1 = (test_f1_scores[1] + test_f1_scores[2]) / 2.0
        test_overall_acc = np.mean(np.array(test_preds) == np.array(test_labels))
        
        # 5. Print results
        print(f"\nGaussian Noise Level: std={noise_std}")
        print(f"Test Overall Acc: {test_overall_acc:.4f}")
        print("-" * 65)
        print(f"{'Class':<20} | {'Recall':<10} | {'Precision':<10}  | {'F1-Score':<10}")
        print("-" * 65)
        print(f"{'0 (Healthy)':<20} | {test_recall[0]:.4f}      | {test_precision[0]:.4f}       | {test_f1_scores[0]:.4f}")
        print(f"{'1 (Mild/Mod)':<20} | {test_recall[1]:.4f}      | {test_precision[1]:.4f}       | {test_f1_scores[1]:.4f}")
        print(f"{'2 (Sev/Prolif)':<20} | {test_recall[2]:.4f}      | {test_precision[2]:.4f}       | {test_f1_scores[2]:.4f}")
        print("-" * 65)
        print(f"Test Avg Sick F1: {test_avg_sick_f1:.4f}")
        
        # 6. Save results
        result = {
            'noise_std': noise_std,
            'accuracy': test_overall_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1_scores,
            'avg_sick_f1': test_avg_sick_f1,
            'preds': test_preds,
            'labels': test_labels
        }
        all_results.append(result)
        
        # 7. Plot confusion matrix
        cm = confusion_matrix(result['labels'], result['preds'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
        plt.title(f'Confusion Matrix (Noise std={noise_std})\nTest Avg Sick F1: {result["avg_sick_f1"]:.4f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    # 8. Print summary table
    print("\n" + "="*70)
    print("Gaussian Noise Test Results Summary")
    print("="*70)
    print(f"{'Noise Std':<15} {'Accuracy':<10} {'F1-Healthy':<12} {'F1-Mild/Mod':<14} {'F1-Sev/Prolif':<15} {'Avg Sick F1':<12}")
    print("-" * 80)
    print(f"{'0.000':<15} {clean_result['accuracy']:<10.4f} {clean_result['f1'][0]:<12.4f} "
          f"{clean_result['f1'][1]:<14.4f} {clean_result['f1'][2]:<15.4f} {clean_result['avg_sick_f1']:<12.4f}")
    for result in all_results:
        noise_std = result['noise_std']
        f1_scores = result['f1']
        print(f"{noise_std:<15.3f} {result['accuracy']:<10.4f} {f1_scores[0]:<12.4f} "
              f"{f1_scores[1]:<14.4f} {f1_scores[2]:<15.4f} {result['avg_sick_f1']:<12.4f}")
    
    print("\n" + "="*70)
    print("Gaussian Noise Test Completed!")
    print("="*70)
    return all_results


# ====================================================
# Define function to add Gaussian blur
# ====================================================

class AddGaussianBlur:

   
    def __init__(self, sigma=0.1, kernel_size=None):
        if kernel_size is None:
            # Auto-calculate: kernel_size = 2 * ceil(3 * sigma) + 1
            self.kernel_size = 2 * math.ceil(3 * sigma) + 1
            self.kernel_size = max(3, self.kernel_size)  # Minimum 3
        else:
            self.kernel_size = kernel_size
            
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        self.sigma = sigma

        
    def __call__(self, tensor):
        # tensor shape: [C, H, W], value range: [0, 1]
        
        blurred = TF.gaussian_blur(tensor, kernel_size=self.kernel_size, sigma=self.sigma)
        return blurred
    
    def __repr__(self):
        return f"{self.__class__.__name__}(sigma={self.sigma}, kernel_size={self.kernel_size})"

# ====================================================
# Blur Test Function 
# ====================================================

def test_blur(clean_result, blur_levels, model, test_df):
    model.eval()
    all_results = []
    print("\n" + "="*70)
    print("Starting Gaussian Blur Test")
    print("="*70)

    # ====================================================
    #Test model performance under different Gaussian blur levels
    # ====================================================

    for sigma in blur_levels: 
        # 1. Create transform with blur
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            AddGaussianBlur(sigma=sigma),  #Call function to add blur
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # 2. Create DataLoader with blur transform
        test_dataset = APTOSFastDataset(test_df, CONFIG['image_dir'], transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # 3. Inference on blurred images
        test_preds, test_labels = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(test_loader, desc="Testing...", leave=True):
                imgs = imgs.to(CONFIG['device'])
                outputs = model(imgs)
                _, p = torch.max(outputs, 1)
                test_preds.extend(p.cpu().numpy())
                test_labels.extend(lbls.numpy())
        
        # 4. Calculate metrics
        test_metrics = precision_recall_fscore_support(test_labels, test_preds, 
                                                       labels=[0, 1, 2], zero_division=0)
        test_precision, test_recall, test_f1_scores = test_metrics[0], test_metrics[1], test_metrics[2]
        test_avg_sick_f1 = (test_f1_scores[1] + test_f1_scores[2]) / 2.0
        test_overall_acc = np.mean(np.array(test_preds) == np.array(test_labels))
        
        # 5. Print results
        print(f"\nGaussian Blur Level: sigma={sigma}")
        print(f"Test Overall Acc: {test_overall_acc:.4f}")
        print("-" * 65)
        print(f"{'Class':<20} | {'Recall':<10} | {'Precision':<10}  | {'F1-Score':<10}")
        print("-" * 65)
        print(f"{'0 (Healthy)':<20} | {test_recall[0]:.4f}      | {test_precision[0]:.4f}       | {test_f1_scores[0]:.4f}")
        print(f"{'1 (Mild/Mod)':<20} | {test_recall[1]:.4f}      | {test_precision[1]:.4f}       | {test_f1_scores[1]:.4f}")
        print(f"{'2 (Sev/Prolif)':<20} | {test_recall[2]:.4f}      | {test_precision[2]:.4f}       | {test_f1_scores[2]:.4f}")
        print("-" * 65)
        print(f"Test Avg Sick F1: {test_avg_sick_f1:.4f}")
        
        # 6. Save results
        result = {
            'sigma': sigma,
            'accuracy': test_overall_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1_scores,
            'avg_sick_f1': test_avg_sick_f1,
            'preds': test_preds,
            'labels': test_labels
        }
        all_results.append(result)
        
        # 7. Plot confusion matrix
        cm = confusion_matrix(result['labels'], result['preds'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
        plt.title(f'Confusion Matrix (Blur sigma={sigma})\nTest Avg Sick F1: {result["avg_sick_f1"]:.4f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    # 8. Print summary table
    print("\n" + "="*70)
    print("Gaussian Blur Test Results Summary")
    print("="*70)
    print(f"{'Sigma':<15} {'Accuracy':<10} {'F1-Healthy':<12} {'F1-Mild/Mod':<14} {'F1-Sev/Prolif':<15} {'Avg Sick F1':<12}")
    print("-" * 80)
    print(f"{'0.000':<15} {clean_result['accuracy']:<10.4f} {clean_result['f1'][0]:<12.4f} "
          f"{clean_result['f1'][1]:<14.4f} {clean_result['f1'][2]:<15.4f} {clean_result['avg_sick_f1']:<12.4f}")
    for result in all_results:
        sigma = result['sigma']
        f1_scores = result['f1']
        print(f"{sigma:<15.3f} {result['accuracy']:<10.4f} {f1_scores[0]:<12.4f} "
              f"{f1_scores[1]:<14.4f} {f1_scores[2]:<15.4f} {result['avg_sick_f1']:<12.4f}")
    
    print("\n" + "="*70)
    print("Gaussian Blur Test Completed!")
    print("="*70)
    
    return all_results

# ====================================================
# Execute Tests
# ====================================================

# Print system information
print(f"Device: {CONFIG['device']}")
print(f"Model: {CONFIG['model_name']}")
print(f"Test samples: {len(test_df)}")
print(f"Batch size: {CONFIG['batch_size']}")

# ====================================================
# Execute Clean Data Tests
# ====================================================

# Clean data test (baseline)
print("\n" + "="*70)
print("CLEAN DATA TEST (BASELINE)")
print("="*70)

# 1. Inference on clean test set
test_preds, test_labels = [], []
with torch.no_grad():
    for imgs, lbls in tqdm(test_loader, desc="Testing clean data set...", leave=True):
        imgs = imgs.to(CONFIG['device'])
        outputs = model(imgs)
        _, p = torch.max(outputs, 1)
        test_preds.extend(p.cpu().numpy())
        test_labels.extend(lbls.numpy())

# 2. Calculate metrics for clean data
test_metrics = precision_recall_fscore_support(test_labels, test_preds, labels=[0, 1, 2], zero_division=0)
test_precision, test_recall, test_f1_scores = test_metrics[0], test_metrics[1], test_metrics[2]
test_avg_sick_f1 = (test_f1_scores[1] + test_f1_scores[2]) / 2.0
test_overall_acc = np.mean(np.array(test_preds) == np.array(test_labels))

# 3. Save clean data test results
clean_result = {
    'type': 'clean',
    'accuracy': test_overall_acc,
    'precision': test_precision,
    'recall': test_recall,
    'f1': test_f1_scores,
    'avg_sick_f1': test_avg_sick_f1,
    'preds': test_preds,
    'labels': test_labels
}

# 4. Print clean data test report
print(f"\nClean Data Test")
print(f"Test Overall Acc: {test_overall_acc:.4f}")
print("-" * 65)
print(f"{'Class':<20} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
print("-" * 65)
print(f"{'0 (Healthy)':<20} | {test_recall[0]:.4f}      | {test_precision[0]:.4f}         | {test_f1_scores[0]:.4f}")
print(f"{'1 (Mild/Mod)':<20} | {test_recall[1]:.4f}      | {test_precision[1]:.4f}         | {test_f1_scores[1]:.4f}")
print(f"{'2 (Sev/Prolif)':<20} | {test_recall[2]:.4f}      | {test_precision[2]:.4f}         | {test_f1_scores[2]:.4f}")
print("-" * 65)
print(f"Test Avg Sick F1: {test_avg_sick_f1:.4f}")
print("#"*65)

# 5. Plot confusion matrix for clean data tset
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
plt.title(f'Confusion Matrix (Clean Data)\nTest Avg Sick F1: {test_avg_sick_f1:.4f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# ====================================================
# Execute Robustness Tests
# ====================================================


# Define the noise level we want to test
noise_levels=[0.01, 0.03, 0.05, 0.07, 0.10]
# Execute the Gaussian noise tset
all_noise_results = test_noisy(clean_result, noise_levels, model, test_df)

# Define the blur level we want to test
blur_levels = [0.1, 0.2, 0.4, 0.6, 0.8]
# Execute the Gaussian blur tset
all_blur_results = test_blur(clean_result, blur_levels, model, test_df)


