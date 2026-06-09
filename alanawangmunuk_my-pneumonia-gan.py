 #安装必要库（Kaggle已预装大部分，只需额外安装pydicom）
!pip install pydicom --quiet

# === 导入所有需要的库 ===
import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 进度条工具

# === 设置Kaggle专用路径 ===
# 输入路径（数据集位置）
INPUT_DIR = "/kaggle/input/rsna-pneumonia-detection-challenge"
# 输出路径（处理后的文件保存位置）
OUTPUT_DIR = "/kaggle/working"

# === 1. 提取分类标签 ===
print("步骤1/5: 处理标签...")
df_class = pd.read_csv(os.path.join(INPUT_DIR, "stage_2_detailed_class_info.csv"))

# 筛选有效类别
valid_classes = ['Normal', 'Lung Opacity']
df_valid = df_class[df_class['class'].isin(valid_classes)].copy()

# 创建标签映射
df_valid['label'] = df_valid['class'].map({'Normal': 0, 'Lung Opacity': 1})

# 添加DICOM文件路径
df_valid['dcm_path'] = df_valid['patientId'].apply(
    lambda x: os.path.join(INPUT_DIR, "stage_2_train_images", f"{x}.dcm")
)

# 检查文件是否存在
df_valid['exists'] = df_valid['dcm_path'].apply(os.path.exists)
print(f"有效图像数量: {df_valid['exists'].sum()}/{len(df_valid)}")

# 保存预处理标签
df_valid[['patientId', 'dcm_path', 'label']].to_csv(
    os.path.join(OUTPUT_DIR, 'preprocessed_labels.csv'), index=False
)

# === 2. DICOM转图像 + 预处理 ===
print("\n步骤2/5: 转换DICOM文件...")
os.makedirs(os.path.join(OUTPUT_DIR, "processed_images"), exist_ok=True)

def preprocess_dicom(dcm_path, output_size=256):
    """处理DICOM文件的函数"""
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(float)
    
    # 归一化到0-255
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = img.astype(np.uint8)
    
    # 转换为3通道
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    
    # 调整尺寸
    h, w = img.shape[:2]
    scale = output_size / max(h, w)
    new_h, new_w = int(h*scale), int(w*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # 边缘填充
    pad_h = (output_size - new_h) // 2
    pad_w = (output_size - new_w) // 2
    img_padded = cv2.copyMakeBorder(
        img_resized, 
        pad_h, output_size - new_h - pad_h, 
        pad_w, output_size - new_w - pad_w, 
        cv2.BORDER_CONSTANT, 
        value=0
    )
    return img_padded

# 批量处理所有图像（使用进度条）
for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid)):
    if not row['exists']: 
        continue
    
    output_path = os.path.join(OUTPUT_DIR, "processed_images", f"{row['patientId']}.png")
    # 如果文件已处理则跳过
    if not os.path.exists(output_path):
        try:
            img = preprocess_dicom(row['dcm_path'])
            Image.fromarray(img).save(output_path)
        except Exception as e:
            print(f"处理 {row['patientId']} 失败: {str(e)}")

# === 3. 数据分析 ===
print("\n步骤3/5: 数据分析...")
# 类别分布
plt.figure(figsize=(10,5))
df_valid['label'].value_counts().plot.pie(
    autopct='%1.1f%%', 
    labels = ['Normal', 'Pneumonia'],
    colors=['lightgreen', 'lightcoral']
)
plt.title("Class Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.jpg'))
plt.show()

# === 4. 划分数据集 ===
print("\n步骤4/5: 划分数据集...")
train_df, test_df = train_test_split(
    df_valid, 
    test_size=0.2, 
    stratify=df_valid['label'],
    random_state=42
)
val_df, test_df = train_test_split(
    test_df, 
    test_size=0.5, 
    stratify=test_df['label'],
    random_state=42
)

print(f"训练集: {len(train_df)} 张")
print(f"验证集: {len(val_df)} 张")
print(f"测试集: {len(test_df)} 张")

# 保存划分结果
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_split.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val_split.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_split.csv'), index=False)

# === 5. 可视化样本 ===
print("\n步骤5/5: 生成样本图像...")
def plot_samples(df, title):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        sample = df.iloc[i]
        img_path = os.path.join(OUTPUT_DIR, "processed_images", f"{sample['patientId']}.png")
        img = Image.open(img_path)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"{sample['class']} (ID: {sample['patientId']})")
        axes[i].axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.jpg"))

plot_samples(train_df[train_df['label']==0], "Example of Normal Sample")
plot_samples(train_df[train_df['label']==1], "Example of pneumonia sample")

print("\n=== 预处理完成！ ===")
print(f"处理后的图像保存在: {os.path.join(OUTPUT_DIR, 'processed_images')}")
print(f"标签文件保存在: {OUTPUT_DIR}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.nn.utils import spectral_norm
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========== 超参数配置 ==========
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
LATENT_DIM = 128
N_CRITIC = 3
LAMBDA_GP = 100
IMG_SIZE = 128
CHANNELS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 模型架构 ==========
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.conv(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = IMG_SIZE // 4
        self.l1 = nn.Sequential(
            spectral_norm(nn.Linear(LATENT_DIM, 128 * self.init_size ** 2))
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 64, 3, stride=1, padding=1)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, CHANNELS, 3, stride=1, padding=1)),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(CHANNELS, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, 4, stride=1, padding=0))
        )

    def forward(self, img):
        return self.model(img)

# ========== 关键函数 ==========
def compute_gradient_penalty(D, real_samples, fake_samples, lambda_gp):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

def save_image(tensor, filename, nrow=8, padding=2):
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, normalize=True)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

# ========== 数据准备 ==========
def prepare_data():
    data_dir = "/kaggle/working/processed_images"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs("/kaggle/working/samples", exist_ok=True)
    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
    
    # 示例：从Kaggle数据集解压（根据实际情况修改）
    if len(os.listdir(data_dir)) == 0:
        print("数据目录为空，请确保已添加数据")
        print("示例解压命令（取消注释使用）：")
        print("# !unzip /kaggle/input/your-dataset.zip -d /kaggle/working/processed_images")
        return False
    return True

# ========== 训练流程 ==========
def train():
    if not prepare_data():
        return

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    class XRayDataset(torch.utils.data.Dataset):
        def __init__(self, img_dir, transform=None):
            self.img_dir = img_dir
            self.transform = transform
            self.img_list = [f for f in os.listdir(img_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
            if not self.img_list:
                raise ValueError(f"目录 {img_dir} 中未找到有效图像")
            
        def __len__(self):
            return len(self.img_list)
            
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_list[idx])
            try:
                with Image.open(img_path) as img:
                    image = img.convert('L')
                    if self.transform:
                        image = self.transform(image)
                    return image
            except Exception as e:
                print(f"加载 {img_path} 失败: {str(e)}")
                return torch.zeros(1, IMG_SIZE, IMG_SIZE)
    
    try:
        dataset = XRayDataset("/kaggle/working/processed_images", transform)
        print(f"成功加载 {len(dataset)} 张图像")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=2, pin_memory=True)
    
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=3e-5, betas=(0.5, 0.9))
    
    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, real_imgs in enumerate(progress_bar):
            real_imgs = real_imgs.to(DEVICE)
            
            # 训练判别器
            optimizer_D.zero_grad()
            z = torch.randn(real_imgs.size(0), LATENT_DIM).to(DEVICE)
            fake_imgs = generator(z).detach()
            
            gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, LAMBDA_GP)
            d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs)) + gp
            d_loss.backward()
            optimizer_D.step()
            
            # 训练生成器
            if i % N_CRITIC == 0:
                optimizer_G.zero_grad()
                gen_imgs = generator(z)
                g_loss = -torch.mean(discriminator(gen_imgs))
                g_loss.backward()
                optimizer_G.step()
            
            progress_bar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}",
                "GP": f"{gp.item():.4f}",
                "λ_GP": LAMBDA_GP
            })

        # 保存检查点
        if epoch % 2 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch
            }, f"/kaggle/working/checkpoints/epoch_{epoch}.pth")
            
            # 生成示例图像
            with torch.no_grad():
                test_z = torch.randn(16, LATENT_DIM).to(DEVICE)
                gen_imgs = generator(test_z)
                save_image(gen_imgs, f"/kaggle/working/samples/epoch_{epoch}.png")
                
                # 实时显示最新生成图像
                plt.figure(figsize=(10,10))
                plt.imshow(utils.make_grid(gen_imgs, nrow=4, padding=2, normalize=True).permute(1,2,0).cpu())
                plt.axis('off')
                plt.title(f"Epoch {epoch} Generated Samples")
                plt.show()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"训练中断: {str(e)}")
        if 'generator' in locals():
            torch.save(generator.state_dict(), "/kaggle/working/emergency_generator.pth")
            print("已保存紧急备份模型")


!ls /kaggle/working/checkpoints/  
!ls /kaggle/working/  # 查看是否有generator.pth


# 创建最终模型目录
!mkdir -p /kaggle/working/final_model

# 复制最新检查点并重命名
!cp /kaggle/working/checkpoints/epoch_48.pth /kaggle/working/final_model/model.pth

# 生成元数据文件
with open('/kaggle/working/final_model/dataset-metadata.json', 'w') as f:
    f.write("""{
    "title": "RSNA_Pneumonia_GAN_Final_Model",
    "id": "alanawangmunuk/pneumonia-gan-final",
    "licenses": [{
        "name": "CC0-1.0"
    }]
}""")

# 验证
!ls -lh /kaggle/working/final_model


!stat /kaggle/working/final_model/model.pth

