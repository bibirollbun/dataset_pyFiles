import os
import math
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from torchvision import models
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC


if not os.path.exists("/kaggle/working/birdclef-2025"):
    os.makedirs("/kaggle/working/birdclef-2025")
os.listdir('/kaggle/working')


class CFG:
    seed = 42
    img_size = [128, 384]
    batch_size = 64
    duration = 15  # 秒
    sample_rate = 32000
    audio_len = duration * sample_rate
    dim = sample_rate * 5
    nfft = 2028
    window = 2048
    hop_length = audio_len // (img_size[1] - 1)
    fmin = 20
    fmax = 16000
    epochs = 300  # 减少epoch便于测试
    # preset = 'efficientnetv2_b2_imagenet'
    BASE_PATH_work = '/kaggle/working/birdclef-2025'
    # model_weight = "/kaggle/input/best_model_weights.pth/pytorch/default/1/best_model_weights.pth"
    # BASE_PATH_input = '/kaggle/input/birdclef-2025'
    BASE_PATH_input = '/kaggle/input/birdclef-2025'
    augment = True
    class_names = sorted(os.listdir(f'{BASE_PATH_input}/train_audio/'))  # 确保路径正确
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v: k for k, v in label2name.items()}
    


class MixUp(nn.Module):
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha

    def forward(self, images, labels):
        if self.alpha <= 0 or torch.rand(1) > 0.35:
            return images, labels

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]

        mixed_images = lam * images + (1 - lam) * shuffled_images
        mixed_labels = torch.stack([
            lam * labels,
            (1 - lam) * shuffled_labels
        ], dim=1)
        return mixed_images, mixed_labels

class RandomMask(nn.Module):
    def __init__(self, height_ratio, width_ratio, is_time_mask=True):
        super().__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio
        self.is_time_mask = is_time_mask

    def forward(self, img):
        if torch.rand(1) > 0.35:
            return img

        c, h, w = img.shape
        if self.is_time_mask:
            mask_width = int(w * np.random.uniform(*self.width_ratio))
            mask_start = np.random.randint(0, w - mask_width)
            img[:, :, mask_start:mask_start + mask_width] = 0
        else:
            mask_height = int(h * np.random.uniform(*self.height_ratio))
            mask_start = np.random.randint(0, h - mask_height)
            img[:, mask_start:mask_start + mask_height, :] = 0
        return img
class AudioAugmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.augmenters = nn.ModuleList([
            MixUp(alpha=0.4),
            RandomMask((1.0, 1.0), (0.06, 0.12), True),
            RandomMask((0.06, 0.1), (1.0, 1.0), False)
        ])

    def forward(self, img, label):
        if img.ndim == 3 and img.shape[2] in [1, 3]:
            img = img.permute(2, 0, 1)

        for aug in self.augmenters:
            if isinstance(aug, MixUp):
                img, label = aug(img, label)
            else:
                img = aug(img)

        if img.shape[0] in [1, 3]:
            img = img.permute(1, 2, 0)
        return img, label


def build_decoder(with_labels=True):
    def get_audio(filepath: str) -> torch.Tensor:
        """
        PyTorch实现音频加载与预处理

        :param filepath: .ogg音频文件路径
        :return: 单声道音频张量，形状为 [num_samples]
        """
        # 读取.ogg文件
        waveform, sample_rate = torchaudio.load(filepath)

        # 单声道转换（如果音频是立体声，则取左声道）
        if waveform.shape[0] > 1:  # 检查是否为立体声
            waveform = waveform[0:1, :]  # 取第一个声道

        # 压缩维度并标准化
        waveform = waveform.squeeze(0)  # 移除通道维度，形状变为 [samples]
        return waveform



    def crop_or_pad(audio: torch.Tensor,
                    target_len: int,
                    pad_mode: str = "constant") -> torch.Tensor:
        """
        音频长度统一化 (PyTorch 实现)

        动态调整音频长度至固定长度：
        1. 音频过短：随机分配填充位置进行补零
        2. 音频过长：随机选取起始位置截取片段
        3. 支持多种填充模式（constant/reflect/replicate）

        参数：
            audio: 输入音频张量，形状为 [L]
            target_len: 目标长度（样本点数）
            pad_mode: 填充模式，支持 "constant"/"reflect"/"replicate"

        返回：
            调整后的音频张量，形状为 [target_len]
        """
        audio_len = audio.size(0)
        diff_len = abs(target_len - audio_len)

        # 短音频补零
        if audio_len < target_len:
            pad1 = torch.randint(0, diff_len + 1, ()).item()
            pad2 = diff_len - pad1
            audio = F.pad(audio, (pad1, pad2), mode=pad_mode)

        # 长音频随机截取
        elif audio_len > target_len:
            start_idx = torch.randint(0, diff_len + 1, ()).item()
            audio = audio[start_idx: start_idx + target_len]

        # 确保形状正确 [target_len]
        return audio.reshape(-1)[:target_len]

    def apply_preproc(spec: torch.Tensor) -> torch.Tensor:
        """
        频谱图标准化与归一化 (PyTorch 实现)

        功能说明：
        1. Z-Score标准化：消除量纲差异，使数据均值为0，标准差为1
        2. Min-Max归一化：压缩值域到[0,1]，提升模型训练稳定性
        3. 鲁棒性处理：对静音段等特殊情况降级处理

        参数：
            spec: 输入频谱图张量，形状任意

        返回：
            预处理后的张量，保持原始形状
        """
        # ================= Z-Score 标准化 =================
        mean = torch.mean(spec)
        std = torch.std(spec)

        # 处理零标准差情况（静音段）
        if torch.is_nonzero(std):
            spec = (spec - mean) / std
        else:  # 退化为中心化操作
            spec = spec - mean

        # ================= Min-Max 归一化 =================
        min_val = torch.min(spec)
        max_val = torch.max(spec)
        delta = max_val - min_val

        # 处理零值域情况（全零或全同值）
        if torch.is_nonzero(delta):
            spec = (spec - min_val) / delta
        else:  # 退化到零基线
            spec = spec - min_val

        return spec

    def get_target(target: torch.Tensor) -> torch.Tensor:
        """
        One-Hot 标签编码 (PyTorch 实现)

        功能：
        1. 将整数类别标签转换为 One-Hot 编码
        2. 确保输出形状为 [CFG.num_classes]
        3. 支持标量输入和批量输入（需与损失函数兼容）

        参数：
            target: 输入标签张量，形状为 []（标量）或 [batch_size]

        返回：
            One-Hot 编码张量，形状为 [CFG.num_classes] 或 [batch_size, CFG.num_classes]
        """
        # 确保输入为整型张量
        target = target.long()

        # 生成 One-Hot 编码
        one_hot = torch.nn.functional.one_hot(target, num_classes=CFG.num_classes)

        # 调整形状和数据类型
        return one_hot.reshape(-1, CFG.num_classes).float().squeeze()

    def decode(path: str) -> torch.Tensor:
        """
        频谱图生成与处理 (PyTorch 实现)

        功能：
        1. 加载音频并统一长度
        2. 生成Mel频谱图
        3. 标准化与归一化
        4. 三通道适配

        参数：
            path: 音频文件路径

        返回：
            3通道频谱图张量，形状为 [3, CFG.img_size[0], CFG.img_size[1]]
        """
        # 加载音频 (假设已实现 get_audio 返回单声道张量)
        audio = get_audio(path)  # 形状 [L]

        # 统一音频长度
        audio = crop_or_pad(audio, CFG.dim)  # 形状 [CFG.dim]

        # 生成Mel频谱图
        mel_spec_transform = MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=CFG.nfft,
            hop_length=CFG.hop_length,
            n_mels=CFG.img_size[0],
            power=2.0  # 与TensorFlow的MelSpectrogram默认参数对齐[7](@ref)
        )
        spec = mel_spec_transform(audio)  # 形状 [n_mels, time_steps]

        # 转换为对数刻度（模拟TensorFlow的return_decibel=True）
        spec = torchaudio.functional.amplitude_to_DB(spec, multiplier=20, amin=1e-5, db_multiplier=0)

        # 标准化与归一化
        spec = apply_preproc(spec)

        # 三通道适配 (模仿tf.tile)
        spec = spec.unsqueeze(0)  # 添加通道维度 [1, n_mels, time_steps]
        spec = spec.repeat(3, 1, 1)  # 复制为3通道 [3, n_mels, time_steps]

        # 调整形状至目标尺寸（需确保 hop_length 参数设置正确）
        spec = spec[..., :CFG.img_size[1]]  # 截断时间轴至目标长度
        return spec

    def decode_with_labels(path, label):
        label = get_target(label)
        return decode(path), label

    return decode_with_labels if with_labels else decode



class MixUp(nn.Module):
    """
    MixUp：以alpha=0.4的参数混合两个样本的频谱图和标签，创造新的训练样本
    """
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha

    def forward(self, images, labels):
        if self.alpha <= 0 or torch.rand(1) > 0.35:
            return images, labels

        # 生成混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)

        # 随机打乱样本顺序
        indices = torch.randperm(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]

        # 混合图像和标签
        mixed_images = lam * images + (1 - lam) * shuffled_images
        mixed_labels = torch.stack([
            lam * labels,
            (1 - lam) * shuffled_labels
        ], dim=1)

        return mixed_images, mixed_labels


class RandomMask(nn.Module):
    def __init__(self, height_ratio, width_ratio, is_time_mask=True):
        super().__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio
        self.is_time_mask = is_time_mask

    def forward(self, img):
        if torch.rand(1) > 0.35:
            return img

        # 频谱图维度处理（假设输入为[C, Freq, Time]）
        c, h, w = img.shape

        # 时间掩码（垂直条）
        if self.is_time_mask:
            mask_width = int(w * np.random.uniform(*self.width_ratio))
            mask_start = np.random.randint(0, w - mask_width)
            img[:, :, mask_start:mask_start + mask_width] = 0
        # 频率掩码（水平条）
        else:
            mask_height = int(h * np.random.uniform(*self.height_ratio))
            mask_start = np.random.randint(0, h - mask_height)
            img[:, mask_start:mask_start + mask_height, :] = 0

        return img


class AudioAugmenter(nn.Module):
    """可序列化的增强流水线"""

    def __init__(self):
        super().__init__()
        self.augmenters = nn.ModuleList([
            MixUp(alpha=0.4),
            RandomMask((1.0, 1.0), (0.06, 0.12), True),
            RandomMask((0.06, 0.1), (1.0, 1.0), False)
        ])

    def forward(self, img, label):
        # 统一维度格式为 [C, H, W]
        if img.ndim == 3 and img.shape[2] in [1, 3]:  # [H,W,C] -> [C,H,W]
            img = img.permute(2, 0, 1)

        for aug in self.augmenters:
            if isinstance(aug, MixUp):
                img, label = aug(img, label)
            else:
                img = aug(img)

        # 恢复原始维度格式
        if img.shape[0] in [1, 3]:  # [C,H,W] -> [H,W,C]
            img = img.permute(1, 2, 0)

        return img, label


def build_augmenter():
    """构建可序列化的增强器"""
    return AudioAugmenter()


    class ImageClassifier(nn.Module):
        def __init__(self, num_classes, preset='efficientnet_v2_s'):
            super().__init__()
            # 加载预训练主干网络
            self.backbone = models.efficientnet_v2_s(pretrained=True)
            in_features = self.backbone.classifier[1].in_features

            # 替换分类头
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes)
            )

        def forward(self, x):
            return self.backbone(x)


os.listdir("/kaggle/working")


df = pd.read_csv(f'{CFG.BASE_PATH_input}/train.csv')
df['filepath'] = CFG.BASE_PATH_input + '/train_audio/' + df.filename
df['target'] = df.primary_label.map(CFG.name2label)
df['filename'] = df.filepath.map(lambda x: x.split('/')[-1])
df['xc_id'] = df.filepath.map(lambda x: x.split('/')[-1].split('.')[0])
train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=CFG.seed)
train_df = train_df.reset_index(drop=True).copy()
valid_df = valid_df.reset_index(drop=True).copy()
print(f'{CFG.BASE_PATH_work}/train_df.csv')
train_df.to_csv(f'{CFG.BASE_PATH_work}/train_df.csv', index=False)
valid_df.to_csv(f'{CFG.BASE_PATH_work}/valid_df.csv', index=False)
# print(df)
# Display rwos
# df.head(2)
train_mel_df = []
valid_mel_df = []
train_labels = []
valid_labels = []
if not os.path.exists(f'{CFG.BASE_PATH_work}/train_mel_path.npy'):
    """训练集"""
    for (idx, row_i) in tqdm(train_df.iterrows(), total=train_df.shape[0], desc="训练集"):
        # print(row_i)
        train_label = row_i['target']
        file_path = row_i['filepath']
        train_mel_path = file_path.replace("train_audio", "train_mel")
        train_mel_path = train_mel_path.replace(CFG.BASE_PATH_input, CFG.BASE_PATH_work)
        train_mel_path = train_mel_path.replace(".ogg", ".npy")
        train_label = torch.tensor([int(train_label)])
        decode_with_labels = build_decoder()
        mel, target = decode_with_labels(file_path, train_label)
        train_labels.append(target)
        # torch.save(mel, mel_path)
        if not os.path.exists(os.path.dirname(train_mel_path)):
            os.makedirs(os.path.dirname(train_mel_path))
        np.save(train_mel_path, mel.numpy())
        train_mel_df.append([train_mel_path])
        # if idx == 3:
        #     break
    train_mel_df = pd.DataFrame(train_mel_df, columns=["mel_path"])
    train_mel_df.to_csv(f'{CFG.BASE_PATH_work}/train_mel_path.csv', index=None)
    train_labels = np.array(train_labels)
    np.save(f'{CFG.BASE_PATH_work}/train_labels.npy', train_labels)

if not os.path.exists(f'{CFG.BASE_PATH_work}/valid_mel_path.npy'):
    """测试集"""
    for (idx, row_i) in tqdm(valid_df.iterrows(), total=valid_df.shape[0], desc="测试集"):
        # print(row_i)
        valid_label = row_i['target']
        file_path = row_i['filepath']
        valid_mel_path = file_path.replace("train_audio", "valid_mel")
        valid_mel_path = valid_mel_path.replace(CFG.BASE_PATH_input, CFG.BASE_PATH_work)
        valid_mel_path = valid_mel_path.replace(".ogg", ".npy")
        valid_label = torch.tensor([int(valid_label)])
        decode_with_labels = build_decoder()
        mel, target = decode_with_labels(file_path, valid_label)
        valid_labels.append(target)
        # torch.save(mel, mel_path)
        if not os.path.exists(os.path.dirname(valid_mel_path)):
            os.makedirs(os.path.dirname(valid_mel_path))
        np.save(valid_mel_path, mel.numpy())
        valid_mel_df.append([valid_mel_path])
        # if idx == 3:
        #     break
    valid_mel_df = pd.DataFrame(valid_mel_df, columns=["mel_path"])
    valid_mel_df.to_csv(f'{CFG.BASE_PATH_work}/valid_mel_path.csv', index=None)
    valid_labels = np.array(valid_labels)
    np.save(f'{CFG.BASE_PATH_work}/valid_labels.npy', valid_labels)


# 初始化配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据路径
train_df = pd.read_csv(f'{CFG.BASE_PATH_work}/train_df.csv')
valid_df = pd.read_csv(f'{CFG.BASE_PATH_work}/valid_df.csv')

train_mel_path = pd.read_csv(f'{CFG.BASE_PATH_work}/train_mel_path.csv').to_numpy().ravel().tolist()
valid_mel_path = pd.read_csv(f'{CFG.BASE_PATH_work}/valid_mel_path.csv').to_numpy().ravel().tolist()

# train_mel_path = train_df['filepath'].to_numpy().ravel().tolist()
# valid_mel_path = valid_df['filepath'].to_numpy().ravel().tolist()

# 加载标签
train_labels = torch.LongTensor(np.load(f"{CFG.BASE_PATH_work}/train_labels.npy")).argmax(dim=1)
valid_labels = torch.LongTensor(np.load(f"{CFG.BASE_PATH_work}/valid_labels.npy")).argmax(dim=1)


# train_mel_path[0]
valid_mel_path[0]


class AudioDataset(Dataset):
    def __init__(self,
                 spec_paths: list,  # 新增频谱图路径参数
                 labels=None,
                 augment_fn=None):
        """
        改进后的音频数据集类
        :param spec_paths: 预先生成的梅尔频谱图路径列表
        :param labels: 标签列表（可选）
        :param augment_fn: 数据增强函数
        """
        self.spec_paths = spec_paths
        self.labels = labels
        self.augment_fn = augment_fn or None

    def _load_mel_spec(self, path: str) -> torch.Tensor:
        """加载预存的梅尔频谱图"""
        spec = torch.from_numpy(np.load(path)).float()

        # 通道处理（单通道转三通道）
        if spec.shape[0] == 1:
            spec = spec.repeat(3, 1, 1)
        return spec

    def __getitem__(self, idx):
        # 加载预存频谱
        spec = self._load_mel_spec(self.spec_paths[idx])
        label = self.labels[idx] if (self.labels is not None) else None
        # 应用数据增强
        if self.augment_fn:
            label = self.labels[idx] if (self.labels is not None) else None
            spec, label = self.augment_fn(spec, label)

        return (spec, label) if (self.labels is not None) else spec

    def __len__(self):
        return len(self.spec_paths)


def build_dataloader(paths, labels=None, batch_size=32,
                     augment_fn=None,
                     cache=False, augment=False, shuffle=2048, num_workers=0):
    """
    PyTorch数据加载管道
    :param shuffle: 设置为True启用随机洗牌，False则关闭（原TF的shuffle参数为缓冲区大小）
    """
    # 创建数据集实例
    dataset = AudioDataset(paths, labels, augment_fn if augment else None)

    # 缓存机制
    if cache:
        # 预加载所有数据到内存（适合小数据集）
        specs, labels = zip(*[dataset[i] for i in range(len(dataset))])
        dataset = TensorDataset(torch.stack(specs), torch.tensor(labels))

    # 构建DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle),  # 原TF的shuffle参数转换为布尔值
        num_workers=num_workers,  # 并行加载进程数（替代TF的AUTOTUNE）
        pin_memory=True,  # 加速GPU传输
        drop_last=True  # 对应原TF的drop_remainder
    )
    return loader


# 创建数据加载器
augmenter = AudioAugmenter()
train_loader = build_dataloader(train_mel_path, train_labels, CFG.batch_size, 
                               augment_fn=None, augment=False)
valid_loader = build_dataloader(valid_mel_path, valid_labels, CFG.batch_size)


# 初始化模型
model = ImageClassifier(CFG.num_classes).to(device)
# 加载自定义权重
# model.load_state_dict(torch.load(CFG.model_weight))


# 损失函数
class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.02):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets.to(logits.device).long()
        log_probs = torch.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



# 优化器与评估指标
optimizer = optim.Adam(model.parameters())
criterion = LabelSmoothCrossEntropy(smoothing=0.02)
auc_metric = MulticlassAUROC(num_classes=CFG.num_classes, average="macro").to(device)

# 学习率调度
def get_lr_scheduler(optimizer, batch_size=8, mode='cos', epochs=30):
    lr_start, lr_max, lr_min = 5e-5, 8e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep = 3, 0

    def lr_lambda(epoch):
        if epoch < lr_ramp_ep:
            return (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            return lr_max
        elif mode == 'cos':
            decay_total_epochs = epochs - lr_ramp_ep - lr_sus_ep + 3
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            return (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
    return LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(optimizer, CFG.batch_size)


# 训练过程
best_auc = 0
for epoch in trange(CFG.epochs, desc="Epochs"):
    # 训练阶段
    model.train()
    # for inputs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    model.eval()
    val_auc = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            auc_metric.update(outputs, labels)
        
        val_auc = auc_metric.compute()
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"{CFG.BASE_PATH_work}best_model_weights.pth")
        auc_metric.reset()
    
    scheduler.step()
    print(f"Epoch {epoch+1}/{CFG.epochs} | Val AUC: {val_auc:.4f} | Best AUC: {best_auc:.4f}")

print("训练完成!")













