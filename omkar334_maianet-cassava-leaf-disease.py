import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),  # Resize to input size of MaiaNet
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flipping
        transforms.RandomVerticalFlip(p=0.5),  # Vertical flipping
        transforms.ToTensor(),  # Convert to tensor before adding noise
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),  # Add Gaussian noise
        transforms.Lambda(lambda x: transforms.functional.erase(x, i=0, j=0, h=50, w=50, v=0.0)),  # Add cutout
    ]
)

# df = pd.read_csv("train.csv")
df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

print(df.label.value_counts())
balanced_df = pd.DataFrame()

for label in df["label"].unique():
    label_df = df[df["label"] == label]
    if len(label_df) > 1000:
        _, sampled_df = train_test_split(label_df, test_size=1000, random_state=42, stratify=label_df["label"])
    balanced_df = pd.concat([balanced_df, sampled_df])


class Dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_id = self.dataframe.iloc[idx]["image_id"]
        label = self.dataframe.iloc[idx]["label"]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        # Move tensors to GPU if available
        image = image.to(device)
        label = torch.tensor(label, device=device)
        return image, label


train_df, temp_df = train_test_split(balanced_df, test_size=0.3, random_state=42, stratify=balanced_df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

train_dataset = Dataset(train_df)
test_dataset = Dataset(test_df)
val_dataset = Dataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)


import torch
import torch.cuda.amp as amp  # For mixed precision training
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.amp import GradScaler, autocast


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr=0.2, num_epochs=80):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Enable cudnn benchmarking for better performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.lr = lr

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.96)
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # Move loss function to GPU

        # Initialize mixed precision training
        self.scaler = torch.amp.GradScaler('cuda')

        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for images, labels in pbar:
            # Clear GPU cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Use mixed precision training
            with amp.autocast('cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Scale the loss and perform backprop
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Delete unnecessary tensors
            del outputs, loss

        self.scheduler.step()

        return total_loss / len(self.train_loader)

    @torch.no_grad()  # More efficient than with torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with amp.autocast('cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Clean up GPU memory
            del outputs, loss

        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = self.calculate_metrics(all_preds, all_labels)

        return avg_loss, metrics

    @staticmethod
    def calculate_metrics(predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def print_metrics(metrics, phase):
        print(f"\n{phase} Metrics:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        print("-" * 50)

    def train(self):
        try:
            for epoch in range(self.num_epochs):
                train_loss = self.train_epoch(epoch)
                val_loss, val_metrics = self.validate()

                print(f"\nEpoch {epoch + 1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
                self.print_metrics(val_metrics, "Validation")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    # Save model state to CPU to avoid GPU memory issues
                    self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

                # Print GPU memory usage if available
                if torch.cuda.is_available():
                    print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            # Save the current best model if training is interrupted
            if self.best_model_state is not None:
                torch.save(self.best_model_state, "interrupted_model.pt")

    def test(self):
        # Load best model state back to GPU
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
        test_loss, test_metrics = self.validate()
        print("\nBest Model Performance on Test Set:")
        self.print_metrics(test_metrics, "Test")



import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import torch
import torch.nn as nn
from torchvision.models import ResNet


def get_freq_indices(method):
    assert method in ["top1", "top2", "top4", "top8", "top16", "top32", "bot1", "bot2", "bot4", "bot8", "bot16", "bot32", "low1", "low2", "low4", "low8", "low16", "low32"]
    num_freq = int(method[3:])
    if "top" in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif "low" in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif "bot" in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method="top16"):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer("weight", self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, "x must been 4 dimensions, but got " + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part : (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaiaNet(nn.Module):
    def __init__(self, num_classes):
        super(MaiaNet, self).__init__()
        self.head = HeadBlock(3, 64)  # Input: 448×448×3 -> 112×112×64
        self.anti_aliasing_1 = AntiAliasingBlock(64, 64, downsample=False)  # 112×112×64 -> 112×112×64
        self.maia_1 = MaiaBlock(64, 256)  # 112×112×64 -> 112×112×256
        self.anti_aliasing_2 = AntiAliasingBlock(256, 512, downsample=True)  # 112×112×256 -> 56×56×512
        self.maia_2 = MaiaBlock(512, 512)  # 56×56×512 -> 56×56×512
        self.anti_aliasing_3 = AntiAliasingBlock(512, 1024, downsample=True)  # 56×56×512 -> 28×28×1024
        self.maia_3 = MaiaBlock(1024, 1024)  # 28×28×1024 -> 28×28×1024
        self.maia_4 = MaiaBlock(1024, 2048, downsample=True)  # 14×14×2048 -> 14×14×2048
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Converts 14x14x2048 to 1x1x2048
        self.fc = nn.Linear(2048, num_classes)  # Fully connected layer (2048 -> num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, verbose=False):
        if verbose:
            print("Input:", x.shape)
        x = self.head(x)
        if verbose:
            print("Head:", x.shape)
        x = self.anti_aliasing_1(x)
        if verbose:
            print("Anti-aliasing 1:", x.shape)
        x = self.maia_1(x)
        if verbose:
            print("MAIA 1:", x.shape)
        x = self.anti_aliasing_2(x)
        if verbose:
            print("Anti-aliasing 2:", x.shape)
        x = self.maia_2(x)
        if verbose:
            print("MAIA 2:", x.shape)
        x = self.anti_aliasing_3(x)
        if verbose:
            print("Anti-aliasing 3:", x.shape)
        x = self.maia_3(x)
        if verbose:
            print("MAIA 3:", x.shape)
        x = self.maia_4(x)
        if verbose:
            print("MAIA 4:", x.shape)

        x = self.global_pool(x)  # Shape: (batch_size, 2048, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (batch_size, 2048)
        x = self.fc(x)  # Shape: (batch_size, num_classes)

        return x


class HeadBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeadBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class MultiAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiAttention, self).__init__()

        # https://github.com/hujie-frank/SENet/blob/master/README.md
        self.se = SELayer(in_channels, reduction=16)

        # https://github.com/cfzd/FcaNet/blob/master/model/fcanet.py
        self.fca = MultiSpectralAttentionLayer(in_channels, 7, 7, reduction=16, freq_sel_method="top16")

    def forward(self, x):
        x = self.se(x)
        x = self.fca(x)
        return x


class AntiAliasingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(AntiAliasingBlock, self).__init__()

        self.downsample = downsample

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.down_conversion = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        stride = 2 if self.downsample else 1
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.ma = MultiAttention(out_channels)
        self.ibn = nn.InstanceNorm2d(out_channels)

        self.residual_conv = None
        if in_channels != out_channels or downsample:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.residual_conv = None

    def forward(self, x):
        out = self.block1(x)
        out = self.down_conversion(out)
        out = self.block2(out)
        out = self.ma(out)
        if self.residual_conv:
            x = self.residual_conv(x)
        out = out + x
        out = self.ibn(out)
        out = F.relu(out)
        return out


class MaiaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(MaiaBlock, self).__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        self.ma = MultiAttention(out_channels)
        self.ibn = nn.InstanceNorm2d(out_channels)

        if in_channels != out_channels or downsample:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.residual_conv = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.ma(out)

        if self.residual_conv:
            x = self.residual_conv(x)
        out = out + x
        out = self.ibn(out)
        out = F.relu(out)
        return out


if __name__ == "__main__":
    model = MaiaNet(num_classes=5).to(device)
    x = torch.randn(1, 3, 448, 448).to(device)
    output = model(x)
    print(output.shape)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm


trainer = Trainer(model, train_loader, test_loader, val_loader)


trainer.train()

