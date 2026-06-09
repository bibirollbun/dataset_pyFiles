import torch
from torch import nn
import torchvision
import os
from torch.utils.data import random_split


!pip install decord --quiet


from decord import VideoReader, cpu
import pandas as pd

def compute_video_lengths(csv_path, video_dir):
    df = pd.read_csv(csv_path)
    lengths = []
    for path in df['path']:
        vr = VideoReader(os.path.join(video_dir, path), ctx=cpu(0))
        lengths.append(len(vr))
    return lengths


lengths = compute_video_lengths('/kaggle/input/what-on-the-video/train.csv', '/kaggle/input/what-on-the-video/train')


import matplotlib.pyplot as plt

plt.hist(lengths, bins=30)
plt.xlabel("Number of frames in a video")
plt.ylabel("Number of videos")
plt.title("Video length distribution")
plt.show()


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


class_df = pd.read_csv('/kaggle/input/what-on-the-video/train.csv')
class_df[9:12]


class_df.loc[class_df['path'] == 'Oregon cumulous cloud over Cannon Beach_preview.mp4', 'labels'] = 'cloud, water'
class_df[9:12]


# caching video data is very important for speed-up
cache_dir = '/kaggle/working/cache/'


class VideoDataset(Dataset):
    def __init__(self, df, video_dir, transform=None, num_frames=16, cache_dir=cache_dir):
        self.df = df
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames
        self.cache_dir = cache_dir or os.path.join(video_dir, "_cached")

        os.makedirs(self.cache_dir, exist_ok=True)

        all_labels = set()
        for label_str in self.df['labels']:
            all_labels.update(label_str.split(", "))
        self.classes = sorted(list(all_labels))
        self.class2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx2class = {idx: cls for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def _get_cache_path(self, filename):
        base = os.path.splitext(os.path.basename(filename))[0]
        return os.path.join(self.cache_dir, f"{base}_{self.num_frames}.pt")

    def _extract_frames(self, video_file):
        vr = VideoReader(video_file, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(np.int32)
        else:
            indices = np.concatenate([
                np.arange(total_frames),
                np.full(self.num_frames - total_frames, total_frames - 1)
            ]).astype(np.int32)

        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]

        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames.permute(1, 0, 2, 3)])
            frames = frames.permute(1, 0, 2, 3)

        return frames  # [C, T, H, W]

    def __getitem__(self, idx):
        video_filename = self.df.iloc[idx]['path']
        label_str = self.df.iloc[idx]['labels']
        video_file = os.path.join(self.video_dir, video_filename)
        cache_file = self._get_cache_path(video_filename)

        if os.path.exists(cache_file):
            frames = torch.load(cache_file)
        else:
            frames = self._extract_frames(video_file)
            torch.save(frames, cache_file)

        # binary mask for classes
        target = torch.zeros(len(self.class2idx), dtype=torch.float32)
        for label in label_str.split(", "):
            target[self.class2idx[label]] = 1.0

        return frames, target



train_dataset = VideoDataset(
    df=class_df,
    video_dir='/kaggle/input/what-on-the-video/train',
    num_frames=16,
    transform=None,
    cache_dir=cache_dir
)


# making sure there are 9 classes
print(train_dataset.idx2class)


from matplotlib.animation import FuncAnimation

def animate_video_from_dataset(dataset, idx, interval=200):
    """
    Animates a video from dataset (doesn't actually animate in kaggle for some reason, maybe I did smt wrong).

    interval â€” interval between frames in ms
    """
    video, label = dataset[idx]  # video: [C, T, H, W]
    lbl = ', '.join([
        dataset.idx2class[idx] for idx in range(len(label)) if label[idx] == 1
    ])
    if isinstance(video, torch.Tensor):
        video = video.cpu()
    
    T = video.shape[1]
    frames = video.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
    frames = np.clip(frames, 0, 1)

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    plt.title(f"Video #{idx} â€” Label: {lbl}")
    plt.axis("off")

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        return [im]

    ani = FuncAnimation(fig, update, frames=T, interval=interval, blit=True)
    plt.show()



animate_video_from_dataset(train_dataset, 20, 10)


def pack_slowfast_pathway(frames, alpha=4):
    fast = frames  # original
    slow = frames[:, :, ::alpha, :, :]  # more rare
    return [slow, fast]


from tqdm import tqdm
import torch
import gc
from time import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class Trainer:
    def __init__(self, model, optimizer, criterion, device, save_path="best_model.pth", model_name='not_slowfast'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.model_name = model_name
        
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
    
        loop = tqdm(train_loader, desc="ğŸš‚ Training", leave=False)
        for videos, labels in loop:
            try:
                videos, labels = videos.to(self.device), labels.to(self.device)

                if self.model_name == "slowfast":
                    videos = pack_slowfast_pathway(videos)

                self.optimizer.zero_grad()
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
    
                loop.set_postfix(loss=loss.item())

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("âš ï¸� Out of memory! Clearing cache and skipping batch.")
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise e
    
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)

                if self.model_name == "slowfast":
                    videos = pack_slowfast_pathway(videos)

                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
                targets = labels.cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets)

        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        return avg_loss, f1

    def fit(self, train_loader, val_loader, num_epochs):
        start = time()
        for epoch in range(1, num_epochs + 1):
            epoch_start = time()
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_f1 = self.validate(val_loader)
    
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1s.append(val_f1)
    
            clear_output(wait=True)
            self._plot_live(epoch, num_epochs)
            epoch_finish = time()
            print(f"Epoch {epoch}/{num_epochs} (finished in {epoch_finish - epoch_start:.2f}s) | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}\n"
                  f"Time passed: {epoch_finish - start:.0f}s (Avg Time Per Epoch: {(epoch_finish - start) / epoch:.2f}s)"
                )
    
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print("ğŸ’¾ Saved new best model!")

    def _plot_live(self, current_epoch, total_epochs):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()



def predict_and_save(
    model, 
    test_dir, 
    output_csv, 
    idx2class, 
    model_weights_path=None,
    num_frames=32, 
    transform=None,
    threshold=0.5, 
    device='cuda',
    model_name='not_slowfast'
):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)) if model_name == 'slowfast' else transforms.Resize((112, 112)),
            transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])

    if model_weights_path is not None:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    model.to(device)

    results = []
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.mp4')])

    for idx, filename in enumerate(tqdm(test_files, desc="ğŸ”� Predicting")):
        video_path = os.path.join(test_dir, filename)

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(np.int32)
        else:
            indices = np.concatenate([
                np.arange(total_frames),
                np.full(num_frames - total_frames, total_frames - 1)
            ]).astype(np.int32)

        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]

        # âœ… ĞŸÑ€Ğ°Ğ²Ğ¸Ğ»ÑŒĞ½Ğ¾Ğµ Ğ¿Ñ€Ğ¸Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ğµ transform Ğ¿Ğ¾ ĞºĞ°Ğ´Ñ€Ğ°Ğ¼
        if transform:
            frames = frames.permute(1, 0, 2, 3)  # [T, C, H, W]
            frames = torch.stack([transform(frame) for frame in frames])  # [T, C, H, W]
            frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]

        frames = frames.unsqueeze(0).to(device)  # [1, C, T, H, W]

        if model_name == "slowfast":
            frames = pack_slowfast_pathway(frames)

        with torch.no_grad():
            logits = model(frames)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            predicted_indices = [i for i, p in enumerate(probs) if p >= threshold]
            
            if not predicted_indices:
                predicted_indices = [int(probs.argmax())]
            
            predicted_labels = [idx2class[i] for i in predicted_indices]

        label_str = ", ".join(predicted_labels)
        results.append((idx, filename, label_str))

    df = pd.DataFrame(results, columns=["index", "file_name", "label"])
    df.to_csv(output_csv, index=False)
    print(f"ğŸ“� Saved predictions to: {output_csv}")



!pip install pytorchvideo --quiet


from pytorchvideo.models.hub import slowfast_r50, i3d_r50, x3d_xs
from torchvision.models.video import mc3_18, r3d_18
import pytorchvideo.models.hub as hb


print(dir(hb))


def get_model(name: str, num_classes = 9, device = device):
    if name == 'slowfast':
        model = slowfast_r50(pretrained=True)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)

    elif name == 'i3d':
        model = i3d_r50(pretrained=True)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)

    elif name == 'x3d':
        model = x3d_xs(pretrained=True)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)
    elif name == 'r3d':
        model = r3d_18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'mc3':
        model = mc3_18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"name should be r3d, mc3, slowfast, i3d, or x3d")

    return model.to(device)


def run(model_name, num_frames, bs=8,lr=1e-4, num_epochs=15, split_percentage=0.9):
    frame_transform = transforms.Compose([
        transforms.Resize((112, 112)) if model_name != 'slowfast' else transforms.Resize((224, 224)),  # Ñ€Ğ°Ğ·Ğ¼ĞµÑ€, Ğ½Ğ° ĞºĞ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ¼ Ğ¾Ğ±ÑƒÑ‡Ğ°Ğ»Ñ�Ñ� r3d_18
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])
    train_dataset = VideoDataset(
        df=class_df,
        video_dir='/kaggle/input/what-on-the-video/train',
        transform=frame_transform,
        num_frames=num_frames,
        cache_dir=cache_dir + 'slowfast'
    )
    train_size = int(split_percentage * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    model = get_model(model_name)
    
    train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=bs, shuffle=False, num_workers=0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    trainer = Trainer(model, optimizer, criterion, device, 
                      save_path=f"/kaggle/working/{model_name}_frames{num_frames}_bs{bs}_lr{lr}.pth",
                     model_name=model_name)
    trainer.fit(train_loader, val_loader, num_epochs=num_epochs)
    return model


def run_and_save(model_name, num_frames, bs=8,lr=1e-4, num_epochs=15, split_percentage=0.9):
    model = run(
        model_name=model_name,
        num_frames=num_frames,
        bs=bs,
        lr=lr,
        num_epochs=num_epochs,
        split_percentage=split_percentage,
    )
    predict_and_save(
        model,
        "/kaggle/input/what-on-the-video/test",
        f"/kaggle/working/BEST_{model_name}_frames{num_frames}_bs{bs}_lr{lr}.csv",
        train_dataset.idx2class,
        model_weights_path = f"/kaggle/working/{model_name}_frames{num_frames}_bs{bs}_lr{lr}.pth",
        num_frames=num_frames,
        model_name=model_name
    )
    predict_and_save(
        model,
        "/kaggle/input/what-on-the-video/test",
        f"/kaggle/working/LAST_{model_name}_frames{num_frames}_bs{bs}_lr{lr}.csv",
        train_dataset.idx2class,
        # model_weights_path = f"/kaggle/working/{model_name}_frames{num_frames}_bs{bs}_lr{lr}.pth",
        num_frames=num_frames,
        model_name=model_name
    )


run_and_save(
    model_name='r3d',
    num_frames=32,
    bs=8,
    lr=1e-4,
    num_epochs=15,
    split_percentage=0.8
)


!rm -rf /kaggle/working/cache

