import pandas as pd
train_df = pd.read_pickle('/kaggle/input/mnist-rotation/train.pkl')
test_df = pd.read_pickle('/kaggle/input/mnist-rotation/test.pkl')


train_df


test_df


import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import rotate as scipy_rotate


def show_image(image, title=None):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    if title is not None:
        plt.title(title)
    plt.show()

def rotate(img: np.ndarray, angle: int) -> np.ndarray:
    if not (-120 <= angle <= 120):
        raise ValueError("Angle must be between -120 and 120 degrees.")
    
    rotated_img = scipy_rotate(
        img,
        angle=angle,
        reshape=False,      # сохраняем размер 28x28
        order=1,            # билинейная интерполяция для аккуратного поворота
        mode='constant',    # пиксели вне исходного изображения заполняются константой
        cval=0.0            # заполняем черным (0)
    )
    return rotated_img


class NoizeGenerator:
    def __init__(
        self,
        discrete_noise_proba=0.02,
        beta_alpha=0.3,
        beta_beta=0.3,
        gaussian_sigma=0.0,
        shift_prob=1.0,
        seed=None,
    ):
        """
        discrete_noise_proba: вероятность заменить пиксель значением из бета-распределения
        beta_alpha, beta_beta: параметры бета-распределения (бимодальность при < 1)
        gaussian_sigma: стандартное отклонение для нормального шума
        shift_prob: вероятность случайного сдвига изображения на 1 пиксель
        """
        self.discrete_noise_proba = discrete_noise_proba
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.gaussian_sigma = gaussian_sigma
        self.shift_prob = shift_prob
        self.rng = np.random.default_rng(seed)

    def apply_beta_noise(self, img):
        mask = self.rng.random(img.shape) < self.discrete_noise_proba
        beta_noise = (
            self.rng.beta(self.beta_alpha, self.beta_beta, size=img.shape) * 255
        )
        noisy_img = img.copy()
        noisy_img[mask] = beta_noise[mask]
        return noisy_img

    def apply_gaussian_noise(self, img):
        if self.gaussian_sigma > 0:
            noise = self.rng.normal(loc=0.0, scale=self.gaussian_sigma, size=img.shape)
            img = img + noise
            img = np.clip(img, 0, 255)
        return img

    def apply_random_shift(self, img):
        direction = self.rng.choice(["up", "down", "left", "right"])
        shifted = np.zeros_like(img)

        if direction == "up":
            shifted[:-1, :] = img[1:, :]
        elif direction == "down":
            shifted[1:, :] = img[:-1, :]
        elif direction == "left":
            shifted[:, :-1] = img[:, 1:]
        elif direction == "right":
            shifted[:, 1:] = img[:, :-1]

        return shifted

    def transform_image(self, image):
        tmp_image = image.astype(np.float32)
        tmp_image = self.apply_beta_noise(tmp_image)
        tmp_image = self.apply_gaussian_noise(tmp_image)

        if self.rng.random() < self.shift_prob:
            tmp_image = self.apply_random_shift(tmp_image)
        return tmp_image.astype(np.uint8)

    def transform_dataset(self, X):
        n_samples = X.shape[0]
        X_aug = np.zeros_like(X)

        for i in range(n_samples):
            img = self.transform_image(X[i].reshape(28, 28))
            X_aug[i] = img.flatten()
        return X_aug
    

noize_gen = NoizeGenerator(
    discrete_noise_proba=0.2,
    beta_alpha=0.3,
    beta_beta=0.3,
    gaussian_sigma=40,
    shift_prob=1.0,
    seed=1,
)


allowed_angles = np.arange(-120, 121, 30)
val = len(train_df)
for i in range(val):
    if i%1000 == 0:
        print(i)
    for angle in allowed_angles:
        train_df.loc[len(train_df)] = [noize_gen.transform_image(rotate(train_df.loc[i,'image'], angle)),train_df.loc[i,'label'],angle]


import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn


class CustomDataset(Dataset):
    def __init__(self,images,labels,angles):
        self.images = images
        self.labels = labels
        self.angles = angles
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()  # float32
            if img.ndim == 2:
                img = img.unsqueeze(0)  # (1, H, W)
            elif img.ndim == 3:
                img = img.permute(2, 0, 1)
        label = torch.tensor(self.labels[idx],dtype = torch.long)
        angle = torch.tensor(self.angles[idx], dtype = torch.long)
        return img,label,angle


angle_list = sorted(train_df['angle'].unique())  # e.g. [-120, -90, ..., 120]
angle2class = {angle: idx for idx, angle in enumerate(angle_list)}
train_df['angle_class'] = train_df['angle'].map(angle2class)


data = CustomDataset(train_df['image'],train_df['label'],train_df['angle_class'])
dataloader = DataLoader(data,batch_size = 8, shuffle = True)


import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_labels=10, num_angles=9):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (64, 7, 7)
        )

        self.flatten = nn.Flatten()
        self.image_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.label_embed = nn.Embedding(num_labels, 32)

        self.fc = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_angles)
        )

    def forward(self, x_img, x_label):
        x = self.conv(x_img)
        x = self.flatten(x)
        x = self.image_fc(x)

        label_embed = self.label_embed(x_label)
        x = torch.cat([x, label_embed], dim=1)
        x = self.fc(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_labels=10, num_angles=9).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels, angle_classes in dataloader:
        images = images.to(device).float()
        labels = labels.to(device)
        angle_classes = angle_classes.to(device)
        optimizer.zero_grad()
        outputs = model(images, labels)  # (batch, 12)
        loss = criterion(outputs, angle_classes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")


class CustomDataset1(Dataset):
    def __init__(self,images,labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()  # float32
            if img.ndim == 2:
                img = img.unsqueeze(0)  # (1, H, W)
            elif img.ndim == 3:
                img = img.permute(2, 0, 1)
        label = torch.tensor(self.labels[idx],dtype = torch.long)
        return img,label


data_test = CustomDataset1(test_df['image'],test_df['label'])
dataloader_test = DataLoader(data_test,batch_size = 8, shuffle = False)


angle_classes = sorted(train_df['angle'].unique())  # [0, 30, ..., 330]
angle_to_idx = {angle: idx for idx, angle in enumerate(angle_classes)}
idx_to_angle = {idx: angle for angle, idx in angle_to_idx.items()}


pred = []
model.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        images = images.to(device).float()
        labels = labels.to(device)

        logits = model(images, labels)
        pred_class = torch.argmax(logits, dim=1)
        pred_angles = [idx_to_angle[i.item()] for i in pred_class]
        for i in pred_angles:
            pred.append(i)


sample_submission = pd.read_csv('/kaggle/input/mnist-rotation/sample_submission.csv')


submission = pd.DataFrame({
    'ID': sample_submission['ID'],
    'angle': pred
})


submission


submission.to_csv('submission.csv',index = False)

