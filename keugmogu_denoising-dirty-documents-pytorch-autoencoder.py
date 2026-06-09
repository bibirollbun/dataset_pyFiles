!unzip /kaggle/input/denoising-dirty-documents/train.zip -d /content/denoising_data 
!unzip /kaggle/input/denoising-dirty-documents/test.zip -d /content/denoising_data
!unzip /kaggle/input/denoising-dirty-documents/train_cleaned.zip -d /content/denoising_data


import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2
import math
import torchvision.transforms.functional as TF
import torch.nn.functional as F


#GPU setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


test_dir = "/content/denoising_data/test"
train_dir = "/content/denoising_data/train"
train_cleaned_dir = "/content/denoising_data/train_cleaned"


def load_img_from_folder(folder):
    images = [] # list initialization named images
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"): # PNG file only
            img_path = os.path.join(folder, filename) # image path == folder(directory) + filename
            img = cv2.imread(img_path) # Read file thorugh OpenCV(BGR type)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR -> grayscale
            images.append(img)
            filenames.append(filename)

    # Sort images and filenames based on the filnames
    _, sorted_images = zip(*sorted(zip(filenames, images)))
    return sorted_images

# load images

train_imgs = load_img_from_folder(train_dir)
train_cleaned_imgs = load_img_from_folder(train_cleaned_dir)
test_imgs = load_img_from_folder(test_dir)


# Check if there are different shapes of images in each variable.
for img in train_imgs:
    if img.shape != (420,540):
        display(img)
        break


for img in test_imgs:
    if img.shape != (420,540):
        display(img)
        break


for img in train_cleaned_imgs:
    if img.shape != (420,540):
        display(img)
        break


np.unique([img.shape for img in train_imgs])


np.unique([img.shape for img in test_imgs])


np.unique([img.shape for img in train_cleaned_imgs])


def pad_image(img_list, target_shape):

    # target height, width
    target_h, target_w = target_shape

    # search and store
    pad_list = []
    for img in img_list:
        # height, width
        h, w = img.shape

        # top, bottom, left, right pad
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left

        # pad apply to the image
        padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        pad_list.append(padded_img)

    return pad_list



train_pad = pad_image(train_imgs, (420, 540))
test_pad = pad_image(test_imgs, (420, 540))
train_cleaned_pad = pad_image(train_cleaned_imgs, (420, 540))


# make sure padded data and original data match with their length
len(train_imgs), len(train_pad), len(test_imgs), len(test_pad), len(train_cleaned_imgs), len(train_cleaned_pad)


# size check
np.unique([img.shape for img in train_pad]), np.unique([img.shape for img in test_pad]), np.unique([img.shape for img in train_cleaned_pad])


class Pair_Dataset(Dataset):
    # Recieves train, train_cleaned(as target) and transform
    def __init__(self, train, train_cleaned, transform=None):
        self.train = train
        self.train_cleaned = train_cleaned
        self.transform = transform

    # Length of either train or train_cleaned
    def __len__(self):
        return len(self.train)

    # Access specific data using an index
    def __getitem__(self, index):
        o_img = self.train[index]
        c_img = self.train_cleaned[index]

        # apply transform if provided
        if self.transform:
            o_img = self.transform(o_img)
            c_img = self.transform(c_img)

        return o_img, c_img, index


class Test_Dataset(Dataset):
    def __init__(self, test_pad, transforms=None):
        self.test = test_pad
        self.transform = transforms
    def __len__(self):
        return len(self.test)
    def __getitem__(self, index):
        t_img = self.test[index]

        if self.transform:
            t_img = self.transform(t_img)

        return t_img


# Define the transformation pipeline
transforms = v2.Compose(
    [
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True) # sclae=True will apply normalization
    ]
)
X_train, X_val, y_train, y_val = train_test_split(train_pad, train_cleaned_pad, train_size=0.2, random_state=42)

train_dataset = Pair_Dataset(X_train, y_train, transforms)
val_dataset = Pair_Dataset(X_val, y_val, transforms)
test_dataset = Test_Dataset(test_pad, transforms)

train_loader = DataLoader(train_dataset, batch_size=9, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=9, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=9, shuffle=False)


for train, cleaned, idx in train_loader:
    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    for i in range(3):
        axes[i, 0].imshow(train[i].permute(1,2,0).cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cleaned[i].permute(1,2,0).cpu().numpy(), cmap='gray')
        axes[i, 1].set_title(f"Cleaned {i+1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
    break


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder -> Downsample the image by applying convolution operations through multiple layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 420x540 -> 210x270
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 210x270 -> 105x135
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 105x135 -> 52x67
            nn.ReLU()
        )

        # Decoder -> Reconstruct the original image by reversing the convolution operations applied during encoding (Upsample)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0), # 52x67 -> 105x135
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 105x135 -> 210x270
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 210x270 -> 420x540
            nn.Sigmoid() # Sigmoid operation applied for the outputs which are normalized values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


!pip install torchinfo


from torchinfo import summary

model = AutoEncoder()

summary(model, input_size=(1, 28, 28))


# mse will be calculated within the train_model function
# i_max = bit-size of the images (8 bits = 0 to 255)
def psnr(mse):
    i_max = 255.0
    psnr_value = 10 * torch.log10(i_max**2/mse)
    return psnr_value


# average of mse, rmse, psnr need to be calculated through each batch
def train_model(model, data_loader, criterion, optimizer, epochs=5, device=device):
    model.train()

    for epoch in range(epochs):
        reconst = []
        all_indices = []

        train_loss = 0.0
        rmse_loss = 0.0
        total_psnr = 0.0

        for train, target, idx in data_loader:
            train, target = train.to(device), target.to(device)

            optimizer.zero_grad()

            # forward
            outputs = model(train)
            loss = criterion(outputs, target) # MSELoss
            if isinstance(criterion, nn.MSELoss):
                rmse = torch.sqrt(loss)
                rmse_loss += rmse.mean().item()
                psnr_values = psnr(loss)
                total_psnr += psnr_values.mean().item()

            # backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            reconst.append(outputs)
            all_indices.append(idx)

        print(f"Epoch: {epoch+1}/{epochs}, Train Loss(avg): {train_loss/len(data_loader):.4f}, RMSE(avg): {rmse_loss/len(data_loader):.4f}, PSNR(avg): {total_psnr/len(data_loader):.4f}")

    reconst = torch.cat(reconst, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    return reconst, all_indices


def val_test_model(model, data_loader, criterion, optimizer, device=device):
    model.eval()

    reconst = []
    all_indices = []

    val_loss = 0.0
    rmse_loss = 0.0
    total_psnr = 0.0

    with torch.no_grad():
        for val, target, _ in data_loader:
            val, target = val.to(device), target.to(device)

            # forward
            outputs = model(val)
            loss = criterion(outputs, target) # MSELoss

            if isinstance(criterion, nn.MSELoss):
                rmse = torch.sqrt(loss)
                rmse_loss += rmse.mean().item()
                psnr_values = psnr(loss)
                total_psnr += psnr_values.mean().item()

            val_loss += loss.item()

            reconst.append(outputs)

    print(f"Val Loss(avg): {val_loss/len(data_loader):.4f}, RMSE(avg): {rmse_loss/len(data_loader):.4f}, PSNR(avg): {total_psnr/len(data_loader):.4f}")

    reconst = torch.cat(reconst, dim=0)

    return reconst, val_loss/len(data_loader)


def test_model(model, data_loader, criterion, device=device):
    model.eval()

    reconst = []

    test_loss = 0.0
    rmse_loss = 0.0
    total_psnr = 0.0

    with torch.no_grad():
        for test in test_loader:
            test = test.to(device)
            outputs = model(test)

            loss = criterion(outputs, test)

            if isinstance(criterion, nn.MSELoss):
                rmse = torch.sqrt(loss)
                rmse_loss += rmse
                psnr_values = psnr(loss)
                total_psnr += psnr_values.item()

            test_loss += loss.item()

            reconst.append(outputs)
    print(f"Test Loss(avg): {test_loss/len(data_loader):.4f}, RMSE(avg): {rmse_loss/len(data_loader):.4f}, PSNR(avg): {total_psnr/len(data_loader):.4f}")
    reconst = torch.cat(reconst, dim=0)

    return reconst


def get_optimizer(model, optimizer_type, LR):
    if optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=LR)
    elif optimizer_type == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=LR, momentum=0.9, alpha=0.99)
    elif optimizer_type == 'Adadelta':
        return optim.Adadelta(model.parameters(), lr=LR)
    elif optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=LR)
    elif optimizer_type == 'NAdam':
        return optim.NAdam(model.parameters(), lr=LR)
    else:
        raise ValueError("Invalid optimizer type")


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        print(f"initialized weights for {m}")


def run_experiments(model, data_loader, optimizer_type, LR=0.001):
    model.apply(init_weight)

    optimizer = get_optimizer(model, optimizer_type, LR=LR)

    criterion = nn.MSELoss()

    print(f"Training with {optimizer_type} optimizer!")
    t_output, train_idx = train_model(model, data_loader, criterion, optimizer, epochs)
    v_output, _ = val_test_model(model, val_loader, criterion, optimizer)

    return t_output, v_output, train_idx


def idx_trace_visualization(dataset, output, idx=None):
    fig, axes = plt.subplots(3, 3, figsize=(8, 9))
    for i in range(3):
        if idx is not None:
            index = idx[i].item()
        else:
            index = i
        original_image = dataset[index][0].permute(1,2,0).cpu().numpy().squeeze()
        reconstructed_image = output[index].cpu().detach().numpy().squeeze()
        cleaned_image = dataset[index][1].permute(1,2,0).cpu().numpy().squeeze()

        axes[i, 0].imshow(original_image, cmap='gray')
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed_image, cmap='gray')
        axes[i, 1].set_title(f"Reconstuctured {i+1}")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(cleaned_image, cmap='gray')
        axes[i, 2].set_title(f"Cleaned {i+1}")
        axes[i, 2].axis('off')

    print("Train_Visualization_______")
    plt.tight_layout()
    plt.show()


def test_visualization(dataset, output):
    fig, axes = plt.subplots(3, 2, figsize=(8, 9))


    for i in range(3):
        axes[i, 0].imshow(test_dataset[i].permute(1,2,0).cpu().detach().numpy().squeeze(), cmap='gray')
        axes[i, 0].set_title(f"Test {i+1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(output[i].permute(1,2,0).cpu().detach().numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title(f"test_output {i+1}")
        axes[i, 1].axis('off')

    print("Test_Visualization_______")
    plt.tight_layout()
    plt.show()


epochs = 20
proto_model=AutoEncoder().to(device)

t_output, v_output, train_idx = run_experiments(proto_model, train_loader, 'SGD', LR=1.1)


idx_trace_visualization(train_dataset, t_output, train_idx)


idx_trace_visualization(val_dataset, v_output)


criterion=nn.MSELoss()

test_output = test_model(proto_model, test_loader, criterion, device)
test_visualization(test_dataset, test_output)


proto_model=AutoEncoder().to(device)

t_output, v_output, train_idx = run_experiments(proto_model, train_loader, 'RMSprop', LR=0.001)


idx_trace_visualization(train_dataset, t_output, train_idx)


idx_trace_visualization(val_dataset, v_output)


proto_model=AutoEncoder().to(device)

t_output, v_output, train_idx = run_experiments(proto_model, train_loader, 'Adadelta', LR=1.5)


idx_trace_visualization(train_dataset, t_output, train_idx)


idx_trace_visualization(val_dataset, v_output)


criterion=nn.MSELoss()

test_output = test_model(proto_model, test_loader, criterion, device)
test_visualization(test_dataset, test_output)


proto_model=AutoEncoder().to(device)

t_output, v_output, train_idx = run_experiments(proto_model, train_loader, 'Adam')


idx_trace_visualization(train_dataset, t_output, train_idx)


idx_trace_visualization(val_dataset, v_output)


criterion=nn.MSELoss()

test_output = test_model(proto_model, test_loader, criterion, device)
test_visualization(test_dataset, test_output)


proto_model=AutoEncoder().to(device)

t_output, v_output, train_idx = run_experiments(proto_model, train_loader, 'NAdam')


idx_trace_visualization(train_dataset, t_output, train_idx)


idx_trace_visualization(val_dataset, v_output)


criterion=nn.MSELoss()

test_output = test_model(proto_model, test_loader, criterion, device)
test_visualization(test_dataset, test_output)


# Dropout applied
class AutoEncoder2(nn.Module):
    def __init__(self,d_prob=0.5):
        super(AutoEncoder2, self).__init__()

        # Encoder -> Downsample the image by applying convolution operations through multiple layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # 420x540 -> 210x270
            nn.ReLU(),
            # The data is already normalized, batch normalization for input layer may not be neccesary
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 210x270 -> 105x135
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=d_prob),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 105x135 -> 52x67
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=d_prob)
        )

        # Decoder -> Reconstruct the original image by reversing the convolution operations applied during encoding (Upsample)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0), # 52x67 -> 105x135
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 105x135 -> 210x270
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 210x270 -> 420x540
            nn.Sigmoid() # Sigmoid operation applied for the outputs, which are normalized values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = AutoEncoder2(d_prob=0.5).to(device)

summary(model, input_size=(9,1, 420, 540))


class EarlyStopiing:
    def __init__(self, patience=5, delta=0.0002, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_weights = None
        self.path = path

    def step(self, current_loss, model):
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.best_model_weights = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered! Val_loss is not improving for {self.patience} times")
                torch.save(self.best_model_weights, self.path)
                return True
        return False



# average of mse, rmse, psnr need to be calculated through each batch
def train_final_model(model, data_loader, criterion, optimizer, epochs=5, device=device, scheduler=None, patience=10):
    model.train()

    early_stop = EarlyStopiing(patience=patience)

    for epoch in range(epochs):
        reconst = []
        all_indices = []

        train_loss = 0.0
        rmse_loss = 0.0
        total_psnr = 0.0

        for train, target, idx in data_loader:
            train, target = train.to(device), target.to(device)

            optimizer.zero_grad()

            # forward
            outputs = model(train)
            loss = criterion(outputs, target) # MSELoss

            if isinstance(criterion, nn.MSELoss):
                rmse = torch.sqrt(loss)
                rmse_loss += rmse.mean().item()
                psnr_values = psnr(loss)
                total_psnr += psnr_values.mean().item()

            # backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            reconst.append(outputs)
            all_indices.append(idx)
        print(f"Epoch: {epoch+1}/{epochs}, Train Loss(avg): {train_loss/len(data_loader):.4f}, RMSE(avg): {rmse_loss/len(data_loader):.4f}, PSNR(avg): {total_psnr/len(data_loader):.4f}")

        # Scheduler application
        if scheduler is not None:
            model.eval()
            v_output, val_loss = val_test_model(model, val_loader, criterion, optimizer, device)
            print("________________________\n")
            scheduler.step(val_loss)
            model.train()

        # Early Stopping application
        model.eval()
        stop = early_stop.step(val_loss, model)
        model.train()

        if stop:
            break

    model.load_state_dict(early_stop.best_model_weights)
    reconst = torch.cat(reconst, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    return reconst, v_output, all_indices


epochs = 200
model = AutoEncoder2(d_prob=0.2).to(device)
optimizer = get_optimizer(model, "Adam", LR=0.003)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)

model.apply(init_weight)

t_output, v_output, train_idx = train_final_model(model, train_loader, criterion, optimizer, epochs, device, scheduler)


idx_trace_visualization(train_dataset, t_output, train_idx)


idx_trace_visualization(val_dataset, v_output)


criterion = nn.MSELoss()
test_output = test_model(model, test_loader, criterion)


test_visualization(test_dataset, test_output)


import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

target_size = (420, 540)
def compute_padding(original_size, target_size):
    o_h, o_w = original_size
    t_h, t_w = target_size

    pad_top = (t_h - o_h) // 2 if t_h > o_h else 0
    pad_bottom = t_h - o_h - pad_top if t_h > o_h else 0
    pad_left = (t_w - o_w) // 2 if t_w > o_w else 0
    pad_right = t_w - o_w - pad_left if t_w > o_w else 0

    return pad_top, pad_bottom, pad_left, pad_right


def remove_padding(reconst, original_size, target_size):
    o_h, o_w = original_size
    pad_top, _, pad_left, _ = compute_padding(original_size, target_size)

    return reconst[:, pad_top:pad_top+o_h, pad_left:pad_left+o_w]


test_file_paths = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)


submission_data = []

for file_path in test_file_paths:
    image_id = os.path.splitext(os.path.basename(file_path))[0]

    original_image = Image.open(file_path).convert('L')
    o_w, o_h = original_image.size
    original_size = (o_h, o_w)

    idx = test_file_paths.index(file_path)
    reconst = test_output[idx]

    cropped_pred = remove_padding(reconst, original_size, target_size)

    pred_np = cropped_pred.squeeze().cpu().numpy()

    for row in range(o_h):
        for col in range(o_w):
            pixel_id = f"{image_id}_{row+1}_{col+1}"
            pixel_value = pred_np[row, col]
            submission_data.append((pixel_id, pixel_value))


submission_file = 'submission.csv'
with open(submission_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'value'])
    for row in submission_data:
        writer.writerow(row)

print(f"Submission file '{submission_file}' has been created!")


len(submission_data)


test_file_paths




