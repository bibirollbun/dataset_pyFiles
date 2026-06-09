import os
from glob import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import torchvision.models as models

# 配置参数
IMG_SIZE = 224
COLOR_TYPE = 1  # 1表示灰度，3表示彩色
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集定义
class DrivingDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.images = []
        self.labels = []
        if mode == 'train':
            for class_id in range(NUM_CLASSES):
                files = glob(f'{root_dir}/imgs/train/c{class_id}/*.jpg')
                for file in files:
                    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE if COLOR_TYPE == 1 else cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    if COLOR_TYPE == 1:
                        img = np.expand_dims(img, axis=-1)
                    self.images.append(img)
                    self.labels.append(class_id)
        else:
            files = sorted(glob(f'{root_dir}/imgs/test/*.jpg'))
            for file in files:
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE if COLOR_TYPE == 1 else cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                if COLOR_TYPE == 1:
                    img = np.expand_dims(img, axis=-1)
                self.images.append(img)
                self.labels.append(-1)
        self.images = np.array(self.images, dtype=np.float32) / 255.0
        if COLOR_TYPE == 1:
            self.images = self.images.transpose((0, 3, 1, 2))
        else:
            self.images = self.images.transpose((0, 3, 1, 2))
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx])

# MobileNetV2模型定义
class MobileNetNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, color_type=COLOR_TYPE):
        super(MobileNetNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        # 若为灰度图，修改第一层输入通道
        if color_type == 1:
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # 修改最后分类层
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)

# 训练过程指标可视化
def plot_metrics(train_loss, val_loss, train_acc, val_acc, train_r2, val_r2, train_mse, val_mse, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(train_r2, label='Train R2')
    plt.plot(val_r2, label='Val R2')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()
    plt.title('Training and Validation R2')
    plt.savefig(os.path.join(save_dir, 'r2.png'))
    plt.close()

    plt.figure()
    plt.plot(train_mse, label='Train MSE')
    plt.plot(val_mse, label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Training and Validation MSE')
    plt.savefig(os.path.join(save_dir, 'mse.png'))
    plt.close()

# Grad-CAM热力图叠加显示
def show_gradcam_on_image(img, mask, alpha=0.5):
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    img = np.transpose(img, (1, 2, 0))
    img = np.uint8(255 * img)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    plt.figure(figsize=(4, 4))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title('Grad-CAM')
    plt.show()

# Grad-CAM生成函数
def generate_gradcam(model, input_tensor, target_class, conv_layer=None):
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    # 默认用最后一层卷积层
    if conv_layer is None:
        last_conv = model.model.features[-1][0]
    else:
        last_conv = conv_layer
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_f = last_conv.register_forward_hook(forward_hook)
    handle_b = last_conv.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    class_idx = target_class if target_class is not None else pred_class
    score = output[0, class_idx]
    model.zero_grad()
    score.backward()

    grads_val = gradients[0]
    activations_val = activations[0]
    weights = grads_val.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations_val).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_f.remove()
    handle_b.remove()
    return cam

# 训练与验证主流程
def train_model(model, train_loader, val_loader, epochs=10, save_dir='results'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_r2s, val_r2s = [], []
    train_mses, val_mses = [], []
    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_r2s.append(r2_score(all_labels, all_preds))
        train_mses.append(mean_squared_error(all_labels, all_preds))

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        val_imgs_list = []
        val_labels_list = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                val_imgs_list.append(imgs.cpu())
                val_labels_list.append(labels.cpu())
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_r2s.append(r2_score(all_labels, all_preds))
        val_mses.append(mean_squared_error(all_labels, all_preds))
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Train R2: {train_r2s[-1]:.4f} | Val R2: {val_r2s[-1]:.4f} | "
              f"Train MSE: {train_mses[-1]:.4f} | Val MSE: {val_mses[-1]:.4f}")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_MobileNet_model.pth'))
            print(f"Best model saved at epoch {epoch+1} with val_acc={val_acc:.4f}")

            # 只在保存最优模型时输出按类别准确率
            class_correct = [0 for _ in range(NUM_CLASSES)]
            class_total = [0 for _ in range(NUM_CLASSES)]
            for label, pred in zip(all_labels, all_preds):
                if 0 <= label < NUM_CLASSES:
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
            print("Per-class accuracy (only for best model):")
            for i in range(NUM_CLASSES):
                acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                print(f"  Class {i}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")

    # 随机选取10张验证集图片生成Grad-CAM热力图
    val_imgs_all = torch.cat(val_imgs_list, dim=0)
    val_labels_all = torch.cat(val_labels_list, dim=0)
    idxs = np.random.choice(val_imgs_all.shape[0], 10, replace=False)
    for idx in idxs:
        img = val_imgs_all[idx].numpy()
        label = val_labels_all[idx].item()
        cam = generate_gradcam(model, val_imgs_all[idx], target_class=label)
        show_gradcam_on_image(img, cam)
    return train_losses, val_losses, train_accs, val_accs, train_r2s, val_r2s, train_mses, val_mses

# 主函数
def main():
    root_dir = '../input/state-farm-distracted-driver-detection'
    dataset = DrivingDataset(root_dir, mode='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    print("Training MobileNetNet...")
    model = MobileNetNet().to(DEVICE)
    save_dir = 'results'
    train_losses, val_losses, train_accs, val_accs, train_r2s, val_r2s, train_mses, val_mses = train_model(
        model, train_loader, val_loader, EPOCHS, save_dir=save_dir)
    plot_metrics(train_losses, val_losses, train_accs, val_accs, train_r2s, val_r2s, train_mses, val_mses, save_dir=save_dir)

if __name__ == "__main__":
    main()

