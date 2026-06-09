import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image  # 添加这一行，用于使用 Image 类
%matplotlib inline

# 定义相关参数
img_width_inception = 299
img_height_inception = 299
img_width_resnet = 224
img_height_resnet = 224
batch_size = 32
epochs = 120
num_classes = 5

# 读取 CSV 文件
train_csv_path = '/kaggle/input/pre-aptos/train.csv'
test_csv_path = '/kaggle/input/pre-aptos/val.csv'
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# 定义数据生成函数
def data_generator(df, batch_size):
    num_samples = len(df)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_df = df.iloc[offset:offset + batch_size]
            batch_images_inception = []
            batch_images_resnet = []
            batch_labels = []
            for index, row in batch_df.iterrows():
                img_path = row['path']
                img_inception = load_img(img_path, target_size=(img_width_inception, img_height_inception))
                img_inception = img_inception.resize((img_width_inception, img_height_inception))
                img_inception = img_to_array(img_inception) / 255.0
                img_resnet = load_img(img_path, target_size=(img_width_resnet, img_height_resnet))
                img_resnet = img_resnet.resize((img_width_resnet, img_height_resnet))
                img_resnet = img_to_array(img_resnet) / 255.0
                label = row['label']
                label = tf.keras.utils.to_categorical(label, num_classes=num_classes)
                batch_images_inception.append(img_inception)
                batch_images_resnet.append(img_resnet)
                batch_labels.append(label)
            batch_images_inception = np.array(batch_images_inception)
            batch_images_resnet = np.array(batch_images_resnet)
            batch_labels = np.array(batch_labels)
            batch_images_inception = tf.convert_to_tensor(batch_images_inception)
            batch_images_resnet = tf.convert_to_tensor(batch_images_resnet)
            batch_labels = tf.convert_to_tensor(batch_labels)
            yield (batch_images_inception, batch_images_resnet), batch_labels

# 生成训练和测试数据生成器
train_generator = data_generator(train_df, batch_size)
test_generator = data_generator(test_df, batch_size)

# 加载预训练的 Inception V3 和 ResNet50 模型，去除最后一层
inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width_inception, img_height_inception, 3))
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width_resnet, img_height_resnet, 3))

# 冻结预训练模型的所有层
for layer in inception_base.layers:
    layer.trainable = False
for layer in resnet_base.layers:
    layer.trainable = False

# 定义输入层
input_inception = Input(shape=(img_width_inception, img_height_inception, 3))
input_resnet = Input(shape=(img_width_resnet, img_height_resnet, 3))

# 提取特征
inception_features = inception_base(input_inception)
resnet_features = resnet_base(input_resnet)

# 展平特征
inception_flat = Flatten()(inception_features)
resnet_flat = Flatten()(resnet_features)

# 拼接特征
concatenated_features = Concatenate()([inception_flat, resnet_flat])

# 添加全连接层进行分类
x = Dense(512, activation='relu')(concatenated_features)
output = Dense(num_classes, activation='softmax')(x)

# 定义最终模型
model = Model(inputs=[input_inception, input_resnet], outputs=output)

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 定义早停回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 自定义回调函数，每 n 个周期保存一次模型权重和训练结果
class CustomCheckpoint(Callback):
    def __init__(self, save_freq):
        super(CustomCheckpoint, self).__init__()
        self.save_freq = save_freq
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        # 保存训练结果
        self.history['loss'].append(logs['loss'])
        self.history['accuracy'].append(logs['accuracy'])
        self.history['val_loss'].append(logs['val_loss'])
        self.history['val_accuracy'].append(logs['val_accuracy'])

        if (epoch + 1) % self.save_freq == 0:
            # 保存模型权重
            model.save_weights(f'model_epoch_{epoch + 1}.weights.h5')
            print(f'Epoch {epoch + 1}: 模型权重已保存到 model_epoch_{epoch + 1}.weights.h5')
            # 保存训练结果
            with open(f'training_history_epoch_{epoch + 1}.json', 'w') as f:
                json.dump(self.history, f)
            print(f'Epoch {epoch + 1}: 训练结果已保存到 training_history_epoch_{epoch + 1}.json')

# 设置每 5 个周期保存一次模型权重
custom_checkpoint = CustomCheckpoint(save_freq=5)

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(test_df) // batch_size,
    callbacks=[early_stopping, custom_checkpoint]
)

# 打印训练过程
for epoch in range(len(history.history['loss'])):
    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train Loss: {history.history["loss"][epoch]:.4f}, Train Accuracy: {history.history["accuracy"][epoch]:.4f}')
    print(f'Val Loss: {history.history["val_loss"][epoch]:.4f}, Val Accuracy: {history.history["val_accuracy"][epoch]:.4f}')

# 保存最终模型权重
model.save_weights('final_model_weights.weights.h5')

# 保存训练历史信息
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

# 定义可视化函数
def visualize_training_results():
    try:
        with open('training_history.json', 'r') as f:
            history = json.load(f)
        plt.figure(figsize=(12, 4))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # 保存可视化结果
        plt.savefig('training_results_visualization.png')
        plt.show()
    except FileNotFoundError:
        print("训练历史文件未找到，请先训练模型。")

# 调用可视化函数进行可视化
visualize_training_results()


import os
import shutil

# 定义要删除的文件路径
file_path = '/kaggle/working/processed_train_images'
# 检查文件是否存在
if os.path.exists(file_path):
    # 删除文件
    shutil.rmtree(file_path)
    print(f"文件 {file_path} 已成功删除。")
else:
    print(f"文件 {file_path} 不存在。")


import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 定义相关参数
img_width_inception = 299
img_height_inception = 299
img_width_resnet = 224
img_height_resnet = 224
num_classes = 5

# 构建模型
input_inception = Input(shape=(img_width_inception, img_height_inception, 3))
input_resnet = Input(shape=(img_width_resnet, img_height_resnet, 3))

inception_base = InceptionV3(weights='imagenet', include_top=False)(input_inception)
resnet_base = ResNet50(weights='imagenet', include_top=False)(input_resnet)

inception_flat = Flatten()(inception_base)
resnet_flat = Flatten()(resnet_base)

concatenated_features = Concatenate()([inception_flat, resnet_flat])
x = Dense(512, activation='relu')(concatenated_features)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=[input_inception, input_resnet], outputs=output)

# 加载模型权重
model.load_weights('/kaggle/working/model_epoch_5.weights.h5')

def preprocess_image(img_path):
    """
    对输入的图片进行预处理
    :param img_path: 图片的路径
    :return: 处理后的 Inception 和 ResNet 输入
    """
    img_inception = load_img(img_path, target_size=(img_width_inception, img_height_inception))
    img_inception = img_to_array(img_inception) / 255.0
    img_inception = np.expand_dims(img_inception, axis=0)

    img_resnet = load_img(img_path, target_size=(img_width_resnet, img_height_resnet))
    img_resnet = img_to_array(img_resnet) / 255.0
    img_resnet = np.expand_dims(img_resnet, axis=0)

    return img_inception, img_resnet

# 替换为你要测试的图片路径
image_path = '/kaggle/input/22222/colored_images/data1/2/31411_left.png'
img_inception, img_resnet = preprocess_image(image_path)

# 进行预测
predictions = model.predict([img_inception, img_resnet])
predicted_class = np.argmax(predictions)

print(f"预测的类别是: {predicted_class}")
    


import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



import os
import sys
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from PIL import Image

# 定义 FocalLoss 类
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.5, gamma=3):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return focal_loss.mean()

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(0.1, 0.1, 0.1),
                                     transforms.RandomRotation(degrees=15),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # 加载训练集和验证集
    train_csv_path = '/kaggle/input/pre-aptos/train.csv'
    val_csv_path = '/kaggle/input/pre-aptos/val.csv'
    train_dataset = CustomDataset(csv_file=train_csv_path, transform=data_transform["train"])
    val_dataset = CustomDataset(csv_file=val_csv_path, transform=data_transform["val"])

    # 获取类别列表并生成类别字典
    class_list = sorted(train_dataset.data['label'].unique())
    cla_dict = {i: class_list[i] for i in range(len(class_list))}
    cla_dict = {int(key): int(value) if isinstance(value, np.int64) else value for key, value in cla_dict.items()}
    # 将类别字典写入JSON文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 300
    save_path = '/kaggle/working/AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            predict_y = torch.max(outputs, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_accurate = train_acc / len(train_loader.dataset)
        train_loss = running_loss / train_steps
        train_losses.append(train_loss)
        train_accuracies.append(train_accurate)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_running_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_running_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / len(val_dataset)
        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accurate)

        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, train_loss, train_accurate, val_loss, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

    plt.figure(figsize=(12, 6))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # 保存可视化结果
    plt.savefig('/kaggle/working/training_results.png')
    plt.close()


if __name__ == '__main__':
    main()
    


import shutil
import os

# 定义要压缩的文件夹路径
folder_to_compress = '/kaggle/working/processed_train_images'
# 定义压缩文件的输出路径和名称
zip_file_path = '/kaggle/working/processed_train_images_zip'

# 压缩文件夹
shutil.make_archive(zip_file_path, 'zip', folder_to_compress)


import os
os.chdir('/kaggle/working')
print(os.getcwd())
print(os.listdir("/kaggle/working"))
from IPython.display import FileLink
FileLink('/kaggle/working/processed_train_images_zip.zip')


import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(12, 6))
    # 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# 保存可视化结果
plt.savefig('/kaggle/working/training_results.png')
plt.show()


import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

train_images_dir = '/kaggle/input/diabetic-retinopathy-resized/resized_train_cropped/resized_train_cropped'
output_dir = '/kaggle/working/processed_train_images'
os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpeg')]
batch_size = 5000  # 设置批次大小，可以根据内存情况调整

for i in tqdm(range(0, len(image_files), batch_size)):
    batch_files = image_files[i:i + batch_size]
    for img_file in batch_files:
        img_path = os.path.join(train_images_dir, img_file)
        image = cv2.imread(img_path)

        if image is not None:
            processed_image = load_ben_color(image)
            output_path = os.path.join(output_dir, img_file)
            # 使用PIL库来保存RGB图像
            img_to_save = Image.fromarray(processed_image.astype('uint8'), 'RGB')
            img_to_save.save(output_path)
        else:
            print(f"无法读取图像: {img_path}")

print("对train数据集所有图片的数据预处理并保存完成！")

