import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm


import sys
!cp -r ../input/openai-clip/CLIP/CLIP-main /tmp/

# Kaggle likes to unpack .gz files in datasets... so we have to pack it back
!gzip -c /tmp/CLIP-main/clip/bpe_simple_vocab_16e6.txt > /tmp/CLIP-main/clip/bpe_simple_vocab_16e6.txt.gz
sys.path.append('/tmp/CLIP-main')


! pip install ftfy
! pip install hiddenlayer


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import re
from clip.simple_tokenizer import SimpleTokenizer
#import faiss
import matplotlib.pyplot as plt

%matplotlib inline


_tokenizer = SimpleTokenizer()

# Copied from https://github.com/openai/CLIP/blob/beba48f35392a73c6c47ae67ddffced81ad1916d/clip/clip.py#L164
# but with relaxed exception
def tokenize(texts, context_length: int = 77) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        n = min(len(tokens), context_length)
        result[i, :n] = torch.tensor(tokens)[:n]
        if len(tokens) > context_length:
            result[i, -1] = tokens[-1]

    return result


RE_EMOJI = re.compile(r"\\x[A-Za-z0-9./]+", flags=re.UNICODE)

def strip_emoji(text):
    return RE_EMOJI.sub(r'', text)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("../input/openai-clip/ViT-B-32.pt", device=device, jit=False)


from PIL import Image

# 打开图片
image = Image.open('/kaggle/input/ai-vs-human-generated-dataset/train_data/a6dcb93f596a43249135678dfcfc17ea.jpg')


image


from PIL import Image

# 打开图片
image = Image.open('/kaggle/input/ai-vs-human-generated-dataset/train_data/041be3153810433ab146bc97d5af505c.jpg')


np.array(image).shape


fake_text = ['It is a fake image', 'It is an unreal image', 'It is a fabricated image', 'It is a counterfeit image', 'It is an artificial image']
real_text = ['It is a real image', 'It is a genuine image', 'It is a true image', 'It is a real-life image','It is a factual image']


# fake_text = ['It is a fake image']
# real_text = ['It is a real image']


fake_text = tokenize(fake_text)
real_text = tokenize(real_text)


# fake_texts_features = torch.mean(model.encode_text(fake_text.to(device)),dim=0).unsqueeze(0)
# real_texts_features = torch.mean(model.encode_text(real_text.to(device)),dim=0).unsqueeze(0)
# text_features = torch.cat([fake_texts_features,real_texts_features],dim=0)


fake_texts_features = torch.mean(model.encode_text(fake_text.to(device)),dim=0).unsqueeze(0)





# Define paths to dataset files
path = '/kaggle/input/ai-vs-human-generated-dataset'
train_csv = '/kaggle/input/detect-ai-vs-human-generated-images/train.csv'
test_csv = '/kaggle/input/detect-ai-vs-human-generated-images/test.csv'

# Load the training and test datasets
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)

# Print dataset shapes
print(f'Training dataset shape: {train.shape}')
print(f'Test dataset shape: {test.shape}')

# Preprocess column names for consistency
train = train[['file_name', 'label']]
train.columns = ['id', 'label']

# Display columns for reference
print("Train columns:", train.columns)
print("Test columns:", test.columns)


train_len =  round(0.95*len(train))


train_df = train.loc[:train_len]
val_df =  train.loc[train_len:]


name = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/002cbbdc87f0484db60ed0c261c53e7b.jpg'
image = np.array(Image.open(name)).shape








ans = []
for i in range(len(test)):
    name = test['id'][i]
    name = '/kaggle/input/ai-vs-human-generated-dataset/'+name
    image = np.array(Image.open(name)).shape
    if(764 in image):
        count+=1
    ans.append(image)


ans


# Split the training data into training and validation sets (95% train, 5% validation)
# train_df, val_df = train_test_split(
#     train, 
#     test_size=0.05, 
#     random_state=42,  
#     stratify=train['label'] 
# )

# Print shapes of the splits
print(f'Train shape: {train_df.shape}')
print(f'Validation shape: {val_df.shape}')

# Check class distribution in both sets
print("\nTrain class distribution:")
print(train_df['label'].value_counts(normalize=True))

print("\nValidation class distribution:")
print(val_df['label'].value_counts(normalize=True))


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义高斯滤波器
def gaussian_filter(kernel_size, sigma=1.0):
    """
    创建一个高斯滤波器核。
    :param kernel_size: 滤波器的大小（必须是奇数）
    :param sigma: 高斯分布的标准差
    :return: 高斯滤波器核
    """
    x = torch.arange(0, kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g = g[:, None] * g[None, :]
    return g[None, None, :, :]  # 扩展为 [1, 1, kernel_size, kernel_size]

# 加载图像并进行预处理
def load_and_preprocess_image(image_path, image_size=224):
    """
    加载并预处理图像。
    :param image_path: 图像路径
    :param image_size: 图像大小
    :return: 预处理后的图像张量
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # 添加批次维度

# 应用高斯滤波
def apply_gaussian_filter(image_tensor, kernel_size=7, sigma=3.0):
    """
    对图像应用高斯滤波。
    :param image_tensor: 输入图像张量 [batch_size, channels, height, width]
    :param kernel_size: 高斯滤波器的大小
    :param sigma: 高斯分布的标准差
    :return: 滤波后的图像张量
    """
    # 创建高斯滤波器
    gaussian_kernel = gaussian_filter(kernel_size, sigma).to(image_tensor.device)
    gaussian_kernel = gaussian_kernel.repeat(image_tensor.size(1), 1, 1, 1)  # 扩展到与通道数一致

    # 应用滤波器
    filtered_image = F.conv2d(image_tensor, gaussian_kernel, padding=kernel_size // 2, groups=image_tensor.size(1))
    return filtered_image

# 显示图像
def show_image(image_tensor, title=""):
    """
    显示图像。
    :param image_tensor: 图像张量 [batch_size, channels, height, width]
    :param title: 图像标题
    """
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# # 主函数
# if __name__ == "__main__":
#     # 图像路径
#     image_path = "path_to_your_image.jpg"  # 替换为你的图像路径

#     # 加载并预处理图像
#     image_tensor = load_and_preprocess_image(image_path)

#     # 应用高斯滤波
#     filtered_image = apply_gaussian_filter(image_tensor, kernel_size=5, sigma=1.0)

#     # 显示原始图像和滤波后的图像
#     show_image(image_tensor, title="Original Image")
#     show_image(filtered_image, title="Filtered Image")


# filter_data = apply_gaussian_filter(data)


# show_image(filter_data[5,:,:,:])


# show_image(data[5,:,:,:])


# final_data = torch.abs(data[5,:,:,:]-filter_data[5,:,:,:])
# min_val = final_data.min()
# max_val = final_data.max()
# normalized_image = (final_data - min_val) / (max_val - min_val)


# show_image(normalized_image)


# show_image(torch.abs(data[3,:,:,:]-data[3,:,:,:]))


# Training augmentations
train_transforms = transforms.Compose([
    transforms.Resize([234,234]),  # Resize to match ConvNeXt preprocessing
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# Validation and Test transforms
val_test_transforms = transforms.Compose([
    transforms.Resize([224,224]),  # Resize to 232 as per ConvNeXt documentation
    # transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])


# Dataset class for training and validation
class AIImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        # image = preprocess(Image.open(img_name))#.convert('RGB'))#.convert('RGB')
        image = Image.open(img_name).convert('RGB')#.convert('RGB')
        
        if self.transform:
           image = self.transform(image)
        filter_data = apply_gaussian_filter(image.unsqueeze(0))
        image = torch.abs( image -  filter_data.squeeze() )
        min_val = image.min()
        max_val = image.max()
        image = (image - min_val) / (max_val - min_val)
        
        # image = apply_gaussian_filter(image.unsqueeze(0),sigma=0.5).squeeze()
        
        label = self.dataframe.iloc[idx, 1]
        return image, label

# Dataset class for inference (validation and test)
class TestAIImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')#.convert('RGB')

        
        #img = preprocess(Image.open(img_path))#.convert('RGB'))#convert("RGB")
        if self.transform:
           img = self.transform(img)
        filter_data = apply_gaussian_filter(img.unsqueeze(0))
        img = torch.abs( img -  filter_data.squeeze() )
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val)
        # img = apply_gaussian_filter(img.unsqueeze(0),sigma=0.5).squeeze()
        return img, os.path.basename(img_path)  # Return image and filename


# Create datasets
train_dataset = AIImageDataset(train_df, root_dir=path, transform=train_transforms)

# For validation, create a list of file paths and store labels separately
val_file_list = [os.path.join(path, fname) for fname in val_df['id']]
val_labels = val_df['label'].values  # Store labels separately for later use
val_dataset = TestAIImageDataset(file_list=val_file_list, transform=val_test_transforms)

# For testing, create a list of file paths
test_file_list = [os.path.join(path, fname) for fname in test['id']]
test_dataset = TestAIImageDataset(file_list=test_file_list, transform=val_test_transforms)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)



model = model.to(dtype=torch.float32)


# for param in model.parameters():
#     param.requires_grad = True

# Define loss function, optimizer, and learning rate scheduler
optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'lr': 1e-5},  # Lower LR for backbone
])

criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5, gamma=0.7)





# # Load pretrained ConvNeXt Base model
# model = models.convnext_base(weights="DEFAULT")

# # Freeze all layers initially
# for param in model.features.parameters():
#     param.requires_grad = False

# # Unfreeze the last two stages 
# for param in model.features[-2:].parameters(): 
#     param.requires_grad = True

# # Replace the classifier head with a custom one
# model.classifier = nn.Sequential(
#     nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
#     nn.Flatten(),                  # Flatten the tensor
#     nn.BatchNorm1d(1024),          # Add BatchNorm here
#     nn.Linear(1024, 512),          # First fully connected layer
#     nn.ReLU(),                     # Activation function
#     nn.Dropout(0.4),               # Dropout for regularization
#     nn.Linear(512, 2)              # Output layer (binary classification)
# )

# # Move the model to gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Define loss function, optimizer, and learning rate scheduler
# optimizer = torch.optim.AdamW([
#     {'params': model.features[-2:].parameters(), 'lr': 1e-5},  # Lower LR for backbone
#     {'params': model.classifier.parameters(), 'lr': 1e-4}      # Higher LR for classifier
# ])

# criterion = nn.CrossEntropyLoss()
# scheduler = StepLR(optimizer, step_size=5, gamma=0.7)


print("Training Start")









# Training Loop
epochs = 12

train_losses, train_accuracies, val_losses, val_accuracies, val_f1s = [], [], [], [], []

for epoch in range(epochs):
    # -- Training --
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        # if(epoch==0):
        #     break
        data, label = data.to(device).float(), label.to(device)
        
        # Check data for NaN or Inf
        if torch.isnan(data).any() or torch.isinf(data).any():
            raise ValueError("Data contains NaN or Inf values")
        
        optimizer.zero_grad()
        
        fake_texts_features = torch.mean(model.encode_text(fake_text.to(device)), dim=0).unsqueeze(0)
        real_texts_features = torch.mean(model.encode_text(real_text.to(device)), dim=0).unsqueeze(0)
        text_features = torch.cat([real_texts_features,fake_texts_features], dim=0)
        
        image_features = model.encode_image(data) 
        image_features_final = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_features_final = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        output = image_features_final @ text_features_final.T  /0.07  # Increased temperature
        # print(output)
        
        loss = criterion(output, label)
        
        image_features_consistency = image_features_final @ image_features_final.T
        label2 =  ((label - 0.5).reshape(-1,1) @  (label - 0.5).reshape(1,-1))>0.0 + 0.0
        image_features_consistency2 = image_features_consistency * label2.float() + (1-label2.float())*-100
        loss2 = torch.mean(-1 * torch.log( torch.sum(torch.exp(image_features_consistency2),dim=0)/ torch.sum(torch.exp(image_features_consistency),dim=0)),dim=0)
        # break
        
        
        # print(f"Loss: {loss.item()}")
        
        (loss+loss2).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        epoch_loss += loss.item()
        preds = output.argmax(dim=1)
        acc = (preds == label).float().mean().item()
        epoch_accuracy += acc
    
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    # break
    # -- Validation --
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_pred_classes = []  # To store predictions
    val_labels_list = []   # To store true labels
    
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
            data = data.to(device)
            # with torch.no_grad():
            fake_texts_features = torch.mean(model.encode_text(fake_text.to(device)), dim=0).unsqueeze(0)
            real_texts_features = torch.mean(model.encode_text(real_text.to(device)), dim=0).unsqueeze(0)
            text_features = torch.cat([real_texts_features,fake_texts_features], dim=0)
            image_features = model.encode_image(data) 
            image_features_final = image_features/(image_features.norm(dim=-1, keepdim=True)+ 1e-8)
            text_features_final = text_features/(text_features.norm(dim=-1, keepdim=True)+ 1e-8)
            output = image_features_final @ text_features_final.T  /0.07  # Increased temperature
            # output = model(data)
            
            # Get true labels from val_df
            batch_labels = val_labels[i * val_loader.batch_size : (i + 1) * val_loader.batch_size]
            batch_labels = torch.tensor(batch_labels, device=device)
            
            # Compute loss
            loss = criterion(output, batch_labels)
            val_loss += loss.item()
            
            # Compute predictions and accuracy
            preds = output.argmax(dim=1)
            acc = (preds == batch_labels).float().mean().item()
            val_acc += acc
            
            # Store predictions and true labels
            val_pred_classes.extend(preds.cpu().numpy())
            val_labels_list.extend(batch_labels.cpu().numpy())
    
    # Compute average validation metrics
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_f1 = f1_score(val_labels_list, val_pred_classes, average='binary')  # Binary classification
    
    # Append metrics
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_f1s.append(val_f1)
    
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
    )
    
    # Step the learning rate scheduler
    scheduler.step()
    # Generate predictions and logits for the test set
    model.eval()
    test_logits = []  # To store logits
    test_pred_classes = []
    
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc="Generating Test Predictions"):
            data = data.to(device)
            fake_texts_features = torch.mean(model.encode_text(fake_text.to(device)),dim=0).unsqueeze(0)
            real_texts_features = torch.mean(model.encode_text(real_text.to(device)),dim=0).unsqueeze(0)
            text_features = torch.cat([real_texts_features,fake_texts_features], dim=0)
            image_features = model.encode_image(data) 
            image_features_final = image_features/(image_features.norm(dim=-1, keepdim=True)+ 1e-8)
            text_features_final = text_features/(text_features.norm(dim=-1, keepdim=True)+ 1e-8)
            output = image_features_final @ text_features_final.T  /0.07  # Increased temperature
            # output = model(data)
            #output = model(data)  # Raw logits (before softmax)
            
            # Save logits
            test_logits.extend(output.cpu().numpy())  # Store raw logits
            
            # Get predicted class (0 or 1)
            preds = output.argmax(dim=1)
            test_pred_classes.extend(preds.cpu().numpy())

    # Convert logits to a DataFrame
    logits_df = pd.DataFrame(test_logits, columns=['logit_class_0', 'logit_class_1'])
    logits_df['id'] = test['id'].values  # Add image IDs for reference
    
    # Save logits to a CSV file
    logits_df.to_csv('test_logits'+str(epoch)+'.csv', index=False)
    
    # Add predictions to the test DataFrame
    test['label'] = test_pred_classes
    test[['id', 'label']].to_csv('submission'+str(epoch)+'.csv', index=False)
    print(test['label'].value_counts())
    print("Test logits saved to 'test_logits.csv'")
    print("Test predictions saved to 'submission.csv'")








loss


# Generate predictions and logits for the test set
model.eval()
test_logits = []  # To store logits
test_pred_classes = []

with torch.no_grad():
    for data, _ in tqdm(test_loader, desc="Generating Test Predictions"):
        data = data.to(device)
        fake_texts_features = torch.mean(model.encode_text(fake_text.to(device)),dim=0).unsqueeze(0)
        real_texts_features = torch.mean(model.encode_text(real_text.to(device)),dim=0).unsqueeze(0)
        text_features = torch.cat([real_texts_features,fake_texts_features], dim=0)
        image_features = model.encode_image(data) 
        image_features_final = image_features/(image_features.norm(dim=-1, keepdim=True)+ 1e-8)
        text_features_final = text_features/(text_features.norm(dim=-1, keepdim=True)+ 1e-8)
        output = image_features_final @ text_features_final.T  /0.07  # Increased temperature
            # output = model(data)
        #output = model(data)  # Raw logits (before softmax)
        
        # Save logits
        test_logits.extend(output.cpu().numpy())  # Store raw logits
        
        # Get predicted class (0 or 1)
        preds = output.argmax(dim=1)
        test_pred_classes.extend(preds.cpu().numpy())

# Convert logits to a DataFrame
logits_df = pd.DataFrame(test_logits, columns=['logit_class_0', 'logit_class_1'])
logits_df['id'] = test['id'].values  # Add image IDs for reference

# Save logits to a CSV file
logits_df.to_csv('test_logits.csv', index=False)

# Add predictions to the test DataFrame
test['label'] = test_pred_classes
test[['id', 'label']].to_csv('submission.csv', index=False)

print("Test logits saved to 'test_logits.csv'")
print("Test predictions saved to 'submission.csv'")
print(pd.read_csv('submission.csv')['label'].value_counts())


for i in range(len(test)):
    print(test.loc[i])
    if(i>20):
        break


from PIL import Image

# 打开图片
image = Image.open('/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/ef29ead63754441b82b56c1a22082fdf.jpg')
image


# # Generate predictions for the test set
# model.eval()
# test_pred_classes = []

# with torch.no_grad():
#     for data, _ in tqdm(test_loader, desc="Generating Test Predictions"):
#         data = data.to(device)
#         output = model(data)
#         preds = output.argmax(dim=1)  # Get predicted class (0 or 1)
#         test_pred_classes.extend(preds.cpu().numpy())

# # Add predictions to the test DataFrame
# test['label'] = test_pred_classes

# # Save predictions to a CSV file
# test[['id', 'label']].to_csv('submission.csv', index=False)
# print("Test predictions saved to 'submission.csv'")


pd.read_csv('submission.csv')['label'].value_counts()




