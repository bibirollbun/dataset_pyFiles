import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
'''
Specific Objectives
1) Accurately classify a given image from testing dataset into different diseased category or a healthy leaf; 
2) Accurately distinguish between many diseases, sometimes more than one on a single leaf; 
3) Deal with rare classes and novel symptoms; 
4) Address depth perception—angle, light, shade, physiological age of the leaf; 
5) Incorporate expert knowledge in identification, annotation, quantification, and guiding computer vision to search for relevant features during learning.
'''



# 데이터 경로
data_path = '/kaggle/input/plant-pathology-2020-fgvc7/'
image_path = '/kaggle/input/plant-pathology-2020-fgvc7/images/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')


cols = list(train.columns)
cols


# 데이터 분할
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 클래스 정의
class PlantDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform, image_dir):
        self.dataframe = dataframe
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_id'] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        label = row[['healthy', 'multiple_diseases', 'rust', 'scab']].values.astype(float)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


# ResNet50 모델 로드
model = models.resnet50(pretrained=True)

# 출력층 수정
num_classes = 4
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# 데이터로더 생성
train_dataset = PlantDataset(train_df, transform, image_path)
val_dataset = PlantDataset(val_df, transform, image_path)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 손실 함수와 최적화기 정의
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}")


sigmoid = nn.Sigmoid()
# 평가 모드로 설정
model.eval()
val_preds = []
val_labels = []
# 예측 수행
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        labels = labels.cpu().numpy()
        outputs = model(images)
        probabilities = sigmoid(outputs)  # 확률 값 계산
        val_preds.extend(probabilities)
        val_labels.extend(labels)

val_preds = np.array(val_preds)
val_labels = np.array(val_labels)
    
# ROC-AUC 계산
roc_auc = roc_auc_score(val_labels, val_preds, average='macro')
print(roc_auc)


# 테스트 데이터로더 생성
test[['healthy', 'multiple_diseases', 'rust', 'scab']] = None
test_dataset = PlantDataset(test, transform, image_path)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 평가 모드 설정
model.eval()

# 예측 결과 저장
predictions = []

with torch.no_grad():
    for images, _ in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)  # 모델 출력값
        probabilities = sigmoid(outputs).cpu().numpy()  # 확률로 변환
        predictions.extend(probabilities)


# 제출 데이터프레임 생성
submission_df = pd.DataFrame(predictions, columns=['healthy', 'multiple_diseases', 'rust', 'scab'])
submission_df.insert(0, 'image_id', test['image_id'])

# CSV 저장
submission_file = 'submission.csv'
submission_df.to_csv(submission_file, index=False)

print(f"Submission file saved to: {submission_file}")


from IPython.display import HTML
import base64

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(submission_df, filename='submission.csv')

