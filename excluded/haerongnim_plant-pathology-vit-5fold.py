!pip install transformers timm


from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd



train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
submission = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")


base_path='/kaggle/input/plant-pathology-2020-fgvc7/images/'
def generate_image_path(image_id):
    return f"{base_path}{image_id}.jpg"

# Apply the function to create the 'img' column
train['img'] = train['image_id'].apply(generate_image_path)
test['img'] = test['image_id'].apply(generate_image_path)


train.head()


submission.head()


#데이터 증강
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 수평 플립
    transforms.RandomVerticalFlip(),    # 수직 플립 (원하는 경우)
    transforms.RandomRotation(30),      # 30도 이내로 회전
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Shift(이동) 및 Scale(크기 조정)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기 및 대비 조정
    transforms.GaussianBlur(3),  # Gaussian Blur
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])


from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch.nn as nn

# 사전 학습된 ViT 모델 로드
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",  # Pretrained 모델
    num_labels=4  # Healthy, Multiple Diseases, Rust, Scab
)

# 출력 레이어에 Sigmoid 추가
model.classifier = nn.Sequential(
    nn.Dropout(0.5), #dropout 추가
    nn.Linear(model.classifier.in_features, 4),
    nn.Sigmoid()  # 멀티라벨 예측을 위한 Sigmoid
)

model = model.to("cuda")



class PlantPathologyDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, is_test=False):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test  # is_test 속성 추가

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = f"{self.image_dir}/{row['image_id']}.jpg"
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # 테스트 데이터에서는 레이블을 반환하지 않음
        if self.is_test:
            return image, row['image_id']  # 레이블 없이 이미지와 image_id만 반환
        else:
            labels = row[['healthy', 'multiple_diseases', 'rust', 'scab']].values.astype("float32")
            return image, labels



# Train Dataset
train_csv_path = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"  # Train CSV 파일 경로
train_image_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images"    # 이미지가 섞여 있는 디렉터리
train_dataset = PlantPathologyDataset(train_csv_path, train_image_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Test Dataset
test_csv_path = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"    # Test CSV 파일 경로
test_image_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images"    # 동일한 이미지 디렉터리 사용
test_dataset = PlantPathologyDataset(test_csv_path, test_image_dir, transform=test_transforms, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) #L2 정규화 추가

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to("cuda"), labels.to("cuda")
        
        # Forward pass
        outputs = model(images).logits  # ViT는 logits 속성 반환
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")



model.eval()
predictions = []
image_ids = []

for images, image_names in test_loader:
    images = images.to("cuda")
    with torch.no_grad():
        outputs = model(images).logits
        predictions.extend(outputs.cpu().numpy())
        image_ids.extend(image_names)  # image_names는 image_id

# 예측 결과를 DataFrame으로 저장
submission = pd.DataFrame(predictions, columns=['healthy', 'multiple_diseases', 'rust', 'scab'])
submission.insert(0, 'image_id', image_ids)
submission.to_csv('submission_1.csv', index=False)


