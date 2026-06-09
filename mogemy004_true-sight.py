!pip install datasets
!pip install -U -q evaluate transformers datasets>=2.14.5 accelerate>=0.27 2>/dev/null
!pip install huggingface_hub


!apt-get install git-lfs
!git lfs install
!git clone https://huggingface.co/datasets/yashduhan/DeepFakeDetection
!GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/yashduhan/DeepFakeDetection


!cd DeepFakeDetection && git lfs pull
!ls DeepFakeDetection


import zipfile

with zipfile.ZipFile("/kaggle/working/DeepFakeDetection/DeepFakeDetection.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/dataset_extracted")
    
!ls /kaggle/working/dataset_extracted


from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

data_dir = "/kaggle/working/dataset_extracted"

dataset = datasets.ImageFolder(root=data_dir, transform=ToTensor())
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(len(dataset))
print(dataset.classes)


!ls /kaggle/working/dataset_extracted/DeepFakeDetection | head
!ls /kaggle/working/dataset_extracted/DeepFakeDetection | wc -l



import os
import shutil

src = "/kaggle/working/dataset_extracted/DeepFakeDetection"
dst_real = "/kaggle/working/data/real"
dst_fake = "/kaggle/working/data/fake"

os.makedirs(dst_real, exist_ok=True)
os.makedirs(dst_fake, exist_ok=True)

for file in os.listdir(src):
    if "real" in file.lower():
        shutil.move(os.path.join(src, file), os.path.join(dst_real, file))
    elif "fake" in file.lower():
        shutil.move(os.path.join(src, file), os.path.join(dst_fake, file))



!ls /kaggle/working/data/real | head
!ls /kaggle/working/data/fake | head


data_dir = "/kaggle/working/data"

dataset = datasets.ImageFolder(root=data_dir, transform=ToTensor())
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(dataset.classes)
print(len(dataset))



import os

path1 = "/kaggle/input/deepfake-detection-challenge"
print(os.listdir(path1))

path2 = "/kaggle/input/deep-fake-detection-dfd-entire-original-dataset"
print(os.listdir(path2))

path3 = "/kaggle/input/deepfake-and-real-images"
print(os.listdir(path3))



train_dir = "/kaggle/input/deepfake-detection-challenge/train_sample_videos"

print(len(os.listdir(train_dir)))
print(os.listdir(train_dir)[:10])

test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos"

print(len(os.listdir(test_dir)))
print(os.listdir(test_dir)[:10])


