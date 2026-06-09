# Импорт библиотек
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Проверка содержимого директории
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Пути к данным
data_dir = '/kaggle/input/image-matching-challenge-2025/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
submission_file = os.path.join(data_dir, 'sample_submission.csv')

# Содержимое на sample_submission.csv
print("\nСодержимое sample_submission.csv:")
sample_submission = pd.read_csv(submission_file)
print(sample_submission.head())

# Класс для загрузки изображений
class ImageDataset:
    def __init__(self, img_dir, img_size=(640, 480)):
        self.img_paths = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.endswith(('.jpg', '.png')):
                    self.img_paths.append(os.path.join(root, f))
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        return img, self.img_paths[idx]

# Функция для извлечения признаков и сопоставления с SIFT
def match_images_sift(img1_path, img2_path):
    sift = cv2.SIFT_create()
    
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    
    if desc1 is None or desc2 is None:
        return np.array([]), np.array([])
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Применяем тест отношения Лоу
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    mkpts0 = np.array([kp1[m.queryIdx].pt for m in good])
    mkpts1 = np.array([kp2[m.trainIdx].pt for m in good])
    
    return mkpts0, mkpts1

# Функция для вычисления фундаментальной матрицы
def compute_fundamental_matrix(mkpts0, mkpts1):
    if len(mkpts0) < 8:  # Минимум 8 точек для RANSAC
        return None
    F, mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.FM_RANSAC)
    return F

# Подготовка тестовых данных
test_dataset = ImageDataset(test_dir)

# Извлечение дескрипторов для кластеризации (упрощённо)
sift = cv2.SIFT_create()
descriptors = []
img_paths = []
for idx in range(len(test_dataset)):
    img, path = test_dataset[idx]
    _, desc = sift.detectAndCompute(img, None)
    if desc is not None:
        desc_mean = desc.mean(axis=0)  # Усредняем дескрипторы
        descriptors.append(desc_mean)
        img_paths.append(path)
    else:
        descriptors.append(np.zeros(128))  # Заглушка для пустых дескрипторов
descriptors = np.array(descriptors)

# Кластеризация с DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
clusters = dbscan.fit_predict(descriptors)

# Группировка изображений по кластерам
cluster_dict = {}
for idx, cluster_id in enumerate(clusters):
    if cluster_id != -1:  # Игнорируем шум (-1)
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(img_paths[idx])

# Обработка парного соответствия внутри кластеров
matches_dict = {}
for cluster_id, img_list in cluster_dict.items():
    for i in range(len(img_list)):
        for j in range(i + 1, len(img_list)):
            mkpts0, mkpts1 = match_images_sift(img_list[i], img_list[j])
            F = compute_fundamental_matrix(mkpts0, mkpts1)
            
            if F is not None:
                pair_key = f"{os.path.basename(img_list[i])};{os.path.basename(img_list[j])}"
                matches_dict[pair_key] = F.flatten().tolist()

# Создание submission.csv
submission_data = []
for pair, F in matches_dict.items():
    submission_data.append({
        'image_pair': pair,
        'fundamental_matrix': ' '.join(map(str, F))
    })

submission = pd.DataFrame(submission_data)
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Файл submission.csv успешно создан!")




