# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%capture
!pip install laspy


import os
import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, entropy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import PowerTransformer
import lightgbm as lgb
import csv
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report



TRAIN_DIR = "/kaggle/input/hutechaichallenge2024-mc/Train"  
TEST_DIR = "/kaggle/input/hutechaichallenge2024-mc/Test"   
OUTPUT_CSV = "/kaggle/working/sumbmission.csv"


# =============== MAPPING LABELS ===============
LABEL_MAP = {
    "Fir": 1,    # lãnh sam
    "Pine": 2,   # thông
    "Spruce": 3, # vân sam
    "Alder": 4,  # trăn
    "Aspen": 5,  # dương
    "Birch": 6,  # bạch dương
    "Tilia": 7   # đoạn
}



def load_lidar_data(filepath):
    las = laspy.read(filepath)
    points = np.vstack((las.x, las.y, las.z)).T
    point_count = len(points)

    # Thống kê cơ bản
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    min_val = np.min(points, axis=0)
    max_val = np.max(points, axis=0)
    median = np.median(points, axis=0)
    range_val = max_val - min_val
    kurt_val = kurtosis(points, axis=0)
    skew_val = skew(points, axis=0)
    percentiles = np.percentile(points, [25, 50, 75], axis=0).flatten()
    
    # Phân bố theo trục Z
    z_bins = np.histogram(points[:, 2], bins=10, density=True)[0]
    vertical_profile = z_bins / (np.sum(z_bins) + 1e-6)
    vertical_range = max_val[2] - min_val[2]
    height_entropy = entropy(z_bins + 1e-6)
    
    # Tính canopy_height nếu có đủ điểm
    if len(points) > 0:
        canopy_height = max_val[2] - np.median(points[points[:, 2] > np.percentile(points[:, 2], 75), 2])
    else:
        canopy_height = 0

    xy_range = np.sqrt((max_val[0] - min_val[0])**2 + (max_val[1] - min_val[1])**2)
    density = point_count / ((xy_range + 1e-6) * (vertical_range + 1e-6))
    
    canopy_ratio = canopy_height / (vertical_range + 1e-6) if vertical_range != 0 else 0
    crown_width = np.max([max_val[0] - min_val[0], max_val[1] - min_val[1]]) if point_count > 0 else 0

    # Sử dụng DBSCAN để lấy số cụm và tỉ lệ nhiễu
    if point_count > 10:
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(points)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        noise_ratio = list(clustering.labels_).count(-1) / point_count
    else:
        n_clusters = 1
        noise_ratio = 0

    features = np.concatenate([
        mean, std, min_val, max_val, median, range_val, kurt_val, skew_val, percentiles,
        vertical_profile,
        [density, height_entropy, vertical_range, point_count, canopy_height, xy_range],
        [canopy_ratio, crown_width, n_clusters, noise_ratio]
    ])
    return features, points

def get_data_from_folder(folder):
    data, labels = [], []
    for tree_type in os.listdir(folder):
        tree_path = os.path.join(folder, tree_type)
        if os.path.isdir(tree_path) and tree_type in LABEL_MAP:
            for file in os.listdir(tree_path):
                if file.lower().endswith(".las"):
                    filepath = os.path.join(tree_path, file)
                    features, _ = load_lidar_data(filepath)
                    data.append(features)
                    labels.append(LABEL_MAP[tree_type])
    return np.array(data), np.array(labels)



def train_model():
    print("Đang đọc dữ liệu Train...")
    train_data, train_labels = get_data_from_folder(TRAIN_DIR)
    print("Kích thước dữ liệu train:", train_data.shape)
    
    # Chuẩn hóa dữ liệu
    scaler = PowerTransformer()
    train_data_scaled = scaler.fit_transform(train_data)
    
    # Tạo DataFrame với tên cột cho các đặc trưng
    features_cols = [f"f{i}" for i in range(train_data_scaled.shape[1])]
    train_df = pd.DataFrame(train_data_scaled, columns=features_cols)
    
    # Khởi tạo mô hình LightGBM
    model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        verbosity=-1
    )
    
    print("Đang huấn luyện mô hình LightGBM...")
    model.fit(train_df, train_labels)
    print("Huấn luyện hoàn tất!")
    
    # Tính toán độ chính xác trên tập huấn luyện
    pred_train = model.predict(train_df)
    acc = accuracy_score(train_labels, pred_train)
    print("Training Accuracy: {:.2f}%".format(acc * 100))
    print("Classification Report:\n", classification_report(train_labels, pred_train))
    
    return scaler, model, features_cols



def predict_on_test(scaler, model, features_cols):
    results = []
    for tree_type in os.listdir(TEST_DIR):
        tree_path = os.path.join(TEST_DIR, tree_type)
        if os.path.isdir(tree_path):
            for file in os.listdir(tree_path):
                if file.lower().endswith(".las"):
                    filepath = os.path.join(tree_path, file)
                    features, _ = load_lidar_data(filepath)
                    features_scaled = scaler.transform([features])
                    test_df = pd.DataFrame(features_scaled, columns=features_cols)
                    pred_label = model.predict(test_df)[0]
                    filename_no_ext = os.path.splitext(file)[0]
                    results.append((filename_no_ext, pred_label))
    return results



def save_results_to_csv(results, output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for (filename, pred_label) in results:
            writer.writerow([filename, pred_label])



if __name__ == "__main__":
    # Huấn luyện mô hình và in ra độ chính xác trên tập Train
    scaler, lgb_model, features_cols = train_model()
    
    # Dự đoán trên tập Test
    print("Đang dự đoán trên dữ liệu Test...")
    test_results = predict_on_test(scaler, lgb_model, features_cols)
    print(f"Số lượng file dự đoán: {len(test_results)}")
    
    # Lưu kết quả dự đoán vào file CSV
    save_results_to_csv(test_results, OUTPUT_CSV)
    print(f"Đã lưu kết quả dự đoán vào file '{OUTPUT_CSV}'.")


