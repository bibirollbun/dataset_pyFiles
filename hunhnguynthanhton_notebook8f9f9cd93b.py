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


# %% [code]
import os
import laspy
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# %% [code]
def load_lidar_data(file_path):
    
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    return points

def extract_features(points):
    
    # Thống kê cơ bản
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    min_val = np.min(points, axis=0)
    max_val = np.max(points, axis=0)
    median = np.median(points, axis=0)
    
    # Các giá trị phần trăm
    perc_25 = np.percentile(points, 25, axis=0)
    perc_75 = np.percentile(points, 75, axis=0)
    
    # Thống kê phân bố: skewness và kurtosis cho mỗi tọa độ
    skewness = np.array([skew(points[:, i]) for i in range(points.shape[1])])
    kurt = np.array([kurtosis(points[:, i]) for i in range(points.shape[1])])
    
    # Tính mật độ điểm theo mặt phẳng XY
    count = points.shape[0]
    area = (max_val[0] - min_val[0]) * (max_val[1] - min_val[1])
    density = count / area if area > 0 else count
    
    # Nối các đặc trưng lại thành một vector
    features = np.concatenate([mean, std, min_val, max_val, median, perc_25, perc_75, skewness, kurt, [density]])
    return features

def prepare_data(folder, label):
   
    file_names = []
    features = []
    labels = []
    
    for root, dirs, files in os.walk(folder):
        for filename in tqdm(files, desc=f"Processing files in {folder}"):
            if filename.lower().endswith('.las'):
                file_path = os.path.join(root, filename)
                try:
                    points = load_lidar_data(file_path)
                    feat = extract_features(points)
                    features.append(feat)
                    file_names.append(filename)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return np.array(features), np.array(labels), file_names

def save_predictions_to_csv(file_names, predictions, output_file):
    """
    Lưu kết quả dự đoán (tên file và nhãn) vào file CSV.
    """
    result = {'name': file_names, 'label': predictions}
    df = pd.DataFrame(result)
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")



# %% [code]
# Đường dẫn dữ liệu training
folder_coniferous = r'/kaggle/input/hutechaichallenge2024-bc/Train/Coniferous'
folder_deciduous  = r'/kaggle/input/hutechaichallenge2024-bc/Train/Deciduous'

# Gán nhãn: 0 cho cây lá kim, 1 cho cây lá rộng
X_coniferous, y_coniferous, _ = prepare_data(folder_coniferous, 0)
X_deciduous, y_deciduous, _ = prepare_data(folder_deciduous, 1)

# Kết hợp dữ liệu của cả hai nhóm
X = np.concatenate([X_coniferous, X_deciduous], axis=0)
y = np.concatenate([y_coniferous, y_deciduous], axis=0)

print("Total samples:", X.shape[0])



# %% [code]
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80/20) với stratify để đảm bảo cân bằng nhãn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])



# %% [code]
# Xây dựng pipeline gồm StandardScaler và RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Lưới tham số mở rộng cho GridSearchCV
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)



# %% [code]
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print("Classification report:\n", classification_report(y_test, y_pred))

# Hiển thị Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)



# %% [code]
model_path = r'/kaggle/working/best_rf_model.pkl'
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")



# %% [code]
# Đường dẫn tới thư mục Test
test_folder = r'/kaggle/input/hutechaichallenge2024-bc/Test'
test_file_names = []
test_predictions = []

for root, dirs, files in os.walk(test_folder):
    for filename in tqdm(files, desc="Predicting test files"):
        if filename.lower().endswith('.las'):
            file_path = os.path.join(root, filename)
            try:
                points = load_lidar_data(file_path)
                feat = extract_features(points).reshape(1, -1)
                pred = best_model.predict(feat)
                test_file_names.append(os.path.splitext(filename)[0])
                test_predictions.append(pred[0])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Lưu kết quả dự đoán vào file CSV
output_csv = r'/kaggle/working/submission.csv'
save_predictions_to_csv(test_file_names, test_predictions, output_csv)


