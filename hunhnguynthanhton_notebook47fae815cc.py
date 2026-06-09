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
import laspy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# Hàm đọc dữ liệu từ file .las và trích xuất tọa độ 3D (x, y, z)
def load_lidar_data(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    return points

# Hàm duyệt qua thư mục (bao gồm các thư mục con) và trích xuất đặc trưng từ file .las
def prepare_data(folder, label):
    file_names = []
    features = []
    labels = []
    
    # Duyệt qua tất cả các thư mục con trong thư mục gốc
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith('.las'):  # không phân biệt chữ hoa thường
                file_path = os.path.join(root, filename)
                points = load_lidar_data(file_path)
                # Sử dụng trung bình tọa độ (x, y, z) làm đặc trưng
                features.append(np.mean(points, axis=0))
                file_names.append(filename)
                labels.append(label)
    
    return np.array(features), np.array(labels), file_names



# Sử dụng raw string để đảm bảo đường dẫn đúng (không bị lỗi escape ký tự)
folder_coniferous = r'/kaggle/input/hutechaichallenge2024-bc/Test/Coniferous'
folder_deciduous  = r'/kaggle/input/hutechaichallenge2024-bc/Test/Deciduous'

# Dữ liệu cho cây lá kim (Coniferous) với nhãn 0 và cây lá rộng (Deciduous) với nhãn 1
X_coniferous, y_coniferous, files_coniferous = prepare_data(folder_coniferous, 0)
X_deciduous,  y_deciduous,  files_deciduous  = prepare_data(folder_deciduous, 1)

# Kết hợp dữ liệu từ cả 2 nhóm
X = np.concatenate([X_coniferous, X_deciduous], axis=0)
y = np.concatenate([y_coniferous, y_deciduous], axis=0)

print("Tổng số mẫu:", X.shape[0])



# Chia dữ liệu thành tập huấn luyện và kiểm tra (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dữ liệu huấn luyện: {X_train.shape[0]} mẫu")
print(f"Dữ liệu kiểm tra: {X_test.shape[0]} mẫu")

# Sử dụng mô hình RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))



# Hàm lưu kết quả dự đoán vào file CSV
def save_predictions_to_csv(file_names, predictions, output_file):
    result = {'name': file_names, 'label': predictions}
    df = pd.DataFrame(result)
    df.to_csv(output_file, index=False)
    print(f"Kết quả đã được lưu vào {output_file}")

# Dự đoán cho tất cả các file .las trong thư mục test
test_folder = r'/kaggle/input/hutechaichallenge2024-bc/Test'
file_names_pred = []
predictions = []

for root, dirs, files in os.walk(test_folder):
    for filename in files:
        if filename.lower().endswith('.las'):
            file_path = os.path.join(root, filename)
            points = load_lidar_data(file_path)
            # Dùng trung bình tọa độ (x, y, z) làm đặc trưng để dự đoán
            features = np.mean(points, axis=0).reshape(1, -1)
            result = clf.predict(features)
            file_names_pred.append(os.path.splitext(filename)[0])
            predictions.append(result[0])

# Đường dẫn lưu file CSV (sử dụng raw string)
output_csv = r'/kaggle/working/submission.csv'
save_predictions_to_csv(file_names_pred, predictions, output_csv)


