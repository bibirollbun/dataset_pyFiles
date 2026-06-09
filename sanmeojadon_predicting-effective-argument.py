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


!pip install -U autogluon > /dev/null


import pandas as pd
from autogluon.tabular import TabularPredictor
import os



data_path = '/kaggle/input/feedback-prize-effectiveness'
train_csv_path = os.path.join(data_path, 'train.csv')
train_df = pd.read_csv(train_csv_path)
print("Dữ liệu huấn luyện có", len(train_df), "hàng.")



from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(train_df, test_size = 0.2, random_state = 3, stratify = train_df['discourse_effectiveness'] )


label = 'discourse_effectiveness'


predictor = TabularPredictor(
    label=label,
    eval_metric='log_loss'
).fit(
    train_data = train_data,
    time_limit = 600
)


predictor.evaluate(test_data)


import pandas as pd
from autogluon.tabular import TabularPredictor
import os

# --- Bước 1: Tải dữ liệu ---
# Đường dẫn tới dữ liệu trên môi trường Kaggle
data_path = '/kaggle/input/feedback-prize-effectiveness'
train_csv_path = os.path.join(data_path, 'train.csv')
test_csv_path = os.path.join(data_path, 'test.csv')
sample_submission_path = os.path.join(data_path, 'sample_submission.csv')

print("Đang tải dữ liệu...")
# Tải dữ liệu huấn luyện
train_df = pd.read_csv(train_csv_path)

# Tải dữ liệu kiểm tra
test_df = pd.read_csv(test_csv_path)

# Tải file submission mẫu để biết định dạng
sample_submission_df = pd.read_csv(sample_submission_path)

print("Tải dữ liệu thành công!")
print("Dữ liệu huấn luyện có", len(train_df), "hàng.")
print("Dữ liệu kiểm tra có", len(test_df), "hàng.")


# --- Bước 2: Chuẩn bị dữ liệu ---
# Biến mục tiêu (target) mà chúng ta cần dự đoán
label = 'discourse_effectiveness'

# AutoGluon có thể tự động xử lý các cột không cần thiết,
# nhưng để rõ ràng, chúng ta có thể chỉ chọn những cột quan trọng.
# Các feature chính là nội dung văn bản và loại của yếu tố lập luận.
# AutoGluon sẽ tự động xem 'discourse_text' là văn bản và 'discourse_type' là biến phân loại.
# Chúng ta không cần xóa các cột khác, AutoGluon sẽ tự bỏ qua chúng nếu không hữu ích.

# Hiển thị 5 dòng đầu của dữ liệu huấn luyện
print("\n5 dòng đầu của dữ liệu huấn luyện:")
print(train_df.head())


# --- Bước 3: Huấn luyện mô hình với AutoGluon ---
# Khởi tạo TabularPredictor
# - label: Tên cột mục tiêu cần dự đoán.
# - path: Thư mục để lưu lại các mô hình đã huấn luyện.
# - eval_metric: Thước đo để đánh giá mô hình, dùng 'log_loss' theo yêu cầu cuộc thi.
predictor = TabularPredictor(
    label=label,
    path='autogluon_models',
    eval_metric='log_loss'
)

# Bắt đầu huấn luyện
# - train_data: DataFrame chứa dữ liệu huấn luyện.
# - presets: Cấu hình có sẵn. 'best_quality' sẽ ưu tiên độ chính xác cao nhất,
#            nhưng tốn nhiều thời gian. Bạn có thể dùng 'high_quality' hoặc
#            'medium_quality_faster_train' để chạy nhanh hơn.
# - time_limit: Thời gian tối đa cho phép huấn luyện (tính bằng giây).
#               Rất quan trọng trong các cuộc thi có giới hạn thời gian chạy notebook.
#               Ví dụ: 2 giờ = 7200 giây.
#               Trên Kaggle, bạn có khoảng 9 tiếng (khoảng 32400 giây).
#               Chúng ta đặt 8 tiếng để có thời gian dự phòng.
time_limit_seconds = 1200

print("\nBắt đầu huấn luyện mô hình...")
predictor.fit(
    train_data=train_df,
    presets='best_quality', # Tối ưu cho độ chính xác
    time_limit=time_limit_seconds
)

print("Huấn luyện hoàn tất!")


# --- Bước 4: Xem lại kết quả và các mô hình đã huấn luyện ---
print("\nBảng xếp hạng các mô hình:")
leaderboard = predictor.leaderboard(train_df, silent=True)
print(leaderboard)


# --- Bước 5: Dự đoán trên tập kiểm tra (Test set) ---
print("\nBắt đầu dự đoán trên tập test...")
# Sử dụng predict_proba() để lấy xác suất cho mỗi lớp, vì cuộc thi yêu cầu log_loss.
predictions_proba = predictor.predict_proba(test_df)

print("Dự đoán hoàn tất!")
print("5 dòng đầu của kết quả dự đoán (xác suất):")
print(predictions_proba.head())


# --- Bước 6: Tạo tệp submission ---
print("\nTạo tệp submission...")
# Kết quả dự đoán có các cột là 'Ineffective', 'Adequate', 'Effective'.
# Chúng ta chỉ cần gán các giá trị này vào đúng cột trong file submission.
submission_df = test_df[['discourse_id']].copy()
submission_df[['Ineffective', 'Adequate', 'Effective']] = predictions_proba

# Lưu tệp submission
submission_df.to_csv('submission.csv', index=False)

print("\nTệp submission.csv đã được tạo thành công!")
print(submission_df.head())




