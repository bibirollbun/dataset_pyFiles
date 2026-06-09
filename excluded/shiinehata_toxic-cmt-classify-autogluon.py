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


#---------------------------------------------------
# PHẦN 1: CÀI ĐẶT VÀ CHUẨN BỊ DỮ LIỆU
#---------------------------------------------------

# 1.1. Cài đặt thư viện AutoGluon
# Lệnh này cần thiết vì AutoGluon không có sẵn trên Kaggle
!pip install autogluon --quiet

import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

print("="*50)
print("PHẦN 1: ĐANG TẢI VÀ CHUẨN BỊ DỮ LIỆU")
print("="*50)

# 1.2. Khai báo đường dẫn và tải dữ liệu
path_toxic_comment = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv'
path_unintended_bias = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv'

try:
    df_toxic = pd.read_csv(path_toxic_comment)
    df_bias = pd.read_csv(path_unintended_bias)

    # Ghép nối và chuẩn hóa dữ liệu như các bước trước
    df_toxic_subset = df_toxic[['comment_text', 'toxic']]
    df_bias_subset = df_bias[['comment_text', 'toxic']]
    full_train_df = pd.concat([df_toxic_subset, df_bias_subset], ignore_index=True)
    full_train_df['toxic'] = full_train_df['toxic'].apply(lambda x: 1 if x >= 0.5 else 0)
    
    # Để chạy nhanh hơn cho ví dụ này, chúng ta sẽ lấy một mẫu nhỏ
    # BỎ CHÚ THÍCH DÒNG DƯỚI ĐÂY NẾU BẠN MUỐN CHẠY TRÊN TOÀN BỘ DỮ LIỆU (sẽ mất rất nhiều thời gian)
    full_train_df = full_train_df.sample(n=100000, random_state=42)
    
    print(f"Đã tạo DataFrame training tổng hợp với {len(full_train_df)} mẫu.")

except FileNotFoundError as e:
    print(f"\nLỖI: Không tìm thấy file training. Quy trình dừng lại.")
    print(f"Chi tiết lỗi: {e}")
    exit()

#---------------------------------------------------
# PHẦN 2: CHIA DỮ LIỆU (80% TRAIN, 20% TEST)
#---------------------------------------------------
print("\n" + "="*50)
print("PHẦN 2: CHIA DỮ LIỆU THÀNH TẬP TRAIN VÀ TEST")
print("="*50)

# Chia dữ liệu thành 80% train và 20% test (để đánh giá cuối cùng)
# stratify=full_train_df['toxic'] rất quan trọng để đảm bảo tỉ lệ nhãn 'toxic'
# là như nhau trong cả hai tập train và test.
train_data, test_data = train_test_split(
    full_train_df,
    test_size=0.2,
    random_state=42,
    stratify=full_train_df['toxic']
)

print(f"Kích thước tập Train: {train_data.shape}")
print(f"Kích thước tập Test: {test_data.shape}")
print(f"Phân phối nhãn trong tập Train:\n{train_data['toxic'].value_counts(normalize=True)}")
print(f"Phân phối nhãn trong tập Test:\n{test_data['toxic'].value_counts(normalize=True)}")


#---------------------------------------------------
# PHẦN 3: HUẤN LUYỆN VỚI AUTOGLUON
#---------------------------------------------------
print("\n" + "="*50)
print("PHẦN 3: BẮT ĐẦU HUẤN LUYỆN VỚI AUTOGLUON")
print("="*50)

# Khởi tạo TabularPredictor
# AutoGluon sẽ tự động xử lý cột 'comment_text' như một đặc trưng văn bản
predictor = TabularPredictor(
    label='toxic',                # Cột mục tiêu cần dự đoán
    problem_type='binary',        # Loại bài toán: phân loại nhị phân
    eval_metric='roc_auc',        # Thước đo để tối ưu, phù hợp với cuộc thi
    path='./ag_models_toxic'      # Thư mục để lưu các mô hình đã huấn luyện
)

# Huấn luyện mô hình
# AutoGluon sẽ thử nhiều mô hình khác nhau và kết hợp chúng lại
# time_limit là giới hạn thời gian huấn luyện (tính bằng giây)
# presets='best_quality' để có kết quả tốt nhất, bạn có thể dùng 'high_quality' hoặc 'medium_quality' để nhanh hơn
predictor.fit(
    train_data,
    time_limit=1800, # Giới hạn thời gian 30 phút. Tăng lên để có kết quả tốt hơn.
    presets='high_quality'
)

#---------------------------------------------------
# PHẦN 4: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST
#---------------------------------------------------
print("\n" + "="*50)
print("PHẦN 4: ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH")
print("="*50)

# Xem bảng xếp hạng các mô hình đã được huấn luyện
print("Bảng xếp hạng các mô hình (đánh giá trên tập validation nội bộ của AutoGluon):")
leaderboard = predictor.leaderboard(silent=True)
print(leaderboard)

# Đánh giá hiệu suất trên tập test 20% mà chúng ta đã tách ra
print("\nĐánh giá trên tập test (20% dữ liệu giữ lại):")
performance = predictor.evaluate(test_data)
print(performance)

