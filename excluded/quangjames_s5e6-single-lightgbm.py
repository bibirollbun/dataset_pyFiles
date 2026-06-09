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


from IPython.display import display, HTML
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import torch


train_file ='/kaggle/input/playground-series-s5e6/train.csv'
test_file ='/kaggle/input/playground-series-s5e6/test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
print(train_data.info())
print(test_data.info())


id_column = 'id'
target_column = 'Fertilizer Name'
numeric_features = [col for col in train_data.select_dtypes(include = ['int64', 'float64']).columns
                   if col != target_column and col != id_column]
categorical_features = [col for col in train_data.select_dtypes(include = ['object', 'category']).columns
                       if col != target_column and col != id_column]
print(f'All numerical features are: {numeric_features}')
print(f'All categorical features are: {categorical_features}')


print(train_data.describe(include='all'))


target_values = train_data[target_column].value_counts()
display(HTML("<span style='color: red; font-weight: bold;'> Distribution of Target values </span>"))
print(target_values)
total_target_values = target_values.sum()
display(HTML(f"<span style ='color:green; font-weight:bold'> Target values total: {total_target_values:,} </span>"))

percentages = target_values / total_target_values * 100
# Visualization
display(HTML("<span style= 'color:blue; font-weight: bold'> Target values distribution visualization </span>"))
plt.figure(figsize=(10, 4))
bars = plt.bar(target_values.index, target_values.values, color='cornflowerblue')

for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

plt.title('Target Distribution', fontsize=14)
plt.xlabel(f'{target_column}', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Đọc dữ liệu
train_file = '/kaggle/input/playground-series-s5e6/train.csv'
test_file = '/kaggle/input/playground-series-s5e6/test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Data preparation
train_df = train_data.copy().drop('id', axis=1)
test_ids = test_data['id'].copy()
test_df = test_data.copy().drop('id', axis=1)

# Hàm thêm đặc trưng
# def add_features(df):
#     df['Temp_Humidity'] = df['Temparature'] / (df['Humidity'] + 1e-5)
#     df['Sum_Temp_Humidity'] = df['Temparature'] + df['Humidity']
#     df['Temp_Moisture'] = df['Temparature'] / (df['Moisture'] + 1e-5)
#     df['Sum_Temp_Moisture'] = df['Temparature'] + df['Moisture']
#     df['Moisture_Humidity'] = df['Moisture'] / (df['Humidity'] + 1e-5)
#     df['Sum_Temp_Humidity'] = df['Moisture'] + df['Humidity']
#     df['Total_Moisture_Humidity'] = df['Temparature'] + df['Moisture'] + df['Humidity']
    
#     # df['Nitrogen_Potassium'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
#     # df['Sum_Nitrogen_Potassium'] = df['Nitrogen'] + df['Potassium']
#     # df['Nitrogen_Phosphorous'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
#     # df['Sum_Nitrogen_Phosphorous'] = df['Nitrogen'] + df['Phosphorous']
#     # df['Potassium_Phosphorous'] = df['Potassium'] / (df['Phosphorous'] + 1e-5)
#     # df['Sum_Potassium_Phosphorous'] = df['Potassium'] + df['Phosphorous']
#     # df['Total_Nutrients'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
#     # df['Soil_Crop'] = df['Soil Type'] + "_" + df['Crop Type']
#     # df['Temp_Humidity'] = df['Temparature'] * df['Humidity']    
#     # df['Potassium_Phosphorous'] = df['Potassium'] / (df['Phosphorous'] + 1e-5)
#     return df

# train_df = add_features(train_df)
# test_df = add_features(test_df)

# Định nghĩa các biến
id_column = 'id'
target_column = 'Fertilizer Name'
numeric_features = [col for col in train_df.select_dtypes(include=['int64', 'float64']).columns 
                    if col != target_column and col != id_column]
categorical_features = [col for col in train_df.select_dtypes(include=['object']).columns 
                        if col != target_column and col != id_column]

# # Outlier handling
# def clip_outliers(df, column):
#     q1 = df[column].quantile(0.25)
#     q3 = df[column].quantile(0.75)
#     iqr = q3 - q1
#     upper_bound = q3 + 1.5 * iqr
#     lower_bound = q1 - 1.5 * iqr
#     df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
#     return df

# for col in numeric_features:
#     train_df[col] = np.log1p(train_df[col])
#     test_df[col] = np.log1p(test_df[col])
#     train_df = clip_outliers(train_df, col)
#     test_df = clip_outliers(test_df, col)

# Kiểm tra giá trị thiếu
print("Missing values in train_df:\n", train_df.isnull().sum())
print("Missing values in test_df:\n", test_df.isnull().sum())

label_encoders = {}

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Encode Target column
le_target = LabelEncoder()
train_df[target_column] = le_target.fit_transform(train_df[target_column])
label_encoders[target_column] = le_target

# Tách đặc trưng và mục tiêu
X = train_df.drop(target_column, axis=1)
y = train_df[target_column]

# Chia tập train thành train và validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check size
print(f"Columns of train: {X.columns.tolist()}")
print(f"Size of train: {X.shape}")
print(f"Size of test: {test_df.shape}")
print(f"Size of train for training: {X_train.shape}")
print(f"Size of validation for training: {X_val.shape}")


# Hàm tính MAP@3
def map_at_3(y_true, y_pred_prob, k=3):
    map_scores = []
    for true_label, pred_prob in zip(y_true, y_pred_prob):
        top_k_indices = np.argsort(pred_prob)[::-1][:k]
        true_label_binary = np.zeros(len(pred_prob))
        true_label_binary[true_label] = 1
        relevant = [1 if idx == true_label else 0 for idx in top_k_indices]
        precisions = []
        num_relevant = 0
        for i, rel in enumerate(relevant):
            if rel == 1:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))
        map_scores.append(np.mean(precisions) if precisions else 0)
    return np.mean(map_scores)

# # Thiết lập KFold
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Thiết lập StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Danh sách lưu trữ
train_map3_scores = []
val_map3_scores = []
models = []

# Lưu mô hình tốt nhất
best_model = None
best_val_map3 = 0.0

# Huấn luyện với KFold
fold = 1
for train_idx, val_idx in kf.split(X,y):
    print(f"\nFold {fold}")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Tạo dataset cho LightGBM
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold)
    
    # Khởi tạo LightGBM
    params = {
        'objective': 'multiclass',
        'num_class': len(le_target.classes_),
        'boosting_type': 'gbdt',
        'num_leaves': 40,
        'learning_rate': 0.03,
        'max_depth': 5,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'device': 'gpu'
    }
    
    # Huấn luyện
    model = lgb.train(
        params,
        train_data,
        num_boost_round=10000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=True)]
    )
    
    # Dự đoán
    train_pred_prob = model.predict(X_train_fold)
    val_pred_prob = model.predict(X_val_fold)
    
    # Tính MAP@3
    train_map3 = map_at_3(y_train_fold, train_pred_prob)
    val_map3 = map_at_3(y_val_fold, val_pred_prob)
    
    train_map3_scores.append(train_map3)
    val_map3_scores.append(val_map3)
    
    print(f"Fold {fold} - Train MAP@3: {train_map3:.4f}, Val MAP@3: {val_map3:.4f}")
    
    if val_map3 > best_val_map3:
        best_val_map3 = val_map3
        best_model = model
    
    fold += 1

# Tính trung bình MAP@3
mean_train_map3 = np.mean(train_map3_scores)
mean_val_map3 = np.mean(val_map3_scores)
print(f"\nMean Train MAP@3: {mean_train_map3:.4f}")
print(f"Mean Val MAP@3: {mean_val_map3:.4f}")

print('LightGBM training was completed!')


# Predicting probability on test set using best model
test_pred_prob = best_model.predict(test_df)  # Dùng trực tiếp test_df, không cần DMatrix

# Get the top 3 most probable labels for each sample
top_3_indices = np.argsort(test_pred_prob, axis=1)[:, -3:][:, ::-1] 

# Decode top 3 labels from number to original name (Fertilizer Name)
top_3_labels = []
for i in range(len(top_3_indices)):
    labels = label_encoders[target_column].inverse_transform(top_3_indices[i])
    # Kết hợp các nhãn thành một chuỗi, phân tách bằng dấu cách
    top_3_labels.append(" ".join(labels))

# Tạo DataFrame submission
submission_df = pd.DataFrame({
    'id': test_ids,  # Sử dụng test_ids đã lưu trước đó
    'Fertilizer Name': top_3_labels
})

# Lưu file submission
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' has been created!")


# Check submission
print(submission_df.head())
print(submission_df.shape)

