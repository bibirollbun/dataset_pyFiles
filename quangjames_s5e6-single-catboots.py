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


import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


#------------- TRAIN DATASET
train_file = '/kaggle/input/playground-series-s5e6/train.csv'

train_data = pd.read_csv(train_file)

display(HTML("<span style='color: red; font-weight: bold;'>Overview information of Train dataset</span>"))
print(train_data.info())
display(HTML("<span style='color: blue; font-weight: bold;'>10 first samples of Train dataset</span>"))
print(train_data.head(10))

#------------- TEST DATASET

test_file = '/kaggle/input/playground-series-s5e6/test.csv'

test_data = pd.read_csv(test_file)

display(HTML("<span style='color: red; font-weight: bold;'>Overview information of Test dataset</span>"))
print(test_data.info())
display(HTML("<span style='color: blue; font-weight: bold;'>10 first samples of Test dataset</span>"))
print(test_data.head(10))


id_column = 'id'
target_column = 'Fertilizer Name'
numeric_features = [col for col in train_data.select_dtypes(include=['int64', 'float64']).columns if col != target_column and col != id_column]
print(numeric_features)
categorical_features = [col for col in train_data.select_dtypes(include=['object']).columns if col != target_column and col != id_column]
print(categorical_features)


target_counts = train_data[target_column].value_counts()
print(target_counts)

target_total_count = target_counts.sum()
print("Total of target_counts:", target_total_count)

# Tính phần trăm
percentages = target_counts / target_total_count * 100

# Vẽ biểu đồ cột
plt.figure(figsize=(10, 6))
bars = plt.bar(target_counts.index, target_counts.values, color='cornflowerblue')

# Thêm nhãn phần trăm trên mỗi cột
for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

# Cấu hình biểu đồ
plt.title('Target Distribution', fontsize=14)
plt.xlabel(f'{target_column}', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Read dataset
train_file = '/kaggle/input/playground-series-s5e6/train.csv'
test_file = '/kaggle/input/playground-series-s5e6/test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Data preparation
train_df = train_data.copy().drop(id_column, axis=1)
test_ids = test_data[id_column].copy()
test_df = test_data.copy().drop(id_column, axis=1)

# # Add more features
# def add_features(df):
#     #df['Temp_Humidity'] = df['Temparature'] / (df['Humidity'] + 1e-5)
#     df['Sum_Temp_Humidity'] = df['Temparature'] + df['Humidity']
#     #df['Temp_Moisture'] = df['Temparature'] / (df['Moisture'] + 1e-5)
#     df['Sum_Temp_Moisture'] = df['Temparature'] + df['Moisture']
#     #df['Moisture_Humidity'] = df['Moisture'] / (df['Humidity'] + 1e-5)
#     df['Sum_Temp_Humidity'] = df['Moisture'] + df['Humidity']
#     df['Total_Moisture_Humidity'] = df['Temparature'] + df['Moisture'] + df['Humidity']
    
#     # df['Nitrogen_Potassium'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
#     # df['Sum_Nitrogen_Potassium'] = df['Nitrogen'] + df['Potassium']
#     # df['Nitrogen_Phosphorous'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
#     # df['Sum_Nitrogen_Phosphorous'] = df['Nitrogen'] + df['Phosphorous']
#     # df['Potassium_Phosphorous'] = df['Potassium'] / (df['Phosphorous'] + 1e-5)
#     # df['Sum_Potassium_Phosphorous'] = df['Potassium'] + df['Phosphorous']
#     # df['Total_Nutrients'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
#     df['Soil_Crop'] = df['Soil Type'] + "_" + df['Crop Type']
#     # df['Temp_Humidity'] = df['Temparature'] * df['Humidity']    
#     # df['Potassium_Phosphorous'] = df['Potassium'] / (df['Phosphorous'] + 1e-5)
#     return df

# train_df = add_features(train_df)
# test_df = add_features(test_df)

# Variable defination
id_column = 'id'
target_column = 'Fertilizer Name'
numeric_features = [col for col in train_df.select_dtypes(include=['int64', 'float64']).columns 
                   if col != target_column and col != id_column]
categorical_features = [col for col in train_df.select_dtypes(include=['object']).columns 
                       if col != target_column and col != id_column]

# Check for missing values
print("Missing values in train_df:\n", train_df.isnull().sum())
print("Missing values in test_df:\n", test_df.isnull().sum())

# Synchronize column structure between train_df and test_df
all_columns = numeric_features + categorical_features + [target_column] 
for col in all_columns:
    if col not in test_df.columns:
        test_df[col] = np.nan

# Rearrange the column order for test_df to match train_df
test_df = test_df[all_columns]

# Split the target before transforming (only for train_df)
X = train_df.drop(target_column, axis=1)  # Đặc trưng
y = train_df[target_column]  # Nhãn

# Preprocessor definition (chỉ áp dụng cho numeric_features)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', PowerTransformer(method='yeo-johnson'), numeric_features)
    ],
    remainder='passthrough'  # Giữ nguyên các cột categorical
)

# Pipeline definition
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit và transform đồng thời cho train và test
X_transformed = pipeline.fit_transform(X)
test_df_transformed = pipeline.transform(test_df.drop(target_column, axis=1))  # Loại bỏ target khỏi test_df

# Chuyển đổi mảng NumPy về DataFrame để dễ quản lý
feature_names = numeric_features + categorical_features
X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
test_df_transformed_df = pd.DataFrame(test_df_transformed, columns=feature_names)

# Encode Target column
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
label_encoders = {target_column: le_target}

# Tách đặc trưng và mục tiêu
X = X_transformed_df
y = y_encoded

# Chia tập train thành train và validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check size
print(f"Columns of train: {X.columns.tolist()}")
print(f"Size of train: {X.shape}")
print(f"Size of test: {test_df_transformed_df.shape}")
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

# Thiết lập KFold
#kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Danh sách lưu trữ
train_map3_scores = []
val_map3_scores = []
models = []

# Định nghĩa biến hạng mục (CatBoost sẽ xử lý trực tiếp)
cat_features = [X.columns.get_loc(col) for col in categorical_features]  # Chuyển tên cột thành chỉ số

# Lưu mô hình tốt nhất
best_model = None
best_val_map3 = 0.0

# Huấn luyện với KFold
fold = 1
for train_idx, val_idx in kf.split(X, y):
    print(f"\nFold {fold}")
    
    # Sử dụng .iloc để index theo hàng
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y[train_idx]  # y là mảng NumPy, dùng indexing trực tiếp
    y_val_fold = y[val_idx]
    
    # Tạo Pool cho CatBoost
    train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_features)
    val_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_features)
    
    # Khởi tạo CatBoost với tham số ban đầu
    model = CatBoostClassifier(
        iterations=20000,
        depth=5,
        learning_rate=0.01,
        l2_leaf_reg=8,
        random_seed=42,
        cat_features=cat_features,
        task_type='GPU',
        early_stopping_rounds=400,
        verbose=200
    )
    
    # Huấn luyện
    model.fit(train_pool, eval_set=val_pool)
    
    # Dự đoán
    train_pred_prob = model.predict_proba(X_train_fold)
    val_pred_prob = model.predict_proba(X_val_fold)
    
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

print('CatBoost training was completed!')


from catboost import Pool

# Tạo Pool cho test_df_transformed_df với cat_features
cat_features_indices = [test_df_transformed_df.columns.get_loc(col) for col in categorical_features]
test_pool = Pool(
    data=test_df_transformed_df,
    cat_features=cat_features_indices
)

# Predicting probability on test set using best model
test_pred_prob = best_model.predict_proba(test_pool)  # Sử dụng predict_proba

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

