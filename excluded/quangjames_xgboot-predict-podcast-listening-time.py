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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from category_encoders import TargetEncoder
from itertools import combinations
from xgboost import XGBRegressor, DMatrix, callback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import xgboost as xgb
import torch



# Pre-processing function
# is_train = True: Input data is Train set
# is_train = False: Input data is Test set

# Dictionary toàn cục để lưu LabelEncoder cho các tổ hợp đặc trưng
# Biến toàn cục
label_encoders = {}
target_encoder = None
scaler = None

def preprocess_data(df, is_train=True, encoder_file='label_encoders.pkl', target_encoder_file='target_encoder.pkl', scaler_file='scaler.pkl'):
    global label_encoders, target_encoder, scaler
    df_processed = df.copy()    

    df_processed = df_processed.drop_duplicates()
    print("Start Data Pre-Processing..........")
    print(f"Total columns now: {len(df_processed.columns)}")
    print("Step 1: Process data imputation.........")
    genre_means = df_processed.groupby('Genre')['Episode_Length_minutes'].mean()
    df_processed['Episode_Length_minutes'] = df_processed['Episode_Length_minutes'].fillna(
                                            df_processed['Genre'].map(genre_means))

    df_processed['Host_Popularity_Bin'] = pd.qcut(df_processed['Host_Popularity_percentage'], q=5, duplicates='drop')
    host_means = df_processed.groupby('Host_Popularity_Bin', observed=False)['Guest_Popularity_percentage'].mean()
    df_processed['Guest_Popularity_percentage'] = df_processed.apply(
        lambda row: host_means[row['Host_Popularity_Bin']] if pd.isna(row['Guest_Popularity_percentage']) else row['Guest_Popularity_percentage'],
        axis=1
    )
    df_processed = df_processed.drop(columns=['Host_Popularity_Bin'])

    df_processed['Number_of_Ads'] = df_processed['Number_of_Ads'].fillna(round(df_processed['Number_of_Ads'].median()))
    print(f"Total columns now: {len(df_processed.columns)}")
    
    print("Step 2: Process Outlier values (Outlier Removal).......")
    df_processed['Episode_Length_minutes'] = np.clip(df_processed['Episode_Length_minutes'], 0, 120)
    df_processed['Host_Popularity_percentage'] = np.clip(df_processed['Host_Popularity_percentage'], 20, 100)
    df_processed['Guest_Popularity_percentage'] = np.clip(df_processed['Guest_Popularity_percentage'], 0, 100)
    df_processed.loc[df_processed['Number_of_Ads'] > 3, 'Number_of_Ads'] = 0
    print(f"Total columns now: {len(df_processed.columns)}")
    
    print("Step 3: Add more new features by feature engineering.........")
    df_processed['Episode_Length_minutes_sqrt'] = np.sqrt(df_processed['Episode_Length_minutes'])
    df_processed['Episode_Length_minutes_squared'] = df_processed['Episode_Length_minutes'] ** 2
    df_processed['Is_Weekend'] = df_processed['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df_processed['Episode_Number'] = df_processed['Episode_Title'].str.extract('(\d+)').astype(int)
    #df_processed['Episode_Title_Length'] = df_processed['Episode_Title'].str.len()

    print("One-hot encode Genre........")
    genre_encoded = pd.get_dummies(df_processed['Genre'], prefix='Genre')
    df_processed = pd.concat([df_processed, genre_encoded], axis=1)
    for col in genre_encoded.columns:
        df_processed[f'{col}_Length_Interaction'] = genre_encoded[col] * df_processed['Episode_Length_minutes']
    print(f"Total columns now: {len(df_processed.columns)}")
    
    print("Step 4: Basic Categorical Encoding.......")
    day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    df_processed['Publication_Day'] = df_processed['Publication_Day'].map(day_mapping)
    
    time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
    df_processed['Publication_Time'] = df_processed['Publication_Time'].map(time_mapping)
    
    sentiment_map = {'Negative': 1, 'Neutral': 2, 'Positive': 3}
    df_processed['Episode_Sentiment'] = df_processed['Episode_Sentiment'].map(sentiment_map)
    print(f"Total columns now: {len(df_processed.columns)}")
    
    print("Step 5: Feature combination and encoding...........")
    columns_to_encode = [
        'Episode_Length_minutes', 'Episode_Title', 'Publication_Time', 'Host_Popularity_percentage',
        'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Podcast_Name', 'Genre',
        'Guest_Popularity_percentage'
    ]
    pair_size = [2, 3]
    
    for col in columns_to_encode:
        if col not in df_processed.columns:
            raise ValueError(f"Cột {col} không tồn tại trong DataFrame")
    
    if is_train and os.path.exists(encoder_file):
        os.remove(encoder_file)
    elif not is_train and os.path.exists(encoder_file):
        with open(encoder_file, 'rb') as f:
            label_encoders = pickle.load(f)
    
    new_columns = {}
    for r in pair_size:
        combos = list(combinations(columns_to_encode, r))
        print(f"Total number of {r}-column combinations: {len(combos)}")
        for idx, cols in enumerate(tqdm(combos, desc=f"Processing {r}-column combinations")):
            col_name = '_'.join(cols)
            combined = df_processed[cols[0]].astype(str)
            for col in cols[1:]:
                combined += '_' + df_processed[col].astype(str)
            if is_train:
                le = LabelEncoder()
                new_columns[col_name] = le.fit_transform(combined) + 1
                label_encoders[col_name] = le
            else:
                if col_name not in label_encoders:
                    raise ValueError(f"Không tìm thấy LabelEncoder cho {col_name}. Đảm bảo đã xử lý train trước.")
                le = label_encoders[col_name]
                try:
                    new_columns[col_name] = le.transform(combined) + 1
                except ValueError:
                    unique_values = set(combined)
                    known_values = set(le.classes_)
                    unknown_mask = combined.isin(unique_values - known_values)
                    new_columns[col_name] = le.transform(combined.where(~unknown_mask, le.classes_[0])) + 1
    
    new_columns_df = pd.DataFrame(new_columns, index=df_processed.index)
    df_processed = pd.concat([df_processed, new_columns_df], axis=1)   
    print(f"Total columns now: {len(df_processed.columns)}")

    if is_train:
        with open(encoder_file, 'wb') as f:
            pickle.dump(label_encoders, f)

    print("Step 6: Target Encoding and Outline columns removal")
    if is_train:
        target_encoder = TargetEncoder()
        for col in ['Genre', 'Podcast_Name', 'Publication_Day']:
            df_processed[col + '_te'] = target_encoder.fit_transform(df_processed[[col]], df_processed['Listening_Time_minutes'])
        with open(target_encoder_file, 'wb') as f:
            pickle.dump(target_encoder, f)
    else:
        if os.path.exists(target_encoder_file):
            with open(target_encoder_file, 'rb') as f:
                target_encoder = pickle.load(f)
        else:
            raise ValueError(f"File {target_encoder_file} không tồn tại. Đảm bảo đã xử lý tập train trước.")
        for col in ['Genre', 'Podcast_Name', 'Publication_Day']:
            df_processed[col + '_te'] = target_encoder.transform(df_processed[[col]])
    
    df_processed = df_processed.drop(columns=['Genre', 'Podcast_Name', 'Publication_Day'])
    print(f"Total columns now: {len(df_processed.columns)}")

    print("Step 7: Remove unnecessary columns")
    df_processed = df_processed.drop(columns=['Episode_Title'])
    print(f"Total columns now: {len(df_processed.columns)}")

    print("Step 8: Numeric encoding")
    numeric_cols = [
        col for col in df_processed.columns
        if col not in ['id', 'Listening_Time_minutes'] and df_processed[col].dtype in [np.float64, np.int64]
    ]
    
    if is_train:
        if os.path.exists(scaler_file):
            os.remove(scaler_file)
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
        else:
            raise ValueError(f"File {scaler_file} không tồn tại. Đảm bảo đã xử lý tập train trước.")
        
        scaler_cols = scaler.feature_names_in_
        missing_cols = [col for col in scaler_cols if col not in numeric_cols]
        extra_cols = [col for col in numeric_cols if col not in scaler_cols]
        if missing_cols or extra_cols:
            for col in missing_cols:
                df_processed[col] = 0
                numeric_cols.append(col)
            numeric_cols = [col for col in numeric_cols if col not in extra_cols]
        
        numeric_cols = [col for col in scaler_cols if col in numeric_cols]
        df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])
    print(f"Total columns now: {len(df_processed.columns)}")

    if is_train:
        df_processed = df_processed.drop(columns=['id'])

    print("Pre-processing was completed!")
    df_processed = df_processed.astype('float32')
    return df_processed


# Đọc dữ liệu
train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')

# Tiền xử lý train
train_processed = preprocess_data(train_data, is_train=True)

# Chuẩn bị dữ liệu huấn luyện
X = train_processed.drop(columns=['Listening_Time_minutes'])
y = train_processed['Listening_Time_minutes']


test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_processed = preprocess_data(test_data, is_train=False, encoder_file='label_encoders.pkl', target_encoder_file='target_encoder.pkl')


# Kiểm tra phân phối y
y = train_processed['Listening_Time_minutes']
print("Phân phối Listening_Time_minutes:")
print(y.describe())
print("Số lượng y < 1:", (y < 1).sum())

# Clip y để giảm outlier
y = np.clip(y, 0, y.quantile(0.99))

# 2. Kiểm tra và căn chỉnh cột
train_columns = set(X.columns)
test_columns = set(test_processed.columns)
missing_in_test = train_columns - test_columns
missing_in_train = test_columns - train_columns
print(f"Cột thiếu trong test: {missing_in_test}")
print(f"Cột thừa trong test: {missing_in_train}")

# Thêm cột thiếu trong test
for col in missing_in_test:
    test_processed[col] = 0
    print(f"Đã thêm cột {col} vào test với giá trị 0")

# Loại bỏ cột thừa trong test (trừ 'id')
for col in missing_in_train:
    if col != 'id':
        test_processed = test_processed.drop(columns=[col])
        print(f"Đã xóa cột thừa {col} khỏi test")

# Đảm bảo thứ tự cột
common_columns = X.columns
test_processed = test_processed[[col for col in common_columns if col in test_processed.columns] + ['id']]

# 3. Chia dữ liệu thành tập train và validation (8:2)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Kích thước tập train: {X_train.shape}")
print(f"Kích thước tập validation: {X_val.shape}")


# Kiểm tra GPU và phiên bản XGBoost
print("GPU available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("XGBoost version:", xgb.__version__)

# Learning Rate Scheduler
def lr_decay(epoch):
    lr_start = 0.03  # Giảm từ 0.03 xuống 0.01
    lr_end = 0.005
    decay_speed = 0.01
    lr = lr_end + (lr_start - lr_end) * np.exp(-decay_speed * epoch)
    return lr

# Trực quan hóa quá trình huấn luyện
def plot_training_progress(eval_results):
    plt.figure(figsize=(10, 6))
    train_rmse = eval_results['validation_0']['rmse']  # Train set là validation_0
    val_rmse = eval_results['validation_1']['rmse']    # Val set là validation_1
    epochs = range(1, len(train_rmse) + 1)
    plt.plot(epochs, train_rmse, label='Train RMSE')
    plt.plot(epochs, val_rmse, label='Validation RMSE')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('RMSE')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

# Cross-validation
seed = 42
cv = KFold(n_splits=7, random_state=seed, shuffle=True)
val_rmse_scores = []
val_mae_scores = []
val_r2_scores = []
val_mape_scores = []
train_rmse_scores = []  # Thêm để lưu RMSE của train
test_predictions = np.zeros(len(test_data))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"Fold {fold + 1}/7")
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Chuyển sang NumPy array để tối ưu
    X_train_np = X_train_fold.to_numpy()
    X_val_np = X_val_fold.to_numpy()
    y_train_np = y_train_fold.to_numpy()
    y_val_np = y_val_fold.to_numpy()
    X_test_np = test_processed.drop(columns=['id']).to_numpy()

    # Khai báo mô hình với tối ưu GPU
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,  # Giảm learning_rate
        max_depth=17,  # Giảm từ 12 xuống 10 để giảm overfitting
        min_child_weight=150,
        reg_alpha=15,
        reg_lambda=10,
        subsample=0.9,
        colsample_bytree=0.6,
        colsample_bynode=0.5,
        random_state=seed,
        early_stopping_rounds=100,  # Tăng từ 30 lên 50
        tree_method='hist',
        device='cuda',
        grow_policy='depthwise',
        eval_metric='rmse',
        callbacks=[callback.LearningRateScheduler(lr_decay)]
    )

    # Huấn luyện mô hình với NumPy array
    xgb_model.fit(
        X_train_np,
        y_train_np,
        eval_set=[(X_train_np, y_train_np), (X_val_np, y_val_np)],
        verbose=False  # Bật log để xem chi tiết
    )

    # Lấy kết quả đánh giá
    eval_results = xgb_model.evals_result()

    # Trực quan hóa
    plot_training_progress(eval_results)

    # Lấy RMSE của train từ eval_results
    train_rmse = eval_results['validation_0']['rmse'][-1]  # RMSE cuối cùng của train
    val_rmse = eval_results['validation_1']['rmse'][-1]    # RMSE cuối cùng của validation

    # Dự đoán và đánh giá trên tập validation
    val_pred = xgb_model.predict(X_val_np)
    rmse = np.sqrt(mean_squared_error(y_val_np, val_pred))
    mae = mean_absolute_error(y_val_np, val_pred)
    r2 = r2_score(y_val_np, val_pred)
    mape = np.mean(np.abs((y_val_np - val_pred) / np.where(y_val_np < 1, 1, y_val_np))) * 100

    val_rmse_scores.append(rmse)
    val_mae_scores.append(mae)
    val_r2_scores.append(r2)
    val_mape_scores.append(mape)
    train_rmse_scores.append(train_rmse)  # Lưu RMSE của train

    print(f"Fold {fold + 1} - Train RMSE: {train_rmse}, Val RMSE: {rmse}, MAE: {mae}, R²: {r2}, MAPE: {mape}%")

    # Dự đoán trên tập test với NumPy array
    test_pred = xgb_model.predict(X_test_np)
    test_predictions += test_pred / 7  # Trung bình dự đoán từ 7 fold

# Đánh giá trung bình trên cross-validation
print("\nCross-Validation Results:")
print(f"Average Train RMSE: {np.mean(train_rmse_scores)} (±{np.std(train_rmse_scores)})")
print(f"Average Val RMSE: {np.mean(val_rmse_scores)} (±{np.std(val_rmse_scores)})")
print(f"Average MAE: {np.mean(val_mae_scores)} (±{np.std(val_mae_scores)})")
print(f"Average R²: {np.mean(val_r2_scores)} (±{np.std(val_r2_scores)})")
print(f"Average MAPE: {np.mean(val_mape_scores)}% (±{np.std(val_mape_scores)}%)")

print("Training is completed!")



# 7. Feature importance
feature_importance = xgb_model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(12, 8))
plt.bar(feature_names[sorted_idx[:40]], feature_importance[sorted_idx[:40]])
plt.xticks(rotation=45, ha='right')
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.show()

# 8. Chọn đặc trưng quan trọng (top 20%)
important_features = feature_names[feature_importance > np.percentile(feature_importance, 80)]
print(f"Số đặc trưng quan trọng: {len(important_features)}")



# Chuẩn bị file submit
submission = pd.DataFrame({
    'id': test_data['id'],
    'Listening_Time_minutes': test_predictions
})
submission.to_csv('predict_submission.csv', index=False)
print("Đã xuất file predict_submission.csv.")

