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


import numpy as np
import seaborn as sns
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder,PowerTransformer
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_log_error


train_dataset = '/kaggle/input/playground-series-s5e5/train.csv'
test_dataset = '/kaggle/input/playground-series-s5e5/test.csv'

train_data = pd.read_csv(train_dataset)
test_data = pd.read_csv(test_dataset)

print(train_data.info())
print(train_data.head(5))

print(test_data.info())
print(test_data.head(5))


# Check outiine values
numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp','Calories']
# Check for infinity (inf)
for col in numeric_cols:
    has_inf = train_data[col].isin([float('inf'), -float('inf')]).sum()
    has_nan = train_data[col].isna().sum()
    print(f"Column {col}: {has_inf} Inf values:, {has_nan} NaN values")

# Outliers boxplot 
plt.figure(figsize=(12, 6))
train_data[numeric_cols].boxplot()
plt.title('Outlier Boxplots of Numeric''s columns')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
for col in ['Duration', 'Heart_Rate', 'Body_Temp', 'Calories']:
    plt.figure(figsize=(6, 4))
    
    # Thay thế inf và -inf thành NaN, sau đó loại bỏ các giá trị NaN
    clean_data = train_data[col].replace([np.inf, -np.inf], np.nan).dropna()
    
    sns.histplot(clean_data, kde=True)
    
    plt.title(f'Distribution of {col}')
    plt.show()


# Data preparation
train_df_processed = train_data.copy().drop('id', axis=1)
test_ids = test_data['id']  
test_df_processed = test_data.copy().drop('id', axis=1)

# Outlier handling
def clip_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# Áp dụng cho các cột quan trọng
for col in ['Heart_Rate', 'Duration', 'Body_Temp']:
    train_df_processed = clip_outliers(train_df_processed, col)
    test_df_processed = clip_outliers(test_df_processed, col)

# Outlier handling for Calories (only train)
q1 = train_df_processed['Calories'].quantile(0.25)
q3 = train_df_processed['Calories'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr
train_df_processed['Calories'] = train_df_processed['Calories'].clip(lower=lower_bound, upper=upper_bound)

train_df_processed['Duration'] = np.log1p(train_df_processed['Duration'])
test_df_processed['Duration'] = np.log1p(test_df_processed['Duration'])
train_df_processed['Body_Temp'] = np.log1p(train_df_processed['Body_Temp'])
test_df_processed['Body_Temp'] = np.log1p(test_df_processed['Body_Temp'])

# Create new features
def add_features(df):
    df['Heart_Rate_Duration'] = df['Heart_Rate'] * df['Duration']
    df['Age_Weight'] = df['Age'] * df['Weight']
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Log_Weight'] = np.log1p(df['Weight'])  
    df['Sqrt_Height'] = np.sqrt(df['Height'])
    # Process Duration feature
    df['Log_Duration'] = np.log1p(df['Duration'])
    df['Sqrt_Duration'] = np.sqrt(df['Duration'])
    df['Duration_Age'] = df['Duration'] * df['Age']
    df['Duration_Weight'] = df['Duration'] * df['Weight']
    df['Duration_Height'] = df['Duration'] * df['Height']
    df['Duration_Body_Temp'] =  df['Duration'] * df['Body_Temp']
    df['Heart_Rate_BMI'] = df['Heart_Rate'] * df['BMI']
    df['Duration_Squared'] = df['Duration'] ** 2
    df['Heart_Rate_Squared'] = df['Heart_Rate'] ** 2
    df['Heart_Rate_per_Duration'] = df['Heart_Rate'] / df['Duration'].replace(0, 1)  # Tránh chia cho 0
    df['Body_Temp_Heart_Rate'] = df['Body_Temp'] * df['Heart_Rate']
    df['Duration_Bin'] = pd.cut(df['Duration'], bins=[0, 7, 12, 17, 22, 27, 30], labels=[0, 1, 2, 3, 4, 5])
    df['Heart_Rate_Duration_Log'] = np.log1p(df['Heart_Rate'] * df['Duration'])
    
    return df

train_df_processed = add_features(train_df_processed)
test_df_processed = add_features(test_df_processed)

# numeric_features and categorical_features definition
numeric_features = [col for col in train_df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist() if col != 'Calories'] 
#numeric_features = [feat for feat in numeric_features if feat not in ['Height', 'BMI', 'Weight', 'Heart_Rate_BMI', 'Body_Temp']]
#print(numeric_features)
#numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Heart_Rate_Duration', 
#                    'Age_Weight', 'BMI', 'Duration_Age', 'Heart_Rate_BMI', 'Duration_Squared', 'Heart_Rate_Squared']
categorical_features = ['Sex', 'Duration_Bin']

# X and y definition
X = train_df_processed.drop('Calories', axis=1)
y = np.log1p(train_df_processed['Calories'])  #Log transformation for Calories
X_test = test_df_processed  

# Preprocessor definition
preprocessor = ColumnTransformer(
    transformers=[
        ('num', PowerTransformer(method='yeo-johnson'), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Pipeline definition
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit và transform
X = pipeline.fit_transform(X)
X_test = pipeline.transform(X_test)

# Check size
#print(f"Column of train: {X.columns()}")
print(f"Size of train: {X.shape}")
print(f"Size of test: {X_test.shape}")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Size of train for training: {X_train.shape}")
print(f"Size of validation for training: {X_val.shape}")


# Check multicollinearity
# Use Variance Inflation Factor (VIF) to check and remove features causing multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

numeric_cols = [col for col in train_df_processed.select_dtypes(include=['int64', 'float64']).columns if col != 'Calories']
vif_df = calculate_vif(train_df_processed[numeric_cols])
print(vif_df)

# Loại bỏ các đặc trưng có VIF > 10
#high_vif_features = vif_df[vif_df['VIF'] > 10]['feature'].tolist()
#train_df_processed.drop(columns=high_vif_features, inplace=True)
#test_df_processed.drop(columns=high_vif_features, inplace=True)


for col in ['Duration', 'Heart_Rate', 'Body_Temp', 'Calories']:
    plt.figure(figsize=(6, 4))
    
    # Thay thế inf và -inf thành NaN, sau đó loại bỏ các giá trị NaN
    clean_data = train_df_processed[col].replace([np.inf, -np.inf], np.nan).dropna()
    
    sns.histplot(clean_data, kde=True)
    
    plt.title(f'Distribution of {col}')
    plt.show()


# Hàm tính RMSLE (đơn giản hóa và tái sử dụng từ sklearn)
def rmsle(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)  # Hoàn nguyên log1p
    y_pred = np.expm1(y_pred_log)
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Callback tùy chỉnh để lưu lịch sử RMSLE
class RMSLEHistoryCallback(xgb.callback.TrainingCallback):
    def __init__(self, train_data, val_data, y_train, y_val):
        self.train_data = train_data
        self.val_data = val_data
        self.y_train = y_train
        self.y_val = y_val
        self.train_rmsle_history = []
        self.val_rmsle_history = []

    def after_iteration(self, model, epoch, evals_log):
        # Giảm tần suất tính RMSLE để tăng tốc
        if epoch % 100 == 0:  # Chỉ tính mỗi 100 vòng lặp
            train_pred = model.predict(self.train_data)
            val_pred = model.predict(self.val_data)
            train_rmsle = rmsle(self.y_train, train_pred)
            val_rmsle = rmsle(self.y_val, val_pred)
            self.train_rmsle_history.append(train_rmsle)
            self.val_rmsle_history.append(val_rmsle)
        return False

def calculate_weights(y):
    weights = np.ones(len(y))
    high_cal_indices = y > 100  # Ngưỡng tùy chọn
    weights[high_cal_indices] = 4.0  # Tăng trọng số cho giá trị cao
    return weights
    
# Thiết lập KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Danh sách lưu trữ RMSLE
train_rmsle_scores = []
val_rmsle_scores = []

# List of models from each fold (for ensemble)
models = []

# Tham số XGBoost (tinh chỉnh để giảm thời gian huấn luyện)
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.0075,
    'subsample': 0.95,
    'colsample_bytree': 0.8,
    'lambda': 0.75,
    'alpha': 0.5,
    'min_child_weight': 10,
    'max_bin': 512,
    'gamma': 0.1,
    'grow_policy': 'lossguide',
    'random_state': 42,    
    'eval_metric': 'rmse',
    'tree_method': 'hist', 
    'device': 'cuda'
}

# Lưu mô hình tốt nhất từ các fold
best_model = None
best_val_rmsle = float('inf')

# Huấn luyện với KFold
fold = 1
for train_idx, val_idx in kf.split(X):
    print(f"\nFold {fold}")
    
    # Chia dữ liệu
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Tạo dataset cho XGBoost
    weights = calculate_weights(y[train_idx])
    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)
    
    # Khởi tạo callback
    rmsle_callback = RMSLEHistoryCallback(train_data, val_data, y_train, y_val)
    
    # Huấn luyện mô hình
    evals = [(train_data, 'train'), (val_data, 'val')]
    evals_result = {}
    model = xgb.train(
        xgb_params,
        train_data,
        num_boost_round=100000,  # Giảm số vòng lặp để tăng tốc
        evals=evals,
        early_stopping_rounds=300,  # Giảm early stopping rounds
        evals_result=evals_result,
        verbose_eval=500,
        callbacks=[rmsle_callback]
    )
    models.append(model)
    
    # Dự đoán và tính RMSLE
    y_train_pred = model.predict(train_data)
    y_val_pred = model.predict(val_data)
    
    train_rmsle = rmsle(y_train, y_train_pred)
    val_rmsle = rmsle(y_val, y_val_pred)
    
    train_rmsle_scores.append(train_rmsle)
    val_rmsle_scores.append(val_rmsle)
    
    print(f"Fold {fold} - Train RMSLE: {train_rmsle:.4f}, Val RMSLE: {val_rmsle:.4f}")
    
    # Lưu mô hình tốt nhất
    if val_rmsle < best_val_rmsle:
        best_val_rmsle = val_rmsle
        best_model = model
    
    # Vẽ đồ thị sau mỗi fold
    plt.figure(figsize=(8, 6))
    plt.plot(rmsle_callback.train_rmsle_history, label='Train RMSLE', linestyle='--', color='#1f77b4')
    plt.plot(rmsle_callback.val_rmsle_history, label='Val RMSLE', color='#ff7f0e')
    plt.title(f'Training Progress (RMSLE) - Fold {fold}')
    plt.xlabel('Iteration (x100)')
    plt.ylabel('RMSLE')
    plt.legend()
    plt.grid(True)
    #plt.savefig(f'xgb_fold_{fold}_training_progress.png')  # Lưu đồ thị
    plt.show()
    #plt.close()  # Đóng figure để tránh hiển thị chồng chéo
    
    fold += 1

# Tính RMSLE trung bình
mean_train_rmsle = np.mean(train_rmsle_scores)
mean_val_rmsle = np.mean(val_rmsle_scores)
print(f"\nMean Train RMSLE: {mean_train_rmsle:.4f}")
print(f"Mean Val RMSLE: {mean_val_rmsle:.4f}")

print('XGBoot training was completed!')


residuals = np.expm1(y_val) - np.expm1(y_val_pred)
plt.scatter(np.expm1(y_val), residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('True Calories')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# Get the importance of all features
feature_importance = model.get_score(importance_type='gain')

feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

importance_list = list(zip(feature_names, feature_importance))

importance_list.sort(key=lambda x: x[1], reverse=True)

print("Feature Importance List (All Features):")
for feature, importance in importance_list:
    print(f"{feature}: {importance}")

print("\nTop 10 Most Important Features:")
for feature, importance in importance_list[:10]:
    print(f"{feature}: {importance}")


sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
print("Feature Importance:", sorted_importance)
important_features = [f for f, score in sorted_importance if score > 1.0]

xgb.plot_importance(model, max_num_features=10)
plt.show()


# Dự đoán trên tập test
test_data = xgb.DMatrix(X_test)
y_test_pred_log = best_model.predict(test_data)
y_test_pred = np.expm1(y_test_pred_log)
y_test_pred = np.clip(y_test_pred, 0, None)

submission = pd.DataFrame({
    'id': test_ids,
    'Calories': y_test_pred
})
submission.to_csv('xgb_submission.csv', index=False)

print('Predict and save the submission were completed!')




