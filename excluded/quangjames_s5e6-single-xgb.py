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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder


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


submission_file ='/kaggle/input/playground-series-s5e6/sample_submission.csv'
submission_data = pd.read_csv(submission_file)
print(submission_data.head(5))


id_column = 'id'
target_column = 'Fertilizer Name'
numeric_features = [col for col in train_data.select_dtypes(include=['int64', 'float64']).columns if col != target_column and col != id_column]
print(numeric_features)
categorical_features = [col for col in train_data.select_dtypes(include=['object']).columns if col != target_column and col != id_column]
print(categorical_features)


print(train_data.describe(include='all'))


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


for col in categorical_features:
    counts = train_data[col].value_counts()
    print(f"Value counts for column '{col}':")
    print(counts)
    print("-" * 30)


display(HTML("<span style='color: blue; font-weight: bold;'>Check NaN values of Train dataset</span>"))
print(train_data.isnull().sum())
display(HTML("<span style='color: blue; font-weight: bold;'>Check NaN values of Test dataset</span>"))
print(test_data.isnull().sum())


# Check for infinity (inf)
for col in numeric_features:
    has_inf = train_data[col].isin([float('inf'), -float('inf')]).sum()
    has_nan = train_data[col].isna().sum()
    print(f"Column {col}: {has_inf} Inf values:, {has_nan} NaN values")

# Outliers boxplot 
plt.figure(figsize=(12, 6))
train_data[numeric_features].boxplot()
plt.title('Outlier Boxplots of Numeric''s columns')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

for col in numeric_features:
    plt.figure(figsize=(6, 4))
    plt.boxplot(train_data[col].dropna())  # loại bỏ NaN để tránh lỗi khi vẽ
    plt.title(f'Boxplot 0f {col}')
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


warnings.simplefilter(action='ignore', category=FutureWarning)
for col in numeric_features:
    plt.figure(figsize=(6, 4))
    
    # Replace inf and -inf to NaN, remove NaN then
    clean_data = train_data[col].replace([np.inf, -np.inf], np.nan).dropna()
    
    sns.histplot(clean_data, kde=True)
    
    plt.title(f'Distribution of {col}')
    plt.show()


corr_matrix = train_data[numeric_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


#Read dataset
train_file = '/kaggle/input/playground-series-s5e6/train.csv'
test_file = '/kaggle/input/playground-series-s5e6/test.csv'
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Data preparation
train_df = train_data.copy().drop(id_column, axis=1)
test_ids = test_data[id_column].copy()
test_df = test_data.copy().drop(id_column, axis=1)

def add_features(df):
    df['Nitrogen_Potassium'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    df['Nitrogen_Phosphorous'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    #df['Soil_Crop'] = df['Soil Type'] + "_" + df['Crop Type']
    df['Temp_Humidity'] = df['Temparature'] * df['Humidity']
    df['Total_Nutrients'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
    df['Total_Moisture'] = df['Temparature'] + df['Humidity'] + df['Moisture']
    #df['Humidity_Moisture'] = df['Humidity'] / (df['Moisture'] + 1e-5)
    #df['Potassium_Phosphorous'] = df['Potassium'] / (df['Phosphorous'] + 1e-5)
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# Define variables
id_column = 'id'
target_column = 'Fertilizer Name'
numeric_features = [col for col in train_df.select_dtypes(include=['int64', 'float64']).columns 
                    if col != target_column and col != id_column]
categorical_features = [col for col in train_df.select_dtypes(include=['object']).columns 
                        if col != target_column and col != id_column]

# Outlier handling
def clip_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

for col in numeric_features:
    train_df[col] = np.log1p(train_df[col])
    test_df[col] = np.log1p(test_df[col])
    train_df = clip_outliers(train_df, col)
    test_df = clip_outliers(test_df, col)

# # Kiểm tra giá trị thiếu
# print("Missing values in train_df:\n", train_df.isnull().sum())
# print("Missing values in test_df:\n", test_df.isnull().sum())

# Process Categorical features
label_encoders = {}
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

# Create X, y
X = train_df.drop(target_column, axis=1)
y = train_df[target_column]

# Normalize digital features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])
test_df[numeric_features] = scaler.transform(test_df[numeric_features])

# Split the train set into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check size
print(f"Columns of train: {X.columns.tolist()}")
print(f"Size of train: {X.shape}")
print(f"Size of test: {test_df.shape}")
print(f"Size of train for training: {X_train.shape}")
print(f"Size of validation for training: {X_val.shape}")


all_train_features = X.columns.tolist()
corr_matrix = X[all_train_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


# MAP@3 function
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

# Callback options
class MAP3HistoryCallback(xgb.callback.TrainingCallback):
    def __init__(self, train_data, val_data, y_train, y_val):
        self.train_data = train_data
        self.val_data = val_data
        self.y_train = y_train
        self.y_val = y_val
        self.train_map3_history = []
        self.val_map3_history = []

    def after_iteration(self, model, epoch, evals_log):
        if epoch % 100 == 0:
            train_pred_prob = model.predict(self.train_data)
            val_pred_prob = model.predict(self.val_data)
            train_map3 = map_at_3(self.y_train, train_pred_prob)
            val_map3 = map_at_3(self.y_val, val_pred_prob)
            self.train_map3_history.append(train_map3)
            self.val_map3_history.append(val_map3)
        return False

# Define KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Danh sách lưu trữ
train_map3_scores = []
val_map3_scores = []
models = []

# XGBoost options
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoders[target_column].classes_),
    'max_depth': 6,  
    'learning_rate': 0.025, 
    'subsample': 0.95,
    'colsample_bytree': 0.85,
    'lambda': 2.0,
    'alpha': 0.75,
    'min_child_weight': 20,  
    'max_bin': 128,
    'gamma': 0.2,
    'grow_policy': 'lossguide',
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'device': 'cuda'
}

# Save best model
best_model = None
best_val_map3 = 0.0

# Training with KFold
fold = 1
for train_idx, val_idx in kf.split(X):
    print(f"\nFold {fold}")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = xgb.DMatrix(X_train_fold, label=y_train_fold)
    val_data = xgb.DMatrix(X_val_fold, label=y_val_fold)
    
    map3_callback = MAP3HistoryCallback(train_data, val_data, y_train_fold, y_val_fold)
    
    evals = [(train_data, 'train'), (val_data, 'val')]
    evals_result = {}
    model = xgb.train(
        xgb_params,
        train_data,
        num_boost_round=10000,  # Tăng lên 10000
        evals=evals,
        early_stopping_rounds=200,
        evals_result=evals_result,
        verbose_eval=200,
        callbacks=[map3_callback]
    )
    models.append(model)
    
    train_pred_prob = model.predict(train_data)
    val_pred_prob = model.predict(val_data)
    
    train_map3 = map_at_3(y_train_fold, train_pred_prob)
    val_map3 = map_at_3(y_val_fold, val_pred_prob)
    
    train_map3_scores.append(train_map3)
    val_map3_scores.append(val_map3)
    
    print(f"Fold {fold} - Train MAP@3: {train_map3:.4f}, Val MAP@3: {val_map3:.4f}")
    
    if val_map3 > best_val_map3:
        best_val_map3 = val_map3
        best_model = model
    
    # plt.figure(figsize=(8, 6))
    # plt.plot(map3_callback.train_map3_history, label='Train MAP@3', linestyle='--', color='#1f77b4')
    # plt.plot(map3_callback.val_map3_history, label='Val MAP@3', color='#ff7f0e')
    # plt.title(f'Training Progress (MAP@3) - Fold {fold}')
    # plt.xlabel('Iteration (x100)')
    # plt.ylabel('MAP@3')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    fold += 1

mean_train_map3 = np.mean(train_map3_scores)
mean_val_map3 = np.mean(val_map3_scores)
print(f"\nMean Train MAP@3: {mean_train_map3:.4f}")
print(f"Mean Val MAP@3: {mean_val_map3:.4f}")

print('XGBoost training was completed!')


# Convert test_df to DMatrix for prediction with XGBoost
test_data = xgb.DMatrix(test_df)

# Predicting probability on test set using best model
test_pred_prob = best_model.predict(test_data)

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




# Kiểm tra submission
print(submission_df.head())
print(submission_df.shape)

