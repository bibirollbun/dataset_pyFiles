import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import lightgbm as lgb
from datetime import datetime


# データの読み込み
# Load the datasets
try:
    # Kaggleノートブックでの正しいファイルパスを指定
    # Specify the correct file paths for Kaggle notebooks
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
    print("データセットが正常に読み込まれました。")
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("ファイルが見つかりません。Kaggleノートブックにデータセットが追加されているか、パスが正しいか確認してください。")
    print("Files not found. Please ensure the dataset is added to your Kaggle notebook and the paths are correct.")
    exit()


# データセットの結合（前処理を効率化するため）
# Combine datasets for consistent preprocessing
# Make a copy to avoid SettingWithCopyWarning later
train_ids = train_df['id']
test_ids = test_df['id']


# Drop 'id' column as it's not a feature
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# ターゲット変数を分離
# Separate the target variable
X = train_df.drop('Personality', axis=1)
y = train_df['Personality']

# テストデータにはターゲット変数がないため、特徴量のみを扱う
# Test data only contains features, no target variable
X_test = test_df.copy()

# 欠損値の確認
# Check for missing values
print("\n訓練データの欠損値の数:")
print("Missing values in training data:")
print(X.isnull().sum())
print("\nテストデータの欠損値の数:")
print("Missing values in test data:")
print(X_test.isnull().sum())


# 欠損値の補完
# Impute missing values

# 数値列のリスト
# List of numerical columns
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

# カテゴリカル列のリスト
# List of categorical columns
categorical_cols = ['Stage_fear', 'Drained_after_socializing']


# 数値列の欠損値を中央値で補完
# Impute missing numerical values with the median
for col in numerical_cols:
    if col in X.columns:
        median_val = X[col].median()
        # 修正: inplace=True を使わず、直接割り当てる
        # Fix: Assign directly instead of using inplace=True
        X[col] = X[col].fillna(median_val)
    if col in X_test.columns:
        # Use median from training data to avoid data leakage
        median_val_test = X[col].median() # Use median from training data for test set
        # 修正: inplace=True を使わず、直接割り当てる
        # Fix: Assign directly instead of using inplace=True
        X_test[col] = X_test[col].fillna(median_val_test)


# カテゴリカル列の欠損値を最頻値で補完
# Impute missing categorical values with the mode
for col in categorical_cols:
    if col in X.columns:
        mode_val = X[col].mode()[0]
        # 修正: inplace=True を使わず、直接割り当てる
        # Fix: Assign directly instead of using inplace=True
        X[col] = X[col].fillna(mode_val)
    if col in X_test.columns:
        # Use mode from training data to avoid data leakage
        mode_val_test = X[col].mode()[0] # Use mode from training data for test set
        # 修正: inplace=True を使わず、直接割り当てる
        # Fix: Assign directly instead of using inplace=True
        X_test[col] = X_test[col].fillna(mode_val_test)


print("\n欠損値補完後の訓練データの欠損値の数:")
print("Missing values in training data after imputation:")
print(X.isnull().sum())
print("\n欠損値補完後のテストデータの欠損値の数:")
print("Missing values in test data after imputation:")
print(X_test.isnull().sum())


# カテゴリカル変数のエンコーディング
# Encode categorical variables

# 'Stage_fear'と'Drained_after_socializing'を数値に変換
# Convert 'Stage_fear' and 'Drained_after_socializing' to numerical
# 'No' -> 0, 'Yes' -> 1
for df in [X, X_test]:
    df['Stage_fear'] = df['Stage_fear'].map({'No': 0, 'Yes': 1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No': 0, 'Yes': 1})


# ターゲット変数 'Personality' を数値に変換
# Convert target variable 'Personality' to numerical
# 'Introvert' -> 0, 'Extrovert' -> 1
le = LabelEncoder()
y_encoded = le.fit_transform(y) # y_encoded will be 0s and 1s


# モデルの訓練
# Train the model
# LightGBM Classifierを使用
# Using LightGBM Classifier
model = lgb.LGBMClassifier(random_state=42, n_jobs=-1) # n_jobs=-1 for parallel processing
model.fit(X, y_encoded)


# テストデータでの予測
# Predict on the test data
predictions_encoded = model.predict(X_test)


# 予測結果を元のカテゴリカルな名前に戻す
# Convert predictions back to original categorical names
predictions = le.inverse_transform(predictions_encoded)


# 提出ファイルの作成
# Create the submission file
submission_df = pd.DataFrame({'id': test_ids, 'Personality': predictions})


# 現在の日時を取得し、ファイル名に含める
# Get current datetime and include it in the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_filename = f'submission_{timestamp}.csv'

# 提出ファイルをCSVとして保存
# Save the submission file as CSV
submission_df.to_csv(submission_filename, index=False)

print(f"\n予測が完了し、'{submission_filename}' が生成されました。")
print(f"Predictions complete, '{submission_filename}' generated.")
print("\nすべての予測結果:")
print("All prediction results:")
print(submission_df) # submission_df全体を出力





