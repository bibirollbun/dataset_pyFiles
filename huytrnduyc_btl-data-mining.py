# --- CELL 1: SETUP ---
import numpy as np
import pandas as pd
import gc # Dọn rác bộ nhớ
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
# Tắt tất cả các cảnh báo
warnings.filterwarnings('ignore')
# Cấu hình hiển thị
pd.set_option('display.max_columns', None)

# HÀM QUAN TRỌNG: Giúp giảm dung lượng RAM xuống 50%
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('RAM ban đầu: {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print('RAM sau khi giảm: {:.2f} MB'.format(end_mem))
    return df


# --- CELL 2: LOAD DATA ---
print("Đang đọc dữ liệu...")
# Load tập TRAIN
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

# Load tập TEST
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

print("Đang gộp bảng (Merge)...")
df_train = train_transaction.merge(train_identity, on='TransactionID', how='left')
df_test = test_transaction.merge(test_identity, on='TransactionID', how='left')

# Xóa biến tạm để giải phóng RAM ngay lập tức
del train_transaction, train_identity, test_transaction, test_identity
gc.collect()

print(f"Kích thước Train: {df_train.shape}")
print(f"Kích thước Test: {df_test.shape}")

# Giảm bộ nhớ
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)


# --- CELL 3: EDA ---
plt.figure(figsize=(8, 5))
sns.countplot(x='isFraud', data=df_train)
plt.title('Distribution: Fraud (1) vs Normal (0)')
plt.show()

# Calculate percentage
fraud_ratio = df_train['isFraud'].value_counts(normalize=True)[1] * 100
print(f"Fraud ratio: {fraud_ratio:.2f}% (Imbalanced Data)")

# View top 10 columns with the most missing data
missing = df_train.isnull().sum() / len(df_train)
missing = missing[missing > 0].sort_values(ascending=False).head(10)
print("\nTop 10 columns with the most missing data:")
print(missing)


# --- CELL 3.1: ADVANCED EDA - TRANSACTION AMOUNT ---
plt.figure(figsize=(12, 5))

# Plot transaction amount distribution (using log scale for visibility)
sns.histplot(x='TransactionAmt', data=df_train, hue='isFraud', kde=True, log_scale=True, common_norm=False)

plt.title('Transaction Amount Distribution (Log Scale)')
plt.xlabel('Transaction Amount (Log)')
plt.ylabel('Density')
plt.show()

print("Observation: If the Orange line (Fraud) peaks at low or high amounts, that is a sign!")


# --- CELL 3.2: ADVANCED EDA - TIME ---
# Create hour column (Assuming TransactionDT starts at 00:00)
df_train['hour'] = (df_train['TransactionDT'] // 3600) % 24

plt.figure(figsize=(14, 6))
# Plot overlapping histograms
sns.histplot(data=df_train, x='hour', hue='isFraud', common_norm=False, stat='density', kde=True)

plt.title('Transaction Behavior by Hour of Day')
plt.xlabel('Hour (0 - 23)')
plt.xlim(0, 23)
plt.show()

print("Observation: Check if the Orange line spikes during any specific time frame?")


# --- CELL 3.3: ADVANCED EDA - CATEGORICAL ---
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Fraud Rate by Product Type
sns.barplot(x='ProductCD', y='isFraud', data=df_train, ax=ax[0], palette='magma')
ax[0].set_title('Fraud Rate by Product (ProductCD)')
ax[0].set_ylabel('Fraud Rate (0.0 - 1.0)')

# Plot 2: Fraud Rate by Card Type
sns.barplot(x='card4', y='isFraud', data=df_train, ax=ax[1], palette='viridis')
ax[1].set_title('Fraud Rate by Card Type (Card4)')
ax[1].set_ylabel('Fraud Rate')

plt.show()

print("Observation: Any column that spikes indicates that category is extremely risky!")


# --- CELL 4: PREPROCESSING (AGGRESSIVE FIX) ---
from sklearn.preprocessing import LabelEncoder

print("Starting data processing (Deep Clean)...")

# 1. Remove useless columns
cols_to_drop = ['TransactionID', 'TransactionDT']
for col in cols_to_drop:
    if col in df_train.columns:
        df_train = df_train.drop(col, axis=1)
    if col in df_test.columns:
        df_test = df_test.drop(col, axis=1)

# 2. Iterate through ALL columns to process
# (Do not use pre-filtered lists to avoid missing any columns)
for col in df_train.columns:
    # Skip Label column (isFraud)
    if col == 'isFraud':
        continue

    # Check if the column is text (object)
    # Or if it is text in Test (sometimes Train is numeric but Test contains text)
    if df_train[col].dtype == 'object' or (col in df_test.columns and df_test[col].dtype == 'object'):
        print(f"Encoding text column: {col}")
        
        # Fill missing values with 'unknown' and convert all to string (str)
        df_train[col] = df_train[col].fillna('unknown').astype(str)
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna('unknown').astype(str)

        # Label Encoding (Convert Text -> Number)
        le = LabelEncoder()
        # Learn vocabulary from both Train and Test
        temp_data = list(df_train[col])
        if col in df_test.columns:
            temp_data += list(df_test[col])
        
        le.fit(temp_data)
        
        # Transform
        df_train[col] = le.transform(df_train[col])
        if col in df_test.columns:
            df_test[col] = le.transform(df_test[col])
            
    else:
        # If numeric column: Fill -999 for missing values
        df_train[col] = df_train[col].fillna(-999)
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna(-999)

print("Processing done! All data is now numeric.")


# --- CELL 5: MODEL PREPARATION ---
X = df_train.drop('isFraud', axis=1)
y = df_train['isFraud']

# IMPORTANT: Align Columns
# Ensure X_test has the exact same column order and count as X
X_test = df_test.reindex(columns=X.columns, fill_value=0)

# Final RAM cleanup before training
del df_train, df_test
gc.collect()

# Split Validation set (First 80% for train, last 20% for time-based validation)
split_idx = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")


# --- CELL 6: TRAINING ---
print("Training Random Forest (Please wait)...")

clf = RandomForestClassifier(
    n_estimators=100,      # Number of trees: 100
    max_depth=15,          # Depth: 15 (to prevent overfitting)
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,             # Use all Kaggle CPUs
    verbose=1
)

clf.fit(X_train, y_train)

print("Training finished!")


# --- CELL 7: EVALUATION & SUBMISSION (ADDED F1-SCORE) ---
from sklearn.metrics import f1_score, classification_report

# 1. Evaluate on Validation set
print("Evaluating...")

# Get Probability to calculate ROC-AUC
val_preds_prob = clf.predict_proba(X_val)[:, 1]

# Convert probability to binary labels 0/1 to calculate F1-Score
# Default threshold is 0.5 (Greater than 50% is considered fraud)
val_preds_label = (val_preds_prob > 0.5).astype(int)

# Calculate metrics
auc_score = roc_auc_score(y_val, val_preds_prob)
f1 = f1_score(y_val, val_preds_label)

print(f"=== ROC-AUC SCORE (Validation): {auc_score:.4f} ===")
print(f"=== F1 SCORE (Validation):      {f1:.4f} ===")

# (Optional) Print detailed Precision and Recall for better team visibility
print("\nDetailed Report (Classification Report):")
print(classification_report(y_val, val_preds_label, target_names=['Normal', 'Fraud']))

# 2. Predict on real Test set for Kaggle submission
print("\nCreating submission file...")
test_preds = clf.predict_proba(X_test)[:, 1]

# 3. Create submission.csv file
submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
submission['isFraud'] = test_preds
submission.to_csv('submission.csv', index=False)

print("File 'submission.csv' saved. You can submit this file!")


# --- CELL 8: FEATURE IMPORTANCE ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Get feature importance from the trained model (clf)
# X.columns contains the data column names
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, X.columns)), columns=['Value','Feature'])

# 2. Get Top 20 most important features
top_20 = feature_imp.sort_values(by="Value", ascending=False).head(20)

# 3. Plot the chart
plt.figure(figsize=(10, 8))
sns.barplot(x="Value", y="Feature", data=top_20, palette="viridis")
plt.title('Top 20 Most Important Features (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

print("Explanation: The feature at the very top is the most important.")
print("Example: TransactionAmt (Amount) or card1 (Card Type) usually appear in the top.")


# --- CELL 9: CONFUSION MATRIX (FIXED) ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Re-predict probabilities on Validation set
# (Recalculate here to avoid dependency on variable names from step 7)
val_probs = clf.predict_proba(X_val)[:, 1]

# 2. Convert to 0/1 labels with threshold 0.5
val_preds_label = (val_probs > 0.5).astype(int)

# 3. Create confusion matrix
cm = confusion_matrix(y_val, val_preds_label)

# 4. Plot figure
fig, ax = plt.subplots(figsize=(6, 6))
# display_labels: Labels displayed on the plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fraud'])

# values_format='d': Display integers (e.g., 100) instead of scientific notation (1e2)
disp.plot(ax=ax, cmap='Blues', values_format='d') 

plt.title('Confusion Matrix')
plt.grid(False) # Turn off grid for better aesthetics
plt.show()

# 5. Calculate Recall Score (Important)
# Recall = What % of actual fraudsters did we catch?
# Formula: True Positive / (True Positive + False Negative)
recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print(f"Recall (Ability to catch fraudsters): {recall:.2%}")

