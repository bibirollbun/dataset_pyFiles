!pip uninstall -y imbalanced-learn
!pip install imbalanced-learn==0.10.1


import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder


test = pd.read_csv('/kaggle/input/xg-data/dataset/test_split.csv')
train = pd.read_csv('/kaggle/input/xg-data/dataset/train_split.csv')


# List created from Data page on competition 
cat_cols = ['ProductCD',
                'card1','card2','card3','card5','card6',
                'addr1','addr2',
                'P_emaildomain','R_emaildomain',
                'M1','M2','M3','M4','M6','M7','M8','M9',
                'DeviceType','DeviceInfo']

# use list comprehension for id columns 
id_feats = [f'id_{x}' for x in range(12,39)]
#id_feats = [f'id_{x}' for x in range(12,27)]  + [f'id_{x}' for x in range(28,39)]
#for c in ['id_14','id_21','id_30','id_32','id_34', 'id_33']:
#    id_feats.remove(c)
# combine lists for final categorical feature list
cat_cols = cat_cols + id_feats

cat_cols = train.select_dtypes(include='object').columns.tolist()
cat_cols += [col for col in train.columns if '_emaildomain_' in col]

print(len(cat_cols))
print(cat_cols)


def label_encode(train, cat_cols):
    for col in cat_cols:
        if col in train.columns:
            train[col] = train[col].astype(str).fillna('unknown')
            le = LabelEncoder()
            try:
                train[col] = le.fit_transform(train[col])
            except:
                print(f"Bỏ qua cột {col} do lỗi mã hóa")
    return train


train = label_encode(train, cat_cols)
test = label_encode(test, cat_cols)


num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [col for col in train.columns if col not in cat_cols + ['TransactionID','isFraud']]

print(len(num_cols))
print(num_cols)


features = num_cols + cat_cols
new_feats = [col for col in train.columns if any(key in col for key in ['to_mean_', 'to_std_', 'emaildomain_'])]
features += new_feats

temp = train['isFraud']
for col in train.columns:
    if col not in features:
        train = train.drop(col, axis=1)


num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [col for col in train.columns if col not in cat_cols + ['TransactionID','isFraud']]

if 'isFraud' in num_cols:
    num_cols.remove('isFraud')

# 1. Xác định các cột số có >25% giá trị bị thiếu
num_cols_to_drop = [col for col in num_cols if train[col].isnull().mean() > 0.4]



train['isFraud'] = temp


from sklearn.model_selection import train_test_split

y = train['isFraud']
features = train.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore').columns
X_train, X_val, y_train, y_val= train_test_split(
   train[features], y, test_size=0.2, stratify=y, random_state=42
)


from sklearn.model_selection import train_test_split

# Chia dữ liệu ra val trước để giữ nguyên raw data
X_temp_raw, X_val_raw, y_temp_raw, y_val_raw = train_test_split(
    train, train['isFraud'], test_size=0.2, random_state=42, stratify=train['isFraud']
)

# Chỉ tiền xử lý cho tập train+test
train = X_temp_raw.copy()  # Dữ liệu cần tiền xử lý




# Bỏ các feature không dùng
for col in test.columns:
    if col not in train.columns:
        test = test.drop(col, axis=1)

# Đảm bảo X_test dùng đúng features như train
X_test = test.drop('isFraud', axis=1, errors='ignore')
y_test = test['isFraud']



fraud_ratio = y_train.value_counts(normalize=True)
print("Tỷ lệ nhãn trong tập huấn luyện:")
print(fraud_ratio*100)



import xgboost as xgb
from sklearn.metrics import roc_auc_score


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.03,
    'n_estimators': 3999,  
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'tree_method': 'gpu_hist',       
    'use_label_encoder': False,
    'verbosity': 1
}

model = xgb.XGBClassifier(**params)

# Huấn luyện mô hình với tập đã resample
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],        
    early_stopping_rounds=100,
    verbose=200
)




importance_df = pd.DataFrame({
    'feature': model.get_booster().feature_names,
    'importance': model.feature_importances_
})

# Sắp xếp theo mức độ quan trọng giảm dần
top_50 = importance_df.sort_values(by='importance', ascending=False).head(50)

# Vẽ biểu đồ
plt.figure(figsize=(12, 8))
plt.barh(top_50['feature'][::-1], top_50['importance'][::-1], color='steelblue')
plt.xlabel("Feature Importance (Gain)", fontsize=12)
plt.title("Top 50 Feature Importances - XGBoost", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



bottom_50 = importance_df.sort_values(by='importance', ascending=True).head(50)

plt.figure(figsize=(12, 8))
plt.barh(bottom_50['feature'], bottom_50['importance'], color='steelblue')
plt.xlabel("Feature Importance (Gain)", fontsize=12)
plt.title("Bottom 50 Feature Importances - XGBoost", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


y_proba_test = model.predict_proba(X_test)[:, 1]


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Dự đoán xác suất trên tập validation
y_pred_test = model.predict_proba(X_test)[:, 1]

y_pred_test_binary = (y_pred_test >= 0.4).astype(int)

# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred_test_binary)
precision = precision_score(y_test, y_pred_test_binary)
recall = recall_score(y_test, y_pred_test_binary)
f1 = f1_score(y_test, y_pred_test_binary)
auc = roc_auc_score(y_test, y_pred_test)

# In ra các chỉ số đánh giá
print("Validation Accuracy:", accuracy)
print("Validation Precision:", precision)
print("Validation Recall:", recall)
print("Validation F1 Score:", f1)
print("Validation AUC Score:", auc)


y_pred_binary = (y_pred_test > 0.4).astype(int)
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Tính các giá trị FPR, TPR
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
roc_auc = auc(fpr, tpr)

# Vẽ ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


