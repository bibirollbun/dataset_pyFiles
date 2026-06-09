import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder


import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

# Gộp dữ liệu
train = train_trans.merge(train_id, on='TransactionID', how='left')

# Chia thành train và test (80% train, 20% test, có thể điều chỉnh test_size)
train_set, test_set = train_test_split(train, test_size=0.2, random_state=42, stratify=train['isFraud'])

# Lưu ra file CSV
train_set.to_csv('/kaggle/working/train_split.csv', index=False)
test_set.to_csv('/kaggle/working/test_split.csv', index=False)



print(train.shape[0])


from sklearn.model_selection import train_test_split
import pandas as pd

# Giả sử bạn đã đọc file train.csv từ trước
# train = pd.read_csv("train.csv")

# 1. Tách nhãn
y = train['isFraud']
features = train.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore').columns

# 2. Chia tập dữ liệu thành 80% train_test và 20% val
X_train, X_test, y_train, y_test = train_test_split(
    train[features], y, test_size=0.2, stratify=y, random_state=42
)

train = pd.concat()
# 3. Từ 80% còn lại, tiếp tục chia thành 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)



# List created from Data page on competition 
cat_cols = ['ProductCD',
                'card1','card2','card3','card4','card5','card6',
                'addr1','addr2',
                'P_emaildomain','R_emaildomain',
                'M1','M2','M3','M4','M5','M6','M7','M8','M9',
                'DeviceType','DeviceInfo']

# use list comprehension for id columns 
id_feats = [f'id_{x}' for x in range(12,39)]

# combine lists for final categorical feature list
cat_cols = cat_cols + id_feats

print(len(cat_cols))
print(cat_cols)


num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [col for col in train.columns if col not in cat_cols + ['TransactionID','isFraud'] ]

print(len(num_cols))
print(num_cols)


tem = train['isFraud']
train = train.drop('isFraud', axis=1)


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Hàm chia an toàn
def safe_divide(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)

# 1. Tạo đặc trưng tỷ lệ theo nhóm
def create_ratio_features(train):
    ratio_features = {}
    for col in ['TransactionAmt', 'TransactionDT', 'D15', 'D1', 'C1', 'C13', 'C14', 'C11', 'C10']:
        if col not in train.columns:
            continue
        base_cols = {
            'TransactionAmt': ['card1', 'card2', 'card4', 'card6'],
            'TransactionDT': ['card1', 'card2', 'card4', 'card6'],
            'D15': ['card1', 'card2'],
            'D1': ['card1', 'card2'],
            'C1': ['card1', 'card2', 'card4', 'addr1', 'card6'],
            'C13': ['card1', 'card2', 'card4', 'addr1', 'card6'],
            'C14': ['card2'],
            'C11': ['card1', 'card2', 'card4', 'addr1'],
            'C10': ['card6'],
        }
        for base in base_cols.get(col, []):
            if base in train.columns:
                mean_name = f'{col}_to_mean_{base}'
                std_name  = f'{col}_to_std_{base}'
                group = train.groupby(base)[col]
                ratio_features[mean_name] = safe_divide(train[col], group.transform('mean'))
                ratio_features[std_name]  = safe_divide(train[col], group.transform('std'))
    return ratio_features

# 2. Tách domain email thành nhiều phần
def split_email_domain(train):
    email_features = {}
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in train.columns:
            splits = train[col].str.split('.', expand=True, n=2)
            for i in range(splits.shape[1]):
                email_features[f'{col}_{i+1}'] = splits[i]
    return email_features

# 3. Tạo các tổ hợp đặc trưng logic
def create_uid_features(train):
    uid_features = pd.DataFrame(index=train.index)
    uid_features['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
    uid_features['uid2'] = uid_features['uid'] + '_' + train['addr1'].astype(str)
    uid_features['uid5'] = train['card2'].astype(str) + '_' + train['addr1'].astype(str)
    uid_features['addr1_C1'] = train['addr1'].astype(str) + '_' + train['C1'].astype(str)
    uid_features['addr1_C13'] = train['addr1'].astype(str) + '_' + train['C13'].astype(str)
    uid_features['addr1_C14'] = train['addr1'].astype(str) + '_' + train['C14'].astype(str)
    uid_features['card1_P_emaildomain'] = train['card1'].astype(str) + '_' + train['P_emaildomain'].astype(str)
    uid_features['P_R_emaildomain'] = train['P_emaildomain'].astype(str) + '_' + train['R_emaildomain'].astype(str)
    if 'DeviceType' in train.columns and 'DeviceInfo' in train.columns:
        uid_features['DeviceType_DeviceInfo'] = train['DeviceType'].astype(str) + '_' + train['DeviceInfo'].astype(str)
    return uid_features

# 4. Thống kê theo nhóm
def create_group_stats(train, group_cols, stat_cols):
    for gcol in group_cols:
        if gcol not in train.columns:
            continue
        for scol in stat_cols:
            if scol in train.columns:
                train[scol] = pd.to_numeric(train[scol], errors='coerce')
                train[f'{gcol}_{scol}_mean'] = train.groupby(gcol)[scol].transform('mean')
                train[f'{gcol}_{scol}_std'] = train.groupby(gcol)[scol].transform('std')
    return train

def label_encode(train, comb_cols):
    le = LabelEncoder()
    for col in comb_cols:
        if col in train.columns:
            train[col] = train[col].astype(str).fillna('nan')
            train[col] = le.fit_transform(train[col])
    return train

def process_features(train):
    # B1. Tạo đặc trưng tỷ lệ
    ratio_feats = create_ratio_features(train)
    train = pd.concat([train, pd.DataFrame(ratio_feats)], axis=1)

    # B2. Tách domain email
    email_feats = split_email_domain(train)
    train = pd.concat([train, pd.DataFrame(email_feats)], axis=1)

    # B3. Tổ hợp logic
    uid_feats = create_uid_features(train)
    train = pd.concat([train, uid_feats], axis=1)

    # B4. Thống kê theo nhóm
    group_cols1 = ['uid', 'uid2', 'uid5']
    stat_cols1 = ['TransactionAmt', 'D15', 'D1', 'TransactionDT', 'C1', 'C13', 'C14', 'C11', 'C10']
    train = create_group_stats(train, group_cols1, stat_cols1)

    group_cols2 = ['addr1_C1']
    stat_cols2 = ['TransactionAmt', 'D15', 'D1', 'TransactionDT', 'C13', 'C14', 'C11', 'C10']
    train = create_group_stats(train, group_cols2, stat_cols2)

    # B5. Label Encoding các đặc trưng kết hợp
    comb_cols = [
        'uid', 'uid2', 'card1_P_emaildomain', 'P_R_emaildomain', 'DeviceType_DeviceInfo',
        'addr1_C1', 'addr1_C13', 'addr1_C14'
    ]
    train = label_encode(train, comb_cols)

    return train

train = process_features(train)



train['isFraud'] = tem


print(train.shape[1])


if 'isFraud' in num_cols:
    num_cols.remove('isFraud')

# 1. Xác định các cột số có >25% giá trị bị thiếu
num_cols_to_drop = [col for col in num_cols if train[col].isnull().mean() > 0.4]

# 2. Xóa những cột số bị thiếu quá nhiều
train.drop(columns=num_cols_to_drop, inplace=True)

# 3. Với các cột số còn lại, điền missing bằng median
num_cols_to_fill = [col for col in num_cols if col not in num_cols_to_drop]
for col in num_cols_to_fill:
    train[col] = train[col].fillna(train[col].median())


cat_cols = train.select_dtypes(include='object').columns.tolist()
cat_cols += [col for col in train.columns if '_emaildomain_' in col]

for col in cat_cols:
    if col in train.columns:
        train[col] = train[col].astype(str).fillna('unknown')
        le = LabelEncoder()
        try:
            train[col] = le.fit_transform(train[col])
        except:
            print(f"Bỏ qua cột {col} do lỗi mã hóa")


#li = ['V284', 'V100', 'V111', 'V34', 'V59', 'V302', 'V101', 'V60', 'V95', 'V26', 'V93', 'id_25', 'V79', 'V94', 'V97', 'id_28', 'V134', 'V319', 'V105', 'V40', 'V58', 'V74', 'V25', 'V301', 'V57', 'DeviceType', 'V49', 'id_16', 'V42', 'V286', 'id_24', 'V73', 'V72', 'V125', 'V123', 'V66', 'V108', 'V292', 'V47', 'V33', 'V39', 'V137', 'V135', 'V43', 'V132', 'V290', 'id_23', 'id_21', 'id_26', 'id_34', 'id_37', 'V52', 'V29', 'V115', 'V71', 'V85', 'V289', 'V287', 'V109', 'V295', 'V35', 'V24', 'V23', 'V303', 'V30', 'M7', 'C7', 'V321', 'V124', 'id_29', 'V298', 'V37', 'V316', 'M8', 'V96', 'R_emaildomain_3', 'V103', 'id_15', 'card4', 'V293', 'id_12', 'V291', 'V36', 'V19', 'P_emaildomain_3', 'V126', 'V75', 'V13', 'C3', 'V77', 'V129', 'V102', 'V99', 'V61', 'V69', 'V81', 'V56', 'V306', 'V127', 'V128']
#for col in train.columns:
#    if col in li:
        #print('yes')
#        train = train.drop(col, axis=1)



from sklearn.model_selection import train_test_split
import pandas as pd

# Giả sử bạn đã đọc file train.csv từ trước
# train = pd.read_csv("train.csv")

# 1. Tách nhãn
y = train['isFraud']
features = train.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore').columns

# 2. Chia tập dữ liệu thành 80% train_test và 20% val
X_temp, X_val, y_temp, y_val = train_test_split(
    train[features], y, test_size=0.2, stratify=y, random_state=42
)

# 3. Từ 80% còn lại, tiếp tục chia thành 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# ✅ Kết quả:
# - X_train: 60% của toàn bộ data → dùng để huấn luyện mô hình
# - X_test: 20% của toàn bộ data → dùng để đánh giá trong quá trình phát triển
# - X_val: 20% của toàn bộ data → chỉ dùng để đánh giá cuối cùng (không dùng trong training hoặc tuning)



# Tạo dataset cho LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


# Tham số LightGBM 
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,      
    'num_leaves': 256,           
    'max_depth': -1,
    'min_child_samples': 50,     
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.3,            
    'reg_lambda': 0.3,
    'n_estimators': 4000        
}


# Train
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    valid_names=["test"],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
)



import matplotlib.pyplot as plt
import lightgbm as lgb

# Vẽ biểu đồ top 50 đặc trưng quan trọng nhất theo 'gain'
fig, ax = plt.subplots(figsize=(12, 10))
lgb.plot_importance(
    model,
    max_num_features=50,
    importance_type='gain',
    ax=ax,
    height=0.45,
    color='mediumseagreen',
    title=None,
    xlabel='Feature Importance (Gain)'
)

# Thêm tiêu đề, lưới và căn chỉnh
ax.set_title(" Top 50 Most Important Features (by Gain)")
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd

# Lấy importance theo 'gain'
importance_df = pd.DataFrame({
    'feature': model.feature_name(),
    'gain': model.feature_importance(importance_type='gain')
})

# Sắp xếp theo gain giảm dần
importance_df = importance_df.sort_values(by='gain', ascending=False)

# Lấy từ vị trí 51 đến 100
next_50 = importance_df.iloc[50:100]

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(next_50['feature'], next_50['gain'], color='cornflowerblue')
ax.set_xlabel('Feature Importance (Gain)')
ax.set_title('Features Ranked 51–100 by Gain')
ax.invert_yaxis()  # Để feature quan trọng nhất ở trên
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd

# Lấy importance theo 'gain'
importance_df = pd.DataFrame({
    'feature': model.feature_name(),
    'gain': model.feature_importance(importance_type='gain')
})

# Sắp xếp theo gain giảm dần
importance_df = importance_df.sort_values(by='gain', ascending=False)

# Lấy từ vị trí 100 đến 150
next_50 = importance_df.iloc[100:150]

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(next_50['feature'], next_50['gain'], color='cornflowerblue')
ax.set_xlabel('Feature Importance (Gain)')
ax.set_title('Features Ranked 51–100 by Gain')
ax.invert_yaxis()  # Để feature quan trọng nhất ở trên
plt.tight_layout()
plt.show()



import pandas as pd

# Lấy thông tin importance theo 'gain'
importance_df = pd.DataFrame({
    'feature': model.feature_name(),
    'gain': model.feature_importance(importance_type='gain')
})

# Sắp xếp theo importance giảm dần
importance_df.sort_values(by='gain', ascending=False, inplace=True)
importance_df.reset_index(drop=True, inplace=True)

# Thêm cột xếp hạng
importance_df['rank'] = importance_df.index + 1



import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

# Lấy thông tin đặc trưng và độ quan trọng
feature_importance = pd.DataFrame({
    'feature': model.feature_name(),
    'importance_gain': model.feature_importance(importance_type='gain')
})

# Lấy 50 đặc trưng kém quan trọng nhất
least_important = feature_importance.sort_values(by='importance_gain', ascending=True).head(100)

# Vẽ biểu đồ.
fig, ax = plt.subplots(figsize=(12, 10))
least_important.plot.barh(
    x='feature',
    y='importance_gain',
    ax=ax,
    color='salmon',
    edgecolor='black'
)

ax.set_title("Bottom 100 Least Important Features (by Gain)")
ax.set_xlabel("Feature Importance (Gain)")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.show()



#print(least_important['feature'])
temp = least_important['feature'].tolist()
print(temp)


y_proba_val = model.predict(X_val)


print("Validation AUC Score:", roc_auc_score(y_val, y_proba_val))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Gán nhãn: nếu xác suất > 0.4 thì là fraud (1)
y_pred_val = (y_proba_val > 0.4).astype(int)
print("Evaluation Metrics (Threshold = 0.4):")
print("Accuracy :", round(accuracy_score(y_val, y_pred_val), 4))
print("Precision:", round(precision_score(y_val, y_pred_val), 4))
print("Recall   :", round(recall_score(y_val, y_pred_val), 4))
print("F1-score :", round(f1_score(y_val, y_pred_val), 4))
print("Validation AUC Score:",round(roc_auc_score(y_val, y_proba_val),4))


y_pred_binary = (y_pred_val > 0.4).astype(int)
cm = confusion_matrix(y_val, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



import shap
# Lọc các giao dịch gian lận trong tập validation
X_val_pos = X_val[y_val == 0].copy()

# Lấy mẫu nếu quá lớn
X_val_pos_sample = X_val_pos.sample(n=20, random_state=42)

# Tạo SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val_pos_sample)

# Hiển thị SHAP summary plot cho positive cases
shap.summary_plot(shap_values, X_val_pos_sample, plot_type='bar')



import shap
# Lọc các giao dịch gian lận trong tập validation
X_val_pos = X_val[y_val == 1].copy()

# Lấy mẫu nếu quá lớn
X_val_pos_sample = X_val_pos.sample(n=20, random_state=42)

# Tạo SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val_pos_sample)

# Hiển thị SHAP summary plot cho positive cases
shap.summary_plot(shap_values, X_val_pos_sample, plot_type='bar')



shap.summary_plot(shap_values, X_val_pos_sample, max_display=100)  # plot_type='dot' mặc định

