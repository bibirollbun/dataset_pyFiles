import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


data_4_6_delay = pd.read_csv("delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv")
data_4_6_not_delay = pd.read_csv("not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv")
data_7_9_delay = pd.read_csv("delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv")
data_7_9_not_delay = pd.read_csv("not_delay_7_9_CONDITION_PRODUCT_SUPPLIER (1).csv")



data_4_6 = pd.concat([data_4_6_delay, data_4_6_not_delay], axis = 0, ignore_index= True)
data_7_9 = pd.concat([data_7_9_delay, data_7_9_not_delay], axis = 0, ignore_index= True)
print(data_4_6.info())
print(data_7_9.info())


common_columns = data_4_6.columns.intersection(data_7_9.columns)

common_columns


data_4_6 = data_4_6[common_columns]
data_7_9 = data_7_9[common_columns]


data = pd.concat([data_4_6, data_7_9], axis = 0, ignore_index= True)


data.info()


data.describe()


for col in data.columns:
    if data[col].isna().sum() > 0:
        print(col, data[col].isna().sum())
        data[col] = data[col].fillna('Unknow')


data['Ship Mode'].value_counts()


data['SUPPLIER_DIV'].value_counts()


for col in data.columns:
    print(col)
    print(data[col].unique())



def histplot_func(data, col, bins):
    sns.histplot(data = data, x = col, kde = True, bins = bins)
    plt.show()
def count_encoding(data, col):
    count_encoding = data[col].value_counts().to_dict()
    data[col] = data[col].map(count_encoding)
    #histplot_func(data, col, 100)

# Tính Information Value
def calculate_woe_iv(df, feature, target):
    # Nhóm dữ liệu theo bin, đếm Good và Bad
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped['Good'] = grouped['count'] - grouped['sum']  # Số mẫu Good
    grouped['Bad'] = grouped['sum']  # Số mẫu Bad
    
    # Tính tỷ lệ
    total_good = grouped['Good'].sum()
    total_bad = grouped['Bad'].sum()
    grouped['Pct_Good'] = grouped['Good'] / total_good
    grouped['Pct_Bad'] = grouped['Bad'] / total_bad
    
    # Tính WoE
    grouped['WoE'] = np.log(grouped['Pct_Good'] / grouped['Pct_Bad'].replace(0, 1e-6))
    
    # Tính IV
    grouped['IV'] = (grouped['Pct_Good'] - grouped['Pct_Bad']) * grouped['WoE']
    iv_total = grouped['IV'].sum()
    
    return grouped, iv_total


# Hàm tính Null Importances
def get_null_importances(X, y, cat_features,  features, n_splits=5, n_shuffles=10):
    # Khởi tạo K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Ma trận lưu importance thực tế
    real_importances = np.zeros(len(features))
    
    # Danh sách lưu null importances cho mỗi lần xáo trộn
    null_importances = np.zeros((len(features), n_shuffles * n_splits))
    
    # Huấn luyện trên 5 folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Khởi tạo CatBoostClassifier
        model = CatBoostClassifier(
            iterations=100,
            scale_pos_weight=40,  # Xử lý mất cân bằng 1:40
            random_seed=42,
            verbose=0
        )
        
        # Huấn luyện trên dữ liệu thực
        model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)
        real_importances += model.get_feature_importance() / n_splits
        
        # Tạo null importances
        for shuffle_iter in range(n_shuffles):
            X_shuffled = X_train.copy()
            for col in features:
                # Xáo trộn từng cột
                X_shuffled[col] = np.random.permutation(X_shuffled[col].values)
            
            # Huấn luyện trên dữ liệu xáo trộn
            model.fit(X_shuffled, y_train, eval_set=(X_val, y_val))
            null_importances[:, fold * n_shuffles + shuffle_iter] = model.get_feature_importance()
    
    return real_importances, null_importances


# Hàm chọn thuộc tính dựa trên Null Importances
def select_important_features(real_importances, null_importances, features, threshold_factor=2):
    null_mean = null_importances.mean(axis=1)
    null_std = null_importances.std(axis=1)
    threshold = null_mean + threshold_factor * null_std
    
    selected_features = [
        (features[i], real_importances[i]) 
        for i in range(len(features)) 
        if real_importances[i] > threshold[i]
    ]
    return selected_features





#data.info()

# Những thuộc tính categorical 
# VSD, Order Date, SO_TIME -> Count_encode 
data = data.drop(columns= 'SUBSIDIARY_CD')
count_encoding_col = ['Order date', 'GLOBAL_NO', 'CLASSIFY_CD', 'CUST_CD', 'BRAND_CD',
                      'INNER_CD', 'SUPPLIER_CD', 'PRODUCT_CD', 'SHIP DECISION NO', 'SOUF_RCV_NO', 'QTUF_RCV_NO',
                      'VSD', 'REASON_CD', 'SO_TIME']

categorical_col = ['OTHER AREA SHIP DIV', 'DELI_DIV', 'Ship Mode', 'Stock class', 'Consider count hodiday Saturday',
                   'PRODUCT ATTRIBUTION', 'SPECIAL DIV', 'LOGICAL PLANT', 'DIRECT SHIP FLG', 'SPECIAL_DIV', 
                   'PACKING RANK', 'SUPPLIER_DIV']

numerical_col = ['Sales order line number', 'SO QTY', 'ALLOCATION QTY', 'SUPPLIER INV AMOUNT', 'PURCHASE AMOUNT',
                 'PACK QTY', 'WEIGHT PER PIECE', 'SO_DAY_OF_MONTH', 'SO_DAY_OF_WEEK']







for col in numerical_col:
    print(col)
    print(data[col].unique())




data_iv = data.copy()
for col in categorical_col:
    count_encoding(data_iv, col)
for col in count_encoding_col:
    count_encoding(data_iv, col)


data_temp = pd.DataFrame(columns= ['Col_Name', 'Value'])
for col in data.columns:
    result, iv = calculate_woe_iv(data_iv, col, 'label')
    data_temp.loc[len(data_temp)] = [col, round(iv, 2)]

data_temp = data_temp.sort_values(by='Value', ascending=False)

# Hiển thị bảng kết quả
print("\nBảng IV sắp xếp từ lớn đến nhỏ:")
data_temp



corr_matrix = data_iv[categorical_col].corr()
plt.figure(figsize=(10, 8))  # Kích thước biểu đồ
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap Categorical')
plt.show()


corr_matrix = data_iv[count_encoding_col].corr()
plt.figure(figsize=(10, 8))  # Kích thước biểu đồ
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap Categorical')
plt.show()


corr_matrix = data_iv[numerical_col].corr()
plt.figure(figsize=(10, 8))  # Kích thước biểu đồ
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap Numerical')
plt.show()



X = data_iv.drop('label', axis=1)
y = data_iv['label']
features = X.columns.tolist()

cat_features = categorical_col + count_encoding_col

# Tính Null Importances
real_importances, null_importances = get_null_importances(X, y, cat_features, features, n_splits=5, n_shuffles=10)

# Tạo bảng kết quả
importance_df = pd.DataFrame({
    'Feature': features,
    'Real_Importance': real_importances,
})
importance_df = importance_df.sort_values(by='Real_Importance', ascending=False)

# Hiển thị kết quả
print("\nBảng Feature Importance (sắp xếp từ lớn đến nhỏ):")
print(importance_df.round(4).to_string(index=False))


importance_df


drop_col = [ 'BRAND_CD', 'ALLOCATION QTY', 'SPECIAL DIV', , 'Sales order line number','LOGICAL PLANT', 
            'PURCHASE AMOUNT', 'DIRECT SHIP FLG', 'SHIP DECISION NO', 'PACK QTY','GLOBAL_NO', 
            'SUPPLIER_DIV', 'SPECIAL_DIV', 'SO_DAY_OF_MONTH', 'SO_DAY_OF_WEEK', 'SO_TIME', 'QTUF_RCV_NO', 
            'SOUF_RCV_NO',, 'Stock class', 'Order date', 'VSD', 'SO_TIME', 'REASON_CD']
data_t = data.copy()

data_t = data_t.drop(columns= drop_col)


cat_features


# Thay thế dấu cách và xử lý NaN
data_t = data_t.replace(' ', 'Unknow')

# Chuyển các cột phân loại thành chuỗi
for col in cat_features:
    if col not in drop_col:
        data_t[col] = data_t[col].astype(str)


from sklearn.utils import resample

# Hàm thực hiện undersampling
def undersample_data(X, y, majority_class_size, random_state=42):
    X_majority = X[y == 0]
    y_majority = y[y == 0]
    X_minority = X[y == 1]
    y_minority = y[y == 1]
    
    X_majority_downsampled, y_majority_downsampled = resample(
        X_majority, y_majority,
        n_samples=majority_class_size,
        random_state=random_state,
        replace=False
    )
    
    X_balanced = pd.concat([X_majority_downsampled, X_minority], axis=0)
    y_balanced = pd.concat([y_majority_downsampled, y_minority], axis=0)
    
    return X_balanced, y_balanced

# Chuẩn bị dữ liệu
X = data_t.drop('label', axis=1)
y = data_t['label']

n_minority = sum(y == 1) 
n_majority = n_minority * 20 
X_balanced, y_balanced = undersample_data(X, y, n_majority, random_state=42)

# Kiểm tra kết quả
print(f"Số mẫu sau undersampling: {len(y_balanced)}")
print(f"Tỷ lệ lớp (not_delay:delay): {sum(y_balanced == 0)}:{sum(y_balanced == 1)}")



for col in categorical_col:
    if col not in drop_col:
        data_t[col] = data_t[col].astype('category')
for col in count_encoding_col:
    if col not in drop_col:
        data_t[col] = data_t[col].astype('category')
    


data_t.info()


from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

X = X_balanced
y = y_balanced

# 1. Chia tập train (60%) và tạm (40%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# 2. Chia tập tạm thành dev (20%) và test (20%)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Xác định các cột phân loại
cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

cat_features



train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
dev_pool = Pool(data=X_dev, label=y_dev, cat_features=cat_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

# Huấn luyện mô hình
model = CatBoostClassifier(verbose=100, learning_rate= 0.1, n_estimators= 1000, random_seed= 42, early_stopping_rounds= 100)
model.fit(train_pool, eval_set=dev_pool)


'''
from sklearn.model_selection import RandomizedSearchCV

model = CatBoostClassifier(
    verbose=100,
    iterations=1000,
    learning_rate=0.1,
    random_seed=42,
    cat_features=cat_features
    early_stopping_rounds=100
)

param_dist = {
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'bagging_temperature': [0, 0.3, 0.6, 1],
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,  # Số mẫu thử
    scoring='f1_macro',  
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print("Best params:", random_search.best_params_)
'''


from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


X_balanced


data_pilot = pd.read_csv('test.csv').drop(columns = 'ID')
data_pilot


print(data_pilot.columns[13])


X_balanced['WEIGHT PER PIECE'].dtype


cat_features


for col in data_pilot.columns:
    if col not in cat_features:
        print(col, data_pilot[col].unique())


data_pilot['WEIGHT PER PIECE'] = data_pilot['WEIGHT PER PIECE'].replace('Unknow', 0)
data_pilot['WEIGHT PER PIECE'] = data_pilot['WEIGHT PER PIECE'].astype(float)


y_pred = model.predict(data_pilot)


submit = pd.DataFrame(columns = ['ID', 'label'])
submit['label'] = y_pred
submit['ID'] = range(1, len(y_pred) + 1)


submit.to_csv('submission.csv', index=False)


submit

