import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score


# delay
data_delay_4_6 = pd.read_csv('/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
data_delay_7_9 = pd.read_csv('/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')

# not delay
data_not_delay_4_6 = pd.read_csv('/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
data_not_delay_7_9 = pd.read_csv('/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')

# test data
data_10 = pd.read_csv('/PILOT_10.csv')


print('Shape of delay 4_6: ', data_delay_4_6.shape)
print('Shape of delay 7_9: ', data_delay_7_9.shape)
print('Shape of not delay 4_6: ', data_not_delay_4_6.shape)
print('Shape of not delay 7_9: ', data_not_delay_7_9.shape)


# merge data 4_6
full_data_4_6 = pd.concat([data_delay_4_6, data_not_delay_4_6], ignore_index=True)
full_data_4_6 = full_data_4_6.sample(frac=1).drop_duplicates().reset_index()

# merge data 7_9
full_data_7_9 = pd.concat([data_delay_7_9, data_not_delay_7_9], ignore_index=True)
full_data_7_9 = full_data_7_9.sample(frac=1).drop_duplicates().reset_index()


useful_features = ['Order date', 'SO QTY', 'SO_TIME', 'SO_DAY_OF_MONTH', 'SO_DAY_OF_WEEK',
                'WEIGHT PER PIECE',
                'SUPPLIER_CD','SUPPLIER INV AMOUNT','SUPPLIER_DIV','VSD','Ship Mode',
                'DELI_DIV', 'Stock class','PACKING RANK','label']

useful_features1 = ['Order date', 'SO QTY', 'SO_TIME', 'SO_DAY_OF_MONTH', 'SO_DAY_OF_WEEK',
                'WEIGHT PER PIECE',
                'SUPPLIER_CD','SUPPLIER INV AMOUNT','SUPPLIER_DIV','VSD','Ship Mode',
                'DELI_DIV', 'Stock class','PACKING RANK']


# Lọc các thuộc tính useful cho tập data
full_data_4_6 = full_data_4_6[useful_features]
full_data_7_9 = full_data_7_9[useful_features]

full_data = pd.concat([full_data_4_6, full_data_7_9], ignore_index=True)
data_10 = data_10[useful_features1]


full_data = full_data.replace(r'^\s*$', np.nan, regex=True)
data_10 = data_10.replace(r'^\s*$', np.nan, regex=True)

# train
full_data = full_data.dropna(subset=['SUPPLIER_DIV','Ship Mode'])

# test
data_10['Ship Mode'] = data_10['Ship Mode'].fillna(data_10['Ship Mode'].mode()[0])
data_10['SUPPLIER_DIV'] = data_10['SUPPLIER_DIV'].fillna(data_10['SUPPLIER_DIV'].mode()[0])


data_10.info()


def constrain_numeric_type(data):
    convert_dict = {
        'SO QTY' : int,
        'SUPPLIER INV AMOUNT' : int,
        #'label' : 'category',
        'WEIGHT PER PIECE'  : int,
        'SUPPLIER_DIV'  : int,
        'SO_DAY_OF_MONTH' : int,
        'SO_DAY_OF_WEEK' : int,
        'SO_TIME' : int
      }
    data_constrain_value = data.astype(convert_dict)
    data_constrain_value.info()
    return data_constrain_value

full_data = constrain_numeric_type(full_data)
full_data['label'] = full_data['label'].astype(int)

data_10 = constrain_numeric_type(data_10)


def constrain_type(data):
    convert_dict = {
        'Order date': 'datetime64[ns]',
        'SUPPLIER_CD': 'category',
        'Stock class':'category',
        'SO QTY' : 'int32',
        'SUPPLIER INV AMOUNT' : 'float32',
        'PACKING RANK' : 'category',
        'VSD' : 'datetime64[ns]',
        'DELI_DIV' : 'category',
        #'label' : 'category',
        'Ship Mode' : 'category',
        'WEIGHT PER PIECE'  : 'float32',
        'SUPPLIER_DIV'  : 'category',
        'SO_DAY_OF_MONTH' : 'int32',
        'SO_DAY_OF_WEEK' : 'int32',
        'SO_TIME' : 'int32',
      }
    data_constrain_value = data.astype(convert_dict)
    data_constrain_value.info()
    return data_constrain_value

full_data = constrain_type(full_data)
full_data['label'] = full_data['label'].astype('category')

data_10 = constrain_type(data_10)


encoder = TargetEncoder(cols='SUPPLIER_CD')
full_data['SUPPLIER_CD'] = encoder.fit_transform(full_data['SUPPLIER_CD'], full_data['label'])
data_10['SUPPLIER_CD'] = encoder.transform(data_10['SUPPLIER_CD'])


# chỉnh sửa, tạo đặc trưng cho data
def feature_engineer(data):

    feature_date = ['Order date', 'VSD']

    # Tạo đặc trưng khoảng cách ngày nhận
    data['day_range'] = (pd.to_datetime(data['VSD']) - pd.to_datetime(data['Order date'])).dt.days

    # Tạo đặc trưng ngày, tháng, năm
    data['WAITING_DAY'] = pd.to_datetime(data['day_range']).dt.day
    data['SO_MONTH'] = pd.to_datetime(data['Order date']).dt.month

    # Loại bỏ những cột không còn cần thiết
    data = data.drop(feature_date, axis=1)

    return data


full_data = feature_engineer(full_data)
data_10 = feature_engineer(data_10)


X = full_data.drop('label', axis=1)
y = full_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Shape of X_train (full_data): ', X_train.shape)
print('Shape of X_test (full_data): ', X_test.shape)

print('Shape of data_10: ', data_10.shape)


model_lgb = LGBMClassifier(n_estimators=1220, learning_rate=0.15, num_leaves=200, min_child_samples=90,
                           objective='binary', subsample=0.9, colsample_bytree=0.85, reg_lambda=0.02,
                           metric='binary_logloss', random_state=42, is_unbalance=True, max_depth=10
                           )

# model_rf = RandomForestClassifier()

# model_cb = CatBoostClassifier(cat_features=categorical_features)


model_lgb.fit(X_train, y_train)
lgb_pred = model_lgb.predict(X_test)

# model_rf.fit(X_train, y_train)
# rf_pred = model_rf.predict(X_test)

# model_cb.fit(X_train, y_train)
# cb_pred = model_cb.predict(X_test)


# Hàm tính toán và in ra kết quả
def result(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return accuracy, f1

# In kết quả từng mô hình
# print("Logistic Regression:")
# lr_result = result(lr_pred, y_test)
# print('*' * 100)

print("LightGBM:")
lgb_result = result(lgb_pred, y_test)
print('*' * 100)

# print("Random Forest:")
# rf_result = result(rf_pred, y_test)
# print('*' * 100)

# # print("Decision Tree:")
# # dt_result = result(dt_pred, y_test)
# # print('*' * 100)

# # print("Naive Bayes:")
# # nb_result = result(nb_pred, y_test)
# # print('*' * 100)

# print("CatBoost:")
# cb_result = result(cb_pred, y_test)
# print('*' * 100)

# # Vẽ biểu đồ trực quan accuracy và f1 score của từng mô hình
# models = ['LightGBM', 'RandomForest', 'CatBoost']
# accuracy_scores = [lgb_result[0], rf_result[0], cb_result[0]]
# f1_scores = [lgb_result[1], rf_result[1], cb_result[1]]

# ind = np.arange(len(models))
# width = 0.35

# fig, ax = plt.subplots(figsize=(8, 6))
# rects1 = ax.bar(ind - width/2, accuracy_scores, width, label='Accuracy')
# rects2 = ax.bar(ind + width/2, f1_scores, width, label='F1 Score')


# ax.set_ylabel('Scores')
# ax.set_title('Accuracy và F1 Score của các mô hình')
# ax.set_xticks(ind)
# ax.set_xticklabels(models)
# ax.set_ylim(0, 1.2)
# ax.legend()
# ax.bar_label(rects1, fmt='%.2f', padding=3)
# ax.bar_label(rects2, fmt='%.2f', padding=3)

# plt.tight_layout()
# plt.show()


best_model_lgb = LGBMClassifier(n_estimators=1220, learning_rate=0.15, num_leaves=200, min_child_samples=90,
                           objective='binary', subsample=0.9, colsample_bytree=0.85, reg_lambda=0.02,
                           metric='binary_logloss', random_state=42, is_unbalance=True, max_depth=10
                           )

# Huấn luyện lại mô hình Random Forest với bộ siêu tham số tốt nhất trên toàn bộ tập dữ liệu X và y
best_model_lgb.fit(X, y)


# Dự đoán trên data_10 bằng mô hình đã huấn luyện lại
result = best_model_lgb.predict(df_test_encoded)
len(result)


# xuất kết quả ra file result
result = pd.DataFrame({
    'ID': range(1, len(result) + 1),
    'label': result
})

result.to_csv('lgb.csv', index=False)

