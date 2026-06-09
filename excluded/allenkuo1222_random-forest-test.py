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


train=pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s4e2/test.csv")


train.head(10)


# 呼叫pandas column
trainy=train["NObeyesdad"]#只想要某一個columns
trainx=train.drop("NObeyesdad",axis=1,inplace=False)


trainx.info()


test.info()


train.columns


train.select_dtypes(include=['int64', 'float64'])


train.select_dtypes(include=['int64', 'float64']).columns


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Observe numerical feature distribution
def manifest(data1,data2):

    num_columns = data1.select_dtypes(include=['int64', 'float64']).columns
    num_features = len(num_columns)
    n_cols=4
    n_rows= int(np.ceil(num_features/n_cols))


    plt.figure(figsize=(n_cols * 4, n_rows * 3))# 呼叫一張底圖
    for i, column in enumerate(num_columns, 1):
        plt.subplot(n_rows, n_cols, i)            # subplot(rowm,column, 現在在第幾張圖)
        sns.histplot(data1[column], color='blue', label='train') # plt第一個varieble (1d array) 數值分布圖(x ->數值，y->數值有幾個)
        sns.histplot(data2[column], color='green', label='test')
        plt.title(f'{column}') #string "abcsde"   f'{a}=6'
        plt.xlabel(column)
        plt.ylabel('Frequency')
    plt.suptitle('Feature distribution',fontsize=16,weight='bold',y=1.005)
    plt.tight_layout()
    plt.show()


manifest(train,test)


from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from category_encoders import MEstimateEncoder
# from sklearn.preprocessing import MinMaxScaler

#m_estimator_encoder = MEstimateEncoder()
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
#scaler = StandardScaler()


train_object=trainx.select_dtypes(include=['object'])


train_object.head(10)


train_object_name=trainx.select_dtypes(include=['object']).columns


encoded_data = one_hot_encoder.fit_transform(train_object[train_object_name])


encoded_data#numpy array


encoded_cols = one_hot_encoder.get_feature_names_out()
encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols,index=train_object.index)


encoded_df.head(5)


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
normal_scaler = MinMaxScaler()
scaler = StandardScaler()

def encoding_1(data_x, test_x):
    data_x_encoded = data_x.copy()
    test_x_encoded = test_x.copy()

    object_features = data_x_encoded.select_dtypes(include=['object']).columns
    numerical_features = data_x_encoded.select_dtypes(include=['int', 'float']).columns

    # One-Hot Encode
    encoded_data = one_hot_encoder.fit_transform(data_x[object_features])
    test_encoded_data = one_hot_encoder.transform(test_x[object_features])
    encoded_cols = one_hot_encoder.get_feature_names_out(object_features)
    
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=data_x.index)
    encoded_df_test = pd.DataFrame(test_encoded_data, columns=encoded_cols, index=test_x.index)

    data_x_encoded = pd.concat([data_x_encoded.drop(columns=object_features), encoded_df], axis=1)
    test_x_encoded = pd.concat([test_x_encoded.drop(columns=object_features), encoded_df_test], axis=1)
     # Fit scalers on training data only
    data_x_encoded[numerical_features] = normal_scaler.fit_transform(data_x[numerical_features])
    data_x_encoded[numerical_features] = scaler.fit_transform(data_x_encoded[numerical_features])
    
    test_x_encoded[numerical_features] = normal_scaler.transform(test_x[numerical_features])
    test_x_encoded[numerical_features] = scaler.transform(test_x_encoded[numerical_features])

    return data_x_encoded, test_x_encoded
# Now you can transform both training and test data
train_x ,test_x= encoding_1(trainx,test)


train_x.head(10)


test_x.head(10)


trainy


target_mapping = {
                  'Insufficient_Weight':0,
                  'Normal_Weight':1,
                  'Overweight_Level_I':2,
                  'Overweight_Level_II':3, 
                  'Obesity_Type_I':4,
                  'Obesity_Type_II':5 ,
                  'Obesity_Type_III':6
             
}
train_y=trainy.map(target_mapping)


train_y


manifest(train_x,test_x)


from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(
    n_estimators=1000,      # 決策樹數量
    max_depth=None,        # 樹的最大深度（None 代表直到葉節點）
    random_state=42        # 固定隨機種子以重現結果
    ,verbose=0
)


random_forest_model.fit(train_x, train_y)


from sklearn.metrics import accuracy_score
train_predict=random_forest_model.predict(train_x)
accuracy = accuracy_score(train_predict, train_y)


print(accuracy)


test_x.head(5)


testy=random_forest_model.predict(test_x)


train_x.index


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
# 進行「多數決（majority vote）」來產生最終預測結果，然後再將數字結果對應到原始的分類名稱。


# Assuming train_x and trainy are your data and labels
X = train_x
y = train_y

# Set up k-fold cross validation
k=5
skf = StratifiedKFold(n_splits=k)
# Initialize the model with your parameters
random_forest_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    random_state=42,
    verbose=0
)
# Perform k-fold cross validation
fold_accuracies = []
test_preds=[]
for fold, (train_idx, val_idx) in enumerate(skf.split(X,y)):
    # Split data for this fold
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train the model
    random_forest_model.fit(X_train_fold, y_train_fold)
    
    # Evaluate the model
    y_pred = random_forest_model.predict(X_val_fold)
    accuracy = accuracy_score(y_val_fold, y_pred)
    fold_accuracies.append(accuracy)
    
    # Print fold results
    print(f"Fold {fold+1}/{k} - Accuracy: {accuracy:.4f}")
    test_pred = random_forest_model.predict(test_x)
    test_preds.append(test_pred)
# Print average accuracy
print(f"\nAverage Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Standard Deviation: {np.std(fold_accuracies):.4f}")


# 假設 test_preds_array 為 shape = (k, n_samples)
# 每一欄代表同一個 sample 的多次預測結果
test_preds_array = np.array(test_preds)

# 對每一欄進行多數決，結果是一個一維陣列 (n_samples,)
final_preds_numeric = []
def majority_vote_with_dict(arr):
    vote_counts = {}  # 用 dictionary 當作 counter
    for item in arr:
        if item in vote_counts:
            vote_counts[item] += 1
        else:
            vote_counts[item] = 1
    # 找出出現最多次的 key
    most_common_label = max(vote_counts, key=vote_counts.get)
    return most_common_label
test_preds_array = np.array(test_preds)  # shape = (k, n_samples)
final_preds_numeric = []

for i in range(test_preds_array.shape[1]):  # 對每一欄做多數決
    column = test_preds_array[:, i]
    vote_result = majority_vote_with_dict(column)
    final_preds_numeric.append(vote_result)

final_preds_numeric = np.array(final_preds_numeric)
 


# 對應數字到原始類別名稱
reverse_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}
final_preds_labels = [reverse_mapping[p] for p in final_preds_numeric]


sample_submision=pd.read_csv('/kaggle/input/playground-series-s4e2/sample_submission.csv')


sample_submision['NObeyesdad']=final_preds_labels


sample_submision.to_csv('submission.csv',index=False)


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np
import pandas as pd

# 假設 train_x, train_y, test_x 都已定義好
X = train_x
y = train_y

k = 5
skf = StratifiedKFold(n_splits=k)

# 初始化 XGBoost 模型
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='mlogloss',  # 避免警告
    random_state=42,
    verbosity=0
)

# 儲存每折準確率與 test 預測
fold_accuracies = []
test_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # 訓練模型
    xgb_model.fit(X_train_fold, y_train_fold)

    # 驗證集預測與準確率
    y_pred = xgb_model.predict(X_val_fold)
    accuracy = accuracy_score(y_val_fold, y_pred)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold+1}/{k} - Accuracy: {accuracy:.4f}")

    # 測試集預測
    test_pred = xgb_model.predict(test_x)
    test_preds.append(test_pred)

# 多數決：針對每個 test_x 樣本，選擇出現最多的分類
test_preds_array = np.array(test_preds)  # shape: (k, len(test_x))


# 對每一欄進行多數決，結果是一個一維陣列 (n_samples,)
final_preds_numeric = []
def majority_vote_with_dict(arr):
    vote_counts = {}  # 用 dictionary 當作 counter
    for item in arr:
        if item in vote_counts:
            vote_counts[item] += 1
        else:
            vote_counts[item] = 1
    # 找出出現最多次的 key
    most_common_label = max(vote_counts, key=vote_counts.get)
    return most_common_label
test_preds_array = np.array(test_preds)  # shape = (k, n_samples)
final_preds_numeric = []

for i in range(test_preds_array.shape[1]):  # 對每一欄做多數決
    column = test_preds_array[:, i]
    vote_result = majority_vote_with_dict(column)
    final_preds_numeric.append(vote_result)

final_preds_numeric = np.array(final_preds_numeric)


reverse_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}
final_preds_labels = [reverse_mapping[p] for p in final_preds_numeric]


sample_submision=pd.read_csv('/kaggle/input/playground-series-s4e2/sample_submission.csv')


sample_submision['NObeyesdad']=final_preds_labels







