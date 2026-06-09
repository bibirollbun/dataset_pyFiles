import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


train=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in ["ID", "efs", "efs_time"]]
numerical_cols = [col for col in numerical_cols if col not in ["ID", "efs", "efs_time"]]
hla_cols = [col for col in train.columns if "hla" in col.lower()]


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))  
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le  


hla_bins = {
    'hla_match_c_high': [0.0, 1.0, 2.0],
    'hla_high_res_8': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'hla_low_res_6': [2.0, 3.0, 4.0, 5.0, 6.0],
    'hla_high_res_6': [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'hla_high_res_10': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'hla_match_dqb1_high': [0.0, 1.0, 2.0],
    'hla_nmdp_6': [2.0, 3.0, 4.0, 5.0, 6.0],
    'hla_match_c_low': [0.0, 1.0, 2.0],
    'hla_match_drb1_low': [1.0, 2.0],
    'hla_match_dqb1_low': [0.0, 1.0, 2.0],
    'hla_match_a_high': [0.0, 1.0, 2.0],
    'hla_match_b_low': [0.0, 1.0, 2.0],
    'hla_match_a_low': [0.0, 1.0, 2.0],
    'hla_match_b_high': [0.0, 1.0, 2.0],
    'hla_low_res_8': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'hla_match_drb1_high': [0.0, 1.0, 2.0],
    'hla_low_res_10': [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

for col, bins in hla_bins.items():
    train[col] = pd.cut(train[col], bins=bins, labels=False, include_lowest=True)

train[hla_cols] = train[hla_cols].fillna(-1)


for col, bins in hla_bins.items():
    test[col] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)

test[hla_cols] = test[hla_cols].fillna(-1)


scaler = StandardScaler()
hla_scaled = scaler.fit_transform(train[hla_cols])
pca = PCA(n_components=len(hla_cols))  
train_hla_pca = pca.fit_transform(hla_scaled)

optimal_components = 6  
pca_opt = PCA(n_components=optimal_components)
train_hla_pca_opt = pca_opt.fit_transform(hla_scaled)

pca_columns = [f"PCA_{i+1}" for i in range(optimal_components)]
train_pca_df = pd.DataFrame(train_hla_pca_opt, columns=pca_columns)

train = pd.concat([train.drop(columns=hla_cols), train_pca_df], axis=1)


hla_scaled_t = scaler.transform(test[hla_cols])  

test_hla_pca = pca.transform(hla_scaled_t) 

test_hla_pca_opt = pca_opt.transform(hla_scaled_t)  

pca_columns_t = [f"PCA_{i+1}" for i in range(optimal_components)]
test_pca_df = pd.DataFrame(test_hla_pca_opt, columns=pca_columns_t)

test = pd.concat([test.drop(columns=hla_cols), test_pca_df], axis=1)


import pandas as pd
from sklearn.impute import KNNImputer

numeric_to_categorical = ['comorbidity_score', 'karnofsky_score']

bins_dict = {
    'karnofsky_score': [0, 10, 20, 30, 40, 60, 80, 100], 
    'comorbidity_score': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  
}


for col in numeric_to_categorical:
    train[col] = pd.cut(train[col], bins=bins_dict[col], labels=False, include_lowest=True)

train[numeric_to_categorical] = train[numeric_to_categorical].fillna(-1)


for col in numeric_to_categorical:
    test[col] = pd.cut(test[col], bins=bins_dict[col], labels=False, include_lowest=True)

test[numeric_to_categorical] = test[numeric_to_categorical].fillna(-1)


from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5)

train['donor_age'] = knn_imputer.fit_transform(train[['donor_age']])
test['donor_age'] = knn_imputer.transform(test[['donor_age']])


# from lifelines import WeibullAFTFitter
# from sklearn.model_selection import train_test_split
# import pandas as pd

# X = train.drop(columns=['efs', 'efs_time', 'ID'])
# y = train[['efs', 'efs_time']]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train_data = pd.concat([X_train, y_train], axis=1)

# aft_model = WeibullAFTFitter()
# aft_model.fit(train_data, duration_col='efs_time', event_col='efs')


# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from lifelines.utils import concordance_index
# import pandas as pd

# # Assuming you've already performed PCA and preprocessing
# # X: Features after PCA, y: Target variable

# # Example: Drop the target columns and any irrelevant columns
# X = train.drop(columns=['efs', 'efs_time', 'ID'])  # After PCA
# y = train[['efs', 'efs_time']]

# # Log transform the 'efs_time' to reduce skewness
# y['log_efs_time'] = np.log1p(y['efs_time'])  # Log transformation to avoid log(0)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # XGBoost requires a DMatrix object, so we convert our data to that format
# dtrain = xgb.DMatrix(X_train, label=y_train['log_efs_time'])
# dtest = xgb.DMatrix(X_test, label=y_test['log_efs_time'])

# # Define the model parameters
# params = {
#     'objective': 'reg:squarederror',  # For regression tasks
#     'eval_metric': 'rmse',  # You can change this metric depending on your needs
#     'max_depth': 6,
#     'eta': 0.1,
#     'subsample': 1,
#     'colsample_bytree': 0.8
# }

# # Train the XGBoost model
# num_round = 100  # Number of boosting rounds
# bst = xgb.train(params, dtrain, num_round)

# # Make predictions (risk scores)
# y_pred = bst.predict(dtest)

# # Inverse log-transform to get the actual survival times back
# y_pred_actual = np.expm1(y_pred)  # Inverse of log1p (log(x + 1))

# # Evaluate performance using RMSE (Root Mean Squared Error)
# rmse = mean_squared_error(np.expm1(y_test['log_efs_time']), y_pred_actual, squared=False)
# print(f'RMSE: {rmse}')

# # Calculate the concordance index (C-index)
# c_index = concordance_index(y_test['efs_time'], y_pred_actual, event_observed=y_test['efs'])
# print(f'C-index: {c_index}')



import xgboost as xgb
import numpy as np
import pandas as pd

X_train = train.drop(columns=['efs', 'efs_time', 'ID'])  
y_train = train[['efs', 'efs_time']] 
y_train['log_efs_time'] = np.log1p(y_train['efs_time']) 

dtrain = xgb.DMatrix(X_train, label=y_train['log_efs_time'])


params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse', 
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 1,
    'colsample_bytree': 0.8
}

num_round = 100  
bst = xgb.train(params, dtrain, num_round)
X_test = test.drop(columns=['ID']) 
dtest_actual = xgb.DMatrix(X_test)
y_pred_actual = bst.predict(dtest_actual)
y_pred_actual = np.expm1(y_pred_actual) 
y_pred_actual = -y_pred_actual

submission_df = pd.DataFrame({
    'ID': test['ID'],
    'prediction': y_pred_actual
})

submission_df.to_csv('submission.csv', index=False)
print("Test predictions saved successfully for XGBoost.")





