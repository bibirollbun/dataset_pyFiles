import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col='id')
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


sample_df.head()


import warnings
warnings.filterwarnings('ignore')


# info of the data
print("\n <==Train Data Info==>")
print(train_df.info())

print("\n <==Test Data Info==>")
print(test_df.info())

# head of data
print("\n <==Head of Train Data==>")
display(train_df.head())

print("\n <===Head of Test Data==>")
display(test_df.head())

# Shape of  data
print("\n Shape of train data: ", train_df.shape)

print("\n Shape of test data: ", test_df.shape)


# missing values in data
print("\n <==Missing Values in train data==>")
print(train_df.isna().sum())

print("\n <==Missing Values in test data===>")
print(test_df.isna().sum())

# check if duplicate rows 
print("\n No.of Duplicate Rows in train Data:",train_df.duplicated().sum())

print("\n No.of Duplicate Rows in test Data:",test_df.duplicated().sum())


print("\n <==Columns in training data==>")
print(train_df.columns)
print("---"*10)
print("\n <==Columns in testing data==>")
print(test_df.columns)

target = list(set(train_df.columns)- set(test_df.columns))[0]
print(f'\n Target column is: {target}')


# Categorical and Numerical Columns in training data
cat_cols = [col for col in train_df.columns if train_df[col].dtype=='object' and col is not target]
num_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64','float64']]

print("\n <==Categorical columns in training data==>")
print(cat_cols)

print("\n <==Numerical columns in training data==>")
print(num_cols)


print("\n <==Unique values in each nominal Categorical columns of training==>")
print(train_df[cat_cols].nunique())


print("\n <==value counts for low cardinality category columns:")
for col in cat_cols:
    print(f"\n  Value count for {col}: ")
    print(train_df[col].value_counts())


# skewness
print("\n <==skeweness of Numerical columns in train_df data==>")
print(train_df[num_cols].skew().sort_values())


print("\n <==correlation of Numerical columns with target==>")

target_corr = train_df.select_dtypes(include=['int64','float64']).corr()
target_corr
# .sort_values(ascending=False)


y = train_df['Fertilizer Name']
X = train_df.drop('Fertilizer Name',axis=1)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val =train_test_split(X,y,test_size=0.2,random_state=42)


print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)


from sklearn.preprocessing import OneHotEncoder

oh_encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')

encoded_train = pd.DataFrame(oh_encoder.fit_transform(X_train[cat_cols]),
                             columns = oh_encoder.get_feature_names_out(),
                             index=X_train.index)

encoded_val = pd.DataFrame(oh_encoder.transform(X_val[cat_cols]),
                             columns = oh_encoder.get_feature_names_out(),
                             index=X_val.index)

X_train_encoded = pd.concat([X_train.drop(['Soil Type','Crop Type'],axis=1),encoded_train],axis=1)
X_val_encoded = pd.concat([X_val.drop(['Soil Type','Crop Type'],axis=1),encoded_val],axis=1)


print(X_train_encoded.shape, y_train.shape)
print(X_val_encoded.shape, y_val.shape)


X_train_encoded.head()


train_df['Fertilizer Name'].unique()


from sklearn.preprocessing import LabelEncoder

target_encoder = LabelEncoder()

y_train_encoded = target_encoder.fit_transform(y_train)
y_val_encoded = target_encoder.transform(y_val)


target_encoder.classes_


# stop


import xgboost as xgb

dtrain = xgb.DMatrix(X_train_encoded, label=y_train_encoded)
dval = xgb.DMatrix(X_val_encoded, label=y_val_encoded)

params = {
    'objective': 'multi:softprob',
    'num_class': len(target_encoder.classes_),
    'eval_metric': 'mlogloss',
    'eta': 0.1,                # Faster learning
    'max_depth': 6,            # Shallower tree = faster
    'min_child_weight': 3,  
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 2.0,              # L2 regularization
    'alpha': 1.0,               # L1 regularization
    'gamma': 1.0,               # Minimum loss reduction for further partitioning
    'seed': 42
}


model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,        
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=20,
    verbose_eval=20
)


# ========== Evaluate with MAP@3 ==========
def mapk(actual, predicted, k=3):
    """Compute mean average precision at k (MAP@k)."""
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break  # only the first correct prediction counts
        return score
        
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# predicting top 3 probable targets for validation
dval = xgb.DMatrix(X_val_encoded)

pred_probs_val = model.predict(dval)

# Choose the class with the highest predicted probability
sorted_val_pred_idx = np.argsort(pred_probs_val, axis=1)

k =3  # k<=7
top_k_val_idx = sorted_val_pred_idx[:,-k:][:,::-1]

print("shape of top 3 probable target indexes:", top_k_val_idx.shape)


y_val_encoded


top_k_val_idx


# Calculate MAP@3
map3_score = mapk(y_val_encoded, top_k_val_idx, k=3)
print(f"\nðŸ“Š Mean Average Precision @3 (MAP@3): {map3_score:.5f}")


print(test_df.shape)
test_df.head()


# encoding the test data
encoded_test = pd.DataFrame(oh_encoder.transform(test_df[cat_cols]),
                             columns = oh_encoder.get_feature_names_out(),
                             index=test_df.index)

test_df_encoded = pd.concat([test_df.drop(['Soil Type','Crop Type'],axis=1),encoded_test],axis=1)

print(test_df_encoded.shape)
test_df_encoded.head()


# predicting top 3 probable targets

dtest = xgb.DMatrix(test_df_encoded)

pred_probs = model.predict(dtest)

# Choose the class with the highest predicted probability
sorted_pred_idx = np.argsort(pred_probs, axis=1)

k =3  # k<=7
top_k_idx = sorted_pred_idx[:,-k:][:,::-1]

print("shape of top 3 probable target indexes of validation:", top_k_idx.shape)


# Convert the predicted class indices back to actual fertilizer names

top_k_preds = target_encoder.inverse_transform(top_k_idx.reshape(-1,1)).reshape(top_k_idx.shape)
print("shape of top k preds:",top_k_preds.shape)
print("Top 3 fertilizers predicted:")
top_k_preds


 [" ".join(row) for row in top_k_preds][:10]


sample_df.head()


# Creating the submission file
output = pd.DataFrame({
    'id': test_df.index,
    'Fertilizer Name': [" ".join(row) for row in top_k_preds] 
})

output.to_csv('xgb_map_submission.csv', index=False)

print(f'\n top 3 fertilizers prediction of test_df: \n',output.head())

print(f'\nSubmission file using XGB model created successfully!')
print('-----------------------------------\n')

















pred_probs[:5,:]


np.argsort(pred_probs, axis=1)


arr = [[0,5,2,3,4],
        [1,5,7,2,5]]

top_k_preds = np.argsort(arr, axis=1)[:,-3:][:,::-1]
top_k_preds
# arr[sorted_probs]


top_k_preds.reshape(-1,1)


top_3_labels = target_encoder.inverse_transform(top_k_preds.ravel().reshape(-1,1)).reshape(top_k_preds.shape)
top_3_labels 


target_encoder.inverse_transform(top_k_preds.reshape(-1, 1)).reshape(top_k_preds.shape)


[' '.join(row) for row in top_3_labels]




