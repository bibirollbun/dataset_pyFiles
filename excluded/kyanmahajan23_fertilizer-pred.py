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


#DATA UNDERSTANDING

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt







test_Data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train_Data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')



train_Data.head()


idd= test_Data['id']


#functions for data_Analaysis
def data_analyse(data, target_col):
    numerical_cols = data.select_dtypes(include = np.number).columns.tolist();
    fig, axes = plt.subplots(int(len(numerical_cols)), 2, figsize= (15,20));
    for i, col in enumerate(numerical_cols):
        
        sns.histplot(data=data, x=col, hue=target_col,  ax=axes[i, 0])
        sns.boxplot(data=data, y=col, x=target_col, ax=axes[i, 1])

    plt.subplots_adjust(
    left=0.1,   # space from the left edge
    right=0.9,  # space from the right edge
    top=0.9,    # space from the top
    bottom=0.1, # space from the bottom
    wspace=0.4, # width space between columns
    hspace=0.4  # height space between rows
)

    plt.show()



   




 print(train_Data.select_dtypes(include = np.number).columns.tolist())


# data_analyse(train_Data, 'Fertilizer Name')


# sns.heatmap(
#     train_Data.select_dtypes(include = np.number).corr()
# )


Y_train = train_Data['Fertilizer Name']



#feature engineering
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

engineer_data =  ColumnTransformer( 
    [
        ("scale", StandardScaler(), train_Data.select_dtypes(include = np.number).columns.tolist()),
        ("encode", OneHotEncoder(), ["Soil Type", "Crop Type"])
        
    ]
)
engineer_data.fit(train_Data.drop(['Fertilizer Name'], axis =1))
train_Data_ =  engineer_data.transform(train_Data)




label = LabelEncoder();
Y_train = label.fit_transform(Y_train)


test_trans = engineer_data.transform(test_Data)


def map_at_3(y_true, y_pred_proba, k=3):
    """Calculate Mean Average Precision at K (MAP@K) metric"""
    map_score = 0.0
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    for i in range(len(y_true)):
        # Get indices of top k predictions
        top_k_preds = np.argsort(y_pred_proba[i])[-k:][::-1]
        if y_true[i] in top_k_preds:
            # Find position of true label in top k predictions
            rank = np.where(top_k_preds == y_true[i])[0][0] + 1
            map_score += 1.0 / rank
    return map_score / len(y_true)

def log_map_at_3(y_true, y_pred_proba, k=3):
    """Log MAP@3 metric with additional information"""
    score = map_at_3(y_true, y_pred_proba, k)
    print(f"MAP@{k}: {score:.6f}")
    return score


from sklearn.ensemble import RandomForestClassifier


lgb_params = {
    'objective': 'multiclass',
    'num_class': 7,
    'device': 'gpu',
    'colsample_bytree': 0.4366677273946288,
    'learning_rate': 0.026164161953515117,
    'max_depth': 12,
    'min_child_samples': 67,
    'n_estimators': 100,
    'n_jobs': -1,
    'num_leaves': 243,
    'random_state': 42,
    'reg_alpha': 6.38288560443373,
    'reg_lambda': 9.392999314379155,
    'subsample': 0.7989164499431718,
    'verbose': -1
}
xgb_params = {
    'tree_method': 'hist',
    'n_estimators': 500,
    'objective': 'multi:softprob',
    'random_state': 32,
    'enable_categorical': True,
    'verbosity': 0,
    'eval_metric': 'mlogloss',
    'booster': 'gbtree',
    "device": "cuda",
    'n_jobs': -1,
    'learning_rate': 0.1,
    'num_class': 7,
    'lambda': 0.05656209749983576,
    'alpha': 5.620898657099113,
    'colsample_bytree': 0.2587327850345624, 
    'subsample': 0.8276149323901826,
    'max_depth': 20,
    'min_child_weight': 10
    }



from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

xgb_pred = []
lgbb_pred = []
xgb_test_pred = [];
lgbb_test_pred = [];

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(cv.split(train_Data_, Y_train)):
    X_train = train_Data_[train_idx]
    X_val = train_Data_[test_idx]
    y_train = Y_train[train_idx]
    y_val = Y_train[test_idx]

    print(f"xbb starts")
    print(f"{fold+1} starts")
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], early_stopping_rounds = 50)
    y_pred_xgb = xgb.predict(X_val)
    xgb_pred.append(y_pred_xgb)
    xgb_test_pred.append(xgb.predict(test_trans))
    
    print(f"XGB done")

    print(f"rf starts")
    rf = LGBMClassifier(**lgb_params)
    rf.fit( X_train, y_train )
    y_pred_lgb = rf.predict(X_val)
    lgbb_pred.append(y_pred_lgb)
    lgbb_test_pred.append(rf.predict(test_trans))
    print(f"lgbb done")

    print(f"Fold {fold + 1} done.")







import pickle


 
# Save
with open('my_list.pkl', 'wb') as f:
    pickle.dump(xgb_pred, f)

with open('my_list2.pkl', 'wb') as f:
    pickle.dump(lgbb_pred, f)


xgb_pred_ = (np.array(xgb_pred))


xgb_pred_ = xgb_pred_.reshape(750000,1)


(np.array(xgb_pred)).shape


lgbb_pred_ = (np.array(lgbb_pred))


lgbb_pred_ = lgbb_pred_.reshape(750000,1)


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np

# Step 1: Prepare meta-features (out-of-fold predictions)
# Assuming xgb_oof and lgb_oof are (n_samples,) arrays aligned with Y_train
meta_features = np.hstack([xgb_pred_, lgbb_pred_])  # shape: (n_samples, 2)

# Step 2: Split meta-features for training meta-model
X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
    meta_features, Y_train, test_size=0.2, random_state=42
)

# Step 3: Train meta-model
meta_model = XGBClassifier(**xgb_params)
meta_model.fit(X_meta_train, y_meta_train)

# Step 4: Predict and evaluate
y_pred = meta_model.predict(X_meta_val)
print(log_map_at_3(y_pred, y_meta_val))


np.array(xgb_test_pred).shape


test_trans.shape


from scipy.stats import mode

xgb_test_pred_ = np.stack(xgb_test_pred)  # shape: (n_folds, 1_250_000)
final_pred = mode(xgb_test_pred_, axis=0)[0].flatten() 


from scipy.stats import mode
lgb_test_pred_ = np.stack(lgbb_test_pred)  # shape: (n_folds, 1_250_000)
final_pred_ = mode(lgb_test_pred_, axis=0)[0].flatten() 


test_data = np.hstack([ final_pred.reshape(-1,1), final_pred_.reshape(-1,1)])


test_data.shape


y_pred_final  = meta_model.predict(test_data)


y_pred_final


y_pred_final_ = label.inverse_transform(y_pred_final)


submission = pd.DataFrame({
    'id': idd,
    'Fertilizer Name': y_pred_final_
})



submission.to_csv("subb.csv", index=False)


test_trans


 xgb = XGBClassifier(**xgb_params)
 xgb.fit(train_Data_, Y_train)
 


y_predd = xgb.predict(test_trans);
y_predd_final = label.inverse_transform(y_predd);
submissionn = pd.DataFrame({
    'id': idd,
    'Fertilizer Name': y_predd_final
})



submissionn.to_csv('subbb.csv', index=False)


lg = LGBMClassifier(**lgb_params)
lg.fit(train_Data_, Y_train)
y_predd_ = lg.predict(test_trans);
y_predd_final_ = label.inverse_transform(y_predd_);
submissionnn = pd.DataFrame({
    'id': idd,
    'Fertilizer Name': y_predd_final_
})
submissionnn.to_csv('subbbb.csv', index=False)

