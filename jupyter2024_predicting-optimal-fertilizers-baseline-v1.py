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


# Import package 
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
import xgboost as xgb
import optuna


# Import Data 
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


# 1.将目标变量使用LabelEncoder 编码
label_encoder = LabelEncoder()
train['Fertilizer_encoder'] = label_encoder.fit_transform(train['Fertilizer Name'])

# 2.对土地类型编码
soil_encoder = TargetEncoder(cols=['Soil Type'],smoothing=1.0)
train['soil_encoder'] = soil_encoder.fit_transform(train['Soil Type'],train['Fertilizer_encoder'])
test['soil_encoder'] = soil_encoder.transform(test['Soil Type'])
# 3. 农作物类型编码
crop_encoder = TargetEncoder(cols=['Crop Type'],smoothing=1.0)
train['crop_encoder'] = crop_encoder.fit_transform(train['Crop Type'],train['Fertilizer_encoder'])
test['crop_encoder'] = crop_encoder.transform(test['Crop Type'])

# 氮磷钾非线性转换
train['NK'] = train['Nitrogen'] * train['Potassium']
train['NP'] = train['Nitrogen'] * train['Phosphorous']
train['PK'] = train['Potassium'] * train['Phosphorous']
train['NP_ratio'] = train['Nitrogen'] / (train['Phosphorous'] + 1e-6)
train['NK_ratio'] = train['Nitrogen'] / (train['Potassium'] + 1e-6)
train['PK_ratio'] = train['Potassium'] / (train['Phosphorous'] + 1e-6)

test['NK'] = test['Nitrogen'] * test['Potassium']
test['NP'] = test['Nitrogen'] * test['Phosphorous']
test['PK'] = test['Potassium'] * test['Phosphorous']
test['NP_ratio'] = test['Nitrogen'] / (test['Phosphorous'] + 1e-6)
test['NK_ratio'] = test['Nitrogen'] / (test['Potassium'] + 1e-6)
test['PK_ratio'] = test['Potassium'] / (test['Phosphorous'] + 1e-6)

# 温度交付
train['temp_humidity'] = train['Temparature'] * train['Humidity'] /100 
test['temp_humidity'] = test['Temparature'] * test['Humidity'] /100 
# 水汽交付
train['water_stress'] = train['Moisture'] / (train['Temparature'] + 1e-6)
test['water_stress'] = test['Moisture'] / (test['Temparature'] + 1e-6)

# 基于环境境条件聚类
env_cluster = KMeans(n_clusters=5).fit_predict(train[['Temparature','Humidity','Moisture']])
train['env_cluster'] = env_cluster
env_cluster = KMeans(n_clusters=5).fit_predict(test[['Temparature','Humidity','Moisture']])
test['env_cluster'] = env_cluster

# 基于土壤特征聚类
soil_cluster = KMeans(n_clusters=3).fit_predict(train[['Nitrogen','Potassium','Phosphorous','soil_encoder']])
train['soil_cluster'] = soil_cluster
    
test_soil_cluster = KMeans(n_clusters=3).fit_predict(test[['Nitrogen','Potassium','Phosphorous','soil_encoder']])
test['soil_cluster'] = test_soil_cluster


#  Filter feature
train_new = train.drop(columns=['id','Soil Type','Crop Type','Fertilizer Name'],axis=1)
X = train_new.drop(columns=['Fertilizer_encoder'],axis=1)
y = train_new['Fertilizer_encoder']
test_new = test.drop(columns=['id','Soil Type','Crop Type'],axis=1)


# Divide the dataset 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create data in DMatrix format.
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(test_new)


# =====================
# The calculation function of MAP@3
# =====================
def calculate_map_at_3(y_true, y_pred_probs, k=3):
    """
    Mean Average Precision at K (MAP@K)
    
    params:
    y_true: true_label_index (n_samples,)
    y_pred_probs: Prediction probability matrix (n_samples, n_classes)
    k: Consider the top K predictions.
    
    return:
    map_at_k: MAP@K
    """
    # Obtain the top-k prediction indices
    top_k_indices = np.argsort(-y_pred_probs, axis=1)[:, :k]
    
    # Calculate the AP@K for each sample
    ap_scores = []
    for i in range(len(y_true)):
        true_label = y_true.iloc[i]
        predicted_labels = top_k_indices[i]
        
        # Calculate AP@K
        relevant_count = 0
        precision_sum = 0
        for j in range(k):
            if predicted_labels[j] == true_label:
                relevant_count += 1
                precision_sum += relevant_count / (j + 1)
        
        ap = precision_sum / min(k, relevant_count) if relevant_count > 0 else 0
        ap_scores.append(ap)
    
    # Calculate MAP@K
    map_at_k = np.mean(ap_scores)
    return map_at_k


def objective(trial):
    # Define the hyperparameter search space
    params = {
        'objective': 'multi:softprob',
        'num_class': 7,
        'eval_metric': 'merror',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'lambda': trial.suggest_float('lambda', 0, 10),
        'alpha': trial.suggest_float('alpha', 0, 10),
        'seed': 42
    }
    
    # train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'test')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    # predict MAP@3
    y_pred_proba = model.predict(dval)
    return calculate_map_at_3(y_val, y_pred_proba, k=3)

# Run optimization
# study = optuna.create_study(direction='maximize')  
# study.optimize(objective, n_trials=50)


# output best_params
# print("best_paramss:")
# for key, value in study.best_params.items():
#     print(f"  {key}: {value}")
# print(f" best MAP@3 score: {study.best_value:.4f}")


best_params = {'max_depth': 6,
 'eta': 0.18886924341876193,
 'subsample': 0.69906449493703,
 'colsample_bytree': 0.8178065891884582,
 'min_child_weight': 9.93375550867301,
 'gamma': 0.025621156165335,
 'lambda': 3.3054862574054122,
 'alpha': 6.704729679693543,
 'objective': 'multi:softprob',
 'num_class': 7,
 'eval_metric': 'merror',
 'seed': 42}


#  Model 
model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'test')],
    early_stopping_rounds=20,
    verbose_eval=False
)


# Predict Data 
pred_probs = model.predict(dtest)    
# get top-3 index
top_k_indices = np.argsort(-pred_probs, axis=1)[:, :3]
    
# Convert indices to original labels
predictions = []
for indices in top_k_indices:
    # Convert numeric labels to original fertilizer names
    labels = label_encoder.inverse_transform(indices)
    predictions.append(' '.join(labels))


# Save Results
result_df = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': predictions
})
result_df.to_csv('/kaggle/working/submission.csv', index=False)
print("The prediction results have been saved to 'submission.csv'")

