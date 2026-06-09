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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch


train_df=pd.read_csv('/kaggle/input/cmi-bfrb-detection-knn-imputed-dataset/train_knn_imputed_columned.csv')
train_demo_df=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')


tot_train_df=pd.merge(train_df,train_demo_df,on='subject',how='left') # merging wrt to subject(the person)
tot_train_df=tot_train_df.drop(columns=['row_id','subject','orientation','behavior','phase'])
X=tot_train_df.drop(columns=['sequence_id','sequence_type','gesture','sequence_counter'])


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le=LabelEncoder()
y=le.fit_transform(tot_train_df['gesture'])
X_temp,X_test,y_temp,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)
X_train,X_valid,y_train,y_valid=train_test_split(
    X_temp,y_temp, test_size=0.25,stratify=y_temp,random_state=42
)


from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation

cb_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function='MultiClass',
    early_stopping_rounds=20,
    l2_leaf_reg=0.40,
    random_state=42,
    verbose=50,
    task_type='GPU',
    devices='0'
)

xgb_model = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    use_label_encoder=False,
    min_child_weight=5,
    colsample_bytree=0.5,
    subsample=0.8,
    n_estimators=2000,
    random_state=42,
    max_depth=6,
    learning_rate=0.1,
    num_class=len(le.classes_),
    early_stopping_rounds=50,
    tree_method='auto',
    device='cuda'
)

lgbm_model =LGBMClassifier(
    boosting_type='gbdt',
    objective='multiclass',
    num_class=len(le.classes_),
    metric='multi_logloss',

    learning_rate=0.03,
    n_estimators=200,
    num_leaves=256,
    max_depth=12,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_gain_to_split=0.01,

    lambda_l1=0.1,
    lambda_l2=0.1,

    random_state=42,

    # GPU if available
    device='gpu'
)

model=lgbm_model
model.fit(
    X_train,y_train,
    eval_set=[(X_valid,y_valid)],
    eval_metric='multi_logloss',
    callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=10)
    ]
)


from sklearn.metrics import accuracy_score,f1_score

y_pred=model.predict(X_test)
acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred,average='macro')

print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1 Score: {f1:.4f}")


import polars as pl
import kaggle_evaluation.cmi_inference_server


def merging(test,test_demographics):
    merged = pd.merge(
        test,
        test_demographics,
        on='subject',
        how='left'
    )
    merged=merged.drop(columns=['row_id','subject'])
    return merged


def NaN_Val(merged):
    sensor_col = [col for col in merged.columns if col.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
    demo_col = ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    
    for col in sensor_col:
        if merged[col].isnull().any():
            merged[col].fillna(0, inplace=True)
            
    for col in demo_col:
        if merged[col].isnull().any():
            if col in ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']:
                merged[col].fillna(merged[col].median(), inplace=True)
            elif col in ['adult_child','sex', 'handedness']:
                merged[col].fillna(merged[col].mode()[0] , inplace=True)
    
    X_final = merged.drop(columns=['sequence_id','sequence_counter'])
    return X_final


def predict(test: pd.DataFrame, demo: pd.DataFrame)-> pd.DataFrame:
    test = test.to_pandas()
    demo = demo.to_pandas()
    merged=merging(test,demo)
    X_final=NaN_Val(merged)
    y_pred = model.predict(X_final)
    print(y_pred)
    
    pred_gesture=le.inverse_transform(y_pred.ravel())
    mode_pred = pd.Series(pred_gesture).mode()
    ans = mode_pred.iloc[0]
    return str(ans)



test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_demo_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")

merged_df = merging(test_df, test_demo_df)
X_finall = NaN_Val(merged_df)
y_pred_final = model.predict(X_finall)

pred_gestures_final = le.inverse_transform(y_pred_final.ravel()) 

submission_df = pd.DataFrame({'sequence_id': test_df['sequence_id'], 'gesture': pred_gestures_final})
submission_df.to_parquet('submission.parquet', index=False)


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )




