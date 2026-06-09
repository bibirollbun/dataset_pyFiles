# %load_ext cudf.pandas
# %load_ext cuml.accel


import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import cuml
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from cuml.ensemble import RandomForestRegressor as cuRF
from tqdm import tqdm
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv",).drop(['id'],axis=1).drop_duplicates()
train=train.reset_index(drop=True)
test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sub=pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")



kf=KFold(n_splits=20,shuffle=True,random_state=42)

# Target Encoding
for i,(train_index, test_index) in enumerate(kf.split(train)):
    X_train=train.loc[train_index]
    means=X_train.groupby('Sex')['Calories'].agg('mean')
    train.loc[test_index,'TE_Sex']=train.loc[test_index,'Sex'].map(means)

full_means=train.groupby('Sex')['Calories'].agg('mean')
test['TE_Sex']=test['Sex'].map(full_means)


# BMR
# vectorized BMR calculation:
def compute_bmr(df):
    return np.where(df['Sex'] == 'Male',
                    10 * df['Weight'] + 6.25 * df['Height']/100 - 5 * df['Age'] + 5,
                    10 * df['Weight'] + 6.25 * df['Height']/100 - 5 * df['Age'] - 161)

train['BMR'] = compute_bmr(train)
test['BMR'] = compute_bmr(test)


## BMI
train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)
test['BMI'] = test['Weight'] / ((test['Height'] / 100) ** 2)


# Duration / Heart Rate
train["Duration_per_HeartRate"] = train["Duration"] / (train["Heart_Rate"] + 1e-5)
test["Duration_per_HeartRate"] = test["Duration"] / (test["Heart_Rate"] + 1e-5)


train["Duration_x_HeartRate"] = train["Duration"] * train["Heart_Rate"]
test["Duration_x_HeartRate"] = test["Duration"] * test["Heart_Rate"]


# Create features by multiplying columns (inspired by https://www.kaggle.com/code/emircansoftware/predict-calories-eda-xgboost)
# cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

# for i in range(len(cols)):
#     for j in range(i + 1, len(cols)):
#         col1 = cols[i]
#         col2 = cols[j]
#         train[f"{col1}_mul_{col2}"] = train[col1] * train[col2]
#         test[f"{col1}_mul_{col2}"] = test[col1] * test[col2]


# Label Encoding of 'Sex' column
train['Sex']=train['Sex'].map({"male":1,"female":0})
test['Sex']=test['Sex'].map({"male":1,"female":0})

train.head()


features=train.drop(['Calories'],axis=1).columns
features


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


kf=KFold(n_splits=18,shuffle=True,random_state=42)

def xgboost_regressor(train, y, kf,n_estimators):
    oof = np.zeros(len(train))  # Out-of-Fold predictions
    best_iterations = []
    with tqdm(total=kf.get_n_splits(train)) as pbar:
        for fold_idx, (train_index, test_index) in enumerate(kf.split(train)):
            X_train, y_train = train.loc[train_index], y.loc[train_index]
            X_valid,y_valid = train.loc[test_index],y.loc[test_index]
       

            # Train model
            model=  XGBRegressor(
                                n_estimators=n_estimators,
                                learning_rate=0.05,
                                max_depth=6,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_lambda=40,
                                early_stopping_rounds=20,
                                tree_method='hist',
                                eval_metric='rmse',
                device='cuda',
                            )
            model.fit(X_train, y_train,
             eval_set=[(X_valid, y_valid)],verbose=False)

            # Predict on validation set
            y_pred_valid = model.predict(X_valid, iteration_range=(0, model.best_iteration + 1))

            
            # Store validation predictions
            oof[test_index] = y_pred_valid
            best_iterations.append(model.best_iteration)
            # print(f"Fold {len(best_iterations)}: best_iteration = {model.best_iteration}")
            
            pbar.update(1)

    return oof, model, best_iterations


y=train['Calories']
y_log = np.log1p(y)
oof_preds, xgb_model, best_iters=xgboost_regressor(train[features], y_log, kf,n_estimators=1000)

cv_score=np.sqrt(mean_squared_error(oof_preds, y_log))
print("\nCv Score \n",cv_score)



avg_best_iter = int(sum(best_iters) / len(best_iters))
print(f"\nAverage best_iteration to use for final model: {avg_best_iter}")




final_model=XGBRegressor(
                                n_estimators=avg_best_iter,
                                learning_rate=0.05,
                                max_depth=6,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                tree_method='hist',
                                eval_metric='rmse',
                device='cuda',
                            )
final_model.fit(train[features], y_log)


test_predictions=np.expm1(final_model.predict(test[features]))
sub.to_csv(f"submission.csv",index=False)
sub['Calories']=test_predictions
sub.head()




