# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
import numpy as np

train_path="/kaggle/input/playground-series-s5e5/train.csv"
test_path="/kaggle/input/playground-series-s5e5/test.csv"

train_csv= pd.read_csv(train_path)
train_csv.head(5)


import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):
    df = df.copy()

    # Drop irrelevant columns if present
    df.drop(columns=[col for col in ['id', 'User_ID'] if col in df.columns], inplace=True)

    # Ensure all relevant columns are numeric (avoids type errors)
    numeric_cols = ['Age', 'Weight', 'Height', 'Duration', 'Heart_Rate', 'Body_Temp']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Encode Sex or Gender
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 2}).fillna(df['Sex'])
    if 'Gender' in df.columns:
        df['Sex'] = df['Gender'].map({'female': 1, 'male': 2}).fillna(df['Gender'])
        df.drop(columns=['Gender'], inplace=True)

    df['Sex'] = pd.to_numeric(df['Sex'], errors='coerce')

    # Create combined categorical feature
    df['AgeSex'] = df['Age'].astype(str) + df['Sex'].astype(str)
    df['AgeSex'] = LabelEncoder().fit_transform(df['AgeSex']) + 1

    for col in ['Sex', 'Age', 'AgeSex']:
        df['CAT_' + col] = df[col].astype('category')
        # Feature: Body Mass Index (BMI)
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

    # Heart rate per age
    df['Heart_Rate_per_Age'] = df['Heart_Rate'] / df['Age']

    # Interaction between temperature and heart rate
    df['Temp_Heart_Interaction'] = df['Body_Temp'] * df['Heart_Rate']

    # Mean and std of Heart Rate by Sex
    group_stats = df.groupby('Sex')['Heart_Rate'].agg(['mean', 'std']).rename(
        columns={'mean': 'Mean_HR_by_Sex', 'std': 'Std_HR_by_Sex'}
    )
    df = df.merge(group_stats, on='Sex', how='left')

    # Difference from group mean
    df['HR_above_group_mean'] = df['Heart_Rate'] - df['Mean_HR_by_Sex']

    
     # Log and root transformations
    df['Log_Weight'] = np.log1p(df['Weight'])
    df['Sqrt_Height'] = np.sqrt(df['Height'])

    # Polynomial terms
    df['Age_squared'] = df['Age'] ** 2
    df['Weight_cubed'] = df['Weight'] ** 3

    # Interaction terms
    features = ['Age', 'Weight', 'Height', 'Body_Temp', 'Heart_Rate', 'Duration', 'Sex', 'AgeSex']
    for comb in itertools.combinations(features, 2):
        col_name = f"{comb[0]} * {comb[1]}"
        df[col_name] = df[comb[0]] * df[comb[1]]

    return df

train_fe = feature_engineering(train_csv)
train_fe["Calories"]=np.log1p(train_fe['Calories'])
train_fe


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# cat_features = 

x=train_fe.drop("Calories",axis=1)
y=train_fe.Calories
feature_cols = x.columns 

xgb_model = XGBRegressor(**{'tree_method': 'hist',
                            'n_estimators': 1500,
                            'objective': 'reg:squarederror',
                            'random_state': 42,
                            'enable_categorical': True,
                            'verbosity': 0,
                            # 'early_stopping_rounds': 100,
                            'eval_metric': 'rmse',
                            'booster': 'gbtree',
                            "device": "cuda",
                            'n_jobs': -1,
                            'max_depth': 8,
                            'min_child_weight': 10, 
                            'subsample': 0.8260966788901262,
                            'reg_alpha': 0.27469472188551974, 
                            'reg_lambda': 0.5613776857654753, 
                            'colsample_bytree': 0.7965527339281658,
                            'learning_rate': 0.01,
                           })

cat_model = CatBoostRegressor(**{'verbose': 0,
                                 'random_state':  42,
                                 # 'cat_features': cat_features,
                                 # 'early_stopping_rounds': 100,
                                 'eval_metric': "RMSE",
                                 'n_estimators' : 1500,
                                 'objective': 'RMSE', 
                                 'learning_rate': 0.01,
                                 "task_type": "GPU",
                              })

lgb_model = LGBMRegressor(**{'random_state':  42,
                              # 'early_stopping_round': 100,
                              # 'categorical_feature': cat_features,
                              'verbose': -1,
                              'boosting_type': 'gbdt',
                              'n_estimators': 1500,
                              'eval_metric': 'rmse',
                              'objective': 'regression_l2',
                              "device": "gpu",
                              'learning_rate': 0.01,
                              'max_depth': 10,
                              'num_leaves': 928, 
                              'min_child_samples': 8,
                              'min_child_weight': 18, 
                              'colsample_bytree': 0.4009405711855729,
                              'reg_alpha': 0.22713546532680443,
                              'reg_lambda': 0.6266447966186705,
                              })


def select_features_with_rfecv(model, x, y, n_splits=5, scoring='neg_mean_squared_error', random_state=42):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=cv,
        scoring=scoring,
        min_features_to_select=1
    )
    
    rfecv.fit(x, y)
    
    selected_features = x.columns[rfecv.support_].tolist()
    
    return selected_features, rfecv


# Perform feature selection using each model
# catboost_features, catboost_rfecv = select_features_with_rfecv(cat_model, x, y)
# xgboost_features, xgboost_rfecv = select_features_with_rfecv(xgb_model, x, y)
# lightgbm_features, lightgbm_rfecv = select_features_with_rfecv(lgb_model, x,  y)


# catboost_features


from sklearn.model_selection import KFold
import numpy as np

xg_selected_features=['Sex', 'Duration', 'Temp_Heart_Interaction', 'HR_above_group_mean',
       'Age * Heart_Rate', 'Age * Duration', 'Age * Sex', 'Weight * Sex',
       'Height * Sex', 'Heart_Rate * Duration', 'Duration * Sex',
       'Sex * AgeSex']

light_selected_features=['Heart_Rate','Body_Temp','Temp_Heart_Interaction','HR_above_group_mean','Age * Weight','Age * Body_Temp',
                         'Age * Heart_Rate','Age * Duration','Age * Sex','Weight * Heart_Rate','Weight * Sex','Height * Heart_Rate',
                         'Height * Duration','Height * Sex','Body_Temp * Duration','Heart_Rate * Duration','Heart_Rate * Sex',
                         'Duration * Sex','Duration * AgeSex']

cat_selected_features = xg_selected_features + light_selected_features
cat_selected_features = list(dict.fromkeys(cat_selected_features))
# cat_selected_features=['Sex', 'Duration', 'Temp_Heart_Interaction', 'HR_above_group_mean',
#        'Age * Heart_Rate', 'Age * Duration', 'Age * Sex', 'Weight * Sex',
#        'Height * Sex', 'Heart_Rate * Duration', 'Duration * Sex',
#        'Sex * AgeSex']


test_csv=pd.read_csv(test_path)
test_fe= feature_engineering(test_csv)


def train_oof_model(X, y, test_data, selected_features, base_model, n_splits=10):
    X = X[selected_features]
    test_data = test_data[selected_features]
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_data))
    models = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = base_model.__class__(**base_model.get_params())
        model.fit(X_train, y_train)
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(test_data) / n_splits
        models.append(model)
    
    return models, oof_preds, test_preds

xgb_models, xgb_oof, xgb_test = train_oof_model(x, y, test_fe, xg_selected_features, xgb_model)
cat_models, cat_oof, cat_test = train_oof_model(x, y, test_fe, cat_selected_features, cat_model)
lgb_models, lgb_oof, lgb_test = train_oof_model(x, y, test_fe, light_selected_features, lgb_model)




import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import scipy

# Stack OOF predictions from each base model (shape: [n_samples, 3])
X_meta = np.vstack([xgb_oof, cat_oof, lgb_oof]).T

# Stack test predictions (shape: [n_test_samples, 3])
X_meta_test = np.vstack([xgb_test, cat_test, lgb_test]).T


# Stack OOF predictions (train)
X_meta = np.vstack([xgb_oof, cat_oof, lgb_oof]).T

# Stack test predictions
X_meta_test = np.vstack([xgb_test, cat_test, lgb_test]).T

# Train Ridge meta-model
ridge = Ridge(alpha=9.800121553858055, fit_intercept=False,solver='svd')
ridge.fit(X_meta, y)
# Predict on training and test sets
stacked_train_preds = ridge.predict(X_meta)
stacked_test_preds = ridge.predict(X_meta_test)

# Evaluate
rmse = mean_squared_error(y, stacked_train_preds, squared=False)
print("Stacked model RMSE on training:", rmse)


# X_meta_test


final_preds = np.expm1(stacked_test_preds)

submission = pd.DataFrame({"id": test_csv["id"],'Calories': final_preds})
submission.to_csv('submission.csv', index=False)



pd.read_csv("/kaggle/working/submission.csv")

