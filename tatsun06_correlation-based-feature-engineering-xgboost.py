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
df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')



df_train


df_train['Sex']=df_train['Sex'].map({'male':0,'female':1})
df_train.drop(columns='id',inplace=True)


import pandas as pd

#Extract only columns of numeric type
numeric_cols = df_train.select_dtypes(include=['number']).columns.tolist()
# Exclude the objective variable from the numeric type column
if 'Calories' in numeric_cols:
    numeric_cols.remove('Calories')

# Calculate the correlation between the original column and Calories
original_correlations = {}
for col in numeric_cols:
    original_correlations[col] = abs(df_train[col].corr(df_train['Calories']))

correlations = {}

# Perform operations on each numerical sequence pair
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col1 = df_train[numeric_cols[i]]
        col2 = df_train[numeric_cols[j]]
        name1 = numeric_cols[i]
        name2 = numeric_cols[j]

        corr_original_1 = original_correlations.get(name1, 0)
        corr_original_2 = original_correlations.get(name2, 0)
        # mul
        new_col_mul = col1 * col2
        corr_mul = new_col_mul.corr(df_train['Calories'])
        if abs(corr_mul) > corr_original_1 and abs(corr_mul) > corr_original_2 and abs(corr_mul) >= 0.3:
            correlations[f'{name1}*{name2}'] = corr_mul
        df_train[f'{name1}*{name2}'] = df_train[name1]*df_train[name2]

        # # div (Add a small value to avoid division by zero)
        # epsilon = 1e-7
        # new_col_div1 = col1 / (col2 + epsilon)
        # corr_div1 = new_col_div1.corr(df_train['Calories'])
        # if abs(corr_div1) > corr_original_1 and abs(corr_div1) > corr_original_2 and abs(corr_div1) >= 0.3:
        #     correlations[f'{name1}/{name2}'] = corr_div1
        #     df_train[f'{name1}/{name2}'] = df_train[name1]/df_train[name2]

# Display sorted by absolute value of correlation
sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)

for k, v in sorted_correlations:
    print(f'Correlation with Calories for {k}: {v:.4f}')


df_train_copy = df_train.copy()
df_train_copy['BMI'] = df_train_copy['Weight']/(df_train_copy['Height'] ** 2)
df_train_copy['BSA'] = 0.007184 * (df_train_copy['Weight']**0.425)*(df_train_copy['Height']**0.725)
df_train_copy['RelativeIntenstiy'] = df_train_copy['Heart_Rate']/(220-df_train_copy['Age'])*100
def calculate_lbm(row):
    weight_kg = row['Weight']
    height_cm = row['Height']
    sex = row['Sex']

    if sex == 0:  # man
        lbm = 0.32810 * weight_kg + 0.33929 * height_cm - 29.5336
    else:  # woman
        lbm = 0.29569 * weight_kg + 0.41813 * height_cm - 43.2933
    return lbm
df_train_copy['LBM'] = df_train_copy.apply(calculate_lbm, axis=1)
def calculate_lbm(row):
    weight_kg = row['Weight']
    height_cm = row['Height']
    sex = row['Sex']
    age = row['Age']

    if sex == 0:  # man
        num = 13.397 * weight_kg + 4.799 * height_cm  -5.677 * age +88.362
    else:  # woman
        num = 9.247 * weight_kg + 3.098 * height_cm - 4.33*age + 447.593
    return num
df_train_copy['BMR'] = df_train_copy.apply(calculate_lbm, axis=1)
new_cols = ['BMI','BSA','LBM','BMR','RelativeIntenstiy']
for col in new_cols:
    print(f"{col}:{df_train_copy[col].corr(df_train_copy['Calories'])}")


# df_train['BMI'] = df_train['Weight']/(df_train['Height'] ** 2)
# df_train['BSA'] = 0.007184 * (df_train['Weight']**0.425)*(df_train['Height']**0.725)
df_train['RelativeIntenstiy'] = df_train['Heart_Rate']/(220-df_train['Age'])*100
# def calculate_lbm(row):
#     weight_kg = row['Weight']
#     height_cm = row['Height']
#     sex = row['Sex']

#     if sex == 0:  # man
#         lbm = 0.32810 * weight_kg + 0.33929 * height_cm - 29.5336
#     else:  # woman
#         lbm = 0.29569 * weight_kg + 0.41813 * height_cm - 43.2933
#     return lbm
# df_train['LBM'] = df_train.apply(calculate_lbm, axis=1)
# def calculate_lbm(row):
#     weight_kg = row['Weight']
#     height_cm = row['Height']
#     sex = row['Sex']
#     age = row['Age']

#     if sex == 0:  # man
#         num = 13.397 * weight_kg + 4.799 * height_cm  -5.677 * age +88.362
#     else:  # woman
#         num = 9.247 * weight_kg + 3.098 * height_cm - 4.33*age + 447.593
#     return num
# df_train['BMR'] = df_train.apply(calculate_lbm, axis=1)


import pandas as pd

numeric_cols = df_train.select_dtypes(include=['number']).columns.tolist()

if 'Calories' in numeric_cols:
    numeric_cols.remove('Calories')


original_correlations = {}
for col in numeric_cols:
    original_correlations[col] = abs(df_train[col].corr(df_train['Calories']))

correlations = {}

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col1 = df_train[numeric_cols[i]]
        col2 = df_train[numeric_cols[j]]
        name1 = numeric_cols[i]
        name2 = numeric_cols[j]

        corr_original_1 = original_correlations.get(name1, 0)
        corr_original_2 = original_correlations.get(name2, 0)
        # mul
        new_col_mul = col1 * col2
        corr_mul = new_col_mul.corr(df_train['Calories'])
        if abs(corr_mul) > corr_original_1 and abs(corr_mul) > corr_original_2 and abs(corr_mul) >= 0.3:
            correlations[f'{name1}*{name2}'] = corr_mul

        # # div
        # epsilon = 1e-7
        # new_col_div1 = col1 / (col2 + epsilon)
        # corr_div1 = new_col_div1.corr(df_train['Calories'])
        # if abs(corr_div1) > corr_original_1 and abs(corr_div1) > corr_original_2:
        #     correlations[f'{name1}/{name2}'] = corr_div1
        #     df_train[f'{name1}/{name2}'] = df_train[name1]/df_train[name2]


sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)

for k, v in sorted_correlations:
    print(f'Correlation with Calories for {k}: {v:.4f}')


df_train['Duration*Heart_Rate*Heart_Rate*RelativeIntenstiy'] = df_train['Duration'] * (df_train['Heart_Rate'] ** 2) * df_train['RelativeIntenstiy']


df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e5/test.csv')
df_test['Sex']=df_test['Sex'].map({'male':0,'female':1})
df_test.drop(columns='id',inplace=True)

import pandas as pd

numeric_cols = df_test.select_dtypes(include=['number']).columns.tolist()

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col1 = df_test[numeric_cols[i]]
        col2 = df_test[numeric_cols[j]]
        name1 = numeric_cols[i]
        name2 = numeric_cols[j]

        corr_original_1 = original_correlations.get(name1, 0)
        corr_original_2 = original_correlations.get(name2, 0)
        # mul
        new_col_mul = col1 * col2
        df_test[f'{name1}*{name2}'] = df_test[name1]*df_test[name2]



# df_test['BMI'] = df_test['Weight']/(df_test['Height'] ** 2)
# df_test['BSA'] = 0.007184 * (df_test['Weight']**0.425)*(df_test['Height']**0.725)
df_test['RelativeIntenstiy'] = df_test['Heart_Rate']/(220-df_test['Age'])*100
# df_test['LBM'] = df_test.apply(calculate_lbm, axis=1)
# df_test['BMR'] = df_test.apply(calculate_lbm, axis=1)

df_test['Duration*Heart_Rate*Heart_Rate*RelativeIntenstiy'] = df_test['Duration'] * (df_test['Heart_Rate'] ** 2) * df_test['RelativeIntenstiy']




from sklearn.model_selection import train_test_split
import numpy as np
X = df_train.drop('Calories', axis=1)
y = np.log1p(df_train['Calories'])


import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

FOLDS = 10

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

X_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(df_train))
pred = np.zeros(len(df_test))

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': 2
}

for i, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")

    x_train = X_scaled.iloc[train_idx]
    y_train = y.iloc[train_idx]
    x_valid = X_scaled.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]
    x_test = X_test_scaled

    start = time.time()
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    dtest = xgb.DMatrix(x_test)
    
    model = xgb.train(xgb_params, dtrain,
                      num_boost_round=1000,
                      evals=[(dvalid, 'valid')],
                      verbose_eval=100,
                      early_stopping_rounds=100) # early stopping
    
    oof[valid_idx] = model.predict(dvalid)
    pred += model.predict(dtest) / FOLDS

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"time: {time.time() - start:.1f} second")

full_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nCV Total RMSE: {full_rmse:.4f}")


pred_scale = np.expm1(pred)
submission = pd.DataFrame({'id': df_test.index + 750000, 'Calories': pred_scale})
submission.to_csv("submission.csv", index=False)
submission.head()

