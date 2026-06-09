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


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error


train_csv = pd.read_csv('/kaggle/input/2024-4-big-data-analytics-certification-kr/train.csv')
test_csv = pd.read_csv('/kaggle/input/2024-4-big-data-analytics-certification-kr/test.csv')
sample_csv = pd.read_csv('/kaggle/input/2024-4-big-data-analytics-certification-kr/sample_submission.csv')


train_csv.head()


test_csv.head()


sample_csv.head()


print("\nMissing values in train dataset:")
missing_train = test_csv.isnull().sum().reset_index()
missing_train.columns = ['Feature', 'Missing Count']
display(missing_train[missing_train['Missing Count'] > 0]) if missing_train['Missing Count'].sum() > 0 else print("No missing values found.")


# drop "Id"
test_ids = test_csv['id'].copy()

# save the label
y = train_csv['Age']

# drop the useless columns
train_csv = train_csv.drop(columns=['id', 'Age'])
test_csv  = test_csv.drop(columns=['id'])


train_csv.head()


# preprocessing "Sex" column - One Hot Encoding
categorical_cols = ['Sex']

train_csv = pd.get_dummies(train_csv, columns=categorical_cols, drop_first=True)
test_csv  = pd.get_dummies(test_csv,  columns=categorical_cols, drop_first=True)


# be sure the datasets have the same columns
missing_cols = set(train_csv.columns) - set(test_csv.columns)
for c in missing_cols:
    test_csv[c] = 0
test_csv = test_csv[train_csv.columns]


test_csv.head()


train_csv.head()


def add_engineered_features(df):
    df = df.copy()
    eps = 1e-6  
    
    df['ratio_Weight_Length']  = df['Weight'] / (df['Length'] + eps)
    df['ratio_Shell_Viscera']  = df['Shell Weight'] / (df['Viscera Weight'] + eps)
    df['ratio_Shucked_Weight'] = df['Shucked Weight'] / (df['Weight'] + eps)
    

    df['volume_proxy'] = df['Length'] * df['Diameter'] * df['Height']
    
 
    for col in ['Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']:
        df[f'log_{col}'] = np.log1p(df[col])
    

    df['ratio_Height_Length']   = df['Height'] / (df['Length'] + eps)
    df['ratio_Diameter_Length'] = df['Diameter'] / (df['Length'] + eps)
    df['ratio_Height_Diameter'] = df['Height'] / (df['Diameter'] + eps)
    

    df['Length_sq']   = df['Length'] ** 2
    df['Diameter_sq'] = df['Diameter'] ** 2
    df['Height_sq']   = df['Height'] ** 2
    

    df['Weight_sq']       = df['Weight'] ** 2
    df['ShellWeight_sq']  = df['Shell Weight'] ** 2
    df['VisceraWeight_sq'] = df['Viscera Weight'] ** 2
    
    df['Weight_cu']      = df['Weight'] ** 3
    df['ShellWeight_cu'] = df['Shell Weight'] ** 3
    

    df['total_weight'] = (
        df['Weight'] + df['Shucked Weight'] + df['Viscera Weight'] + df['Shell Weight']
    )
    
 
    df['prop_ShellWeight']  = df['Shell Weight'] / (df['total_weight'] + eps)
    df['prop_VisceraWeight'] = df['Viscera Weight'] / (df['total_weight'] + eps)
    df['prop_ShuckedWeight'] = df['Shucked Weight'] / (df['total_weight'] + eps)
    

    df['diff_Weight_Shucked']   = df['Weight'] - df['Shucked Weight']
    df['diff_Shell_Viscera']    = df['Shell Weight'] - df['Viscera Weight']
    

    df['Length_by_Weight']    = df['Length'] * df['Weight']
    df['Diameter_by_Height']  = df['Diameter'] * df['Height']
    df['Weight_by_ShellWeight'] = df['Weight'] * df['Shell Weight']
    

    df['volume_proxy_sq']   = df['volume_proxy'] ** 2
    df['volume_proxy_cbrt'] = df['volume_proxy'] ** (1/3)
    


    if 'Sex_I' in df.columns and 'Sex_M' in df.columns:
        for col in ['Length', 'Diameter', 'Height', 'Weight', 'Shell Weight']:
            df[f'{col}_by_SexI'] = df[col] * df['Sex_I']
            df[f'{col}_by_SexM'] = df[col] * df['Sex_M']
    

    df['sqrt_Weight']       = np.sqrt(df['Weight'])
    df['cbrt_ShellWeight']  = np.cbrt(df['Shell Weight'])
    

    return df

train_csv = add_engineered_features(train_csv)
test_csv  = add_engineered_features(test_csv)


train_csv.head()


test_csv.head()


# scale numeric features
scaler = StandardScaler()
scaler.fit(train_csv)

X_scaled    = scaler.transform(train_csv) 
test_scaled = scaler.transform(test_csv)   


# split the data
kf = KFold(n_splits=5, shuffle=True, random_state=42)


lgbmr = LGBMRegressor(random_state = 42)

param_distributions = {
    'n_estimators': [100, 300, 500, 800, 1200],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'max_depth': [3, 5, 7, 9, 12],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [5, 10, 20, 40],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0.0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.0, 0.1, 0.5, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=lgbmr,
    param_distributions=param_distributions,
    n_iter=100,           
    scoring='neg_mean_absolute_error',
    cv=kf,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


random_search.fit(X_scaled, y)


best_mae = -random_search.best_score_
best_params = random_search.best_params_
print(f"The best MAE (CV, LGBM): {best_mae:.4f}")
print("The best hyperparameters LGBM:")
for key, val in best_params.items():
    print(f"  {key}: {val}")



# train
best_lgbm = LGBMRegressor(
    random_state=42,
    **best_params
)

best_lgbm.fit(X_scaled, y)


test_preds = best_lgbm.predict(test_scaled)
test_preds = np.clip(test_preds, a_min=0, a_max=None)


submission = pd.DataFrame({
    'id': test_ids,
    'Age': test_preds
})
submission.to_csv('submission.csv', index=False)
submission.head()

