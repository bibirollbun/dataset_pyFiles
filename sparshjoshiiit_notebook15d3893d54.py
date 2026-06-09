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


df=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


df.head(10)


df.shape


df.describe()


for col in df.columns:
    if df[col].isnull().sum()>0:
        print(f'{col} contains null values')
    else:
        continue
print('Done')   #the output below tells that there are no Nan values in any column


import matplotlib.pyplot as plt
value_counts = df['y'].value_counts()
plt.bar(value_counts.index, value_counts.values)
plt.show()


target=df['y']
df=df.drop(columns=['id','y'])





# if we find some value pdays != -1, then from there we can find the data's reference date.However this is not helping in any way..
max_days = -1
max_days_index = -1
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

for index, entry in enumerate(df['pdays']):
    if entry != -1:
        if entry > max_days:
            max_days = entry
            max_days_index = index

if max_days_index != -1:
    print(f"Max pdays found was: {max_days} at row index: {max_days_index}")

    day_of_contact = df['day'][max_days_index]
    month_of_contact_str = df['month'][max_days_index]
    
    print(f"Original contact was on day: {day_of_contact}, month: '{month_of_contact_str}'")
    print("-" * 20)

    present_date = day_of_contact + max_days
    present_month_str = month_of_contact_str

    while True:
        current_month_index = months.index(present_month_str)

        # Your logic for months with 31 days
        if present_month_str in ['jan', 'mar', 'may', 'jul', 'aug', 'oct', 'dec']:
            if present_date > 31:
                present_date -= 31
                # Move to the next month (e.g., from 'jan' to 'feb')
                present_month_str = months[(current_month_index + 1) % 12]
            else:
                break 

        elif present_month_str in ['apr', 'jun', 'sep', 'nov']:
            if present_date > 30:
                present_date -= 30
                present_month_str = months[(current_month_index + 1) % 12]
            else:
                break 

        elif present_month_str == 'feb':
            if present_date > 28:
                present_date -= 28
                present_month_str = months[(current_month_index + 1) % 12]
            else:
                break 
        else:
            break

    print("Calculation Result:")
    print(f"Final Calculated Date: {present_date}")
    print(f"Final Calculated Month: '{present_month_str}'")

else:
    print("No client was previously contacted (all pdays are -1).")


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=40, shuffle=True, random_state=42)


def feature_engineering(df):
    df['was_contacted_before'] = np.where(df['pdays'] == -1, 'no', 'yes')
    
    df['age_bins'] = pd.cut(x=df['age'],
                            bins=[17, 33, 39, 48, 95],
                            labels=['18-33', '34-39', '40-48', '49-95'])
    
    df['campaign_bins'] = pd.cut(x=df['campaign'],
                                 bins=[0, 1, 2, 3, 63],
                                 labels=['1_contact', '2_contacts', '3_contacts', '4+_contacts'])
    
    df['balance_bins'] = pd.cut(x=df['balance'],
                                bins=[-float('inf'), 0, 634, 1390, float('inf')],
                                labels=['Negative_or_Zero', 'Low', 'Medium', 'High'])
    
    df['job_and_marital'] = df['job'] + '_' + df['marital']
    df['education_and_loan'] = df['education'] + '_' + df['loan']
    df['home_and_loan'] = df['housing'] + '_' + df['loan']
    df['marital_and_loan'] = df['marital'] + '_' + df['loan']
    df['marital_and_house'] = df['marital'] + '_' + df['housing']
    df['duration_is_zero'] = np.where(df['duration'] == 0, 'yes', 'no')
    df['duration_per_contact'] = df['duration'] / (df['campaign'] + 1)
    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['duration_x_pdays'] = df['duration'] * df['pdays']
    df['duration_div_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['is_month_end'] = (df['day'] >= 26).astype(int)
    df['is_month_start'] = (df['day'] <= 5).astype(int)
    month_to_quarter = {
        'jan': 1, 'feb': 1, 'mar': 1, 'apr': 2, 'may': 2, 'jun': 2,
        'jul': 3, 'aug': 3, 'sep': 3, 'oct': 4, 'nov': 4, 'dec': 4
    }
    df['quarter'] = df['month'].map(month_to_quarter)
    housing_map = {'yes': 1, 'no': 0, 'unknown': 0}
    loan_map = {'yes': 1, 'no': 0, 'unknown': 0}
    df['debt_pressure_index'] = (
        df['housing'].map(housing_map) + df['loan'].map(loan_map)
    ) / (df['balance'].abs() + 1)
    was_last_successful = (df['poutcome'] == 'success').astype(int)
    df['success_streak_proxy'] = was_last_successful / (df['pdays'] + 1)
    is_entrepreneurial_job = df['job'].isin(['self-employed', 'student', 'unemployed'])
    is_high_balance = df['balance'] > df['balance'].quantile(0.80)
    df['entrepreneurial_spirit'] = (is_entrepreneurial_job & is_high_balance).astype(int)
    is_non_desk_job = df['job'].isin(['blue-collar', 'services', 'housemaid', 'student'])
    df['work_life']=(is_non_desk_job).astype(int)
    
    return df
df=feature_engineering(df)



df.columns





X=df
y=target


import numpy as np
inf_cols = [col for col in X.columns if X[col].isin([np.inf, -np.inf]).any()]

if inf_cols:
    print("Columns with infinite values found:", inf_cols)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)



'''import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
def objective(trial):
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False for compatibility
    ])
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='passthrough')
    params = {
        'objective': 'binary',
        'verbosity': -1,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), 
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), 
    }

    model = lgb.LGBMClassifier(**params)

    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', model)])
    score = cross_val_score(full_pipeline, X, y, cv=skf, scoring='roc_auc').mean()

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
'''


import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,FunctionTransformer
from sklearn.impute import SimpleImputer


num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
cat_cols = [col for col in X.columns if X[col].dtype in ['object', 'category']]
num_transformer_part=SimpleImputer(strategy='mean')
num_transformer = Pipeline(steps=[
    ('imputer', num_transformer_part),
    ('log_transformer', FunctionTransformer(np.log1p, validate=False))  # log1p to handle 0s
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

# Fit and transform the data
X_transformed = preprocessor.fit_transform(X)



import lightgbm as lgb

# The exact parameters from the competitor's solution
lgbm_params={'n_estimators': 2342, 'learning_rate': 0.011408274303382475, 'num_leaves': 215, 'max_depth': -1, 'min_child_samples': 25, 'subsample': 0.8997392998387105, 'colsample_bytree': 0.5776609723229261, 'reg_alpha': 1.545045708114096, 'reg_lambda': 6.663333236094877, 'max_bin': 267}

# Create the LightGBM model instance
lgbm_model = lgb.LGBMClassifier(**lgbm_params)

print("LightGBM model created successfully with the specified parameters.")
print(lgbm_model)



lgbm_model.fit(X_transformed,y)



test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test=feature_engineering(test)
test=test.drop(columns=['id'])


test=preprocessor.transform(test)


y_pred=lgbm_model.predict_proba(test)



print(y_pred)


test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


y_pred_positive_class = y_pred[:, 1]


y_pred1=y_pred.flatten()


print(y_pred1)


submission = pd.DataFrame({
    "id": test['id'],  
    "y": y_pred_positive_class
})
submission.to_csv("/kaggle/working/submission.csv", index=False)


submission=pd.read_csv('/kaggle/input/submit3/submission3.csv')


submission.to_csv('/kaggle/working/submission.csv',index=False)




