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
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow import keras
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


import seaborn as sns
import matplotlib.pyplot as plt 


dt = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/train.csv')
dt.replace([np.inf, -np.inf], np.nan, inplace=True)
dt_test = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/test.csv')
dt_test.replace([np.inf, -np.inf], np.nan, inplace=True)


print("Shape of train data : ", dt.shape)
print("Shape of test data : ", dt_test.shape)


dt.describe(include="all")


dt.drop_duplicates(inplace=True)
dt_test.drop_duplicates(inplace=True)


print("% of Null values in train data set : ")
print(dt.isnull().sum() / dt.shape[0]*100)
print("% of Null values in test data set : ")
print(dt_test.isnull().sum() / dt_test.shape[0]*100)


label_counts = dt['Y'].value_counts()
plt.figure(figsize=(3, 3))
plt.pie(label_counts, labels=['Label 0 (Non-phishing URL)', 'Label 1 (Phishing URL)'], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Distribution of Labels 0 and 1')
plt.show()


dt.columns


cols = ['x_3','x_5']
dt.drop(columns=cols, inplace=True)
dt_test.drop(columns=cols, inplace=True)


X = dt.drop(['id', 'Y'], axis=1)
Xt = dt_test.drop(['id'], axis=1)
y = dt['Y'].map({True: 1, False: 0})


# Handling Missing Values :- 
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=10, weights='distance')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
Xt_imputed = pd.DataFrame(imputer.fit_transform(Xt), columns=Xt.columns)


sns.kdeplot(dt['x_1'])


X_imputed


def winsorize_outliers(df, lower=0.01, upper=0.99):
    df_win = df.copy()
    for col in df.columns:
        if not col.endswith('_missing'):  
            lower_bound = df[col].quantile(lower)
            upper_bound = df[col].quantile(upper)
            df_win[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df_win

X_treated = winsorize_outliers(X_imputed, lower=0.005, upper=0.995)
Xt_treated = winsorize_outliers(Xt_imputed, lower=0.005, upper=0.995)


X_treated.corrwith(dt['Y'])   ## :) Watching linear correlation


X_treated.corr()


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_treated, y)

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(20).plot(kind='barh')
plt.title("Top 20 Important Features")
plt.show()


## Taking the union of linear and Non_linear coorelation we are droping these columns 
drop_columns = ['x_6','x_7','x_17','x_18','x_20','x_21','x_14']
X_fe = X_treated.drop(columns=drop_columns)
Xt_fe = Xt_treated.drop(columns=drop_columns)


X_fe = X_fe[['x_1', 'x_2','x_4', 'x_8', 'x_15', 'x_16', 'x_19', 'x_9', 'x_10', 'x_12', 'x_13', 'x_11']]
Xt_fe = Xt_fe[['x_1', 'x_2','x_4', 'x_8', 'x_15', 'x_16', 'x_19', 'x_9', 'x_10', 'x_12', 'x_13', 'x_11']]


X_fe.columns  = ['x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','x_9','x_10','x_11','x_12']
Xt_fe.columns  = ['x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','x_9','x_10','x_11','x_12']


X_fe.describe()


X_before_fe = X_fe.copy()
Xt_before_fe = Xt_fe.copy()


## FEATURE CONSTRUCTION :- 
def construct(data) :
    first_7_cols = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5','x_6','x_7']
    all_x_cols = data.columns
    for i in range(1, 8):
        for j in range(i+1, 8):
            data[f'x{i}_x{j}_prod'] = data[f'x_{i}'] * data[f'x_{j}']
            data[f'x{i}_x{j}_sum'] = data[f'x_{i}'] + data[f'x_{j}']
            data[f'x{i}_x{j}_diff'] = data[f'x_{i}'] - data[f'x_{j}']

    for i in [8,9,10,11]:
        for j in [8,9,10,11]:
            if i != j:
                data[f'x{i}_x{j}_ratio'] = data[f'x_{i}'] / (data[f'x_{j}'].abs() + 1e-8)

    for col in first_7_cols:
        data[f'{col}_sq'] = data[col] ** 2
        data[f'{col}_cb'] = data[col] ** 3
        data[f'{col}_sqrt'] = np.sqrt(np.abs(data[col]))
        data[f'{col}_log'] = np.log1p(np.abs(data[col]))

    data['mean_first7'] = data[first_7_cols].mean(axis=1)
    data['std_first7'] = data[first_7_cols].std(axis=1)
    data['max_first7'] = data[first_7_cols].max(axis=1)
    data['min_first7'] = data[first_7_cols].min(axis=1)
    data['range_first7'] = data['max_first7'] - data['min_first7']
    data['median_first7'] = data[first_7_cols].median(axis=1)
    data['skew_first7'] = data[first_7_cols].skew(axis=1)
    data['kurt_first7'] = data[first_7_cols].kurtosis(axis=1)

    large_cols = ['x_8','x_9','x_10','x_11']
    data['sum_large'] = data[large_cols].sum(axis=1)
    data['mean_large'] = data[large_cols].mean(axis=1)
    data['std_large'] = data[large_cols].std(axis=1)
    data['max_large'] = data[large_cols].max(axis=1)
    data['min_large'] = data[large_cols].min(axis=1)

    # All features statistics
    data['mean_all'] = data[all_x_cols].mean(axis=1)
    data['std_all'] = data[all_x_cols].std(axis=1)
    data['max_all'] = data[all_x_cols].max(axis=1)
    data['min_all'] = data[all_x_cols].min(axis=1)

    for col in first_7_cols:
        data[f'{col}_4th'] = data[col] ** 4
        data[f'{col}_inv'] = 1 / (data[col].abs() + 1e-8)
        data[f'{col}_exp'] = np.exp(np.clip(data[col], -10, 10))  # Clip to avoid overflow
        data[f'{col}_sin'] = np.sin(data[col])
        data[f'{col}_cos'] = np.cos(data[col])
        data[f'{col}_tanh'] = np.tanh(data[col])



construct(X_fe)
construct(Xt_fe)


print(f"Created {X_fe.shape[1] - X_before_fe.shape[1]} new features")
print(f"Total features: {X_fe.shape[1]}")


print(f"Created {Xt_fe.shape[1] - Xt_before_fe.shape[1]} new features")
print(f"Total features: {Xt_fe.shape[1]}")


cols = X_fe.columns



from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_fe = pt.fit_transform(X_fe)
Xt_fe = pt.transform(Xt_fe)


X_fe = pd.DataFrame(X_fe, columns=cols)
Xt_fe = pd.DataFrame(Xt_fe, columns=cols)


def selection(data) :
    from sklearn.feature_selection import SelectFromModel

    from catboost import CatBoostClassifier

    selector = CatBoostClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=0
    )
    selector.fit(data, y)

    # Use SelectFromModel to select important features
    sfm = SelectFromModel(selector, threshold='0.75*mean')
    sfm.fit(data, y)
    selected_features = data.columns[sfm.get_support()].tolist()
    return selected_features


selected_features = selection(X_fe)
X_selected = X_fe[selected_features]
Xt_selected = Xt_fe[selected_features]


len(selected_features)


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_selected), 
    columns=selected_features
)
Xt_scaled = pd.DataFrame(
    scaler.transform(Xt_selected), 
    columns=selected_features
)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")


# Model 1 :- 
from catboost import CatBoostClassifier
X_sample = X_train.sample(n=30000, random_state=64)
y_sample = y_train.loc[X_sample.index]
cat_model1 = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.005,
    depth=8,
    l2_leaf_reg=5,
    bagging_temperature=0.5,
    random_strength=0.5,
    eval_metric='AUC',
    random_seed=64,
    verbose=False,
    early_stopping_rounds=100
)

print("\nTraining CatBoost...")
cat_model1.fit(X_sample, y_sample, eval_set=(X_test, y_test), verbose=False)
cat_final1 = cat_model1.predict_proba(Xt_scaled)[:, 1]
cat_proba1 = cat_model1.predict_proba(X_test)[:, 1]
cat_auc = roc_auc_score(y_test, cat_proba1)
print(f"CatBoost ROC AUC: {cat_auc:.6f}")



# Sample a subset of training data (same as before)
X_sample = X_train.sample(n=30000, random_state=321)
y_sample = y_train.loc[X_sample.index]

# Define the Random Forest model
rf_model1 = RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=321,
    n_jobs=-1
)

print("\nTraining Random Forest...")
rf_model1.fit(X_sample, y_sample)

# Predictions
rf_final1 = rf_model1.predict_proba(Xt_scaled)[:, 1]
rf_proba1 = rf_model1.predict_proba(X_test)[:, 1]

# Evaluate AUC
rf_auc = roc_auc_score(y_test, rf_proba1)
print(f"Random Forest ROC AUC: {rf_auc:.6f}")


# Model 2 :-
X_sample = X_train.sample(n=32000, random_state=123)
y_sample = y_train.loc[X_sample.index] 
cat_model2 = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.005,
    depth=8,
    l2_leaf_reg=5,
    bagging_temperature=0.5,
    random_strength=0.5,
    eval_metric='AUC',
    random_seed=123,
    verbose=False,
    early_stopping_rounds=100
)

print("\nTraining CatBoost...")
cat_model2.fit(X_sample, y_sample, eval_set=(X_test, y_test), verbose=False)
cat_final2 = cat_model2.predict_proba(Xt_scaled)[:, 1]
cat_proba2 = cat_model2.predict_proba(X_test)[:, 1]
cat_auc = roc_auc_score(y_test, cat_proba2)
print(f"CatBoost ROC AUC: {cat_auc:.6f}")


# Sample a subset of training data (same as before)
X_sample = X_train.sample(n=32000, random_state=309)
y_sample = y_train.loc[X_sample.index]

# Define the Random Forest model
rf_model2 = RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=309,
    n_jobs=-1
)

print("\nTraining Random Forest...")
rf_model2.fit(X_sample, y_sample)

# Predictions
rf_final2 = rf_model2.predict_proba(Xt_scaled)[:, 1]
rf_proba2 = rf_model2.predict_proba(X_test)[:, 1]

# Evaluate AUC
rf_auc = roc_auc_score(y_test, rf_proba2)
print(f"Random Forest ROC AUC: {rf_auc:.6f}")


x_new = pd.DataFrame({
    'cat1' : cat_proba1,
    'rad1' : rf_proba1,
    'cat2' : cat_proba2,
    'rad2' : rf_proba2
})
x_new


xt_new = pd.DataFrame({
    'cat1' : cat_final1,
    'rad1' : rf_final1,
    'cat2' : cat_final2,
    'rad2' : rf_final2
})
xt_new


X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
    x_new, y_test, test_size=0.4, random_state=321, stratify=y_test
)

print(f"\nTrain size: {X_new_train.shape[0]}, Test size: {X_new_test.shape[0]}")


# Model 4 :- Logistic Regression (No Scaling)
from sklearn.linear_model import LogisticRegression

lr_model4 = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=2000,
    random_state=321,
    class_weight='balanced',    # Handle class imbalance if needed
    verbose=0,
    n_jobs=-1
)

print("\nTraining Logistic Regression...")
lr_model4.fit(X_new_train, y_new_train)
lr_final_new = lr_model4.predict_proba(xt_new)[:, 1]
lr_proba_new = lr_model4.predict_proba(X_new_test)[:, 1]
lr_auc = roc_auc_score(y_new_test, lr_proba_new)
print(f"Logistic Regression ROC AUC: {lr_auc:.6f}")


probabilities = lr_final_new
id = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/test.csv')['id']
submission = pd.DataFrame({
    'id': id,
    'Y': probabilities
})

# Save to CSV - Only 2 columns: id and Y
submission.to_csv('/kaggle/working/team_tricksters.csv', index=False)




