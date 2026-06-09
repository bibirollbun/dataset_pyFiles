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


train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


import warnings
warnings.filterwarnings('ignore')



train.head()


train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})
train = train.drop_duplicates(subset=train.columns).reset_index(drop=True)
train = train.groupby(['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])['Calories'].min().reset_index()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
# Distribution of Heart Rate
plt.figure(figsize=(10, 6))
sns.histplot(train['Heart_Rate'], kde=True, color='skyblue')
plt.title('Distribution of Heart Rate')
plt.xlabel('Heart Rate')
plt.ylabel('Frequency')
plt.show()



# Relationship between Heart Rate and Duration
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Calories',y='Duration', data=train, color='orange')
plt.title('Heart Rate vs Duration')
plt.xlabel('Heart Rate')
plt.ylabel('Duration (minutes)')
plt.show()



# Relationship between Heart Rate and Calories
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Heart_Rate', y='Calories', data=train, color='green')
plt.title('Heart Rate vs Calories Burned')
plt.xlabel('Heart Rate')
plt.ylabel('Calories Burned')
plt.show()



train['BMR'] = np.where(
    train['Sex'] == 'male',
    88.362 + (13.397 * train['Weight']) + (4.799 * train['Height']) - (5.677 * train['Age']),
    447.593 + (9.247 * train['Weight']) + (3.098 * train['Height']) - (4.330 * train['Age'])
)



test['BMR'] = np.where(
    test['Sex'] == 'male',
    88.362 + (13.397 * test['Weight']) + (4.799 * test['Height']) - (5.677 * test['Age']),
    447.593 + (9.247 * test['Weight']) + (3.098 * test['Height']) - (4.330 * test['Age'])
)



train['Predicted_Calories_Burned'] = train['BMR'] * (train['Heart_Rate'] * train['Duration']) / 1000



test['Predicted_Calories_Burned'] = test['BMR'] * (test['Heart_Rate'] * test['Duration']) / 1000



def add_features(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']

    df['Sex_Reversed'] = 1 - df['Sex']
    for dur in df['Duration'].unique():
        df[f'HR_Dur_{int(dur)}'] = np.where(df['Duration'] == dur, df['Heart_Rate'], 0)
        df[f'Temp_Dur_{int(dur)}'] = np.where(df['Duration'] == dur, df['Body_Temp'], 0)
    for age in df['Age'].unique():
        df[f'HR_Age_{int(age)}'] = np.where(df['Age'] == age, df['Heart_Rate'], 0)
        df[f'Temp_Age_{int(age)}'] = np.where(df['Age'] == age, df['Body_Temp'], 0)

    for f1 in ['Duration', 'Heart_Rate', 'Body_Temp']:
        for f2 in ['Sex', 'Sex_Reversed']:
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]

    for col in ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']:
        for agg in ['min', 'max']:
            agg_val = train.groupby('Sex')[col].agg(agg).rename(f'Sex_{col}_{agg}')
            df = df.merge(agg_val, on='Sex', how='left')

    df.drop(columns=['Sex_Reversed'], inplace=True)
    return df

train = add_features(train)
test = add_features(test)


# Combining Body Temperature with Heart Rate and Duration
train['Body_Temperature_Interaction'] = train['Body_Temp'] * train['Heart_Rate'] * train['Duration']
test['Body_Temperature_Interaction'] = test['Body_Temp'] * test['Heart_Rate'] * test['Duration']



import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb

# Load your datasets
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# =========
# Data Preparation
# =========
excluded_cols = ['Calories', 'ID'] if 'ID' in train.columns else ['Calories']
features = [col for col in train.columns if col in test.columns and col not in excluded_cols]

X = train[features].copy()
y = np.log1p(train['Calories'])  # log1p transform
X_test = test[features].copy()

# LightGBM-specific data (category encoding)
X_lgb, X_test_lgb = X.copy(), X_test.copy()
for col in X_lgb.select_dtypes(include='object'):
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# XGBoost-specific label encoding
def label_encode(train_df, val_df, test_df):
    for col in train_df.select_dtypes(include='object'):
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        map_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        val_df[col] = val_df[col].map(map_dict).fillna(-1).astype(int)
        test_df[col] = test_df[col].map(map_dict).fillna(-1).astype(int)
    return train_df, val_df, test_df

# =========
# Base Models Cross-validation
# =========
NUM_MODELS = 9
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], NUM_MODELS))
test_preds = np.zeros((X_test.shape[0], NUM_MODELS))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 1. LightGBM
    lgb_train, lgb_val = X_train.copy(), X_val.copy()
    for col in lgb_train.select_dtypes(include='object'):
        lgb_train[col] = lgb_train[col].astype('category')
        lgb_val[col] = lgb_val[col].astype('category')
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=10, colsample_bytree=0.7,
        subsample=0.9, random_state=42, verbose=-1
    )
    lgb_model.fit(
        lgb_train, y_train,
        eval_set=[(lgb_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    oof_preds[val_idx, 0] = lgb_model.predict(lgb_val)
    test_preds[:, 0] += lgb_model.predict(X_test_lgb) / kf.n_splits

    # 2. XGBoost
    xgb_train, xgb_val, xgb_test = X_train.copy(), X_val.copy(), X_test.copy()
    xgb_train, xgb_val, xgb_test = label_encode(xgb_train, xgb_val, xgb_test)

    xgb_model = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=10, colsample_bytree=0.7,
        subsample=0.9, gamma=0.01, max_delta_step=2, enable_categorical=True,
        eval_metric='rmse', random_state=42
    )
    xgb_model.fit(xgb_train, y_train, eval_set=[(xgb_val, y_val)],
                  early_stopping_rounds=100, verbose=False)
    oof_preds[val_idx, 1] = xgb_model.predict(xgb_val)
    test_preds[:, 1] += xgb_model.predict(xgb_test) / kf.n_splits

    # 3. Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    oof_preds[val_idx, 2] = ridge.predict(X_val)
    test_preds[:, 2] += ridge.predict(X_test) / kf.n_splits

    # 4. PCA + Ridge
    pca_ridge = make_pipeline(
        StandardScaler(), PCA(n_components=0.95), Ridge(alpha=1.0)
    )
    pca_ridge.fit(X_train, y_train)
    oof_preds[val_idx, 3] = pca_ridge.predict(X_val)
    test_preds[:, 3] += pca_ridge.predict(X_test) / kf.n_splits

    # 5. Lasso
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.1))
    lasso.fit(X_train, y_train)
    oof_preds[val_idx, 4] = lasso.predict(X_val)
    test_preds[:, 4] += lasso.predict(X_test) / kf.n_splits

    # 6. ElasticNet
    enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5))
    enet.fit(X_train, y_train)
    oof_preds[val_idx, 5] = enet.predict(X_val)
    test_preds[:, 5] += enet.predict(X_test) / kf.n_splits

    # 7. ExtraTrees
    et = ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=42)
    et.fit(X_train, y_train)
    oof_preds[val_idx, 6] = et.predict(X_val)
    test_preds[:, 6] += et.predict(X_test) / kf.n_splits

    # 8. KNN
    knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=8))
    knn.fit(X_train, y_train)
    oof_preds[val_idx, 7] = knn.predict(X_val)
    test_preds[:, 7] += knn.predict(X_test) / kf.n_splits

    # 9. SVR
    svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1))
    svr.fit(X_train, y_train)
    oof_preds[val_idx, 8] = svr.predict(X_val)
    test_preds[:, 8] += svr.predict(X_test) / kf.n_splits

# =========
# Meta-Model (Ridge)
# =========
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
meta_model = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1)
meta_model.fit(oof_preds, y)

print("Best alpha for Ridge meta-model:", meta_model.best_params_)

# Evaluate
oof_meta_preds = meta_model.predict(oof_preds)
rmse = mean_squared_error(y, oof_meta_preds, squared=False)
print(f"Stacked OOF RMSE: {rmse:.5f}")

# Final Predictions
final_test_log_preds = meta_model.predict(test_preds)
final_test_preds = np.expm1(final_test_log_preds)



import pandas as pd

# If 'ID' is in X_test
submission = pd.DataFrame({
    'id': sub['id'],  # or X_test['ID'] if it's a column
    'Calories': final_test_preds     # replace 'target' with the actual target column name
})



submission.to_csv("submission.csv", index=False)














