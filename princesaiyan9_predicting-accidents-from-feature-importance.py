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


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
print("Reading train,test, submission is Complete")


train.head()


test.head()
#so the accident risk feature must be filled by the model after learning the relationships



#no need to check for null values since in the describe data it has already been made clear
#just for satisfaction we can do it
train.isnull().sum()


#so we have to learn the relationships between the different features here, let's say there is a target variable y=f(X)+ϵ 
# f is the model here to describe f(X) is our target 
#we already know that doing a simple slope -> b0+b1x1+b2x2+b3x3 will give us the behaviour of coefficients
# since the target probability is between 0 and 1 we can use logistic to flatten our function more
# so adding a sigmoid function to the slope will help acheieve it
# P(accident) = 1 / (1 + e^-(b0+b1x1+b2x2+b3x3))

#let us describe the lables again
train.describe()
#we see some categorical labels are not available
train.head(1)


import matplotlib.pyplot as plt
import seaborn as sns

num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

plt.figure(figsize=(15, 4))
for i, col in enumerate(num_cols):
    plt.subplot(1, len(num_cols), i+1)
    sns.histplot(train[col], bins=20, kde=True, color='skyblue')
    plt.title(col)
plt.tight_layout()
plt.show()

cat_cols = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'school_season']

plt.figure(figsize=(18, 8))
for i, col in enumerate(cat_cols):
    plt.subplot(3, 3, i+1)
    sns.countplot(x=col, data=train, palette='pastel')
    plt.xticks(rotation=45)
    plt.title(col)
plt.tight_layout()
plt.show()



#we shall explore relationships now , to see features and accident risk?
plt.figure(figsize=(15, 4))
for i, col in enumerate(num_cols):
    plt.subplot(1, len(num_cols), i+1)
    sns.scatterplot(x=col, y='accident_risk', data=train)
    plt.title(f'{col} vs accident_risk')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
corr = train[num_cols + ['accident_risk']].corr()
sns.heatmap(corr, annot=True, cmap='PiYG')
plt.title('Correlation Matrix')
plt.show()


#describe the training data again
train.head(5)


import pandas as pd
import numpy as np

def feature_engineering(df):
    df = df.copy()

    
    bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    df[bool_cols] = df[bool_cols].astype(int)

    num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df.fillna("unknown", inplace=True)

    
    df['rush_hour'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)

    
    df['log_accidents'] = np.log1p(df['num_reported_accidents'])
    df['log_curvature'] = np.log1p(df['curvature'])
    df['speed_curvature_ratio'] = df['speed_limit'] / (1 + df['curvature'])
    df['road_complexity'] = df['num_lanes'] * df['curvature']
    df['accident_density'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)

    
    cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    
    lighting_cols = [c for c in df.columns if 'lighting_' in c]
    weather_cols = [c for c in df.columns if 'weather_' in c]
    df['visibility_factor'] = df[lighting_cols].sum(axis=1) + df[weather_cols].sum(axis=1)

    return df


train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

# --- Define features and target ---
X = train_fe.drop(columns=['accident_risk', 'id'])
y = train_fe['accident_risk']

# Align test columns with train (important!)
X_test = test_fe.reindex(columns=train_fe.drop(columns=['accident_risk','id']).columns, fill_value=0).values

print("Feature engineering complete.")
print(f"Train shape: {X.shape}, Test shape: {test_fe.shape}")



from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# XGBoost: optimized for GPU histogram, stability, and speed
xgb_model = XGBRegressor(
    n_estimators=500,                # slightly higher for better convergence
    learning_rate=0.03,              # slower LR, more robust
    max_depth=7,                     # deeper trees handle complex data better
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.2,                  # L2 regularization
    reg_alpha=0.3,                   # L1 regularization
    random_state=42,
    tree_method="gpu_hist",          # ✅ latest recommended GPU algorithm
    predictor="gpu_predictor",       # GPU inference
    gpu_id=0,                        # ensures it picks the first GPU
    n_jobs=-1,
    verbosity=0                      # silent mode, replaces verbose param
)

# LightGBM: tuned for GPU and numerical stability
lgb_model = LGBMRegressor(
    n_estimators=600,                # more trees since GPU can handle speed
    learning_rate=0.03,
    num_leaves=63,                   # balanced for accuracy & overfit control
    subsample=0.85,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.2,
    min_child_samples=30,            # avoids overfitting small leaf nodes
    random_state=42,
    device="gpu",                    # ✅ native GPU training
    gpu_platform_id=0,
    gpu_device_id=0,
)



from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

# ensure feature order
feature_cols = train_fe.drop(columns=['accident_risk', 'id']).columns.tolist()

# numpy arrays
X_np = train_fe[feature_cols].to_numpy()
y_np = train_fe['accident_risk'].to_numpy()
X_test_np = test_fe.reindex(columns=feature_cols, fill_value=0).to_numpy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((X_np.shape[0], 2))
test_preds = np.zeros((X_test_np.shape[0], 2))

def fit_model(model, X_train, y_train):
    name = model.__class__.__name__.lower()
    
    if "xgb" in name:
        model.fit(X_train, y_train, verbose=0)     # ✅ fix here
    elif "lgb" in name:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

# training each model
for i, model in enumerate([xgb_model, lgb_model]):
    fold_preds = np.zeros(X_np.shape[0])
    test_fold_preds = np.zeros(X_test_np.shape[0])

    for train_idx, val_idx in kf.split(X_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        fit_model(model, X_train, y_train)          # ✅ use helper with proper verbose
        fold_preds[val_idx] = model.predict(X_val)
        test_fold_preds += model.predict(X_test_np) / kf.n_splits

    oof_preds[:, i] = fold_preds
    test_preds[:, i] = test_fold_preds

# meta-level stacking
meta = Ridge(alpha=1.0)
meta.fit(oof_preds, y_np)
final_preds = meta.predict(test_preds)
final_preds = np.clip(final_preds, 0, 1)


submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': final_preds
})
submission.to_csv('submission.csv', index=False)





