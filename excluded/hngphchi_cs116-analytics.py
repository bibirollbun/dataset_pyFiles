import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import LabelEncoder


# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.info()


num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']

import matplotlib.pyplot as plt
import seaborn as sns

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], bins=50, kde=True)
    plt.title(f'Phân phối của {col}')
    plt.xlabel(col)
    plt.ylabel('Tần suất')
    plt.tight_layout()
    plt.show()


for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot của {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()



features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

for col in features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=train, x=col, y='Calories', hue='Sex', alpha=0.3)
    plt.title(f'Calories vs {col}')
    plt.tight_layout()
    plt.show()




plt.figure(figsize=(6, 4))
sns.histplot(data=train, x='Calories', hue='Sex', kde=True, bins=50, element='step')
plt.title('Phân phối Calories theo giới tính')
plt.xlabel('Calories')
plt.ylabel('Số lượng')
plt.tight_layout()
plt.show()




plt.figure(figsize=(10, 8))
sns.heatmap(train[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan giữa các đặc trưng')
plt.tight_layout()
plt.show()



X = train.drop(['Calories', 'id'], axis=1)
y = np.log1p(train['Calories'])  # log transform for RMSLE

X_test = test.drop(['id'], axis=1)


# Label Encoding (if any categorical)
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# Prepare arrays for storing predictions
lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))


# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)


from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Load data
X = train.drop(['Calories', 'id'], axis=1)
y = np.log1p(train['Calories'])  # log transform for RMSLE
X_test = test.drop(['id'], axis=1)

# Label Encoding
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])



# Final Prediction: simple average ensemble
final_preds = np.expm1((lgb_preds + cat_preds) / 2)  # reverse log1p
#final_preds = np.expm1(lgb_preds)  # reverse log1p


# Submission
submission['Calories'] = final_preds
submission

