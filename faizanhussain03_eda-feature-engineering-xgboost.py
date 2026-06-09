# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

from scipy.stats import ks_2samp, chi2_contingency

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from xgboost import XGBRegressor
import optuna

# Silence unnecessary warnings for cleaner output
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# Display the first few rows of the training data
train_df.head()


# Split features and target
X_train = train_df.drop(columns=["accident_risk", "id"])
y_train = train_df["accident_risk"]

# Prepare the test set (drop id)
X_test = test_df.drop(columns=["id"])


# Dataset shape
print(f'Training dataset has {train_df.shape[0]} rows and {train_df.shape[1]} columns')
print(f'Test dataset has {test_df.shape[0]} rows and {test_df.shape[1]} columns')

# Check for Null values
null_values=0
null_values+=train_df.isnull().sum().sum()+test_df.isnull().sum().sum()
print(f'Total null values: {null_values}')



train_df.info()


# Columns separation
num_cols = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]
cat_cols = ["road_type", "lighting", "weather", "time_of_day"]
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]

# Target column
target = "accident_risk"


# target Distribution plot
plt.hist(train_df[target],bins=20)
plt.title("Histogram of accident_risk column")
plt.show()

train_df[target].describe()



plt.figure(figsize=(8, 3))
sns.boxplot(x=train_df[target],width=0.4,fliersize=3)

# quantile value
q1 = train_df[target].quantile(0.25)
q2 = train_df[target].quantile(0.50)
q3 = train_df[target].quantile(0.75)

IQR=q3-q1
lbound = q1 - 1.5 * IQR
ubound = q3 + 1.5 * IQR
lb=0
ub=0
for t in train_df[target]:
    if t<lbound:
        lb += 1
    elif t>ubound:
        ub += 1
print(f'No. of outier below lower bound {lb}')
print(f'No. of outier below upper bound {ub}')

plt.title("Accident Risk Distribution", fontsize=13, weight="bold")
plt.xlabel("Accident risk", fontsize=11)
plt.xlim(0, 1)
plt.xticks(np.arange(0, 1.10, 0.10)) 
plt.grid(axis="x", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()


# Histogram of numerical columns

fig, axes = plt.subplots(2, 2, figsize=(8, 4))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(train_df[col], bins=20, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()


num_cols = [col for col in num_cols if col not in ["num_lanes", "speed_limit"]]
cat_cols.extend(["num_lanes", "speed_limit"])


# Correlation map between numerical features and target
corr = train_df[['curvature','num_reported_accidents',target]].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr,annot=True)
plt.title("Correlation Matrix (Numerical Variables + Target)")
plt.show()



fig, axes = plt.subplots(2, 3, figsize=(18,9))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(train_df,x=col,ax=axes[i],palette="Blues")
    axes[i].set_title(col, fontsize=12, weight="bold")
    
plt.tight_layout()

plt.show()
    


fig, axes = plt.subplots(2, 2, figsize=(10,5))
axes = axes.flatten()

for i, col in enumerate(bool_cols):
    sns.countplot(train_df,x=col,ax=axes[i],palette="Blues")
    axes[i].set_title(col, fontsize=12)
plt.tight_layout()

plt.show()


def feature_engineering(df):

    df = df.copy()

    # With Interactions we will create new columns which can be helpful for model building
    df['speed_curvature'] = df['speed_limit']*df['curvature']
    df["curvature_night"] = df["curvature"] * (df["lighting"] == "night").astype(int)
    df['curvature_squared'] = df['curvature'] ** 2
    df['speed_sq'] = df['speed_limit'] ** 2

    # Nonlinear transforms & logs (safe)
    df['accidents_log'] = np.log1p(df['num_reported_accidents'])
    df['curv_log'] = np.log1p(df['curvature'])
    df['speed_log'] = np.log1p(df['speed_limit'])
    df['inv_speed'] = 1.0 / (df['speed_limit'] + 1.0)

    # Ratios / density per lane
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['curv_per_lane'] = df['curvature'] / (df['num_lanes'] + 1)
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)

     # Combined risk indices
    df['danger_score'] = (df['speed_limit'] / 100.0) * (df['curvature'] ** 2)
    df['risk_density'] = df['speed_curvature'] / (df['num_lanes'] + 1.0)
    df['accident_density'] = df['accidents_per_lane'] * df['speed_curvature']

    # Polynomial / smoother mixes
    # Use np.where to protect from negative inside sqrt though curvature and speed_limit are non-negative in domain
    df['poly_mix1'] = np.sqrt(np.maximum(df['curvature'] * df['speed_limit'], 0))
    df['poly_mix2'] = (df['num_reported_accidents'] ** 0.3) * df['speed_limit']

    # Statistical combos
    df['risk_index'] = (df['speed_curvature'] * df['accidents_per_lane']) / (df['speed_limit'] + 1.0)
    df['stability_score'] = (df['num_lanes'] / (1.0 + df['curvature'])) * df['speed_limit']

    # Binary derived flags
    df['tight_lane'] = (df['num_lanes'] <= 2).astype(int)
    df['sharp_curve'] = (df['curvature'] > 0.6).astype(int)
    df['high_speed_zone'] = (df['speed_limit'] > 80).astype(int)
    df['critical_zone'] = ((df['sharp_curve'] == 1) & (df['high_speed_zone'] == 1)).astype(int)

    # Clean up infinite / extremely large values (if any)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df
    


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)


train_df.head()


train_df.info()


train_df[cat_cols], test_df[cat_cols] = train_df[cat_cols].astype("category"), test_df[cat_cols].astype("category")


dtrain = xgb.DMatrix(train_df.drop(columns=['id',target]), label=train_df[target], enable_categorical=True)



xgb_params  = {
    'tree_method': 'hist', 
    'device': 'cuda',
    'eval_metric': 'rmse',
    'random_state': 42,
    'max_bin': 512, 
    'min_child_weight': 3,
    'max_delta_step': 1, 
    'max_depth': 11,
    'learning_rate': 0.010453775390437146,
    'subsample': 0.8162196077561874,
    'colsample_bytree': 0.8057453252225478,
    'gamma': 0.011515371568909936,
    'reg_alpha': 0.1153674139991063,
    'reg_lambda': 0.4029264986439234,
    'colsample_bylevel': 0.8675078626084138,
    'colsample_bynode': 0.8804930677965951,
    'scale_pos_weight': 0.3615894752587659,

    
    
    

}

# Run cross-validation
cv_results = xgb.cv(
    params=xgb_params,
    dtrain=dtrain,
    nfold=5,
    num_boost_round=2000,
    metrics='rmse',
    verbose_eval=100,
    early_stopping_rounds=50
)

# Display last few CV results
print(cv_results.tail())

# Extract best boosting round
best_round = cv_results['test-rmse-mean'].idxmin()
best_rmse = cv_results['test-rmse-mean'][best_round]
print(f"Best round: {best_round}, Best CV RMSE: {best_rmse:.7f}")


last_round = len(cv_results) - 1
xgb_params["n_estimators"] = last_round + 10


X_train = train_df.drop(columns=['id',target])
y_train = train_df[target]
model= XGBRegressor(**xgb_params,enable_categorical=True)
model.fit(X_train,y_train)

pred=model.predict(test_df.drop(columns='id'))





sub = pd.DataFrame({
    "id": test_df["id"],
    target: pred
})
sub.to_csv("submission.csv", index=False)







