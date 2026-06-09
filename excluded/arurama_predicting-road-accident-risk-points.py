import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor#
warnings.filterwarnings('ignore', category=FutureWarning)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#help(pd.read_csv)
df_submit = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df_train.head()


df_train.info()
df_train.isnull().sum().sum()


df_train.head()


df_train["accident_risk"].hist()          # histogram
plt.show()
#df_train.value_counts().plot(kind="bar")
#plt.show()


plt.figure(figsize=(12, 12))
sns.heatmap(df_train.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()


num_df = df_train.select_dtypes(include=[np.number])
corr_matrix = num_df.corr(method='pearson')  # default
target_corr = corr_matrix["accident_risk"].sort_values(ascending=False)
print(target_corr)


target_corr = corr_matrix["accident_risk"].sort_values(ascending=False)
print(target_corr)



# Constants
g = 9.81  # gravity in m/s²
kmh_to_ms = 0.2778  # conversion factor

# Add new column with formula: V² * 0.2778 * K / g
df_train['centri_factor'] = (df_train['speed_limit']**2) * kmh_to_ms * df_train['curvature'] / g

# Add new column with formula: V² * 0.2778 * K / g
df_test['centri_factor'] = (df_test['speed_limit']**2) * kmh_to_ms * df_test['curvature'] / g


df_train_xgb = xgb.DMatrix(df_train.drop(columns=['accident_risk','road_type','lighting','weather','time_of_day']),
                     label=df_train["accident_risk"],
                     enable_categorical=True)



# XGBoost Cross-Validation for Road Accident Risk Prediction

## XGBoost Parameters

xgb_params = {
    'tree_method': 'hist',                # Histogram-based tree method
    'device': 'cuda',                     # GPU acceleration
    'eval_metric': 'rmse',                # Evaluation metric (regression)
    'random_state': 42,                   # Reproducibility
    'max_bin': 512,                       # Maximum number of bins for histogram
    'min_child_weight': 3,                # Minimum sum of instance weight in a child
    'max_delta_step': 1,                   # Max delta step
    'max_depth': 11,                       # Maximum tree depth
    'learning_rate': 0.010453775390437146,# Learning rate (eta)
    'subsample': 0.8162196077561874,      # Row sampling
    'colsample_bytree': 0.8057453252225478,# Column sampling per tree
    'gamma': 0.011515371568909936,        # Minimum loss reduction to make a split
    'reg_alpha': 0.1153674139991063,      # L1 regularization
    'reg_lambda': 0.4029264986439234,     # L2 regularization
    'colsample_bylevel': 0.8675078626084138,# Column sampling per level
    'colsample_bynode': 0.8804930677965951,# Column sampling per node
    'scale_pos_weight': 0.3615894752587659,# Balance positive/negative weight
}

# Run cross-validation
cv_results = xgb.cv(
    params=xgb_params,
    dtrain=df_train_xgb,
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


X_train = df_train.drop(columns=['accident_risk','road_type','lighting','weather','time_of_day'])
X_train.head()




y_train = df_train['accident_risk']
y_train.head()



X_train = df_train.drop(columns=['accident_risk','road_type','lighting','weather','time_of_day'])
y_train = df_train['accident_risk']
model= XGBRegressor(**xgb_params,enable_categorical=True)
model.fit(X_train,y_train)
xgb_df_test = df_test.drop(columns=['road_type','lighting','weather','time_of_day'])
pred=model.predict(xgb_df_test)


sub = pd.DataFrame({
    "id": xgb_df_test["id"],
    'accident_risk': pred
})
sub.to_csv("submission.csv", index=False)

