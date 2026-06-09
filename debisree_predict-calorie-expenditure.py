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


# import libraries:

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import shap
from shap import  summary_plot

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import LabelEncoder

import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from tqdm import tqdm 

from sklearn.metrics import mean_squared_error, r2_score,  mean_squared_log_error

from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.simplefilter("ignore")
pd.options.mode.chained_assignment = None  


# Import Data

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print(train.shape)
print(test.shape)


train.head()


#missing value?

train.isnull().sum()


test.isnull().sum()


# set id as index:

train.set_index('id', inplace= True)
test.set_index('id', inplace= True)

train.head()


# Target:

sns.histplot(train['Calories'], kde= True)
plt.show()


sns.histplot(data=train, x='Calories', kde=True, hue='Sex')
plt.show()


train['log_Calories'] = np.log1p(train['Calories'])


sns.histplot(train['log_Calories'], kde= True)
plt.show()

sns.histplot(data=train, x='log_Calories', kde=True, hue='Sex')
plt.show()


sns.countplot(data= train, x= 'Sex')
plt.show()


# Age:

sns.histplot(train['Age'], kde= True)
plt.show()

sns.histplot(data=train, x='Age', kde=True, hue='Sex')
plt.show()


sns.histplot(train['Height'], kde= True)
plt.show()

sns.histplot(data=train, x='Height', kde=True, hue='Sex')
plt.show()



sns.histplot(train['Weight'], kde= True)
plt.show()

sns.histplot(data=train, x='Weight', kde=True, hue='Sex')
plt.show()


sns.histplot(train['Duration'], kde= True)
plt.show()

sns.histplot(data=train, x='Duration', kde=True, hue='Sex')
plt.show()


sns.histplot(train['Heart_Rate'], kde= True)
plt.show()

sns.histplot(data=train, x='Heart_Rate', kde=True, hue='Sex')
plt.show()


sns.histplot(train['Body_Temp'], kde= True)
plt.show()

sns.histplot(data=train, x='Body_Temp', kde=True, hue='Sex')
plt.show()


# Feature Creation - BMI

#convert Height in m:

train['Height'] = train['Height']/100
train['bmi'] = train['Weight']/ (train['Height']**2.0)

test['Height'] = test['Height']/100
test['bmi'] = test['Weight']/ (test['Height']**2.0)


train.head()


train['hr_dur'] = train['Heart_Rate'] * train['Duration']
train['age_hr_ratio'] = train['Heart_Rate'] / train['Age']
train['bmi_dur'] = train['bmi'] * train['Duration']
train['weight_dur'] = train['Weight'] * train['Duration']
train['temp_hr'] = train['Body_Temp'] * train['Heart_Rate']


test['hr_dur'] = test['Heart_Rate'] * test['Duration']
test['age_hr_ratio'] = test['Heart_Rate'] / test['Age']
test['bmi_dur'] = test['bmi'] * test['Duration']
test['weight_dur'] = test['Weight'] * test['Duration']
test['temp_hr'] = test['Body_Temp'] * test['Heart_Rate']


sns.histplot(train['bmi'], kde= True)
plt.show()

sns.histplot(data=train, x='bmi', kde=True, hue='Sex')
plt.show()


sns.histplot(train['hr_dur'], kde= True)
plt.show()

sns.histplot(data=train, x='hr_dur', kde=True, hue='Sex')
plt.show()


#BMI Ranks

train['bmi_rank'] = pd.qcut(train['bmi'], q=3, labels=['low', 'medium', 'high'])
test['bmi_rank'] = pd.qcut(test['bmi'], q=3, labels=['low', 'medium', 'high'])

train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')

train['bmi_rank'] = train['bmi_rank'].astype('category')
test['bmi_rank'] = test['bmi_rank'].astype('category')

# Create interaction feature


train['sex_bmi_rank'] = train['Sex'].astype(str) + '_' + train['bmi_rank'].astype(str)
test['sex_bmi_rank'] = test['Sex'].astype(str) + '_' + test['bmi_rank'].astype(str)



#plot:

plt.figure(figsize=(10,5))
sns.boxplot(x= 'sex_bmi_rank', y= 'Calories', data= train)
plt.show()


# sns.histplot(train['age_hr_ratio'], kde= True)
# plt.show()

# sns.histplot(data=train, x='age_hr_ratio', kde=True, hue='Sex')
# plt.show()


# sns.histplot(train['bmi_dur'], kde= True)
# plt.show()

# sns.histplot(data=train, x='bmi_dur', kde=True, hue='Sex')
# plt.show()


# sns.histplot(train['weight_dur'], kde= True)
# plt.show()

# sns.histplot(data=train, x='weight_dur', kde=True, hue='Sex')
# plt.show()


# sns.histplot(train['temp_hr'], kde= True)
# plt.show()

# sns.histplot(data=train, x='temp_hr', kde=True, hue='Sex')
# plt.show()


# Relation with Calories:



# def plot_numerical_vs_calorie(df, target='Calories'):
#     """Plot scatterplots with regression lines for each numerical feature vs. Calorie."""
    
#     # Select numerical features excluding the target
#     numerical_cols = df.select_dtypes(include='number').columns
#     feature_cols = [col for col in numerical_cols if col != target]

#     for col in feature_cols:
#         plt.figure(figsize=(8, 5))
#         sns.regplot(x=col, y=target, data=df, scatter_kws={'alpha':0.5})
#         plt.title(f'{col} vs {target}')
#         plt.xlabel(col)
#         plt.ylabel(target)
#         plt.tight_layout()
#         plt.show()
        
#     return 


# plot_numerical_vs_calorie(train)



train = pd.get_dummies(train, columns = ['Sex'], drop_first =False)
test = pd.get_dummies(test, columns = ['Sex'], drop_first =False)


# Keep Sex column as a categorical one:
# train['Sex'] = train['Sex'].astype('category')
# test['Sex'] = test['Sex'].astype('category')


train['sex_bmi_rank'] = train['sex_bmi_rank'].astype('category')
test['sex_bmi_rank'] = test['sex_bmi_rank'].astype('category')


# train.drop(['Height', 'bmi', 'weight_dur', 'Weight'], axis =1, inplace=True)
# test.drop(['Height', 'bmi', 'weight_dur', 'Weight'], axis =1, inplace=True)

train.drop(['bmi', 'bmi_rank'], axis =1, inplace=True)
test.drop(['bmi', 'bmi_rank'], axis =1, inplace=True)


# Linear correlation:



#  Compute correlation matrix
# corr_matrix = train_encoded.corr()

# # Plot heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Matrix - Features with the target")
# plt.show()


# def detect_and_cap_outliers_safe(df, method='iqr', lower_quantile=0.01, upper_quantile=0.99, verbose=True, zero_floor_columns=None):
#     """
#     Detects and caps outliers, with optional flooring at zero for specific columns (like heartrate, body_temp).
#     """
#     df_capped = df.copy()
#     numeric_cols = df.select_dtypes(include='number').columns

#     if zero_floor_columns is None:
#         zero_floor_columns = []

#     for col in numeric_cols:
#         if method == 'iqr':
#             Q1 = df[col].quantile(0.25)
#             Q3 = df[col].quantile(0.75)
#             IQR = Q3 - Q1
#             lower = Q1 - 1.5 * IQR
#             upper = Q3 + 1.5 * IQR
#         elif method == 'quantile':
#             lower = df[col].quantile(lower_quantile)
#             upper = df[col].quantile(upper_quantile)
#         else:
#             raise ValueError("method must be 'iqr' or 'quantile'")

#         # Apply zero floor for features that must not go negative
#         if col in zero_floor_columns and lower < 0:
#             lower = 0

#         outliers = ((df[col] < lower) | (df[col] > upper)).sum()
#         if verbose:
    #         print(f"{col}: {outliers} outliers capped (bounds: {lower:.2f} to {upper:.2f})")

    #     df_capped[col] = df[col].clip(lower=lower, upper=upper)

    # return df_capped



# train_capped = detect_and_cap_outliers_safe(
#     train_encoded,
#     method='iqr',
#     zero_floor_columns=['Age', 'Duration', 'Calories', 'Heart_Rate', 'Weight', 'Height', 'bmi']
# )



# test_capped = detect_and_cap_outliers_safe(
#     test_encoded,
#     method='iqr',
#     zero_floor_columns=['Age', 'Duration',  'Heart_Rate', 'Weight', 'Height', 'bmi']
# )


# train-test split

target = "Calories"
X = train.drop(['Calories', 'log_Calories'], axis=1)


y = train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import numpy as np
import matplotlib.pyplot as plt

# === Step 0: Prepare log-transformed target ===
train['log_Calories'] = np.log1p(train['Calories'])  # log(1 + x) to avoid log(0)

X = train.drop(['Calories', 'log_Calories'], axis=1)
y = train['log_Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure categorical columns (if any) are of correct type
if 'Sex' in X_train.columns:
    X_train['Sex'] = X_train['Sex'].astype('category')
    X_test['Sex'] = X_test['Sex'].astype('category')

# === Step 1: Cross-validation to find best iterations ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"--- Fold {fold} ---")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = XGBRegressor(
        n_estimators=4000,
        learning_rate=0.005,
        max_depth=4,
        min_child_weight=10,
        colsample_bytree=0.6,
        subsample=0.6,
        gamma=1.0,
        reg_alpha=5.0,
        reg_lambda=10.0,
        eval_metric="rmse",
        early_stopping_rounds=50,
        enable_categorical=True,
        tree_method="hist",
        verbosity=0,
        random_state=42
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    best_iter = model.best_iteration or model.n_estimators
    best_iterations.append(best_iter)
    print(f"Best iteration = {best_iter}")

# === Step 2: Train final model with avg best iteration ===
mean_best_iter = int(np.mean(best_iterations))
print(f"\nMean Best Iteration from CV: {mean_best_iter}")

final_model = XGBRegressor(
    n_estimators=mean_best_iter,
    learning_rate=0.005,
    max_depth=4,
    min_child_weight=10,
    colsample_bytree=0.6,
    subsample=0.6,
    gamma=1.0,
    reg_alpha=5.0,
    reg_lambda=10.0,
    eval_metric="rmse",
    enable_categorical=True,
    tree_method="hist",
    verbosity=0,
    random_state=42
)

final_model.fit(X_train, y_train)

# === Step 3: Predict and inverse-transform ===
y_train_pred_log = final_model.predict(X_train)
y_test_pred_log = final_model.predict(X_test)

y_train_pred = np.expm1(y_train_pred_log)
y_test_pred = np.expm1(y_test_pred_log)

y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)

# === Step 4: Evaluation on original scale ===
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))

train_rmsle = np.sqrt(mean_squared_log_error(y_train_actual, y_train_pred))
test_rmsle = np.sqrt(mean_squared_log_error(y_test_actual, y_test_pred))

print(f"\nFinal Evaluation (Original Scale):")
print(f"Train RMSE:  {train_rmse:.4f} | Test RMSE:  {test_rmse:.4f}")
print(f"Train RMSLE: {train_rmsle:.4f} | Test RMSLE: {test_rmsle:.4f}")





def get_important_shap_features_tree(model, X, sample_size=1000, plot=True):
    """
    Use TreeExplainer to compute SHAP values and return important features.
    
    Parameters:
    - model: tree-based model (e.g., XGBoost, LightGBM, CatBoost)
    - X: DataFrame of features
    - sample_size: number of samples to use for SHAP computation
    - plot: if True, show SHAP summary bar plot
    
    Returns:
    - List of feature names with non-zero SHAP contribution
    """
    # Subsample if dataset is large
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    # Create TreeExplainer and compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_scores = pd.Series(mean_abs_shap, index=X_sample.columns)
    shap_scores = shap_scores[shap_scores > 0].sort_values(ascending=False)

    # Optional plot
    if plot:
        shap.summary_plot(shap_values, X_sample, plot_type="bar")
        shap.summary_plot(shap_values, X_sample)


    return 


get_important_shap_features_tree(final_model, X, sample_size=1000, plot=True)





# === Predict on test data (log scale) ===
test_pred_log = final_model.predict(test)

# === Convert predictions back to original scale ===
test_pred = np.expm1(test_pred_log)

# === Create submission file ===
submission = pd.DataFrame({
    'id': test.index,
    'Calories': test_pred
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission.head())


#Test Prediction values 


plt.figure(figsize=(6,4))
plt.hist(test_pred, bins=100)
plt.title("Test Predictions")
plt.show()




