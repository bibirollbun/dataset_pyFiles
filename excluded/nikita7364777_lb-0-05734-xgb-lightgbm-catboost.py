from IPython.display import Image
Image("/kaggle/input/rmsle-png/RMSLE.png")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
fig, axes = plt.subplots(1, 1, figsize = (10, 5))
# Hist
sns.histplot(df_train['Calories'], bins = 30, kde = True, ax = axes, color = 'blue')
axes.set_title('Hist Calories (Target)');


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
fig, axes = plt.subplots(1, 1, figsize = (10, 5))
# Hist
sns.histplot(np.log1p(df_train['Calories']), bins = 30, kde = True, ax = axes, color = 'blue')
axes.set_title('Hist Calories (Log-Target)');


# Base
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as py
from plotly.offline import init_notebook_mode
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

#Statistics
from scipy.stats import skew
from scipy import stats
from scipy.stats import norm, cramervonmises, anderson, kstest, norm, cramervonmises, randint
from statsmodels.stats.diagnostic import lilliefors, normal_ad, het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera, durbin_watson
! pip install arch
from arch.unitroot import VarianceRatio
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import featuretools as ft
from sklearn.inspection import permutation_importance

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder
#import cudf

#Models ML (Linear and Tree)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from xgboost import XGBRegressor
from xgboost import plot_importance
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#Model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

#Stacking
from sklearn.ensemble import StackingRegressor


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_train.head()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
df_test.head()


df_orig = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv")
df_orig = df_orig.rename(columns = {'User_ID': 'id', 'Gender': 'Sex'})
df_orig.head()


df_train.info(), df_test.info(), df_orig.info()


#df_train = pd.concat([df_train, df_orig], axis = 0, ignore_index = True)
#df_train.head()


df_train.describe(exclude = np.number).T


round(df_train.describe(exclude = 'object').T, 2).style.background_gradient(axis = 1, low = 0.3, high = 1.0)


plt.figure(figsize=(15, 5))
numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.drop(['id', 'Calories'])
df_numeric = df_train[['id'] + list(numeric_cols)]
colors = plt.cm.rainbow(np.linspace(0, 1, len(numeric_cols)))

for idx, col in enumerate(numeric_cols):
    plt.scatter(df_train['id'][(df_train['id'] < 0.75 * 1e7)], 
                df_train[col][(df_train['id']  < 0.75 * 1e7)], 
                color=colors[idx], 
                label=col,
                alpha=0.8, s = 0.5)

plt.xlabel('ID', fontsize=12)
plt.ylabel('Numerical values', fontsize=12)
plt.title('Feature dependencies on ID for train.csv', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))
numeric_cols = df_orig.select_dtypes(include=['int64', 'float64']).columns.drop(['id', 'Calories'])
df_numeric = df_orig[['id'] + list(numeric_cols)]
colors = plt.cm.rainbow(np.linspace(0, 1, len(numeric_cols)))

for idx, col in enumerate(numeric_cols):
    plt.scatter(df_orig['id'], 
                df_orig[col], 
                color=colors[idx], 
                label=col,
                alpha=0.8, s = 0.5)

plt.xlabel('ID', fontsize=12)
plt.ylabel('Numerical values', fontsize=12)
plt.title('Feature dependencies on ID for calories.csv', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 1, figsize = (15, 10))
# Hist
sns.histplot(df_train['Calories'], bins = 30, kde = True, ax = axes[0], color = 'blue')
axes[0].set_title('Hist Calories (Target)')
# BoxPlot
sns.boxplot(x = df_train['Calories'], ax = axes[1], color = 'blue')
axes[1].set_title('Boxplot Calories (Target)')


def nan_values(df):
    for i in df.columns:
        if df[i].isna().sum() > 0:
            print(f"For column - {i}, we have {df[i].isna().sum()} nan values")
        else:
            print(f"Our column {i} have zero nan values")
            print(f"Ideal!")


nan_values(df_train)
nan_values(df_test)


# Checking the normality of the target variable distribution
shapiro_test = stats.shapiro(df_train['Calories'])
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue:.3f}")
# If the p-value is < 0.05, the distribution is NOT normal.
# # We got a near-zero value, which tells us that the remnants of our target are distributed normally, therefore the final remnants of the model will also be distributed abnormally
# Let's try to prolog the target variable to improve the normal distribution


# Checking the normality of the log(target) variable distribution
shapiro_test = stats.shapiro(np.log(df_train['Calories']))
print(f"Shapiro-Wilk p-value: {shapiro_test.pvalue:.3f}")
# Also has a zero value, which means we are working with what we have


def z_metrics(df):
    results = []
    for feature in df.columns:
        mean = df[feature].mean()
        std = df[feature].std()
        df[f"{feature}_normal"] = (df[feature] - mean) / std
        # Calculating anomalies
        anomalies = df[np.abs(df[f"{feature}_normal"]) > 3]
        n_anomalies = len(anomalies)
        percentage = n_anomalies / len(df) * 100
        results.append({'Feature': feature,
                        'Number of anomalies': n_anomalies,
                        'Percentage of anomalies': round(percentage, 2)})
    report_df = pd.DataFrame(results, columns=['Feature', 'Number of anomalies', 'Percentage of anomalies'])
    return report_df

z_report = z_metrics(df_train.drop(columns=['id', 'Sex']))
z_report
# The test results show that the largest percentage of abnormal values will be found in the Body_Temp predictor, let's plot it.


plt.figure(figsize=(13, 7))
plt.scatter(df_train['id'], 
           df_train['Body_Temp'],
           color = 'red',
           label="Body_Temp",
           s = 0.5,
           alpha=0.6)
plt.title("Body_Temp graphs for train.csv")
plt.xlabel("id")
plt.ylabel("Body_Temp");


# Categorical Variable Analysis (Sex)
gender_dist = df_train['Sex'].value_counts(normalize=True)
fig = px.pie(gender_dist, 
       names=gender_dist.index, 
       title='Gender Distribution',
       color_discrete_sequence=px.colors.qualitative.Pastel)
init_notebook_mode(connected=True)
py.iplot(fig)


# 4. Numerical Features Distribution vs Sex
num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data = df_train, x = col, kde = True, hue = 'Sex', color='skyblue')
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()


# Correlation Analysis
corr_matrix = df_train[num_cols + ['Calories']].corr()
fig = px.imshow(round(corr_matrix, 2),
                text_auto=True,
                color_continuous_scale='rainbow',
                title='Feature Correlation Matrix')
fig.update_layout(width=800, height=800)
init_notebook_mode(connected=True)
py.iplot(fig)


# Target vs Features Relationships (Sampled 1%)
fig = px.scatter_matrix(df_train,
                        dimensions=['Age', 'Duration', 'Heart_Rate', 'Calories'],
                        color='Sex',
                        title='Pairwise Relationships',
                        opacity=0.5)
fig.update_layout(width = 1300, height = 800)
fig.update_traces(diagonal_visible = False)
init_notebook_mode(connected=True)
py.iplot(fig)


# 3D Visualization
fig = px.scatter_3d(df_train.sample(frac=0.2),
                    x='Age',
                    y='Duration',
                    z='Heart_Rate',
                    color='Calories',
                    title='3D Relationship: Age vs Duration vs Heart Rate',
                    color_continuous_scale='Viridis')
fig.update_layout(width = 500, height = 500)
init_notebook_mode(connected=True)
py.iplot(fig)


# Outlier Detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train[num_cols], orient='h', palette='Set2')
plt.title('Numerical Features Boxplot Comparison')
plt.xlabel('Values')
plt.show()


#1. Initializing EntitySet
es = ft.EntitySet(id='workout_data')

#2. Adding an entity
try:
    es = es.add_dataframe(
        dataframe_name='workouts',
        dataframe=df_train.drop(columns=['Calories']), 
        index='id'
    )
    print("âœ… The 'workouts' entity has been added.")
except Exception as e:
    print(f"â�Œ Error: {e}")
    raise

#3. Feature generation
try:
    trans_primitives = ['multiply_numeric', 'divide_numeric', 'subtract_numeric']
    #groupby_trans_primitives = ['sum', 'mean', 'max']

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name='workouts',
        trans_primitives = trans_primitives,
        #groupby_trans_primitives = groupby_trans_primitives,
        max_depth=2,
        verbose=True
    )
    print("âœ… The signs are generated.")
except Exception as e:
    print(f"â�Œ Error in DFS: {e}")
    raise

# 3. Automatic selection of features by patterns
enhanced_df = feature_matrix
# ---------------------------------------------------------------------- I can't create aggregated attributes in any way


# from (thank you): https://www.kaggle.com/code/andrewsokolovsky/catboost-xgboost-lightgbm-rmsle-0-05684
def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature_1 = numerical_features[i]
            feature_2 = numerical_features[j]
            cross_term_name = f"{feature_1}_x_{feature_2}"
            df_new[cross_term_name] = df_new[feature_1] * df_new[feature_2]
    return df_new


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
df_train = add_feature_cross_terms(df_train, numerical_features)
df_test = add_feature_cross_terms(df_test, numerical_features)


df_train.info()


df_train = df_train.rename(columns = {'Heart_Rate_x_Body_Temp': 'Metabolic load',
                                      'Age_x_Duration': 'Age_Duration_Interaction'})
df_test = df_test.rename(columns = {'Heart_Rate_x_Body_Temp': 'Metabolic load',
                                      'Age_x_Duration': 'Age_Duration_Interaction'})


def create_manual_features(df):
    # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ ĞºĞ¾Ğ¿Ğ¸Ñ� DataFrame Ñ‡Ñ‚Ğ¾Ğ±Ñ‹ Ğ¸Ğ·Ğ±ĞµĞ¶Ğ°Ñ‚ÑŒ side effects
    df = df.copy()
    # =============================================
    # 1. Anthropometric characteristics
    # =============================================
    df['BMI'] = df['Weight'] / (df['Height'] / 100)**2  # Body mass index
    df['Height_Weight_Ratio'] = df['Height'] / df['Weight']  # Height-weight ratio
    
    # =============================================
    # 2. Physiological indicators
    # =============================================
    df['HR_Duration_Ratio'] = df['Heart_Rate'] / df['Duration']  # Load intensity
    # df['Metabolic_Load'] = df['Heart_Rate'] * df['Body_Temp']  # Metabolic load
    
    # =============================================
    # 3. Time and statistical indicators
    # =============================================
    # Logarithm for normalization
    df['Log_Duration'] = np.log1p(df['Duration'])
    
    # =============================================
    # 4. Group statistics by gender
    # =============================================
    # Aggregates by gender
    sex_agg = df.groupby('Sex').agg({'Age': ['mean', 'std'],
                                     'Heart_Rate': ['mean', 'max'],
                                     'Duration': ['sum', 'median']})
    # Renaming the columns
    sex_agg.columns = ['Sex_' + '_'.join(col).upper() for col in sex_agg.columns]
    # Merge with the original data
    df = df.merge(sex_agg, on = 'Sex', how = 'left')
    
    # =============================================
    # 5. Time series (cumulative amounts)
    # =============================================
    # Sorting by id before calculation
    df = df.sort_values('id')
    
    # Cumulative indicators by gender
    df['Cum_Duration_By_Sex'] = df.groupby('Sex')['Duration'].cumsum()
    
    # =============================================
    # 6. Interactions of features
    # =============================================
    df['HR_BMI_Interaction'] = df['Heart_Rate'] * df['BMI']
    #df['Age_Duration_Interaction'] = df['Age'] * df['Duration']
    
    # =============================================
    # 7. Categorical transformations
    # =============================================
    # 1) Create an instance of OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # 2. Convert the "Sex" column
    # First, we convert the column to a numpy array and reshape it to make it two-dimensional
    sex_encoded = encoder.fit_transform(df[['Sex']]) # Two-dimensional array, even if there is one column
    # 3. We get the names of the new columns
    encoded_columns = encoder.get_feature_names_out(['Sex'])
    # 4. Create a new Data Frame from the encoded values
    sex_encoded_df = pd.DataFrame(sex_encoded, columns = encoded_columns, index = df.index)
    #5. Combine the encoded data with the original DataFrame
    df = pd.concat([df, sex_encoded_df], axis = 1)
    # 6. Delete the original "Sex" column
    df = df.drop('Sex', axis=1)
    
    return df


# Applying the function to the training data
df_train = create_manual_features(df_train)


df_train.info()


# Applying the function to the test data
df_test = create_manual_features(df_test)


df_test.info()


# We separate the data for modeling and scale it
y_full = df_train['Calories'].reset_index(drop = True)
X_full = df_train.drop(columns = ['id', 'Calories']).reset_index(drop = True)
X_test = df_test.drop(columns = ['id'])

scaler = StandardScaler().set_output(transform = "pandas")
X_full = scaler.fit_transform(X_full)
X_test = scaler.transform(X_test)

# We check the importance of features on a model with standard hyperparameter values
default_param = {'objective': 'reg:squaredlogerror',
                 'eval_metric': 'rmsle',
                 'tree_method': 'gpu_hist',
                 'device': 'cuda',
                 'seed': 42}

trained_model_XGB = XGBRegressor(**default_param).fit(X_full, y_full)


# 1. Correct definition of RMSLE (getting rid of negative prediction values)
def root_mean_squared_log_error(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)  # Ğ“Ğ°Ñ€Ğ°Ğ½Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ½ĞµĞ¾Ñ‚Ñ€Ğ¸Ñ†Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ñ‹Ğµ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ�
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

# 2. Calculating permutation importance based on the direction of the metric
results = permutation_importance(trained_model_XGB,
                                 X_full,
                                 y_full,
                                 scoring = make_scorer(root_mean_squared_log_error, greater_is_better = False),
                                 n_repeats = 10,
                                 random_state = 42)

# 3. Handling negative importance values
importance = pd.DataFrame({'feature': X_full.columns,
                           'importance': results.importances_mean})
# Absolute values for visualization
importance['abs_importance'] = np.abs(importance['importance'])
# Sorting by absolute importance
importance = importance.sort_values('abs_importance', ascending = True)

# 4. Visualization of the top N features
top_n = 25
plt.figure(figsize=(12, 8))
bars = plt.barh(importance['feature'].head(top_n)[::-1],
                importance['abs_importance'].head(top_n)[::-1],
                color = 'cyan')

# 5. Add annotations with real values
for bar in bars:
    width = bar.get_width()
    plt.text(width * 1.02, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va = 'center')

plt.xlabel('Absolute Importance')
plt.title('Permutation Importance XGB (Top 25 Features)')
plt.gca().invert_yaxis()
plt.grid(axis = 'x', alpha = 0.3)
plt.tight_layout()
plt.show()


feature_names = X_full.columns
importances = trained_model_XGB.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(12, 8))
plt.barh(y = sorted_features[-25:], width=sorted_importances[-25:], color='cyan',edgecolor='black')
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("XGB Feature Importance", fontsize=14, pad = 20)
plt.grid(axis='x', linestyle='--', alpha=0.7)
for i, v in enumerate(sorted_importances[-25:]):
    plt.text(v + 0.001, i, f'{v:.3f}', color='black',va='center',fontsize=9)
plt.tight_layout()
plt.show()


features_to_drop = ['Sex_male', 'Sex_female', 'Sex_DURATION_MEDIAN', 'Sex_DURATION_SUM', 'Sex_HEART_RATE_MAX', 'Sex_HEART_RATE_MEAN', 'Sex_AGE_STD', 'Log_Duration']

predictors = set(X_full.columns) - set(features_to_drop)
predictors = list(predictors)


vif_data = pd.DataFrame()
vif_data["feature"] = X_full[predictors].columns
vif_data["VIF"] = [variance_inflation_factor(X_full[predictors].values, i) for i in range(X_full[predictors].shape[1])]
print("\nMulticollinearity check (For Tree and Boosting Models VIF > 1 000 000 is a problem (I found this statement in one of the articles on multicollinearity.) AND for Linear Models VIF > 10 (5) have a strong influnce on weight coefficients:")
print(vif_data.sort_values("VIF", ascending=False))


# Best RMSLE = 0.6298 calories - Optuna
# The heteroscedasticity in the data is too high, so we use logorithmization (you can also use the Box-Cox transform)
def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int("n_estimators", 500, 5000, step = 100),
        'max_depth': trial.suggest_int("max_depth", 6, 15, step = 2),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.9, log=True),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
        'subsample': trial.suggest_float("subsample", 0.5, 0.95),
        'gamma': trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 0.95),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-6, 1e-1, log=True),
        'objective': 'reg:squaredlogerror',
        'eval_metric': 'rmsle',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'seed': 42}

    model = XGBRegressor(**xgb_params)
    
    # Cross-validation configuration
    cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
    rmsle_scores_valid = []
    y_pred_val = np.zeros(len(X_full))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full[predictors], y_full)):
        print(f"\n Fold XGB (Optim) {fold + 1}")
        # Data separation
        X_train = X_full[predictors].iloc[idx_train].copy()
        X_valid = X_full[predictors].iloc[idx_valid].copy()
        y_train = y_full.iloc[idx_train].copy()
        y_valid = y_full.iloc[idx_valid].copy()

        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  early_stopping_rounds = 500,
                  verbose = 100)

        y_pred_val[idx_valid] = model.predict(X_valid)

        fold_rmsle_valid = np.sqrt(mean_squared_log_error(y_valid, y_pred_val[idx_valid]))
        rmsle_scores_valid.append(fold_rmsle_valid)
        print(f"Fold XGB (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

    overall_rmsle_valid = np.sqrt(mean_squared_log_error(y_full, y_pred_val))
    print(f"\nğŸ�¯ Overall CV XGB (Optim) RMSLE: {overall_rmsle_valid:.5f}")

    return overall_rmsle_valid


#sampler = TPESampler(seed = 42)
#study_1 = optuna.create_study(direction = "minimize", sampler=sampler)
#study_1.optimize(objective, n_trials = 25)


# Best RMSLE = 0.06007 calories
# Final TEST prediction RMSLE = 0.05826 (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int("n_estimators", 500, 5000, step = 100),
        'max_depth': trial.suggest_int("max_depth", 6, 15, step = 2),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.9, log=True),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
        'subsample': trial.suggest_float("subsample", 0.5, 0.95),
        'gamma': trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 0.95),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-6, 1e-1, log=True),
        'objective': 'reg:squaredlogerror',
        'eval_metric': 'rmsle',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'seed': 42}

    model = XGBRegressor(**xgb_params)
    
    # Cross-validation configuration
    cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
    rmsle_scores_valid = []
    y_pred_val = np.zeros(len(X_full))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full[predictors], np.log1p(y_full))):
        print(f"\n Fold XGB (Optim) {fold + 1}")
        # Data separation
        X_train = X_full[predictors].iloc[idx_train].copy()
        X_valid = X_full[predictors].iloc[idx_valid].copy()
        y_train = np.log1p(y_full.iloc[idx_train].copy())
        y_valid = np.log1p(y_full.iloc[idx_valid].copy())

        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  #early_stopping_rounds = 500,
                  verbose = 100)

        y_pred_val[idx_valid] = model.predict(X_valid)

        fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val[idx_valid])))
        rmsle_scores_valid.append(fold_rmsle_valid)
        print(f"Fold XGB (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

    overall_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val)))
    print(f"\nğŸ�¯ Overall CV XGB (Optim) RMSLE: {overall_rmsle_valid:.5f}")

    return overall_rmsle_valid


#sampler = TPESampler(seed = 42)
#study_2 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_2.optimize(objective, n_trials = 25)


xgb_params_1 = {'n_estimators': 3800, 'max_depth': 10, 'learning_rate': 0.0030243734646021444, 
                'reg_alpha': 0.0002801815566698886, 'subsample': 0.7032510340614875, 
                'gamma': 0.0006301399943418406, 'colsample_bytree': 0.47221963917204685, 
                'min_child_weight': 2, 'reg_lambda': 0.002198536815574321}
xgb_params_2 = {'objective': 'reg:squaredlogerror', 'eval_metric': 'rmsle',
                'tree_method': 'gpu_hist', 'device': 'cuda', 'seed': 42}
model_1 = XGBRegressor(**xgb_params_1, **xgb_params_2)
model_1.fit(X_full[predictors], np.log1p(y_full))
y_xgb_train_pred = model_1.predict(X_full[predictors])
rmsle_xgb_train_pred = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_xgb_train_pred)))
print(f"Root mean squared log error on the full dataset = {round(rmsle_xgb_train_pred, 5)} calories")


# Setting the initial parameters for validation
cv = KFold(n_splits = 5, random_state = 41, shuffle = True)
rmsle_scores_valid_xgb = []
rmsle_scores_train_xgb = []
y_pred_val_xgb = np.zeros(len(X_full))
y_pred_train_xgb = np.zeros(len(X_full))
y_pred_test_xgb = np.zeros(len(X_test))

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full[predictors], np.log1p(y_full))):
    print(f"\n Fold XGB (Final) {fold + 1}")
    # Separating the training and radiation data from the source dataset
    X_train = X_full[predictors].iloc[idx_train].copy()
    X_valid = X_full[predictors].iloc[idx_valid].copy()
    X_test = X_test[predictors].copy()
    y_train = np.log1p(y_full.iloc[idx_train].copy())
    y_valid = np.log1p(y_full.iloc[idx_valid].copy())

    model_1.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              #early_stopping_rounds = 500,
              verbose = 100)
    
    # We try to extract maximum information from the radiation, so we calculate predictions based on the training data.
    y_pred_val_xgb[idx_valid] = model_1.predict(X_valid)
    y_pred_train_xgb[idx_train] = model_1.predict(X_train)
    y_pred_test_xgb += model_1.predict(X_test)

    # We define RMSLE on both training and validation sets
    # The variables rmsle_scores_valid and rmsle_scores_train will be used for plotting graphs.
    fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val_xgb[idx_valid])))
    fold_rmsle_train = np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(y_pred_train_xgb[idx_train])))
    rmsle_scores_valid_xgb.append(fold_rmsle_valid)
    rmsle_scores_train_xgb.append(fold_rmsle_train)
    print(f"Fold XGB (Final) {fold + 1} RMSLE on valid data: {fold_rmsle_valid:.5f}")
    print(f"Fold XGB (Final) {fold + 1} RMSLE on train data: {fold_rmsle_train:.5f}")

# It is much more reliable to calculate the error on the already calculated data, rather than averaging it over 5 fouls.
overall_rmsle_valid_xgb = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val_xgb)))
overall_rmsle_train_xgb = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_train_xgb)))
print(f"\nğŸ�¯ Overall CV XGB (Final) RMSLE on valid data: {overall_rmsle_valid_xgb:.5f}")
print(f"\nğŸ�¯ Overall CV XGB (Final) RMSLE on train data: {overall_rmsle_train_xgb:.5f}")
# Since the predicted data on the test set was summed up every fold (5), therefore we divide our sum by the number of folds
y_pred_test_xgb /= 5


# Create figure
fig = go.Figure()

# Add fold validation scores
fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_train_xgb,
                         mode = 'lines+markers', name = 'Train RMSLE_xgb per Fold',
                         line=dict(color = 'blue', dash = 'dash'), marker = dict(size = 8)))

fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_valid_xgb,
                         mode = 'lines+markers', name = 'Valid RMSLE_xgb per Fold',
                         line = dict(color = 'red', dash = 'dash'), marker = dict(size = 8)))

# Add overall horizontal lines
fig.add_shape(type="line", 
              x0 = 0.5, y0 = overall_rmsle_train_xgb, 
              x1 = 5.5, y1 = overall_rmsle_train_xgb,
              line=dict(color="blue", width = 1), name='Overall Train RMSLE_xgb')

fig.add_shape(type="line",
              x0=0.5, y0=overall_rmsle_valid_xgb,
              x1=5.5, y1=overall_rmsle_valid_xgb,
              line=dict(color="red", width = 1), name='Overall Valid RMSLE_xgb')

# Add annotations for overall scores
fig.add_annotation(x=5.3, y=0.050,
                   text=f'Train: {overall_rmsle_train_xgb:.5f}',
                   showarrow=False, font=dict(color='blue'))
    
fig.add_annotation(x = 5.3, y = 0.061,
                   text=f'Valid: {overall_rmsle_valid_xgb:.5f}',
                   showarrow=False, font=dict(color='red'))

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Cross-Validation RMSLE_xgb Scores</b>',
        font=dict(size=24, family='Arial'),
        x = 0.5
    ),
    xaxis=dict(
        title='Fold Number',
        tickmode = 'array',
        tickvals=list(range(1, 6)),
        gridcolor='lightgrey',
        title_font=dict(size=16)
    ),
    yaxis=dict(
        title='RMSLE',
        gridcolor='lightgrey',
        title_font=dict(size=16),
        range=[0.045, 0.062]
    ),
    plot_bgcolor='white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    ),
    margin=dict(l=80, r=80, t=100, b=80),
    height=600,
    width=1300
)

init_notebook_mode(connected=True)
py.iplot(fig)


results_xgb = model_1.evals_result()
plt.figure(figsize=(10,5))
plt.plot(results_xgb["validation_0"]["rmsle"], label="Training loss")
plt.plot(results_xgb["validation_1"]["rmsle"], label="Validation loss")
plt.axvline(2000, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.title('Graphics of Loss function (RMSLE) for XGBoostModels')
plt.legend();


# Best RMSLE = 0.5998 calories
# Final TEST prediction RMSLE = 0.05793 calories (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
def objective(trial):
    lgbm_params = {'n_estimators': trial.suggest_int("n_estimators", 500, 3000, step = 100),
                   'num_leaves': trial.suggest_int("num_leaves", 31, 127, step = 32),
                   'max_depth': trial.suggest_int("max_depth", 3, 12),
                   'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.9, log=True),
                   'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 1e-1, log = True),
                   'reg_lambda': trial.suggest_float("reg_lambda", 1e-6, 1e-1, log = True),
                   'subsample': trial.suggest_float("subsample", 0.6, 0.95),
                   'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 0.95),
                   'min_child_weight': trial.suggest_float("min_child_weight", 1e-4, 1e-1, log=True),
                   'device': 'cpu',
                   'random_state': 42,
                   'verbose': -1}

    model = LGBMRegressor(**lgbm_params)
    
    # Cross-validation configuration
    cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
    rmsle_scores_valid = []
    y_pred_val = np.zeros(len(X_full))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full[predictors], np.log1p(y_full))):
        print(f"\n Fold LGBM (Optim) {fold + 1}")
        # Data separation
        X_train = X_full[predictors].iloc[idx_train].copy()
        X_valid = X_full[predictors].iloc[idx_valid].copy()
        y_train = np.log1p(y_full.iloc[idx_train].copy())
        y_valid = np.log1p(y_full.iloc[idx_valid].copy())

        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric='rmsle',
                  callbacks=[lgb.early_stopping(500, verbose = 100)])

        y_pred_val[idx_valid] = model.predict(X_valid)

        fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val[idx_valid])))
        rmsle_scores_valid.append(fold_rmsle_valid)
        print(f"Fold LGBM (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

    overall_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val)))
    print(f"\nğŸ�¯ Overall CV LGBM (Optim) RMSLE: {overall_rmsle_valid:.5f}")

    return overall_rmsle_valid


#sampler = TPESampler(seed = 42)
#study_3 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_3.optimize(objective, n_trials = 25)


lgbm_params_1 = {'n_estimators': 2700, 'num_leaves': 95, 'max_depth': 6, 
                  'learning_rate': 0.020651360831882216, 'reg_alpha': 0.0006941535518885974, 
                  'reg_lambda': 0.05544369867596478, 'subsample': 0.8701734043330248, 
                  'colsample_bytree': 0.7943431723843197, 'min_child_weight': 0.008843767026866556}
lgbm_params_2 = {'device': 'cpu',
                'random_state': 42,
                'verbose': -1}
model_2 = LGBMRegressor(**lgbm_params_1, **lgbm_params_2)
model_2.fit(X_full[predictors], np.log1p(y_full))
y_lgbm_train_pred = model_2.predict(X_full[predictors])
rmsle_lgbm_train_pred = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_lgbm_train_pred)))
print(f"Root mean squared log error on the full dataset = {round(rmsle_lgbm_train_pred, 5)} calories")


# Setting the initial parameters for validation
cv = KFold(n_splits = 5, random_state = 41, shuffle = True)
rmsle_scores_valid_lgbm = []
rmsle_scores_train_lgbm = []
y_pred_val_lgbm = np.zeros(len(X_full))
y_pred_train_lgbm = np.zeros(len(X_full))
y_pred_test_lgbm = np.zeros(len(X_test))

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full[predictors], np.log1p(y_full))):
    print(f"\n Fold LightGBM (Final) {fold + 1}")
    # Separating the training and radiation data from the source dataset
    X_train = X_full[predictors].iloc[idx_train].copy()
    X_valid = X_full[predictors].iloc[idx_valid].copy()
    X_test  = X_test[predictors].copy()
    y_train = np.log1p(y_full.iloc[idx_train].copy())
    y_valid = np.log1p(y_full.iloc[idx_valid].copy())

    model_2.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric='rmsle',
                callbacks=[lgb.early_stopping(500, verbose = 100)])
    
    # We try to extract maximum information from the radiation, so we calculate predictions based on the training data.
    y_pred_val_lgbm[idx_valid] = model_2.predict(X_valid)
    y_pred_train_lgbm[idx_train] = model_2.predict(X_train)
    y_pred_test_lgbm += model_2.predict(X_test)

    # We define RMSLE on both training and validation sets
    # The variables rmsle_scores_valid and rmsle_scores_train will be used for plotting graphs.
    fold_rmsle_valid_lgbm = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val_lgbm[idx_valid])))
    fold_rmsle_train_lgbm = np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(y_pred_train_lgbm[idx_train])))
    rmsle_scores_valid_lgbm.append(fold_rmsle_valid_lgbm)
    rmsle_scores_train_lgbm.append(fold_rmsle_train_lgbm)
    print(f"Fold LightGBM (Final) {fold + 1} RMSLE on valid data: {fold_rmsle_valid_lgbm:.5f}")
    print(f"Fold LightGBM (Final) {fold + 1} RMSLE on train data: {fold_rmsle_train_lgbm:.5f}")


# It is much more reliable to calculate the error on the already calculated data, rather than averaging it over 5 fouls.
overall_rmsle_valid_lgbm = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val_lgbm)))
overall_rmsle_train_lgbm = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_train_lgbm)))
print(f"\nğŸ�¯ Overall CV LightGBM (Final) RMSLE on valid data: {overall_rmsle_valid_lgbm:.5f}")
print(f"\nğŸ�¯ Overall CV LightGBM (Final) RMSLE on train data: {overall_rmsle_train_lgbm:.5f}")
# Since the predicted data on the test set was summed up every fold (5), therefore we divide our sum by the number of folds
y_pred_test_lgbm /= 5


# Create figure
fig = go.Figure()

# Add fold validation scores
fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_train_lgbm,
                         mode = 'lines+markers', name = 'Train RMSLE_lgbm per Fold',
                         line=dict(color = 'blue', dash = 'dash'), marker = dict(size = 8)))

fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_valid_lgbm,
                         mode = 'lines+markers', name = 'Valid RMSLE_lgbm per Fold',
                         line = dict(color = 'red', dash = 'dash'), marker = dict(size = 8)))

# Add overall horizontal lines
fig.add_shape(type="line", 
              x0 = 0.5, y0 = overall_rmsle_train_lgbm, 
              x1 = 5.5, y1 = overall_rmsle_train_lgbm,
              line=dict(color="blue", width = 1), name='Overall Train RMSLE_lgbm')

fig.add_shape(type="line",
              x0=0.5, y0=overall_rmsle_valid_lgbm,
              x1=5.5, y1=overall_rmsle_valid_lgbm,
              line=dict(color="red", width = 1), name='Overall Valid RMSLE_lgbm')

# Add annotations for overall scores
fig.add_annotation(x=5.3, y=0.050,
                   text=f'Train: {overall_rmsle_train_lgbm:.5f}',
                   showarrow=False, font=dict(color='blue'))
    
fig.add_annotation(x = 5.3, y = 0.061,
                   text=f'Valid: {overall_rmsle_valid_lgbm:.5f}',
                   showarrow=False, font=dict(color='red'))

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Cross-Validation RMSLE_LightGBM Scores</b>',
        font=dict(size=24, family='Arial'),
        x = 0.5
    ),
    xaxis=dict(
        title='Fold Number',
        tickmode = 'array',
        tickvals=list(range(1, 6)),
        gridcolor='lightgrey',
        title_font=dict(size=16)
    ),
    yaxis=dict(
        title='RMSLE',
        gridcolor='lightgrey',
        title_font=dict(size=16),
        range=[0.045, 0.062]
    ),
    plot_bgcolor='white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    ),
    margin=dict(l=80, r=80, t=100, b=80),
    height=600,
    width=1300
)

init_notebook_mode(connected=True)
py.iplot(fig)


results_lgbm = model_2.evals_result_
plt.figure(figsize=(10,5))
plt.plot(results_lgbm['training']['l2'], label="Training loss")
plt.plot(results_lgbm['valid_1']['l2'], label="Validation loss")
plt.axvline(200, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.title('Graphics of Loss function (RMSLE) for LightGBMModels')
plt.legend();


# Best RMSLE = 0.05948 calories
# Final TEST prediction RMSLE = 0.05730 calories (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
def objective(trial):
    cat_params = {'iterations': trial.suggest_int("n_estimators", 500, 3000, step=100),
        'depth': trial.suggest_int("depth", 3, 12),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.9, log=True),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-6, 1e-1, log=True),
        'subsample': trial.suggest_float("subsample", 0.6, 0.95),
        'rsm': trial.suggest_float("rsm", 0.6, 0.95),
        'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 100),
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU'}

    model = CatBoostRegressor(**cat_params)
    
    # Cross-validation configuration
    cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
    rmsle_scores_valid = []
    y_pred_val = np.zeros(len(X_full))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full[predictors], np.log1p(y_full))):
        print(f"\n Fold CatBoost (Optim) {fold + 1}")
        # Data separation
        X_train = X_full[predictors].iloc[idx_train].copy()
        X_valid = X_full[predictors].iloc[idx_valid].copy()
        y_train = np.log1p(y_full.iloc[idx_train].copy())
        y_valid = np.log1p(y_full.iloc[idx_valid].copy())

        model.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=500,
            verbose=100)

        y_pred_val[idx_valid] = model.predict(X_valid)

        fold_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val[idx_valid])))
        rmsle_scores_valid.append(fold_rmsle_valid)
        print(f"Fold CatBoost (Optim) {fold + 1} RMSLE: {fold_rmsle_valid:.5f}")

    overall_rmsle_valid = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val)))
    print(f"\nğŸ�¯ Overall CV CatBoost (Optim) RMSLE: {overall_rmsle_valid:.5f}")

    return overall_rmsle_valid


#sampler = TPESampler(seed = 42)
#study_4 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_4.optimize(objective, n_trials = 25)


cat_params_1 = {'n_estimators': 2800, 
                'depth': 11, 
                'learning_rate': 0.012008496949716967, 
                'l2_leaf_reg': 0.023051836331648992, 
                'subsample': 0.69569069665225, 
                'rsm': 0.9087310464907349, 
                'min_data_in_leaf': 72}
cat_params_2 = {'random_seed': 42,
                'verbose': False,
                'task_type': 'CPU'}
model_3 = CatBoostRegressor(**cat_params_1, **cat_params_2)
model_3.fit(X_full[predictors], np.log1p(y_full))
y_cat_train_pred = model_3.predict(X_full[predictors])
rmsle_cat_train_pred = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_cat_train_pred)))
print(f"Root mean squared log error on the full dataset = {round(rmsle_cat_train_pred, 5)} calories")


# Setting the initial parameters for validation
cv = KFold(n_splits = 5, random_state = 41, shuffle = True)
rmsle_scores_valid_cat = []
rmsle_scores_train_cat = []
y_pred_val_cat = np.zeros(len(X_full))
y_pred_train_cat = np.zeros(len(X_full))
y_pred_test_cat = np.zeros(len(X_test))

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full[predictors], np.log1p(y_full))):
    print(f"\n Fold CatBoost (Final) {fold + 1}")
    # Separating the training and radiation data from the source dataset
    X_train = X_full[predictors].iloc[idx_train].copy()
    X_valid = X_full[predictors].iloc[idx_valid].copy()
    X_test  = X_test[predictors].copy()
    y_train = np.log1p(y_full.iloc[idx_train].copy())
    y_valid = np.log1p(y_full.iloc[idx_valid].copy())

    model_3.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=500,
                verbose=100)
    
    # We try to extract maximum information from the radiation, so we calculate predictions based on the training data.
    y_pred_val_cat[idx_valid] = model_3.predict(X_valid)
    y_pred_train_cat[idx_train] = model_3.predict(X_train)
    y_pred_test_cat += model_3.predict(X_test)

    # We define RMSLE on both training and validation sets
    # The variables rmsle_scores_valid and rmsle_scores_train will be used for plotting graphs.
    fold_rmsle_valid_cat = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val_cat[idx_valid])))
    fold_rmsle_train_cat = np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(y_pred_train_cat[idx_train])))
    rmsle_scores_valid_cat.append(fold_rmsle_valid_cat)
    rmsle_scores_train_cat.append(fold_rmsle_train_cat)
    print(f"Fold CatBoost (Final) {fold + 1} RMSLE on valid data: {fold_rmsle_valid_cat:.5f}")
    print(f"Fold CatBoost (Final) {fold + 1} RMSLE on train data: {fold_rmsle_train_cat:.5f}")


# It is much more reliable to calculate the error on the already calculated data, rather than averaging it over 5 fouls.
overall_rmsle_valid_cat = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val_cat)))
overall_rmsle_train_cat = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_train_cat)))
print(f"\nğŸ�¯ Overall CV XGB (Final) RMSLE on valid data: {overall_rmsle_valid_cat:.5f}")
print(f"\nğŸ�¯ Overall CV XGB (Final) RMSLE on train data: {overall_rmsle_train_cat:.5f}")
# Since the predicted data on the test set was summed up every fold (5), therefore we divide our sum by the number of folds
y_pred_test_cat /= 5


# Create figure
fig = go.Figure()

# Add fold validation scores
fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_train_cat,
                         mode = 'lines+markers', name = 'Train RMSLE_cat per Fold',
                         line=dict(color = 'blue', dash = 'dash'), marker = dict(size = 8)))

fig.add_trace(go.Scatter(x = list(range(1, 6)), y = rmsle_scores_valid_cat,
                         mode = 'lines+markers', name = 'Valid RMSLE_cat per Fold',
                         line = dict(color = 'red', dash = 'dash'), marker = dict(size = 8)))

# Add overall horizontal lines
fig.add_shape(type="line", 
              x0 = 0.5, y0 = overall_rmsle_train_cat, 
              x1 = 5.5, y1 = overall_rmsle_train_cat,
              line=dict(color="blue", width = 1), name='Overall Train RMSLE_cat')

fig.add_shape(type="line",
              x0=0.5, y0=overall_rmsle_valid_cat,
              x1=5.5, y1=overall_rmsle_valid_cat,
              line=dict(color="red", width = 1), name='Overall Valid RMSLE_cat')

# Add annotations for overall scores
fig.add_annotation(x=5.3, y=0.050,
                   text=f'Train: {overall_rmsle_train_cat:.5f}',
                   showarrow=False, font=dict(color='blue'))
    
fig.add_annotation(x = 5.3, y = 0.061,
                   text=f'Valid: {overall_rmsle_valid_cat:.5f}',
                   showarrow=False, font=dict(color='red'))

# Update layout
fig.update_layout(
    title=dict(
        text='<b>Cross-Validation RMSLE_CatBoost Scores</b>',
        font=dict(size=24, family='Arial'),
        x = 0.5
    ),
    xaxis=dict(
        title='Fold Number',
        tickmode = 'array',
        tickvals=list(range(1, 6)),
        gridcolor='lightgrey',
        title_font=dict(size=16)
    ),
    yaxis=dict(
        title='RMSLE',
        gridcolor='lightgrey',
        title_font=dict(size=16),
        range=[0.045, 0.062]
    ),
    plot_bgcolor='white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    ),
    margin=dict(l=80, r=80, t=100, b=80),
    height=600,
    width=1300
)

init_notebook_mode(connected=True)
py.iplot(fig)


# Here I decided to create a custom metric for CatBoost, since the CatBoostRegressor functionality does not allow adding RMSLE to the loss function
# By default, CatBoost uses the RMSE loss function
# The results are the same

class RMSLE(object):
    def is_max_optimal(self):
        return False
    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        approx = approxes[0]
        
        # Converting predictions and goals from logarithmic form
        approx_exp = np.expm1(approx)
        target_exp = np.expm1(target)
        
        # Calculating the RMSLE
        error = np.sqrt(np.mean(np.square(np.log1p(approx_exp) - np.log1p(target_exp))))
        return error, 1  # We return the error and weight

    def get_final_error(self, error, weight):
        return error

cat_params_2 = {
    'random_seed': 42,
    'verbose': False,
    'task_type': 'CPU',
    'eval_metric': RMSLE()
}

# Initializing the model with the new parameters
model_3 = CatBoostRegressor(**cat_params_1, **cat_params_2)

cv = KFold(n_splits = 5, random_state = 41, shuffle = True)
rmsle_scores_valid_cat = []
rmsle_scores_train_cat = []
y_pred_val_cat = np.zeros(len(X_full))
y_pred_train_cat = np.zeros(len(X_full))
y_pred_test_cat = np.zeros(len(X_test))
# To store the history of metrics
train_loss_history = []
val_loss_history = []

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full[predictors], np.log1p(y_full))):
    print(f"\n Fold CatBoost {fold + 1}")
    
    # Data preparation
    X_train = X_full[predictors].iloc[idx_train]
    X_valid = X_full[predictors].iloc[idx_valid]
    y_train = np.log1p(y_full.iloc[idx_train])
    y_valid = np.log1p(y_full.iloc[idx_valid])

    # Training with tracking metrics
    model_3.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=500,
        verbose=100,
        use_best_model=True
    )
    
    # Saving the history of metrics
    results = model_3.get_evals_result()
    train_loss = results['learn']['RMSLE']
    val_loss = results['validation']['RMSLE']
    
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    
    # We try to extract maximum information from the radiation, so we calculate predictions based on the training data.
    y_pred_val_cat[idx_valid] = model_3.predict(X_valid)
    y_pred_train_cat[idx_train] = model_3.predict(X_train)
    y_pred_test_cat += model_3.predict(X_test)

    # We define RMSLE on both training and validation sets
    # The variables rmsle_scores_valid and rmsle_scores_train will be used for plotting graphs.
    fold_rmsle_valid_cat = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val_cat[idx_valid])))
    fold_rmsle_train_cat = np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(y_pred_train_cat[idx_train])))
    rmsle_scores_valid_cat.append(fold_rmsle_valid_cat)
    rmsle_scores_train_cat.append(fold_rmsle_train_cat)
    print(f"Fold CatBoost (Final) {fold + 1} RMSLE on valid data: {fold_rmsle_valid_cat:.5f}")
    print(f"Fold CatBoost (Final) {fold + 1} RMSLE on train data: {fold_rmsle_train_cat:.5f}")

# It is much more reliable to calculate the error on the already calculated data, rather than averaging it over 5 fouls.
overall_rmsle_valid_cat = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_val_cat)))
overall_rmsle_train_cat = np.sqrt(mean_squared_log_error(np.expm1(np.log1p(y_full)), np.expm1(y_pred_train_cat)))
print(f"\nğŸ�¯ Overall CV CatBoost (Final) RMSLE on valid data: {overall_rmsle_valid_cat:.5f}")
print(f"\nğŸ�¯ Overall CV CatBoost (Final) RMSLE on train data: {overall_rmsle_train_cat:.5f}")
# Since the predicted data on the test set was summed up every fold (5), therefore we divide our sum by the number of folds
y_pred_test_cat /= 5


avg_train_loss = np.mean(train_loss_history, axis=0)
avg_val_loss = np.mean(val_loss_history, axis=0)
plt.figure(figsize=(10,5))
plt.plot(avg_train_loss, label="Training RMSLE")
plt.plot(avg_val_loss, label="Validation RMSLE")
plt.axvline(np.argmin(avg_val_loss), color="gray", linestyle="--", label="Best Iteration")
plt.xlabel("Number of trees")
plt.ylabel("RMSLE")
plt.title('Training and Validation RMSLE Across Folds CatBoost')
plt.legend()
plt.show()


estimators = [('XGBoost', model_1),
              ('LightGBM', model_2),
              ('CatBoost', model_3)]
model_4 = StackingRegressor(estimators = estimators, final_estimator = Lasso(alpha = 0.001))


# (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
# We will not use this model
# RMSLE on test data when stacking = 0.05775 calories
# The best forecast based on test data ({xgb+lgbm+cat}/3) = 0.05730 calories
'''
cv = KFold(n_splits = 5, random_state = 41, shuffle = True)
rmsle_scores_valid_stacking = []
rmsle_scores_train_stacking = []
y_pred_val_stacking = np.zeros(len(X_full))
y_pred_train_stacking = np.zeros(len(X_full))
y_pred_test_stacking = np.zeros(len(X_test))

for fold, (idx_train, idx_valid) in enumerate(cv.split(X_full[predictors], np.log1p(y_full))):
    print(f"\n Fold Stacking (Final) {fold + 1}")
    # Separating the training and radiation data from the source dataset
    X_train = X_full[predictors].iloc[idx_train].copy()
    X_valid = X_full[predictors].iloc[idx_valid].copy()
    X_test  = X_test[predictors].copy()
    y_train = np.log1p(y_full.iloc[idx_train].copy())
    y_valid = np.log1p(y_full.iloc[idx_valid].copy())

    model_4.fit(X_train, y_train)
    
    # We try to extract maximum information from the radiation, so we calculate predictions based on the training data.
    y_pred_val_stacking[idx_valid] = model_4.predict(X_valid)
    y_pred_train_stacking[idx_train] = model_4.predict(X_train)
    y_pred_test_stacking += model_4.predict(X_test)

    # We define RMSLE on both training and validation sets
    # The variables rmsle_scores_valid and rmsle_scores_train will be used for plotting graphs.
    fold_rmsle_valid_stacking = root_mean_squared_log_error(np.expm1(y_valid), np.expm1(y_pred_val_stacking[idx_valid]))
    fold_rmsle_train_stacking = root_mean_squared_log_error(np.expm1(y_train), np.expm1(y_pred_train_stacking[idx_train]))
    rmsle_scores_valid_stacking.append(fold_rmsle_valid_stacking)
    rmsle_scores_train_stacking.append(fold_rmsle_train_stacking)
    print(f"Fold Stacking (Final) {fold + 1} RMSLE on valid data: {fold_rmsle_valid_stacking:.5f}")
    print(f"Fold Stacking (Final) {fold + 1} RMSLE on train data: {fold_rmsle_train_stacking:.5f}")
y_pred_test_stacking /= 5
'''


y_final = (y_pred_test_xgb + y_pred_test_lgbm + y_pred_test_cat)/3


# The Heteroscedasticity Test
X_train_const = add_constant(X_full[predictors])
y_train_final_pred = (y_pred_train_xgb + y_pred_train_lgbm + y_pred_train_cat)/3
residuals_train = y_full - y_train_final_pred
if X_train_const.shape[1] < 2:
    raise ValueError("After removing the features, there are less than 2 variables left. The test is not possible.")
bp_test = het_breuschpagan(residuals_train, X_train_const)
print(f"\nBreusch-Pagan Test: p-value = {bp_test[1]:.10f}")
if bp_test[1] < 0.05:
    print("Heteroskedasticity is present (p < 0.05)")
else:
    print("No heteroskedasticity was detected")


# The Darbin-Watson autocorrelation test
dw_test = durbin_watson(residuals_train)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}")
if dw_test < 1.5 or dw_test > 2.5:
    print("Autocorrelation of residues detected")
else:
    print("No residue autocorrelation was detected")


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub.head()


sub['Calories'] = np.clip(np.expm1((y_final)/(1)), 1, 314)
sub.head()


sub.info()


sub.to_csv('submission.csv', index = False)

