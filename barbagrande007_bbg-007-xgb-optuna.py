import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import lightgbm as lgb
import xgboost as xgb

import optuna


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
submit = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.head()


train.info()


def create_summary(df):
    describe = df.describe().transpose()
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary["MissingValues"] = df.isna().sum()
    summary["UniqueValues"] = df.nunique()
    summary["Value_1"] = df.iloc[0]
    summary["Value_2"] = df.iloc[1]
    summary["Value_3"] = df.iloc[2]
    summary = pd.concat([summary, describe], axis=1)
    
    return summary

create_summary(train)


# Add features
def add_features(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['BMI'] = df['BMI'].clip(lower=15, upper=40)

    # Body Surface Area (BSA) calculation
    df['BSA'] = np.where(
        df['Sex'] == 'female',
        0.000975482 * (df['Weight'] ** 0.46) * (df['Height'] ** 1.08),
        0.000579479 * (df['Weight'] ** 0.38) * (df['Height'] ** 1.24)
    )
    
    # Weight-to-Height Ratio
    df['Weight_Height_Ratio'] = df['Weight'] / df['Height']
    
    # Duration per Age
    df['Duration_per_Age'] = df['Duration'] / df['Age']
    
    # Heart Rate to Body Temperature Ratio
    df['HeartRate_BodyTemp_Ratio'] = df['Heart_Rate'] / df['Body_Temp']
    
    # Age Group (Categorical Feature)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior'])

    df['BodyFat'] = (1.20 * df['BMI']) + (0.23 * df['Age']) - 16.2
    df['BodyFat'] = df['BodyFat'].clip(lower=0, upper=100)

    ## Binning features
    
    # Bin weights
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

    # Define bin edges
    female_bins = [0, 50, 60, 70, 80, np.inf]
    male_bins   = [0, 60, 70, 80, 90, np.inf]

    # Apply sex-specific binning
    df['Weight_Bin'] = np.where(
        df['Sex'] == "female",
        pd.cut(df['Weight'], bins=female_bins, labels=labels),
        pd.cut(df['Weight'], bins=male_bins, labels=labels)
    )

    # Height bin labels
    height_labels = ['Very Short', 'Short', 'Average', 'Tall', 'Very Tall']

    # Bin edges for height by sex
    female_height_bins = [0, 150, 160, 170, 180, np.inf]
    male_height_bins   = [0, 160, 170, 180, 190, np.inf]

    # Apply sex-specific height binning
    df['Height_Bin'] = np.where(
        df['Sex'] == "female",
        pd.cut(df['Height'], bins=female_height_bins, labels=height_labels),
        pd.cut(df['Height'], bins=male_height_bins, labels=height_labels)
    )

    # Binning BMI
    bmi_labels = ['Severely Underweight', 'Underweight', 'Normal', 'Overweight', 'Obese']
    df['BMI_Bin'] = pd.cut(
        df['BMI'],
        bins=[0, 16, 18.5, 24.9, 29.9, 40],
        labels=bmi_labels,
        right=False
    )

    return df

train = add_features(train)
test = add_features(test)
train.head()


cols = train.drop(columns=['Calories','Sex','Age_Group', 'Weight_Bin', 'Height_Bin', 'BMI_Bin']).columns.tolist()
ncols = 3
nrows = int(np.ceil(len(cols) / ncols))

fig, ax = plt.subplots(nrows, ncols, figsize=(20,15))
ax = ax.flatten()

plt.suptitle('Correlation of features with Calories', fontsize=24)

for idx, col in enumerate(cols):
    sns.scatterplot(x=train[col], y=train['Calories'], hue=train['Sex'], ax=ax[idx])

plt.show()


X = train.drop(['Calories'], axis=1)
y = train['Calories']
X_test = test.copy()


num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=X['Weight_Bin'],random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

X = preprocessor.fit_transform(X)

X_train.shape, X_val.shape, X_test.shape


# Try various models
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'HistGradientBoostingRegressor': HistGradientBoostingRegressor(),
    'LightGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
}

rmsle_scores = []

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, None)
    
    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_val)
    rmsle = np.sqrt(np.mean((log_pred - log_true) ** 2))

    rmsle_scores.append((name, rmsle))
    total_time = time.time() - start_time
    print(f"{name} - RMSLE: {rmsle:.4f} - Time: {total_time:.2f}s")

# Visualize MSE scores
rmse_df = pd.DataFrame(rmsle_scores, columns=['Model', 'RMSLE']).sort_values(by='RMSLE', ascending=True).reset_index(drop=True)
rmse_df


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_true)
    return np.sqrt(np.mean((log_pred - log_true) ** 2))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
cv = KFold(n_splits=10, shuffle=True, random_state=51)


optuna.logging.set_verbosity(optuna.logging.WARNING)

# Hyperparameter tuning for the best model
def objective(trial):
    params = {
        'objective': 'count:poisson',
        'eval_metric': 'logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        # 'tree_method': 'gpu_hist',
        'device': 'gpu',
        'random_state': 51,
    }
    
    model = xgb.XGBRegressor(**params)

    cv_score = -1 * cross_val_score(model, X_train, y_train, cv=cv, scoring=rmsle_scorer)
    avg_rmsle = np.mean(cv_score)

    return avg_rmsle


study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective, n_trials=50, show_progress_bar=True)

best_params_xgb = study_xgb.best_params
best_value_xgb = study_xgb.best_value
print("Best parameters: ", best_params_xgb)
print("Best RMSLE: ", best_value_xgb)


best_model_xgb = xgb.XGBRegressor(**best_params_xgb)
best_model_xgb.fit(X_train, y_train)
y_pred_xgb = best_model_xgb.predict(X_val)

y_pred_xgb = y_pred_xgb.clip(1, None)

    
log_pred_xgb = np.log1p(y_pred_xgb)
log_true = np.log1p(y_val)
rmsle_xgb = np.sqrt(np.mean((log_pred_xgb - log_true) ** 2))

print(f"RMSLE: {rmsle_xgb:.4f}")


xgb_model = xgb.XGBRegressor(**best_params_xgb)
xgb_model.fit(X, y)


xgb_test_pred = xgb_model.predict(X_test)
xgb_test_pred = xgb_test_pred.clip(1, None)


plt.figure(figsize=(10, 6))
plt.title('Distribution of Actual vs Predicted Calories', fontsize=20)
sns.kdeplot(xgb_test_pred, label='Predicted', color='orange')
sns.kdeplot(y, label='Actual', color='blue')
plt.legend()
plt.show()


submit['Calories'] = xgb_test_pred
submit.to_csv('submission.csv', index=False)
print("Submission file created successfully.")




