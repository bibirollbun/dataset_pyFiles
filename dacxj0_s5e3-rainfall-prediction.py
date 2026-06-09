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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
original_df = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")


train_df.info()


test_df.info()


train_df.describe()


train_df.shape


original_df.shape


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, RidgeClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif,RFE
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns


base_corr = train_df.corr()
sns.heatmap(base_corr, cmap="jet", annot=False)


X_base = train_df.drop(["id", "rainfall"], axis=1)
y_base = train_df["rainfall"]

X_train_base, X_valid_base, y_train_base, y_valid_base = train_test_split(X_base, y_base, test_size=0.3, random_state=42)

print(X_train_base.shape, X_valid_base.shape, y_train_base.shape, y_valid_base.shape)


base_model = LogisticRegression(max_iter=150, random_state=42, n_jobs=-1, class_weight="balanced")

base_model.fit(X_train_base, y_train_base)


base_pred = base_model.predict(X_valid_base)

roc_auc_score(y_valid_base, base_pred)


cross_val_score(base_model, X_train_base, y_train_base, cv=5, scoring="roc_auc", n_jobs=-1)


print(classification_report(y_valid_base, base_pred))


test_df['winddirection'] = test_df['winddirection'].fillna( test_df['winddirection'].mean())


X_test_base = test_df.drop("id", axis=1)

y_pred_result_base = base_model.predict_proba(X_test_base)


base_result_df = pd.DataFrame({
    "id": test_df['id'],
    "rainfall": y_pred_result_base[:,1]
})

base_result_df.to_csv("submission.csv", index=False)


import math
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)

def apply_feature_engineering(df, test=False, last_known_rainfall=None):
    # Day of the year features
    df['sinday'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cosday'] = np.cos(2 * np.pi * df['day'] / 365)
    
    def get_season(day):
        if day <= 80 or day > 355:   # Winter
            return "Winter"
        elif 80 < day <= 172:        # Spring
            return "Spring"
        elif 172 < day <= 265:       # Summer
            return "Summer"
        else:                        # Fall
            return "Fall"
    df['season'] = df['day'].apply(get_season)
    # Interaction features
    df['temprange'] = df['maxtemp'] - df['mintemp']
    df['humiditytemp'] = df['humidity'] * df['temparature']
    df['dewpointdepression'] = df['temparature'] - df['dewpoint']
    df['cloudcategory'] = pd.cut(df['cloud'], bins=[0, 3, 6, 9], labels=['Low', 'Medium', 'High'])
    
    df['windU'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
    df['windV'] = df['windspeed'] * np.sin(np.radians(df['winddirection']))

    df['humiditysquared'] = df['humidity'] ** 2
    df['tempcloudinteraction'] = df['temparature'] * df['cloud']
    df['pressurehumidity'] = df['pressure'] * df['humidity']

    # Clusters
    features = ['pressure', 'maxtemp', 'mintemp', 'humidity', 'cloud']
    if test:
        df['weathercluster'] = kmeans.predict(df[features])
    else:
        df['weathercluster'] = kmeans.fit_predict(df[features])

  
    if not test:
        df['rainfall_lag1'] = df['rainfall'].shift(1)
        df['rainfall_rolling_mean_3'] = df['rainfall'].rolling(window=3).mean()
    else:
        df['rainfall_lag1'] = np.nan
        df['rainfall_rolling_mean_3'] = np.nan
        if last_known_rainfall is not None:
            df.loc[df.index[0], 'rainfall_lag1'] = last_known_rainfall
            df.loc[df.index[0], 'rainfall_rolling_mean_3'] = last_known_rainfall

apply_feature_engineering(train_df)
last_known_rainfall = train_df['rainfall'].iloc[-1]


num_df = train_df.select_dtypes(include=['number'])
new_corr = num_df.corr().abs()

sns.heatmap(new_corr, cmap="jet", annot=False, linewidth=5.0)


num_df = train_df.drop(["id", "rainfall"], axis=1).select_dtypes(include="number")
cat_df = train_df.select_dtypes(exclude="number")


num_features = list(num_df.columns)
cat_features = list(cat_df.columns)

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

cat_pipe = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num_feats", num_pipe, num_features),
    ("cat_feates",cat_pipe, cat_features)
], verbose_feature_names_out=True)

preprocessor.set_output(transform="pandas")


features = num_features+cat_features
print(features)
print(len(features))


from imblearn.over_sampling import RandomOverSampler
from collections import Counter

X = train_df[features]
y = train_df["rainfall"]

print(sorted(Counter(y).items()))


ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

print(sorted(Counter(y_resampled).items()))


X_transformed = preprocessor.fit_transform(X_resampled)

model = LogisticRegression(random_state=42, solver="liblinear")

selector = RFE(model, n_features_to_select=10)
X_new = selector.fit_transform(X_transformed, y_resampled)
feature_names = preprocessor.get_feature_names_out()

# Get selected feature names
selected_features = feature_names[selector.get_support()].tolist()
print("Selected Features:", selected_features)


X_train, X_valid, y_train, y_valid = train_test_split(X_transformed[selected_features], y_resampled, test_size=0.3, random_state=42)


import optuna

def objective(trial):
    C = trial.suggest_float('C', 1e-5, 1e5, log=True)
    solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42 )

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)


model = LogisticRegression(random_state=42, n_jobs=-1, **study.best_params)
model.fit(X_train, y_train)


pred = model.predict_proba(X_valid)

roc_auc_score(y_valid,pred[:, 1] )


import optuna

def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    loss = trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'])
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
    eta0 = trial.suggest_float('eta0', 1e-4, 1.0, log=True)  # Initial learning rate
    max_iter = trial.suggest_int('max_iter', 500, 2000)  # Maximum number of epochs

    # Elasticnet requires a mixing parameter (l1_ratio)
    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    else:
        l1_ratio = 0.5  # Default value (not used for other penalties)

    # Create the SGDClassifier model with the suggested hyperparameters
    model = SGDClassifier(
        alpha=alpha,
        loss=loss,
        penalty=penalty,
        learning_rate=learning_rate,
        eta0=eta0,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        random_state=42
    )


    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    return np.mean(scores)

study_sgd = optuna.create_study(direction="maximize")
study_sgd.optimize(objective, n_trials=100, show_progress_bar=True)


model_sgd = SGDClassifier(random_state=42, n_jobs=-1, **study_sgd.best_params)
model_sgd.fit(X_train, y_train)


pred = model_sgd.predict_proba(X_valid)

roc_auc_score(y_valid,pred[:, 1] )


import optuna
from xgboost import XGBClassifier

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'  # For binary/multi-class classification
    }

    model = XGBClassifier(**params)

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    return np.mean(scores)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective, n_trials=100, show_progress_bar=True)


model_xgb = XGBClassifier(n_jobs=-1, **study_xgb.best_params)
model_xgb.fit(X_train, y_train)


pred = model_xgb.predict_proba(X_valid)

roc_auc_score(y_valid,pred[:, 1] )


model_stack = StackingClassifier(
    estimators=[
        ('lr',  LogisticRegression(random_state=42, n_jobs=-1, **study.best_params)),
        ('sgd', SGDClassifier(random_state=42, n_jobs=-1, **study_sgd.best_params)),
        ('xgb', XGBClassifier(random_state=42, n_jobs=-1, **study_xgb.best_params))
    ],
    final_estimator=KNeighborsClassifier(),
    stack_method='predict_proba',
    cv=5,
    n_jobs=-1
)


model_stack.fit(X_train, y_train)


pred = model_stack.predict_proba(X_valid)

roc_auc_score(y_valid,pred[:, 1] )


apply_feature_engineering(test_df, test=True, last_known_rainfall=last_known_rainfall)
X_test = test_df.drop("id", axis=1)

X_test_transformed = preprocessor.transform(X_test)

y_pred = model_stack.predict_proba(X_test_transformed[selected_features])


base_result_df = pd.DataFrame({
    "id": test_df['id'],
    "rainfall": y_pred[:, 1] 
})

base_result_df.to_csv("submission.csv", index=False)




