import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import optuna
import seaborn as sns
import numpy as np


%%time

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train.head()


print(train.isnull().sum())


print(test.isnull().sum())


print(train.duplicated().sum())


print(test.duplicated().sum())


test.fillna(test.median(numeric_only=True), inplace=True)


train.describe()


test.describe()


plt.figure(figsize=(12, 8))
corr_matrix = train.drop(columns=["id"]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Heatmap Korelasi Fitur")
plt.show()



train['rainfall'].value_counts()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['day'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Day')
plt.title('Boxplot Day vs Rainfall')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['pressure'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Pressure')
plt.title('Boxplot Pressure vs Rainfall')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['maxtemp'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Max Temp')
plt.title('Boxplot Max Temp vs Rainfall')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['temparature'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Temparature')
plt.title('Boxplot Temparature vs Rainfall')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['mintemp'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Min Temp')
plt.title('Boxplot Min Temp vs Rainfall')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['dewpoint'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Dewpoint')
plt.title('Boxplot Dewpoint vs Rainfall')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['humidity'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Humidity')
plt.title('Boxplot Humidity vs Rainfall')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['cloud'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Cloud')
plt.title('Boxplot Cloud vs Rainfall')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['sunshine'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Sunshine')
plt.title('Boxplot Sunshine vs Rainfall')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['winddirection'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Wind Direction')
plt.title('Boxplot Wind Direction vs Rainfall')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['rainfall'], y=train['windspeed'], palette='coolwarm')
plt.xlabel('Rainfall')
plt.ylabel('Wind Speed')
plt.title('Boxplot Wind Speed vs Rainfall')
plt.show()


train["temp_range"] = train["maxtemp"] - train["mintemp"]
train["humidity_dew_ratio"] = train["humidity"] / (train["dewpoint"] + 1e-6)
train["wind_effect"] = train["windspeed"] * train["winddirection"]
train['temp_range'] = train['maxtemp'] - train['mintemp']
train['cloud_sun_ratio'] = train['cloud'] / (train['sunshine'] + 0.00001)
train['temp_humidity'] = train['temparature'] * train['humidity']
train['dewpoint_temparature'] = train['temparature'] - train['dewpoint']
train['cloud_prosentase'] = train['cloud'] / 100
train['sunshine_prosentase'] = train['sunshine'] / 100
train['cloud_speed_humidity_dewpoint'] = train['cloud'] + train['windspeed'] + train['humidity'] + train['dewpoint']

test["temp_range"] = test["maxtemp"] - test["mintemp"]
test["humidity_dew_ratio"] = test["humidity"] / (test["dewpoint"] + 1e-6)
test["wind_effect"] = test["windspeed"] * test["winddirection"]
test['temp_range'] = test['maxtemp'] - test['mintemp']
test['cloud_sun_ratio'] = test['cloud'] / (test['sunshine'] + 0.00001)
test['temp_humidity'] = test['temparature'] * test['humidity']
test['dewpoint_temparature'] = test['temparature'] - test['dewpoint']
test['cloud_prosentase'] = test['cloud'] / 100
test['sunshine_prosentase'] = test['sunshine'] / 100
test['cloud_speed_humidity_dewpoint'] = test['cloud'] + test['windspeed'] + test['humidity'] + test['dewpoint']


X = train.drop(columns=["id", "rainfall"])
y = train["rainfall"]


test_final = test.drop(columns=["id"])


# Define the objective function for Optuna
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01),
        'n_estimators': trial.suggest_int('n_estimators', 400, 800),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
    }
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=111,
        **params
    )
    
    # Use Stratified K-Fold Cross Validation as an alternative to OOB
    kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=111)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
    
    return cv_scores.mean()  # Mean AUC-ROC as the optimization metric

# Run Optuna hyperparameter optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

# Best parameters found by Optuna
best_params = study.best_params
print("Best Parameters:", best_params)

# Train the final model using the best parameters
optuna_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=111,
    **best_params
)

optuna_model.fit(X, y)

# OOB-like estimate using cross-validation on the full dataset
final_cv_scores = cross_val_score(optuna_model, X, y, cv=7, scoring='roc_auc')
print(f"OOB-like AUC-ROC Score (Cross-Validation): {final_cv_scores.mean():.4f}")


feature_importances = optuna_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance dari XGBoost')
plt.gca().invert_yaxis()
plt.show()


sample.head()


sample_sub = sample.drop(columns=["rainfall"])


sample_sub["rainfall"] = optuna_model.predict_proba(test_final)[:, 1]
sample_sub.to_csv('submission.csv', index=False)
sample_sub.tail()

