
pip install imbalanced-learn



import warnings
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

from xgboost import XGBClassifier
import xgboost as xgb 

from sklearn.metrics import (
    accuracy_score, fbeta_score, f1_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_csv('/kaggle/input/2025-ness-statathon/train_2025.csv')
test = pd.read_csv('/kaggle/input/2025-ness-statathon/test_2025.csv')
submission = pd.read_csv('/kaggle/input/2025-ness-statathon/sample_submission.csv')


test_copy = test.copy()


df


test


df.describe(include='all')


df = df.drop(['claim_day_of_week','claim_number','zip_code','claim_date'],axis=1)
test = test.drop(['claim_day_of_week','claim_number','zip_code','claim_date'],axis=1)


df.info()


x_values = df[['age_of_driver', 'safty_rating', 'annual_income', 'past_num_of_claims',
               'claim_est_payout', 'age_of_vehicle', 'vehicle_weight', 'vehicle_price']]



plt.figure(figsize=(10, 8))
sns.countplot(x='fraud', data=df)
plt.title('Fraud Distribution')
plt.xlabel('Fraud')
plt.ylabel('Count')
plt.show()



plt.figure(figsize=(10, 8))
corr_matrix = df.select_dtypes(include='number').corr()
sns.heatmap(corr_matrix,cmap='Blues')
plt.show()



fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))

for i, x_value in enumerate(x_values):
    ax = axes.flatten()[i] 
    sns.boxplot(data=df, x='fraud', y=x_value, hue='fraud', ax=ax,palette=["#006992", "#ff7d00"])
    ax.set_title(f'{x_value.capitalize()}')
    ax.set_ylabel(x_value.capitalize())
plt.tight_layout()
plt.show()



fig, axis = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))

for ax, x_value in zip(axis.flat, x_values):
    sns.histplot(data=df, x=x_value, hue="fraud", kde=True, ax=ax, bins=20, alpha=0.6)
    ax.set_title(f'{x_value.capitalize()}')
plt.tight_layout()
plt.show()



fig, axis = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
y_value = 'vehicle_price'

for ax, x_value in zip(axis.flat, x_values):
    sns.scatterplot(data=df, x=x_value, y=y_value, hue='fraud', ax=ax)
    ax.set_title(f'{x_value.capitalize()} and {y_value.capitalize()}')

plt.tight_layout()
plt.show()


df['price_weight_ratio'] = df['vehicle_price'] / df['vehicle_weight']
df['driver_profile_score'] = (df['age_of_driver'] * 0.2) + (df['safty_rating'] * 0.5) + (df['high_education_ind'] * 20) - (df['past_num_of_claims'] * 10)

test['price_weight_ratio'] = test['vehicle_price'] / test['vehicle_weight']
test['driver_profile_score'] = (test['age_of_driver'] * 0.2) + (test['safty_rating'] * 0.5) + (test['high_education_ind'] * 20) - (test['past_num_of_claims'] * 10)


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
numerical_columns = [col for col in numerical_columns if col != 'fraud']

categorical_columns = df.select_dtypes(include=['object', 'category']).columns
categorical_columns = [col for col in categorical_columns if col != 'fraud']


def remove_outliers(df):
    cleaned_df = df.copy()
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype in ['float64', 'int64']:

            Q1 = cleaned_df[col].quantile(0.15)
            Q3 = cleaned_df[col].quantile(0.85)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            cleaned_df[col] = cleaned_df[col].apply(
                lambda x: x if pd.isnull(x) or (lower_limit <= x <= upper_limit) else None
            )
    
    return cleaned_df

df = remove_outliers(df)
test = remove_outliers(test)


def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':  
            df[column] = df[column].fillna(df[column].mode()[0]) 
        else:
            df[column] = df[column].fillna(df[column].mean())     
    return df

df = fill_missing_values(df)
test = fill_missing_values(test)


def one_hot_encode_columns(df, categorical_columns):
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df_encoded

df = one_hot_encode_columns(df,categorical_columns)
test = one_hot_encode_columns(test,categorical_columns)


test


numerical_columns


def scale_numerical_features(df, numerical_columns):
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

df = scale_numerical_features(df,numerical_columns)
test = scale_numerical_features(test,numerical_columns)


X = df.drop('fraud', axis=1)
y = df['fraud']


adasyn = ADASYN(sampling_strategy='auto',n_neighbors=5,random_state=42)
X, y = adasyn.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


# def objective(trial):
#    param = {
#        'objective': 'binary:logistic', 
#        'eval_metric': 'logloss', 
#        'max_depth': trial.suggest_int('max_depth', 3, 12),  
#        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),  
#        'n_estimators': trial.suggest_int('n_estimators', 50, 300),  
#        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
#        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
#        'gamma': trial.suggest_uniform('gamma', 0, 10), 
#        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
#    }
 
#    model = xgb.XGBClassifier(**param)
#    model.fit(X_train, y_train)
#    y_pred = model.predict(X_test)
#    accuracy = accuracy_score(y_test, y_pred)
#    return accuracy 

# study = optuna.create_study(direction='maximize') 
# study.optimize(objective, n_trials=500) 

# print(f"Best hyperparameters: {study.best_params}")
# print(f"Best accuracy: {study.best_value}")


best_params = {
    'max_depth': 12,
    'learning_rate': 0.06621299537054244,
    'n_estimators': 296,
    'subsample': 0.875470345850089,
    'colsample_bytree': 0.8210871979872257,
    'gamma': 0.0015885082070873358,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}


model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Metrics
accuracy = accuracy_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Results
print(f"Accuracy: {accuracy:.4f}")
print(f"F2-score: {f2:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Recall: {recall:.4f}")


# ROC metrics 
prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, prob)
roc_auc = roc_auc_score(y_test, prob)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



# Precision-Recall metrics
precision, recall_pr, _ = precision_recall_curve(y_test, prob)
avg_precision = average_precision_score(y_test, prob)


plt.figure(figsize=(10, 8))
plt.plot(recall_pr, precision, label=f'AP = {avg_precision:.4f}', color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(12, 8))
xgb.plot_importance(model)
plt.title('Feature Importance')
plt.grid(True)
plt.show()


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.grid(False)
plt.show()


y_test = model.predict(test).flatten()

submission = pd.DataFrame({
    'claim_number': test_copy['claim_number'],
    'target': y_test
})
submission.to_csv('submission4021', index=False)

