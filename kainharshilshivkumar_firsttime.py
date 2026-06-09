import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
import lightgbm as lgb



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

ids = test['id']


num_cols = train.select_dtypes(include = 'number').columns.tolist()
cat_cols = train.select_dtypes(include = 'object').columns

imputer = IterativeImputer(max_iter = 10, random_state = 42, initial_strategy = 'mean')
train[num_cols] = imputer.fit_transform(train[num_cols])
train[cat_cols] = train[cat_cols].fillna('Missing')

# print(train[num_cols].isnull().sum())

# ---------------------------------------------------------------------------------------------

num_cols = test.select_dtypes(include = 'number').columns.tolist()
cat_cols = test.select_dtypes(include = 'object').columns

imputer = IterativeImputer(max_iter = 10, random_state = 42, initial_strategy = 'mean')
test[num_cols] = imputer.fit_transform(test[num_cols])
test[cat_cols] = test[cat_cols].fillna('Missing')

# test.isnull().sum()


for col in ['Stage_fear', 'Drained_after_socializing']:
    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=train,
        x=col,
        hue='Personality',
        palette=['#1f77b4', '#ff7f0e'],
        edgecolor='black'
    )
    
    plt.title(f'Distribution of {col} by Personality', fontsize=14)
    plt.xlabel(f'{col} (0=No, 1=Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(
        title='Personality',
        labels=['Introvert (0)', 'Extrovert (1)'],
        loc='upper right'
    )
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(6, 4))
sns.heatmap(train[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.show()


plt.figure(figsize=(14, 6))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(data=train, y=col)
    plt.title(f"Boxplot: {col}")
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(
        data=train,
        x='Personality',
        y=col,
        palette='Set2',
        linewidth=1.2,
        fliersize=4
    )
    plt.title(f'{col} by Personality', fontsize=14, fontweight='semibold', color='#2E4057')
    plt.xlabel('Personality', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


train['Stage_fear'] = train['Stage_fear'].replace({
    'No' : 0,
    'Yes' : 1,
    'Missing' : 2
})

train['Drained_after_socializing'] = train['Drained_after_socializing'].replace({
    'No' : 0,
    'Yes' : 1,
    'Missing' : 2
})

# -------------------------------------------------------------------------------------

test['Stage_fear'] = test['Stage_fear'].replace({
    'No' : 0,
    'Yes' : 1,
    'Missing' : 2
})

test['Drained_after_socializing'] = test['Drained_after_socializing'].replace({
    'No' : 0,
    'Yes' : 1,
    'Missing' : 2
})


train = train.drop('id', axis = 1)

scaler = StandardScaler()

train_data = train.drop(columns = ['Personality'], axis = 1)
scaler.fit_transform(train_data)
# train_data.head()

# -----------------------------------------------------------------

test_data = test.drop('id', axis = 1)
scaler.transform(test_data)
# test_data.head()


encoder = LabelEncoder()
y = encoder.fit_transform(train['Personality'])  

X = train_data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(n_estimators = 200, random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    print(f"\n Classification Report for {name}:")
    print(classification_report(y_valid, y_pred, target_names=encoder.classes_))
    print('-------------------------------------------')


# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='logloss',
#     use_label_encoder=False,
#     random_state=42
# )

# param_grid = {
#     'n_estimators': [500, 750, 1000, 1250, 1500],
#     'learning_rate': [0.005, 0.006, 0.006358, 0.007, 0.01],
#     'max_depth': [6, 7, 8, 9, 10],
#     'subsample': [0.8, 0.85, 0.8854, 0.9, 1.0],
#     'colsample_bytree': [0.5, 0.6, 0.7],
#     'reg_lambda': [0.6, 0.7, 0.8, 0.8295, 0.9, 1.0],
#     'reg_alpha': [4, 5, 5.5, 5.5149, 6, 7],
#     'gamma': [0.0, 0.02, 0.0395, 0.05, 0.1],
#     'min_child_weight': [1, 2, 3, 4]
# }

# xgb_search = RandomizedSearchCV(
#     estimator=xgb_model,
#     param_distributions=param_grid,
#     n_iter=100,  
#     scoring='accuracy',
#     cv=5,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# xgb_search.fit(X_train, y_train)

# # print("Best XGBoost Parameters:")
# # print(xgb_search.best_params_)

# # ----------------------------------------------------------------------------------

# lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)

# lgb_param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 5, 7, 10, 15, 20],
#     'num_leaves': [20, 31, 40, 50],
#     'min_child_samples': [10, 20, 30],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }

# lgb_search = RandomizedSearchCV(
#     estimator=lgb_model,
#     param_distributions=lgb_param_grid,
#     n_iter=25,
#     scoring='accuracy',
#     cv=6,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# # lgb_search.fit(X_train, y_train)
# # print(" Best LightGBM Parameters:", lgb_search.best_params_)

# # ------------------------------------------------------------------------------------------------------

# knn_model = KNeighborsClassifier()

# knn_param_grid = {
#     'n_neighbors': list(range(3, 21)),
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }

# knn_search = RandomizedSearchCV(
#     estimator=knn_model,
#     param_distributions=knn_param_grid,
#     n_iter=20,
#     scoring='accuracy',
#     cv=6,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# # knn_search.fit(X_train, y_train)
# # print("Best KNN Parameters:", knn_search.best_params_)


fi_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_estimators=500,
    learning_rate=0.007,
    max_depth=6,
    subsample=0.8854,
    colsample_bytree=0.7,
    reg_lambda=0.7,
    reg_alpha=5.5149,
    gamma=0.02,
    min_child_weight=4
)


fi_model.fit(X_train, y_train)
y_pred = fi_model.predict(X_valid)
print(f"\n Classification Report for {name}:")
print(classification_report(y_valid, y_pred, target_names=encoder.classes_))
print('-------------------------------------------')


test_pred = fi_model.predict(test_data)
test_pred_labels = encoder.inverse_transform(test_pred)

print(test_pred_labels[:10])


importances = fi_model.feature_importances_
features = X_valid.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)


plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Feature Importance - XGBoost")
plt.tight_layout()
plt.show()


final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_estimators=500,
    learning_rate=0.007,
    max_depth=6,
    subsample=0.8854,
    colsample_bytree=0.7,
    reg_lambda=0.7,
    reg_alpha=5.5149,
    gamma=0.02,
    min_child_weight=4
)

fitting_final = final_model.fit(X_train, y_train)
test_pred_final = final_model.predict(test_data)
test_pred_labels_final = encoder.inverse_transform(test_pred_final)

submission = pd.DataFrame({
    "id": ids,    
    "price": test_pred_labels_final
})

submission.to_csv("submission.csv", index = False)
print(submission.head())

