import numpy as np
import pandas as pd

import optuna
from xgboost import XGBRegressor

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error,r2_score

import joblib
import os




TRAIN_FILE_PATH = "/kaggle/input/playground-series-s5e5/train.csv"
TEST_FILE_PATH = "/kaggle/input/playground-series-s5e5/test.csv"


train_df = pd.read_csv(TRAIN_FILE_PATH)
test_df = pd.read_csv(TEST_FILE_PATH)


train_df.head()


test_df.head()


train_df.info()


train_df.describe().T


numerical_columns = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Calories"]
categorical_columns = ["Sex"]
train_df = train_df[numerical_columns+categorical_columns]


for col in numerical_columns: 
    plt.figure(figsize=(10, 5))
    plt.hist(train_df[col], bins=50, edgecolor='black')
    plt.title(f"Histogram of {col} values")
    plt.xlabel(f"{col} Value")
    plt.grid(True)
    plt.show()


female_count = train_df[categorical_columns].value_counts()["female"]
male_count = train_df[categorical_columns].value_counts()["male"]


labels = ['Female', 'Male']
sizes = [female_count, male_count]

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Sex Distribution')
plt.axis('equal')
plt.show()


train_df.isna().sum()


test_df.isna().sum()


columns = list(train_df.columns)
columns.remove("Calories")
columns.remove("Sex")


columns


def add_feature_cross_terms(df, features):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    new_features = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            new_col = f"{f1}_x_{f2}"
            df[new_col] = df[f1] * df[f2]
            new_features.append(new_col)

    print("New cross term columns:")
    print(new_features)
    return df, new_features


import itertools
def add_interaction_features(df, features):
    df_new = df.copy()
    new_features = []
    for f1, f2 in itertools.combinations(features, 2):
        col1 = f"{f1}_plus_{f2}"
        col2 = f"{f1}_minus_{f2}"
        col3 = f"{f2}_minus_{f1}"
        col4 = f"{f1}_div_{f2}"
        col5 = f"{f2}_div_{f1}"
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-5)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-5)
        new_features.extend([col1, col2, col3, col4, col5])
    

    
    return df_new,new_features




def add_statistical_features(df, features):
    df_new = df.copy()
    df_new["row_mean"] = df[features].mean(axis=1)
    df_new["row_std"] = df[features].std(axis=1)
    df_new["row_max"] = df[features].max(axis=1)
    df_new["row_min"] = df[features].min(axis=1)
    df_new["row_median"] = df[features].median(axis=1)
    new_features = ["row_mean", "row_std", "row_max", "row_min", "row_median"]
    return df_new,new_features


train_df,new_features = add_feature_cross_terms(train_df,columns)
test_df,new_features = add_feature_cross_terms(test_df,columns)


train_df,new_features_interaction = add_interaction_features(train_df,columns)
test_df,new_features_interaction = add_interaction_features(test_df,columns)


train_df,new_features_statistical = add_interaction_features(train_df,columns)
test_df,new_features_statistical = add_interaction_features(test_df,columns)


train_df[numerical_columns+new_features+new_features_interaction].corr()


correlation_matrix = train_df[numerical_columns+new_features].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Analysis between numerical variables")
plt.tight_layout()
plt.show()


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

for i in range(0, len(numerical_features), 2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    scatter1 = ax1.scatter(train_df[numerical_features[i]], train_df['Calories'],
                          c=train_df[numerical_features[i]], cmap='viridis')
    ax1.set_xlabel(numerical_features[i])
    ax1.set_ylabel('Calories')
    ax1.set_title(f'Relationship between {numerical_features[i]} and Calories')
    plt.colorbar(scatter1, ax=ax1)
    
    scatter2 = ax2.scatter(train_df[numerical_features[i+1]], train_df['Calories'],
                          c=train_df[numerical_features[i+1]], cmap='viridis')
    ax2.set_xlabel(numerical_features[i+1])
    ax2.set_ylabel('Calories')
    ax2.set_title(f'Relationship between {numerical_features[i+1]} and Calories')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Sex', y='Calories', data=train_df, palette={'female': '#FF69B4', 'male': '#4169E1'})
plt.title('Calories Distribution by Sex')
plt.show()



target = "Calories"
target = train_df[target]
features = train_df.drop(columns=["Calories"])


label_encoder = LabelEncoder()
features['Sex'] = label_encoder.fit_transform(features['Sex'])



features.head()


target.head()


X = features.copy()
y = target.copy()

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'tree_method': 'gpu_hist',  # Use GPU acceleration
        'predictor': 'gpu_predictor'  # Use GPU for predictions
    }
    
    model = XGBRegressor(**params, random_state=42)
    
    scores = cross_val_score(
        model, X, y,
        cv=5,
        scoring='neg_root_mean_squared_error'
    )
    
    return -scores.mean()

"""

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("\nBest parameters:", best_params)
print("Best RMSE:", -study.best_value)

best_params.update({
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
})
best_xgb = XGBRegressor(**best_params, random_state=42)
best_xgb.fit(X, y)
"""



best_params = {
    'n_estimators': 546,
    'max_depth': 10,
    'learning_rate': 0.019948578686450974,
    'subsample': 0.7108735360872366,
    'colsample_bytree': 0.7641958633892596,
    'min_child_weight': 2,
    'gamma': 0.0019653264747787856,
    'tree_method': 'gpu_hist',       # âœ… GPU kullanÄ±mÄ±nÄ± aktif eder
    'predictor': 'gpu_predictor',    # (opsiyonel ama Ã¶nerilir)
    'verbosity': 1                   # (opsiyonel: eÄŸitim sÄ±rasÄ±nda log verir)
}



X = features.copy()
y = target.copy()

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

rmlse_scores = []
r2_scores = []

def rmlse(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

if not os.path.exists('models'):
    os.makedirs('models')

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    model_path = os.path.join('models', f'xgboost_fold_{fold+1}.joblib')
    joblib.dump(model, model_path)
    
    y_pred = model.predict(X_val)
    
    rmlse_score = rmlse(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    rmlse_scores.append(rmlse_score)
    r2_scores.append(r2)
    
    print(f'Fold {fold+1}:')
    print(f'RMLSE: {rmlse_score:.4f}')
    print(f'R2 Score: {r2:.4f}\n')

print('Average Metrics:')
print(f'Average RMLSE: {np.mean(rmlse_scores):.4f} (+/- {np.std(rmlse_scores):.4f})')
print(f'Average R2: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})')



ids = test_df[["id"]]
features_test = test_df.drop(columns=["id"]).copy()
features_test['Sex'] = label_encoder.fit_transform(features_test['Sex'])


model = XGBRegressor(**best_params, random_state=42)
model.fit(X,y)


cols_X = X.columns.tolist()
cols = features_test.columns.tolist()
features_test = features_test[cols_X]
features_test


predictions_test = model.predict(features_test)


predictions_test


df = pd.DataFrame(ids)
df.columns=["id"]
df["Calories"] = predictions_test
df.head()
df.to_csv("submission_df_raw_cross_terms_feature_engineering_xgb.csv",index=False)


plt.hist(predictions_test,bins=100)
plt.title("Test Preds Histogram")
plt.show()


