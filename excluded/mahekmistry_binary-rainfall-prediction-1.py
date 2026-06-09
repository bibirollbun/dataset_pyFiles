import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


print(train_df.info())


print(train_df.describe())


print(train_df.isnull().sum())


print(test_df.isnull().sum())


train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)


X = train_df.drop(columns=['id', 'rainfall'])  # Dropping ID & Target
y = train_df['rainfall']  # Target
X_test = test_df.drop(columns=['id'])


# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='auc', random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
}


auc_scores = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_pred)
    auc_scores[name] = auc_score
    print(f'{name} AUC: {auc_score}')


# Select best model
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]
print(f'Best Model: {best_model_name}')


test_predictions = best_model.predict_proba(X_test_scaled)[:, 1]


submission = pd.DataFrame({"id": test_df["id"], "rainfall": test_predictions})
submission.to_csv("sample_submission1_lor.csv", index=False)
print("Submission file saved!")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# Create interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)


# Convert back to DataFrame
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly.get_feature_names_out(X.columns))



# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_poly_df, y, test_size=0.2, random_state=42, stratify=y)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_poly_df)


# Model Training
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='auc', random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
}


auc_scores = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train_resampled)
    y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_pred)
    auc_scores[name] = auc_score
    print(f'{name} AUC: {auc_score}')


# Select best model
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]
print(f'Best Model: {best_model_name}')


# Make final predictions on test set
test_predictions = best_model.predict_proba(X_test_scaled)[:, 1]


# Save submission file
submission = pd.DataFrame({"id": test_df["id"], "rainfall": test_predictions})
submission.to_csv("sample_submission2_lor.csv", index=False)
print("Submission file saved!")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import optuna


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# EDA - Checking missing values & distributions
print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())


# Handling missing values (if any)
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

# Define features and target
X = train_df.drop(columns=['id', 'rainfall'])  # Dropping ID & Target
y = train_df['rainfall']  # Target
X_test = test_df.drop(columns=['id'])


# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)



# Hyperparameter tuning using Optuna
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'auc',
        'use_label_encoder': False
    }
    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train_scaled, y_train_resampled)
    y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
    return roc_auc_score(y_val, y_val_pred)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=20)
best_params_xgb = study_xgb.best_params
print(f'Best XGBoost params: {best_params_xgb}')


# Train XGBoost with best parameters
best_xgb = xgb.XGBClassifier(**best_params_xgb, random_state=42)
best_xgb.fit(X_train_scaled, y_train_resampled)



# Predictions using best XGBoost model
test_predictions = best_xgb.predict_proba(X_test_scaled)[:, 1]


# Save submission file
submission = pd.DataFrame({"id": test_df["id"], "rainfall": test_predictions})
submission.to_csv("submission1_xgb_optuna.csv", index=False)
print("Submission file saved!")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# EDA - Checking missing values & distributions
print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())

# Handling missing values (if any)
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

# Define features and target
X = train_df.drop(columns=['id', 'rainfall'])  # Dropping ID & Target
y = train_df['rainfall']  # Target
X_test = test_df.drop(columns=['id'])


# Feature Selection using RFE
log_reg = LogisticRegression(max_iter=500, class_weight='balanced')
rfe = RFE(log_reg, n_features_to_select=10)  # Selecting top 10 features
X_selected = rfe.fit_transform(X, y)
X_test_selected = rfe.transform(X_test)


# Convert back to DataFrame
selected_features = X.columns[rfe.support_]
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)


# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_selected_df, y, test_size=0.2, random_state=42, stratify=y)



# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_selected_df)


# Model Training
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='auc', random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
}


auc_scores = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train_resampled)
    y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_pred)
    auc_scores[name] = auc_score
    print(f'{name} AUC: {auc_score}')


# Select best model
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]
print(f'Best Model: {best_model_name}')



# Make final predictions on test set
test_predictions = best_model.predict_proba(X_test_scaled)[:, 1]


# Save submission file
submission = pd.DataFrame({"id": test_df["id"], "rainfall": test_predictions})
submission.to_csv("submission3_lor_rfe.csv", index=False)
print("Submission file saved!")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# EDA - Checking missing values & distributions
print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())

# Handling missing values (if any)
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

# Define features and target
X = train_df.drop(columns=['id', 'rainfall'])  # Dropping ID & Target
y = train_df['rainfall']  # Target
X_test = test_df.drop(columns=['id'])

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to find optimal number of components
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model Training
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='auc', random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
}

auc_scores = {}
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_pred)
    auc_scores[name] = auc_score
    print(f'{name} AUC: {auc_score}')


# Select best model
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]
print(f'Best Model: {best_model_name}')




# Make final predictions on test set
test_predictions = best_model.predict_proba(X_test_pca)[:, 1]

# Save submission file
submission = pd.DataFrame({"id": test_df["id"], "rainfall": test_predictions})
submission.to_csv("submission4_lor_pca.csv", index=False)
print("Submission file saved!")







