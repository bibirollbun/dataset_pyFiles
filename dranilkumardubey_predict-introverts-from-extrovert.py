# === 1. Imports ===
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_log_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# Combine train + test for consistent transformation
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# Encode target
train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# Concatenate for joint preprocessing
df_combined = pd.concat([train.drop(columns='Personality'), test], axis=0).reset_index(drop=True)

# Encode object (categorical) features
for col in df_combined.columns:
    if df_combined[col].dtype == 'object':
        df_combined[col] = LabelEncoder().fit_transform(df_combined[col].astype(str))

# Handle missing values: impute all numeric/categorical with median
imputer = SimpleImputer(strategy='median')
df_combined_imputed = pd.DataFrame(imputer.fit_transform(df_combined), columns=df_combined.columns)

# Now safe to apply PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df_combined_imputed)
df_combined_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out())

# Remove low variance features
selector = VarianceThreshold(threshold=0.01)
df_combined_selected = selector.fit_transform(df_combined_poly)

# Split back into train/test
X = df_combined_selected[:len(train)]
X_test = df_combined_selected[len(train):]
y = train['Personality']



def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model = XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=5)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y.iloc[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(accuracy_score(y.iloc[val_idx], preds))
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)
xgb_best = XGBClassifier(**study.best_params)



# Base models
cat = CatBoostClassifier(verbose=0, random_state=42)
lgb = LGBMClassifier(random_state=42)

stack_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_best),
        ('cat', cat),
        ('lgb', lgb)
    ],
    final_estimator=XGBClassifier(n_estimators=200, learning_rate=0.1),
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1,
    passthrough=True
)

stack_model.fit(X, y)



cv = StratifiedKFold(n_splits=5)
y_preds = cross_val_predict(stack_model, X, y, cv=cv, method='predict')

print("CV Accuracy:", accuracy_score(y, y_preds))
print("CV F1 Score:", f1_score(y, y_preds))
print("CV Recall:", recall_score(y, y_preds))
print("CV MSLE:", mean_squared_log_error(y, y_preds))



stack_model.fit(X, y)
final_preds = stack_model.predict(X_test)

# Convert predictions to labels
final_labels = ['Introvert' if i == 0 else 'Extrovert' for i in final_preds]
sample_submission['Personality'] = final_labels
sample_submission.to_csv('submission.csv', index=False)



xgb_best.fit(X, y)
importances = pd.Series(xgb_best.feature_importances_, index=poly.get_feature_names_out()[selector.get_support()])
importances.sort_values().tail(20).plot(kind='barh', figsize=(10,6))
plt.title("Top 20 XGBoost Feature Importances")
plt.show()





