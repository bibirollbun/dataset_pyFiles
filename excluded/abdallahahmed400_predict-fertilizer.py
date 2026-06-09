import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, precision_score, recall_score
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train_df.head(10)


train_df.info()


train_df.columns


categorical_cols = train_df.select_dtypes(include=['object']).columns


categorical_cols


for col in categorical_cols:

  plt.figure(figsize=(5,5))
  sns.countplot(x=col,data=train_df)
  plt.xticks(rotation=90)
  plt.show()


for col in categorical_cols:
  print(train_df[col].value_counts())
  print('#'*40)



train_df.isnull().sum()


train_df.drop(columns=['id'], inplace=True)


X = train_df.drop(['Fertilizer Name'], axis=1)
y = train_df['Fertilizer Name']


X


scaler = StandardScaler()


num_cols = X.select_dtypes(exclude="object").columns


num_cols


for col in num_cols:
    X[col] = scaler.fit_transform(X[[col]])



X


label_encoder  = LabelEncoder()


y_encodered = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encodered, test_size=0.2, random_state=42, stratify=y_encodered
)


categorical_features = X.columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Not needed here but good practice
)


# Define objective function for Optuna
def objective(trial):
    # Preprocessing
    X_processed = preprocessor.fit_transform(X_train)

    # XGBoost parameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist'
    }



    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_processed, y_train)):
        X_train_cv, X_val_cv = X_processed[train_idx], X_processed[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

        model = XGBClassifier(
            **params,
            random_state=42)

        model.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            verbose=False
        )

        y_pred = model.predict_proba(X_val_cv)
        score = log_loss(y_val_cv, y_pred)
        cv_scores.append(score)

        # Prune unpromising trials
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(cv_scores)


# Create Optuna study
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
)


study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials or 1 hour


print(f"Best trial value (Log Loss): {study.best_trial.value:.4f}")
print("Best parameters:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")


# # Train final model with best parameters
# best_params = {k: v for k, v in study.best_trial.params.items()
#                if k not in ['use_gpu']}
best_params = {'n_estimators': 218, 
                        'max_depth': 10, 
                        'learning_rate': 0.1205712628744377, 
                        'subsample': 0.8394633936788146, 
                        'colsample_bytree': 0.6624074561769746, 
                        'gamma': 0.7799726016810132, 
                        'min_child_weight': 1, 
                        'reg_alpha': 8.661761457749352, 
                        'reg_lambda': 6.011150117432088}

final_model = XGBClassifier(**best_params, random_state=42)



# Full preprocessing
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Train final model
final_model.fit(
    X_train_processed, y_train,
    eval_set=[(X_test_processed, y_test)],
    early_stopping_rounds=20,
    verbose=True
)


# Evaluate
y_pred = final_model.predict(X_test_processed)
y_pred_proba = final_model.predict_proba(X_test_processed)

test_accuracy = accuracy_score(y_test, y_pred)
test_log_loss = log_loss(y_test, y_pred_proba)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Log Loss: {test_log_loss:.4f}")


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.show()


# Feature importance
plt.figure(figsize=(12, 8))
sorted_idx = final_model.feature_importances_.argsort()
feature_names = preprocessor.get_feature_names_out()
plt.barh(
    np.array(feature_names)[sorted_idx][-20:],
    final_model.feature_importances_[sorted_idx][-20:]
)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()


test_df.head()
test_features = test_df.drop(columns=['id'], errors='ignore')
test_features.head()


cat_cols = test_features.columns.tolist()


test_df = preprocessor.fit_transform(test_features)


ohe = preprocessor.named_transformers_['cat']
ohe_feature_names = ohe.get_feature_names_out(categorical_features)

# Get remaining feature names (passthrough)
non_cat_features = [col for col in X.columns if col not in categorical_features]

# Combine all feature names
all_feature_names = list(ohe_feature_names) + non_cat_features

# Convert to DataFrame
df_final = pd.DataFrame(test_df.toarray(), columns=all_feature_names)
df_final.head()


print("Transformed shape:", test_df.shape)
print("Feature names count:", len(all_feature_names))



xgb_test_probs = final_model.predict_proba(test_df)
top3_indices = np.argsort(xgb_test_probs, axis=1)[:, -3:][:, ::-1]


df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


top3_str = [' '.join(label_encoder.inverse_transform(row)) for row in top3_indices]
#final submission
submission = pd.DataFrame({
    "id": df["id"],
    "Fertilizer Name": top3_str
})
submission.to_csv("submission.csv", index=False)
print('done submission')

