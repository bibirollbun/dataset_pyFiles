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


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import optuna
import warnings
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
dt=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
org1=pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv")
org2=pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")
org3=pd.read_csv("/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv")


df.head()


dt.head()


org1.head()


org2.head()


org3.head()


df.info()


org2.info()


org = pd.concat([org1,org2], ignore_index=True)
org


df=df.drop(columns=['id'])
dt=dt.drop(columns=['id'])



org = org.rename(columns={"Personality": "P2"})

org = org.drop_duplicates(subset=[
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
    'Post_frequency'
])



df = df.merge(org, how='left')
dt = dt.merge(org, how='left')


dt


X = df.drop(columns=['Personality'])
y = df['Personality']
target_encoder = LabelEncoder()
y = pd.Series(target_encoder.fit_transform(y))
print("Label encoded Class:", target_encoder.classes_)


combined_df =pd.concat([df.drop("Personality", axis=1), dt], ignore_index=True)



numerical_col = combined_df.select_dtypes(include=np.number)
categorical_col = combined_df.select_dtypes(include="object")
print("numerical:-", numerical_col.columns.values)
print(" ")
print("categorical:-" ,categorical_col.columns.values)


combined_df["Time_spent_Alone"] = np.sqrt(combined_df["Time_spent_Alone"])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
combined_df[numerical_col.columns] = scaler.fit_transform(combined_df[numerical_col.columns])



for col in numerical_col.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=combined_df, x=col, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.show()



for col in categorical_col.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=combined_df, x=col, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.show()



corr_matrix = numerical_col.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
combined_df.boxplot(figsize=(12, 6), rot=45)
plt.title("Boxplot of all numeric features")
plt.show()



for col in combined_df.select_dtypes(include='number').columns:
            combined_df[col] = combined_df[col].fillna(combined_df[col].mean())
for col in combined_df.select_dtypes(include='object').columns:
            combined_df[col] = combined_df[col].fillna("missing")



from sklearn.preprocessing import OrdinalEncoder
cat_cols = combined_df.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
combined_df[cat_cols] = encoder.fit_transform(combined_df[cat_cols])
combined_df


best_params = {
    'n_estimators': 1013,
    'max_depth': 3,
    'learning_rate': 0.04473761810915283,
    'subsample': 0.7472021066686094,
    'colsample_bytree': 0.6526442450606929,
    'gamma': 4.987525774261538,
    'reg_lambda': 0.1016293050091594,
    'reg_alpha': 0.8381641826774137,
    'min_child_weight': 10,
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'device': 'cuda',
    'random_state': 42
}

model = xgb.XGBClassifier(**best_params)


X = combined_df.iloc[:len(df)] 
X_test = combined_df.iloc[len(df):]
y = df["Personality"]


target_encoder = LabelEncoder()
y = pd.Series(target_encoder.fit_transform(y))
print("Label encoded Class:", target_encoder.classes_)


models_xgb, scores , pred_xgb = [], [] , []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n[INFO] Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    # pred_xgb.append(model.predict_proba(X_val)[:,1])
    acc = accuracy_score(y_val, model.predict(X_val))
    print(f"[INFO] Accuracy: {acc:.6f}")
    models_xgb.append(model)
    scores.append(acc)

print("\n[INFO] Mean CV Accuracy:", np.mean(scores)) #[INFO] Mean CV Accuracy: 0.9690672519477793



# import optuna
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.metrics import accuracy_score
# import numpy as np

# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1200),
#         'max_depth': trial.suggest_int('max_depth', 2, 20),
#         'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
#         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
#         'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
#         'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
#         'random_state': 42,
#         'n_jobs': -1
#     }

#     model = RandomForestClassifier(**params)
    
#     # Use StratifiedKFold to preserve class balance in CV
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
#     return scores.mean()
# study_random = optuna.create_study(direction='maximize')  # maximize accuracy
# study_random.optimize(objective, n_trials=50)

# print("Best trial:")
# print(study_random.best_trial.params)



dt=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")



# import optuna
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# import numpy as np

# def objective(trial):
#     params = {
#         'objective': 'binary', # Don't forget the objective for LGB!
#         'n_estimators': 1000,
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 512),
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
#         'random_state': 42,
#         'n_jobs': -1,
#         'metric': 'auc' # Use a metric for early stopping and evaluation
#     }

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     val_scores = []

#     for train_idx, val_idx in skf.split(X, y):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         model = LGBMClassifier(**params)
#         model.fit(X_train, y_train)

#         preds = model.predict(X_val)
#         acc = accuracy_score(y_val, preds)
#         val_scores.append(acc)

#     return np.mean(val_scores)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# print("\n[OPTUNA] Best Accuracy: ", study.best_value)
# print("[OPTUNA] Best Params:")
# for key, value in study.best_params.items():
#     print(f"  {key}: {value}")



# import optuna
# from catboost import CatBoostClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# import numpy as np

# def objective(trial):
#     params = {
#         'iterations': 1000, # Max iterations, early stopping will determine actual
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'depth': trial.suggest_int('depth', 4, 10),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
#         'border_count': trial.suggest_int('border_count', 32, 255),
#         'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
#         'eval_metric': 'AUC', # Optimize for AUC directly in CatBoost
#         'loss_function': 'Logloss',
#         'verbose': 0, # Suppress output during trials
#         'random_seed': 42,
#         'task_type': 'CPU', 
#     }

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     val_scores = []

#     for train_idx, val_idx in skf.split(X, y):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         model = CatBoostClassifier(**params)
#         model.fit(X_train, y_train)

#         preds = model.predict(X_val)
#         acc = accuracy_score(y_val, preds)
#         val_scores.append(acc)

#     return np.mean(val_scores)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# print("\n[OPTUNA] Best Accuracy: ", study.best_value)
# print("[OPTUNA] Best Params:")
# for k, v in study.best_params.items():
#     print(f"  {k}: {v}")



from catboost import CatBoostClassifier
import lightgbm as lgb
cat_best_params = {
'learning_rate': 0.013053487300374228,
 'depth': 4,
 'l2_leaf_reg': 2.8111991395141525,
 'bagging_temperature': 0.22579915083440136,
 'border_count': 155,
 'random_strength': 0.021646182508250927
}
lgb_best_params = {
    'learning_rate': 0.02268302087741931,
    'num_leaves': 428,
    'max_depth': 3,
    'min_child_samples': 49,
    'subsample': 0.7048596009171635,
    'colsample_bytree': 0.6768910271965423,
    'reg_alpha': 0.00021713652036488824,
    'reg_lambda': 0.2098003550818892,
    'verbose': -1  # use -1 to suppress LightGBM logs
}




from sklearn.ensemble import RandomForestClassifier

Random_best_params = {
    'n_estimators': 903,
    'max_depth': 10,
    'min_samples_split': 13,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'criterion': 'log_loss',
    'random_state': 42,
    'n_jobs': -1 
}
model_random = RandomForestClassifier(**Random_best_params)



model_cat = CatBoostClassifier(**cat_best_params ,verbose=0)
model_lgb = lgb.LGBMClassifier(**lgb_best_params)
model_random = RandomForestClassifier(**Random_best_params)



models_rand, scores , pred_rand= [], [] , []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n[INFO] Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model_random.fit(X_train, y_train)
    # pred_rand.append(model_random.predict_proba(X_val)[:,1])
    acc = accuracy_score(y_val, model_random.predict(X_val))
    print(f"[INFO] Accuracy: {acc:.6f}")
    models_rand.append(model_random)
    scores.append(acc)

print("\n[INFO] Mean CV Accuracy:", np.mean(scores)) #[INFO] Mean CV Accuracy: 0.9690672519477793



models_lgb, scores  = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n[INFO] Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model_lgb.fit(X_train, y_train)
    # pred_lgb.append(model_lgb.predict_proba(X_val)[:,1])

    acc = accuracy_score(y_val, model_lgb.predict(X_val))
    print(f"[INFO] Accuracy: {acc:.6f}")
    models_lgb.append(model_lgb)
    scores.append(acc)

print("\n[INFO] Mean CV Accuracy:", np.mean(scores)) #[INFO] Mean CV Accuracy: 0.9690672519477793



models_cat, scores  = [], [] 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n[INFO] Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model_cat.fit(X_train, y_train)
    acc = accuracy_score(y_val, model_cat.predict(X_val))
    print(f"[INFO] Accuracy: {acc:.6f}")
    models_cat.append(model_cat)
    scores.append(acc)

print("\n[INFO] Mean CV Accuracy:", np.mean(scores)) #[INFO] Mean CV Accuracy: 0.9690672519477793



cat_avg = np.mean([model.predict_proba(X)[:, 1] for model in models_cat], axis=0)
xgb_avg = np.mean([model.predict_proba(X)[:, 1] for model in models_xgb], axis=0)
lgb_avg = np.mean([model.predict_proba(X)[:, 1] for model in models_lgb], axis=0)
rand_avg = np.mean([model.predict_proba(X)[:, 1] for model in models_rand], axis=0)



models_lgb


import numpy as np
import optuna
from sklearn.metrics import accuracy_score

# Assume these are saved from CV or test:
# Each of shape (n_samples,)
probs_cat = np.mean([model.predict_proba(X)[:, 1] for model in models_cat], axis=0)
probs_xgb = np.mean([model.predict_proba(X)[:, 1] for model in models_xgb], axis=0)
probs_lgb = np.mean([model.predict_proba(X)[:, 1] for model in models_lgb], axis=0)
probs_rand = np.mean([model.predict_proba(X)[:, 1] for model in models_rand], axis=0)

# Stack model outputs
all_probs = np.vstack([probs_cat, probs_xgb, probs_lgb, probs_rand]).T  # shape: (n_samples, 4)

def objective(trial):
    # Suggest weights for each model (normalized)
    w_cat = trial.suggest_float("w_cat", 0, 1)
    w_xgb = trial.suggest_float("w_xgb", 0, 1)
    w_lgb = trial.suggest_float("w_lgb", 0, 1)
    w_rand = trial.suggest_float("w_rand", 0, 1)

    weights = np.array([w_cat, w_xgb, w_lgb, w_rand])
    weights /= np.sum(weights)  # Normalize weights

    # Suggest threshold
    threshold = trial.suggest_float("threshold", 0.3, 0.7)

    # Weighted ensemble prediction
    final_probs = np.dot(all_probs, weights)
    final_preds = (final_probs >= threshold).astype(int)

    acc = accuracy_score(y, final_preds)
    return acc

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Output best config
print("Best Parameters (weights + threshold):")
print(study.best_params)
print("Best Accuracy:", study.best_value)



best_params = study.best_params

# Extract weights and normalize again (just in case)
weights = np.array([
    best_params["w_cat"],
    best_params["w_xgb"],
    best_params["w_lgb"],
    best_params["w_rand"]
])
weights /= weights.sum()

best_threshold = best_params["threshold"]



# Final ensemble prediction
final_probs = np.dot(all_probs, weights)
final_preds = (final_probs >= best_threshold).astype(int)



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y, final_preds))
print("Classification Report:\n", classification_report(y, final_preds))
print("Confusion Matrix:\n", confusion_matrix(y, final_preds))



# Get averaged probabilities from each model on X_test
probs_cat_test = np.mean([model.predict_proba(X_test)[:, 1] for model in models_cat], axis=0)
probs_xgb_test = np.mean([model.predict_proba(X_test)[:, 1] for model in models_xgb], axis=0)
probs_lgb_test = np.mean([model.predict_proba(X_test)[:, 1] for model in models_lgb], axis=0)
probs_rand_test = np.mean([model.predict_proba(X_test)[:, 1] for model in models_rand], axis=0)

# Stack and apply weights + threshold
all_probs_test = np.vstack([probs_cat_test, probs_xgb_test, probs_lgb_test, probs_rand_test]).T
final_probs_test = np.dot(all_probs_test, weights)
final_preds_test = (final_probs_test >= best_threshold).astype(int)




Personality = target_encoder.inverse_transform(final_preds_test)
Personality


submission = pd.DataFrame({
    "id": dt["id"],
    "Personality": Personality
})

submission.to_csv("submission.csv", index=False)



from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
voting = VotingClassifier(
    estimators=[
        ('xgb', xgb.XGBClassifier(**best_params)),
        ('cat', CatBoostClassifier(**cat_best_params, verbose=0)),
        ('lgb', LGBMClassifier(**lgb_best_params)),
        ('random', RandomForestClassifier(**Random_best_params)
)
    ],
    voting='soft'
)
voting.fit(X, y)


pred = voting.predict(X_test)
Personality = target_encoder.inverse_transform(pred)

Personality


submission = pd.DataFrame({
    "id": dt["id"],
    "Personality": Personality
})
submission.to_csv("submission.csv", index=False)




