!pip install pytorch-tabnet -q

import pandas as pd
import numpy as np
import torch
import os
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, train_test_split

from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from itertools import product
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df_raw = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


print(len(train_df_raw))
display(train_df_raw.head())


original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')
print(len(original))
display(original.head())


column_mapping = {
    'Time spent alone': 'Time_spent_Alone',
    'Stage fear': 'Stage_fear',
    'Social event attendence': 'Social_event_attendance',
    'Going outside': 'Going_outside',
    'Drained after socializing': 'Drained_after_socializing',
    'Friends circle size': 'Friends_circle_size',
    'Post frequency': 'Post_frequency'
}

original.rename(columns=column_mapping, inplace=True)
original = original.rename(columns={'Personality': 'match_p'})

merge_cols = [col for col in original.columns if col in train_df_raw.columns and col != 'id' and col != 'Personality']


original_dedup = original.drop_duplicates(subset=merge_cols)


train_df = train_df_raw.merge(original_dedup, on=merge_cols, how='left')
test_df = test_df.merge(original_dedup, on=merge_cols, how='left')


train_df.describe()


train_df.isnull().sum()


plt.figure(figsize=(8, 6))

ax = sns.countplot(
    x='Personality',
    data=train_df,
    palette=['#4374B3', '#FF8F33']
)


for p in ax.patches:
    ax.annotate(
        f'{p.get_height()}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='center',
        xytext=(0, 9),         
        textcoords='offset points'
    )

plt.title('Extroverts vs Introverts', fontsize=16)
plt.xlabel('Type', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.show()

print("\nPercnets personality:")
print(train_df['Personality'].value_counts(normalize=True) * 100)


numeric_df = train_df.select_dtypes(include=np.number)

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation heat map')
plt.show()


y = train_df['Personality'].apply(lambda x: 1 if x == 'Extrovert' else 0)
X = train_df.drop(columns=['id', 'Personality'])
X_submission = test_df.drop(columns=['id'])

submission_ids = test_df['id']


for df in [X, X_submission]:
    df['match_p_is_null'] = df['match_p'].isna().astype(int)
    
    df['Stage_fear'] = df['Stage_fear'].fillna('unknown')
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('unknown')
    df['match_p'] = df['match_p'].fillna('unknown')
    
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].mean())


X_columns = X.columns
categorical_features_names = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

encoders = {}
for col in categorical_features_names:
    le = LabelEncoder()
    all_categories = pd.concat([X[col], X_submission[col]]).astype(str).unique()
    le.fit(all_categories)
    X[col] = le.transform(X[col].astype(str))
    X_submission[col] = le.transform(X_submission[col].astype(str))
    encoders[col] = le



imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X_columns)
X_submission = pd.DataFrame(imputer.transform(X_submission), columns=X_columns)


scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_submission[numerical_cols] = scaler.transform(X_submission[numerical_cols])


for col in categorical_features_names:
    X[col] = X[col].round().astype(int)
    X_submission[col] = X_submission[col].round().astype(int)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


catboost_params = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 10,
    'l2_leaf_reg': 10,
    'random_seed': 42,
    'verbose': 100,
    'auto_class_weights': 'Balanced',
    'random_strength': 1.5,
    'bootstrap_type': 'Bayesian',
    'eval_metric': 'AUC',
}


if use_gpu:
    catboost_params['task_type'] = 'GPU'
    catboost_params['devices'] = '0'


X_np = X.to_numpy()
y_np = y.to_numpy()

cat_idxs = [X.columns.get_loc(col) for col in categorical_features_names]
cat_dims = [len(encoders[col].classes_) for col in categorical_features_names]

tabnet_params_cv = {
    'n_d': 16, 
    'n_a': 24, 
    'n_steps': 5, 
    'gamma': 1.5,
    'cat_idxs': [X.columns.get_loc(col) for col in categorical_features_names],
    'cat_dims': [len(encoders[col].classes_) for col in categorical_features_names],
    'cat_emb_dim': 1,
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=0.02),
    'scheduler_params': dict(mode="max", patience=5, factor=0.5),
    'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'mask_type': 'entmax',
    'device_name': device,
    'verbose': 0
}


catboost_scores = []
tabnet_scores = []
tabnet_best_epochs = []

for fold, (train_idx, val_idx) in enumerate(tqdm(cv.split(X, y), total=cv.get_n_splits(), desc="CV Folds")):
    print(f"\n--- Fold {fold+1}/{cv.get_n_splits()} ---")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train_np, X_val_np = X_train.to_numpy(), X_val.to_numpy()
    y_train_np, y_val_np = y_train.to_numpy(), y_val.to_numpy()

    print("Train CatBoost...")
    cat_model = CatBoostClassifier(**catboost_params)
    cat_model.fit(X_train, y_train, cat_features=categorical_features_names, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)
    cat_preds = cat_model.predict_proba(X_val)[:, 1]
    cat_score = roc_auc_score(y_val, cat_preds)
    catboost_scores.append(cat_score)
    print(f"CatBoost Fold {fold+1} AUC: {cat_score:.5f}")

    print("Train TabNet...")
    tab_model = TabNetClassifier(**tabnet_params_cv)
    tab_model.fit(X_train_np, y_train_np, eval_set=[(X_val_np, y_val_np)], eval_metric=['auc'], max_epochs=100, patience=15, batch_size=1024, weights=1, drop_last=False)
    tab_preds = tab_model.predict_proba(X_val_np)[:, 1]
    tab_score = roc_auc_score(y_val_np, tab_preds)
    tabnet_scores.append(tab_score)
    tabnet_best_epochs.append(tab_model.best_epoch)
    print(f"TabNet Fold {fold+1} AUC: {tab_score:.5f}")



final_tabnet_epochs = int(np.mean(tabnet_best_epochs)) + 5

best_score_catboost = np.mean(catboost_scores)
best_score_tabnet = np.mean(tabnet_scores)

print(f"Best CatBoost score {best_score_catboost}")
print(f"Best TabNet score {best_score_tabnet}")


final_catboost_model = CatBoostClassifier(**catboost_params)
final_catboost_model.fit(X, y, cat_features=categorical_features_names, verbose=0)


final_tabnet_model = TabNetClassifier(**tabnet_params_cv)
final_tabnet_model.fit(
    X_train=X_np, y_train=y_np,
    max_epochs=final_tabnet_epochs,
    batch_size=1024,
    weights=1,
    drop_last=False
)


catboost_preds = final_catboost_model.predict_proba(X_submission)[:, 1]
X_submission_np = X_submission.to_numpy()
tabnet_preds = final_tabnet_model.predict_proba(X_submission_np)[:, 1]

power = 10
score_catboost_powered = best_score_catboost ** power
score_tabnet_powered = best_score_tabnet ** power

total_powered_score = score_catboost_powered + score_tabnet_powered

weight_catboost = score_catboost_powered / total_powered_score
weight_tabnet = score_tabnet_powered / total_powered_score

ensemble_preds = weight_catboost * catboost_preds + weight_tabnet * tabnet_preds

binary_preds = (ensemble_preds > 0.5).astype(int)

final_labels = pd.Series(binary_preds).map({1: 'Extrovert', 0: 'Introvert'})

submission_df = pd.DataFrame({'id': submission_ids, 'Personality': final_labels})

submission_df.to_csv('submission_ensemble.csv', index=False)

display(submission_df.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


eval_cat_model = CatBoostClassifier(**catboost_params)
eval_cat_model.fit(X_train, y_train, cat_features=categorical_features_names, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=0)
cat_preds_proba_eval = eval_cat_model.predict_proba(X_test)[:, 1]


X_train_np, X_test_np = X_train.to_numpy(), X_test.to_numpy()
y_train_np, y_test_np = y_train.to_numpy(), y_test.to_numpy()

eval_tab_model = TabNetClassifier(**tabnet_params_cv)
eval_tab_model.fit(
    X_train=X_train_np, y_train=y_train_np,
    eval_set=[(X_test_np, y_test_np)],
    eval_metric=['auc'],
    max_epochs=final_tabnet_epochs,
    patience=15,
    batch_size=1024,
    weights=1,
    drop_last=False
)
tab_preds_proba_eval = eval_tab_model.predict_proba(X_test_np)[:, 1]

y_pred_proba = weight_catboost * cat_preds_proba_eval + weight_tabnet * tab_preds_proba_eval
y_pred = (y_pred_proba > 0.5).astype(int)


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Introvert (0)', 'Extrovert (1)'],
            yticklabels=['Introvert (0)', 'Extrovert (1)'])
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.show()


roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC on train: {roc_auc:.4f}")


report = classification_report(y_test, y_pred, target_names=['Introvert (0)', 'Extrovert (1)'])
print(report)

