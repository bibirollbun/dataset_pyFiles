# ğŸŒŸ Let's get started â€” sabse pehle required libraries ko import karte hain!
import os
import numpy as np
import pandas as pd
import gc
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display            # for head()

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler  # scaler optional (future)
from sklearn.decomposition import PCA                              # optional (future)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import lightgbm as lgb

warnings.simplefilter(action='ignore')

# ============================================
# ğŸ“¦ 1) House-keeping & common settings
# ============================================
SEED   = 42
TARGET = "Fertilizer Name"          
np.random.seed(SEED)



# ============================================
#  2) Load data
# ===========================================

PLAYGROUND_DIR = "/kaggle/input/playground-series-s5e6"   #  Playground Series Season 5 Episode 6
FERTILIZER_DIR = "/kaggle/input/fertilizer-prediction"    #  Fertilizer Prediction dataset

#  Training set
df_train = pd.read_csv(f"{PLAYGROUND_DIR}/train.csv")

#  Test set (isko model pe chala kar submission banayenge)
df_test  = pd.read_csv(f"{PLAYGROUND_DIR}/test.csv")

# Original fertilizer-prediction data (feature engineering ke ideas yahan se nikal sakte hain)
df_fert  = pd.read_csv(f"{FERTILIZER_DIR}/Fertilizer Prediction.csv")

#  Submission template (yehi format Kaggle ko chahiye hota hai)
df_sub   = pd.read_csv(f"{PLAYGROUND_DIR}/sample_submission.csv")



# ============================================
# ğŸ‘€ Quick EDA â€” Data se dosti karte hain pehle ğŸ¤�
# ============================================

print("ğŸ”� Doing a little EDA before jumping into modeling...")

# 1ï¸�âƒ£ Let's see how our training data looks
print("ğŸ“� Train Data Sample:")
display(df_train.head())

print("ğŸ“� Shape of train:", df_train.shape)
print("ğŸ“� Shape of test:", df_test.shape)
print("ğŸ“� Shape of external/original data:", df_fert.shape)

# 2ï¸�âƒ£ Check if any columns have missing values
print("\nâ�“ Checking missing values:")
print(df_train.isnull().sum())

# 3ï¸�âƒ£ Unique fertilizer labels â€” ye hi to predict krna hai ğŸ�¯
print("\nğŸ§ª Unique Fertilizer Labels:")
print(df_train['Fertilizer Name'].value_counts())

# 4ï¸�âƒ£ Plot â€” Target distribution (thoda imbalance ho to pata chale)
plt.figure(figsize=(12, 6))
sns.countplot(data=df_train, y='Fertilizer Name', order=df_train['Fertilizer Name'].value_counts().index)
plt.title("Distribution of Fertilizer Classes in Train Set")
plt.xlabel("Count")
plt.ylabel("Fertilizer Name")
plt.tight_layout()
plt.show()



# ============================================
# ğŸ”§ 4) Feature Engineering
# ============================================
# ğŸ§ª Original fertilizer data ka backup â€“ augment karne ke liye
df_fert_aug = df_fert.copy()

# ğŸ”� Data augmentation â€“ original data ko 6x duplicate karke bada set bana rahe hain
for _ in range(6):
    df_fert = pd.concat([df_fert, df_fert_aug], axis=0)

# ğŸ”� Sirf numerical features nikal rahe hain (except 'id')
num_cols = [
    col for col in df_train.select_dtypes(include=['int64', 'float64']).columns 
    if col != 'id'
]

# Har dataframe pe transformation apply â€“ binning, renaming, type optimization
for df in [df_train, df_test, df_fert]:
    
    #  Numerical columns ko binned categories mein convert kar rahe hain
    for col in num_cols:
        df[f'{col}_Binned'] = df[col].astype(str).astype('category')

    #  Typo fix: 'Temparature' â†’ 'Temperature'
    df.rename(columns={'Temparature': 'Temperature'}, inplace=True)
    
    #  Memory optimization â€“ int64 â†’ int8, float64 â†’ float16
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int8')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float16')




# ============================================
# ğŸ”¢ 5) Data Preprocessing
# ============================================
# ğŸ�·ï¸�  Categorical columns nikal rahe hain (target ko exclude kar ke)
cat_features = [
    col for col in df_train.select_dtypes(include=['object', 'category']).columns
    if col != "Fertilizer Name"
]

# ğŸ”� Label Encoding â€“ saare categorical features ke liye
for col in cat_features:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_fert[col] = le.transform(df_fert[col])
    df_test[col]  = le.transform(df_test[col])

# ğŸ�¯ Target column "Fertilizer Name" ka bhi encoding
target_le = LabelEncoder()
df_train["Fertilizer Name"] = target_le.fit_transform(df_train["Fertilizer Name"])
df_fert["Fertilizer Name"]  = target_le.transform(df_fert["Fertilizer Name"])

# ğŸ§½ Category conversion â€“ memory optimize aur model ko help mile
for col in cat_features:
    df_train[col] = df_train[col].astype("category")
    df_fert[col]  = df_fert[col].astype("category")
    df_test[col]  = df_test[col].astype("category")

# ğŸ�¯ Feature-target split
X_train = df_train.drop(columns=["id", "Fertilizer Name"])
y_train = df_train["Fertilizer Name"]

# ğŸ“¦ Test set (submission ke liye)
X_test = df_test.drop(columns=["id"])

# ğŸŒ± Original fertilizer data split (SSL ya augmentation ke kaam aa sakta hai)
X_fert = df_fert.drop(columns=["Fertilizer Name"])
y_fert = df_fert["Fertilizer Name"]



# ============================================
# 6) Modelling
# ============================================
# ğŸ�¯ MAP@K metric â€“ model ki top-K prediction ki accuracy ko evaluate karta hai
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# âš™ï¸�  Model configs â€“ XGBoost, LightGBM (with & without GOSS)
model_configs = {
    'xgb': {
        'model': XGBClassifier,
        'params': {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train)),
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'max_bin': 128,
            'colsample_bytree': 0.3,
            'colsample_bylevel': 1,
            'colsample_bynode': 1,
            'tree_method': 'hist',
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'device': 'cuda',
            'enable_categorical': True,
            'n_estimators': 10000,
            'early_stopping_rounds': 50
        }
    },
    'lgb_goss': {
        'model': LGBMClassifier,
        'params': {
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'boosting_type': 'goss',
            'device': 'gpu',
            'colsample_bytree': 0.3275,
            'learning_rate': 0.02670,
            'max_depth': 9,
            'min_child_samples': 84,
            'n_estimators': 10000,
            'n_jobs': -1,
            'num_leaves': 229,
            'random_state': 42,
            'reg_alpha': 6.87997,
            'reg_lambda': 4.7391,
            'subsample': 0.5411,
            'categorical_feature': cat_features,
            'verbose': -1
        }
    },
    'lgb': {
        'model': LGBMClassifier,
        'params': {
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'device': 'gpu',
            'colsample_bytree': 0.4366,
            'learning_rate': 0.02617,
            'max_depth': 11,
            'min_child_samples': 67,
            'n_estimators': 10000,
            'n_jobs': -1,
            'num_leaves': 243,
            'random_state': 42,
            'reg_alpha': 6.38283,
            'reg_lambda': 9.39295,
            'subsample': 0.79898,
            'categorical_feature': cat_features,
            'verbose': -1
        }
    }
}




# ============================================
#  7) cross validation
# ============================================
# Cross-validation config (7 folds)
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

#  Storage for predictions and scores
oof_preds = {name: np.zeros((len(X_train), y_train.nunique())) for name in model_configs}
test_preds = {name: np.zeros((len(X_test), y_train.nunique())) for name in model_configs}
map3_scores = {name: [] for name in model_configs}

#  Training loop â€“ har model + har fold
for name, config in model_configs.items():
    print(f"\n Training model: {name.upper()}")

    model = config['model'](**config['params'])

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        x_tr, x_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        #  Augmenting training set with original fertilizer data
        x_tr = pd.concat([x_tr, X_fert], axis=0, ignore_index=True)
        y_tr = pd.concat([y_tr, y_fert], axis=0, ignore_index=True)

        #  Model training
        if name == 'xgb':
            model.fit(x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_val, y_val)], verbose=0)
        else:
            model.fit(
                x_tr, y_tr,
                eval_set=[(x_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=100)]
            )

        # OOF + Test prediction
        oof_preds[name][valid_idx] = model.predict_proba(x_val)
        test_preds[name] += model.predict_proba(X_test) / skf.n_splits

        # ğŸ“Š Fold-wise MAP@3 score
        top_3 = np.argsort(oof_preds[name][valid_idx], axis=1)[:, -3:][:, ::-1]
        score = mapk([[label] for label in y_val], top_3)
        map3_scores[name].append(score)
        print(f"ğŸ“ˆ Fold {fold+1} MAP@3: {score:.5f}")

    #  Final score for model
    avg_score = np.mean(map3_scores[name])
    print(f"{name.upper()} Average MAP@3: {avg_score:.5f}")


#===============================
# 8) stacking ensemble 
#==================================

# OOF aur Test Predictions ko concat karke meta-features banaye stacking ke liye
stack_train = np.hstack([oof_preds[name] for name in oof_preds])     # Train meta features
stack_test  = np.hstack([test_preds[name] for name in test_preds])  # Test meta features

# Meta-model (LGBM as final blender)
meta_model = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_train)),
    learning_rate=0.03,
    n_estimators=10000,
    random_state=42,
    verbose=-1
)

print("Training stacking ensemble...")

#  Final prediction containers
final_oof  = np.zeros((len(y_train), y_train.nunique()))
final_test = np.zeros((len(X_test), y_train.nunique()))
ensemble_scores = []

# Stacking cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(skf.split(stack_train, y_train)):
    x_tr, x_val = stack_train[train_idx], stack_train[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    #  Meta model training
    meta_model.fit(
        x_tr, y_tr,
        eval_set=[(x_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )

    # Meta predictions
    final_oof[valid_idx] = meta_model.predict_proba(x_val)
    final_test += meta_model.predict_proba(stack_test) / skf.n_splits

    # MAP@3 evaluation
    top_3 = np.argsort(final_oof[valid_idx], axis=1)[:, -3:][:, ::-1]
    score = mapk([[lab] for lab in y_val], top_3)
    ensemble_scores.append(score)

    print(f" Ensemble Fold {fold+1}: MAP@3 = {score:.5f}")

# ğŸ�� Final average score
avg_ensemble_score = np.mean(ensemble_scores)
print(f"\n Ensemble Average MAP@3: {avg_ensemble_score:.5f}")



#==================================
# 9) Saving and submissions
#=================================
# Results save karne ke liye directory bana rahe hain
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

#  Stacking ensemble predictions save (Offline analysis ke liye useful)
np.save(f"{output_dir}/stacking_oof.npy", final_oof)
np.save(f"{output_dir}/stacking_test.npy", final_test)

#  Har base model ke OOF aur test predictions bhi separately save kar rahe hain
for name in oof_preds:
    np.save(f"{output_dir}/{name}_oof.npy", oof_preds[name])
    np.save(f"{output_dir}/{name}_test.npy", test_preds[name])

# Final test predictions ko label form mein convert karna (Top-3)
top_3 = np.argsort(final_test, axis=1)[:, -3:][:, ::-1]
labels = target_le.inverse_transform(top_3.ravel()).reshape(top_3.shape)

# Kaggle submission file ready kar rahe hain
submission_df = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in labels]
})
submission_df.to_csv("submission.csv", index=False)

# Score report save kar rahe hain (sab model aur ensemble ke MAP@3 scores)
with open(f"{output_dir}/scores.txt", "w") as f:
    for name, scores in map3_scores.items():
        f.write(f"{name} MAP@3 Scores: {scores}\n")
        f.write(f"{name} Average MAP@3: {np.mean(scores):.5f}\n")
    f.write(f"Ensemble MAP@3 Scores: {ensemble_scores}\n")
    f.write(f"Ensemble Average MAP@3: {np.mean(ensemble_scores):.5f}\n")





