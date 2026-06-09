# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold # Best for classification/imbalance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
df_train.drop('id',axis=1,inplace=True)
df_test.drop('id',axis=1,inplace=True)


numerical_cols = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate']

# Determine grid size for subplots
num_cols = len(numerical_cols)
num_rows = (num_cols + 1) // 2  # Adjust as needed for layout

plt.figure(figsize=(12, 4 * num_rows)) # Adjust figure size

for i, col in enumerate(numerical_cols):
    plt.subplot(num_rows, 2, i + 1) # 2 columns per row
    sns.histplot(df_train[col], kde=True) # Example: histogram
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


categorical_cols = ['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade' ]

cat_cols_count = len(categorical_cols)
cat_rows = (cat_cols_count + 1) // 2  # Adjust as needed for layout, e.g., 3 rows for 6 plots

# Create figure and a set of subplots
fig, axes = plt.subplots(nrows=cat_rows, ncols=2, figsize=(15, 4 * cat_rows))

# Flatten the axes array for easier iteration
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    # Use axes-level function (sns.countplot) and specify the 'ax'
    sns.countplot(data=df_train, x=col, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].tick_params(axis='x', rotation=45) # Rotate labels if they overlap

# Hide any unused subplots if the total count is odd
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif

TARGET = 'loan_paid_back'

# 1. Deduplication and combine for uniform FE
print("ğŸ”� Removing duplicates in train/test...")
df_train = df_train.drop_duplicates().reset_index(drop=True)
df_test = df_test.drop_duplicates().reset_index(drop=True)

print("ğŸ”„ Combining train and test for consistent processing...")
df_train['is_train'] = 1
df_test['is_train'] = 0
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
print(f"â†’ Combined shape: {df_all.shape}")

# 2. Label Encoding (all categoricals, keep mapping for interpretability if needed)
cat_cols = df_all.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    df_all[col] = LabelEncoder().fit_transform(df_all[col].astype(str))
print(f"âœ“ Encoded {len(cat_cols)} categoricals")

# 3. Select only a few simple key categoricals (not high-card!), and top numerics by MI
cat_counts = {c: df_all[c].nunique() for c in cat_cols}
main_cats = [c for c, u in cat_counts.items() if 2 <= u <= 8][:2]    # use 2 lowest-card simple cats only (tune count as needed)
print(f"â˜† Main categoricals: {main_cats}")

num_cols = [c for c in df_all.columns if c not in cat_cols + [TARGET, 'is_train'] and np.issubdtype(df_all[c].dtype, np.number)]
mi = mutual_info_classif(df_all.loc[df_all.is_train==1, num_cols], df_all.loc[df_all.is_train==1, TARGET])
main_nums = [num_cols[i] for i in np.argsort(mi)[::-1][:5]]           # Only top 5 for max generalization
print(f"â˜† Selected numerics (by MI): {main_nums}")

# 4. Numeric interactions (only for main_nums, and only sum/prod/ratioâ€”not all pairs)
print("â�— Creating numeric interactions (sum, prod, ratio)...")
for i in range(len(main_nums)):
    for j in range(i+1, len(main_nums)):
        c1, c2 = main_nums[i], main_nums[j]
        df_all[f"{c1}_plus_{c2}"] = df_all[c1] + df_all[c2]
        df_all[f"{c1}_times_{c2}"] = df_all[c1] * df_all[c2]
        df_all[f"{c1}_div_{c2}"]   = df_all[c1] / (df_all[c2] + 1e-5)  # safe division

print("âœ“ Interactions done.")

# 5. Binning/transforms (main nums only)
print("ğŸ§® Numeric transforms/quantile bins/log/sqrt...")
for col in main_nums:
    try:
        df_all[f'{col}_bin'] = pd.qcut(df_all[col].rank(method='first'), 5, labels=False, duplicates='drop')
    except Exception as e:
        print(f"   - Skipped binning for {col}: {e}")
    df_all[f'{col}_log']  = np.log1p(np.abs(df_all[col]))
    df_all[f'{col}_sqrt'] = np.sqrt(np.abs(df_all[col]))
print("âœ“ Transforms complete.")

# 6. Restore final train/test sets
df_train = df_all[df_all['is_train']==1].drop(columns=['is_train'])
df_test  = df_all[df_all['is_train']==0].drop(columns=['is_train', TARGET], errors='ignore')
print(f"âœ”ï¸� Final train: {df_train.shape} | test: {df_test.shape}")

# 7. Standard scaling for selected numerics (for strong stacking/meta-models)
scaler = StandardScaler()
df_train[main_nums] = scaler.fit_transform(df_train[main_nums])
df_test[main_nums]  = scaler.transform(df_test[main_nums])
print(f"âœ… Features ready: {df_train.shape[1]-1} (excluding target)")


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')

def blended_feature_selection(df_train, df_test, target_col='loan_paid_back', n_features=50,
                             mi_thresh=0.03, lgb_thresh=0.03, et_thresh=0.03, min_sources=2):
    print("\n=*= FEATURE SELECTION PIPELINE (Multiple Importances, Robust) =*=")
    y = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])
    X_test = df_test.copy()

    # Remove constant features
    constant_features = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
    if constant_features:
        print(f"âœ“ Dropped {len(constant_features)} constant features: {constant_features}")
    X_train = X_train.drop(columns=constant_features)
    X_test  = X_test.drop(columns=constant_features, errors='ignore')

    # Remove highly correlated features
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [col for col in upper_tri.columns if (upper_tri[col] > 0.99).any()]
    if high_corr:
        print(f"âœ“ Dropped {len(high_corr)} highly correlated features: {high_corr}")
    X_train = X_train.drop(columns=high_corr)
    X_test  = X_test.drop(columns=high_corr, errors='ignore')

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    Xf = X_train[num_cols].fillna(0)  # For selectors

    print("ğŸ”� Calculating importances and selecting features...")
    # Mutual Info
    mi_scores = mutual_info_classif(Xf, y, random_state=42)
    mi_scores_norm = mi_scores / (np.max(mi_scores) + 1e-10)
    mi_features = [f for f, s in zip(num_cols, mi_scores_norm) if s > mi_thresh]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, importance_type='gain')
    lgb_model.fit(Xf, y)
    lgb_imp = lgb_model.feature_importances_
    lgb_imp_norm = lgb_imp / (np.max(lgb_imp) + 1e-10)
    lgb_features = [f for f, s in zip(num_cols, lgb_imp_norm) if s > lgb_thresh]

    # Extra Trees
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_model.fit(Xf, y)
    et_imp = et_model.feature_importances_
    et_imp_norm = et_imp / (np.max(et_imp) + 1e-10)
    et_features = [f for f, s in zip(num_cols, et_imp_norm) if s > et_thresh]

    # Create robust selector: keep only features identified by at least two selectors
    selectors = {'mi': mi_features, 'lgb': lgb_features, 'et': et_features}
    def count_sources(feat):
        return sum([feat in selectors[s] for s in selectors])

    print("  Counting agreement of sources for each feature...")
    agreement = {f: count_sources(f) for f in num_cols}
    robust_features = [f for f, cnt in agreement.items() if cnt >= min_sources]
    print(f"â˜… Features with agreement from at least {min_sources} selectors: {len(robust_features)}")

    # Average rank filter (only rank robust features)
    imp_df = pd.DataFrame({'feature': num_cols,
                           'mi': mi_scores_norm, 'lgb': lgb_imp_norm, 'et': et_imp_norm})
    for c in ['mi', 'lgb', 'et']:
        imp_df[f'{c}_rank'] = imp_df[c].rank(ascending=False)
    imp_df['avg_rank'] = imp_df[[f'{c}_rank' for c in ['mi','lgb','et']]].mean(axis=1)

    imp_df = imp_df[imp_df['feature'].isin(robust_features)]
    imp_df = imp_df.sort_values('avg_rank')
    selected = imp_df['feature'].tolist()[:n_features]

    print(f"âœ… Robust selected features ({len(selected)}):\n - {selected[:10]}{' ...' if len(selected) > 10 else ''}")
    X_final = X_train[selected].copy()
    X_test_final = X_test[selected].copy()
    print(f"âœ�ï¸� X_final: {X_final.shape}, X_test_final: {X_test_final.shape}")

    return X_final, X_test_final, y

# Usage
X_final, X_test_final, y = blended_feature_selection(
    df_train, df_test, target_col='loan_paid_back', n_features=50,
    mi_thresh=0.03, lgb_thresh=0.03, et_thresh=0.03, min_sources=2)


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

X = X_final
y = y
X_test = X_test_final

NFOLDS = 5
SEED_LIST = [42, 123, 999]

model_configs = {
    'lgb': (
        lgb.LGBMClassifier(force_col_wise=True, verbosity=-1),
        {'n_estimators': [150, 200], 'max_depth': [5, 7, 9], 'learning_rate': [0.03, 0.05, 0.07]}
    ),
    'xgb': (
        xgb.XGBClassifier(use_label_encoder=False, objective='binary:logistic'),
        {'n_estimators': [150, 200], 'max_depth': [5, 7], 'learning_rate': [0.03, 0.05]}
    ),
    'cat': (
        CatBoostClassifier(verbose=False),
        {'iterations': [150, 200], 'depth': [5, 7], 'learning_rate': [0.03, 0.05]}
    ),
}

# For each model, for each seed, store oof/test [seed, n_samples]
oof_preds = {m: np.zeros((len(SEED_LIST), len(X))) for m in model_configs}
test_preds = {m: np.zeros((len(SEED_LIST), len(X_test))) for m in model_configs}

for si, SEED in enumerate(SEED_LIST):
    print(f"\n==================== SEED {SEED} ====================")
    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold}/{NFOLDS} (Seed {SEED}) ===")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        for name, (base_model, param_dist) in model_configs.items():
            print(f"ğŸ”� Training {name.upper()} on fold {fold} (seed {SEED})...")
            model_cv = RandomizedSearchCV(
                base_model.set_params(random_state=SEED),
                param_dist,
                cv=3, scoring='roc_auc', n_iter=3, n_jobs=-1, random_state=SEED
            )
            model_cv.fit(X_train, y_train)
            best_params = model_cv.best_params_
            print(f"    Best params: {best_params}")

            if name == 'lgb':
                final_model = lgb.LGBMClassifier(**best_params, random_state=SEED, force_col_wise=True, verbosity=-1)
                final_model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='auc'
                )
            elif name == 'xgb':
                final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, objective='binary:logistic', random_state=SEED)
                final_model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=20,
                    eval_metric='auc',
                    verbose=False
                )
            else:  # cat
                final_model = CatBoostClassifier(**best_params, random_state=SEED, verbose=False)
                final_model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=20,
                    verbose=False
                )

            pred_valid = final_model.predict_proba(X_valid)[:, 1]
            pred_test = final_model.predict_proba(X_test)[:, 1]

            oof_preds[name][si, valid_idx] = pred_valid
            test_preds[name][si] += pred_test / NFOLDS

# Average OOF/test predictions over seeds
final_oof = {m: arr.mean(axis=0) for m, arr in oof_preds.items()}
final_test = {m: arr.mean(axis=0) for m, arr in test_preds.items()}

# Combined meta-feature DataFrames
print("\n========== LEVEL 0 MODEL PERFORMANCE ==========")
for model in final_oof:
    auc = roc_auc_score(y, final_oof[model])
    print(f"âœ… Model: {model.upper()} | OOF ROC-AUC: {auc:.5f}")

oof_stack = pd.DataFrame({f'{m}_oof': p for m, p in final_oof.items()}, index=X_final.index)
test_stack = pd.DataFrame({f'{m}_oof': p for m, p in final_test.items()}, index=X_test_final.index)
print("\nğŸ“¦ Shape of OOF meta-features:", oof_stack.shape)
print("ğŸ“¦ Shape of Test meta-features:", test_stack.shape)



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class SimpleStacker:
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state

    def get_model_and_grid(self, model_name):
        if model_name == 'lr':
            return LogisticRegression(random_state=self.random_state, max_iter=300), {
                'C': [0.05, 0.1, 0.5, 1.0]
            }
        elif model_name == 'ridge':
            return RidgeClassifier(random_state=self.random_state), {
                'alpha': [0.1, 1.0, 10.0]
            }
        elif model_name == 'lgb':
            return lgb.LGBMClassifier(random_state=self.random_state, force_col_wise=True, verbosity=-1), {
                'n_estimators': [100, 150],
                'num_leaves': [10, 20, 31],
                'learning_rate': [0.02, 0.05]
            }

    def fit_predict(self, oof_stack, test_stack, y_train):
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        meta_oof = pd.DataFrame(index=oof_stack.index)
        meta_test = pd.DataFrame(index=test_stack.index)
        meta_scores = {}

        # Only key meta-modelsâ€”no tree ensemble meta!
        model_names = ['lr', 'ridge', 'lgb']

        for model_name in model_names:
            print(f"\nTraining {model_name.upper()} with GridSearchCV")
            oof_preds = np.zeros(len(oof_stack))
            test_preds = np.zeros(len(test_stack))
            scores = []

            base_model, param_grid = self.get_model_and_grid(model_name)

            for fold, (tr_idx, va_idx) in enumerate(skf.split(oof_stack, y_train), 1):
                X_tr, X_va = oof_stack.iloc[tr_idx], oof_stack.iloc[va_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

                grid = GridSearchCV(base_model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
                grid.fit(X_tr, y_tr)
                best_model = grid.best_estimator_

                if hasattr(best_model, 'predict_proba'):
                    val_pred = best_model.predict_proba(X_va)[:, 1]
                    test_pred = best_model.predict_proba(test_stack)[:, 1]
                else:
                    val_pred = best_model.decision_function(X_va)
                    test_pred = best_model.decision_function(test_stack)
                    val_pred = (val_pred - val_pred.min()) / (val_pred.max() - val_pred.min() + 1e-8)
                    test_pred = (test_pred - test_pred.min()) / (test_pred.max() - test_pred.min() + 1e-8)

                oof_preds[va_idx] = val_pred
                test_preds += test_pred / self.n_folds

                score = roc_auc_score(y_va, val_pred)
                scores.append(score)
                print(f"  Fold {fold}: AUC = {score:.4f}")

            meta_oof[model_name] = oof_preds
            meta_test[model_name] = test_preds
            meta_scores[model_name] = np.mean(scores)
            print(f" > Mean OOF AUC for {model_name.upper()}: {meta_scores[model_name]:.5f}")

        # Simple uniform average or best model â€” or blend by OOF AUC
        best_model_name = max(meta_scores, key=lambda x: meta_scores[x])
        print(f"\nFinal Stacker: Best meta-model by OOF AUC is {best_model_name.upper()} ({meta_scores[best_model_name]:.5f})")
        final_oof = meta_oof[best_model_name].values
        final_test = meta_test[best_model_name].values

        print(f"\nFinal Stacked AUC: {roc_auc_score(y_train, final_oof):.5f}")
        return final_oof, final_test

# Usage
stacker = SimpleStacker(n_folds=5, random_state=42)
final_oof, final_test = stacker.fit_predict(oof_stack, test_stack, y)


# Build submission
df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.DataFrame({"id": df_test["id"], "loan_paid_back": final_test})
submission.to_csv("submission.csv", index=False)
print("Saved final submission.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, brier_score_loss, roc_curve
)
from sklearn.calibration import calibration_curve

class Step6_ModelEvaluation:
    def __init__(self, target_col='loan_paid_back'):
        self.target_col = target_col

    def evaluate_all_models(self, y_true, predictions_dict, model_type="Level 0"):
        results = {}
        for name, preds in predictions_dict.items():
            preds_binary = (preds > 0.5).astype(int)
            auc = roc_auc_score(y_true, preds)
            acc = accuracy_score(y_true, preds_binary)
            f1 = f1_score(y_true, preds_binary)
            prec = precision_score(y_true, preds_binary)
            recall = recall_score(y_true, preds_binary)
            brier = brier_score_loss(y_true, preds)
            results[name] = {'auc': auc, 'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': recall, 'brier': brier}
            print(f"[{model_type}] {name}: AUC={auc:.4f} | Acc={acc:.4f} | F1={f1:.4f} | Prec={prec:.4f} | Recall={recall:.4f} | Brier={brier:.4f}")
        return results

    def model_rank_table(self, results):
        res_df = pd.DataFrame(results).T
        print(res_df.sort_values('auc', ascending=False)[['auc', 'accuracy', 'f1', 'precision', 'recall', 'brier']])

    def plot_roc_curves(self, y_true, predictions_dict):
        plt.figure(figsize=(10, 7))
        for name, preds in predictions_dict.items():
            fpr, tpr, _ = roc_curve(y_true, preds)
            auc = roc_auc_score(y_true, preds)
            plt.plot(fpr, tpr, label=f"{name} (AUC: {auc:.3f})")
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves")
        plt.legend(); plt.show()

    def plot_calibration_curves(self, y_true, predictions_dict):
        plt.figure(figsize=(10, 7))
        for name, preds in predictions_dict.items():
            prob_true, prob_pred = calibration_curve(y_true, preds, n_bins=10, strategy='uniform')
            plt.plot(prob_pred, prob_true, marker='o', label=name)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel("Mean Predicted Value"); plt.ylabel("Fraction of Positives"); plt.title("Calibration Curves")
        plt.legend(); plt.show()

    def plot_prediction_distribution(self, predictions_dict):
        plt.figure(figsize=(10, 7))
        for name, preds in predictions_dict.items():
            plt.hist(preds, bins=50, alpha=0.5, label=name, density=True)
        plt.xlabel("Predicted Probability"); plt.ylabel("Density"); plt.title("Prediction Distributions")
        plt.legend(); plt.show()

    def compare_ensembles(self, y_true, predictions_dict):
        preds_df = pd.DataFrame(predictions_dict)
        aucs = {col: roc_auc_score(y_true, preds_df[col]) for col in preds_df.columns}
        best_model = max(aucs, key=aucs.get)
        remaining = set(preds_df.columns) - {best_model}
        current = preds_df[[best_model]].copy()
        best_auc = aucs[best_model]
        selected = [best_model]
        improved = True
        while improved and remaining:
            improved = False
            next_best = None
            next_auc = best_auc
            for cand in remaining:
                avg_pred = current.mean(axis=1) * (len(current.columns) / (len(current.columns)+1)) + preds_df[cand] / (len(current.columns)+1)
                auc_val = roc_auc_score(y_true, avg_pred)
                if auc_val > next_auc:
                    next_auc = auc_val
                    next_best = cand
            if next_best is not None:
                selected.append(next_best)
                current[next_best] = preds_df[next_best]
                best_auc = next_auc
                remaining.remove(next_best)
                improved = True
        final_ensemble_pred = current.mean(axis=1)
        print(f"Greedy ensemble selected models: {selected}")
        return selected, final_ensemble_pred

# Assuming oof_stack columns: 'lgb_oof', 'xgb_oof', 'cat_oof'
# And final_oof from meta-stacker

evaluator = Step6_ModelEvaluation(target_col='loan_paid_back')

# Evaluate individual L0 models
level0_results = evaluator.evaluate_all_models(
    y_true=y,
    predictions_dict={col: oof_stack[col] for col in oof_stack.columns},
    model_type="Level 0"
)

# Evaluate meta-stacker
meta_results = evaluator.evaluate_all_models(
    y_true=y,
    predictions_dict={'stacked_meta': final_oof},
    model_type="Meta"
)

print("\nğŸ”¢ Model Ranking Table (by AUC):")
evaluator.model_rank_table(level0_results)

# Visualization: L0 + meta only
plot_dict = {col: oof_stack[col] for col in oof_stack.columns}
plot_dict['stacked_meta'] = final_oof

print("\nğŸ“ˆ ROC Curves")
evaluator.plot_roc_curves(y, plot_dict)
print("\nğŸ“‰ Calibration Curves")
evaluator.plot_calibration_curves(y, plot_dict)
print("\nğŸ“Š Prediction Distributions")
evaluator.plot_prediction_distribution(plot_dict)

# Greedy ensemble comparison (optional for curiosity)
print("\nğŸ”§ Greedy Ensemble Search")
selected_models, ensemble_preds = evaluator.compare_ensembles(y, {col: oof_stack[col] for col in oof_stack.columns})
ensemble_auc = roc_auc_score(y, ensemble_preds)
print(f"\nGreedy Ensemble AUC: {ensemble_auc:.4f}")
print(f"Meta-model better than ensemble? {meta_results['stacked_meta']['auc'] > ensemble_auc}")

