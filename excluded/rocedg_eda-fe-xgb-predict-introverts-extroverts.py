# Import necessary libraries
import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import lightgbm as lgb
from cuml.preprocessing.TargetEncoder import TargetEncoder
from catboost import CatBoostClassifier

sns.set_theme(style="whitegrid")
sns.set_palette('pastel')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# Function to load data
base_path="/kaggle/input/playground-series-s5e7/"

"""Load and return train, test and sample submission dataframes"""
train = pd.read_csv(f"{base_path}train.csv").set_index("id")
test = pd.read_csv(f"{base_path}test.csv").set_index("id")
submission = pd.read_csv(f"{base_path}sample_submission.csv").set_index("id")


train.shape


train["Personality"] = train["Personality"].map({"Extrovert": 1, "Introvert": 0})
train.head()


train.describe()


train.isnull().sum()


# Create a temporary cleaned DataFrame for visualization only
train_nonan = train.replace([np.inf, -np.inf], np.nan).dropna()

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    for i, column in enumerate(train_nonan.columns):
        ax = axes[i]

        if train_nonan[column].dtype == 'object':
            sns.countplot(data=train_nonan, x=column, hue='Personality', palette='pastel', edgecolor='black', ax=ax)
        elif column == 'Personality':
            sns.countplot(data=train_nonan, x=column, palette='pastel', edgecolor='black', ax=ax)
        else:
            sns.histplot(data=train_nonan, x=column, hue='Personality',  palette='pastel', edgecolor='black', ax=ax)

        ax.set_title(f'Distribution of {column}', fontsize=14)
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle('Feature Distributions by Personality Type (Introvert = 0, Extrovert = 1)', fontsize=18)
plt.show()


# Identify columns
numeric_cols = ['Time_spent_Alone','Social_event_attendance', 'Going_outside',
                'Friends_circle_size','Post_frequency']
cat_cols     = ['Stage_fear','Drained_after_socializing']


from scipy.stats import pointbiserialr, chi2_contingency

def cramers_v(chi2, n, dof):
    k = dof + 1  # approximate smallest dimension categories
    return np.sqrt(chi2 / (n * (k - 1)))

def correlation_analysis_and_plot(df, numeric_cols, cat_cols, target='Personality'):
    correlation_results = {}

    for column in df.columns:
        if column == target:
            continue

        if column in numeric_cols:
            subset = df[[target, column]].dropna()
            if subset.empty:
                continue
            corr, p = pointbiserialr(subset[target], subset[column])
            correlation_results[column] = {
                'Test': 'Point Biserial Correlation',
                'Correlation': corr,
                'p-value': p
            }

        elif column in cat_cols:
            ct = pd.crosstab(df[column], df[target])
            if ct.empty:
                continue
            chi2, p, dof, expected = chi2_contingency(ct)
            v = cramers_v(chi2, ct.sum().sum(), dof)
            correlation_results[column] = {
                'Test': 'Chi-Squared Test',
                'Chi2 Statistic': chi2,
                'p-value': p,
                'Cramers V': v
            }

    # Prepare DataFrame for plotting
    rows = []
    for feature, res in correlation_results.items():
        if res['Test'] == 'Point Biserial Correlation':
            corr_val = abs(res['Correlation'])
        elif res['Test'] == 'Chi-Squared Test':
            corr_val = res.get('Cramers V', 0)
        else:
            corr_val = 0
        rows.append({
            'Feature': feature,
            'Correlation_Magnitude': corr_val,
            'Test': res['Test']
        })

    df_corr = pd.DataFrame(rows)

    df_corr_plot = df_corr.sort_values(by='Correlation_Magnitude', ascending=False)[:25]

    # Plot
    plt.figure(figsize=(12, max(6, len(df_corr_plot) * 0.3)))
    sns.barplot(
        data=df_corr_plot,
        y='Feature',
        x='Correlation_Magnitude',
        hue='Test',
        dodge=False,
        palette='muted'
    )
    plt.title(f'Correlation Magnitude with {target} by Feature (TOP 50)')
    plt.xlabel('Correlation Magnitude (Absolute Point Biserial or CramÃ©r\'s V)')
    plt.ylabel('Feature')
    plt.xlim(0, 1)
    plt.legend(title='Test Type')
    plt.tight_layout()
    plt.show()

    return df_corr_plot.head()


correlation_analysis_and_plot(train_nonan, numeric_cols, cat_cols, target='Personality')


def cpu_impute(df, numeric_cols, categorical_cols, n_neighbors=5):
    """
    Perform KNN imputation on numeric columns and mode imputation on categoricals,
    adding missing-indicator flags for each. Operates purely on pandas DataFrame.
    """
    df = df.copy()

    # Add missing flags for numeric columns
    for col in numeric_cols:
        df[col + '_is_missing'] = df[col].isna().astype(int)

    # Numeric: KNN Imputer
    knn = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_cols] = knn.fit_transform(df[numeric_cols])

    # Categorical: fill with mode + missing flags
    for col in categorical_cols:
        df[col + '_is_missing'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df

numeric_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
cat_cols = ['Stage_fear','Drained_after_socializing']

train_imputed = cpu_impute(train, numeric_cols, cat_cols)
test_imputed  = cpu_impute(test, numeric_cols, cat_cols)


train_imputed.head()


# Add constant feature
def add_constant(df, value=0):
    df['constant_zero_feature'] = value
    return df

train_imputed = add_constant(train_imputed, 0)
test_imputed  = add_constant(test_imputed, 0)


from sklearn.preprocessing import MinMaxScaler

### 3. Feature Engineering ###
def preprocess_and_fe(df):
    df = df.copy()

    # 3.1 Binaryâ€�encode original categoricals
    for c in cat_cols:
        df[c] = df[c].map({'Yes':1,'No':0})

    # 3.2 Thresholdâ€�based binary features on RAW values
    df['Time_spent_Alone_lt4']   = (df['Time_spent_Alone'] <  4).astype(int)
    df['Time_spent_Alone_gt5']   = (df['Time_spent_Alone'] >  5).astype(int)
    
    df['Social_event_attendance_lt3'] = (df['Social_event_attendance'] < 3).astype(int)
    df['Social_event_attendance_gt3'] = (df['Social_event_attendance'] > 3).astype(int)
    
    df['Going_outside_gt3']      = (df['Going_outside'] > 3).astype(int)
    df['Going_outside_lt2']      = (df['Going_outside'] < 3).astype(int)
    
    df['Friends_circle_size_gt5']= (df['Friends_circle_size'] > 5).astype(int)
    df['Friends_circle_size_lt5']= (df['Friends_circle_size'] < 5).astype(int)
    
    df['Post_frequency_gt3']     = (df['Post_frequency'] > 3).astype(int)
    df['Post_frequency_lt3']     = (df['Post_frequency'] < 3).astype(int)

    # 3.3 Minâ€�Max scale numerics
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Step 3.4: Sum of all numeric features
    df['numeric_sum'] = df[numeric_cols].sum(axis=1)

    # 3.5 Pairwise sum & product
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i+1:]:
            df[f'{c1}_plus_{c2}']  = df[c1] + df[c2]
            df[f'{c1}_times_{c2}'] = df[c1] * df[c2]

    # 3.6 Total missing count
    miss_cols = [col for col in df.columns if col.endswith('_is_missing')]
    df['missing_count'] = df[miss_cols].sum(axis=1)

    return df

# Apply to your dataset
train_fe = preprocess_and_fe(train_imputed)
test_fe  = preprocess_and_fe(test_imputed)


cat_cols_updated = [
    # Binary-encoded original categoricals
    'Stage_fear',
    'Drained_after_socializing',
    'constant_zero_feature',

    # Missing indicators
    'Time_spent_Alone_is_missing',
    'Social_event_attendance_is_missing',
    'Going_outside_is_missing',
    'Friends_circle_size_is_missing',
    'Post_frequency_is_missing',
    'Stage_fear_is_missing',
    'Drained_after_socializing_is_missing',

    # Threshold-based binary features
    'Time_spent_Alone_lt4',
    'Time_spent_Alone_gt5',
    'Social_event_attendance_lt3',
    'Social_event_attendance_gt3',
    'Going_outside_gt3',
    'Going_outside_lt2',
    'Friends_circle_size_gt5',
    'Friends_circle_size_lt5',
    'Post_frequency_gt3',
    'Post_frequency_lt3'
]

# Start with scaled original numerics
numeric_cols_updated = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency',

    # Aggregated/statistical numerics
    'numeric_sum',
    'missing_count'
]

# Add pairwise sums and products
for i, c1 in enumerate(numeric_cols[:]):
    for c2 in numeric_cols[i+1:]:
        numeric_cols_updated.append(f'{c1}_plus_{c2}')
        numeric_cols_updated.append(f'{c1}_times_{c2}')



# Now call the function on your expanded feature dataset:
correlation_analysis_and_plot(train_fe, numeric_cols_updated, cat_cols_updated, target='Personality')


y = train["Personality"]

train_fe.head()


from sklearn.model_selection import KFold

def smooth_target_encoding(train, test, cols, target, n_splits=5, alpha=20, random_state=42):
    """
    train, test : DataFrames
    cols        : list of categorical columns to encode
    target      : name of the target column (in train only)
    n_splits    : number of CV folds
    alpha       : smoothing parameter (higher â†’ more global mean weight)
    """
    # prepare output
    train_enc = pd.DataFrame(index=train.index)
    test_enc  = pd.DataFrame(index=test.index)
    
    # global mean
    global_mean = train[target].mean()
    
    # CV splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for col in cols:
        # outâ€‘ofâ€‘fold encodings
        oof = pd.Series(index=train.index, dtype=float)
        
        for tr_idx, val_idx in kf.split(train):
            tr = train.iloc[tr_idx]
            val = train.iloc[val_idx]
            
            # compute perâ€‘category mean & count on this foldâ€™s train
            stats = tr.groupby(col)[target].agg(['mean','count'])
            stats['smooth'] = (
                stats['mean'] * stats['count'] + global_mean * alpha
            ) / (stats['count'] + alpha)
            
            # map to validation fold, fill unseen with global mean
            oof.iloc[val_idx] = (
                val[col]
                  .map(stats['smooth'])
                  .fillna(global_mean)
            )
        
        train_enc[col] = oof
        
        # now fit on full train â†’ apply to test
        stats_full = train.groupby(col)[target].agg(['mean','count'])
        stats_full['smooth'] = (
            stats_full['mean'] * stats_full['count'] + global_mean * alpha
        ) / (stats_full['count'] + alpha)
        
        test_enc[col] = (
            test[col]
              .map(stats_full['smooth'])
              .fillna(global_mean)
        )
    
    return train_enc.add_prefix('TE_'), test_enc.add_prefix('TE_')


# Prepare data for TE
train_te = pd.DataFrame(index=train_fe.index)
test_te  = pd.DataFrame(index=test_fe.index)


# Single-column TE
train_te, test_te = smooth_target_encoding(
    train = train_fe.assign(Personality=y),
    test  = test_fe,
    cols  = cat_cols_updated,
    target= 'Personality',
    n_splits=5,
    alpha=20
)

# merge in your new TE features
train_final = pd.concat([train_fe, train_te], axis=1)
test_final  = pd.concat([test_fe , test_te ], axis=1)


from itertools import combinations

# Step 1: Generate pairwise columns and collect names
pairwise_cols = []
for c1, c2 in combinations(cat_cols_updated, 2):
    new_col = f'{c1}_{c2}'
    train_fe[new_col] = train_fe[c1].astype(str) + '-' + train_fe[c2].astype(str)
    test_fe[new_col] = test_fe[c1].astype(str) + '-' + test_fe[c2].astype(str)
    pairwise_cols.append(new_col)

train_pair_te, test_pair_te = smooth_target_encoding(
    train=train_fe.assign(Personality=y),
    test=test_fe,
    cols=pairwise_cols,
    target='Personality',
    n_splits=5,
    alpha=20
)

train_f = pd.concat([train_final, train_pair_te], axis=1)
test_f = pd.concat([test_final, test_pair_te], axis=1)


train_f.columns


X = train_f.drop(columns=['Personality'])
y = train_f['Personality']


import time
import optuna
import xgboost as xgb

def objective(trial, X, y):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',   # updated per warning
        'device': 'cuda',        # new param for GPU
        'random_state': 42,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,  # pass here, not in fit
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.02, log=True),  # updated
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'enable_categorical': False,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    losses = []

    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict_proba(X_val)[:, 1]
        loss = log_loss(y_val, preds)
        losses.append(loss)
        print(f"  Fold {fold} logloss: {loss:.5f}")

    mean_loss = np.mean(losses)
    elapsed = time.time() - start_time
    print(f"Trial completed. Mean logloss: {mean_loss:.5f}, Time elapsed: {elapsed:.2f} seconds\n")

    return mean_loss


def tune_xgb_params(X, y, n_trials=50):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print('Best trial:')
    trial = study.best_trial
    print(f'  Logloss: {trial.value}')
    print('  Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    
    best_params = trial.params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['tree_method'] = 'hist'
    best_params['random_state'] = 42
    best_params['n_estimators'] = 1000
    best_params['enable_categorical'] = False

    return best_params


best_params = tune_xgb_params(X, y, n_trials=50)


from sklearn.metrics import log_loss, accuracy_score

# Remove early_stopping_rounds since there's no validation set
final_xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'n_estimators': 1000,
    'enable_categorical': False,
}
final_xgb_params.update(best_params)

# 1. Create final model
final_model = xgb.XGBClassifier(**final_xgb_params)

# 2. Fit model on full training data
final_model.fit(X, y, verbose=True)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = []
ll_scores  = []

for train_idx, val_idx in kf.split(X, y):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
    
    m = xgb.XGBClassifier(**final_xgb_params)
    m.fit(X_tr, y_tr)
    
    proba_va = m.predict_proba(X_va)[:,1]
    preds_va = (proba_va > 0.5).astype(int)
    
    acc_scores.append( accuracy_score(y_va, preds_va) )
    ll_scores.append(  log_loss(y_va, proba_va) )

print(f"CV Accuracy: {np.mean(acc_scores):.4f} Â± {np.std(acc_scores):.4f}")
print(f"CV Logâ€‘Loss : {np.mean(ll_scores):.4f} Â± {np.std(ll_scores):.4f}")


# Plot feature importance
plt.figure(figsize=(12, 20))  # adjust size for readability
xgb.plot_importance(final_model,
                    max_num_features=25,       # top 50 features
                    importance_type='gain',    # options: 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
                    show_values=False,         # optionally hide importance values
                    height=0.5)                # bar height
plt.title('Top 50 Feature Importances')
plt.tight_layout()
plt.show()


cat_cols_updated2 = [
    # Binary-encoded original categoricals
    'Stage_fear',
    'Drained_after_socializing',
    'constant_zero_feature',

    # Missing indicators
    'Time_spent_Alone_is_missing',
    'Social_event_attendance_is_missing',
    'Going_outside_is_missing',
    'Friends_circle_size_is_missing',
    'Post_frequency_is_missing',
    'Stage_fear_is_missing',
    'Drained_after_socializing_is_missing',

    # Threshold-based binary features
    'Time_spent_Alone_lt4',
    'Time_spent_Alone_gt5',
    'Social_event_attendance_lt3',
    'Social_event_attendance_gt3',
    'Going_outside_gt3',
    'Going_outside_lt2',
    'Friends_circle_size_gt5',
    'Friends_circle_size_lt5',
    'Post_frequency_gt3',
    'Post_frequency_lt3'
]

numeric_cols_updated2 = [col for col in test_f.columns if col not in cat_cols_updated2]


correlation_analysis_and_plot(train_f, numeric_cols_updated2, cat_cols_updated2, target='Personality')


test_preds_proba = final_model.predict_proba(test_f)[:, 1]
test_preds = (test_preds_proba > 0.5).astype(int)


# 6. Submission
submission['Personality'] = test_preds
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv('submission.csv')
print("Saved submission.csv")

# 7. Count introverts
num_introverts = (submission['Personality'] == 'Introvert').sum()
print(f"Number of Introverts: {num_introverts}")

