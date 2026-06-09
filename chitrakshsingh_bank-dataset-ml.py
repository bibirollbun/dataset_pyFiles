import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fgr
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, skew, kurtosis 


sns.set(style="whitegrid", palette="pastel")
plt.rcParams['figure.figsize'] = (10, 6)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb
import optuna


from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, Dataset, train
from catboost import CatBoostClassifier


import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train_df.head()


train_df.shape


test_df.shape


train_df.info()


for col in train_df.columns:
    unique_values = train_df[col].unique()
    print(f'Unique values in {col}: {unique_values}')



num_cols = [
    'age',
    'balance',
    'day',
    'duration',
    'campaign',
    'pdays',
    'previous'
]

cat_cols = [
    'job',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
    'contact',
    'month',
    'poutcome',
    'y'
]



for col in num_cols:
    col_skew = skew(train_df[col].dropna())
    col_kurt = kurtosis(train_df[col].dropna())

    if abs(col_skew) < 0.5:
        dist_type = "Approximately Normal"
    elif col_skew > 0.5:
        dist_type = "Right-Skewed"
    else:
        dist_type = "Left-Skewed"

    print(f"{col}: Skew = {col_skew:.2f}, Kurtosis = {col_kurt:.2f} â†’ {dist_type} distribution")

    plt.figure(figsize=(6, 4))
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f'{col} Distribution ({dist_type})')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index)
    plt.title(f'Frequency of Categories in {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=6)
    
    plt.show()


plt.figure(figsize=(10, 8))
corr = train_df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix (Numerical Features)")
plt.tight_layout()
plt.show()


sns.pairplot(train_df, vars=['duration', 'balance', 'age'], hue='y', plot_kws={'alpha': 0.5})
plt.show()


n_cols = 3
n_rows = int(np.ceil(len(cat_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
for i, col in enumerate(cat_cols):
    ax = axes[i // n_cols, i % n_cols]
    sns.countplot(data=train_df, x=col, hue='y', order=train_df[col].value_counts().index, ax=ax)
    ax.set_title(f'{col} vs Target (y)')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


n_cols = 3
n_rows = int(np.ceil(len(num_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
for i, col in enumerate(num_cols):
    ax = axes[i // n_cols, i % n_cols]
    sns.boxplot(data=train_df, x='y', y=col, ax=ax)
    ax.set_title(f'{col} distribution by Target (y)')
plt.tight_layout()
plt.show()


def clean_classwise(df, skew_thresh=0.5):
    cleaned_parts = []
    for label in df['y'].unique():
        subset = df[df['y'] == label].copy()
        skewed_info = {}

        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if col == 'y': continue
            col_skew = skew(subset[col].dropna())
            skewed_info[col] = col_skew

        normal_cols = [col for col, s in skewed_info.items() if abs(s) <= skew_thresh]
        skewed_cols = [col for col, s in skewed_info.items() if abs(s) > skew_thresh]

        for col in normal_cols:
            mean = subset[col].mean()
            std = subset[col].std()
            subset = subset[(subset[col] >= mean - 3 * std) & (subset[col] <= mean + 3 * std)]

        for col in skewed_cols:
            Q1 = subset[col].quantile(0.25)
            Q3 = subset[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            extreme_upper = subset[col].quantile(0.99)
            subset = subset[(subset[col] >= lower_bound) & ((subset[col] <= upper_bound) | (subset[col] > extreme_upper))]

        cleaned_parts.append(subset)

    return pd.concat(cleaned_parts, ignore_index=True)

train_df = clean_classwise(train_df)
print(train_df['y'].value_counts())

n_cols = 3
n_rows = int(np.ceil(len(num_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
for i, col in enumerate(num_cols):
    ax = axes[i // n_cols, i % n_cols]
    sns.boxplot(data=train_df, x='y', y=col, ax=ax)
    ax.set_title(f'{col} distribution by Target (y)')
plt.tight_layout()
plt.show()


for col in cat_cols:
    contingency = pd.crosstab(train_df[col], train_df['y'])
    chi2, p, _, _ = chi2_contingency(contingency)
    print(f"Chi-square test for {col} vs y: p-value = {p:.4f}")

for col in num_cols:
    group0 = train_df[train_df['y'] == 0][col]
    group1 = train_df[train_df['y'] == 1][col]
    f_stat, p = f_oneway(group0, group1)
    print(f"ANOVA for {col} vs y: p-value = {p:.4f}")


train_df['balance_duration'] = train_df['balance'] * train_df['duration']
train_df['balance_campaign_ratio'] = train_df['balance'] / (train_df['campaign'] + 1e-5)
train_df['duration_pdays_diff'] = train_df['duration'] - train_df['pdays']
train_df['age_balance'] = train_df['age'] * train_df['balance']

train_df['job_balance_mean'] = train_df.groupby('job')['balance'].transform('mean')
train_df['marital_duration_std'] = train_df.groupby('marital')['duration'].transform('std')
train_df['education_age_median'] = train_df.groupby('education')['age'].transform('median')

train_df['log_balance'] = np.log1p(train_df['balance'])
train_df['log_duration'] = np.log1p(train_df['duration'])
train_df['log_pdays'] = np.log1p(train_df['pdays'])

train_df['recent_contact'] = (train_df['pdays'] < 30).astype(int)
train_df['never_contacted'] = (train_df['pdays'] == 999).astype(int)

train_df['job_admin'] = (train_df['job'] == 'admin.').astype(int)
train_df['edu_secondary'] = (train_df['education'] == 'secondary').astype(int)

train_df['poutcome_success'] = (train_df['poutcome'] == 'success').astype(int)
train_df['poutcome_failure'] = (train_df['poutcome'] == 'failure').astype(int)

train_df['is_old_high_balance'] = ((train_df['age'] > 50) & (train_df['balance'] > 1000)).astype(int)
train_df['low_duration_no_prev'] = ((train_df['duration'] < 100) & (train_df['previous'] == 0)).astype(int)


df = train_df.copy()
df = df.dropna() 

X_mi = pd.get_dummies(df.drop(columns=['y']), drop_first=True)
y_mi = df['y']


X_mi.replace([np.inf, -np.inf], np.nan, inplace=True)
X_mi.dropna(inplace=True)
y_mi = y_mi.loc[X_mi.index] 


mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features='auto', random_state=42)
mi_df = pd.Series(mi_scores, index=X_mi.columns).sort_values(ascending=False)


mi_df.plot(kind='barh', figsize=(10, 12))
plt.title("Mutual Information Scores")
plt.xlabel("Score")
plt.tight_layout()
plt.show()


df = train_df.copy()
df = df.dropna(subset=['y'])

X = df.drop(columns=['y'])
y = df['y']

X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()
y = y.loc[X.index]

X = pd.get_dummies(X, drop_first=True)
X = X.loc[:, ~X.columns.duplicated()]  # Remove duplicate columns

RANDOM_STATE = 42

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
)


def objective(trial):
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": y_train_full.nunique(),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 31, 512),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        "verbosity": -1,
        "seed": RANDOM_STATE
    }

    # Use a 80/20 validation split for the trial
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, stratify=y_train_full, test_size=0.2, random_state=RANDOM_STATE
    )

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
    params,
    dtrain,
    valid_sets=[dval],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

    preds = model.predict(X_val)
    pred_labels = np.argmax(preds, axis=1)

    return 1.0 - f1_score(y_val, pred_labels, average='weighted')  # minimize (1 - F1)



study = optuna.create_study(direction="minimize", study_name="lgbm_classifier_opt")
study.optimize(objective, n_trials=50)

best_params = study.best_trial.params
best_params.update({
    "objective": "multiclass",
    "num_class": y_train_full.nunique(),
    "verbosity": -1,
    "random_state": RANDOM_STATE
})


print("âœ… Best parameters found:", best_params)


final_model = LGBMClassifier(**best_params)
final_model.fit(X_train_full, y_train_full)

train_preds = final_model.predict(X_train_full)
test_preds = final_model.predict(X_test)



def print_scores(y_true, y_pred, dataset='Test'):
    print(f"\nðŸ“Š {dataset} Set Scores:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred))

print_scores(y_train_full, train_preds, "Training")
print_scores(y_test, test_preds, "Testing")


