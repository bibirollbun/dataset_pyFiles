import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')



df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


df_train.head()


df_train.shape


df_train.describe()


df_train.isnull().sum()


import pandas as pd

summary = pd.DataFrame({
    "dtype": df_train.dtypes.astype(str),
    "non_null": df_train.notna().sum(),
    "missing_%": (df_train.isna().mean() * 100).round(2),
    "unique": df_train.nunique(dropna=True)
})

def example_val(s):
    x = s.dropna()
    return x.iloc[0] if not x.empty else None

summary["example"] = [example_val(df_train[c]) for c in df_train.columns]

# Nice ordering
summary = summary[["dtype", "non_null", "missing_%", "unique", "example"]]
summary



# 0) Make sure age is numeric (non-numeric -> NaN)
df_train['age'] = pd.to_numeric(df_train['age'], errors='coerce')

# 1) Define bins and labels
bins   = [-np.inf, 18, 25, 35, 45, 55, 65, np.inf]   # left-closed, right-open with right=False
labels = ['<18', '18–24', '25–34', '35–44', '45–54', '55–64', '65+']

# 2) Create an ordered categorical column
df_train['age_group'] = pd.cut(
    df_train['age'],
    bins=bins,
    labels=labels,
    right=False,      # [18,25) gives 18–24
    ordered=True
)

# 3) Quick summary (counts + %)
age_summary = (
    df_train['age_group']
      .value_counts(dropna=False)
      .sort_index()
      .rename_axis('age_group')
      .reset_index(name='count')
)
age_summary['percent'] = (age_summary['count'] / len(df_train) * 100).round(1)
age_summary


sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)

x = pd.to_numeric(df_train['age'], errors='coerce').dropna()
n = x.size
mu, sd = x.mean(), x.std(ddof=1)
skew = stats.skew(x, bias=False)
kurt = stats.kurtosis(x, fisher=True, bias=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# --- Left: Histogram + KDE + fitted Normal PDF + mean/median lines
sns.histplot(
    x, bins=20, stat="density", kde=False,
    color="#6f42c1", edgecolor="white", alpha=0.8, ax=axes[0]
)
sns.kdeplot(x, ax=axes[0], linewidth=2)
xs = np.linspace(x.min(), x.max(), 300)
pdf = stats.norm.pdf(xs, loc=mu, scale=sd if sd > 0 else 1)
axes[0].plot(xs, pdf, lw=1.8, label="Normal PDF", color="#d62728")

axes[0].axvline(mu, color="black", ls="--", lw=1.6, label=f"Mean {mu:.1f}")
axes[0].axvline(np.median(x), color="#1f77b4", ls=":", lw=1.8, label=f"Median {np.median(x):.1f}")

axes[0].set_title(f"Distribution of Age (n={n})")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Density")
axes[0].legend(frameon=False, ncol=2)

# --- Right: Q-Q plot vs Normal
(osm, osr), (slope, intercept, r) = stats.probplot(x, dist="norm")
axes[1].scatter(osm, osr, s=18, alpha=0.75)
line_x = np.array([np.min(osm), np.max(osm)])
line_y = slope * line_x + intercept
axes[1].plot(line_x, line_y, lw=2)
axes[1].set_title(f"Q-Q Plot vs Normal (R² ≈ {r**2:.3f})")
axes[1].set_xlabel("Theoretical quantiles")
axes[1].set_ylabel("Ordered values")

sns.despine()
plt.tight_layout()
plt.show()



px.bar(
    age_summary,
    x='age_group',
    y='count',
    color='count',
    text='count',
    title='Age Group Distribution',
    labels={'age_group': 'Age Group', 'count': 'Number of Customers'},
    color_discrete_sequence=['#636EFA']
).update_traces(textposition='outside').update_layout(yaxis_range=[0, age_summary['count'].max() * 1.1])


df_train.head()


df_train['job'].unique()


job_unique = []
for col in df_train['job'].unique():
    job_unique.append(col)

counts = []

for col in job_unique:
    total = (df_train['job'] == col).sum()
    counts.append(total)


job_to_count = dict(zip(job_unique, counts))

job_counts = pd.Series(counts, index=job_unique, name='count').sort_values(ascending=False)

job_counts_df = job_counts.reset_index(name='job')
job_counts_df


plt.figure(figsize=(12,10))
px.bar(job_counts_df, x='index', y='job', color='job', title='Job Distribution', labels={'index': 'Job', 'job': 'Count'})


df_train.head()


print("The unique values in education column are: ", df_train['education'].unique())
print("The unique values in marital column are: ", df_train['education'].value_counts().unique())


vc = df_train['education'].value_counts(dropna=False)
labels = vc.index.astype(str)
sizes  = vc.values

max_idx = int(np.argmax(sizes))
pull    = [0.12 if i == max_idx else 0 for i in range(len(sizes))]

fig = go.Figure(
    data=[go.Pie(labels=labels, values=sizes, pull=pull, textinfo='percent+label')]
)
fig.update_layout(title="Education distribution (largest slice pulled out)")
fig.show()


import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*use_inf_as_na.*")

categorical_columns = ['marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

plt.figure(figsize=(14,14))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=column, data=df_train, palette='Set2',hue='y')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# only numerical columns 
numerical_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
numerical_cols
# correlation matrix

corr_matrix = df_train[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis', square=True)
plt.title('Correlation Matrix of Numerical Features')
plt.show()



print("Loading data...")

train = df_train
test = df_test

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

X = train.drop('y', axis=1)
y = train['y']
X_test = test.copy()

print(f"\nTarget distribution:")
print(y.value_counts(normalize=True))

def preprocess_data(X_train, X_test):
    
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                          'loan', 'contact', 'month', 'poutcome']
    
    
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        
        combined_data = pd.concat([X_train_processed[feature], X_test_processed[feature]])
        le.fit(combined_data)
        
        X_train_processed[feature] = le.transform(X_train_processed[feature])
        X_test_processed[feature] = le.transform(X_test_processed[feature])
        label_encoders[feature] = le
    
    
    X_train_processed['age_balance'] = X_train_processed['age'] * X_train_processed['balance']
    X_test_processed['age_balance'] = X_test_processed['age'] * X_test_processed['balance']
    
    X_train_processed['duration_campaign'] = X_train_processed['duration'] * X_train_processed['campaign']
    X_test_processed['duration_campaign'] = X_test_processed['duration'] * X_test_processed['campaign']
    
   
    X_train_processed['age_group'] = pd.cut(X_train_processed['age'], bins=5, labels=False)
    X_test_processed['age_group'] = pd.cut(X_test_processed['age'], bins=5, labels=False)
    
    X_train_processed['balance_group'] = pd.cut(X_train_processed['balance'], bins=10, labels=False)
    X_test_processed['balance_group'] = pd.cut(X_test_processed['balance'], bins=10, labels=False)
    
    
    X_train_processed = X_train_processed.fillna(-1)
    X_test_processed = X_test_processed.fillna(-1)
    
    return X_train_processed, X_test_processed, label_encoders

print("\nApplying preprocessing...")
X_processed, X_test_processed, label_encoders = preprocess_data(X, X_test)

print(f"Processed train shape: {X_processed.shape}")
print(f"Processed test shape: {X_test_processed.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

print("\nPreprocessing completed successfully!")
print("Ready for model training...")


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

print("="*50)
print("MODEL 1: LightGBM Classifier")
print("="*50)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'verbosity': -1,
    'random_state': 42,
    'n_estimators': 1000,
    'early_stopping_rounds': 100
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_predictions = np.zeros(len(X_processed))
test_predictions = np.zeros(len(X_test_processed))

print("Training LightGBM with 5-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y)):
    print(f"\nFold {fold + 1}/5")
    
    
    X_fold_train, X_fold_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    
    val_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test_processed, num_iteration=model.best_iteration)
    
    oof_predictions[val_idx] = val_pred
    test_predictions += test_pred / 5
    
    fold_score = roc_auc_score(y_fold_val, val_pred)
    cv_scores.append(fold_score)
    print(f"Fold {fold + 1} ROC AUC: {fold_score:.6f}")

overall_cv_score = roc_auc_score(y, oof_predictions)
print(f"\n" + "="*30)
print(f"LightGBM Results:")
print(f"CV Scores: {[f'{score:.6f}' for score in cv_scores]}")
print(f"Mean CV Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
print(f"Overall OOF Score: {overall_cv_score:.6f}")
print(f"="*30)

lgb_oof = oof_predictions.copy()
lgb_test = test_predictions.copy()
lgb_score = overall_cv_score

print(f"\nLightGBM training completed!")
print(f"Best ROC AUC Score: {lgb_score:.6f}")


import pandas as pd
import numpy as np

# load sample to get the exact id order & column names
sample = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")   # has columns like: id,y

# plug in your probabilities (keep probabilities, don't threshold)
preds = np.clip(lgb_test, 1e-6, 1 - 1e-6)       # optional, avoids 0/1 extremes
assert len(preds) == len(sample), "Length mismatch between preds and sample_submission!"

sample.iloc[:, 1] = preds     # assumes 2nd column is the target (e.g., 'y')
sample.to_csv("submission.csv", index=False)
print("Saved submission.csv"); 

