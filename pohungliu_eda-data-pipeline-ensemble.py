import pandas as pd
import numpy as np

import re
import sys
import math

import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno
import scipy.stats as stats
from patsy import dmatrices
import statsmodels.api as sm 


from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# Setting up the environment
plt.style.use('seaborn-v0_8-darkgrid')

if sys.platform == 'win32':
    print('Win')
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
elif sys.platform == 'darwin':
    print('Mac')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rcParams['axes.unicode_minus']=False
# setting pandas display options
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

# pd.set_option('display.max_columns', 20)  # or 1000
# pd.set_option('display.max_rows', 80)  # or 1000
# pd.set_option('display.max_colwidth', 20)  # or 199


# train_df_path = './train.csv'
train_df_path = '/kaggle/input/playground-series-s5e8/train.csv'

# test_df_path = './test.csv'
test_df_path = '/kaggle/input/playground-series-s5e8/test.csv'


train_df = pd.read_csv(train_df_path)
display(train_df.head())
display(train_df.info())


test_df = pd.read_csv(test_df_path )
display(test_df.head())
display(test_df.info())



data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
display(data.head())
display(data.info())


data_with_na = data.drop('y', axis=1)  # Exclude the target variable
data_with_na = data_with_na[data_with_na.isnull().any(axis=1)]
display(data_with_na.head())
data_na_count = data_with_na.isnull().sum()
# add a column for the percentage of missing values
data_na_percent = data_na_count / len(data) * 100
data_na_summary = pd.DataFrame({
    'Missing Values': data_na_count,
    'Percentage': data_na_percent
}).sort_values(by='Missing Values', ascending=False)
display(data_na_summary)

# Visualizing missing values
msno.matrix(data, figsize=(12, 6))
display(plt.show())


# Load the dataset
df = pd.read_csv(train_df_path)

# Encode subscription target
df['subscribed'] = df['y']

# Drop the original target column
palette = sns.color_palette("tab10")

# Compute rates
age_rate = df.groupby('age')['subscribed'].mean().reset_index()
job_rate = df.groupby('job')['subscribed'].mean().reset_index().sort_values('subscribed', ascending=False)
education_rate = df.groupby('education')['subscribed'].mean().reset_index().sort_values('subscribed', ascending=False)
marital_rate = df.groupby('marital')['subscribed'].mean().reset_index().sort_values('subscribed', ascending=False)

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Relationship Between Age and Subscription Rate
sns.lineplot(
    data=age_rate,
    x='age',
    y='subscribed',
    marker='o',
    color=palette[0],
    ax=axes[0, 0]
)
axes[0, 0].set_title('Relationship Between Age and Subscription Rate')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Subscription Rate')

# 2. Impact of Job Type on Subscription Rates
sns.barplot(
    data=job_rate,
    x='job',
    y='subscribed',
    order=job_rate['job'],
    palette=palette,
    ax=axes[0, 1]
)
axes[0, 1].set_title('Impact of Job Type on Subscription Rates')
axes[0, 1].set_xlabel('Job Type')
axes[0, 1].set_ylabel('Subscription Rate')
labels = axes[0, 1].get_xticklabels()
axes[0, 1].set_xticklabels(labels, ha='left')  # ha='right' / 'center' / 'left'
axes[0, 1].tick_params(axis='x', rotation=-45)

# 3. Effect of Education Level on Subscription Rates
sns.barplot(
    data=education_rate,
    x='education',
    y='subscribed',
    order=education_rate['education'],
    palette=palette,
    ax=axes[1, 0]
)
axes[1, 0].set_title('Effect of Education Level on Subscription Rates')
axes[1, 0].set_xlabel('Education Level')
axes[1, 0].set_ylabel('Subscription Rate')
axes[1, 0].tick_params(axis='x', rotation=0)

# 4. Influence of Marital Status on Subscription Rates
sns.barplot(
    data=marital_rate,
    x='marital',
    y='subscribed',
    order=marital_rate['marital'],
    palette=palette,
    ax=axes[1, 1]
)
axes[1, 1].set_title('Influence of Marital Status on Subscription Rates')
axes[1, 1].set_xlabel('Marital Status')
axes[1, 1].set_ylabel('Subscription Rate')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()




palette = sns.color_palette("tab10")

# Encode the subscription target
df['subscribed'] = df['y']

# ──────────────────────────────────────
# 1. Relationship between account balance and subscription rate
bins = 300
df['balance_bin'] = pd.cut(df['balance'], bins=bins)
balance_rate = (
    df
    .groupby('balance_bin')['subscribed']
    .mean()
    .reset_index()
)
# Calculate the midpoint of each bin
balance_rate['balance_mid'] = balance_rate['balance_bin'].apply(lambda iv: iv.mid)

# ──────────────────────────────────────
# 2. Effect of loan status on subscription rate
loan_rate = df.groupby('loan')['subscribed'].mean().reset_index()

# ──────────────────────────────────────
# 3. Effect of default history on subscription rate
default_rate = df.groupby('default')['subscribed'].mean().reset_index()

# ──────────────────────────────────────
# Plot a 1×3 grid of charts
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# Chart 1: Account balance vs subscription rate
sns.lineplot(
    data=balance_rate.sort_values('balance_mid'),
    x='balance_mid',
    y='subscribed',
    marker='o',
    ax=axes[0],
    color=palette[0]
)
axes[0].set_title('Account Balance vs Subscription Rate')
axes[0].set_xlabel('Balance Midpoint')
axes[0].set_ylabel('Subscription Rate')

# Chart 2: Loan status vs subscription rate
sns.barplot(
    data=loan_rate,
    x='loan',
    y='subscribed',
    palette=palette,
    ax=axes[1]
)
axes[1].set_title('Loan Status vs Subscription Rate')
axes[1].set_xlabel('Has Loan')
axes[1].set_ylabel('Subscription Rate')

# Chart 3: Default history vs subscription rate
sns.barplot(
    data=default_rate,
    x='default',
    y='subscribed',
    palette=palette,
    ax=axes[2]
)
axes[2].set_title('Default History vs Subscription Rate')
axes[2].set_xlabel('Defaulted Before')
axes[2].set_ylabel('Subscription Rate')

plt.tight_layout()
plt.show()





# 2. Encode subscription target
df['subscribed'] = df['y']

# 3. Set Seaborn styling
sns.set_theme(style="whitegrid")
palette = sns.color_palette("tab10")

# ─── 1. Contact Method vs Subscription Rate ───
contact_rate = (
    df
    .groupby('contact')['subscribed']
    .mean()
    .reset_index()
)

# ─── 2. Contact Duration vs Subscription Rate ───
# Bin 'duration' into 50 quantile-based intervals
df['dur_bin'] = pd.qcut(df['duration'], q=50, duplicates='drop')
dur_rate = (
    df
    .groupby('dur_bin')['subscribed']
    .mean()
    .reset_index()
)
# Compute midpoint of each duration bin
dur_rate['dur_mid'] = dur_rate['dur_bin'].apply(lambda iv: iv.mid)

# ─── 3. Contact Frequency vs Subscription Rate ───
freq_rate = (
    df
    .groupby('campaign')['subscribed']
    .mean()
    .reset_index()
)

# ─── 4. Last Contact Timing (Month & Day) ───
# Create a heatmap: index = day, columns = month
month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
pivot = (
    df
    .pivot_table(index='day', columns='month', values='subscribed', aggfunc='mean')
    .reindex(columns=month_order)
)

# ─── 5. Campaign Outcome vs Subscription Rate ───
pout_rate = (
    df
    .groupby('poutcome')['subscribed']
    .mean()
    .reset_index()
)

# ─── 6. Previous Success (binary) vs Subscription Rate ───
df['prev_succ'] = (df['poutcome'] == 'success').astype(int)
prev_succ_rate = (
    df
    .groupby('prev_succ')['subscribed']
    .mean()
    .reset_index()
)

# ────────────────────────────
# Plot a 2×3 grid of charts
fig, axes = plt.subplots(2, 3, figsize=(21, 12))

# (1) Contact Method
sns.barplot(
    data=contact_rate,
    x='contact', y='subscribed',
    palette=palette, ax=axes[0, 0]
)
axes[0, 0].set_title('Contact Method vs Subscription Rate')
axes[0, 0].set_xlabel('Contact Method')
axes[0, 0].set_ylabel('Subscription Rate')

# (2) Contact Duration
sns.lineplot(
    data=dur_rate.sort_values('dur_mid'),
    x='dur_mid', y='subscribed',
    marker='o', ax=axes[0, 1], color=palette[1]
)
axes[0, 1].set_title('Contact Duration vs Subscription Rate')
axes[0, 1].set_xlabel('Duration (seconds, bin midpoint)')
axes[0, 1].set_ylabel('Subscription Rate')

# (3) Contact Frequency
sns.lineplot(
    data=freq_rate,
    x='campaign', y='subscribed',
    marker='o', ax=axes[0, 2], color=palette[2]
)
axes[0, 2].set_title('Contact Frequency vs Subscription Rate')
axes[0, 2].set_xlabel('Number of Contacts')
axes[0, 2].set_ylabel('Subscription Rate')

# (4) Last Contact Timing Heatmap
sns.heatmap(
    pivot, ax=axes[1, 0],
    cmap='YlGnBu', cbar_kws={'label': 'Subscription Rate'}
)
axes[1, 0].set_title('Subscription Rate by Last Contact (Day × Month)')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Day of Month')

# (5) Previous Campaign Outcome
sns.barplot(
    data=pout_rate,
    x='poutcome', y='subscribed',
    order=['unknown','failure','other','success'],
    palette=palette, ax=axes[1, 1]
)
axes[1, 1].set_title('Previous Campaign Outcome vs Subscription Rate')
axes[1, 1].set_xlabel('Previous Outcome')
axes[1, 1].set_ylabel('Subscription Rate')

# (6) Previous Success Flag
sns.barplot(
    data=prev_succ_rate,
    x='prev_succ', y='subscribed',
    palette=palette, ax=axes[1, 2]
)
axes[1, 2].set_title('Previous Success Flag vs Subscription Rate')
axes[1, 2].set_xlabel('Previous Success (0 = No, 1 = Yes)')
axes[1, 2].set_ylabel('Subscription Rate')

plt.tight_layout()
plt.show()




from scipy.stats import ks_2samp, chi2_contingency

# === 1. Load datasets ===
train = pd.read_csv(train_df_path)
test  = pd.read_csv(test_df_path)

# === 2. Define features ===
numeric_features = ['age', 'balance', 'duration']
categorical_features = ['job', 'marital', 'education', 'contact', 'poutcome']

# === 3. Data cleaning and consistency check ===
# Normalize categorical strings (lowercase + strip spaces)
for col in categorical_features:
    train[col] = train[col].astype(str).str.strip().str.lower()
    test[col] = test[col].astype(str).str.strip().str.lower()

# Check for categories in test set that are not present in train set
for col in categorical_features:
    diff_test = set(test[col].unique()) - set(train[col].unique())
    if diff_test:
        print(f"[Warning] {col} has categories in Test not in Train: {diff_test}")

# === 4. Seaborn visualization settings ===
sns.set_theme(style="whitegrid")
train_color = "#1f77b4"
test_color  = "#ff7f0e"

# === 5. Numeric features distribution (KDE Plot + KS Test) ===
fig, axes = plt.subplots(len(numeric_features), 1, figsize=(10, 5 * len(numeric_features)))
if len(numeric_features) == 1:
    axes = [axes]  # Ensure axes is iterable

for ax, col in zip(axes, numeric_features):
    sns.kdeplot(train[col], label='Train', ax=ax, alpha=0.6, color=train_color)
    sns.kdeplot(test[col],  label='Test',  ax=ax, fill=True, alpha=0.2, color=test_color, linestyle="--")
    ax.set_title(f'Distribution of {col.capitalize()} (Train vs Test)')
    ax.set_xlabel(col.capitalize())
    ax.legend()
    
    # KS Test
    stat, p = ks_2samp(train[col], test[col])
    print(f"[KS Test] {col}: p-value={p:.4f} → {'Different' if p<0.05 else 'Similar'}")

plt.tight_layout()
plt.show()

# === 6. Categorical features distribution (Bar Plot + Chi-square Test) ===
fig, axes = plt.subplots(len(categorical_features), 1, figsize=(12, 4 * len(categorical_features)))
if len(categorical_features) == 1:
    axes = [axes]

for ax, col in zip(axes, categorical_features):
    # Order categories by frequency in train set
    train_counts = train[col].value_counts(normalize=True)
    test_counts  = test[col].value_counts(normalize=True)
    categories_order = train_counts.index
    df_cmp = pd.DataFrame({
        'Train': train_counts,
        'Test':  test_counts
    }).fillna(0).reindex(categories_order)
    
    df_cmp.plot.barh(ax=ax, color=[train_color, test_color])
    ax.set_title(f'Proportion of {col.capitalize()} Categories (Train vs Test)')
    ax.set_xlabel('Proportion')
    ax.set_ylabel(col.capitalize())
    ax.legend()
    
    # Chi-square Test
    combined = pd.concat([
        train[col].value_counts(),
        test[col].value_counts()
    ], axis=1).fillna(0)
    chi2, p, _, _ = chi2_contingency(combined)
    print(f"[Chi-square] {col}: p-value={p:.4f} → {'Different' if p<0.05 else 'Similar'}")

plt.tight_layout()
plt.show()



sns.set_theme(style="whitegrid")

# ─────────────────────────────────────────────────
# 3. Plot: Age distribution by subscription outcome
plt.figure(figsize=(10, 5))
sns.histplot(
    data=train_df,
    x="age",
    hue="y",
    bins=40,
    kde=True,
    palette="coolwarm",
    alpha=0.6
)
plt.title("Age Distribution by Subscription Outcome (y)", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend(title="Subscription Outcome (y)", loc='upper right')
plt.show()

# ─────────────────────────────────────────────────
# 4. Plot: Balance distribution by subscription outcome
plt.figure(figsize=(10, 5))
sns.histplot(
    data=train_df,
    x="balance",
    hue="y",
    bins=40,
    kde=True,
    palette="coolwarm",
    alpha=0.6
)
plt.title("Balance Distribution by Subscription Outcome (y)", fontsize=14)
plt.xlabel("Balance")
plt.ylabel("Count")
plt.legend(title="Subscription Outcome (y)", loc='upper right')
plt.show()

# ─────────────────────────────────────────────────
# 5. Plot: Subscription count by education level
#    a) Compute counts, order by subscription rate
edu_counts = train_df.groupby(["education", "y"]) \
                     .size() \
                     .unstack(fill_value=0)
edu_order = (edu_counts[1] / edu_counts.sum(axis=1)) \
               .sort_values(ascending=False) \
               .index
edu_counts = edu_counts.loc[edu_order]

#    b) Melt into long form
edu_long = edu_counts.reset_index().melt(
    id_vars="education",
    value_vars=[0, 1],
    var_name="subscription",
    value_name="count"
)
edu_long["subscription"] = edu_long["subscription"].map({
    0: "Not Subscribed",
    1: "Subscribed"
})

#    c) Plot bar chart
plt.figure(figsize=(10, 6))
sns.barplot(
    data=edu_long,
    x="education",
    y="count",
    hue="subscription",
    hue_order=["Not Subscribed", "Subscribed"],
    palette=["#1f77b4", "#ff7f0e"]
)
plt.title("Subscription Count by Education Level", fontsize=14)
plt.xlabel("Education Level")
plt.ylabel("Number of People")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Subscription Status", loc='upper right')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────
# 6. Plot: Correlation heatmap for numeric features
numeric_cols = train_df.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=[col for col in ["id", "y"] if col in train_df])
corr_matrix = numeric_cols.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.show()

# ─────────────────────────────────────────────────
# 7. Plot: Subscription outcome proportion
plt.figure(figsize=(5, 5))
sns.countplot(
    data=train_df,
    x="y",
    palette="pastel"
)
plt.title("Subscription Outcome Distribution", fontsize=14)
plt.xlabel("y (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(title="Subscription Outcome", loc='upper right')
plt.show()



from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif


pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

# Reading the dataset
class LoadBankData(BaseEstimator, TransformerMixin):
    def __init__(self, train_path=train_df_path):
        self.train_path = train_path

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        df = pd.read_csv(self.train_path)
        return df


# Combine day and month into a single date column
class CombineDayMonth(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['month_num'] = df['month'].map(self.month_map)
        df['date'] = df['day'].astype(str) + '-' + df['month_num'].astype(str)
        return df.drop(columns=['day', 'month'])


# Label Encoding with Target Relation
class LabelEncodingWithTargetRelation(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='y'):
        self.target_col = target_col
        self.categorical_cols = [
            'job', 'education', 'marital',
            'housing', 'loan', 'default',
            'poutcome', 'contact'
        ]
        self.encoding_maps = {}
        self.fitted = False

    def fit(self, X, y=None):
        df = X.copy()
        y_series = df[self.target_col] if self.target_col in df.columns else y
        
        if y_series is None:
            raise ValueError("y is required when fitting LabelEncodingWithTargetRelation")

        for col in self.categorical_cols:
            order = df.groupby(col)[self.target_col].mean().sort_values()
            mapping = {cat: idx for idx, cat in enumerate(order.index)}
            self.encoding_maps[col] = mapping
        
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("This transformer has not been fitted yet.")
        
        df = X.copy()
        for col in self.categorical_cols:
            mapping = self.encoding_maps[col]
            median_code = np.median(list(mapping.values()))
            df[col] = df[col].map(mapping).fillna(median_code).astype(int)
        return df


# Domain-specific feature engineering
class AddDomainFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        # time related features
        df['is_q2'] = (df['month_num'].isin([4,5,6])).astype(int)
        df['is_q4'] = (df['month_num'].isin([10,11,12])).astype(int)
        df['is_month_start'] = (df['date'].str.split('-').str[0].astype(int) <= 5).astype(int)
        df['is_month_end'] = (df['date'].str.split('-').str[0].astype(int) >= 25).astype(int)

        # Marketing-related features
        df['pdays_recent'] = (df['pdays'] < 10).astype(int)
        df['campaign_per_duration'] = df['campaign'] / (df['duration']+1)
        df['balance_per_duration'] = df['balance'] / (df['duration']+1)
        df['campaign_per_pdays'] = df['campaign'] / (df['pdays']+2)
        
        # Log transformation of balance
        df['balance'] = np.log1p(df['balance'].clip(lower=0) )
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df


# Interaction Features Auto-Generation
class InteractionFeaturesAuto(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='y', save_list_path='interaction_feature_list.txt'):
        self.target_col = target_col
        self.save_list_path = save_list_path
        self.manual_pairs = [
            ('job','age'), 
            ('job', 'education'),
            ('job', 'default'),
            ('job', 'housing'),
            ('job', 'duration'),
            ('job', 'balance'),
            ('education', 'loan'),
            ('education', 'default'),
            ('education', 'duration'),
            ('education', 'balance'),
            ('balance', 'duration'),
            ('campaign_per_duration', 'balance'),
            ('campaign_per_duration', 'age'),
            ('campaign_per_duration', 'default'),
            ('campaign_per_duration', 'age'),
            ('campaign_per_duration', 'marital'),
            ('housing','loan'),
            ('housing','duration'),
            ('loan', 'duration'),
            ('contact', 'pdays'),
            ('contact', 'campaign'),
            ('marital', 'education'),
        ]
        self.valid_pairs = []  

    def fit(self, X, y=None):
        df = X.copy()

        self.valid_pairs = [(f1, f2) for f1, f2 in self.manual_pairs if f1 in df.columns and f2 in df.columns]        
        return self

    def transform(self, X):
        df = X.copy()
        interaction_df = pd.DataFrame(index=df.index)
        
        # Generate interaction features
        for f1, f2 in self.valid_pairs:
            col_name = f"{f1}_X_{f2}"
            interaction_df[col_name] = df[f1] * df[f2]
        
        # Save the interaction feature names
        return pd.concat([df, interaction_df], axis=1)

# Create the Pipeline
bank_pipeline = Pipeline(steps=[
    ('load_data', LoadBankData(train_df_path)),
    ('combine_day_month', CombineDayMonth()),
    ('label_encode', LabelEncodingWithTargetRelation()),
    ('add_domain_features', AddDomainFeatures()),
    ('interaction_features', InteractionFeaturesAuto())
])

processed_bank_df = bank_pipeline.fit_transform(None)


display(processed_bank_df.head())
display(processed_bank_df.shape)





import umap
import matplotlib.pyplot as plt
import seaborn as sns

def plot_umap_projection(df, sample_frac=0.1, n_neighbors=15, min_dist=0.1):
    sampled_df = df.sample(frac=sample_frac, random_state=42)
    
    
    X_sample = sampled_df.drop(columns=['id', 'date', 'y']).values
    y_sample = sampled_df['y'].values
    
    reducer = umap.UMAP(
        metric='euclidean', 
        init='spectral',
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        n_jobs=-1,
        verbose=False,
        random_state=42
    )
    
    
    embedding = reducer.fit_transform(X_sample)
    
    
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("Set1", 2)
    
    for label, color in zip([0, 1], palette):
        mask = (y_sample == label)
        plt.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=[color], label='Subscribed' if label == 1 else 'Not Subscribed',
                    alpha=0.6, s=30, edgecolors='none')
    
    plt.legend()
    plt.title(f'UMAP Projection on Random Sample ({int(sample_frac*100)}%)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.show()

plot_umap_projection(processed_bank_df, sample_frac=0.5, n_neighbors=25, min_dist=0.1)



from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.base import clone

from lightgbm import LGBMClassifier
from lightgbm import early_stopping as lgb_early_stopping
from lightgbm import log_evaluation as lgb_log
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# === 1. Data Preparation ===
X = processed_bank_df.drop(columns=["y", "id", "date"])
y = processed_bank_df["y"]

pos_weight = (len(y) - sum(y)) / sum(y)
early_stop_rounds = 300

# === 2. Base models  ===
models = {
    "LGBM": LGBMClassifier(
        random_state=42,
        n_estimators=6583,
        metric='auc',
        objective='binary',
        max_depth=9,
        learning_rate=0.023,
        reg_alpha=1.08,       
        reg_lambda=0.07,
        verbose=-1
    ),
    "XGB_tree": XGBClassifier(
        n_estimators=6134,
        objective='binary:logistic',
        random_state=42,
        subsample=0.8,
        learning_rate=0.02975,
        booster='gbtree',
        reg_alpha=1.0         
    ),
    "CAT": CatBoostClassifier(
        random_state=42,
        eval_metric="Logloss",
        n_estimators=6000,
        learning_rate=0.0211,
        depth=10,
        min_data_in_leaf=8,
        task_type="CPU",
        l2_leaf_reg=3,        
        verbose=0
    ),
    "HGB": HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.03,
        max_depth=8,
        l2_regularization=0.5,   
        random_state=42
    ),
}

# === 3. Cross-validation setup ===
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_models = {name: [] for name in models}
oof_preds = pd.DataFrame(index=np.arange(len(X)))
test_preds = pd.DataFrame()

X_train_all, X_valid_all, y_train_all, y_valid_all = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

for name, base_model in models.items():
    print(f"[{name}] training with fold-averaging ...")
    oof_pred = np.zeros(len(X))
    test_pred_folds = np.zeros((len(X_valid_all), n_splits))  # 儲存每折 test 預測

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # === 每折各類別數量與權重 ===
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        
        # A：scale_pos_weight = neg/pos
        spw = (n_neg / n_pos) if n_pos > 0 else 1.0

        w_pos = len(y_tr) / (2.0 * n_pos)
        w_neg = len(y_tr) / (2.0 * n_neg)
        sw_tr = np.where(y_tr.values == 1, w_pos, w_neg)


        model = clone(base_model)
        if "LGBM" in name:
            model.set_params(scale_pos_weight=spw)   
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=["auc", "average_precision"],
                callbacks=[
                    lgb_early_stopping(stopping_rounds=early_stop_rounds),
                    lgb_log(0)
                ]
            )
        elif "XGB" in name:
            model.set_params(scale_pos_weight=spw)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif "CAT" in name:
            model.set_params(scale_pos_weight=spw)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=early_stop_rounds,
                verbose=500
            )
        else:  # HGB

            model.fit(X_tr, y_tr, sample_weight=sw_tr)

        # OOF forecast
        oof_pred[val_idx] = model.predict_proba(X_val)[:, 1]
        # Save test predictions for this fold
        test_pred_folds[:, fold] = model.predict_proba(X_valid_all)[:, 1]
        fold_models[name].append(model)
        
    # === Save OOF and test predictions ===
    test_pred_mean = test_pred_folds.mean(axis=1)

    oof_preds[name] = oof_pred
    test_preds[name] = test_pred_mean
    print(f"{name} OOF ROC AUC: {roc_auc_score(y, oof_pred):.4f}")

# === 4. Meta model training ===
meta_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    random_state=42
)
meta_model.fit(oof_preds, y)
meta_oof_pred = meta_model.predict_proba(oof_preds)[:, 1]
meta_test_pred = meta_model.predict_proba(test_preds)[:, 1]

print(f"Meta Model (Logit) ROC AUC: {roc_auc_score(y, meta_oof_pred):.4f}")

# === 5. （  Weighted Blending） ===
weights = np.ones(len(models)) / len(models)
blend_pred = test_preds.values @ weights


# === 6. Evaluation and Visualization ===
plt.figure(figsize=(12,5))

# ROC
plt.subplot(1,2,1)
for col in oof_preds.columns:
    fpr, tpr, _ = roc_curve(y, oof_preds[col])
    plt.plot(fpr, tpr, linestyle="--", label=f"{col} (AUC={roc_auc_score(y, oof_preds[col]):.3f})")
fpr_meta, tpr_meta, _ = roc_curve(y, meta_oof_pred)
plt.plot(fpr_meta, tpr_meta, label=f"Meta (AUC={roc_auc_score(y, meta_oof_pred):.3f})", linewidth=2)
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()

# PR
plt.subplot(1,2,2)
for col in oof_preds.columns:
    prec, rec, _ = precision_recall_curve(y, oof_preds[col])
    plt.plot(rec, prec, linestyle="--", label=f"{col} (AP={average_precision_score(y, oof_preds[col]):.3f})")
prec_meta, rec_meta, _ = precision_recall_curve(y, meta_oof_pred)
plt.plot(rec_meta, prec_meta, label=f"Meta (AP={average_precision_score(y, meta_oof_pred):.3f})", linewidth=2)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

label_encoder_fitted      = bank_pipeline.named_steps['label_encode']
interaction_features_fitted = bank_pipeline.named_steps['interaction_features']

bank_pipeline_test = Pipeline(steps=[
    ('load_data',           LoadBankData(test_df_path)),
    ('combine_day_month',   CombineDayMonth()),
    ('add_domain_features', AddDomainFeatures()),
    ('label_encode',        label_encoder_fitted),
    ('interaction_features', interaction_features_fitted)
])

processed_test_df = bank_pipeline_test.transform(None)


test_ids = processed_test_df["id"]
X_test   = processed_test_df.drop(columns=["id", "date"], errors="ignore")

# === 7. Final Predictions on Test Set ===
test_preds_final = pd.DataFrame(index=np.arange(len(X_test)))
for name, model_list in fold_models.items():
    
    fold_stack = np.column_stack([m.predict_proba(X_test)[:, 1] for m in model_list])
    
    test_preds_final[name] = fold_stack.mean(axis=1)

# Meta model prob
meta_test_pred_prob  = meta_model.predict_proba(test_preds_final)[:, 1]


submission_df = pd.DataFrame({
    "id": test_ids,
    "y":  meta_test_pred_prob,     
})

submission_df.to_csv("submission.csv", index=False)
display(submission_df.head())


