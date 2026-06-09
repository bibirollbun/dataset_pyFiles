import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Set plotting style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Set a consistent random state for reproducibility
RANDOM_STATE = 42

# Assume df1 is already loaded in the environment.
df1 = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


# Dataset shape, data types, and non-null counts
print("Dataset Shape:", df1.shape)
print("\nData Types and Non-Null Counts:")
df1.info()

# Basic descriptive statistics for numerical columns
print("\nBasic Descriptive Statistics for Numerical Columns:")
print(df1.describe())

# Identify missing values (NaN) in numerical columns
missing_numerical = df1.select_dtypes(include=np.number).isnull().sum()
print("\nMissing Values (NaN) in Numerical Columns:")
print(missing_numerical[missing_numerical > 0])

# Count 'unknown' values in specified categorical columns
categorical_cols_for_unknown = ['education', 'contact', 'poutcome', 'job']
unknown_counts = {col: (df1[col] == 'unknown').sum() for col in categorical_cols_for_unknown if col in df1.columns}
print("\n'unknown' Counts in Specified Categorical Columns:")
for col, count in unknown_counts.items():
    if count > 0:
        print(f"{col}: {count} ({count/len(df1)*100:.2f}%)")


plt.figure(figsize=(8, 6))
ax = sns.countplot(x='y', data=df1, palette='viridis')
plt.title('Distribution of Target Variable (y)')
plt.xlabel('Subscribed to Term Deposit (0: No, 1: Yes)')
plt.ylabel('Count')

# Add percentage and count annotations
total = len(df1)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n({p.get_height()})'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=10, color='black')
plt.tight_layout()
plt.show()


numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
num_plots_per_row = 4
num_rows_num_uni = int(np.ceil(len(numerical_features) * 2 / num_plots_per_row))
fig_num_uni, axes_num_uni = plt.subplots(num_rows_num_uni, num_plots_per_row, figsize=(20, num_rows_num_uni * 5), constrained_layout=True)
axes_num_uni = axes_num_uni.flatten()

for i, feature in enumerate(numerical_features):
    # Histogram and KDE plot
    sns.histplot(df1[feature], kde=True, ax=axes_num_uni[i*2], bins=50)
    axes_num_uni[i*2].set_title(f'Distribution of {feature}')

    # Box plot
    sns.boxplot(x=df1[feature], ax=axes_num_uni[i*2+1], color='skyblue')
    axes_num_uni[i*2+1].set_title(f'Box Plot of {feature}')

# Hide unused subplots
for j in range(len(numerical_features) * 2, len(axes_num_uni)):
    fig_num_uni.delaxes(axes_num_uni[j])

fig_num_uni.suptitle('Univariate Analysis of Numerical Features', fontsize=16, y=1.02)
plt.show()


categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
n_cols_cat_uni = 3
n_rows_cat_uni = int(np.ceil(len(categorical_features) / n_cols_cat_uni))
fig_cat_uni, axes_cat_uni = plt.subplots(n_rows_cat_uni, n_cols_cat_uni, figsize=(18, n_rows_cat_uni * 5), constrained_layout=True)
axes_cat_uni = axes_cat_uni.flatten()

for i, feature in enumerate(categorical_features):
    ax = sns.countplot(y=feature, data=df1, ax=axes_cat_uni[i], palette='crest', order=df1[feature].value_counts().index)
    axes_cat_uni[i].set_title(f'Distribution of {feature}')
    axes_cat_uni[i].set_xlabel('Count')
    axes_cat_uni[i].set_ylabel('')

    # Add percentage annotations
    total = len(df1)
    for p in ax.patches:
        width = p.get_width()
        percentage = f'{width / total * 100:.1f}%'
        x = width
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y), ha='left', va='center', fontsize=9, color='black')

# Hide unused subplots
for j in range(len(categorical_features), len(axes_cat_uni)):
    fig_cat_uni.delaxes(axes_cat_uni[j])

fig_cat_uni.suptitle('Univariate Analysis of Categorical Features', fontsize=16, y=1.02)
plt.show()


numerical_features_biv = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
n_cols_num_biv = 3
n_rows_num_biv = int(np.ceil(len(numerical_features_biv) / n_cols_num_biv))
fig_num_biv, axes_num_biv = plt.subplots(n_rows_num_biv, n_cols_num_biv, figsize=(18, n_rows_num_biv * 6), constrained_layout=True)
axes_num_biv = axes_num_biv.flatten()

for i, feature in enumerate(numerical_features_biv):
    sns.violinplot(x='y', y=feature, data=df1, ax=axes_num_biv[i], palette='coolwarm')
    axes_num_biv[i].set_title(f'{feature} vs. Target (y)')
    axes_num_biv[i].set_xlabel('Subscribed (0: No, 1: Yes)')
    axes_num_biv[i].set_ylabel(feature)

# Hide unused subplots
for j in range(len(numerical_features_biv), len(axes_num_biv)):
    fig_num_biv.delaxes(axes_num_biv[j])

fig_num_biv.suptitle('Bivariate Analysis: Numerical Features vs. Target', fontsize=16, y=1.02)
plt.show()


categorical_features_biv = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
n_cols_cat_biv = 3
n_rows_cat_biv = int(np.ceil(len(categorical_features_biv) / n_cols_cat_biv))
fig_cat_biv, axes_cat_biv = plt.subplots(n_rows_cat_biv, n_cols_cat_biv, figsize=(18, n_rows_cat_biv * 6), constrained_layout=True)
axes_cat_biv = axes_cat_biv.flatten()

for i, feature in enumerate(categorical_features_biv):
    # Calculate subscription rates (proportion of y=1)
    subscription_rates = df1.groupby(feature)['y'].value_counts(normalize=True).unstack().fillna(0)
    if 1 in subscription_rates.columns:
        subscription_rates_y1 = subscription_rates[1].sort_values(ascending=False)
        sns.barplot(x=subscription_rates_y1.values, y=subscription_rates_y1.index, ax=axes_cat_biv[i], palette='viridis')
        axes_cat_biv[i].set_title(f'Subscription Rate by {feature}')
        axes_cat_biv[i].set_xlabel('Proportion of Subscribers (y=1)')
        axes_cat_biv[i].set_ylabel('')

        # Add annotations
        for index, value in enumerate(subscription_rates_y1.values):
            axes_cat_biv[i].text(value + 0.01, index, f'{value:.2%}', va='center', fontsize=9, color='black')
    else:
        axes_cat_biv[i].set_title(f'No subscribers for {feature}')
        axes_cat_biv[i].text(0.5, 0.5, 'No y=1 observations', ha='center', va='center', transform=axes_cat_biv[i].transAxes)

# Hide unused subplots
for j in range(len(categorical_features_biv), len(axes_cat_biv)):
    fig_cat_biv.delaxes(axes_cat_biv[j])

fig_cat_biv.suptitle('Bivariate Analysis: Categorical Features vs. Target (Subscription Rates)', fontsize=16, y=1.02)
plt.show()


numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
numerical_df = df1[numerical_features + ['y']] # Include target for correlation
correlation_matrix = numerical_df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()


# Make a copy of the original df1 to work on
df = df1.copy()

# Drop 'id' column
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Identify categorical columns for encoding
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# One-Hot Encode categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Numerical Feature Transformation (log1p)
features_to_transform = ['balance', 'duration', 'campaign', 'pdays', 'previous']
for feature in features_to_transform:
    if feature == 'balance':
        # Shift balance to handle non-positive values before log1p
        min_balance = df['balance'].min()
        shift_amount = abs(min_balance) + 1 if min_balance <= 0 else 0
        df[feature] = np.log1p(df[feature] + shift_amount)
    elif feature == 'pdays':
        # Shift pdays to handle -1 before log1p
        df[feature] = np.log1p(df[feature] + 2)
    else:
        df[feature] = np.log1p(df[feature])

# Feature Engineering: Create 'was_contacted_before'
# Derived from the ORIGINAL 'pdays' to capture 'never contacted' (-1) semantics
df['was_contacted_before'] = (df1['pdays'] != -1).astype(int)

print("Shape after preprocessing:", df.shape)
print("\nFirst 5 rows after preprocessing:")
print(df.head())
print("\nData types after preprocessing:")
df.info()
print("\nDescriptive statistics after transformations:")
print(df.describe())


# Numerical features to apply StandardScaler
numerical_features_to_scale = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Apply StandardScaler
scaler = StandardScaler()
df[numerical_features_to_scale] = scaler.fit_transform(df[numerical_features_to_scale])

print("Shape after scaling:", df.shape)
print("\nFirst 5 rows with scaled numerical columns:")
print(df[numerical_features_to_scale].head())
print("\nDescriptive statistics for scaled numerical columns:")
print(df[numerical_features_to_scale].describe())

# Visualize distributions of scaled features
features_to_visualize_scaled = numerical_features_to_scale[:min(len(numerical_features_to_scale), 6)]
num_plots_per_row = 3
num_rows_vis = int(np.ceil(len(features_to_visualize_scaled) / num_plots_per_row))
fig_vis, axes_vis = plt.subplots(num_rows_vis, num_plots_per_row, figsize=(18, num_rows_vis * 5), constrained_layout=True)
axes_vis = axes_vis.flatten() if num_rows_vis > 1 else [axes_vis]

for i, feature in enumerate(features_to_visualize_scaled):
    sns.histplot(df[feature], kde=True, ax=axes_vis[i], bins=50)
    axes_vis[i].set_title(f'Distribution of Scaled {feature}')
    axes_vis[i].set_xlabel(f'Scaled {feature}')
    axes_vis[i].set_ylabel('Frequency')

# Hide any unused subplots
for j in range(len(features_to_visualize_scaled), len(axes_vis)):
    fig_vis.delaxes(axes_vis[j])

fig_vis.suptitle('Distributions of Scaled Numerical Features', fontsize=16, y=1.02)
plt.show()


# Separate features (X) and target (y)
y = df['y']
X = df.drop('y', axis=1)

print(f"Shape of features (X): {X.shape}")
print(f"Shape of target (y): {y.shape}")

# Create dual feature sets: with and without 'duration'
X_with_duration = X.copy()
if 'duration' in X.columns:
    X_without_duration = X.drop('duration', axis=1)
    print(f"Shape of X_without_duration (duration dropped): {X_without_duration.shape}")
else:
    print("Warning: 'duration' column not found in X. X_without_duration not created.")


# Initialize StratifiedKFold for robust cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

print(f"Initialized StratifiedKFold with n_splits={n_splits}, shuffle=True, random_state={RANDOM_STATE}.")
print("StratifiedKFold object:", skf)





import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
def cat_handling_simple(df):
    """Basic categorical handling: Label or One-Hot encode + mapping."""
    
    # 1. Binary columns: yes/no â†’ 1/0
    for col in ['housing', 'loan', 'default']:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    # 2. Month mapping
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    df['month'] = df['month'].map(month_map)

    # 3. One-Hot encode multiclass columns
    df = pd.get_dummies(df, columns=['job','education','marital','contact'], drop_first=True)

    # 4. Label encode poutcome
    df['poutcome'] = LabelEncoder().fit_transform(df['poutcome'].astype(str))

    return df
df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df_test['y'] = -1
df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all = cat_handling_simple(df_all)
dftrain = df_all[df_all['y'] != -1].reset_index(drop=True)
dftest  = df_all[df_all['y'] == -1].drop(columns=['y']).reset_index(drop=True)
X_final = dftrain.drop(columns=['y', 'id'])
y_final = dftrain['y']
X_test = dftest.drop(columns=['id'])
N_MODELS = 20
test_preds = np.zeros((X_test.shape[0], N_MODELS), dtype=np.float32)

# print(f"Training {N_MODELS} XGBoost models")
for i in tqdm(range(N_MODELS), desc="Training Models"):
    
    model = XGBClassifier(
        max_depth=10,
        learning_rate=0.05147421740531445,
        subsample=0.8438286745821164,
        colsample_bytree=0.60892638044329,
        reg_alpha=9.92999981052385,
        reg_lambda=1.2468573262301848e-08,
        n_estimators=534,
        use_label_encoder=False,
        eval_metric='auc',
        tree_method='hist',
        device='cuda',
        random_state=42 + i  # Different seed
    )
    
    model.fit(X_final, y_final)
    test_preds[:, i] = model.predict_proba(X_test)[:, 1]
dftest['Pred'] = test_preds.mean(axis=1)
submission = dftest[['id', 'Pred']]
submission.to_csv("submission.csv", index=False)

print(" submission.csv")





