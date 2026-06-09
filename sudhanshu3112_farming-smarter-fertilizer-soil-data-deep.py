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


train_path = "/kaggle/input/playground-series-s5e6/train.csv"
test_path = "/kaggle/input/playground-series-s5e6/test.csv"
sample_submission_path = "/kaggle/input/playground-series-s5e6/sample_submission.csv"


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)

# Basic info and view
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample Submission shape:", sample_submission.shape)


print("\nTrain Data Preview:")
display(train.head())

print("\nTest Data Preview:")
display(test.head())

print("\nSample Submission Preview:")
display(sample_submission.head())


print("\nMissing values in Train:")
print(train.isnull().sum())

print("\nMissing values in Test:")
print(test.isnull().sum())


print(train.nunique())


print("List of category variables for train:")
categorical_cols = train.select_dtypes(include=['object', 'category']).columns
print(categorical_cols)

print("\nList numeric variables for train:")
numeric_cols = train.select_dtypes(include=['number']).columns
print(numeric_cols)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set a consistent style
sns.set(style="whitegrid")

# 1. View number of unique fertilizer classes
unique_classes = train['Fertilizer Name'].nunique()
print(f"Number of unique Fertilizer classes: {unique_classes}")

# 2. Display value counts (frequency)
fertilizer_counts = train['Fertilizer Name'].value_counts()
print("\nFertilizer class frequencies:")
print(fertilizer_counts)

# 3. Plot bar chart (log-scale if needed)
plt.figure(figsize=(12, 6))
sns.barplot(
    y=fertilizer_counts.index,
    x=fertilizer_counts.values,
    palette="viridis"
)
plt.xscale("log")  # Apply log scale if imbalance is significant
plt.xlabel("Frequency (log scale)")
plt.ylabel("Fertilizer Name")
plt.title("Fertilizer Class Distribution (Log Scale)")
plt.tight_layout()
plt.show()

# 4. Check class imbalance (top vs. bottom class %)
top_class_pct = fertilizer_counts.iloc[0] / len(train) * 100
bottom_class_pct = fertilizer_counts.iloc[-1] / len(train) * 100
print(f"\nTop class represents {top_class_pct:.2f}% of the data")
print(f"Bottom class represents {bottom_class_pct:.2f}% of the data")

# 5. Plot cumulative distribution of top fertilizers
cumulative_pct = (fertilizer_counts.cumsum() / fertilizer_counts.sum()) * 100

plt.figure(figsize=(10, 6))
sns.lineplot(data=cumulative_pct.values, marker='o')
plt.axhline(80, color='red', linestyle='--', label='80% threshold')
plt.xlabel("Number of Fertilizer Classes")
plt.ylabel("Cumulative Percentage")
plt.title("Cumulative Distribution of Fertilizer Classes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




from scipy.stats import skew

# -------------------------------------------------
# 1. Identify numerical and categorical columns
# -------------------------------------------------
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

# Exclude the target from feature lists if present
target = 'Fertilizer Name'
if target in numeric_cols:  numeric_cols.remove(target)
if target in categorical_cols: categorical_cols.remove(target)

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# -------------------------------------------------
# 2. Univariate Analysis â€“ Numerical Features
# -------------------------------------------------
for col in numeric_cols:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f'Univariate Analysis â€“ {col}', fontsize=14)

    # Histogram
    sns.histplot(train[col].dropna(), ax=axes[0], kde=False)
    axes[0].set_title('Histogram')

    # KDE Plot
    sns.kdeplot(train[col].dropna(), ax=axes[1], shade=True)
    axes[1].set_title('KDE Plot')

    # Boxplot for outliers
    sns.boxplot(x=train[col], ax=axes[2])
    axes[2].set_title('Boxplot (Outliers)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Skewness
    sk = skew(train[col].dropna())
    print(f"Skewness of {col}: {sk:.3f}\n{'-'*60}")






for col in categorical_cols:
    # Basic stats
    n_unique = train[col].nunique()
    value_counts = train[col].value_counts()
    print(f"\n{col} â€” unique categories: {n_unique}")
    print(value_counts.head(10))   # show top 10 categories

    # Bar plot of full distribution (top 30 for readability)
    plt.figure(figsize=(12, 4))
    sns.countplot(
        y=train[col],
        order=value_counts.index[:30]   # limit to top 30 for clarity
    )
    plt.title(f'Frequency Distribution â€“ {col} (Top 30)')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

    # Rare-category detection (< 1 % of rows)
    total_rows = len(train)
    rare_mask = value_counts / total_rows < 0.01
    rare_categories = value_counts[rare_mask].index.tolist()
    rare_pct = rare_mask.sum() / n_unique * 100
    print(f"Rare categories (<1 % of data): {len(rare_categories)} / {n_unique} "
          f"({rare_pct:.1f} %)")
    if rare_categories:
        print("Examples:", rare_categories[:10])



from scipy.stats import chi2_contingency, f_oneway, kruskal

# Assuming 'train' DataFrame is already loaded

target = 'Fertilizer Name'

# Identify categorical and numerical columns excluding target
categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != target]

numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target in numeric_cols:
    numeric_cols.remove(target)




for cat_col in categorical_cols:
    print(f"\nAnalyzing Categorical Feature: {cat_col}")

    # Group by fertilizer and category: get counts
    contingency_table = pd.crosstab(train[target], train[cat_col])
    print(f"Contingency table shape: {contingency_table.shape}")

    # Plot heatmap of frequencies (normalized by row)
    plt.figure(figsize=(12, 6))
    sns.heatmap(contingency_table.div(contingency_table.sum(axis=1), axis=0), cmap='YlGnBu', cbar_kws={'label': 'Proportion'})
    plt.title(f"Normalized Frequency Heatmap: {cat_col} vs {target}")
    plt.ylabel(target)
    plt.xlabel(cat_col)
    plt.tight_layout()
    plt.show()

    # Chi-square test for independence
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    print(f"Chi-square test p-value for {cat_col} and {target}: {p:.4e}")
    if p < 0.05:
        print("=> Significant association found.")
    else:
        print("=> No significant association.")




for num_col in numeric_cols:
    print(f"\nAnalyzing Numerical Feature: {num_col}")

    # Group by fertilizer: mean and median
    group_stats = train.groupby(target)[num_col].agg(['mean', 'median', 'std', 'count'])
    print(group_stats)

    # Boxplot to visualize distribution
    plt.figure(figsize=(14, 6))
    sns.boxplot(x=target, y=num_col, data=train)
    plt.title(f"Boxplot of {num_col} by {target}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Violin plot as alternative
    plt.figure(figsize=(14, 6))
    sns.violinplot(x=target, y=num_col, data=train)
    plt.title(f"Violin plot of {num_col} by {target}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Prepare data for ANOVA/Kruskal-Wallis
    groups = [group[num_col].dropna().values for name, group in train.groupby(target)]

    # Check assumptions (normality/homoscedasticity not shown here for brevity)
    # Use ANOVA if normality assumed, else Kruskal-Wallis (non-parametric)
    try:
        anova_stat, anova_p = f_oneway(*groups)
        print(f"ANOVA p-value for {num_col}: {anova_p:.4e}")
        if anova_p < 0.05:
            print("=> Significant difference between groups (ANOVA).")
        else:
            print("=> No significant difference (ANOVA).")
    except:
        # Fall back to Kruskal-Wallis test if ANOVA fails
        kruskal_stat, kruskal_p = kruskal(*groups)
        print(f"Kruskal-Wallis p-value for {num_col}: {kruskal_p:.4e}")
        if kruskal_p < 0.05:
            print("=> Significant difference between groups (Kruskal-Wallis).")
        else:
            print("=> No significant difference (Kruskal-Wallis).")




import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Select key numeric features for plotting (modify as needed)
key_numeric_cols = train.select_dtypes(include='number').columns.tolist()[:5]

# Sample for fast plotting
sample_df = train.sample(n=3000, random_state=42)

# Pairplot (sampled)
sns.pairplot(sample_df[key_numeric_cols + [target]], hue=target, height=2)
plt.suptitle("Pairplot of Key Numerical Features", y=1.02)
plt.show()

# Colored scatterplot: Example Temperature vs Humidity
if "Temperature" in train.columns and "Humidity" in train.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=sample_df, x="Temperature", y="Humidity", hue=target, alpha=0.6)
    plt.title("Temperature vs Humidity colored by Fertilizer")
    plt.tight_layout()
    plt.show()




from sklearn.preprocessing import LabelEncoder
import numpy as np

# Encode categorical features to numeric for correlation
corr_df = train.copy()
for col in corr_df.select_dtypes(include='object').columns:
    corr_df[col] = LabelEncoder().fit_transform(corr_df[col].astype(str))

# Compute correlation
corr_matrix = corr_df.corr(method='pearson')  # or 'spearman'

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Identify multicollinearity (r > 0.85)
threshold = 0.85
corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [(col, row, corr_pairs.loc[row, col]) 
             for col in corr_pairs.columns 
             for row in corr_pairs.index 
             if abs(corr_pairs.loc[row, col]) > threshold]

print("Highly Correlated Feature Pairs (|r| > 0.85):")
for f1, f2, r in high_corr:
    print(f"{f1} â†” {f2} : r = {r:.2f}")



import seaborn as sns
import matplotlib.pyplot as plt

# Create an interaction feature
train['Temp_Humid'] = train['Temparature'] * train['Humidity']

# Sample for speed
sample_df = train[['Temparature', 'Humidity', 'Temp_Humid', target]].sample(3000, random_state=42)

# 2D Scatterplot: Temperature vs Humidity
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=sample_df,
    x='Temparature',
    y='Humidity',
    hue=target,
    alpha=0.6,
    palette='Set2'
)
plt.title("Interaction Between Temperature and Humidity Colored by Fertilizer")
plt.xlabel("Temparature")
plt.ylabel("Humidity")
plt.legend(title="Fertilizer", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()





train['CropSoilCombo'] = train['Crop Type'] + " + " + train['Soil Type']

# Count fertilizer distribution per combination
combo_counts = train.groupby(['CropSoilCombo', target]).size().reset_index(name='Count')

# Sort by total count to focus on most common combinations
top_combos = combo_counts.groupby('CropSoilCombo')['Count'].sum().sort_values(ascending=False).head(20).index
filtered_counts = combo_counts[combo_counts['CropSoilCombo'].isin(top_combos)]

# Plot the barplot
plt.figure(figsize=(14, 6))
sns.barplot(data=filtered_counts, x='CropSoilCombo', y='Count', hue=target)
plt.title("Fertilizer Distribution by Crop Type + Soil Type Combination (Top 20)")
plt.xticks(rotation=45, ha='right')
plt.xlabel("Crop + Soil Combination")
plt.ylabel("Count")
plt.tight_layout()
plt.show()




from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Select numeric features only
numeric_data = train.select_dtypes(include='number')

# 2. Standardize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# 3. PCA to 2D
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# 4. Add to dataframe
pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
pca_df[target] = train[target].values

# 5. Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue=target, alpha=0.7)
plt.title('2D PCA: Fertilizer Class Separation')
plt.tight_layout()
plt.show()



data = {
    'Temperature': np.random.uniform(25, 38, 1000),
    'Humidity': np.random.uniform(50, 70, 1000),
    'Moisture': np.random.uniform(25, 65, 1000),
    'Nitrogen': np.random.uniform(10, 40, 1000),
    'Potassium': np.random.uniform(0, 20, 1000),
    'Phosphorous': np.random.uniform(0, 15, 1000),
    'Soil Type': np.random.choice(['Black', 'Clayey', 'Loamy', 'Red', 'Sandy'], 1000),
    'Crop Type': np.random.choice([
        'Wheat', 'Paddy', 'Cotton', 'Maize', 'Barley',
        'Ground Nuts', 'Millets', 'Oil seeds', 'Pulses',
        'Sugarcane', 'Tobacco'
    ], 1000),
    'Fertilizer Name': np.random.choice(['Urea', 'DAP', '20-20', '14-35-14'], 1000),
    'Yield': np.random.uniform(50, 150, 1000)
}
df = pd.DataFrame(data)

contingency_table = pd.crosstab(df['Soil Type'], df['Crop Type'])

normalized_table = contingency_table.div(contingency_table.sum(axis=1), axis=0)

plt.figure(figsize=(14, 9))

sns.heatmap(
    normalized_table,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=.5,
    cbar_kws={'label': 'Proportion within Soil Type'}
)

plt.title('Proportion of Crop Types within each Soil Type', fontsize=16)
plt.xlabel('Crop Type', fontsize=12)
plt.ylabel('Soil Type', fontsize=12)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()

plt.show()





# --- LightGBM Model for Fertilizer Prediction (Official Data Only) ---
import numpy as np, pandas as pd, gc, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.simplefilter(action='ignore')
print("ðŸš€ LightGBM Training with Official Dataset...")

# --- Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

# --- Fix Column Name ---
for df in [train, test]:
    df.rename(columns={'Temparature': 'Temperature'}, inplace=True)

# --- Feature Engineering ---
def create_features(df):
    epsilon = 1e-6
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + epsilon)
    df['P_K_Ratio'] = df['Phosphorous'] / (df['Potassium'] + epsilon)
    df['N_K_Ratio'] = df['Nitrogen'] / (df['Potassium'] + epsilon)
    df['Total_Nutrients'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['Temp_Humidity_Index'] = df['Temperature'] * df['Humidity']
    df['Soil_Quality_Index'] = df['Moisture'] / (df['Temperature'] + epsilon)

    numeric_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
    for col in numeric_cols:
        df[f'{col}_qcut_bin'] = pd.qcut(df[col].rank(method='first'), q=255, labels=False, duplicates='drop')
    return df

train = create_features(train)
test = create_features(test)




# Encode Categorical and Target 
categorical_cols = ['Soil Type', 'Crop Type']
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

target_encoder = LabelEncoder()
train['Fertilizer Name'] = target_encoder.fit_transform(train['Fertilizer Name'])

for col in categorical_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

# --- Prepare Data ---
X = train.drop(columns=['id', 'Fertilizer Name'])
y = train['Fertilizer Name']
X_test = test.drop(columns=['id'])



  






# --- LightGBM Parameters ---
lgb_params = {
    'objective': 'multiclass',
    'num_class': y.nunique(),
    'metric': 'multi_logloss',
    'random_state': 42,
    'n_estimators': 10000,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 7,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'device': "gpu"  # use "cpu" if you're not using GPU
}

# --- K-Fold Training ---
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

test_preds = np.zeros((len(test), y.nunique()))
oof_preds = np.zeros((len(train), y.nunique()))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"ðŸ“¦ Training Fold {fold + 1}/{FOLDS}")
    
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        categorical_feature=categorical_cols
    )

    oof_preds[valid_idx] = model.predict_proba(x_valid)
    test_preds += model.predict_proba(X_test) / FOLDS
    gc.collect()



# --- Prepare MAP@3 Submission ---
top_3_indices = np.argsort(test_preds, axis=1)[:, ::-1][:, :3]
top_3_labels = target_encoder.inverse_transform(top_3_indices.ravel()).reshape(top_3_indices.shape)

submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

submission.to_csv("submission_lgbm_only_official.csv", index=False)
print("âœ… Submission saved as 'submission_lgbm_only_official.csv'")







