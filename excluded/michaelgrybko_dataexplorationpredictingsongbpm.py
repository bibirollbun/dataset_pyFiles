import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
print("\n-------------------------- Head Training Data -----------------------------")
print(train.head())
print("\n-------------------------- Head Testing Data ------------------------------")
print(test.head())
print("\n---------------------- Head Sample Submission Data ------------------------")
print(submission.head())


# Remove 'id' Variable from Training and Testing Data
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


print("\n------------------------------- Column Names -------------------------------")
print(train.columns)
print(test.columns)
print("\n-------------------------------- Data Types --------------------------------")
print(train.dtypes)
print(test.dtypes)
print("\n-------------------------- Training Data Description --------------------------")
print(train.describe())
print("\n-------------------------- Testing Data Description ---------------------------")
print(test.describe())


# Remove 'BeatPerMinute' from train data for side-by-side plots
train_plot_df = train.drop('BeatsPerMinute', axis=1)


# Box and Whiskers plots to detect outliers
# Loop through each column
for col in train_plot_df.columns:
    # Combine train and test for each variable
    combined = pd.concat([
        pd.DataFrame({col: train_plot_df[col], 'Dataset': 'Train'}),
        pd.DataFrame({col: test[col], 'Dataset': 'Test'})
    ])

    # Side-by-side boxplots
    plt.figure(figsize=(5, 3))
    ax = sns.boxplot(
        data=combined,
        x="Dataset",
        y=col,
        palette={"Train": "cornflowerblue", "Test": "limegreen"},
        width=0.5
    )

    # Style
    plt.title(f"{col}", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold")
    plt.xlabel("")  
    plt.ylabel("")
    
    plt.tight_layout()
    plt.show()


# Histograms to inspect predictor distributions
# Loop through each column
for col in train_plot_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Histogram for training data
    sns.histplot(data=train_plot_df, x=col, bins=50, color='cornflowerblue',stat="probability",
                 edgecolor='black', alpha=0.5, kde=True, ax=axes[0])
    axes[0].set_title(f'Train: {col}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].set_ylabel('Probability')

    # Histogram for testing data
    sns.histplot(data=test, x=col, bins=50, color='limegreen', stat="probability",
                 edgecolor='black', alpha=0.5, kde=True, ax=axes[1])
    axes[1].set_title(f'Test: {col}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel("")
    axes[1].set_ylabel('Probability')

    sns.despine()
    plt.tight_layout()
    plt.show()


fig, axes = plt.subplots(2,1, figsize=(10, 6), sharex=True)

# Histogram
sns.histplot(data=train, x='BeatsPerMinute', color='royalblue',alpha=0.5,
             edgecolor='black', bins=50, kde=True, ax=axes[0])
axes[0].set_title('Distribution and Boxplot of Response Variable, BeatsPerMinute', 
                  fontweight='bold', fontsize=16)
axes[0].set_ylabel('Frequency', fontweight='bold')

# Horizontal boxplot
sns.boxplot(data=train, x='BeatsPerMinute', ax=axes[1], color='royalblue')
axes[1].axvline(train['BeatsPerMinute'].mean(), color='red', linestyle='--', label='Mean')
axes[1].legend(loc='upper right')
axes[1].set_xlabel('BeatsPerMinute', fontweight='bold', fontsize=16)

plt.tight_layout()
plt.show()


# Check for correlated predictors
corr_matrix = train.corr()

plt.figure(figsize=(12,5)),

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.tick_params(axis='x', labelrotation=35)

plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)

plt.show()


X = train.drop('BeatsPerMinute', axis=1)
y = train['BeatsPerMinute']

# add a constant for the intercept to use statsmodels
X_const = sm.add_constant(X)

# build robust model (Huber Regressor)
rlm_model = sm.RLM(y, X_const, M = sm.robust.norms.HuberT())
rlm_results = rlm_model.fit()

print(rlm_results.summary())


# scale data
std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X)


explained_variance = []
n_comp = np.arange(10)
for n in n_comp:
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    explained_variance.append(np.sum(pca.explained_variance_ratio_))

print(explained_variance)


plt.figure(figsize=(6,8))
plt.plot(n_comp, explained_variance, marker ='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio') 
plt.title('PCA Results')

