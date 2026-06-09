import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style='whitegrid', palette="muted", font_scale=1.1)

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train.head()


train.info()


numerical_features = []
categorical_features = []

for feature in train.columns:
    if train[feature].dtype == 'object' or feature in ['family_history_diabetes','hypertension_history','cardiovascular_history']:
        categorical_features.append(feature)
    else:
        numerical_features.append(feature)


numerical_features


numerical_features.remove('id')
numerical_features.remove('diagnosed_diabetes')


categorical_features


train['diagnosed_diabetes'].value_counts()


fig, axes = plt.subplots(1,2,figsize=(14,6))
sns.countplot(x='diagnosed_diabetes', data = train, ax=axes[0])
axes[0].set_title('Distribution of Target Variable', fontweight='bold', size=20)
axes[0].set_xticks(ticks=[0, 1],labels=['No', 'Yes'])
axes[0].set_xlabel("diagnosed_diabetes") 

train['diagnosed_diabetes'].value_counts(sort=False).plot(kind='pie', ax=axes[1], explode=(0.0, 0.1), autopct="%.2f%%", labels=['Yes', 'No'], pctdistance=0.75)
axes[1].add_artist(plt.Circle((0, 0), 0.5, fc='w'))
axes[1].set_title('Pie Chart of Target Variable', fontweight='bold', size=20)
axes[1].set_ylabel("")


train[numerical_features].hist(bins=100, figsize=(15,15), layout=(7,3))
plt.suptitle("Feature Distributions") # sup => super => main title
plt.subplots_adjust(hspace=1.0, wspace=0.3) # plt.tight_layout(rect=[0, 0, 1, 0.96])  also available
plt.show()


fig, axes = plt.subplots(len(numerical_features), 2, figsize=(20, 30))
axes = axes.flatten()

for idx, feature in enumerate(numerical_features):
    axes[idx].hist(train[train['diagnosed_diabetes'] == 0][feature].dropna(), 
                   bins=50, 
                   alpha=0.6, 
                   label='Response = 0',
                   color='blue',
                   edgecolor='black')
    
    axes[idx].hist(train[train['diagnosed_diabetes'] == 1][feature].dropna(), 
                   bins=50, 
                   alpha=0.6, 
                   label='Response = 1',
                   color='red',
                   edgecolor='black')
    
    axes[idx].set_title(feature)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

# Hide unused subplots
for idx in range(len(numerical_features), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Feature Distributions by Response Variable", fontsize=16)
plt.subplots_adjust(hspace=1.0, wspace=0.3)
plt.show()


# Create a correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = train[numerical_features].corr()
mask = np.triu(correlation_matrix) # masking the upper triangle for clarity
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            mask=mask, vmin=-1, vmax=1, annot_kws={"size": 8})
plt.title('Correlation Matrix of Features', fontsize=16)
plt.show()


fig, axes = plt.subplots(len(categorical_features), 1, figsize=(20, 40))
axes = axes.flatten()

for idx, cat in enumerate(categorical_features):
    # Get normalized proportions
    response_0 = train[train['diagnosed_diabetes'] == 0][cat].value_counts(normalize=True).sort_index() # proportion is better for comparison
    response_1 = train[train['diagnosed_diabetes'] == 1][cat].value_counts(normalize=True).sort_index()
    
    combined = pd.DataFrame({
        'Response = 0': response_0,
        'Response = 1': response_1
    }).fillna(0)
    
    combined.plot(kind='bar', ax=axes[idx], color=['blue', 'red'], alpha=0.7, legend=True)
    # axes[idx].set_title(cat)
    axes[idx].set_ylabel('Proportion')
    axes[idx].tick_params(axis='x', rotation=45)

plt.subplots_adjust(hspace=1.0, wspace=0.3)


import lightgbm as lgb


RANDOM_SEED = 42

def get_feature_importance(X, y):
    """Get feature importance from a LightGBM model"""

    # Convert categorical columns to 'category' dtype
    X_processed = X.copy()
    for col in X_processed.select_dtypes(include=['object']).columns: # type handling for light GBM
        X_processed[col] = X_processed[col].astype('category')
    
    model = lgb.LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)
    model.fit(X_processed, y)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df

# Get feature importance
X = train[numerical_features + categorical_features]
y = train['diagnosed_diabetes']

importance_df = get_feature_importance(X, y)

# Visualize feature importance
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importance from LightGBM', fontsize=16)
plt.tight_layout()
plt.show()

print("Top 10 most important features:")
print(importance_df.head(10))




