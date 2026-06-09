# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer
from scipy.stats import ttest_ind, mannwhitneyu, pointbiserialr, chi2_contingency, zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report    


# Customizations
sns.set_theme(style="darkgrid")  # Setting the style for seaborn

# Color palette
custom_palette = {
    'Extrovert': '#ff7f0e',  # orange for extroverts
    'Introvert': '#1f77b4',  # blue for introverts
    'Ambivert': '#2ca02c',  # green for ambiverts just in caseğŸ˜€
}


# Loading the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')  


train


test.sample(10)


sample_submission.head()


print(f"Training set has {train.shape[0]} rows and {train.shape[1]} columns.")


# Convert column names to lowercase
train.columns = train.columns.str.lower()
test.columns = test.columns.str.lower()


train.info()


test.info()


# 1. Check for duplictes
duplicates = train.duplicated().sum()
print(f"Number of duplicate rows in the training set: {duplicates}")


# 2. Missing values
missing_values = train.isnull().sum()
missing_values = missing_values[missing_values > 0].reset_index()
missing_values.columns = ['feature', 'total_missing']
missing_values["%_missing"] = round((missing_values['total_missing'] / train.shape[0]) * 100, 1)
missing_values = missing_values.sort_values(by='%_missing', ascending=False)

print("Missing values in training set:")
missing_values



# Visual heatmap
msno.matrix(train, figsize=(10, 5), color=(0.2, 0.4, 0.6), fontsize=8, label_rotation=90)
plt.title('Missing Values Matrix')
plt.show()


train.describe().round(2)


# Summary stats for categorical features
train.describe(include=['object']).T


# 1. Impute missing numerical values with median using SimpleImputer
numerical_features = train.select_dtypes(include=['float64', 'int64']).columns
# Exlude 'id' column from numerical features
numerical_features = numerical_features[numerical_features != 'id']

median_imputer = SimpleImputer(strategy='median')
train[numerical_features] = median_imputer.fit_transform(train[numerical_features])

# 2. Impute missing categorical values with mode using SimpleImputer
categorical_features = train.select_dtypes(include=['object']).columns
categorical_features = categorical_features[categorical_features != 'personality']

mode_imputer = SimpleImputer(strategy='most_frequent')
train[categorical_features] = mode_imputer.fit_transform(train[categorical_features])



# Transform test with imputer
test[numerical_features] = median_imputer.transform(test[numerical_features])

test[categorical_features] = mode_imputer.transform(test[categorical_features])


# Check if there are any missing values left
missing_values_after = train.isnull().sum()
missing_values_after


train.info()


train.describe().round(2)


train[categorical_features].describe()


# Visualizing the distribution of numerical features - Histograms and Boxplots
n = len(numerical_features) 
fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(14, n * 5), sharex=False, sharey=False)

for i, feature in enumerate(numerical_features):
    sns.histplot(train[feature], kde=True, ax=axes[i, 0], color='blue')
    axes[i, 0].set_title(f'Histogram of {feature}')
    axes[i, 0].set_xlabel('')
    axes[i, 0].set_ylabel('Frequency')

    sns.boxplot(x=train[feature], ax=axes[i, 1], color='lightgreen')
    axes[i, 1].set_title(f'Boxplot of {feature}')
    axes[i, 1].set_xlabel('')


# Visualize the categorical features
categorical_features = train.select_dtypes(include='object').columns
for feature in categorical_features:
    plt.figure(figsize=(4, 4))
    sns.countplot(data=train, x=feature, palette='Blues_r', hue=feature)
    plt.title(f'Distribution of {feature}')
    plt.show()  


# Numeric Vs. Target Variable
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train['personality'], y=train[feature], palette=custom_palette, hue=train['personality'])
    plt.title(f'Boxplot of {feature} by Personality')
    plt.xlabel('Personality')
    plt.ylabel(feature)
    plt.show()


# Statistical Tests
extroverts = train[train['personality'] == 'Extrovert']
introverts = train[train['personality'] == 'Introvert']

print("Statistical Test Results:\n")

for feature in numerical_features:
    extrovert_vals = extroverts[feature]
    introvert_vals = introverts[feature]
    
    # T-test (for means)
    t_stat, t_pval = ttest_ind(extrovert_vals, introvert_vals, equal_var=False)
    
    # Mann-Whitney U test (for medians)
    u_stat, u_pval = mannwhitneyu(extrovert_vals, introvert_vals, alternative='two-sided')
    
    print(f"Feature: {feature}")
    print(f"  T-test (means):      t-stat = {t_stat:.3f}, p-value = {t_pval:.4f}")
    if t_pval < 0.05:
        print(f"  Significant difference in mean values of {feature} between Extroverts and Introverts.")
    else:
        print(f"  No significant difference in mean values of {feature} between Extroverts and Introverts.")

    print(f"  Mann-Whitney (meds): u-stat = {u_stat:.3f}, p-value = {u_pval:.4f}")
    if u_pval < 0.05:
        print(f"  Significant difference in median values of {feature} between Extroverts and Introverts.")
    else:
        print(f"  No significant difference in median values of {feature} between Extroverts and Introverts.")
    print("-" * 50)



train[numerical_features].corr().round(2)


# Correlation matrix for numerical features
plt.figure(figsize=(10, 6))
sns.heatmap(train[numerical_features].corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Numerical Features')
plt.show()


le_personality = LabelEncoder()
train['personality_encoded'] = le_personality.fit_transform(train['personality'])


train['personality_encoded'].value_counts()


# Point Biserial Correlation for numerical features vs. personality
for feature in numerical_features:
    corr, p_value = pointbiserialr(train['personality_encoded'], train[feature])
    print(f"Feature: {feature}")
    print(f'  Point Biserial Correlation: {corr:.3f}, p-value: {p_value:.4f}')
    if p_value < 0.05:
        print(f"  Significant correlation between {feature} and personality.")
    else:
        print(f"  No significant correlation between {feature} and personality.")
    print("-" * 50)


# Cross-tabulation of categorical features with personality
for feature in categorical_features:
    if feature != 'personality':
        crosstab = pd.crosstab(train[feature], train['personality'])
        print(f"\nCrosstab for {feature} and personality:")
        print(crosstab)
        
        # Chi-square test
        chi2, p, _, _ = chi2_contingency(crosstab)
        print(f"Chi-square test: chi2 = {chi2:.3f}, p-value = {p:.4f}")
        
        if p < 0.05:
            print(f"  Significant association between {feature} and personality.")
        else:
            print(f"  No significant association between {feature} and personality.")
        print("-" * 50)


# Z-score based outlier detection
outlier_summary_3sigma = {}
outlier_summary_2sigma = {}

for col in numerical_features:
    z = np.abs(zscore(train[col].dropna()))
    outlier_summary_3sigma[col] = (z > 3).sum()   # 3Ïƒ rule
    outlier_summary_2sigma[col] = (z > 2).sum()   # 2Ïƒ rule

# Convert to DataFrames
outliers_3sigma_df = pd.Series(outlier_summary_3sigma, name="Outliers (>3Ïƒ)").sort_values(ascending=False).to_frame()
outliers_2sigma_df = pd.Series(outlier_summary_2sigma, name="Outliers (>2Ïƒ)").sort_values(ascending=False).to_frame()

# Display bar-style summaries
print("Outlier Summary (Z > 3):")
display(outliers_3sigma_df.style.bar(color="#bb4212", align='zero'))

print("Outlier Summary (Z > 2):")
display(outliers_2sigma_df.style.bar(color="#0868a4", align='zero'))


train.info()


# Encode categorical variables
le_dict = {}
for col in ['stage_fear', 'drained_after_socializing']:
    le = LabelEncoder()
    train[col + '_enc'] = le.fit_transform(train[col])
    le_dict[col] = le  # Save encoder for use on test set


for col in ['stage_fear', 'drained_after_socializing']:
    test[col + '_enc'] = le_dict[col].transform(test[col])


train.head()


test.head()


test = test.drop(columns=['stage_fear', 'drained_after_socializing'])


pca_df = train.copy()
X_pca = pca_df.drop(columns=['id', 'personality', 'personality_encoded', 
                             'stage_fear', 'drained_after_socializing'])
y_pca = pca_df['personality_encoded']

# Standardize the features
scaler = StandardScaler()
X_pca_scaled = scaler.fit_transform(X_pca)

# PCA
pca = PCA(n_components=2)  
X_pca_transformed = pca.fit_transform(X_pca_scaled)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=X_pca_transformed, columns=['PC1', 'PC2'])
pca_df['personality'] = pca_df.index.map(lambda x: y_pca[x])
pca_df['personality_label'] = pca_df['personality'].map({0: 'Extrovert', 1: 'Introvert'})

pca_df


# Summary stats of pca
pca_stats = pca_df.groupby('personality_label').agg(
    mean_pc1 = ('PC1', 'mean'),
    std_pc1 = ('PC1', 'std'),
    mean_pc2 = ('PC2', 'mean'),
    std_pc2 = ('PC2', 'std')
)

pca_stats.round(2)


# Visualize PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', 
                hue='personality_label', palette=custom_palette, 
                alpha=0.7)
plt.title('PCA of Personality Traits')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Personality', loc='upper right')
plt.show()


train.info()


# Features and labels
X = train.drop(columns=['id', 'personality', 'personality_encoded', 
                             'stage_fear', 'drained_after_socializing'])
y = train['personality_encoded']


# 1. Store test ids before dropping the column
test_ids = test['id'].copy()

# 2. Drop 'id' so it doesn't interfere with prediction
test = test.drop('id', axis=1)


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Standardize the features using MinMaxScaler
minmax_scaler = MinMaxScaler()

X_train_mm_scaled = minmax_scaler.fit_transform(X_train)
X_val_mm_scaled = minmax_scaler.transform(X_val)

# Check shape
print(f"Shape of X_train_mm_scaled: {X_train_mm_scaled.shape}") 
print(f"Shape of X_val_mm_scaled: {X_val_mm_scaled.shape}")


# Transform test
X_test_mm_scaled = minmax_scaler.transform(test)


X_train_mm_scaled[1]


# Run LazyClassifier
#clf_mm = LazyClassifier(verbose=0, ignore_warnings=True, random_state=42, custom_metric=None)


#models_mm, predictions_mm = clf_mm.fit(X_train_mm_scaled, X_val_mm_scaled, y_train, y_val)


#models_mm_sorted = models_mm.sort_values(by='Accuracy', ascending=False)
#models_mm_sorted.head(10)


# Final logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_mm_scaled, y_train)


y_pred = log_reg.predict(X_test_mm_scaled)
y_pred_labels = le_personality.inverse_transform(y_pred)


submission = pd.DataFrame({
    'id': test_ids,
    'Personality': y_pred_labels
})


submission.head()


submission.to_csv('submission.csv', index=False)

