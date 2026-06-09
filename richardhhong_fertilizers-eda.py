import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


import warnings
warnings.simplefilter("ignore")


df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


df_train.head()


df_train.info()


df_test.info()


df_train.isna().sum()


print(f"Unique Soil types in train: {sorted(df_train['Soil Type'].unique())}")
print(f"Unique Soil types in test: {sorted(df_test['Soil Type'].unique())}")


print(f"Unique Crop types in train: {sorted(df_train['Crop Type'].unique())}")
print(f"Unique Crop types in test: {sorted(df_test['Crop Type'].unique())}")


print(f"Unique Fertilizers: {df_train['Fertilizer Name'].unique()}")


df_train.describe()


df_test.describe()


numerical_features = list(df_train.select_dtypes(include=['int64']).columns)
numerical_features.remove('id')
categorical_features = list(df_test.select_dtypes(include=['object']).columns)
outcome = ['Fertilizer Name']


sns.histplot(df_train['Fertilizer Name'])
plt.title('Distribution of Fertilizer Name')
plt.show()


# distributions of numerical features in df_train
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.histplot(df_train[feature], ax=axes[i], bins=df_train[feature].nunique())
    axes[i].set_title(f'Distribution of {feature} in Train')

plt.tight_layout()
plt.show()


# distributions of numerical features in df_test
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.histplot(df_test[feature], ax=axes[i], bins=df_test[feature].nunique())
    axes[i].set_title(f'Distribution of {feature} in Test')

plt.tight_layout()
plt.show()


# Comparing distributions of numerical features in train and test
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.histplot(df_test[feature], ax=axes[i], alpha=0.3, fill=True, 
                 bins=min(df_test[feature].nunique(), 43), stat='probability', label='test')
    sns.histplot(df_train[feature], ax=axes[i], alpha=0.3, fill=True, 
                 bins=min(df_train[feature].nunique(), 43), stat='probability', label='train')
    axes[i].set_title(f'Distribution of {feature}')

plt.legend()
plt.tight_layout()
plt.show()


# Scatterplots
sns.pairplot(data=df_train, vars=numerical_features, corner=True)
plt.show()


# Correlation Matrix
corr_matrix = df_train[numerical_features].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.xticks(rotation=45)
plt.title('Correlation of Features in df_train')
plt.show()


# variance inflation factors
X = df_train[numerical_features].copy()
X_vif = add_constant(X)

vif_data = pd.DataFrame({
    'Feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i)
            for i in range(X_vif.shape[1])]
})

vif_data = vif_data[vif_data['Feature'] != 'const']

print(vif_data.sort_values(by='VIF', ascending=False).reset_index(drop=True))


# distributions of categorical features in training data
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    sns.histplot(df_train[feature], ax=axes[i])
    axes[i].set_title(f'Distribution of {feature} in Train')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# distributions of categorical features in test data
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    sns.histplot(df_test[feature], ax=axes[i])
    axes[i].set_title(f'Distribution of {feature} in Test')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# distributions of categorical features in test data
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    sns.histplot(df_test[feature], ax=axes[i], label='test', alpha=0.3, fill=True, stat='probability')
    sns.histplot(df_train[feature], ax=axes[i], label='train', alpha=0.3, fill=True, stat='probability')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].legend()

plt.tight_layout()
plt.show()


# distributions of categorical features in training data
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    counts = df_train.groupby([feature, 'Fertilizer Name']).size().rename('Count')
    total = counts.groupby(level=0).transform('sum')
    prop_df = (counts / total).reset_index(name='Proportion')

    sns.barplot(data=prop_df, x=feature, y='Proportion', hue='Fertilizer Name', ax=axes[i])
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., prop={'size': 8})

plt.tight_layout()
plt.show()


# soil type and crop type
plt.figure(figsize=(12, 8))

counts = df_train.groupby(['Soil Type', 'Crop Type']).size().rename('Count')
total = counts.groupby(level=0).transform('sum')
prop_df = (counts / total).reset_index(name='Proportion')

sns.barplot(data=prop_df, x='Crop Type', y='Proportion', hue='Soil Type')
plt.title('Ah')
# axes[i].tick_params(axis='x', rotation=45)
# axes[i].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., prop={'size': 8})

plt.tight_layout()
plt.show()


# Relationship between All Numeric and Soiltype
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.boxplot(x=df_train['Soil Type'], y=df_train[feature], ax=axes[i])
    axes[i].set_title(f'Relationship Between Soil Type and {feature}')

plt.tight_layout()
plt.show()


# Relationship between All Numeric and Crop Type
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.boxplot(x=df_train['Crop Type'], y=df_train[feature], ax=axes[i])
    axes[i].set_title(f'Relationship Between Crop Type and {feature}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Mutual information
all_columns = df_train.drop(columns=['id']).columns
mi_matrix = pd.DataFrame(index=all_columns, columns=all_columns, dtype=float)

for i in all_columns:
    for j in all_columns:
        mi_matrix.loc[i, j] = mutual_info_score(df_train[i], df_train[j])

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(mi_matrix.astype(float), annot=True, fmt=".2f", cmap="viridis")
plt.title("Mutual Information Matrix")
plt.show()


original_numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

def make_feature_set1(df):
    df_copy = df.copy() 
    df_copy['Temp_Humidity_Interaction'] = df_copy['Temparature'] * df_copy['Humidity']
    df_copy['N_P_Ratio'] = df_copy['Nitrogen'] / (df_copy['Phosphorous'].replace(0, 1))
    df_copy['K_P_Ratio'] = df_copy['Potassium'] / (df_copy['Phosphorous'].replace(0, 1))
    df_copy['Soil_Crop_Combination'] = df_copy['Soil Type'].astype(str) + '_' + df_copy['Crop Type'].astype(str)

    # Binning numerical features (as strings for categorical handling)
    for col in original_numerical_cols:
        df_copy[f'{col}_Binned'] = df_copy[col].astype(str)

    return df_copy

df_feature_set1 = make_feature_set1(df_train)


fs1 = ['Temp_Humidity_Interaction', 'N_P_Ratio', 'K_P_Ratio', 'Soil_Crop_Combination']
fs1_num = ['Temp_Humidity_Interaction', 'N_P_Ratio', 'K_P_Ratio']
fs1_cat = ['Soil_Crop_Combination']


df_feature_set1[fs1].describe()


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(fs1):
    sns.histplot(df_feature_set1[feature], ax=axes[i], bins=30)
    axes[i].set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()


df_feature_set1['Soil_Crop_Combination'].nunique()


# relationship between new features and outcome
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(fs1_num):
    sns.boxplot(df_feature_set1, x='Fertilizer Name', y=feature, ax=axes[i])
    if i != 0:
        axes[i].set_ylim(0, 6)
    axes[i].set_title(f'Relationship Between Outcome and {feature}')

axes[3].remove()
plt.tight_layout()
plt.show()


# relationship between outcome and Soil Crop Combination within Fertilizer Name
ct = pd.crosstab(df_feature_set1['Fertilizer Name'], df_feature_set1['Soil_Crop_Combination'])
ct_norm = ct.div(ct.sum(axis=1), axis=0)

plt.figure(figsize=(15, 12))
sns.heatmap(ct_norm, cmap="YlGnBu", linewidths=0.5)
plt.title("Relationship Between Fertilizer Name and Soil Crop Combination within Fertilizer Name")
plt.xlabel("Soil_Crop_Combination")
plt.ylabel("Fertilizer Name")
plt.tight_layout()
plt.show()


# relationship between outcome and Soil Crop Combination within Soil Crop Combination
ct = pd.crosstab(df_feature_set1['Fertilizer Name'], df_feature_set1['Soil_Crop_Combination'])
ct_norm = ct.div(ct.sum(axis=0), axis=1)

plt.figure(figsize=(15, 12))
sns.heatmap(ct_norm, cmap="YlGnBu", linewidths=0.5)
plt.title("Relationship Between Fertilizer Name and Soil Crop Combination within Soil Crop Comb")
plt.xlabel("Soil_Crop_Combination")
plt.ylabel("Fertilizer Name")
plt.tight_layout()
plt.show()

