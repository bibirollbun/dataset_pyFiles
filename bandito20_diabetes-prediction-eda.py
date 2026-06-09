import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
%matplotlib inline


df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv", index_col='id')


df.head()


df.info()


missing_data = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing Percentage": (df.isna().sum() / len(df)) * 100
})

missing_data = missing_data[missing_data["Missing Count"] > 0]

if missing_data.empty:
    print("No missing values found")
else:
    display(missing_data)


df.duplicated().sum()


diab_count  = df['diagnosed_diabetes'].value_counts()
diab_percent = df['diagnosed_diabetes'].value_counts(normalize=True) * 100

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.countplot(data=df, x='diagnosed_diabetes')
plt.title("Diagnosed Diabetes - Count")

plt.subplot(1, 2, 2)
plt.pie(diab_count, labels=diab_count.index, autopct='%1.1f%%', startangle=90)
plt.title("Diagnosed Diabetes - Percentage")

plt.tight_layout()
plt.show()


num_cols = df.select_dtypes(include=['number']).columns.tolist()

df[num_cols].describe()


n_cols = 3
n_rows = (len(num_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    df[col].hist(bins=40, ax=axes[i], edgecolor='black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


cat_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    value_counts = df[col].value_counts()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    value_counts.head(10).plot(kind='bar')
    plt.title(f"Top Categories - {col}")
    plt.xticks(rotation=45)

    # plt.subplot(1, 2, 2)
    # churn_by_cat = df.groupby(col)['Churn'].value_counts(normalize=True).unstack()
    # if 'Yes' in churn_by_cat.columns:
    #     churn_by_cat['Yes'].sort_values(ascending=False).head(10).plot(kind='bar')
    #     plt.title(f"Churn Rate by {col}")
    #     plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if num_cols:
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, x='diagnosed_diabetes', y=col, ax=axes[i])
        axes[i].set_title(f'{col} vs Churn')
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


key_categoricals = ['ethnicity', 'gender', 'income_level', 
                    'smoking_status', 'employment_status', 'family_history_diabetes']

if key_categoricals:
    n_cols = 2
    n_rows = (len(key_categoricals) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for i , col in enumerate(key_categoricals):
        churn_pivot = pd.crosstab(df[col], df['diagnosed_diabetes'], normalize='index') * 100
        churn_pivot.plot(kind='bar', ax=axes[i], stacked=True)
        axes[i].set_title(f'Churn Distribution by {col}')
        axes[i].set_ylabel('Percentage')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend(title='diagnosed_diabetes')

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


plt.figure(figsize=(15, 10))

correlation_matrix = df[num_cols].corr()

maks = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=maks, annot=True, cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title("Correlation Matrix - Numerical Features")
plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()

le = LabelEncoder()

for col in cat_cols:
    df_encoded[col] = le.fit_transform(df[col].astype(str))

df_encoded = df_encoded.fillna(0)

X = df_encoded.drop(['diagnosed_diabetes'], axis=1)
y = df_encoded['diagnosed_diabetes']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance, y='Feature', x='Importance')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()


print("="*50)
print("EDA KEY INSIGHTS SUMMARY")
print("="*50)

print(f"\n1. DATASET OVERVIEW:")
print(f"   - Total records: {df.shape[0]:,}")
print(f"   - Total features: {df.shape[1]}")
print(f"   - Churn rate: {churn_percent[1]:.1f}%")

print(f"\n2. DATA QUALITY:")
print(f"   - Missing values: {df.isnull().sum().sum()}")
print(f"   - Duplicate rows: {df.duplicated().sum()}")

print(f"\n3. FEATURE TYPES:")
print(f"   - Numerical features: {len(num_cols)}")
print(f"   - Categorical features: {len(cat_cols)}")





