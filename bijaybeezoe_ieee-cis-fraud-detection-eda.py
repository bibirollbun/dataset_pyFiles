import os 
os.listdir('/kaggle/input/')


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

df_train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
df_train_txn = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
df_test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
df_test_txn = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

df_sample = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')


df_train_id.head()


df_train_txn.head()


df_test_id.head()


df_test_txn.head()


print("Full Train Set")
df_train = pd.merge(df_train_txn, df_train_id, on = 'TransactionID', how = 'left' )
df_train.head(2)



print("Full Test Set")
df_test = pd.merge(df_test_txn, df_test_id, on = 'TransactionID', how = 'left')
df_test.head(2)


print("Shape of train_identity", df_train_id.shape)
print("Shape of test_identity", df_test_id.shape)
print("Shape of train_transaction", df_train_txn.shape)
print("Shape of test_transaction", df_test_txn.shape)

print("\n\nShape of train_full", df_train.shape)
print("Shape of test_full", df_test.shape)


df = df_train


pd.set_option('display.max_columns', None)      
pd.set_option('display.max_rows', None)         
pd.set_option('display.max_colwidth', None)      
pd.set_option('display.width', 1000)           
pd.set_option('display.expand_frame_repr', False)


df.info(verbose=True, show_counts=True)


df.describe(include='all').T


df.isnull().sum()


import missingno as msno

msno.matrix(df)



df_sample = df.sample(5000, random_state=42)

cols_sample = df.columns[:50]
df_subset = df[cols_sample]

msno.matrix(df_sample)
msno.heatmap(
    df_subset,  
    figsize=(12, 8),
    fontsize=6,
    cmap='coolwarm'
)


df.duplicated().sum()



target_counts = df['isFraud'].value_counts()

print(f"Class Distribution : \n {target_counts}")
print(f"Class ratio: {target_counts[1]/target_counts[0]:.4f}")
sns.countplot(x = 'isFraud', data = df)
plt.show()


num_cols = df.select_dtypes(include=['int64', 'float64']).drop('isFraud', axis=1).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print("=== Skewness & Kurtosis (Numerical Features) ===")
for col in num_cols:
    if df[col].nunique() > 1:   
        print(f"{col:<20} | Skew: {df[col].skew():.2f} | Kurtosis: {df[col].kurt():.2f}")

print("\n=== Plotting First 10 Numerical Features ===")
for col in num_cols[:10]:
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df[col].dropna(), kde=True, color='skyblue', alpha=0.7)
    plt.title(f"{col}\nSkew: {df[col].skew():.2f} | Kurt: {df[col].kurt():.2f}")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col].dropna(), color='lightcoral')
    plt.title(f"{col} - Boxplot")

    plt.tight_layout()
    plt.show()
    plt.close()  


top_n = 25   

for col in cat_cols[:15]:
    plt.figure(figsize=(14, 10))  # Reasonable size
    
    plt.subplot(3, 1, 1)
    sns.countplot(x=col, data=df, order=df[col].value_counts().index, palette='Blues_d')
    plt.title(f"{col} - Frequency")
    plt.xticks(rotation=45, ha='right')

    plt.subplot(3, 1, 2)
    sns.countplot(x=col, hue='isFraud', data=df, order=df[col].value_counts().index, palette='Set2')
    plt.title(f"{col} vs isFraud")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='isFraud')

    plt.subplot(3, 1, 3)
    target_mean = df.groupby(col)['isFraud'].mean()
    top_categories = target_mean.sort_values(ascending=False).head(top_n)
    top_categories.plot(kind='bar', color='coral', edgecolor='black')
    plt.title(f"Mean isFraud by {col} (Top {top_n})")
    plt.ylabel("Mean isFraud")
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    plt.close() 


from scipy.stats import f_oneway, ttest_ind

results = {}

for col in cat_cols:
    unique_val = df[col].dropna().unique()
    
    if len(unique_val) ==2:
        group1 = df[df[col] == unique_val[0]]['isFraud']
        group2 = df[df[col] == unique_val[1]]['isFraud']
        stats, p = ttest_ind(group1, group2)
        test_type = "ttest"
    else:
        groups = [df[df[col]==val]['isFraud'] for val in unique_val]
        stats, p = f_oneway(*groups)
        test_type = 'ANOVA'

    results[col] = {'Test': test_type,  
                   'Stats':stats,
                   'p-values':p
                   }
results_df = pd.DataFrame(results).T.sort_values('p-values')
print(results_df)



sample_df = df[num_cols].sample(n=5000, random_state=42)
corr = sample_df.corr(method='spearman')

mask = (corr > 0.7) | (corr < -0.7)
plt.figure(figsize=(12, 10))
sns.heatmap(corr[mask], cmap='coolwarm', center=0)
plt.title("High Correlation (Spearman) | Sampled 5k rows")
plt.show()


