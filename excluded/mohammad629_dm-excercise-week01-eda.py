import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


import pandas as pd

# =========================
# 1. Ø®ÙˆØ§Ù†Ø¯Ù† Data Dictionary
# =========================
print("=== Carvana Data Dictionary (20 Ø®Ø· Ø§ÙˆÙ„) ===")
with open("/kaggle/input/DontGetKicked/Carvana_Data_Dictionary.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20:  # Ù�Ù‚Ø· 20 Ø®Ø· Ø§ÙˆÙ„ Ø±Ùˆ Ù†Ø´ÙˆÙ† Ø¨Ø¯Ù‡
            break
        print(line.strip())

# =========================
# 2. Ø®ÙˆØ§Ù†Ø¯Ù† Ø¯ÛŒØªØ§Ø³Øª Ø§ØµÙ„ÛŒ
# =========================
# ØªÙˆØ¬Ù‡: Ø§Ø³Ù… Ù�Ø§ÛŒÙ„ Ø¯ÛŒØªØ§Ø³Øª Ø§ØµÙ„ÛŒ Ø¯Ø± Ø§ÛŒÙ† Ù…Ø³Ø§Ø¨Ù‚Ù‡ Ù…Ø¹Ù…ÙˆÙ„Ø§ 'training.csv' Ù‡Ø³Øª
df = pd.read_csv("/kaggle/input/DontGetKicked/training.csv")

# Ù†Ù…Ø§ÛŒØ´ Ø§Ø·Ù„Ø§Ø¹Ø§Øª Ø§ÙˆÙ„ÛŒÙ‡ Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ…
print("\n=== Dataset Info ===")
df.info()

# Ù†Ù…Ø§ÛŒØ´ 5 Ø±Ø¯ÛŒÙ� Ø§ÙˆÙ„
print("\n=== Head of Dataset ===")
print(df.head())

# Ù†Ù…Ø§ÛŒØ´ Ø¢Ù…Ø§Ø± ØªÙˆØµÛŒÙ�ÛŒ Ø¨Ø±Ø§ÛŒ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ
print("\n=== Describe (numeric columns) ===")
print(df.describe().T)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# 1. Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ø¯ÛŒØªØ§Ø³Øª
df = pd.read_csv("/kaggle/input/DontGetKicked/training.csv")
df['PurchDate'] = pd.to_datetime(df['PurchDate'], errors='coerce')

# ===========================
# 2. Ø¨Ø±Ø±Ø³ÛŒ Ù…Ù‚Ø§Ø¯ÛŒØ± Ú¯Ù…Ø´Ø¯Ù‡
# ===========================
print("=== Missing Values ===")
print(df.isnull().sum())

# Ù†Ù…Ø§ÛŒØ´ Ú¯Ø±Ø§Ù�ÛŒÚ©ÛŒ missing values
msno.matrix(df)
plt.show()

# ===========================
# 3. ØªØ­Ù„ÛŒÙ„ Ø³ØªÙˆÙ† Ù‡Ø¯Ù� (IsBadBuy)
# ===========================
print("\n=== Target Distribution ===")
print(df['IsBadBuy'].value_counts(normalize=True))

sns.countplot(data=df, x='IsBadBuy')
plt.title("Target Variable Distribution")
plt.show()

# ===========================
# 4. ØªØ­Ù„ÛŒÙ„ Ù…ØªØºÛŒØ±Ù‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ
# ===========================
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
df[numeric_cols].hist(figsize=(15,10), bins=30)
plt.show()

# correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# ===========================
# 5. ØªØ­Ù„ÛŒÙ„ Ù…ØªØºÛŒØ±Ù‡Ø§ÛŒ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ
# ===========================
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts().head(10))  # top 10 values
    sns.countplot(y=col, data=df, order=df[col].value_counts().iloc[:10].index)
    plt.show()



# Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ø¯ÛŒØªØ§Ø³Øª
df = pd.read_csv("/kaggle/input/DontGetKicked/training.csv")

# Ø§Ù†ØªØ®Ø§Ø¨ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Ù†Ù…Ø§ÛŒØ´ Ù…Ù‚Ø§Ø¯ÛŒØ± Ø­Ø¯Ø§Ù‚Ù„ Ùˆ Ø­Ø¯Ø§Ú©Ø«Ø± Ø¨Ø±Ø§ÛŒ Ù‡Ø± Ø³ØªÙˆÙ†
min_max_values = df[numeric_columns].agg(['min', 'max'])
print(min_max_values)



# ØªØ­Ù„ÛŒÙ„ categorical fields Ùˆ Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ø¯Ø³ØªÙ‡â€ŒÙ‡Ø§ÛŒ Ù†Ø§Ù…ØªØ¹Ø§Ø±Ù� ÛŒØ§ Ù†Ø§Ø¯Ø±
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f"\n=== Column: {col} ===")
    print("Unique values:", df[col].nunique())
    print("Top 10 most frequent values:")
    print(df[col].value_counts().head(10))
    
    # Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ø¯Ø³ØªÙ‡â€ŒÙ‡Ø§ÛŒ Ù†Ø§Ø¯Ø± (<1%)
    counts = df[col].value_counts(normalize=True)
    rare_categories = counts[counts < 0.01].index.tolist()
    if rare_categories:
        print(f"Rare categories (<1% of data): {rare_categories}")



import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(8,4))
    
    # Boxplot Ø¨Ø§ Ù…Ø´Ø®Øµ Ú©Ø±Ø¯Ù† outlierÙ‡Ø§
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()
    
    plt.figure(figsize=(8,4))
    # Histogram
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f"Histogram of {col}")
    plt.show()
    
    # Ù…Ø­Ø§Ø³Ø¨Ù‡ IQR Ø¨Ø±Ø§ÛŒ Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ù…Ù‚Ø§Ø¯ÛŒØ± Ù¾Ø±Øª
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
    print(f"{col}: Number of potential outliers = {len(outliers)}")



import pandas as pd

# Ù�Ø±Ø¶ Ø¨Ø± Ø§ÛŒÙ† Ø§Ø³Øª Ú©Ù‡ Ø¯ÛŒØªØ§Ø³Øª Ø´Ù…Ø§ df Ù†Ø§Ù… Ø¯Ø§Ø±Ø¯
# Ù†Ù…Ø§ÛŒØ´ ØªØ¹Ø¯Ø§Ø¯ Ù…Ù‚Ø§Ø¯ÛŒØ± Ú¯Ù…Ø´Ø¯Ù‡ Ø¯Ø± Ù‡Ø± Ø³ØªÙˆÙ†
missing_values = df.isnull().sum()

# Ø¬Ø¯Ø§ Ú©Ø±Ø¯Ù† Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ Ùˆ Ø±Ø¯Ù‡â€ŒØ§ÛŒ
categorical_missing = missing_values[df.select_dtypes(include='object').columns]
continuous_missing = missing_values[df.select_dtypes(include=['int64','float64']).columns]

print("Missing values in categorical columns:\n", categorical_missing)
print("\nMissing values in continuous columns:\n", continuous_missing)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot

# Ù�Ø±Ø¶ Ú©Ù†ÛŒÙ… Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ø´Ù…Ø§ df Ù†Ø§Ù… Ø¯Ø§Ø±Ø¯
continuous_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in continuous_cols:
    print(f"--- Normality test for {col} ---")
    
    # Shapiro-Wilk test
    stat, p = shapiro(df[col])
    print(f"Shapiro-Wilk Test: Statistics={stat:.3f}, p={p:.3f}")
    if p > 0.05:
        print("Probably Gaussian (normal)")
    else:
        print("Probably not Gaussian (not normal)")
    
    # Ù†Ù…ÙˆØ¯Ø§Ø± Ù‡ÛŒØ³ØªÙˆÚ¯Ø±Ø§Ù…
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    
    # Q-Q plot
    plt.subplot(1,2,2)
    probplot(df[col], dist="norm", plot=plt)
    plt.title(f'Q-Q plot of {col}')
    
    plt.tight_layout()
    plt.show()



import pandas as pd

# Ø§Ù†ØªØ®Ø§Ø¨ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Ø¨Ø±Ø±Ø³ÛŒ ØªÙˆØ²ÛŒØ¹ Ù‡Ø± Ø³ØªÙˆÙ† Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ
for col in categorical_cols:
    print(f"\n--- Distribution of {col} ---")
    counts = df[col].value_counts(normalize=True) * 100
    print(counts)
    rare = counts[counts < 1]
    if not rare.empty:
        print("\nâš ï¸� Rare categories (<1%):")
        print(rare)

# Ø¨Ø±Ø±Ø³ÛŒ Ø¹Ø¯Ù… ØªÙˆØ§Ø²Ù† Ø¯Ø± Ø³ØªÙˆÙ† Ù‡Ø¯Ù� (Ù…Ø«Ù„Ø§Ù‹ IsBadBuy)
if 'IsBadBuy' in df.columns:
    print("\n--- Target Distribution (IsBadBuy) ---")
    print(df['IsBadBuy'].value_counts(normalize=True) * 100)



import pandas as pd

# Ø§Ù†ØªØ®Ø§Ø¨ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Ø¢Ø³ØªØ§Ù†Ù‡ Ø¨Ø±Ø§ÛŒ Ø¯Ø³ØªÙ‡â€ŒÙ‡Ø§ÛŒ Ù†Ø§Ø¯Ø± (Û±Ùª)
threshold = 0.01  

# Ù¾ÛŒÙ…Ø§ÛŒØ´ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ
for col in categorical_cols:
    freq = df[col].value_counts(normalize=True)  # Ø¯Ø±ØµØ¯ Ù�Ø±Ø§ÙˆØ§Ù†ÛŒ Ù‡Ø± Ø¯Ø³ØªÙ‡
    rare_categories = freq[freq < threshold].index  # Ø¯Ø³ØªÙ‡â€ŒÙ‡Ø§ÛŒÛŒ Ú©Ù‡ Ú©Ù…ØªØ± Ø§Ø² Ø¢Ø³ØªØ§Ù†Ù‡ Ù‡Ø³ØªÙ†Ø¯
    
    if len(rare_categories) > 0:
        print(f"\nØ³ØªÙˆÙ†: {col}")
        print("Ø¯Ø³ØªÙ‡â€ŒÙ‡Ø§ÛŒ Ù†Ø§Ø¯Ø±:", list(rare_categories))
    
    # Ø¬Ø§ÛŒÚ¯Ø²ÛŒÙ†ÛŒ Ø¯Ø³ØªÙ‡â€ŒÙ‡Ø§ÛŒ Ù†Ø§Ø¯Ø± Ø¨Ø§ 'Other'
    df[col] = df[col].replace(rare_categories, "Other")

print("\nâœ… Ø¯Ø³ØªÙ‡â€ŒÙ‡Ø§ÛŒ Ù†Ø§Ø¯Ø± Ø¯Ø± Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ Ø¨Ø§ 'Other' ØªØ¬Ù…ÛŒØ¹ Ø´Ø¯Ù†Ø¯.")



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ù¾Ø§Ø±Ø§Ù…ØªØ±: Ú†Ù†Ø¯ Ø¨Ø±Ù†Ø¯ Ù¾Ø±ØªÛŒØ±Ø§Ú˜ Ù†Ø´Ø§Ù† Ø¯Ø§Ø¯Ù‡ Ø´ÙˆØ¯ (Ø¨Ø±Ø§ÛŒ Ø¬Ù„ÙˆÚ¯ÛŒØ±ÛŒ Ø§Ø² Ø´Ù„ÙˆØºÛŒ Ù†Ù…ÙˆØ¯Ø§Ø±)
top_n = 15

# 1) Boxplot Ø¨Ø±Ø§ÛŒ ÛŒÚ© Ù…ØªØºÛŒØ± Ø¹Ø¯Ø¯ÛŒ Ø¨Ø± Ø§Ø³Ø§Ø³ Ú©Ù„Ø§Ø³ Ù‡Ø¯Ù�
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.boxplot(x='IsBadBuy', y='VehOdo', data=df, showfliers=False)  # showfliers=False Ø¨Ø±Ø§ÛŒ Ø®ÙˆØ§Ù†Ø§ÛŒÛŒ Ø¨Ù‡ØªØ±
plt.title('VehOdo distribution by IsBadBuy')
plt.xlabel('IsBadBuy (0 = good, 1 = bad)')
plt.ylabel('VehOdo')

# 2) Ù…Ø­Ø§Ø³Ø¨Ù‡Ù” Ù†Ø³Ø¨Øªâ€ŒÙ‡Ø§ Ø¨Ø±Ø§ÛŒ Make (Ø¨Ø¯ÙˆÙ† Ø§Ø³ØªÙ�Ø§Ø¯Ù‡ Ø§Ø² crosstab/unstack)
# Ø§Ù†ØªØ®Ø§Ø¨ top N Ø¨Ø±Ù†Ø¯ Ø¨Ø± Ø§Ø³Ø§Ø³ Ù�Ø±Ú©Ø§Ù†Ø³
top_makes = df['Make'].value_counts().nlargest(top_n).index
df_top = df[df['Make'].isin(top_makes)].copy()

# ØªØ¹Ø¯Ø§Ø¯ Ú©Ù„ Ø¨Ø±Ø§ÛŒ Ù‡Ø± Ø¨Ø±Ù†Ø¯ (Ø¯Ø± Ù…Ø¬Ù…ÙˆØ¹Ù‡ Ø§Ù†ØªØ®Ø§Ø¨ Ø´Ø¯Ù‡)
total_per_make = df_top.groupby('Make').size()

# ØªØ¹Ø¯Ø§Ø¯ Ø¨Ø¯ (IsBadBuy == 1) Ùˆ Ø®ÙˆØ¨ (IsBadBuy == 0) Ø¨Ø±Ø§ÛŒ Ù‡Ø± Ø¨Ø±Ù†Ø¯
bad_counts = df_top[df_top['IsBadBuy'] == 1].groupby('Make').size()
good_counts = df_top[df_top['IsBadBuy'] == 0].groupby('Make').size()

# Ù…Ø­Ø§Ø³Ø¨Ù‡Ù” Ù†Ø³Ø¨Øªâ€ŒÙ‡Ø§ (proportions) Ùˆ Ù¾Ø± Ú©Ø±Ø¯Ù† NaN Ø¨Ø§ 0
prop_bad = bad_counts.div(total_per_make).fillna(0)
prop_good = good_counts.div(total_per_make).fillna(0)

# Ø³Ø§Ø®Øª DataFrame Ù…Ø±ØªØ¨ Ø·Ø¨Ù‚ ØªØ±ØªÛŒØ¨ top_makes
make_props = pd.DataFrame({
    'Good (0)': prop_good,
    'Bad (1)': prop_bad
}).loc[top_makes]

# Ø±Ø³Ù… stacked bar
plt.subplot(1,2,2)
make_props.plot(kind='bar', stacked=True, figsize=(12,5), colormap='tab20', ax=plt.gca())
plt.title(f'Make distribution by IsBadBuy (proportion) â€” top {top_n} makes')
plt.ylabel('Proportion within Make')
plt.xlabel('Make')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

plt.tight_layout()
plt.show()

# Ú†Ø§Ù¾ Ø¬Ø¯ÙˆÙ„ Ù†Ø³Ø¨ØªÙ‡Ø§ Ø¨Ù‡ ØµÙˆØ±Øª Ø¹Ø¯Ø¯ÛŒ Ø¨Ø±Ø§ÛŒ Ø¨Ø§Ø²Ø¨ÛŒÙ†ÛŒ Ø³Ø±ÛŒØ¹
print("\nProportions of Bad (1) and Good (0) for top makes:")
print(make_props.round(3))



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ù�Ø±Ø¶: df Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ø´Ù…Ø§Ø³Øª Ùˆ Ø³ØªÙˆÙ† Ù‡Ø¯Ù� "IsBadBuy" Ù‡Ø³Øª

plt.figure(figsize=(14, 6))

# 1ï¸�âƒ£ Ù…ØªØºÛŒØ± Ø¹Ø¯Ø¯ÛŒ (Ù…Ø«Ù„Ø§Ù‹ VehOdo)
plt.subplot(1, 2, 1)
sns.histplot(data=df, x="VehOdo", hue="IsBadBuy", bins=30, kde=False, multiple="stack")
plt.title("Distribution of VehOdo by IsBadBuy")
plt.xlabel("VehOdo")
plt.ylabel("Count")

# 2ï¸�âƒ£ Ù…ØªØºÛŒØ± Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ (Ù…Ø«Ù„Ø§Ù‹ Make)
plt.subplot(1, 2, 2)
make_counts = pd.crosstab(df['Make'], df['IsBadBuy'])   # Ø¬Ø¯ÙˆÙ„ Ø´Ù…Ø§Ø±Ø´ÛŒ
make_props = make_counts.div(make_counts.sum(axis=1), axis=0)  # Ù†Ø³Ø¨Øª Ø¯Ø±ØµØ¯ÛŒ
make_props.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="coolwarm")
plt.title("Make distribution by IsBadBuy")
plt.xlabel("Make")
plt.ylabel("Proportion")
plt.legend(title="IsBadBuy")

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Ù�Ù‚Ø· Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ Ø±Ùˆ Ø§Ù†ØªØ®Ø§Ø¨ Ù…ÛŒâ€ŒÚ©Ù†ÛŒÙ…
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Ù…Ø­Ø§Ø³Ø¨Ù‡ Ù…Ø§ØªØ±ÛŒØ³ Ù‡Ù…Ø¨Ø³ØªÚ¯ÛŒ
corr_matrix = numeric_df.corr()

# Ø±Ø³Ù… heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Continuous Variables")
plt.show()





