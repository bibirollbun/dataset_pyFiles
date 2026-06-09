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


import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


# ØªØ¹Ø¯Ø§Ø¯ Ø±Ø¯ÛŒÙ� Ùˆ Ø³ØªÙˆÙ†
print(df.shape)
# Ù†Ù…Ø§ÛŒØ´ Ú†Ù†Ø¯ Ø±Ø¯ÛŒÙ� Ø§ÙˆÙ„
df.head()
# Ø§Ø·Ù„Ø§Ø¹Ø§Øª Ú©Ù„ÛŒ
df.info()



pip install ydata-profiling


import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.info()



pip install ydata_profiling


from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Don't Get Kicked Data EDA", explorative=True)
profile.to_file("your-dataset-profile-report.html")





df.info()


df.head()


# Ø§Ù†ØªØ®Ø§Ø¨ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ
num_cols = df.select_dtypes(include='number').columns

# Ø­Ø¯Ø§Ù‚Ù„ Ùˆ Ø­Ø¯Ø§Ú©Ø«Ø± Ø¨Ø±Ø§ÛŒ Ø¨Ø±Ø±Ø³ÛŒ Ù…Ù‚Ø§Ø¯ÛŒØ± ØºÛŒØ±Ø¹Ø§Ø¯ÛŒ
df[num_cols].describe().T



# Ø§Ù†ØªØ®Ø§Ø¨ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ categorical
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Ø¨Ø±Ø±Ø³ÛŒ ØªØ¹Ø¯Ø§Ø¯ Ø¯Ø³ØªÙ‡â€ŒÙ‡Ø§ Ùˆ Ù…Ù‚Ø§Ø¯ÛŒØ± ØºÛŒØ±Ù…Ù†ØªØ¸Ø±Ù‡
for col in cat_cols:
    print(col, df[col].value_counts(normalize=True))  # Ø¯Ø±ØµØ¯ Ù‡Ø± Ø¯Ø³ØªÙ‡



cat_cols = df.select_dtypes(include=['object', 'category']).columns
print(cat_cols)



# ØªØ¹Ø¯Ø§Ø¯ Ùˆ Ø¯Ø±ØµØ¯ missing Ø¯Ø± Ù‡Ø± Ø³ØªÙˆÙ†
missing = df.isnull().sum()
missing_percent = 100 * df.isnull().sum() / len(df)
pd.DataFrame({"missing_count": missing, "missing_percent": missing_percent})



import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

for col in num_cols:  
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()

    # Ø¢Ø²Ù…ÙˆÙ† Ø´Ø§Ù¾ÛŒØ±Ùˆ
    stat, p = shapiro(df[col].dropna())
    print(col, 'p-value:', p)



import pandas as pd
from scipy.stats import shapiro

# Ù�Ø±Ø¶ Ú©Ù† df Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ…Ù‡ Ùˆ num_cols Ù„ÛŒØ³Øª Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒÙ‡
results = []

for col in num_cols:
    data = df[col].dropna()
    # Ø´Ø±Ø· Ø¨Ø±Ø§ÛŒ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒÛŒ Ú©Ù‡ Ø­Ø¯Ø§Ù‚Ù„ 3 Ø¯Ø§Ø¯Ù‡ Ø¯Ø§Ø±Ù† (Ø´Ø§Ù¾ÛŒØ±Ùˆ Ø®Ø·Ø§ Ù…ÛŒØ¯Ù‡ Ø§Ú¯Ø± Ú©Ù…ØªØ± Ø§Ø² 3 Ø¨Ø§Ø´Ù‡)
    if len(data) >= 3:
        stat, p = shapiro(data)
        normality = "âœ… Ù†Ø±Ù…Ø§Ù„" if p >= 0.05 else "â�Œ ØºÛŒØ±Ù†Ø±Ù…Ø§Ù„"
        results.append([col, p, normality])
    else:
        results.append([col, None, "âš ï¸� Ø¯Ø§Ø¯Ù‡ Ú©Ø§Ù�ÛŒ Ù†Ø¯Ø§Ø±Ø¯"])

# Ø³Ø§Ø®Øª Ø¬Ø¯ÙˆÙ„ Ù†ØªØ§ÛŒØ¬
normality_df = pd.DataFrame(results, columns=["Ø³ØªÙˆÙ†", "p-value", "Ù†ØªÛŒØ¬Ù‡"])
display(normality_df)



cat_cols = df.select_dtypes(include=['object', 'category']).columns
cat_cols



for col in cat_cols:
    print(f"\nğŸ“Š Column: {col}")
    value_counts = df[col].value_counts(dropna=False)
    percent = 100 * value_counts / len(df)
    rare = percent[percent < 1]

    print("Top categories:")
    print(value_counts.head())

    if not rare.empty:
        print("\nâš ï¸� Rare categories (<1%):")
        print(rare)



df['IsBadBuy'].value_counts(normalize=True) * 100



from ydata_profiling import ProfileReport

# Ø¬Ø¯Ø§ Ú©Ø±Ø¯Ù† Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ø±Ø§Ø³Ø§Ø³ Target
df_IsBadBuy_0 = df[df.IsBadBuy == 0]
df_IsBadBuy_1 = df[df.IsBadBuy == 1]

# Ø§ÛŒØ¬Ø§Ø¯ Ù¾Ø±ÙˆÙ�Ø§ÛŒÙ„
profile_0 = ProfileReport(df_IsBadBuy_0, title="Don't Get Kicked! EDA 0", minimal=True)
profile_1 = ProfileReport(df_IsBadBuy_1, title="Don't Get Kicked! EDA 1", minimal=True)

# Ù…Ù‚Ø§ÛŒØ³Ù‡ Ø¯Ùˆ Ù¾Ø±ÙˆÙ�Ø§ÛŒÙ„
comparison_report = profile_0.compare(profile_1)

# Ø®Ø±ÙˆØ¬ÛŒ Ø¨Ù‡ Ù�Ø§ÛŒÙ„ HTML
comparison_report.to_file("comparison.html")
from IPython.display import IFrame

IFrame("comparison.html", width=1000, height=600)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ù�Ø±Ø¶ Ú©Ù† num_cols Ù„ÛŒØ³Øª Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø¹Ø¯Ø¯ÛŒ Ù‡Ø³Øª
num_cols = df.select_dtypes(include=['int64','float64']).columns

# Ù…Ø­Ø§Ø³Ø¨Ù‡ Ù…Ø§ØªØ±ÛŒØ³ Ù‡Ù…Ø¨Ø³ØªÚ¯ÛŒ
corr_matrix = df[num_cols].corr()

# Ù†Ù…Ø§ÛŒØ´ Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap of Continuous Fields")
plt.show()



import pandas as pd
import numpy as np

# =========================
# Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ø¯ÛŒØªØ§Ø³Øª
# =========================
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')

# =========================
# Ø¯Ø³ØªÙ‡â€ŒØ¨Ù†Ø¯ÛŒ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§
# =========================
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
target_col = 'IsBadBuy'

# =========================
# 1. Ú†Ú©â€ŒÙ„ÛŒØ³Øª Missing Values
# =========================
missing_count = df.isnull().sum()
missing_percent = 100 * missing_count / len(df)

# =========================
# 2. Ú†Ú©â€ŒÙ„ÛŒØ³Øª Rare Categories
# =========================
rare_dict = {}
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    rare_dict[col] = freq[freq < 0.01].index.tolist()

# =========================
# 3. Numerical Outliers
# =========================
outlier_dict = {}
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
    outlier_dict[col] = len(outliers)

# =========================
# 4. Correlations
# =========================
corr = df[num_cols].corr()

# =========================
# 5. Categorical vs Target imbalance
# =========================
cat_target_imbalance = {}
for col in cat_cols:
    crosstab = pd.crosstab(df[col], df[target_col], normalize='index')*100
    # Ø¨Ø±Ø±Ø³ÛŒ Ø§Ø®ØªÙ„Ø§Ù� Ø¨ÛŒÙ† Ú©Ù„Ø§Ø³ 0 Ùˆ 1
    diff = (crosstab[0] - crosstab[1]).abs().max()
    cat_target_imbalance[col] = diff

# =========================
# 6. Ø³Ø§Ø®Øª Ú†Ú©â€ŒÙ„ÛŒØ³Øª Ù†Ù‡Ø§ÛŒÛŒ
# =========================
checklist = pd.DataFrame({
    "column": num_cols + cat_cols,
    "type": ["numerical"]*len(num_cols) + ["categorical"]*len(cat_cols),
    "missing_count": [missing_count[c] for c in num_cols + cat_cols],
    "missing_percent": [missing_percent[c] for c in num_cols + cat_cols],
    "rare_categories": [", ".join(rare_dict[c]) if c in rare_dict else "" for c in num_cols + cat_cols],
    "outliers_count": [outlier_dict[c] if c in outlier_dict else "" for c in num_cols + cat_cols],
    "target_imbalance_indicator": ["" if c in num_cols else cat_target_imbalance[c] for c in num_cols + cat_cols]
})

# =========================
# Ù†Ù…Ø§ÛŒØ´ Ú†Ú©â€ŒÙ„ÛŒØ³Øª
# =========================
display(checklist.sort_values(by="missing_percent", ascending=False))


