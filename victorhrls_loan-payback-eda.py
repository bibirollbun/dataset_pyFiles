# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
data.head()


data.info()


data.shape


data.isna().sum()


target_dist = data['loan_paid_back'].value_counts(normalize=True)
print("Target Distribution")
print(target_dist)

# 20 % of people do not payback the loans


# numeric correlations

num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
correlations = data[num_cols + ['loan_paid_back']].corr()['loan_paid_back'].abs().sort_values(ascending=False)


plt.figure(figsize = (12,8))
top_corr = correlations.head(5)

sns.barplot(x=correlations[1:].values, y=correlations.index[1:])
plt.title("Top 5 numeric features by correlation with loan paid back")
plt.xlabel("Absolute correlation with the target")
plt.tight_layout()
plt.plot()


categories = ['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']
label_encoder = LabelEncoder()

for cat in categories:
    data[cat] = label_encoder.fit_transform(data[cat])



X_cat = data[categories]
y = data['loan_paid_back']      # Goal

chi2_stats , p_values = chi2(X_cat,y)


for col, chi2_stat , p_val in zip(X_cat.columns, chi2_stats, p_values):
    print(f'{col}: chi2={chi2_stat:.2f}, p-value={p_val:.4f}')
    


p_values_df = pd.DataFrame({
    'Feature' : X_cat.columns,
    'p-value' : p_values
}).sort_values('p-value')

plt.figure(figsize=(12,8))

sns.barplot(data = p_values_df, x = 'p-value' , y='Feature', palette = 'viridis')
plt.axvline(x=0.05, color = 'red', linestyle = '--' , label = 'significance of 0.05')
plt.title("P-values of the test Chi-Square for the categorical labels")
plt.legend()
plt.tight_layout()
plt.show()




