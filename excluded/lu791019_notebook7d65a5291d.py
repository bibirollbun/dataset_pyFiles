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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore")


df = pd.read_csv(os.path.join(dirname, 'train.csv'))
df.info()


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='churn')


sns.displot(df, x='account_length', hue='churn', kde=True)
plt.title('Churn rate vs Account length')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 創建 account_length 的分組
bins = [0, 12, 24, 36, 48, 60, 72, 100, 150, 200, 250]
df['account_length_bins'] = pd.cut(df['account_length'], bins=bins)

# 繪製分組後的流失率
plt.figure(figsize=(10,6))
ax = sns.countplot(data=df, x='account_length_bins', hue='churn', palette={'no': 'blue', 'yes': 'red'})

# 添加數值標籤
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), 
                textcoords='offset points')

# 設定標題與標籤
plt.title('Churn Rate by Account Length', fontsize=14)
plt.xlabel('Account Length (Months)', fontsize=12)
plt.ylabel('Customer Count', fontsize=12)

# 調整圖例位置
plt.legend(title='Churn', loc='upper right', bbox_to_anchor=(1.2, 1))

plt.show()



plt.figure(figsize=(20,10))
sns.countplot(df, x='international_plan', hue='churn')
plt.title("international_paln vs churn")
plt.show()


# 計算每類的流失比例
churn_ratio = df.groupby('international_plan')['churn'].value_counts(normalize=True).unstack()
churn_ratio.plot(kind='bar', stacked=True, figsize=(8,6), colormap='coolwarm')

plt.title("Churn Rate by International Plan Subscription", fontsize=14)
plt.xlabel("International Plan Subscription", fontsize=12)
plt.ylabel("Percentage", fontsize=12)
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.show()



plt.figure(figsize=(20,10))
sns.countplot(df, x='voice_mail_plan', hue='churn')
plt.title("international_paln vs churn")
plt.show()


plt.figure(figsize=(10,6))
sns.displot(df, x='number_customer_service_calls', kde=True)
plt.show()


plt.figure(figsize=(10,6))
sns.displot(df, x='number_customer_service_calls', hue='churn', kde=True)
plt.show()


plt.figure(figsize=(10,6))
sns.displot(df, x='number_vmail_messages', hue='churn', kde=True)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
ax = sns.countplot(data=df, x='number_customer_service_calls', hue='churn', palette={'no': 'blue', 'yes': 'red'})

# 加上標籤
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), 
                textcoords='offset points')

plt.title("Customer Service Calls vs Churn", fontsize=14)
plt.xlabel("Number of Customer Service Calls", fontsize=12)
plt.ylabel("Customer Count", fontsize=12)


plt.legend(title='Churn', loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()



churn_ratio = df.groupby('number_customer_service_calls')['churn'].value_counts(normalize=True).unstack()
churn_ratio.plot(kind='bar', stacked=True, figsize=(8,6), colormap='coolwarm')

plt.title("Churn Rate by Customer Service Calls", fontsize=14)
plt.xlabel("Number of Customer Service Calls", fontsize=12)
plt.ylabel("Percentage", fontsize=12)
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.show()


