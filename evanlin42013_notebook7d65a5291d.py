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


plt.figure(figsize=(20,10))
sns.countplot(df, x='international_plan', hue='churn')
plt.title("international_paln vs churn")
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

