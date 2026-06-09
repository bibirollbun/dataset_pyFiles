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


import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv', index_col=0)
df_test=pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-test.csv',index_col=0)
df_dict=pd.read_excel('/kaggle/input/GiveMeSomeCredit/Data Dictionary.xls')


df.head()


df_test.head()


df_dict.columns


df_dict['Unnamed: 1'][1]


df_dict


df.rename(columns={'SeriousDlqin2yrs': 'event'}, inplace=True)
df['event'] = df['event'].astype(int)



df.head()


df.shape


df.isna().sum()*100/len(df)


df.columns


plt.figure(figsize=(15,5))
plt.grid()
sns.kdeplot(df['MonthlyIncome'], fill=True)
plt.show()


round(df['MonthlyIncome'].describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]),3)


df['NumberOfDependents'].value_counts()


# Step 2: Data preprocessing
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True) 


# Define duration as a proxy: e.g., age or create synthetic durations
df['duration'] = df['age']  # This is a placeholder. In real data, use time to default or observation time.


%%capture
!pip install lifelines


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter



# Step 3: Kaplan-Meier Estimator
kmf = KaplanMeierFitter()
kmf.fit(durations=df['duration'], event_observed=df['event'])



# Plot the survival function
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title("Kaplan-Meier Survival Curve")
plt.xlabel("Time (Age used as duration)")
plt.ylabel("Probability of Survival (No Default)")
plt.grid()
plt.show()


# Step 4: Cox Proportional Hazards Model
cph_df = df[['duration', 'event', 'RevolvingUtilizationOfUnsecuredLines', 
             'DebtRatio', 'age', 'NumberOfOpenCreditLinesAndLoans', 'MonthlyIncome']]



cph = CoxPHFitter()
cph.fit(cph_df, duration_col='duration', event_col='event')


# Summary of the Cox model
cph.print_summary()


# Step 5: Visualize effects
cph.plot()
plt.title("Cox Model - Covariate Effects on Hazard Rate")
plt.show()


# Optional: create income group for stratified Kaplan-Meier
df['IncomeGroup'] = pd.qcut(df['MonthlyIncome'], 3, labels=['Low', 'Medium', 'High'])


df['IncomeGroup'].value_counts()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index



# Step 2: Train/Test Split
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42) 


train_df.shape, test_df.shape


# Step 3: Kaplan-Meier Stratified by Income
plt.figure(figsize=(10, 6))
plt.grid()
for group in df['IncomeGroup'].unique():
    kmf = KaplanMeierFitter()
    mask = train_df['IncomeGroup'] == group
    kmf.fit(train_df[mask]['duration'], train_df[mask]['event'], label=str(group))
    kmf.plot_survival_function(ci_show=False)
plt.title("Stratified Kaplan-Meier Curve by Income Group")
plt.xlabel("Duration (Age)")
plt.ylabel("Survival Probability")
plt.grid()
plt.legend(title="Income Group")
plt.show()


# Step 4: Cox Model on Train Set
features = ['duration', 'event', 'RevolvingUtilizationOfUnsecuredLines',
            'DebtRatio', 'age', 'NumberOfOpenCreditLinesAndLoans', 'MonthlyIncome']



cph = CoxPHFitter()
cph.fit(train_df[features], duration_col='duration', event_col='event')
cph.print_summary()
cph.plot()
plt.title("Cox Model - Feature Effects on Hazard Rate")
plt.show()



# Step 5: Evaluate Cox Model on Test Set using Concordance Index
test_pred = cph.predict_partial_hazard(test_df[features[2:]])
ci = concordance_index(test_df['duration'], -test_pred, test_df['event'])
print(f"\nConcordance Index on Test Set: {ci:.4f}")


# Use test data and trained Cox model
test_df['risk_score'] = cph.predict_partial_hazard(test_df[cph.params_.index])

# 1️⃣ Create deciles of risk scores
test_df['risk_decile'] = pd.qcut(test_df['risk_score'], 10, labels=False)

# 2️⃣ Plot KM survival curves by risk decile
plt.figure(figsize=(12, 6))
for d in sorted(test_df['risk_decile'].unique()):
    kmf = KaplanMeierFitter()
    mask = test_df['risk_decile'] == d
    kmf.fit(durations=test_df[mask]['duration'], 
            event_observed=test_df[mask]['event'], 
            label=f"Decile {d+1}")
    kmf.plot_survival_function(ci_show=False)

plt.title("Survival Curves by Predicted Risk Decile")
plt.xlabel("Time (Duration)")
plt.ylabel("Survival Probability")
plt.legend(title="Risk Decile", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()




