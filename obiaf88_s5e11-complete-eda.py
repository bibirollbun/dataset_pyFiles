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


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv(r'/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e11/sample_submission.csv')


df = train


cat_features = df.select_dtypes('object')
num_features = df.select_dtypes('number')


cat_features.columns


num_features.columns


t = (df['loan_paid_back'].value_counts() / df['loan_paid_back'].shape[0])*100


fig,ax = plt.subplots(figsize = (10,5))
bars = ax.bar(t.index,t, color = ['green','red'])
ax.bar_label(bars,padding =3)
ax.set_xticks([0,1],['No paid back','Paid back'])
ax.set_ylabel("% of loan paid/no paid back")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0,95])
ax.set_title("Overall percentages of loans payback");


fig, ax = plt.subplots(3, 2, figsize=(10, 10))
plt.suptitle("Loan payback for categorical variables", fontsize = 12)
for a in  zip(ax.flatten(), cat_features):
    t = (df.groupby(f'{a[1]}')['loan_paid_back'].sum() / df[df['loan_paid_back'] == 1].shape[0])*100
    bars = a[0].bar(t.index,t)
    a[0].bar_label(bars,padding =3,rotation = 90, fontsize = 9)
    a[0].spines['right'].set_visible(False)
    a[0].spines['top'].set_visible(False)
    a[0].set_ylim([0,95])
    a[0].tick_params(rotation = 90)
    a[0].set_title(f"Loan payback for {a[1]}")
plt.tight_layout()


df.groupby('gender')['loan_paid_back'].mean()


columns_to_plot = ['gender', 'marital_status', 'education_level']
g = sns.PairGrid(df, y_vars="loan_paid_back",
                 x_vars= columns_to_plot,
                 height=5, aspect=.5)

# Draw a seaborn pointplot onto each Axes
g.map(sns.pointplot, color="xkcd:plum")
plt.suptitle("Pointplot of categorical variables", fontsize = 20, y = 1.1)
for ax in zip(g.axes[-1,:],columns_to_plot):
    ax[0].tick_params(rotation = 90)
    ax[0].set_title(ax[1])


columns_to_plot = ['employment_status','loan_purpose', 'grade_subgrade']
g = sns.PairGrid(df, y_vars="loan_paid_back",
                 x_vars= columns_to_plot,
                 height=5, aspect=1.2)

# Draw a seaborn pointplot onto each Axes
g.map(sns.pointplot, color="xkcd:plum")
for ax in zip(g.axes[-1,:],columns_to_plot):
    ax[0].tick_params(rotation = 90)
    ax[0].set_title(ax[1])


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


cols_to_scale = [col for col in num_features.columns if col not in 'loan_paid_back']


num_features_scaled = pd.DataFrame(scaler.fit_transform(num_features[cols_to_scale]), columns = scaler.get_feature_names_out())


num_features_pairplot = pd.concat([num_features_scaled, num_features['loan_paid_back']], axis = 1 )


num_features_pairplot = num_features_pairplot.replace([np.inf, -np.inf], np.nan)
num_features_pairplot['loan_paid_back'] = num_features_pairplot['loan_paid_back'].astype('category')


sns.pairplot(num_features_pairplot, hue = 'loan_paid_back')


fig, ax = plt.subplots(3,2,figsize = (10,10))

for a in zip(ax.flatten(), [col for col in num_features.columns if col not in 'loan_paid_back']):
    sns.histplot(data = num_features, x = a[1], hue ='loan_paid_back',  ax = a[0])
    sns.despine(ax = a[0])
    a[0].set_title(a[1])
    a[0].tick_params(rotation = 45)
fig.delaxes(ax.flatten()[-1])
plt.suptitle("Histplot of numerical features", fontsize = 20, y = 1)
plt.tight_layout()


df_corr = num_features[[col for col in num_features.columns if col not in 'loan_paid_back']].corr()


fig,ax = plt.subplots(figsize = (10,8))
sns.color_palette("coolwarm", as_cmap=True)
sns.heatmap(df_corr,annot = True, ax = ax, linewidths=.5, fmt = '.1%', cmap = 'gist_heat')
ax.set_title("Correlation for numerical variables", pad = 10);


combinations = []


for num in [col for col in num_features.columns if col not in 'loan_paid_back']:
    for cat in cat_features.columns:
        combinations.append((num,cat))


fig,ax = plt.subplots(15,2, figsize = (20,40))

for a in zip(ax.flatten(),combinations):
    sns.violinplot(data=df, x=a[1][1], y=a[1][0], hue="loan_paid_back",
               split = True, fill = False, innser = 'point'
               ,palette="pastel", ax = a[0])
    a[0].set_title(f"{a[1][1]} VS {a[1][0]}")
    sns.despine()
plt.suptitle("Categorical vs Numerical Features", fontsize = 20, y = 1)
plt.tight_layout()


df['loan_income_ratio'] = df['loan_amount'] / df['annual_income']
df['loan_rate'] = (df['interest_rate']/100) * df['loan_amount']
df['debt'] = df['annual_income'] * df['debt_to_income_ratio']
df['loan_rate_debt_ratio'] = df['loan_rate'] / df['debt']
df['credit_score_debt_ratio'] = df['credit_score'] / df['debt']


fig, ax = plt.subplots(2,2,figsize = (10,5))
sns.scatterplot(data = df, x = 'debt_to_income_ratio' , y = 'loan_income_ratio', hue = 'loan_paid_back', ax =  ax[0,0], style = 'loan_paid_back')
ax[0,0].set_title("Loan_income_ratio vs Debt_to_income_ratio")

sns.scatterplot(data = df, x = 'loan_income_ratio' , y = 'loan_rate_debt_ratio', hue = 'loan_paid_back', ax =  ax[0,1], style = 'loan_paid_back')
ax[0,1].set_title("Loan_income_ratio vs Loan_rate_debt_ratio")

sns.scatterplot(data = df, x = 'loan_income_ratio' , y = 'credit_score_debt_ratio', hue = 'loan_paid_back', ax =  ax[1,0], style = 'loan_paid_back')
ax[1,0].set_title("Loan_income_ratio vs Credit_score_debt_ratio")

sns.scatterplot(data = df, x = 'loan_rate_debt_ratio' , y = 'credit_score_debt_ratio', hue = 'loan_paid_back', ax =  ax[1,1], style = 'loan_paid_back')
ax[1,1].set_title("Loan_rate_debt_ratio vs Credit_score_debt_ratio")

sns.despine()
plt.tight_layout()

