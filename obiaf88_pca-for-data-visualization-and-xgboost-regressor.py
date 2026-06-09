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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.simplefilter("ignore")
from scipy import stats
from statsmodels.stats.descriptivestats import describe
pd.set_option('display.max_columns', 500)
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error


train_df=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_df_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train_df.head(2)


test_df.head(2)


train_df.info(), test_df.info()


np.round(train_df.isnull().sum()/train_df.shape[0],2), np.round(test_df.isnull().sum()/test_df.shape[0],2)


fig, ax = plt.subplots(figsize = (10,5))
sns.heatmap(train_df[[col for col in train_df.columns if col not in ('Sex','id')]].corr(),annot = True, ax= ax)
ax.set_title('Correlation between features', fontsize = 18, pad = 20)
plt.tight_layout()


sns.displot(data = train_df, x = 'Calories', bins = 100, kde = True)
plt.title('Distribution of calories');


describe(train_df['Calories']).T


fig , ax = plt.subplots(3,2 ,figsize = (15,10))
plt.suptitle('Calories vs numerical features')
ax[0][0].set_title('Calories vs Duration')
sns.regplot(data=train_df,x="Duration", y="Calories", ax = ax[0][0], line_kws = {'color':'red'})
ax[0][1].set_title('Calories vs Heart_Rate')
sns.regplot(data=train_df,x="Heart_Rate", y="Calories", ax = ax[0][1], line_kws = {'color':'red'})
ax[1][0].set_title('Calories vs Body Temp')
sns.regplot(data=train_df,x="Body_Temp", y="Calories", ax = ax[1][0], line_kws = {'color':'red'})
ax[1][1].set_title('Calories vs Weight')
sns.regplot(data=train_df,x="Weight", y="Calories", ax = ax[1][1], line_kws = {'color':'red'})
ax[2][0].set_title('Calories vs Age')
sns.regplot(data=train_df,x="Age", y="Calories", ax = ax[2][0], line_kws = {'color':'red'})
ax[2][1].set_title('Calories vs Height')
sns.regplot(data=train_df,x="Height", y="Calories", ax = ax[2][1], line_kws = {'color':'red'})

plt.tight_layout()


v = ['Duration','Heart_Rate','Body_Temp']


for var in v:
    mod = sm.OLS(train_df[f'{var}'], train_df['Calories'])
    res = mod.fit()
    print(f"\033[91mRegression result between {var} and Calories\033[91m \n")
    print("\033[90m)",res.summary(),"\033[90m)")


plt.title("Calories vs Sex")
sns.violinplot(data=train_df, x="Sex", y="Calories")
sns.despine();


train_df_pca = train_df[[col for col in train_df.columns if col not in ('BMI','id','Calories')]]


train_df_pca = pd.concat([train_df_pca, pd.get_dummies(train_df['Sex'], dtype = int)], axis = 1)


train_df_pca.drop('Sex', axis = 1, inplace = True)


train_df_pca


scaler = StandardScaler()


train_df_pca_scaled = pd.DataFrame(scaler.fit_transform(train_df_pca), columns = scaler.get_feature_names_out())


train_df_pca_scaled


pca_full = PCA().fit(train_df_pca_scaled)


fig, ax = plt.subplots(figsize = (10,6))
ax.plot(np.cumsum(pca_full.explained_variance_ratio_))
ax.set_xlabel("Nr components")
ax.set_ylabel("Explained variance ratio")
ax.set_title("Explained variance ratio vs nr_components", pad = 20)
ax.axvline(4, color = 'green')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate("Four components explains about 98% of the variance",xy=(4, 0.9861), xytext=(6, 0.8),
            arrowprops=dict(facecolor='black', shrink=0.02), color = 'red')
;


pca_2_comp = PCA(n_components=2)  
X_pca_2_comp = pca_2_comp.fit_transform(train_df_pca_scaled)
df_pca_2_comp = pd.DataFrame(X_pca_2_comp, columns = ['PC1','PC2'])
df_pca_2_comp['Calories'] = np.log1p(train_df['Calories'])


fig, ax = plt.subplots(figsize = (10,5))
sns.scatterplot(df_pca_2_comp, x='PC1', y='PC2', hue='Calories', palette = sns.color_palette("viridis", as_cmap=True), ax = ax)


pca = PCA(n_components=4)  
X_pca = pca.fit_transform(train_df_pca_scaled)


df_pca = pd.DataFrame(X_pca, columns = ['PC1','PC2','PC3','PC4'])


df_pca['Calories'] = np.log1p(train_df['Calories'])


df_pca


from itertools import combinations


# with this line of code we calculate the different combinations between components.
for p in list(zip([(i,j) for i in range(3) for j in range(2)], list(combinations(['PC1','PC2','PC3','PC4'], 2)))):
    print(p[1][0],p[1][1], p[0])


fig, ax = plt.subplots(3,2, figsize = (15,9))
plt.suptitle("PCA Components vs Calories")
for p in list(zip([(i,j) for i in range(3) for j in range(2)], list(combinations(['PC1','PC2','PC3','PC4'], 2)))):
    sns.scatterplot(df_pca, x=p[1][0], y=p[1][1], hue='Calories', palette = sns.color_palette("viridis", as_cmap=True), ax = ax[p[0]])
    ax[p[0]].set_title(f"Calories vs {p[1][0]} and {p[1][1]}")
plt.tight_layout()
;


test_df.head(1)


test_df = test_df[[col for col in test_df.columns if col not in ('id','Sex')]]


test_df_scaled = pd.DataFrame(scaler.fit_transform(test_df), columns = scaler.get_feature_names_out())


test_df_pca = pca.fit_transform(test_df_scaled)


df_test_pca = pd.DataFrame(test_df_pca, columns = ['PC1','PC2','PC3','PC4'])


df_test_pca.head(2)


#df_pca is the dataframe of training data
X_train, X_test, y_train, y_test = train_test_split(df_pca[[col for col in df_pca.columns if col not in ('Calories')]], df_pca['Calories'], test_size=0.33, random_state=42)


xgb_regressor=xgb.XGBRegressor(eval_metric='rmsle')


param_grid = {"max_depth":    [4, 5, 6],
              "n_estimators": [500, 600, 700],
              "learning_rate": [0.01, 0.015]}


search_best_params = GridSearchCV(xgb_regressor, param_grid, cv=5).fit(X_train, y_train)
print("The best hyperparameters are ",search_best_params.best_params_)


regressor=xgb.XGBRegressor(learning_rate = search_best_params.best_params_["learning_rate"],
                           n_estimators  = search_best_params.best_params_["n_estimators"],
                           max_depth     = search_best_params.best_params_["max_depth"],
                           eval_metric='rmsle')

regressor.fit(X_train, y_train)


print(f"RMSLE is: {np.sqrt( mean_squared_log_error(regressor.predict(X_test), y_test) )}"  )


calories_predicted = regressor.predict(df_test_pca)


calories_predicted = np.expm1(calories_predicted)


submission = pd.DataFrame({'id':test_df_submission['id'], 'Calories': calories_predicted})


assert submission.shape[0] == test_df_submission.shape[0]


submission


# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
print("Submission created")

