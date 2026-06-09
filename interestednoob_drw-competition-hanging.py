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


#getting data analysis and ML packages
import pandas
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #transform different types
from sklearn.datasets import fetch_openml
import numpy
from numpy import sqrt
from numpy import sum
from numpy import square
import seaborn
import matplotlib
import statsmodels
import time

from sklearn.decomposition import PCA
import xgboost as xgb
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from mlxtend.regressor import StackingRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


!pip install ipympl

# Restart the kernel after installing for the changes to take effect.
# After restarting, you can run the cell again.


df_train = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
#replace = pandas.read_csv('/kaggle/input/x1-x780/pls-762-to-5.csv')
#df_test = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')
#df_sample = pandas.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')


df_train = df_train.drop(df_train.loc[:,'X1':'X780'].columns, axis=1)


df_train


df_train['lag_1'] = df_train['label'].shift(periods=1)
df_train['lag_2'] = df_train['label'].shift(periods=2)
df_train['lag_3'] = df_train['label'].shift(periods=3)
df_train['lag_4'] = df_train['label'].shift(periods=4)
df_train['lag_5'] = df_train['label'].shift(periods=5)
df_train['lag_6'] = df_train['label'].shift(periods=6)
df_train['lag_7'] = df_train['label'].shift(periods=7)
df_train['lag_8'] = df_train['label'].shift(periods=8)
df_train['lag_9'] = df_train['label'].shift(periods=9)
df_train['lag_10'] = df_train['label'].shift(periods=10)
df_train['lag_11'] = df_train['label'].shift(periods=11)
df_train['lag_12'] = df_train['label'].shift(periods=12)
df_train['lag_13'] = df_train['label'].shift(periods=13)
df_train['lag_14'] = df_train['label'].shift(periods=14)
df_train['lag_15'] = df_train['label'].shift(periods=15)
df_train['lag_16'] = df_train['label'].shift(periods=16)
df_train.loc[:,'lag_1':'lag_16'] = df_train.loc[:,'lag_1':'lag_16'].replace({np.nan: df_train['label'].mean()})


## Dropping infs
infcol = df_train.columns[np.isinf(df_train).any(axis=0)].tolist()
print(infcol)

df_train = df_train.drop(columns=infcol)


## converting index from datetime to numbers. Cannot use pipeline bc not applying to entire df, would need columnTransformer, too lazy
df_train.index = range(0,len(df_train.index))


df_train = df_train.join(replace)


df_train = df_train.rename(columns={'0':'PLS_1','1':'PLS_2','2':'PLS_3','3':'PLS_4','4':'PLS_5'})


df_train.isna().sum()


df_train.columns.values


df_train.dtypes


df_train.describe()


## Correlation (Done)
temp = df_train.iloc[0:44640,0:900]
seaborn.heatmap(temp.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


## Histogram (Done)
output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)

for col in df_train.columns:
    if df_train[col].isin([numpy.inf, -numpy.inf]).any().any() == False:
        fig, ax = plt.subplots()
        df_train[col].plot(kind='hist', ax=ax,bins=20)
        plt.title(f"{col} histogram")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_histogram.png")


#Histogram and KDE but with separated y axis by twinning the x axis (Done)
output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)

for col in df_train.columns:
    if df_train[col].isin([numpy.inf, -numpy.inf]).any().any() == False:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        df_train[col].plot(kind='hist', bins=20, ax=ax)
        df_train[col].plot(kind='kde',ax=ax2,color='red', linestyle='--')
        plt.title(f"{col} Density and Histogram Plot")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_Density and histogram.png")


# Tell histogram to change y axis to density bc I intend to plot kde also, and use histogram as plotted axis
ax = df_train['X354'].plot(kind='hist',stacked=False, density=True)
df_train['X354'].plot.kde(ax=ax)
plt.title(f"X354 Density Plot")
plt.show()


# KDE (Done)
output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)

for col in df_train.columns:
    ax = df_train[col].plot.kde()
    plt.title(f"{col} Density Plot")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{col}_KDE.png")


# Which ones are normal?
from scipy import stats

for col in df_train.columns:
    if df_train[col].isin([numpy.inf, -numpy.inf]).any().any() == False:
        p = stats.kstest(df_train[col],stats.norm.cdf).pvalue
        pl = stats.kstest(df_train[col],stats.norm.cdf, alternative='less').pvalue
        pg = stats.kstest(df_train[col],stats.norm.cdf, alternative='greater').pvalue
        if p > 0.005:
            print(col, ':Not rejected, RV is normally distributed')
        elif pl > 0.005:
            print(col, ':Not rejected, RV is normally distributed, leaning towards greater value')
        elif pg > 0.005:
            print(col, ':Not rejected, RV is normally distributed, leaning towards lesser value \n=====')


# Which ones are normal (Detailed)?
from scipy import stats

for col in df_train.columns:
    if df_train[col].isin([numpy.inf, -numpy.inf]).any().any() == False:
        p = stats.kstest(df_train[col],stats.norm.cdf).pvalue
        pl = stats.kstest(df_train[col],stats.norm.cdf, alternative='less').pvalue
        pg = stats.kstest(df_train[col],stats.norm.cdf, alternative='greater').pvalue
        if p <= 0.005:
            print(col, ':Rejected, Not normal @ 0.5% α')
        elif p > 0.005:
            print(col, ':Not rejected, RV is normally distributed')
        if pl <= 0.005:
            print(col, ':Rejected, Not normal @ 0.5% α (less test)')
        elif pl > 0.005:
            print(col, ':Not rejected, RV is normally distributed, leaning towards greater value')
        if pg <= 0.005:
            print(col, ':Rejected, Not normal @ 0.5% α (greater test) \n=====')
        elif pg > 0.005:
            print(col, ':Not rejected, RV is normally distributed, leaning towards lesser value \n=====')


# Which ones are correlated significantly?
from scipy import stats

for col in df_train.columns:
    if df_train[col].isin([numpy.inf, -numpy.inf]).any().any() == False:
        p = stats.pearsonr(df_train[col],df_train['label']).pvalue
        pl = stats.pearsonr(df_train[col],df_train['label'], alternative='less').pvalue
        pg = stats.pearsonr(df_train[col],df_train['label'], alternative='greater').pvalue
        if p <= 0.005:
            print(col, ':Rejected, Rho != 0 \n=====')
        if pl <= 0.005:
            print(col, ':Rejected, Rho < 0, Rho is likely negative \n=====')
        if pg <= 0.005:
            print(col, ':Rejected, Rho > 0, Rho is likely positive \n=====')


## defunc
cols=[]
for col in df_train.columns:
    if df_train[col].isin([numpy.inf, -numpy.inf]).any().any() == True:
        cols.append(col)
cols


import os

output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)

for col in df_train.columns:
    ax = df_train.plot.scatter(x=col, y='label')
    plt.title(f"{col} vs label")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{col}_vs_label.png")



### output all the graphs to a zip file
!zip -r output_folder.zip  /kaggle/working/


scaler = StandardScaler().fit(df_train[['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890','year','month','day']])
temp = scaler.transform(df_train[['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890','year','month','day']])
temp = pandas.DataFrame(temp).rename(columns={0:'bid_qty_std',1:'ask_qty_std',2:'buy_qty_std',3:'sell_qty_std',4:'volume_std',5:'X17_std',6:'X25_std',7:'X198_std',8:'X205_std',9:'X413_std',10:'X415_std',11:'X421_std',12:'X594_std',13:'X596_std',14:'X606_std',15:'X610_std',16:'X663_std',17:'X667_std',18:'X675_std',19:'X681_std',20:'X687_std',21:'X693_std',22:'X865_std',23:'X875_std',24:'X876_std',25:'X879_std',26:'X880_std',27:'X881_std',28:'X882_std',29:'X883_std',30:'X890_std',31:'year_std',32:'month_std',33:'day_std'})
df_train = df_train.join(temp)


columns=['X345-int-X387',
        'X345-int-X429',
        'X387-int-X429',
        'X339-int-X381',
        'X339-int-X423',
        'X381-int-X423',
        'X333-int-X375',
        'X333-int-X417',
        'X375-int-X417',
        'X92-int-X134',
        'X92-int-X176',
        'X134-int-X176',
        'X68-int-X110',
        'X68-int-X152',
        'X110-int-X152',
        'X435-int-X438',
        'X188-int-X191',
        'X657-int-X660',
        'X669-int-X672',
        'X685-int-X691',
        'X688-int-X694',
        'X681-int-X684',
        'X651-int-X654',
        'X676-int-X682',
        'X673-int-X679',
        'X664-int-X670',
        'X645-int-X648',
        'X663-int-X666',
        'X639-int-X642',
        'X675-int-X678',
        'X652-int-X658',
        'X661-int-X667',
        'X693-int-X696',
        'X687-int-X690',
        'X640-int-X646',
        'X649-int-X655',
        'X637-int-X643',
        'X615-int-X618',
        'X621-int-X624',
        'X628-int-X634',
        'X627-int-X630',
        'X298-int-X300',
        'X625-int-X631',
        'X299-int-X301',
        'X633-int-X636',
        'X51-int-X53',
        'X432-int-X435',
        'X482-int-X489',
        'X45-int-X47',
        'X52-int-X54',
        'X300-int-X302',
        'X613-int-X619',
        'X483-int-X490',
        'X481-int-X488',
        'X616-int-X622',
        'X185-int-X188',
        'X432-int-X438',
        'X301-int-X303',
        'X431-int-X434',
        'X7-int-X8',
        'X484-int-X491',
        'X292-int-X294',
        'X185-int-X191',
        'X46-int-X48',
        'X434-int-X437',
        'X293-int-X295',
        'X286-int-X288',
        'X39-int-X41',
        'X487-int-X494',
        'X235-int-X242',
        'X239-int-X246',
        'X240-int-X247',
        'X486-int-X493',
        'X485-int-X492',
        'X236-int-X243',
        'X40-int-X42',
        'X234-int-X241',
        'X287-int-X289',
        'X238-int-X245',
        'X237-int-X244',
        'X53-int-X55',
        'X6-int-X7',
        'X280-int-X282',
        'X281-int-X283',
        'X54-int-X56',
        'X254-int-X255',
        'X47-int-X49',
        'X298-int-X302',
        'X756-int-X757',
        'X885-int-X886',
        'X299-int-X303',
        'X15-int-X16',
        'X752-int-X753',
        'X873-int-X874',
        'X253-int-X254',
        'X33-int-X35',
        'X294-int-X296',
        'X295-int-X297',
        'X48-int-X50',
        'X431-int-X437']


## make interaction variables
for ip in columns:
    c1, c2 = ip.split('-int-')
    df_train[ip] = df_train[c1]*df_train[c2]


## Dropping perfectly one-half of all perfectly correlated pairs of RVs
df_train = df_train.drop(['X104',
                            'X62',
                            'X116',
                            'X74',
                            'X122',
                            'X80',
                            'X128',
                            'X86',
                            'X140',
                            'X98',
                            'X351',
                            'X309',
                            'X357',
                            'X315',
                            'X363',
                            'X321',
                            'X369',
                            'X327',
                            'date'], axis=1)


X = df_train.loc[:, df_train.columns != 'label']
y = df_train['label'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)

scaler = StandardScaler().fit(X_train[['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890']])
temp = scaler.transform(X_train[['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890']])
temp = pandas.DataFrame(temp).rename(columns={0:'bid_qty_std',1:'ask_qty_std',2:'buy_qty_std',3:'sell_qty_std',4:'volume_std',5:'X17_std',6:'X25_std',7:'X198_std',8:'X205_std',9:'X413_std',10:'X415_std',11:'X421_std',12:'X594_std',13:'X596_std',14:'X606_std',15:'X610_std',16:'X663_std',17:'X667_std',18:'X675_std',19:'X681_std',20:'X687_std',21:'X693_std',22:'X865_std',23:'X875_std',24:'X876_std',25:'X879_std',26:'X880_std',27:'X881_std',28:'X882_std',29:'X883_std',30:'X890_std'})
temp.index = X_train.index
X_train = X_train.join(temp)
X_train = X_train.drop(columns=['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890'])

scaler = StandardScaler().fit(X_test[['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890','year','month','day']])
temp = scaler.transform(X_test[['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890','year','month','day']])
temp = pandas.DataFrame(temp).rename(columns={0:'bid_qty_std',1:'ask_qty_std',2:'buy_qty_std',3:'sell_qty_std',4:'volume_std',5:'X17_std',6:'X25_std',7:'X198_std',8:'X205_std',9:'X413_std',10:'X415_std',11:'X421_std',12:'X594_std',13:'X596_std',14:'X606_std',15:'X610_std',16:'X663_std',17:'X667_std',18:'X675_std',19:'X681_std',20:'X687_std',21:'X693_std',22:'X865_std',23:'X875_std',24:'X876_std',25:'X879_std',26:'X880_std',27:'X881_std',28:'X882_std',29:'X883_std',30:'X890_std',31:'year_std',32:'month_std',33:'day_std'})
temp.index = X_test.index
X_test = X_test.join(temp)
X_test = X_test.drop(columns=['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890'])

#X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values


from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

sgd = SGDRegressor()
par = PassiveAggressiveRegressor()

models = [('SGD Reg', sgd), ('PA Reg', par)]


for model_name, model in models:
    ave_r2, ave_mae, ave_mape, ave_rmse, ave_mdae = [],[],[],[],[]
    for i in range(0,2):
        kfold = model_selection.KFold(n_splits=5, random_state=i, shuffle=True) 
	
    	# execute cross val to est skill of ML model (cross_validate faster)
        start = time.time()
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring = ('r2','neg_root_mean_squared_error','neg_mean_absolute_error','neg_mean_absolute_percentage_error','neg_median_absolute_error'),return_train_score=False)
        end = time.time() 
        duration =  end-start
        
        ave_r2.append(cv_results['test_r2'].mean())
        ave_mae.append(cv_results['test_neg_mean_absolute_error'].mean())
        ave_mape.append(cv_results['test_neg_mean_absolute_percentage_error'].mean())
        ave_rmse.append(cv_results['test_neg_root_mean_squared_error'].mean())
        ave_mdae.append(cv_results['test_neg_median_absolute_error'].mean())

    # print results
    import statistics
    msg1 = "%s: mean R2: %f (SD: %f) \nMAE: %f (SD: %f) \nMAPE: %f (SD: %f) \nRMSE: %f (SD: %f) \nMDAE: %f (SD: %f)" % (model_name, statistics.mean(ave_r2),statistics.stdev(ave_r2), statistics.mean(ave_mae),statistics.stdev(ave_mae), statistics.mean(ave_mape),statistics.stdev(ave_mape), statistics.mean(ave_rmse),statistics.stdev(ave_rmse), statistics.mean(ave_mdae),statistics.stdev(ave_mdae))
    print(msg1, "\n Time ~", duration)


from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

etr = ExtraTreesRegressor()
hgbr =  HistGradientBoostingRegressor()
rfr = RandomForestRegressor()

models = [('Extra Trees Reg Ensemble', etr),('Random Forest Reg', rfr),('Hist GB Reg',hgbr)]


for model_name, model in models:
    ave_r2, ave_mae, ave_mape, ave_rmse, ave_mdae = [],[],[],[],[]
    for i in range(0,2):
        kfold = model_selection.KFold(n_splits=5, random_state=i, shuffle=True) # stratified for classing
	
    	# execute cross val to est skill of ML model (cross_validate faster)
        start = time.time()
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring = ('r2','neg_root_mean_squared_error','neg_mean_absolute_error','neg_mean_absolute_percentage_error','neg_median_absolute_error'),return_train_score=False)
        end = time.time() 
        duration =  end-start
        
        ave_r2.append(cv_results['test_r2'].mean())
        ave_mae.append(cv_results['test_neg_mean_absolute_error'].mean())
        ave_mape.append(cv_results['test_neg_mean_absolute_percentage_error'].mean())
        ave_rmse.append(cv_results['test_neg_root_mean_squared_error'].mean())
        ave_mdae.append(cv_results['test_neg_median_absolute_error'].mean())

    # print results
    import statistics
    msg1 = "%s: mean R2: %f (SD: %f) \nMAE: %f (SD: %f) \nMAPE: %f (SD: %f) \nRMSE: %f (SD: %f) \nMDAE: %f (SD: %f)" % (model_name, statistics.mean(ave_r2),statistics.stdev(ave_r2), statistics.mean(ave_mae),statistics.stdev(ave_mae), statistics.mean(ave_mape),statistics.stdev(ave_mape), statistics.mean(ave_rmse),statistics.stdev(ave_rmse), statistics.mean(ave_mdae),statistics.stdev(ave_mdae))
    print(msg1, "\n Time ~", duration)


from sklearn.linear_model import Lasso, LinearRegression, ElasticNet

lsr = Lasso()
lr = LinearRegression()
elst = ElasticNet()

models = [('Lasso', lsr), ('Linear', lr), ('ElasticNet', elst)]


for model_name, model in models:
    ave_r2, ave_mae, ave_mape, ave_rmse, ave_mdae = [],[],[],[],[]
    for i in range(0,6):
        kfold = model_selection.KFold(n_splits=5) # not stratified, no class
	
    	# execute cross val to est skill of ML model (cross_validate faster)
        start = time.time()
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring = ('r2','neg_root_mean_squared_error','neg_mean_absolute_error','neg_mean_absolute_percentage_error','neg_median_absolute_error'),return_train_score=False)
        end = time.time() 
        duration =  end-start
        
        ave_r2.append(cv_results['test_r2'].mean())
        ave_mae.append(cv_results['test_neg_mean_absolute_error'].mean())
        ave_mape.append(cv_results['test_neg_root_mean_squared_error'].mean())
        ave_rmse.append(cv_results['test_neg_mean_absolute_percentage_error'].mean())
        ave_mdae.append(cv_results['test_neg_median_absolute_error'].mean())

    # print results
    import statistics
    msg1 = "%s: mean R2: %f (SD: %f) \nMAE: %f (SD: %f) \nMAPE: %f (SD: %f) \nRMSE: %f (SD: %f) \nMDAE: %f (SD: %f)" % (model_name, statistics.mean(ave_r2),statistics.stdev(ave_r2), statistics.mean(ave_mae),statistics.stdev(ave_mae), statistics.mean(ave_mape),statistics.stdev(ave_mape), statistics.mean(ave_rmse),statistics.stdev(ave_rmse), statistics.mean(ave_mdae),statistics.stdev(ave_mdae))
    print(msg1, "\n Time ~", duration)


from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Perceptron, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn import metrics

lsr = Lasso()
lr = LinearRegression()
elst = ElasticNet()
svr_rbf = SVR(kernel='rbf')

kfold = model_selection.KFold(n_splits=5) 

stk = StackingRegressor(estimators=[('Lasso', lsr), ('Linear', lr), ('ElasticNet', elst)], final_estimator=svr_rbf, cv=kfold)

# Training the stacking classifier

stk.fit(X_train, y_train)

# Evaluate and visualize the fit

print("RMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f" % (metrics.root_mean_squared_error(y_test,stk.predict(X_test)), metrics.r2_score(y_test,stk.predict(X_test)), metrics.mean_absolute_percentage_error(y_test,stk.predict(X_test)),metrics.median_absolute_error(y_test,stk.predict(X_test)),metrics.mean_absolute_error(y_test,stk.predict(X_test)) ))

with plt.style.context(('seaborn-whitegrid')):
    plt.scatter(X_train, y_train, c='lightgray')
    plt.plot(X_train, stk.predict(X_train), c='darkgreen', lw=2)

plt.show()


from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Perceptron, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn import metrics

lr = LinearRegression()
svr_lin = SVR(kernel='linear')
elst = ElasticNet()
svr_rbf = SVR(kernel='rbf')

kfold = model_selection.KFold(n_splits=5) 

stk2 = StackingRegressor(estimators=[('ElasticNet', elst), ('Linear', lr), ('Linear SVR', svr_lin)], final_estimator=svr_rbf, cv=kfold)

# Training the stacking classifier

stk2.fit(X_train, y_train)

# Evaluate and visualize the fit

print("RMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f" % (metrics.root_mean_squared_error(y_test,stk2.predict(X_test)), metrics.r2_score(y_test,stk2.predict(X_test)), metrics.mean_absolute_percentage_error(y_test,stk2.predict(X_test)),metrics.median_absolute_error(y_test,stk2.predict(X_test)),metrics.mean_absolute_error(y_test,stk2.predict(X_test)) ))

with plt.style.context(('seaborn-whitegrid')):
    plt.scatter(X_train, y_train, c='lightgray')
    plt.plot(X_train, stk.predict(X_train), c='darkgreen', lw=2)

plt.show()


## defunc
X = df_train.loc[:, df_train.columns != 'label']
y = df_train.loc[:, df_train.columns == 'label']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)

scaler = StandardScaler().fit(X_train)
temp = scaler.transform(X_train)
rename_dict = dict(zip(temp.columns,X_train.columns))
X_train = pandas.DataFrame(temp).rename(columns=rename_dict)

scaler = StandardScaler().fit(X_test)
temp = scaler.transform(X_test)
rename_dict = dict(zip(temp.columns, X_test.columns))
X_test = pandas.DataFrame(temp).rename(columns=rename_dict)


X = df_train.loc[:, df_train.columns != 'label']
y = df_train['label'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)


X_train.columns


from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Lasso, LinearRegression, ElasticNet, Ridge
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn import metrics
import sklearn.pipeline
import gc

gc.collect()
#sgd = Pipeline([('std', StandardScaler()), ('sgd',SGDRegressor())])
#par = Pipeline([('std', StandardScaler()), ('par',PassiveAggressiveRegressor())])
#etr = Pipeline([('std', StandardScaler()), ('etr',ExtraTreesRegressor())])
hgbr =  Pipeline([('std', StandardScaler()), ('hgbr',HistGradientBoostingRegressor())])
#rfr = Pipeline([('std', StandardScaler()), ('rfr',RandomForestRegressor())])
lsr = Pipeline([('std', StandardScaler()), ('lsr',Lasso())])
lr = Pipeline([('std', StandardScaler()), ('lr',LinearRegression())])
elst = Pipeline([('std', StandardScaler()), ('elst',ElasticNet())])
rd = Pipeline([('std', StandardScaler()), ('rd',Ridge())])

models = [('Hist GB Reg',hgbr), ('Lasso', lsr), ('Linear', lr), ('ElasticNet', elst), ('Ridge', rd)]

for model_name, model in models:
    ave_r2, ave_mae, ave_mape, ave_rmse, ave_mdae, ave_time = [],[],[],[],[],[]
    for i in range(0,5):
        kfold = model_selection.KFold(n_splits=5, random_state=i, shuffle=True) 
	
    	# execute cross val to est skill of ML model (cross_validate faster)
        start = time.time()
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring = ('r2','neg_root_mean_squared_error','neg_mean_absolute_error','neg_mean_absolute_percentage_error','neg_median_absolute_error'),return_train_score=False)
        end = time.time() 
        duration =  end-start
        
        ave_r2.append(cv_results['test_r2'].mean())
        ave_mae.append(cv_results['test_neg_mean_absolute_error'].mean())
        ave_mape.append(cv_results['test_neg_mean_absolute_percentage_error'].mean())
        ave_rmse.append(cv_results['test_neg_root_mean_squared_error'].mean())
        ave_mdae.append(cv_results['test_neg_median_absolute_error'].mean())
        ave_time.append(duration)

    # print results
    import statistics
    msg1 = "%s: mean R2: %f \nMAE: %f \nMAPE: %f \nRMSE: %f \nMDAE: %f" % (model_name, statistics.mean(ave_r2), statistics.mean(ave_mae), statistics.mean(ave_mape), statistics.mean(ave_rmse), statistics.mean(ave_mdae))
    print(msg1, "\n Time ~", statistics.mean(ave_time))
    gc.collect()



from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_absolute_percentage_error
import gc

lsr = Pipeline([('std', StandardScaler()), ('lsr',Lasso())])
rd = Pipeline([('std', StandardScaler()), ('lr',Ridge())])
elst = Pipeline([('std', StandardScaler()), ('elst',ElasticNet(l1_ratio=0.2, alpha=0.9))])
par = Pipeline([('std', StandardScaler()), ('par',PassiveAggressiveRegressor())])
hgbr =  HistGradientBoostingRegressor()
sgd = Pipeline([('std', StandardScaler()), ('sgd',SGDRegressor())])

kfold = model_selection.KFold(n_splits=5) 

stk = Pipeline([('std',StandardScaler()),('stacker',StackingRegressor(estimators=[('Lasso', lsr), ('Ridge', rd), ('ElasticNet', elst)], final_estimator=rd, cv=kfold))])
stk2 = Pipeline([('std',StandardScaler()),('stacker',StackingRegressor(estimators=[('PAR', par), ('Ridge', rd), ('SGD', sgd)], final_estimator=hgbr, cv=kfold))])


# Training the stacking classifier
for stack in [stk,stk2]:
    start = time.time()
    stack.fit(X_train, y_train)
    end = time.time()
    duration = end - start
    
    # Evaluate and visualize the fit
    #with plt.style.context(('seaborn-whitegrid')):
    #    plt.scatter(X_train[['volume']], y_train, c='darkred')
    #    plt.scatter(X_train[['volume']], stack.predict(X_train), c='darkgreen')
    #plt.show()
    
    print("MSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (mean_squared_error(y_test,stack.predict(X_test)), r2_score(y_test,stack.predict(X_test)), mean_absolute_percentage_error(y_test,stack.predict(X_test)),median_absolute_error(y_test,stack.predict(X_test)),mean_absolute_error(y_test,stack.predict(X_test)), duration ))    
    gc.collect()


## Updated to above, removal of SVR, too slow @ 12 months dataset, defunc now
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_absolute_percentage_error

lsr = Pipeline([('std', StandardScaler()), ('lsr',Lasso())])
rd = Pipeline([('std', StandardScaler()), ('lr',Ridge())])
elst = Pipeline([('std', StandardScaler()), ('elst',ElasticNet())])
svr_rbf = SVR(kernel='rbf')
svr_lin = SVR(kernel='linear')
hgbr =  Pipeline([('std', StandardScaler()), ('hgbr',HistGradientBoostingRegressor())])

kfold = model_selection.KFold(n_splits=5) 

stk = Pipeline([('std',StandardScaler()),('stacker',StackingRegressor(estimators=[('Lasso', lsr), ('Ridge', rd), ('ElasticNet', elst)], final_estimator=svr_rbf, cv=kfold))])
stk2 = Pipeline([('std',StandardScaler()),('stacker',StackingRegressor(estimators=[('ElasticNet', elst), ('Histogram based GB', hgbr), ('Linear SVR', svr_lin)], final_estimator=svr_rbf, cv=kfold))])


# Training the stacking classifier
for stack in [stk,stk2]:
    #stack.fit(X_train, y_train)
    # execute cross val to est skill of ML model (cross_validate faster)
    start = time.time()
    cv_results = model_selection.cross_validate(stack, X_train, y_train, cv=kfold, scoring = ('r2','neg_root_mean_squared_error','neg_mean_absolute_error','neg_mean_absolute_percentage_error','neg_median_absolute_error'),return_train_score=False)
    end = time.time()
    duration =  end-start
    
    # Evaluate and visualize the fit
    
    #print("MSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f" % (mean_squared_error(y_test,stack.predict(X_test)), r2_score(y_test,stack.predict(X_test)), mean_absolute_percentage_error(y_test,stack.predict(X_test)),median_absolute_error(y_test,stack.predict(X_test)),mean_absolute_error(y_test,stack.predict(X_test)) ))
    msg1 = "%s: mean R2: %f \nMAE: %f \nMAPE: %f \nRMSE: %f \nMDAE: %f" % (model_name, statistics.mean(cv_results['test_r2']), statistics.mean(cv_results['test_neg_mean_absolute_error']), statistics.mean(cv_results['test_neg_mean_absolute_percentage_error']), statistics.mean(cv_results['test_neg_root_mean_squared_error']), statistics.mean(cv_results['test_neg_median_absolute_error']))
    print(msg1, "\n Time ~", statistics.mean(duration))
    
    gc.collect()


with plt.style.context(('seaborn-whitegrid')):
        plt.scatter(X_train, y_train, c='darkred')
        plt.scatter(X_train, stack.predict(X_train), c='darkgreen', lw=2)
    
    plt.show()


from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Perceptron, Lasso, LinearRegression, ElasticNet

param_grid_sgd = {
    'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet']
}

param_grid = {
    'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'],
    'epsilon': [0.1, 0.2, 0.5],
    'C': [0.2,1,2]
}

model = PassiveAggressiveRegressor(tol=1e-4)

kfold = model_selection.KFold(n_splits=5)
start = time.time()
# Define the grid search we want to run. Run it with six cpus in parallel.
gs_cv = model_selection.GridSearchCV(model, param_grid, cv=kfold, n_jobs=6, verbose=100)

# Run the grid search - on only the training data
gs_cv.fit(X_train, y_train)

# Print the parameters that gave us the best result
print(gs_cv.best_params_)
end = time.time()
print(end - start)


##ensemble results: {'l2_regularization': 0.9, 'learning_rate': 0.3, 'min_samples_leaf': 20}
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

param_grid = {
    'l2_regularization': [0,0.3,0.6,0.9],
    'learning_rate': [0,0.3,0.6,0.9],
    'min_samples_leaf': [20,40]
}

model = hgbr =  HistGradientBoostingRegressor()

kfold = model_selection.KFold(n_splits=5)
start = time.time()
# Define the grid search we want to run. Run it with six cpus in parallel.
gs_cv = model_selection.GridSearchCV(model, param_grid, cv=kfold, n_jobs=6, verbose=100)

# Run the grid search - on only the training data
gs_cv.fit(X_train, y_train)
end = time.time()

# Print the parameters that gave us the best result
print(gs_cv.best_params_)
print("RMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,gs_cv.predict(X_test))), r2_score(y_test,gs_cv.predict(X_test)), mean_absolute_percentage_error(y_test,gs_cv.predict(X_test)),median_absolute_error(y_test,gs_cv.predict(X_test)),mean_absolute_error(y_test,gs_cv.predict(X_test)),end-start ))    
    


##ensemble results2: {'l2_regularization': 0.3, 'learning_rate': 0.2, 'max_leaf_nodes': 51, 'min_samples_leaf': 10}
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

param_grid = {
    'l2_regularization': [0.3,0.9,1],
    'learning_rate': [0.1,0.2,0.3],
    'min_samples_leaf': [10,20],
    'max_leaf_nodes': [21,31,51]
}

model = hgbr =  HistGradientBoostingRegressor()

kfold = model_selection.KFold(n_splits=5)
start = time.time()
# Define the grid search we want to run. Run it with six cpus in parallel.
gs_cv = model_selection.GridSearchCV(model, param_grid, cv=kfold, n_jobs=6, verbose=100)

# Run the grid search - on only the training data
gs_cv.fit(X_train, y_train)
end = time.time()

# Print the parameters that gave us the best result
print(gs_cv.best_params_)
print("RMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,gs_cv.predict(X_test))), r2_score(y_test,gs_cv.predict(X_test)), mean_absolute_percentage_error(y_test,gs_cv.predict(X_test)),median_absolute_error(y_test,gs_cv.predict(X_test)),mean_absolute_error(y_test,gs_cv.predict(X_test)),end-start ))    



##ensemble results3: {'max_iter': 900, 'tol': 1e-11}
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

param_grid = {
    'max_iter': [50,100,200,400,800,900,1000],
    'tol': [1e-11,1e-12]
}

model = hgbr =  HistGradientBoostingRegressor(l2_regularization = 0.9, learning_rate = 0.3, min_samples_leaf = 20)

kfold = model_selection.KFold(n_splits=5)
start = time.time()
# Define the grid search we want to run. Run it with six cpus in parallel.
gs_cv = model_selection.GridSearchCV(model, param_grid, cv=kfold, n_jobs=6, verbose=100)

# Run the grid search - on only the training data
gs_cv.fit(X_train, y_train)
end = time.time()

# Print the parameters that gave us the best result
print(gs_cv.best_params_)
print("RMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,gs_cv.predict(X_test))), r2_score(y_test,gs_cv.predict(X_test)), mean_absolute_percentage_error(y_test,gs_cv.predict(X_test)),median_absolute_error(y_test,gs_cv.predict(X_test)),mean_absolute_error(y_test,gs_cv.predict(X_test)),end-start ))    
    


##ensemble results4: {'l2_regularization': 0, 'learning_rate': 0.3, 'max_iter': 800, 'min_samples_leaf': 20, 'tol': 1e-12}
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

param_grid = {
    'l2_regularization': [0,0.3,0.6,0.9],
    'learning_rate': [0,0.3,0.6,0.9],
    'min_samples_leaf': [20,40],
    'max_iter': [50,100,200,400,800,900,1000],
    'tol': [1e-11,1e-12]
}

model = hgbr =  HistGradientBoostingRegressor()

kfold = model_selection.KFold(n_splits=5)
start = time.time()
# Define the grid search we want to run. Run it with six cpus in parallel.
gs_cv = model_selection.GridSearchCV(model, param_grid, cv=kfold, n_jobs=6, verbose=100)

# Run the grid search - on only the training data
gs_cv.fit(X_train, y_train)
end = time.time()

# Print the parameters that gave us the best result
print(gs_cv.best_params_)
print("RMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,gs_cv.predict(X_test))), r2_score(y_test,gs_cv.predict(X_test)), mean_absolute_percentage_error(y_test,gs_cv.predict(X_test)),median_absolute_error(y_test,gs_cv.predict(X_test)),mean_absolute_error(y_test,gs_cv.predict(X_test)),end-start ))    
    


## Partial fitting (Defunc)
#getting data analysis and ML packages
import pandas
import gc
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #transform different types
from sklearn.datasets import fetch_openml
import numpy
from numpy import sqrt
from numpy import sum
from numpy import square
import seaborn
import matplotlib
import statsmodels
import time

import xgboost as xgb
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from mlxtend.regressor import StackingRegressor
import lightgbm as lgbm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import warnings

warnings.filterwarnings("ignore")

%load_ext cuml.accel
%load_ext cudf.pandas

!pip install ipympl

# Restart the kernel after installing for the changes to take effect.
# After restarting, you can run the cell again.


model = SGDRegressor(n_jobs=6)

for s,i in [(0,44640),(44640,87840),(87840,132480),(132480,175680),(175680,220320),(220320,264960),(264960,308160),(308160,352800),(352800,396000),(396000,440640),(440640,485280),(485280,527040)]:
    df_train = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
    df_train = df_train[s:i]
    df_train['year']   = df_train.index.year
    df_train['month']  = df_train.index.month
    df_train['day']    = df_train.index.day
    df_train['hour']   = df_train.index.hour
    df_train['minute'] = df_train.index.minute
    df_train['weekday'] = df_train.index.weekday
    df_train['date'] = df_train.index.date
    df_train['second_cont'] = (df_train.index - df_train.index[0]).total_seconds()
    df_train['minute_cont'] = (df_train.index - df_train.index[0]).total_seconds() / 60
    df_train['hour_cont'] = (df_train.index - df_train.index[0]).total_seconds() / 3600
    gc.collect()

    ## Dropping infs
    infcol = df_train.columns[np.isinf(df_train).any(axis=0)].tolist()
    df_train = df_train.drop(columns=infcol)
    gc.collect()
    ## converting index from datetime to numbers. Cannot use pipeline bc not applying to entire df, would need columnTransformer, too lazy
    df_train.index = range(0,len(df_train.index))

    
    X = df_train.loc[:, df_train.columns != 'label']
    y = df_train.loc[:, df_train.columns == 'label']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)
    
    scaler = StandardScaler().fit(X_train[['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890']])
    temp = scaler.transform(X_train[['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890']])
    temp = pandas.DataFrame(temp).rename(columns={0:'bid_qty_std',1:'ask_qty_std',2:'buy_qty_std',3:'sell_qty_std',4:'volume_std',5:'X17_std',6:'X25_std',7:'X198_std',8:'X205_std',9:'X413_std',10:'X415_std',11:'X421_std',12:'X594_std',13:'X596_std',14:'X606_std',15:'X610_std',16:'X663_std',17:'X667_std',18:'X675_std',19:'X681_std',20:'X687_std',21:'X693_std',22:'X865_std',23:'X875_std',24:'X876_std',25:'X879_std',26:'X880_std',27:'X881_std',28:'X882_std',29:'X883_std',30:'X890_std'})
    temp.index = X_train.index
    X_train = X_train.join(temp)
    
    scaler = StandardScaler().fit(X_test[['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890','year','month','day']])
    temp = scaler.transform(X_test[['bid_qty','ask_qty','buy_qty','sell_qty','volume','X17','X25','X198','X205','X413','X415','X421','X594','X596','X606','X610','X663','X667','X675','X681','X687','X693','X865','X875','X876','X879','X880','X881','X882','X883','X890','year','month','day']])
    temp = pandas.DataFrame(temp).rename(columns={0:'bid_qty_std',1:'ask_qty_std',2:'buy_qty_std',3:'sell_qty_std',4:'volume_std',5:'X17_std',6:'X25_std',7:'X198_std',8:'X205_std',9:'X413_std',10:'X415_std',11:'X421_std',12:'X594_std',13:'X596_std',14:'X606_std',15:'X610_std',16:'X663_std',17:'X667_std',18:'X675_std',19:'X681_std',20:'X687_std',21:'X693_std',22:'X865_std',23:'X875_std',24:'X876_std',25:'X879_std',26:'X880_std',27:'X881_std',28:'X882_std',29:'X883_std',30:'X890_std',31:'year_std',32:'month_std',33:'day_std'})
    temp.index = X_test.index
    X_test = X_test.join(temp)
    gc.collect()
    for epoch in range(1000):
        model.partial_fit(X_train, y_train)
        print('Epoch: %f RMSE: %.4f \nR2: %.4f \nMAE: %.4f' % (epoch, metrics.root_mean_squared_error(y_test,model.predict(X_test)), metrics.r2_score(y_test,model.predict(X_test)),metrics.mean_absolute_error(y_test,model.predict(X_test)) ) ) 


#Linear
#getting data analysis and ML packages
import pandas
import gc
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_absolute_percentage_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #transform different types
from sklearn.datasets import fetch_openml
import numpy
from numpy import sqrt
from numpy import sum
from numpy import square
import seaborn
import matplotlib
import statsmodels
import time
import numpy as np

import xgboost as xgb
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from mlxtend.regressor import StackingRegressor
import lightgbm as lgbm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import warnings

warnings.filterwarnings("ignore")


!pip install ipympl

# Restart the kernel after installing for the changes to take effect.
# After restarting, you can run the cell again.

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Ridge, Lasso, LinearRegression, ElasticNet

# feature derive from index
df_train = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
replace = pandas.read_csv('/kaggle/input/x1-x780/pls-762-to-5.csv')
df_train = df_train.drop(df_train.loc[:,'X1':'X780'].columns, axis=1)

df_train['year']   = df_train.index.year
df_train['month']  = df_train.index.month
df_train['day']    = df_train.index.day
df_train['hour']   = df_train.index.hour
df_train['minute'] = df_train.index.minute
df_train['weekday'] = df_train.index.weekday
df_train['second_cont'] = (df_train.index - df_train.index[0]).total_seconds()
df_train['minute_cont'] = (df_train.index - df_train.index[0]).total_seconds() / 60
df_train['hour_cont'] = (df_train.index - df_train.index[0]).total_seconds() / 3600
gc.collect()

# lag k
df_train['lag_1'] = df_train['label'].shift(periods=1)
df_train['lag_2'] = df_train['label'].shift(periods=2)
df_train['lag_3'] = df_train['label'].shift(periods=3)
df_train['lag_4'] = df_train['label'].shift(periods=4)
df_train['lag_5'] = df_train['label'].shift(periods=5)
df_train['lag_6'] = df_train['label'].shift(periods=6)
df_train['lag_7'] = df_train['label'].shift(periods=7)
df_train['lag_8'] = df_train['label'].shift(periods=8)
df_train['lag_9'] = df_train['label'].shift(periods=9)
df_train['lag_10'] = df_train['label'].shift(periods=10)
df_train['lag_11'] = df_train['label'].shift(periods=11)
df_train['lag_12'] = df_train['label'].shift(periods=12)
df_train['lag_13'] = df_train['label'].shift(periods=13)
df_train['lag_14'] = df_train['label'].shift(periods=14)
df_train['lag_15'] = df_train['label'].shift(periods=15)
df_train['lag_16'] = df_train['label'].shift(periods=16)
df_train.loc[:,'lag_1':'lag_16'] = df_train.loc[:,'lag_1':'lag_16'].replace({np.nan: df_train['label'].mean()})

## Dropping infs
#infcol = df_train.columns[np.isinf(df_train).any(axis=0)].tolist()
#df_train = df_train.drop(columns=infcol)
gc.collect()
## converting index from datetime to numbers. Cannot use pipeline bc not applying to entire df, would need columnTransformer, too lazy
df_train.index = range(0,len(df_train.index))

# join
df_train = df_train.join(replace)
df_train = df_train.rename(columns={'0':'PLS_1','1':'PLS_2','2':'PLS_3','3':'PLS_4','4':'PLS_5'})

# split
X = df_train.loc[:, df_train.columns != 'label']
y = df_train['label'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)

gc.collect()

rd = Pipeline([('std', StandardScaler()), ('rd',Ridge())])
par = Pipeline([('std', StandardScaler()), ('par',PassiveAggressiveRegressor())])
sgd = Pipeline([('std', StandardScaler()), ('sgd',SGDRegressor())])

start1 = time.time()
rd.fit(X_train, y_train)
end1 = time.time()

start2 = time.time()
par.fit(X_train, y_train)
end2 = time.time()

start3 = time.time()
sgd.fit(X_train, y_train)
end3 = time.time()

start4 = time.time()
vt = VotingRegressor(estimators=[('PAR', par), ('Ridge', rd), ('SGD', sgd)])
vt.fit(X_train, y_train)
end4 = time.time()

#start5 = time.time()
#br = BaggingRegressor(estimator=HistGradientBoostingRegressor(l2_regularization = 0.9, learning_rate = 0.3), n_estimators=5, max_samples=0.1, oob_score = True)
#br.fit(X_train, y_train)
#end5 = time.time()

print("RD \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,rd.predict(X_test))), r2_score(y_test,rd.predict(X_test)), mean_absolute_percentage_error(y_test,rd.predict(X_test)),median_absolute_error(y_test,rd.predict(X_test)),mean_absolute_error(y_test,rd.predict(X_test)),end1-start1 ))
print("PAR \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,par.predict(X_test))), r2_score(y_test,par.predict(X_test)), mean_absolute_percentage_error(y_test,par.predict(X_test)),median_absolute_error(y_test,par.predict(X_test)),mean_absolute_error(y_test,par.predict(X_test)),end2-start2 ))
print("SGD \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,sgd.predict(X_test))), r2_score(y_test,sgd.predict(X_test)), mean_absolute_percentage_error(y_test,sgd.predict(X_test)),median_absolute_error(y_test,sgd.predict(X_test)),mean_absolute_error(y_test,sgd.predict(X_test)),end3-start3 ))
print("VT \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,vt.predict(X_test))), r2_score(y_test,vt.predict(X_test)), mean_absolute_percentage_error(y_test,vt.predict(X_test)),median_absolute_error(y_test,vt.predict(X_test)),mean_absolute_error(y_test,vt.predict(X_test)),end4-start4 ))



pandas.DataFrame({'y-true':y_test,
                  'rd-pred':rd.predict(X_test),
                  'par-pred':par.predict(X_test),
                  'sgd-pred':sgd.predict(X_test),
                  'vt-pred':vt.predict(X_test)
    
}).corr()


# ensemble
#getting data analysis and ML packages
import pandas
import gc
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_absolute_percentage_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #transform different types
from sklearn.datasets import fetch_openml
import numpy
from numpy import sqrt
from numpy import sum
from numpy import square
import seaborn
import matplotlib
import statsmodels
import time
import numpy as np

import xgboost as xgb
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from mlxtend.regressor import StackingRegressor
import lightgbm as lgbm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import warnings

warnings.filterwarnings("ignore")


!pip install ipympl

# Restart the kernel after installing for the changes to take effect.
# After restarting, you can run the cell again.

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Ridge, Lasso, LinearRegression, ElasticNet

# feature derive from index
df_train = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
replace = pandas.read_csv('/kaggle/input/x1-x780/pls-762-to-5.csv')
df_train = df_train.drop(df_train.loc[:,'X1':'X780'].columns, axis=1)

#df_train['year']   = df_train.index.year
#df_train['month']  = df_train.index.month
#df_train['day']    = df_train.index.day
#df_train['hour']   = df_train.index.hour
#df_train['minute'] = df_train.index.minute
#df_train['weekday'] = df_train.index.weekday
#df_train['second_cont'] = (df_train.index - df_train.index[0]).total_seconds()
#df_train['minute_cont'] = (df_train.index - df_train.index[0]).total_seconds() / 60
#df_train['hour_cont'] = (df_train.index - df_train.index[0]).total_seconds() / 3600
gc.collect()

# lag k
df_train['lag_1'] = df_train['label'].shift(periods=1)
df_train['lag_2'] = df_train['label'].shift(periods=2)
df_train['lag_3'] = df_train['label'].shift(periods=3)
df_train['lag_4'] = df_train['label'].shift(periods=4)
df_train['lag_5'] = df_train['label'].shift(periods=5)
df_train['lag_6'] = df_train['label'].shift(periods=6)
df_train['lag_7'] = df_train['label'].shift(periods=7)
df_train['lag_8'] = df_train['label'].shift(periods=8)
df_train['lag_9'] = df_train['label'].shift(periods=9)
df_train['lag_10'] = df_train['label'].shift(periods=10)
df_train['lag_11'] = df_train['label'].shift(periods=11)
df_train['lag_12'] = df_train['label'].shift(periods=12)
df_train['lag_13'] = df_train['label'].shift(periods=13)
df_train['lag_14'] = df_train['label'].shift(periods=14)
df_train['lag_15'] = df_train['label'].shift(periods=15)
df_train['lag_16'] = df_train['label'].shift(periods=16)
df_train.loc[:,'lag_1':'lag_16'] = df_train.loc[:,'lag_1':'lag_16'].replace({np.nan: df_train['label'].mean()})

## Dropping infs
#infcol = df_train.columns[np.isinf(df_train).any(axis=0)].tolist()
#df_train = df_train.drop(columns=infcol)
gc.collect()
## converting index from datetime to numbers. Cannot use pipeline bc not applying to entire df, would need columnTransformer, too lazy
df_train.index = range(0,len(df_train.index))

# join
df_train = df_train.join(replace)
df_train = df_train.rename(columns={'0':'PLS_1','1':'PLS_2','2':'PLS_3','3':'PLS_4','4':'PLS_5'})

# split
X = df_train.loc[:, df_train.columns != 'label']
y = df_train['label'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)

gc.collect()

start1 = time.time()
v1 = HistGradientBoostingRegressor(l2_regularization= 0.6, learning_rate= 0.3, max_iter= 200, min_samples_leaf= 20, tol= 1e-12)
v1.fit(X_train, y_train)
end1 = time.time()

start2 = time.time()
v2 = HistGradientBoostingRegressor(l2_regularization = 0.9, learning_rate = 0.3)
v2.fit(X_train, y_train)
end2 = time.time()

start3 = time.time()
v3 = HistGradientBoostingRegressor(l2_regularization= 0.6, learning_rate= 0.3, max_iter= 200, min_samples_leaf= 20, tol= 1e-12)
v3.fit(X_train, y_train)
end3 = time.time()

start4 = time.time()
rd = Pipeline([('std', StandardScaler()), ('lr',Ridge())])
hgbr =  HistGradientBoostingRegressor(l2_regularization= 0.6, learning_rate= 0.3, max_iter= 200, min_samples_leaf= 20, tol= 1e-12)
vt = VotingRegressor(estimators=[('rd', rd),('histo', hgbr)])
vt.fit(X_train, y_train)
end4 = time.time()

#start5 = time.time()
#br = BaggingRegressor(estimator=HistGradientBoostingRegressor(l2_regularization = 0.9, learning_rate = 0.3), n_estimators=5, max_samples=0.1, oob_score = True)
#br.fit(X_train, y_train)
#end5 = time.time()

print("V1 \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,v1.predict(X_test))), r2_score(y_test,v1.predict(X_test)), mean_absolute_percentage_error(y_test,v1.predict(X_test)),median_absolute_error(y_test,v1.predict(X_test)),mean_absolute_error(y_test,v1.predict(X_test)),end1-start1 ))
print("V2 \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,v2.predict(X_test))), r2_score(y_test,v2.predict(X_test)), mean_absolute_percentage_error(y_test,v2.predict(X_test)),median_absolute_error(y_test,v2.predict(X_test)),mean_absolute_error(y_test,v2.predict(X_test)),end2-start2 ))
print("V3 \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,v3.predict(X_test))), r2_score(y_test,v3.predict(X_test)), mean_absolute_percentage_error(y_test,v3.predict(X_test)),median_absolute_error(y_test,v3.predict(X_test)),mean_absolute_error(y_test,v3.predict(X_test)),end3-start3 ))
print("VT \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,vt.predict(X_test))), r2_score(y_test,vt.predict(X_test)), mean_absolute_percentage_error(y_test,vt.predict(X_test)),median_absolute_error(y_test,vt.predict(X_test)),mean_absolute_error(y_test,vt.predict(X_test)),end4-start4 ))
#print("BR \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,br.predict(X_test))), r2_score(y_test,br.predict(X_test)), mean_absolute_percentage_error(y_test,br.predict(X_test)),median_absolute_error(y_test,br.predict(X_test)),mean_absolute_error(y_test,br.predict(X_test)),end5-start5 ))


pandas.DataFrame({'y-true':y_test,
                  'v1-pred':v1.predict(X_test),
                  'v2-pred':v2.predict(X_test),
                  'v3-pred':v3.predict(X_test),
                  'vt-pred':vt.predict(X_test)
    
}).corr()


#ensemble for loop
import pandas
import gc
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_absolute_percentage_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #transform different types
from sklearn.datasets import fetch_openml
import numpy
from numpy import sqrt
from numpy import sum
from numpy import square
import seaborn
import matplotlib
import statsmodels
import time
import numpy as np

import xgboost as xgb
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from mlxtend.regressor import StackingRegressor
import lightgbm as lgbm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Lasso, LinearRegression, ElasticNet, Ridge
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn import metrics
import gc

for i in range(100):
    start1 = time.time()
    v1 = HistGradientBoostingRegressor(l2_regularization= 0.6, learning_rate= 0.3, max_iter= 200, min_samples_leaf= 20, tol= 1e-12)
    v1.fit(X_train, y_train)
    end1 = time.time()
    
    start2 = time.time()
    v2 = HistGradientBoostingRegressor(l2_regularization= 0.6, learning_rate= 0.3, max_iter= 200, min_samples_leaf= 20, tol= 1e-12)
    v2.fit(X_train, y_train)
    end2 = time.time()
    
    start3 = time.time()
    v3 = HistGradientBoostingRegressor(l2_regularization= 0, learning_rate= 0.3, max_iter= 800, min_samples_leaf= 20, tol = 1e-12)
    v3.fit(X_train, y_train)
    end3 = time.time()
    
    start4 = time.time()
    rd = Pipeline([('std', StandardScaler()), ('lr',Ridge())])
    hgbr =  HistGradientBoostingRegressor(l2_regularization= 0.6, learning_rate= 0.3, max_iter= 200, min_samples_leaf= 20, tol= 1e-12)
    vt = VotingRegressor(estimators=[('rd', rd),('histo', hgbr)])
    vt.fit(X_train, y_train)
    end4 = time.time()

    print("V1 \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,v1.predict(X_test))), r2_score(y_test,v1.predict(X_test)), mean_absolute_percentage_error(y_test,v1.predict(X_test)),median_absolute_error(y_test,v1.predict(X_test)),mean_absolute_error(y_test,v1.predict(X_test)),end1-start1 ))
    print("V2 \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,v2.predict(X_test))), r2_score(y_test,v2.predict(X_test)), mean_absolute_percentage_error(y_test,v2.predict(X_test)),median_absolute_error(y_test,v2.predict(X_test)),mean_absolute_error(y_test,v2.predict(X_test)),end2-start2 ))
    print("V3 \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,v3.predict(X_test))), r2_score(y_test,v3.predict(X_test)), mean_absolute_percentage_error(y_test,v3.predict(X_test)),median_absolute_error(y_test,v3.predict(X_test)),mean_absolute_error(y_test,v3.predict(X_test)),end3-start3 ))
    print("VT \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,vt.predict(X_test))), r2_score(y_test,vt.predict(X_test)), mean_absolute_percentage_error(y_test,vt.predict(X_test)),median_absolute_error(y_test,vt.predict(X_test)),mean_absolute_error(y_test,vt.predict(X_test)),end4-start4 ))

    ok = input("If Break, then Yes")
    if ok == 'Yes':
        break


## pickling
import joblib
joblib.dump(v2, '/kaggle/working/best-histo.pkl')
joblib.dump(v3, '/kaggle/working/2nd-histo.pkl')
joblib.dump(vt, '/kaggle/working/vt-rd-histo.pkl')


## loading
temp = joblib.load('/kaggle/working/2nd-histo.pkl')
print(temp.predict(X_train.iloc[0:1,:]) )

temp = joblib.load('/kaggle/working/vt-rd-histo.pkl')
print(temp.predict(X_train.iloc[0:1,:]) )

temp = joblib.load('/kaggle/working/best-histo.pkl')
print(temp.predict(X_train.iloc[0:1,:]) )


## checking
print(v3.predict(X_train.iloc[0:1,:]), vt.predict(X_train.iloc[0:1,:]), v2.predict(X_train.iloc[0:1,:]) )


import pandas
import gc
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_absolute_percentage_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #transform different types
from sklearn.datasets import fetch_openml
import numpy
from numpy import sqrt
from numpy import sum
from numpy import square
import seaborn
import matplotlib
import statsmodels
import time
import numpy as np

import xgboost as xgb
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from mlxtend.regressor import StackingRegressor
import lightgbm as lgbm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Lasso, LinearRegression, ElasticNet, Ridge
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn import metrics
import gc

##linear for loop
rd = Pipeline([('std', StandardScaler()), ('rd',Ridge())])
par = Pipeline([('std', StandardScaler()), ('par',PassiveAggressiveRegressor(C= 2,epsilon= 0.2,loss= 'squared_epsilon_insensitive'))])
sgd = Pipeline([('std', StandardScaler()), ('sgd',SGDRegressor())])

for i in range(100):
    start1 = time.time()
    rd.fit(X_train, y_train)
    end1 = time.time()
    
    start2 = time.time()
    par.fit(X_train, y_train)
    end2 = time.time()
    
    start3 = time.time()
    sgd.fit(X_train, y_train)
    end3 = time.time()
    
    start4 = time.time()
    vt = VotingRegressor(estimators=[('PAR', par), ('Ridge', rd), ('SGD', sgd)])
    vt.fit(X_train, y_train)
    end4 = time.time()
    
    #start5 = time.time()
    #br = BaggingRegressor(estimator=HistGradientBoostingRegressor(l2_regularization = 0.9, learning_rate = 0.3), n_estimators=5, max_samples=0.1, oob_score = True)
    #br.fit(X_train, y_train)
    #end5 = time.time()
    
    print("RD \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,rd.predict(X_test))), r2_score(y_test,v1.predict(X_test)), mean_absolute_percentage_error(y_test,rd.predict(X_test)),median_absolute_error(y_test,rd.predict(X_test)),mean_absolute_error(y_test,rd.predict(X_test)),end1-start1 ))
    print("PAR \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,par.predict(X_test))), r2_score(y_test,v2.predict(X_test)), mean_absolute_percentage_error(y_test,par.predict(X_test)),median_absolute_error(y_test,par.predict(X_test)),mean_absolute_error(y_test,par.predict(X_test)),end2-start2 ))
    print("SGD \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,sgd.predict(X_test))), r2_score(y_test,v3.predict(X_test)), mean_absolute_percentage_error(y_test,sgd.predict(X_test)),median_absolute_error(y_test,sgd.predict(X_test)),mean_absolute_error(y_test,sgd.predict(X_test)),end3-start3 ))
    print("VT \nRMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f \n~time: %.4f" % (np.sqrt(mean_squared_error(y_test,vt.predict(X_test))), r2_score(y_test,vt.predict(X_test)), mean_absolute_percentage_error(y_test,vt.predict(X_test)),median_absolute_error(y_test,vt.predict(X_test)),mean_absolute_error(y_test,vt.predict(X_test)),end4-start4 ))

    pandas.DataFrame({'y-true':y_test,
                  'v1-pred':v1.predict(X_test),
                  'v2-pred':v2.predict(X_test),
                  'v3-pred':v3.predict(X_test),
                  'vt-pred':vt.predict(X_test)
    
    }).corr()
    
    ok = input("If Break, then Yes")
    if ok == 'Yes':
        break


import joblib

joblib.dump(sgd, '/kaggle/working/sgd.pkl')
joblib.dump(rd, '/kaggle/working/rd.pkl')


## loading
temp = joblib.load('/kaggle/working/sgd.pkl')
print(temp.predict(X_train.iloc[0:1,:]) )

temp = joblib.load('/kaggle/working/rd.pkl')
print(temp.predict(X_train.iloc[0:1,:]) )


## checking
print(sgd.predict(X_train.iloc[0:1,:]), rd.predict(X_train.iloc[0:1,:]) )


X = df_train.loc[:, df_train.columns != 'label']
y = df_train['label']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)


import joblib
rd = joblib.load("/kaggle/input/all-models/scikitlearn/default/1/rd.pkl")
sgd = joblib.load("/kaggle/input/all-models/scikitlearn/default/1/sgd.pkl")
vt = joblib.load("/kaggle/input/all-models/scikitlearn/default/1/vt-rd-histo.pkl")
best = joblib.load("/kaggle/input/all-models/scikitlearn/default/1/best-histo.pkl")
best2 = joblib.load("/kaggle/input/all-models/scikitlearn/default/1/2nd-histo.pkl")

for name, i in [('ridge',rd), ('SGD',sgd), ('Voter',vt), ('Histo',best), ('2nd Histo',best2)]:
    with plt.style.context(('seaborn-whitegrid')):
        plt.scatter(X_train[['volume']], y_train, c='darkred')
        plt.scatter(X_train[['volume']], i.predict(X_train), c='darkgreen')
        plt.title(f"{name} volume vs label")
        plt.xlabel('volume')
        plt.ylabel('label')
    plt.show()

combined = (rd.predict(X_train) + sgd.predict(X_train) + vt.predict(X_train) + best.predict(X_train) + best2.predict(X_train))/5
with plt.style.context(('seaborn-whitegrid')):
        plt.scatter(X_train[['volume']], y_train, c='darkred')
        plt.scatter(X_train[['volume']], combined, c='darkgreen')
        plt.title("combined volume vs label")
        plt.xlabel('volume')
        plt.ylabel('label')
plt.show()


rr = pandas.DataFrame(rd.predict(df_train.iloc[:,df_train.columns != 'label']) )
rsgd = pandas.DataFrame(sgd.predict(df_train.iloc[:,df_train.columns != 'label']), index = rr.index )
rvt = pandas.DataFrame(vt.predict(df_train.iloc[:,df_train.columns != 'label']) , index = rr.index )
rbest = pandas.DataFrame(best.predict(df_train.iloc[:,df_train.columns != 'label']) , index = rr.index )
r2best = pandas.DataFrame(best2.predict(df_train.iloc[:,df_train.columns != 'label']) , index = rr.index )


rr = rd.predict(X_train.iloc[0:1,:])[0]
rsgd = sgd.predict(X_train.iloc[0:1,:])[0]
rvt = vt.predict(X_train.iloc[0:1,:])[0]
rbest = best.predict(X_train.iloc[0:1,:])[0]
r2best = best2.predict(X_train.iloc[0:1,:])[0]
combined =  (rr + rsgd + rvt + rbest + r2best) / 5
temp = pandas.DataFrame({'y':[rr, rsgd, rvt, rbest, r2best, combined, 0.188167]})
temp.plot(kind='barh')


import joblib
hist = joblib.load('/kaggle/input/model-new/scikitlearn/default/1/2nd-histo.pkl')
pandas.DataFrame(X.iloc[0,:]).T
hist.predict(pandas.DataFrame(X.iloc[0,:]).T)
hist.predict(pandas.DataFrame(X.iloc[0,:]).T)[0]


predictions = pandas.DataFrame([0,0,0,0],columns=['Predictions'])
predictions = pandas.DataFrame({'Predictions':[0,0,0,0]})
predictions


df_test = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')


df_test['lag_1'] = df_test['label'].shift(periods=1)
df_test['lag_2'] = df_test['label'].shift(periods=2)
df_test['lag_3'] = df_test['label'].shift(periods=3)
df_test['lag_4'] = df_test['label'].shift(periods=4)
df_test['lag_5'] = df_test['label'].shift(periods=5)
df_test['lag_6'] = df_test['label'].shift(periods=6)
df_test['lag_7'] = df_test['label'].shift(periods=7)
df_test['lag_8'] = df_test['label'].shift(periods=8)
df_test['lag_9'] = df_test['label'].shift(periods=9)
df_test['lag_10'] = df_test['label'].shift(periods=10)
df_test['lag_11'] = df_test['label'].shift(periods=11)
df_test['lag_12'] = df_test['label'].shift(periods=12)
df_test['lag_13'] = df_test['label'].shift(periods=13)
df_test['lag_14'] = df_test['label'].shift(periods=14)
df_test['lag_15'] = df_test['label'].shift(periods=15)
df_test['lag_16'] = df_test['label'].shift(periods=16)
df_test.loc[:,'lag_1':'lag_16'] = df_test.loc[:,'lag_1':'lag_16'].replace({np.nan: 0.03612583479917425})
## Dropping infs
infcol = df_test.columns[np.isinf(df_test).any(axis=0)].tolist()
print(infcol)

df_test = df_test.drop(columns=infcol)
## converting index from datetime to numbers. Cannot use pipeline bc not applying to entire df, would need columnTransformer, too lazy
df_test.index = range(0,len(df_test.index))

import joblib
pls = joblib.load('/kaggle/input/all-models/scikitlearn/default/1/pls.pkl')
replace = pls.transform(df_test.loc[:, 'X1':'X780'])
replace = pandas.DataFrame(replace)
df_test = df_test.drop(df_test.loc[:,'X1':'X780'].columns, axis=1)
df_test = df_test.join(replace)
df_test = df_test.rename(columns={0:'PLS_1',1:'PLS_2',2:'PLS_3',3:'PLS_4',4:'PLS_5'})

X = df_test.loc[:, df_test.columns != 'label']
y = df_test['label'].values

hist = joblib.load('/kaggle/input/model-new/scikitlearn/default/1/2nd-histo.pkl')
histb = joblib.load('/kaggle/input/model-new/scikitlearn/default/1/best-histo.pkl')
rd = joblib.load('/kaggle/input/model-new/scikitlearn/default/1/rd.pkl')
vt = joblib.load('/kaggle/input/model-new/scikitlearn/default/1/vt-rd-histo.pkl')


predictions = []
for i in range(len(X)):
    predictions.append(hist.predict(pandas.DataFrame(X.iloc[i,:]).T)[0])
    X.loc[i+1,'lag_1':'lag_16'] = X.loc[i,'lag_1':'lag_16'].replace({0: hist.predict(pandas.DataFrame(X.iloc[i,:]).T)[0]})
submission = pandas.DataFrame(predictions, columns=['Predictions-histo2'])



pandas.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')





from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class ExpSmoothingWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, trend=None, seasonal=None, seasonal_periods=None):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

    def fit(self, X, y):
        self.model_ = ExponentialSmoothing(
            y, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods
        ).fit()
        return self

    def predict(self, X):
        n_periods = len(X)
        last_idx = self.model_.fittedvalues.shape[0] - 1
        forecast = self.model_.predict(start=last_idx + 1, end=last_idx + n_periods)
        return forecast


exp = ExpSmoothingWrapper(trend='add', seasonal='add', seasonal_periods=23903)


# feature derive from index
df_train = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
df_test = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')
df_test['label'] = range(0,538150,1)
df_test.index = pandas.date_range(start = "2024/03/01", freq= 'min', periods = 538150)
df_train = df_train.drop(df_train.loc[:,df_train.columns != 'label'], axis=1)
df_test = df_test.drop(df_test.loc[:,df_test.columns != 'label'], axis=1)
## split
train, test = df_train.iloc[:484149], df_train.iloc[484149:]


#check
df_test


#check
df_train


# defunc
# when .loc dataframe and single column, will make dataframe into matrix, cannot .loc anymore.
# if .loc[:,'label'].values will make 2D dataframe (n,1) into 2D matrix (n,1) into 1D array (n,) , no column name or index
y = df_train.loc[:,'label']
y.loc[:,'label']


exp.fit(train.index,train)


test['pred'] = exp.predict(test).values
fig, ax = plt.subplots()
plt.plot(test.index, test['pred'])
plt.plot(test.index, test['label'])
plt.title("time series")
plt.tight_layout()



import joblib

joblib.dump(exp, '/kaggle/working/exp.pkl')


# feature derive from index
df_test = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')
df_test['label'] = range(0,538150,1)
df_test.index = pandas.date_range(start = "2024/03/01", freq= 'min', periods = 538150)
df_test = df_test.drop(df_test.loc[:,df_test.columns != 'label'], axis=1)
#df_test.to_parquet('/kaggle/working/df_test.parquet')
df_train = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
df_train = df_train.drop(df_train.loc[:,df_train.columns != 'label'], axis=1)
## split
df_train = df_train.iloc[0:114912,:]
train, test = df_train.iloc[0:71712,:], df_train.iloc[71712:]
#train, test = df_train.iloc[:484149], df_train.iloc[484149:]
#df_train.to_parquet('/kaggle/working/df_train.parquet')


## defunc
import pyarrow as pa
import pyarrow.parquet as pq
import statsmodels.formula.api as smf

class DataSet(dict):
    def __init__(self, path):
        self.parquet = pq.ParquetFile(path)

    def __getitem__(self, key):
        try:
            return self.parquet.read([key]).to_pandas()[key]
        except:
            raise KeyError

df_train = DataSet('/kaggle/input/removed-drw/df_train.parquet')
df_test = DataSet('/kaggle/input/removed-drw/df_test.parquet')


from statsmodels.tsa.holtwinters import ExponentialSmoothing

exp = statsmodels.tsa.holtwinters.ExponentialSmoothing(endog=train, trend='add', seasonal='add', seasonal_periods=23904).fit(optimized=True)


exp.forecast(steps=30)


exp.save('/kaggle/working/exp.pkl')


fexp.predict(params, start='2024-02-01 00:00:00', end='2024-02-29 23:59:00')

