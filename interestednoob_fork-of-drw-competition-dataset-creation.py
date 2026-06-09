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


df_train = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
#df_test = pandas.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')
#df_sample = pandas.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')


df_train = df_train.drop(['bid_qty'	,'ask_qty'	,'buy_qty'	,'sell_qty','volume'], axis = 1)


df_train


#df_train = df_train[0:44640]


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



## converting index from datetime to numbers. Cannot use pipeline bc not applying to entire df, would need columnTransformer, too lazy
df_train.index = range(0,len(df_train.index))


df_train['label'].mean()


df_train['lag_1'] = df_train['label'].shift(periods=1)
df_train['lag_2'] = df_train['label'].shift(periods=2)
df_train['lag_3'] = df_train['label'].shift(periods=3)
df_train['lag_4'] = df_train['label'].shift(periods=4)
df_train['lag_5'] = df_train['label'].shift(periods=5)
df_train.loc[:,'lag_1':'lag_5'] = df_train.loc[:,'lag_1':'lag_5'].replace({np.nan: df_train['label'].mean()})


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


df_train


## Dropping infs
infcol = df_train.columns[np.isinf(df_train).any(axis=0)].tolist()
print(infcol)

df_train = df_train.drop(columns=infcol)


pca = PCA(n_components=10)
pca.fit(df_train.loc[0:132480,'X1':'X780'])
print(pca.explained_variance_ratio_)
print(pca.components_)
print(pca.singular_values_)

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Compute communalities (sum of squared loadings for each variable)
communalities = np.sum(loadings**2, axis=1)

print("Communalities:", communalities)
print("loadings:", loadings)


pca = PCA(n_components=24)
pca.fit(df_train.loc[0:132480,'X1':'X780'])
print(pca.explained_variance_ratio_.sum())


## Inter-factor correl
temp = pandas.DataFrame(pca.transform(df_train.loc[0:132480,'X1':'X780']))
seaborn.heatmap(temp.corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


seaborn.heatmap(pandas.DataFrame(temp.corrwith(df_train.loc[0:132480,'label'])), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


temp.corrwith(df_train.loc[0:132480,'label']).sort_values(ascending=True)


from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(rotation='varimax')
fit = fa.fit(df_train.loc[0:132480,'X1':'X780'])
fa.components_


loadings = fa.components_.T  # transpose to get features x factors

# Variance explained per factor (sum of squared loadings per factor)
explained_var_per_factor = np.sum(loadings**2, axis=0)

# Total variance explained by all factors
total_explained_variance = np.sum(explained_var_per_factor)


print("Explained variance per factor:", explained_var_per_factor)
print("Total explained variance:", total_explained_variance)


proportion_explained = explained_var_per_factor/total_explained_variance
var = pandas.DataFrame(proportion_explained)
var.loc[0:52,0].sum()


pandas.DataFrame(fa.transform(df_train.loc[0:132480,'X1':'X780'])).loc[:,0:24]


## Inter-factor correl
temp = pandas.DataFrame(fa.transform(df_train.loc[0:132480,'X1':'X780'])).loc[:,0:52].corr()
seaborn.heatmap(temp, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


seaborn.heatmap(pandas.DataFrame(temp.corrwith(df_train.loc[0:132480,'label'])), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


temp.corrwith(df_train.loc[0:132480,'label']).sort_values(ascending=True)


from sklearn.cross_decomposition import PLSRegression, PLSSVD

X = df_train.loc[0:132480,'X1':'X780']
y = df_train.loc[0:132480,'label']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#pls = PLSRegression(n_components=300)
#pls.fit(X_train, y_train)
plsvd = PLSSVD(n_components=1)
plsvd.fit(X_train, y_train)


# SVD correl
temp = plsvd.transform(X_train)
temp = pandas.DataFrame(temp)
temp.index = y_train.index
temp.join(y_train).corr()


from sklearn.cross_decomposition import PLSRegression, PLSSVD

X = df_train.loc[:,'X1':'X780']
y = df_train.loc[:,'label']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

pls = PLSRegression(n_components=5)
pls.fit(X_train, y_train)


r2_score(y_test, pls.predict(X_test))


r2_score(df_train.loc[1250:2250,'label'], pls.predict(df_train.loc[1250:2250,'X1':'X780']))


# Variance explained in y per component
explained_var_y = np.var(pls.y_scores_, axis=0, ddof=0)
total_var_y = np.var(y_train, ddof=0)
explained_ratio_y = explained_var_y / total_var_y
cumulative_var = np.cumsum(explained_ratio_y)

# Plot scree (elbow) plot
plt.figure(figsize=(30, 30))
plt.plot(range(1, 276), cumulative_var, marker='v')
#plt.axhline(0.7, color='r', linestyle='--', label='70% threshold')
plt.xlabel('Number of PLS Components')
plt.ylabel('Cumulative Variance Explained in y')
plt.title('PLS Scree Plot (Elbow Method)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/kaggle/working/PLS-scree.png", dpi=300, bbox_inches='tight')
plt.show()


pls = PLSRegression(n_components=762)
pls.fit(X_train, y_train)
explained_var_y = np.var(pls.y_scores_, axis=0, ddof=0)
total_var_y = np.var(y_train, ddof=0)
explained_ratio_y = explained_var_y / total_var_y
print(explained_ratio_y.sum())


# Evaluate the model performance
import shap

# Assuming 'pls_model' is your trained PLS regression model
# Assuming 'X_train' is your training set features
# Assuming 'X_test' is your test set features

# Create a SHAP explainer for the PLS regression model using KernelExplainer
explainer = shap.KernelExplainer(pls.predict, X_train)

# Calculate SHAP values for the entire test set
shap_values = explainer.shap_values(X_test)

# Summary plot for all instances
shap.summary_plot(shap_values, X_test)


## Inter-factor correl
temp = pandas.DataFrame(pls.transform(X_train)).iloc[:,0:201].corr()
seaborn.heatmap(temp, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Inter-latent variable Correlation Heatmap 1m')
plt.savefig("/kaggle/working/inter-factor-pls.png")
plt.show()


## Factor-target correl
seaborn.heatmap(pandas.DataFrame(temp.corrwith(y_train)), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Latent variables-target Correlation Heatmap 1m')
plt.savefig("/kaggle/working/target-factor-pls.png")
plt.show()


pandas.DataFrame(temp.corrwith(y_train)).sort_values(0,ascending=False)


from sklearn.decomposition import KernelPCA

X = df_train.loc[0:44640,'X1':'X780']
y = df_train.loc[0:44640,'label']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

Kpca = KernelPCA(n_components=5)
Kpca.fit(X_train)


## Inter-factor correl
temp = pandas.DataFrame(Kpca.transform(X_train)).iloc[:,0:5].corr()
seaborn.heatmap(temp, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Inter-Factor KPCA Correlation Heatmap')
plt.savefig("/kaggle/working/inter-factor-kpca.png")
plt.show()


## Factor-target correl
seaborn.heatmap(pandas.DataFrame(temp.corrwith(y_train)), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Factor-target Correlation Heatmap')
plt.savefig("/kaggle/working/target-factor-kpca.png")
plt.show()


pandas.DataFrame(temp.corrwith(y_train)).sort_values(0,ascending=False)


explained_var_ratio = Kpca.eigenvalues_ / np.sum(Kpca.eigenvalues_)
print(explained_var_ratio)


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


## converting index from datetime to numbers. Cannot use pipeline bc not applying to entire df, would need columnTransformer, too lazy
df_train.index = range(0,len(df_train.index))


X = df_train.loc[:, df_train.columns != 'label']
y = df_train['label'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)


from statsmodels.tsa.seasonal import seasonal_decompose

# assuming `volume_series` is a pandas Series with a datetime index
decomp = seasonal_decompose(df_train['label'], model='add', period=87648)  # or 7 if daily


decomp.plot()
plt.savefig('/kaggle/working/season-87648.png')



from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df_train['label'], lags=86400)

plt.savefig('/kaggle/working/ACF-86400-lag.png')


from statsmodels.tsa.seasonal import STL

stl = STL(df_train['label'], period=175000)
result = stl.fit()
result.plot()

plt.savefig('/kaggle/working/STL-175000.png')


from scipy.signal import periodogram

freqs, power = periodogram(df_train.loc[:,'label'].dropna())
plt.figure(figsize=(30, 10))
plt.semilogy(1 / freqs, power)
plt.xlabel("Period (in time steps)")
plt.ylabel("Spectral Power")
plt.title("Spectral Analysis: Suggesting Seasonal Periods")

plt.savefig('/kaggle/working/spectral.png')


from scipy.signal import find_peaks

# Ignore first few low periods
period = 1 / freqs
valid_range = (period > 2) & (period < len(df_train.loc[:,'label'].dropna()) // 2)
peaks, _ = find_peaks(power[valid_range], height=np.percentile(power[valid_range], 95))

suggested_periods = period[valid_range][peaks]
print("ðŸ“Œ Suggested seasonal periods (by spectral peak):", suggested_periods[:5])


hexp = Pipeline([('std', StandardScaler()), ('exp', ExpSmoothingWrapper(trend='add', seasonal='add', seasonal_periods=3))])


from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn import metrics

sgd = Pipeline([('std', StandardScaler()), ('sgd',SGDRegressor())])
par = Pipeline([('std', StandardScaler()), ('par',PassiveAggressiveRegressor())])
#etr = Pipeline([('std', StandardScaler()), ('etr',ExtraTreesRegressor())])
hgbr =  Pipeline([('std', StandardScaler()), ('hgbr',HistGradientBoostingRegressor())])
rfr = Pipeline([('std', StandardScaler()), ('rfr',RandomForestRegressor())])
lsr = Pipeline([('std', StandardScaler()), ('lsr',Lasso())])
lr = Pipeline([('std', StandardScaler()), ('lr',LinearRegression())])
elst = Pipeline([('std', StandardScaler()), ('elst',ElasticNet())])


models = [('SGD Reg', sgd), ('PA Reg', par),('Histogram based GB', hgbr), ('Random Forest', rfr), ('Lasso', lsr), ('Linear', lr), ('ElasticNet', elst)]


for model_name, model in models:
    ave_r2, ave_mae, ave_mape, ave_rmse, ave_mdae = [],[],[],[],[]
    for i in range(0,1):
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
    msg1 = "%s: mean R2: %f \nMAE: %f \nMAPE: %f \nRMSE: %f \nMDAE: %f" % (model_name, statistics.mean(ave_r2), statistics.mean(ave_mae), statistics.mean(ave_mape), statistics.mean(ave_rmse), statistics.mean(ave_mdae))
    print(msg1, "\n Time ~", duration)


from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Perceptron, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn import metrics

lsr = Lasso()
lr = LinearRegression()
elst = ElasticNet()
svr_rbf = SVR(kernel='rbf')
svr_lin = SVR(kernel='linear')

kfold = model_selection.KFold(n_splits=5) 

stk = Pipeline([('std',StandardScaler()),('stacker',StackingRegressor(estimators=[('Lasso', lsr), ('Linear', lr), ('ElasticNet', elst)], final_estimator=svr_rbf, cv=kfold))])
vt = Pipeline([('std',StandardScaler()),('voter',VotingRegressor(estimators=[('Lasso', lsr), ('Linear', lr), ('ElasticNet', elst)], cv=kfold))])


# Training the stacking classifier
for stack in [stk,stk2]:
    stack.fit(X_train, y_train)
    
    # Evaluate and visualize the fit
    
    print("RMSE: %.4f \nR2: %.4f \nMAPE: %.4f \nMDAE: %.4f \nMAE: %.4f" % (metrics.root_mean_squared_error(y_test,stack.predict(X_test)), metrics.r2_score(y_test,stack.predict(X_test)), metrics.mean_absolute_percentage_error(y_test,stack.predict(X_test)),metrics.median_absolute_error(y_test,stack.predict(X_test)),metrics.mean_absolute_error(y_test,stack.predict(X_test)) ))
    
    with plt.style.context(('seaborn-whitegrid')):
        plt.scatter(X_train, y_train, c='lightgray')
        plt.plot(X_train, stack.predict(X_train), c='darkgreen', lw=2)
    
    plt.show()


exp = ExpSmoothingWrapper(trend='add', seasonal='add', seasonal_periods=12)



from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Perceptron, Lasso, LinearRegression, ElasticNet

param_grid = {
    'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet']
}

model = SGDRegressor(n_jobs=6)

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


from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Perceptron, Lasso, LinearRegression, ElasticNet

param_grid = {
    'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'],
    'epsilon': [0.1, 0.2, 0.5],
    'C': [0.2,1,2]
}

model = PassiveAggressiveRegressor(n_jobs=6, tol=1e-4)

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




