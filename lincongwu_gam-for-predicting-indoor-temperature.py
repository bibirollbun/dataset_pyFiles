# Import data handling & numerical libraries
import pandas as pd
import numpy as np
from copy import copy
import datetime

# Import Data Visualization libraries
import seaborn as sb
import matplotlib.pyplot as plt

#import libraries for muting unnecessary warnings if needed
import warnings
warnings.filterwarnings('ignore')


from IPython.display import Image
url = '../input/smart-homes-temperature-time-series-forecasting/Solar house sensors and actuators map.png'
Image(url,width=700, height=700)


#Reading datasets
df_train=pd.read_csv("../input/smart-homes-temperature-time-series-forecasting/train.csv")
df_test=pd.read_csv("../input/smart-homes-temperature-time-series-forecasting/test.csv")


df_train.info()


df_train.describe()


# sort by dates
df_train.sort_index(inplace = True)

#creating datetime list with boundaries of raw data series, hourly frequency
datelist = pd.date_range(datetime.datetime(2012,3,13,11,45,0), datetime.datetime(2012,4,11,6,30,0), freq='15min').tolist()

#extracting raw data series indices
idx_list = df_train.index.to_list()

#checking for anomalies by comparing the two
print(idx_list == datelist)
#searching for anomalies
print("\n No. of elements in full list:", len(datelist), "\n No. of indices:", len(idx_list), "\n No. of elements in set of indices:", len(set(idx_list)))


import plotly.express as px
fig = px.line(df_train['Indoor_temperature_room'])
fig.show()


import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import cv
# drop the target column from the training data
train = df_train.drop(['Id', 'Indoor_temperature_room', 'Date', 'Time'], axis=1)
test = df_test.drop(['Id', 'Date', 'Time'], axis = 1)
# select only the numerical features
# add the train/test labels
train["AV_class"] = 0
test["AV_class"]  = 1

# make one big dataset
train_test = pd.concat([train, test], axis=0, ignore_index=True)

# shuffle
train_test_shuffled = train_test.sample(frac=1)

# create our DMatrix (the XGBoost data structure)
X = train_test_shuffled.drop(['AV_class'], axis=1)
y = train_test_shuffled['AV_class']
XGBdata = xgb.DMatrix(data=X,label=y)

# our XGBoost parameters
params = {"objective":"binary:logistic",
          "eval_metric":"logloss",
          'learning_rate': 0.05,
          'max_depth': 5, }

# perform cross validation with XGBoost
cross_val_results = cv(dtrain=XGBdata, params=params, 
                       nfold=5, metrics="auc", 
                       num_boost_round=200,early_stopping_rounds=20,
                       as_pandas=True)

# print out the final result
print((cross_val_results["test-auc-mean"]).tail(1))


classifier = XGBClassifier(eval_metric='logloss',use_label_encoder=False)
classifier.fit(X, y)
fig, ax = plt.subplots(figsize=(12,4),dpi=100)
plt.title('Feature Importances', size = 18, color='purple', loc='center', backgroundcolor='lavender', pad='10.0')
plot_importance(classifier, ax=ax, color='#087E8B', edgecolor= 'cyan', linewidth=3)
plt.show();


from scipy import stats
features_list = test.columns.values.tolist()
for feature in features_list:
    statistic, pvalue = stats.kstest(train[feature], test[feature]) #Kolmogorov-Smirnov test 
    print("p-value %.2f" %pvalue, "for the feature",feature)


!pip install -q ptitprince
import ptitprince as pt
num_feats=[col for col in df_test.columns if df_test[col].dtypes != 'object'  and col !='Day_of_the_week' and col != 'Id']
fig=plt.figure(figsize=(30,80))
for i, col in enumerate(num_feats):
    plt.subplot(len(num_feats),2,1*i+1)
    pt.RainCloud(data=df_train, y=df_train[col], bw=0.1, cut=0, hue=['dodgerblue'], orient='h', label="train data", palette=['dodgerblue'], alpha = .65)
    pt.RainCloud(data=df_test,  y=df_test[col], bw=0.1, cut=0, orient='h',label="test data", hue=['crimson'], palette=['crimson'], alpha = .65)
    legend = plt.legend()
    plt.title(f'{col} distribution')
plt.tight_layout()


df_train.Date.unique(), df_test.Date.unique()


# To extract seasons
def month2seasons(x):
    if x in [12, 1, 2, 3]:
        season = 0
    elif x in [6, 7, 8]:
        season = 1
    elif x in [4, 5]:
        season = 2
    elif x in [9, 10, 11]:
        season = 3
    return season

for df in (df_train, df_test):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')   
    df['DateTime'] = df['Date'] + ' ' + df['Time']
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].apply(lambda x : x.hour)
    df['Month'] = df['DateTime'].apply(lambda x : x.month)
    df['Minutes']=df['DateTime'].apply(lambda x: x.hour *60 + x.minute).astype(float)
    df["CO2_avg"] = (df["CO2_(dinning-room)"] + df["CO2_room"])/2 # To extract average of CO@ features
    # To extract average of Meteo Sun light features
    df['Meteo_Sun_light_AVG'] = (df['Meteo_Sun_light_in_west_facade'] + df['Meteo_Sun_light_in_east_facade'] + df['Meteo_Sun_light_in_south_facade'])/3
    df['Season'] = df['Month'].apply(month2seasons)
    df.drop(['Day_of_the_week', 'Date','Time', "CO2_(dinning-room)", "CO2_room", 'Month','Meteo_Sun_light_in_west_facade',
        'Meteo_Sun_light_in_east_facade', 'Meteo_Sun_light_in_south_facade'], axis = 1, inplace = True)
df_train.shape, df_train.columns


minutes=df_train[['Indoor_temperature_room','Minutes']].groupby(['Minutes']).mean().reset_index()
minutes.columns = ['Minutes', 'mean_temp']
#minutes['Minutes']= minutes['Minutes'].astype(int)
scaled_temp = minutes['mean_temp'] - minutes['mean_temp'].mean()


# Define a function to plot the entire dataframe to performs data visualization
def display_plot(x, y, fig_title):
    plt.figure(figsize = (20,7))
    plt.title(fig_title, loc='center', fontsize=20)
    sb.barplot(x = x, y = y, palette = 'cool') 
    plt.tight_layout();
display_plot(minutes['Minutes'],minutes['mean_temp'], "Mean indoor room temperature")
display_plot(minutes['Minutes'],scaled_temp, "Scaled Mean indoor room temperature")

fig,ax=plt.subplots(figsize = (12,5),dpi=100)
#ax.plot(minutes['Minutes'],minutes['mean_temp'])
import matplotlib.patheffects as pe
ax.plot(minutes['Minutes'],minutes['mean_temp'], lw = 3, color='#087E8B', 
         path_effects=[pe.SimpleLineShadow(shadow_color='cyan'), pe.Stroke(linewidth=5, foreground='cyan'),pe.Normal()])
plt.title('Sine wave pattern recognized', loc = 'Center', fontsize=18, color='purple', style='normal', backgroundcolor='lavender', pad='10.0')
plt.show()


a1=np.arange(0,1400)
a2=-np.sin(2*np.pi*a1/1400) # sin (2pift)
display_plot(a1,a2, "Sinosoidal wave pattern\n $\mathcal{A}\mathrm{sin}(2 \omega t)$")
fig,ax=plt.subplots(figsize = (12,5),dpi=100)
ax.plot(a1,a2, lw = 3, color='#087E8B', 
         path_effects=[pe.SimpleLineShadow(shadow_color='cyan'), pe.Stroke(linewidth=5, foreground='cyan'),pe.Normal()])
mono_font = {'fontname':'monospace'}
plt.title('Sine wave with period 1400', loc = 'Center', fontsize=18, **mono_font,
          color='purple', style='oblique', backgroundcolor='lavender', pad='10.0')
plt.title("$\mathcal{A}\mathrm{sin}(2 \pi f t)$", fontsize=16, loc = 'right', pad='10.0')
plt.show()


df_train1 = df_train.drop(['Id', 'DateTime','hour'], axis = 1)
X = df_train1.drop('Indoor_temperature_room', axis = 1)
y = df_train1['Indoor_temperature_room']
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle = True, test_size = 0.2)


!pip install pygam --quiet
from pygam import LinearGAM
gam = LinearGAM(n_splines=10).fit(X, y)
gam.summary()


## plotting
plt.figure();
fig, axes = plt.subplots(3,ncols =int(len(X.columns)/3), figsize = (30, 6));

titles = X.columns
for i, (col,ax) in enumerate(zip(X.columns, axes.flatten())):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    #if i == 0:
        #ax.set_ylim(-30,30)
    ax.set_title(titles[i]);
plt.tight_layout()


def model_train_evaluation(y, ypred, model_name): 
       
    # Model Evaluation metrics
    from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score, r2_score, mean_absolute_percentage_error
    print("\n \n Model Evaluation Report: ")
    print('Mean Absolute Error(MAE) of', model_name,':', mean_absolute_error(y, ypred))
    print('Mean Squared Error(MSE) of', model_name,':', mean_squared_error(y, ypred))
    print('Root Mean Squared Error (RMSE) of', model_name,':', mean_squared_error(y, ypred, squared = False))
    print('Mean absolute percentage error (MAPE) of', model_name,':', mean_absolute_percentage_error(y, ypred))
    print('Explained Variance Score (EVS) of', model_name,':', explained_variance_score(y, ypred))
    print('R2 of', model_name,':', (r2_score(y, ypred)).round(2))
    print('\n \n')
    
    # Actual vs Predicted Plot
    f, ax = plt.subplots(figsize=(12,6),dpi=100);
    plt.scatter(y, ypred, label="Actual vs Predicted")
    # Perfect predictions
    plt.xlabel('Indoor Temperature in celsius')
    plt.ylabel('Indoor Temperature in celsius')
    plt.title('Expection vs Prediction')
    plt.plot(y,y,'r', label="Perfect Expected Prediction")
    plt.legend()
    f.text(0.95, 0.06, 'AUTHOR: RINI CHRISTY',
         fontsize=12, color='green',
         ha='left', va='bottom', alpha=0.5);
    
    print('\n \n \n \n')
    fig,ax=plt.subplots(figsize=(15,8))
    plt.plot(y.values, lw = 4, label='Actual values', color = 'blue')
    plt.plot(ypred, label='Predicted values', color = 'red')
    plt.legend(loc='best')
    plt.title(f'Actual vs Predicted for {model_name}')
    plt.show()


yhat = gam.predict(X)
model_train_evaluation(y, yhat, 'GAM with n_splines=10')


gam = LinearGAM(n_splines=40).fit(X, y)
## plotting
plt.figure();
fig, axes = plt.subplots(3,ncols =int(len(X.columns)/3), figsize = (30, 6));

titles = X.columns
for i, (col,ax) in enumerate(zip(X.columns, axes.flatten())):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    #if i == 0:
        #ax.set_ylim(-30,30)
    ax.set_title(titles[i]);
plt.tight_layout()


yhat = gam.predict(X)
model_train_evaluation(y, yhat, 'GAM (n_splines=40) prediction using training set ')


ypred = gam.predict(X_val)
model_train_evaluation(y_val, ypred, 'GAM prediction using validation set')


gam.summary()


residuals = y_val.values-ypred
plt.scatter(ypred, residuals);
plt.axhline(0, color='red')
plt.xlabel('Predicted Values');
plt.ylabel('Residuals');


# line plot of residuals
residuals = pd.DataFrame(residuals)
residuals.plot()
#plt.ylim(-0.5, 0.5)
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.xlim(-10, 10)
plt.show()
# summary stats of residuals
print(residuals.describe())


from scipy import stats
from statsmodels.stats.diagnostic import normal_ad
import statsmodels
sw_result = stats.shapiro(residuals)#Shapiro Wilk test
KS_result = stats.kstest(residuals, 'norm')#Kolmogorov-Smirnov test 
JB_result = stats.jarque_bera(residuals) #Jarque-Bera test
#AD_result = stats.anderson(residuals)#Anderson-Darling test
ad_result = normal_ad(np.array(residuals), axis=0)#Anderson-Darling test
dag_result = stats.normaltest(residuals, axis=0, nan_policy='propagate')#D’Agostino’s K-squared test
lf_result = statsmodels.stats.diagnostic.lilliefors(residuals)#Lilliefors test
#print(f'\n \n Residulas for Indoor_temperature_room')
#print(residuals)
print(f'\n \n Shapiro Wilk test results for Indoor_temperature_room')
print(sw_result)
print(f'\n \n Kolmogorov-Smirnov test results for Indoor_temperature_room')
print(KS_result)
print(f'\n \n Jarque-Bera test results for Indoor_temperature_room')
print(JB_result)
#print(f'\n \n Anderson-Darling test results for Indoor_temperature_room')
#print(AD_result)
print(f'\n \n Anderson-Darling test results for Indoor_temperature_room')
print(ad_result)
print(f'\n \n D’Agostino’s K-squared test results for Indoor_temperature_room')
print(dag_result)
print(f'\n \n Lilliefors test results for Indoor_temperature_room')
print(lf_result)


def test_model(col):
    s = []
    for i in col:
        a = [1,i]
        s.append(a)
    return (np.array(s))
Het = df_train.iloc[8:,]

exog = test_model(y_val)
# Method 1
from statsmodels.stats.diagnostic import het_breuschpagan
breusch_pagan_test = het_breuschpagan(residuals, exog)
print(breusch_pagan_test)
print ('\n Het_breuschpagan-test p_value:', breusch_pagan_test[1])
if breusch_pagan_test[1] > 0.05:
    print("The residuals are not heteroscedastic.")
if breusch_pagan_test[1] < 0.05:
    print("The residuals are heteroscedastic.")


# Method 2
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import statsmodels.tools.tools as smt
name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(residuals, exog_het= exog )
lzip(name, test)


name = ["F statistic", "p-value"]
GQ_test = sms.het_goldfeldquandt(residuals, exog)
print(lzip(name, test))
if GQ_test[1] > 0.05:
    print("The residuals are not heteroscedastic.")
if GQ_test[1] < 0.05:
    print("The residuals are heteroscedastic.")


from statsmodels.stats.diagnostic import het_white
#define labels to use for output of White's test
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
white_test = het_white(residuals.values,exog) 
#print results of White's test
print(dict(zip(labels, white_test)))
if white_test[1] > 0.05:
    print("The residuals are not heteroscedastic.")
if white_test[1] < 0.05:
    print("The residuals are heteroscedastic.")


X_test = df_test.drop(['Id','DateTime','hour'], axis = 1)
X_test.shape, X_test.columns, X_train.shape, X_train.columns  


df_test['forecast'] = gam.predict(X_test)
#test['forecast'] = forecast
Final = df_test[['Id', 'forecast']]
Final.columns = ['Id', 'Indoor_temperature_room']
Final.to_csv('submission.csv', index = False)
Final


plt.figure(figsize= (12,8))
plt.plot(Final['Indoor_temperature_room']);


%%html
<marquee style=’width: 90%; height:70%; color: #0bda11;’>
    <b> Thanks for reading. Hope you enjoyed it as much as I did working on it.  Please consider upvoting if you like it.</b></marquee>

