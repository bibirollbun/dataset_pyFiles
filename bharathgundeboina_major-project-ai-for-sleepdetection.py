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


!pip install pandarallel


!pip install polars plotly pyarrow scikit-learn colorama tqdm


import gc 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import polars as pl
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline

from colorama import Fore, Style, init

import pyarrow as pa
import datetime as dt
import torch

from tqdm import tqdm
tqdm.pandas()

from pyarrow.parquet import ParquetFile
from datetime import datetime as dts
from pandarallel import pandarallel 
pandarallel.initialize(progress_bar=True)


import warnings
warnings.filterwarnings('ignore')


class PATHS:
    MAIN_DIR = "/kaggle/input/child-mind-institute-detect-sleep-states/"
    SUBMISSION = MAIN_DIR + "sample_submission.csv"
    TRAIN_EVENTS = MAIN_DIR + "train_events.csv"
    TRAIN_SERIES = MAIN_DIR + "train_series.parquet"
    TEST_SERIES = MAIN_DIR + "test_series.parquet"



class CMIDataReader:
    def __init__(self, demo_mode=False):
        self.paths = {
            "submission": PATHS.SUBMISSION,
            "train_events": PATHS.TRAIN_EVENTS,
            "train_series": PATHS.TRAIN_SERIES,
            "test_series": PATHS.TEST_SERIES,
        }
        self.demo_mode = demo_mode

    def load_data(self, data_name):
        path = self.paths.get(data_name)
        if path is None:
            raise ValueError(f"Invalid data name: {data_name}")

        # Read CSV or Parquet based on file extension
        if path.endswith('.parquet'):
            if self.demo_mode:
                data = pd.read_parquet(path, engine='pyarrow', columns=None).head(10000)
            else:
                data = pd.read_parquet(path, engine='pyarrow')
        else:
            if self.demo_mode:
                data = pd.read_csv(path, nrows=10000)
            else:
                data = pd.read_csv(path)

        # For files with 'timestamp' column, parse datetime and set index
        if 'timestamp' in data.columns:
            data = data.dropna(subset=['timestamp'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
            data = data.set_index('timestamp')

        return data


# Usage example
reader = CMIDataReader(demo_mode=True)  # Set demo_mode=True for quick test



submission = reader.load_data("submission")


test_series = reader.load_data("test_series")


train_events = reader.load_data("train_events")



train_series = reader.load_data("train_series")



train_series.describe().T


train_events.describe().T



print(train_series.info())

print(train_events.info())


print("\nTrain Events Dataset Info:")
print(f"Shape: {train_events.shape}")

print("\nTrain Series Dataset Info:")
print(f"Shape: {train_series.shape}")

print("\nTest Series Dataset Info:")
print(f"Shape: {test_series.shape}")



train_series.reset_index(inplace = True)
train_events.reset_index(inplace = True)


train_series


train_events


# Ensure it's in datetime format
train_events["timestamp"] = pd.to_datetime(train_events["timestamp"], utc = True)

# Extract datetime components
train_events["year"] = train_events["timestamp"].dt.year
train_events["month"] = train_events["timestamp"].dt.month
train_events["day"] = train_events["timestamp"].dt.day
train_events["hour"] = train_events["timestamp"].dt.hour


train_events


df_onset = train_events[train_events['event'] == 'onset']
df_wakeup = train_events[train_events['event'] == 'wakeup']

fig = make_subplots(rows=1, cols=2, subplot_titles=('<b>Hourly Distribution of Onset Events</b>',
                                                    '<b>Hourly Distribution of Wakeup Events</b>',
                                                   ))

fig.add_trace(go.Histogram(x=df_onset['hour'].dropna(), nbinsx=24), row=1, col=1)
fig.add_trace(go.Histogram(x=df_wakeup['hour'].dropna(), nbinsx=24), row=1, col=2)
fig.update_layout(
    showlegend=False,
    width=800,
    height=400,
    autosize=False,
    margin=dict(t=20, b=4, l=5, r=4),
    template="plotly_white"
    
)
fig.show()


df_onset = train_events[train_events['event'] == 'onset']
df_wakeup = train_events[train_events['event'] == 'wakeup']

fig = make_subplots(rows=1, cols=2, subplot_titles=('<b>day Distribution of Onset Events</b>',
                                                    '<b>day Distribution of Wakeup Events</b>',
                                                   ))

fig.add_trace(go.Histogram(x=df_onset['day'].dropna(), nbinsx=24), row=1, col=1)
fig.add_trace(go.Histogram(x=df_wakeup['day'].dropna(), nbinsx=24), row=1, col=2)
fig.update_layout(
    showlegend=False,
    width=800,
    height=400,
    autosize=False,
    margin=dict(t=20, b=4, l=5, r=4),
    template="plotly_white"
    
)
fig.show()


plt.figure(figsize=(10,4))
# plt.subplot(121)
plt.title('Hour distribution with onset and wakeup')
sns.histplot(x=train_events.dropna().hour, hue=train_events.dropna().event, stat='density', bins=24, binrange=(-0.5, 23.5))
sns.kdeplot(train_events.dropna().hour, bw_adjust=0.45)


plt.figure(figsize=(10,4))
# plt.subplot(121)
plt.title('day distribution with onset and wakeup')
sns.histplot(x=train_events.dropna().day, hue=train_events.dropna().event, stat='density', bins=24, binrange=(-0.5, 23.5))
sns.kdeplot(train_events.dropna().day, bw_adjust=0.45)


plt.figure(figsize=(10,4))
# plt.subplot(121)
plt.title('month distribution with onset and wakeup')
sns.histplot(x=train_events.dropna().month, hue=train_events.dropna().event, stat='density', bins=24, binrange=(-0.5, 23.5))
sns.kdeplot(train_events.dropna().month, bw_adjust=0.45)


mask_nonull= (~train_events['step'].isnull()) & (~train_events['timestamp'].isnull()) 
train_events_nonull = train_events[mask_nonull]

# Group by 'series_id' and calculate sleep duration for each night
sleep_duration_df = train_events_nonull.groupby([train_events_nonull['series_id'], train_events_nonull['night']])['timestamp'].agg(['min', 'max']).reset_index()
sleep_duration_df = sleep_duration_df.rename(columns={'min': 'onset', 'max': 'wakeup'})
sleep_duration_df['sleep_duration'] = ((sleep_duration_df['wakeup'] - sleep_duration_df['onset']).dt.seconds / 3600).round(0)

dfg = sleep_duration_df['sleep_duration'].value_counts().reset_index()
dfg.columns = ['sleep_duration', 'number_of_observations']
fig = px.bar(dfg, x='sleep_duration', y='number_of_observations', 
             title='Sleep hour distribution, Training seria (hours)'
            ) 
fig.show()


train_series.info()


sns.boxplot(x = train_series['anglez'])
plt.title('box plot for train series enmo data')


train_series['anglez'] = pd.to_numeric(train_series['anglez'], errors='coerce').astype('float32')

# Drop NaNs before plotting
sns.histplot(train_series['anglez'].dropna(), bins=50)

# Plot labels
plt.title('distribution  of train_series anglez data')
plt.xlabel('anglez')
plt.ylabel('Frequency')
plt.show()


sns.boxplot(x = train_series['enmo'])
plt.title('box plot for train series enmo data')


train_series['enmo'] = pd.to_numeric(train_series['enmo'], errors='coerce').astype('float32')

# Drop NaNs before plotting
sns.histplot(train_series['enmo'].dropna(), bins=50,kde = True )

# Plot labels
plt.title('distribution  of train_series enmo data')
plt.xlabel('anglez')
plt.ylabel('Frequency')
plt.show()


train_events


# Create 'missing_event' column: True if 'step' is missing, else False
train_events['missing_event'] = train_events['step'].isnull()

# Group by 'series_id' and aggregate
aggregated = train_events.groupby('series_id').agg(
    tot_events=('event', 'count'),
    tot_nights=('night', 'last'),
    tot_missing_events=('missing_event', 'sum')
).reset_index()

# Compute 'tot_missing_nights'
aggregated['tot_missing_nights'] = (aggregated['tot_missing_events'] // 2).astype('int32')

# Compute 'tot_recorded_events' and 'tot_recorded_nights'
aggregated['tot_recorded_events'] = aggregated['tot_events'] - aggregated['tot_missing_events']
aggregated['tot_recorded_nights'] = aggregated['tot_nights'] - aggregated['tot_missing_nights']

# Plot: Nights of sleep per user
fig, ax = plt.subplots(figsize=(10, 6))
aggregated['tot_recorded_nights'].hist(bins=50, ax=ax, color='skyblue')
ax.set_title('Total Recorded Nights of Sleep per User')
ax.set_ylabel('Nights of Sleep')
ax.set_xlabel('Number of Users')

# Describe the 'tot_recorded_nights' column
aggregated[['tot_recorded_nights']].describe().T



# Plot missing nights of sleep per user
fig, ax = plt.subplots(figsize=(10, 6))
aggregated['tot_missing_nights'].hist(bins=50, ax=ax, color='maroon')
ax.set_title('Total Missing Nights of Sleep per User')
ax.set_ylabel('Missing Nights of Sleep')
ax.set_xlabel('Number of Users')

# Describe
aggregated[['tot_missing_nights']].describe().T




def window(df, win_size):
    ind=df.index[df['event'].isna()==False]
    c=0
    df['window']=np.nan
    for i in tqdm(ind):
        a=i-win_size
        b=i+win_size
        df['window'].loc[a:i]=int(c)
        c=c+1
        df['window'].loc[i:b]=int(c)
        c=c+1
    df['window'].dropna(inplace=True)
    return df[df['window'].isna()==False]

    


def inactive_periods(df):
    print("shape before application: ",df.shape)
    df['diff_anglez']=df['anglez'].diff()
    df=df[(df['enmo']!=0.0) | (df['diff_anglez']!=0.0)]
    print("shape after application: ",df.shape)
    df.drop('diff_anglez', inplace=True, axis=1)
    print("shape after completion: ",df.shape)
    print("removed ")
    return df



def clustering(df):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler    
    X=df[['anglez','enmo']]
#Scalling the data
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
#perform clustering
    model=KMeans(n_clusters=4,algorithm="elkan" )
    model.fit(X_scaled)
    return model.labels_



def rollingstd(series_df):
# Creating columns with nans
    series_df['sd_enmo_1']=np.nan    # 1 min rolling std: enmo
    series_df['sd_anglez_1']=np.nan  # 1 min rolling std: anglez
    series_df['m_enmo_2']=np.nan     # 2 min rolling mean: enmo
    series_df['m_anglez_2']=np.nan   # 2 min rolling std: anglez 
    print('anglez rolling std 12')
    series_df['sd_anglez_1'] = (series_df.groupby('series_id')['anglez']
                      .rolling(12)
                      .std()
                      .reset_index(level=0, drop=True))
    print('anglez rolling std 2')
    series_df['sd_anglez_1'][series_df['sd_anglez_1'].isna()==True] = (series_df.groupby('series_id')['anglez']
                      .rolling(2)
                      .std()
                      .reset_index(level=0, drop=True))
    print('enmo rolling std 12')
    series_df['sd_enmo_1'] = (series_df.groupby('series_id')['enmo']
                      .rolling(12)
                      .std()
                      .reset_index(level=0, drop=True))
    print('enmo rolling mean 24')
    series_df['m_enmo_2'] = (series_df.groupby('series_id')['enmo']
                      .rolling(24)
                      .mean()
                      .reset_index(level=0, drop=True))
    print('anglez rolling mean 24')
    series_df['m_anglez_2'] = (series_df.groupby('series_id')['anglez']
                      .rolling(24)
                      .mean()
                      .reset_index(level=0, drop=True))
    print('enmo rolling std 2')
    print('Nans in sd_emno_1: ',series_df['sd_enmo_1'].isnull().sum())
    series_df['sd_enmo_1'][series_df['sd_enmo_1'].isna()==True] = (series_df.groupby('series_id')['enmo']
                      .rolling(2)
                      .std()
                      .reset_index(level=0, drop=True))
    print('enmo rolling mean 2')
    series_df['m_enmo_2'][series_df['m_enmo_2'].isna()==True] = (series_df.groupby('series_id')['enmo']
                      .rolling(2)
                      .mean()
                      .reset_index(level=0, drop=True))
    print('anglez rolling mean 2')
    series_df['m_anglez_2'][series_df['m_anglez_2'].isna()==True] = (series_df.groupby('series_id')['anglez']
                      .rolling(2)
                      .mean()
                      .reset_index(level=0, drop=True))
#Series wise rolling std and mean
# filling rest of nans
    print('Nans in sd_emno_1: ',series_df['sd_enmo_1'].isnull().sum())
    series_df['sd_enmo_1'].fillna(0.0, inplace=True)
    series_df['sd_anglez_1'].fillna(0.0, inplace=True)
    series_df['m_enmo_2'].fillna(0.0, inplace=True)
    series_df['m_anglez_2'].fillna(0.0, inplace=True)
    print('Nans after removal: ',series_df['sd_enmo_1'].isnull().sum())

    return(series_df)


def scale(X):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X)
    return (scaler.transform(X))


df_series = train_series
df_events = train_events


print(type(df_series))  # should output: <class 'pandas.core.frame.DataFrame'>
print(type(df_events))



# Merging the datasets
print('Merging the training datasets...')
events = df_events[['series_id', 'step', 'event']]
series_df = pd.merge(df_series, events, on=["step", "series_id"], how='left')
series_df['sleep']=np.nan
series_df.loc[series_df["event"]=="onset", "sleep"] = 1
series_df.loc[series_df["event"]=="wakeup", "sleep"] = 0
series_df['sleep'].fillna(method='ffill', inplace=True)
series_df['sleep'].fillna(value=0, inplace=True)
print('Datasets Merged...')
print('______________________________________')

# Removing the periods of inactivity
print('Removing the periods of Inactivity...')
series_df=inactive_periods(series_df)
print('______________________________________')

# Forming Windows
win_size=720  #60mins
print('Creating Windows each size: ',win_size)
series_df=window(series_df,win_size)
print('Windows formed...')
print('______________________________________')

# Adding the columns of Standard Deviation (1 min)
print('Adding columns to account for deviation in enmo and anglez 1 min rolling...')
series_df=rollingstd(series_df)
series_df['sd_anglez_1']=pd.to_numeric(series_df['sd_anglez_1'])
series_df['sd_enmo_1']=pd.to_numeric(series_df['sd_enmo_1'])
series_df['m_anglez_2']=pd.to_numeric(series_df['m_anglez_2'])
series_df['m_enmo_2']=pd.to_numeric(series_df['m_enmo_2'])
print('Std columns added...')
print('______________________________________')

# Clustering the Data
print('Clustering the data based on enmo and anglez...')
series_df['cluster']=(clustering(series_df)+1)/4
print('Added clusters...')


series_df.head()


series_df['cluster'].unique()


figure= px.imshow(series_df[['sd_anglez_1','sd_enmo_1','m_anglez_2','m_enmo_2','anglez','enmo','cluster','sleep']].corr(),text_auto=True, width=1200, height=1200)
figure.show()


from sklearn.model_selection import train_test_split
X=series_df[['sd_anglez_1','sd_enmo_1','anglez','m_anglez_2','m_enmo_2','enmo','cluster']]
y=series_df[['sleep']]
X_scaled=scale(X)
X_train, X_test, y_train, y_test =train_test_split(X_scaled,y,test_size=0.2, random_state=42)



X_train.shape


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


X.to_csv('X_data.csv')


y.to_csv('y_data.csv')


series_df.to_csv('final_series_df.csv')


y_test[['sleep']].value_counts()


def evaluate(y_test,ypred):
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    print("Accuracy: ",accuracy_score(y_test,y_pred)) 
    print("Precision Score : ", precision_score(y_test,y_pred)) #precision measures the proportion of true positive predictions among all positive instances. how many of survived predicted actually survived, doesn't verifies 0's 70 survived as preicted whereas actually 92 survived so 70/92 will be the precision.  if we predicted 70 survived, so presion will tell how many of those 70 predicted survived matches the actual row by row data. It checkes all positives and verifies if the answer is true for each row?
    print("Recall Score: ", recall_score(y_test,y_pred, average='macro')) #Recall measures the proportion of true positive predictions among all actual positive instalnces. If we predicted 100 survived correctly whereas actually 100 survived out of which 67 predicted correctly so recall will be 0.67
    print("F1 Score: ",f1_score(y_test,y_pred)) #mean of recall and precision
    cm = confusion_matrix(y_test, y_pred)
    figure= px.imshow(cm,text_auto=True, width=1200, height=1200)
    figure.show()



from sklearn.ensemble import RandomForestClassifier
rf =RandomForestClassifier(n_jobs=-1,verbose=1) 
print ('Training the model')
rf.fit(X_train,y_train)
print ('Saving the model')
from joblib import dump, load
dump(rf, 'rf_model.joblib')
y_pred=rf.predict(X_test)
evaluate(y_test,y_pred)


def evaluate_train_data(y_train, train_pred):
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    print("Accuracy: ",accuracy_score(y_train, train_pred)) 
    print("Precision Score : ", precision_score(y_train, train_pred)) #precision measures the proportion of true positive predictions among all positive instances. how many of survived predicted actually survived, doesn't verifies 0's 70 survived as preicted whereas actually 92 survived so 70/92 will be the precision.  if we predicted 70 survived, so presion will tell how many of those 70 predicted survived matches the actual row by row data. It checkes all positives and verifies if the answer is true for each row?
    print("Recall Score: ", recall_score(y_train, train_pred, average='macro')) #Recall measures the proportion of true positive predictions among all actual positive instalnces. If we predicted 100 survived correctly whereas actually 100 survived out of which 67 predicted correctly so recall will be 0.67
    print("F1 Score: ",f1_score(y_train, train_pred)) #mean of recall and precision
    cm = confusion_matrix(y_train, train_pred)
    figure= px.imshow(cm,text_auto=True, width=1200, height=1200)
    figure.show()



train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)



evaluate_train_data(y_train, train_pred)


from sklearn.metrics import accuracy_score


def check_model_fit(train_acc, test_acc):
    print(f"\n Train Accuracy: {train_acc:.4f}")
    print(f" Test Accuracy:  {test_acc:.4f}")
    if train_acc > 0.95 and (train_acc - test_acc) > 0.1:
        print(" Model is likely OVERFITTING.")
    elif train_acc < 0.70 and test_acc < 0.70:
        print(" Model is likely UNDERFITTING.")
    else:
        print(" Model is likely a GOOD FIT.")
check_model_fit(
    train_acc=accuracy_score(y_train, train_pred),
    test_acc=accuracy_score(y_test, test_pred)
)


print('Removing the periods of Inactivity...')
test_series = inactive_periods(test_series)
print('______________________________________')

print('Adding Features...')
test_series = rollingstd(test_series)
test_series['sd_anglez_1'] = pd.to_numeric(test_series['sd_anglez_1'])
test_series['sd_enmo_1'] = pd.to_numeric(test_series['sd_enmo_1'])
test_series['m_anglez_2'] = pd.to_numeric(test_series['m_anglez_2'])
test_series['m_enmo_2'] = pd.to_numeric(test_series['m_enmo_2'])
print('Features added...')
print('______________________________________')

print('Clustering the data based on enmo and anglez...')
test_series['cluster'] = (clustering(test_series) + 1) / 4
print('Added clusters...')

X_test = test_series[['sd_anglez_1', 'sd_enmo_1', 'anglez', 'm_anglez_2', 'm_enmo_2', 'enmo', 'cluster']]
y_pred = rf.predict(scale(X_test))
X_test = []



test_series.reset_index(inplace = True)


result_df=test_series[['series_id', 'step','timestamp']]
result_df['sleep']=y_pred
result_df['timestamp']=result_df[['timestamp']].progress_apply(lambda x: pd.to_datetime(x,utc=True))
df=result_df.copy()
df.index=df['timestamp']
mean = df.groupby([df['series_id'], df.index.floor('5min')])['sleep'].mean()  # Calculating the mean of predictions over an interval of 5 mins (Due to the nature of Testing Data Series). 
mean=mean.reset_index()
mean['timestamp']=mean['timestamp']- pd.to_timedelta('5m') # Since the event is recorded at the end of the interval so subtracting 5 mins (Due to the nature of Testing Data Series) so it records the event at the start of the interval
summary=pd.merge(result_df,mean,on=["timestamp","series_id"],how='left')  # merging the means into the original data based on timestamps and series ID.
summary=summary[summary['sleep_y'].isna()==False]  # removing the Nan's of prediction mean. That'll ensure that we have a row every 5 mins (Due to the nature of Testing Data Series).
# Creating Event Column
summary['event']=np.nan
summary.loc[summary["sleep_y"]==1, "event"] = 'onset'  # the mean prediction will be 1 if predicted onset for 30 mins consecutive
summary.loc[summary["sleep_y"]==0, "event"] = 'wakeup' # the mean prediction will be 0 if predicted wakeup for 30 mins consecutive. Any duration in between will be considered disturbance as will be less tan 30 mins.
summary=summary[summary['event'].isna()==False] # Removing the rows with no event recorded. 
submission=summary[['series_id','step','event','sleep_y']]  # Creating Submission
submission = submission.rename(columns={'sleep_y': 'score'})  # Renaming a column

submission.to_csv('submission.csv')  # Saving the csv file


submission


import pickle


pickle.dump(rf, open("model.pkl", "wb"))




model = pickle.load(open('model.pkl','rb'))




