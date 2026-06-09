import numpy as np
import pandas as pd

#data_path
data_path = '/kaggle/input/bike-sharing-demand/'

train = pd.read_csv(data_path + 'train.csv')  #train data
test  = pd.read_csv(data_path + 'test.csv')   #test data
submission = pd.read_csv(data_path + 'sampleSubmission.csv')  # Sample data to submit


# train data : 10886 rows, 12 - features
# test data  : 6493 rows, 9 -  features 
train.shape, test.shape


train.head()


test.head()


submission.head()


train.info()


test.info()


def resumetable(df):
    '''
    feature summary
    '''
    print(f'Data shape : {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns = ['Data Type'])
    summary = summary.reset_index()

    summary = summary.rename(columns = {'index': '피처'})
    summary['NaN Count'] = df.isnull().sum().values
    summary['Unique Count'] = df.nunique().values
    summary['The First value'] = df.loc[0].values
    summary['The Second value'] = df.loc[1].values
    summary['The Third value'] = df.loc[2].values

    return summary


summary = resumetable(train)
summary


print(train['datetime'][100])  # 100th data
print(train['datetime'][100].split())  # Split on the basis of space.
print(f"Date : {train['datetime'][100].split()[0]}")
print(f"Time : {train['datetime'][100].split()[1]}")


print(train['datetime'][100].split()[0])
print(train['datetime'][100].split()[0].split('-'))
print(train['datetime'][100].split()[0].split('-')[0])  # year
print(train['datetime'][100].split()[0].split('-')[1])  # month
print(train['datetime'][100].split()[0].split('-')[2])  # day


#time, hour,  minute, seconds
print(train['datetime'][100].split()[1])  #time
print(train['datetime'][100].split()[1].split(':'))
print(train['datetime'][100].split()[1].split(':')[0]) # hour
print(train['datetime'][100].split()[1].split(':')[1]) # minute
print(train['datetime'][100].split()[1].split(':')[2]) # second


train['date'] = train['datetime'].apply(lambda x : x.split()[0])  # date feature
#year, month, day, hour, minute, seconds
train['year'] = train['datetime'].apply(lambda x : x.split()[0].split('-')[0]) # year
train['month'] = train['datetime'].apply(lambda x : x.split()[0].split('-')[1]) #month
train['day'] = train['datetime'].apply(lambda x : x.split()[0].split('-')[2]) #day
train['hour'] = train['datetime'].apply(lambda x : x.split()[1].split(":")[0]) #hour
train['minute'] = train['datetime'].apply(lambda x : x.split()[1].split(":")[1]) #minute
train['seconds'] = train['datetime'].apply(lambda x : x.split()[1].split(":")[2]) #seconds


train.head()


from datetime import datetime
import calendar

print(train['datetime'][100]) # date
print(datetime.strptime(train['date'][100], '%Y-%m-%d'))  # datetime format
print(datetime.strptime(train['date'][100], '%Y-%m-%d').weekday()) #change day into integer
print(calendar.day_name[datetime.strptime(train['date'][100], '%Y-%m-%d').weekday()])  # day name


#train['weekday'] = train['date'].apply(lambda x:calendar.day_name[datetime.strptime(x,"%Y-%m-%d").weekday()])
train['weekday'] = train['date'].apply(lambda dateString:calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])


train.head()


train['season'] = train['season'].map({1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'})
train['weather'] = train['weather'].map({1:'Clear', 2:'Mist, Few clouds', 3:'Light Snow, Rain, Thunderstorm', 4:'Heavy Rain, Thunderstorm, Snow, Fog'})


train.head()


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline


mpl.rc('font', size=15)
sns.displot(train['count'])   #The distribution of target values


# Logarithm convert for skewed data
sns.displot(np.log(train['count']))


#로그함수
a = np.log(4) # e - 2.718를 거듭제곱하여 4가 되게 하는 수 (Logarithm)
a


#지수함수로 변환
np.exp(a)


#Step 1 - font 
mpl.rc('font', size=14)  # font size
mpl.rc('axes', titlesize=15) # each axes' title size 
figure, axes = plt.subplots(nrows=3, ncols=2)  # 3 * 2
plt.tight_layout()     #Reserve space betwen graph
figure.set_size_inches(10, 9) # set 10*9 inches of overall figure

#Step 2 - assign subplots
sns.barplot(x='year', y='count', data = train, ax = axes[0,0])
sns.barplot(x='month', y='count', data = train, ax = axes[0,1])
sns.barplot(x='day',   y='count', data=train, ax = axes[1,0])
sns.barplot(x='hour',  y='count', data = train, ax = axes[1,1])
sns.barplot(x='minute', y='count', data =train, ax=axes[2,0])
sns.barplot(x='seconds', y='count', data =train, ax=axes[2,1])

# Step 3 - 

#Step 3-1 : title set
axes[0,0].set(title = 'Rental amounts by year')
axes[0,1].set(title= 'Rental amounts by month')
axes[1,0].set(title = 'Rental amounts by day')
axes[1,1].set(title = 'Rental amounts by hour')
axes[2,0].set(title = 'Rental amounts by minute')
axes[2,1].set(title = 'Rental amounts by seconds')

# Step3-2 : label rotation
axes[1,0].tick_params(axis = 'x', labelrotation=90)
axes[1,1].tick_params(axis = 'x', labelrotation=90)


axes


#Step 1 : m * n Figure
figures, axes = plt.subplots(nrows=2, ncols=2)
plt.tight_layout()
figure.set_size_inches(10, 9)

#Step 2 : Assign subplots
#season, weather, holiday, workingday
sns.boxplot(x='season', y='count', data = train, ax = axes[0,0])
sns.boxplot(x='weather', y='count', data = train, ax = axes[0,1])
sns.boxplot(x='holiday', y='count', data = train, ax = axes[1,0])
sns.boxplot(x='workingday', y='count', data = train, ax = axes[1,1])

#Step 3 - detail set
#Step 3-1 : Set subplots title
axes[0,0].set(title = 'Box Plot On Count Across Season')
axes[0,1].set(title = 'Box Plot On Count Across weather')
axes[1,0].set(title = 'Box Plot On Count Across holiday')
axes[1,1].set(title = 'Box Plot On Count Across workingday')

#3-2 :
axes[0,1].tick_params(axis='x', labelrotation=10)


#Step 1 - m * n Figure 
mpl.rc('font', size=11)
figure, axes = plt.subplots(nrows=5)  # 5 * 1
figure.set_size_inches(12, 18)

#Step 2 : Assign subplots
sns.pointplot(x='hour', y='count', data=train, hue='workingday', ax = axes[0])
sns.pointplot(x='hour', y='count', data=train, hue='holiday', ax = axes[1])
sns.pointplot(x='hour', y='count', data=train, hue='weekday', ax = axes[2])
sns.pointplot(x='hour', y='count', data=train, hue='season',  ax = axes[3])
sns.pointplot(x='hour', y='count', data=train, hue='weather', ax = axes[4])


#Step 1 - m * n figure 
mpl.rc('font', size=15)
figure, axes = plt.subplots(nrows=2, ncols=2)
plt.tight_layout()
figure.set_size_inches(7,6)

#Step 2: Assign subplots
# Temperature, actual temperature, Windspeed, Humidity
sns.regplot(x='temp', y='count', data=train, ax=axes[0,0], scatter_kws={'alpha':0.2}, line_kws={'color':'blue'})
sns.regplot(x='atemp', y='count', data=train, ax=axes[0,1], scatter_kws={'alpha':0.2}, line_kws={'color':'blue'})
sns.regplot(x='windspeed', y='count', data=train, ax=axes[1,0], scatter_kws={'alpha':0.2}, line_kws={'color':'blue'})
sns.regplot(x='humidity', y='count', data=train, ax=axes[1,1], scatter_kws={'alpha':0.2}, line_kws={'color':'blue'})


#code snippet
train[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr()


#Step 1 - Figure preparation
corrMat = train[['temp','atemp','humidity', 'windspeed', 'count']].corr()
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)

#Step 2 
sns.heatmap(data = corrMat, annot=True)   # annot=True : Correlation density 
ax.set(title = 'Heatmap of Numerical Data')

