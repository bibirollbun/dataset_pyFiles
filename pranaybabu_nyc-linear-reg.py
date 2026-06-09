# Initial Python environment setup...
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to

print(os.listdir('../input'))


train_df =  pd.read_csv('../input/train.csv', nrows = 10_000_000)
train_df.dtypes


train_df.head()


test_df = pd.read_csv('../input/test.csv')
test_df.dtypes


test_df.head()


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_df)
add_travel_vector_features(test_df)


print(train_df.isnull().sum())


print(test_df.isnull().sum())


print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))


plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')


print('Old size: %d' % len(train_df))
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
print('New size: %d' % len(train_df))


print('Old size: %d' % len(test_df))
test_df = test_df[(test_df.abs_diff_longitude < 5.0) & (test_df.abs_diff_latitude < 5.0)]
print('New size: %d' % len(test_df))


ls1 = list(train_df['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_df['pickup_time'] = ls1

ls1 = list(test_df['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_df['pickup_time'] = ls1


train_df.head()


test_df.head()


ls1 = list(train_df['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_df['Weekday'] = ls1

ls1 = list(test_df['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_df['Weekday'] = ls1



train_df.head()


test_df.head()


train_df.drop('pickup_datetime',inplace=True,axis=1)
test_df.drop('pickup_datetime',inplace=True,axis=1)


train_df['Weekday'].replace(to_replace=[i for i in range(0,7)],value = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],inplace=True)
test_df['Weekday'].replace(to_replace=[i for i in range(0,7)],value = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],inplace=True)


train_df.head()


test_df.head()


train_one_hot = pd.get_dummies(train_df['Weekday'])
test_one_hot = pd.get_dummies(test_df['Weekday'])
train_df = pd.concat([train_df,train_one_hot],axis=1)
test_df = pd.concat([test_df,test_one_hot],axis=1)


train_df.head()


test_df.head()


train_df.drop('Weekday',axis=1,inplace=True)
test_df.drop('Weekday',axis=1,inplace=True)


ls1 = list(train_df['pickup_time'])
for i in range(len(ls1)):
    z = ls1[i].split(':')
    ls1[i] = int(z[0])*100+int(z[1])
train_df['pickup_time'] = ls1

ls1 = list(test_df['pickup_time'])
for i in range(len(ls1)):
    z = ls1[i].split(':')
    ls1[i] = int(z[0])*100+int(z[1])
test_df['pickup_time'] = ls1


train_df.head()


test_df.head()


R = 6373.0
lat1 = np.asarray(np.radians(train_df['pickup_latitude']))
lon1 = np.asarray(np.radians(train_df['pickup_longitude']))
lat2 = np.asarray(np.radians(train_df['dropoff_latitude']))
lon2 = np.asarray(np.radians(train_df['dropoff_longitude']))

dlon = lon2-lon1
dlat = lat2-lat1
ls1=[]
a = np.sin(dlat/2)**2 +np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
distance = R*c

train_df['Distance'] = np.asarray(distance)*0.621

lat1 = np.asarray(np.radians(test_df['pickup_latitude']))
lon1 = np.asarray(np.radians(test_df['pickup_longitude']))
lat2 = np.asarray(np.radians(test_df['dropoff_latitude']))
lon2 = np.asarray(np.radians(test_df['dropoff_longitude']))

dlon = lon2-lon1
dlat = lat2-lat1
ls1=[]
a = np.sin(dlat/2)**2 +np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
distance = R*c

test_df['Distance'] = np.asarray(distance)*0.621





R = 6373.0
lat1 = np.asarray(np.radians(train_df['pickup_latitude']))
lon1 = np.asarray(np.radians(train_df['pickup_longitude']))
lat2 = np.asarray(np.radians(train_df['dropoff_latitude']))
lon2 = np.asarray(np.radians(train_df['dropoff_longitude']))

lat3 = np.zeros(len(train_df['pickup_latitude']))+np.radians(40.6413)
lon3 = np.zeros(len(train_df['pickup_longitude']))+np.radians(-73.7781)
dlon_pickup = lon3-lon1
dlat_pickup = lat3-lat1
d_lon_dropoff = lon3-lon2
d_lat_dropoff = lat3-lat2
a1 = np.sin(dlat_pickup/2)**2 +np.cos(lat1)*np.cos(lat3)*np.sin(dlon_pickup/2)**2
c1 = 2*np.arctan2(np.sqrt(a1),np.sqrt(1-a1))
distance1 = R*c1
train_df['Pickup_Distance_airport'] = np.asarray(distance1)*0.621

a2 = np.sin(d_lat_dropoff/2)**2 +np.cos(lat2)*np.cos(lat3)*np.sin(d_lon_dropoff/2)**2
c2 = 2*np.arctan2(np.sqrt(a2),np.sqrt(1-a2))
distance2 = R*c2
train_df['Dropoff_Distance_airport'] = np.asarray(distance2)*0.621

lat1 = np.asarray(np.radians(test_df['pickup_latitude']))
lon1 = np.asarray(np.radians(test_df['pickup_longitude']))
lat2 = np.asarray(np.radians(test_df['dropoff_latitude']))
lon2 = np.asarray(np.radians(test_df['dropoff_longitude']))

lat3 = np.zeros(len(test_df['pickup_latitude']))+np.radians(40.6413)
lon3 = np.zeros(len(test_df['pickup_latitude']))+np.radians(-73.7781)
dlon_pickup = lon3-lon1
dlat_pickup = lat3-lat1
d_lon_dropoff = lon3-lon2
d_lat_dropoff = lat3-lat2
a1 = np.sin(dlat_pickup/2)**2 +np.cos(lat1)*np.cos(lat3)*np.sin(dlon_pickup/2)**2
c1 = 2*np.arctan2(np.sqrt(a1),np.sqrt(1-a1))
distance1 = R*c1
test_df['Pickup_Distance_airport'] = np.asarray(distance1)*0.621

a2 = np.sin(d_lat_dropoff/2)**2 +np.cos(lat2)*np.cos(lat3)*np.sin(d_lon_dropoff/2)**2
c2 = 2*np.arctan2(np.sqrt(a2),np.sqrt(1-a2))
distance2 = R*c2
test_df['Dropoff_Distance_airport'] = np.asarray(distance2)*0.621


train_df['Distance'] = np.round(train_df['Distance'],2)
train_df['Pickup_Distance_airport'] = np.round(train_df['Pickup_Distance_airport'],2)
train_df['Dropoff_Distance_airport'] = np.round(train_df['Dropoff_Distance_airport'],2)
test_df['Distance'] = np.round(test_df['Distance'],2)
test_df['Pickup_Distance_airport'] = np.round(test_df['Pickup_Distance_airport'],2)
test_df['Dropoff_Distance_airport'] = np.round(test_df['Dropoff_Distance_airport'],2)


train_df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)
test_df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)


train_df['abs_diff_longitude']=np.abs(train_df['abs_diff_longitude'])-np.mean(train_df['abs_diff_longitude'])
train_df['abs_diff_longitude']=train_df['abs_diff_longitude']/np.var(train_df['abs_diff_longitude'])


train_df['abs_diff_latitude']=np.abs(train_df['abs_diff_latitude'])-np.mean(train_df['abs_diff_latitude'])
train_df['abs_diff_latitude']=train_df['abs_diff_latitude']/np.var(train_df['abs_diff_latitude'])


test_df['abs_diff_longitude']=np.abs(test_df['abs_diff_longitude'])-np.mean(test_df['abs_diff_longitude'])
test_df['abs_diff_longitude']=test_df['abs_diff_longitude']/np.var(test_df['abs_diff_longitude'])


test_df['abs_diff_latitude']=np.abs(test_df['abs_diff_latitude'])-np.mean(test_df['abs_diff_latitude'])
test_df['abs_diff_latitude']=test_df['abs_diff_latitude']/np.var(test_df['abs_diff_latitude'])


train_df.shape


test_df.shape


train_df.head()


from sklearn.model_selection import train_test_split
X = train_df.drop(['key','fare_amount'],axis=1)
y = train_df['fare_amount']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.01,random_state=80)


from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))


pred = np.round(lr.predict(test_df.drop('key',axis=1)),2)


pd.read_csv('../input/sample_submission.csv').head()


Submission = pd.DataFrame(data=pred,columns=['fare_amount'])
Submission['key'] = test_df['key']
Submission = Submission[['key','fare_amount']]


Submission.set_index('key',inplace=True)


Submission.to_csv('Submission.csv')




