import pandas as pd
train_file=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
train_file


train_file.columns


train_file.drop(['id','Episode_Title'], axis=1, inplace=True)


train_file=train_file.dropna()


Publication_Day_List=train_file['Publication_Day'].unique().tolist()
Publication_Day_List, len(Publication_Day_List)


train_file.groupby('Publication_Day')['Listening_Time_minutes'].mean()


(train_file.groupby('Publication_Day')['Listening_Time_minutes'].mean()).plot(kind='hist', title='Avg Listening time (minutes)')


train_file_copy=train_file.copy(deep=True)
train_file_copy['Pub Mon to Wed']=train_file_copy['Publication_Day'].apply(lambda x: 1 if x in ['Monday', 'Tuesday', 'Wednesday'] else 0)
train_file_copy['Pub other days']=train_file_copy['Publication_Day'].apply(lambda x: 0 if x in ['Monday', 'Tuesday', 'Wednesday'] else 1)
train_file_copy.drop(columns=['Publication_Day'], inplace=True)


Genre_List=train_file['Genre'].unique().tolist()
Genre_List, len(Genre_List)


train_file.groupby('Genre')['Listening_Time_minutes'].mean()


(train_file.groupby('Genre')['Listening_Time_minutes'].mean()).plot(kind='hist', title='Avg Listening time (minutes)')


train_file_copy['High Intrigue']=train_file_copy['Genre'].apply(lambda x: 1 if x in ['True Crime', 'Technology', 'Music', 'Health', 'Education', 'Business'] else 0)
train_file_copy['Moderate Intrigue']=train_file_copy['Genre'].apply(lambda x: 1 if x in ['Sports', 'Lifestyle'] else 0)
train_file_copy['Low Intrigue']=train_file_copy['Genre'].apply(lambda x: 1 if x in ['Comedy', 'News'] else 0)
train_file_copy.drop(columns=['Genre'], inplace=True)


Publication_Time_List=train_file['Publication_Time'].unique().tolist()
Publication_Time_List, len(Publication_Time_List)


(train_file.groupby('Publication_Time')['Listening_Time_minutes'].mean()).plot(kind='hist', title='Avg Listening time (minutes)')


train_file.groupby('Publication_Time')['Listening_Time_minutes'].mean()


train_file_copy['Night&Matinee']=train_file_copy['Publication_Time'].apply(lambda x: 1 if x in ['Afternoon', 'Night'] else 0)
train_file_copy['B4&AfterWork']=train_file_copy['Publication_Time'].apply(lambda x: 1 if x in ['Evening', 'Morning'] else 0)
train_file_copy.drop(columns=['Publication_Time'], inplace=True)


train_file.groupby('Episode_Sentiment')['Listening_Time_minutes'].mean()


train_file_copy.loc[train_file_copy['Episode_Sentiment']=='Negative', 'Episode_Sentiment']=1
train_file_copy.loc[train_file_copy['Episode_Sentiment']=='Neutral', 'Episode_Sentiment']=1*(1.02)
train_file_copy.loc[train_file_copy['Episode_Sentiment']=='Positive', 'Episode_Sentiment']=1*(1.02)*(1.02)
train_file_copy


import gc

n = gc.collect()
print("Number of unreachable objects collected by GC:", n)


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


Podcast_Names= train_file_copy['Podcast_Name']
vectorizer=TfidfVectorizer(stop_words='english')

vectorized_Podcast_Names=vectorizer.fit_transform(Podcast_Names)
print(vectorized_Podcast_Names[0]), vectorized_Podcast_Names


#Reducing dimentionality using PCA

pca=PCA(n_components=2)
low_dim_Podcast_Names=pca.fit_transform(vectorized_Podcast_Names.toarray())


low_dim_Podcast_Names


#Applying K-means clustering
no_clusters=3
kmeans=KMeans(n_clusters=no_clusters,n_init=5,max_iter=300,random_state=42)
kmeans.fit(low_dim_Podcast_Names)


results=pd.DataFrame()
results['Podcast_Name']=Podcast_Names
results['Cluster']=kmeans.labels_
print(results.sample(5)),
results


len(results), len(results[results['Cluster']==0]), len(results[results['Cluster']==1]), len(results[results['Cluster']==2])


train_file_copy['Podcast_Name_cluster']=results['Cluster']


len(train_file_copy[train_file_copy['Podcast_Name_cluster']==0]), len(train_file_copy[train_file_copy['Podcast_Name_cluster']==1]), len(train_file_copy[train_file_copy['Podcast_Name_cluster']==2])


train_file_copy.groupby('Podcast_Name_cluster')['Listening_Time_minutes'].mean()


train_file_copy.drop(columns=['Podcast_Name_cluster'], inplace=True)
train_file_copy.drop(columns=['Podcast_Name'], inplace=True)


import matplotlib.pyplot as plt


plt.subplot(3, 3, 1)
(train_file_copy.groupby('Episode_Length_minutes')['Listening_Time_minutes'].mean()).plot(kind='hist', title='Avg Listening time (minutes) for given Episode_Length_minutes')
plt.subplot(3, 3, 4)
(train_file_copy.groupby('Host_Popularity_percentage')['Listening_Time_minutes'].mean()).plot(kind='hist', title='Avg Listening time (minutes) for a given Host_Popularity_percentage')
plt.subplot(3, 3, 7)
(train_file_copy.groupby('Guest_Popularity_percentage')['Listening_Time_minutes'].mean()).plot(kind='hist', title='Avg Listening time (minutes) for a given Guest_Popularity_percentage')


train_file_copy.groupby('Number_of_Ads')['Listening_Time_minutes'].mean()


train_file_copy[train_file_copy['Number_of_Ads']>3]


#These rows with Number of adverts ~ 100 seem garbage data. Hence deleting these rows

train_file_copy.drop(train_file_copy[train_file_copy['Number_of_Ads'] > 12].index, inplace=True)


import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split


# Separating features from the target
train_file_copy.columns


X=train_file_copy[['Episode_Length_minutes', 'Host_Popularity_percentage','Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment','Pub Mon to Wed', 'Pub other days','High Intrigue', 'Moderate Intrigue', 'Low Intrigue', 'Night&Matinee','B4&AfterWork']]
Y=train_file_copy[['Listening_Time_minutes']]


# Split data into test and train sets
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.2, random_state=42)
len(X_train), len(X_test), len(y_train), len(y_test)


# Split data into test and train sets
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.2, random_state=42)

scalar=StandardScaler()
scalar.fit(X_train)
X_train_scaled=scalar.transform(X_train)
X_test_scaled=scalar.transform(X_test)
len(X_train), len(X_test), len(y_train), len(y_test), X_train.shape, y_train.shape


# Define NN architecture
import keras
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
model=Sequential()
model.add(Dense(128, input_dim=12, activation='relu'))
model.add(Dense(32, activation='relu'))
#output layer
model.add(Dense(1, activation='linear'))


# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


model.summary()


# Train the model
history=model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.2)


from matplotlib import pyplot as plt
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1, len(loss)+1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.xlabel='Epochs'
plt.ylabel='Loss'
plt.title='Training and Validation loss'
plt.legend()
plt.show()
acc=history.history['mae']
val_acc=history.history['val_mae']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
#plt.title('Training and Validation MAE')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Evaluate the model on the test data
test_loss = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", test_loss)


y_pred = model.predict(X_test_scaled)


len(y_test), len(y_pred), type(y_test), type(y_pred), y_pred.shape


#y_test['Predicted_Listening_Time']=y_pred
#y_test.drop(columns='Predicted_Listening_Time', inplace=True)
y_pred


test_file=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_file.head()


test_file.drop(columns=['Podcast_Name','Episode_Title'], inplace=True)
test_file.head()


test_file['Pub Mon to Wed']=test_file['Publication_Day'].apply(lambda x: 1 if x in ['Monday', 'Tuesday', 'Wednesday'] else 0)
test_file['Pub other days']=test_file['Publication_Day'].apply(lambda x: 0 if x in ['Monday', 'Tuesday', 'Wednesday'] else 1)
test_file.drop(columns=['Publication_Day'], inplace=True)


test_file['High Intrigue']=test_file['Genre'].apply(lambda x: 1 if x in ['True Crime', 'Technology', 'Music', 'Health', 'Education', 'Business'] else 0)
test_file['Moderate Intrigue']=test_file['Genre'].apply(lambda x: 1 if x in ['Sports', 'Lifestyle'] else 0)
test_file['Low Intrigue']=test_file['Genre'].apply(lambda x: 1 if x in ['Comedy', 'News'] else 0)
test_file.drop(columns=['Genre'], inplace=True)


test_file['Night&Matinee']=test_file['Publication_Time'].apply(lambda x: 1 if x in ['Afternoon', 'Night'] else 0)
test_file['B4&AfterWork']=test_file['Publication_Time'].apply(lambda x: 1 if x in ['Evening', 'Morning'] else 0)
test_file.drop(columns=['Publication_Time'], inplace=True)


test_file.loc[test_file['Episode_Sentiment']=='Negative', 'Episode_Sentiment']=1
test_file.loc[test_file['Episode_Sentiment']=='Neutral', 'Episode_Sentiment']=1*(1.02)
test_file.loc[test_file['Episode_Sentiment']=='Positive', 'Episode_Sentiment']=1*(1.02)*(1.02)


test_file.head()


test_file.interpolate(method='linear', inplace=True)
test_file.head()


n = gc.collect()
print("Number of unreachable objects collected by GC:", n)


test_file.columns


X_pred=X=test_file[['Episode_Length_minutes', 'Host_Popularity_percentage','Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment','Pub Mon to Wed', 'Pub other days', 'High Intrigue','Moderate Intrigue','Low Intrigue','Night&Matinee','B4&AfterWork']]
X_pred.head()


X_pred_scaled=scalar.transform(X_pred)
Y_pred_output = model.predict(X_pred_scaled)


type(Y_pred_output), Y_pred_output.shape


Y_pred_output


Output_df=test_file[['id']]
Output_df['Listening_Time_minutes']=Y_pred_output
#Output_df=Output_df.set_index('id')
Output_df.head(3)


Output_df.to_csv('predictions.csv', index=False)




