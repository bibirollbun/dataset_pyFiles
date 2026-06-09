import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



train_df=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv') 
test_df=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_df.info()


train_df.head()


train_df.describe()


test_df.describe()


long_Podcast_train=train_df['Episode_Length_minutes']>200
long_Podcast_test=test_df['Episode_Length_minutes']>200
print(f"The number of long podcasts in the training data is：{long_Podcast_train.sum()}")
print(f"The number of long podcasts in the test data is：{long_Podcast_test.sum()}")




train_df['Episode_Length_minutes']=train_df.groupby(['Podcast_Name','Episode_Title'])['Episode_Length_minutes'].transform(lambda x: x.fillna(x.mean()))
train_df['Guest_Popularity_percentage']=train_df['Guest_Popularity_percentage'].fillna(0)
train_df['Number_of_Ads']=train_df['Number_of_Ads'].fillna(0)



def clean_podcast_data(df,verbose=True):
    original_len=len(df)
  
    df=df[df['Episode_Length_minutes']>0]
    
    df=df[df['Listening_Time_minutes']>0]
    df=df[df['Listening_Time_minutes']<=3*df['Episode_Length_minutes']]
 
    df=df.reset_index(drop=True)
    if verbose:
        cleaned_len=len(df)
        print(f"Data volume before cleaning:{original_len}")
        print(f"Data volume after cleaning:{cleaned_len}")
        print(f"Clear data volume:{original_len-cleaned_len}")
    return df

train_df=clean_podcast_data(train_df)
    
    


train_pairs = set(zip(train_df['Podcast_Name'],train_df['Episode_Title'] ))
test_pairs = set(zip(test_df['Podcast_Name'],test_df['Episode_Title'] ))
unseen_pairs = test_pairs- train_pairs 
print(f"test dataset have {len(unseen_pairs)} new episode")


podcast_count_train=train_df.groupby('Podcast_Name')['Podcast_Name'].count()
podcast_count_test=test_df.groupby('Podcast_Name')['Podcast_Name'].count()

each_podcast_total=podcast_count_train + podcast_count_test
test_perzent=podcast_count_test/each_podcast_total
print(f"The mean is：{test_perzent.mean()}  The standard deviation is：{test_perzent.std()}")


# Distribution of numerical columns
num_cols=train_df.select_dtypes(include='number').columns
num_cols=[col for col in num_cols if col.lower()!='id']
for col in num_cols:
    sns.histplot(train_df[col],bins=50,kde=True)
    plt.title(f"Distribution of{col}" )
    plt.show()



avg_listening_time=train_df.groupby('Podcast_Name')['Listening_Time_minutes'].mean()
avg_listening_time=avg_listening_time.sort_values(ascending=False)
plt.figure(figsize=(16,8))
sns.barplot(x=avg_listening_time.index,y=avg_listening_time.values)
plt.title('avg_listening_time')
plt.xlabel('Podcast_Name')
plt.ylabel('time(minutes)')
plt.tight_layout()
plt.show()


train_df['listening_percentage']=train_df['Listening_Time_minutes']/train_df['Episode_Length_minutes']
percentage_by_pocast=train_df.groupby('Podcast_Name')['listening_percentage'].mean()
percentage_by_pocast=percentage_by_pocast.sort_values(ascending=False)
plt.figure(figsize=(16,8))
sns.barplot(x=percentage_by_pocast.index,y=percentage_by_pocast.values)
plt.title('listening_percentage')
plt.xlabel('Podcast_Name')
plt.ylabel('listening_percentage')
#plt.tight_layout()
plt.show()



#Number of listeners per podcast
top_podcast=train_df['Podcast_Name'].value_counts()
plt.figure(figsize=(16,8))
sns.barplot(x=top_podcast.index,y=top_podcast.values)
plt.title('Top Podcast')
plt.xlabel('Podcast_Name')
plt.ylabel('linstener')
plt.tight_layout()
plt.show()



# Fluctuation of listening time of a single program
listen_stat=train_df.groupby('Podcast_Name')['Listening_Time_minutes'].agg(['mean','std']).reset_index()
plt.figure(figsize=(10,5))
sns.scatterplot(data=listen_stat, x='mean',y='std')
top=listen_stat.sort_values(by='mean',ascending=False).head(5)
for i in range(top.shape[0]):
    plt.text(x=top['mean'].iloc[i],y=top['std'].iloc[i],s=top['Podcast_Name'].iloc[i])
plt.title('Podcast listening Time: mean vs Satandard Deviation')
plt.xlabel('Aberage listening Time(minutes)')
plt.ylabel('Satandard Deviation ')
plt.grid(True)
plt.tight_layout()
plt.show()


#Listening rate of different columns
Genre =train_df.groupby('Genre')['Listening_Time_minutes'].agg(['count','mean']).reset_index()
plt.figure(figsize=(14,5))
sns.scatterplot(data=Genre, x='count',y='mean')
top=Genre.sort_values(by='count',ascending=False)
for i in range(top.shape[0]):
    plt.text(x=top['count'].iloc[i],y=top['mean'].iloc[i],s=top['Genre'].iloc[i])
plt.title('Podcast listening Time: listener vs Duaration')
plt.xlabel('linstener')
plt.ylabel('average listening Time ')
plt.grid(True)
plt.tight_layout()
plt.show()
print(top)


import scipy.stats as stats
from scipy.stats import f_oneway
grouped = train_df.groupby('Podcast_Name')['Listening_Time_minutes']
samples = [group for name, group in grouped]

f_stat, p_val =stats.f_oneway(*samples)
print(f'F-statistic：{f_stat:.4f}, P-value:{ p_val:.4f}')



from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey=pairwise_tukeyhsd(endog=train_df['Listening_Time_minutes'],
                        groups=train_df['Podcast_Name'],
                        alpha=0.05)



tukey.plot_simultaneous(figsize=(10,12))
plt.show()



# Joke_Junction
Joke_Junction=train_df.loc[train_df['Podcast_Name']=='Joke Junction']
plt.figure(figsize=(14,5))
#sns.scatterplot(data=train_df, x=train_df['Episode_Length_minutes'],y=train_df['Listening_Time_minutes'],hue=train_df['Episode_Sentiment'])
sns.kdeplot(data=Joke_Junction,
            x='Episode_Length_minutes',
            y='Listening_Time_minutes', 
            fill=True,
            cmap='inferno',
           thresh=0.05,
           levels=100)
plt.title('Podcast lenth vs listening time')
plt.xlabel('Podcast lenth')
plt.ylabel('listening Time ')
plt.grid(True)
plt.tight_layout()
plt.show()



Study_Sessions=train_df.loc[train_df['Podcast_Name']=='Study Sessions']

plt.figure(figsize=(14,5))
#sns.scatterplot(data=train_df, x=train_df['Episode_Length_minutes'],y=train_df['Listening_Time_minutes'],hue=train_df['Episode_Sentiment'])
sns.kdeplot(data=Study_Sessions,
            x='Episode_Length_minutes',
            y='Listening_Time_minutes', 
            fill=True,
            cmap='inferno',
           thresh=0.05,
           levels=100)
plt.title('Podcast lenth vs listening time')
plt.xlabel('Podcast lenth')
plt.ylabel('listening Time ')
plt.grid(True)
plt.tight_layout()
plt.show()


# episode16 
episode16=Study_Sessions.loc[Study_Sessions['Episode_Title']=='Episode 16']
sns.kdeplot(data=episode16,
            x='Episode_Length_minutes',
            y='Listening_Time_minutes', 
            fill=True,
            cmap='inferno',
           thresh=0.05,
           levels=100)
plt.title('Podcast lenth vs listening time')
plt.xlabel('Podcast lenth')
plt.ylabel('listening Time ')
plt.grid(True)
plt.tight_layout()
plt.show()


# Release time vs. listening length
time_listeninglength=train_df.groupby(['Publication_Day','Publication_Time'])['Listening_Time_minutes'].mean().unstack()
plt.figure(figsize=(16,8))
sns.heatmap(time_listeninglength, annot=True)
plt.title('Release time vs. listening length')
plt.show()


# Release day and listening length
daylisteninglength=train_df.groupby(['Podcast_Name','Publication_Day'])['Listening_Time_minutes'].mean().unstack()
plt.figure(figsize=(16,8))
sns.heatmap(data=daylisteninglength, annot=True)
plt.title('Release day and listening length')

plt.show()



# Average listening time of each program released at different time periods
top_podcast_total=train_df.groupby(['Podcast_Name','Publication_Time'])['Listening_Time_minutes'].mean().unstack()
plt.figure(figsize=(16,8))
sns.heatmap(data=top_podcast_total, annot=True)
plt.title('Average listening time of each program released at different time periods')

plt.show()


# The impact of different sentiment on listening time
top_podcast_total=train_df.groupby(['Podcast_Name','Episode_Sentiment'])['Listening_Time_minutes'].mean().unstack()
plt.figure(figsize=(16,8))
sns.heatmap(data=top_podcast_total, annot=True)
plt.title('The impact of different sentiment on listening time')

plt.show()


test_df['Episode_Length_minutes']=test_df.groupby(['Podcast_Name','Episode_Title'])['Episode_Length_minutes'].transform(lambda x: x.fillna(x.mean()))
test_df['Guest_Popularity_percentage']=test_df['Guest_Popularity_percentage'].fillna(0)
test_df['Number_of_Ads']=test_df['Number_of_Ads'].fillna(0)



train_df['num_listener']=train_df.groupby('Podcast_Name')['Podcast_Name'].transform('count')
popularity_map=train_df[['Podcast_Name','num_listener']].drop_duplicates()
test_df=test_df.merge(popularity_map,on='Podcast_Name',how='left')

def Podcast_type(row):
    if row['Number_of_Ads']>=10:
        return 'ads'
    else:
        return 'normal'

train_df['Podcast_type']=train_df.apply(Podcast_type,axis=1)
test_df['Podcast_type']=test_df.apply(Podcast_type,axis=1)



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# data preprocessing
#label encoding

la_podcast=LabelEncoder()
#la_episode=LabelEncoder()
la_genre=LabelEncoder()
la_pubd=LabelEncoder()
la_pubt=LabelEncoder()
la_episen=LabelEncoder()
p_type=LabelEncoder()

train_df['Podcast_label']=la_podcast.fit_transform(train_df['Podcast_Name'])
#train_df['Episode_label']=la_podcast.fit_transform(train_df['Podcast_Name']+'_'+train_df['Episode_Title'])
train_df['Genre_label']=la_genre.fit_transform(train_df['Genre'])
train_df['PubD_label']=la_pubd.fit_transform(train_df['Publication_Day'])
train_df['PubT_label']=la_pubt.fit_transform(train_df['Publication_Time'])
train_df['EpiSen_label']=la_episen.fit_transform(train_df['Episode_Sentiment'])
train_df['PT_label']=p_type.fit_transform(train_df['Podcast_type'])
    



test_df['Podcast_label']=la_podcast.transform(test_df['Podcast_Name'])
#test_df['Episode_label']=la_podcast.transform(test_df['Podcast_Name']+'_'+test_df['Episode_Title'])
test_df['Genre_label']=la_genre.transform(test_df['Genre'])
test_df['PubD_label']=la_pubd.transform(test_df['Publication_Day'])
test_df['PubT_label']=la_pubt.transform(test_df['Publication_Time'])
test_df['EpiSen_label']=la_episen.transform(test_df['Episode_Sentiment'])
test_df['PT_label']=p_type.transform(test_df['Podcast_type'])



def comb_label_encode(train_1,test_1,columns,new_col_name,unknow_id=-1):
    train_comb=train_1[columns].astype(str).agg('_'.join,axis=1)
    test_comb=test_1[columns].astype(str).agg('_'.join,axis=1)
    label_map={val: idx for idx,val in enumerate(train_comb.unique())}
    train_1[new_col_name]=train_comb.map(label_map)
    test_1[new_col_name]=test_comb.map(label_map).fillna(-1).astype(int)
    return train_1,test_1,label_map

train_df,test_df,Episode_label = comb_label_encode(train_df,test_df,
                                                   columns=['Podcast_Name','Episode_Title'],
                                                   new_col_name='Episode_label',unknow_id=-1)

    
    


subset_cols_1=['Listening_Time_minutes','Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads','Podcast_label',
            'Genre_label',	'PubD_label','PubT_label','EpiSen_label','PT_label','num_listener']
sns.heatmap(train_df[subset_cols_1].corr(),annot=True,fmt='.2f',cmap='coolwarm')



train=train_df[subset_cols_1]
test=test_df[['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads','Podcast_label',
            'Genre_label',	'PubD_label','PubT_label','EpiSen_label','PT_label','num_listener']]


train.head()


X=train.drop(columns='Listening_Time_minutes',axis=1)
y=train['Listening_Time_minutes']
X_train, X_val, y_train, y_val  = train_test_split(X,y, test_size=0.2, random_state=778) 


X_train.shape


test.shape


X_train


y_train


from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu', input_shape=[12]),
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(4, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1),
])

model.compile(
    optimizer='adam',
    loss='mae',
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.01,
    restore_best_weights=True,)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=10000,
    epochs=40,
    verbose=0,
)

# Show the learning curves
history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot();


y_pre_val=model(X_val).numpy()
y_pre_val=y_pre_val.flatten()
sns.histplot(x=y_val,color='blue',kde=True)
sns.histplot(x=y_pre_val,color='orange',kde=True)

#plt.scatter(range(len(y_pre_val)),y_pre_val,color='red',s=10,alpha=0.6)
plt.xlabel('Actual listening time distribution VS Model prediction results')

plt.grid(True)
plt.show()



prediction=model.predict(test)


prediction=np.round(prediction,3)
prediction=prediction.flatten()
prediction.shape


prediction[:10]


output = pd.DataFrame({'id': test_df.id, 'Listening_Time_minutes': prediction})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

