import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')


df.head()


df.shape


df.isnull().sum()


df.info()


#distribution of podcast listening time
plt.figure(figsize=(12, 6))
sns.histplot(df['Listening_Time_minutes'], bins=30, kde=True, color='purple')
plt.title('Distribution of Podcast Listening Time', fontsize=16, fontweight='bold')
plt.xlabel('Listening Time (minutes)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


#impact of ads on listening time
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Number_of_Ads', y='Listening_Time_minutes', palette='coolwarm')
plt.title('Impact of Ads on Listening Time', fontsize=16, fontweight='bold')
plt.xlabel('Number of Ads', fontsize=14)
plt.ylabel('Listening Time (minutes)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


#podcast genre distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Genre', palette='viridis', order=df['Genre'].value_counts().index)
plt.title('Podcast Genre Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Number of Podcasts', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#episode length vs. listening time
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Episode_Length_minutes', y='Listening_Time_minutes', color='orange')
plt.title('Episode Length vs. Listening Time', fontsize=16, fontweight='bold')
plt.xlabel('Episode Length (minutes)', fontsize=14)
plt.ylabel('Listening Time (minutes)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


#number of episodes published by day of the week
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Publication_Day', palette='pastel', order=df['Publication_Day'].value_counts().index)
plt.title('Number of Episodes Published by Day of the Week', fontsize=16, fontweight='bold')
plt.xlabel('Day of the Week', fontsize=14)
plt.ylabel('Number of Episodes', fontsize=14)
plt.xticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#correlation heatmap
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Numeric Features', fontsize=16, fontweight='bold')
plt.show()


#filling the missing values in Guest_Popularity_percentage and Episode_Length_minutes with the median
df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median(), inplace=True)

#filling the missing value in Number_of_Ads with the mode
df['Number_of_Ads'].fillna(df['Number_of_Ads'].mode()[0], inplace=True)


#dropping the columns that are not used in the model
df.drop(columns=['Episode_Title', 'Publication_Time', 'id'], inplace=True)


#visualizing outliers
columns_for_outlier = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                         'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']

sns.set(style="whitegrid")
plt.figure(figsize=(16, 10))
for i, col in enumerate(columns_for_outlier):
    plt.subplot(2, 3, i+1) 
    sns.boxplot(data=df, y=col)
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()


#identifying outliers
def summarize_outliers(data, columns):
    outlier_summary = {}
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        
        outlier_summary[column] = {
            'num_outliers': len(outliers),
            'outlier_values': outliers[column].values
        }
    return outlier_summary

outlier_summary = summarize_outliers(df, columns_for_outlier)
for col, summary in outlier_summary.items():
    print(f"{col}:")
    print(f"  Number of Outliers: {summary['num_outliers']}")
    print(f"  Outlier Values: {summary['outlier_values']}")
    print("\n")


#removing outlier for Episode_Length_minutes
df = df[df['Episode_Length_minutes'] != 325.24]

#removing outliers for Number_of_Ads
outlier_threshold = df['Number_of_Ads'].quantile(0.75) + 1.5 * (df['Number_of_Ads'].quantile(0.75) - df['Number_of_Ads'].quantile(0.25))
df = df[df['Number_of_Ads'] <= outlier_threshold]


#one-hot encoding
df=pd.get_dummies(df,drop_first=True)


#adding new features
df['Interaction'] = df['Host_Popularity_percentage'] * df['Episode_Length_minutes']

#new feature that is the square of Episode_Length_minutes
df['Episode_Length_minutes_squared'] = df['Episode_Length_minutes'] ** 2


#features and target
x = df.drop('Listening_Time_minutes', axis=1)
y = df['Listening_Time_minutes']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#defining the model architecture
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

#compiling the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MeanAbsoluteError'])


#early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)


#training the model
history = model.fit(x_train,y_train,epochs=20,batch_size=32,validation_split=0.2,callbacks=[early_stopping],verbose=0)


model.summary()


pred = model.predict(x_test)
rmse = mean_squared_error(y_test, pred, squared=False)


rmse


#RMSE over epochs
rmse_per_epoch = [val_loss ** 0.5 for val_loss in history.history['val_loss']]
plt.figure(figsize=(10, 6))
plt.plot(rmse_per_epoch, label='Validation RMSE', color='blue', lw=2)
plt.title("RMSE Over Epochs", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.grid(alpha=0.3)
plt.legend(fontsize=10, shadow=True)
plt.tight_layout()
plt.show()


test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

#preprocessing test data
submission = pd.DataFrame({'id': test['id']})
test = test.drop(columns=['id', 'Episode_Title', 'Publication_Time'])

#one-hot encoding the test data and align columns
test = pd.get_dummies(test, drop_first=True)
test = test.reindex(columns=x.columns, fill_value=0)

#filling the missing values in Guest_Popularity_percentage and Episode_Length_minutes with the median
test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(), inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].median(), inplace=True)

#filling the missing value in Number_of_Ads with the mode
test['Number_of_Ads'].fillna(test['Number_of_Ads'].mode()[0], inplace=True)

#adding new features
df['Interaction'] = df['Host_Popularity_percentage'] * df['Episode_Length_minutes']

#new feature that is the square of Episode_Length_minutes
df['Episode_Length_minutes_squared'] = df['Episode_Length_minutes'] ** 2

#scaling
test_scaled = scaler.transform(test)


#making predictions on test data
predictions = model.predict(test_scaled)

#adding predicted values to df and create submission file
submission['Listening_Time_minutes'] = predictions.flatten()
submission.to_csv('submission.csv', index=False)

