import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df.head()


df_test.head()


df.shape


df.columns


df.drop(columns=['id'],inplace=True)


df.info()


df.describe()


df.duplicated().sum()


plt.figure(figsize=(8,6))
plt.title('Distribution of accident_risk',fontweight='bold',fontsize=14)
sns.histplot(data=df, x = "accident_risk",bins = 60, color='skyblue',edgecolor='black',linewidth=1.4,stat='density')
sns.kdeplot(data=df, x='accident_risk',color='red',linewidth=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('accident_risk',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.show()


avg_lighting_risk = df.groupby('lighting')['accident_risk'].mean().reset_index()
sizes = avg_lighting_risk['accident_risk'].values
labels = avg_lighting_risk['lighting'].values
colors = ['gold', 'orange', 'blue']  
plt.figure(figsize= (8,6))
plt.title("Accident Risk based on Lighting",fontsize=14,fontweight="bold")
plt.pie(sizes,labels = labels,autopct='%1.1f%%',colors = colors, pctdistance=.75, labeldistance=1.1)
plt.axis('equal')
plt.show()


plt.title('Correlation Heatmap',fontsize=14,fontweight='bold')
sns.heatmap(df.corr(numeric_only=True),annot=True,fmt='.2f')
plt.show()


avg_road_type = df.groupby('road_type')['accident_risk'].mean().reset_index()
plt.figure(figsize=(8,6))
plt.title("Accident Risk by Road Type",fontweight='bold',fontsize = 14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylabel('Road Type',fontsize=12)
plt.xlabel('Average accident_risk',fontsize=12)
plt.barh(avg_road_type['road_type'],avg_road_type['accident_risk'],edgecolor='black',color='violet')
plt.show()


avg_weather_risk_sorted=df.groupby('num_reported_accidents')['accident_risk'].mean().reset_index().sort_values('accident_risk',ascending=False)
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=avg_weather_risk_sorted,
    x='accident_risk',
    y='num_reported_accidents',
    s = 200,
    color = 'purple'
)
for risk in avg_weather_risk_sorted['accident_risk']:
    plt.axvline(x=risk, color='grey', linestyle='--', linewidth=0.5, zorder=0)
plt.title('Average Accident Risk by Number of Reported accidents', fontsize=14, fontweight='bold')
plt.xlabel('Average Accident Risk', fontsize=12)
plt.ylabel('Number of accidents', fontsize=12)
plt.xlim(0, avg_weather_risk_sorted['accident_risk'].max() * 1.1) 
plt.grid(axis='x', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()


def category(x):
    if x>0.66:
        return "High"
    elif x<=0.66 and x>0.33:
        return "Medium"
    else:
        return "Low"

df['curvature_type'] = df['curvature'].apply(category)
avg_curvature_risk = df.groupby('curvature_type')['accident_risk'].mean().reset_index().sort_values('accident_risk',ascending=False)
plt.barh(avg_curvature_risk['curvature_type'],avg_curvature_risk['accident_risk'],edgecolor='black',color='violet')
plt.title("Accident Risk based on Curvature Type",fontweight='bold',fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylabel('Curvature Type',fontsize=12)
plt.xlabel('Average accident_risk',fontsize=12)
plt.show()
df.drop(columns = ['curvature_type'],inplace=True)


avg_time_risk = df.groupby('time_of_day')['accident_risk'].mean().reset_index().sort_values('accident_risk',ascending=False)
plt.barh(avg_time_risk['time_of_day'],avg_time_risk['accident_risk'],edgecolor='black',color='violet')
plt.title("Accident Risk based on Time of Day",fontweight='bold',fontsize=14)
plt.ylabel('Time of Day',fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('Average accident_risk',fontsize=12)
plt.show()


from sklearn.preprocessing import StandardScaler,LabelEncoder
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


cols = ['road_type','lighting','weather','time_of_day']

for col in cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    df_test[col] = le.transform(df_test[col])


x_train = df.iloc[:,0:12]
y_train = df['accident_risk']
x_test =df_test.iloc[:,1:]


ss = StandardScaler()
x_train_s = ss.fit_transform(x_train)
x_test_s = ss.transform(x_test)


model = Sequential([
    Dense(50,activation = 'relu',input_dim=x_train_s.shape[1]),
    Dense(20, activation = 'relu'),
    Dense(1,activation = 'sigmoid')]
)


model.summary()


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


model.compile(optimizer = 'adam',loss=rmse,metrics = ['mae',tf.keras.metrics.RootMeanSquaredError(name='rmse')])


model.fit(x_train_s,y_train,epochs = 10,validation_split = 0.2)


predictions = model.predict(x_test_s)


submission = pd.DataFrame({
    'PassengerId': df_test['id'],
    'Survived': predictions.ravel()
})
submission.to_csv('submission.csv', index=False)
print("submission.csv created!")

