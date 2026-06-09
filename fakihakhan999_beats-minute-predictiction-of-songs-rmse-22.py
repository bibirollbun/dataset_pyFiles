import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score

import warnings
warnings.filterwarnings('ignore')


train= pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test= pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample= pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train.info()


test.info()


train.head()


print(train.isnull().sum())
print(test.isnull().sum())


print(train.duplicated().sum())
print(test.duplicated().sum())


test.head()


# removing Id as it is unique identifier

train.drop('id',axis=1,inplace =True)
test.drop('id',axis=1,inplace =True)


train.columns


for col in list(train.columns):
    print(f'\n {col} has Unique Values :\t  {train[col].nunique()}')


plt.figure(figsize=(12,10))
inx=1
for col in list(train.columns):
    plt.subplot(3,4,inx)
    sns.histplot(x=col,data=train)
    inx=inx+1

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,10))
inx=1
for col in list(train.columns):
    plt.subplot(3,4,inx)
    sns.boxplot(x=col,data=train)
    inx=inx+1

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,10))
inx=1
for col in list(test.columns):
    plt.subplot(3,3,inx)
    sns.boxplot(x=col,data=test)
    inx=inx+1

plt.tight_layout()
plt.show()


train.columns


test.columns


for col in ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood','TrackDurationMs']:
    train[col] = winsorize(train[col], limits=[0.1, 0.1]) 
    test[col] = winsorize(test[col], limits=[0.1, 0.1]) 

train['BeatsPerMinute'] = winsorize(train['BeatsPerMinute'], limits=[0.1, 0.1]) 


plt.figure(figsize=(12,10))
inx=1
for col in list(train.columns):
    plt.subplot(3,4,inx)
    sns.boxplot(x=col,data=train)
    inx=inx+1

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,10))
inx=1
for col in list(test.columns):
    plt.subplot(3,3,inx)
    sns.boxplot(x=col,data=test)
    inx=inx+1

plt.tight_layout()
plt.show()


train.describe().loc[['mean','min','max']].T   


corr = train.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

plt.savefig('Correlation Heatmap')


X=train.drop(['BeatsPerMinute'],axis=1)
y=train['BeatsPerMinute']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)

sc = MinMaxScaler()
df_scaled = train.copy()

sc.fit(x_train) # picks max value fromx train
sc_x_train=sc.transform(x_train) # scales all x_test and train wih max of x_train
sc_x_test=sc.transform(x_test)


models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=10),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}



results = []

for name, model in models.items():
    model.fit(sc_x_train, y_train)
    y_pred = model.predict(sc_x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

# --- Put results into DataFrame ---
results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
print(results_df)

