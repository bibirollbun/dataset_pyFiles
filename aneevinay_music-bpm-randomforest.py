import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


train= pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test= pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample= pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train.shape


test.shape


train.head()


test.head()


train.info()


train.isnull().sum()


train.duplicated().sum()


train['BeatsPerMinute'].skew()


train.hist(['BeatsPerMinute'],bins=30,color='red',edgecolor='black')
plt.title('Histogram of BeatsPerMinute')
plt.show()


sns.boxplot(x='BeatsPerMinute',data=train,color='violet')


train.describe().loc[['mean','min','max']].T   # Teranspose the table


corr = train.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

plt.savefig('Correlation Heatmap')


corr['BeatsPerMinute'].sort_values(ascending=False)


train.hist(figsize=(20,15), bins=30, color='Red', edgecolor='black')
plt.suptitle("Histogram of Features")

plt.savefig('Histogram of features')


X=train.drop(['id','BeatsPerMinute'],axis=1)
y=train['BeatsPerMinute']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)


model1=RandomForestRegressor(n_estimators=120, 
                             max_depth=15,
                             max_features='sqrt',
                             min_samples_split=5,
                             min_samples_leaf=2,
                             n_jobs=-1,
                             random_state=10)


model1.fit(X_train,y_train)


importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model1.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(8,5))
plt.barh(importance['Feature'], importance['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance (RandomForest)")
plt.show()


y_pred=model1.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse


test_ids = test["id"]
model1.fit(X, y)


X_test_final = test.drop(columns=["id"])
y_pred_final = model1.predict(X_test_final)
submission = pd.DataFrame({
    "id": test_ids,
    "bpm": y_pred_final
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv is ready!")




