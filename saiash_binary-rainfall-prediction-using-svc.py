import numpy as np 
import pandas as pd


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


df.info()


df.head()


df['day'].nunique()


df.columns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[['pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection','windspeed']]
y = df['rainfall']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=1)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}
rf = RandomForestClassifier(random_state=4)
# grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

grid.fit(x_train_scaled,y_train)


print(f"Best Parameters: {grid.best_params_}")
print(f"Best Score: {grid.best_score_}")


predic = grid.predict(x_test_scaled)
print(classification_report(y_test,predic))
print(confusion_matrix(y_test, predic))


dft = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
dft.head()


dft.info()


dft['winddirection'] = dft['winddirection'].fillna(df['winddirection'].mean())


idd = dft['id']
xt = dft[[ 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']]
xt_scaled = scaler.transform(xt)


sub = grid.predict(xt_scaled)


submi = pd.DataFrame({
    'id':idd,
    'rainfall':sub
})


submi = submi.set_index('id')


submi.to_csv("rainfall_submit.csv")

