import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore")
plt.style.use('dark_background')


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df


df.info()


df.describe()


df.drop(['id','day'],axis=1,inplace=True)
df


plt.figsize=(12,12)
sns.heatmap(df.corr(),cmap='cividis',annot=True,annot_kws={'size': 7})
plt.show()


sns.pairplot(df)
plt.show()


correlations = df.corr()['rainfall'].drop('rainfall').dropna().sort_values()
colors = correlations.apply(lambda x: 'red' if x<=0 else 'green')

plt.figure(figsize=(5,4))
correlations.plot(kind='bar',color=colors)
plt.xlabel('Features')
plt.ylabel('Correlation with rainfall')
plt.show()


from matplotlib.colors import ListedColormap
cmap = ListedColormap(['orange','lightblue'])
scatter = plt.scatter(data=df,x='cloud',y='sunshine',c='rainfall',cmap=cmap)
plt.xlabel('Cloud')
plt.ylabel('Sunshine')
cbar = plt.colorbar(scatter,ticks=[0,1])
cbar.ax.set_yticklabels(['No Rain','Rain'])
plt.show()


df['temp_range'] = df['maxtemp']-df['mintemp']
df.drop(['maxtemp','mintemp'],axis=1,inplace=True)
sns.heatmap(df.corr(),cmap='cividis',annot=True,annot_kws={'size': 7})
plt.show()


df


from sklearn.model_selection import train_test_split
X = df.drop('rainfall',axis=1)
y = df['rainfall']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = pd.DataFrame(sc.fit_transform(X_train),columns=X_train.columns)
X_test_scaled = pd.DataFrame(sc.transform(X_test),columns=X_test.columns)


X_train_scaled


X_test_scaled


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Multi-layer Perceptron': MLPClassifier()
}


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
param_grids = {
    'Logistic Regression':{'C':[0.1,1,10],'solver':['liblinear','lbfgs']},
    'Support Vector Machine':{'C':[0.1,1,10],'kernel':['linear','rbf']},
    'K-Nearest Neighbors':{'n_neighbors':[3,5,7],'weights':['uniform','distance']},
    'Decision Tree':{'max_depth':[None,5,10],'min_samples_split':[2,5,10]},
    'Random Forest':{'n_estimators':[50,100,200],'max_depth':[None,5,10]},
    'Gradient Boosting':{'n_estimators':[50,100,200],'learning_rate':[0.01,0.1,1]},
    'Gaussian Naive Bayes':{},
    'Linear Discriminant Analysis':{'solver':['svd','lsqr','eigen']},
    'Quadratic Discriminant Analysis':{},
    'Multi-layer Perceptron':{'hidden_layer_sizes':[(50,),(100,)],'activation':['relu','tanh']}
}


best_models = {}

for model_name, model in models.items():
    print(f"Performing GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(estimator=model,param_grid=param_grids[model_name],cv=5,scoring='accuracy')
    grid_search.fit(X_train_scaled,y_train)

    best_models[model_name] = grid_search.best_estimator_

    print(f"Best parameters for {model_name} : {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {model_name} : {grid_search.best_score_:.4f}")

    y_pred = grid_search.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test,y_pred)
    print(f"Test set accuracy for {model_name} : {test_accuracy:.4f}")
    print("-"*25)


rf = RandomForestClassifier(max_depth = 10, n_estimators = 100)
rf.fit(X_train_scaled,y_train)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_df


test_df.info()


test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].mean())


test_df.info()


X_test_df = test_df.copy()
X_test_df.drop(['id','day'],axis=1,inplace=True)
X_test_df['temp_range'] = X_test_df['maxtemp']-X_test_df['mintemp']
X_test_df.drop(['maxtemp','mintemp'],axis=1,inplace=True)
X_test_df


X_test_df_scaled = pd.DataFrame(sc.transform(X_test_df),columns=X_test_df.columns)
X_test_df_scaled


y_pred = rf.predict(X_test_df_scaled)
y_pred


sol = pd.DataFrame({'id':test_df['id'],'rainfall':y_pred})
sol


#sol.to_csv('Sumbission2.csv',index=False)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


model = Sequential()
model.add(Dense(256,activation='relu',input_shape=(9,)))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))

model.summary()


model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])


history = model.fit(X_train_scaled,y_train,epochs=200,batch_size=32,validation_split=0.2,verbose=1)


test_loss,test_accuracy = model.evaluate(X_test_scaled,y_test,verbose=0)
test_accuracy


preds = model.predict(X_test_df_scaled)
#preds = (preds>0.5).astype(int)
preds


sol = pd.DataFrame({'id':test_df['id'],'rainfall':preds.flatten()})
sol


sol.to_csv('Submission4.csv',index=False)

