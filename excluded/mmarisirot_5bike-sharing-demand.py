import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error



def cross_val_test(model, dataX, dataY, model_type='regression', n_cv=5):

    cv = KFold(n_splits=n_cv, shuffle=True)
    if model_type == 'regression':
        scorer = {'r2':make_scorer(r2_score),
                'mae': make_scorer(mean_absolute_error),
                'mse': make_scorer(mean_squared_error),
                'mape': make_scorer(mean_absolute_percentage_error)}
    if model_type == 'classification':
        scorer = {'f1':make_scorer(f1_score, average='micro'),
          }
    scores = cross_validate(model, dataX, dataY, scoring=scorer, cv=cv, return_train_score=True)
    return scores


df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
df.head()


df.info()


df['datetime'] = pd.to_datetime(df['datetime'])

df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day_of_month'] = df['datetime'].dt.day
df['day_of_week'] = df['datetime'].dt.day_of_week
df['hour'] = df['datetime'].dt.hour
df.drop(['datetime', 'casual','registered'], axis = 1, inplace = True)

df.dtypes


df.describe()


df.duplicated().sum()



num_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'count']

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Розподіл {col}')
    plt.show()


plt.figure(figsize=(8, 4))
sns.boxplot(x='season', y='count', data=df)
plt.title('Розподіл кількості оренди по сезонах')
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x='hour', y='count', data=df)
plt.title('Розподіл по годинах')
plt.show()


corr = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Кореляційна матриця")
plt.show()





df['demand_level'] = pd.cut(df['count'],
                            bins=[-1, df['count'].median(), df['count'].max()],
                            labels=['dark_period', 'light_period'])

print(df['demand_level'].value_counts())

X = df.drop(['count', 'demand_level'], axis=1)
y = df['demand_level']


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, random_state=1
)

clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=1)
clf.fit(X_train, y_train)


cross_val_test(clf, X_train, y_train, model_type='classification')


plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=X_train.columns,
          class_names=clf.classes_,
          filled=True, 
          rounded=True, 
          max_depth=3)
plt.title("Візуалізація дерева рішень", fontsize=16)
plt.show()


from sklearn.model_selection import GridSearchCV
from sklearn.inspection import DecisionBoundaryDisplay
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_leaf': [1, 5, 10],
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)


X_vis = X_train[['hour', 'atemp']].sample(frac=0.3, random_state=1)
y_vis = y_train.loc[X_vis.index]

clf_vis = DecisionTreeClassifier(max_depth=3, random_state=1)
clf_vis.fit(X_vis, y_vis)

disp = DecisionBoundaryDisplay.from_estimator(
    clf_vis,
    X_vis,
    response_method="predict",
    xlabel='hour',
    ylabel='atemp',
    alpha=0.5,
    cmap=plt.cm.coolwarm
)
disp.ax_.scatter(
    X_vis['hour'], 
    X_vis['atemp'], 
    c=y_vis.cat.codes, 
    edgecolor="k", 
    cmap=plt.cm.coolwarm
)

plt.title("Decision Boundary of Decision Tree Classifier")
plt.show()


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)


cross_val_test(clf, X_train, y_train, model_type='classification')

