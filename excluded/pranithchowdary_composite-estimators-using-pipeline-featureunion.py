import pandas as pd


!unzip /kaggle/input/spooky-author-identification/train.zip


horror_train_data = pd.read_csv('/kaggle/working/train.csv')


horror_train_data.head()


!unzip /kaggle/input/spooky-author-identification/test.zip


horror_test_data= pd.read_csv('/kaggle/working/test.csv')


horror_test_data.info()


horror_train_data.info()


horror_train_data = horror_train_data[['text','author']]


from sklearn.pipeline import make_pipeline


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


pipelines = []
for model in [LogisticRegression(), DecisionTreeClassifier(), MultinomialNB(), LinearSVC()]:
    pipeline = make_pipeline(
              CountVectorizer(stop_words='english'),
              TfidfTransformer(),
              model)
    pipelines.append(pipeline)


pipelines[1].steps[2]


from sklearn.model_selection import train_test_split


trainX,testX,trainY,testY = train_test_split(horror_train_data.text, horror_train_data.author)


for pipeline in pipelines:
    pipeline.fit(trainX, trainY)


for pipeline in pipelines:
    print (pipeline.score(testX, testY))


horror_test_data.info()


results = []
for pipeline in pipelines:
    result = pipeline.predict(horror_test_data.text)
    results.append(result)


results


pipelines[0].steps[0][1].transform(horror_test_data.text)


from sklearn.model_selection import GridSearchCV


svc_pipe =  make_pipeline(
              CountVectorizer(stop_words='english'),
              TfidfTransformer(),
              LinearSVC())


dt_pipe = make_pipeline(
              CountVectorizer(stop_words='english'),
              TfidfTransformer(),
              DecisionTreeClassifier())


svc_pipe


svc_pipe.steps


import numpy as np
params = {
    'linearsvc__C': list(np.logspace(1,20,20))
}


dt_pipe.steps[2]


params = {
    'countvectorizer__max_features':[5000,7500,10000],
    'decisiontreeclassifier__max_depth':[100,200]
}


gs = GridSearchCV(dt_pipe,cv=5,param_grid=params, n_jobs=-1)


gs.fit(trainX,trainY)


gs.best_params_


gs.best_score_


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


boston = load_boston()


X = boston.data


y = boston.target


regressor = LinearRegression()


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


regressor.fit(X_train, y_train)


print('R2 score: {0:.2f}'.format(regressor.score(X_test, y_test)))


pred = regressor.predict(X_test)


from sklearn.metrics import mean_absolute_error, r2_score


mean_absolute_error(y_pred=pred, y_true=y_test)


from sklearn.preprocessing import PowerTransformer,QuantileTransformer


pt = PowerTransformer()


qt = QuantileTransformer(output_distribution='normal')


#X_tf = pt.fit_transform(X)
#OR
X_tf = qt.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_tf, y, random_state=0)


regressor = LinearRegression()


regressor.fit(X_train, y_train)


print('R2 score: {0:.2f}'.format(regressor.score(X_test, y_test)))


pred = regressor.predict(X_test)


mean_absolute_error(y_pred=pred, y_true=y_test)


from sklearn.compose import TransformedTargetRegressor


regr = TransformedTargetRegressor(regressor=regressor,transformer=qt)


regr.fit(X_train, y_train)


pred = regr.predict(X_test)


mean_absolute_error(y_pred=pred, y_true=y_test)


r2_score(y_pred=pred, y_true=y_test)


emp_data = pd.read_csv('https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/HR_comma_sep.csv.txt')


emp_data.head()


emp_data.rename(columns={'sales':'dept'}, inplace=True)


num_cols = ['number_project','average_montly_hours','time_spend_company']


bin_cols = ['Work_accident','promotion_last_5years']


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, LabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self,key):
        self.key = key
        
    def fit(self,X,Y=None):
        return self
    
    def transform(self,X,Y=None):
        return X[self.key]


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)


pipeline_dept = Pipeline([
    ('selector', ItemSelector('dept')),
    ('lb', MyLabelBinarizer()),
])


pipeline_dept.fit_transform(emp_data)


class MultiItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self,keys):
        self.keys = keys
        
    def fit(self,X,Y=None):
        return self
    
    def transform(self,X,Y=None):
        return X[self.keys]


class SalaryMapper(BaseEstimator, TransformerMixin):
    
    def fit(self,X,Y=None):
        return self
    
    def transform(self,X,Y=None):
        db = {'low':1,'medium':2,'high':3}
        print (type(X))
        r = X.str.strip().replace(db)
        return r.values.reshape(-1,1)


pipeline_salary = Pipeline([
    ('selector',ItemSelector('salary')),
    ('sm',SalaryMapper())
])


pipeline_numbers = Pipeline([
    ('selector',MultiItemSelector(num_cols)),
    ('scaling', MinMaxScaler())
])


pipeline_bin = Pipeline([
    ('selector',MultiItemSelector(bin_cols))
])


fu = FeatureUnion([
    ('dept_pipe',pipeline_dept),
    ('salary_pipe',pipeline_salary),
    ('numbers_pipe',pipeline_numbers),
    ('bin_pipe',pipeline_bin)
])


pipeline = Pipeline([
    ('union',fu),
    #('feature_selector',SelectKBest(k=15)),
    ('classifier',RandomForestClassifier(n_estimators=10))
])


from sklearn.model_selection import train_test_split


trainX,testX, trainY,testY = train_test_split(emp_data.drop('left',axis=1), emp_data.left)


pipeline.fit(trainX,trainY)


pipeline.predict(testX)


pipeline.score(testX,testY)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


titanic_data = pd.read_csv('https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/titanic-train.csv.txt', index_col='PassengerId')


titanic_data.head()


num_cols = ['Age','Fare']
cat_cols = ['Embarked','Sex','Pclass']


pipeline_num = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaling',StandardScaler())
])


pipeline_cat = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoding', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', pipeline_num, num_cols),
        ('cat', pipeline_cat, cat_cols)])


pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                ('classifier',RandomForestClassifier(n_estimators=10))])


X = titanic_data.drop('Survived',axis=1)


Y = titanic_data.Survived


trainX,testX,trainY,testY = train_test_split(X,Y)


pipeline.fit(trainX,trainY)


pipeline.score(testX,testY)


pipeline.steps


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__n_estimators': [10,15,20],
}


from sklearn.model_selection import GridSearchCV


grid_search = GridSearchCV(pipeline, param_grid, cv=5, iid=False)
grid_search.fit(trainX,trainY)


grid_search.score(testX,testY)


grid_search.best_params_

