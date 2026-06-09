# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder,StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import log_loss, make_scorer
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head(2)


test.head(2)


class NewNumericalFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, columns = None):
        self.columns = columns
        return columns

    def fit(self,X, y= None):
        return self

    def transform(self, X):
        X_new = pd.DataFrame(index = X.index)
        num_cols = [col for col in X.select_dtypes(include = np.number).columns if col not in ['id']]
        # non linear transformation like Sqrt, Log, Exp
        for i in num_cols:
            X_new[f'Sqrt_{i}'] = np.sqrt(X[i])
            X_new[f'Log_{i}'] = np.log(X[i]+1)
            X_new[f'Exp_{i}'] = np.exp(X[i])
        #realtive position features
        features = []
        features_names = []
        
        for i in range(len(X_new.columns)):
            for j in range(i+1, len(X_new.columns)):
                col1, col2 = X_new.columns[i],X_new.columns[j]
                diff = X_new[col1] - X_new[col2]
                mod = X_new[col1] % (X_new[col2] + 1e-6)
                ratio = X_new[col1] / (X_new[col2] + 1e-6)
                
                features.append(diff)
                features_names.append(f"{col1}_minus_{col2}")

                features.append(ratio)
                features_names.append(f'{col1}_div_{col2}')

                features.append(mod)
                features_names.append(f'{col1}_mod_{col2}')
        X_relative = pd.DataFrame(index = X.index)
        X_relative = pd.concat(features, axis=1)
        X_relative.columns = features_names
        X_new = pd.concat([X_new, X_relative], axis = 1)
        X_final = np.hstack((X[num_cols].values,X_new.values))
        return X_final


X = train[[col for col in train.columns if col not in ['Fertilizer Name','id']]]
y = train['Fertilizer Name']


y.value_counts()/y.shape


num_columns = X.select_dtypes('number').columns
cat_columns = X.select_dtypes('object').columns


num_pipe = Pipeline([
    ('num_feat', FeatureUnion([('new_feat',NewNumericalFeatures()),('poly',PolynomialFeatures(degree=3, interaction_only=True, include_bias=False))])),
    ('scaler',StandardScaler())
])


col_transformer = ColumnTransformer(
    remainder = 'passthrough',
    transformers = [
        ('num', num_pipe, num_columns),
        ('cat',OneHotEncoder(), cat_columns )
    ]
)


col_transformer


X_transformed = col_transformer.fit_transform(X)
test_transformed = col_transformer.fit_transform(test[[col for col in test.columns if col not in ['id']]])


X_transformed.shape, test_transformed.shape


X_train , X_test ,y_train, y_test = train_test_split(X_transformed, y ,test_size = 100000,train_size= 100000, stratify = y, random_state = 42)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


model = CatBoostClassifier(loss_function='MultiClass', verbose=0,thread_count=2)


param_dist = {
    'iterations': [200, 500],
    'learning_rate': [0.05, 0.1],
    'depth': [4, 6],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64],
    'bagging_temperature': [0, 0.5]
}


random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=5,  
    scoring='neg_log_loss',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=2 
)


random_search.fit(X_train, y_train)
print("Best parameters: " ,random_search.best_params_)


y_pred = random_search.predict(X_test)


print(classification_report(y_test, y_pred,target_names=random_search.classes_))


assert(X_transformed.shape[0], len(y))


random_search.best_estimator_.fit(X_transformed,y)
proba = random_search.best_estimator_.predict_proba(test_transformed)


proba[:2]


random_search.best_estimator_.classes_


ordered_classes = random_search.best_estimator_.classes_[np.argsort(-proba)]


ordered_classes[:3]


res = []


for i in range(ordered_classes.shape[0]):
    res.append((test['id'].iloc[i], ' '.join(ordered_classes[i,:3])))


submission = pd.DataFrame(res,columns = ['id','Fertilizer Name'])  


submission.head()


assert test.shape[0] == submission.shape[0]


submission.to_csv('submission.csv', index=False)
print("Submission created")

