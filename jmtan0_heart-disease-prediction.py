import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv")
df_train.info()
df_train.describe(include='all')


df_test = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv")
df_test.info()
df_test.describe(include='all')


cat_col = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG',
           'ExerciseAngina', 'ST_Slope']
num_col = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']


plt.figure(figsize=(14, 14))
for i, f in enumerate(cat_col, start=1):
    plt.subplot(3, 2, i)
    
    grouped_data = df_train.groupby([f, 'HeartDisease']).size().unstack()
    
    grouped_data.plot(kind='bar', stacked=False, color=['skyblue', 'orange'], ax=plt.gca())
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.ylabel('Count')
    plt.legend(labels=['No HD', 'HD'], loc='best')


for f in num_col:
    present_0 = df_train[df_train['HeartDisease'] == 0]
    present_1 = df_train[df_train['HeartDisease'] == 1]
    
    # Plot the histograms
    plt.hist(present_0[f], bins=10, alpha=0.5, label='Low Risk on HD')
    plt.hist(present_1[f], bins=10, alpha=0.5, label='High Risk on HD')
    plt.xlabel(f)
    plt.legend()
    plt.show()


df_train = pd.get_dummies(data=df_train, columns=cat_col, drop_first=True)
df_train.describe(include='all')


def suspicious_data(df):

    # replace suspicious data with NaN
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df.loc[df['Cholesterol'] < 400, 'Cholesterol'] = np.nan
    df.loc[df['Oldpeak'] < 0, 'Oldpeak'] = np.nan

    # imputation for missing data
    col = df.columns[df.isnull().any()].tolist()
    
    ## scale the data before KNN
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[col])

    ## apply KNN imputation
    imputer = KNNImputer()
    imputed_values = imputer.fit_transform(scaled_values)
    
    ## inverse scale to get back to original values
    imputed_inverse = scaler.inverse_transform(imputed_values)
    
    ## convert to df and update only the imputed_inverse
    df_imputed = pd.DataFrame(imputed_inverse, columns=col, index=df.index)
    df[col] = df_imputed[col]
    
    return df


def handle_outliers(df):
    col = df.select_dtypes(include=['float'])
    for i in col:
        if (df[i].skew() > -1) or (df[i].skew() < 1):
            series = df[i]
            z_scores = stats.zscore(series)
            lower = series[z_scores > -3].min()
            upper = series[z_scores < 3].max()
            df[i] = series.clip(lower, upper)
        
        else:
            series = df[i]
    
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
    
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
    
            df[i].clip(lower_bound, upper_bound)
    
    return df


def transformation(df):
    col = df.select_dtypes(include=['float'])
    for i in col:
        if -1 < df[i].skew() < 1:
            std_scaler = StandardScaler()
            df[i] = std_scaler.fit_transform(df[[i]])
            df.drop(i, axis=1, inplace=True)
    
        else:
            minmax_scaler = MinMaxScaler()
            df[i] = minmax_scaler.fit_transform(df[[i]])
            df.drop(i, axis=1, inplace=True)

    return df


class ModelRunner:
    def __init__(self, model, name=None, param_grid=None, search_type=None, n_iter=20):
        self.model = model
        self.name = name if name else type(model).__name__
        self.param_grid = param_grid # dict of hyperparameters
        self.search_type = search_type   # "grid", "random"
        self.n_iter = n_iter
        self.best_model = None # will store tuned model
        self.best_parameters = None

    def tune(self, X, y, cv=5, scoring="roc_auc"):
        if self.param_grid is None:
            self.model.fit(X, y)
            self.best_model = self.model
            self.searcher = None
            return self

        # Decide search method
        if self.search_type == "grid":
            searcher = GridSearchCV(self.model, self.param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        elif self.search_type == "random":
            searcher = RandomizedSearchCV(self.model, self.param_grid, cv=cv, 
                                          scoring=scoring, n_jobs=-1, n_iter=self.n_iter,
                                          random_state=42)
        else:
            raise ValueError("search_type must be 'grid' or 'random' if param_grid is given")
        searcher.fit(X, y)
        self.best_model = searcher.best_estimator_
        self.best_parameters = searcher.best_params_
        self.searcher = searcher   # store the whole object
        return self
    
    def evaluate(self, X, y, cv=10):
        model_to_eval = self.best_model if self.best_model else self.model
        r = cross_validate(model_to_eval, X, y, cv=cv,
                           scoring=['accuracy', 'roc_auc'],
                           return_train_score=True)        
        return {
            "name": self.name,
            "train_accuracy": r['train_accuracy'].mean(),
            "test_accuracy": r['test_accuracy'].mean(),
            "train_roc_auc": r['train_roc_auc'].mean(),
            "test_roc_auc": r['test_roc_auc'].mean()
        }

    def fit_predict(self, X_train, y_train, X_test):
        model_to_eval = self.best_model if self.best_model else self.model
        m = model_to_eval.fit(X_train, y_train)
        return m.predict(X_test)


models = [ModelRunner(LogisticRegression(), 'Logistic Regression'),
          ModelRunner(RandomForestClassifier(), 'Random Forest'),
          ModelRunner(KNeighborsClassifier(), 'K-Nearest Neighbor'),
          ModelRunner(SGDClassifier(), 'Stochastic Gradient Descent'),
          ModelRunner(SVC(), 'Support Vector Classification'),
          ModelRunner(GradientBoostingClassifier(), 'Gradient Boosting'),
          ModelRunner(HistGradientBoostingClassifier(), 'Histogram-based Gradient Boosting'),
          ModelRunner(XGBClassifier(), 'eXtreme Gradient Boosting'),
          ModelRunner(LGBMClassifier(verbose=-1), 'Light Gradient Boosting'),
          ModelRunner(CatBoostClassifier(verbose=False), 'Categorical Boosting')]


X = df_train.drop(['HeartDisease'], axis=1)
y = df_train['HeartDisease']

results_data = [m.evaluate(X, y) for m in models]
results_df = pd.DataFrame(results_data)
results_df


models = [ModelRunner(RandomForestClassifier(), 'Random Forest'),
          ModelRunner(GradientBoostingClassifier(), 'Gradient Boosting'),
          ModelRunner(HistGradientBoostingClassifier(), 'Histogram-based Gradient Boosting'),
          ModelRunner(XGBClassifier(), 'eXtreme Gradient Boosting'),
          ModelRunner(LGBMClassifier(verbose=-1), 'Light Gradient Boosting'),
          ModelRunner(CatBoostClassifier(verbose=False), 'Categorical Boosting')]

df_train_v2 = suspicious_data(df_train)
df_train_v2 = handle_outliers(df_train_v2)

X = df_train_v2.drop(['HeartDisease'], axis=1)
y = df_train_v2['HeartDisease']

results_data_6 = [m.evaluate(X, y) for m in models]

# models that need transformation
models_trans = [ModelRunner(LogisticRegression(), 'Logistic Regression'),
                ModelRunner(KNeighborsClassifier(), 'K-Nearest Neighbor'),
                ModelRunner(SGDClassifier(), 'Stochastic Gradient Descent'),
                ModelRunner(SVC(), 'Support Vector Classification')]

df_train_v2_trans = transformation(df_train_v2)

X_trans = df_train_v2_trans.drop(['HeartDisease'], axis=1)
y_trans = df_train_v2_trans['HeartDisease']

results_data_4 = [m.evaluate(X_trans, y_trans) for m in models_trans]
results_df = pd.DataFrame(results_data_6 + results_data_4)
results_df


models = [ModelRunner(RandomForestClassifier(random_state=369),
                      name='Random Forest', search_type='grid',
                      param_grid={'n_estimators': [100, 200, 500],
                                  'max_depth': [3, 5, 10, 20]}),
          ModelRunner(GradientBoostingClassifier(random_state=369),
                      name='Gradient Boosting', search_type='grid',
                      param_grid={'n_estimators': [100, 200, 500],
                                  'learning_rate': [0.01, 0.2],
                                  'max_depth': [3, 5, 10]}),
          ModelRunner(HistGradientBoostingClassifier(random_state=369),
                      name='Histogram-based Gradient Boosting', search_type='grid',
                      param_grid={'max_iter': [100, 200, 500],
                                  'learning_rate': [0.01, 0.2],
                                  'max_depth': [3, 5, 10]}),
          ModelRunner(XGBClassifier(random_state=369),
                      name='eXtreme Gradient Boosting', search_type='grid',
                      param_grid={'n_estimators': [100, 300, 600, 1000],
                                  'max_depth': [3, 5, 10],
                                  'eta': [0.01, 0.1, 0.3]}),
          ModelRunner(LGBMClassifier(verbose=-1, random_state=369),
                      name='Light Gradient Boosting', search_type='grid',
                      param_grid={'num_leaves': [20, 50, 100, 200],
                                  'max_depth': [3, 8, 12, 15],
                                  'learning_rate': [0.01, 0.2],
                                  'n_estimators': [100, 300, 800, 1000]}),
          ModelRunner(CatBoostClassifier(verbose=False, random_state=369),
                      name='Categorical Boosting', search_type='grid',
                      param_grid={'depth': [4, 7, 10],
                                  'learning_rate': [0.01, 0.2],
                                  'iterations': [500, 1000, 2000]})]

results_data_6 = []
for m in models:
    runner = m.tune(X, y)
    result = runner.evaluate(X, y)
    print(f'{runner.name}: {runner.best_parameters}')
    results_data_6.append(result)

# models that need transformation
models_trans = [ModelRunner(LogisticRegression(max_iter=5000, random_state=369),
                            name='Logistic Regression', search_type="grid",
                            param_grid = {'C': [0.01, 0.1, 1, 10],
                                          'penalty': ['l1', 'l2', 'elasticnet'],
                                          'solver': ['saga']}),
                ModelRunner(KNeighborsClassifier(),
                            name='K-Nearest Neighbor', search_type='grid',
                            param_grid = {'n_neighbors': [3, 15],
                                          "metric": ["minkowski"],
                                          "p": [1, 2]}),
                ModelRunner(SGDClassifier(random_state=369),
                            name='Stochastic Gradient Descent', search_type='grid',
                            param_grid = {'alpha': [1e-5, 1e-2],
                                          'penalty': ['l1', 'l2', 'elasticnet']}),
                ModelRunner(SVC(random_state=369),
                            name='Support Vector Classification', search_type='grid',
                            param_grid = {'C': [0.01, 0.1, 1, 10],
                                          'kernel': ['linear', 'rbf', 'poly']})]

results_data_4 = []
for m in models_trans:
    runner = m.tune(X_trans, y_trans)
    result = runner.evaluate(X_trans, y_trans)
    print(f'{runner.name}: {runner.best_parameters}')
    results_data_4.append(result)

results_df = pd.DataFrame(results_data_6 + results_data_4)
results_df


models = [ModelRunner(RandomForestClassifier(random_state=369),
                      name='Random Forest', search_type='grid',
                      param_grid={'n_estimators': [500, 800, 1000],
                                  'max_depth': [8, 10, 12]}),
          ModelRunner(GradientBoostingClassifier(random_state=369),
                      name='Gradient Boosting', search_type='grid',
                      param_grid={'n_estimators': [500, 800, 1000],
                                  'learning_rate': [0.001, 0.005, 0.01],
                                  'max_depth': [1, 2, 3]}),
          ModelRunner(HistGradientBoostingClassifier(random_state=369),
                      name='Histogram-based Gradient Boosting', search_type='grid',
                      param_grid={'max_iter': [500, 800, 1000],
                                  'learning_rate': [0.001, 0.005, 0.01],
                                  'max_depth': [1, 2, 3]}),
          ModelRunner(XGBClassifier(random_state=369),
                      name='eXtreme Gradient Boosting', search_type='grid',
                      param_grid={'n_estimators': [500, 600, 700],
                                  'max_depth': [1, 2, 3],
                                  'eta': [0.001, 0.005, 0.01]}),
          ModelRunner(LGBMClassifier(verbose=-1, random_state=369),
                      name='Light Gradient Boosting', search_type='grid',
                      param_grid={'num_leaves': [5, 15, 20],
                                  'max_depth': [1, 2, 3],
                                  'learning_rate': [0.001, 0.005, 0.01],
                                  'n_estimators': [1000, 1200, 1500]}),
          ModelRunner(CatBoostClassifier(verbose=False, random_state=369),
                      name='Categorical Boosting', search_type='grid',
                      param_grid={'depth': [6, 7, 8],
                                  'learning_rate': [0.001, 0.005, 0.01],
                                  'iterations': [800, 1000, 1200]})]

results_data_6 = []
for m in models:
    runner = m.tune(X, y)
    result = runner.evaluate(X, y)
    print(f'{runner.name}: {runner.best_parameters}')
    results_data_6.append(result)

# models that need transformation
models_trans = [ModelRunner(LogisticRegression(max_iter=5000, random_state=369),
                            name='Logistic Regression', search_type="grid",
                            param_grid = {'C': [10, 12, 15],
                                          'penalty': ['l2'],
                                          'solver': ['saga']}),
                ModelRunner(KNeighborsClassifier(),
                            name='K-Nearest Neighbor', search_type='grid',
                            param_grid = {'n_neighbors': [13, 15, 18],
                                          "metric": ["minkowski"],
                                          "p": [1]}),
                ModelRunner(SGDClassifier(random_state=369),
                            name='Stochastic Gradient Descent', search_type='grid',
                            param_grid = {'alpha': [1e-7, 1e-6, 1e-5],
                                          'penalty': ['l1']}),
                ModelRunner(SVC(random_state=369),
                            name='Support Vector Classification', search_type='grid',
                            param_grid = {'C': [10, 12, 15],
                                          'kernel': ['linear']})]

results_data_4 = []
for m in models_trans:
    runner = m.tune(X_trans, y_trans)
    result = runner.evaluate(X_trans, y_trans)
    print(f'{runner.name}: {runner.best_parameters}')
    results_data_4.append(result)

results_df = pd.DataFrame(results_data_6 + results_data_4)
results_df


models = [ModelRunner(RandomForestClassifier(random_state=369),
                      name='Random Forest', search_type='grid',
                      param_grid={'n_estimators': [500], # from basic
                                  'max_depth': [10], # from basic
                                  'min_samples_split': [2, 5, 10],
                                  'min_samples_leaf': [1, 2, 4],
                                  'max_features': ['sqrt', 'log2', None]}),
          ModelRunner(GradientBoostingClassifier(random_state=369),
                      name='Gradient Boosting', search_type='grid',
                      param_grid={'n_estimators': [800], # from basic
                                  'learning_rate': [0.005], # from basic
                                  'max_depth': [3], # from basic
                                  'min_samples_split': [2, 5, 10],
                                  'min_samples_leaf': [1, 2, 4],
                                  'subsample': [0.6, 0.8, 1.0]}),
          ModelRunner(HistGradientBoostingClassifier(random_state=369),
                      name='Histogram-based Gradient Boosting', search_type='grid',
                      param_grid={'max_iter': [1000], # from basic
                                  'learning_rate': [0.01], # from basic
                                  'max_depth': [3], # from basic
                                  'max_leaf_nodes': [15, 31, 63],
                                  'min_samples_leaf': [20, 50, 100],
                                  'l2_regularization': [0.0, 0.1, 1.0]}),
          ModelRunner(XGBClassifier(random_state=369),
                      name='eXtreme Gradient Boosting', search_type='grid',
                      param_grid={'n_estimators': [500], # from basic
                                  'max_depth': [3], # from basic
                                  'learning_rate': [0.01], # from basic
                                  'subsample': [0.6, 0.8, 1.0],
                                  'colsample_bytree': [0.6, 0.8, 1.0],
                                  'gamma': [0, 1, 5]}),
          ModelRunner(LGBMClassifier(verbose=-1, random_state=369),
                      name='Light Gradient Boosting', search_type='grid',
                      param_grid={'num_leaves': [5],# from basic
                                  'max_depth': [3], # from basic
                                  'learning_rate': [0.005], # from basic
                                  'n_estimators': [1500],# from basic
                                  'min_child_samples': [20, 50, 100],
                                  'subsample': [0.6, 0.8, 1.0],
                                  'colsample_bytree': [0.6, 0.8, 1.0]}),
          ModelRunner(CatBoostClassifier(verbose=False, random_state=369),
                      name='Categorical Boosting', search_type='grid',
                      param_grid={'depth': [7], # from basic
                                  'learning_rate': [0.01], # from basic
                                  'iterations': [800], # from basic
                                  'l2_leaf_reg': [1, 3, 5, 7],
                                  'bagging_temperature': [0, 0.5, 1]})]

results_data_6 = []
for m in models:
    runner = m.tune(X, y)
    result = runner.evaluate(X, y)
    print(f'{runner.name}: {runner.best_parameters}')
    results_data_6.append(result)

# models that need transformation
models_trans = [ModelRunner(LogisticRegression(max_iter=5000, random_state=369),
                            name='Logistic Regression', search_type="grid",
                            param_grid = {'C': [15], # from basic
                                          'penalty': ['l2'], # from basic
                                          'solver': ['saga']}), # from basic
                ModelRunner(KNeighborsClassifier(),
                            name='K-Nearest Neighbor', search_type='grid',
                            param_grid = {'n_neighbors': [15],# from basic
                                          'weights': ['uniform', 'distance'],
                                          'metric': ['minkowski'],# from basic
                                          'p': [1]}),# from basic
                ModelRunner(SGDClassifier(random_state=369),
                            name='Stochastic Gradient Descent', search_type='grid',
                            param_grid = {'alpha': [1e-5],# from basic
                                          'penalty': ['l1'], # from basic
                                          'l1_ratio': [0, 0.15, 0.5, 0.85, 1],
                                          'loss': ['hinge', 'log_loss', 'modified_huber'],}),
                ModelRunner(SVC(random_state=369),
                            name='Support Vector Classification', search_type='grid',
                            param_grid = {'C': [15],# from basic
                                          'kernel': ['linear'],# from basic
                                          'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                                          'degree': [2, 3, 4]})]

results_data_4 = []
for m in models_trans:
    runner = m.tune(X_trans, y_trans)
    result = runner.evaluate(X_trans, y_trans)
    print(f'{runner.name}: {runner.best_parameters}')
    results_data_4.append(result)

results_df = pd.DataFrame(results_data_6 + results_data_4)
results_df


models = [ModelRunner(RandomForestClassifier(random_state=369),
                      name='Random Forest', search_type='grid',
                      param_grid={'n_estimators': [500], # from basic
                                  'max_depth': [10], # from basic
                                  'min_samples_split': [0, 1, 2],
                                  'min_samples_leaf': [0, 2, 3],
                                  'max_features': ['sqrt', 'log2', None]}),
          ModelRunner(GradientBoostingClassifier(random_state=369),
                      name='Gradient Boosting', search_type='grid',
                      param_grid={'n_estimators': [800], # from basic
                                  'learning_rate': [0.005], # from basic
                                  'max_depth': [3], # from basic
                                  'min_samples_split': [8,10,15],
                                  'min_samples_leaf': [0, 1, 2],
                                  'subsample': [1, 2, 5]}),
          ModelRunner(HistGradientBoostingClassifier(random_state=369),
                      name='Histogram-based Gradient Boosting', search_type='grid',
                      param_grid={'max_iter': [1000], # from basic
                                  'learning_rate': [0.01], # from basic
                                  'max_depth': [3], # from basic
                                  'max_leaf_nodes': [5, 10, 15],
                                  'min_samples_leaf': [0, 10, 20],
                                  'l2_regularization': [0.8, 1.0, 1.5]}),
          ModelRunner(XGBClassifier(random_state=369),
                      name='eXtreme Gradient Boosting', search_type='grid',
                      param_grid={'n_estimators': [500], # from basic
                                  'max_depth': [3], # from basic
                                  'learning_rate': [0.01], # from basic
                                  'subsample': [0, 0.2, 0.6],
                                  'colsample_bytree': [0, 0.2, 0.6],
                                  'gamma': [0.5, 1, 2]}),
          ModelRunner(LGBMClassifier(verbose=-1, random_state=369),
                      name='Light Gradient Boosting', search_type='grid',
                      param_grid={'num_leaves': [5],# from basic
                                  'max_depth': [3], # from basic
                                  'learning_rate': [0.005], # from basic
                                  'n_estimators': [1500],# from basic
                                  'min_child_samples': [1, 10, 20],
                                  'subsample': [0.1, 0.25, 0.6],
                                  'colsample_bytree': [0.9, 0.99, 1]}),
          ModelRunner(CatBoostClassifier(verbose=False, random_state=369),
                      name='Categorical Boosting', search_type='grid',
                      param_grid={'depth': [7], # from basic
                                  'learning_rate': [0.01], # from basic
                                  'iterations': [800], # from basic
                                  'l2_leaf_reg': [2, 3, 4],
                                  'bagging_temperature': [0, 0.01, 0.1]})]

results_data_6 = []
for m in models:
    runner = m.tune(X, y)
    result = runner.evaluate(X, y)
    print(f'{runner.name}: {runner.best_parameters}')
    results_data_6.append(result)

# models that need transformation
models_trans = [ModelRunner(LogisticRegression(max_iter=5000, random_state=369),
                            name='Logistic Regression', search_type="grid",
                            param_grid = {'C': [15], # from basic
                                          'penalty': ['l2'], # from basic
                                          'solver': [None, 'saga']}), # from basic
                ModelRunner(KNeighborsClassifier(),
                            name='K-Nearest Neighbor', search_type='grid',
                            param_grid = {'n_neighbors': [15],# from basic
                                          'weights': [None, 'distance'],
                                          'metric': [None, 'minkowski'],# from basic
                                          'p': [1]}),# from basic
                ModelRunner(SGDClassifier(random_state=369),
                            name='Stochastic Gradient Descent', search_type='grid',
                            param_grid = {'alpha': [1e-5],# from basic
                                          'penalty': ['l1'], # from basic
                                          'l1_ratio': [None, 0, 0.05],
                                          'loss': [None, 'hinge']}),
                ModelRunner(SVC(random_state=369),
                            name='Support Vector Classification', search_type='grid',
                            param_grid = {'C': [15],# from basic
                                          'kernel': [None, 'linear'],# from basic
                                          'gamma': [None, 'scale', 0.001, 100],
                                          'degree': [1, 2, 2.5]})]

results_data_4 = []
for m in models_trans:
    runner = m.tune(X_trans, y_trans)
    result = runner.evaluate(X_trans, y_trans)
    print(f'{runner.name}: {runner.best_parameters}')
    results_data_4.append(result)

results_df = pd.DataFrame(results_data_6 + results_data_4)
results_df


df_test = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv")
df_test = pd.get_dummies(data=df_test, columns=cat_col, drop_first=True)

models = [ModelRunner(RandomForestClassifier(), 'Random Forest'),
          ModelRunner(GradientBoostingClassifier(), 'GradientBoostingClassifier'),
          ModelRunner(HistGradientBoostingClassifier(), 'HistGradientBoostingClassifier'),
          ModelRunner(XGBClassifier(), 'eXtreme Gradient Boosting'),
          ModelRunner(LGBMClassifier(verbose=-1), 'Light Gradient Boosting'),
          ModelRunner(CatBoostClassifier(verbose=False), 'Categorical Boosting')]

df_test_v2 = suspicious_data(df_test)
df_test_v2 = handle_outliers(df_test_v2)

for m in models:
    predictions = m.fit_predict(X, y, df_test_v2)
    sub = pd.DataFrame({"id": range(len(predictions)),
                        "HeartDisease": predictions})
    sub.to_csv(f"{m.name}_submission_v2.csv", index=False)


# models that need transformation
models = [ModelRunner(LogisticRegression(), 'Logistic Regression'),
          ModelRunner(KNeighborsClassifier(), 'K-Nearest Neighbor'),
          ModelRunner(SGDClassifier(), 'Stochastic Gradient Descent'),
          ModelRunner(SVC(), 'Support Vector Classification')]

df_test_v2_trans = transformation(df_test_v2)

for m in models:
    predictions = m.fit_predict(X_trans, y_trans, df_test_v2_trans)
    sub = pd.DataFrame({"id": range(len(predictions)),
                        "HeartDisease": predictions})
    sub.to_csv(f"{m.name}_submission_v2.csv", index=False)

