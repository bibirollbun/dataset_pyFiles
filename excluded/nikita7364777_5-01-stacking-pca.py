# 1. Import Packages
#Basics package
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, PowerTransformer

#Transformers and Pipeline
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config #(for scheme vizualization)

#Feature engineering
from sklearn.feature_selection import mutual_info_regression
import holidays

#Models ML (Linear and Tree)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR


#Model evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


#Stacking
from sklearn.ensemble import StackingRegressor


# 2. First Look to Data
df_train = pd.read_csv('train.csv', index_col = 'id')
df_test = pd.read_csv('test.csv', index_col = 'id')


pipe_data = df_train.copy() # For pipelines
pipe_test = df_test.copy() # For pipelines predictions


df_train.head()


# date -> pd.to_datetime, для остальных признаков можем применить OneHot, так как unique значений меньше 6
df_train.describe(exclude=np.number).T


df_train.describe().T


# Присутствуют нулевые значения
df_train.info()


#Распределение целевой переменной (что вижу то и говорю (качественно) - самые дешевые продаются чаще, чем самые дорогие)
plt.figure(figsize=(5, 5))
sns.histplot(df_train['num_sold'], kde=True, bins=30)
plt.title('Distribution of num_sold')
plt.xlabel('num_sold')
plt.ylabel('Frequency')
plt.show()


# 3. Missing Values (присутствуют только в целевой переменной) (я выбрал следующий метод избавления - убрал строки где отсутствует информация. Пробовал заменять на медианные и средние, ошибка в этом случае выше)
missing = pd.DataFrame(df_train.isnull().sum().sort_values(ascending = False))
missing.columns = ['Count_Null']
missing['Percent_Null'] = round(missing[0:]*100/230130, 2)
missing.style.background_gradient('seismic')


# Все категориальные признаки (only nominal without ordinal CF)
categorical_features = [feature
                        for feature in df_train.columns
                            if df_train[feature].dtype == "object"]
categorical_features.remove('date')


# Преобразуем дату к формату date-time
df_train['date'] = pd.to_datetime(df_train['date'])


# Целевая переменная
y = df_train['num_sold']


# 4. Exploratory Data Analysis
# Категориальные признаки (чистить тут ничего не нужно, всё располагается в нужых диапазонах, вылеты отсуттвуют (можно выкинуть индексы num_sold от 5 до 6к, тут на усмотрение, оишбка я думаю особо не поменяется))
fig, ax = plt.subplots(1, len(categorical_features), figsize=(20, 8))
ax = ax.flatten() if len(categorical_features) > 1 else [ax]

for var, subplot in zip(categorical_features, ax):
    sns.boxplot(x=var, y='num_sold', data=df_train, ax=subplot)
    subplot.tick_params(axis='x', rotation=45)
    subplot.set_title(f'Boxplot of {var} vs num_sold')
    
plt.tight_layout()
plt.show()


# На графике присутствует сезонность + 6 раз меняется тренд
# Для более детального анализа можно вычислить тренд по отдельным участкам и учесть его
# Тренд можно вычислить применив полином 1-2 порядка к отрезку данных (в данном случае я вижу 6 отрезков)
# Учитывается он простым вычитанием из отрезка наблюденных данных (это будет называться локальная составляющая данных или наблюденные данные с учетом трендовой составляющей)
# Про методы борьбы с сезонностью не буду рассказывать, это всё можно прочитать самостотоятельно
plt.figure(figsize=(24,6))
df_train.groupby('date')['num_sold'].sum().plot(xlabel='Date',  ylabel='Number of Products Sold',  title='Total Sales Over Time')
plt.grid()
plt.show()


# Норвегия лидирует по продажам (за 6 лет)
plt.figure(figsize=(24, 6))
sns.lineplot(x=df_train['date'].dt.year, y=df_train['num_sold'], hue=df_train['country'], estimator='sum')
plt.title('Sales Trends by Country (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Country')
plt.grid()
plt.show()


plt.figure(figsize=(24, 6))
sns.lineplot(x=df_train['date'].dt.year, y=df_train['num_sold'], hue=df_train['store'], estimator='sum')
plt.title('Sales Trends by Store (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Store')
plt.grid()
plt.show()


plt.figure(figsize=(24, 6))
sns.lineplot(x=df_train['date'].dt.year, y=df_train['num_sold'], hue=df_train['product'], estimator='sum')
plt.title('Sales Trends by product (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Number of Products Sold')
plt.legend(title='Store')
plt.grid()
plt.show()


# Вычисление взимосвязи категориальных признаков
mutual_df_categorical = df_train[categorical_features]
for column in mutual_df_categorical:
    mutual_df_categorical[column], _ = mutual_df_categorical[column].factorize()
mutual_info = mutual_info_regression(mutual_df_categorical, y.fillna(y.median()), random_state = 42)

mutual_info = pd.Series(mutual_info)
mutual_info.index = mutual_df_categorical.columns
pd.DataFrame(mutual_info.sort_values(ascending=False), columns = ["MI_regression_Cat"] ).style.background_gradient("rainbow")


# Вычисляем новый признак - праздничные дни
# Для этого нам нужно составить словарь типа Country^ {Date: Holiday, ...., Date: Holiday}
# Далее заполняем этот словарь нулями и там где у нас будут присутствовать праздиники ставим 1 (0 значит праздников нет)
extract_country = dict(zip(np.sort(df_train.country.unique()), ["CA", "FI", "IT", "KE", "NO", "SG"]))


extract_country


holidays_dict = {c: holidays.country_holidays(a, years=range(2010, 2020))
                 for c, a in extract_country.items()}


holidays_dict


# 5. Feature Engineering (добавляем новые числовые признаки)
df_train["is_holiday"] = 0 # Там где отсутствуют праздники
for c in holidays_dict:
    df_train.loc[df_train.country == c, "is_holiday"] = df_train.date.isin(holidays_dict[c]).astype(int)
df_train['day'] = df_train.date.dt.day.astype('int64')
df_train['month'] = df_train.date.dt.month.astype('int64')
df_train['year'] = df_train.date.dt.year.astype('int64')
df_train['quarter'] = df_train.date.dt.quarter.astype('int64')
df_train['dayofyear'] = df_train.date.dt.dayofyear.astype('int64')
df_train['weekday'] = df_train.date.dt.weekday.astype('int64')
df_train['sine_day'] = np.sin(2 * np.pi * df_train['day'] / 31).astype('float64')
df_train['cos_day'] = np.cos(2 * np.pi * df_train['day'] / 31).astype('float64')
df_train['sine_month'] = np.sin(2 * np.pi * df_train['month'] / 12).astype('float64')
df_train['cos_month'] = np.cos(2 * np.pi * df_train['month'] / 12).astype('float64')
df_train['sine_year'] = np.sin(2 * np.pi * df_train['year']/ 3).astype('float64')
df_train['cos_year'] = np.cos(2 * np.pi * df_train['year']/ 3).astype('float64')
df_train['sine_quarter'] = np.sin(2 * np.pi * df_train['quarter'] / 4).astype('float64')
df_train['cos_quarter'] = np.cos(2 * np.pi * df_train['quarter'] / 4).astype('float64')
df_train['sine_dayofyear'] = np.sin(2 * np.pi * df_train['dayofyear'] / 366).astype('float64')
df_train['cos_dayofyear'] = np.cos(2 * np.pi * df_train['dayofyear'] / 366).astype('float64')
df_train['sine_weekday'] = np.sin(2 * np.pi * df_train['weekday'] / 7).astype('float64')
df_train['cos_weekday'] = np.cos(2 * np.pi * df_train['weekday'] / 7).astype('float64')


df_train.info()


pipe_data['date'] = pd.to_datetime(pipe_data['date'])
# Присвоили целевую переменную
y = pipe_data.num_sold
# Избавились от ненужных для моделирования данных + засейвим некоторые преобразования (это в целом можно поместить в конвертор)
extract_country = dict(zip(np.sort(pipe_data.country.unique()), ["CA", "FI", "IT", "KE", "NO", "SG"]))
holidays_dict = {c: holidays.country_holidays(a, years=range(2010, 2020))
                 for c, a in extract_country.items()}
pipe_data["is_holiday"] = 0
for c in holidays_dict:
    pipe_data.loc[pipe_data.country == c, "is_holiday"] = pipe_data.date.isin(holidays_dict[c]).astype(int)
pipe_data['day'] = pipe_data.date.dt.day.astype('int64')
pipe_data['month'] = pipe_data.date.dt.month.astype('int64')
pipe_data['year'] = pipe_data.date.dt.year.astype('int64')
pipe_data['quarter'] = pipe_data.date.dt.quarter.astype('int64')
pipe_data['dayofyear'] = pipe_data.date.dt.dayofyear.astype('int64')
pipe_data['weekday'] = pipe_data.date.dt.weekday.astype('int64')
pipe_data = pipe_data.dropna()
pipe_data = pipe_data.drop_duplicates()
pipe_data = pipe_data.drop(['date', 'num_sold'], axis = 1)


pipe_data.info()


pipe_test['date'] = pd.to_datetime(pipe_test['date'])
# Аналогично для тестовых данных
# Избавились от ненужных для моделирования данных + засейвим некоторые преобразования (это в целом можно поместить в конвертор)
extract_country = dict(zip(np.sort(pipe_test.country.unique()), ["CA", "FI", "IT", "KE", "NO", "SG"]))
holidays_dict = {c: holidays.country_holidays(a, years=range(2010, 2020))
                 for c, a in extract_country.items()}
pipe_test["is_holiday"] = 0
for c in holidays_dict:
    pipe_test.loc[pipe_test.country == c, "is_holiday"] = pipe_test.date.isin(holidays_dict[c]).astype(int)
pipe_test['day'] = pipe_test.date.dt.day.astype('int64')
pipe_test['month'] = pipe_test.date.dt.month.astype('int64')
pipe_test['year'] = pipe_test.date.dt.year.astype('int64')
pipe_test['quarter'] = pipe_test.date.dt.quarter.astype('int64')
pipe_test['dayofyear'] = pipe_test.date.dt.dayofyear.astype('int64')
pipe_test['weekday'] = pipe_test.date.dt.weekday.astype('int64')
pipe_test = pipe_test.dropna()
pipe_test = pipe_test.drop_duplicates()
pipe_test = pipe_test.drop(['date'], axis = 1)


pipe_test.info()


num_vars = df_train.select_dtypes("number").columns.to_list()


# Составляем корреляционную матрицу часловых признаков (прежде всего это линейная зависимость)
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    df_train[num_vars].corr(),
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    cmap="seismic",
    annot_kws={"fontsize": 8},
    cbar_kws={"shrink": 1},
    ax = ax
)
cbar_ax = fig.axes[-1]
cbar_ax.tick_params(labelsize=8)
ax.tick_params(labelsize=8)
ax.tick_params(labelsize=8)
plt.title("Correlation Heatmap", fontdict={"fontsize": 14}, pad=20)
plt.show()


# Соответственно выделим новые признаки
new_categorical_features = [features 
                            for features in df_train.columns
                               if df_train[features].dtype == 'object']

new_numerical_features = list(set(df_train.columns) - set(new_categorical_features))
new_numerical_features.remove('date')
new_numerical_features.remove('num_sold')


new_categorical_features, len(new_categorical_features)


new_numerical_features, len(new_numerical_features)


len(df_train.columns)
# med_num_sold, num_sold, date - убираем из набора, остается 21 признак
# new_categorical_features+new_numerical_features = 18 числовых + 3 категориальных = 21 признак


# 2D гистограмма численных признаков (дисперсия хромает, в препроцессинге требуется определить признаки с выскокой skew 
# + применить PowerTransformer для пайплайнов линейных моделей)
fig, ax = plt.subplots(5, 4, figsize=(15, 15))
for var, subplot in zip(new_numerical_features, ax.flatten()):
  subplot.hist2d(df_train[var], df_train['med_num_sold'], cmap='rainbow')
  subplot.set_xlabel(var)
  subplot.set_ylabel('num_sold')
  subplot.set_title(f'Hist2d of {var} vs num_sold')
plt.tight_layout()
plt.show()


# 6. Preprocessing Data
# Для древовидных моделей
# Во многих ноутбуках на Kaggle нормализация в дреововидных моделях не инициализируется, хотя по хорошему надо
numerical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'median'))]) # Для численных признаков нам в целом не нужен данный трансформер, так как пропуски у нас только в y
categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'constant', fill_value = 'Not_Have_Value')), 
                                            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))])


tree_preprocessor = ColumnTransformer(remainder=numerical_transformer, 
                                      transformers=[('categorical_transformer', categorical_transformer, new_categorical_features)])


set_config(display="diagram")
tree_preprocessor


# Для линейных моделей
numerical_transformer2 = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'median')), 
                                          ('Scaller', StandardScaler())])
categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'constant', fill_value = 'Not_Have_Value')), 
                                            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))])


df_train.columns


df_train = df_train.drop(['date', 'num_sold'], axis = 1)


# Поскольку линейные модели предполагают нормальное распределение, skewed features (или анизотропные признаки с повышенной дисперсией)
# снижают их производительность. Для этого к ним применяется PowerTransformer, который ставит признаки в норм. распределение. Рассмотрим только skewed features (>0.005).
skew_features = df_train.select_dtypes(exclude=['object']).skew().sort_values(ascending=False)
skew_features = pd.DataFrame({'Skew' : skew_features})
skew_features.style.background_gradient('seismic')


skewed_features = ['is_holiday', 'cos_day', 'sine_month', 'cos_month', 'day', 'sine_quarter']


skewness_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                       ('PowerTransformer', PowerTransformer(method='yeo-johnson', standardize=True)),])


linear_preprocessor = ColumnTransformer(remainder=numerical_transformer2, transformers=[('skewness_transformer', skewness_transformer, skewed_features),
                                                                                        ('categorical_transformer', categorical_transformer, new_categorical_features)])


set_config(display="diagram")
linear_preprocessor


#Создадим конвейер для обработки численных признаков из этапа Feature Engineering
class FeatureCreator1(BaseEstimator, TransformerMixin):
    def __init__(self, add_attributes=True):
        self.add_attributes = add_attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.add_attributes:
            #Copy from Feature Engineering
            X_copy = X.copy()
            X_copy['sine_day'] = np.sin(2 * np.pi * X_copy['day'] / 31).astype('float64')
            X_copy['cos_day'] = np.cos(2 * np.pi * X_copy['day'] / 31).astype('float64')
            X_copy['sine_month'] = np.sin(2 * np.pi * X_copy['month'] / 12).astype('float64')
            X_copy['cos_month'] = np.cos(2 * np.pi * X_copy['month'] / 12).astype('float64')
            X_copy['sine_year'] = np.sin(2 * np.pi * X_copy['year']/ 3).astype('float64')
            X_copy['cos_year'] = np.cos(2 * np.pi * X_copy['year']/ 3).astype('float64')
            X_copy['sine_quarter'] = np.sin(2 * np.pi * X_copy['quarter'] / 4).astype('float64')
            X_copy['cos_quarter'] = np.cos(2 * np.pi * X_copy['quarter'] / 4).astype('float64')
            X_copy['sine_dayofyear'] = np.sin(2 * np.pi * X_copy['dayofyear'] / 366).astype('float64')
            X_copy['cos_dayofyear'] = np.cos(2 * np.pi * X_copy['dayofyear'] / 366).astype('float64')
            X_copy['sine_weekday'] = np.sin(2 * np.pi * X_copy['weekday'] / 7).astype('float64')
            X_copy['cos_weekday'] = np.cos(2 * np.pi * X_copy['weekday'] / 7).astype('float64')
            return X_copy
        else:
            return X_copy


Convertor = FeatureCreator1(add_attributes = True)


# Проделаем алгоритм кросс валидации (котоырй используется ниже) вручную, чтобы проверить что всё работает (шаг 1)
#pp1 = make_pipeline(Convertor)
#pipe_d1 = pp1.fit_transform(pipe_data)
#pipe_d1 = pd.DataFrame(pipe_d1)
# Теперь применяем препроцессор (шаг 2)
#pipe_d2 = linear_preprocessor.fit_transform(pipe_d1)
#pipe_d2 = pd.DataFrame(pipe_d2)
# Далее эти данные помещаются в древовидную модель или линейную модель или нейрсоеть с заверткой в sklearn (шаг 3)
#pipe_d2


# 10 итераций грузит пару минут
def objective(trial):
    max_iter = trial.suggest_int("max_iter", 1000, 4000)
    alpha =  trial.suggest_float("alpha", 1e-4, 1000, log=True) 
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.05)
    tol =  trial.suggest_float("tol", 1e-6, 1e-3, log=True)

    ElasticNet_regressor = ElasticNet(max_iter=max_iter, alpha=alpha,tol=tol, l1_ratio=l1_ratio, random_state = 42)
   
    ElasticNet_pipeline = make_pipeline(Convertor,
                                        linear_preprocessor, 
                                        ElasticNet_regressor)
    
    ElasticNet_model = TransformedTargetRegressor(regressor=ElasticNet_pipeline, func=np.log1p, inverse_func=np.expm1)
    
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state = 42)
    score = cross_val_score(ElasticNet_model, pipe_data, y.dropna(), scoring= make_scorer(mean_absolute_percentage_error),  cv=ss)
    score = score.mean()
    return score


sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=10)


plot_optimization_history(study)


plot_slice(study)


plot_contour(study)


plot_param_importances(study)


# 10 итераций грузит пару минут
def objective(trial):
    max_iter = trial.suggest_int("max_iter", 1000, 6000)
    alpha =  trial.suggest_float("alpha", 1e-4, 1000, log=True) 
    tol =  trial.suggest_float("tol", 1e-6, 1e-3, log=True)

    Lasso_regressor = Lasso(max_iter=max_iter, alpha=alpha,tol=tol, random_state = 42)
    Lasso_pipeline = make_pipeline(Convertor, linear_preprocessor, Lasso_regressor)
    Lasso_model = TransformedTargetRegressor(regressor=Lasso_pipeline, func=np.log1p, inverse_func=np.expm1)
    
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state= 42)
    score = cross_val_score(Lasso_model, pipe_data, y.dropna(), scoring= make_scorer(mean_absolute_percentage_error),  cv=ss)
    score = score.mean()
    return score


sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=10)


plot_optimization_history(study)


plot_slice(study)


plot_contour(study)


plot_param_importances(study)


# 10 итераций грузит пару минут
def objective(trial):
    max_iter = trial.suggest_int("max_iter", 1000, 6000)
    alpha =  trial.suggest_float("alpha", 1e-4, 1000, log=True) 
    tol =  trial.suggest_float("tol", 1e-6, 1e-3, log=True)

    Ridge_regressor = Ridge(max_iter=max_iter, alpha=alpha, tol= tol, random_state = 42)
    Ridge_pipeline = make_pipeline(Convertor, linear_preprocessor, Ridge_regressor)
    Ridge_model = TransformedTargetRegressor(regressor=Ridge_pipeline, func=np.log1p, inverse_func=np.expm1)
    
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state = 42)
    score = cross_val_score(Ridge_model, pipe_data, y.dropna(), scoring= make_scorer(mean_absolute_percentage_error),  cv=ss)
    score = score.mean()
    return score


sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=10)


plot_optimization_history(study)


plot_slice(study)


plot_contour(study)


plot_param_importances(study)


# 10 итераций грузит очень долго (порядка 6 часов). Иногда даже комп вырубается.
def objective(trial):
    param = {
         "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "sigmoid"]),
         "C": trial.suggest_float("C", 1e-3, 1e-1, log=True),
         "epsilon": trial.suggest_float("epsilon", 1e-4, 1e-1, log=True)}

    SVR_regressor = SVR(**param)
    SVR_pipeline = make_pipeline(Convertor, linear_preprocessor, SVR_regressor)
    SVR_model = TransformedTargetRegressor(regressor=SVR_pipeline, func=np.log1p, inverse_func=np.expm1)
    
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state = 42)
    score = cross_val_score(SVR_model, pipe_data, y.dropna(), scoring= make_scorer(mean_absolute_percentage_error),  cv=ss)     
    score = score.mean()
    return score


sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=10)


plot_optimization_history(study)


plot_slice(study)


plot_contour(study)


plot_param_importances(study)


from sklearn.decomposition import PCA


# 10 итераций грузит очень долго (порядка 8 часов).
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=100)
    max_depth = trial.suggest_int("max_depth", 7, 15, step=2)
    learning_rate =  trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-6, 10, log=True)
    #subsample = trial.suggest_float("subsample", 0.5, 0.9)
    #gamma = trial.suggest_float("gamma", 1e-3, 1e-1, log=True)
    #colsample_bytree = trial.suggest_float("colsample_bytree", 0.22, 0.9)
    #min_child_weight = trial.suggest_int("min_child_weight", 1, 3)
    #reg_lambda = trial.suggest_float("reg_lambda", 1e-6, 10, log=True)

    xgb_regressor = XGBRegressor(n_estimators = n_estimators,
                                 reg_alpha=reg_alpha,
                                 #reg_lambda = reg_lambda,
                                 #subsample=subsample,
                                 #colsample_bytree=colsample_bytree,
                                 max_depth=max_depth,
                                 #min_child_weight =min_child_weight,
                                 learning_rate=learning_rate,
                                 #gamma=gamma,
                                 eval_metric = 'mape',
                                 booster = 'gbtree',
                                 random_state = 42)

    pca = PCA(n_components = 5) #-------------------------------------------------------------------------------------------------------Proba
    xgb_pipeline = make_pipeline(Convertor, tree_preprocessor, pca, xgb_regressor)

    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state = 42)
    score = cross_val_score(xgb_pipeline, pipe_data, y.dropna(), scoring= make_scorer(mean_absolute_percentage_error),  cv=ss)
    score = score.mean()
    return score


sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=25)


plot_optimization_history(study)


plot_slice(study)


plot_contour(study)


plot_param_importances(study)


# 10 итераций грузит очень долго (порядка 6 часов).
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step= 50)
    max_depth = trial.suggest_int("max_depth", 6, 16, step=2)
    learning_rate =  trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    #subsample = trial.suggest_float("subsample", 0.40, 0.95)
    #min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 20, step=2)
    #min_samples_split = trial.suggest_float("min_samples_split", 0.3, 0.9)
    #min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.1, 0.9)

    gbm_regressor = GradientBoostingRegressor(n_estimators = n_estimators,
                                              max_depth=max_depth,
                                              learning_rate=learning_rate,
                                              #subsample=subsample,
                                              #min_samples_leaf=min_samples_leaf,
                                              #min_samples_split = min_samples_split,
                                              #min_impurity_decrease = min_impurity_decrease,
                                              random_state = 42)

    gbm_pipeline = make_pipeline(Convertor, tree_preprocessor, gbm_regressor)

    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state = 42)
    score = cross_val_score(gbm_pipeline, pipe_data, y.dropna(), scoring= make_scorer(mean_absolute_percentage_error),  cv=ss)
    score = score.mean()
    return score


sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=15)


plot_optimization_history(study)


plot_slice(study)


plot_contour(study)


plot_param_importances(study)


# 10 итераций грузит очень долго (порядка 5 часов).
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 2000, 7000, step=500)
    max_depth = trial.suggest_int("max_depth", 6, 20, step = 3)
    learning_rate =  trial.suggest_float("learning_rate", 1e-3, 1, log=True)
    #min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 10, 30, step = 2)
    #subsample = trial.suggest_float("subsample", 0.4, 0.9)        
    max_bin = trial.suggest_int("max_bin", 100, 300, step=20),
    #feature_fraction = trial.suggest_float("feature_fraction", 0.1, 0.5)

    lgbm_regressor = LGBMRegressor(n_estimators = n_estimators,
                                   max_depth=max_depth,
                                   learning_rate=learning_rate, 
                                   #min_data_in_leaf=min_data_in_leaf,
                                   #subsample=subsample,
                                   max_bin=max_bin,
                                   #feature_fraction=feature_fraction,
                                   verbosity = -1,
                                   random_state = 42)

    lgbm_pipeline = make_pipeline(Convertor,tree_preprocessor, lgbm_regressor)

    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state = 42)
    score = cross_val_score(lgbm_pipeline, pipe_data, y.dropna(), scoring= make_scorer(mean_absolute_percentage_error),  cv=ss)
    score = score.mean()
    return score


sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=25)


plot_optimization_history(study)


plot_slice(study)


plot_contour(study)


plot_param_importances(study)


# 10 итераций грузит очень долго (порядка 19 часов).
#"colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 0.9),
#"random_strength": trial.suggest_float("random_strength", 1e-2, 1, log=True),
#"subsample": trial.suggest_float("subsample", 0.6, 1)
def objective(trial):
    cat_param = {
        "iterations" : trial.suggest_int("iterations", 500, 2000, step=500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "depth": trial.suggest_int("depth", 4, 10, step = 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log = True),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Bayesian"]),
        "early_stopping_rounds": 100,
        "task_type": "GPU"
    }

    catboost_regressor = CatBoostRegressor(**cat_param, random_state = 42, verbose = False, loss_function = "MAPE")
    catboost_pipeline = make_pipeline(Convertor, tree_preprocessor, catboost_regressor)

    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state = 42)
    score = cross_val_score(catboost_pipeline, pipe_data, y.dropna(), scoring= make_scorer(mean_absolute_percentage_error),  cv=ss)
    score = score.mean()
    return score


sampler = TPESampler(seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=25)


plot_optimization_history(study)


plot_slice(study)


plot_contour(study)


plot_param_importances(study)


# Stacking (стакаем все модели и надеемся уменьшить ошибку)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Tree models

xgb_tunned = XGBRegressor(n_estimators = 100, max_depth = 15, learning_rate = 0.08691089486124978, 
                          reg_alpha = 0.09278524624474564, random_state = 42)
pipe_xgb = Pipeline(steps=[('tree_preprocessor', tree_preprocessor),
                           ('regressor1', xgb_tunned)])

gbm_tunned = GradientBoostingRegressor(n_estimators = 450, max_depth = 16, 
                                       learning_rate = 0.029106359131330698, random_state = 42)
pipe_gbm = Pipeline(steps=[('tree_preprocessor', tree_preprocessor),
                           ('regressor2', gbm_tunned)])

lgbm_tunned = LGBMRegressor(n_estimators = 6000, max_depth = 12, 
                            learning_rate = 0.034184504082053084, max_bin = 240, verbosity = -1,
                            random_state = 42)
pipe_lgbm = Pipeline(steps=[('tree_preprocessor', tree_preprocessor),
                            ('regressor3', lgbm_tunned)])
'''
catboost_tunned = CatBoostRegressor()
pipe_catboost = Pipeline(steps=[('tree_preprocessor', tree_preprocessor),
                                ('regressor4', catboost_tunned)])
'''
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Linear models
elasticnet_tunned = ElasticNet(max_iter = 2777, alpha = 0.00021142332035497166, l1_ratio = 0.6000000000000001, tol = 3.247673570627449e-06)
pipe_Elasticnet = Pipeline(steps=[('linear_preprocessor', linear_preprocessor),
                                  ('regressor5', elasticnet_tunned)])
TargetTransformedElasticnet = TransformedTargetRegressor(regressor=pipe_Elasticnet, func=np.log1p, inverse_func=np.expm1)

lasso_tunned = Lasso(max_iter = 4541, alpha = 0.000139345022513376, tol = 0.0008123245085588687)
pipe_Lasso = Pipeline(steps=[('linear_preprocessor', linear_preprocessor),
                             ('regressor6', lasso_tunned)])
TargetTransformedLasso = TransformedTargetRegressor(regressor=pipe_Lasso, func=np.log1p, inverse_func=np.expm1)

ridge_tunned = Ridge(max_iter = 4541, alpha = 0.000139345022513376, tol = 0.0008123245085588687)
pipe_Ridge = Pipeline(steps=[('linear_preprocessor', linear_preprocessor),
                             ('regressor7', ridge_tunned)])
TargetTransformedRidge = TransformedTargetRegressor(regressor=pipe_Ridge, func=np.log1p, inverse_func=np.expm1)
'''
svr_tunned = SVR(kernel = 'linear', C =0.004207988669606638, epsilon = 0.00029375384576328325)
pipe_SVR = Pipeline(steps=[('linear_preprocessor', linear_preprocessor),
                           ('regressor8', svr_tunned)])
TargetTransformedSVR = TransformedTargetRegressor(regressor=pipe_SVR, func=np.log1p, inverse_func=np.expm1)
'''


estimators = [
            #("pipe_xgb", pipe_xgb)]#,
            #("pipe_gbm", pipe_gbm),
            #("pipe_lgbm", pipe_lgbm)]#,
            #("pipe_catboost", pipe_catboost),
            #("TargetTransformedElasticnet", TargetTransformedElasticnet),
            #("TargetTransformedLasso", TargetTransformedLasso),
            #("TargetTransformedRidge", TargetTransformedRidge)]
            #("TargetTransformedSVR", TargetTransformedSVR)]


#Сюда вставить нужно значение alpha
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=Lasso(alpha = 0.01, random_state = 42))


#Искать alpha будем через GridSearchCV
#grid_params = {'stacking_regressor__final_estimator__alpha': [0.0001, 0.01, 1, 10]}
#ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state= 42)
#stack_search = GridSearchCV(final_pipe, param_grid = grid_params,scoring= make_scorer(mean_absolute_error), cv = ss, n_jobs = -1)
#stack_search.fit(pipe_data, y.dropna())


final_pipe = Pipeline(steps=[('Convertor', Convertor),
                             ('stacking_regressor', stacking_regressor)])


stacked_regressor = final_pipe.fit(pipe_data, y.dropna())


y_preds = stacked_regressor.predict(pipe_test)


# Output export
output = pd.DataFrame({'id': pipe_test.index, 'num_sold': y_preds})
output.to_csv('submission_LGBM.csv', index=False)


output.head()

