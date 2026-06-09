import pandas as pd
import numpy as np
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import KNNImputer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import optuna

import warnings
warnings.simplefilter("ignore")

np.random.seed(42)
random.seed(42)


train = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv', index_col='id')


train.head(3)


train.info()


train['IsActiveMember'].value_counts(normalize=True)


train.hist(figsize=(20, 15), layout=(-1, 2));


train.describe()


plt.figure(figsize=(12, 8))
sns.heatmap(
    data=train.corr(numeric_only=True).round(3),
    square=True,
    annot=True
);


X = train.drop(columns=['CustomerId', 'Surname', 'Exited'])
y = np.array(train['Exited'])


def new_features(df: pd.DataFrame):
    df = (
        pd.concat([
            df, 
            pd.get_dummies(df['Gender'], drop_first=True).astype('int'), 
            pd.get_dummies(df['Geography'], drop_first=True).astype('int')
        ], 
        axis=1
        )
    )
    
    bins = [17, 38, 42, 45, 48, 72]
    labels = ['18-38', '39-42', '43-45', '46-48', '49-72']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    df['WealthIndex'] = df['Balance'] * df['CreditScore'] / 10000
    df['ZeroBalance'] = (df['Balance'] == 0).astype('int')
    df['SalaryPerYear'] = (df['EstimatedSalary'] / df['Age'])
    
    df_diff = (
        df.groupby('AgeGroup')
        .agg(EstimatedSalaryDiff=('EstimatedSalary', 'median'))
        .reset_index()
    )
    
    df = (
        df
        .merge(
            df_diff, 
            right_on='AgeGroup', 
            left_on='AgeGroup', how='left')
    )
    df['EstimatedSalaryDiff'] = df['EstimatedSalaryDiff'] / df['EstimatedSalary']
    df['AvgCountryBalance'] = df.groupby('Geography')['Balance'].transform('mean')
    df['AgeTenureRatio'] = df['Age'] / (df['Tenure'] + 1)
    df['HasCardAndActive'] = (df['HasCrCard'] == 1) & (df['IsActiveMember'] == 1)
    df['LogEstimatedSalary'] = np.log1p(df['EstimatedSalary'])
    df['AgeSquared'] = df['Age'] ** 2
    
    conditions = [(df["NumOfProducts"]==1), (df["NumOfProducts"]==2), (df["NumOfProducts"]>2)]
    values = ["One Product","Two Products", "More Than 2 Products"]
    df["QuantityProducts"] = np.select(conditions, values)

    return df


X = new_features(X).drop(['Germany', 'Spain', 'Male', 'NumOfProducts', 'EstimatedSalary', 'Age'], axis=1)


cat_features_onehot = [
    'Geography',
    'AgeGroup', 
    'QuantityProducts'
] 

cat_features_ordinal = [
    'Gender', 
    'HasCrCard', 
    'IsActiveMember',
    'ZeroBalance', 
    'HasCardAndActive'
]       

num_features = [
    'CreditScore', 
    'Tenure', 
    'Balance', 
    'WealthIndex', 
    'SalaryPerYear', 
    'EstimatedSalaryDiff',
    'AvgCountryBalance', 
    'AgeTenureRatio', 
    'LogEstimatedSalary',
    'AgeSquared', 
]


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features_onehot),
        ("cat_bin", OrdinalEncoder(), cat_features_ordinal)
    ],
    remainder='passthrough'
)


X_transformed = preprocessor.fit_transform(X)


X_transformed.shape


X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'loss_function': 'Logloss', 
        'verbose': False,
        'random_state': 42
    }
    
    model = CatBoostClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


best_value = study.best_value
print("ROC-AUC:", best_value)


best_params = study.best_params


best_model = CatBoostClassifier(**best_params, random_state=42)


best_model.fit(X_train, y_train)


test = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv', index_col='id')


test = new_features(test).drop(['CustomerId', 'Surname', 'Germany',
                                'Spain', 'Male', 'NumOfProducts',
                                'EstimatedSalary', 'Age'], axis=1)


y_transformed = preprocessor.fit_transform(test)


y_predict = best_model.predict_proba(y_transformed)[:][:,1]


submission = (
    pd.DataFrame([i for i in range(15000, 25000)], y_predict)
    .reset_index()[[0, 'index']]
    .rename(columns={0: 'id', 'index': 'Exited'})
)
submission.to_csv('submission.csv', index=False)

