import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mstats
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats.mstats import winsorize
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json


path = r'/kaggle/input/who-life-expectancy-prediction/training data.csv'


# load data set
data = pd.read_csv(path)
data.columns = data.columns.str.strip() 
data = data.dropna(subset=['Life expectancy'])
y = data['Life expectancy']
X = data.drop(columns='Life expectancy')


data.head()
data.info()


health_features = [
    "Adult Mortality",
    "infant deaths",
    "Alcohol",
    "Hepatitis B",
    "Measles",
    "BMI",
    "under-five deaths",
    "Polio",
    "Total expenditure",
    "Diphtheria",
    "HIV/AIDS",
    "thinness 1-19 years",
    "thinness 5-9 years"
]

economic_features = [
    "GDP",
    "percentage expenditure",
    "Income composition of resources"
]

other_features = [
    "Country",
    "Year",
    "Status",
    "Population",
    "Schooling"
]


# Visualize the coeffiction of dataset
numeric_df = data.select_dtypes(include=['float64', 'int64'])

corr_matrix = numeric_df.corr()
plt.figure(figsize=(14, 8))  
sns.heatmap(corr_matrix, 
            annot=True,  
            fmt=".2f",   
            cmap='coolwarm', 
            linewidths=0.5,  
            vmin=-1, vmax=1) 
plt.title("Heatmap - The correlation between the values.", fontsize=14)
plt.show()


class OneHotEncoderDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.columns
        self.encoder.fit(X)
        return self

    def transform(self, X):
        encoded_array = self.encoder.transform(X)
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.encoder.get_feature_names_out(self.columns),
            index=X.index
        )
        return encoded_df

    def get_feature_names_out(self, input_features=None):
        return self.encoder.get_feature_names_out(self.columns)


# modeling

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model=None, top_n=10, feature_names=None):
        self.top_n = top_n
        self.model = model if model is not None else RandomForestRegressor()
        self.feature_names = feature_names
        self.selected_features_ = None

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = X.columns

        self.model.fit(X, y)
        importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
        self.selected_features_ = importances.nlargest(self.top_n).index.tolist()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            idxs = [self.feature_names.index(f) for f in self.selected_features_]
            return X[:, idxs]

numerical_col = X.select_dtypes(include=['int64','float64']).columns.tolist()
category_col = X.select_dtypes(include='object').columns.tolist()

# remove outlier


numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

processor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numerical_col),
    ('cat', OneHotEncoderDF(), category_col)
])


model = Pipeline(steps=[
    ('feature_selector', TopFeatureSelector(top_n=10)),   # select top 10 feature with RF
    ('xgb_model', XGBRegressor(colsample_bytree= 0.7064654114449902, 
    n_estimators= 6875,
    gamma= 0.18279975644986343, 
    learning_rate= 0.01223882034412844, 
    max_depth= 5, 
    reg_alpha= 0.05893643440612434, 
    reg_lambda= 0.7420718776978183, 
    subsample= 0.9234945896121194))                         # training with XGBRegressor
])


# final pipeline

life_expectancy_predictor = Pipeline(steps=[
    ('preprocessing', processor),
    ('modeling', model)
])


life_expectancy_predictor.fit(X,y)


# load testing data

test = pd.read_csv(r"/kaggle/input/who-life-expectancy-prediction/testing data.csv")
test.columns = test.columns.str.strip()

rows_id = test['Row_id']

# process testing data

test_processed = life_expectancy_predictor.named_steps['preprocessing'].transform(test.drop(columns='Row_id'))

# make predictions
prediction = life_expectancy_predictor.named_steps['modeling'].predict(test_processed)

output = pd.DataFrame({'Row_id' : rows_id, 'Prediction' : prediction})

output.to_csv('submission.csv', index=False)

