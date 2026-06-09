import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from scipy.stats.mstats import winsorize
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import json


train_path = r'/kaggle/input/who-life-expectancy-prediction/training data.csv'
test_path  = r'/kaggle/input/who-life-expectancy-prediction/testing data.csv'


data = pd.read_csv(train_path)
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


class SkewedWinsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorizes all numeric columns in a DataFrame based on the direction of skewness.

    Parameters:
    - df: pd.DataFrame
        The input DataFrame.
    - skew_threshold: float, default=0.5
        Threshold for treating a distribution as significantly skewed.
    - limit: float, default=0.01
        The proportion of data to winsorize (e.g., 0.01 means trimming 1% from one or both tails).

    Returns:
    - pd.DataFrame
        A new DataFrame with winsorized numeric columns.
    """
    def __init__(self, skew_threshold=0.5, limit=0.01):
        self.skew_threshold = skew_threshold
        self.limit = limit

    def fit(self, X, y=None):
        self.feature_names_ = X.columns if hasattr(X, 'columns') else None
        return self

    def transform(self, X, y=None):
        X_winsorized = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = X[col]
            col_skew = skew(col_data.dropna())
            lower, upper = col_data.quantile([self.limit, 1 - self.limit])

            if abs(col_skew) < self.skew_threshold:
                X_winsorized[col] = np.clip(col_data, lower, upper)
            elif col_skew > 0:
                X_winsorized[col] = np.clip(col_data, col_data.min(), upper)
            else:
                X_winsorized[col] = np.clip(col_data, lower, col_data.max())

        return X_winsorized

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.feature_names_



class ToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names)



# modeling
class SHAPFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model=None, top_n=10):
        self.top_n = top_n
        self.model = model if model is not None else XGBRegressor()
        self.selected_features_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer(X)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_importance = pd.Series(mean_abs_shap, index=X.columns)
        self.selected_features_ = feature_importance.sort_values(ascending=False).head(self.top_n).index.tolist()
        return self

    def transform(self, X):
        return X[self.selected_features_].values


numerical_col = X.select_dtypes(include=['int64','float64']).columns.tolist()
category_col = X.select_dtypes(include='object').columns.tolist()


numeric_pipeline = Pipeline([
    ('winsorize', SkewedWinsorizer(skew_threshold=0.5, limit=0.01)),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

column_transformer = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numerical_col),
    ('cat', OneHotEncoderDF(), category_col)
])

# Define the main model with specified hyperparameters
xgb_model_final = XGBRegressor(
    colsample_bytree=0.7064654114449902, 
    n_estimators=6875,
    gamma=0.18279975644986343, 
    learning_rate=0.01223882034412844, 
    max_depth=5, 
    reg_alpha=0.05893643440612434, 
    reg_lambda=0.7420718776978183, 
    subsample=0.9234945896121194
)

# Build the pipeline
model = Pipeline([
    ('feature_selector', SHAPFeatureSelector(top_n=10)),  # SHAP-based feature selection
    ('xgb_model', xgb_model_final)                        # Final model
])


column_transformer.fit(X)

feature_names = []
for name, trans, cols in column_transformer.transformers_:
    if hasattr(trans, 'get_feature_names_out'):
        out_names = trans.get_feature_names_out()
    else:
        out_names = cols
    feature_names.extend(out_names)



processor = Pipeline([
    ('column_transform', column_transformer),
    ('to_df', ToDataFrame(feature_names=feature_names))
])



life_expectancy_predictor = Pipeline([
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

