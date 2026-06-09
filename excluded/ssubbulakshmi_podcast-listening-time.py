# Import required libraries
import numpy as np
import pandas as pd

# For visualization techniques
import seaborn as sns
import matplotlib.pyplot as plt

# For data splits
from sklearn.model_selection import train_test_split


# Load train and test data from the input CSV files
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
X_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")




# Get all the column info and missing  values from train df
train_df.info()


# Split input features and target variable
X = train_df.drop(columns=['Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']


# View sample input data
X.sample(5)


# View sample target data
y.sample(5)


# Find number of missing values in the columns of X
X.info()


print(f"Number of unique values: {X['Podcast_Name'].nunique()}")
print(f"Unique values: {X['Podcast_Name'].unique()}")


print(f"Number of unique values: {X['Episode_Title'].nunique()}")
print(f"Unique values: {X['Episode_Title'].unique()}")


X['Episode_Title'].value_counts()


print(f"Number of unique values: {X['Genre'].nunique()}")
print(f"Unique values: {X['Genre'].unique()}")


print(f"Number of unique values: {X['Publication_Day'].nunique()}")
print(f"Unique values: {X['Publication_Day'].unique()}")


X['Publication_Day'].value_counts()


print(f"Number of unique values: {X['Publication_Time'].nunique()}")
print(f"Unique values: {X['Publication_Time'].unique()}")


X['Publication_Time'].value_counts()


print(f"Number of unique values: {X['Episode_Sentiment'].nunique()}")
print(f"Unique values: {X['Episode_Sentiment'].unique()}")


X['Episode_Sentiment'].value_counts()


sns.histplot(X['Episode_Length_minutes'], kde=True, bins=100)
plt.title('Distribution of Episode length in minutes')
plt.show()


sns.boxplot(X['Episode_Length_minutes'])


X['Episode_Length_minutes'].describe()


Q1 = X['Episode_Length_minutes'].quantile(0.25) # 25th percentile
Q3 = X['Episode_Length_minutes'].quantile(0.75) # 75th percentile
print(f"Q1 = {Q1}, Q3 = {Q3}")

IQR = Q3 - Q1
print(f"IQR = {IQR}")

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)
print(f"Lower bound = {lower_bound}, Upper bound = {upper_bound}")


# Outlier in Episode_Length_minutes
X[X['Episode_Length_minutes'] < lower_bound]


# Outlier in Episode_Length_minutes
X[X['Episode_Length_minutes'] > upper_bound]


sns.boxplot(X['Host_Popularity_percentage'])


# 5-point summary of Host_Popularity_percentage
X['Host_Popularity_percentage'].describe()


Q1 = X['Host_Popularity_percentage'].quantile(0.25)
Q3 = X['Host_Popularity_percentage'].quantile(0.75)
print(f"Q1 = {Q1}, Q3 = {Q3}")

IQR = Q3 - Q1
print(f"IQR = {IQR}")

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)
print(f"Lower bound = {lower_bound}, Upper bound = {upper_bound}")


sns.histplot(X['Guest_Popularity_percentage'], kde=True, bins=100)
plt.title('Distribution of Guest popularity percentage')
plt.show()


sns.boxplot(X['Guest_Popularity_percentage'])


Q1 = X['Guest_Popularity_percentage'].quantile(0.25)
Q3 = X['Guest_Popularity_percentage'].quantile(0.75)
print(f"Q1 = {Q1}, Q3 = {Q3}")

IQR = Q3 - Q1
print(f"IQR = {IQR}")

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)
print(f"Lower bound = {lower_bound}, Upper bound = {upper_bound}")


# 5-point summary of Guest_Popularity_percentage
X['Guest_Popularity_percentage'].describe()


sns.histplot(X['Number_of_Ads'], kde=True, bins=100)
plt.title('Distribution of Number of Ads')
plt.show()


sns.boxplot(X['Number_of_Ads'])


Q1 = X['Number_of_Ads'].quantile(0.25)
Q3 = X['Number_of_Ads'].quantile(0.75)
print(f"Q1 = {Q1}, Q3 = {Q3}")

IQR = Q3 - Q1
print(f"IQR = {IQR}")

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)
print(f"Lower bound = {lower_bound}, Upper bound = {upper_bound}")


X['Number_of_Ads'].describe()


# Split into train(85%) and val(15%) splits
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=1000)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# For dropping unnecessary features
class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1) # drop by axis=1==>column wise


# For text preprocessing
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, preproc_functions=None):
        # preproc_functions dictionary with key as feature name mapped with preprocessing functions for that feature
        self.preproc_functions = preproc_functions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Find string columns and apply general preprocessing steps
        str_cols = X.select_dtypes(include=['object']).columns
        
        for col in str_cols:
            X[col] = (
                X[col]
                .astype(str) # String type conversion
                .str.lower() # convert to lower case
                .str.strip() # Strip unnecessary spaces
                .str.replace(' ', '_') # Replace space between the words and replace with underscore
            )
        if self.preproc_functions:
            # Apply configured preprocessing function for the specific features
            for feature, preproc_func in self.preproc_functions.items():
                X[feature] = X[feature].apply(preproc_func)

        return X


class AutoImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='median')  # For numeric columns
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')  # For categorical columns
    
    def fit(self, X, y=None):
        # Automatically infer numeric and categorical columns based on the input data
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Fit imputers on the respective columns
        self.numeric_imputer.fit(X[self.numeric_cols])
        self.categorical_imputer.fit(X[self.categorical_cols])
        
        return self
    
    def transform(self, X):
        
        # Apply imputation
        X[self.numeric_cols] = self.numeric_imputer.transform(X[self.numeric_cols])
        X[self.categorical_cols] = self.categorical_imputer.transform(X[self.categorical_cols])
        
        return X



class FeatureEnggTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_functions=None):
        self.feature_functions = feature_functions or {} # Dictionary with key: new feature to be created and value is the tuple having feature name and function to be applied on feature
        self.statistics = {}  # Store statistics like Q1, Q3 for each feature

    def fit(self, X, y=None): 
        if self.feature_functions:
            func_mappings = list(self.feature_functions.values())
            # Statistics learnt from train data
            for func_mapping in func_mappings:
                feature = func_mapping[0]
                Q1 = X[feature].quantile(0.25)
                Q3 = X[feature].quantile(0.75)
                self.statistics[feature] = {'Q1': Q1, 'Q3': Q3}
        return self

    def transform(self, X):
        for new_feature, (existing_feature, func) in self.feature_functions.items():
            if existing_feature in X.columns:
                Q1 = self.statistics.get(existing_feature, {}).get('Q1', 0)
                Q3 = self.statistics.get(existing_feature, {}).get('Q3', 0)
                X[new_feature] = X[existing_feature].apply(func, args=(Q1, Q3))  # Pass Q1, Q3 to the function
        return X


def categorize_episode_duration(x, Q1, Q3):
    """
    Categorize the episode duration based on Q1 and Q3 values.
    """
    if x < Q1:
        return 'short'
    elif x <= Q3:
        return 'regular'
    else:
        return 'long'

def categorize_popularity(x, Q1, Q3):
    """
    Categorize popularity based on Q1 and Q3 values.
    """
    if x < Q1:
        return 'less_popular'
    elif x <= Q3:
        return 'popular'
    else:
        return 'more_popular'


class DynamicEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_mappings=None):
        self.ordinal_mappings = ordinal_mappings or {}
        self.encoder = None
        self.nominal_cols = []
        self.ordinal_cols = []

    def _create_encoder(self, X):
        self.ordinal_cols = list(self.ordinal_mappings.keys())
        self.nominal_cols = [
            col for col in X.select_dtypes(include='object').columns
            if col not in self.ordinal_cols
        ]

        nominal_encoder = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        ordinal_encoder = Pipeline([
            ('encoder', OrdinalEncoder(categories=[self.ordinal_mappings[col] for col in self.ordinal_cols]))
        ])

        return ColumnTransformer(transformers=[
            ('nominal', nominal_encoder, self.nominal_cols),
            ('ordinal', ordinal_encoder, self.ordinal_cols)
        ], remainder='passthrough')

    def fit(self, X, y=None):
        self.encoder = self._create_encoder(X)
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X)



ordinal_mappings = {
    "Episode_Sentiment": ['negative', 'neutral', 'positive'], # Episode_Sentiment
    "episode_duration": ['short', 'regular', 'long'], # episode_duration
    "guest_popularity": ['less_popular', 'popular', 'more_popular'], # guest_popularity
    "host_popularity": ['less_popular', 'popular', 'more_popular'], # host_popularity
}

# Define functions for feature engineering
feature_functions = {
    # Keys are new features
    'episode_duration': ('Episode_Length_minutes', categorize_episode_duration),
    'guest_popularity': ('Guest_Popularity_percentage', categorize_popularity),
    'host_popularity': ('Host_Popularity_percentage', categorize_popularity)
}


from xgboost import XGBRegressor

pipeline = Pipeline(steps=[
    ('dropper', FeatureDropper(columns_to_drop=['id', 'Episode_Title', 'Podcast_Name'])),
    ('text_preprocessor', TextPreprocessor(preproc_functions=None)),
    ('imputer', AutoImputer()),
    ('feature_engg', FeatureEnggTransformer(feature_functions=feature_functions)),
    ('encoder', DynamicEncoder(ordinal_mappings=ordinal_mappings)),
    ('xgb', XGBRegressor(random_state=1000, n_jobs=-1))
])


pipeline.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error

y_val_pred = pipeline.predict(X_val)  # Predict on validation data

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("RMSE:", rmse)


!pip install hyperopt


from hyperopt import hp

search_space = {
    'xgb__n_estimators': hp.quniform('n_estimators', 200, 500, 50),
    'xgb__max_depth': hp.quniform('max_depth', 3, 20, 2),
    'xgb__learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'xgb__subsample': hp.uniform('subsample', 0.7, 1.0),
    'xgb__colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
    'xgb__gamma': hp.uniform('gamma', 0, 5),  # Minimum loss reduction to make a split
    'xgb__reg_alpha': hp.loguniform('reg_alpha', -3, 1),  # L1 regularization
    'xgb__reg_lambda': hp.loguniform('reg_lambda', -3, 1),  # L2 regularization
    'xgb__min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),  # Min sum of instance weight (hessian) in a child
}


from hyperopt import STATUS_OK
from sklearn.metrics import mean_squared_error
import logging
import numpy as np

# Set up logging for hyperparameter search
logging.basicConfig(level=logging.INFO)

def objective(params):
    # Ensure the correct type for these parameters
    params['xgb__n_estimators'] = int(params['xgb__n_estimators'])  # Cast to integer
    params['xgb__min_child_weight'] = int(params['xgb__min_child_weight'])  # Cast to integer
    params['xgb__max_depth'] = int(params['xgb__max_depth']) # Cast to integer
    
    # Setting the model parameters dynamically
    pipeline.set_params(**params)
    
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_val_pred = pipeline.predict(X_val)
    
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    # Log results for progress tracking
    logging.info(f"Trial params: {params}, RMSE: {rmse}")
    
    # Return loss and status for Hyperopt to use
    return {'loss': rmse, 'status': STATUS_OK}



from hyperopt import fmin, tpe, Trials, space_eval
from numpy.random import default_rng
from sklearn.model_selection import KFold

trials = Trials()
rstate = default_rng(42)

best = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=35,
    trials=trials,
    rstate=rstate
)

print("Best Hyperparameters:", best)
# Decode the best parameters found
best_params = space_eval(search_space, best)

# Log the best hyperparameters
print("Best Hyperparameters:", best_params)


best


# Ensure the correct type for these parameters
best_params['xgb__n_estimators'] = int(best_params['xgb__n_estimators'])  # Cast to integer
best_params['xgb__min_child_weight'] = int(best_params['xgb__min_child_weight'])  # Cast to integer
best_params['xgb__max_depth'] = int(best_params['xgb__max_depth']) # Cast to integer
    
# Apply to pipeline
pipeline.set_params(**best_params)
pipeline.fit(X_train, y_train)


y_val_pred = pipeline.predict(X_val)  # Predict on validation data

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("RMSE:", rmse)


X_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


X_test.head()


test_id = X_test['id']
test_id


X_test


y_test = pipeline.predict(X_test)  # Predict on test data


# Submission sample data
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


sample_submission['Listening_Time_minutes'] = y_test


sample_submission.head()


sample_submission.info()


sample_submission.to_csv('submission.csv', index=False)


pd.read_csv("submission.csv")




