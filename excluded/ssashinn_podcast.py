# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, roc_curve
from sklearn.compose import make_column_transformer
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold, TimeSeriesSplit as TSS)
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, plot_importance, cv
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
import optuna

from yellowbrick.regressor import ResidualsPlot, PredictionError

pd.set_option('display.max_columns', 100)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
original = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')

target = 'Listening_Time_minutes'

train.drop('id', inplace=True, axis=1)
test.drop('id', inplace=True, axis=1)

train.head()


original_raw_shape = original.shape

original = original.drop_duplicates()
print(f'The length of the original dataset has change from {original_raw_shape[0]} to {original.shape[0]}')


train.describe(include='all')



for name, data in zip(["train", "test", "original"], [train, test, original]):
    print(f"Missing Percentage in {name} dataset\n")
    print((data.isna().sum()/len(data))*100)


print(f'There are {original[target].isna().sum()} missing targets in the external data. We decided to drop them.')

original = original.dropna(subset=target)

# Concatenate original data with synthetics ones
train = pd.concat([train, original], axis=0, ignore_index=True)


def get_avg_episode_length(df):
    return df.groupby('Podcast_Name')['Episode_Length_minutes'].mean().to_dict()

def impute_episode_length(df, avg_dict):
    return df.assign(
        Episode_Length_minutes=df.apply(
            lambda row: avg_dict.get(row['Podcast_Name'], row['Episode_Length_minutes']) 
                        if pd.isna(row['Episode_Length_minutes']) 
                        else row['Episode_Length_minutes'],
            axis=1
        )
    )

train_avg = get_avg_episode_length(train)
test_avg = get_avg_episode_length(test)

train = impute_episode_length(train, train_avg)
test = impute_episode_length(test, test_avg)


# Filling missing values of ads column
train['Number_of_Ads'].fillna(train['Number_of_Ads'].mode()[0], inplace=True)

# Filling missing values of Guest_Popularity_percentage column
train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median(), inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].median(), inplace=True)


num_feat = train.select_dtypes('number').columns.tolist()
print(f'numeric features: {num_feat}\n')
cat_feat = train.select_dtypes(exclude='number').columns.tolist()
print(f'categoric features: {cat_feat}')
target_enc_cols = ['Episode_Title', 'Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']



def numerical_distrib_analysis(data, numerical_features):
    for feature in numerical_features:
        plt.figure(figsize=(12, 5))

        # Histogram with KDE curve
        plt.subplot(1, 2, 1)
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f"Histogram of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        # Boxplot to detect outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[feature])
        plt.title(f"Boxplot of {feature}")

        plt.tight_layout()
        plt.show()

        # Additional statistics
        print(f"\nStatistics for {feature}:")
        print(f"Skewness: {data[feature].skew():.2f}")
        print(f"Missing Values: {data[feature].isnull().sum()}")

numerical_distrib_analysis(train, num_feat)


def categorical_distrib_analysis(data, categorical_features, top_n=10):
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))

        unique_count = data[feature].nunique()

        if unique_count > top_n:
            # Show only the top_n most frequent categories
            top_categories = data[feature].value_counts().nlargest(top_n)
            sns.barplot(x=top_categories.index, y=top_categories.values, palette="pastel")
            plt.title(f"Top {top_n} Categories of {feature}")
        else:
            # Show all categories
            sns.countplot(x=data[feature], order=data[feature].value_counts().index, palette="pastel")
            plt.title(f"Distribution of {feature}")

        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

        # Print stats
        print(f"Feature: {feature}")
        print(f"Number of Unique Values: {unique_count}")
        print(f"Missing Values: {data[feature].isnull().sum()}\n")


categorical_distrib_analysis(train, cat_feat)


def numerical_correlation_analysis(data, numerical_features, target):
    for feature in numerical_features:
        if feature != target:
            # Scatter plot: feature vs target
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=data[feature], y=data[target], alpha=0.5)
            plt.title(f"{feature} vs {target}")
            plt.xlabel(feature)
            plt.ylabel(target)
            plt.show()

    # Correlation matrix
    correlation_matrix = data[numerical_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()


numerical_correlation_analysis(train, num_feat, "Listening_Time_minutes")



def categorical_correlation_analysis(data, categorical_features, target, high_cardinality_threshold=10):
    for feature in categorical_features:
        if data[feature].nunique() <= high_cardinality_threshold:
            # Boxplot: target distribution per category
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data[feature], y=data[target], palette='husl')
            plt.title(f"{feature} vs {target}")
            plt.xlabel(feature)
            plt.ylabel(target)
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f"Skipping {feature}: too many unique values ({data[feature].nunique()})\n")

categorical_correlation_analysis(train, cat_feat, 'Listening_Time_minutes')



X = train.copy()
y = X.pop(target)

X.shape

num_feat = test.select_dtypes('number').columns.tolist()


class OutlierHandler(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = df.copy()
        for col in df.select_dtypes(include=['number']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df


class Feature_Eng(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = df.copy()
        df['Episode_Sentiment'] = df['Episode_Sentiment'].map({'Positive':1, 'Negative':-1, 'Neutral':0})
        df = df.astype({'Podcast_Name': 'category',
                       'Genre': 'category',
                       'Publication_Day': 'category',
                       'Publication_Time': 'category',
                       'Episode_Sentiment': 'category'})
        df['Episode_Numb'] = df['Episode_Title'].apply(lambda x: int(x.split(' ')[1]))
        # df.drop(columns=['Episode_Title'], inplace=True)
        df['Popularity_Average'] = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage'])/2
        df['Ad_Density'] = df.apply(
            lambda row: row['Number_of_Ads'] / row['Episode_Length_minutes']
            if row['Episode_Length_minutes'] != 0 else 0,
            axis=1)
        return df




import category_encoders as ce  

class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.encoder = None

    def fit(self, X, y):
        self.encoder = ce.TargetEncoder(cols=self.cols)
        self.encoder.fit(X[self.cols], y)
        return self

    def transform(self, X):
        X_encoded = X.copy()
        X_encoded[self.cols] = self.encoder.transform(X[self.cols])
        return X_encoded

column_trans = make_column_transformer(
        (RobustScaler(), num_feat),
        remainder = 'passthrough',
        sparse_threshold = 0
    )

prep_pipeline = Pipeline([
    ('outlier_handler', OutlierHandler()),
    ('feature_eng', Feature_Eng()),
    ('target_enc', TargetEncoderTransformer(cols=target_enc_cols)),
    ('column_trans', column_trans)
])

prep_pipeline


def objective_xgb(trial):
    xgb_param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001,0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),  # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),  # L2 regularization
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    model = Pipeline([
    ('prep', prep_pipeline),
    ('xgb', XGBRegressor(**xgb_best_grid, verbose=50))
])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def Run_Pass_xgb_study(n_trials=1):
    if n_trials > 1:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_xgb, n_trials=n_trials, timeout=36000, show_progress_bar=True)
        best_study_params = study.best_params

        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")
        best_study_params = {'n_estimators': 930, 
                              'learning_rate': 0.031629338917482916, 
                              'max_depth': 15, 
                              'subsample': 0.9468611176708397, 
                              'colsample_bytree': 0.8042585491461298, 
                              'reg_alpha': 9.193697404267928, 
                              'reg_lambda': 2.4499368941014024, 
                              'min_child_weight': 3}

    print(f"Best parameters: {best_study_params}")
    return best_study_params

xgb_best_params = Run_Pass_xgb_study(n_trials=1)


def objective_rf(trial):
    rf_param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 1200, 20),
        "max_depth": trial.suggest_int("max_depth", 15, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 7,14),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
    }

    model = Pipeline([
    ('prep', prep_pipeline),
    ('rf', RandomForestRegressor(**rf_param_grid, n_jobs=-1, verbose =1))
])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def Run_Pass_rf_study(n_trials=1):
    if n_trials > 1:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_rf, n_trials=n_trials, timeout=36000, show_progress_bar=True)
        best_study_params = study.best_params

        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")
        best_study_params = {
        'n_estimators': 800, 
        'max_depth': 30, 
        'min_samples_split': 10, 
        'min_samples_leaf': 3}
        #score: 12.75
    print(f"Best parameters: {best_study_params}")
    return best_study_params

rf_best_params = Run_Pass_rf_study(n_trials=1)


my_spliter = KFold(n_splits=5, shuffle=True)

xgb_model = XGBRegressor(**xgb_best_params, verbose=1, n_jobs=-1)
rf_model = RandomForestRegressor(**rf_best_params, n_jobs=-1)

meta_model = LinearRegression()

stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model)
    ],
    final_estimator=meta_model
)
model = Pipeline([
    ('prep', prep_pipeline),  
    ('stacking', stacking_model)  
])

# cv_splits = my_spliter.split(X, y)
# scores = []

# for f, (train_idx, val_idx) in enumerate(cv_splits, start=1):
#     X_train, X_val = X.loc[train_idx], X.loc[val_idx]
#     y_train, y_val = y.loc[train_idx], y.loc[val_idx]

#     print(f'Fold_{f}')

#     model.fit(X_train, y_train)
    
#     val_pred = model.predict(X_val)
#     score = np.sqrt(mean_squared_error(y_val, val_pred))
#     scores.append(score)
    
#     print(f'RMSE: {score:.8f}\n')

#     print(f'\nAverage RMSE: {np.mean(scores):.8f} ± {np.std(scores):.8f}\n')


# final_model = Pipeline([
#             ('outlier_handler', OutlierHandler()),
#             ('feature_eng', Feature_Eng()),
#             ('column_trans', column_trans),
#             ('xgb', XGBRegressor(**xgb_best_params))
#         ])

final_model = Pipeline([
    ('prep', prep_pipeline),
    ('stacking',stacking_model)
])
final_model.fit(X, y)


# X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

# fig, ax = plt.subplots(figsize=(6, 4))
# resid = ResidualsPlot(final_model)
# resid.fit(X_tr, y_tr)
# resid.score(X_va, y_va)
# resid.poof();


# Load the sample submission file
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv').copy()

# Predict the target in the test data
sub['Listening_Time_minutes'] = final_model.predict(test).tolist()
# sub_0['Listening_Time_minutes'] = test_preds_series

# Safe the csv submission file
sub.to_csv('submission.csv', index=False)

# Display the submission file
display(sub.head(10))
print('The file is ready for submission!')

