import numpy as np
import pandas as pd


try:
    training_dataset = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
    test_dataset = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
except FileNotFoundError:
    print(f"File not found")


training_dataset.head()


training_dataset.info()


categorical_columns = training_dataset.select_dtypes(include='object').columns
categorical_columns


training_dataset['Genre'].unique()   # one hot encoding


training_dataset['Publication_Day'].unique() # treating as cycle with sin/cos


training_dataset['Publication_Time'].unique() # treating as cycle with sin/cos


training_dataset['Episode_Sentiment'].unique() # ordinal -1, 0, 1


from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor


X = training_dataset.iloc[:, :-1]
y = training_dataset.iloc[:, -1]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}

def cycle_encoder(data, mapping):
    data_mapped = data.map(mapping)

    # scaling to 360 degrees so that it forms a circle (cycle) and normalizing
    alpha = data_mapped * np.pi * 2 / len(mapping)

    data_transformed = pd.DataFrame({
        'sin': np.sin(alpha),
        'cos': np.cos(alpha)
    })

    return data_transformed


sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

def sentiment_encoder(data, mapping):
    data = data.map(mapping)
    data = data.to_frame()
    return data


def episode_encoder(data):
    data = data.str.lower()
    data = data.str.replace('episode ', '')
    return data.astype(int).to_frame()


# pipeline for numeric columns
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])


# pipeline for text columns 
text_pipeline = Pipeline(steps=[
    ('tfidf', TfidfVectorizer()),
    ('svd', TruncatedSVD(n_components=50))  # Apply SVD to reduce dimensionality for text data
])


preprocessor = ColumnTransformer(
transformers = [
    ('genre_ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Genre']),
    ('publication_day', FunctionTransformer(lambda x: cycle_encoder(x, day_mapping)), 'Publication_Day'),
    ('publication_time', FunctionTransformer(lambda x: cycle_encoder(x, time_mapping)), 'Publication_Time'),
    ('sentiment_ord', FunctionTransformer(lambda x: sentiment_encoder(x, sentiment_mapping)), 'Episode_Sentiment'),
    ('episode_no', FunctionTransformer(lambda x: episode_encoder(x)), 'Episode_Title'),
    ('podcast_tfidf', text_pipeline, 'Podcast_Name'),
    ('num_pipeline', num_pipeline, ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Host_Popularity_percentage', 'Number_of_Ads'])
    ],
    remainder='passthrough'
)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_regression, k=50)),
    ('regressor', RandomForestRegressor(n_estimators=50, n_jobs=-1))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_val)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_regression_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Model Evaluation Metrics:")
    print(f"--------------------------")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

evaluate_regression_model(y_val, y_pred)


y_pred = pipeline.predict(test_dataset)


df = pd.DataFrame({'id': test_dataset['id'], 'Listening_Time_minutes': y_pred})

df.to_csv('predictions.csv', index=False)

