import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA

import xgboost as xgb
import lightgbm as lgb

import optuna
from gensim.models import Word2Vec


sns.set_style("whitegrid")
sns.set_palette("Blues")


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train.head()


train.info()


def create_summary(df):
    describe = df.describe().transpose()
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary["MissingValues"] = df.isna().sum()
    summary["UniqueValues"] = df.nunique()
    summary["Value_1"] = df.iloc[0]
    summary["Value_2"] = df.iloc[1]
    summary["Value_3"] = df.iloc[2]
    summary = pd.concat([summary, describe], axis=1)
    
    return summary

create_summary(train)


create_summary(test)


# Fill na for Guest_Popularity_percentage = 0
train['Guest_Popularity_percentage'].fillna(0, inplace=True)
test['Guest_Popularity_percentage'].fillna(0, inplace=True)

# Fill na for Number of ads
train['Number_of_Ads'].fillna(0, inplace=True)
test['Number_of_Ads'].fillna(0, inplace=True)

# Fill na for Episode_Length_minutes with median values
train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(), inplace=True)
test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(), inplace=True)



train.isna().sum(), test.isna().sum()


cols = 3
rows = int(np.ceil(len(train.columns) / cols))

fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(20,20))
ax = ax.flatten()

plt.suptitle("Visualize all features",size=24, y=1.01)

for i,col in enumerate(train.columns):
    if train[col].dtype == float or train[col].dtype == int:
        sns.boxplot(data=train,y=col,ax=ax[i],orient="vertical")
        ax[i].set_title(f"{col}")
    else:
        sns.countplot(data=train,x=col,ax=ax[i])
        ax[i].set_title(f"{col}")
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)

# Remove empty subplots
for i in range(len(train.columns), len(ax)):
    fig.delaxes(ax[i])

plt.tight_layout()
plt.show()



# Prepare data for Word2Vec
# Combine Podcast_Name and Genre into a single list of words
train['Combined_Text'] = train['Podcast_Name'] + ' ' + train['Genre']
test['Combined_Text'] = test['Podcast_Name'] + ' ' + test['Genre']

# Tokenize the text into lists of words
train_sentences = train['Combined_Text'].apply(lambda x: x.split()).tolist()
test_sentences = test['Combined_Text'].apply(lambda x: x.split()).tolist()

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=train_sentences + test_sentences, vector_size=20, window=5, min_count=1, workers=4)

# Generate embeddings for Podcast_Name and Genre
def get_word2vec_embeddings(text, model, vector_size):
    words = text.split()
    embeddings = np.zeros(vector_size)
    for word in words:
        if word in model.wv:
            embeddings += model.wv[word]
    return embeddings / len(words) if len(words) > 0 else embeddings

# Apply embeddings to the dataset
train['Podcast_Genre_Embedding'] = train['Combined_Text'].apply(lambda x: get_word2vec_embeddings(x, word2vec_model, 20))
test['Podcast_Genre_Embedding'] = test['Combined_Text'].apply(lambda x: get_word2vec_embeddings(x, word2vec_model, 20))

# Flatten embeddings
train_embeddings = np.vstack(train['Podcast_Genre_Embedding'].values)
test_embeddings = np.vstack(test['Podcast_Genre_Embedding'].values)

# Drop the original text columns
train.drop(columns=['Podcast_Name', 'Combined_Text', 'Podcast_Genre_Embedding'], inplace=True)
test.drop(columns=['Podcast_Name', 'Combined_Text', 'Podcast_Genre_Embedding'], inplace=True)

# Combine embeddings with other features
train = pd.concat([train.reset_index(drop=True), pd.DataFrame(train_embeddings)], axis=1)
test = pd.concat([test.reset_index(drop=True), pd.DataFrame(test_embeddings)], axis=1)

# Ensure column names are strings
train.columns = train.columns.astype(str)
test.columns = test.columns.astype(str)


ordinal_encoder = OrdinalEncoder()

# Encode categorical features
train['Genre_ordinal'] = ordinal_encoder.fit_transform(train[['Genre']])
test['Genre_ordinal'] = ordinal_encoder.transform(test[['Genre']])


train.head()


# Extract the number from episode title
train['Episode_Title'] = train['Episode_Title'].str.extract(r'(\d+)')
test['Episode_Title'] = test['Episode_Title'].str.extract(r'(\d+)')

# Convert to numeric
train['Episode_Title'] = pd.to_numeric(train['Episode_Title'], errors='coerce')
test['Episode_Title'] = pd.to_numeric(test['Episode_Title'], errors='coerce')



# Ordinal encode publication_day into day of the week numbers
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
train['Publication_Day'] = train['Publication_Day'].apply(lambda x: days.index(x))
test['Publication_Day'] = test['Publication_Day'].apply(lambda x: days.index(x))

# Ordinal encode publication_time in to 4 time slots
time = ['Morning', 'Afternoon', 'Evening', 'Night']
train['Publication_Time'] = train['Publication_Time'].apply(lambda x: time.index(x))
test['Publication_Time'] = test['Publication_Time'].apply(lambda x: time.index(x))

# Ordinal encode episode_sentiment into 3 categories
sentiment = ['Negative', 'Neutral', 'Positive']
train['Episode_Sentiment'] = train['Episode_Sentiment'].apply(lambda x: sentiment.index(x))
test['Episode_Sentiment'] = test['Episode_Sentiment'].apply(lambda x: sentiment.index(x))


# Cyclic encoding of Publication_Day and Publication_Time
def cyclic_encode(df, col, max_val):
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df

train = cyclic_encode(train, 'Publication_Day', 7)
test = cyclic_encode(test, 'Publication_Day', 7)
train = cyclic_encode(train, 'Publication_Time', 4)
test = cyclic_encode(test, 'Publication_Time', 4)




train.head()


def create_features(df):
    # Episode_length from mean, std and median
    episode_length_mean = df['Episode_Length_minutes'].mean()
    episode_length_std = df['Episode_Length_minutes'].std()
    episode_length_median = df['Episode_Length_minutes'].median()
    episode_length_min = df['Episode_Length_minutes'].min()
    episode_length_max = df['Episode_Length_minutes'].max()
    episode_length_range = episode_length_max - episode_length_min
    episode_length_iqr = df['Episode_Length_minutes'].quantile(0.75) - df['Episode_Length_minutes'].quantile(0.25)
    episode_length_25 = df['Episode_Length_minutes'].quantile(0.25)
    episode_length_75 = df['Episode_Length_minutes'].quantile(0.75)
    episode_length_90 = df['Episode_Length_minutes'].quantile(0.90)

    # Episode_Length from min and max
    # df['Episode_Length_delta_mean'] = (df['Episode_Length_minutes'] - episode_length_mean)
    # df['Episode_Length_delta_std'] = (df['Episode_Length_minutes'] - episode_length_std)
    # df['Episode_Length_delta_median'] = (df['Episode_Length_minutes'] - episode_length_median)
    # df['Episode_Length_delta_min'] = (df['Episode_Length_minutes'] - episode_length_min)
    # df['Episode_Length_delta_max'] = (df['Episode_Length_minutes'] - episode_length_max)
    # df['Episode_Length_delta_range'] = (df['Episode_Length_minutes'] - episode_length_range)
    df['Episode_Length_delta_iqr'] = (df['Episode_Length_minutes'] - episode_length_iqr)
    df['Episode_Length_delta_25'] = (df['Episode_Length_minutes'] - episode_length_25)
    df['Episode_Length_delta_75'] = (df['Episode_Length_minutes'] - episode_length_75)
    df['Episode_Length_delta_90'] = (df['Episode_Length_minutes'] - episode_length_90)

    # Normalized Episode_Length
    # df['Episode_Length_normalized'] = df['Episode_Length_minutes'] / episode_length_max

    # # Interaction features
    # df['Length_x_Ads'] = df['Episode_Length_minutes'] * df['Number_of_Ads']
    df['Length_/_Ads'] = df['Episode_Length_minutes'] / (df['Number_of_Ads'] + 1e-5)
    # df['Length_x_Guest'] = df['Episode_Length_minutes'] * df['Guest_Popularity_percentage']
    df['Length_/_Guest'] = df['Episode_Length_minutes'] / (df['Guest_Popularity_percentage'] + 1e-5)
    # df['Length_x_Host'] = df['Episode_Length_minutes'] * df['Host_Popularity_percentage']
    df['Length_/_Host'] = df['Episode_Length_minutes'] / (df['Host_Popularity_percentage'] + 1e-5)
    # df['Length_x_Title'] = df['Episode_Length_minutes'] * df['Episode_Title']
    # df['Length_/_Title'] = df['Episode_Length_minutes'] / (df['Episode_Title'] + 1e-5)
    # df['Length_x_Guest_Ads'] = df['Episode_Length_minutes'] * df['Guest_Popularity_percentage'] * df['Number_of_Ads']
    # df['Length_x_Guest_Host'] = df['Episode_Length_minutes'] * df['Guest_Popularity_percentage'] * df['Host_Popularity_percentage']
    # df['Length_x_Host_Ads'] = df['Episode_Length_minutes'] * df['Host_Popularity_percentage'] * df['Number_of_Ads']
    # df['Length_x_Guest_Title'] = df['Episode_Length_minutes'] * df['Guest_Popularity_percentage'] * df['Episode_Title']
    # df['Length_x_Host_Title'] = df['Episode_Length_minutes'] * df['Host_Popularity_percentage'] * df['Episode_Title']
    # df['Length_x_Guest_Host_Ads'] = df['Episode_Length_minutes'] * df['Guest_Popularity_percentage'] * df['Host_Popularity_percentage'] * df['Number_of_Ads']
    # df['Length_x_Guest_Host_Title'] = df['Episode_Length_minutes'] * df['Guest_Popularity_percentage'] * df['Host_Popularity_percentage'] * df['Episode_Title']
    # df['Length_x_Guest_Ads_Title'] = df['Episode_Length_minutes'] * df['Guest_Popularity_percentage'] * df['Number_of_Ads'] * df['Episode_Title']
    # df['Length_x_Host_Ads_Title'] = df['Episode_Length_minutes'] * df['Host_Popularity_percentage'] * df['Number_of_Ads'] * df['Episode_Title']
    # df['Length_x_Genre'] = df['Episode_Length_minutes'] * df['Genre_ordinal']
    # df['Length_x_Sentiment'] = df['Episode_Length_minutes'] * df['Episode_Sentiment']
    # df['Length_x_Publication_Day'] = df['Episode_Length_minutes'] * df['Publication_Day']
    # df['Length_x_Publication_Time'] = df['Episode_Length_minutes'] * df['Publication_Time']

    # Binning features
    df['Episode_Length_bins'] = pd.cut(df['Episode_Length_minutes'], bins=[0, 20, 40, 60, 80, 100, 120, float('inf')], labels=[1, 2, 3, 4, 5, 6, 7], include_lowest=True).astype('category')
    df['Episode_Title_bins'] = pd.cut(df['Episode_Title'], bins=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], include_lowest=True).astype('category')
    df['Episode_Length_bins_num'] = df['Episode_Length_bins'].cat.codes
    df['Episode_Title_bins_num'] = df['Episode_Title_bins'].cat.codes
    df['Host_Popularity_bins'] = pd.cut(df['Host_Popularity_percentage'], bins=[0, 20, 40, 60, 80, 100, 120], labels=[1, 2, 3, 4, 5, 6]).astype('category')
    df['Host_Popularity_bins_num'] = df['Host_Popularity_bins'].cat.codes
    df['Guest_Popularity_bins'] = pd.cut(df['Guest_Popularity_percentage'], bins=[0, 20, 40, 60, 80, 100, 120, float('inf')], labels=[1, 2, 3, 4, 5, 6, 7], include_lowest=True).astype('category')
    df['Guest_Popularity_bins_num'] = df['Guest_Popularity_bins'].cat.codes

    # Add a random column to the dataframe
    # df['Random_Noise'] = np.random.rand(len(df))
    
    # Drop original columns
    df.drop(columns=['Publication_Day', 'Publication_Time'], inplace=True)
    
    return df


train = create_features(train)
test = create_features(test)


X = train.drop('Listening_Time_minutes', axis=1)
y = train['Listening_Time_minutes']
X_test = test.copy()


# Feature Importance
def plot_feature_importance(model, X, y):
    # Create a copy of X to avoid modifying the original DataFrame
    X_copy = X.drop('Genre', axis=1)
    y_copy = y.copy()

    # Fit the model
    model.fit(X_copy, y_copy)
    feature_importances = model.feature_importances_
    feature_names = X_copy.columns
    indices = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(15, 6))
    plt.title("Feature Importances")
    plt.bar(range(X_copy.shape[1]), feature_importances[indices], align="center")
    plt.xticks(range(X_copy.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, X_copy.shape[1]])
    plt.show()

# Plot feature importance for Random Forest
rf_model = RandomForestRegressor(n_estimators=10, random_state=51)
plot_feature_importance(rf_model, X, y)


# # Drop all columns with lower importance than Random_Noise
# threshold = rf_model.feature_importances_[X.columns.get_loc('Random_Noise')]
# low_importance_columns = X.columns[rf_model.feature_importances_ <= threshold]


# X.drop(columns=low_importance_columns, inplace=True)
# X_test.drop(columns=low_importance_columns, inplace=True)


X.shape, X_test.shape


num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            # ('pca', PCA(n_components='mle', svd_solver='full'))
        ]), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=51)


X_train.shape, X_val.shape, y_train.shape, y_val.shape


models = {
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet(),
    'Lasso': Lasso(),
    'DecisionTree': DecisionTreeRegressor(),
    'KNeighbors': KNeighborsRegressor(),
    'LightGBM': lgb.LGBMRegressor(verbose=-1),
    'XGBoost': xgb.XGBRegressor(),
}

results = {}
# Loop through the models and fit them
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

    pipeline.fit(X_train, y_train)
    
    # Cross-validation
    scores = cross_val_score(pipeline, X_val, y_val, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    
    print(f"{name} RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
    results[name] = rmse_scores.mean()
    
    # Predict on validation set
    y_pred = pipeline.predict(X_val)
    
    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"{name} Validation RMSE: {rmse:.2f}, MAE: {mae:.2f}\n")


# Visualize the results in a bar plot
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['RMSE'])
results_df = results_df.sort_values(by='RMSE', ascending=True)
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y='RMSE', data=results_df)
plt.xticks(rotation=45)
plt.title('Model RMSE Comparison')
plt.xlabel('Model')
plt.ylabel('RMSE')

# Add result values on top of bars
for index, value in enumerate(results_df['RMSE']):
    plt.text(index, value + 0.1, f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Set optuna logging level
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):

    # Define hyperparameters to tune
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': 51,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }

    model = lgb.LGBMRegressor(**params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Cross-validation
    scores = -1 * cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(scores).mean()

    return rmse_scores

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, n_jobs=-1)

# Get the best hyperparameters
best_params = study.best_params
best_value = study.best_value
print("Best Hyperparameters: ", best_params)
print("Best RMSE: ", best_value)


best_model = lgb.LGBMRegressor(**best_params, verbose=-1, objective='regression', metric='rmse', random_state=51)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
print(f"Best Model RMSE: {rmse:.2f}, MAE: {mae:.2f}\n")


final_model = lgb.LGBMRegressor(**best_params, verbose=-1, objective='regression', metric='rmse', random_state=51)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', final_model)])
pipeline.fit(X, y)


y_test = pipeline.predict(X_test)


y_test = np.clip(y_test, 0, None)


submission['Listening_Time_minutes'] = y_test


sns.histplot(train['Listening_Time_minutes'], label='Train Data', kde=True, multiple='dodge')
sns.histplot(submission['Listening_Time_minutes'], label='Test Data', kde=True, multiple='dodge')
plt.title('Distribution of Predicted Listening Time')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Density')
plt.legend()
plt.show()


submission.to_csv('submission_bbg007.csv', index=False)




