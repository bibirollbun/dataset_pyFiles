import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib
import io
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


df


df_test.head(10)


podcast_names = df['Podcast_Name'].unique()


podcast_names


podcast_names.size


number_episodes_list = []
for name in podcast_names:
    number_episodes_list.append(df[df['Podcast_Name'] == name]['Episode_Title'].unique().size)
print(number_episodes_list)


df['Episode_Title'] = df['Episode_Title'].str.extract(r'(\d+)').astype(int)
df_test['Episode_Title'] = df_test['Episode_Title'].str.extract(r'(\d+)').astype(int)


df['Publication_Time'].unique()


def publication_time_encoder(df, name_column: str):
    new_publication_time_cos = []
    new_publication_time_sin = []
    for value in df[name_column]:
        if value == 'Morning':
            new_publication_time_cos.append(1)
            new_publication_time_sin.append(0)
        elif value == 'Afternoon':  
            new_publication_time_cos.append(0)
            new_publication_time_sin.append(1)
        elif value == 'Evening':  
            new_publication_time_cos.append(-1)
            new_publication_time_sin.append(0)
        elif value == 'Night':  
            new_publication_time_cos.append(0)
            new_publication_time_sin.append(-1)

    df_new = df.copy()
    df_new = df_new.drop(columns=['Publication_Time'])
    df_new['Publication_Time_cos'] = new_publication_time_cos
    df_new['Publication_Time_sin'] = new_publication_time_sin
    return df_new


df = publication_time_encoder(df, 'Publication_Time')
df_test = publication_time_encoder(df_test, 'Publication_Time')


df


def publication_day_encoder(df_orig, name_column: str):
    df = df_orig.copy()
    
    # Відповідність дня тижня числу (Monday=0, Sunday=6)
    day_to_num = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    df[name_column] = df[name_column].map(day_to_num)

    # Циклічне кодування
    df[name_column + '_cos'] = np.cos(2 * np.pi * df[name_column] / 7)
    df[name_column + '_sin'] = np.sin(2 * np.pi * df[name_column] / 7)

    df = df.drop(columns=[name_column])

    return df


df = publication_day_encoder(df, 'Publication_Day')
df_test = publication_day_encoder(df_test, 'Publication_Day')


df['Episode_Sentiment'].unique()


def episode_sentiment_encoder(df_orig, name_column: str):
    df = df_orig.copy()
    new_column = []
    for value in df[name_column]:
        if value == 'Positive':
            new_column.append(1)
        elif value == 'Negative':  
            new_column.append(-1)
        elif value == 'Neutral':  
            new_column.append(0)

    df[name_column] = new_column
    return df


df = episode_sentiment_encoder(df, 'Episode_Sentiment')
df_test = episode_sentiment_encoder(df_test, 'Episode_Sentiment')


genres = df['Genre'].unique()


genres


genres_distribution_columns = [genre for genre in genres]


def make_genres_distribution(df_orig):
    df = df_orig.copy()
    
    # Створюємо масиви для подкастів і нульових значень
    podcast_names_values = np.array([podcast_names]).T
    genres_distribution_values = np.zeros((len(podcast_names), len(genres_distribution_columns)))
    
    # Об'єднуємо подкастові і жанрові дані
    genres_distribution_concat = np.concatenate((podcast_names_values, genres_distribution_values), axis=1)
    
    # Створюємо DataFrame
    df_gd = pd.DataFrame(genres_distribution_concat, columns=['Podcast_Name'] + genres_distribution_columns)
    
    # Заповнюємо таблицю жанровими даними
    for i, name in enumerate(podcast_names):
        # Отримуємо лічильники жанрів для кожного подкасту
        d = df[df['Podcast_Name'] == name]['Genre'].value_counts().to_dict()
        
        # Для кожного жанру в колонках генеруємо значення
        for genre in genres:
            # Перевіряємо, чи є жанр у словнику, і оновлюємо відповідне значення в таблиці
            df_gd.loc[i, genre] = d.get(genre, 0)
    
    return df_gd


genres_distribution = make_genres_distribution(df)


genres_distribution


df = df.drop(columns=['Genre'])
df_test = df_test.drop(columns=['Genre'])


df_minutes = df[['Episode_Length_minutes', 'Listening_Time_minutes']]
df_minutes = df_minutes.dropna()
df_minutes_describe = df_minutes.describe()
df_minutes_describe['part'] = df_minutes_describe['Listening_Time_minutes'] / df_minutes_describe['Episode_Length_minutes'] * 100


df_minutes_describe


listening_time_minutes_means = {}
for name in podcast_names:
    listening_time_minutes_means[name] = df[df['Podcast_Name'] == name]['Listening_Time_minutes'].mean()


listening_time_minutes_means


listening_time_minutes_mean = df['Listening_Time_minutes'].mean()


df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(listening_time_minutes_mean * 0.7)
df_test['Episode_Length_minutes'] = df_test['Episode_Length_minutes'].fillna(listening_time_minutes_mean * 0.7)


df[df['Number_of_Ads'].isna()]


df[(df['Podcast_Name'] == 'Game Day') & (df['Episode_Title'] == 33) & (df['Publication_Time_cos'] == -1) & (df['Publication_Time_sin'] == 0)]['Number_of_Ads'].median()


df['Number_of_Ads'] = df['Number_of_Ads'].fillna(1.0)


df.plot.box(
    column='Guest_Popularity_percentage',
)
plt.show()


guest_popularity_percentage_mean = df['Guest_Popularity_percentage'].mean()
df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(guest_popularity_percentage_mean)
df_test['Guest_Popularity_percentage'] = df_test['Guest_Popularity_percentage'].fillna(guest_popularity_percentage_mean)


df.isnull().sum()


df


name_to_num = {name: i for i, name in enumerate(podcast_names)}


# Створення енкодера
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Навчання на train + трансформація обох
encoder.fit(df[['Podcast_Name']])

train_encoded = encoder.transform(df[['Podcast_Name']])
test_encoded = encoder.transform(df_test[['Podcast_Name']])

encoder_columns = encoder.get_feature_names_out(['Podcast_Name'])

train_encoded_df = pd.DataFrame(train_encoded, columns=encoder_columns, index=df.index)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder_columns, index=df_test.index)

df = df.drop(columns=['Podcast_Name'])
df = pd.concat([df, train_encoded_df], axis=1)

df_test = df_test.drop(columns=['Podcast_Name'])
df_test = pd.concat([df_test, test_encoded_df], axis=1)


df


df_test.head(10)


X = df.drop(columns=['Listening_Time_minutes', 'id'])
y = df['Listening_Time_minutes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)


def model_regression_report(model, X_test, y_test):
    # Прогнозування
    y_pred = model.predict(X_test)
    
    # Оцінка за допомогою метрик регресії
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    # print(f"Regression report for the model {model_name}:")
    print(f"Root Mean Squared Error (RMSE): {rmse}")


def grid_search_fun(alg, param_grid, X, y):
    grid_search = GridSearchCV(
        alg,
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
    )

    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best accuracy:", grid_search.best_score_)


param_grid_LGBMR = {
    # 'boosting_type': ['gbdt'],
    # 'num_leaves': [261, 271],
    'max_depth': [61, 71, 81],
    # 'num_leaves': [501],
    # 'max_depth': [251],
    # 'learning_rate': [0.1, 0.5],
    # 'n_estimators': [1000, 1100],
    # 'subsample': [1.0, 0.8, 0.6],
    # 'colsample_bytree': [1.0, 0.9, 0.8],
    # 'reg_alpha': [0, 0.1, 0.5],
    # 'reg_lambda': [0, 0.1, 0.5],
    # 'min_child_samples': [40, 50, 60],
    # 'min_split_gain': [0.1, 0.5, 1.0],
}


GrS_LGBMR = 0
if GrS_LGBMR:
    grid_search_fun(
        LGBMRegressor(
            boosting_type = 'gbdt',
            num_leaves = 261,
            max_depth = 71,
            n_estimators = 1000,
            min_child_samples = 50,
            colsample_bytree = 0.8,
        ),
        param_grid_LGBMR,
        X_train,
        y_train,
    )


model_LGBMR = LGBMRegressor(
            boosting_type = 'gbdt',
            num_leaves = 261,
            max_depth = 71,
            n_estimators = 1000,
            min_child_samples = 50,
            colsample_bytree = 0.8,
)
model_LGBMR.fit(X_train, y_train)


model_regression_report(
    model_LGBMR, X_test, y_test
)


df_test.drop(columns=['id'])


y_pred_test = model_LGBMR.predict(df_test.drop(columns=['id']))


df_sample_submission['id'] = df_test['id']
df_sample_submission['Listening_Time_minutes'] = y_pred_test


df_sample_submission


df_sample_submission.to_csv('submission.csv', index=False)

