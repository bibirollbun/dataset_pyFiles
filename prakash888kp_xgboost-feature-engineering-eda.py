# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from cuml.preprocessing import TargetEncoder
from itertools import combinations
from tqdm.auto import tqdm


# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

# Pandas settings
pd.options.mode.copy_on_write = True


def basic_eda(df, name="Dataset"):
    """Perform basic Exploratory Data Analysis."""
    print(f"\n----- {name} EDA -----")
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print(f"Duplicated rows: {df.duplicated().sum()}")
    
    # Plotting missing values heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title(f'Missing Values Heatmap - {name}')
    plt.show()

def process_combinations_fast(df, columns_to_encode, pair_sizes, max_batch_size=2000):
    """Generate feature combinations and encode them."""
    str_df = df[columns_to_encode].astype(str)
    le = LabelEncoder()
    total_new_cols = 0

    for r in pair_sizes:
        print(f"\nProcessing {r}-combinations...")
        n_combinations = np.math.comb(len(columns_to_encode), r)
        print(f"Total {r}-combinations: {n_combinations}")
        
        combos_iter = combinations(columns_to_encode, r)
        batch_cols, batch_names = [], []

        with tqdm(total=n_combinations) as pbar:
            while True:
                batch_cols.clear()
                batch_names.clear()

                for _ in range(max_batch_size):
                    try:
                        cols = next(combos_iter)
                        batch_cols.append(list(cols))
                        batch_names.append('+'.join(cols))
                    except StopIteration:
                        break

                if not batch_cols:
                    break

                for cols, new_name in zip(batch_cols, batch_names):
                    result = str_df[cols[0]].copy()
                    for col in cols[1:]:
                        result += str_df[col]
                    df[new_name] = le.fit_transform(result) + 1
                    pbar.update(1)

                total_new_cols += len(batch_cols)

        print(f"Completed {r}-combinations. Total columns now: {len(df.columns)}")
    
    return df

def learning_rate_scheduler(epoch):
    """Dynamic learning rate scheduler for XGBoost."""
    return 0.05 if epoch < 115 else 0.01


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
# df_original = pd.read_csv("/kaggle/input/orginal-podcast-dataset/podcast_dataset.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


# Concatenate datasets
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# Drop ID column
df.drop(columns=['id'], inplace=True)

# Drop duplicate entries
df = df.drop_duplicates()


df1 = df.copy()
df1


df1["Listening_Eff"] = df1["Listening_Time_minutes"] / df1["Episode_Length_minutes"]
genre = df1.groupby("Genre")["Listening_Eff"].mean().sort_values(ascending=False)
print(genre)


plt.figure(figsize=(10, 6))
sns.barplot(x=genre.values, y=genre.index, palette="viridis")
plt.title("Average Listening Efficiency by Genre")
plt.xlabel("Listening_Time Eff")
plt.ylabel("Genre")
plt.show()


basic_eda(df, "Combined Dataset")


# Outlier treatment
df['Episode_Length_minutes'] = np.clip(df['Episode_Length_minutes'], 0, 120)
df['Host_Popularity_percentage'] = np.clip(df['Host_Popularity_percentage'], 20, 100)
df['Guest_Popularity_percentage'] = np.clip(df['Guest_Popularity_percentage'], 0, 100)
df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = 0



# Categorical Encoding
day_mapping = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
time_mapping = {'Morning':1, 'Afternoon':2, 'Evening':3, 'Night':4}
sentiment_mapping = {'Negative':1, 'Neutral':2, 'Positive':3}


df['Publication_Day'] = df['Publication_Day'].map(day_mapping)
df['Publication_Time'] = df['Publication_Time'].map(time_mapping)
df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_mapping)


# Feature correction
df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '', regex=True).astype(int)



# Label Encoding for object types
le = LabelEncoder()
for col in df.select_dtypes('object').columns:
    df[col] = le.fit_transform(df[col]) + 1



# Polynomial features
for col in ['Episode_Length_minutes']:
    df[f"{col}_sqrt"] = np.sqrt(df[col])
    df[f"{col}_squared"] = df[col] ** 2

# Episode length mean encoding
group_cols = ['Episode_Sentiment', 'Genre', 'Publication_Day', 'Podcast_Name', 'Episode_Title',
              'Guest_Popularity_percentage', 'Host_Popularity_percentage', 'Number_of_Ads']

for col in tqdm(group_cols, desc="Creating group mean features"):
    df[f"{col}_EP"] = df.groupby(col)['Episode_Length_minutes'].transform('mean')

# Combination features
combo_columns = ['Episode_Length_minutes', 'Episode_Title', 'Publication_Time', 'Host_Popularity_percentage', 
                 'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Podcast_Name', 'Genre', 'Guest_Popularity_percentage']
df = process_combinations_fast(df, combo_columns, pair_sizes=[2, 3, 5, 7], max_batch_size=1000)

# Downcasting
df = df.astype('float32')


df_train = df.iloc[:-len(df_test)]
df_test = df.iloc[-len(df_test):].reset_index(drop=True)

df_train = df_train[df_train['Listening_Time_minutes'].notnull()]
target = df_train.pop('Listening_Time_minutes')
df_test = df_test.drop(columns=['Listening_Time_minutes'])



seed = 42
cv = KFold(n_splits=7, random_state=seed, shuffle=True)
pred_test = np.zeros((250000,))

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': seed,
    'max_depth': 19,
    'learning_rate': 0.03,
    'min_child_weight': 50,
    'reg_alpha': 5,
    'reg_lambda': 1,
    'subsample': 0.85,
    'colsample_bytree': 0.6,
    'colsample_bynode': 0.5,
    'device': "cuda"
}

lr_callback = xgb.callback.LearningRateScheduler(learning_rate_scheduler)

for fold, (idx_train, idx_valid) in enumerate(cv.split(df_train)):
    print(f"\n--- Fold {fold+1} ---")
    
    X_train, y_train = df_train.iloc[idx_train], target.iloc[idx_train]
    X_valid, y_valid = df_train.iloc[idx_valid], target.iloc[idx_valid]
    X_test = df_test[X_train.columns].copy()

    features = df_train.columns
    encoder = TargetEncoder(n_folds=5, seed=seed, stat="mean")

    # Apply Target Encoding
    for col in tqdm(features[:20], desc="Target Encoding first 20 features"):
        X_train[f"{col}_te1"] = encoder.fit_transform(X_train[[col]], y_train)
        X_valid[f"{col}_te1"] = encoder.transform(X_valid[[col]])
        X_test[f"{col}_te1"] = encoder.transform(X_test[[col]])

    for col in tqdm(features[20:], desc="Target Encoding remaining features"):
        X_train[col] = encoder.fit_transform(X_train[[col]], y_train)
        X_valid[col] = encoder.transform(X_valid[[col]])
        X_test[col] = encoder.transform(X_test[[col]])

    # DMatrix creation
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)

    # Model Training
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1_000_000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=30,
        verbose_eval=500,
        callbacks=[lr_callback]
    )

    # Validation and Test Predictions
    val_pred = model.predict(dval)
    pred_test += np.clip(model.predict(dtest), 0, 120)
    print("-" * 70)

pred_test /= 7


df_sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
df_sub['Listening_Time_minutes'] = pred_test
df_sub.to_csv('submission.csv', index=False)

