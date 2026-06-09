# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import warnings

warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df.head()


df.isna().sum() / len(df) * 100


# Exclude the number of outliers
df = df[df['Number_of_Ads'] < 4]

# Remove outliers
t25th, s75th = df['Episode_Length_minutes'].quantile(.25), df['Episode_Length_minutes'].quantile(.75)
IQR = s75th - t25th
upper = s75th + 1.5*IQR
lower = t25th - 1.5*IQR

df = df[(df['Episode_Length_minutes'] < upper) & (df['Episode_Length_minutes'] > lower)]


df[df['Episode_Length_minutes'] < df['Listening_Time_minutes']]


num_df = df.select_dtypes(include="number")

cat_df = df.select_dtypes(exclude="number")


si_num = SimpleImputer(strategy="mean")
num_df = pd.DataFrame(si_num.fit_transform(num_df), columns=num_df.columns)

si_cat = SimpleImputer(strategy="constant")
cat_df = pd.DataFrame(si_cat.fit_transform(cat_df), columns=cat_df.columns)


num_df['Episode_Number'] = cat_df['Episode_Title'].str.lstrip('Episode ').astype(int)

# for lag in range(1, 4):
#     num_df[f'Episode_Length_minutes_{lag}'] = num_df['Episode_Length_minutes'].shift(lag).fillna(0)


df_concat = pd.concat([num_df, cat_df], axis=1)

df_episode = df_concat.groupby(by=['Podcast_Name', 'Episode_Number'])['Listening_Time_minutes'].sum().reset_index()

plt.figure(figsize=(30, 8))
sns.lineplot(
    data=df_episode, 
    x="Episode_Number", y="Listening_Time_minutes", 
    hue="Podcast_Name", legend=False,
    alpha=.75,
)
plt.show()


df_ep_count = df_episode.groupby('Episode_Number')['Listening_Time_minutes'].sum().reset_index()

sns.lineplot(
    data=df_ep_count,
    x="Episode_Number", y="Listening_Time_minutes",
)
for each in range(0, 102, 2):
    plt.axvline(each, color='r', linewidth=.3, linestyle='--')
plt.title("Listening Time per episodes")
plt.show()


df_ep_count['Listening_Time_minutes_rolling'] = df_ep_count['Listening_Time_minutes'].rolling(window=3).mean().fillna(0)

sns.lineplot(
    data=df_ep_count,
    x='Episode_Number', y='Listening_Time_minutes_rolling',
)
plt.title('Rolling Mean')
plt.show()


df_ep_count['Listening_Time_minutes_rolling_std'] = df_ep_count['Listening_Time_minutes'].rolling(window=3).std().fillna(0)

sns.lineplot(
    data=df_ep_count,
    x='Episode_Number', y='Listening_Time_minutes_rolling_std',
)
for each in range(0, 102, 2):
    plt.axvline(each, color='r', linewidth=.3, linestyle='--')
plt.title("Rolling STD")
plt.show()


df_ep_count['Listening_Time_minutes_diff'] = df_ep_count['Listening_Time_minutes'].shift(1).fillna(0) - df_ep_count['Listening_Time_minutes']

sns.lineplot(
    data=df_ep_count,
    x='Episode_Number', y='Listening_Time_minutes_diff',
)
for each in range(0, 102, 2):
    plt.axvline(each, color='r', linewidth=.3, linestyle='--')
plt.show()


sns.boxplot(
    data=df,
    x="Episode_Length_minutes",
)
plt.show()


sns.boxplot(
    data=df_ep_count,
    x="Listening_Time_minutes",
)
plt.show()


sns.pairplot(num_df.drop(['id'], axis=1), corner=True)
plt.show()


sns.scatterplot(data=df, x="Episode_Length_minutes", y="Listening_Time_minutes")
plt.show()


def inc_eps(x):
    if x > 125:
        return -1
    return x%25

num_df['Epiode_Length_below_125'] = num_df['Episode_Length_minutes'].apply(inc_eps)


sns.scatterplot(
    data=df, 
    x="Host_Popularity_percentage", 
    y="Listening_Time_minutes",
    hue="Episode_Sentiment",
)
plt.show()


ax = sns.barplot(
    data=df,
    x="Publication_Day",
    y="Listening_Time_minutes",
    hue="Publication_Time",
    order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    hue_order=['Morning', 'Afternoon', 'Evening', 'Night'],
    errorbar=None,
)
ax.set_ylim(bottom=42)
plt.xticks(rotation=45)
plt.show()


ax = sns.barplot(
    data=df,
    x="Genre",
    y="Listening_Time_minutes",
    errorbar=None,
)
ax.set_ylim(bottom=42)
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df,
    x="Podcast_Name",
    y="Listening_Time_minutes",
    errorbar=None,
)
ax.set_ylim(bottom=42)
plt.xticks(rotation=90)
plt.show()


ax = sns.barplot(
    data=df,
    x="Episode_Sentiment",
    y="Listening_Time_minutes",
    errorbar=None,
)
ax.set_ylim(bottom=42)
# plt.xticks(rotation=90)
plt.show()


sns.heatmap(num_df.corr().round(2), annot=True)
plt.show()


cat_df_dropped = cat_df.drop(['Episode_Title'], axis=1)


inits = lambda x: x.strip()[:4].upper()
cat_df_dropped['Genre_Sentiment'] = cat_df_dropped['Genre'].apply(inits) + cat_df_dropped['Episode_Sentiment'].apply(inits)
cat_df_dropped['Day_Time'] = cat_df_dropped['Publication_Day'].apply(inits) + cat_df_dropped['Publication_Time'].apply(inits)

cat_df_dropped['Genre_Time'] = cat_df_dropped['Genre'].apply(inits) + cat_df_dropped['Publication_Time'].apply(inits)
cat_df_dropped['Senitment_Time'] = cat_df_dropped['Episode_Sentiment'].apply(inits) + cat_df_dropped['Publication_Time'].apply(inits)

cat_df_dropped['Genre_Day'] = cat_df_dropped['Genre'].apply(inits) + cat_df_dropped['Publication_Day'].apply(inits)
cat_df_dropped['Senitment_Day'] = cat_df_dropped['Episode_Sentiment'].apply(inits) + cat_df_dropped['Publication_Day'].apply(inits)

cat_df_dropped['Name_Time'] = cat_df_dropped['Podcast_Name'].apply(inits) + cat_df_dropped['Publication_Time'].apply(inits)
cat_df_dropped['Name_Day'] = cat_df_dropped['Podcast_Name'].apply(inits) + cat_df_dropped['Publication_Day'].apply(inits)
cat_df_dropped.head()


le = LabelEncoder()

cat_df_encoded = pd.DataFrame()

for col in cat_df_dropped.columns:
    cat_df_encoded[col] = le.fit_transform(cat_df_dropped[col])

cat_df_encoded.head()


random_state = 42

df_final = pd.concat([num_df, cat_df_encoded], axis=1)

X = df_final.drop(['id', 'Listening_Time_minutes'], axis=1)
y = df_final['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=.2, random_state=random_state,
)


xt = XGBRegressor(
    learning_rate=.1, 
    n_estimators=200, 
    max_depth=13,
    eval_metric="rmse",
    random_state=random_state,
)

xt.fit(X_train, y_train)

y_preds = xt.predict(X_test)

MSE = mean_squared_error(y_test, y_preds)

print(
    f'MSE  :\t {MSE}',
    f'RMSE :\t {MSE**(1/2)}',
    f'R2   :\t {r2_score(y_test, y_preds)}',
    f'MAE  :\t {mean_absolute_error(y_test, y_preds)}',
    sep="\n",
)

features = zip(X.columns, xt.feature_importances_)
features = sorted(dict(features).items(), key=lambda x: abs(x[1]), reverse=True)

print(f"{'Feature':<30} {'Importance':>12} {'Is important?':>16}", end="\n\n")
for col_name, importance in features:
    print(f"{col_name:<30} {importance*100:>10.4f} {'N' if not importance else 'Y':>12}")


pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv').head()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

num_df = test_df.select_dtypes(include="number")

cat_df = test_df.select_dtypes(exclude="number")

si_num = SimpleImputer(strategy="median")
num_df = pd.DataFrame(si_num.fit_transform(num_df), columns=num_df.columns)

si_cat = SimpleImputer(strategy="constant")
cat_df = pd.DataFrame(si_cat.fit_transform(cat_df), columns=cat_df.columns)

cat_df_dropped = cat_df.drop(['Episode_Title'], axis=1)
cat_df_dropped['Genre_Sentiment'] = cat_df_dropped['Genre'].apply(inits) + cat_df_dropped['Episode_Sentiment'].apply(inits)
cat_df_dropped['Day_Time'] = cat_df_dropped['Publication_Day'].apply(inits) + cat_df_dropped['Publication_Time'].apply(inits)

cat_df_dropped['Genre_Time'] = cat_df_dropped['Genre'].apply(inits) + cat_df_dropped['Publication_Time'].apply(inits)
cat_df_dropped['Senitment_Time'] = cat_df_dropped['Episode_Sentiment'].apply(inits) + cat_df_dropped['Publication_Time'].apply(inits)

cat_df_dropped['Genre_Day'] = cat_df_dropped['Genre'].apply(inits) + cat_df_dropped['Publication_Day'].apply(inits)
cat_df_dropped['Senitment_Day'] = cat_df_dropped['Episode_Sentiment'].apply(inits) + cat_df_dropped['Publication_Day'].apply(inits)

cat_df_dropped['Name_Time'] = cat_df_dropped['Podcast_Name'].apply(inits) + cat_df_dropped['Publication_Time'].apply(inits)
cat_df_dropped['Name_Day'] = cat_df_dropped['Podcast_Name'].apply(inits) + cat_df_dropped['Publication_Day'].apply(inits)

num_df['Episode_Number'] = cat_df['Episode_Title'].str.lstrip('Episode ').apply(int)
num_df['Epiode_Length_below_125'] = num_df['Episode_Length_minutes'].apply(inc_eps)
cat_df_encoded = pd.DataFrame()

for col in cat_df_dropped.columns:
    cat_df_encoded[col] = le.fit_transform(cat_df_dropped[col])

df_final = pd.concat([num_df, cat_df_encoded], axis=1).drop(['id'], axis=1)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': xt.predict(df_final)
})

submission.to_csv('submission.csv', index=False)

pd.read_csv('submission.csv').head()

