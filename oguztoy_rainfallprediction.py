

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df1 = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df1.head()


df2 = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
df2.head()


df1.shape, df2.shape


df = pd.concat([df1, df2])
df.sample(5)


df.info()


df.isnull().sum()


imputer = KNNImputer(n_neighbors=7)
df[['winddirection']] = KNNImputer(n_neighbors=5).fit_transform(df[['winddirection'] + df.select_dtypes(include='number').drop(columns='winddirection').columns.tolist()])[:, [0]]


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="Blues");


def wind_category(angle):
    if pd.isna(angle): return np.nan
    if angle < 90:
        return 1
    elif angle < 180:
        return 2
    elif angle < 270:
        return 3
    else:
        return 4

df['wind_dir_cat'] = df['winddirection'].apply(wind_category)


def preprocess_weather_data(df):
    

    df["temp_range"] = df["maxtemp"] - df["mintemp"]
    df["avg_temp"] = (df["maxtemp"] + df["mintemp"]) / 2
    df["temp_dew_diff"] = df["temparature"] - df["dewpoint"]

    df["dew_humidity"] = df["dewpoint"] * df["humidity"]
    df["dew_humidity_ratio"] = df["dewpoint"] / (df["humidity"] + 1)
    df["dew_humidity_per_sun"] = (df["dewpoint"] * df["humidity"]) / (df["sunshine"] + 1)

    df["cloud_to_humidity"] = df["cloud"] / (df["humidity"] + 1)
    df["cloud_windspeed"] = df["cloud"] * df["windspeed"]
    df["cloud_sun_ratio"] = df["cloud"] / (df["sunshine"] + 1)
    df["cloud_humidity_pressure"] = (df["cloud"] * df["humidity"]) / (df["pressure"] + 1)

    df["temp_to_sunshine"] = df["sunshine"] / (df["temparature"] + 1)
    df["wind_temp_interaction"] = df["windspeed"] * df["temparature"]

    df["humidity_sunshine"] = df["humidity"] * df["sunshine"]
    df["cloud_sunshine_interaction"] = df["cloud"] * df["sunshine"]
    
    
    df['month'] = ((df['day'] - 1) // 30 + 1).clip(upper=12)
    df['season'] = df['month'].apply(lambda x: 1 if 3 <= x <= 5
                                     else 2 if 6 <= x <= 8
                                     else 3 if 9 <= x <= 11
                                     else 0)

    df['season_cloud_trend'] = df['cloud'] * df['season']
    df['season_cloud_deviation'] = df['cloud'] - df.groupby('season')['cloud'].transform('mean')
    df['season_temperature'] = df['temparature'] * df['season']


    df.drop(columns=[
        "maxtemp", "mintemp", "temparature", "dewpoint",
        "humidity", "pressure", "winddirection",
        "day", "month", "season"
    ], inplace=True)

    return df
df = preprocess_weather_data(df)


df.head()


abs(df.corr(numeric_only=True)["rainfall"].sort_values(ascending=False))


low_corr_cols = abs(df.corr(numeric_only=True)['rainfall']).loc[lambda x: x < 0.30].index

df.drop(columns=low_corr_cols, inplace=True)


train=df[:2190]
test=df[2190:]


x = train.drop("rainfall", axis=1)
y = train[["rainfall"]]
test = test.drop("rainfall", axis=1)


scaler = MinMaxScaler()
x = scaler.fit_transform(x)
test = scaler.transform(test)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=8)


models = {
    "Logistic Regression": LogisticRegression( max_iter=1000),
    "Random Forest": RandomForestClassifier( n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Neural Network": MLPClassifier( max_iter=100, hidden_layer_sizes=(10)),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric='mlogloss'),
    "CatBoost": CatBoostClassifier( iterations=100, learning_rate=0.14, depth=6, verbose=0)
}


for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"ðŸ”¹ {name} Accuracy: {acc:.4f}")


cb = CatBoostClassifier( iterations=100, learning_rate=0.14, depth=6, verbose=0)
model = cb.fit(x,y)
prediction = model.predict(test)


prediction


submission = pd.DataFrame({"id": df2["id"], "rainfall":prediction})


submission


submission.to_csv("submission.csv", index=False)




