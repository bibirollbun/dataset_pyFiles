#imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')


df.info()


df.head()


df.describe()


df.corr()


num_cols = df.drop(['id','BeatsPerMinute'],axis=1).columns
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[col], y=df['BeatsPerMinute'])
    plt.title(f"{col} vs BPM")
    plt.show()


X_train = df.drop(['id','BeatsPerMinute'],axis=1)
y_train = df['BeatsPerMinute']


X_train.info()


y_train.info()


X_test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


X_test = X_test.drop(['id'],axis=1)


X_test.info()


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_1 = lr.fit(X_train,y_train)
y_predict_1 = model_1.predict(X_test)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': y_predict_1})
submission_df.to_csv('submission.csv', index=False)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


lr_poly = LinearRegression()
model_2 = lr_poly.fit(X_train_poly, y_train)
y_predict_2 = model_2.predict(X_test_poly)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': y_predict_2})
submission_df.to_csv('submission_poly.csv', index=False)


X_train.info()


X_train.isnull().sum()


y_train.info()


X_train.head()


from sklearn.preprocessing import MinMaxScaler

# Normalize only the 'TrackDurationMs' column between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))

# Reshape the column to be 2D for scaling
X_train_audio = X_train['AudioLoudness'].values.reshape(-1, 1)
X_test_audio = X_test['AudioLoudness'].values.reshape(-1, 1)
X_train_track_duration = X_train['TrackDurationMs'].values.reshape(-1, 1)
X_test_track_duration = X_test['TrackDurationMs'].values.reshape(-1, 1)

# Apply scaling
X_train['TrackDurationMs'] = scaler.fit_transform(X_train_track_duration)
X_test['TrackDurationMs'] = scaler.transform(X_test_track_duration)

X_train['AudioLoudness'] = scaler.fit_transform(X_train_audio)
X_test['AudioLoudness'] =  scaler.fit_transform(X_test_audio)


X_train.head()


X_test.head()


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


lr_poly = LinearRegression()
model_3 = lr_poly.fit(X_train_poly, y_train)
y_predict_3 = model_2.predict(X_test_poly)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': y_predict_3})
submission_df.to_csv('submission_poly_1.csv', index=False)

