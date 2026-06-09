import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error

)
pd.set_option("display.max_columns",None)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostRegressor

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
%matplotlib inline


df=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")


df.head()


df.shape


df.isnull().sum()


df.drop(columns=["id"],axis=1,inplace=True)


df.info()


df.describe()


df.head()


features = df.columns.drop('BeatsPerMinute')
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.scatterplot(x=feature, y='BeatsPerMinute', data=df)
    plt.title(f'BPM vs {feature}')
plt.tight_layout()
plt.show()


df['Energy_Binned'] = pd.qcut(df['Energy'], q=3, labels=['Low', 'Medium', 'High'])
plt.figure(figsize=(15, 6))
sns.boxplot(x='Energy_Binned', y='BeatsPerMinute', data=df)
plt.title('BPM Distribution by Energy Levels')
plt.show()


plt.figure(figsize=(15, 6))
sns.histplot(df['BeatsPerMinute'], kde=True)
plt.title('Distribution of Beats Per Minute')
plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['RhythmScore'], df['Energy'], df['BeatsPerMinute'])
ax.set_xlabel('RhythmScore')
ax.set_ylabel('Energy')
ax.set_zlabel('BeatsPerMinute')
plt.title('3D Scatter: BPM vs RhythmScore & Energy')
plt.show()


df['MoodScore_Binned'] = pd.qcut(df['MoodScore'], q=3, labels=['Low', 'Medium', 'High'])
plt.figure(figsize=(15, 6))
sns.violinplot(x='MoodScore_Binned', y='BeatsPerMinute', data=df)
plt.title('BPM Distribution by MoodScore')
plt.show()


df.drop(columns=["Energy_Binned","MoodScore_Binned"],axis=1,inplace=True)


from sklearn.model_selection import train_test_split

# Features and target
X = df.drop("BeatsPerMinute", axis=1)   # replace with your real target column
y = df["BeatsPerMinute"]


X['Log_TrackDurationMs'] = np.log1p(X['TrackDurationMs'])
X['Log_AudioLoudness'] = np.log1p(-X['AudioLoudness'])  


X.drop(columns=["TrackDurationMs","AudioLoudness"],axis=1,inplace=True)


scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
poly_columns = poly.get_feature_names_out(X.columns)
X_poly = pd.DataFrame(X_poly, columns=poly_columns)
y_capped = np.clip(y, y.quantile(0.01), y.quantile(0.99))


X_poly.shape


X_train, X_test, y_train, y_test = train_test_split(X_poly, y_capped, test_size=0.2, random_state=42)



cat_model = CatBoostRegressor(iterations=1000,learning_rate=0.005,depth=6,loss_function='RMSE',random_state=42,verbose=200)

cat_model.fit(X_train, y_train)


# Predictions
y_pred = cat_model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # same as np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
maxerr = max_error(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Explained Variance: {evs:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"Median AE: {medae:.4f}")
print(f"Max Error: {maxerr:.4f}")



plt.figure(figsize=(15, 6))
sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"}, scatter_kws={"alpha": 0.6})

# Reference line (perfect predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],color='blue', linestyle='--', linewidth=2, label="Perfect Prediction")

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.legend()
plt.show()


test_df=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


test_df.head()


Id=test_df.id


test_df.drop(columns=["id"],axis=1,inplace=True)


test_df.isnull().sum()


test_df['Log_TrackDurationMs'] = np.log1p(test_df['TrackDurationMs'])
test_df['Log_AudioLoudness'] = np.log1p(-test_df['AudioLoudness'])
test_df.drop(columns=["TrackDurationMs", "AudioLoudness"], axis=1, inplace=True)
test_df = test_df[X_scaled.columns]
test_scaled = scaler.transform(test_df)
test_df_scaled = pd.DataFrame(test_scaled, columns=X_scaled.columns)
test_poly = poly.transform(test_df_scaled)
poly_columns = poly.get_feature_names_out(X_scaled.columns)
test_df = pd.DataFrame(test_poly, columns=poly_columns)


test_df.shape


y_pred=cat_model.predict(test_df)
df=pd.DataFrame({"id":Id,"BeatsPerMinute":y_pred})
df.to_csv("submission.csv",index=False)


df.head()




