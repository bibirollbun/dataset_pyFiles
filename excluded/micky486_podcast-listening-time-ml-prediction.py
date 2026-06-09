import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


train=pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv")
test=pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")
submission=pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")


train.head()


train.shape


test.head()


test.shape


submission.head()


submission.shape


# Filling NaN values in both train and test dataset 
# Use mode from train for both train and test
train["Episode_Length_minutes"] = train["Episode_Length_minutes"].fillna(train["Episode_Length_minutes"].mode()[0])
test["Episode_Length_minutes"] = test["Episode_Length_minutes"].fillna(train["Episode_Length_minutes"].mode()[0])

train["Guest_Popularity_percentage"] = train["Guest_Popularity_percentage"].fillna(train["Guest_Popularity_percentage"].mode()[0])
test["Guest_Popularity_percentage"] = test["Guest_Popularity_percentage"].fillna(train["Guest_Popularity_percentage"].mode()[0])




train.duplicated().sum()
test.duplicated().sum()


# Drop only 'Episode_Title' — keep 'id' in both
train.drop(columns=["Episode_Title"], inplace=True)
test.drop(columns=["Episode_Title"], inplace=True)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
sns.boxplot(x=train["Listening_Time_minutes"], color='red')
plt.title("Distribution of Listening Time (minutes)")
plt.xlabel("Listening Time (minutes)")
plt.tight_layout()
plt.show()



train.head()


train.columns


train.select_dtypes(include="number").columns


import pandas as pd

# ✅ Step 1: Specify categorical columns
cols_to_encode = ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]

# ✅ Step 2: Select numeric columns explicitly
num_cols = train[['id', 'Episode_Length_minutes', 'Host_Popularity_percentage',
                  'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']]

# ✅ Step 3: Apply pd.get_dummies to encode categorical columns
encoded_cat = pd.get_dummies(train[cols_to_encode], drop_first=True).astype("int")

# ✅ Step 4: Concatenate encoded categorical columns with numeric columns
train_fdf = pd.concat([encoded_cat, num_cols], axis=1)

# ✅ Step 5: Check final shape or preview
print("Final shape:", train_fdf.shape)
train_fdf.head()
train_fdf.isna().sum()
train_fdf.dropna(inplace=True)
train_fdf.head()


train_fdf.columns


model_rf = RandomForestRegressor()
model_gb = GradientBoostingRegressor()
from xgboost import XGBRegressor
model_x=XGBRegressor()


x=train_fdf.drop(columns=["Listening_Time_minutes"])
y=train_fdf["Listening_Time_minutes"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model_x.fit(x_train,y_train)


y_pred=model_x.predict(x_test)
# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Output results
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 4))



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Listening Time (minutes)")
plt.ylabel("Predicted Listening Time (minutes)")
plt.title("Actual vs. Predicted Listening Time")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal line
plt.show()



residuals = y_test - y_pred

plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=30, kde=True, color="purple")
plt.xlabel("Residuals")
plt.title("Distribution of Residuals")
plt.show()



test.info()


test_encoded = pd.get_dummies(test[cols_to_encode], drop_first=True)

# Align with training columns
test_encoded = test_encoded.reindex(columns=train_fdf.drop(columns=["Listening_Time_minutes"]).columns, fill_value=0)

# Convert to int (optional safety)
test_encoded = test_encoded.astype("int")

# ✅ Step 5: Check final shape or preview
print("Final shape:", train_fdf.shape)
train_fdf.head()
train_fdf.isna().sum()
train_fdf.dropna(inplace=True)
train_fdf.head()


test_preds=model_x.predict(test_encoded)


submission.head()


submission["Listening_Time_minutes"]=test_preds
submission=pd.DataFrame({
    "id":test["id"],
    "Listening_Time_minutes":test_preds
})

submission
submission.to_csv("my_submission.csv", index=False)


