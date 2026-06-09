import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
df


df.shape


df.info()


df.describe()


df.isnull().sum()


df.head()


if "id" in df.columns:
    df = df.drop(columns=["id"])
df


df = df.fillna(df.mean())
df


sns.boxplot(x = df.RhythmScore)
plt.grid(True)
plt.tight_layout()
plt.show()


# sns.boxplot(x = df.VocalContent)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


sns.boxplot(x = df.AcousticQuality)
plt.grid(True)
plt.tight_layout()
plt.show()


sns.distplot(x = df.AudioLoudness, kde=True)


sns.distplot(x = df.AcousticQuality, kde=True, color='lightgreen')


sns.boxplot(x = df.InstrumentalScore)


sns.boxplot(x = df.MoodScore)


for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {outliers} outliers")


# Q1 = df['AcousticQuality'].quantile(0.25)
# Q3 = df['AcousticQuality'].quantile(0.75)
# IQR = Q3 - Q1
# IQR
# lower_limit = Q1 - 1.5*IQR
# upper_limit = Q3 + 1.5*IQR
# print('Lower Limit : ',lower_limit)
# print('Upper Limit : ',upper_limit)
# df.loc[df['AcousticQuality'] > upper_limit]
# df.loc[df['AcousticQuality'] < lower_limit]
# df.loc[df['AcousticQuality'] > upper_limit, 'AcousticQuality'] = df['AcousticQuality'].median()


cols=df.columns
for column in cols:
  q1 = df[column].quantile(0.25)
  q3 = df[column].quantile(0.75)
  iqr = q3-q1
  #Bounds:
  lower = q1 - 1.5*iqr
  upper = q3 + 1.5*iqr
  # outliers = df[(df[column]>upper) | (df[column]<lower)]
  # #print(f"Total outliers w.r.t {column} = {outliers.sum}")
  df = df[(df[column]<upper) & (df[column]>lower)]


for col in df.columns:
    plt.figure()
    plt.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


df.hist(figsize=(15,10), bins=30)
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(df["BeatsPerMinute"], bins=40, kde=True, color="blue")
plt.title("Distribution of BPM")
plt.xlabel("Beats Per Minute")
plt.show()



# plt.figure(figsize=(8,5))
# sns.scatterplot(x=df["Energy"], y=df["BeatsPerMinute"], alpha=0.3)
# plt.title("Energy vs BPM")
# plt.xlabel("Energy")
# plt.ylabel("BPM")
# plt.show()


sns.pairplot(df[["BeatsPerMinute","Energy","RhythmScore","MoodScore","TrackDurationMs"]], diag_kind="kde")
plt.show()


X = df.drop(columns=['BeatsPerMinute'])  # features only
y = df['BeatsPerMinute']                  # target

scaler = StandardScaler()
X_scaled= scaler.fit_transform(X)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
test_df


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


X_train, X_test, y_train, y_test = train_test_split( X_scaled,y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

# # --- 1. Train/Validation Evaluation ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_val = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred_val))
print("MSE:", mean_squared_error(y_test, y_pred_val))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_val)))
print("MAE:", mean_absolute_error(y_test, y_pred_val))

# --- 2. Retrain on FULL data for Kaggle submission ---
model.fit(X, y)   # train on full dataset
test_pred = model.predict(test_df.drop("id", axis=1))

# --- 3. Save submission ---
submission = pd.DataFrame({
    "id": test_df["id"],
    "BeatsPerMinute": test_pred
})

submission.to_csv("Robocon4.csv", index=False)
print("✅ Submission file saved as Robocon4.csv")



# from sklearn.tree import DecisionTreeRegressor
# dt = DecisionTreeRegressor(random_state=42)
# dt.fit(X_train, y_train)

# #Predictions
# y_pred_dt = dt.predict(X_test)

# print("R2:", r2_score(y_test, y_pred_dt))
# print("MSE:", mean_squared_error(y_test, y_pred_dt))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dt)))
# print("MAE:", mean_absolute_error(y_test, y_pred_dt))


# X_test_final = test_df.drop(columns=['id'])  # Remove only ID column

#  # Apply same scaling if used during training
# X_test_final_scaled = scaler.transform(X_test_final)
# y_pred = model.predict(X_test_final_scaled)
# y_pred


# rf = RandomForestRegressor(
#     n_estimators=300,        # fewer trees for quick baseline (100–300 later for accuracy)
#     max_depth=15,           # limit tree depth (default = unlimited)
#     min_samples_split=5,    # stop splitting earlier
#     n_jobs=-1,              # use all CPU cores
#     random_state=42
# )
# rf.fit(X_train, y_train)

# y_pred_rf = rf.predict(X_test)


# print("MAE:", mean_absolute_error(y_test, y_pred_rf))
# print("R²:", r2_score(y_test, y_pred_rf))



# !pip install xgboost --quiet


# from xgboost import XGBRegressor

# xgb = XGBRegressor(
#     n_estimators=500,    # more trees
#     learning_rate=0.1,   # slightly higher LR
#     max_depth=5,         # shallower trees to reduce overfitting
#     subsample=0.8,       # row sampling
#     colsample_bytree=0.8,
#     random_state=42
# )
# xgb.fit(X_train, y_train)


# y_pred = xgb.predict(X_test)

# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("R²:", r2_score(y_test, y_pred))



# from sklearn.ensemble import AdaBoostRegressor
# # Train
# ada = AdaBoostRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     random_state=42)
# ada.fit(X_train, y_train)

# # Predict
# y_pred = ada.predict(X_test)

# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("R²:", r2_score(y_test, y_pred))




# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)



# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# # Train Random Forest Regressor
# rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
# rf_reg.fit(X_train, y_train)
# y_pred = rf_reg.predict(X_test)

# # Evaluation
# rf_reg_rmse = mean_squared_error(y_test, y_pred)
# rf_reg_r2 = r2_score(y_test, y_pred)

# print("Random Forest Regression RMSE:", rf_reg_rmse)
# print("Random Forest Regression R²:", rf_reg_r2)


# from sklearn.ensemble import AdaBoostRegressor

# # Train AdaBoost Regressor
# ada_reg = AdaBoostRegressor(n_estimators=100, random_state=42)
# ada_reg.fit(X_train, y_train)
# y_pred_val = ada_reg.predict(X_test)

# print("AdaBoost Validation R²:", r2_score(y_test, y_pred_val))
# print("AdaBoost Validation RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_val)))
# print("AdaBoost Validation MAE:", mean_absolute_error(y_test, y_pred_val))

# # --- 2. Retrain on FULL data for Kaggle submission ---
# ada_reg.fit(X_scaled, y)   # use full dataset now
# test_pred = ada_reg.predict(test_df.drop("id", axis=1))

# # --- 3. Save submission ---
# submission = pd.DataFrame({
#     "id": test_df["id"],
#     "BeatsPerMinute": test_pred
# })

# submission.to_csv("Robocon2.csv", index=False)
# print("✅ Submission file saved as Robocon2.csv")


# from sklearn.ensemble import GradientBoostingRegressor
# # Train on full training data

# gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_reg.fit(X_train, y_train)
# y_pred = gb_reg.predict(X_test)

# gb_reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# gb_reg_r2 = r2_score(y_test, y_pred)

# print("Gradient Boosting Regression RMSE:", gb_reg_rmse)
# print("Gradient Boosting Regression R²:", gb_reg_r2)


# gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_reg.fit(X_scaled, y)   # now trained with feature names

# # Predict on test set
# test_pred = gb_reg.predict(test_df.drop("id", axis=1))

# submission = pd.DataFrame({
#     "id": test_df["id"],
#     "BeatsPerMinute": test_pred
# })

# submission.to_csv("Robocon.csv", index=False)
# print("✅ Submission file saved as Robocon.csv")


