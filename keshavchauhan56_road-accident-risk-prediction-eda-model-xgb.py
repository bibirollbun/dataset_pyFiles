import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Important Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df_sample=pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
df_train= pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test= pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


df_train.head()


df_test.head()


#Checking the dimension of the train & test dataset.
print(f"Df Train Dimension: {df_train.shape}")
print(f"Df Test Dimension: {df_test.shape}")


#Checking the datasetinfo.
df_train.info()


#Checking columns names.
df_train.columns


#Dropping the 'Id' columns as it didn't have relative information.
df_train= df_train.drop('id', axis=1)
df_test= df_test.drop('id', axis=1)


#Statistical Summary of the dataset.
df_train.describe()


#Checking the null values in the Train & Test dataset.
print(f"Null value in the Train dataset: {df_train.isnull().sum().sum()}")
print(f"Null value in the Test dataset: {df_test.isnull().sum().sum()}")


#Checking Duplicate values in traing dataset.
print(f"Number of duplicate values in training datset: {df_train.duplicated().sum()}")


#Droping the duplicate rows.
df_train= df_train.drop_duplicates()
df_train.duplicated().sum()


df_train.head()


# Checking the Accident Risk Distribution.
plt.figure(figsize=(8,6))
sns.histplot(df_train['accident_risk'], bins=20, kde=True, color='orange')
plt.title("Accident Risk Distribution", size=20)
plt.xlabel("Accisdent Risk")
plt.show()


#Creating the function for checking multiplut features vs target variable.
def target_plot(x,y,data):
    plt.figure(figsize=(6,4))
    sns.barplot(x=x, y=y, data=data, estimator="mean", palette="viridis")
    plt.title("Feature VS Target", size=18, fontweight="bold", fontname="Times")
    plt.xlabel("Fetures Variable", size=12)
    plt.ylabel("Target Variable", size=12)
    plt.xticks(rotation=45)
    plt.show()


# Road Type vs Accident Risk.
target_plot("road_type","accident_risk",df_train)


# Lighting vs Accident Risk
target_plot("lighting", "accident_risk", df_train)


# Wather vs Accident Risk
target_plot("weather", "accident_risk", df_train)


# Public Road vs Accident Risk
target_plot("public_road","accident_risk",df_train)


# Road Signs Present vs Accident risk.
target_plot("road_signs_present","accident_risk",df_train)


# Time of day vs Accident risk.
target_plot("time_of_day","accident_risk",df_train)


# Holiday vs Accident risk.
target_plot("holiday","accident_risk",df_train)


# School Season vs Accident risk.
target_plot("school_season","accident_risk",df_train)


# Checking the outliers in the dataset. By creating function for check in multiple columns.
def boxplot(data):
    fig=plt.figure(figsize=(8,6)) 
    sns.boxplot(x=data)
    plt.title("Checking Outlier in datatset")
    plt.show()


boxplot(df_train["speed_limit"])


boxplot(df_train["accident_risk"])


#There are outlier's in the accident_risk. We have to fix that because it may affect our model predict.
#Let's fix the outlier in the accident_risk columns.
target=df_train["accident_risk"]
Q1=target.quantile(0.25)
Q3=target.quantile(0.75)
IQR= Q3-Q1

lower_bound= Q1-1.5*IQR
upper_bound= Q3+1.5*IQR

outliers=target[(target<lower_bound)|(target>upper_bound)]

print("Number of outliers:", len(outliers))
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


# Removing the outlier from the accident_risk columna and makeing new clean dataset.
df_clean = df_train[(target >= lower_bound) & (target <= upper_bound)]


#Now you can see there is no outlier in the accident_risk columns.
boxplot(df_clean["accident_risk"])


# Converting the categrical column using Label Encoding for df_clean.
from sklearn.preprocessing import LabelEncoder

label_encoder={}
cat_col=["road_type","lighting","time_of_day","weather"]

for col in cat_col:
    df_clean[col]= df_clean[col].astype(str).str.strip().fillna("Unknown")
    label_encoder[col]= LabelEncoder()
    df_clean[col]= label_encoder[col].fit_transform(df_clean[col])
    df_clean[col]= df_clean[col].astype(int)


# Same thing we will do for df_test datset.
label_encoder={}
cat_col=["road_type","lighting","time_of_day","weather"]

for col in cat_col:
    df_test[col]= df_test[col].astype(str).str.strip().fillna("Unknown")
    label_encoder[col]= LabelEncoder()
    df_test[col]= label_encoder[col].fit_transform(df_test[col])
    df_test[col]= df_test[col].astype(int)


# Converting boolean value into int using apply function.
bol_col=["road_signs_present","public_road","holiday","school_season"]

for col in bol_col:
    df_clean[col] = df_clean[col].apply(lambda x: 1 if x == True else 0)
    df_test[col] = df_test[col].apply(lambda x: 1 if x == True else 0)


corrdf= df_clean.corr()

plt.figure(figsize=(12,6))
sns.heatmap(corrdf, annot=True, cmap="rainbow")
plt.title("Correlation Heatmap", fontweight='bold', fontsize=20)
plt.show()


#Checking coor relation of independent variable to target variable.
target_corr = corrdf["accident_risk"].abs().sort_values(ascending=False)
selected_features = target_corr[target_corr > 0.1].index
print(target_corr)
print("")
print(selected_features)


# droping the less correlacted feature from the train dataset.
df_clean.drop(["holiday","public_road","road_type","num_lanes","time_of_day","road_signs_present","school_season"],axis=1,inplace=True)
df_clean.head()


# droping the less correlacted feature from the prediction dataset.
df_test.drop(["holiday","public_road","road_type","num_lanes","time_of_day","road_signs_present","school_season"],axis=1,inplace=True)
df_test.head()


X= df_clean.drop(["accident_risk"],axis=1)
y= df_clean["accident_risk"]


# Spliting the data into Train, Test.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, train_size=0.8, random_state=2410)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# First model we are using Linear Regression.
from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(X_train,y_train)

y_predict= model.predict(X_test)


# Evaluatiom prediction using RMSE.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse= np.sqrt(mean_squared_error(y_test, y_predict))
mae= mean_absolute_error(y_test, y_predict)
r2= r2_score(y_test, y_predict)

print("RMSE:", rmse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


#Error calculating.
error= y_test-y_predict

sns.histplot(data=error, kde=True)
plt.title("Checking Error Distribution")
plt.xlabel("Error")
plt.show()


sns.scatterplot(x=y_predict,y=error)
plt.title("Checking the Prediction and Error")
plt.xlabel("Prediction")
plt.ylabel("Error")
plt.show()


# Second model we are using RandomForrestRegressor
from sklearn.ensemble import RandomForestRegressor

rf_model= RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_predict2= rf_model.predict(X_test)


rmse= np.sqrt(mean_squared_error(y_test, y_predict2))
mae= mean_absolute_error(y_test, y_predict2)
r2= r2_score(y_test, y_predict2)

print("RMSE:", rmse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


# Thrid Model we are using XGBRegressor
from xgboost import XGBRegressor 

xgb_model= XGBRegressor()

xgb_model.fit(X_train, y_train)

y_predict3= xgb_model.predict(X_test)


rmse= np.sqrt(mean_squared_error(y_test, y_predict3))
mae= mean_absolute_error(y_test, y_predict3)
r2= r2_score(y_test, y_predict3)

print("RMSE:", rmse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


# From the above the 3 model. We have seen that XGBRegressor model is working excellent as compaer to other two. As you see above RMSE is lower
# side, Mean absolute error is also on lower side and R2 Score is excellent as more then 0.8.

# We decide to go with XGBRegressor now we hypertunne the model.


from sklearn.model_selection import RandomizedSearchCV

n_estimators= [500, 800, 1000]
learning_rate=[0.01, 0.02, 0.03]
max_depth=[5, 7, 9]
subsample=[0.7, 0.8, 0.9, 1.0]
colsample_bytree=[0.7, 0.8, 0.9, 1.0]

param_dist={"n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            }

xgb_random= RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=60, cv=5, scoring="neg_root_mean_squared_error",n_jobs=1,verbose=2)


xgb_random.fit(X_train, y_train)


# Best model
best_xgb = xgb_random.best_estimator_
print(best_xgb)


# Predict on test set
final_pred= best_xgb.predict(X_test)


rmse= np.sqrt(mean_squared_error(y_test, final_pred))
mae= mean_absolute_error(y_test, final_pred)
r2= r2_score(y_test, final_pred)

print("RMSE:", rmse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


from sklearn.model_selection import cross_val_score, KFold


kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Compute cross-validation RMSE directly
scores = cross_val_score(
    best_xgb, X_train, y_train, 
    scoring='neg_root_mean_squared_error', 
    cv=kf, 
    n_jobs=-1)


print("Average RMSE:", -scores.mean())
print("All Fold RMSEs:", -scores)


# Now building the final model.
final_model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=800,       
    learning_rate=0.02,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42 )

# Fit on the entire training dataset
final_model.fit(X_train, y_train)


# Predict on test dataset
y_pred_test = final_model.predict(X_test)

# Seeing first few predictions
print("Test Predictions Sample:", y_pred_test[:10])


#Doing final prediction on test dataset.
test_predictions = final_model.predict(df_test)
df_test['accident_risk_predicted'] = test_predictions


#Here is the prediction on the test dataset.
df_test.head()


#Now as required submitting on the sample data.
df_sample['accident_risk'] = test_predictions


df_sample.head()


df_sample.to_csv('submission.csv', index=False)

