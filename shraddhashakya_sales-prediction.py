#Import Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error

#Load Training Data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

#Convert 'date' to datetime format
train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

#Handle Missing Values in 'num_sold'
train_df["num_sold"] = train_df.groupby(["country", "store", "product"])["num_sold"].transform(lambda x: x.fillna(x.median()))

#Remove zero or negative sales to avoid log errors
train_df = train_df[train_df["num_sold"] > 0].copy()

#Apply Log Transformation to stabilize 'num_sold'
train_df["num_sold"] = np.log1p(train_df["num_sold"])

#Extract Time Features
for df in [train_df, test_df]:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

#Encode Categorical Features
label_encoders = {}
for col in ["country", "store", "product"]:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le  # Store encoder

#Save Label Encoders for Reuse
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

#Define Features & Target Variable
features = ["country", "store", "product", "year", "month", "day", "weekday", "weekofyear"]
X = train_df[features]
y = train_df["num_sold"]

#Split Data into Train & Validation Sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#Handle Missing & Infinite Values
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_valid = X_valid.replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = y_train.fillna(0)
y_valid = y_valid.fillna(0)

#Train XGBoost Model (Optimized for CPU)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", 
                             n_estimators=50, 
                             learning_rate=0.1, 
                             max_depth=6, 
                             tree_method="hist",  
                             device="cpu",  
                             random_state=42)

xgb_model.fit(X_train, y_train)

#Make Predictions on Validation Set
y_pred = xgb_model.predict(X_valid)

#Convert Predictions Back from Log Scale
y_valid_exp = np.expm1(y_valid)  
y_pred_exp = np.expm1(y_pred)

#Evaluate Model Performance Using MAPE
mape_score = mean_absolute_percentage_error(y_valid_exp, y_pred_exp) * 100
print(f"ðŸ“Š Validation MAPE: {mape_score:.2f}%")


#Reload Label Encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

#Encode Test Data (Handling Unseen Categories)
for col in ["country", "store", "product"]:
    test_df[col] = test_df[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

#Select Features for Prediction
X_test = test_df[features]

#Make Predictions on Test Data
test_predictions = xgb_model.predict(X_test)

#Convert Predictions Back from Log Scale
test_predictions = np.expm1(test_predictions)

#Create Submission DataFrame
submission_df = pd.DataFrame({"id": test_df["id"], "num_sold": test_predictions})

#Ensure 98,550 Rows for Submission
assert submission_df.shape[0] == 98550, "ðŸš¨ Submission file does not have the correct number of rows!"

#Save as CSV for Kaggle
submission_df.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' is ready.")


