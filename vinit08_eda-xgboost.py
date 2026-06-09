import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train_df.info()


test_df.info()


train_df.describe()


test_df.describe()


train_df.isna().sum() / train_df.shape[0] *100


test_df.isna().sum() / test_df.shape[0] *100


train_df.drop("id" , axis = 1 , inplace = True)
test_df.drop("id" , axis = 1 , inplace = True)


numeric_cols = [
    'Temparature', 
    'Humidity', 
    'Moisture',
    'Nitrogen', 
    'Potassium', 
    'Phosphorous'
]


for i in numeric_cols:
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_df[i], kde=True , color = "green")
    plt.title(f"Histogram of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")


for i in numeric_cols:
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(train_df[i], color = "green")
    plt.title(f"Boxplot of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")


for i in numeric_cols:
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.kdeplot(train_df[i], fill = True)
    plt.title(f"Kde of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")


soil_data = train_df["Soil Type"].value_counts().sort_values(ascending = False).reset_index()
soil_data


plt.figure(figsize = (12,6))
fig = px.bar(soil_data  , x= "Soil Type" , y = "count" , color = "Soil Type" , title = "Soil Type")
fig.show()


crop_data = train_df["Crop Type"].value_counts().sort_values(ascending = False).reset_index()
crop_data



plt.figure(figsize = (12,6))
fig = px.bar(crop_data  , x= "Crop Type" , y = "count" , color = "Crop Type" , title = "crop Type")
fig.show()


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])


test_df[numeric_cols] = scaler.fit_transform(test_df[numeric_cols])


from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()


train_df["Soil Type"] = encoder.fit_transform(train_df["Soil Type"])


test_df["Soil Type"] = encoder.fit_transform(test_df["Soil Type"])


train_df["Crop Type"] = encoder.fit_transform(train_df["Crop Type"])


test_df["Crop Type"] = encoder.fit_transform(test_df["Crop Type"])


train_df["Fertilizer Name"] = encoder.fit_transform(train_df["Fertilizer Name"])


from sklearn.model_selection import train_test_split


x = train_df.iloc[: , 0:-1]
y = train_df.iloc[: , -1]


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)


from xgboost import XGBClassifier


xgb_model = XGBClassifier(objective='reg:squarederror', n_jobs=-1, random_state=42,
                               n_estimators=100, max_depth=5, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8)


xgb_model.fit(x_train , y_train)



y_pred_probs = xgb_model.predict_proba(x_test)
top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y_test]

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
map3_score = mapk(actual, top_3_preds)
print(f" MAP@3 Score: {map3_score:.5f}")

