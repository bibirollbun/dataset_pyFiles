import pandas as pd
import numpy as np
import seaborn as sns


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


trainDF = train.copy()
testDF = test.copy()


trainDF.head()


trainDF.info()


trainDF.isnull().sum()


import missingno as msno


msno.bar(trainDF);


msno.matrix(trainDF);


msno.heatmap(trainDF);


trainDF_ELM = trainDF["Episode_Length_minutes"]
trainDF_ELM.head()


sns.boxplot(x = trainDF_ELM);


Q1 = trainDF_ELM.quantile(0.25)
Q3 = trainDF_ELM.quantile(0.75)
IQR = Q3-Q1


print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)


low_limit = Q1-1.5*IQR
up_limit = Q3+1.5*IQR


print("Lower Limit:", low_limit)
print("Upper Limit:", up_limit)


mean_ELM = trainDF[(trainDF["Episode_Length_minutes"] >= low_limit) & 
                     (trainDF["Episode_Length_minutes"] <= up_limit)]["Episode_Length_minutes"].mean()



trainDF["Episode_Length_minutes"] = trainDF["Episode_Length_minutes"].apply(
    lambda x: mean_ELM if x < low_limit or x > up_limit else x
)


sns.boxplot(x = trainDF["Episode_Length_minutes"]);


trainDF["Episode_Length_minutes"].isnull().sum()


trainDF["Episode_Length_minutes"] = trainDF["Episode_Length_minutes"].fillna(trainDF["Episode_Length_minutes"].mean())


trainDF["Episode_Length_minutes"].isnull().sum()


trainDF_GPP = trainDF["Guest_Popularity_percentage"]
trainDF_GPP.head()


sns.boxplot(x = trainDF_GPP);


Q1 = trainDF_GPP.quantile(0.25)
Q3 = trainDF_GPP.quantile(0.75)
IQR = Q3-Q1


print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)


low_limit = Q1-1.5*IQR
up_limit = Q3+1.5*IQR


print("Lower Limit:", low_limit)
print("Upper Limit:", up_limit)


trainDF["Guest_Popularity_percentage"] = trainDF["Guest_Popularity_percentage"].fillna(trainDF["Guest_Popularity_percentage"].mean())


trainDF["Guest_Popularity_percentage"].isnull().sum()


sns.boxplot(x = trainDF["Guest_Popularity_percentage"]);


trainDF_NA = trainDF["Number_of_Ads"]
trainDF_NA.head()


sns.boxplot(x = trainDF_NA);


Q1 = trainDF_NA.quantile(0.25)
Q3 = trainDF_NA.quantile(0.75)
IQR = Q3-Q1


print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)


low_limit = Q1-1.5*IQR
up_limit = Q3+1.5*IQR


print("Lower Limit:", low_limit)
print("Upper Limit:", up_limit)


mean_NA = trainDF[(trainDF["Number_of_Ads"] >= low_limit) & 
                     (trainDF["Number_of_Ads"] <= up_limit)]["Number_of_Ads"].mean()


trainDF["Number_of_Ads"] = trainDF["Number_of_Ads"].apply(
    lambda x: mean_NA if x < low_limit or x > up_limit else x
)


sns.boxplot(x = trainDF["Number_of_Ads"]);


trainDF["Number_of_Ads"].isnull().sum()


trainDF["Number_of_Ads"].mean()


trainDF["Number_of_Ads"] = trainDF["Number_of_Ads"].fillna(trainDF["Number_of_Ads"].mean())


trainDF["Number_of_Ads"].isnull().sum()


trainDF_HPP = trainDF["Host_Popularity_percentage"]
trainDF_HPP.head()


sns.boxplot(x = trainDF_HPP);


trainDF_HPP.isnull().sum()


trainDF.head()


trainDF.info()


msno.bar(trainDF);


testDF.head()


testDF.info()


testDF.isnull().sum()


msno.bar(testDF);


msno.matrix(testDF);


msno.heatmap(testDF);


testDF_ELM = testDF["Episode_Length_minutes"]
testDF_ELM.head()


sns.boxplot(x = testDF_ELM);


testDF_ELM_Q1 = testDF_ELM.quantile(0.25)
testDF_ELM_Q3 = testDF_ELM.quantile(0.75)
testDF_ELM_IQR = testDF_ELM_Q3-testDF_ELM_Q1


print("Q1:", testDF_ELM_Q1)
print("Q3:", testDF_ELM_Q3)
print("IQR:", testDF_ELM_IQR)


testDF_ELM_low_limit = testDF_ELM_Q1-1.5*testDF_ELM_IQR
testDF_ELM_up_limit = testDF_ELM_Q3+1.5*testDF_ELM_IQR


print("Lower Limit:", testDF_ELM_low_limit)
print("Upper Limit:", testDF_ELM_up_limit)


testDF_ELM_mean = testDF[(testDF["Episode_Length_minutes"] >= testDF_ELM_low_limit) & 
                     (testDF["Episode_Length_minutes"] <= testDF_ELM_up_limit)]["Episode_Length_minutes"].mean()



testDF["Episode_Length_minutes"] = testDF["Episode_Length_minutes"].apply(
    lambda x: testDF_ELM_mean if x < testDF_ELM_low_limit or x > testDF_ELM_up_limit else x
)


sns.boxplot(x = testDF["Episode_Length_minutes"]);


testDF["Episode_Length_minutes"].isnull().sum()


testDF["Episode_Length_minutes"] = testDF["Episode_Length_minutes"].fillna(testDF["Episode_Length_minutes"].mean())


testDF["Episode_Length_minutes"].isnull().sum()


testDF_HPP = testDF["Host_Popularity_percentage"]
testDF_HPP.head()


sns.boxplot(x = testDF_HPP);


testDF_HPP.isnull().sum()


testDF_GPP = testDF["Guest_Popularity_percentage"]
testDF_GPP.head()


sns.boxplot(x = testDF_GPP);


testDF_GPP.isnull().sum()


testDF["Guest_Popularity_percentage"] = testDF["Guest_Popularity_percentage"].fillna(testDF["Guest_Popularity_percentage"].mean())


testDF["Guest_Popularity_percentage"].isnull().sum()


sns.boxplot(x = testDF["Guest_Popularity_percentage"]);


testDF_NA = testDF["Number_of_Ads"]
testDF_NA.head()


sns.boxplot(x = testDF_NA);


testDF["Number_of_Ads"].max()


testDF.loc[testDF["Number_of_Ads"].idxmax()]


testDF_NA_Q1 = testDF_NA.quantile(0.25)
testDF_NA_Q3 = testDF_NA.quantile(0.75)
testDF_NA_IQR = testDF_NA_Q3-testDF_NA_Q1


print("Q1:", testDF_NA_Q1)
print("Q3:", testDF_NA_Q3)
print("IQR:", testDF_NA_IQR)


testDF_low_limit = testDF_NA_Q1-1.5*testDF_NA_IQR
testDF_up_limit = testDF_NA_Q3+1.5*testDF_NA_IQR


print("Lower Limit:", testDF_low_limit)
print("Upper Limit:", testDF_up_limit)


testDF_mean = testDF[(testDF["Number_of_Ads"] >= testDF_low_limit) & 
                     (testDF["Number_of_Ads"] <= testDF_up_limit)]["Number_of_Ads"].mean()


testDF["Number_of_Ads"] = testDF["Number_of_Ads"].apply(
    lambda x: testDF_mean if x < testDF_low_limit or x > testDF_up_limit else x
)


sns.boxplot(x = testDF["Number_of_Ads"]);


testDF["Number_of_Ads"].isnull().sum()


testDF.head()


testDF.info()


msno.bar(testDF);


msno.bar(trainDF);


y = trainDF["Listening_Time_minutes"]
trainDF = trainDF.drop(["Listening_Time_minutes"], axis=1)
trainDF.head()


X_df = trainDF.copy()


print("Genre Number of Unique: ", X_df["Genre"].nunique())
print("Publication_Day Number of Unique: ", X_df["Publication_Day"].nunique())
print("Episode_Sentiment Number of Unique: ", X_df["Episode_Sentiment"].nunique())
print("Podcast_Name Number of Unique: ", X_df["Podcast_Name"].nunique())
print("Episode_Title Number of Unique: ", X_df["Episode_Title"].nunique())
print("Publication_Time Number of Unique: ", X_df["Publication_Time"].nunique())


dms = pd.get_dummies(X_df[['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']])


X_ = X_df.drop(['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'], axis=1)


X = pd.concat([X_, dms], axis=1)


X.head()


X = X.drop("id", axis=1)


X = X.drop(["Podcast_Name","Episode_Title"], axis=1)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.333333, 
                                                    random_state=42)


xgb_model = XGBRegressor().fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


test_ids = testDF["id"]


print(len(y_pred))
print(len(test_ids))


submission = pd.DataFrame({
    "id": test_ids,
    "Listening_Time_minutes": y_pred
})
submission.to_csv("submission.csv", index=False)

