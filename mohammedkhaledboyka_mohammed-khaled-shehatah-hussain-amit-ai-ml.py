import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.metrics import *
from xgboost import *
from sklearn.preprocessing import *
from sklearn.ensemble import *
import warnings
warnings.filterwarnings("ignore")


Data_Frame_One = pd.read_csv("/kaggle/input/train-and-test-and-submissiom-test-dataset-by-mohammed/train.csv")
Data_Frame_One


Data_Frame_Two = pd.read_csv("/kaggle/input/train-and-test-and-submissiom-test-dataset-by-mohammed/test.csv")
Data_Frame_Two


Data_Frame_One.info()


Data_Frame_One.describe()


Data_Frame_One.isnull().sum()


Data_Frame_One["staff_experience"].fillna(Data_Frame_One["staff_experience"].mode()[0],inplace=True)
Data_Frame_One["staff_experience"].mode()[0]


Data_Frame_One.isnull().sum()


Data_Frame_One["staff_experience"] = LabelEncoder().fit_transform(Data_Frame_One["staff_experience"])
Data_Frame_One["waste_category"] = LabelEncoder().fit_transform(Data_Frame_One["waste_category"])
Data_Frame_One.info()


Data_Frame_One = Data_Frame_One.drop(["ID" , "date"],axis=1)
Data_Frame_One


Data_Frame_One.hist(figsize = (20,20))


sns.heatmap(Data_Frame_One.corr(),annot=True)


Data_Frame_One.dropna()
Data_Frame_One.drop_duplicates()


Scaler = StandardScaler()
X = Data_Frame_One.drop(["food_waste_kg"],axis=1)


X_Scaled = Scaler.fit_transform(X)
Y = Data_Frame_One["food_waste_kg"]
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X_Scaled,Y,train_size=0.8,random_state=30)


Model = LinearRegression()
Model.fit(X_Train , Y_Train)
Y_Pred = Model.predict(X_Test)
Mae = mean_absolute_error(Y_Test , Y_Pred)
Mse = mean_squared_error(Y_Test , Y_Pred)
Root_Mse=(Mse**(0.5))
R2_Score = r2_score(Y_Test , Y_Pred)


print("The Score Of Training =",Model.score(X_Train,Y_Train))
print("The Score Of Testing =",Model.score(X_Test,Y_Test))
print("The R2 Score For Train Dataset = " , R2_Score)
print("The Mean Abselute Error = " , Mae)
print("The Mean Square Error = " , Mse)
print("The Root Mean Square Error = " , Root_Mse)


X_Train,X_Test,Y_Train,Y_Test=train_test_split(X_Scaled,Y,train_size=0.8,random_state=40)
Advanced_Model = XGBRegressor()
Advanced_Model.fit(X_Train , Y_Train)
Y_Pred = Advanced_Model.predict(X_Test)
Mae = mean_absolute_error(Y_Test , Y_Pred)
Mse = mean_squared_error(Y_Test , Y_Pred)
Root_Mse=(Mse**(0.5))
R2_Score = r2_score(Y_Test , Y_Pred)


print("The Score Of Training Round Two =",Advanced_Model.score(X_Train,Y_Train))
print("The Score Of Testing Round Two =",Advanced_Model.score(X_Test,Y_Test))
print("The R2 Score For Train Dataset Round Two = " , R2_Score)
print("The Mean Abselute Error Round Two = " , Mae)
print("The Mean Square Error Round Two = " , Mse)
print("The Root Mean Square Error Round Two = " , Root_Mse)


Data_Frame_Two


Data_Frame_Two.info()


Data_Frame_Two.describe()


Data_Frame_Two.isnull().sum()


Data_Frame_Two["staff_experience"].fillna(Data_Frame_Two["staff_experience"].mode()[0],inplace=True)
Data_Frame_Two["staff_experience"].mode()[0]


Data_Frame_Two.isnull().sum()


Data_Frame_Two["staff_experience"] = LabelEncoder().fit_transform(Data_Frame_Two["staff_experience"])
Data_Frame_Two["waste_category"] = LabelEncoder().fit_transform(Data_Frame_Two["waste_category"])
Data_Frame_Two.info()


Submission_ID_Test = Data_Frame_Two["ID"]
pd.DataFrame(Submission_ID_Test)


Data_Frame_Two = Data_Frame_Two.drop(["ID" , "date"],axis=1)
Data_Frame_Two


Data_Frame_Two.hist(figsize = (20,20))


sns.heatmap(Data_Frame_Two.corr(),annot=True)


Data_Frame_Two.dropna()
Data_Frame_Two.drop_duplicates()


Scaler = StandardScaler()
X_Scaled = Scaler.fit_transform(X)
Test_Scaled = Scaler.transform(Data_Frame_Two)
Test_Predictions = Model.predict(Test_Scaled)


Submission_Test_New_Dataset = pd.DataFrame({"ID" : Submission_ID_Test , "food_waste_kg" :Test_Predictions})
Submission_Test_New_Dataset


Submission_Test_New_Dataset.to_csv("Submission_Test_New_Dataset_By_Mohammed_Khaled_Shehata_Hussain.csv",index=False)
Data_Frame_Three = pd.read_csv("/kaggle/input/train-and-test-and-submissiom-test-dataset-by-mohammed/Submission_Test_New_Dataset_By_Mohammed_Khaled_Shehata_Hussain.xls")
Data_Frame_Three

