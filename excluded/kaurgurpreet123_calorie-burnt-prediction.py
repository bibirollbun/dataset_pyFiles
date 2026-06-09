import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# Train data overview
train.info()


train.isnull().sum()


train.head()


 sns.countplot(x="Sex", data=train) 


 print("Min Age: ", min(train["Age"]))
 print("Max Age: ", max(train["Age"]))
 print("Avg Age: ", sum(train["Age"])/len(train["Age"]))
 sns.histplot(train["Age"])


 # Height stats
 print("Min height: ", min(train["Height"]))
 print("Max height: ", max(train["Height"]))
 print("Avg height: ", sum(train["Height"])/len(train["Height"]))

 # Visualizing distribution
 sns.distplot(train["Height"], kde=True)


 # Weight stats
 print("Min weight: ", min(train["Weight"]))
 print("Max weight: ", max(train["Weight"]))
 print("Avg weight: ", sum(train["Weight"])/len(train["Weight"]))

 # Visualizing distribution
 sns.distplot(train["Weight"], kde=True)


 sns.scatterplot(x="Duration", y="Calories", data=train)
 sns.lineplot(x="Duration", y = "Calories", color = "red", data=train)
 plt.grid(True)
 plt.show()


 sns.scatterplot(x="Heart_Rate", y="Calories", data=train)
 sns.lineplot(x="Heart_Rate", y="Calories", data=train, color="Red")


 sns.scatterplot(x="Body_Temp", y="Calories", data=train)
 sns.lineplot(x="Body_Temp", y="Calories", data=train, color="Red")


 sns.scatterplot(x="Height", y="Calories", data=train)
 sns.lineplot(x="Height", y="Calories", data=train, color="Red")


 sns.scatterplot(x="Weight", y="Calories", data=train)
 sns.lineplot(x="Weight", y="Calories", data=train, color="Red")


# Creating BMI column
train["BMI"] = (train["Weight"] * 100*100 )/(train["Height"] * train["Height"])
train.drop(columns='BMI',inplace=True)


# Get numerical featres and find correlation
numerical_cols= train.select_dtypes(include = ['number'])
print(numerical_cols)


corr= numerical_cols.corr()
sns.heatmap(corr,cmap='coolwarm',annot=True)
plt.figure(figsize=(18,14))
plt.show()


from sklearn.preprocessing import LabelEncoder, StandardScaler
encoder = LabelEncoder()
scaler = StandardScaler()

train["Sex"] = encoder.fit_transform(train["Sex"])



train.info()


from sklearn.model_selection import train_test_split

X = train.drop(columns = ["Calories"])
y = train["Calories"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)


## Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 

# Intialize and train the model
model = LinearRegression()
model=model.fit(X,y)

# Make predictions on validation set
y_pred_val = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred_val)
r_squared = r2_score(y_val, y_pred_val)
print(f"Mean Squared Error on Validation Set: {mse}")
print(f"R-squared on Validation Set: {r_squared}")


# Make predictions on the test dataset
test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test["Sex"] = encoder.fit_transform(test["Sex"])
y_pred_test = model.predict(test)



# Submission CSV
submission = pd.DataFrame({
    'id': test['id'],        
    'Calories': y_pred_test       
})



# Save predictions to CSV file
submission.to_csv('submission.csv', index=False)


