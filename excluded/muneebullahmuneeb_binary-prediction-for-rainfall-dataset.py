import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# lets import the data set using csv file
df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
# lets check the first 5 rows of the data set
df.head()


# lets we see the info of the data
df.info()


# lets wee see the statistical summary of the data
df.describe().T


# lets we see the shape of the dataset
print('The rows of this dataset are: ', df.shape[0], 'and the columns are: ', df.shape[1])


# lets we first check the values counts of the maxtemp
print('This is the sum of value counts of the maximum temperature column: ', df['maxtemp'].value_counts().sum())


# lets make the plot of the 
print('The number of unique value of the maximum temp columns is : ', df['maxtemp'].nunique())


# lets we groupy the maxtemp based on the pressure
# we can use the groupby function from pandas
df.groupby('pressure')['maxtemp'].value_counts()


# lets we plot 
plt.figure(figsize=(10, 6))
sns.scatterplot(df, x = 'maxtemp', y = 'pressure')


# lets we groupby the min temperature based on the rainfall
df.groupby('rainfall')['mintemp'].min()





# lets we groupby the humidity based on the temperature
df.groupby('temparature')['humidity'].value_counts()


# lets we explore the number of unique values in the humidity column
humidity_unique = df['humidity'].nunique()
print(f"Number of unique humidity values: {humidity_unique}")


df.isnull().sum() * 100 / len(df)


sns.heatmap(df.isnull())


import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame for illustration
# df = pd.DataFrame(...)  # Ensure you have your DataFrame defined

# Define the number of columns in your DataFrame
num_cols = len(df.columns)

# Calculate the number of rows needed
num_rows = (num_cols + 1) // 2  # Using integer division to round up

plt.figure(figsize=(20, 10 * num_rows))

colors = ['red', 'green', 'orange', 'blue', 'purple', 'pink', 'brown', 'gray', 'olive', 'cyan', 'yellow', 'black', 'm']

for i, col in enumerate(df.columns):
    plt.subplot(num_rows, 2, i + 1)  # 2 columns
    sns.boxplot(x=df[col], color=colors[i % len(colors)])  # Use modulo to cycle through colors
    plt.title(col)  # Use the column name as the title

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# lets make the correlation matrix 
corr_matrix = df[['pressure', 'temparature', 'humidity', 'windspeed', 'rainfall']].corr()
corr_matrix
# lets plot the corr matrix
import seaborn as sns
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)



# we import all libraries above
# split the data into X and y
X = df.drop('rainfall', axis=1)
y = df['rainfall']
# we split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)                                                    



models = [
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=42)),
    ('XG boost', XGBRegressor(random_state=42)),
    ('Support Vector Regressor', SVR()),
    ('Lasso Regressor', Lasso(random_state=42)),
    ('Ridge Regressor', Ridge(random_state=42))
]

best_model = None
best_accuracy = 0.0


for name, model in models:
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('one encoder', OneHotEncoder(handle_unknown='ignore')),
        ('model', model)
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=5)
    
    mean_accuracy = scores.mean()
    
    
    pipeline.fit(X_train , y_train)
    
    y_pred = pipeline.predict(X_test)
    MSE = mean_squared_error(y_pred, y_test)
    
    print('Mean Squared Error: ', MSE)
    print('Croos validation accuracy: ', mean_accuracy)
    print('Model: ', name)
    
    


model = XGBRegressor()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
print('Mean Squared Error:',mean_squared_error(y_test, y_pred))



# save the model
import pickle

# Assuming 'model' is your trained model
with open('model_filename.pkl', 'wb') as file:
    pickle.dump(model, file)


# load the model
with open('model_filename.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

