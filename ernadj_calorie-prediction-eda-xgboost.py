import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print(df_train.info())
print("\n")
print(df_train.shape)


print(df_test.info())
print("\n")
print(df_test.shape)


df_train.head()


df_test.head()


missing_cols = df_train.columns[df_train.isnull().any()]
print("\nColumns with missing values:")
print(missing_cols)

print("\n")

missing_cols = df_test.columns[df_test.isnull().any()]
print("\nColumns with missing values:")
print(missing_cols)

print("\n")


from sklearn.preprocessing import LabelEncoder

# Step 1: Initialize the label encoder
le = LabelEncoder()

# Step 2: Fit and transform the 'Sex' column - train dataset
df_train['Sex'] = le.fit_transform(df_train['Sex'])

# Step 2: Fit and transform the 'Sex' column - train dataset
df_test['Sex'] = le.fit_transform(df_test['Sex'])


df_train.head()  #checking label encoder for sex column


df_test.head()


dict(zip(le.classes_, le.transform(le.classes_))) #mapping the sex column


from warnings import filterwarnings
filterwarnings('ignore')

# Select relevant numerical columns for the pair plot
cols_to_plot = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# Create the pairplot
sns.pairplot(df_train[cols_to_plot], diag_kind='kde', corner=True)
#sns.pairplot(df_train[cols_to_plot], diag_kind='hist', corner=True)
plt.show()



# taking log value for target variable (Log (1+x))

df_train['Log_Calories'] = np.log1p(df_train['Calories'])


import pandas as p
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

X = df_train.drop(columns=['Calories', 'Log_Calories'])
y = df_train['Log_Calories']

# Split train set for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train XGBoost model
model = XGBRegressor(n_estimators=100, 
                     learning_rate=0.1, 
                     random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)
y_pred = np.maximum(0, y_pred)  # Avoid negative predictions for RMSLE

# 6. Evaluate with RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f'RMSLE: {rmsle:.4f}')


# Predict the target variable on the df_test dataset
y_pred_test = model.predict(df_test)

y_pred_test



y_pred_test = np.expm1(y_pred_test)
y_pred_test


# Ensuring the prediction does not have negative values
import pandas as pd
pd.Series(y_pred_test).describe().round(4)  


# Create a submission dataframe
# Replace 'ID_column' with the actual name of the identifier column in df_test
submission = pd.DataFrame({
    'id': df_test['id'],  # Replace 'id' with exact column in the submission file
    'Calories': y_pred_test
})

# Save the submission dataframe as a CSV file
submission.to_csv('submission.csv', index=False)
submission.head(5)

