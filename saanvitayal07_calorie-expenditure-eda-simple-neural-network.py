import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


df= pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")



print("--- Dataset Head ---")
df.head()


print("\n--- Missing Values Count ---")
print(df.isnull().sum())
print(test.isnull().sum())


df.describe()


df.columns


colums =[  'Height', 'Weight', 'Duration', 'Heart_Rate','Body_Temp']
# if df.isnull().sum().sum() == 0:
#     print("No missing values found.")
# else:
    
    # for i in colums :
    #     df[i] = df[i].replace(np.inf, np.nan)
    #     df[i] = df[i].fillna(df[i].median())
    # df['Age']=df["Age"].fillna(df['Age'].mean())
    # print("Missing values handled.")


for i in colums :
    df[i] = df[i].replace(np.inf, np.nan)
    df[i] = df[i].fillna(df[i].median())
    df['Age']=df["Age"].fillna(df['Age'].mean())
    print("Missing values handled.")


df = pd.get_dummies(df, columns=['Sex'], drop_first=True,dtype = int)
test =pd.get_dummies(test, columns = ['Sex'], drop_first = True , dtype = int)
df.head()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[colums] = scaler.fit_transform(df[colums])
test[colums] = scaler.transform(test[colums])


plt.figure(figsize=(12, 8))
for i, col in enumerate(colums, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


for i in colums :
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3-Q1
    upper = Q3+1.5*IQR
    lower = Q1 - 1.5*IQR
    df[i]=df[i].clip(lower = lower , upper = upper)
Q1_cal = df["Calories"].quantile(0.25)
Q3_cal = df["Calories"].quantile(0.75)
IQR_cal = Q3-Q1
upper_cal = Q3_cal+1.5*IQR_cal
lower_cal = Q1_cal - 1.5*IQR_cal
df["Calories"]=df["Calories"].clip(lower = lower_cal , upper = upper_cal)


for i in colums :
    Q1 = test[i].quantile(0.25)
    Q3 = test[i].quantile(0.75)
    IQR = Q3-Q1
    upper = Q3+1.5*IQR
    lower = Q1 - 1.5*IQR
    test[i]=test[i].clip(lower = lower , upper = upper)


list (df.columns)


plt.figure(figsize=(12, 8))
for i, col in enumerate(list (df.columns), 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


cleaned_df = pd.DataFrame(df)  
cleaned_df.to_csv('cleaned_data.csv', index=False)


test.head(10)


df= df.drop("id", axis = 1)
test.columns
test = test.drop("id",axis = 1)


df.boxplot(column='Calories', by='Sex_male', figsize=(6,4))
plt.title('Calories by Sex')
plt.ylabel('Calories')
plt.show()

print("\nMean Calories by Sex:")
print(df.groupby('Sex_male')['Calories'].mean())


numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
plt.figure(figsize=(8,6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Check skewness
print("\nSkewness of Features:")
print(df[numerical_cols].skew())


df['BMI'] = df['Weight'] / (df['Height'] ** 2)
df['Sex_Weight'] = df['Sex_male'] * df['Weight']

test['BMI'] = test['Weight'] / (test['Height'] ** 2)


corr = df.corr()
plt.figure(figsize = (8,4) )
sns.heatmap(corr, annot=True, cmap='coolwarm',  vmin=-1, vmax=1)


df = df.drop(['Height', 'Sex_Weight'], axis=1)
test = test.drop(["Height"], axis = 1)
# to remove redunncy 


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Duration', y='Heart_Rate', hue='Calories', size='Calories', palette='viridis')
plt.title('Duration vs Heart_Rate (Colored by Calories)')
plt.show()


df["Exercise_Intensity"] = df["Heart_Rate"] *df['Duration']
test["Exercise_Intensity"] = test["Heart_Rate"] *test['Duration']
new_cols = ['BMI', 'Exercise_Intensity']
df[new_cols] = scaler.fit_transform(df[new_cols])
test[new_cols] = scaler.transform(test[new_cols])


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df['Calories'] = scaler.fit_transform(df[['Calories']]) ### in histogram perivlsuy plotted, lot of skewness was observed 


X = df.drop('Calories', axis=1)
y = df['Calories']


X_train , X_temp, y_train, y_temp  = train_test_split(X, y, test_size=0.3, random_state=42)
X_val,X_test , y_val,y_test  = train_test_split(X_temp, y_temp, test_size = 0.9, random_state = 42)
print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")




tf.random.set_seed(30)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer + first hidden layer
    Dropout(0.2), 
    Dense(32, activation='relu'), 
    Dropout(0.2),
    Dense(16, activation='relu'), 
    Dense(1, activation ='softplus') 
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])  # Mean Absolute Error (MAE) for evaluation


model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30, 
    batch_size=32, 
    verbose=1 )


# # to avoid fititgn  my model again n again , i will save it and load it 

# import os
# model_dir1 = '/kaggle/working/model.h5'
# model_dir2 ='/kaggle/input/mymodel/tensorflow2/default/1/my_model.h5'

# from tensorflow.keras.models import load_model
# import tensorflow.keras.metrics
# # import tensorflow.keras.losses
# from tensorflow.keras.losses import mean_squared_error

# model = load_model(model_dir2, custom_objects={
#     'mse': tensorflow.keras.losses.mean_squared_error,
#     'mae': tensorflow.keras.metrics.mean_absolute_error
# })



# model.save('/kaggle/working/my_model.h5')


# model = tf.keras.models.load_model('/kaggle/working/my_model.h5')



y_pred = model.predict(X_test)



y_test_unscaled = scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
y_pred_unscaled = scaler.inverse_transform(y_pred).flatten()


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
print(f"Average Prediction Error (MAE): {mae:.2f} calories")


plt.figure(figsize=(6, 6))
plt.scatter(y_test_unscaled, y_pred_unscaled, alpha=0.5, color='blue')
plt.plot([y_test_unscaled.min(), y_test_unscaled.max()], [y_test_unscaled.min(), y_test_unscaled.max()], 'r--')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Actual vs Predicted Calories')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


results = model.predict(test)


results = np.maximum(results, 0)



r = pd.DataFrame(results)
r.describe()


results.shape


submission =pd.DataFrame({
    "id": pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")["id"],
    "Calories" :results.flatten()
})
submission.to_csv('submission.csv', index=False)



submission 




