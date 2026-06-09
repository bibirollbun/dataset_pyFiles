import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

print(f"Shape of train_df : {train_df.shape}")
print(f"Shape of test_df  : {test_df.shape}")


train_df.head()


test_df.head()


train_df.drop(columns=['id'], inplace=True)
test_ids = test_df['id']
test_df.drop(columns=['id'], inplace=True)


train_df.head()


train_df.shape


test_df.head()


test_df.shape


train_df.info()


train_df.isnull().sum()


test_df.isnull().sum()


columns = ['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color']


for col in columns:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    print(f'Filled the missing values of {col}')

train_df['Weight Capacity (kg)'] = train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].mean()) 


for col in columns:
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])
    print(f'Filled the missing values of {col}')

test_df['Weight Capacity (kg)'] = test_df['Weight Capacity (kg)'].fillna(test_df['Weight Capacity (kg)'].mean()) 

print("All Done")


train_df.isnull().sum()


test_df.isnull().sum()


train_df.head()


le_brand = LabelEncoder()
le_material = LabelEncoder()
le_size = LabelEncoder()
le_laptopCompartment = LabelEncoder()
le_waterproof = LabelEncoder()
le_style = LabelEncoder()
le_color = LabelEncoder()


train_df['Brand'] = le_brand.fit_transform(train_df['Brand'])
train_df['Material'] = le_material.fit_transform(train_df['Material'])
train_df['Size'] = le_size.fit_transform(train_df['Size'])
train_df['Laptop Compartment'] = le_laptopCompartment.fit_transform(train_df['Laptop Compartment'])
train_df['Waterproof'] = le_waterproof.fit_transform(train_df['Waterproof'])
train_df['Style'] = le_style.fit_transform(train_df['Style'])
train_df['Color'] = le_color.fit_transform(train_df['Color'])

test_df['Brand'] = le_brand.fit_transform(test_df['Brand'])
test_df['Material'] = le_material.fit_transform(test_df['Material'])
test_df['Size'] = le_size.fit_transform(test_df['Size'])
test_df['Laptop Compartment'] = le_laptopCompartment.fit_transform(test_df['Laptop Compartment'])
test_df['Waterproof'] = le_waterproof.fit_transform(test_df['Waterproof'])
test_df['Style'] = le_style.fit_transform(test_df['Style'])
test_df['Color'] = le_color.fit_transform(test_df['Color'])


train_df.head()


test_df.head()


scaler_compartments = MinMaxScaler()
scaler_weight = MinMaxScaler()

train_df['Compartments'] = scaler_compartments.fit_transform(train_df[['Compartments']])
train_df['Weight Capacity (kg)'] = scaler_weight.fit_transform(train_df[['Weight Capacity (kg)']])

test_df['Compartments'] = scaler_compartments.fit_transform(test_df[['Compartments']])
test_df['Weight Capacity (kg)'] = scaler_weight.fit_transform(test_df[['Weight Capacity (kg)']])


train_df.head()


test_df.head()


X = train_df.drop(columns=['Price'])
y = train_df['Price']


# Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


GBR = GradientBoostingRegressor(
    n_estimators=150,       
    learning_rate=0.05,    
    max_depth=5,           
    subsample=0.8,         
    min_samples_split=4,   
    min_samples_leaf=2,    
    random_state=42
)

# Train the model
GBR.fit(X_train, y_train)

# Predict
y_pred = GBR.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print results
print(f"MAE  : {mae}")
print(f"MSE  : {mse}")
print(f"RMSE : {rmse}")



predictions = GBR.predict(test_df)


submission = pd.DataFrame({'id': test_ids, 'prediction': predictions})

submission.to_csv('submission.csv', index=False)

