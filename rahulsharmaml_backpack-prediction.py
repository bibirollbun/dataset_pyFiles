import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch


train=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


print(train.head())
print(train.shape)
print(torch.cuda.is_available())


print(train.isnull().sum())


print(test.head)
print(test.isnull().sum())


brands = train["Size"].unique()
for brand in brands:
    sns.kdeplot(train[train["Size"] == brand]["Price"], label=brand)
plt.axvline(x=25, color='r', linestyle='--', label='Division 1')
plt.axvline(x=130, color='g', linestyle='--', label='Division 2')
plt.legend()
plt.show()


train["Brand"]=train['Brand'].fillna(train["Brand"].mode()[0])
test["Brand"]=test['Brand'].fillna(test["Brand"].mode()[0])
train["Material"]=train['Material'].fillna(train["Material"].mode()[0])
test["Material"]=test['Material'].fillna(test["Material"].mode()[0])
train["Size"]=train['Size'].fillna(train["Size"].mode()[0])
test["Size"]=test['Size'].fillna(test["Size"].mode()[0])
train["Laptop Compartment"]=train["Laptop Compartment"].fillna(train["Laptop Compartment"].mode()[0])
test["Laptop Compartment"]=test["Laptop Compartment"].fillna(test["Laptop Compartment"].mode()[0])
train["Waterproof"]=train['Waterproof'].fillna(train["Waterproof"].mode()[0])
test["Waterproof"]=test['Waterproof'].fillna(test["Waterproof"].mode()[0])
train["Size"]=train['Size'].fillna(train["Size"].mode()[0])
test["Size"]=test['Size'].fillna(test["Size"].mode()[0])
train["Style"]=train['Style'].fillna(train["Style"].mode()[0])
test["Style"]=test['Style'].fillna(test["Style"].mode()[0])
train["Color"]=train['Color'].fillna(train["Color"].mode()[0])
test["Color"]=test['Color'].fillna(test["Color"].mode()[0])
train["Weight Capacity (kg)"]=train['Weight Capacity (kg)'].fillna(train["Weight Capacity (kg)"].mean())
test["Weight Capacity (kg)"]=test['Weight Capacity (kg)'].fillna(test["Weight Capacity (kg)"].mean())




print(train.isnull().sum())


print(train.columns)


category = ["Brand", "Material", "Size", "Waterproof", "Laptop Compartment", "Style", "Color"]

for col in category:
    # Get dummies for both train and test sets
    train_dummies = pd.get_dummies(train[col], prefix=col)
    test_dummies = pd.get_dummies(test[col], prefix=col)
    
    # Ensure both train and test have the same columns
    train_dummies, test_dummies = train_dummies.align(test_dummies, join='outer', axis=1, fill_value=0)
    
    # Concatenate dummies to the original DataFrame
    train = pd.concat([train, train_dummies], axis=1)
    test = pd.concat([test, test_dummies], axis=1)

# Drop the original columns
train.drop(columns=category, inplace=True)
test.drop(columns=category, inplace=True)


from xgboost import XGBRegressor
import pandas as pd

# Example train and test DataFrames
# Ensure these are defined in your notebook
# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")

# Define the model
model = XGBRegressor(
    tree_method="gpu_hist",  # Use GPU for training
    max_depth=8,
    colsample_bytree=0.9,
    subsample=0.9,
    n_estimators=2000,
    learning_rate=0.01,
    early_stopping_rounds=25,
    eval_metric="rmse"
)

# Define Xtrain and ytrain
Xtrain = train.drop(columns=["Price"])
ytrain = train["Price"]

# Fit the model
model.fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain)], verbose=True)

# Predict on the test set
predicted = model.predict(test)

print(predicted)



print(predicted)


ids=test["id"]
PredictionDF = pd.DataFrame({'id' : ids, 'Price' : predicted})

PredictionDF.to_csv('submission.csv', index=False)




