import numpy as np # linear algebra
import pandas as pd # data processing


train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
#train_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.head()


test.head()


train.info()


train.describe()


train.nunique()


for df in [train,test]:
    df.drop(['id'],axis=1, inplace=True)
train.head()


import matplotlib.pyplot as plt
import seaborn as sns


#Distributions:
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 2)
sns.histplot(train["Compartments"], bins=10, kde=True, color='green')
plt.title("Compartments Distribution")
plt.xlabel("Number of Compartments")

plt.subplot(1, 3, 3)
sns.histplot(train["Weight Capacity (kg)"], bins=10, kde=True, color='red')
plt.title("Weight Capacity Distribution")
plt.xlabel("Weight Capacity (kg)")

plt.tight_layout()
plt.show()


# Check Price Distribution
plt.figure(figsize=(10, 5))
sns.histplot(train["Price"], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()


#Relation of material with price:
plt.figure(figsize=(8, 6))
sns.boxplot(x='Material', y='Price', data=train)
plt.xticks(rotation=45)
plt.title("Price Distribution by Material")
plt.show()


# Relation between brand and price:
plt.figure(figsize=(8,6))
sns.boxplot(x='Brand', y='Price', data=train)
plt.xticks(rotation=45)
plt.title("Price Distribution by Material")
plt.show()


#Correlation heatmap:
plt.figure(figsize=(8, 5))
corr = train.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# Relation between style and price:
plt.figure(figsize=(8,6))
sns.boxplot(x='Style', y='Price', data=train)
plt.xticks(rotation=45)
plt.title("Price Distribution by Material")
plt.show()


# Material + Brand interaction:
plt.figure(figsize=(10, 4))
sns.boxplot(x="Material", y="Price", hue="Brand", data=train)
plt.xticks(rotation=45)
plt.title("Price by Material & Brand")
plt.show()


# Style + Brand interaction:
plt.figure(figsize=(10, 4))
sns.boxplot(x="Style", y="Price", hue="Brand", data=train)
plt.xticks(rotation=45)
plt.title("Price by Material & Brand")
plt.show()


#Missing data:
missing_data = train.isnull().sum()
missing_data = missing_data[missing_data > 0] 

if not missing_data.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_data.index, y=missing_data.values)
    plt.title('Missing Value Distribution in df_train')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=90)
    plt.show()
else:
    print("No missing values in the dataset.")


train.head()


test.head()


train.isnull().sum()


test.isnull().sum()


#Dealing with missing values

# Categorical columns to fill with mode (most frequent value)
categorical_cols = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']

for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)

# Numerical columns to fill with median (for features like 'Weight Capacity (kg)', 'Laptop Compartment', 'Waterproof')
numerical_cols = ['Weight Capacity (kg)']

for col in numerical_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

# Verify that there are no more missing values
missing_values_after = train.isnull().sum()
print(missing_values_after[missing_values_after > 0])




#Credits: Satya, this code is inspired by his notebook

def feature_engineering(df):
    #Encode size
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Size'] = df['Size'].map(size_mapping)
    
    df['Compartments_per_Size'] = df['Compartments'] / df['Size']    
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments'] 

    #Encode binary cols
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    
    df['Waterproof_Laptop'] = df['Waterproof'] * df['Laptop Compartment']
    
    #Material based
    df['Is_Durable_Material'] = df['Material'].apply(lambda x: 1 if x in ['Leather', 'Nylon'] else 0)
    df['Is_Lightweight_Material'] = df['Material'].apply(lambda x: 1 if x in ['Canvas', 'Nylon'] else 0)
    df['Luxury_Material'] = df['Material'].apply(lambda x: 1 if x == 'Leather' else 0)
    
    #Style based
    df['Professional_Style'] = df['Style'].apply(lambda x: 1 if x in ['Messenger', 'Tote'] else 0)
    df['Casual_Style'] = df['Style'].apply(lambda x: 1 if x in ['Backpack', 'Duffle'] else 0)
    
    #Brand based
    df['Is_Premium_Brand'] = df['Brand'].apply(lambda x: 1 if x in ['Nike', 'Under Armour', 'Adidas'] else 0)
    df['Is_Budget_Brand'] = df['Brand'].apply(lambda x: 1 if x == 'Jansport' else 0)
    
    #Size based 
    df['Is_Small'] = df['Size'].apply(lambda x: 1 if x == 1 else 0)
    df['Is_Medium'] = df['Size'].apply(lambda x: 1 if x == 2 else 0)
    df['Is_Large'] = df['Size'].apply(lambda x: 1 if x == 3 else 0)
    
    # One-Hot Encode categorical columns
    categorical_cols_onehot = ['Brand', 'Material', 'Style', 'Color']
    df = pd.get_dummies(df, columns=categorical_cols_onehot, drop_first=True)  # Drop the first category to avoid multicollinearity
    
    return df

train_preprocessed = feature_engineering(train)
test_preprocessed = feature_engineering(test)


train_preprocessed.dtypes


train_preprocessed.columns, test_preprocessed.columns


#Splitting the dataset:
from sklearn.model_selection import train_test_split

X= train_preprocessed.drop(columns=["Price"], axis =1)
y=train_preprocessed['Price']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state=42)


#Imprting necessary libraries
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


#Defining the parameters,parameters based on tested best values

params = {
    'max_depth': 6,
    'learning_rate': 0.04,
    'min_child_weight': 1,
    'subsample': 0.12,
    'colsample_bylevel': 0.22,
    'colsample_bytree': 0.9,
    'colsample_bynode': 0.55,
    'reg_alpha': 0.65,
    'reg_lambda': 0.4,
    'eval_metric': 'rmse',
    'n_estimators': 100
}

# Initialize the XGBoost Regressor with the parameters
xgb_model = xgb.XGBRegressor(**params)

# Train the model on the training set
xgb_model.fit(X_train, y_train)


# Plot feature importance
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10)
plt.show()


# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Calculate Mean Squared Error (MSE) and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print RMSE and R²
print(f'Root Mean Squared Error (RMSE): {rmse}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')


from catboost import CatBoostRegressor

# Define the CatBoost regressor
catboost_model = CatBoostRegressor(
    iterations=680,      
    learning_rate=0.009,
    depth=5,
    eval_metric='RMSE',
    random_seed=42, 
    verbose=200      
)

# Fit the model on the training data
catboost_model.fit(X_train, y_train)


# Predict using the trained model
predictions = catboost_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"CatBoost RMSE: {rmse}")



from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# Define a meta-model (e.g., Ridge)
meta_model = Ridge()

# Create the stacking model with XGBoost and CatBoost as base models
stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('catboost', catboost_model)], 
    final_estimator=meta_model
)

# Fit the stacking model on your training data
stacking_model.fit(X_train, y_train)

# Make predictions on the test data
stacking_preds = stacking_model.predict(X_test)

# Calculate RMSE for the stacked model
stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_preds))
print(f"Stacked Model RMSE: {stacking_rmse}")


# Predict the prices for the test set
test_predictions = stacking_model.predict(test_preprocessed)


test_predictions


# Load the submission file
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv') 

# Assign the predicted prices to the 'Price' column
submission['Price'] = test_predictions


submission.head(10)


submission.describe()


submission.to_csv("submission.csv", index=False)

