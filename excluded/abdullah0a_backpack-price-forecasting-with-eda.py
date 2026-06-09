import numpy as np
import pandas as pd

# Set random seed for NumPy
np.random.seed(42)



# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')



train_df.shape


test_df.shape


train_df.head()


# Basic information about the dataset
train_df.info()


train_df.isnull().sum()


print(train_df['Price'].max())
print(train_df['Price'].min())


skewness = train_df['Price'].skew()
print(f"Skewness of Price: {skewness}")


train_df['Size'].value_counts()


# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


msno.matrix(train_df)


# Visualize the distribution of the target variable 'Price'
plt.figure(figsize=(10,6))
sns.histplot(train_df['Price'], kde=True, color='blue')
plt.title('Price Distribution')
plt.show()


# Bar plot of price for each brand
plt.figure(figsize=(10, 6))
sns.barplot(x='Brand', y='Price', data=train_df, palette='viridis')
plt.title('Price Comparison by Brand')
plt.xlabel('Brand')
plt.ylabel('Price')
plt.show()


# 3. Box Plot: Comparison of Material Types by Price
plt.figure(figsize=(10, 6))
sns.boxplot(x='Material', y='Price', data=train_df, palette='Set2')
plt.title('Price Comparison by Material Type')
plt.xlabel('Material')
plt.ylabel('Price')
plt.show()


# Machine learning models and tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Function for feature engineering
def create_features(df):

  
    # Define weight capacity bins
    bins = [0, 5, 10, 20, 30]  # Example weight capacity bins
    labels = ['Light', 'Medium', 'Heavy', 'Extra Heavy']  # Corresponding labels

    df['weight_capacity_category'] = pd.cut(df['Weight Capacity (kg)'], bins=bins, labels=labels)

    return df

train_df = create_features(train_df)
test_df  =create_features(test_df)



train_df.head()


train_df.info()


categorical_cols = ['Brand','Size','Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

for col in categorical_cols:
    train_df[col].fillna('Unknown', inplace=True)
    test_df[col].fillna('Unknown', inplace=True)


# Handle missing numerical values (fill with mean or drop rows)

train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].mean(), inplace=True)
test_df['Weight Capacity (kg)'].fillna(test_df['Weight Capacity (kg)'].mean(), inplace=True)



train_df.isnull().sum()


# One-Hot Encoding for categorical variables
train_df = pd.get_dummies(train_df, drop_first=True)
test_df  = pd.get_dummies(test_df, drop_first=True)


X = train_df.drop(columns=['Price', 'id'])
y = train_df['Price']


# Scaling the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split the dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# Initialize the XGBoost Regressor
import xgboost as xgb
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression task
    n_estimators=2000,  # Number of trees
    max_depth=8,  # Maximum depth of each tree
    learning_rate=0.006,  # Step size at each iteration
    subsample=0.7,  # Fraction of samples used for each tree
    colsample_bytree=0.7,  # Fraction of features used for each tree
    random_state=42
)

model.fit(X_train, y_train)



# Predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate performance using RMSE
rmse_gb = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE : {rmse_gb:.4f}")


"""from sklearn.model_selection import cross_val_score

# Perform cross-validation with RMSE as the evaluation metric
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

# Calculate and print the average RMSE across all folds
average_rmse = -cv_scores.mean()  # Convert negative RMSE to positive
print(f"Average RMSE from 5-fold cross-validation: {average_rmse:.4f}")
"""


# Prepare the test dataset
X_test = test_df.drop(columns=['id'])
X_test_scaled = scaler.transform(X_test)

# Predictions using the best model (Random Forest)
final_predictions = model.predict(X_test_scaled)

# Prepare the submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Price': final_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)



submission.head()

