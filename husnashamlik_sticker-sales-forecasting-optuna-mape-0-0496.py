import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import mean_absolute_percentage_error,make_scorer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost
from xgboost import XGBRegressor
import optuna



train_data=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_data=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")



train_data


test_data



print('train_data: ',train_data.shape)
print('test_data: ',test_data.shape)


print('train_data: ',train_data.columns,'\n')
print('test_data: ',test_data.columns)


train_data.describe()


train_data.info()


train_data.drop(columns=['id'],inplace=True)


train_data


train_data.isnull().sum()


train_data.duplicated().sum()


target_col='num_sold'
num_col=train_data.select_dtypes(include=['number']).columns
cat_col=train_data.select_dtypes(include=['object']).columns
print("Target Columns: ",target_col)
print("\nNumrical Column: ",num_col.tolist())
print("\nCategorical Column: ",cat_col.tolist())


num_data=train_data.select_dtypes(include=['number'])
cat_data=train_data.select_dtypes(include=['object'])


print("Categorical Data Dsicription!")
cat_data.describe().T


for c in cat_data:
    col_count=train_data[c].nunique()
    print(f'{c} has {col_count} unique values.')
    print("**"*20)


# Check where missing values occur
missing_rows = train_data[train_data['num_sold'].isnull()]
print(missing_rows.head(),'\n')

# Check percentage of missing values
missing_percentage = train_data['num_sold'].isnull().mean() * 100
print(f"Missing Percentage: {missing_percentage:.2f}%")



train_data['num_sold']=train_data.groupby(['country','store','product'])['num_sold'].transform(lambda x:x.fillna(x.mean()))
train_data['num_sold']=train_data['num_sold'].fillna(train_data['num_sold'].mean())


#train_data=train_data.dropna(subset=['num_sold'])


#after handling missing values
train_data.isnull().sum()


# Convert the date column to datetime
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date']=pd.to_datetime(test_data['date'])
train_data


# Extract date-related features
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day
train_data['day_of_week'] = train_data['date'].dt.dayofweek
train_data['is_weekend'] = train_data['day_of_week'].isin([5, 6]).astype(int)

# Repeat for test data
test_data['year'] = test_data['date'].dt.year
test_data['month'] = test_data['date'].dt.month
test_data['day'] = test_data['date'].dt.day
test_data['day_of_week'] = test_data['date'].dt.dayofweek
test_data['is_weekend'] = test_data['day_of_week'].isin([5, 6]).astype(int)



#After extracting the necessary features, you can drop the original date column since it is no longer needed.

train_data = train_data.drop('date', axis=1)
test_data = test_data.drop('date', axis=1)




train_data


test_data


categorical_cols=['country','store','product']
encoder={}
for feature in categorical_cols:
    
    encoder[feature]=LabelEncoder()
    train_data[feature]=encoder[feature].fit_transform(train_data[feature])
    test_data[feature]=encoder[feature].fit_transform(test_data[feature])


train_data


test_data


#scaler=StandardScaler()

#train_data['num_sold']=scaler.fit_transform(train_data['num_sold'].values.reshape(-1,1))     



X=train_data.drop('num_sold',axis=1)
y=train_data['num_sold']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


models={'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest':RandomForestRegressor(),
        'XGBoost':XGBRegressor()
       }
for model_name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mape=mean_absolute_percentage_error(y_test,y_pred)
    print(f'{model_name} MAPE: {mape}')


# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
    max_depth = trial.suggest_categorical("max_depth", [None, 10, 20, 30, 40])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10, step=1)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4, step=1)
    max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
    
    # Initialize the Random Forest model with suggested parameters
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    
    # Use cross-validation to evaluate the model
    cv_scores = cross_val_score(
        rf, X_train, y_train, cv=3, scoring="neg_mean_absolute_percentage_error", n_jobs=-1
    )
    
    # Return the mean score (negative MAPE)
    return -1 * cv_scores.mean()

# Create an Optuna study
study = optuna.create_study(direction="minimize")  # Minimize MAPE
study.optimize(objective, n_trials=30, n_jobs=-1)  # Run optimization for 50 trials




# Retrieve the best parameters and train the best model
best_params = study.best_params
print(f"Best Parameters: {best_params}")


final_model = RandomForestRegressor(**best_params, random_state=42)
final_model.fit(X_train,y_train)


y_pred=final_model.predict(X_test)
maperror=mean_absolute_percentage_error(y_test,y_pred)
print(f'Test MAPE: {maperror}')


# Assuming final_model is your trained RandomForestRegressor
feature_importances = final_model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': X_train.columns,  # Replace with your feature names
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=importance_df.head(10), 
    palette='viridis'  # You can use 'purple' or other palettes too
)
plt.title('Random Forest Feature Importance', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()



test_transformed=test_data.drop(columns=['id'])
output= pd.DataFrame(test_data['id'])
rf_output = final_model.predict(test_transformed)
output['num_sold']= rf_output
output.to_csv("submission.csv", index = None)
output

