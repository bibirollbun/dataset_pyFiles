import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import xgboost
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import Ridge




train_data=pd.read_csv('/kaggle/input/innovative-ai-challenge-2024/train.csv')
test_data=pd.read_csv('/kaggle/input/innovative-ai-challenge-2024/test.csv')
                


train_data


test_data


print('train_data: ',train_data.shape)
print('test_data: ',test_data.shape)


print('train_data: ',train_data.columns)
print('test_data: ',test_data.columns)


train_data.describe()


train_data.info()


train_data.drop(columns=['id'],inplace=True)


train_data


train_data.isnull().sum()


train_data.duplicated().sum()


target_col='Crop_Yield (kg/ha)'
num_col=train_data.select_dtypes(include=['number']).columns
cat_col=train_data.select_dtypes(include=['object']).columns
print("Target Columns: ",target_col)
print("\nNumrical Column: ",num_col.tolist())
print("\nCategorical Column: ",cat_col.tolist())


num_data=train_data.select_dtypes(include=['number'])
cat_data=train_data.select_dtypes(include=['object'])



print('Numerical Data Distribution!')
num_data.describe().round(2).T


print("Categorical Data Dsicription!")
cat_data.describe().T


for c in cat_data:
    col_count=train_data[c].nunique()
    print(f'{c} has {col_count} unique values.')
    print("**"*20)


for i in cat_col:
    cat_value=train_data[i].value_counts()
    print(f'value count for {i} is :')
    print(cat_value)
    print("-"*20)
    


categorical_cols=['State','Crop_Type','Soil_Type']
encoder={}
for feature in categorical_cols:
    
    encoder[feature]=LabelEncoder()
    train_data[feature]=encoder[feature].fit_transform(train_data[feature])
    test_data[feature]=encoder[feature].fit_transform(test_data[feature])
    


train_data


#categorical_cols=['State','Crop_Type','Soil_Type']
#train_data=pd.get_dummies(train_data,columns=categorical_cols,dtype=int)
#test_data=pd.get_dummies(test_data,columns=categorical_cols,dtype=int)



scaler=StandardScaler()
numerical_cols=['Rainfall','Irrigation_Area']
train_data[numerical_cols]=scaler.fit_transform(train_data[numerical_cols])     
test_data[numerical_cols]=scaler.transform(test_data[numerical_cols])


X=train_data.drop('Crop_Yield (kg/ha)',axis=1)
y=train_data['Crop_Yield (kg/ha)']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# Define models
models = {
    'Ridge Regression': Ridge(),
    'Random Forest': RandomForestRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'XGBoost': XGBRegressor()
}

# Train and evaluate each model using k-fold cross-validation
for model_name, model in models.items():
    # K-fold Cross-Validation with 5 folds
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Convert negative MSE to positive
    mean_mse = np.mean(-cv_scores)
    mean_r2 = np.mean(r2_scores)
    
    # Output results
    print(f'{model_name} mean MSE (using K-fold CV): {mean_mse}')
    print(f'{model_name} mean R2 score (using K-fold CV): {mean_r2 * 100:.2f}%','\n')



param_grid={'n_estimators':[50,100,200,300],
            'max_depth':[None,10,20,30,40],
            'min_samples_split':[2,5,10],
            'min_samples_leaf':[1,2,4],
            'max_features':['auto','sqrt','log2']}
rf=RandomForestRegressor(random_state=42)
random_search=RandomizedSearchCV(estimator=rf,param_distributions=param_grid,
                                cv=5,scoring='neg_mean_squared_error',
                                verbose=2,n_jobs=-1,n_iter=50,random_state=42)
random_search.fit(X_train,y_train)
best_params=random_search.best_params_
best_model=random_search.best_estimator_
y_pred=best_model.predict(X_test)

# Output the results
print(f"Best Parameters: {best_params}")




final_model = RandomForestRegressor(**best_params, random_state=42)
final_model.fit(X_train,y_train)



y_pred=final_model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2score=r2_score(y_test,y_pred)
print(f"Test MSE: {mse:.2f}")
print(f"Test R2 Score: {r2score * 100:.2f}%")


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



test_data


test_transformed=test_data.drop(columns=['id'])
output= pd.DataFrame(test_data['id'])
rf_output = final_model.predict(test_transformed)
output['Crop_Yield (kg/ha)']= rf_output
output.to_csv("/kaggle/working/submission.csv", index = None)
output

