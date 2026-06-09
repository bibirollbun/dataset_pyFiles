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



train_data=pd.read_csv('/kaggle/input/innovative-ai-challenge-2024/train.csv')
test_data=pd.read_csv('/kaggle/input/innovative-ai-challenge-2024/test.csv')


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


numerical_col = ['Rainfall', 'Irrigation_Area', 'Crop_Yield (kg/ha)']
plt.figure(figsize=(12, 8))  # Set the figure size for all subplots
plot_num = 1

for col in numerical_col:
    if plot_num <= 3:  # Limit the number of subplots to match the number of columns
        plt.subplot(2, 2, plot_num)  # Create a 2x2 grid of subplots
        sns.histplot(data=train_data, x=col, kde=True, bins=30, color='green')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plot_num += 1

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()


plt.figure(figsize=(6, 4))
sns.scatterplot(data=train_data, x='Irrigation_Area', y='Crop_Yield (kg/ha)', color='purple')
sns.regplot(data=train_data, x='Irrigation_Area', y='Crop_Yield (kg/ha)', scatter=False, color='orange')
plt.title('Irrigation Area vs Crop Yield')
plt.xlabel('Irrigation Area (ha)')
plt.ylabel('Crop Yield (kg/ha)')
plt.show()


plt.figure(figsize=(6, 4))
sns.scatterplot(data=train_data, x='Rainfall', y='Crop_Yield (kg/ha)', color='green')
sns.regplot(data=train_data, x='Rainfall', y='Crop_Yield (kg/ha)', scatter=False, color='red')
plt.title('Rainfall vs Crop Yield')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Crop Yield (kg/ha)')
plt.show()


categorical_columns = ['State', 'Crop_Type', 'Soil_Type']
plt.figure(figsize=(16, 8))
plot_num=1
for col in categorical_columns:
    if plot_num<=3:
        plt.subplot(2,2,plot_num)
        sns.boxplot(data=train_data, x=col, y='Crop_Yield (kg/ha)', palette='Set3')
        plt.title(f'Crop Yield by {col}')
        plt.xlabel(col)
        plt.ylabel('Crop Yield (kg/ha)')
        plt.xticks(rotation=45)
        plot_num +=1
plt.show()


yearly_yield = train_data.groupby('Year')['Crop_Yield (kg/ha)'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_yield, x='Year', y='Crop_Yield (kg/ha)', marker='o', color='blue')
plt.title('Crop Yield Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Crop Yield (kg/ha)')
plt.show()


import plotly.express as px

fig = px.scatter_3d(train_data, x='Rainfall', y='Irrigation_Area', z='Crop_Yield (kg/ha)', color='Crop_Type')
fig.update_layout(title='Rainfall, Irrigation Area, and Crop Yield')
fig.show()


grouped_data = train_data.groupby(['State', 'Crop_Type'])['Crop_Yield (kg/ha)'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_data, x='State', y='Crop_Yield (kg/ha)', hue='Crop_Type', palette='tab10')
plt.title('Crop Yield by State and Crop Type')
plt.xlabel('State')
plt.ylabel('Average Crop Yield (kg/ha)')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10,6))
sns.pairplot(train_data, vars=['Rainfall', 'Irrigation_Area', 'Crop_Yield (kg/ha)'], hue='Soil_Type', palette='Dark2')
plt.show()


plt.figure(figsize=(12, 8))
sns.lineplot(data=train_data, x='Year', y='Crop_Yield (kg/ha)', hue='State', marker='o', palette='viridis')
plt.title('Crop Yield Trends by State')
plt.xlabel('Year')
plt.ylabel('Crop Yield (kg/ha)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


pivot_table = train_data.pivot_table(index='Soil_Type', columns='Crop_Type', values='Crop_Yield (kg/ha)', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Crop Yield by Soil Type and Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('Soil Type')
plt.show()


from matplotlib import cm
# Define categorical columns for pie chart
categorical_columns = ['State', 'Crop_Type', 'Soil_Type']

# Plotting pie charts
plt.figure(figsize=(20, 10))  # Set overall figure size
plot_num = 1
pastel_colors = cm.Pastel1.colors
for col in categorical_columns:
    plt.subplot(1, 3, plot_num)  # Create subplots
    data = train_data[col].value_counts()  # Get value counts for the column
    plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, colors=pastel_colors[:len(data)])
    plt.title(f"Proportion of {col}")
    plot_num += 1

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Data for crop types
crop_data = train_data['Crop_Type'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(crop_data, labels=crop_data.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors[:len(crop_data)])
plt.title("Distribution of Crop Types")
plt.show()


# Data for soil types
soil_data = train_data['Soil_Type'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(soil_data, labels=soil_data.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors[:len(soil_data)])
plt.title("Distribution of Soil Types")
plt.show()


categorical_cols=['State','Crop_Type','Soil_Type']
encoder={}
for feature in categorical_cols:
    
    encoder[feature]=LabelEncoder()
    train_data[feature]=encoder[feature].fit_transform(train_data[feature])
    test_data[feature]=encoder[feature].fit_transform(test_data[feature])


train_data


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


test_transformed=test_data.drop(columns=['id'])
output= pd.DataFrame(test_data['id'])
rf_output = final_model.predict(test_transformed)
output['Crop_Yield (kg/ha)']= rf_output
output.to_csv("/kaggle/working/submission.csv", index = None)
output




