import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



test_data=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
print('test_data',test_data.head())
print(test_data.shape)
train_data=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
print('------------------------------------')
original_data=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
print('original_data: ',original_data.head())
print(original_data.shape)
print('----------------------------------------')



print('train_data',train_data.head())
print(train_data.shape)


train_data.info()


test_data.describe()


train_data.describe()


original_data.describe()


train_data.isnull().sum()/train_data.shape[0]*100


test_data.isnull().sum()/test_data.shape[0]*100


original_data.isnull().sum()/original_data.shape[0]*100


train_data.columns


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30, 28))  # Initialize figure
plots = 1  # Track subplot position

for col in train_data.columns:
    if train_data[col].dtype == object:  # Check if column is categorical
        plt.subplot(4, 4, plots)  # Set subplot position for boxplot
        
        # Fix: Assign hue to avoid warning
        sns.boxplot(data=train_data, x=col, y='Price', hue=col, palette='pastel', dodge=False)
        plt.xlabel(col, fontsize=28)
        plt.ylabel('Price',fontsize=28)
        plt.title(f'Boxplot for {col} and its Effect on Price')
        plt.xticks(rotation=45,fontsize=18)
        plt.yticks(fontsize=18)
        plots += 1  # Move to the next subplot
        
        # Add a pie chart for the same categorical column
        plt.subplot(4, 4, plots)  # Set subplot position for pie chart
        train_data[col].value_counts().plot.pie(autopct='%1.1f%%', cmap='Pastel1', textprops={'fontsize': 14})
        plt.ylabel('')  # Hide y-label to keep it clean
        plt.title(f'Category Distribution of {col}')
        plots += 1  # Move to the next subplot

plt.tight_layout()
plt.show()





columns = ['Weight Capacity (kg)', 'Compartments']

plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)  
sns.kdeplot(data=train_data, x=columns[0], fill=True, color='blue')
plt.title(f'KDE Plot for {columns[0]} (Train)')


plt.subplot(2, 2, 2)  
sns.kdeplot(data=test_data, x=columns[0], fill=True, color='red')
plt.title(f'KDE Plot for {columns[0]} (Test)')


plt.subplot(2, 2, 3)  
sns.histplot(data=train_data, x=columns[1], kde=True, color='green')
plt.title(f'KDE Plot for {columns[1]} (Train)')

plt.subplot(2, 2, 4)  
sns.histplot(data=test_data, x=columns[1], kde=True, color='purple')
plt.title(f'KDE Plot for {columns[1]} (Test)')

plt.tight_layout()
plt.show()




# Heatmap for train data (includes 'Price')
plt.figure(figsize=(5, 4))
sns.heatmap(train_data[['Weight Capacity (kg)', 'Compartments', 'Price']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Train Data)")
plt.show()

# Heatmap for test data (excludes 'Price')
plt.figure(figsize=(5, 4))
sns.heatmap(test_data[['Weight Capacity (kg)', 'Compartments']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Test Data)")
plt.show()




data=original_data.dropna()
train_data=pd.concat([train_data,data],axis=0).reset_index(drop=True)
train_data.shape


train_data.isnull().sum()


test_data.isnull().sum()


categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_features = ["Weight Capacity (kg)"]

# Fill missing values in categorical columns with mode
for col in train_data.select_dtypes(include=['object']).columns:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

# Fill missing values in numerical columns with median

train_data["Weight Capacity (kg)"] = train_data["Weight Capacity (kg)"].fillna(train_data["Weight Capacity (kg)"].median())
test_data["Weight Capacity (kg)"] = test_data["Weight Capacity (kg)"].fillna(test_data["Weight Capacity (kg)"].median())




import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def perform_feature_engineering(df):
    # Brand Material Interaction - Certain materials may be common for specific brands
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']

    # Brand & Size Interaction - Some brands may produce only specific sizes
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']

    # Has Laptop Compartment - Convert Yes/No to 1/0 for easier analysis
    df['Has_Laptop_Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})

    # Is Waterproof - Convert Yes/No to 1/0 for easier analysis
    df['Is_Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})

    # Compartments Binning - Group compartments into categories
    df['Compartments_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])

    # Weight Capacity Ratio - Normalize weight capacity using the max value
    df['Weight_Capacity_Ratio'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()

    # Interaction Feature: Weight vs. Compartments - Some bags may hold more with less compartments
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)  # Avoid division by zero

    # Style and Size Interaction - Certain styles may correlate with sizes
    df['Style_Size'] = df['Style'] + '_' + df['Size']

    return df

# Apply the function to the training data
train_data = perform_feature_engineering(train_data)

# Apply the function to the test data
test_data = perform_feature_engineering(test_data)






id_test = test_data['id']

columns_to_drop = ['id']
train_data.drop(columns_to_drop, axis=1, inplace=True)
test_data.drop(columns_to_drop, axis=1, inplace=True)




from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
columns_to_encode = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color','Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment','Is_Waterproof', 'Compartments_Category', 'Style_Size']

for col in columns_to_encode:
    train_data[col] = encoder.fit_transform(train_data[col])
    test_data[col] = encoder.fit_transform(test_data[col])


train_data


test_data


columns_to_check = ['Weight Capacity (kg)','Weight_Capacity_Ratio','Weight_to_Compartments']

# Function to remove outliers using IQR and visualize
def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.15)
    Q3 = data[column].quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Calculate the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)
    
    # Plot the distribution with outliers
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[column], color='lightblue', flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
    
    # Highlight Q1 and Q3
    plt.axvline(Q1, color='green', linestyle='--', label='Q1 (10th Percentile)')
    plt.axvline(Q3, color='blue', linestyle='--', label='Q3 (90th Percentile)')
    
    # Highlight lower and upper bounds
    plt.axvline(lower_bound, color='red', linestyle='-', label='Lower Bound')
    plt.axvline(upper_bound, color='red', linestyle='-', label='Upper Bound')

    plt.title(f'Outlier Detection for {column}')
    plt.legend()
    plt.xlabel(column)
    plt.show()
    
    return filtered_data, rows_deleted

# Apply function to each numerical column and visualize
rows_deleted_total = 0

for column in columns_to_check:
    train_data, rows_deleted = remove_outliers_iqr_with_plot(train_data, column)
    rows_deleted_total += rows_deleted
    print(f"Rows deleted for {column}: {rows_deleted}")

print(f"Total rows deleted: {rows_deleted_total}")


remove_outliers_iqr_with_plot(test_data,columns[0])


remove_outliers_iqr_with_plot(train_data,columns[1])


remove_outliers_iqr_with_plot(test_data,columns[1])


from sklearn.preprocessing import StandardScaler
std=StandardScaler()
for i in  ['Weight Capacity (kg)','Weight_Capacity_Ratio','Weight_to_Compartments']:
    train_data[i]=std.fit_transform(train_data[[i]])
    test_data[i]=std.fit_transform(test_data[[i]])


train_data = train_data[[col for col in train_data.columns if col != 'Price'] + ['Price']]
train_data


train_data.shape


from sklearn.model_selection import train_test_split
X = train_data.drop(columns=["Price"], axis =1)
y = train_data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state = 42)




from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna
def objective(trial):

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1)
    }

    model = XGBRegressor(
        tree_method="gpu_hist",
        random_state=0,
        **params
    )

    scores = cross_val_score(
        model, 
        X, 
       y, 
        cv=3,  
        scoring="neg_mean_squared_error"  
    )

    rmse_scores = np.sqrt(-scores)  
    
    return rmse_scores.mean() 



model_xgb = XGBRegressor(
    eval_metric="rmse",
    enable_categorical=True,
    random_state=42,
)




model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],  # Provide validation set
    verbose=False
)





importance = model_xgb.feature_importances_

sorted_idx = np.argsort(importance)[::-1]
features = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh([features[i] for i in sorted_idx], importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  
plt.show()






test_preds = model_xgb.predict(X_test)



submission = pd.DataFrame({ "id": X_test.index, 'target': test_preds })

submission.to_csv("submission.csv", index=False)



