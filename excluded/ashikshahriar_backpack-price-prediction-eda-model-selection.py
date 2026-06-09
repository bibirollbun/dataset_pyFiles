import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


train_data=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_data.head()


train_data.shape


train_data.info()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(train_data.isnull(), cmap='viridis', cbar=False)
plt.title("Missing Values Heatmap")
plt.show()



# Plot distributions of numerical features
train_data.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Distribution of Numerical Columns", fontsize=14)
plt.show()



plt.figure(figsize=(12, 5))
sns.countplot(data=train_data, y='Brand', order=train_data['Brand'].value_counts().index[:10])
plt.title("Top 10 Most Common Brands")
plt.show()



plt.figure(figsize=(10, 5))
sns.countplot(data=train_data, x='Size', order=train_data['Size'].value_counts().index[:10])
plt.title("Most Common Sizes")
plt.xticks(rotation=45)
plt.show()



plt.figure(figsize=(10, 5))
sns.histplot(train_data['Price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.show()



plt.figure(figsize=(10, 5))
sns.boxplot(data=train_data, x='Compartments', y='Price')
plt.title("Price vs Number of Compartments")
plt.show()



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


numerical_cols = train_data.select_dtypes(include=['number']).columns
categorical_cols = train_data.select_dtypes(exclude=['number']).columns

# Impute missing values
num_imputer = SimpleImputer(strategy='median')  # Replace NaNs with median for numerical columns
cat_imputer = SimpleImputer(strategy='most_frequent')  # Replace NaNs with mode for categorical columns

train_data[numerical_cols] = num_imputer.fit_transform(train_data[numerical_cols])
train_data[categorical_cols] = cat_imputer.fit_transform(train_data[categorical_cols])



train_data.isnull().sum()


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select only numerical columns
numerical_cols = train_data.select_dtypes(include=['int64', 'float64']).columns

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_data[numerical_cols])

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Plot PCA result
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=train_data['Price'], cmap='coolwarm', alpha=0.5)
plt.title("PCA - Feature Reduction Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Price')
plt.show()



from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(scaled_data)

# Plot t-SNE result
plt.figure(figsize=(10, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=train_data['Price'], cmap='coolwarm', alpha=0.5)
plt.title("t-SNE - Data Clusters Visualization")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label='Price')
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
# df = pd.read_csv('your_file.csv')  # Uncomment and replace with your file path

# Separate features and target
X = train_data.drop(columns=['Price', 'id'])  # Drop target and ID column
y = train_data['Price']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['number']).columns
categorical_cols = X.select_dtypes(exclude=['number']).columns

# Create preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

# Train and evaluate models
results = {}

for name, model in models.items():
    # Create pipeline with preprocessor and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[name] = {'MAE': mae, 'RMSE': rmse}

# Convert results to DataFrame
results_df = pd.DataFrame(results).T

# Plot comparison
plt.figure(figsize=(10, 5))
sns.barplot(x=results_df.index, y=results_df['RMSE'], palette='viridis')
plt.title("Model Comparison - RMSE")
plt.ylabel("RMSE")
plt.xticks(rotation=30)
plt.show()

# Print results
print(results_df)



test_data=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


test_data.info()




