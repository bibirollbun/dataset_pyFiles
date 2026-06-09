# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")



# Step 1: Load the datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")



print(df_train.info())
print(df_test.info())



!pip install sweetviz


import sweetviz as sv
my_report = sv.analyze([df_train, "All"])
my_report.show_notebook(w="100%", h="full")



# Step 2: Impute missing values (numerical with mean, categorical with most frequent)
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']



# Summary statistics for numeric variables
numeric_summary = df_train[numerical_cols].describe().transpose()
print("Numeric Variables Summary:")
display(numeric_summary)

# Summary statistics for categorical variables
categorical_summary = df_train[categorical_cols].astype('category').describe().transpose()
print("\nCategorical Variables Summary:")
display(categorical_summary)


# Imputation
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

df_train[numerical_cols] = num_imputer.fit_transform(df_train[numerical_cols])
df_test[numerical_cols] = num_imputer.transform(df_test[numerical_cols])

df_train[categorical_cols] = cat_imputer.fit_transform(df_train[categorical_cols])
df_test[categorical_cols] = cat_imputer.transform(df_test[categorical_cols])



# Step 3: Feature Scaling (Numerical Only)
scaler = StandardScaler()
df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])


# Step 4: EDA (Exploratory Data Analysis)

# 4.1 Check missing values after imputation
print("\nMissing values in train dataset:\n", df_train.isnull().sum())
print("\nMissing values in test dataset:\n", df_test.isnull().sum())


# 4.2 Distribution of target variable
plt.figure(figsize=(8,5))
sns.histplot(df_train['Listening_Time_minutes'], bins=30, kde=True, color='blue')
plt.title('Distribution of Listening Time (minutes)')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Frequency')
plt.show()



# 4.3 Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_train[numerical_cols + ['Listening_Time_minutes']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# Step 5: Explanatory Analysis of 'Listening_Time_minutes'

# Group by Genre
genre_group = df_train.groupby('Genre')['Listening_Time_minutes'].mean().sort_values()

plt.figure(figsize=(12,6))
genre_group.plot(kind='barh', color='skyblue')
plt.title('Average Listening Time by Genre')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Genre')
plt.show()


# Step 6: Plot 'Listening_Time_minutes' with numerical and categorical variables

# 6.1 Numerical Variables vs Target
for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df_train[col], y=df_train['Listening_Time_minutes'])
    plt.title(f'Listening Time vs {col}')
    plt.xlabel(col)
    plt.ylabel('Listening Time (minutes)')
    plt.show()


# 6.2 Categorical Variables vs Target
for col in categorical_cols:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df_train[col], y=df_train['Listening_Time_minutes'])
    plt.xticks(rotation=90)
    plt.title(f'Listening Time by {col}')
    plt.xlabel(col)
    plt.ylabel('Listening Time (minutes)')
    plt.show()


# Step 7: Predict 'Listening_Time_minutes' for Test Data

# Preparing data
X = df_train.drop(columns=["Listening_Time_minutes"])
y = df_train["Listening_Time_minutes"]
X_test = df_test.copy()
test_ids = df_test["id"]


# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])


# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
}


# Train/validation split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42)


results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipe.fit(X_train_split, y_train_split)
    y_pred = pipe.predict(X_val_split)
    rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
    results[name] = {"model": pipe, "rmse": rmse}
    print(f"{name} RMSE: {rmse:.4f}")


# Plot Model Comparison
model_names = list(results.keys())
rmse_values = [results[name]["rmse"] for name in model_names]



plt.figure(figsize=(10,6))
bars = plt.bar(model_names, rmse_values, color='lightcoral')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Model Comparison based on RMSE')
plt.xticks(rotation=45)
plt.ylim(0, max(rmse_values) * 1.1)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()


# Best Model
best_model_name = min(results, key=lambda k: results[k]['rmse'])
best_model = results[best_model_name]['model']
print(f"\n✅ Best Model: {best_model_name}")


# Predict on test set
test_predictions = best_model.predict(X_test)


# Prepare submission
sample_submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': test_predictions
})
sample_submission.head(5)



#saving submission file
sample_submission.to_csv("sample_submission.csv", index=False)

