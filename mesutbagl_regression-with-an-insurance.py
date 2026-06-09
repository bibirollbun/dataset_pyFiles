import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import warnings
import pickle
# Suppress warnings
warnings.filterwarnings('ignore')


train_df=pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


train_df.head(10)


test_df.head()


train_df=train_df.drop('id', axis=1)
test_id=test_df['id']
test_df=test_df.drop('id', axis=1)


train_df.info()


train_df.describe()


train_df.shape


train_df.isnull().sum()


train_df['Income_per_Dependent'] = train_df['Annual Income'] / (train_df['Number of Dependents'] + 1)
test_df['Income_per_Dependent'] = test_df['Annual Income'] / (test_df['Number of Dependents'] + 1)


# Convert the date column to datetime format
train_df['Policy Start Date'] = pd.to_datetime(train_df['Policy Start Date'])

# Extract year, month, and day
train_df['year'] = train_df['Policy Start Date'].dt.year


# Convert the date column to datetime format
test_df['Policy Start Date'] = pd.to_datetime(test_df['Policy Start Date'])

# Extract year, month, and day
test_df['year'] = test_df['Policy Start Date'].dt.year


train_df=train_df.drop(columns=['Policy Start Date','Education Level', 'Location', 'Marital Status','Customer Feedback','Property Type'], axis=1)
test_df=test_df.drop(columns=['Policy Start Date','Education Level', 'Location', 'Marital Status','Customer Feedback','Property Type'], axis=1)


train_df.head()


imputer = SimpleImputer(strategy='median')


# List of categorical and numerical columns
categorical_cols = ['Occupation']
numerical_cols = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Credit Score',
                 'Insurance Duration', 'Vehicle Age', 'Previous Claims', 'Income_per_Dependent']  
# Create imputers
categorical_imputer = SimpleImputer(strategy='most_frequent')
numerical_imputer = SimpleImputer(strategy='median')

# Fit and transform categorical columns
train_df[categorical_cols] = categorical_imputer.fit_transform(train_df[categorical_cols])
test_df[categorical_cols] = categorical_imputer.transform(test_df[categorical_cols])

# Fit and transform numerical columns
train_df[numerical_cols] = numerical_imputer.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = numerical_imputer.transform(test_df[numerical_cols])


train_df['Policy Type'] = train_df['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium':2})
test_df['Policy Type'] = test_df['Policy Type'].map({'Basic': 0, 'Comprehensive': 1, 'Premium':2})


train_df['Smoking Status'] = train_df['Smoking Status'].map({'Yes': 1, 'No': 0})
test_df['Smoking Status'] = test_df['Smoking Status'].map({'Yes': 1, 'No': 0})


plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Participants', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Premium Amount', y='Occupation', data=train_df, palette='Set2')
plt.title('Occupation vs Premium Amount', fontsize=16)
plt.xlabel('Premium Amount', fontsize=12)
plt.ylabel('Occupation', fontsize=12)
plt.grid()
plt.show()


plt.figure(figsize=(12, 8))
correlation_matrix = train_df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()


# Group by Year and sum the Premium Amounts
yearly_distribution = train_df.groupby('year')['Premium Amount'].sum().reset_index()

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(yearly_distribution['year'], yearly_distribution['Premium Amount'], color='skyblue')
plt.title('Yearly Distribution of Premium Amounts')
plt.xlabel('Year')
plt.ylabel('Total Premium Amount')
plt.xticks(yearly_distribution['year'])
plt.grid(axis='y')

# Show the plot
plt.tight_layout()
plt.show()


train_df['Age'] = train_df['Age'].astype('float32')
train_df['Number of Dependents'] = train_df['Number of Dependents'].astype('float32')
train_df['Health Score'] = train_df['Health Score'].astype('float32')
train_df['Previous Claims'] = train_df['Previous Claims'].astype('float32')
train_df['Vehicle Age'] = train_df['Vehicle Age'].astype('float32')
train_df['Credit Score'] = train_df['Credit Score'].astype('float32')
train_df['Insurance Duration'] = train_df['Insurance Duration'].astype('float32')


# Features and Target
X = train_df.drop("Premium Amount", axis=1)
y = np.log(train_df["Premium Amount"])


# Preprocessing
numeric_features = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score',
       'Policy Type', 'Previous Claims', 'Vehicle Age', 'Credit Score',
       'Insurance Duration', 'Smoking Status', 'Income_per_Dependent', 'year']
categorical_features = train_df.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ))
])


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Fit Model
pipeline.fit(X_train, y_train)


# Evaluation
y_pred = pipeline.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)


from xgboost import plot_importance
plot_importance(pipeline.named_steps['regressor'])
plt.show()




