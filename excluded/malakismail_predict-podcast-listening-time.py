import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_ids = test['id']


def basic_eda(df):
    print("=== First 5 Rows ===")
    display(df.head())
    
    print("\n=== Data Information ===")
    print(df.info())
    
    print("\n=== Descriptive Statistics ===")
    display(df.describe(include='all').T)
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    print("\n=== Unique Values ===")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")


print("Training Data Analysis:")
basic_eda(train)


print("\nTest Data Analysis:")
basic_eda(test)


# Missing Value Analysis
plt.figure(figsize=(12,6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

missing_data = train.isnull().sum().sort_values(ascending=False)
missing_data = missing_data[missing_data > 0]
print("\nMissing Values Summary:")
print(missing_data)


# Target Variable Analysis
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.histplot(train['Listening_Time_minutes'], kde=True)
plt.title('Target Distribution')

plt.subplot(1, 3, 2)
sns.boxplot(x=train['Listening_Time_minutes'])
plt.title('Target Boxplot')

plt.subplot(1, 3, 3)
sns.scatterplot(x=np.arange(len(train)), y=np.sort(train['Listening_Time_minutes']))
plt.title('Target Value Distribution')
plt.tight_layout()
plt.show()


# Categorical Variables Analysis
cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

plt.figure(figsize=(20, 15))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=col, data=train, order=train[col].value_counts().index)
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Numerical Variables Analysis
num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage',
           'Guest_Popularity_percentage', 'Number_of_Ads']

plt.figure(figsize=(20, 15))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(train[col], kde=True)
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()



# Outlier Detection
plt.figure(figsize=(20, 10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=train[col])
    plt.title(f'{col} Boxplot')
plt.tight_layout()
plt.show()


# Time-related Analysis
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Publication_Day', y='Listening_Time_minutes', data=train)
plt.title('Listening Time by Day of Week')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.boxplot(x='Publication_Time', y='Listening_Time_minutes', data=train)
plt.title('Listening Time by Time of Day')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Correlation Analysis
corr_matrix = train.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# Pairplot for Numerical Features
sns.pairplot(train[num_cols + ['Listening_Time_minutes']])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()


# Genre Analysis
genre_analysis = train.groupby('Genre')['Listening_Time_minutes'].agg(['mean', 'median', 'count'])
genre_analysis = genre_analysis.sort_values('mean', ascending=False)
print("\nGenre Analysis:")
display(genre_analysis)

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_analysis.index, y='mean', data=genre_analysis)
plt.title('Average Listening Time by Genre')
plt.xticks(rotation=45)
plt.ylabel('Mean Listening Time (minutes)')
plt.show()


# Sentiment Analysis
sentiment_analysis = train.groupby('Episode_Sentiment')['Listening_Time_minutes'].agg(['mean', 'median', 'count'])
print("\nSentiment Analysis:")
display(sentiment_analysis)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Episode_Sentiment', y='Listening_Time_minutes', data=train)
plt.title('Listening Time Distribution by Sentiment')
plt.show()


# Publication Time Analysis
time_analysis = train.groupby('Publication_Time')['Listening_Time_minutes'].agg(['mean', 'median', 'count'])
print("\nPublication Time Analysis:")
display(time_analysis)


# Host Popularity Analysis
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Host_Popularity_percentage', y='Listening_Time_minutes', data=train)
plt.title('Listening Time vs Host Popularity')
plt.show()


'''# Advanced Visualization (Interactive)
fig = px.scatter_3d(train, x='Host_Popularity_percentage', 
                   y='Guest_Popularity_percentage', 
                   z='Listening_Time_minutes',
                   color='Genre')
fig.update_layout(title='3D Relationship: Host vs Guest Popularity vs Listening Time')
fig.show()'''


# Categorical Features Analysis
cat_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
plt.figure(figsize=(20, 15))
for i, feature in enumerate(cat_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=feature, y='Listening_Time_minutes', data=train)
    plt.title(f'Listening Time by {feature}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def preprocess_data(df):
    # Handle missing values
    df = df.dropna(subset=['Number_of_Ads'])
    
    # Impute numerical features
    num_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                   'Guest_Popularity_percentage']
    for col in num_features:
        df[col] = df[col].fillna(df[col].median())
    
    # Handle guest popularity
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].clip(upper=100)
    df['Guest_Pop_missing'] = df['Guest_Popularity_percentage'].isnull().astype(int)
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(0)
    
    # Feature engineering
    df['Host_Guest_Ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1)
    df['Publication_Time'] = df['Publication_Time'].map({'Night':0, 'Morning':1, 'Afternoon':2, 'Evening':3})
    
    return df


train = preprocess_data(train)
test = preprocess_data(test)


# Prepare features and target
X = train.drop(['id', 'Podcast_Name', 'Episode_Title', 'Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']
X_test = test.drop(['id', 'Podcast_Name', 'Episode_Title'], axis=1)


# Define preprocessing
numeric_features = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                   'Guest_Popularity_percentage', 'Number_of_Ads', 'Host_Guest_Ratio']
categorical_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1))
])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Validation predictions
val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f'Validation RMSE: {rmse:.4f}')


feature_names = numeric_features + list(model.named_steps['preprocessor']
                                       .named_transformers_['cat']
                                       .get_feature_names_out(categorical_features))

importances = model.named_steps['regressor'].feature_importances_

plt.figure(figsize=(12, 8))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances')
plt.show()


test_pred = model.predict(X_test)
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

