import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

import lightgbm as lgb

import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = PerformanceWarning)
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
train_df.shape, test_df.shape


train_df.info()


null_counts = train_df.isnull().sum()
total_rows = len(train_df)
null_percentages = (null_counts / total_rows) * 100
print("Percentage of Null Values per Column:")
print(null_percentages.sort_values(ascending=False))


train_df.describe().T


print(train_df['Personality'].value_counts())
train_df['Personality'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])


sns.boxplot(x='Personality', y='Time_spent_Alone', data=train_df)
plt.show()


numerical_features = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

sns.pairplot(train_df, hue='Personality', vars=numerical_features, corner=True)
plt.show()


# Create a count plot for 'Stage_fear' colored by 'Personality'
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Stage_fear', hue='Personality')
plt.title('Stage Fear Distribution by Personality')
plt.show()


# --- 1. Define Features (X) ---
# We'll use the original training data and drop the target and id
X = train_df.drop(['id', 'Personality'], axis=1)

# Identify numerical and categorical feature names
numerical_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include='object').columns


# --- 2. Define Preprocessing Steps ---
# Define transformers for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor to apply the transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# --- 3. Apply Preprocessing ---
# Fit the preprocessor to the data and transform it, creating our final array
print("Applying preprocessing to the feature set...")
X_processed = preprocessor.fit_transform(X)

print("Preprocessing complete!")
print(f"Shape of processed data: {X_processed.shape}")


# Initialize and apply t-SNE to reduce the data to 2 components
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_processed)

# Create a new DataFrame with the t-SNE components and the personality labels
tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE Component 1', 'TSNE Component 2'])
tsne_df['Personality'] = train_df['Personality']

# Create the cluster visualization using seaborn
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='TSNE Component 1', y='TSNE Component 2',
    hue='Personality',
    data=tsne_df,
    alpha=0.7,
    s=50
)

plt.title('t-SNE Visualization of Personality Clusters')
plt.show()


numerical_df = train_df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


y_raw = train_df['Personality']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))
])

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring="accuracy", n_jobs=-1)

print(f"Cross-Validation Scores: {cv_scores}")
print("-" * 35)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")


# 2. Train the final model on ALL the training data
print("Training the final model on all data...")
model.fit(X, y)

print("Generating predictions on the test set...")
test_predictions_encoded = model.predict(test_df)

# 4. Create and save the submission file
# Decode predictions from 0/1 back to 'Introvert'/'Extrovert'
final_predictions = label_encoder.inverse_transform(test_predictions_encoded)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Personality': final_predictions
})

# Save the file
submission_df.to_csv('submission.csv', index=False)

print("\nFinal submission file 'submission.csv' created successfully!")
print(submission_df.head())

