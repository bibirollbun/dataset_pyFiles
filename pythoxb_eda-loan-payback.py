import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df


test =pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv') 
test


df.info()


df.describe(include='all')


num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

print("Numerical Columns:", num_cols.tolist())
print("Categorical Columns:", cat_cols.tolist())



for col in num_cols:
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    
    plt.subplot(1,2,2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    
    plt.show()



for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.show()



# Scatterplots between key numeric pairs
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for i in range(len(num_cols)):
    for j in range(i+1, len(num_cols)):
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[num_cols[i]], y=df[num_cols[j]])
        plt.title(f'{num_cols[i]} vs {num_cols[j]}')
        plt.show()



# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Numerical Variables)')
plt.show()



cat_cols = df.select_dtypes(include=['object']).columns

for cat in cat_cols:
    for num in num_cols:
        plt.figure(figsize=(8,4))
        sns.boxplot(x=cat, y=num, data=df)
        plt.title(f'{num} vs {cat}')
        plt.xticks(rotation=45)
        plt.show()



# Categorical vs Categorical (corrected)
for i in range(len(cat_cols)):
    for j in range(i+1, len(cat_cols)):
        ct = pd.crosstab(df[cat_cols[i]], df[cat_cols[j]])
        plt.figure(figsize=(8,5))
        sns.heatmap(ct, annot=False, cmap='YlGnBu')
        plt.title(f'{cat_cols[i]} vs {cat_cols[j]}')
        plt.xlabel(cat_cols[j])
        plt.ylabel(cat_cols[i])
        plt.show()



sns.pairplot(df[num_cols], diag_kind='kde')
plt.suptitle('Pairplot of All Numeric Variables', y=1.02)
plt.show()



plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='annual_income',
    y='loan_amount',
    hue='grade_subgrade',
    alpha=0.7
)
plt.title('Annual Income vs Loan Amount by Grade')
plt.show()



plt.figure(figsize=(10,5))
sns.boxplot(
    data=df,
    x='education_level',
    y='loan_amount',
    hue='marital_status'
)
plt.title('Loan Amount by Education Level and Marital Status')
plt.xticks(rotation=45)
plt.show()



from pandas.plotting import parallel_coordinates

subset_cols = ['grade_subgrade', 'annual_income', 'credit_score', 'loan_amount', 'interest_rate']
plt.figure(figsize=(12,6))
parallel_coordinates(df[subset_cols].sample(500, random_state=42), 'grade_subgrade', colormap='viridis')
plt.title('Parallel Coordinates Plot - Multivariate View')
plt.xticks(rotation=30)
plt.show()



X = df.drop(columns=['loan_paid_back'])  
y = df['loan_paid_back']




num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

print("Numerical Columns:", num_cols.tolist())
print("Categorical Columns:", cat_cols.tolist())



# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

from sklearn.preprocessing import OrdinalEncoder

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])



# LightGBM Classifier pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])


# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit model
model.fit(X_train, y_train)


y_pred = model.predict(X_valid)

r2 = r2_score(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

print(f"RÂ² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")



# Predict probabilities or class labels
test_pred = model.predict(test)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'loan_default': test_pred
})

# Preview first 5 rows
print(submission.head())

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv saved successfully")



import joblib
joblib.dump(model, 'loan_default_model.pkl')


