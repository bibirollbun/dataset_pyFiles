import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


train_df=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


train_df.head()


train_df.info()


train_df.describe()


train_df.isna().sum()


train_df.duplicated().sum()


numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
sns.set(style="whitegrid")
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train_df[col], color='skyblue')
    plt.title(f'Boxplot of {col}', fontsize=14)
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


sns.set(style="whitegrid", palette="pastel")
plt.rcParams['figure.figsize'] = (10, 5)

# Exclude 'id' as it's not useful for visualization
features = train_df.drop(columns=['id', 'Calories']).columns

# Univariate analysis: numerical features
numeric_features = train_df.select_dtypes(include=['int64', 'float64']).drop(columns=['id', 'Calories']).columns

for col in numeric_features:
    plt.figure()
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# Univariate analysis: categorical feature (Sex)
plt.figure()
sns.countplot(x='Sex', data=train_df)
plt.title('Count of Sex Categories')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Target Variable: Calories
plt.figure()
sns.histplot(train_df['Calories'], kde=True, bins=50)
plt.title('Distribution of Calories (Target Variable)')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.show()


# Numerical Features vs Target (Calories)
for col in numeric_features:
    plt.figure()
    sns.scatterplot(data=train_df, x=col, y='Calories', alpha=0.3)
    plt.title(f'{col} vs Calories')
    plt.xlabel(col)
    plt.ylabel('Calories')
    plt.show()

# Categorical Feature vs Target (Boxplot)
plt.figure()
sns.boxplot(x='Sex', y='Calories', data=train_df)
plt.title('Calories Burnt by Sex')
plt.xlabel('Sex')
plt.ylabel('Calories')
plt.show()



encoder=LabelEncoder()
train_df['Sex']=encoder.fit_transform(train_df['Sex'])
train_df.head()


corr = train_df.drop(columns=['id', 'Sex']).corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()


X = train_df.drop(columns=['id', 'Calories'])
y = train_df['Calories']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}


results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    results[name] = {"RMSE": rmse, "R2 Score": r2}


results_df = pd.DataFrame(results).T.sort_values(by="RMSE")
print("Validation Performance:")
print(results_df)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_df['Sex'] = encoder.fit_transform(test_df['Sex'])
test_ids = test_df['id']
X_test = test_df.drop(columns=['id'])


best_model = models['XGBoost']
best_model.fit(X, y)
test_preds = best_model.predict(X_test)


# Load sample submission format
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Fill predictions
sample_submission['Calories'] = test_preds

# Save to CSV
sample_submission.to_csv('final_submission.csv', index=False)




