# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df.head()


df.info()


# Checking for missing values
df.isnull().sum()


# Statistical Summary
df.describe()


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(data=df, x='Age', kde=True, bins=12, ax=axes[0])
sns.boxplot(data=df, x='Age', ax=axes[1])


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(data=df, x='Duration', kde=True, bins=30, ax=axes[0])
sns.boxplot(data=df, x='Duration', ax=axes[1])


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(data=df, x='Heart_Rate', kde=True, bins=12, ax=axes[0])
sns.boxplot(data=df, x='Heart_Rate', ax=axes[1])


sns.countplot(data=df, x='Sex')


sns.barplot(data=df, x='Sex', y='Calories')


sns.violinplot(x='Sex', y='Heart_Rate', data=df)
plt.title("Heart Rate Distribution by Sex")
plt.show()


sns.lineplot(x='Age', y='Heart_Rate', data=df.groupby('Age')['Heart_Rate'].mean().reset_index())
plt.title("Average Heart Rate by Age")
plt.show()


sns.lmplot(x='Duration', y='Calories', data=df, scatter_kws={'alpha':0.2}, line_kws={"color": "red"})
plt.title("Duration vs Calories Burned")
plt.show()


sns.lmplot(x='Heart_Rate', y='Calories', data=df, scatter_kws={'alpha':0.2}, line_kws={"color": "red"})
plt.title("Heart Rate vs Calories Burned")
plt.show()


corr = df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


def feature_engg(df):
    df_engg = df.copy()

    df_engg['Age'] = df_engg['Age'].astype('int32')
    df_engg['Height'] = df_engg['Height'].astype('float32')  
    df_engg['Weight'] = df_engg['Weight'].astype('float32')
    df_engg['Duration'] = df_engg['Duration'].astype('float32')
    df_engg['Heart_Rate'] = df_engg['Heart_Rate'].astype('float32')
    df_engg['Body_Temp'] = df_engg['Body_Temp'].astype('float32')

    if 'Calories' in df_engg.columns:
        df_engg['Calories'] = df_engg['Calories'].astype('float32')

    height_m = df_engg['Height'] / 100.0

    # BMR (Basal Metabolic Rate) Calculation
    # For men: (9.65 Ã— weight in kg) + (573 Ã— height in m) â€“ (5.08 Ã— age in years) + 260
    # For women: (7.38 Ã— weight in kg) + (607 Ã— height in m) â€“ (2.31 Ã— age in years) + 43

    bmr_male = (9.65 * df_engg['Weight']) + (573 * height_m) - (5.08 * df_engg['Age']) + 260
    bmr_female = (7.38 * df_engg['Weight']) + (607 * height_m) - (2.31 * df_engg['Age']) + 43

    df_engg['BMR'] = np.where(df_engg['Sex'] == 'male', bmr_male, bmr_female)
    
    df_engg['BMI'] = df_engg['Weight'] / (height_m ** 2)
    
    df_engg['Activity_Intensity_Proxy'] = df_engg['Heart_Rate'] * df_engg['Duration']

    return df_engg


df_train_processed = feature_engg(df)


X = df_train_processed.drop(columns=['id', 'Calories'])
y = df_train_processed["Calories"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.head()


from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error, r2_score

numeric_features = X.select_dtypes(include=['int32', 'float32']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', PowerTransformer(method='box-cox'), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42
)

# Create pipeline
xgb_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('scaler', StandardScaler()),
    ('model', xgb_model)
])

# Fit model
print("Training XGBoost model...")
xgb_pipeline.fit(X_train, y_train)

y_pred = xgb_pipeline.predict(X_test)
rmsle_val = np.sqrt(mean_squared_log_error(y_test, y_pred))
r2_val = r2_score(y_test, y_pred)
print(f"Validation RMSLE: {rmsle_val:.4f}")
print(f"Validation RÂ²: {r2_val:.4f}")


feature_names = []
for name, transformer, columns in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(columns)
    elif name == 'cat':
        cats = transformer.fit_transform(X_train[columns])
        feature_names.extend([f"{col}_{i}" for col in columns for i in range(cats.shape[1])])

# Print top features if dimensionality matches
if hasattr(xgb_pipeline.named_steps['model'], 'feature_importances_'):
    importances = xgb_pipeline.named_steps['model'].feature_importances_
    if len(feature_names) == len(importances):
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        print("\nTop 10 features:")
        print(feature_imp.head(10))


df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test_id = df_test["id"].copy()

df_test_processed = feature_engg(df_test)

test_preds = xgb_pipeline.predict(df_test_processed)
test_preds_clipped = np.maximum(test_preds, 1)
submission_df = pd.DataFrame({
    "id": test_id,
    "Calories": test_preds_clipped
})

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())

