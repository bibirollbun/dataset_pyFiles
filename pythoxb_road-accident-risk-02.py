import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col='id')
df


df.isnull().sum()


df.describe()



# Separate features and target
X = df.drop(['num_reported_accidents', 'accident_risk'], axis=1)
y = df['num_reported_accidents']



# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns



# Preprocessor: one-hot encode categorical + scale numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],remainder='passthrough'
    
)


# Model: Random Forest (better for large and complex data)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=150,       # number of trees
        max_depth=None,         # let trees grow fully
        random_state=42,
        n_jobs=-1               # use all CPU cores for faster training
    ))
])



# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train the model
model.fit(X_train, y_train)





# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Random Forest Model Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")



# --- Load your test dataset ---
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')

# --- Generate predictions using the trained pipeline ---
test_predictions = model.predict(test_df)

# --- Prepare submission DataFrame ---
submission = pd.DataFrame({
    'id': test_df.index,
    'num_reported_accidents': test_predictions
})

# --- Save to CSV file ---
submission.to_csv('submission.csv', index=False)

print("✅ Submission file created successfully: 'submission.csv'")
submission.head()





