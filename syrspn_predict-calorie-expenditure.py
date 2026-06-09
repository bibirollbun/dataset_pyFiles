import pandas as pd

df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df.head()


df.info()


df.isna().sum()


df.duplicated().sum()


df.describe()


import seaborn as sns
import matplotlib.pyplot as plt

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(20, 8))

for i, col in enumerate(numeric_cols[:8], 1):  
    plt.subplot(2, 4, i)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribusi: {col}')
    plt.xlabel(col)
    plt.xticks(rotation=30)
    plt.ylabel('Frekuensi')

plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 8))

for i, col in enumerate(numeric_cols[:8], 1):
    plt.subplot(2, 4, i)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot: {col}')
    plt.xlabel(col)
    plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


df.describe(include='object')


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
ax = sns.countplot(x='Sex', data=df, palette='pastel')
plt.title('Frekuensi Kategori "Sex"')
plt.xlabel('Sex')
plt.ylabel('Jumlah')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom')

plt.tight_layout()
plt.show()


corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Matriks Korelasi Antar Variabel Numerik')
plt.show()


correlation_with_calories = df[numeric_cols].corr()['Calories'].drop('Calories')
correlation_sorted = correlation_with_calories.sort_values(ascending=False)

print("Korelasi setiap fitur dengan kolom 'Calories':")
correlation_sorted


correlation_sorted_desc = correlation_sorted.sort_values(ascending=True)

plt.figure(figsize=(16, 8))
ax = correlation_sorted_desc.plot(kind='barh', color='skyblue')
plt.title("Korelasi Fitur Numerik dengan Kolom 'Calories'")
plt.xlabel("Korelasi Pearson")
plt.ylabel("Fitur")
plt.grid(axis='x', linestyle='--', alpha=0.7)

for p in ax.patches:
    ax.annotate(f'{p.get_width():.3f}', 
                (p.get_width() + 0.01 * (1 if p.get_width() > 0 else -1), p.get_y() + p.get_height() / 2),
                ha='left' if p.get_width() > 0 else 'right',
                va='center',
                fontsize=9)

plt.tight_layout()
plt.show()


df_cleaned = pd.get_dummies(df, columns=['Sex'], dtype=int)
desired_order = [
    'id', 'Age', 'Sex_female', 'Sex_male', 'Height', 'Weight',
    'Duration', 'Heart_Rate', 'Body_Temp', 'Calories'
]
df_cleaned = df_cleaned.rename(columns={'Sex_female': 'Sex_female', 'Sex_male': 'Sex_male'})
df_cleaned = df_cleaned[desired_order]
df_cleaned.head()


df_cleaned['BMI'] = df_cleaned['Weight'] / (df_cleaned['Height'] / 100) ** 2
df_cleaned['BMI'] = df_cleaned['BMI'].round(1)
df_cleaned = df_cleaned.drop(columns=['Weight', 'Height'])
desired_order = [
    'id', 'Age', 'Sex_female', 'Sex_male', 'BMI',
    'Duration', 'Heart_Rate', 'Body_Temp', 'Calories'
]
df_cleaned = df_cleaned.rename(columns={'Sex_female': 'Sex_female', 'Sex_male': 'Sex_male'})
df_cleaned = df_cleaned[desired_order]
df_cleaned.head()


from sklearn.preprocessing import MinMaxScaler

numerical_features = ['Age', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp']
scaler = MinMaxScaler()
df_cleaned[numerical_features] = scaler.fit_transform(df_cleaned[numerical_features])

df_cleaned.head()


numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
correlation_with_calories = df_cleaned[numeric_cols].corr()['Calories'].drop('Calories')
correlation_sorted = correlation_with_calories.sort_values(ascending=False)

print("Korelasi setiap fitur dengan kolom 'Calories':")
correlation_sorted


correlation_sorted_desc = correlation_sorted.sort_values(ascending=True)

plt.figure(figsize=(16, 8))
ax = correlation_sorted_desc.plot(kind='barh', color='skyblue')
plt.title("Korelasi Fitur Numerik dengan Kolom 'Calories'")
plt.xlabel("Korelasi Pearson")
plt.ylabel("Fitur")
plt.grid(axis='x', linestyle='--', alpha=0.7)

for p in ax.patches:
    ax.annotate(f'{p.get_width():.3f}', 
                (p.get_width() + 0.01 * (1 if p.get_width() > 0 else -1), p.get_y() + p.get_height() / 2),
                ha='left' if p.get_width() > 0 else 'right',
                va='center',
                fontsize=9)

plt.tight_layout()
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


target = "Calories"
X = df_cleaned[["Duration", "Heart_Rate", "Body_Temp", "Age", "BMI", "Sex_male"]]
y = df_cleaned[target]


from sklearn.model_selection import train_test_split
# Split data ke train test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)


# Print coefficients
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

# Predict
y_pred = model.predict(X_test)


import numpy as np
# y_train and y_pred are both 1D arrays of length N
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.2, label='Predicted vs. Actual')
# draw ideal y=x line
lims = [
    np.min([y_train.min(), y_pred.min()]),
    np.max([y_train.max(), y_pred.max()]),
]
plt.plot(lims, lims, 'r--', label='Ideal (y = x)')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Predicted vs. Actual Calories')
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_log_error
# Ensure y_pred is a Series for compatibility
y_pred_series = pd.Series(y_pred, index=y_test.index)

# Use correct parameters depending on data type
y_test_clipped = y_test.clip(lower=0)  # pandas Series
y_pred_clipped = y_pred_series.clip(lower=0)  # also pandas Series now

# Now compute MSLE
msle = mean_squared_log_error(y_test_clipped, y_pred_clipped)
print("MSLE:", msle)
print("RMSLE:", msle**0.5)


# Extract feature and target arrays
X, y = df_cleaned[["Duration", "Heart_Rate", "Body_Temp", "Age", "BMI", "Sex_male"]], df_cleaned[['Calories']]


from sklearn.model_selection import train_test_split
# data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import xgboost as xgb
# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)


# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "hist"}


n = 100
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
modelxgb = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals
)


predicted = modelxgb.predict(dtest_reg)
actual = y_test.values.flatten() 


from sklearn.metrics import mean_squared_log_error
actual = actual.clip(min=0)
predicted = predicted.clip(min=0)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(actual, predicted))
print(f"RMSLE: {rmsle:.4f}")


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline


target = "Calories"
X = df_cleaned[["Duration", "Heart_Rate", "Body_Temp", "Age", "BMI", "Sex_male"]]
y = df_cleaned[target]

print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# Initialize the model
rf_model = RandomForestRegressor(
    n_estimators=100,       # Number of trees
    max_depth=None,         # No limit on tree depth
    random_state=42
)



# Train the model
rf_model.fit(X_train, y_train)


# Predict
y_pred = rf_model.predict(X_test)



# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Regression Performance:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"RÂ²   = {r2:.2f}")


# Memastikan tidak ada data negatid
y_test_clip = np.clip(y_test, a_min=0, a_max=None)
y_pred_clip = np.clip(y_pred, a_min=0, a_max=None)

# Hitung MSLE and RMSLE
msle = mean_squared_log_error(y_test_clip, y_pred_clip)
rmsle = np.sqrt(msle)

print(f"MSLE  = {msle:.4f}")
print(f"RMSLE = {rmsle:.4f}")


df_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Convert to Pandas category
#for col in cats2:
#   X_new[col] = X[col].astype('category')


df_submission = pd.get_dummies(df_submission, columns=['Sex'], dtype=int)
desired_order = [
    'id', 'Age', 'Sex_female', 'Sex_male', 'Height', 'Weight',
    'Duration', 'Heart_Rate', 'Body_Temp'
]
df_submission = df_submission.rename(columns={'Sex_female': 'Sex_female', 'Sex_male': 'Sex_male'})
df_submission = df_submission[desired_order]
df_submission.head()


df_submission['BMI'] = df_submission['Weight'] / (df_submission['Height'] / 100) ** 2
df_submission['BMI'] = df_submission['BMI'].round(1)
df_submission = df_submission.drop(columns=['Weight', 'Height'])
desired_order = [
    'id', 'Age', 'Sex_female', 'Sex_male', 'BMI',
    'Duration', 'Heart_Rate', 'Body_Temp'
]
df_submission = df_submission.rename(columns={'Sex_female': 'Sex_female', 'Sex_male': 'Sex_male'})
df_submission = df_submission[desired_order]
df_submission.head()


df_submission


#coba ke data testing
X_new = df_submission[X_train.columns]  # Ensure same columns





dnew = xgb.DMatrix(X_new, enable_categorical=True)
# Use the already-trained model to predict
predictions = modelxgb.predict(dnew)

print(predictions)


# Save predictions
output = pd.DataFrame(predictions, columns=["Predicted"])
submission = pd.DataFrame({
    "id": df_submission["id"],          
    "Calories": predictions       
})

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")


len(df_submission)


labels = ['Simple Linear Regression', 'XGBoost', 'RandomForestRegressor'] 
rmsle = [0.5645201516323676, 0.0686, 0.0691] 


plt.figure(figsize=(10, 6))
bars = plt.bar(labels, rmsle, color=['skyblue', 'lightgreen', 'orange'], edgecolor='black')

# Annotate each bar with its score
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Styling
plt.title("Model Performance Comparison (RMSLE)", fontsize=14)
plt.ylabel("RMSLE")
plt.xlabel("Model")
plt.ylim(0, max(scores) + 0.1)  # Add margin above highest bar
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# Extract feature and target arrays
X, y = df_cleaned[["Duration", "Heart_Rate", "Body_Temp"]], df_cleaned[['Calories']]


# data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)


# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "hist"}


n = 100
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
modelxgb = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals
)


predicted = modelxgb.predict(dtest_reg)
actual = y_test.values.flatten() 


actual = actual.clip(min=0)
predicted = predicted.clip(min=0)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(actual, predicted))
print(f"RMSLE: {rmsle:.4f}")


from sklearn.model_selection import GridSearchCV


# Extract feature and target arrays
X, y = df_cleaned[["Duration", "Heart_Rate", "Body_Temp", "Age", "BMI", "Sex_male"]], df_cleaned[['Calories']]


# data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)


# Define hyperparameters
#params = {"objective": "reg:squarederror", "tree_method": "hist"}
seed = 42
params = {
    # Core parameters
    'booster': 'gbtree',               # Standard tree-based booster
    'tree_method': 'hist',             # Fast and memory-efficient histogram-based split
    'objective': 'reg:squarederror',   # For regression tasks
    'eval_metric': 'rmse',             # Root Mean Square Error

    # Regularization (helps prevent overfitting)
    'lambda': 1.0,                     # L2 regularization term on weights (default)
    'alpha': 0.0,                      # L1 regularization term on weights (default)

    # Tree growth and model complexity
    'max_depth': 6,                    # Reasonable default (not too shallow or deep)
    'min_child_weight': 1,             # Default, allows smaller splits
    'gamma': 0,                        # No minimum loss reduction for splits

    # Subsampling for robustness and speed
    'subsample': 1.0,                  # Use all rows per tree (set <1.0 if overfitting)
    'colsample_bytree': 1.0,          # Use all features (adjust if overfitting)

    # Learning rate and iterations
    'learning_rate': 0.1,              # Default rate; reduce for slower, safer learning

    # Misc
    'seed': 42,                        # Reproducibility
    'verbosity': 1,                    # Show info messages
    'device': 'cuda'                   # Use GPU if available
}


n = 1000
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

modelxgb = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=100
)


predicted = modelxgb.predict(dtest_reg)
actual = y_test.values.flatten() 


actual = actual.clip(min=0)
predicted = predicted.clip(min=0)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(actual, predicted))
print(f"RMSLE: {rmsle:.4f}")


from itertools import product
from sklearn.metrics import mean_squared_error
# Base params
base_params = {
    'booster': 'gbtree',
    'tree_method': 'hist',
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': 42,
    'device': 'cuda',
    'verbosity': 1
}



# Param grid
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 1.0],
    'lambda': [1, 2],
    'alpha': [0, 0.5]
}


# combinasi parameter
all_combos = list(product(
    param_grid['max_depth'],
    param_grid['learning_rate'],
    param_grid['subsample'],
    param_grid['colsample_bytree'],
    param_grid['lambda'],
    param_grid['alpha']
))


# menampung skor
best_score = float('inf')
best_params = None


# Grid search
for i, (md, lr, ss, cs, lam, alp) in enumerate(all_combos):
    params = base_params.copy()
    params.update({
        'max_depth': md,
        'learning_rate': lr,
        'subsample': ss,
        'colsample_bytree': cs,
        'lambda': lam,
        'alpha': alp
    })

    print(f"\n[{i+1}/{len(all_combos)}] Testing: {params}")

    model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=5000,
        evals=[(dtrain_reg, "train"), (dtest_reg, "val")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    preds = model.predict(dtest_reg)
    score = np.sqrt(mean_squared_error(y_test, preds))   # RMSE

    print(f"--> RMSE: {score:.4f}")

    if score < best_score:
        best_score = score
        best_params = params.copy()

print("\nâœ… Best RMSE:", round(best_score, 4))
print("ðŸ“Œ Best Params:", best_params)


# Parameter terbaik
params = {
    'booster': 'gbtree',
    'tree_method': 'hist',
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': 42,
    'device': 'cuda',
    'verbosity': 1,
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'lambda': 1,
    'alpha': 0.5
}



evals = [(dtrain_reg, 'train'), (dtest_reg, 'validation')]


# Train model dengan early stopping
model1 = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=5000,            
    evals=evals,
    early_stopping_rounds=50,        # berhenti jika tidak ada peningkatan performa dalam 50 tree
    verbose_eval=200                 # setiap 200 tree, print metrics
)


# Hitung rmse
preds = model1.predict(dtest_reg)
score = np.sqrt(mean_squared_error(y_test, preds))
print(f"Final RMSE: {score:.4f}")


predicted = model1.predict(dtest_reg)
actual = y_test.values.flatten() 


actual = actual.clip(min=0)
predicted = predicted.clip(min=0)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(actual, predicted))
print(f"RMSLE: {rmsle:.4f}")


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error


df2 = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df2.head()


#Sebelum feature engineering
correlation_sorted_desc = correlation_sorted.sort_values(ascending=True)

plt.figure(figsize=(16, 8))
ax = correlation_sorted_desc.plot(kind='barh', color='skyblue')
plt.title("Korelasi Fitur Numerik dengan Kolom 'Calories'")
plt.xlabel("Korelasi Pearson")
plt.ylabel("Fitur")
plt.grid(axis='x', linestyle='--', alpha=0.7)

for p in ax.patches:
    ax.annotate(f'{p.get_width():.3f}', 
                (p.get_width() + 0.01 * (1 if p.get_width() > 0 else -1), p.get_y() + p.get_height() / 2),
                ha='left' if p.get_width() > 0 else 'right',
                va='center',
                fontsize=9)

plt.tight_layout()
plt.show()


# Step 0: Create Sex dummies first
df_cleaned2 = pd.get_dummies(df2, columns=['Sex'], dtype=int)

# Step 1: Feature engineering
df_cleaned2['BMI'] = df_cleaned2['Weight'] / (df_cleaned2['Height'] / 100) ** 2
df_cleaned2['Height_m'] = df_cleaned2['Height'] / 100
df_cleaned2['BMI'] = df_cleaned2['BMI'].round(1)

df_cleaned2['Age_bin'] = pd.cut(df_cleaned2['Age'], bins=[0,18,35,50,65,100], labels=False)

df_cleaned2['HR_category'] = pd.cut(df_cleaned2['Heart_Rate'], bins=[0,60,100,200], labels=['low','normal','high'])
df_cleaned2 = pd.get_dummies(df_cleaned2, columns=['HR_category'], drop_first=True)


df_cleaned2['Temp_diff'] = df_cleaned2['Body_Temp'] - 37

# Step 2: Define final column order
desired_order = [
    'id', 'Age', 'Sex_male', 'Height', 'Weight', 'BMI', 'Height_m',
    'Age_bin', 'HR_category_normal', 'HR_category_high', 'Temp_diff',
    'Duration', 'Heart_Rate', 'Body_Temp', 'Calories'
]

# Step 3: Reorder columns and display
df_cleaned2 = df_cleaned2[desired_order]

df_cleaned2.head()



#Setelah Feature Engineering
numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
correlation_with_calories = df_cleaned[numeric_cols].corr()['Calories'].drop('Calories')
correlation_sorted = correlation_with_calories.sort_values(ascending=False)

print("Korelasi setiap fitur dengan kolom 'Calories':")
correlation_sorted


correlation_sorted_desc = correlation_sorted.sort_values(ascending=True)

plt.figure(figsize=(16, 8))
ax = correlation_sorted_desc.plot(kind='barh', color='skyblue')
plt.title("Korelasi Fitur Numerik dengan Kolom 'Calories'")
plt.xlabel("Korelasi Pearson")
plt.ylabel("Fitur")
plt.grid(axis='x', linestyle='--', alpha=0.7)

for p in ax.patches:
    ax.annotate(f'{p.get_width():.3f}', 
                (p.get_width() + 0.01 * (1 if p.get_width() > 0 else -1), p.get_y() + p.get_height() / 2),
                ha='left' if p.get_width() > 0 else 'right',
                va='center',
                fontsize=9)

plt.tight_layout()
plt.show()



# Extract feature and target arrays
X, y = df_cleaned.drop(['id','Calories'],axis=1), df_cleaned[['Calories']]


# data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)


# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "hist"}


n = 100
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
modelxgb = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals
)


predicted = modelxgb.predict(dtest_reg)
actual = y_test.values.flatten() 


actual = actual.clip(min=0)
predicted = predicted.clip(min=0)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(actual, predicted))
print(f"RMSLE: {rmsle:.4f}")


# Extract feature and target arrays
X, y = df_cleaned.drop(['id','Calories'],axis=1), df_cleaned[['Calories']]


# data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)


# Define hyperparameters
#params = {"objective": "reg:squarederror", "tree_method": "hist"}
seed = 42
params = {
    # Core parameters
    'booster': 'gbtree',               # Standard tree-based booster
    'tree_method': 'hist',             # Fast and memory-efficient histogram-based split
    'objective': 'reg:squarederror',   # For regression tasks
    'eval_metric': 'rmse',             # Root Mean Square Error

    # Regularization (helps prevent overfitting)
    'lambda': 1.0,                     # L2 regularization term on weights (default)
    'alpha': 0.0,                      # L1 regularization term on weights (default)

    # Tree growth and model complexity
    'max_depth': 6,                    # Reasonable default (not too shallow or deep)
    'min_child_weight': 1,             # Default, allows smaller splits
    'gamma': 0,                        # No minimum loss reduction for splits

    # Subsampling for robustness and speed
    'subsample': 1.0,                  # Use all rows per tree (set <1.0 if overfitting)
    'colsample_bytree': 1.0,          # Use all features (adjust if overfitting)

    # Learning rate and iterations
    'learning_rate': 0.1,              # Default rate; reduce for slower, safer learning

    # Misc
    'seed': 42,                        # Reproducibility
    'verbosity': 1,                    # Show info messages
    'device': 'cuda'                   # Use GPU if available
}


n = 1000
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
evals_result = {}
modelxgb = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals,
    early_stopping_rounds=50,
    evals_result=evals_result, 
    verbose_eval=100
)


predicted = modelxgb.predict(dtest_reg)
actual = y_test.values.flatten() 


actual = actual.clip(min=0)
predicted = predicted.clip(min=0)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(actual, predicted))
print(f"RMSLE: {rmsle:.4f}")


import itertools
# Fixed parameters (constant across runs)
fixed_params = {
    'booster': 'gbtree',
    'tree_method': 'hist',
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': 42,
    'verbosity': 1,
    'device': 'cuda'
}


# Parameter grid for manual search
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.1, 0.01],
    'subsample': [1.0, 0.7],
    'colsample_bytree': [1.0, 0.7],
    'alpha': [0.0, 0.5]
}


n_rounds = 1000
early_stopping = 50
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]


best_rmsle = float('inf')
best_params = None
best_model = None


# Grid search loop
keys, values = zip(*param_grid.items())
for v in itertools.product(*values):
    params = dict(zip(keys, v))
    params.update(fixed_params)

    evals_result = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=n_rounds,
        evals=evals,
        early_stopping_rounds=early_stopping,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    preds = model.predict(dtest_reg).clip(min=0)
    actual = y_test.values.flatten().clip(min=0)
    
    rmsle = np.sqrt(mean_squared_log_error(actual, preds))
    
    print(f"Params: {params} => RMSLE: {rmsle:.4f}")
    
    if rmsle < best_rmsle:
        best_rmsle = rmsle
        best_params = params
        best_model = model

print(f"\nBest RMSLE: {best_rmsle:.4f}")
print(f"Best Params: {best_params}")


# Best parameters
params = {
    'booster': 'gbtree',
    'tree_method': 'hist',
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': 42,
    'device': 'cuda',
    'verbosity': 1,
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 1.0,
    'alpha': 0.5
}


evals = [(dtrain_reg, 'train'), (dtest_reg, 'validation')]


# Train model with early stopping
evals_result = {}
model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=5000,            # Large number, early stopping will prevent overfitting
    evals=evals,
    evals_result=evals_result, 
    early_stopping_rounds=50,        # Stop if no improvement after 50 rounds
    verbose_eval=200                 # Print metrics every 200 rounds
)


from sklearn.metrics import mean_squared_error
# Predict and evaluate
preds = model.predict(dtest_reg)
score = np.sqrt(mean_squared_error(y_test, preds))
print(f"Final RMSE: {score:.4f}")


predicted = model.predict(dtest_reg)
actual = y_test.values.flatten() 


actual = actual.clip(min=0)
predicted = predicted.clip(min=0)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(actual, predicted))
print(f"RMSLE: {rmsle:.4f}")


df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test.head()


# Step 0: Create Sex dummies first
df_test = pd.get_dummies(df_test, columns=['Sex'], dtype=int)

# Step 1: Feature engineering
df_test['BMI'] = df_test['Weight'] / (df_test['Height'] / 100) ** 2
df_test['Height_m'] = df_test['Height'] / 100
df_test['BMI'] = df_test['BMI'].round(1)

df_test['Age_bin'] = pd.cut(df_test['Age'], bins=[0,18,35,50,65,100], labels=False)

df_test['HR_category'] = pd.cut(df_test['Heart_Rate'], bins=[0,60,100,200], labels=['low','normal','high'])
df_test = pd.get_dummies(df_test, columns=['HR_category'], drop_first=True)


df_test['Temp_diff'] = df_test['Body_Temp'] - 37

# Step 2: Define final column order
desired_order = [
    'id', 'Age', 'Sex_male', 'Height', 'Weight', 'BMI', 'Height_m',
    'Age_bin', 'HR_category_normal', 'HR_category_high', 'Temp_diff',
    'Duration', 'Heart_Rate', 'Body_Temp'
]

# Step 3: Reorder columns and display
df_test = df_test[desired_order]

df_test.head()


# Extract feature and target arrays
X_new = df_test
X_new = X_new.drop(columns=['id'])
dnew = xgb.DMatrix(X_new, enable_categorical=True)
predictions = model.predict(dnew)


#coba ke data testing
#X_new = df_test[X_train.columns]  # Ensure same columns
# Extract text features
#cats2 = X_new.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
#for col in cats2:
#   X_new[col] = X[col].astype('category')


dnew = xgb.DMatrix(X_new, enable_categorical=True)
# Use the already-trained model to predict
predictions = model.predict(dnew)

print(predictions)


# Save predictions
output = pd.DataFrame(predictions, columns=["Predicted"])
submission = pd.DataFrame({
    "id": df_test["id"],          
    "Calories": predictions       
})

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

