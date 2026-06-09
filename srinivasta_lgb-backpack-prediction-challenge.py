import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from category_encoders import TargetEncoder
from xgboost import XGBRegressor



# Load your data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')
train_ex = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')

# Combine train and train_ex
train = pd.concat([train, train_ex], axis=0)
train.reset_index(drop=True, inplace=True)  # Reset index after concatenation



# Impute missing numerical data with the median values from the TRAIN dataset
num_cols = test.select_dtypes(include=['number']).columns
imputation_value = train[num_cols].median()
train[num_cols] = train[num_cols].fillna(imputation_value)
test[num_cols] = test[num_cols].fillna(imputation_value)



# Impute missing object data with 'None'
obj_cols = train.select_dtypes(include=['object']).columns
train[obj_cols] = train[obj_cols].fillna('None')
test[obj_cols] = test[obj_cols].fillna('None')



# Target Encoding with Cross-Validation and Explicit Type Conversion
features = test.columns.tolist()

for col in features:
    # Create a copy to store encoded values
    train_encoded = train[col].copy()  
    test_encoded = test[col].copy()

    # KFold for cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  

    for train_index, val_index in kf.split(train):
        # Fit_transform on training fold, transform on validation fold
        encoder = TargetEncoder(cols=[col], smoothing=5) 
        encoder.fit(train.iloc[train_index][[col]], train.iloc[train_index]['Price'])
        train_encoded.iloc[val_index] = encoder.transform(train.iloc[val_index][[col]])[col].values

    # Fit on entire training data and transform test data
    encoder = TargetEncoder(cols=[col], smoothing=5)
    encoder.fit(train[[col]], train['Price'])
    test_encoded = encoder.transform(test[[col]])[col].values

    # Update original columns with encoded values AND convert to numeric
    train[col] = pd.to_numeric(train_encoded)  # Explicitly convert to numeric
    test[col] = pd.to_numeric(test_encoded)   # Explicitly convert to numeric




# Model Training
X = train.drop(['Price'], axis=1)
y = train['Price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    device="cpu",
    max_depth=5,
    n_estimators=2000,
    learning_rate=0.015,
    random_state=42,
    eval_metric='rmse',
    objective='reg:squarederror' 
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)




# Prediction and Submission
y_pred = model.predict(test)
submission = pd.DataFrame({'id': test.index, 'Price': y_pred})
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")


# prompt: # Trends in Validation RMSE During Model Training
import matplotlib.pyplot as plt

# Trends in Validation RMSE During Model Training

evals_result = model.evals_result()
min_rmse = min(evals_result['validation_0']['rmse'])
min_index = evals_result['validation_0']['rmse'].index(min_rmse)

plt.figure(figsize=(10, 5))
plt.plot(evals_result['validation_0']['rmse'], label='Validation RMSE', color='blue')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('Trends in Validation RMSE During Model Training')
plt.scatter(min_index, min_rmse, color='red', s=50)
plt.text(min_index+50, min_rmse+0.02, f'Validation RMSE(min): {min_rmse:.3f}', color='red', fontsize=11, ha='right')
plt.legend()
plt.grid(True)
plt.show()

# Displaying Minimum Validation RMSE

print(f"Validation RMSE(min): {min_rmse:.4f}")



display(submission)

