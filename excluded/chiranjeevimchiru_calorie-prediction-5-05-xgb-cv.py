import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error, make_scorer



df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print(df_train.info())
print("\n")
print(df_train.shape)


print(df_test.info())
print("\n")
print(df_test.shape)


# Preprocess the Data

# Encode categorical variable 'Sex'
df_train['Sex'] = df_train['Sex'].map({'male': 0, 'female': 1})
df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})


# taking log value for target variable (Log (1+x))

df_train['Log_Calories'] = np.log1p(df_train['Calories'])


# Define features and target
X = df_train.drop(columns=['Log_Calories','Calories'])
y = df_train['Log_Calories']


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the Model
#Initialize and train XGBoost model
model = XGBRegressor(n_estimators=100, 
                     learning_rate=0.1, 
                     random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)
y_pred = np.maximum(0, y_pred)  # Avoid negative predictions for RMSLE

# 6. Evaluate with RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f'RMSLE: {rmsle:.4f}')


#Define custom scorer for RMSLE
def rmsle_scorer(y_test, y_pred):
    return np.sqrt(mean_squared_log_error(y_test, y_pred))

rmsle_cv = make_scorer(rmsle_scorer, greater_is_better=False)

#Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=3, scoring=rmsle_cv)
print("Mean RMSLE from Cross-Validation:", -np.mean(cv_scores))


# Predict the target variable on the df_test dataset
y_pred_test = model.predict(df_test)


y_pred_test = np.expm1(y_pred_test)


# Ensuring the prediction does not have negative values
import pandas as pd
pd.Series(y_pred_test).describe().round(4) 



# Create a submission dataframe
submission = pd.DataFrame({
    'id': df_test['id'],  # Replace 'id' with exact column in the submission file
    'Calories': y_pred_test
})
submission.to_csv('submission.csv', index=False)
submission.head(5)

