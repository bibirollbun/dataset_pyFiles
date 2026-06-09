import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# Import gradient boosting libraries
import lightgbm as lgb
import xgboost as xgb
import catboost as catb


# Setup plotting
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)


# For reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')  


print("______________Train data______________")
print(train.head())
print("______________Test data______________")
print(test.head())
print(sample_submission.head())


# Display information about the datasets
print("Train data shape")
print(train.shape)
print("Test data shape")
print(test.shape)
print("sample_submission data shape")
print(sample_submission.shape)


# basic statistics
print(train['accident_risk'].describe())


train.isnull().sum()


# Create subplots
fig, axs = plt.subplots(1, 5, figsize=(15, 4))  # 1 row, 5 columns

axs[0].hist(train['accident_risk'], bins=20, color='blue', edgecolor='black')
axs[0].set_title('accident_risk')

axs[1].hist(train['num_reported_accidents'], bins=5, color='blue', edgecolor='black')
axs[1].set_title('num_reported_accidents')

axs[2].hist(train['curvature'], bins=20, color='blue', edgecolor='black')
axs[2].set_title('curvature')

axs[3].hist(train['speed_limit'], bins=5, color='blue', edgecolor='black')
axs[3].set_title('speed_limit')

axs[4].hist(train['num_lanes'], bins=5, color='blue', edgecolor='black')
axs[4].set_title('num_lanes')
#accident_risk, num_reported_accidents
plt.tight_layout()
plt.show()



#road_type,lighting ,weather,time_of_day
#following feature need to be encode with "Label Encodeing"

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
road_type_encoded = encoder.fit_transform(train["road_type"])
lighting_encoded = encoder.fit_transform(train["lighting"])
weather_encoded = encoder.fit_transform(train["weather"])
time_of_day_encoded = encoder.fit_transform(train["time_of_day"])


# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(10, 4))  # 1 row, 4 columns

axs[0].hist(road_type_encoded, bins=3, color='orange', edgecolor='black')
axs[0].set_title('road_type')

axs[1].hist(lighting_encoded, bins=3, color='orange', edgecolor='black')
axs[1].set_title('lighting')

axs[2].hist(weather_encoded, bins=3, color='orange', edgecolor='black')
axs[2].set_title('weather')

axs[3].hist(time_of_day_encoded, bins=3, color='orange', edgecolor='black')
axs[3].set_title('time_of_day')

#accident_risk, num_reported_accidents
plt.tight_layout()
plt.show()



#road_signs_present,public_road,holiday,school_season
#following feature need to be encode with "One-Hot Encodeing"
# True → 1, False → 0
road_signs_present_encoded = [int(x) for x in train["road_signs_present"]]  
public_road_encoded = [int(x) for x in train["public_road"]]  
holiday_encoded = [int(x) for x in train["holiday"]]  
school_season_encoded = [int(x) for x in train["school_season"]]  


# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(10, 4))  # 1 row, 4 columns

axs[0].hist(road_signs_present_encoded, bins=2, color='blue', edgecolor='black')
axs[0].set_title('road_signs_present')

axs[1].hist(public_road_encoded, bins=2, color='blue', edgecolor='black')
axs[1].set_title('public_road')

axs[2].hist(holiday_encoded, bins=2, color='blue', edgecolor='black')
axs[2].set_title('holiday')

axs[3].hist(school_season_encoded, bins=2, color='blue', edgecolor='black')
axs[3].set_title('school_season')

#accident_risk, num_reported_accidents
plt.tight_layout()
plt.show()



from sklearn.preprocessing import PolynomialFeatures
def preprocess_dataframe(df: pd.DataFrame, drop_cols: list = None) -> pd.DataFrame:
    df = df.copy()

    # Handle missing values
    # Althought there are no missing data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Step 0: Drop specified columns 
    
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')  # 'ignore' avoids errors if column not found

    # Step 1: Encode boolean columns
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Step 2: Label encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    # Step 3: Polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df[['num_lanes', 'curvature']])
    poly_cols = [f'poly_{i}' for i in range(poly_features.shape[1])]
    df[poly_cols] = poly_features
    
    return df


preprocess_train = preprocess_dataframe(train,drop_cols=['accident_risk','id'])


preprocess_train


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
def get_score(n_estimators):
    """Return the average MAE over 5 CV folds of XGBRegressor model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer(strategy='median')),# more robust to outliers
        ('model', XGBRegressor(n_estimators=n_estimators, learning_rate=0.05) )
    ])
    scores = -1 * cross_val_score(my_pipeline, X_train_final, y_train_fianl,
                              cv=5,
                              scoring='neg_mean_absolute_error')
    cv_mae = scores.mean()
    
    # Fit on full training set
    my_pipeline.fit(X_train_final, y_train_fianl)
    
    # Validation score
    valid_preds = my_pipeline.predict(X_valid)
    valid_mae = mean_absolute_error(y_valid, valid_preds)
    
    #print(f"CV MAE: {cv_mae:.4f}")
    #print(f"Validation MAE: {valid_mae:.4f}")

    return cv_mae, valid_mae



train_data = preprocess_dataframe(train,drop_cols=['id'])
test_data_id = test['id']
test_data = preprocess_dataframe(test,drop_cols=['id'])


train_x = train_data.drop(['accident_risk'] , axis=1)
train_y = train_data['accident_risk']


X_train_final, X_valid, y_train_fianl, y_valid = train_test_split(train_x, train_y, train_size=0.8, test_size=0.2,random_state=0)


#10/22/2025 update for checking if the parameter is overfitting
cv_results = {}
valid_results = {}

for i in range(1, 30):
    n = 10 * i
    cv_mae, valid_mae = get_score(n)
    cv_results[n] = cv_mae
    valid_results[n] = valid_mae



import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(10, 6))
plt.plot(list(cv_results.keys()), list(cv_results.values()), label='CV MAE', marker='o')
plt.plot(list(valid_results.keys()), list(valid_results.values()), label='Validation MAE', marker='s')
plt.xlabel('n_estimators')
plt.ylabel('Mean Absolute Error')
plt.title('Model Performance vs. Number of Trees')
plt.legend()
plt.grid(True)
plt.show()



# Define the model
my_model_1 = XGBRegressor(n_estimators=100, learning_rate=0.05) 

# Fit the model
my_model_1.fit(X_train_final, y_train_fianl)

# Get predictions
predictions_1 = my_model_1.predict(X_valid)

# Calculate MAE
mae_1 =  mean_absolute_error(predictions_1, y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_1)


# Define the model
my_model_2 = XGBRegressor(n_estimators=200, learning_rate=0.05) 

# Fit the model
my_model_2.fit(X_train_final, y_train_fianl)

# Get predictions
predictions_2 = my_model_2.predict(X_valid)

# Calculate MAE
mae_2 =  mean_absolute_error(predictions_2, y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_2)


# Define the model
my_model_3 = XGBRegressor(n_estimators=300, learning_rate=0.05) 

# Fit the model
my_model_3.fit(X_train_final, y_train_fianl)

# Get predictions
predictions_3 = my_model_3.predict(X_valid)

# Calculate MAE
mae_3 =  mean_absolute_error(predictions_3, y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_3)


final_preds = np.round(my_model_3.predict(test_data),3)


print(final_preds)


submission = pd.DataFrame({'id': test_data_id, 'accident_risk': final_preds})
submission.to_csv('submission.csv', index=False)
display(submission.head())

