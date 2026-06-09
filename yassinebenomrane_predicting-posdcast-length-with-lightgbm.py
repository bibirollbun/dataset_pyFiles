import pandas as pd     #importing pandas library
import numpy as np      #importing numpy libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import warnings
warnings.simplefilter("ignore")



train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_df.head()


test_df.head()


train_df.describe()


test_df.describe()


# Show number of missing values per column
print(train_df.isnull().sum())

# To see the percentage of missing values per column:
print((train_df.isnull().sum() / len(train_df)) * 100)


train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median())
train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median())
train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(0)

test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(test_df['Episode_Length_minutes'].median())
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median())
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].fillna(0)



def create_interaction_features(df):
    df_new = df.copy()
    
    # Create a numeric encoding for Genre
    genre_mapping = {genre: i for i, genre in enumerate(df_new['Genre'].unique())}
    df_new['Genre_Numeric'] = df_new['Genre'].map(genre_mapping)
        # Ensure Genre_Numeric is numeric
    df_new['Genre_Numeric'] = df_new['Genre_Numeric'].astype(float)
    
    # Convert Episode_Length_minutes to float if it's not numeric
    df_new['Episode_Length_minutes'] = df_new['Episode_Length_minutes'].astype(float)
    
    # Create the interaction feature
    df_new['Genre_Length_Interaction'] = df_new['Genre_Numeric'] * df_new['Episode_Length_minutes']
    
    # Host and Guest Popularity interaction
    # Handle NaN values in Guest_Popularity_percentage
    df_new['Guest_Popularity_percentage'] = df_new['Guest_Popularity_percentage'].fillna(0)
    
    # interaction feature
    df_new['Host_Guest_Popularity_Interaction'] = df_new['Host_Popularity_percentage'] * df_new['Guest_Popularity_percentage']
    
    # 3. Number of Ads and Episode Length interaction
    df_new['Ads_Length_Interaction'] = df_new['Number_of_Ads'] * df_new['Episode_Length_minutes']
    
    # 4. Sentiment and Host Popularity interaction
    # Convert sentiment to numeric (-1, 0, 1)
    sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    
    df_new['Sentiment_Numeric'] = df_new['Episode_Sentiment'].map(sentiment_mapping).astype(int)
    df_new['Sentiment_Host_Interaction'] = df_new['Sentiment_Numeric'] * df_new['Host_Popularity_percentage']
    
    # 5. Day and Time interaction (after converting them to cyclical features)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_mapping = {day: i for i, day in enumerate(days)}
    df_new['day_numeric'] = df_new['Publication_Day'].map(day_mapping)
        # Convert day_numeric to a numeric type explicitly
    df_new['day_numeric'] = pd.to_numeric(df_new['day_numeric'], errors='coerce')
    df_new['day_sin'] = np.sin(2 * np.pi * df_new['day_numeric'] / 7)
    df_new['day_cos'] = np.cos(2 * np.pi * df_new['day_numeric'] / 7)
    
    times = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_mapping = {time: i for i, time in enumerate(times)}
    df_new['time_numeric'] = df_new['Publication_Time'].map(time_mapping)
        # Convert time_numeric to a numeric type explicitly
    df_new['time_numeric'] = pd.to_numeric(df_new['time_numeric'], errors='coerce')
    df_new['time_sin'] = np.sin(2 * np.pi * df_new['time_numeric'] / 4)
    df_new['time_cos'] = np.cos(2 * np.pi * df_new['time_numeric'] / 4)
    
    # day-time interaction features
    df_new['day_time_sin_interaction'] = df_new['day_sin'] * df_new['time_sin']
    df_new['day_time_cos_interaction'] = df_new['day_cos'] * df_new['time_cos']
    
    
    # Ads density
    df_new['Ads_Per_Minute'] = df_new['Number_of_Ads'] / df_new['Episode_Length_minutes'].clip(1, None)
    
    # binary flags for missing values
    df_new['Guest_Missing'] = df_new['Guest_Popularity_percentage'].isna().astype(int)
    df_new['Length_Missing'] = df_new['Episode_Length_minutes'].isna().astype(int)
    
    return df_new


def prepare_for_lightgbm(df, target_col='Listening_Time_minutes'):
    # Create a copy to avoid modifying the original dataframe
    df_encoded = df.copy()
    
    # List of original categorical columns
    categorical_cols_names = [
        'Genre', 
        'Publication_Day', 
        'Publication_Time', 
        'Episode_Sentiment',
        'Podcast_Name',
        'Episode_Title' 
    ]
    
    # Encode all categorical columns to numeric
    for col in categorical_cols_names:
        if col in df_encoded.columns:
            # Convert to categorical and then to numeric codes
            df_encoded[col] = pd.Categorical(df_encoded[col])
            df_encoded[col] = df_encoded[col].cat.codes
    
    # Add any new categorical columns 
    categorical_cols_names.extend([
        'Genre_Numeric',
        'day_numeric',
        'time_numeric',
        'Guest_Missing',
        'Length_Missing'
    ])
    
    # Remove the target column and any ID columns from features
    columns_to_drop = [target_col, 'id'] if 'id' in df_encoded.columns else [target_col]
    
    if target_col in df_encoded.columns:
        X = df_encoded.drop(columns=columns_to_drop)
        y = df_encoded[target_col]
    else:
        X = df_encoded.drop(columns=['id']) if 'id' in df_encoded.columns else df_encoded
        y = None
    
    # Filter out categorical column names that aren't in X
    categorical_cols_names = [col for col in categorical_cols_names if col in X.columns]
    
    # Get column indices for categorical features (LightGBM needs indices, not names)
    feature_names = X.columns.tolist()
    categorical_cols = [feature_names.index(col) for col in categorical_cols_names if col in feature_names]
    
    return X, y, categorical_cols, feature_names


pip install optuna



#import optuna

#df_with_interactions = create_interaction_features(train_df)
#X, y, categorical_cols, feature_names = prepare_for_lightgbm(df_with_interactions, target_col='Listening_Time_minutes')

# Split into train/validation sets
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Define Optuna objective
#def objective(trial):
#    params = {
#        'objective': 'regression',
#        'metric': 'rmse',
#        'verbosity': -1,
#        'boosting_type': 'gbdt',
#        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
#        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
#        'max_depth': trial.suggest_int('max_depth', 5, 15),
#        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
#        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
#        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
#        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#    }

    # LightGBM Dataset
#    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols,feature_name=feature_names, free_raw_data=False)
#    dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_cols,feature_name=feature_names, reference=dtrain, free_raw_data=False)

    # Train
#    model = lgb.train(
#    params,
#    dtrain,
#    valid_sets=[dtrain, dvalid],
#    valid_names=['train', 'valid'],
#    num_boost_round=1000,
#    callbacks=[
#        lgb.early_stopping(50),
#        lgb.log_evaluation(0)  # Change to 10 if you want to see logs
#    ]
#)


 #   # Predict
 #   preds = model.predict(X_valid)
 #   rmse = mean_squared_error(y_valid, preds, squared=False)
 #   return rmse

# Run Optuna optimization
#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=50)

#print("Best RMSE:", study.best_value)
#print("Best Params:", study.best_params)


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

# --- Step 1: Feature Engineering ---
df_with_interactions = create_interaction_features(train_df)
X, y, categorical_cols, feature_names = prepare_for_lightgbm(df_with_interactions, target_col='Listening_Time_minutes')

# --- Step 2: Train/Validation Split ---
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize base estimator - a simpler version of your model
base_model = lgb.LGBMRegressor(
    learning_rate=0.05,
    n_estimators=100,
    num_leaves=31,
    categorical_feature=categorical_cols
)

# Initialize RFECV with CV folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rfe_selector = RFECV(
    estimator=base_model,
    step=1,  # Remove one feature at a time
    cv=kf,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    min_features_to_select=10  # Don't go below 10 features
)

# Fit the selector
rfe_selector.fit(X, y)

# Get selected features
selected_features = X.columns[rfe_selector.support_]

print(f"RFECV selected {len(selected_features)} features")
print(f"Optimal number of features: {rfe_selector.n_features_}")

# --- Step 3: Train/Validation Split with selected features only ---
X_selected = X[selected_features]
X_train, X_valid, y_train, y_valid = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Update categorical columns for the selected features
categorical_cols_selected = [i for i, col in enumerate(selected_features) 
                          if col in [X.columns[i] for i in categorical_cols]]

# --- Step 4: Train the Model with your best parameters ---
Best_params = {
    'learning_rate': 0.04541706943693706,
    'num_leaves': 116,
    'max_depth': 14,
    'min_data_in_leaf': 104,
    'feature_fraction': 0.7957855478343936,
    'bagging_fraction': 0.9728338065953579,
    'bagging_freq': 3,
    'lambda_l1': 2.2417738239035918e-06,
    'lambda_l2': 1.6700004223673643e-05
}

model = lgb.LGBMRegressor(**Best_params, n_estimators=1000)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_valid, y_valid)],
          eval_names=['train', 'valid'],
          eval_metric='rmse',
          categorical_feature=categorical_cols_selected,
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])


# --- Step 5: Predict on Validation Set and Calculate RMSE ---
preds = model.predict(X_valid)
rmse = mean_squared_error(y_valid, preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")



# --- Step 5: Train Final Model on All Data with Selected Features ---
Best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
})

dtrain_full = lgb.Dataset(X_selected, label=y, 
                         categorical_feature=categorical_cols_selected, 
                         feature_name=selected_features.tolist())
final_model = lgb.train(Best_params, dtrain_full, num_boost_round=1000)



# Predict on the full training data
y_train_pred = final_model.predict(X_selected)

rmse_train = np.sqrt(mean_squared_error(y, y_train_pred))
print(f"Training RMSE: {rmse_train:.2f}")



def prepare_test_data(test_df, selected_features):
    df_features = create_interaction_features(test_df)
    
    # Prepare for LightGBM in the same way as training data
    X_test, _, _, _ = prepare_for_lightgbm(df_features)
    
    # Ensure we have all required columns
    missing_cols = set(selected_features) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0 
        
    # Make sure columns are in the same order as training
    X_test = X_test[selected_features]
    
    return X_test




X_test = prepare_test_data(test_df, selected_features)
y_test_pred = final_model.predict(X_test)


submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_time_minutes': y_test_pred,
})
submission.head()


submission.to_csv("submission.csv", index=False)


import os
print(os.listdir())


