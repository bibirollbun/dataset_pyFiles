import pandas as pd


trainDF = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
testDF = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')


for df_name in ['trainDF', 'testDF']:
    df = globals()[df_name]

    df = pd.get_dummies(df, columns=['Genre'], drop_first=False, dtype=int)
    df = pd.get_dummies(df, columns=['Episode_Sentiment'], drop_first=False, dtype=int)

    # Map day and time
    day_map = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
               'Thursday': 4, 'Friday': 5, 'Saturday': 6}
    time_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}

    df['day_code'] = df['Publication_Day'].map(day_map)
    df['time_code'] = df['Publication_Time'].map(time_map)

    df['publication_daytime'] = df['day_code'] * 10 + df['time_code']
    df.drop(columns=['day_code', 'time_code'], inplace=True)

    # Extract episode number from title
    df['Episode_Title'] = df['Episode_Title'].str.extract(r'(\d+)').astype(float).astype('Int64')

    # Fill nulls properly
    #df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
    #df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median())
    #df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].median())

    # Drop Podcast_Name
    df.drop(columns='Podcast_Name', inplace=True)
    df.drop(columns='Episode_Title', inplace=True)
    df.drop(columns='Publication_Day', inplace=True)
    df.drop(columns='Publication_Time', inplace=True)

    # Save back to globals
    globals()[df_name] = df


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np

def train_rf_imputer(df_not_null, target_col, features, model_type='auto'):
    """Trains a RandomForest model on non-null rows to predict missing values in target_col."""
    y_train = df_not_null[target_col]
    
    # Auto-detect model type
    if model_type == 'auto':
        model_type = 'classification' if y_train.nunique() <= 10 and y_train.dtype in ['int', 'object', 'category'] else 'regression'

    # Choose model
    if model_type == 'regression':
        model = RandomForestRegressor(n_estimators=10, random_state=42)
    elif model_type == 'classification':
        model = RandomForestClassifier(n_estimators=10, random_state=42)
    else:
        raise ValueError("model_type must be 'regression', 'classification', or 'auto'")

    model.fit(df_not_null[features], y_train)
    return model


def predict_missing_values_safely(train_df, test_df, target_col, model_type='auto'):
    """
    Imputes missing values in `target_col`:
    - Trains imputer only on non-null rows of train_df.
    - Applies to missing values in both train_df and test_df.
    - Adds indicator column for missingness.
    """

    print(f"\nâ�¡ï¸� Starting imputation for column: '{target_col}'")
    
    # Add indicator column for missing values
    train_df[target_col + '_was_missing'] = train_df[target_col].isnull().astype(int)
    test_df[target_col + '_was_missing'] = test_df[target_col].isnull().astype(int)

    # Get features (complete columns only, excluding target)
    all_df = pd.concat([train_df, test_df], axis=0)
    complete_features = [col for col in all_df.columns if col != target_col and all_df[col].isnull().sum() == 0]

    if len(complete_features) == 0:
        raise ValueError(f"No complete features available to predict '{target_col}'.")

    print(f"âœ… Features used for imputation: {complete_features}")

    # Train imputer only on train's non-null rows
    df_not_null = train_df[train_df[target_col].notnull()]
    imputer_model = train_rf_imputer(df_not_null, target_col, complete_features, model_type)

    # Impute trainDF (only rows that are null)
    missing_train_mask = train_df[target_col].isnull()
    if missing_train_mask.sum() > 0:
        train_df.loc[missing_train_mask, target_col] = imputer_model.predict(train_df.loc[missing_train_mask, complete_features])
        print(f"ğŸ§© Imputed {missing_train_mask.sum()} missing values in trainDF[{target_col}]")

    # Impute testDF
    missing_test_mask = test_df[target_col].isnull()
    if missing_test_mask.sum() > 0:
        test_df.loc[missing_test_mask, target_col] = imputer_model.predict(test_df.loc[missing_test_mask, complete_features])
        print(f"ğŸ§© Imputed {missing_test_mask.sum()} missing values in testDF[{target_col}]")

    print(f"âœ… Imputation complete for '{target_col}'\n")
    return train_df, test_df


# -------- Apply to your dataset --------

cols_to_impute = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']

for col in cols_to_impute:
    trainDF, testDF = predict_missing_values_safely(trainDF, testDF, col)


trainDF.head()


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# ğŸ§¹ Step 1: Prepare data
X = trainDF.drop(columns=['Listening_Time_minutes'])
y = trainDF['Listening_Time_minutes']

# Step 2: Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Step 4: Define params
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Step 5: Train model with early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=10,
    verbose_eval=True
)

# Step 6: Predict and evaluate
y_pred = model.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")


# Step 6: Convert test data to DMatrix
dtest = xgb.DMatrix(testDF)

# Predict on test set
y_test_pred = model.predict(dtest)

# Prepare the submission dataframe
submission = pd.DataFrame({
    'id': testDF['id'],  # or whatever identifier you have
    'Listening_Time_minutes': y_test_pred
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

