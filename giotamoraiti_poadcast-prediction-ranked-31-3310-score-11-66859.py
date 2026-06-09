!pip install scikit-learn==1.6.1


import os
print(os.listdir("/kaggle/input"))

import warnings
warnings.filterwarnings("ignore", message="The split criterion chosen was not present")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from itertools import combinations


def preprocess_data(train_df, test_df):
    # Data preprocessing
    # Separate features and target
    X_train = train_df.drop(columns=['Listening_Time_minutes', 'id'])  # Drop 'id' and target column
    y_train = train_df['Listening_Time_minutes']

    # Drop 'id' from test data
    X_test = test_df.drop(columns=['id'])

    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Create a preprocessor with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            #('num', Pipeline([
            ('num', 'passthrough', numerical_cols),
                #('imputer', SimpleImputer(strategy='median')),  # Impute missing values with the median
                # ('scaler', StandardScaler())  # Standardize numerical data
            #]), numerical_cols),
            
            ('cat', Pipeline([
                # ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  # Encode categorical data as numbers
            ]), categorical_cols)
        ]
    )

    # Apply transformations to the training and test data (features only)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Combine the column names
    all_columns = numerical_cols + categorical_cols

    # Create DataFrames from the transformed arrays
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=all_columns)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=all_columns)

    return X_train_transformed_df, X_test_transformed_df, y_train


def evaluate_models(X, y, model, X_test, random_state=26):
    # Define KFold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    rmse_scores = []
    y_pred_test = np.zeros(X_test.shape[0])

    for idx_train, idx_valid in cv.split(X, y):
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
        X_test_fold = X_test.copy()
        
        encoded_columns = X.columns[5:] # Encode columns from index 5 and beyond
        encoder = TargetEncoder()
        encoder.set_output(transform='pandas') 
        
        X_train[encoded_columns] = encoder.fit_transform(X_train[encoded_columns], y_train)
        X_valid[encoded_columns] = encoder.transform(X_valid[encoded_columns])
        X_test_fold[encoded_columns] = encoder.transform(X_test_fold[encoded_columns])

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            # callbacks=[lgb.log_evaluation(100)]
        )

        # Predict on validation set
        y_pred = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        rmse_scores.append(rmse)
        
        y_pred_test += model.predict(X_test_fold)
        
        print(f'Fold RMSE: {rmse:.4f}')

    mean_rmse = np.mean(rmse_scores)
    y_pred_test = y_pred_test/5
        
    # Print the mean RMSE for this model
    print(f'Mean RMSE: {mean_rmse:.4f}')
    
    return y_pred_test


def train_models(X, y, model, X_test, random_state=26):
    encoded_columns = X.columns[5:] # Dont target encode original
    encoder = TargetEncoder(random_state=random_state)
    
    X[encoded_columns] = encoder.fit_transform(X[encoded_columns], y_train) #.astype('float32')
    X_test[encoded_columns] = encoder.transform(X_test[encoded_columns]) #.astype('float32')

    model.fit(
        X, y
        )
    
    # Predict
    y_pred_test = model.predict(X_test)

    return y_pred_test


def generate_predictions_and_save(test_df, predictions, output_prefix='predictions'):
    output_df = pd.DataFrame({
        'id': test_df['id'],
        'Listening_Time_minutes': predictions
    })
    output_filename = f"{output_prefix}.csv"
    output_df.to_csv(output_filename, index=False)


# Load the data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
extra_data = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")

# Use extra data for training
train_df = pd.concat([train_df, extra_data], ignore_index=True)

# Drop duplicate samples
train_df = train_df.drop_duplicates()

# Drop target
train_df = train_df.dropna(subset=["Listening_Time_minutes"])


# Add new feature 'Is_Weekend'
train_df['Is_Weekend'] = train_df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
test_df['Is_Weekend'] = test_df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

# Apply the preprocessing (ordinal encoding for categorical columns)
X_train_prepr_fe, X_test_prepr_fe, y_train = preprocess_data(train_df, test_df)

# Identify numerical and categorical columns
numerical_columns = X_train_prepr_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X_train_prepr_fe.select_dtypes(include=['object']).columns.tolist()
all_columns = numerical_columns + categorical_columns

# Convert to dataframes
X_train_prepr_fe = pd.DataFrame(X_train_prepr, columns=all_columns)
X_test_prepr_fe = pd.DataFrame(X_test_prepr, columns=all_columns)


# Convert to category type for tree based models, training data
X_train_prepr_fe['Genre'] = X_train_prepr_fe['Genre'].astype('category')
X_train_prepr_fe['Podcast_Name'] = X_train_prepr_fe['Podcast_Name'].astype('category')
X_train_prepr_fe['Publication_Day'] = X_train_prepr_fe['Publication_Day'].astype('category')
X_train_prepr_fe['Publication_Time'] = X_train_prepr_fe['Publication_Time'].astype('category')
X_train_prepr_fe['Episode_Sentiment'] = X_train_prepr_fe['Episode_Sentiment'].astype('category')

# Convert to category type for tree based models, test data
X_test_prepr_fe['Genre'] = X_test_prepr_fe['Genre'].astype('category')
X_test_prepr_fe['Podcast_Name'] = X_test_prepr_fe['Podcast_Name'].astype('category')
X_test_prepr_fe['Publication_Day'] = X_test_prepr_fe['Publication_Day'].astype('category')
X_test_prepr_fe['Publication_Time'] = X_test_prepr_fe['Publication_Time'].astype('category')
X_test_prepr_fe['Episode_Sentiment'] = X_test_prepr_fe['Episode_Sentiment'].astype('category')


# Replace 'Episode_Title' ordinal encoded column with episode number from the original column
Episode_Title_df = train_df['Episode_Title'].str[8:].astype('category') # Extract episode number
X_train_prepr_fe = X_train_prepr_fe.drop('Episode_Title', axis=1)
X_train_prepr_fe = X_train_prepr_fe.reset_index(drop=True)
Episode_Title_df = Episode_Title_df.reset_index(drop=True)
X_train_prepr_fe = pd.concat([X_train_prepr_fe, Episode_Title_df], axis=1)

# Replace 'Episode_Title' ordinal encoded column with episode number from the original column
# test data
Episode_Title_test_df = test_df['Episode_Title'].str[8:].astype('category')
X_test_prepr_fe = X_test_prepr_fe.drop('Episode_Title', axis=1)
X_test_prepr_fe = X_test_prepr_fe.reset_index(drop=True)
Episode_Title_test_df = Episode_Title_test_df.reset_index(drop=True)
X_test_prepr_fe = pd.concat([X_test_prepr_fe, Episode_Title_test_df], axis=1)


# There are 10 Original Features
# New Features, 5 new + 'Is_Weekend' added before
X_train_prepr_fe['Is_High_Host_Popularity'] = (X_train_prepr_fe['Host_Popularity_percentage'] > 70).astype(int)
X_train_prepr_fe['Is_High_Guest_Popularity'] = (X_train_prepr_fe['Guest_Popularity_percentage'] > 70).astype(int)
X_train_prepr_fe['Host_Guest_Popularity_Gap'] = X_train_prepr_fe['Host_Popularity_percentage'] / X_train_prepr_fe['Guest_Popularity_percentage']
X_train_prepr_fe['Host_Guest_Popularity_Gap'] = X_train_prepr_fe['Host_Guest_Popularity_Gap'].replace([np.inf, -np.inf], np.nan)
X_train_prepr_fe['Ad_Density'] = X_train_prepr_fe['Number_of_Ads'] / X_train_prepr_fe['Episode_Length_minutes']
X_train_prepr_fe['Ad_Density'] = X_train_prepr_fe['Ad_Density'].replace([np.inf, -np.inf], np.nan)
X_train_prepr_fe['Is_Long_Medium_Small_Episode'] = X_train_prepr_fe['Episode_Length_minutes'].apply(lambda x: 2 if x > 60 else 0 if x < 20 else 1)

# Test features
X_test_prepr_fe['Is_High_Host_Popularity'] = (X_test_prepr_fe['Host_Popularity_percentage'] > 70).astype(int)
X_test_prepr_fe['Is_High_Guest_Popularity'] = (X_test_prepr_fe['Guest_Popularity_percentage'] > 70).astype(int)
X_test_prepr_fe['Host_Guest_Popularity_Gap'] = X_test_prepr_fe['Host_Popularity_percentage'] / X_test_prepr_fe['Guest_Popularity_percentage']
X_test_prepr_fe['Host_Guest_Popularity_Gap'] = X_test_prepr_fe['Host_Guest_Popularity_Gap'].replace([np.inf, -np.inf], np.nan)
X_test_prepr_fe['Ad_Density'] = X_test_prepr_fe['Number_of_Ads'] / X_test_prepr_fe['Episode_Length_minutes']
X_test_prepr_fe['Ad_Density'] = X_test_prepr_fe['Ad_Density'].replace([np.inf, -np.inf], np.nan)
X_test_prepr_fe['Is_Long_Medium_Small_Episode'] = X_test_prepr_fe['Episode_Length_minutes'].apply(lambda x: 2 if x > 60 else 0 if x < 20 else 1)

# One more new feature
X_train_prepr_fe['LinearFeat'] = 0.728*X_train_prepr_fe['Episode_Length_minutes']
X_test_prepr_fe['LinearFeat'] = 0.728*X_test_prepr_fe['Episode_Length_minutes']


encode_columns = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 
                  'Publication_Day', 'Publication_Time', 'Guest_Popularity_percentage', 'Episode_Title', 
                  'Podcast_Name', 'Genre']

pair_size = [2, 3, 4, 5, 6, 7]

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r))):
        new_col_name = '_'.join(cols)
        
        X_train_prepr_fe[new_col_name] = X_train_prepr_fe[cols[0]].astype(str)
        for col in cols[1:]:
            X_train_prepr_fe[new_col_name] = X_train_prepr_fe[new_col_name] + '_' + X_train_prepr_fe[col].astype(str)
        X_train_prepr_fe[new_col_name] = X_train_prepr_fe[new_col_name].astype('category')
        
        X_test_prepr_fe[new_col_name] = X_test_prepr_fe[cols[0]].astype(str)
        for col in cols[1:]:
            X_test_prepr_fe[new_col_name] = X_test_prepr_fe[new_col_name] + '_' + X_test_prepr_fe[col].astype(str)
        X_test_prepr_fe[new_col_name] = X_test_prepr_fe[new_col_name].astype('category')


# Convert to category type and optionally float32 for less memory usage
X_train_prepr_fe = X_train_prepr_fe.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)
# X_train_prepr_fe = X_train_prepr_fe.apply(lambda col: col.astype('float32') if col.dtype == 'float64' else col)

X_test_prepr_fe = X_test_prepr_fe.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)
# X_test_prepr_fe = X_test_prepr_fe.apply(lambda col: col.astype('float32') if col.dtype == 'float64' else col)


model1 = lgb.LGBMRegressor(
          n_estimators=1500,
          max_depth=-1,
          num_leaves=2048,
          colsample_bytree=0.7,
          learning_rate=0.01,
          objective='l2',
          metric='rmse', 
          verbosity=-1,
          max_bin=1024,
	      random_state=26)

model2 = xgb.XGBRegressor(
         tree_method='hist',
         device='cuda',
         enable_categorical=True,
         n_estimators=1500,
         learning_rate=0.01,
         max_depth=14,
         colsample_bytree=0.7,
         subsample=0.9,
	     objective='reg:squarederror', 
   	     eval_metric='rmse',
         min_child_weight=10,
	     random_state=26)

# model3 = CatBoostRegressor(
# 	       iterations=1500,         
#          learning_rate=0.01,
#          depth=14,                    
#          loss_function='RMSE',
#          eval_metric='RMSE',
#          verbose=0,       
#          max_bin=1024,              
#          random_state=26)


# Evaluate models using 5 fold cross-validation, lgbm
y_preds = evaluate_models(X_train_prepr_fe, y_train, model1, X_test_prepr_fe)


# Train on the whole dataset, lgbm
y_preds = train_models(X_train_prepr_fe, y_train, model1, X_test_prepr_fe)

# Generate predictions for the best model
generate_predictions_and_save(
    test_df=test_df,
    predictions=y_preds,
    output_prefix='model_preds_lgbm')


# Evaluate models using 5 fold cross-validation, xgboost
y_preds = evaluate_models(X_train_prepr_fe, y_train, model2, X_test_prepr_fe)


# Train on the whole dataset, xgboost
y_preds = train_models(X_train_prepr_fe, y_train, model1, X_test_prepr_fe)

# Generate predictions for the best model
generate_predictions_and_save(
    test_df=test_df,
    predictions=y_preds,
    output_prefix='model_preds_xgb')


train_df = train_df.drop(columns=['id'])
test_df = test_df_initial.drop(columns=['id'])

label = 'Listening_Time_minutes'

# Preprocessing
train_df['Is_Weekend'] = train_df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
test_df['Is_Weekend'] = test_df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

# 10 Original Features
# New Features, 5 new + 'Is_Weekend' added before
train_df['Is_High_Host_Popularity'] = (train_df['Host_Popularity_percentage'] > 70).astype(int)
train_df['Is_High_Guest_Popularity'] = (train_df['Guest_Popularity_percentage'] > 70).astype(int)
train_df['Host_Guest_Popularity_Gap'] = train_df['Host_Popularity_percentage'] / train_df['Guest_Popularity_percentage']
train_df['Host_Guest_Popularity_Gap'] = train_df['Host_Guest_Popularity_Gap'].replace([np.inf, -np.inf], np.nan)
train_df['Ad_Density'] = train_df['Number_of_Ads'] / train_df['Episode_Length_minutes']
train_df['Ad_Density'] = train_df['Ad_Density'].replace([np.inf, -np.inf], np.nan)
train_df['Is_Long_Medium_Small_Episode'] = train_df['Episode_Length_minutes'].apply(lambda x: 2 if x > 60 else 0 if x < 20 else 1)

# Test features
test_df['Is_High_Host_Popularity'] = (test_df['Host_Popularity_percentage'] > 70).astype(int)
test_df['Is_High_Guest_Popularity'] = (test_df['Guest_Popularity_percentage'] > 70).astype(int)
test_df['Host_Guest_Popularity_Gap'] = test_df['Host_Popularity_percentage'] / test_df['Guest_Popularity_percentage']
test_df['Host_Guest_Popularity_Gap'] = test_df['Host_Guest_Popularity_Gap'].replace([np.inf, -np.inf], np.nan)
test_df['Ad_Density'] = test_df['Number_of_Ads'] / test_df['Episode_Length_minutes']
test_df['Ad_Density'] = test_df['Ad_Density'].replace([np.inf, -np.inf], np.nan)
test_df['Is_Long_Medium_Small_Episode'] = test_df['Episode_Length_minutes'].apply(lambda x: 2 if x > 60 else 0 if x < 20 else 1)

# New Features added 
train_df['LinearFeat'] = 0.728*train_df['Episode_Length_minutes']
test_df['LinearFeat'] = 0.728*test_df['Episode_Length_minutes']


# Define train and valid ratio
train_sample = train_df.sample(frac=1.0, random_state=26)

# Define AutoML model
predictor = TabularPredictor(label=label, eval_metric ='rmse', 
                             problem_type="regression")

# Use best quality for better results (it takes more time)
# refit_full=True to train with best parameters found on the whole dataset
predictor.fit(train_sample, presets='best_quality', time_limit=3600*5, refit_full=True, keep_only_best=True, 
              auto_stack=True, save_space=True, verbosity=3, ag_args_fit={'num_gpus': 1})

# Print the results and feature importance
lb=predictor.leaderboard(data=train_sample)
feature_importance = predictor.feature_importance(data=train_sample)

print("Leaderboard!!!")
print(lb)
print("Feature Importance!!!")
print(feature_importance)

# Generte predictions
df = predictor.predict(test_df).to_frame(name=label)
df.to_csv('./autogluon_best_quality_gpu.csv',index=False)
autogluon_df = pd.read_csv("autogluon_best_quality_gpu.csv")
autogluon_df.insert(0, 'id', test_df_initial['id'])
autogluon_df.to_csv("autogluon_preds.csv", index=False)


# model4 = RandomForestRegressor(n_estimators=100, n_jobs=-1)


def ensemble_predictions_four(model1_preds_file, model2_preds_file, model3_preds_file, model4_preds_file, weights=(0.4, 0.3, 0.2, 0.1), output_prefix='best'):
    # Read the prediction files
    preds1_df = pd.read_csv(model1_preds_file)
    preds2_df = pd.read_csv(model2_preds_file)
    preds3_df = pd.read_csv(model3_preds_file)
    preds4_df = pd.read_csv(model4_preds_file)
    
    # Ensure the 'id' column is the same in all prediction files
    if not (preds1_df['id'].equals(preds2_df['id']) and preds1_df['id'].equals(preds3_df['id']) and preds1_df['id'].equals(preds4_df['id'])):
        raise ValueError("The 'id' columns in all prediction files do not match.")
    
    # Get the predictions for all models
    preds1 = preds1_df['Listening_Time_minutes']
    preds2 = preds2_df['Listening_Time_minutes']
    preds3 = preds3_df['Listening_Time_minutes']
    preds4 = preds4_df['Listening_Time_minutes']
    
    # Calculate the weighted ensemble predictions
    ensemble_preds = (
        weights[0] * preds1 + 
        weights[1] * preds2 + 
        weights[2] * preds3 +
        weights[3] * preds4
    )
    
    # Prepare the output DataFrame
    ensemble_df = pd.DataFrame({
        'id': preds1_df['id'],
        'Listening_Time_minutes': ensemble_preds
    })
    
    # Define the filename for the ensemble predictions
    output_filename = f"{output_prefix}_ensemble.csv"
    
    # Save to CSV
    ensemble_df.to_csv(output_filename, index=False)
    print(f'Ensemble predictions saved to {output_filename}')


# Example usage
ensemble_predictions_four('lgbm.csv', 'autogluon.csv', 'xgb.csv', 'rf_estimators_300.csv', weights=(0.5, 0.15, 0.2, 0.15))

