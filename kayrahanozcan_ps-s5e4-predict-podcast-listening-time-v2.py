#%% [markdown]
# # Podcast Listening Time Prediction
# 
# Objective: Predict `Listening_Time_minutes` based on podcast episode features.
# Approach: Feature Engineering (Aggregations, Interactions), Target Encoding (using `category_encoders`), LightGBM with KFold Cross-Validation.

#%% [code]
# Installs
# Ensure category_encoders is installed
!pip install -qq category_encoders
# Optional: Pin scikit-learn if needed
# !pip install -qq scikit-learn==1.6.1 
# Ensure lightgbm is installed
!pip install -qq lightgbm

print("Libraries checked/installed.")

#%% [code]
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
# from sklearn.preprocessing import TargetEncoder # Using category_encoders instead
from category_encoders import TargetEncoder 
from sklearn.metrics import mean_squared_error
from itertools import combinations
from tqdm.notebook import tqdm # Use tqdm.notebook for better notebook progress bars
import warnings
import gc

warnings.simplefilter('ignore')
print("Core imports successful.")

#%% [markdown]
# ## Configuration

#%% [code]
SEED = 42 # Seed for reproducibility
N_SPLITS = 5 # Number of folds for KFold
np.random.seed(SEED) # Set NumPy seed
print(f"Seed set to {SEED}, N_Splits for CV: {N_SPLITS}")

#%% [markdown]
# ## Feature Engineering Function

#%% [code]
def feature_eng(df):
    """Applies initial feature engineering steps: mapping, type casting, episode number extraction."""
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

    if 'Episode_Title' in df.columns and not df['Episode_Title'].isnull().all():
        df['Episode_Num'] = df['Episode_Title'].str.extract(r'(\d+)$', expand=False)
        df['Episode_Num'] = pd.to_numeric(df['Episode_Num'], errors='coerce').fillna(0).astype(int)
        df['Episode_Num'] = df['Episode_Num'].astype('category')
        if 'Episode_Title' in df.columns: df = df.drop(columns=['Episode_Title']) # Drop safely

    cat_cols_map = {
        'Genre': genr_dict, 'Podcast_Name': podc_dict, 'Publication_Day': week_dict,
        'Publication_Time': time_dict, 'Episode_Sentiment': sent_dict
    }
    for col, mapping in cat_cols_map.items():
        if col in df.columns:
            if df[col].dtype == 'object':
                 df[col] = df[col].map(mapping)
            df[col] = df[col].astype('category')

    for col in df.select_dtypes(include='object').columns:
         # Make sure TARGET column defined globally isn't converted if it's string
         if 'TARGET' not in globals() or col != TARGET: 
            df[col] = df[col].astype('category')

    num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

print("Feature engineering function defined.")

#%% [markdown]
# ## Load Data

#%% [code]
print("Loading data...")
input_path = '/kaggle/input/playground-series-s5e4/' 

try:
    df_train = pd.read_csv(f'{input_path}train.csv', index_col='id')
    df_test = pd.read_csv(f'{input_path}test.csv', index_col='id')
    df_subm = pd.read_csv(f'{input_path}sample_submission.csv', index_col='id')
    TARGET = 'Listening_Time_minutes' # Define target column name here
    print("Train shape:", df_train.shape)
    print("Test shape:", df_test.shape)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Data files not found in {input_path}. Please check the path.")
    TARGET = None # Target won't be available if load fails

#%% [markdown]
# ## Apply Initial Feature Engineering

#%% [code]
if 'df_train' in locals(): # Check if data loaded
    print("Applying initial feature engineering...")
    df_train = feature_eng(df_train)
    df_test = feature_eng(df_test)
    print("Initial feature engineering applied.")
else:
    print("Skipping feature engineering as data loading failed.")

#%% [markdown]
# ## Generate Aggregation Features

#%% [code]
if 'df_train' in locals():
    print("Generating aggregation features...")

    numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads']
    base_categorical_cols_agg = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'] 

    categorical_cols_agg = [col for col in base_categorical_cols_agg if col in df_train.columns]
    numerical_cols = [col for col in numerical_cols if col in df_train.columns]

    print(f"Aggregating numerical features {numerical_cols} by categorical features {categorical_cols_agg}")
    
    # Calculate global stats on TRAINING data *before* the loop for filling NaNs later
    global_stats_train = {}
    if TARGET and TARGET in df_train.columns and pd.api.types.is_numeric_dtype(df_train[TARGET]):
        global_stats_train[TARGET] = {'mean': df_train[TARGET].mean(), 'std': df_train[TARGET].std()} # Target stats if needed? Unlikely used here now.
        
    for num_col in numerical_cols:
        if num_col in df_train and pd.api.types.is_numeric_dtype(df_train[num_col]):
            global_stats_train[num_col] = {'mean': df_train[num_col].mean(), 'std': df_train[num_col].std()}
        else: # Handle case where numerical column might be missing after FE?
            global_stats_train[num_col] = {'mean': 0, 'std': 0} 


    temp_target_col = df_train[[TARGET]] if TARGET in df_train.columns else None 
    combined_df = pd.concat([df_train.drop(TARGET, axis=1, errors='ignore'), df_test], sort=False)

    for cat_col in tqdm(categorical_cols_agg, desc="Aggregation Features"):
        for num_col in numerical_cols:
            agg_funcs = ['mean', 'std'] 
            for func in agg_funcs:
                new_col_name = f'{cat_col}_agg_{num_col}_{func}'
                agg_map = combined_df.groupby(cat_col)[num_col].agg(func)
                
                df_train[new_col_name] = df_train[cat_col].map(agg_map)
                df_test[new_col_name] = df_test[cat_col].map(agg_map)

                # --- FIX START ---
                # Ensure the new column is numeric BEFORE filling NaNs
                df_train[new_col_name] = pd.to_numeric(df_train[new_col_name], errors='coerce')
                df_test[new_col_name] = pd.to_numeric(df_test[new_col_name], errors='coerce')

                # Get the pre-calculated global statistic for the original numerical column
                global_stat_value = global_stats_train.get(num_col, {}).get(func, 0) # Default to 0 if not found
                
                # Fill NaNs in the NEWLY CREATED **NUMERICAL** column
                df_train[new_col_name].fillna(global_stat_value, inplace=True) 
                df_test[new_col_name].fillna(global_stat_value, inplace=True) 
                # --- FIX END ---

    print(f"Number of features after aggregations: {df_train.shape[1]}")
    del combined_df, temp_target_col 
    gc.collect()
else:
     print("Skipping aggregations as data not loaded.")

#%% [markdown]
# ## Generate Interaction Features

#%% [code]
if 'df_train' in locals():
    print("Generating interaction features...")

    interaction_base_cols = [
        'Podcast_Name', 'Genre', 'Publication_Day', 
        'Publication_Time', 'Episode_Sentiment', 'Episode_Num' 
    ]
    interaction_base_cols = [col for col in interaction_base_cols if col in df_train.columns]
    print(f"Base columns for interactions: {interaction_base_cols}")

    pair_size = [2] # Focus on 2-way interactions
    interaction_features_names = []

    for r in pair_size:
        for cols in tqdm(list(combinations(interaction_base_cols, r)), desc=f'{r}-way Interactions'):
            new_col_name = '_x_'.join(map(str, cols)) 
            interaction_features_names.append(new_col_name)
            
            df_train[new_col_name] = df_train[list(cols)].astype(str).agg('_'.join, axis=1)
            df_train[new_col_name] = df_train[new_col_name].astype('category') # Set as category
            
            df_test[new_col_name] = df_test[list(cols)].astype(str).agg('_'.join, axis=1)
            df_test[new_col_name] = df_test[new_col_name].astype('category') # Set as category

    print(f"Created {len(interaction_features_names)} interaction features.")
    print(f"Total features after interactions: {df_train.shape[1]}")
    gc.collect()
else:
    print("Skipping interaction features as data not loaded.")

#%% [markdown]
# ## Prepare Data for Model Training

#%% [code]
if 'df_train' in locals() and TARGET in df_train.columns:
    X = df_train.drop(columns=[TARGET])
    y = df_train[TARGET]

    # Align columns after all feature engineering
    missing_cols_in_test = set(X.columns) - set(df_test.columns)
    for c in missing_cols_in_test:
        print(f"Warning: Adding missing column '{c}' to test set, filling with 0.")
        df_test[c] = 0 

    missing_cols_in_train = set(df_test.columns) - set(X.columns)
    if missing_cols_in_train:
         print(f"Warning: Columns in test but not train: {missing_cols_in_train}. Dropping from test.")
         df_test = df_test.drop(columns=list(missing_cols_in_train))

    X_test = df_test[X.columns].copy() # Final alignment

    print(f"Training features shape: {X.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Identify feature types
    target_encode_cols = interaction_features_names # List of interaction features to encode
    
    original_cat_features_for_lgbm = [
         'Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 
         'Episode_Sentiment', 'Episode_Num' 
    ]
    # Ensure these are still categorical after potential earlier processing
    original_cat_features_for_lgbm = [
        col for col in original_cat_features_for_lgbm 
        if col in X.columns and str(X[col].dtype) == 'category' # More robust check for dtype
    ]

    print(f"Target encoding {len(target_encode_cols)} interaction features using category_encoders.")
    print(f"Passing {len(original_cat_features_for_lgbm)} original categorical features directly to LGBM: {original_cat_features_for_lgbm}")

else:
    print("Skipping model preparation as data or target is missing.")

#%% [markdown]
# ## Model Training: LightGBM with KFold CV

#%% [code]
if 'X' in locals() and 'y' in locals(): # Proceed only if X and y were created
    cv = KFold(n_splits=N_SPLITS, random_state=SEED, shuffle=True)

    oof_predictions = np.zeros(len(df_train))
    test_predictions = np.zeros(len(df_test))
    feature_importances = pd.DataFrame()

    lgbm_params = {
        'objective': 'rmse', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'n_estimators': 5000, 'learning_rate': 0.02, 'num_leaves': 1024,
        'max_depth': -1, 'seed': SEED, 'n_jobs': -1, 'verbose': -1,
        'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1,
        'reg_lambda': 0.1, 'max_bin': 1024
    }

    # Training loop
    for fold, (idx_train, idx_valid) in enumerate(cv.split(X, y)):
        print(f"\n===== Fold {fold+1} / {N_SPLITS} =====")
        # Create copies for fold-specific modifications
        X_train, y_train = X.iloc[idx_train].copy(), y.iloc[idx_train].copy()
        X_valid, y_valid = X.iloc[idx_valid].copy(), y.iloc[idx_valid].copy()
        X_test_fold = X_test.copy() 

        # --- Target Encoding using category_encoders ---
        valid_target_encode_cols = [col for col in target_encode_cols if col in X_train.columns]
        if valid_target_encode_cols:
            print(f"Applying category_encoders.TargetEncoder to {len(valid_target_encode_cols)} features for fold {fold+1}...")
            # handle_missing='value': Replaces NaNs introduced before encoding with the global mean
            # handle_unknown='value': Replaces values seen in transform but not fit with the global mean
            # smoothing: Regularizes the encoding based on category frequency (higher value = more regularization)
            encoder = TargetEncoder(
                cols=valid_target_encode_cols, 
                handle_missing='value', 
                handle_unknown='value', 
                smoothing=5.0 # Tune this value; higher means more towards global mean
            ) 
            
            # Fit on Training data
            encoder.fit(X_train, y_train) 
            
            # Transform Train, Validation and Test data
            # Ensure the output columns maintain the correct index
            X_train_encoded = encoder.transform(X_train)
            X_valid_encoded = encoder.transform(X_valid)
            X_test_fold_encoded = encoder.transform(X_test_fold)
            
            # Update the dataframes for the model
            X_train = X_train_encoded
            X_valid = X_valid_encoded
            X_test_fold = X_test_fold_encoded
            
        else:
            print(f"No valid interaction features found for Target Encoding in fold {fold+1}.")
        # --- End Target Encoding ---

        # Define model
        model = lgb.LGBMRegressor(**lgbm_params)
        
        callbacks = [
            lgb.log_evaluation(period=500), 
            lgb.early_stopping(stopping_rounds=100, verbose=False) 
        ]

        # --- Model Fitting ---
        print(f"Fold {fold+1}: Training LightGBM...")
        # Update list of categoricals passed to LGBM based on columns present *after* encoding
        lgbm_cat_features = [col for col in original_cat_features_for_lgbm if col in X_train.columns and str(X_train[col].dtype) == 'category']
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='rmse', 
            categorical_feature=lgbm_cat_features, 
            callbacks=callbacks
        )
        # --- End Model Fitting ---

        # Store predictions
        oof_pred = model.predict(X_valid)
        oof_predictions[idx_valid] = oof_pred
        test_pred = model.predict(X_test_fold)
        test_predictions += test_pred / N_SPLITS 

        # Store importance
        if hasattr(model, 'feature_importances_'):
            fold_importance_df = pd.DataFrame({
                'feature': X_train.columns, # Use columns from the potentially encoded X_train
                'importance': model.feature_importances_, 
                'fold': fold + 1
            })
            feature_importances = pd.concat([feature_importances, fold_importance_df], axis=0)
        
        fold_rmse = mean_squared_error(y_valid, oof_pred, squared=False)
        print(f"Fold {fold+1} OOF RMSE: {fold_rmse:.5f}")

        del X_train, y_train, X_valid, y_valid, X_test_fold, model, encoder
        del X_train_encoded, X_valid_encoded, X_test_fold_encoded # Also delete encoded copies if created
        gc.collect()

    print("\n----- Training Finished -----")

else:
    print("Skipping Model Training as data preparation failed.")

#%% [markdown]
# ## Evaluate OOF Predictions and Create Submission File

#%% [code]
if 'X' in locals() and 'oof_predictions' in locals() and len(oof_predictions) == len(df_train): # Check if training ran
    # Calculate overall OOF RMSE
    overall_oof_rmse = mean_squared_error(y, oof_predictions, squared=False)
    print(f"\nOverall OOF RMSE: {overall_oof_rmse:.5f}")

    # Create submission file
    df_subm['Listening_Time_minutes'] = test_predictions
    
    submission_filename = 'submission.csv'
    df_subm.to_csv(submission_filename)

    print(f"\nSubmission file '{submission_filename}' created successfully.")
    print("Submission file head:")
    print(df_subm.head())
else:
    print("Skipping OOF evaluation and submission file creation.")

#%% [markdown]
# ## Feature Importances

#%% [code]
if not feature_importances.empty:
    print("\nAggregated Feature Importances (Top 50):")
    # Check for duplicate column names potentially introduced by encoding/merging issues
    if feature_importances['feature'].duplicated().any():
         print("Warning: Duplicate feature names found in importance calculation. Grouping by name.")
         
    mean_importances = feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
    print(mean_importances.head(50))

    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 15)) 
        top_n = 50 
        plot_data = mean_importances.head(top_n)
        
        if not plot_data.empty:
            sns.barplot(x=plot_data.values, y=plot_data.index)
            plt.title(f'Top {top_n} Feature Importances (Averaged Across Folds)')
            plt.xlabel('Mean Importance Score'); plt.ylabel('Features')
            plt.tight_layout(); plt.show() 
        else:
            print("No importance data to plot.")
            
    except ImportError:
        print("\nMatplotlib/Seaborn not found. Skipping feature importance plot.")
else:
    print("\nNo feature importance data generated.")

print("\nScript finished.")

