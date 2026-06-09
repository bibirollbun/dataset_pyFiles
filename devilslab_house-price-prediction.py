import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna



#getting the dataset
def load_data(file_path:str ,dataset_name='DataSet'):
    try:
        df=pd.read_csv(file_path,index_col='id')
        print(f"\n---{dataset_name} Inspection ---")
        
        #Display the rows
        print(f'\n{dataset_name} - First 5 Rows ===')
        print(df.head(5))
        
        #Display the shape of the data
        print(f'\n{dataset_name} Shape : {df.shape}')
        
        #display the information about the dataset
        print(f'\n {dataset_name} Info: ')
        print(df.info())
        
        #displaying the basic statistics of the dataset
        print(f'\n {dataset_name} Description:')
        print(df.describe())
        
        #checking the missing values present in the dataset
        print(f"\n{dataset_name}  Missing Values: ")
        print(df.isnull().sum())
        
        return df
    except FileNotFoundError:
        print(f"Error: file {file_path} not found !.Please recheck the file path...")
        return None
    except Exception as e:
        print(f"Error Loading '{dataset_name}': {str(e)}")
        return None


#Loading and Inspecting the dataset

train_path='/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv'
test_path='/kaggle/input/prediction-interval-competition-ii-house-price/test.csv'

train_df = load_data(train_path ,dataset_name='Training Data')

test_df=load_data(test_path,dataset_name='Test Data')



def inspect_dataset(train_df:pd.DataFrame,test_df:pd.DataFrame,target_column='sale_price'):
    '''
        perform EDA on the house price dataset

        Parameters: 
            train_df(pd.DataFrame): Training dataset with features and target
            test_df(pd.DataFrame): Test Dataset about target.
            target_column(str):Target Column Name, default 'Sale_price'.
            
    '''

    #Target Distribution
    plt.figure(figsize=(10,6))
    sns.histplot(train_df[target_column],kde=True,bins=50)
    plt.title('Distribution of Sale Price')
    plt.xlabel('Sale Price')
    plt.ylabel('Count')
    plt.show()

    #printing the summary statistics for target column [sale price]
    print(f'\nSale Price Summary Statisrtics: ')
    print(train_df[target_column].describe())

    #claculating the skewness to check if sale_price is skewed
    skewness=train_df[target_column].skew()
    print(f"\nSale Price Skewness: {skewness: .4f} (Positive skew > 0,Consider LOG TRANSFORMATION if high")

    #Missing Values
    try:
        print("\nMissing Values in Train (%):")
        missing_train = train_df.isnull().sum()[train_df.isnull().sum() > 0] / len(train_df) * 100
        if missing_train.empty:
            print("No missing values in train dataset.")
        else:
            print(missing_train)
    except:
        print("Error computing missing values in train dataset.")
    
    try:
        print("\nMissing Values in Test (%):")
        missing_test = test_df.isnull().sum()[test_df.isnull().sum() > 0] / len(test_df) * 100
        if missing_test.empty:
            print("No missing values in test dataset.")
        else:
            print(missing_test)
    except:
        print("Error computing missing values in test dataset.")

    #Correlation with numerical features
    numerical_cols=train_df.select_dtypes(include=['int64','float64']).columns
    cor_matrix=train_df[numerical_cols].corr()

    #Correlation with Numerical Features
    plt.figure(figsize=(12,8))
    sns.heatmap(cor_matrix,cmap='coolwarm',annot=False)
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()


    #let us print top 5 related features with sale_price
    print(f"\n Top 5 Features realated with Sale Price:")
    top_corr=cor_matrix[target_column].sort_values(ascending=False).head(5)
    print(top_corr)

    #CATEGORICAL FEATURE ANALYSIS
    categorical_cols=train_df.select_dtypes(include=['object','category']).columns
    print("\nCategoroical Features - Unique Values:")
    for col in categorical_cols:
        print(f"{col}: {train_df[col].nunique()} unique values")
        print(train_df[col].value_counts().head())


    #key distribution
    key_features=['area','year_built','beds','bath_full']
    for feature in key_features:
        plt.figure(figsize=(10,8))
        sns.scatterplot(x=train_df[feature],y=train_df[target_column])
        plt.title(f"Sale Price vs {feature}")
        plt.xlabel(feature)
        plt.ylabel('Sale Price')
        plt.show()


inspect_dataset(train_df, test_df, target_column='sale_price')


def preprocess_data(train_df:pd.DataFrame , test_df:pd.DataFrame , target_column='sale+price'):
    '''
        we are going to :
            1.preprocess train and test dataset
            2.handle the missing values
            3.encodeing categoricals and
            4.scaling numericals

        Parameters:
            train_df(pd.DataFrame):Training Dataset(200,000 roes , 45 features , sale_price).
            test_df(pd.DataFrame):Test Dataset (similar features , no sale price)
            target_column(str):Target column, default 'sale_price'(int 64),
            
    '''

    #printing column names for debugging
    print(f"Train Data Column: {train_df.columns.tolist()}")
    print(f"Test Data Column: {test_df.columns.tolist()}")

    #validating the target column
    if target_column not in train_df.columns:
        raise ValueError(f"Target Column '{target_column}' not found in training dataðŸ˜±.\nAvailaible olumns: {train_df.columns.tolist()}")


    #Detecting numerical and categorical columns
    numerical_cols=train_df.select_dtypes(include=['int64','float64']).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    categorical_cols=train_df.select_dtypes(include=['object','category']).columns


    #Handling missing values
    for col in numerical_cols:
        try:
            median_value=train_df[col].median()
            train_df[col].fillna(median_value,inplace=True)
            test_df[df].fillna(median_value,inplace=True)
        except:
            print(f"No missing values in numerical columns: {col}")

    for col in categorical_cols:
        try:
            if col in ['subdivision']:
                train_df[col].fillna('None',inplace=True)
                test_df[col].fillna('None',inplace=True)
            else:
                mode_value=train_df[col].mode()[0]
                train_df[col].fillna(mode_value,inplace=True)
                test_df[col].fillna(mode_value,inplace=True)
        except:
            print(f"No missing values in categorical columns: {col}")

    x_train=train_df.drop(columns=[target_column])
    y_train=train_df[target_column]


    skewness=y_train.skew()
    if skewness>1:
        print(f"Log-Tranforming {target_column} due to high skewness ({skewness:.4f})")
        y_train=np.log1p(y_train)

    x_test=test_df.copy()


    #let's define preproseccing pipeline
    low_cardinality_cols=[col for col in categorical_cols if train_df[col].nunique()<10]
    high_cardinality_cols=[col for col in categorical_cols if train_df[col].nunique()>=10]

    preprocessor=ColumnTransformer(
        transformers=[
            #numerical pipeline:Scale Features
            ('num',Pipeline([
                ('scaler',StandardScaler())
            ]),numerical_cols),
            #for low cardinality categorical we will be using One-hot encode
            ('cat_low',Pipeline([
                ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
                
            ]),low_cardinality_cols),

            #for high cardinality categorical we will use Target encode
            ('cat_high',Pipeline([
                ('encoder',TargetEncoder(smoothing=1.0))
            ]),high_cardinality_cols)
        ])
    #fit and transform
    x_train_preprocessed=preprocessor.fit_transform(x_train,y_train)
    x_test_preprocessed=preprocessor.transform(x_test)



    #getting feature names after preprocessign
    feature_names=(
        numerical_cols +
        list(preprocessor.named_transformers_['cat_low']['encoder'].get_feature_names_out(low_cardinality_cols)) +
        high_cardinality_cols
    )
    # Convert to DataFrame
    x_train_preprocessed = pd.DataFrame(x_train_preprocessed, columns=feature_names, index=x_train.index)
    x_test_preprocessed = pd.DataFrame(x_test_preprocessed, columns=feature_names, index=x_test.index)

    # Check for NaN values post-transformation
    if x_train_preprocessed.isna().any().any():
        print("Warning: NaN values found in x_train_preprocessed. Imputing with median.")
        x_train_preprocessed = x_train_preprocessed.fillna(x_train_preprocessed.median())
    if x_test_preprocessed.isna().any().any():
        print("Warning: NaN values found in x_test_preprocessed. Imputing with median.")
        x_test_preprocessed = x_test_preprocessed.fillna(x_train_preprocessed.median())

    print(f"\nPreprocessed Training Data Shape: {x_train_preprocessed.shape}")
    print(f"Preprocessed Test Data Shape: {x_test_preprocessed.shape}")
    print("\nPreprocessed Training Data Sample:")
    print(x_train_preprocessed.head())
    
    return x_train_preprocessed, x_test_preprocessed, y_train, preprocessor


x_train, x_test, y_train, preprocessor = preprocess_data(train_df, test_df, target_column='sale_price')


import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna

def train_and_evaluate_models(x_train, y_train, x_test, k_folds=5):
    """
    Train regression models, evaluate with RMSE, and estimate prediction intervals.

    Parameters:
        x_train (pd.DataFrame): Preprocessed training features.
        y_train (pd.Series): Target (sale_price, possibly log-transformed).
        x_test (pd.DataFrame): Preprocessed test features.
        k_folds (int): Number of CV folds, default=5.
    """
    # Initialize models
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBRegressor': XGBRegressor(random_state=42)
    }
    cv_scores = {}
    test_predictions = {}
    test_intervals = {}

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
            x_train_fold = x_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            x_val_fold = x_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]
    
            model.fit(x_train_fold, y_train_fold)
            y_pred = model.predict(x_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            fold_scores.append(rmse)
            print(f"{model_name} - Fold {fold+1} RMSE: {rmse:.4f}")
    
        cv_scores[model_name] = np.mean(fold_scores)
        print(f"{model_name} - Mean CV RMSE: {cv_scores[model_name]:.4f} Â± {np.std(fold_scores):.4f}")
        
        # Train on full data
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        test_predictions[model_name] = test_pred
        
        # Prediction intervals: Mean Â± 2*std of CV residuals
        residuals = []
        for train_idx, val_idx in kf.split(x_train):
            model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            y_pred_val = model.predict(x_train.iloc[val_idx])
            residuals.extend(y_train.iloc[val_idx] - y_pred_val)
        std_residuals = np.std(residuals)
        test_intervals[model_name] = {
            'lower': test_pred - 2 * std_residuals,
            'upper': test_pred + 2 * std_residuals
        }
    
    # Tune XGBRegressor with Optuna
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        xgb = XGBRegressor(**param, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(x_train):
            xgb.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            y_pred = xgb.predict(x_train.iloc[val_idx])
            scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], y_pred)))
        return np.mean(scores)
    
    print("\nTuning XGBRegressor with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print(f"Best XGBRegressor Parameters: {study.best_params}")
    
    tuned_xgb = XGBRegressor(**study.best_params, random_state=42)
    tuned_xgb.fit(x_train, y_train)
    test_predictions['Tuned_XGBRegressor'] = tuned_xgb.predict(x_test)
    residuals = []
    for train_idx, val_idx in kf.split(x_train):
        tuned_xgb.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
        y_pred_val = tuned_xgb.predict(x_train.iloc[val_idx])
        residuals.extend(y_train.iloc[val_idx] - y_pred_val)
    std_residuals = np.std(residuals)
    test_intervals['Tuned_XGBRegressor'] = {
        'lower': test_predictions['Tuned_XGBRegressor'] - 2 * std_residuals,
        'upper': test_predictions['Tuned_XGBRegressor'] + 2 * std_residuals
    }
    cv_scores['Tuned_XGBRegressor'] = study.best_value
    
    return models, cv_scores, test_predictions, test_intervals


models, cv_scores, test_predictions, test_intervals = train_and_evaluate_models(
    x_train, y_train, x_test, k_folds=5
)


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def create_submission(test_df, test_predictions, test_intervals, log_transformed=False, model_name='Tuned_XGBRegressor'):
    """
    Create submission file with id, pi_lower, and pi_upper for the competition.

    Parameters:
        test_df (pd.DataFrame): Original test dataset (for index or id column).
        test_predictions (dict): Model predictions for x_test.
        test_intervals (dict): Prediction intervals (lower, upper) for x_test.
        log_transformed (bool): Whether y_train was log-transformed in Step 3.
        model_name (str): Model to use for submission (default: Tuned_XGBRegressor).
    """
    # Validate model name
    if model_name not in test_predictions:
        raise ValueError(f"Model {model_name} not found in test_predictions. Available: {list(test_predictions.keys())}")
    
    # Get predictions and intervals
    predictions = test_predictions[model_name]
    intervals = test_intervals[model_name]
    
    # Reverse log-transformation if applied
    if log_transformed:
        print("Reversing log-transformation for predictions and intervals...")
        predictions = np.expm1(predictions)
        intervals['lower'] = np.expm1(intervals['lower'])
        intervals['upper'] = np.expm1(intervals['upper'])

    # Create submission DataFrame
    # Check if test_df has an 'id' column; otherwise, use index
    if 'id' in test_df.columns:
        submission = pd.DataFrame({
            'id': test_df['id'],
            'pi_lower': intervals['lower'],
            'pi_upper': intervals['upper']
        })
    else:
        submission = pd.DataFrame({
            'id': test_df.index,
            'pi_lower': intervals['lower'],
            'pi_upper': intervals['upper']
        })

    # Ensure non-negative bounds (house prices can't be negative)
    submission['pi_lower'] = submission['pi_lower'].clip(lower=0)
    submission['pi_upper'] = submission['pi_upper'].clip(lower=0)

    # Save submission file
    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")
    print("Submission head:")
    print(submission.head())

    # Evaluate interval quality on validation set
    def evaluate_intervals(x_train, y_train, model, log_transformed):
        x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        model.fit(x_tr, y_tr)
        val_pred = model.predict(x_val)
        residuals = y_val - val_pred
        std_residuals = np.std(residuals)
        val_intervals = {
            'lower': val_pred - 2 * std_residuals,
            'upper': val_pred + 2 * std_residuals
        }
        if log_transformed:
            val_pred = np.expm1(val_pred)
            val_intervals['lower'] = np.expm1(val_intervals['lower'])
            val_intervals['upper'] = np.expm1(val_intervals['upper'])
            y_val = np.expm1(y_val)
        # Coverage: % of true values within intervals
        coverage = np.mean((y_val >= val_intervals['lower']) & (y_val <= val_intervals['upper']))
        # Mean interval width
        width = np.mean(val_intervals['upper'] - val_intervals['lower'])
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"\nValidation Metrics for {model_name}:")
        print(f"Coverage (90% target): {coverage:.4f}")
        print(f"Mean Interval Width: {width:.2f}")
        print(f"Validation RMSE: {rmse:.4f}")
        return coverage, width, rmse

    # Run evaluation if model is available
    if model_name in models:
        coverage, width, rmse = evaluate_intervals(x_train, y_train, models[model_name], log_transformed)
    else:
        print(f"Model {model_name} not found for interval evaluation.")

    return submission


submission = create_submission(test_df, test_predictions, test_intervals, log_transformed=True, model_name='Tuned_XGBRegressor')




