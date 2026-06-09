!pip install -q autogluon


# Import Library
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor


def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new




# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")



# Define numerical features
numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]



# Add cross terms to capture feature interactions
train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)




# Prepare training data with log-transformed target
train_data = train.copy()
train_data['log_Calories'] = np.log1p(train_data['Calories'])
train_data = train_data.drop(columns=['id', 'Calories'])




# Prepare test data, keeping 'id' for submission
test_data = test.copy()





# Initialize AutoGluon TabularPredictor
predictor = TabularPredictor(
    label='log_Calories',
    problem_type='regression',
    eval_metric='root_mean_squared_error',
    path='autogluon_model'
)



# Fit the model with bagging for robustness
predictor.fit(
    train_data,
    presets='best_quality',
    num_bag_folds=5,
    time_limit=7200  # Increase Time for better output
)

# Generate predictions on test data
log_preds = predictor.predict(test_data)

# Transform predictions back to original scale and clip
preds = np.expm1(log_preds)
preds = np.clip(preds, 1, 314)

# Submission file
submission['Calories'] = preds
submission.to_csv('submission.csv', index=False)

