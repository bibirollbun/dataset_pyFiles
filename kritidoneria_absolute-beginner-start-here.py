!pip install pycaret

import warnings
warnings.filterwarnings("ignore")
import gc

# Data manipulation
import pandas as pd
import numpy as np

# Machine learning
from sklearn.metrics import mean_squared_error
import os


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# Save IDs for submission
test_ids = test['id']

# Drop 'id' column
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

# Shuffle train data for randomness
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

train.head()


from pycaret.regression import *


reg = setup(data = train, 
             target = 'accident_risk',
            # pca=True,
            # remove_multicollinearity=True,
            session_id = 2025,
            fold=3,
            # feature_selection=True,
            # remove_outliers=True,
            # outliers_threshold=0.05,
            # polynomial_features=True,
            # normalize=True,
            # n_jobs =2,
            use_gpu=True)



gc.collect()


# compare_models(sort='RMSE')


blender = tune_model(blend_models(compare_models(n_select = 3,include=['xgboost','catboost','knn','rf'])))
# blender = tune_model(blend_models(compare_models(n_select = 3)))
# blender = tune_model(stack_models(compare_models(n_select = 3,include=['lightgbm','xgboost','catboost','knn','rf'])))


test_predictions = predict_model(blender, data = test)
test_predictions.head()


# Prepare submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': test_predictions['prediction_label']
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
submission.head()




