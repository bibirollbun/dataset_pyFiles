!pip install autogluon


import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor



import pandas as pd

train = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
test = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')

print(train.head())
print(train.info())



# Drop ID or plate if not useful for learning
train_data = train.drop(columns=['id', 'plate'])
test_data = test.drop(columns=['id', 'plate'])

# Make sure target column is present in train
label = 'price'

# AutoGluon will handle dtypes like category and datetime



from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)



predictor = TabularPredictor(
    label=label,
    problem_type='regression',  # ðŸ‘ˆ Force it to treat this as regression
    eval_metric='mean_absolute_error'  # or use 'root_mean_squared_error'
).fit(
    train_df,
    time_limit=600,
    presets='best_quality'
)



val_preds = predictor.predict(val_df)



def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

smape_score = smape(val_df[label].values, val_preds.values)
print("Validation SMAPE:", smape_score)



# Predict on the actual test set
test_preds = predictor.predict(test_data)

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'price': test_preds
})

submission.to_csv('submission.csv', index=False)


