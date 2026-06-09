!pip install autogluon
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def weighted_mae(y_true, y_pred, weights):
    """
    Calculate Weighted Mean Absolute Error
    """
    return np.sum(np.abs(y_true - y_pred) * weights) / np.sum(weights)



# Read data
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
test_weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')


train = train.merge(inventory, on=['unique_id', 'warehouse'], how='left')
test = test.merge(inventory, on=['unique_id', 'warehouse'], how='left')
train = train.merge(calendar, on=['date', 'warehouse'], how='left')
test = test.merge(calendar, on=['date', 'warehouse'], how='left')


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
train['sales'] = train['sales'].fillna(train['sales'].mean())


for df in [train, test]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)


if 'availability' in train.columns:
    train = train.drop('availability', axis=1)
if 'availability' in test.columns:
    test = test.drop('availability', axis=1)


categorical_columns = ['warehouse', 'name', 'L1_category_name', 'L2_category_name', 'L3_category_name']
for col in categorical_columns:
    if col in train.columns:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)



def custom_wmae(y_true, y_pred, sample_weight=None):
    """
    Custom WMAE scorer for model selection
    Returns negative WMAE (negative because AutoGluon tries to maximize metrics)
    """
    return -1 * weighted_mae(y_true, y_pred, sample_weight if sample_weight is not None else np.ones_like(y_true))


from autogluon.core.metrics import make_scorer
custom_wmae_scorer = make_scorer(
    name='custom_wmae',
    score_func=custom_wmae,
    optimum=0,
    greater_is_better=True,  # Because we return negative WMAE
)


predictor = TabularPredictor(
    label='sales',
    eval_metric=custom_wmae_scorer,  # Use the registered scorer
    path='ag_models'
)



# Sample training data
train_sample = train.sample(frac=1.0, random_state=42)

# Fit the model
predictor.fit(
    train_sample,
    time_limit=3600*10,
    verbosity=2,
    excluded_model_types=['KNN'],
    ag_args_fit={'num_gpus': 2})



try:
    print("\n=== Model Performance Evaluation ===")
    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard)
except Exception as e:
    print(f"Error in leaderboard evaluation: {str(e)}")

try:
    print("\n=== Feature Importance ===")
    feature_importance = predictor.feature_importance(data=train_sample)
    print(feature_importance)
except Exception as e:
    print(f"Error in feature importance calculation: {str(e)}")




# Make predictions
predictions = predictor.predict(test)



# Create submission file with ID combining unique_id and date
test['date_str'] = test['date'].dt.strftime('%Y-%m-%d')
submission = pd.DataFrame({
    'id': test['unique_id'].astype(str) + '_' + test['date_str'],
    'sales_hat': predictions
})


try:
    # Calculate WMAE for validation set using test weights distribution
    unique_ids_in_test = test_weights['unique_id'].unique()
    val_weights = np.where(
        train_sample['unique_id'].isin(unique_ids_in_test),
        train_sample['unique_id'].map(test_weights.set_index('unique_id')['weight'].to_dict()).fillna(1),
        1
    )

    val_pred = predictor.predict(train_sample)
    wmae_score = weighted_mae(
        train_sample['sales'].values,
        val_pred.values,
        val_weights
    )
    print(f"\nValidation WMAE: {wmae_score:.4f}")
except Exception as e:
    print(f"Error in validation WMAE calculation: {str(e)}")

try:
    print("\n=== Validation Metrics by Model ===")
    for model_name in predictor.get_model_names():
        try:
            model_pred = predictor.predict(train_sample, model=model_name)
            model_wmae = weighted_mae(
                train_sample['sales'].values,
                model_pred.values,
                val_weights
            )
            print(f"{model_name} WMAE: {model_wmae:.4f}")
        except Exception as e:
            print(f"Error in calculating WMAE for model {model_name}: {str(e)}")
            continue
except Exception as e:
    print(f"Error in model-wise validation: {str(e)}")



# Save predictions
submission.to_csv('submission.csv', index=False)

# Print final model information
try:
    print("\n=== Model Information ===")
    print("Features used:", train.columns.tolist())
    print("Number of training samples:", len(train_sample))
    print("Number of test samples:", len(test))
except Exception as e:
    print(f"Error in printing model information: {str(e)}")

