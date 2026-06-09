# Install AutoGluon
!pip install -q ray==2.10.0
!pip install autogluon.tabular
!pip install -U ipywidgets 


# Import packages
import pandas as pd 
import warnings
import shutil
from autogluon.tabular import TabularDataset, TabularPredictor
warnings.filterwarnings("ignore")


# Read data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')

train_df.head()


train_df = train_df.drop(['id'] , axis=1)
test_id = test_df['id']
test_df = test_df.drop(columns=['id'], axis=1)


train_df.shape


# Check duplicates
train_df.duplicated().sum()


# Check null values
train_df.isnull().sum()


train_df = TabularDataset(train_df)
test_df = TabularDataset(test_df)


label = 'Price'
TIME_LIMIT = 3600 * 11


predictor = TabularPredictor(
    label=label,
    eval_metric='rmse',
    problem_type="regression"
)



predictor.fit(
    train_data=train_df,
    time_limit=TIME_LIMIT,
       verbosity=3,
    presets='best_quality',
    ag_args_fit={
        'num_gpus': 1
    }
)
results = predictor.fit_summary()
print(results)


predictor.leaderboard()


test_preds = predictor.predict(test_df)


importances = predictor.feature_importance(train_df)
print("Feature importances:")
print(importances.head(20))


sub = pd.DataFrame ({'id': test_id,'Price': test_preds})
sub.to_csv('submission.csv', index=False)
print(sub.head(10))

