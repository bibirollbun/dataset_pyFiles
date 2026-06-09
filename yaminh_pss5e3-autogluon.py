import pandas as pd


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')


train.head()


train.isna().sum()


!pip install ray==2.10.0


!pip install autogluon.tabular --no-cache-dir -q
!pip install -U ipywidgets


from autogluon.tabular import TabularPredictor
from autogluon.common import space
import warnings

warnings.simplefilter("ignore")

predictor = TabularPredictor(
    path='/kaggle/working/Autogluon',
    problem_type='binary',  # Explicit classification
    eval_metric='roc_auc',         # Classification metric
    label='rainfall',         # New categorical label
    verbosity=2
)

predictor.fit(
    train_data=train,
    time_limit=3600 * 5,
    presets='best_quality',
    excluded_model_types=['KNN'],  # Removed RF
    ag_args_fit={'num_cpus': 4}
)


# Check class labels (critical step!)
print("Class labels:", predictor.class_labels)  # Will show e.g., [0, 1] or ['0', '1']


predictor.leaderboard(silent=True).style.background_gradient(subset=['score_val'], cmap='viridis')


predictor = TabularPredictor.load("/kaggle/working/Autogluon")


# Get probability predictions
positive_class = predictor.class_labels[1]  # Get the name of the positive class
probability_predictions = predictor.predict_proba(test)[positive_class]


probability_predictions


submission =  pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission.head()


# Create submission
submission = pd.DataFrame({
    'id': submission['id'],  # Use the 'id' COLUMN
    'rainfall': probability_predictions.values  # .values avoids index alignment issues
})


submission.head()


submission.to_csv('submission.csv', index=False)

