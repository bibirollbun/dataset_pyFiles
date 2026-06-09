!pip install autogluon
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label="Listening_Time_minutes").fit("/kaggle/input/playground-series-s5e4/train.csv")
predictions = predictor.predict("/kaggle/input/playground-series-s5e4/test.csv")


from autogluon.tabular import TabularPredictor
import pandas as pd

test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# ðŸ“¦ Format for submission
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': predictions
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved: submission.csv")


