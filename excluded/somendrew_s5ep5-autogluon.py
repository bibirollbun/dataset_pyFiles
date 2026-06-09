# !pip install --upgrade scikit-learn



# !pip install autogluon
!uv pip install --system --quiet scikit-learn==1.5.2 autogluon





from autogluon.tabular import TabularPredictor

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error



import warnings
warnings.filterwarnings("ignore")



train_main = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

train_df = train_main.drop_duplicates()

print(train_df.shape),print(test.shape)

train_df['Calories'] = np.log1p(train_df['Calories'])



from autogluon.core.metrics import make_scorer


predictor = TabularPredictor(
    problem_type= 'regression',
    label='Calories',
    eval_metric='root_mean_squared_error',
    verbosity = 0
).fit(
    train_df,
    presets="best_quality",
    hyperparameters='default',
    excluded_model_types=['KNN', 'NN_TORCH']
)



predictor.fit_summary(show_plot = True)


# predictor.feature_importance(train_df)


models = predictor.model_names()
models


ag_preds = predictor.predict(test)
preds_inverted = np.expm1(ag_preds)



submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission['Calories'] = preds_inverted
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv!")




