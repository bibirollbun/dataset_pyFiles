!pip install fedot -q


import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv(r'/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e5/test.csv')

TARGET = 'Calories'


automl = Fedot(
    problem='regression',
    preset='best_quality',
    timeout=60,
    with_tuning=True,
    n_jobs=-1,
    seed=42,
    initial_assumption=PipelineBuilder() \
    .add_node('scaling')
    .add_branch('catboostreg', 'xgboostreg', 'lgbmreg')
    .join_branches('ridge')
    .build(),
)


automl.fit(features=train, target=TARGET)


y_pred = automl.predict(test)


submission = pd.read_csv(r'/kaggle/input/playground-series-s5e5/sample_submission.csv')


submission[TARGET] = y_pred


submission.to_csv('submission.csv', index=False)




