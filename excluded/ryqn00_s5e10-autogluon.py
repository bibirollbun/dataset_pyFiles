%%capture
!pip install autogluon.tabular scikit-learn==1.5.2 "ray>=2.10.0,<2.45.0"


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from autogluon.tabular import TabularPredictor

import warnings
warnings.filterwarnings('ignore')


# LOAD DATA
categorical_features = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']
train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e10/train.csv').drop(columns = 'id')
train_df = pd.get_dummies(train_df, columns=categorical_features, prefix_sep='_', drop_first=False)
test_df  = pd.read_csv(r'/kaggle/input/playground-series-s5e10/test.csv').drop(columns = 'id')
test_df = pd.get_dummies(test_df, columns=categorical_features, prefix_sep='_', drop_first=False)


# AUTOGLUON

# DEFINE AUTOGLUON
predictor = TabularPredictor(label = 'accident_risk',
                         problem_type = 'regression',
                         eval_metric = 'rmse')

# TRAIN AUTOGLUON
predictor.fit(train_df,
              presets = 'extreme',
              num_cpus = 1,
              verbosity = 2,
              ag_args_fit={'num_gpus': 1})


predictor.leaderboard()


# CHECKING BEST MODEL 

best_model = predictor.model_best

print(f'Best Model : {best_model}')


predictor.feature_importance(train_df)


# CHECK SUBMISSION

# TEST DATA PREDICTION
y_pred = predictor.predict(test_df)

submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')

submission['accident_risk'] = y_pred

submission


# GET OOF (OUT-OF-FOLD) PREDICTION

# GET OOF
oof_predictions = predictor.predict_oof()

# CONVERT TO DATAFRAME
y_pred = oof_predictions.to_frame(name = 'oof_prediction')  # ---> RETURN DATAFRAME
oof_df = pd.DataFrame(y_pred)

oof_df


# SAVE PREDICTED DATA
submission.to_csv(r'submission.csv', index = False)
oof_df.to_csv(r'oof_data_autogluon1.csv', index = False)

