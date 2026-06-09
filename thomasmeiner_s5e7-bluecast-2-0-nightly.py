!python --version
!pip uninstall bluecast --y --q
!pip install --q --no-deps --find-links=/kaggle/input/bluecast-nightly  bluecast==2.0.0
!pip uninstall scikit-learn --y --q
!pip install scikit-learn==1.4.0 --q
!pip install dash --q


import numpy as np 
import pandas as pd 
from bluecast.eda.analyse import create_eda_dashboard_classification


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
target = "Personality"
train.sample(5, random_state=54)


train.info()


from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import CatboostTuneParamsConfig, TrainingConfig
from bluecast.ml_modelling.catboost import CatboostModel
from bluecast.experimentation.tracking import ExperimentTracker


# Create and configure CatboostModel
train_config = TrainingConfig()
train_config.cat_encoding_via_ml_algorithm = True # this is usually False as Xgboost is default
train_config.hyperparameter_tuning_rounds = 200 # reduced for illustration purposes
train_config.calculate_shap_values = False
train_config.autotune_on_device = "gpu"

catboost_config = CatboostTuneParamsConfig()

# set up an experiment tracker: by default it would be temporary, here we let it save to Kaggle
experiment_tracker = ExperimentTracker(db_path="/kaggle/working/experiment_tracker.duckdb")

bluecast = BlueCastCV(
        class_problem="binary",
        ml_model=CatboostModel(
            class_problem="binary",
        ),
        conf_xgboost=catboost_config,
        conf_training=train_config,
        experiment_tracker=experiment_tracker
    )


help(bluecast.fit_eval)


bluecast.fit_eval(train, target_col=target)


predicted_probas, predicted_classes = bluecast.predict(test)


bluecast.experiment_tracker.get_hyperparameter_results()


bluecast.experiment_tracker.get_evaluation_results()


sample_submission[target] = predicted_classes
sample_submission.to_csv('submission.csv', index=False)
sample_submission

