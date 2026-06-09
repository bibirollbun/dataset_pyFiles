# BlueCast 3.0 – install and environment
!python --version
!pip uninstall bluecast -y -q
!pip install -q --no-deps --find-links=/kaggle/input/bluecast-nightly bluecast==3.0.0
!pip uninstall scikit-learn -y -q
!pip install scikit-learn==1.4.0 -q
!pip install dash -q


import numpy as np
import pandas as pd


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
target = "y"
train.sample(5, random_state=54)


train.info()


from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig
from bluecast.ensemble.ensemble_config import EnsembleConfig
from bluecast.experimentation.tracking import ExperimentTracker


experiment_tracker = ExperimentTracker(db_path="/kaggle/working/experiment_tracker.duckdb")

# More folds + repeats = more diverse base models for hill climbing to select from
training_config = TrainingConfig(
    autotune_on_device="gpu",
    out_of_fold_dataset_store_path="/kaggle/working/",
    bluecast_cv_train_n_model=(5, 2),  # 5 folds x 2 repeats = 10 models
)

ensemble_config = EnsembleConfig(
    ensemble_strategy="hill_climbing",
    hc_blending_method="rank",         # rank-transform normalises different model scales
    hc_allow_negative_weights=True,    # allows anti-correlated models to improve diversity
    hc_weight_min=-0.3,
    hc_weight_max=0.5,
    hc_weight_step=0.01,              # fine-grained weight search
    hc_tolerance=1e-7,                 # keep adding models as long as there's any improvement
)

bluecast_pipeline = BlueCastCV(
    class_problem="binary",
    experiment_tracker=experiment_tracker,
    conf_training=training_config,
    ensemble_config=ensemble_config,
)


bluecast_pipeline.fit_eval(train, target_col=target)


# Inspect which models hill climbing selected and their weights
if bluecast_pipeline.hill_climbing_ensemble:
    hc = bluecast_pipeline.hill_climbing_ensemble
    print(f"Hill climbing selected {len(hc.selected_indices)} out of "
          f"{len(bluecast_pipeline.bluecast_models)} fold models\n")
    for step in hc.history:
        print(f"  Step {step['iteration']:2d}: {step['model']:<20s} "
              f"weight={step['weight']:+.4f}  score={step['score']:.6f}")


# Predict uses the hill climbing weights automatically
predicted_probas, predicted_classes = bluecast_pipeline.predict(test)


bluecast_pipeline.experiment_tracker.get_hyperparameter_results()


bluecast_pipeline.experiment_tracker.get_evaluation_results()


from bluecast.evaluation.error_analysis import ErrorAnalyserClassificationCV

analyser_cv = ErrorAnalyserClassificationCV(bluecast_pipeline)
loaded_data = analyser_cv.read_data_from_bluecast_cv_instance()
loaded_data


analyser_cv.analyse_segment_errors()


sample_submission[target] = predicted_probas
sample_submission.to_csv("submission.csv", index=False)
sample_submission

