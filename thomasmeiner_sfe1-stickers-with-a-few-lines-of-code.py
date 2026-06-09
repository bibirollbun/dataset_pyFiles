from IPython.core.display import HTML

# Define custom CSS directly in Python variable
custom_css = """
<style>
  :root {
    --header1_color: #204709;
    --header2_color: #42841F;
    --header3_color: #6EAF4B;
    --keyword_color: #cc241d; /* import */
    --string_color: #79740e;
    --number_color: #b16286;
    --def_color: #689d6a; /* class name */
    --property_color: #458588; /* python properties */
    --builtin_color: #689d6a;
    --comment_color: #9f9f9f;
    --comment_color_2: #458588; /* equals sign */
    --operator_color: #a221f2;
    --font_color: #3c3836; /* general font */
    --variable2_color: #b16286; /*self keyworda */
    --box_color: #fffdee; /* Remove opacity */
  }

  /* Add the following style for headers with background color */
  h1,
  .h1 {
    font-family: "Trebuchet MS", sans-serif;
    font-size: 2em !important;
    letter-spacing: 1px;
    color: var(--header1_color);
    border-bottom: 3px solid var(--header1_color);
    background-color: #000080;
    padding: 0.5em;
    color: #ffff00 !important;
  }

  h2,
  .h2 {
    font-family: "Trebuchet MS";
    font-size: 1.7em !important;
    color: var(--header2_color);
    background-color: #000080;
    padding: 0.5em;
    color: #ffff00 !important;
  }

  h3,
  .h3 {
    font-family: "Trebuchet MS";
    font-size: 1.4em !important;
    color: var(--header3_color);
    background-color: #000080;
    padding: 0.5em;
    color: #ffff00 !important;
  }

  /* Rest of your existing styles... */

  body[data-jp-theme-light="true"] .jp-Notebook .CodeMirror.cm-s-jupyter {
    background-color: var(--box_color) !important;
  }

  div.input_area {
    background-color: var(--box_color) !important;
  }
</style>
"""

# Apply custom CSS
HTML(custom_css)


%%capture
!pip install optuna-integration --no-index --find-links=file:/kaggle/input/optuna-integration/optuna_integration-3.6.0-py3-none-any.whl


#!pip install bluecast --no-index --find-links=file:/kaggle/input/bluecast/bluecast-1.6.4-py3-none-any.whl


%%capture
!pip install bluecast


import itertools
import numpy as np
import pandas as pd
import re
from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from sklearn.model_selection import train_test_split


target = "num_sold"

# Data Loading
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")

# naive target filling
train[target] = train[target].fillna(0)


automl = BlueCastCVRegression(class_problem="regression")
automl.conf_training.autotune_on_device = "cpu"

debug = False
DO_ERROR_ANALYSIS = True


if debug:
    automl.conf_training.autotune_model = False
    automl.conf_training.calculate_shap_values = False
    train = train.sample(1000, random_state=80).reset_index(drop=True)
else:
    automl.conf_training.hypertuning_cv_repeats = 1
    automl.conf_training.hyperparameter_tuning_max_runtime_secs = 60 * 60 * 1
    automl.conf_training.calculate_shap_values = False
    automl.conf_training.hyperparameter_tuning_rounds = 10


if not DO_ERROR_ANALYSIS:
    automl.fit(train.copy(), target_col=target)
else:
    automl.conf_training.out_of_fold_dataset_store_path = "/kaggle/working/" # only when using fit_eval afterwards
    automl.fit_eval(train.copy(), target_col=target)


#probs = automl.predict_p_values(test)
y_hat = automl.predict(test)
y_hat


submission[target] = y_hat
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: 'submission.csv'")
submission

