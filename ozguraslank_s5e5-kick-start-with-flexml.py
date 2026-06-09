import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip3 install flexml


from flexml import Regression
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df.head()


reg = Regression(
    train_df,
    target_col="Calories",
    normalize="minmax_scaler",      # Default is None
    encoding_method="label_encoder" # There is only 'Sex' column, label_encoder is enough. Default is 'onehot_encoder'
)


# FlexML warned about ID column, let's drop it
reg = Regression(
    train_df,
    target_col="Calories",
    normalize="minmax_scaler",     
    encoding_method="label_encoder",
    drop_columns=["id"]
)


reg.start_experiment(experiment_size="quick", eval_metric="RMSE")
# Since FlexML doesn't support 'RMSLE' metric, let's go with 'RMSE'
# 'quick' mode only runs most-used and fastest models, 'wide' runs all available models


reg.tune_model(tuning_method="optuna") # default is 'randomized_search'


preds = reg.predict(test_df)
print(preds)


preds_df = test_df[['id']]
preds_df['Calories'] = pd.DataFrame(preds, columns=["Calories"])
preds_df.head()


preds_df.to_csv("submission.csv", index=False)
# LB 0.05842

