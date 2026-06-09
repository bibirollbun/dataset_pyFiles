from IPython.display import clear_output


!pip install klib pingouin smltk==3.0.0
clear_output(wait=True)


import klib
import os
import pandas as pd
from smltk.data_analysis import DataAnalysis
from smltk.data_processing import DataProcessing
import warnings
clear_output(wait=True)


# RAW_DATA = "../data/raw/hepatic-horizon-multi-class-cirrhosis" # local
# os.makedirs(RAW_DATA, exist_ok=True)
RAW_DATA = "/kaggle/input/hepatic-horizon-multi-class-cirrhosis-outcome-pre" # Kaggle
# RAW_DATA = "/content/sample_data/" # Colab

da = DataAnalysis()
dp = DataProcessing()
warnings.filterwarnings("ignore")


test = pd.read_csv(f"{RAW_DATA}/test.csv")
test


train = pd.read_csv(f"{RAW_DATA}/train.csv")
train


features = da.get_eda("Status", train.drop(columns=["id"]), {"sample.frac": 1})


features.keys()


for feature in features.keys():
    if feature == "data_missing":
        display(pd.DataFrame(
                features[feature].values(),
                columns=[feature],
                index=features[feature].keys()
            ).sort_values(feature).round(2)
        )
    else:
        print(feature, features[feature])


[categorical_features, train] = dp.transform_categories(train)
[categorical_features, test] = dp.transform_categories(test, categorical_features)


klib.corr_mat(train.drop(columns="id"))


klib.corr_plot(train.drop(columns="id"))


features = da.get_eda("Status", train.drop(columns=["id"]), {"sample.frac": 1})
clear_output(wait=True)


features

