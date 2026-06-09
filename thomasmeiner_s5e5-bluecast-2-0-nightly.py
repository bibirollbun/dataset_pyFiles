!pip install scikit-learn --upgrade -q


%%capture
!pip install bluecast --find-links=file:/kaggle/input/bluecast-nightly/bluecast-2.0.0-py3-none-any.whl


import numpy as np 
import pandas as pd 
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression


train = pd.read_csv(r'/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e5/sample_submission.csv')

target = "Calories"

train[target] = np.log1p(train[target])


# this is taken from here: https://www.kaggle.com/code/jiaoyouzhang/calorie-only-xgboost/notebook
numerical_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)


automl = BlueCastCVRegression(
    class_problem="regression", 
)
automl.conf_training.autotune_on_device = "gpu"
automl.conf_training.hypertuning_cv_repeats = 2

automl.fit(train.copy(), target_col=target)
y_preds = automl.predict(test)
submission[target] = y_preds
submission[target] = submission[target].clip(0)
submission[target] = np.expm1(submission[target])
submission.to_csv("submission.csv", index=False)
submission

