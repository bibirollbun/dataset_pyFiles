# Import libraries
import warnings
warnings.filterwarnings("ignore")
!pip install --upgrade scikit-learn
!pip install --upgrade seaborn

# data manipulation
import numpy as np
import pandas as pd

# machine learning libraries
import sklearn as skl
import lightgbm as lgb
import xgboost as xgb

# reuse my kaggle tabular data functions
import urllib.request

url = "https://raw.githubusercontent.com/2awesome-rob/iron_fungi/main/my_kaggle_functions.py"
urllib.request.urlretrieve(url, "my_kaggle_functions.py")
import my_kaggle_functions as mkf


PATH = "/kaggle/input/playground-series-s5e11/"
print(f"Home path is: {PATH}")

DEVICE, CORES  = mkf.set_globals(verbose = True)
XY, features, targets, target = mkf.load_tabular_data(PATH)


XY[target] = XY[target].astype(int)

mkf.summarize_data(XY, targets)
mkf.plot_target_eda(XY, target, title = f'{target} distribution')


mkf.summarize_data(XY, features)


XY=mkf.clean_strings(XY, features, string_length=3)


mkf.plot_features_eda(
    XY, features, target, high_label="paid", low_label="default"
)


training_features = [f for f in XY.columns if f not in targets and
                    XY[f].dtype != "category"]

XY = mkf.get_feature_interactions(XY, training_features)


mapper = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5}
XY['grade'] = XY['grade_subgrade'].str[0].map(mapper)

XY['loan_to_income'] = XY['loan_amount'] / XY['annual_income']


training_features = ['annual_income','debt_to_income_ratio', 'loan_to_income']
XY = mkf.get_transformed_features(
    XY, training_features, skl.preprocessing.PowerTransformer())

training_features = ['credit_score','interest_rate']
XY = mkf.get_transformed_features(
    XY, training_features, skl.preprocessing.StandardScaler())

training_features = ['loan_amount']
XY = mkf.get_transformed_features(
    XY, training_features, skl.preprocessing.RobustScaler())


training_features = [f for f in XY.columns if f not in targets and
                    XY[f].dtype != "category"]

XY = mkf.get_embeddings(
    XY, training_features, skl.decomposition.PCA(n_components=8), "pca_", verbose=True, target=target)


training_features = [f for f in XY.columns if f not in targets and
                     XY[f].dtype == "category"]

XY = pd.get_dummies(XY, columns=training_features, drop_first = True)


model = lgb.LGBMClassifier(verbose = -1, n_jobs=CORES)
TASK = "probability_roc_auc"


### Split to train and validate
training_features = [f for f in XY.columns if f not in targets]
X_train, y_train, X_val,  y_val, X_test, y_test = mkf.split_training_data(
    XY, training_features, target, validation_size = 0.2)

feature_importance = mkf.get_feature_importance(
    X_train, X_val, y_train, y_val, task=TASK)

important_features = [f for f in feature_importance.index if feature_importance[f] != 0]


trained_model, _ = mkf.train_and_score_model(
    X_train[important_features], X_val[important_features], y_train, y_val,
    model = model, task = TASK, verbose = True)


predictions = mkf.submit_predictions(
    X_test[important_features], y_test, target, [trained_model], task =TASK, path=PATH)

