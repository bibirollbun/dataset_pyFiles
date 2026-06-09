!pip install feature-engine "scikit-learn<1.7" --quiet


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import os


warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")#.drop("id", axis=1)
train.head()


train.describe()


def stateless_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Ratio_Alone_to_Outside"] = df.Time_spent_Alone / df.Going_outside
    df["Ratio_SocialEvent_to_Outside"] = df.Social_event_attendance / df.Going_outside
    df["Ratio_Post_to_Outside"] = df.Post_frequency / df.Going_outside
    df["Ratio_FriendSize_to_SocialEvent"] = df.Friends_circle_size / df.Social_event_attendance
    df["Ratio_FriendSize_to_Post"] = df.Friends_circle_size / df.Post_frequency
    df["Ratio_Post_to_Alone"] = df.Post_frequency / df.Time_spent_Alone	
    df.replace(np.inf, np.nan, inplace=True)
    return df
    
train = stateless_preprocess(train)


train.eq(np.inf).sum()


target = "Personality"

X = train.drop(target, axis=1)
y = train[target]

categorical = X.select_dtypes(object).columns.tolist()
numbers = X.select_dtypes("number").columns.tolist()


from feature_engine.encoding import WoEEncoder, CountFrequencyEncoder, OrdinalEncoder, MeanEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn import set_config
from feature_engine.imputation import AddMissingIndicator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from feature_engine.imputation import DropMissingData


set_config(transform_output="pandas")


class CategoricalCombiner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X.iloc[:, 0].astype(str) + '_' + X.iloc[:, 1].astype(str))
        elif isinstance(X, np.ndarray):
            return (X[:, 0].astype(str) + '_' + X[:, 1].astype(str)).reshape(-1, 1)
        else:
            raise TypeError("Input should be a pandas DataFrame or NumPy array")

class DropMissingColumns(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5): # Drop columns with more than 'threshold' missing values
        self.threshold = threshold
        self.columns_to_drop = []

    def fit(self, X, y=None):
        missing_percentage = X.isnull().sum() / len(X)
        self.columns_to_drop = missing_percentage[missing_percentage > self.threshold].index.tolist()
        return self

    def transform(self, X):
        print(self.columns_to_drop)
        return X.drop(columns=self.columns_to_drop)


obj_col = make_column_selector(dtype_include="object")
add_1 = FunctionTransformer(lambda col: col + 1)
missing = AddMissingIndicator()
imp_cat = SimpleImputer(strategy="constant", fill_value="Missing")
imp_num = SimpleImputer(strategy="constant", fill_value=99)
woe = WoEEncoder(fill_value=0)
count = CountFrequencyEncoder("count")
frequency = CountFrequencyEncoder("frequency")
ordinal = OrdinalEncoder()
mean = MeanEncoder()

add_1_ct = make_column_transformer(
    (add_1, make_column_selector(dtype_include="number")),
    remainder="passthrough",
    n_jobs=-1,
    verbose_feature_names_out=False,
    force_int_remainder_cols=False,

) 

imputing1 = make_column_transformer(
    (imp_cat, obj_col),
    (imp_num, make_column_selector(dtype_include="number")),
    remainder="passthrough",
    n_jobs=-1,
    verbose_feature_names_out=True,
    force_int_remainder_cols=False,
)
imputing2 = make_column_transformer(
    (imp_cat, obj_col),
    (imp_num, make_column_selector(dtype_include="number")),
    remainder="passthrough",
    n_jobs=-1,
    verbose_feature_names_out=True,
    force_int_remainder_cols=False,
)
cat_encoding = make_column_transformer(
    (woe, obj_col),
    (count, obj_col),
    (frequency, obj_col),
    (ordinal, obj_col),
    (mean, obj_col),
    remainder="passthrough",
    n_jobs=-1
)

cat_combo = make_column_transformer(
    (CategoricalCombiner(), obj_col),
    remainder="passthrough",
)
replace_inf = make_column_transformer(
    (FunctionTransformer(lambda x: x.replace([-np.inf, np.inf], 99)), make_column_selector(dtype_include="number")),
    remainder="passthrough",
)

models = [
    (
        "ExtraTreesClassifier",
        ExtraTreesClassifier(
            250, 
            n_jobs=-1,
            max_features="sqrt",
            class_weight="balanced_subsample"
        )
    ),
    (
        "RandomForestClassifier",
        RandomForestClassifier(
            250, 
            n_jobs=-1,
            max_features="sqrt",
            class_weight="balanced_subsample"
        )
    ),
    (
        "LogisticRegression",
        LogisticRegression(
            C=0.1, 
            n_jobs=-1,
            random_state=42, 
            max_iter=250, 
            class_weight="balanced"
        )
    ),
    
]
pipelines = {
    name: 
    make_pipeline(
        add_1_ct,
        missing,
        imputing1,
        cat_combo,
        cat_encoding,
        replace_inf,
        imputing2, 
        MinMaxScaler(), 
        model
    )
for name, model in models}


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

folds = 5
cv = StratifiedKFold(folds, shuffle=True, random_state=42)
label = LabelEncoder().fit(y)
y_encode = label.transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# y_train = label.transform(y_train)
# y_test = label.transform(y_test)

# pipeline.fit(X_train, y_train)


from tqdm import tqdm_notebook


scores = dict()
scores_calibrated = dict()
for name, pipe in tqdm_notebook(list(pipelines.items())):
    score = cross_val_score(pipe, X, y_encode, cv=cv, n_jobs=-1, error_score="raise")
    scores[name] = score
    
    calibrated = CalibratedClassifierCV(pipe, method='isotonic', n_jobs=-1)
    score_calibrated = cross_val_score(calibrated, X, y_encode, cv=cv, n_jobs=-1)
    scores_calibrated[name] = score_calibrated


score_df = pd.DataFrame(scores)
score_calibrated_df = pd.DataFrame(scores_calibrated)


best_original = score_df.mean().sort_values()[:1]
best_original


best_calibrated = score_calibrated_df.mean().sort_values()[:1]
best_calibrated


if best_original.values[0] < best_calibrated.values[0]:
    print("Best model is calibrated", best_calibrated.index[0])
    model = CalibratedClassifierCV(pipelines[best_calibrated.index[0]], method="isotonic", n_jobs=-1)
else:
    print("Best model is uncalibrated", best_original.index[0])
    model = pipelines[best_original.index[0]]

model.fit(X, y_encode)


from sklearn.model_selection import TunedThresholdClassifierCV

threshold_model = TunedThresholdClassifierCV(
    estimator=model,
    cv=cv,
    scoring='neg_log_loss',
    response_method='predict_proba',
    random_state=42,
    thresholds=200
)
threshold_model.fit(X, y_encode)


best_threshold = threshold_model.best_threshold_
best_threshold


from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))
ConfusionMatrixDisplay.from_estimator(model, X, y_encode, ax=axes[0], cmap="Greens", colorbar=False)
ConfusionMatrixDisplay.from_estimator(threshold_model, X, y_encode, ax=axes[1], cmap="Greens", colorbar=False)

axes[0].set_title("Pre tuned threshold model")
axes[1].set_title("Post tuned threshold model")


cr_pre = classification_report(label.inverse_transform(y_encode), label.inverse_transform(model.predict(X)), digits=4)
cr_post = classification_report(label.inverse_transform(y_encode), label.inverse_transform(threshold_model.predict(X)), digits=4)

print("--------------------------")
print("Pre tuned model threshold")
print("--------------------------")
print(cr_pre)
print("--------------------------")
print("Post tuned model threshold")
print("--------------------------")
print(cr_post)


pretune_accuracy = accuracy_score(y_encode, model.predict(X))
posttune_accuracy = accuracy_score(y_encode, threshold_model.predict(X))
use_pre = pretune_accuracy > posttune_accuracy

if use_pre: 
    final_model = threshold_model
else:
    final_model = model


print("pretune_accuracy            :", pretune_accuracy)
print("posttune_accuracy           :", posttune_accuracy)
print("Use pretuned model threshold?", use_pre)


sub = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sub_id = pd.DataFrame(sub.pop("id"))
sub = stateless_preprocess(sub)
sub_id["Personality"] = label.inverse_transform(final_model.predict(sub))
sub_id.head()


sub_id.to_csv("submission.csv", index=False)

