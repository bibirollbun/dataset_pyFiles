# basic processing libraries
import numpy as np
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn methods
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

# gradient boosting
import xgboost

# hyperparameter tuning
import optuna

print(f"XGBoost version: {xgboost.__version__}")


!ls /kaggle/input/


!ls /kaggle/input/playground-series-s5e8


train_csv_path = "/kaggle/input/playground-series-s5e8/train.csv"
test_csv_path = "/kaggle/input/playground-series-s5e8/test.csv"


labeled_dataframe = pd.read_csv(train_csv_path, index_col="id")
test_dataframe = pd.read_csv(test_csv_path, index_col="id")

labeled_dataframe.head()


labeled_dataframe.nunique().sort_values(ascending=False)


def merge_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    # life status
    dataframe["socioeconomic_status"] = dataframe["job"] + " " + dataframe["education"]
    dataframe["age_group"] = pd.cut(dataframe["age"],
                                    bins=[0, 25, 45, 65, 100],
                                    labels=["young", "adult", "middle-age", "senior"])
    dataframe["age_marital"] = dataframe["age_group"].astype(str) + " " + dataframe["marital"]
    # dataframe["status"] = dataframe["job"] + " " + dataframe["marital"]

    # contact-related information
    dataframe["last_contact_info"] = dataframe["contact"] + " " + dataframe["day"].astype(str) + " " + dataframe["month"]
    dataframe["previously_contacted"] = (dataframe["pdays"] != -1).astype(int)
    dataframe["previous_contact_outcome"] = dataframe["previous"].astype(str) + " " + dataframe["poutcome"]
    # dataframe["total_contacts"] = dataframe["campaign"] + dataframe["previous"]

    # credit and loan information
    # dataframe["credit_info"] = dataframe["default"] + " " + dataframe["housing"] + " " + dataframe["loan"]

    dataframe.drop(columns="age_group", inplace=True)
    
    return dataframe


labeled_dataframe = merge_columns(labeled_dataframe)
test_dataframe = merge_columns(test_dataframe)


labeled_features = labeled_dataframe.drop(columns="y")
labels = labeled_dataframe["y"]

train_data, validation_data, train_labels, validation_labels = train_test_split(labeled_features, labels, test_size=0.2)

print(f"Train data shape: {train_data.shape}, validation data shape: {validation_data.shape}")
print(f"Train labels shape: {train_labels.shape}, validation labels shape: {validation_labels.shape}")


label_distribution = labeled_dataframe["y"].value_counts(normalize=True)

plt.bar(x=label_distribution.index.astype(str),
        height=label_distribution)
plt.xlabel("Labels", fontsize=13)
plt.ylabel("Distribution", fontsize=13);


correlations_wrt_label = labeled_dataframe.select_dtypes("number").corr()["y"]

correlations_wrt_label


plt.barh(y=correlations_wrt_label.index,
         width=correlations_wrt_label)

plt.title("Column correlations wrt label column")
plt.xlabel("Correlation Coefficient", fontsize=13)
plt.ylabel("Columns", fontsize=13);


labeled_dataframe["job"][labeled_dataframe["y"] == 1].value_counts(ascending=True).plot(kind="barh");

plt.title("Number of subscriptions wrt jobs")
plt.xlabel("Number of people that subscribed to the term deposit", fontsize=13, labelpad=17)
plt.ylabel("People's jobs", fontsize=13)
plt.tight_layout();


labeled_dataframe["job"][labeled_dataframe["y"] == 0].value_counts(ascending=True).plot(kind="barh");

plt.title("Number of non-subscribed people wrt their jobs")
plt.xlabel("Number of people that didn't subscribe", fontsize=13, labelpad=17)
plt.ylabel("People's jobs", fontsize=13)
plt.tight_layout();


labeled_dataframe["age_marital"].value_counts(normalize=True, ascending=True).plot(kind="barh")

plt.title("Most common age-marital status")
plt.xlabel("Distribution of counts", fontsize=13)
plt.ylabel("Age-marital status", fontsize=13);


labeled_dataframe[labeled_dataframe["y"] == 1]["age_marital"].value_counts(normalize=True, ascending=True).plot(kind="barh")

plt.title("Age-marital status VS having a term deposit subscription")
plt.xlabel("Counts distribution", fontsize=13)
plt.ylabel("Age-marital status", fontsize=13);


# tuned_xgb_params = {'n_estimators': 2950,
#  'learning_rate': 0.2176681093411927,
#  'max_depth': 7,
#  'subsample': 0.9384205711303508,
#  'colsample_bytree': 0.7607465770513036}
# gradient_booster = xgboost.XGBClassifier(**tuned_xgb_params, device="cuda")

gradient_booster = xgboost.XGBClassifier(
    n_estimators=2700,
    learning_rate=3e-1,
    max_depth=5,
    subsample=0.93,
    colsample_bytree=0.76,
    device="cuda"
)

gradient_booster


def train_and_predict(booster,
                     train_data,
                     train_labels,
                     validation_data,
                     validation_labels,
                     test_dataframe) -> None:
    
    pipeline = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        StandardScaler(with_mean=False),
        booster
    )
    pipeline.fit(train_data, train_labels)
    
    validation_predictions = pipeline.predict_proba(validation_data)[:, 1]
    validation_roc_auc_score = roc_auc_score(validation_labels, validation_predictions)
    print(f"Validation ROC AUC score: {validation_roc_auc_score:.4f}")

    test_predictions = pipeline.predict_proba(test_dataframe)[:, 1]
    submission_dataframe = pd.DataFrame({"y": test_predictions},
                                        index=test_dataframe.index)

    submission_filename = "submission.csv"
    submission_dataframe.to_csv(submission_filename)
    print(f"Submission file created. Name: {submission_filename}")


train_and_predict(
    booster=gradient_booster,
    train_data=train_data,
    train_labels=train_labels,
    validation_data=validation_data,
    validation_labels=validation_labels,
    test_dataframe=test_dataframe
)

