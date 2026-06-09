from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


original_df = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


numerical_features = [x for x in train_df.describe().columns if x != "id"]
categorical_features = [x for x in train_df.columns if x not in numerical_features and x != "id"]
numerical_features, categorical_features


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def univariate_feature_plots(feature_name):
    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature_name], kde = True, bins = 30)
    plt.title(f"Histogram of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x = train_df[feature_name])
    plt.title(f"Box plot of {feature_name}")

    plt.tight_layout()
    plt.show()

    print(f"Statistics: for {feature_name}", end = '\n')
    print(f"Skewness: {train_df[feature_name].skew():.2f}")
    print(f"Number of missing values: {train_df[train_df[feature_name].isnull()].shape[0]}")


for feature in numerical_features:
    univariate_feature_plots(feature)


def plot_distribution_of_categorical_feature(feature_name):
    counts = train_df[feature_name].value_counts() ## 
    plt.figure(figsize = (10, 10))
    plt.subplot(2, 1, 1)
    plt.pie(counts, labels = counts.index, autopct = "%1.2f%%")
    plt.title(f"Distribution of {feature_name}")
    plt.axis("equal")

    plt.subplot(2, 1, 2)
    sns.countplot(x = feature_name, data = train_df)
    plt.show()

    print(f"Number of unique {feature_name} values: {counts.shape[0]}")
    print(f"Missing Values in {feature_name}: {train_df[train_df[feature_name].isnull()].shape[0]}")


target_feature = categorical_features[-1]
for cat in categorical_features[:-1]: # except target.
    plot_distribution_of_categorical_feature(cat)


numerical_df = train_df[numerical_features]
sns.pairplot(numerical_df, corner = True, plot_kws={'alpha': 0.5})
plt.suptitle('Pairwise Scatter Plots', y = 1.02)
plt.show()


def plot_heatmap_and_distribution_of_categorical_features(feature_name):
    plt.figure(figsize = (12, 12))
    plt.subplot(2, 1, 1)
    sns.countplot(x = feature_name, hue = target_feature, data = train_df)
    plt.title(f"Distribution of {target_feature} across {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.xticks(rotation = 45)
    plt.legend(title = target_feature, bbox_to_anchor = (1.05, 1), loc = "upper left")
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    sns.heatmap(pd.crosstab(train_df[feature_name], train_df[target_feature]), annot = True, fmt = "d")
    plt.title(f"{feature_name} vs. {target_feature} (Counts)")
    plt.ylabel(feature_name)
    plt.xlabel(target_feature)
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 0)
    plt.tight_layout()
    plt.show()


for feature in categorical_features[:-1]:
    plot_heatmap_and_distribution_of_categorical_features(feature)


## one-hot encoding of the categorical features.
oe = OrdinalEncoder()
train_df[categorical_features[:-1]] = oe.fit_transform(train_df[categorical_features[:-1]])
test_df[categorical_features[:-1]] = oe.transform(test_df[categorical_features[:-1]])

## transform the label as well.
target_feature = categorical_features[-1]
le = LabelEncoder()
train_df[target_feature] = le.fit_transform(train_df[target_feature])
train_df.head()


from scipy.stats import wasserstein_distance

def collate_wasserstein_distance(feature_name):
    matrix = np.zeros((len(le.classes_), len(le.classes_)))
    for i in range(len(le.classes_)):
        data_i = train_df[train_df[target_feature] == i][feature_name]
        for j in range(len(le.classes_)):
            data_j = train_df[train_df[target_feature] == j][feature_name]
            if j == i:
                continue
            distance_ij = wasserstein_distance(data_i, data_j)
            matrix[i][j] = distance_ij
            matrix[j][i] = distance_ij
    
    return pd.DataFrame(matrix, index = le.classes_, columns = le.classes_)


for feature in numerical_features:
    df = collate_wasserstein_distance(feature)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=le.classes_, yticklabels=le.classes_)
    
    plt.title(f"Wasserstein Distance Heatmap for {feature}")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.show()


def predict_and_score_xgboost_multi(model):
    y_pred_probs = model.predict_proba(test_x)
    top3_probs = np.argsort(y_pred_probs, axis = 1)[:, -3:][:, ::-1]
    get_best_and_full_accuracy_xgboost(test_y, top3_probs)

    return top3_probs


def _get_score(actual, predicted):
    score = 0.0
    hits = 0
    seen = set()
    for i, pred in enumerate(predicted):
        if pred == np.int64(actual) and pred not in seen:
            hits += 1
            score += hits / (i + 1.0)
            seen.add(pred)
    
    return score ## since actual is ONE entity.


def get_best_and_full_accuracy_xgboost(ptest_y, topk_probs):
    test_yl = ptest_y.tolist()
    first_acc_l = [x for idx, x in enumerate(test_yl) if np.int64(x) == topk_probs[idx][0]]
    score_accl_l = [_get_score(x ,topk_probs[idx]) for idx, x in enumerate(test_yl)]

    print(f"First accuracy: {(len(first_acc_l) / len(test_yl)):.2f}") ## only based on the highest class.
    print(f"Score accuracy: {(np.mean(score_accl_l)):.2f}") ## based on the function defined earlier.


full_y, full_x = train_df[target_feature], train_df.drop(columns = [target_feature])
train_x, test_x, train_y, test_y = train_test_split(full_x, full_y, test_size = 0.2, random_state = 42, stratify = full_y)


model = XGBClassifier(
    objective = "multi:softprob",
    num_class = len(np.unique(train_y)),
    learning_rate = 0.045,
    max_depth = 7,
    colsmaple_bytree = 0.6,
    colsample_bylevel = 0.8,
    subsample = 0.8
)

model.fit(train_x, train_y)


stratified_top3_probs = predict_and_score_xgboost_multi(model)


min_class_count = train_df[target_feature].value_counts().min()
balanced_df = train_df.groupby(target_feature).sample(n=min_class_count, random_state=42)
bfull_y, bfull_x = balanced_df[target_feature], balanced_df.drop(columns = [target_feature])
btrain_x, btest_x, btrain_y, btest_y = train_test_split(bfull_x, bfull_y, test_size = 0.2, random_state = 42, stratify = bfull_y)


model_b = XGBClassifier(
    objective = "multi:softprob",
    num_class = len(np.unique(btrain_y)),
    learning_rate = 0.045,
    max_depth = 7,
    colsmaple_bytree = 0.6,
    colsample_bylevel = 0.8,
    subsample = 0.8
)

model_b.fit(btrain_x, btrain_y)


balanced_top3_probs = predict_and_score_xgboost_multi(model_b)


from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(top3_probs, ty):
    y_pred = [x[0] for x in top3_probs]
    y_true = [np.int64(x) for x in ty.to_list()]
    cmat = confusion_matrix(y_true, y_pred)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names = le.classes_))

    plt.figure(figsize = (6, 5))
    sns.heatmap(cmat, annot = True, fmt = 'd', xticklabels = le.classes_, yticklabels = le.classes_, cmap = "Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


## Atleast one of the labels is right == Eager.
def plot_eager_confusion_matrix(top3_probs, ty):
    y_true = [np.int64(x) for x in test_y.to_list()]
    y_pred = []
    for idx, ans in enumerate(y_true):
        flag = 0
        for pans in top3_probs[idx]:
            if pans == ans:
                y_pred.append(pans)
                flag = 1
                break
        if flag == 0:
            y_pred.append(top3_probs[idx][0])
    cmat = confusion_matrix(y_true, y_pred)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names = le.classes_))

    plt.figure(figsize = (6, 5))
    sns.heatmap(cmat, annot = True, fmt = 'd', xticklabels = le.classes_, yticklabels = le.classes_, cmap = "Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


plot_confusion_matrix(stratified_top3_probs, test_y)
plot_eager_confusion_matrix(stratified_top3_probs, test_y)


plot_confusion_matrix(balanced_top3_probs, test_y)
plot_eager_confusion_matrix(balanced_top3_probs, test_y)


train_df_append = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv").drop(columns = ["id"])
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv").drop(columns = ["id"])
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

train_df.columns, test_df.columns, train_df_append.columns


train_df = pd.concat([train_df, train_df_append], ignore_index = True)
train_df.tail(10)


## one-hot encoding of the categorical features.
oe = OrdinalEncoder()
train_df[categorical_features[:-1]] = oe.fit_transform(train_df[categorical_features[:-1]])
test_df[categorical_features[:-1]] = oe.transform(test_df[categorical_features[:-1]])
for col in categorical_features[:-1]:
    train_df[col] = train_df[col].astype("category")
    test_df[col] = test_df[col].astype("category")
## transform the label as well.
target_feature = categorical_features[-1]
le = LabelEncoder()
train_df[target_feature] = le.fit_transform(train_df[target_feature])
train_df.head()


full_y, full_x = train_df[target_feature], train_df.drop(columns = [target_feature])
train_x, test_x, train_y, test_y = train_test_split(full_x, full_y, test_size = 0.2, random_state = 42, stratify = full_y)


model_xg_multi = XGBClassifier(
    objective = "multi:softprob",
    num_class = len(np.unique(train_y)),
    n_estimators = 2000,
    learning_rate = 0.05,
    max_depth = 12,
    colsmaple_bytree = 0.467,
    early_stopping_rounds = 100,
    reg_alpha = 2.7,
    reg_lambda = 1.4,
    gamma = 0.26,
    enable_categorical = True,
    tree_method = 'hist',
    max_delta_step = 4,
    subsample = 0.86,
    random_state = 13,
    device = "cuda"
)
model_xg_multi.fit(train_x, train_y, eval_set = [(test_x, test_y)], verbose = 0)


top3_xg_multi = predict_and_score_xgboost_multi(model_xg_multi)


def create_balanced_splits():
    train_splits = {}
    for t in range(len(le.classes_)):
        pos_idx = train_y[train_y == t].index
        pos_samples = train_x.loc[pos_idx]
        pos_labels = train_y.loc[pos_idx]

        n_pos = len(pos_idx)
        neg_idx = train_y[train_y != t].index
        neg_x = train_x.loc[neg_idx]
        neg_y = train_y.loc[neg_idx]
        
        neg_x_sampled, _, neg_y_sampled, _ = train_test_split(neg_x, neg_y, train_size = n_pos, stratify = neg_y, random_state = 42)
        binary_x = pd.concat([pos_samples, neg_x_sampled])
        binary_y = pd.Series([1] * n_pos + [0] * n_pos, index = binary_x.index)

        ## shuffle.
        binary_x = binary_x.sample(frac = 1, random_state = 42)
        binary_y = binary_y.loc[binary_x.index]

        train_splits[t] = (binary_x, binary_y)
    
    return train_splits


train_splits = create_balanced_splits()


models_xg_bins = {}
for t_x, (btx, bty) in train_splits.items():
    model_tx = XGBClassifier(
        objective="binary:logistic",
        n_estimators = 2000,
        learning_rate = 0.05,
        max_depth = 4,
        colsmaple_bytree = 0.467,
        reg_alpha = 2.7,
        reg_lambda = 1.4,
        gamma = 0.26,
        enable_categorical = True,
        tree_method = 'hist',
        max_delta_step = 4,
        subsample = 0.86,
        random_state = 13,
        device = "cuda",
        eval_metric="logloss"     # Optional but recommended for binary classification
    )

    model_tx.fit(btx, bty)
    models_xg_bins[t_x] = model_tx


## Convert the predictions to the right consumable format.
def interpret_predictions(predictions, k = 3):
    final_pred_array = np.zeros((test_x.shape[0], len(le.classes_)))
    for t_x, pred_arr in predictions.items():
        final_pred_array[:, t_x] = predictions[t_x][:, 1]
    topk_probs = np.argsort(final_pred_array, axis = 1)[:, -k:][:, ::-1]
    return topk_probs

def predict_and_score_xgboost_bins(models_dict):
    predictions = {}
    for t_x, model_tx in models_dict.items():
        predictions[t_x] = model_tx.predict_proba(test_x)
    top3_probs = interpret_predictions(predictions)
    get_best_and_full_accuracy_xgboost(test_y, top3_probs)    

    return top3_probs


top3_xg_bins = predict_and_score_xgboost_bins(models_xg_bins)


plot_confusion_matrix(top3_xg_bins, test_y)
plot_eager_confusion_matrix(top3_xg_bins, test_y)


test_x.head(), test_df.head()


sample_predictions = {}
for t_x, model_tx in models_xg_bins.items():
    sample_predictions[t_x] = model_tx.predict_proba(test_df)


pred_array = np.zeros((test_df.shape[0], len(le.classes_)))
for t_x, pred_arr in sample_predictions.items():
    pred_array[:, t_x] = sample_predictions[t_x][:, 1]
sample_top_probs = np.argsort(pred_array, axis = 1)[:, -3:][:, ::-1]
sample_top_probs.shape, sample_top_probs[0]


sample_top_prob_labels = [[le.classes_[cl] for cl in prob_array] for prob_array in sample_top_probs]
sample_top_prob_labels[0], len(sample_top_prob_labels)


sample_submission_df.columns


sample_submission_df[target_feature] = [" ".join(k) for k in sample_top_prob_labels]
sample_submission_df.head()


sample_submission_df.to_csv("submission.csv", index = False)




