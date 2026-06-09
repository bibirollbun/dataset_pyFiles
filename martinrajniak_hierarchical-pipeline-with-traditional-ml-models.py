import os, gc
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier

import xgboost as xgb

import matplotlib.pyplot as plt

from IPython.display import display, Latex

np.random.seed(42)

# MAP@K metric (K=3 per competition)
def apk(actual, predicted, k=3):
    """Average precision at k for a single observation.
    `actual` is a single-element list [true_label] (competition has one truth per row).
    `predicted` is an ordered list of labels (top-k predictions).
    """
    if not actual:
        return 0.0
    if k < len(predicted):
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p == actual[0]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            break  # only one relevant label per row
    return score

def mapk(actual_list, predicted_list, k=3):
    return np.mean([apk([a], p, k) for a, p in zip(actual_list, predicted_list)])

def topk_from_proba(class_labels, proba_row, k=3):
    idx = np.argsort(-proba_row)[:k]
    return [class_labels[i] for i in idx]

print("Libraries loaded.")


# Detect Kaggle input path
KAGGLE_INPUT = "/kaggle/input/map-charting-student-math-misunderstandings"
LOCAL_INPUT = "../input/map-charting-student-math-misunderstandings"
HERE = os.getcwd()

if os.path.exists(KAGGLE_INPUT):
    DATA_DIR = KAGGLE_INPUT
elif os.path.exists(LOCAL_INPUT):
    DATA_DIR = LOCAL_INPUT
else:
    # Fallback to working directory (for local testing with provided CSVs)
    DATA_DIR = "data"

print("DATA_DIR =", DATA_DIR)


train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))


print(train_df.shape)
train_df.head(3)


print(test_df.shape)
test_df.head(3)


group_by_question_df = train_df[["QuestionId", "MC_Answer"]].groupby("QuestionId")["MC_Answer"].agg(set).reset_index()
group_by_question_df


# Verify that all questions have exactly 4 possible answers
(group_by_question_df["MC_Answer"].apply(len) == 4).all()


# Mathematical expression are correctly displayed with Latex function only when wrapped in $ sign
def to_latex(text):
    return text.replace('\\(', '$').replace('\\)', '$')


for index, question_row in group_by_question_df.iterrows():
    question = train_df.loc[train_df.QuestionId == question_row.QuestionId].iloc[0].QuestionText
    clean_question = to_latex(question)
    display(Latex(f"Question {question_row.QuestionId}: {clean_question}"))

    # Letter to multi-choice answer relationship doesn't have to be correct
    answer_letters = "ABCD"
    answers = " ".join([f"({answer_letters[i]}) {to_latex(answer)}" for i, answer in enumerate(list(question_row.MC_Answer))])
    display(Latex(f"Answers: {answers}"))

    # Newline
    print()


train_df["Misconception"] = train_df["Misconception"].fillna("NA")

misconception_classes = np.unique(train_df["Misconception"])
print(misconception_classes)
print(len(misconception_classes))


def plot_classes(y):
    # Find the unique classes and their counts
    classes, counts = np.unique(y, return_counts=True)

    # Sort the counts and classes in descending order
    sorted_indices = np.argsort(counts)[::-1]
    sorted_classes = classes[sorted_indices]
    sorted_counts = counts[sorted_indices]

    # Create a bar chart for visualization
    plt.figure(figsize=(20, 12))
    plt.bar(sorted_classes.astype(str), sorted_counts, color="skyblue")
    plt.title("Distribution of Classes")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()

    print("Counts (sorted):", sorted_counts)


plot_classes(train_df["Category"])


# Verify that every QuestionId has only one QuestionText
is_one_to_one = train_df.groupby("QuestionId")["QuestionText"].nunique().eq(1).all()
is_one_to_one


# Let's split Category to Correct_Answer and Explanation_Eval to deal with them separately
train_df[["Correct_Answer", "Explanation_Eval"]] = train_df["Category"].str.split("_", expand=True)
# Let's remove for now - so we don't have to update it in two places
train_df = train_df.drop("Category", axis=1)
train_df.head(2)


# Make sure all Misconceptions occur only when Explanation_Eval is Misconception
train_df[["Explanation_Eval", "Misconception"]].groupby("Explanation_Eval").agg(set)


# Find the unique classes and their counts
classes, counts = np.unique(train_df["Misconception"][train_df["Misconception"] != "NA"], return_counts=True)

# Sort the counts and classes in descending order
sorted_indices = np.argsort(counts)[::-1]
sorted_classes = classes[sorted_indices]
sorted_counts = counts[sorted_indices]

# Create a bar chart for visualization
plt.figure(figsize=(20, 12))
plt.bar(sorted_classes.astype(str), sorted_counts, color="skyblue")
plt.title("Distribution of Misconception Classes")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()

print("Counts (sorted):", sorted_counts)


# Group Questions and Correct Answers
def group_questions_with_correct_answers():
    return train_df[train_df["Correct_Answer"] == "True"][["QuestionId", "MC_Answer"]].groupby("QuestionId").agg(set).reset_index()

question_correct_answer = group_questions_with_correct_answers()
question_correct_answer


# Check that there is exactly one correct answer
question_correct_answer[question_correct_answer["MC_Answer"].apply(len) != 1]


# Based on the question, we know that correct answer is `6`
for row in train_df.loc[
    (train_df.QuestionId == 31778)
    & (train_df.MC_Answer == "\\( 9 \\)")
    & (train_df.Correct_Answer == "True")
][["StudentExplanation", "Explanation_Eval"]].itertuples():
    explanation = row.StudentExplanation
    misconception = row.Explanation_Eval
    print(explanation, misconception)


# Misconception classification seems to be fine
# Let's only change Correct_Answer to False
train_df.loc[
    (train_df.QuestionId == 31778)
    & (train_df.MC_Answer == "\\( 9 \\)")
    & (train_df.Correct_Answer == "True"),
    "Correct_Answer",
] = "False"

# Let's verify we didn't break anything
train_df.head(2)


# Verify that problem is fixed
# Refresh dataframe
question_correct_answer = group_questions_with_correct_answers()
question_correct_answer[question_correct_answer["MC_Answer"].apply(len) != 1]


# Group questions with wrong answers
def group_questions_with_incorrect_answers():
    return train_df[train_df["Correct_Answer"] == "False"][["QuestionId", "MC_Answer"]].groupby("QuestionId").agg(set).reset_index()

question_wrong_answer = group_questions_with_incorrect_answers()
question_wrong_answer


# Check if correct answer is not among the wrong answers
# That would mean that MC_Answer is correct for QuestionId, but Category says otherwise
for questionId in question_wrong_answer.QuestionId:
    correct_answers = question_correct_answer[question_correct_answer.QuestionId == questionId].MC_Answer.iloc[0]
    wrong_answers = question_wrong_answer[question_wrong_answer.QuestionId == questionId].MC_Answer.iloc[0]

    if list(correct_answers)[0] in wrong_answers:
        print(questionId, wrong_answers) 


# Correct answer is `6`
# Let's check explanations first to see whether student mistakenly chose 6, or there is error in category
for row in train_df.loc[
    (train_df.QuestionId == 31778)
    & (train_df.MC_Answer == "\\( 6 \\)")
    & (train_df.Correct_Answer == "False")
][["StudentExplanation", "Explanation_Eval"]].itertuples():
    explanation = row.StudentExplanation
    misconception = row.Explanation_Eval
    print(explanation, misconception)


# It seems that most of them wanted to choose 9 - so let's replace 6 with 9 as answer.
train_df.loc[
    (train_df.QuestionId == 31778)
    & (train_df.MC_Answer == "\\( 6 \\)")
    & (train_df.Correct_Answer == "False"),
    "MC_Answer",
] = "\\( 9 \\)"

# Check that we didn't break anything
train_df.head(2)


# Verify fix
# Refresh dataframe
question_wrong_answer = group_questions_with_incorrect_answers()
for questionId in question_wrong_answer.QuestionId:
    correct_answers = question_correct_answer[question_correct_answer.QuestionId == questionId].MC_Answer.iloc[0]
    wrong_answers = question_wrong_answer[question_wrong_answer.QuestionId == questionId].MC_Answer.iloc[0]

    if list(correct_answers)[0] in wrong_answers:
        print(questionId, wrong_answers) 


# Based on the discussion here: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/589400.
# It seems public data have all the questions (secret test set doesn't include new questions).
# So we can safely use correct answer flag in input data

def add_correct_answer(dataframe):
    dataframe["Correct_Answer"] = [
        list(question_correct_answer[question_correct_answer["QuestionId"] == answer["QuestionId"]]["MC_Answer"].iloc[0])[0] == answer["MC_Answer"]
        for index, answer in dataframe[["QuestionId", "MC_Answer"]].iterrows()
    ]
    return dataframe

test_df = add_correct_answer(test_df)
test_df.head(3)


# Let's bring back category
train_df["Category"] = train_df.Correct_Answer + "_" + train_df.Explanation_Eval
train_df.head(2)


# Text features: concatenate question, MC answer, Is correct answer, and explanation
def combine_text(df):
    return (
        df["QuestionText"].fillna("")
        + " [MC] "
        + df["MC_Answer"].fillna("")
        + " [CORRECT] "
        + df["Correct_Answer"].astype(str)
        + " [EXPL] "
        + df["StudentExplanation"].fillna("")
    )


def create_sgd_pipeline():
    # Word-level pipeline
    word_pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="word",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=250_000,
                    min_df=2,
                ),
            ),
        ]
    )

    # Character-level pipeline
    char_pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=250_000,
                    min_df=2,
                ),
            ),
        ]
    )

    # Combine both pipelines using FeatureUnion
    combined_features = FeatureUnion(
        [("word_features", word_pipeline), ("char_features", char_pipeline)]
    )

    pipeline = Pipeline(
        [
            ("features", combined_features),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-5,
                    max_iter=10_000,
                    tol=1e-4,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    return pipeline


def create_xgboost_pipeline():
    # Word-level pipeline
    word_pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="word",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=250_000,
                    min_df=2,
                ),
            ),
        ]
    )

    # Character-level pipeline
    char_pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=250_000,
                    min_df=2,
                ),
            ),
        ]
    )

    # Combine both pipelines using FeatureUnion
    combined_features = FeatureUnion(
        [("word_features", word_pipeline), ("char_features", char_pipeline)]
    )

    xgb_params = {
        "objective": "multi:softprob",
        "device": "cuda",
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "colsample_bynode": 0.9,
        "colsample_bylevel": 0.9,
        "grow_policy": "lossguide",
        "lambda": 0.1,
        "tree_method": "hist",
        "eta": 0.06,
        "max_depth": 2, # 10
        "n_estimators": 5, # 500
    }

    pipeline = Pipeline(
        [
            ("features", combined_features),
            (
                "clf",
                xgb.XGBClassifier(**xgb_params),
            ),
        ]
    )
    return pipeline


def cross_validation(X, y, model):

    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof_true = []
    oof_pred_single = []
    oof_pred_topk = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model.fit(X_tr, y_tr)

        # Predict probabilities for validation
        proba = model.predict_proba(X_va)
        classes = model.named_steps["clf"].classes_

        # Single predictions (highest probability)
        pred_single = model.predict(X_va)

        # Top-k predictions for MAP@3
        pred_topk = [topk_from_proba(classes, row, k=3) for row in proba]

        oof_true.extend(list(y_va))
        oof_pred_single.extend(pred_single)
        oof_pred_topk.extend(pred_topk)

        fold_map3 = mapk(list(y_va), pred_topk, k=3)
        print(f"Fold {fold} MAP@3: {fold_map3:.5f}")
        gc.collect()

    cv_map3 = mapk(oof_true, oof_pred_topk, k=3)
    print(f"\nCV MAP@3 (mean over out-of-fold): {cv_map3:.5f}")

    return oof_true, oof_pred_single, oof_pred_topk


def analyze_topk_predictions(y_true, y_pred_topk, k=3):
    """Analyze how often true labels appear in top-k predictions"""
    correct_at_k = []
    
    for true_label, pred_topk in zip(y_true, y_pred_topk):
        if true_label in pred_topk:
            position = pred_topk.index(true_label) + 1  # 1-indexed
            correct_at_k.append(position)
        else:
            correct_at_k.append(0)  # Not in top-k
    
    print(f"\n=== Top-{k} Prediction Analysis ===")
    for pos in range(1, k+1):
        count = sum(1 for x in correct_at_k if x == pos)
        pct = count / len(correct_at_k) * 100
        print(f"Correct at position {pos}: {count} ({pct:.1f}%)")
    
    not_in_topk = sum(1 for x in correct_at_k if x == 0)
    pct_not_in_topk = not_in_topk / len(correct_at_k) * 100
    print(f"Not in top-{k}: {not_in_topk} ({pct_not_in_topk:.1f}%)")


def analyze_misclassifications(y_true, y_pred, classes, encoder=None):
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Percentage confusion matrix
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create summary table for each class
    summary_data = []

    for i, true_class in enumerate(classes):
        total_samples = cm[i].sum()
        correct_predictions = cm[i, i]
        misclassifications = total_samples - correct_predictions

        if total_samples > 0:
            misclassification_rate = (misclassifications / total_samples) * 100
            accuracy = (correct_predictions / total_samples) * 100

            # Find most frequent wrong prediction
            wrong_predictions = [
                (classes[j], cm[i, j], cm_pct[i, j])
                for j in range(len(classes))
                if i != j and cm[i, j] > 0
            ]

            if wrong_predictions:
                # Sort by count (not percentage) to find most frequent error
                most_frequent_error = max(wrong_predictions, key=lambda x: x[1])
                most_frequent_error_class = most_frequent_error[0]
                most_frequent_error_count = most_frequent_error[1]
                most_frequent_error_pct = most_frequent_error[2]
            else:
                most_frequent_error_class = "None"
                most_frequent_error_count = 0
                most_frequent_error_pct = 0.0

            summary_data.append(
                {
                    "True_Class": encoder.inverse_transform([true_class])[0] if encoder else true_class,
                    "Total_Samples": total_samples,
                    "Correct": correct_predictions,
                    "Accuracy_%": accuracy,
                    "Misclassified": misclassifications,
                    "Misclassification_%": misclassification_rate,
                    "Most_Frequent_Error": encoder.inverse_transform([most_frequent_error_class])[0] if encoder else most_frequent_error_class,
                    "Error_Count": most_frequent_error_count,
                    "Error_%_of_Class": most_frequent_error_pct,
                }
            )

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Sort by misclassification rate (worst performing classes first)
    summary_df = summary_df.sort_values("Misclassified", ascending=False)

    print("=== Class-by-Class Misclassification Analysis ===")
    print("(Sorted by misclassification - worst first)")
    print()
    print(summary_df.to_string(index=False, float_format="%.1f"))


X_cat = combine_text(train_df)

y_cat = train_df["Category"].astype(str).values

encoder_cat = LabelEncoder()
y_cat_enc = encoder_cat.fit_transform(y_cat)

print("Unique classes:", len(np.unique(y_cat)))
print("Example label:", y_cat[0])
print("Text example:", X_cat.iloc[0][:300])


plot_classes(y_cat)


oof_true_cat, oof_pred_single_cat, oof_pred_topk_cat = cross_validation(X_cat, y_cat_enc, create_sgd_pipeline())


analyze_topk_predictions(oof_true_cat, oof_pred_topk_cat, k=3)


all_classes = sorted(set(oof_true_cat))
analyze_misclassifications(oof_true_cat, oof_pred_single_cat, all_classes, encoder_cat)


# Use only misconception samples
misconception_train_df = train_df[train_df.Explanation_Eval == "Misconception"]

X_misc = combine_text(misconception_train_df)
y_misc = misconception_train_df["Misconception"].astype(str).values

encoder_misc = LabelEncoder()
y_misc_enc = encoder_misc.fit_transform(y_misc)

print("Unique classes:", len(np.unique(y_misc)))
print("Example label:", y_misc[0])
print("Text example:", X_misc.iloc[0][:300])


plot_classes(y_misc)


oof_true_misc, oof_pred_single_misc, oof_pred_topk_misc = cross_validation(X_misc, y_misc_enc, create_xgboost_pipeline())


analyze_topk_predictions(oof_true_misc, oof_pred_topk_misc, k=3)


all_classes = sorted(set(oof_true_misc))
analyze_misclassifications(oof_true_misc, oof_pred_single_misc, all_classes, encoder_misc)


# Build single target label "Category:Misconception"
train_df["target"] = (
    train_df["Category"].astype(str) + ":" + train_df["Misconception"].astype(str)
)

# Cannot do Target since there is a class with only 1 sample
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df.Misconception)

X_val = combine_text(val_df)
y_val = val_df.target.values


plot_classes(train_df["target"].values)


# Category Classifier
X_cat_tr = combine_text(train_df)

encoder_cat = LabelEncoder()
y_cat_tr = encoder_cat.fit_transform(train_df["Category"].astype(str).values)

cat_model = create_sgd_pipeline()
cat_model.fit(X_cat_tr, y_cat_tr)


# Misconception Classifier
X_misc_tr = combine_text(train_df[train_df.Explanation_Eval == "Misconception"])

encoder_misc = LabelEncoder()
y_misc_tr = encoder_misc.fit_transform(train_df[train_df.Explanation_Eval == "Misconception"]["Misconception"].astype(str).values)

misc_model = create_xgboost_pipeline()
misc_model.fit(X_misc_tr, y_misc_tr, clf__verbose=50)


def stacking_pred_topk(cat_model, encoder_cat, misc_model, encoder_misc, X):
    cat_classes_enc = cat_model.named_steps["clf"].classes_
    cat_classes = encoder_cat.inverse_transform(cat_classes_enc)

    pred_topk = []
    for sample in X:
        cat_proba = cat_model.predict_proba([sample])
        cat_pred_topk = [topk_from_proba(cat_classes, row, k=3) for row in cat_proba]
        result = []
        for pred in cat_pred_topk[0]:
            misconception = "NA"
            if "Misconception" in pred:
                misc_pred_single = misc_model.predict([sample])
                misconception = encoder_misc.inverse_transform(misc_pred_single)[0]
            result.append(pred + ":" + misconception)
        pred_topk.append(result)
    return pred_topk


pred_topk = stacking_pred_topk(cat_model, encoder_cat, misc_model, encoder_misc, X_val)


cv_map3 = mapk(y_val, pred_topk, k=3)
print(f"\nCV MAP@3 (mean over out-of-fold): {cv_map3:.5f}")


analyze_topk_predictions(y_val, pred_topk, k=3)


pred_single = np.array(pred_topk)[:, 0]
all_classes = sorted(set(y_val))
analyze_misclassifications(y_val, pred_single, all_classes)


category_classifier = create_sgd_pipeline()
category_classifier.fit(X_cat, y_cat_enc)


misconception_classifier = create_xgboost_pipeline()
misconception_classifier.fit(X_misc, y_misc_enc)


X_test = combine_text(test_df)
pred_topk = stacking_pred_topk(category_classifier, encoder_cat, misconception_classifier, encoder_misc, X_test)


# Build submission
sub = pd.DataFrame({
    'row_id': test_df.index + 36696,  # Kaggle's sample_submission starts at this ID; we re-index safely
    'Category:Misconception': [' '.join(t) for t in pred_topk]
})
sub.head()


# Use the provided sample_submission to ensure exact row_id ordering
sample_path = os.path.join(DATA_DIR, 'sample_submission.csv')
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    if 'row_id' in sample.columns:
        sub = sample[['row_id']].merge(sub, on='row_id', how='left')
        # If any rows didn't merge (shouldn't happen), fill with a safe default
        default_label = 'True_Correct:NA'
        sub['Category:Misconception'] = sub['Category:Misconception'].fillna(default_label)
        print('Aligned with sample_submission.')
else:
    print('sample_submission.csv not found; using generated row_id sequence.')

sub.head(3)


SUB_PATH = 'submission.csv'
sub.to_csv(SUB_PATH, index=False)
print('Saved:', os.path.abspath(SUB_PATH))
sub.head()

