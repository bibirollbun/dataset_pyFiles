# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install nltk
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


RANDOM_STATE = 42
K_FOLDS = 5


dataset = pd.read_table("../input/ml-olympiad-tfugsurabaya-2024/train.tsv")
dataset


XTrain = dataset["REVIEW"]
XTrain.head()


y = dataset["LABEL"]
y.head()


!pip install PySastrawi

# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


from sklearn.feature_extraction.text import CountVectorizer

stopwordsCombined = stopwords.words("indonesian") + stopwords.words("english")

vectorizers = {
    "unigram": CountVectorizer(
        tokenizer=word_tokenize,
        ngram_range=(1, 1),
        min_df=1,
        stop_words=stopwordsCombined,
        preprocessor=stemmer.stem,
    ),
    "bigram": CountVectorizer(
        tokenizer=word_tokenize,
        ngram_range=(2, 2),
        min_df=1,
        stop_words=stopwordsCombined,
        preprocessor=stemmer.stem,
    ),
    "trigram": CountVectorizer(
        tokenizer=word_tokenize,
        ngram_range=(3, 3),
        min_df=1,
        stop_words=stopwordsCombined,
        preprocessor=stemmer.stem,
    ),
    "unigram-bigram": CountVectorizer(
        tokenizer=word_tokenize,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=stopwordsCombined,
        preprocessor=stemmer.stem,
    ),
    "unigram-bigram-trigram": CountVectorizer(
        tokenizer=word_tokenize,
        ngram_range=(1, 3),
        min_df=1,
        stop_words=stopwordsCombined,
        preprocessor=stemmer.stem,
    ),
}


from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

kFoldIndices = pd.DataFrame(columns=["Fold", "Train Indices", "Test Indices"])

foldIndex = 0
for train_index, test_index in kf.split(XTrain, y):
    print(
        f"Fold: {foldIndex}, Train set length: {len(train_index)}, Test set length: {len(test_index)}"
    )
    kFoldIndices = pd.concat(
        [
            kFoldIndices,
            pd.DataFrame(
                {
                    "Fold": [foldIndex],
                    "Train Indices": [train_index],
                    "Test Indices": [test_index],
                }
            ),
        ],
        ignore_index=True,
    )
    foldIndex += 1


import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_kfold(cv, X, y, ax, nSplits, xLimMax=100):
    """
    Plots the indices for a cross-validation object.

    Parameters:
    cv: Cross-validation object
    X: Feature set
    y: Target variable
    ax: Matplotlib axis object
    nSplits: Number of folds in the cross-validation
    xLimMax: Maximum limit for the x-axis
    """

    # Set color map for the plot
    cmap_cv = plt.cm.coolwarm
    cv_split = cv.split(X=X, y=y)

    for i_split, (train_idx, test_idx) in enumerate(cv_split):
        # Create an array of NaNs and fill in training/testing indices
        indices = np.full(len(X), np.nan)
        indices[test_idx], indices[train_idx] = 1, 0

        # Plot the training and testing indices
        ax_x = range(len(indices))
        ax_y = [i_split + 0.5] * len(indices)
        ax.scatter(
            ax_x, ax_y, c=indices, marker="_", lw=10, cmap=cmap_cv, vmin=-0.2, vmax=1.2
        )

    # Set y-ticks and labels
    y_ticks = np.arange(nSplits) + 0.5
    ax.set(
        yticks=y_ticks,
        yticklabels=range(nSplits),
        xlabel="X index",
        ylabel="Fold",
        ylim=[nSplits, -0.2],
        xlim=[0, xLimMax],
    )

    # Set plot title and create legend
    ax.set_title("KFold", fontsize=14)
    legend_patches = [
        Patch(color=cmap_cv(0.8), label="Testing set"),
        Patch(color=cmap_cv(0.02), label="Training set"),
    ]
    ax.legend(handles=legend_patches, loc=(1.03, 0.8))


# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 3))
plot_kfold(kf, XTrain, y, ax, K_FOLDS, XTrain.shape[0])
plt.tight_layout()
fig.subplots_adjust(right=0.6)


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

scores = pd.DataFrame(
    columns=[
        "features",
        "Naive Bayes (Average)",
        "Naive Bayes (Standard Deviation)",
        "Naive Bayes (Consistency Score)",
        "SVM (Average)",
        "SVM (Standard Deviation)",
        "SVM (Consistency Score)",
    ],
)


def cross_validation(estimator, X, y, cv):
    scores = cross_val_score(
        estimator,
        X,
        y,
        scoring="f1_weighted",
        cv = cv,
    )

    return scores.mean(), scores.std()


for vectorizer_key in vectorizers.keys():
    print(f"Evaluating models with {vectorizer_key} features:")

    pipelineNB = Pipeline(
        [
            ("vectorizer", vectorizers[vectorizer_key]),
            ("classifier", MultinomialNB()),
        ]
    )

    pipelineSVM = Pipeline(
        [
            ("vectorizer", vectorizers[vectorizer_key]),
            ("classifier", SVC(random_state=RANDOM_STATE, kernel="linear")),
        ]
    )

    nbScoreMean, nbScoreStd = cross_validation(pipelineNB, XTrain, y, kf)
    svmScoreMean, svmScoreStd = cross_validation(pipelineSVM, XTrain, y, kf)

    """
    Consistency score is based on the harmonic mean of average score and (1 - standard deviation).
    It rewards models that not only perform well on average but also have low variability across folds.    
    """

    scores = pd.concat(
        [
            scores,
            pd.DataFrame(
                {
                    "features": [vectorizer_key],
                    "Naive Bayes (Average)": [nbScoreMean],
                    "Naive Bayes (Standard Deviation)": [nbScoreStd],
                    "Naive Bayes (Consistency Score)": [
                        2
                        * (1 - nbScoreStd)
                        * nbScoreMean
                        / (1 - nbScoreStd + nbScoreMean)
                    ],
                    "SVM (Average)": [svmScoreMean],
                    "SVM (Standard Deviation)": [svmScoreStd],
                    "SVM (Consistency Score)": [
                        2
                        * (1 - svmScoreStd)
                        * svmScoreMean
                        / (1 - svmScoreStd + svmScoreMean)
                    ],
                }
            ),
        ],
        ignore_index=True,
    )

scores



# Train the model and feature with the highest F1 scores

from sklearn.metrics import classification_report

featureSelected = "unigram-bigram"

pipelineSelected = Pipeline(
    [
        ("vectorizer", vectorizers[featureSelected]),
        ("classifier", MultinomialNB()),
    ]
)

pipelineSelected.fit(XTrain, y)

yTrain = pipelineSelected.predict(XTrain)

report = classification_report(y, yTrain)

print(
    f"Classification Report for {pipelineSelected.__class__.__name__} with {featureSelected} features:\n"
)
print(report)


testDataset = pd.read_table("../input/ml-olympiad-tfugsurabaya-2024/test.tsv")
testDataset.head()


XTest = testDataset["REVIEW"].copy()
XTest.head()


yTest = pipelineSelected.predict(XTest)
yTest


# Save predictions to a CSV file
output = pd.DataFrame({"ID": testDataset.ID, "LABEL": yTest})
output.to_csv("submission.csv", index=False)
output.head()

