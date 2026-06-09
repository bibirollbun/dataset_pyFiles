# Loading data
import pandas as pd
import seaborn as sns
import regex as re
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

train_tokens = pd.read_csv(
    "/kaggle/input/extracting-features-with-spacy/training_with_tokens.csv"
)
training_data_labels = pd.read_csv(
    r"/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
)
test_tokens = pd.read_csv(
    "/kaggle/input/extracting-features-with-spacy/test_with_tokens.csv"
)
LEN_VECTOR = 300  # Length of the token vectors
token_vectors = [f"avg_token_vector_{i}" for i in range(LEN_VECTOR)]

train_tokens.head()


def count_none_latin_letters(text):
    # Search for things that are NOT
    # \p{Latin} Latin letters
    # \s empty spaces
    # \p{S} Symbols
    # \p{P} Punitions
    # \p{N} Numbers
    # \p{Greek} greek letters (boy do scientists love themselves some greek letters)
    # \Âµ for some reason Âµ is not part of \p{Greek}? Weird
    return len(re.findall(r"[^\p{Latin}\s\p{S}\p{P}\p{N}\p{Greek}\Âµ]+", text))


def repeats_word_three_times(text: str) -> Tuple[bool, list]:
    repeating_phrases = re.findall(r"([^\w].{4,})\1\1", text.lower())
    if len(repeating_phrases) > 0:
        return True, repeating_phrases
    else:
        return False, []


def preprocess_dataset(dataset: pd.DataFrame):
    dataset.loc[:, "is_empty"] = pd.Series([False] * len(dataset), dtype=bool)
    dataset.loc[:, "has_weird_letters"] = pd.Series([False] * len(dataset), dtype=bool)
    dataset.loc[:, "repeated_word"] = pd.Series([False] * len(dataset), dtype=object)

    for i, row in dataset.iterrows():
        text = row["text"]
        # Empty strings are fake
        if pd.isna(text) or len(text) == 0:
            dataset.loc[i, "is_empty"] = True
            continue
        else:
            dataset.loc[i, "is_empty"] = False

        # Did you use weird letters
        # If both are the same we continue
        count1 = count_none_latin_letters(text)
        dataset.loc[i, "has_weird_letters"] = count1 > 0

        # Repeating words
        # If you repeat a word more then 3 and it is the most repeated
        repeats_1 = repeats_word_three_times(text)
        dataset.loc[i, "repeated_word"] = repeats_1[0]


preprocess_dataset(train_tokens)
preprocess_dataset(test_tokens)
train_tokens.head()


import umap

train_x = train_tokens[token_vectors]
test_x = test_tokens[token_vectors]

embedding = umap.UMAP().fit_transform(pd.concat([test_x, train_x]))


ax = sns.scatterplot(
    x=embedding[:, 0],
    y=embedding[:, 1],
    hue=["unknown"] * len(test_tokens) + list(train_tokens.is_real),
)
ax.set_title("UMAP Projection with is_real Highlighted")


ax = sns.scatterplot(
    x=embedding[:, 0],
    y=embedding[:, 1],
    hue=list(test_tokens.is_empty) + list(train_tokens.is_empty),
)
ax.set_title("UMAP Projection with is_empty Highlighted")


ax = sns.scatterplot(
    x=embedding[:, 0],
    y=embedding[:, 1],
    hue=list(test_tokens.has_weird_letters) + list(train_tokens.has_weird_letters),
)
ax.set_title("UMAP Projection with has_weird_letters Highlighted")


ax = sns.scatterplot(
    x=embedding[:, 0],
    y=embedding[:, 1],
    hue=list(test_tokens.repeated_word) + list(train_tokens.repeated_word),
)
ax.set_title("UMAP Projection with Repeated Words Highlighted")


train_filtered = train_tokens.loc[
    (train_tokens.is_empty == False)
    & (train_tokens.has_weird_letters == False)
    & (train_tokens.repeated_word == False)
]
test_filtered = test_tokens.loc[
    (test_tokens.is_empty == False)
    & (test_tokens.has_weird_letters == False)
    & (test_tokens.repeated_word == False)
]

train_x2 = train_filtered[token_vectors]
test_x2 = test_filtered[token_vectors]

train_x2


import umap

embedding2 = umap.UMAP().fit_transform(pd.concat([test_x2, train_x2]))


plt.figure(figsize=(10, 10))

ax = sns.scatterplot(
    x=embedding2[:, 0],
    y=embedding2[:, 1],
    hue=[2] * len(test_filtered) + list(train_filtered.is_real),
    palette={0: "red", 1: "green", 2: [0.8, 0.8, 0.8]},  # turquoise
)
ax.set_title(
    "UMAP Projection without Weird Letters, Empty Strings or Repeated Words Highlighted"
)


all_difference_vectors = []
is_from_real = []

# Testing data
article_ids = sorted([int(i) for i in test_tokens.article_id.unique()])
num_test_data = len(article_ids)

for i in article_ids:
    vec1 = test_tokens.loc[(test_tokens.article_id == i) & (test_tokens.file_id == 1)]
    vec2 = test_tokens.loc[(test_tokens.article_id == i) & (test_tokens.file_id == 2)]

    difference_vector = vec1[token_vectors].values - vec2[token_vectors].values
    is_from_real.append("unknown")

    all_difference_vectors.append(difference_vector)

# Training data
article_ids = sorted([int(i) for i in train_tokens.article_id.unique()])
num_train_data = len(article_ids)

for i in article_ids:
    vec1 = train_tokens.loc[
        (train_tokens.article_id == i) & (train_tokens.file_id == 1)
    ]
    vec2 = train_tokens.loc[
        (train_tokens.article_id == i) & (train_tokens.file_id == 2)
    ]

    difference_vector = vec1[token_vectors].values - vec2[token_vectors].values
    if vec1.is_real.values[0]:
        is_from_real.append("real")
    else:
        is_from_real.append("fake")

    all_difference_vectors.append(difference_vector)

# Reshape output
all_difference_vectors = np.array(all_difference_vectors)[:, 0, :]
is_from_real = np.array(is_from_real)

print(all_difference_vectors.shape)


import umap

embedding3 = umap.UMAP().fit_transform(all_difference_vectors)

plt.figure(figsize=(16, 16))

ax = sns.scatterplot(
    x=embedding3[:, 0],
    y=embedding3[:, 1],
    hue=is_from_real,
    palette={"fake": "red", "real": "green", "unknown": [0.8, 0.8, 0.8]},
)
ax.set_title(
    "UMAP Projection of Difference Vectors with Real and Fake Articles Highlighted"
)


from sklearn.neighbors import NearestNeighbors

embedding_train = embedding3[num_test_data:]
embedding_test = embedding3[:num_test_data]

for n_neighbors in range(2, 20, 2):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(
        embedding_train
    )

    # Training results
    distances, indices = nbrs.kneighbors(embedding_train)

    all_correct = 0
    for indic in indices:
        predicted = (
            is_from_real[indic[1:] + num_test_data]
            == is_from_real[indic[0] + num_test_data]
        ).mean()
        all_correct += predicted.round()

    print(
        f"acc training: {all_correct/num_train_data:.3f} for {nbrs.n_neighbors - 1} neighbors"
    )


nbrs_train = NearestNeighbors(n_neighbors=8, algorithm="ball_tree").fit(embedding_train)


distances, indices = nbrs_train.kneighbors(embedding_train)
for article_id, indic in enumerate(indices):
    avg_fake = (is_from_real[indic[1:] + num_test_data] == "fake").mean()
    uncertainty = np.abs(avg_fake.round() - avg_fake)
    real_text_id = avg_fake.round() + 1

    training_data_labels.loc[article_id, "prediction"] = real_text_id
    training_data_labels.loc[article_id, "uncertainty"] = uncertainty

training_data_labels.loc[:, "is_correct"] = (
    training_data_labels.prediction == training_data_labels.real_text_id
)

print(f"acc: {training_data_labels.is_correct.sum() / len(training_data_labels)}")

training_data_labels.head()


sns.histplot(
    data=training_data_labels, x="uncertainty", hue="is_correct", multiple="stack"
)


nbrs = NearestNeighbors(n_neighbors=7, algorithm="ball_tree").fit(embedding_train)


# This gives a test acc of 0.86307
submission = pd.DataFrame(columns=["id", "real_text_id", "uncertainty"])

distances, indices = nbrs.kneighbors(embedding_test)
for article_id, indic in enumerate(indices):
    avg_fake = (is_from_real[indic + num_test_data] == "fake").mean()
    uncertainty = np.abs(avg_fake.round() - avg_fake)
    real_text_id = avg_fake.round() + 1

    training_data_labels.loc[article_id, "real_text_id"] = real_text_id
    training_data_labels.loc[article_id, "uncertainty"] = uncertainty

    submission = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "id": article_id,
                        "real_text_id": int(real_text_id),
                        "uncertainty": uncertainty,
                    }
                ]
            ),
            submission,
        ]
    )

submission = submission.sort_values(by="id")
submission.head()


# Adding an extra filter for the submission increases the acc to 0.87136
submission_filtered = submission.copy()
submission_filtered.loc[:, "overwritten"] = False

for i, row in submission_filtered.iterrows():
    file1 = test_tokens[(test_tokens.article_id == row.id) & (test_tokens.file_id == 1)]
    file2 = test_tokens[(test_tokens.article_id == row.id) & (test_tokens.file_id == 2)]

    # Empty is always fake
    if file1.is_empty.values[0]:
        submission_filtered.loc[submission_filtered.id == row.id, "real_text_id"] = 2
        submission_filtered.loc[submission_filtered.id == row.id, "overwritten"] = True
        continue
    elif file2.is_empty.values[0]:
        submission_filtered.loc[submission_filtered.id == row.id, "real_text_id"] = 1
        submission_filtered.loc[submission_filtered.id == row.id, "overwritten"] = True
        continue

    # Weird letters is always fake
    if file1.has_weird_letters.values[0]:
        submission_filtered.loc[submission_filtered.id == row.id, "real_text_id"] = 2
        submission_filtered.loc[submission_filtered.id == row.id, "overwritten"] = True
        continue
    elif file2.has_weird_letters.values[0]:
        submission_filtered.loc[submission_filtered.id == row.id, "real_text_id"] = 1
        submission_filtered.loc[submission_filtered.id == row.id, "overwritten"] = True
        continue

    # Repeated words is always fake
    if file1.repeated_word.values[0]:
        submission_filtered.loc[submission_filtered.id == row.id, "real_text_id"] = 2
        submission_filtered.loc[submission_filtered.id == row.id, "overwritten"] = True
        continue
    elif file2.repeated_word.values[0]:
        submission_filtered.loc[submission_filtered.id == row.id, "real_text_id"] = 1
        submission_filtered.loc[submission_filtered.id == row.id, "overwritten"] = True
        continue

submission_filtered[["id", "real_text_id"]].to_csv("submission.csv", index=False)
submission_filtered.to_csv("prediction.csv", index=False)
submission_filtered.head()

