! python -m spacy download en_core_web_lg


# ðŸ“¦ Import needed package
from pathlib import Path

import numpy as np
import regex as re
import pandas as pd

import spacy


# Loading training data
training_data_labels = pd.read_csv(
    r"/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
)
training_data = pd.DataFrame(columns=["article_id", "file_id", "text", "is_real"])
for i, row in training_data_labels.iterrows():
    article_id = int(row.id)
    real_text_id = row.real_text_id
    fake_text_id = 1 if real_text_id == 2 else 2

    # Get file paths to text
    files_path = Path(
        rf"/kaggle/input/fake-or-real-the-impostor-hunt/data/train/article_{str(article_id).zfill(4)}"
    )
    real_text_path = files_path / f"file_{real_text_id}.txt"
    fake_text_path = files_path / f"file_{fake_text_id}.txt"

    # file 1
    file_1_path = files_path / "file_1.txt"
    file_1 = file_1_path.read_text()
    is_real = 1 if real_text_id == 1 else 0

    training_data = pd.concat(
        [
            pd.DataFrame(
                [[article_id, 1, file_1, is_real]], columns=training_data.columns
            ),
            training_data,
        ],
        ignore_index=True,
    )

    # file 2
    file_2_path = files_path / "file_2.txt"
    file_2 = file_2_path.read_text()
    is_real = 1 if real_text_id == 2 else 0

    training_data = pd.concat(
        [
            pd.DataFrame(
                [[article_id, 2, file_2, is_real]], columns=training_data.columns
            ),
            training_data,
        ],
        ignore_index=True,
    )
# Show the first few rows of the training data
training_data.head()


# Load test data
test_path = Path(r"/kaggle/input/fake-or-real-the-impostor-hunt/data/test")
test_data = pd.DataFrame(columns=["article_id", "file_id", "text"])
for test_path_article in test_path.glob("**/article_*"):
    article_id = int(re.findall("\d+", test_path_article.name)[0])

    # File 1
    file_1_path = test_path_article / "file_1.txt"
    file_1 = file_1_path.read_text()
    test_data = pd.concat(
        [pd.DataFrame([[article_id, 1, file_1]], columns=test_data.columns), test_data],
        ignore_index=True,
    )

    # File 2
    file_2_path = test_path_article / "file_2.txt"
    file_2 = file_2_path.read_text()
    test_data = pd.concat(
        [pd.DataFrame([[article_id, 2, file_2]], columns=test_data.columns), test_data],
        ignore_index=True,
    )

test_data.head()


NLP_EN = spacy.load("en_core_web_lg")
LEN_VECTOR = 300


def encode_text(text: str) -> np.ndarray:
    """
    Encode the text using spaCy's language model.
    Returns a 300-dimensional vector.
    """
    tokens = NLP_EN(text)
    tot_vector = np.zeros((LEN_VECTOR), dtype=np.float32)
    num_tokens = 0
    for token in tokens:
        if token.is_stop:
            continue
        if token.pos_ not in ["ADJ", "ADV", "NOUN", "PROPN"]:
            continue
        if not token.has_vector or token.vector_norm == 0.0:
            continue
        tot_vector += token.vector
        num_tokens += 1

    if num_tokens == 0:
        return np.zeros((LEN_VECTOR), dtype=np.float32)
    else:
        return tot_vector / num_tokens


token_vectors = [f"avg_token_vector_{i}" for i in range(LEN_VECTOR)]

for i, row in training_data.iterrows():
    avg_token_vector = encode_text(row.text)
    training_data.loc[i, token_vectors] = avg_token_vector

training_data.to_csv("training_with_tokens.csv", index=False)
training_data.head()


for i, row in test_data.iterrows():
    avg_token_vector = encode_text(row.text)
    test_data.loc[i, token_vectors] = avg_token_vector

test_data.to_csv("test_with_tokens.csv", index=False)
test_data.head()


# Nearest neighbor search
from sklearn.neighbors import NearestNeighbors

for num_neight in range(2, 96, 2):
    # Fit the model
    train_array = np.array(training_data[token_vectors])
    nbrs = NearestNeighbors(n_neighbors=num_neight, algorithm="ball_tree").fit(
        train_array
    )

    # Find the nearest neighbors for each training data point
    distances, indices = nbrs.kneighbors(train_array)

    all_correct = 0
    for indic in indices:
        prediction = training_data.iloc[indic[1:]].is_real.mode()[0]
        # First neighbor it always itself
        true_y = training_data.iloc[indic[0]].is_real
        all_correct += prediction == true_y

    print(
        f"acc training: {all_correct/len(training_data):.3f} with {num_neight-1} neighbors"
    )


nbrs_test = NearestNeighbors(n_neighbors=45, algorithm="ball_tree").fit(train_array)


submission = pd.DataFrame(columns=["id", "real_text_id"])
test_array = np.array(test_data[token_vectors])

distances, indices = nbrs_test.kneighbors(test_array)
for article_id in range(test_data.article_id.max() + 1):
    index_file_1 = test_data[
        (test_data.article_id == article_id) & (test_data.file_id == 1)
    ].index
    index_file_2 = test_data[
        (test_data.article_id == article_id) & (test_data.file_id == 2)
    ].index

    indic_1 = indices[index_file_1]
    indic_2 = indices[index_file_2]

    pred_1 = training_data.iloc[indic_1[0]].is_real.mean()
    pred_2 = training_data.iloc[indic_2[0]].is_real.mean()

    # Get the highest prediction
    if pred_1 > pred_2:
        submission = pd.concat(
            [pd.DataFrame([{"id": article_id, "real_text_id": 1}]), submission]
        )
        continue
    elif pred_2 > pred_1:
        submission = pd.concat(
            [pd.DataFrame([{"id": article_id, "real_text_id": 2}]), submission]
        )
        continue

    # If prediction are equal get the lowest distant
    dist_1 = distances[index_file_1].mean()
    dist_2 = distances[index_file_2].mean()
    if dist_2 > dist_1:
        submission = pd.concat(
            [pd.DataFrame([{"id": article_id, "real_text_id": 1}]), submission]
        )
        continue
    else:
        submission = pd.concat(
            [pd.DataFrame([{"id": article_id, "real_text_id": 2}]), submission]
        )
        continue

submission = submission.sort_values(by="id")
submission.to_csv("submission.csv", index=False)
submission.head()

