import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import os, re, random
from typing import List, Tuple


data_prefix = "/kaggle/input/fake-or-real-the-impostor-hunt/data/"

def get_data_from_files(data_folder: str) -> List[Tuple[int, str, str]]:
    """
    Load paired texts found in $data_folder:
    Assumes pairs are in $data_folder/article_(\d{4})/file_1.txt and file_2.txt
    Returns: List of (article_number, text1, text2)
    """
    pairs: List[Tuple[int, str, str]] = []

    # Find all subfolders matching "article_####"
    pair_folders = []
    for name in os.listdir(data_folder):
        match = re.match(r"article_(\d{4})$", name)
        if match:
            pair_folders.append((int(match.group(1)), name))

    # Sort folders by numeric value
    pair_folders.sort(key=lambda x: x[0])

    for article_num, pair_folder in pair_folders:
        # Read both text files
        file_1_path: str = os.path.join(data_folder, pair_folder, "file_1.txt")
        file_2_path: str = os.path.join(data_folder, pair_folder, "file_2.txt")

        try:
            with open(file_1_path, 'r', encoding='utf-8') as f1, open(file_2_path, 'r', encoding='utf-8') as f2:
                pairs.append((article_num, f1.read().strip(), f2.read().strip()))

        except FileNotFoundError as e:
            print(f"Could not read files in {pair_folder}: {e}")
            exit(1)
        except Exception as e:
            print(f"Error processing {pair_folder}: {e}")
            exit(1)

    assert(len(pairs) == pairs[-1][0] + 1), \
        "Articles numbers are not running continuously from 0!"
    print(f"Successfully loaded {len(pairs)} pairs of text samples from {data_folder}.")
    return pairs

def get_labels_from_file(labels_file: str) -> pd.DataFrame:
    labels_df: pd.DataFrame = pd.read_csv(labels_file)
    print(f"Successfully loaded {len(labels_df)} labels from {labels_file}")
    return labels_df

# Load the training 
training_labels: pd.DataFrame = get_labels_from_file(data_prefix + "train.csv")
training_pairs: List[Tuple[int, str, str]] = get_data_from_files(data_prefix + "train")
first_training_texts = [pair[1] for pair in training_pairs]
second_training_texts = [pair[2] for pair in training_pairs]


# Load the testing data
# When splitting pairs for testing, ignore the article_num
testing_pairs: List[Tuple[int, str, str]] = get_data_from_files(data_prefix + "test")
id_of_testing_pairs = [pair[0] for pair in testing_pairs]
first_testing_texts = [pair[1] for pair in testing_pairs]
second_testing_texts = [pair[2] for pair in testing_pairs]

# Create a transformed version of training_pairs where real text is always first, 
# Use this only for analysis
real_first_training_pairs = []
for (article_num, text1, text2), label_row in zip(training_pairs, training_labels.itertuples(index=False)):
    if label_row.real_text_id == 1:
        real_first_training_pairs.append((text1, text2))
    else:
        real_first_training_pairs.append((text2, text1))


# Initialize SciBERT
# I have tried other models like ELECTRA and DeBERTa, 
# but SciBERT performs the best on this problem.

model_name: str = "allenai/scibert_scivocab_uncased"
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
model: BertModel = BertModel.from_pretrained(model_name)

# model.eval()


def pair_format(id: int) -> str: 
    return (f"[Article {id}]\n" +
            f"[Version 1]\n{training_pairs[id][1]}\n" +
            f"[Version 2]\n{training_pairs[id][2]}\n\n\n")

def save_pairs_to_file(ids: List[int], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        for i in ids:
            f.write(pair_format(i))
        
random_indices = random.sample(range(len(training_pairs)), 10)
# save_pairs_to_file(random_indices, "sample_pairs.txt")


real_lengths = [len(pair[0].split()) for pair in real_first_training_pairs]
fake_lengths = [len(pair[1].split()) for pair in real_first_training_pairs]

stats_df = pd.DataFrame({
    "Real Texts": pd.Series(real_lengths).describe(),
    "Fake Texts": pd.Series(fake_lengths).describe()
})
print(stats_df)



def get_bert_embeddings(texts: List[str]) -> np.ndarray:
    batch_size = 32  # Adjust based on your GPU memory
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        with torch.no_grad():
            # Tokenize batch - tensors will be on CPU by default
            inputs = tokenizer(
                batch_texts, 
                return_tensors='pt', 
                max_length=512, 
                truncation=True, 
                padding=True
            )
            
            # Forward pass - everything stays on CPU
            outputs = model(**inputs)
            
            # Extract [CLS] embeddings - already on CPU
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(embeddings)
        
    return np.concatenate(all_embeddings, axis=0)

first_training_embeddings: np.ndarray = get_bert_embeddings(first_training_texts)
second_training_embeddings: np.ndarray = get_bert_embeddings(second_training_texts) 


all_embeddings: List[np.ndarray] = []
all_labels: List[int] = []

for i in range(len(training_labels)):
    # add a 1-label if this is the embedding of real text otherwise 0-label
    all_embeddings.append(first_training_embeddings[i])
    all_labels.append(1 if training_labels.iloc[i].real_text_id == 1 else 0)
    # add a 1-label if this is the embedding of real text otherwise 0-label
    all_embeddings.append(second_training_embeddings[i])
    all_labels.append(1 if training_labels.iloc[i].real_text_id == 2 else 0)


classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42, max_iter=500)
    }
    
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in classifiers.items():
    print(f"Training {name}...")
    # Perform 5-fold cross-validation on the full training data
    scores = cross_val_score(clf, all_embeddings, all_labels, cv=cv, scoring='accuracy')
    print(f"{name} Cross-validation scores for each fold:", scores)
    print(f"{name} Mean accuracy:", scores.mean())


# Let's train the SVM model on the entire training set
svm_clf = SVC(kernel='rbf', random_state=42, probability=True)
svm_clf.fit(all_embeddings, all_labels)

# Evaluate the model on the training set without pair information
all_predictions = svm_clf.predict(all_embeddings)
accuracy = accuracy_score(all_labels, all_predictions)
print(f"SVM (RBF) accuracy without pair information: {accuracy:.4f}")

# Now evaluate using the fact that a pair has one real text and one fake text
scores_first_texts = svm_clf.decision_function(first_training_embeddings)
scores_second_texts = svm_clf.decision_function(second_training_embeddings)

# choose the one with the higher score as the real text
pair_predictions = np.where(scores_first_texts > scores_second_texts, 1, 2)
accuracy = accuracy_score(training_labels["real_text_id"].values, pair_predictions)
print(f"SVM (RBF) accuracy with pair information: {accuracy:.4f}")

# Get boolean mask of wrong predictions
wrong_mask = pair_predictions != training_labels["real_text_id"].values
# Extract the IDs of wrongly predicted pairs
wrong_ids = training_labels.loc[wrong_mask, "id"].values
save_pairs_to_file(wrong_ids, "misclassified_pairs.txt")

# Now evaluate using the fact that a pair has one real text and one fake text
proba_first_texts = svm_clf.predict_proba(first_training_embeddings)
proba_second_texts = svm_clf.predict_proba(second_training_embeddings)

for id in wrong_ids:
    print(f"Article {id}")
    print(f"Score text 1: {scores_first_texts[id]}, Score text 2: {scores_second_texts[id]}")
    print(f"Proba text 1: {proba_first_texts[id]}, Proba text 2: {proba_second_texts[id]}")



first_test_embeddings: np.ndarray = get_bert_embeddings(first_testing_texts)
second_test_embeddings: np.ndarray = get_bert_embeddings(second_testing_texts)

# Now evaluate using the fact that a pair has one real text and one fake text
scores_first_texts = svm_clf.decision_function(first_test_embeddings)
scores_second_texts = svm_clf.decision_function(second_test_embeddings)

# choose the one with the higher score as the real text
pair_predictions = np.where(scores_first_texts > scores_second_texts, 1, 2)
submission_df = pd.DataFrame({
    "id": id_of_testing_pairs,
    "real_text_id": pair_predictions
})
submission_df.to_csv("submission.csv", index=False)


