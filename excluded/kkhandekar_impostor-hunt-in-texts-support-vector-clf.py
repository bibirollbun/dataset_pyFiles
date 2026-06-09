#
# Libraries
#

# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, string, re, random, gc, pickle, math,warnings
import json, glob
from itertools import *
from datetime import date
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from datasets import Dataset
from datasets import DatasetDict
from pathlib import Path

# Transformers
from transformers import AutoTokenizer,AutoModel

# PyTorch
import torch

# NLTK / Spacy
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nlp = spacy.load('en_core_web_sm')

# Sklearn
from sklearn.model_selection import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.metrics import pairwise
from sklearn.preprocessing import *
from sklearn.utils import *
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.svm import *
from sklearn.decomposition import *

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Setting
pd.set_option('max_colwidth',None)
seed = 905
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

stopw = pd.read_json('/kaggle/input/english-stopwords/stop_words_english.json')
stopw = stopw.Stopwords.tolist()



#
# Config
#

class Config:
    BASE_PATH = "/kaggle/input/fake-or-real-the-impostor-hunt/data/"
    TRAIN_PATH = os.path.join(BASE_PATH, "train")
    TEST_PATH = os.path.join(BASE_PATH, "test")
    TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
    SUBMISSION_FILE = "submission.csv"
    PCA_COMPONENT = 20
    SEED = 905
    SPLIT_SIZE = 0.1

config = Config()



#
# ~~~ Courtesy - @Qwerty ~~~
#

#
# Custom Function - Read Text Files
#

def read_text_files_robust(df, path):
    texts_1, texts_2 = [], []
    all_dirs = glob.glob(os.path.join(path, 'article_*'))
    # Create a mapping from article_id (int) to its directory path for quick lookup
    dir_map = {int(os.path.basename(p).replace('article_', '')): p for p in all_dirs}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading files from {os.path.basename(path)}"):
        article_id = row['id']
        dir_path = dir_map.get(article_id)
        
        if dir_path:
            # Try to read both files, append empty string if a file is missing
            try:
                with open(os.path.join(dir_path, 'file_1.txt'), 'r', encoding='utf-8') as f:
                    texts_1.append(f.read())
            except FileNotFoundError:
                texts_1.append("")
            
            try:
                with open(os.path.join(dir_path, 'file_2.txt'), 'r', encoding='utf-8') as f:
                    texts_2.append(f.read())
            except FileNotFoundError:
                texts_2.append("")
        else:
            # If the article directory itself is not found
            texts_1.append("")
            texts_2.append("")

    df['text_1'] = texts_1
    df['text_2'] = texts_2
    return df


#
# Custom Function - Load Data
#

def load_data():
    train_df = pd.read_csv(config.TRAIN_CSV)
    
    # Create test_df from the directory names in the test folder
    test_dirs = glob.glob(os.path.join(config.TEST_PATH, 'article_*'))
    if not test_dirs:
        raise FileNotFoundError(f"No 'article_*' directories found in {config.TEST_PATH}")
    test_ids = [int(os.path.basename(p).replace('article_', '')) for p in test_dirs]
    test_df = pd.DataFrame(sorted(test_ids), columns=['id'])

    # Read the text files for both train and test sets
    train_df = read_text_files_robust(train_df, config.TRAIN_PATH)
    test_df = read_text_files_robust(test_df, config.TEST_PATH)
    
    return train_df, test_df


#
# ~~~ Courtesy - @Nilesh2042 ~~~
#


#
# Custom Function - pooling
#

def mean_pooling(last_hidden_state, attention_mask):
    """
    Applies mean pooling on the output of the transformer model,
    considering only the non-padded tokens.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    count = mask.sum(dim=1)
    return summed / count

        
#
# Custom Function - extract mean pooling vector
#

def extract_mean_pooling_vector(text, tokenizer, model, max_len=512, stride=256, device="cuda"):
    """
    Extracts a mean-pooled sentence vector from a long input text
    using a sliding window approach to handle overflow.
    """
    # Tokenize with overflow to cover long texts
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length"
    )

    input_ids_chunks = encoded["input_ids"]
    attention_mask_chunks = encoded["attention_mask"]

    # Move model to device
    model = model.to(device)
    model.eval()

    chunk_vectors = []

    with torch.no_grad():
        for input_ids, attention_mask in zip(input_ids_chunks, attention_mask_chunks):
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_vector = mean_pooling(outputs.last_hidden_state, attention_mask)
            chunk_vectors.append(pooled_vector.squeeze(0))

    # Average all chunk vectors into a single representation
    final_vector = torch.stack(chunk_vectors).mean(dim=0)

    return final_vector.cpu()

    
#
# Custom Function - extract features
#

def extract_features(dataset, tokenizer, model):
    """
    Extracts interaction-based features from each text pair.

    Returns:
        features: numpy array of shape [num_samples, feature_dim]
        ids: list of sample IDs
    """
    features = []
    ids = []

    for row in tqdm(dataset, desc="Extracting features"):
        vec1 = extract_mean_pooling_vector(row['text_1'], tokenizer, model)
        vec2 = extract_mean_pooling_vector(row['text_2'], tokenizer, model)

        # Compute interaction vectors
        diff = vec1 - vec2
        prod = vec1 * vec2

        # Concatenate all parts
        final_vec = torch.cat([vec1, vec2, diff, prod])
        features.append(final_vec.numpy())
        ids.append(row['id'])

    return np.array(features), ids


#
# Custom Function - Clean string
# 

def clean_string(text, stem="None"):

    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    # Note: that this line can be augmented and used over
    # to replace any characters with nothing or a space
    text = re.sub(r'\n', ' ', text)

    # Remove email address
    pattern = r'\S*@\S*\s?'
    text = re.sub(pattern, "", text)

    # Remove URLs
    pattern = r"https?://\S+" or "http?://\S+"
    text = re.sub(pattern, "", text)
	
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove special chars
    text_filtered = [re.sub(r'[^a-zA-Z0-9\s]', ' ', w) for w in text_filtered]
    
    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', ' ', w) for w in text_filtered]

    # Remove Whitespaces
    text_filtered = [w.strip() for w in text_filtered]


    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer() 
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    # Word > 3 letters only
    text_stemmed = [w for w in text_stemmed if len(w) >= 3]
	
    final_string = ' '.join(text_stemmed)
    return final_string


#
# Load Data
#

train, test = load_data()

# update real_text_id to label
train['label'] = train['real_text_id'].apply(lambda x: 1 if x == 1 else 0)

# drop column
train.drop('real_text_id',axis=1,inplace=True)

# view
print(f"Training size: {train.shape} | Testing size: {test.shape}")
train.head()


#
# Preprocessing Text - Clean
#
train_df = train.copy()
test_df = test.copy()

train_df['text_1'] = train_df['text_1'].apply(lambda x: clean_string(x,stem='Spacy'))
train_df['text_2'] = train_df['text_2'].apply(lambda x: clean_string(x,stem='Spacy'))

test_df['text_1'] = test_df['text_1'].apply(lambda x: clean_string(x,stem='Spacy'))
test_df['text_2'] = test_df['text_2'].apply(lambda x: clean_string(x,stem='Spacy'))

# view
train_df.head()


#
# Convert to HuggingFace datasets
#

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into DatasetDict
raw_datasets = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# view
print(raw_datasets)


#
# Load pre-trained model
#

checkpoint = "distilbert-base-uncased"   # or distilgpt2 / distilbert-base-uncased-finetuned-sst-2-english
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model_ = AutoModel.from_pretrained(checkpoint, num_labels=2)


#
# Extract features & define PCA - train dataset
#

x_train_raw, train_ids = extract_features(raw_datasets["train"], tokenizer, model_)
x_test_raw, test_ids = extract_features(raw_datasets["test"], tokenizer, model_)

pca_model = PCA(n_components=config.PCA_COMPONENT)


# 
# Fit on training features
#
x_train = pca_model.fit_transform(x_train_raw)
x_test = pca_model.fit_transform(x_test_raw)


#
# Extract labels
#
y_train = np.array([ex["label"] for ex in raw_datasets["train"]])


#
# Define Estimators
#

svm_models = {
    "SVC": SVC(random_state=config.SEED),
    "LinearSVC": LinearSVC(random_state=config.SEED),
    "NuSVC" : NuSVC(random_state=config.SEED)
}


#
# K-Fold cross validation
#
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
results = {}

for name, model in svm_models.items():
    # Store metrics across folds
    accuracies, precisions = [], []

    # 5-fold CV loop
    for train_idx, val_idx in kf.split(x_train, y_train):
        x_tr, x_val = x_train[train_idx], x_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Train model and predict
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_val)

        # Compute metrics -- balanced accuracy
        acc = balanced_accuracy_score(y_val, y_pred)

        # Compute metrics -- precision_score
        precision = precision_score(y_val, y_pred, average='macro')

        # Append metrics
        accuracies.append(acc)
        precisions.append(precision)
    
    # Average metrics across folds
    results[name] = {
        "Accuracy": np.mean(accuracies),
        "Precision": np.mean(precisions),
    }


# Create DataFrame and sort by F1-score
df_results = pd.DataFrame(results).T.sort_values(by="Accuracy", ascending=False).round(4)
print(df_results)


#
# Hyperparameter estimator - Optuna
#

# optuna objective 
def objective(trial):

    # train & split
    x_tr, x_val, y_tr, y_val = train_test_split(x_train,y_train,test_size=config.SPLIT_SIZE,random_state=config.SEED,stratify=y_train)

    # define params
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    C = trial.suggest_float("C", 1e-5, 1e5, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    degree = trial.suggest_int('degree',3,5,1)

    
    # define model
    if kernel == "poly":
        classifier_obj = SVC(kernel=kernel,
                                     C=C,
                                     gamma=gamma,
                                     class_weight = 'balanced',
                                     random_state = config.SEED,
                                     degree = degree)
    
    else:
        classifier_obj = SVC(kernel=kernel,
                                     C=C,
                                     gamma=gamma,
                                     class_weight = 'balanced',
                                     random_state = config.SEED)
        

    # Train model and predict
    #classifier_obj.fit(x_tr, y_tr)
    #y_pred = classifier_obj.predict(x_val)

    # pruner
    for step in range(100):
        classifier_obj.fit(x_tr, y_tr)
        intermediate_value = balanced_accuracy_score(y_val, classifier_obj.predict(x_val))

        trial.report(intermediate_value, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    # Compute metrics -- balanced accuracy
    #bac = balanced_accuracy_score(y_val, y_pred)

    return balanced_accuracy_score(y_val, classifier_obj.predict(x_val))


#
# Run trials
#

study = optuna.create_study(direction="maximize",study_name='OptimiseSVM',
                            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5))

study.optimize(objective, n_trials=200, gc_after_trial=True, show_progress_bar=True)

# view
print(f"Best value achieved: {study.best_value}\n" )
print(f"Best Params: {study.best_params}")


#
# Best Params & Optimised Model
#

# best params from trial
best_params = study.best_params

# additional params
best_params.update({'class_weight': "balanced", 
                    'random_state' : config.SEED})


# optimised model
model_opt = SVC(**best_params)

# summary
model_opt


#
# Re-Train & Prediction (test)
#

# retrain
model_opt.fit(x_train,y_train)

# predict
y_pred_test = model_opt.predict(x_test)


#
# Submission File
#

# submission dataframe
submission = pd.DataFrame({"id": test_df.id.tolist(),
                           "real_text_id": y_pred_test.tolist()
                            })

# convert "real_text_id" to required format(1 & 2)
submission['real_text_id'] = submission['real_text_id'] + 1

# export to csv
submission.to_csv('submission.csv',index=False)

# view
submission.head()


