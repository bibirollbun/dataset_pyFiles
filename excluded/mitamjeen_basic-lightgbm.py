# Install the library for calculating text statistics
!pip install textstat -q

# Import necessary libraries
import pandas as pd
import numpy as np
import os
import glob
from tqdm.auto import tqdm
import textstat
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')


class Config:
    BASE_PATH = "/kaggle/input/fake-or-real-the-impostor-hunt/data/"
    TRAIN_PATH = os.path.join(BASE_PATH, "train")
    TEST_PATH = os.path.join(BASE_PATH, "test")
    TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
    SUBMISSION_FILE = "submission.csv"
    N_SPLITS = 10
    RANDOM_STATE = 42
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'num_leaves': 20,
        'max_depth': 5,
        'seed': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("NLTK data downloaded.")

# Run the download function
download_nltk_data()

# Instantiate the config class
config = Config()



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

def load_data(config):
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

# Load the data
train_df, test_df = load_data(config)
print("Train DataFrame head:")
print(train_df.head())
print("\nTest DataFrame head:")
print(test_df.head())



def get_text_features(text):
    # Ensure text is a string and not empty
    if not isinstance(text, str) or not text.strip():
        # Define all possible feature names to return a dict of zeros
        zero_features = [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'avg_sentence_length', 'unique_word_count', 'ttr', 'stopword_count', 
            'stopword_ratio', 'punctuation_count', 'flesch_reading_ease', 
            'flesch_kincaid_grade', 'gunning_fog', 'smog_index', 'coleman_liau_index', 
            'automated_readability_index', 'dale_chall_readability_score', 'linsear_write_formula'
        ]
        return {feat: 0 for feat in zero_features}

    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    word_count = len(words)
    
    # Handle cases where tokenization results in zero words
    if word_count == 0:
        return get_text_features("") # Recurse with empty string to get zero dict

    stop_words = set(stopwords.words('english'))
    
    features = {
        'char_count': len(text),
        'word_count': word_count,
        'sentence_count': len(sentences),
        'avg_word_length': np.mean([len(w) for w in words]),
        'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in sentences]),
        'unique_word_count': len(set(w.lower() for w in words)),
        'ttr': len(set(w.lower() for w in words)) / word_count if word_count > 0 else 0,
        'stopword_count': sum(1 for w in words if w.lower() in stop_words),
        'stopword_ratio': sum(1 for w in words if w.lower() in stop_words) / word_count if word_count > 0 else 0,
        'punctuation_count': sum(1 for char in text if char in string.punctuation),
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'smog_index': textstat.smog_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'linsear_write_formula': textstat.linsear_write_formula(text)
    }
    return features

def create_features(df):
    # Get features for text_1 and text_2
    features_1 = df['text_1'].apply(get_text_features).apply(pd.Series)
    features_2 = df['text_2'].apply(get_text_features).apply(pd.Series)
    
    feature_cols = list(features_1.columns)
    
    # Create difference and ratio features
    for col in tqdm(feature_cols, desc="Creating comparison features"):
        df[f'{col}_diff'] = features_1[col].astype(float) - features_2[col].astype(float)
        # Add a small epsilon to the denominator to avoid division by zero
        df[f'{col}_ratio'] = features_1[col].astype(float) / (features_2[col].astype(float) + 1e-9)
        
    final_feature_cols = [f'{col}_diff' for col in feature_cols] + [f'{col}_ratio' for col in feature_cols]
    return df, final_feature_cols

# Create features for train and test sets
train_df, feature_cols = create_features(train_df)
test_df, _ = create_features(test_df)

print(f"\nCreated {len(feature_cols)} features.")
print("Feature columns example:", feature_cols[:5])



def train_and_predict(train_df, test_df, feature_cols, config):
    X = train_df[feature_cols]
    # The target is 0 if real_text_id is 1, and 1 if real_text_id is 2.
    # This aligns with the model's binary classification output.
    y = train_df['real_text_id'].apply(lambda x: 0 if x == 1 else 1) 
    X_test = test_df[feature_cols]

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"===== Fold {fold+1} =====")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
        
        callbacks = [lgb.early_stopping(100, verbose=False)]
        
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='logloss',
                  callbacks=callbacks)
        
        # Predict probabilities for the validation set
        val_preds_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds_proba
        
        # Add this fold's test predictions to the total
        test_preds += model.predict_proba(X_test)[:, 1] / config.N_SPLITS

    oof_accuracy = accuracy_score(y, np.round(oof_preds))
    print(f"\nOverall CV Accuracy: {oof_accuracy:.5f}")
    
    return test_preds

# Run the training and get predictions
test_predictions_proba = train_and_predict(train_df, test_df, feature_cols, config)



# Convert probabilities (0 to 1) to class predictions (0 or 1)
final_predictions_class = (test_predictions_proba > 0.5).astype(int)

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id']})
# Convert class predictions (0 or 1) back to the required format (1 or 2)
submission_df['real_text_id'] = final_predictions_class + 1

# Save the submission file
submission_df.to_csv(config.SUBMISSION_FILE, index=False)

print(f"\nSubmission file created successfully: {config.SUBMISSION_FILE}")
print("Submission file head:")
print(submission_df.head())


