# --- Standard Libraries ---
import re
import warnings

# --- Data Handling ---
import pandas as pd
import numpy as np

# --- Text Processing ---
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# --- Sparse Matrix Handling ---
from scipy import sparse  # For hstack and csr_matrix

# --- Model Evaluation ---
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# --- Machine Learning Models ---
import xgboost as xgb


# --- NLTK Data Downloads ---
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    print("NLTK 'wordnet' downloaded.")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    print("NLTK 'punkt' downloaded.")


def calculate_map3_score(true_labels, predicted_prob_arrays, label_encoder):
    """
    Calculates the MAP@3 score.
    true_labels: Series/array of true encoded labels.
    predicted_prob_arrays: Numpy array of predicted probabilities for each class.
    label_encoder: Fitted LabelEncoder object to decode predicted labels.
    """
    score = 0.
    num_samples = len(true_labels)
    
    # Get top 3 predicted class indices for each sample
    top3_indices = predicted_prob_arrays.argsort(axis=1)[:, -3:][:, ::-1]
    
    # Decode true labels to original category:misconception format
    # Ensure true_labels is an array for inverse_transform
    true_decoded_labels = label_encoder.inverse_transform(np.asarray(true_labels))
    
    # Decode predicted labels for MAP@3 calculation
    predicted_decoded_labels_list = []
    for indices_row in top3_indices:
        predicted_decoded_labels_list.append(label_encoder.inverse_transform(indices_row).tolist())

    for t, p_list in zip(true_decoded_labels, predicted_decoded_labels_list):
        if t == p_list[0]: score += 1.
        elif len(p_list) > 1 and t == p_list[1]: score += 1/2
        elif len(p_list) > 2 and t == p_list[2]: score += 1/3
    return score / num_samples

def get_math_features(text_series):
    """
    Extracts mathematical features from a text Series.
    """
    features = pd.DataFrame()
    features['frac_count'] = text_series.apply(lambda t: len(re.findall(r'FRAC_\d+_\d+|\\frac', str(t))))
    features['number_count'] = text_series.apply(lambda t: len(re.findall(r'\b\d+\b', str(t))))
    features['operator_count'] = text_series.apply(lambda t: len(re.findall(r'[\+\-\*\/\=\<\>]', str(t))))
    return features

def fast_text_lemmatize(text):
    """
    Lemmatizes text using WordNetLemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    # Tokenize before lemmatizing for better results
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

def generate_engineered_features(df):
    """
    Creates various engineered features from the dataframe.
    """
    df['mc_answer_len'] = df['MC_Answer'].astype(str).str.len()
    df['explanation_len'] = df['StudentExplanation'].astype(str).str.len()
    df['question_len'] = df['QuestionText'].astype(str).str.len()
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1e-6)

    # Math features for QuestionText
    q_math_features = get_math_features(df['QuestionText'].astype(str))
    q_math_features.columns = [f'q_{c}' for c in q_math_features.columns]
    df = pd.concat([df, q_math_features], axis=1)

    # Math features for MC_Answer
    mc_math_features = get_math_features(df['MC_Answer'].astype(str))
    mc_math_features.columns = [f'mc_{c}' for c in mc_math_features.columns]
    df = pd.concat([df, mc_math_features], axis=1)

    # Add interaction features
    df['q_text_x_exp_text_len'] = df['question_len'] * df['explanation_len']
    df['q_num_x_mc_num'] = df['q_number_count'] * df['mc_number_count']

    return df

def advanced_clean(text):
    """
    Advanced text cleaning: lowercasing, handling math formats, and removing non-alphanumeric characters.
    Similar to basic_clean but kept separate as per request.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'FRAC_\1_\2', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)
    return text.strip()


# Load data
train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample_submission = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Ensure original columns are available for generate_engineered_features later if needed
# Create copies to preserve original data for specific TF-IDF vectorizers
train_df_original = train_df.copy()
test_df_original = test_df.copy()


# Preprocessing target variable
train_df['Misconception'] = train_df['Misconception'].fillna('NA').astype(str)
train_df['target_cat'] = train_df['Category'] + ":" + train_df['Misconception']

le_target = LabelEncoder()
train_df['target_encoded'] = le_target.fit_transform(train_df['target_cat'])
target_classes = le_target.classes_
n_classes = len(target_classes)

print(f"Total number of unique target classes: {n_classes}")

# --- Feature Engineering (using the provided create_features logic, adjusted) ---
print("Generating engineered features...")
train_df = generate_engineered_features(train_df)
test_df = generate_engineered_features(test_df) # Use test_df here

# Combine text features for main TF-IDF
train_df['combined_text'] = "Question: " + train_df_original['QuestionText'].astype(str) + " Answer: " + train_df_original['MC_Answer'].astype(str) + " Explanation: " + train_df_original['StudentExplanation'].astype(str)
test_df['combined_text'] = "Question: " + test_df_original['QuestionText'].astype(str) + " Answer: " + test_df_original['MC_Answer'].astype(str) + " Explanation: " + test_df_original['StudentExplanation'].astype(str)

# Apply cleaning and lemmatization to combined text
print("Applying text cleaning and lemmatization...")
train_df['cleaned_text'] = train_df['combined_text'].apply(advanced_clean).apply(fast_text_lemmatize)
test_df['cleaned_text'] = test_df['combined_text'].apply(advanced_clean).apply(fast_text_lemmatize)

print("Creating TF-IDF features...")

# TF-IDF for cleaned combined text (word-based)
tfidf_word = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=15000) # Increased max_features
tfidf_word.fit(pd.concat([train_df['cleaned_text'], test_df['cleaned_text']]))
train_tfidf_word = tfidf_word.transform(train_df['cleaned_text'])
test_tfidf_word = tfidf_word.transform(test_df['cleaned_text'])

# TF-IDF for StudentExplanation (as per example)
tfidf_expl = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=5000) # Increased max_features
tfidf_expl.fit(pd.concat([train_df_original['StudentExplanation'].astype(str), test_df_original['StudentExplanation'].astype(str)]))
train_tfidf_expl = tfidf_expl.transform(train_df_original['StudentExplanation'].astype(str))
test_tfidf_expl = tfidf_expl.transform(test_df_original['StudentExplanation'].astype(str))

# Character TF-IDF for cleaned combined text
char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=5000) # Increased max_features
char_tfidf.fit(pd.concat([train_df['cleaned_text'], test_df['cleaned_text']]))
train_char = char_tfidf.transform(train_df['cleaned_text'])
test_char = char_tfidf.transform(test_df['cleaned_text'])

# Numeric features for concatenation
numeric_cols = [
    'mc_answer_len', 'explanation_len', 'question_len',
    'explanation_to_question_ratio',
    'q_frac_count', 'q_number_count', 'q_operator_count', # Using 'q_' prefix now
    'mc_frac_count', 'mc_number_count', 'mc_operator_count', # Using 'mc_' prefix now
    'q_text_x_exp_text_len', 'q_num_x_mc_num' # Interaction features
]

# Ensure numerical features are float type and handle NaNs/Infs before converting to sparse
for col in numeric_cols:
    if col not in train_df.columns:
        print(f"Warning: Column '{col}' not found in train_df. This might be an issue with generate_engineered_features.")
        train_df[col] = 0 # Add dummy column to prevent error
    if col not in test_df.columns:
        print(f"Warning: Column '{col}' not found in test_df. This might be an issue with generate_engineered_features.")
        test_df[col] = 0 # Add dummy column to prevent error


X_numeric = sparse.csr_matrix(train_df[numeric_cols].fillna(0).values.astype(np.float32))
X_numeric_test = sparse.csr_matrix(test_df[numeric_cols].fillna(0).values.astype(np.float32))


# Stack all features
X_train = sparse.hstack([train_tfidf_word, train_tfidf_expl, train_char, X_numeric]).tocsr()
X_test = sparse.hstack([test_tfidf_word, test_tfidf_expl, test_char, X_numeric_test]).tocsr()

y_train = train_df['target_encoded'].values

print(f"Combined feature shape (Train): {X_train.shape}")
print(f"Combined feature shape (Test): {X_test.shape}")


# --- XGBoost KFold (using xgb.DMatrix and xgb.train) ---
print("Training XGBoost model with Cross-Validation (using DMatrix)...")

oof_preds = np.zeros((len(train_df), n_classes))
test_preds = np.zeros((len(test_df), n_classes))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'objective': 'multi:softprob',
    'num_class': n_classes, # Crucial to tell XGBoost the total number of classes
    'eval_metric': 'mlogloss',
    'max_depth': 10, # Adjusted from example to fit current optimal range
    'learning_rate': 0.03, # Adjusted
    'subsample': 0.8, # Adjusted slightly
    'colsample_bytree': 0.8, # Adjusted slightly
    'tree_method': 'gpu_hist',
    'gpu_id': 0, # Assuming GPU is available and ID is 0
    'random_state': 42,
    'n_jobs': -1 # Use all available cores
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"--- Fold {fold+1}/{skf.n_splits} ---")
    
    # Create DMatrix objects for the current fold
    dtrain = xgb.DMatrix(X_train[trn_idx], label=y_train[trn_idx])
    dvalid = xgb.DMatrix(X_train[val_idx], label=y_train[val_idx])

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000, # Increased max rounds, rely on early stopping
        evals=[(dvalid, 'valid')],
        early_stopping_rounds=100, # Increased early stopping rounds
        verbose_eval=100 # Print progress every 100 rounds
    )
    
    # Predict OOF probabilities
    oof_preds[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    
    # Accumulate test predictions
    test_preds += model.predict(xgb.DMatrix(X_test), iteration_range=(0, model.best_iteration)) / skf.n_splits

# --- Evaluation ---
oof_logloss = log_loss(y_train, oof_preds)
print(f"\nOOF Log Loss: {oof_logloss:.4f}")

oof_map3_score = calculate_map3_score(y_train, oof_preds, le_target)
print(f"OOF MAP@3 Score: {oof_map3_score:.4f}")

# --- Submission ---
# Get top 3 predictions from averaged test predictions
top3_indices = test_preds.argsort(axis=1)[:, -3:][:, ::-1] # Sort descending and take top 3

test_predictions_labels = []
for indices in top3_indices:
    pred_labels = [target_classes[i] for i in indices]
    test_predictions_labels.append(' '.join(pred_labels))

sample_submission['Category:Misconception'] = test_predictions_labels
submission_filename = "submission.csv" #submission_xgboost_dmatrix_full_features.csv
sample_submission.to_csv(submission_filename, index=False)
print(f"\nSubmission file created: {submission_filename}")
print("XGBoost model training complete with improved features and DMatrix handling.")

