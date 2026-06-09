import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import time
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report


TRAIN_DATA_PATH = r'/kaggle/input/mercor-ai-detection/train.csv'
TEST_DATA_PATH = r'/kaggle/input/mercor-ai-detection/test.csv'
SAMPLE_SUBMISSION_PATH = r'/kaggle/input/mercor-ai-detection/sample_submission.csv'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
g = torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at {path} does not exist.")
    
    df = pd.read_csv(path)
    print(f"Data loaded successfully from {path}. Shape: {df.shape}")
    print("-"*50)
    print("Columns in the dataset:", df.columns.tolist())
    print("-"*50)
    print("First 5 rows of the dataset:")
    print(df.head())
    print("-"*50)
    return df


train_df = load_data(TRAIN_DATA_PATH)


test_df = load_data(TEST_DATA_PATH)


def plot_distribution(df: pd.DataFrame, column: str):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


plot_distribution(train_df, 'is_cheating')  


def plot_seq_len_distribution(df: pd.DataFrame, column: str = 'answer'):
    length = df[column].apply(lambda x: str(x).split()).apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(length, bins=30, kde=True)
    plt.title(f'Sequence Length Distribution of {column}')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()


plot_seq_len_distribution(train_df)


plot_seq_len_distribution(train_df, 'topic')


def box_plot_seq_len(df: pd.DataFrame, column: str = 'answer'):
    length = df[column].apply(lambda x: str(x)).apply(len)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=length, color='lightblue')
    plt.title(f'Box Plot of Sequence Lengths in {column}')
    plt.xlabel('Length')
    plt.show()


box_plot_seq_len(train_df, 'answer')
box_plot_seq_len(train_df, 'topic')


def percentile_lengths(df: pd.DataFrame, column: str = 'answer', percentiles: list = [25, 75]):
    length = df[column].apply(lambda x: str(x).split()).apply(len)
    results = {}
    for p in percentiles:
        results[f'{p}th_percentile'] = np.percentile(length, p)
        print(f"{p}th Percentile Length in {column}: {results[f'{p}th_percentile']}")
    print("-"*50)
    return results


train_results = percentile_lengths(train_df, 'answer')
topic_results = percentile_lengths(train_df, 'topic')


def count_length_outliers(df: pd.DataFrame, column: str):
    length = df[column].apply(lambda x: str(x).split()).apply(len)
    Q1 = length.quantile(0.25)
    Q3 = length.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    low_outliers = df[length < lower_bound]
    high_outliers = df[length > upper_bound]
    print(f"Number of low length outliers in {column}: {low_outliers.shape[0]}")
    print(f"Number of high length outliers in {column}: {high_outliers.shape[0]}")
    print("-"*50)
    return low_outliers, high_outliers

def same_outliers(low1, high1, low2, high2):
    low_common = pd.merge(low1, low2, how='inner')
    high_common = pd.merge(high1, high2, how='inner')
    print(f"Number of common low length outliers: {low_common.shape[0]}")
    print(f"Number of common high length outliers: {high_common.shape[0]}")
    print("-"*50)


low_answer, high_answer = count_length_outliers(train_df, 'answer')
low_topic, high_topic = count_length_outliers(train_df, 'topic')
same_outliers(low_answer, high_answer, low_topic, high_topic)


def drop_outliers(df: pd.DataFrame, low1, high1, low2, high2) -> pd.DataFrame:
    low_common = pd.merge(low1, low2, how='inner').index
    high_common = pd.merge(high1, high2, how='inner').index
    cleaned_df = df.drop(index=low_common.union(high_common))
    print(f"Data shape after dropping common outliers: {cleaned_df.shape}")
    return cleaned_df


cleaned_train_df = drop_outliers(train_df, low_answer, high_answer, low_topic, high_topic)


def concat_texts(df: pd.DataFrame, col1: str = 'topic', col2: str = 'answer', sep: str = ' [SEP] ') -> pd.DataFrame:
    df['full_text'] = df[col1].astype(str) + sep + df[col2].astype(str)
    print(f"Concatenated '{col1}' and '{col2}' into 'full_text'.")
    return df


cleaned_train_df = concat_texts(cleaned_train_df, 'topic', 'answer', ' [SEP] ')


X = cleaned_train_df['full_text'].values
y = cleaned_train_df['is_cheating'].values


stratified_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)


def train_and_evaluate_fold(count, train_idx, val_idx, X=X, y=y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,4))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    y_val_pred = model.predict(X_val_tfidf)
    y_val_proba = model.predict_proba(X_val_tfidf)[:, 1]
    
    accuracy = accuracy_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    f1 = f1_score(y_val, y_val_pred)
    
    print(f"Fold {count} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}")
    
    print(classification_report(y_val, y_val_pred))
    
    return {
        'count': count,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'val_preds': y_val_pred,
        'val_true': y_val
    }


start = time.time()
results = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate_fold)(count, train, val) 
    for count, (train, val) in tqdm(enumerate(stratified_kf.split(X, y)), total=stratified_kf.get_n_splits())
)
end = time.time()
print(f"Total time for cross-validation: {end - start:.2f} seconds")


def process_results(results):
    accuracies = [res['accuracy'] for res in results]
    roc_aucs = [res['roc_auc'] for res in results]
    f1_scores = [res['f1_score'] for res in results]
    
    print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Average ROC AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    all_val_preds = np.concatenate([res['val_preds'] for res in results])
    all_val_true = np.concatenate([res['val_true'] for res in results])
    print(all_val_preds.shape, all_val_true.shape)
    
    print("Overall Classification Report:")
    print(classification_report(all_val_true, all_val_preds))


process_results(results)


def train_final_model(X, y):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,4))
    X_tfidf = vectorizer.fit_transform(X)
    
    model = MultinomialNB()
    model.fit(X_tfidf, y)
    
    predictions = model.predict(X_tfidf)
    accuracy = accuracy_score(y, predictions)
    roc_auc = roc_auc_score(y, model.predict_proba(X_tfidf)[:, 1])
    f1 = f1_score(y, predictions)   
    print(f"Final Model - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}")
    print("-"*50)
    print("Final Model Classification Report:")
    print(classification_report(y, predictions))

    
    joblib.dump(vectorizer, 'final_vectorizer.joblib')
    joblib.dump(model, 'final_model.joblib')
    
    print("Final model and vectorizer saved.")


train_final_model(X, y)


sample_submission_df = load_data(SAMPLE_SUBMISSION_PATH)


model = joblib.load('final_model.joblib')
print("Final model loaded.")

vectorizer = joblib.load('final_vectorizer.joblib')
print("Final vectorizer loaded.")


def make_submission(test_df, model, vectorizer, submission_df, output_path='submission.csv'):
        
    def check_submissiondf_testdf_ids(submission_df, test_df):
        if not submission_df['id'].equals(test_df['id']):
            raise ValueError("IDs in submission file do not match IDs in test data.")
        print("IDs in submission file match IDs in test data.")
        
    check_submissiondf_testdf_ids(submission_df, test_df)
    
    concatated_test_df = concat_texts(test_df, 'topic', 'answer', ' [SEP] ')
    X_test = concatated_test_df['full_text'].values
    
    X_test_tfidf = vectorizer.transform(X_test)
    
    test_preds = model.predict(X_test_tfidf)
    
    submission_df['is_cheating'] = test_preds
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}.")


make_submission(test_df, model, vectorizer, sample_submission_df, output_path='submission.csv')




