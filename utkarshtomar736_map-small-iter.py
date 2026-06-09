import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import label_ranking_average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns


# Custom MAP@3 function
def compute_map_at_3(y_true, y_proba, classes, k=3):
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:][:, ::-1]
    aps = []
    for i in range(len(y_true)):
        true_label_idx = np.where(classes == y_true.iloc[i])[0][0]
        relevant_ranks = np.where(top_k_preds[i] == true_label_idx)[0] + 1
        if len(relevant_ranks) > 0:
            rank = relevant_ranks[0]
            ap = 1.0 / rank
        else:
            ap = 0.0
        aps.append(ap)
    return np.mean(aps)


# Function for small-batch experiments
def small_batch_eval(train_data, experiment_name, model_class, vec_params={}, model_params={}, fe_params={}, batch_frac=0.1):
    sample = train_data.sample(frac=batch_frac, random_state=42)
    
    # Preprocessing
    sample['StudentExplanation'] = sample['StudentExplanation'].fillna('no_explanation')
    sample['QuestionText'] = sample['QuestionText'].fillna('')
    sample['MC_Answer'] = sample['MC_Answer'].fillna('')
    sample['Category'] = sample['Category'].fillna('UnknownCategory')
    sample['Misconception'] = sample['Misconception'].fillna('UnknownMisconception')
    sample['target'] = sample['Category'] + ':' + sample['Misconception']
    
    # Group rare classes
    counts = sample['target'].value_counts()
    rare = counts[counts < 2].index
    sample['target'] = sample['target'].apply(lambda x: 'Other' if x in rare else x)
    
    # Basic text
    sample['text'] = sample['QuestionText'] + ' ' + sample['MC_Answer'] + ' ' + sample['StudentExplanation']
    
    # Optional FE: Add length (example)
    if fe_params.get('add_length', False):
        sample['explanation_length'] = sample['StudentExplanation'].str.len()
        # For simplicity, concatenate as string (better: hstack numeric)
        sample['text'] += ' length_' + sample['explanation_length'].astype(str)
    
    X = sample['text']
    y = sample['target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    vectorizer = TfidfVectorizer(**vec_params)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    model = model_class(**model_params)
    model.fit(X_train_vec, y_train)
    
    proba = model.predict_proba(X_val_vec)
    classes = np.array(model.classes_)
    map3 = compute_map_at_3(y_val.reset_index(drop=True), proba, classes)
    
    print(f"{experiment_name} MAP@3: {map3:.4f}")
    return map3


# Example runs (load your train_data first)
train_data = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')


# Baseline repeat
br_score = small_batch_eval(train_data, 'Baseline LR', LogisticRegression, vec_params={'max_features':10000, 'ngram_range':(1,2)}, model_params={'max_iter':300, 'multi_class':'ovr', 'random_state':42})
br_score


# Iteration 1: Add class weights
class_wts = small_batch_eval(train_data, 'LR w/ Balanced Weights', LogisticRegression, vec_params={'max_features':10000, 'ngram_range':(1,2)}, model_params={'max_iter':300, 'multi_class':'ovr', 'class_weight':'balanced', 'random_state':42})
class_wts


# Iteration 2: Naive Bayes (simpler model)
nb = small_batch_eval(train_data, 'Naive Bayes', MultinomialNB, vec_params={'max_features':10000, 'ngram_range':(1,2)})
nb


# Iteration 3: FE - Add length
add_l = small_batch_eval(train_data, 'LR + Length FE', LogisticRegression, vec_params={'max_features':10000, 'ngram_range':(1,2)}, model_params={'max_iter':300, 'multi_class':'ovr', 'random_state':42}, fe_params={'add_length':True})
add_l


# Plot results (manual list; collect from runs)
experiments = ['Baseline LR', 'LR w/ Weights', 'Naive Bayes', 'LR + Length FE']
scores = [br_score, class_wts, nb, add_l]  # Replace with your actual scores
sns.barplot(x=experiments, y=scores)
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.title('MAP@3 Comparison on Small Batch')
plt.show()




