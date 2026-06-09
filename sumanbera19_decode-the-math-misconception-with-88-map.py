# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from scipy.sparse import hstack, csr_matrix
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import random
import warnings
warnings.filterwarnings("ignore")


# 1. load data
end_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
end_df['text'] = end_df['QuestionText'] + " " + end_df['MC_Answer']+" "+ end_df['StudentExplanation']

df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv") 

df['text'] = df['QuestionText'] + " " + df['MC_Answer']+" "+ df['StudentExplanation']
df['Misconception'] = df['Misconception'].fillna('NA').astype(str) #fillna NA
df['target_cat'] = df.apply(
    lambda x: x['Category'] + ":" + x['Misconception'], axis=1
) # Category:Misconception 


df


# 2. feature (the code copy from @abdmental01 MAP - XGB notebook to process feature)
re_frac_slash = re.compile(r'(\d+)\s*/\s*(\d+)')
re_frac_latex = re.compile(r'\\frac\{([^\}]+)\}\{([^\}]+)\}')
re_newlines = re.compile(r'\n+')
re_spaces = re.compile(r'\s+')
re_punct = re.compile(r'[^a-zA-Z0-9\s_]')


def txt_clean(text):
    text = re_frac_slash.sub(r'FRAC_\1_\2', text)
    text = re_frac_latex.sub(r'FRAC_\1_\2', text)
    text = re_newlines.sub(' ', text)
    text = re_spaces.sub(' ', text)
    text = re_punct.sub('', text)
    return text.strip().lower()


def extract_math_features(text):
    text = text.lower()

    features = {}
    features['frac_count'] = len(re.findall(r'FRAC_\d+_\d+|\\frac', text))
    features['number_count'] = len(re.findall(r'\b\d+\b', text))
    features['operator_count'] = len(re.findall(r'[\+\-\*\/\=]', text))
    features['multiply_sign_count'] = len(re.findall(r'[\*Ã—Â·]|times', text))
    features['power_count'] = len(re.findall(r'\^|\*\*|\b[sS]quared\b|\b[cC]ubed\b', text))

    return features


def fast_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def create_features(df, is_train=True):
    df['mc_answer_len'] = df['MC_Answer'].astype(str).str.len()
    df['explanation_len'] = df['StudentExplanation'].astype(str).str.len()
    df['question_len'] = df['QuestionText'].astype(str).str.len()
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)

    for col in ['QuestionText', 'MC_Answer']:
        math_features = df[col].apply(extract_math_features).apply(pd.Series)
        prefix = 'mc_' if col == 'MC_Answer' else ''
        math_features.columns = [f'{prefix}{c}' for c in math_features.columns]
        df = pd.concat([df, math_features], axis=1)

    return df

print(f"Train shape: {df.shape}")


df = create_features(df, is_train=True)
end_df = create_features(end_df, is_train=False)

df['cleaned_text'] = df['text'].apply(txt_clean).apply(fast_lemmatize)
end_df['cleaned_text'] = end_df['text'].apply(txt_clean).apply(fast_lemmatize)

# encoded target_cat
le = LabelEncoder()
df['target_encoded'] = le.fit_transform(df['target_cat'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Number of target classes: {n_classes}")


# 3. Extract TF-IDF features from text 
tfidf = TfidfVectorizer(
    max_features=1000,  # Limit features  
    ngram_range=(1, 2),  # use unigram + bigram 
    stop_words='english',  # Remove common
    min_df=5  # Ignore low word 
)

total_embedding = pd.concat([df['cleaned_text'], end_df['cleaned_text']])
tfidf.fit(total_embedding)

train_embed = tfidf.transform(df['cleaned_text'])
test_embed = tfidf.transform(end_df['cleaned_text'])


# 4. List of numerical feature columns
num_cols = ['mc_answer_len', 'explanation_len', 'question_len','explanation_to_question_ratio', 'frac_count',
            'number_count','operator_count', 'mc_frac_count', 'mc_number_count','mc_operator_count']

num_fe = [f for f in num_cols if f in df.columns]

train_num = df[num_fe].fillna(0).values
test_num = end_df[num_fe].fillna(0).values


# Horizontally stack sparse matrices:
train_ = sparse.hstack([train_embed, sparse.csr_matrix(train_num)])
test_ = sparse.hstack([test_embed, sparse.csr_matrix(test_num)])

print(f"Train Final Shape: {train_.shape}")
print(f"Test Final Shape: {test_.shape}")

X_features = hstack([train_embed, csr_matrix(train_num)])  # Combine sparse matrices
 
y = df['target_encoded'].values  # Ensure y is a NumPy array
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)



rf_model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=0)

xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=n_classes,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

# === Voting Ensemble (RandomForest + XGBoost) ===
# Soft voting ensemble


ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model)
    ],
    voting='soft',
    weights=[2, 1],
    verbose=True
)

ensemble_model.fit(X_train, y_train)


# === Evaluate ===



y_pred_category = ensemble_model.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred_category))


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def get_top3_predictions(clf, X, classes):
    # Function to get Top 3 predictions with probabilities for each sample
    
    probas = clf.predict_proba(X)
    top3_indices = np.argsort(probas, axis=1)[:, -3:][:, ::-1] 
    top3_classes = [[classes[idx] for idx in sample_indices] for sample_indices in top3_indices]
    return top3_classes


# 1. Get class names from trained classifiers
category_classes = ensemble_model.classes_
 
top3_categories = get_top3_predictions(ensemble_model, X_test, category_classes)


# 2. Generate final prediction strings 
final_predictions = []
for i in range(X_test.shape[0]):
    cat_pred = top3_categories[i]  # Category predict
    
    combined_pred=[]
    for k in cat_pred:
        combined_pred.append(target_classes[k])

    # Combine all predictions into a string separated by spaces 
    pred_str = " ".join(combined_pred)
    final_predictions.append(pred_str)

submission_test = pd.DataFrame({
    "Predicted": final_predictions
})


submission_test['Predicted'] = submission_test['Predicted'].apply(lambda x: [tag.strip() for tag in x.split(' ')])
y_test_list = [[le.inverse_transform([label])[0]] for label in y_test]

score=mapk(y_test_list,submission_test['Predicted'])
print(submission_test['Predicted'].iloc[:5])
print(f"\nðŸ“Š test MAP@3 Score: {score}")


# load end test data 
end_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv") 
end_df['text'] = end_df['QuestionText'] + " " + end_df['MC_Answer']+" "+ end_df['StudentExplanation']
end_df_test = tfidf.transform(end_df['text'])
end_df.head()


def get_top3_predictions(clf, X, classes):
    # Function to get Top 3 predictions with probabilities for each sample
    probas = clf.predict_proba(X)
    top3_indices = np.argsort(probas, axis=1)[:, -3:][:, ::-1] 
    top3_classes = [[classes[idx] for idx in sample_indices] for sample_indices in top3_indices]
    return top3_classes
 
# 1. Get class names from trained classifiers
category_classes = ensemble_model.classes_
top3_categories = get_top3_predictions(ensemble_model, test_, category_classes)
 
# 2. Generate final prediction strings 
final_predictions = []
for i in range(test_.shape[0]):
    cat_pred = top3_categories[i]  # Category predict
    
    combined_pred=[]
    for k in cat_pred:
        combined_pred.append(target_classes[k])

    # Combine all predictions into a string separated by spaces 
    pred_str = " ".join(combined_pred)
    final_predictions.append(pred_str)


# 3. Create submission file
submission_df = pd.DataFrame({
    "row_id": end_df['row_id'],  # Use end test row IDs
    "Category:Misconception": final_predictions
})
submission_df.to_csv("submission.csv", index=False)
print("Submission file generated: submission.csv")
submission_df.head()




