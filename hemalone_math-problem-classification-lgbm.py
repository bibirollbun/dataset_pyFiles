%%capture
!pip install --force-reinstall --no-cache-dir scikit-learn==1.4.1.post1 imbalanced-learn==0.12.0



import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')
test_df = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')


train_df.head()


test_df.head()


train_df.shape, test_df.shape


sns.countplot(data = train_df, x = 'label')


train_df['length'] = train_df['Question'].astype(str).apply(len)
sns.histplot(train_df['length'], bins=50)
plt.title("Text Length Distribution")
plt.show()


train_df['numeric_token_counts'] = train_df['Question'].str.findall(r'[\d+\-*/=()<>^√π÷%\.]').str.len()
sns.histplot(train_df['length'], bins=50)
plt.title("Numeric Token Distribution")
plt.show()


def clean_math_text(text):
    text = str(text)
    
    math_symbols = {
        r'\$': ' dollar ',
        r'\=': ' equals ',
        r'\<': ' less than ',
        r'\>': ' greater than ',
        r'\+': ' plus ',
        r'\-': ' minus ',
        r'\*': ' times ',
        r'\/': ' divided by ',
        r'\^': ' to the power of ',
        r'\√': ' square root ',
        r'\π': ' pi ',
        r'\∑': ' sum ',
        r'\∫': ' integral ',
        r'\∞': ' infinity '
    }
    
    for pattern, replacement in math_symbols.items():
        text = re.sub(pattern, replacement, text)
    
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = re.sub(r'\{([^}]*)\}', r' \1 ', text)
    
    text = re.sub(r"[^a-zA-Z0-9\s\.\?\!]", " ", text)
    
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)  
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)  
    
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    return text
    
train_df['Question'] = train_df['Question'].apply(clean_math_text)
test_df['Question'] = test_df['Question'].apply(clean_math_text)


def extract_math_features(text):
    features = {
        'num_count': len(re.findall(r'\d+', text)),
        'equation_count': len(re.findall(r'equals|equation|formula|solve for', text)),
        'function_count': len(re.findall(r'function|f\(x\)|derivative|integral', text)),
        'geometry_count': len(re.findall(r'angle|triangle|circle|area|volume', text)),
        'algebra_count': len(re.findall(r'variable|polynomial|matrix|vector', text)),
        'calculus_count': len(re.findall(r'derivative|integral|limit|differentiation', text)),
        'char_count': len(text),
        'word_count': len(text.split()),
        'is_proof': int('prove' in text or 'show that' in text),
        'is_compute': int('compute' in text or 'calculate' in text),
        'is_find': int('find' in text or 'determine' in text),
    }
    return pd.Series(features)

math_features_train = train_df['Question'].apply(extract_math_features)
math_features_test = test_df['Question'].apply(extract_math_features)


tfidf = TfidfVectorizer(max_features=15000,  ngram_range=(1,2))
X_text = tfidf.fit_transform(train_df['Question'])
X_test_text = tfidf.transform(test_df['Question'])


X_combined = hstack([X_text, csr_matrix(math_features_train.values)])
X_test_combined = hstack([X_test_text, csr_matrix(math_features_test.values)])
y = train_df['label']


X_dense_for_smote = X_combined.toarray()
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_dense_for_smote, y)
X_resampled_sparse = csr_matrix(X_resampled)


print(f"Before SMOTE: {dict(pd.Series(y).value_counts())}")
print(f"After SMOTE:  {dict(pd.Series(y_resampled).value_counts())}")


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y_resampled)


X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def kfold_lgbm(X, y, n_splits=5):
    oof_preds = np.zeros(len(y), dtype=int)
    models = []
    f1_micro_scores = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(np.unique(y)),
            learning_rate=0.1,
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        f1_m = f1_score(y_val, preds, average='micro')
        print(f"F1 Micro: {f1_m:.4f}")

        f1_micro_scores.append(f1_m)
        models.append(model)

    overall_f1_m = f1_score(y, oof_preds, average='micro')
    print(f"\nOverall OOF F1 Micro: {overall_f1_m:.4f}")

    return {
        'models': models,
        'oof_preds': oof_preds,
        'fold_f1_micro': f1_micro_scores,
        'overall_f1_micro': overall_f1_m
    }


results_lgbm = kfold_lgbm(X_resampled_sparse, y_encoded)


y_true = np.array(y_encoded)
y_pred = np.array(results_lgbm['oof_preds'])

unique_label_indices = np.unique(np.concatenate((y_true, y_pred)))
label_names = [str(cls) for cls in le.inverse_transform(unique_label_indices)]

print("\nClassification Report (LightGBM):")
print(classification_report(
    y_true,
    y_pred,
    labels=unique_label_indices,
    target_names=label_names
))

conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_label_indices)

plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix,
    annot=True, fmt='d', cmap='YlGnBu',
    xticklabels=label_names,
    yticklabels=label_names
)
plt.title("OOF Confusion Matrix (LightGBM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()



test_features_combined = hstack([
    tfidf.transform(test_df['Question']),
    math_features_test.values
])

test_preds_encoded = results_lgbm['models'][0].predict(test_features_combined)

test_preds = le.inverse_transform(test_preds_encoded)

submission = pd.DataFrame({
    'id': test_df['id'],
    'label': test_preds
})

submission.to_csv('submission.csv', index=False)

