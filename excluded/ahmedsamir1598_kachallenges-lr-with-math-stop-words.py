import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV


CLASS_NAMES = [
    "Algebra",
    "Geometry and Trigonometry",
    "Calculus and Analysis",
    "Probability and Statistics",
    "Number Theory",
    "Combinatorics and Discrete Math",
    "Linear Algebra",
    "Abstract Algebra and Topology"
]

def math_text_preprocessor(text):
    # Math-specific preprocessing
    text = re.sub(r'\$(.*?)\$', r' MATH_EXPR \1 MATH_EXPR ', text)  # Preserve math expressions
    text = re.sub(r'\\\w+', ' ', text)  # Remove LaTeX commands but keep content
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation except math symbols
    text = re.sub(r'\d+', ' NUM ', text)  # Normalize numbers
    return text.lower().strip()

train_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")  
test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")    

X_train = train_df["Question"].apply(math_text_preprocessor).values
y_train = train_df["label"].values
X_test  = test_df["Question"].apply(math_text_preprocessor).values

math_stop_words = {'find', 'prove', 'show', 'calculate', 'determine', 'let', 'given', 'solve'}

# Create TF-IDF pipeline with math-specific features
tfidf_pipe = make_pipeline(
    TfidfVectorizer(
        stop_words=list(math_stop_words),
        ngram_range=(1, 2),  # Add bigrams
        max_features=25000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,  # Use 1 + log(tf)
        analyzer='word',
        token_pattern=r'\b[^\d\W]+\b'  # Exclude pure numbers
    ),
    FunctionTransformer(lambda x: x.tocsc()) 
)


NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Initialize arrays
oof_preds = np.zeros(len(X_train), dtype=int)
test_preds = np.zeros((len(X_test), NUM_FOLDS), dtype=int)

optimal_logreg = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    random_state=42
)

# Updated cross-validation loop
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFold {fold+1}/{NUM_FOLDS}")
    
    # Split data
    X_trn, X_val = X_train[trn_idx], X_train[val_idx]
    y_trn, y_val = y_train[trn_idx], y_train[val_idx]
    
    # Create pipeline directly with tfidf and optimized logistic regression
    model = make_pipeline(tfidf_pipe, optimal_logreg)
    
    # Fit model
    model.fit(X_trn, y_trn)
    
    # Validation predictions
    y_val_pred = model.predict(X_val)
    oof_preds[val_idx] = y_val_pred
    
    # Test predictions stacking
    test_preds[:, fold] = model.predict(X_test)
    
    # Fold metrics
    print(classification_report(y_val, y_val_pred, target_names=CLASS_NAMES))
    fold_f1 = f1_score(y_val, y_val_pred, average="micro")
    print(f"Fold {fold+1} F1 (micro): {fold_f1:.4f}")




# Ensemble predictions (majority vote)
final_test_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=test_preds)

# OOF evaluation
oof_f1 = f1_score(y_train, oof_preds, average="micro")
print("\nOverall OOF Metrics:")
print(classification_report(y_train, oof_preds, target_names=CLASS_NAMES))
print(f"Overall OOF F1 (micro): {oof_f1:.4f}")

submission = pd.DataFrame({"id": test_df["id"], "label": final_test_preds})
submission.to_csv("submission.csv", index=False)
print("\nSubmission saved to submission.csv")




