import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load data
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


def compute_similarity_features(df, vectorizer=None, fit=False):
    """
    Compute similarity features between body and positive/negative examples
    """
    features = []
    
    if fit:
        # Fit vectorizer on all text
        all_texts = []
        for col in ["body", "positive_example_1", "positive_example_2", 
                    "negative_example_1", "negative_example_2"]:
            all_texts.extend(df[col].tolist())
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        vectorizer.fit(all_texts)
    
    for idx, row in df.iterrows():
        # Get vectors
        body_vec = vectorizer.transform([row["body"]])
        pos1_vec = vectorizer.transform([row["positive_example_1"]])
        pos2_vec = vectorizer.transform([row["positive_example_2"]])
        neg1_vec = vectorizer.transform([row["negative_example_1"]])
        neg2_vec = vectorizer.transform([row["negative_example_2"]])
        
        # Compute similarities
        sim_pos1 = cosine_similarity(body_vec, pos1_vec)[0][0]
        sim_pos2 = cosine_similarity(body_vec, pos2_vec)[0][0]
        sim_neg1 = cosine_similarity(body_vec, neg1_vec)[0][0]
        sim_neg2 = cosine_similarity(body_vec, neg2_vec)[0][0]
        
        # Aggregate features
        avg_pos_sim = (sim_pos1 + sim_pos2) / 2
        avg_neg_sim = (sim_neg1 + sim_neg2) / 2
        
        features.append({
            "pos_neg_diff": avg_pos_sim - avg_neg_sim
        })
    
    return pd.DataFrame(features), vectorizer


# Compute features
print("Computing similarity features...")
train_features, vectorizer = compute_similarity_features(train_df, fit=True)
test_features, _ = compute_similarity_features(test_df, vectorizer=vectorizer, fit=False)

print(f"Train features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")


# Normalize predictions to 0-1 range
min_val = train_features["pos_neg_diff"].min()
max_val = train_features["pos_neg_diff"].max()

test_pred = (test_features["pos_neg_diff"] - min_val) / (max_val - min_val)
test_pred = np.clip(test_pred, 0, 1)

print(f"Predictions shape: {test_pred.shape}")
print(f"Predictions range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")


# Create submission
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "rule_violation": test_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission file created!")
print(submission.head(10))

