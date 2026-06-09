!pip install -q sentence-transformers

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr



# Load training, test, and submission data
train_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
sample_submission = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv")

# Preview
train_df.head()



# Load model locally
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")





anchor_raw = train_df["anchor"].tolist()
target_raw = train_df["target"].tolist()

anchor_lower = [a.lower() for a in anchor_raw]
target_lower = [t.lower() for t in target_raw]

anchor_embeddings_1 = model.encode(anchor_raw, convert_to_tensor=True)
anchor_embeddings_2 = model.encode(anchor_lower, convert_to_tensor=True)

target_embeddings_1 = model.encode(target_raw, convert_to_tensor=True)
target_embeddings_2 = model.encode(target_lower, convert_to_tensor=True)

# Average both embeddings
anchor_embeddings = (anchor_embeddings_1 + anchor_embeddings_2) / 2
target_embeddings = (target_embeddings_1 + target_embeddings_2) / 2



from sklearn.linear_model import Ridge

# Use cosine similarity as a feature
X_train = util.cos_sim(anchor_embeddings, target_embeddings).diagonal().cpu().numpy().reshape(-1, 1)
y_train = train_df["score"].values

# Train ridge regression
reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)



# Encode test set anchor and target
test_anchor_embeddings = model.encode(test_df["anchor"].tolist(), convert_to_tensor=True)
test_target_embeddings = model.encode(test_df["target"].tolist(), convert_to_tensor=True)

X_test = util.cos_sim(test_anchor_embeddings, test_target_embeddings).diagonal().cpu().numpy().reshape(-1, 1)
test_scores = reg.predict(X_test)



sample_submission["score"] = np.clip(test_scores, 0, 1)



from sentence_transformers import util
import numpy as np

# Compute cosine similarity
test_scores = util.cos_sim(test_anchor_embeddings, test_target_embeddings).diagonal().cpu().numpy()

# Optional: clip values to stay within 0–1 range
test_scores = np.clip(test_scores, 0, 1)

# Assign to submission dataframe
sample_submission["score"] = test_scores

# Save submission file
sample_submission.to_csv("submission.csv", index=False)

# Preview
sample_submission.head()


