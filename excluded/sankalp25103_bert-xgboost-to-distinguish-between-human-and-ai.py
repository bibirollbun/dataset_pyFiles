import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import matplotlib.pyplot as plt


train_df = pd.read_csv("mercor-ai-detection/train.csv", engine="python", on_bad_lines="skip")
test_df = pd.read_csv("mercor-ai-detection/test.csv", engine="python", on_bad_lines="skip")

# Combine topic and answer for both train and test
train_df["text"] = train_df["topic"].fillna("") + " " + train_df["answer"].fillna("")
test_df["text"] = test_df["topic"].fillna("") + " " + test_df["answer"].fillna("")

print("âœ… Train shape:", train_df.shape)
print("âœ… Test shape:", test_df.shape)



plt.figure(figsize=(6, 4))
train_df["is_cheating"].value_counts().plot(kind="bar", color=["#4caf50", "#f44336"])
plt.title("Class Balance (Human vs AI)")
plt.xlabel("is_cheating (0 = Human, 1 = AI)")
plt.ylabel("Count")
plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("ğŸ”¹ Using device:", device)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")
model.to(device)
model.eval()



def get_roberta_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        # Use [CLS] token representation
        cls_emb = output.last_hidden_state[:, 0, :]
        embeddings.append(cls_emb.cpu().numpy())
    return np.vstack(embeddings)



print("ğŸ”¹ Generating embeddings for train data...")
X_train_emb = get_roberta_embeddings(train_df["text"].tolist())

print("ğŸ”¹ Generating embeddings for test data...")
X_test_emb = get_roberta_embeddings(test_df["text"].tolist())


y_train = train_df["is_cheating"].values
dtrain = xgb.DMatrix(X_train_emb, label=y_train)
dtest = xgb.DMatrix(X_test_emb)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.1,
    "max_depth": 6,
    "seed": 42
}



print("ğŸš€ Training XGBoost on full data...")
model_xgb = xgb.train(params, dtrain, num_boost_round=250, verbose_eval=25)



print("ğŸ”¹ Predicting on test data...")
test_pred_prob = model_xgb.predict(dtest)



test_pred_label = (test_pred_prob >= 0.5).astype(int)


submission = pd.DataFrame({
    "id": test_df["id"],
    "is_cheating": test_pred_label
})

submission.to_csv("submission.csv", index=False)
print("\nâœ… submission.csv created successfully!")
print(submission.head())


submission_prob = pd.DataFrame({
    "id": test_df["id"],
    "is_cheating": test_pred_prob
})
submission_prob.to_csv("submission_prob.csv", index=False)
print("\nğŸ“� submission_prob.csv saved (with probabilities)")

