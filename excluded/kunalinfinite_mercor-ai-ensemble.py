import pandas as pd
deberta_sub=pd.read_csv("/kaggle/input/mercor-ai-deberta-small/submission.csv")
roberta_sub=pd.read_csv("/kaggle/input/mercor-ai-roberta/submission.csv")


display(deberta_sub)
display(roberta_sub)


assert all(deberta_sub["id"] == roberta_sub["id"]), "IDs are not aligned!"



import pandas as pd

# Load both submissions
deberta_sub = pd.read_csv("/kaggle/input/mercor-ai-deberta-small/submission.csv")
roberta_sub = pd.read_csv("/kaggle/input/mercor-ai-roberta/submission.csv")

# Define weights
w1, w2 = 0.5, 0.5  # DeBERTa, RoBERTa

# Weighted average
final = deberta_sub.copy()
final["is_cheating"] = deberta_sub["is_cheating"] * w1 + roberta_sub["is_cheating"] * w2

# Save final ensemble
final.to_csv("weighted_ensemble.csv", index=False)
print("✅ Weighted ensemble saved as weighted_ensemble.csv")


