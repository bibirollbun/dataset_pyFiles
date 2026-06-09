import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

path_to_ds = "/kaggle/input/15-juli-2025-drw/"
file_short_names = ['0.89178', '0.90038', '0.95002', '0.83975', '0.86767', '0.88377']

params = [0.1, 0.15, 0.4, 0.05, 0.1, 0.2]

def iBlend(path_to_ds, file_short_names, sls):
    subms = []
    
    for name in file_short_names:
        filename = f"submission {name}.csv"
        df = pd.read_csv(path_to_ds + filename)
        df.columns = ['row_id', name]
        subms.append(df)

    df_subms = subms[0]
    for i in range(1, len(subms)):
        df_subms = df_subms.merge(subms[i], on="row_id")

    print("Submissions Scores:")
    for i, name in enumerate(file_short_names):
        print(f"{name}: weight = {sls[i]}")

    corr_matrix = df_subms.drop(columns="row_id").corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix between Submission Files")
    plt.show()

    df_subms["target"] = 0
    for i, name in enumerate(file_short_names):
        df_subms["target"] += sls[i] * df_subms[name]

    final = df_subms[["row_id", "target"]].copy()
    return final

submission = iBlend(path_to_ds, file_short_names, params)

submission.to_csv("iBlend_submission.csv", index=False)
print("✅ Submission saved as iBlend_submission.csv")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_names = [
    "submission 0.89178.csv",
    "submission 0.90038.csv",
    "submission 0.95002.csv",
    "submission 0.83975.csv",
    "submission 0.86767.csv",
    "submission 0.88377.csv"
]

weights = [0.1, 0.15, 0.4, 0.05, 0.1, 0.2]

scores = [float(f.split()[1].replace(".csv", "")) for f in file_names]
names = [f.split()[1].replace(".csv", "") for f in file_names]

df = pd.DataFrame({
    "File Name": file_names,
    "Score": scores,
    "Weight": weights
})
df["Score × Weight"] = df["Score"] * df["Weight"]
weighted_score = df["Score × Weight"].sum()

df.to_csv("submission_score_weights.csv", index=False)

plt.figure(figsize=(10, 5))
bars = plt.bar(names, scores, color='skyblue')
plt.xlabel("Submission Score")
plt.ylabel("RMSE")
plt.title("Individual Submission Scores")

plt.axhline(weighted_score, color='red', linestyle='--', label=f'Blend Score = {weighted_score:.5f}')

for bar, score in zip(bars, scores):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{score:.5f}', ha='center', va='bottom', fontsize=9)

plt.legend()
plt.tight_layout()

plt.savefig("submission_score_plot.png")
plt.show()


