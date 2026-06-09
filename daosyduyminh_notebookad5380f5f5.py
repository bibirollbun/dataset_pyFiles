import pandas as pd
from pathlib import Path

# ===== ĐƯỜNG DẪN 2 FILE SUBMISSION =====
PATH_A = "/kaggle/input/thh-greedy-goose-enhanced/submission.csv"
PATH_B = "/kaggle/input/genetic-algorithm-a-type-of-black-box-optimizatio/submission.csv"

# ===== LOAD FILE =====
df_a = pd.read_csv(PATH_A)
df_b = pd.read_csv(PATH_B)

# Đảm bảo model_id giống nhau
assert all(df_a["model_id"] == df_b["model_id"]), "Mismatch model_id!"

# Lấy danh sách cột trigger (225 cột sau model_id)
trigger_cols = [c for c in df_a.columns if c != "model_id"]

# ===== ENSEMBLE =====
# Trung bình đơn giản
df_mean = df_a.copy()
df_mean[trigger_cols] = (df_a[trigger_cols] + df_b[trigger_cols]) / 2.0

# Trung bình có trọng số (ví dụ 0.7 cho A, 0.3 cho B)
wA, wB = 0.2, 0.7
df_weighted = df_a.copy()
df_weighted[trigger_cols] = wA * df_a[trigger_cols] + wB * df_b[trigger_cols]

# ===== SAVE FILE =====
out_mean = Path("./submission_mean.csv")
out_weighted = Path("./submission.csv")

# df_mean.to_csv(out_mean, index=False)
df_weighted.to_csv(out_weighted, index=False)

print(f"Saved:\n- {out_mean}\n- {out_weighted}")





