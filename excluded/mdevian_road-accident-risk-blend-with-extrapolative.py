import pandas as pd, numpy as np

# 1) Load & align by id
files = [
         "/kaggle/input/road-accident-playground/submission (37).csv",
         "/kaggle/input/road-accident-playground/submission (40).csv",
        ]
cols  = []  
dfs = []
for f in files:
    df = pd.read_csv(f)
    if "id" not in df: raise ValueError("butuh kolom id")

    pred = next((c for c in ["accident_risk","prediction","pred","y","target","score","prob"]
                 if c in df.columns), None)
    if pred is None:  
        num = [c for c in df.select_dtypes(include=[np.number]).columns if c!="id"]
        pred = num[0]
    cols.append(pred)
    df = df[["id", pred]].rename(columns={pred: f})
    dfs.append(df)

M = dfs[0]
for df in dfs[1:]:
    M = M.merge(df, on="id", how="inner")
X = M[files].astype(float).fillna(0.0)

# 2)  blend
# mean
M["blend_mean"] = X.mean(1)
# median
M["blend_median"] = X.median(1)
# rank-avg (0..1), 
R = X.rank(axis=0, method="average", pct=True)
M["blend_rank"] = R.mean(1)

# 3) Minimum-Variance weights (shrinkage kecil)
C = np.cov(X.values, rowvar=False)
lam = 1e-3
C_reg = C + lam*np.eye(C.shape[0])
w = np.linalg.solve(C_reg, np.ones(C.shape[0]))
w = w / w.sum()
M["blend_mve"] = X.values @ w

# 4) choose one
out = M[["id","blend_mve"]].rename(columns={"blend_mve":"accident_risk"})
out.to_csv("submission_blend.csv", index=False)


print("X shape:", X.shape)  # (N, M)
corr = pd.DataFrame(np.corrcoef(X.values, rowvar=False), index=files, columns=files)
print("\nKorelasi antar submission:")
print(corr.round(6))

base = X[files[0]].values
max_diff = {f: float(np.max(np.abs(X[f].values - base))) for f in files}
print("\nmax|diff| dibanding", files[0], ":", max_diff)

print("\nBobot MVE (sebelum perbaikan):", dict(zip(files, np.round(w,6))))


import numpy as np, pandas as pd

base = X[files[0]].values
dists = {f: float(np.max(np.abs(X[f].values - base))) for f in files}
pair = sorted(dists, key=dists.get, reverse=True)[:2]  
print("two candidate:", pair, " | max|diff|:", dists[pair[0]], dists[pair[1]])

Ws = [0.2, 0.4, 0.6, 0.8]
for w in Ws:
    w_pair = {pair[0]: w, pair[1]: 1.0 - w}
    rest = [f for f in files if f not in pair]
    if rest:
        w_pair = {k: v*0.9 for k,v in w_pair.items()}
        rest_w = 0.1/len(rest)
        for r in rest:
            w_pair[r] = rest_w

    w_vec = np.array([w_pair[f] for f in files], dtype=float)
    w_vec = w_vec / w_vec.sum()
    pred = X.values @ w_vec

    out = M[["id"]].copy()
    out["accident_risk"] = pred
    out.to_csv(f"submission_grid_{w:.2f}.csv", index=False, float_format="%.9f")
    print(f"Saved: submission_grid_{w:.2f}.csv  | w={dict(zip(files, np.round(w_vec,3)))}")


ref_name = files[-1]  
R = X.rank(axis=0, method="average", pct=True)
ravg = R.mean(axis=1).values
ref_sorted = np.sort(X[ref_name].values)
idx = np.clip((len(ref_sorted)-1)*ravg, 0, len(ref_sorted)-1).astype(int)
rank_qmap = ref_sorted[idx]
out1 = M[["id"]].copy()
out1["accident_risk"] = rank_qmap
out1.to_csv("submission_rank_qmap.csv", index=False, float_format="%.9f")

C = np.cov(X.values, rowvar=False)
tau = float(np.mean(np.diag(C))) if X.shape[1] > 1 else 1.0
lam = 1e-2 * tau
C_reg = C + lam*np.eye(C.shape[0])
w_mve = np.linalg.solve(C_reg, np.ones(C.shape[0])); w_mve /= w_mve.sum()
mve = X.values @ w_mve
print("Bobot MVE adaptif:", dict(zip(files, np.round(w_mve,4))))
out3 = M[["id"]].copy()
out3["accident_risk"] = mve
out3.to_csv("submission.csv", index=False, float_format="%.9f")

