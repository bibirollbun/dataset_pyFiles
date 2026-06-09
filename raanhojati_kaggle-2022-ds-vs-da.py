# Import the data and build a schema map from the first two rows of the CSV (row 0 = QIDs, row 1 = text)

import re, pandas as pd

path ="/kaggle/input/kaggle-survey-2022/kaggle_survey_2022_responses.csv"
df_full = pd.read_csv(path,low_memory=False)
text_row = df_full.iloc[0]
hdr = text_row.to_dict()

# Multi-select option label map: Qxx_k -> "Python", "R", ...
schema_option_label = {
    col: re.sub(r'.*Selected Choice\s*-\s*', '', str(hdr.get(col, col))).strip()
    for col in df_full.columns
    if re.match(r'^Q\d+_', col)
}

print("Q12 option labels:", {k:v for k,v in schema_option_label.items() if k.startswith("Q12_")})


# Drop the text of questions and reindex

df_full = df_full.iloc[1:].copy()                              # drop question-text row
df_full.index = pd.RangeIndex(start=3, stop=3 + len(df_full))  # index labels = CSV rows
pd.set_option('display.max_columns', None)   # show all columns
df_full.head(2)


# Count the respondent roles

vc=df_full["Q23"].value_counts()
print(vc)
psDS = round(100*vc.iloc[0]/vc.sum(),1)
psDA = round(100*vc.iloc[1]/vc.sum(),1)
print("  ")
print("percent of DS =", psDS)
print("percent of DA =", psDA)


# Reduce the rows of dataframe to DS and DA roles

ROLE = "Q23"
DA = "Data Analyst (Business, Marketing, Financial, Quantitative, etc)"
DS = "Data Scientist"
df_dsda = df_full[df_full[ROLE].isin([DS, DA])].copy()
df_dsda["target"] = (df_dsda[ROLE] == DS).astype(int)   # DS=1, DA=0
print(df_dsda.shape)
print(df_dsda[ROLE].value_counts(dropna=False))
df_dsda.head(2)


# Create binary variables

# --- Inputs you already have ---
# df_dsda: your survey DataFrame for DS and DA only
# schema_option_label: dict {child_code -> human label}, e.g., {"Q12_1":"Python", "Q12_2":"R", ...}

import pandas as pd

INCLUDED_PARENTS = {"Q12","Q13","Q14","Q15","Q17","Q18","Q31","Q33","Q34","Q35","Q36"}  

# 1) Build parent -> children map (exclude "None")
parent_map = {}
for code, label in schema_option_label.items():
    p = code.split("_")[0]
    if p in INCLUDED_PARENTS:
        if label == "None": 
            continue
        parent_map.setdefault(p, []).append(code)

# 2) child columns in the same order they appear in the schema
child_cols = []
for p in parent_map.keys():   # Q12, Q13, ...
    for c in parent_map[p]:
        child_cols.append(c)
X_children = df_dsda[child_cols].notna().astype("uint8")
print(X_children.head(2))

# 3) Target y 
y = df_dsda["Q23"].map({"Data Scientist":1, "Data Analyst (Business, Marketing, Financial, Quantitative, etc)":0}).dropna()
# Align matrices to y if you create it here:
X_children = X_children.loc[y.index]


# Filter to prevalent variables

# === Inputs expected ===
# X_children : DataFrame of child dummies ONLY (Q12, Q13, Q14, Q15, Q17, Q18, Q31, Q33, Q34, Q35, Q36)
# y          : Series (0=DA, 1=DS)
# parent_map : dict {"Q12":[child_cols...], "Q13":[...], ...}  (exclude "None" children)

# Global 10% filter + per-class support floor (prevents separation) ----
GLOBAL_MIN_PREV = 0.10   # 10% of ALL DS+DA rows
NMIN_PER_CLASS  = 50     
DS = (y==1); DA = (y==0)
prev = X_children.mean()
keep = prev[prev >= GLOBAL_MIN_PREV].index.tolist()
def has_support(col):
    xDS = ((X_children[col]==1) & DS).sum()
    xDA = ((X_children[col]==1) & DA).sum()
    return (xDS >= NMIN_PER_CLASS) and (xDA >= NMIN_PER_CLASS)
keep = [c for c in keep if has_support(c)]
Xf = X_children[keep].copy()
print(Xf.head(2))


# Show high correlations

import pandas as pd, numpy as np, matplotlib.pyplot as plt

R = Xf.corr(method="pearson")
labels = R.columns
fig, ax = plt.subplots(figsize=(11,11))
im = ax.imshow(R.values, vmin=-1, vmax=1, aspect="auto")
fig.colorbar(im, ax=ax)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45,  ha="right", fontsize=8)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
ax.set_title("Correlation")
plt.tight_layout()


# --- Apply backward stepwise elimination (remove worst p one at a time) ---

import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

alpha = 0.05  # target p-value threshold

# start from all Xf columns
start_cols = list(Xf.columns)
cols = start_cols.copy()

while True:
    fit = sm.Logit(y, sm.add_constant(Xf[cols])).fit(disp=0)
    pvals = fit.pvalues.drop("const")
    worst_feat = pvals.idxmax()
    worst_p = float(pvals.max())
    if worst_p <= alpha:
        break
    cols.remove(worst_feat)      # drop only the single worst feature and loop

# Final refit and summary table
fit_final = sm.Logit(y, sm.add_constant(Xf[cols])).fit(disp=0)
final_cols_sm = cols
final_tab = (
    pd.DataFrame({
        "feature": fit_final.params.drop("const").index,
        "coef":    fit_final.params.drop("const").values,
        "se":      fit_final.bse.drop("const").values,
        "p":       fit_final.pvalues.drop("const").values,
    })
)
print(final_tab)

# CV metrics
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=10000, n_jobs=-1, class_weight="balanced")
auc = cross_val_score(clf, Xf[final_cols_sm], y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
ll  = -cross_val_score(clf, Xf[final_cols_sm], y, cv=cv, scoring="neg_log_loss", n_jobs=-1).mean()
print(" ")
print(f"AUC (5-fold): {auc:.3f}    LogLoss (5-fold): {ll:.3f}")


# Add the DS and DA rates

import pandas as pd

final_tab["label"] = final_tab["feature"].map(lambda k: schema_option_label.get(k, k))

# Compute counts/rates on selected children
Xsel=Xf[final_cols_sm]
ds_mask, da_mask = (y == 1), (y == 0)

rates = pd.DataFrame({
    "feature": Xsel.columns,
    "DS_n":    Xsel.loc[ds_mask].sum().astype(int).values,
    "DA_n":    Xsel.loc[da_mask].sum().astype(int).values,
    "DS_rate": Xsel.loc[ds_mask].mean().values,
    "DA_rate": Xsel.loc[da_mask].mean().values,
})

# Merge and clean
final_tab = final_tab.merge(rates, on="feature", how="left")


# ===== report (with human labels) =====
import pandas as pd
from IPython.display import display

# Columns to display 
cols = ["feature", "label", "coef", "se", "p", "DS_n", "DA_n", "DS_rate", "DA_rate"]

print("\nFinal LR table:")
fmt = {
    "DS_rate": "{:.2f}",
    "DA_rate": "{:.2f}",
    "coef": "{:.3f}", "se": "{:.5f}", "p": "{:.2e}",
}
display(final_tab[cols].style.format(fmt))




