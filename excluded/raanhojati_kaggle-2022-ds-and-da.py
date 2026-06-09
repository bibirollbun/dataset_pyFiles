# Import the data and make a copy of the text row
import re, pandas as pd
path ="/kaggle/input/kaggle-survey-2022/kaggle_survey_2022_responses.csv"
df_full = pd.read_csv(path,low_memory=False)
text_row = df_full.iloc[0]
pd.set_option('display.max_columns', None)   # show all columns
df_full.head(2)


# Drop the text of questions and reindex
df_full = df_full.iloc[1:].copy()                              # drop question-text row
df_full.index = pd.RangeIndex(start=3, stop=3 + len(df_full))  # index labels = CSV rows
df_full.head(2)


vc=df_full["Q23"].value_counts()
print(vc)
psDS = round(100*vc.iloc[0]/vc.sum(),1)
psDA = round(100*vc.iloc[1]/vc.sum(),1)
print("percent of DS =", psDS)
print("percent of DA =", psDA)


### Reducing the rows to DS and DA roles
ROLE = "Q23"
DA = "Data Analyst (Business, Marketing, Financial, Quantitative, etc)"
DS = "Data Scientist"
df_dsda = df_full[df_full[ROLE].isin([DS, DA])].copy()
print(df_dsda.shape)
print(df_dsda[ROLE].value_counts(dropna=False))
df_dsda.head(2)


# build schema maps from the index and text row
hdr = text_row.to_dict()
# Multi-select option label map: Qxx_k -> "Python", "R", ...
schema_option_label = {
    col: re.sub(r'.*Selected Choice\s*-\s*', '', str(hdr.get(col, col))).strip()
    for col in df_full.columns
    if re.match(r'^Q\d+_', col)
}
# Example of schema for Q12 children
print("Q12 option labels:", {k:v for k,v in schema_option_label.items() if k.startswith("Q12_")})


# DS vs DA comparison of tools used
import numpy as np, pandas as pd, re

THRESH = 0.50
INCLUDE = {"Q12","Q13","Q14","Q15","Q17","Q18","Q28","Q31","Q33","Q34","Q35","Q36"}
role_col = "Q23"

# 0) create a dictionary and set for "None" children
NONE_IDX = {
    "Q12":14, "Q13":13, "Q14":15, "Q15":14, "Q17":14, "Q18":13,
    "Q28":7, "Q31":11, "Q33":4, "Q34":7, "Q35":15, "Q36":14
}
none_children = {f"{q}_{i}" for q, i in NONE_IDX.items()}

# 1) Create a respondent id row index
df = df_dsda.reset_index(drop=True).reset_index(names="rid")
df["rid"] += 1  # start at 1 instead of 0 (optional)
cols = [c for c in df.columns if "_" in str(c) and str(c).split("_",1)[0] in INCLUDE]

# 2) Create a Long form + role flags
m = df[["rid", role_col] + cols].melt(id_vars=["rid", role_col], var_name="child", value_name="val")
m["present"] = m["val"].notna() & ~m["child"].isin(none_children)
m["parent"] = m["child"].astype(str).str.partition("_")[0]
lower_role = m[role_col].astype(str).str.lower()
m["grp"] = np.where(lower_role.str.contains("data scientist"), "DS",
            np.where(lower_role.str.contains("data analyst"), "DA", np.nan))
m["present"] = m["val"].notna()

# 3) Compute Child-level % and parent-level "Overall" %
pct = (m.groupby(["grp","parent","child"])["present"].mean()*100).round(1).reset_index()
overall = (m.groupby(["grp","parent","rid"])["present"].any()
             .groupby(["grp","parent"]).mean()*100).round(1).reset_index(name="overall")

# 4) Create Child labels
def child_label(c):
    t = str(hdr.get(c, ""))
    return t.split(" - Selected Choice - ",1)[1] if " - Selected Choice - " in t else c
pct["label"] = pct["child"].map(child_label)

# 5) Build compact DS/DA strings per parent (with Overall fallback)
parents = sorted(set(pct["parent"]).union(set(overall["parent"])),
                 key=lambda q: int(q[1:])) 
rows = []
for p in parents:
    row = {"parent": p}
    for g in ["DS","DA"]:
        kids = pct[(pct["grp"]==g) & (pct["parent"]==p) & (pct["present"]>=THRESH*100)] \
                 .sort_values("present", ascending=False)
        if kids.empty:
            ov = overall[(overall["grp"]==g) & (overall["parent"]==p)]
            ov_pct = float(ov["overall"].iloc[0]) if not ov.empty else None
            row[g] = f"Overall ({ov_pct:.1f}%)" if (ov_pct is not None and ov_pct >= THRESH*100) else ""
        else:
            row[g] = ", ".join(f"{lab} ({pr:.1f}%)" for lab, pr in zip(kids["label"], kids["present"]))
    rows.append(row)

#6 ) Create and display a side-by_side table
compare = pd.DataFrame(rows, columns=["parent","DS","DA"])
compare = compare[
    (compare["DS"].str.strip() != "") |
    (compare["DA"].str.strip() != "")
]
pd.set_option("display.max_colwidth", None)  # avoid "..." truncation
display(
    compare.style
      .set_table_styles([{"selector":"table","props":[("table-layout","fixed"),("width","100%")]}])
      .set_properties(subset=pd.IndexSlice[:, ["DS"]],
                      **{"width":"45%", "max-width":"45%", "white-space":"normal",
                         "word-wrap":"break-word", "word-break":"break-word"})
)




