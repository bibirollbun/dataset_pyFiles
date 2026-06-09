import pandas as pd

RESP = "/kaggle/input/kaggle-survey-2022/kaggle_survey_2022_responses.csv"

df_raw        = pd.read_csv(RESP, low_memory=False)  # includes the long question row
questions_row = df_raw.iloc[0]                       # the long question text (row 1)
df_full       = df_raw.iloc[1:].reset_index(drop=True)  # the actual responses
print(df_full.shape)


questions_row


import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    display(df_full.head(2)) 


vc = df_full["Q2"].value_counts()
n_60p   = vc["60-69"] + vc["70+"]    # count age ≥ 60
n_total = vc.sum()                                # non-null Q2 rows
pct_60p = round(100 * n_60p / n_total,1)
print("no. of 60+ =", n_60p, "    total =", n_total, "    percent =", pct_60p)


import re, numpy as np, pandas as pd

def parse_coding_bounds(val):
    # Convert Q11 text to (low, high) numeric bounds in years.
    s = str(val)
    if pd.isna(val) or s.startswith("I have never written code"):
        return (-1.0, -1.0)
    if s.startswith("< 1"):
        return (0.0, 1.0)
    if s.startswith("1-3"):
        return (1.0, 3.0)
    if s.startswith("3-5"):
        return (3.0, 5.0)
    if s.startswith("5-10"):
        return (5.0, 10.0)
    if s.startswith("10-20"):
        return (10.0, 20.0)
    if s.startswith("20+"):
        return (20.0, float('inf'))

# parsing compensation (Q29).
def has_valid_comp(x):
    if pd.isna(x): return False
    else: return True
    
def parse_comp_range(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    t = str(s).strip().replace(",", "").replace("$","").replace(">","")
    nums = t.split('-')
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    if len(nums) == 1:
        n = int(nums[0])
        return (n, n)
    return (np.nan, np.nan)


bounds = df_full["Q11"].apply(parse_coding_bounds)
df_full["coding_low"]  = bounds.str[0]
df_full["coding_high"] = bounds.str[1]

df_full["valid_comp"] = df_full["Q29"].apply(has_valid_comp)
bounds = df_full["Q29"].apply(parse_comp_range)
df_full["comp_low"] = bounds.str[0]
df_full["comp_high"] = bounds.str[1]

age_ok = (df_full["Q2"] == "60-69") | (df_full["Q2"] == "70+")

coding_ge_1   = df_full["coding_low"] >= 1
coding_leq_10 = df_full["coding_high"] <= 10

q23 = df_full["Q23"].astype(str).fillna("")

exclude_titles_patterns = [
    "Developer Advocate",
    "Manager (Program, Project, Operations, Executive-level, etc)",
    "Engineer (non-software)",
    "Statistician",
    "Teacher / professor",
    "Currently not employed",
]

df_full["exclude_title"] = df_full["Q23"].isin(exclude_titles_patterns)

df_full["Q28_handson_flag"] = df_full["Q28_7"] == "None of these activities are an important part of my role at work"

mask = (
    age_ok
    & coding_ge_1
    & coding_leq_10
    & df_full["valid_comp"]
    & ~df_full["exclude_title"]
    & ~df_full["Q28_handson_flag"]
    & (df_full["comp_high"].fillna(np.inf) < 90000)
)

cohort = df_full[mask].copy()

print("Cohort size:", cohort.shape[0])  
cohort


print("percent of 60+ considered late_career transitioner = ", round(100*35/653,1))


# Job titles (Top 10)
import matplotlib.pyplot as plt
s = cohort["Q23"].value_counts()
title_counts = s.nlargest(10).sort_values(ascending=True)

labels = title_counts.index
vals   = title_counts

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(labels)), vals, height=0.5)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Count", fontsize=9)
ax.set_title("Job titles - Top 10 (Q23)", fontsize=10, pad=6)

# Put small value labels inside when possible, otherwise just outside
def place_count_label():
    for i, (b, v) in enumerate(zip(bars, vals)):
        x = b.get_width()
        if x >= 2:
            ax.text(x - 0.2, b.get_y() + b.get_height()/2, f"{int(v)}",
                    va="center", ha="right", color="white", fontsize=8)
        else:
            ax.text(x + 0.2, b.get_y() + b.get_height()/2, f"{int(v)}",
                    va="center", ha="left", color="black", fontsize=8)
place_count_label()
plt.tight_layout(pad=0.8)
plt.show()


# Industries (Top 10) 
s = cohort["Q24"].value_counts()
industry_counts = s.nlargest(10).sort_values(ascending=True)

labels = industry_counts.index
vals   = industry_counts

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(labels)), vals, height=0.5)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Count", fontsize=9)
ax.set_title("Industries- Top 10 (Q24)", fontsize=10, pad=6)

place_count_label()

plt.tight_layout(pad=0.8)
plt.show()


# Compact horizontal bar for Coding experience (Q11)
import numpy as np
import matplotlib.pyplot as plt

order = ["< 1 years","1-3 years","3-5 years","5-10 years","10-20 years","20+ years"]
coding_counts = cohort["Q11"].value_counts().reindex(order).dropna().astype(int)

labels = coding_counts.index
vals   = coding_counts

fig, ax = plt.subplots(figsize=(8, 4))   
bars = ax.barh(range(len(labels)), vals, height=0.5)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Count", fontsize=9)
ax.set_title("Coding experience (Q11)", fontsize=10, pad=6)

place_count_label()

plt.tight_layout(pad=0.8)
plt.show()


# Compact horizontal bar: Company size (Q25) — ordered
import numpy as np
import matplotlib.pyplot as plt

order = [
"0-49 employees",
"50-249 employees",
"250-999 employees",
"1000-9,999 employees",
"10,000 or more employees"]

company_size_counts = cohort["Q25"].value_counts().reindex(order).dropna().astype(int)

labels = company_size_counts.index
vals   = company_size_counts

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(labels)), vals, height=0.5)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Count", fontsize=9)
ax.set_title("Company size (Q25)", fontsize=10, pad=6)

place_count_label()

plt.tight_layout(pad=0.8)
plt.show()


# Countries (Top 10) — ordered horizontal bar (with compact names)
from textwrap import shorten
def compact_country(x):
    if x == "United States of America": return "USA"
    if x == "United Kingdom of Great Britain and Northern Ireland": return "UK"
    return shorten(str(x), width=20, placeholder="…")

s = cohort["Q4"].map(compact_country).value_counts()
country_counts = s.nlargest(10).sort_values(ascending=True)

labels = country_counts.index
vals   = country_counts

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(labels)), vals, height=0.5)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Count", fontsize=9)
ax.set_title("Countries- Top 10 (Q4)", fontsize=10, pad=6)

place_count_label()

plt.tight_layout(pad=0.8)
plt.show()


from textwrap import shorten
# Top 10 countries
top10_countries = cohort["Q4"].value_counts().head(10).index.tolist()
df_top = cohort[cohort["Q4"].isin(top10_countries)].copy()

# Compact country labels for readability
def compact_country(x: str) -> str:
    if x == "United States of America":
        return "USA"
    if x == "United Kingdom of Great Britain and Northern Ireland":
        return "UK"
    # shorten very long names while keeping them distinguishable
    return shorten(str(x), width=18, placeholder="…")

df_top["CountryLabel"] = df_top["Q4"].map(compact_country)

# Crosstab and preserve the (compacted) top-10 order
ct = pd.crosstab(df_top["CountryLabel"], df_top["Q23"])
row_order = [compact_country(c) for c in top10_countries]
ct = ct.reindex(row_order)

# Horizontal stacked bar with larger canvas and legend outside
plt.figure(figsize=(10, 6))
ypos = np.arange(len(ct.index))
left = np.zeros(len(ct))

for col in ct.columns:
    vals = ct[col].values
    plt.barh(ypos, vals, left=left, label=col)
    left += vals

plt.title("Titles within Countries (stacked counts)")
plt.yticks(ypos, ct.index)
plt.xlabel("Count")

# Leave room on the right for the legend
plt.tight_layout(rect=[0, 0, 0.78, 1])
plt.legend(title="Title", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
plt.show()


# Compact horizontal bar: Compensation (Q29) — ordered with percents
import numpy as np
import matplotlib.pyplot as plt

bin_order = [
    "$0-999","1,000-1,999","2,000-2,999","3,000-3,999","4,000-4,999",
    "5,000-7,499","7,500-9,999",
    "10,000-14,999","15,000-19,999","20,000-24,999","25,000-29,999",
    "30,000-39,999","40,000-49,999","50,000-59,999","60,000-69,999",
    "70,000-79,999","80,000-89,999","90,000-99,999",
    "100,000-124,999","125,000-149,999","150,000-199,999",
    "200,000-249,999","250,000-299,999","300,000-499,999",
    "$500,000-999,999", ">$1,000,000",
]

comp_counts = cohort["Q29"].value_counts().reindex(bin_order).dropna().astype(int)

labels = comp_counts.index
vals   = comp_counts

share  = vals / vals.sum() * 100

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(labels)), vals, height=0.5)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Count", fontsize=9)
ax.set_title("Compensation (Q29)", fontsize=10, pad=6)

# percent labels just outside the bars
for i, (b, p) in enumerate(zip(bars, share)):
    ax.text(b.get_width() + 0.2, b.get_y() + b.get_height()/2,
            f"{p:.0f}%", va="center", ha="left", fontsize=8)

for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.xaxis.grid(False)
plt.tight_layout(pad=0.8)
plt.show()




