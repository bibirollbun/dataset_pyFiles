import numpy as np
import pandas as pd

# Synthetic dataset for HEART-Rx

def generate_users(n_users=5000, weeks=4, seed=42):
    np.random.seed(seed)

    users = []
    for user_id in range(n_users):
        age = np.random.randint(18, 36)
        sex = np.random.choice(["M", "F"])
        bmi = np.random.normal(24, 4)
        family_history = np.random.choice([0, 1], p=[0.8, 0.2])
        smoker = np.random.choice([0, 1], p=[0.85, 0.15])

        baseline_risk = (
            0.02 * (age - 18)
            + 0.15 * family_history
            + 0.10 * smoker
            + 0.03 * (bmi - 22)
        )

        for w in range(weeks):
            sleep_hours = np.clip(np.random.normal(6.5, 1.2), 3, 10)
            sleep_var = abs(np.random.normal(1.0, 0.5))
            steps = max(0, np.random.normal(7000, 2500))
            night_activity = np.random.uniform(0, 0.6)
            stress = np.clip(np.random.normal(4.5, 2), 0, 10)
            sentiment = np.random.uniform(-1, 1)

            chest_pain = np.random.choice([0, 1], p=[0.93, 0.07])
            dizziness = np.random.choice([0, 1], p=[0.9, 0.1])
            tachy = np.random.choice([0, 1], p=[0.92, 0.08])

            risk_score = (
                baseline_risk
                + 0.04 * (7 - sleep_hours)
                + 0.03 * sleep_var
                + 0.00005 * (8000 - steps)
                + 0.05 * stress
                + 0.2 * chest_pain
                + 0.15 * tachy
            )

            risk_prob = 1 / (1 + np.exp(-risk_score))
            label = np.random.binomial(1, min(max(risk_prob, 0.01), 0.6))

            users.append([
                user_id,
                w,
                age,
                sex,
                bmi,
                family_history,
                smoker,
                sleep_hours,
                sleep_var,
                steps,
                night_activity,
                stress,
                sentiment,
                chest_pain,
                dizziness,
                tachy,
                label,
            ])

    cols = [
        "user_id","week","age","sex","bmi","family_history","smoker",
        "sleep_hours","sleep_var","steps","night_activity","stress","sentiment",
        "chest_pain","dizziness","tachy","label"
    ]

    return pd.DataFrame(users, columns=cols)

if __name__ == "__main__":
    df = generate_users(n_users=2000, weeks=4)
    df.to_csv("synthetic_heart_risk.csv", index=False)
    print("Synthetic dataset saved as synthetic_heart_risk.csv with shape:", df.shape)


pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import google.generativeai as genai
import json
import time
import pandas as pd
from tqdm import tqdm



from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY  


import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])



genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("models/gemini-2.5-flash-preview-05-20")
print("ðŸš€ Gemini model ready.")



import json, re

def robust_json_parse(text):
    text = text.strip()
    try: 
        return json.loads(text)
    except:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass

    return {}



def batch_lifestyle_agent(batch_df):
    rows = batch_df.to_dict(orient="records")
    n = len(rows)

    prompt = f"""
You are a lifestyle risk assessment agent.

Process ALL rows exactly in order.

Return ONLY a JSON list with EXACTLY {n} items.
Each item corresponds directly to the same index as the input rows.

Each item must have:
  - lifestyle_risk (0-1)
  - reasoning (short text)

Rows:
{json.dumps(rows)}
"""

    response = model.generate_content(prompt)
    out = robust_json_parse(response.text)
    return align_outputs(out, n)



def batch_symptom_agent(batch_df):
    rows = batch_df.to_dict(orient="records")
    n = len(rows)

    prompt = f"""
You are a symptom triage agent.

Return ONLY a JSON list with EXACTLY {n} items.

Each output item corresponds to the same index input row.

Each item must contain:
  - symptom_risk (0-1)
  - reasoning

Rows:
{json.dumps(rows)}
"""

    response = model.generate_content(prompt)
    out = robust_json_parse(response.text)
    return align_outputs(out, n)



def batch_orchestrator(lifestyle_list, symptom_list):
    prompt = f"""
You are the orchestration agent.
Combine lifestyle + symptom risks for ALL rows.

Return ONLY JSON list:
[
  {{"final_risk": <0-1>, "explanation": "..."}},
  ...
]

Lifestyle:
{json.dumps(lifestyle_list)}

Symptoms:
{json.dumps(symptom_list)}
"""

    response = model.generate_content(prompt)
    return robust_json_parse(response.text)



def align_outputs(output_list, n):
    if not isinstance(output_list, list):
        output_list = []
    while len(output_list) < n:
        output_list.append({"risk": 0, "reasoning": "missing"})
    if len(output_list) > n:
        output_list = output_list[:n]
    return output_list



for m in genai.list_models():
    print(m.name)



import os, json, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

RNG = 42
np.random.seed(RNG)

def generate_competition_dataset(n_users=2500, weeks=4, seed=RNG):
    np.random.seed(seed)
    rows = []
    for user_id in range(n_users):
        age = np.random.randint(18, 80)
        sex = np.random.choice([0,1])
        bmi = np.clip(np.random.normal(25, 4), 15, 45)
        family_history = np.random.choice([0,1], p=[0.75,0.25])
        smoker = np.random.choice([0,1], p=[0.8,0.2])
        chronic_hypertension = np.random.choice([0,1], p=[0.9,0.1])

        baseline = -4.0
        baseline += 0.04*(age-30)
        baseline += 0.06*(bmi-22)
        baseline += 0.8*family_history
        baseline += 0.9*smoker
        baseline += 1.0*chronic_hypertension

        for w in range(weeks):
            sleep_hours = np.clip(np.random.normal(7.0, 1.0), 3, 10)
            sleep_var = abs(np.random.normal(0.8, 0.6))
            steps = max(0, np.random.normal(7500, 2200))
            night_activity = np.random.beta(1.5, 8)
            stress = np.clip(np.random.normal(3.5, 2.0), 0, 10)
            sentiment = np.clip(np.random.normal(0.1, 0.6), -1, 1)
            hr_rest = np.clip(np.random.normal(70 - 0.1*(age-30) + 3*smoker, 8), 40, 120)
            hr_variability = np.clip(np.random.normal(50 - 0.2*(age-30) - 5*smoker, 10), 5, 100)
            chest_pain = np.random.choice([0,1], p=[0.96,0.04])
            dizziness = np.random.choice([0,1], p=[0.95,0.05])
            tachy = np.random.choice([0,1], p=[0.95,0.05])

            logit = baseline
            logit += 0.35*(7 - sleep_hours)
            logit += 0.25*(sleep_var)
            logit += 0.00012*(9000 - steps)
            logit += 0.3*stress/2.0
            logit += 0.6*(chest_pain)
            logit += 0.5*(tachy)
            logit += 0.4*(dizziness)
            logit += 0.02*(hr_rest - 60)/10.0
            logit += -0.25*(hr_variability/50.0)

            prob = 1/(1+np.exp(-logit))
            label = np.random.binomial(1, np.clip(prob*(1-0.08) + 0.04, 0.01, 0.99))

            rows.append([
                user_id, w, age, sex, bmi, family_history, smoker, chronic_hypertension,
                sleep_hours, sleep_var, steps, night_activity, stress, sentiment,
                hr_rest, hr_variability, chest_pain, dizziness, tachy, prob, label
            ])

    cols = [
        "user_id","week","age","sex","bmi","family_history","smoker","chronic_ht",
        "sleep_hours","sleep_var","steps","night_activity","stress","sentiment",
        "hr_rest","hr_variability","chest_pain","dizziness","tachy","true_prob","label"
    ]
    return pd.DataFrame(rows, columns=cols)

def fe(df):
    d = df.copy()
    d["steps_log"] = np.log1p(d["steps"])
    d["sleep_deficit"] = np.clip(7 - d["sleep_hours"], 0, 7)
    d["stress_norm"] = d["stress"]/10.0
    d["sentiment_scaled"] = (d["sentiment"]+1)/2.0
    d["symptom_sum"] = d[["chest_pain","dizziness","tachy"]].sum(axis=1)
    d["age_buckets"] = pd.cut(d["age"], bins=[17,30,45,60,100], labels=[0,1,2,3]).astype(int)
    d["bmi_cat"] = pd.cut(d["bmi"], bins=[14,18.5,25,30,100], labels=[0,1,2,3]).astype(int)
    d["smoker_and_family"] = d["smoker"] * d["family_history"]
    return d

if os.path.exists("synthetic_heart_risk_competition.csv"):
    df = pd.read_csv("synthetic_heart_risk_competition.csv")
    df_fe = fe(df)
else:
    df = generate_competition_dataset()
    df.to_csv("synthetic_heart_risk_competition.csv", index=False)
    df_fe = fe(df)



import os, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

def generate_competition_dataset(n_users=2500, weeks=4, seed=RNG):
    np.random.seed(seed)
    rows = []
    for user_id in range(n_users):
        age = np.random.randint(18, 80)
        sex = np.random.choice([0,1])
        bmi = np.clip(np.random.normal(25, 4), 15, 45)
        family_history = np.random.choice([0,1], p=[0.75,0.25])
        smoker = np.random.choice([0,1], p=[0.8,0.2])
        chronic_hypertension = np.random.choice([0,1], p=[0.9,0.1])
        baseline = -4.0
        baseline += 0.04*(age-30)
        baseline += 0.06*(bmi-22)
        baseline += 0.8*family_history
        baseline += 0.9*smoker
        baseline += 1.0*chronic_hypertension
        for w in range(weeks):
            sleep_hours = np.clip(np.random.normal(7.0, 1.0), 3, 10)
            sleep_var = abs(np.random.normal(0.8, 0.6))
            steps = max(0, np.random.normal(7500, 2200))
            night_activity = np.random.beta(1.5, 8)
            stress = np.clip(np.random.normal(3.5, 2.0), 0, 10)
            sentiment = np.clip(np.random.normal(0.1, 0.6), -1, 1)
            hr_rest = np.clip(np.random.normal(70 - 0.1*(age-30) + 3*smoker, 8), 40, 120)
            hr_variability = np.clip(np.random.normal(50 - 0.2*(age-30) - 5*smoker, 10), 5, 100)
            chest_pain = np.random.choice([0,1], p=[0.96,0.04])
            dizziness = np.random.choice([0,1], p=[0.95,0.05])
            tachy = np.random.choice([0,1], p=[0.95,0.05])
            logit = baseline
            logit += 0.35*(7 - sleep_hours)
            logit += 0.25*(sleep_var)
            logit += 0.00012*(9000 - steps)
            logit += 0.3*stress/2.0
            logit += 0.6*(chest_pain)
            logit += 0.5*(tachy)
            logit += 0.4*(dizziness)
            logit += 0.02*(hr_rest - 60)/10.0
            logit += -0.25*(hr_variability/50.0)
            prob = 1/(1+np.exp(-logit))
            label = np.random.binomial(1, np.clip(prob*(1-0.08) + 0.04, 0.01, 0.99))
            rows.append([
                user_id, w, age, sex, bmi, family_history, smoker, chronic_hypertension,
                sleep_hours, sleep_var, steps, night_activity, stress, sentiment,
                hr_rest, hr_variability, chest_pain, dizziness, tachy, label
            ])
    cols = [
        "user_id","week","age","sex","bmi","family_history","smoker","chronic_ht",
        "sleep_hours","sleep_var","steps","night_activity","stress","sentiment",
        "hr_rest","hr_variability","chest_pain","dizziness","tachy","label"
    ]
    return pd.DataFrame(rows, columns=cols)

def fe(df):
    d = df.copy()
    d["steps_log"] = np.log1p(d["steps"])
    d["sleep_deficit"] = np.clip(7 - d["sleep_hours"], 0, 7)
    d["stress_norm"] = d["stress"]/10.0
    d["sentiment_scaled"] = (d["sentiment"]+1)/2.0
    d["symptom_sum"] = d[["chest_pain","dizziness","tachy"]].sum(axis=1)
    d["age_buckets"] = pd.cut(d["age"], bins=[17,30,45,60,100], labels=[0,1,2,3]).astype(int)
    d["bmi_cat"] = pd.cut(d["bmi"], bins=[14,18.5,25,30,100], labels=[0,1,2,3]).astype(int)
    d["smoker_and_family"] = d["smoker"] * d["family_history"]
    return d

if os.path.exists("synthetic_heart_risk_competition.csv"):
    df = pd.read_csv("synthetic_heart_risk_competition.csv")
else:
    df = generate_competition_dataset()
    df.to_csv("synthetic_heart_risk_competition.csv", index=False)

df_fe = fe(df)
df_fe = df_fe.sort_values(["user_id","week"]).reset_index(drop=True)

def safe_prev_stats(series):
    arr = np.asarray(series)
    if len(arr) == 0:
        return (np.nan, np.nan, 0.0)
    return (np.nan, np.nan, 0.0)

def compute_prev_features(df, cols):
    out = df.copy()
    grp = out.groupby("user_id")
    for c in cols:
        prev_mean = grp[c].expanding().mean().shift(1).reset_index(level=0, drop=True)
        prev_std = grp[c].expanding().std().shift(1).reset_index(level=0, drop=True)
        out["prev_" + c + "_mean"] = prev_mean
        out["prev_" + c + "_std"] = prev_std.fillna(0.0)
        def prev_slope(s):
            vals = s.values
            if len(vals) < 2:
                return np.nan
            try:
                coef = np.polyfit(np.arange(len(vals)), vals, 1)[0]
                return coef
            except:
                return np.nan
        prev_trend = grp[c].apply(lambda s: pd.Series(s).expanding().apply(lambda x: (np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x)>1 else np.nan).shift(1)).reset_index(level=0, drop=True)
        out["prev_" + c + "_trend"] = prev_trend
    return out

cols_to_prev = ["sleep_hours","steps","stress","hr_rest","hr_variability"]
df_fe = compute_prev_features(df_fe, cols_to_prev)
global_fill = {}
for c in df_fe.columns:
    if c.startswith("prev_"):
        global_fill[c] = df_fe[c].median()
df_fe = df_fe.fillna(global_fill)

users = df_fe["user_id"].unique()
train_u, test_u = train_test_split(users, test_size=0.2, random_state=RNG)
train = df_fe[df_fe["user_id"].isin(train_u)].reset_index(drop=True)
test  = df_fe[df_fe["user_id"].isin(test_u)].reset_index(drop=True)

lifestyle_feats = [
    "sleep_hours","sleep_var","steps","night_activity","stress","sentiment",
    "steps_log","sleep_deficit","stress_norm","sentiment_scaled",
    "prev_sleep_hours_mean","prev_sleep_hours_std","prev_sleep_hours_trend",
    "prev_steps_mean","prev_steps_std","prev_steps_trend",
    "prev_stress_mean","prev_stress_std","prev_stress_trend"
]

symptom_feats = [
    "chest_pain","dizziness","tachy","symptom_sum","age","chronic_ht","family_history",
    "prev_hr_rest_mean","prev_hr_rest_std","prev_hr_rest_trend",
    "prev_hr_variability_mean","prev_hr_variability_std","prev_hr_variability_trend"
]

train = train.loc[:, ~train.columns.duplicated()]
test  = test.loc[:,  ~test.columns.duplicated()]

gkf = GroupKFold(n_splits=5)
oof_l = np.zeros(len(train))
test_pred_l = np.zeros(len(test))
for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, groups=train["user_id"])):
    X_tr = train.iloc[tr_idx][lifestyle_feats]
    y_tr = train.iloc[tr_idx]["label"]
    X_val = train.iloc[val_idx][lifestyle_feats]
    m = LGBMClassifier(n_estimators=800, learning_rate=0.04, num_leaves=31, random_state=RNG+fold)
    m.fit(X_tr, y_tr)
    oof_l[val_idx] = m.predict_proba(X_val)[:,1]
    test_pred_l += m.predict_proba(test[lifestyle_feats])[:,1] / gkf.n_splits
    joblib.dump(m, f"lifestyle_fold_{fold}.joblib")
lgb_l = LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=31, random_state=RNG)
lgb_l.fit(train[lifestyle_feats], train["label"])
test_pred_full_l = lgb_l.predict_proba(test[lifestyle_feats])[:,1]

oof_s = np.zeros(len(train))
test_pred_s = np.zeros(len(test))
for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, groups=train["user_id"])):
    X_tr = train.iloc[tr_idx][symptom_feats]
    y_tr = train.iloc[tr_idx]["label"]
    X_val = train.iloc[val_idx][symptom_feats]
    m = LGBMClassifier(n_estimators=700, learning_rate=0.05, num_leaves=24, random_state=RNG+fold)
    m.fit(X_tr, y_tr)
    oof_s[val_idx] = m.predict_proba(X_val)[:,1]
    test_pred_s += m.predict_proba(test[symptom_feats])[:,1] / gkf.n_splits
    joblib.dump(m, f"symptom_fold_{fold}.joblib")
lgb_s = LGBMClassifier(n_estimators=900, learning_rate=0.04, num_leaves=28, random_state=RNG)
lgb_s.fit(train[symptom_feats], train["label"])
test_pred_full_s = lgb_s.predict_proba(test[symptom_feats])[:,1]

train_meta = pd.DataFrame({
    "oof_l": oof_l,
    "oof_s": oof_s,
    "age": train["age"],
    "bmi": train["bmi"],
    "smoker": train["smoker"],
    "family_history": train["family_history"]
})
test_meta = pd.DataFrame({
    "oof_l": test_pred_l,
    "oof_s": test_pred_s,
    "age": test["age"],
    "bmi": test["bmi"],
    "smoker": test["smoker"],
    "family_history": test["family_history"]
})

meta_feats = ["oof_l","oof_s","age","bmi","smoker","family_history"]
meta = LogisticRegression(max_iter=1000)
meta.fit(train_meta[meta_feats], train["label"])
meta_test_pred = meta.predict_proba(test_meta[meta_feats])[:,1]

fusion_oof_auc = roc_auc_score(train["label"], train_meta["oof_l"]*0.5 + train_meta["oof_s"]*0.5)
fusion_meta_auc = roc_auc_score(test["label"], meta_test_pred)

fusion_full = LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=40, subsample=0.9, colsample_bytree=0.9, random_state=RNG)
fusion_full.fit(train_meta[meta_feats], train["label"])

full = df_fe.copy()
full = full.sort_values(["user_id","week"]).reset_index(drop=True)
full = compute_prev_features(full, cols_to_prev)
for c in full.columns:
    if c.startswith("prev_") and full[c].isnull().any():
        full[c] = full[c].fillna(full[c].median())

full_life = lgb_l.predict_proba(full[lifestyle_feats])[:,1]
full_symp = lgb_s.predict_proba(full[symptom_feats])[:,1]
full_meta_df = pd.DataFrame({
    "oof_l": full_life,
    "oof_s": full_symp,
    "age": full["age"],
    "bmi": full["bmi"],
    "smoker": full["smoker"],
    "family_history": full["family_history"]
})
full["fusion_base_score"] = fusion_full.predict_proba(full_meta_df[meta_feats])[:,1]
full["final_risk"] = meta.predict_proba(full_meta_df[meta_feats])[:,1]

try:
    import shap
    expl = shap.TreeExplainer(fusion_full)
    sample = full.sample(n=min(500,len(full)), random_state=RNG)
    def shap_explain_row(i):
        r = full.iloc[i:i+1]
        vals = np.abs(expl.shap_values(r[meta_feats])[0][0])
        top_idx = np.argsort(-vals)[:3]
        return ", ".join([meta_feats[k] + f"({round(float(r.iloc[0][meta_feats[k]]),3)})" for k in top_idx])
    full["explanation"] = [shap_explain_row(i) for i in range(len(full))]
except:
    def simple_explain(r):
        a=[]
        a.append("Lifestyle inc" if r["oof_l"]>0.6 else "Lifestyle ok")
        a.append("Symptoms noted" if r["oof_s"]>0.4 else "Symptoms minor")
        return "; ".join(a)
    full["explanation"] = full.apply(lambda r: simple_explain({"oof_l": r["oof_l"] if "oof_l" in r else full_life[r.name], "oof_s": r["oof_s"] if "oof_s" in r else full_symp[r.name]}), axis=1)

joblib.dump(lgb_l, "lifestyle_model_final.joblib")
joblib.dump(lgb_s, "symptom_model_final.joblib")
joblib.dump(fusion_full, "fusion_model_final.joblib")
joblib.dump(meta, "fusion_meta_logit.joblib")

full[["user_id","week","final_risk","explanation","label"]].to_csv("multiagent_output_comp_final.csv", index=False)

print("Base lifestyle OOF AUC:", round(float(roc_auc_score(train["label"], oof_l)),4))
print("Base symptom OOF AUC:", round(float(roc_auc_score(train["label"], oof_s)),4))
print("Fusion (meta) AUC on test:", round(float(fusion_meta_auc),4))
print("Saved: multiagent_output_comp_final.csv")



user_preds = full.groupby("user_id")["final_risk"].mean()
user_labels = full.groupby("user_id")["label"].max()
from sklearn.metrics import roc_auc_score
print("user-level AUC:", roc_auc_score(user_labels, user_preds))



from sklearn.calibration import CalibratedClassifierCV
cal = CalibratedClassifierCV(fusion_full, cv=3, method="isotonic")
cal.fit(train_meta[meta_feats], train["label"])
cal_test = cal.predict_proba(test_meta[meta_feats])[:,1]
print("Calibrated test AUC:", roc_auc_score(test["label"], cal_test))


