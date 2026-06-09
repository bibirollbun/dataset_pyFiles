!pip install lightgbm codecarbon



import pandas as pd
import os

# Competition dataset klasÃ¶rÃ¼nÃ¼ listele
os.listdir("/kaggle/input")



import pandas as pd

train_path = "/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/train.csv"
meta_path = "/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/metaData.csv"

train = pd.read_csv(train_path)
meta = pd.read_csv(meta_path)

train.head(), train.shape, train.columns



from codecarbon import EmissionsTracker
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

# Load dataset
df = pd.read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/train.csv")

X = df[['feature_1', 'feature_2']]
y = df['target']

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# === BASELINE PIPELINE ===
baseline_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=5, include_bias=False)),   # âœ… daha bÃ¼yÃ¼k pipeline
    ("scaler", StandardScaler()),                                # âœ… normalize
    ("lr", LogisticRegression(max_iter=2000))                    # âœ… gerÃ§ek eÄŸitim sÃ¼resi
])

# === ENERGY TRACKER ===
tracker = EmissionsTracker()
tracker.start()

# TRAIN BASELINE
baseline_pipeline.fit(X_train, y_train)

# STOP ENERGY
baseline_emissions = tracker.stop()

# Predictions
preds = baseline_pipeline.predict(X_val)
baseline_f1 = f1_score(y_val, preds, average="macro")

# === OUTPUT ===
print("=== BASELINE POLY-LOGISTIC ENERGY REPORT ===")
print("CO2 Emissions (kg):", baseline_emissions)
print("Baseline F1:", baseline_f1)



import pandas as pd
import numpy as np

np.random.seed(42)

N = 1000

df = pd.DataFrame({
    "hour": np.random.randint(0, 24, N),
    "temperature": np.random.uniform(-5, 35, N),
    "demand_mw": np.random.uniform(20_000, 45_000, N),
    "wind_percent": np.random.uniform(0, 1, N),
    "solar_percent": np.random.uniform(0, 1, N),
    "weekday": np.random.randint(0, 7, N)
})

# Karbon yoÄŸunluÄŸu kural tabanlÄ± simÃ¼lasyon (bilimsel olarak mantÄ±klÄ±)
ci = (
    0.4 * (df["demand_mw"] / df["demand_mw"].max()) -
    0.3 * df["wind_percent"] -
    0.2 * df["solar_percent"] +
    0.1 * (df["hour"].isin([18,19,20]).astype(int)) +
    np.random.normal(0, 0.05, N)
)

df["carbon_intensity"] = ci

# sÄ±nÄ±f etiketlerine Ã§evir
df["ci_class"] = pd.cut(
    df["carbon_intensity"],
    bins=[-999, 0.25, 0.55, 999],
    labels=["low", "medium", "high"]
)

df.head()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

features = ["hour", "temperature", "demand_mw", "wind_percent", "solar_percent", "weekday"]
X = df[features]
y = df["ci_class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

model = LogisticRegression(max_iter=300)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)
print(classification_report(y_test, preds))



def carbon_scheduler(predicted_class):
    if predicted_class == "high":
        return "Task postponed (high carbon period)"
    elif predicted_class == "medium":
        return "Run only if necessary"
    else:
        return "Task executed now (low carbon period)"

# Ã–rnek tahmin
sample = X_test.iloc[0:1]
sample_scaled = sc.transform(sample)
pred_class = model.predict(sample_scaled)[0]

print("Predicted carbon:", pred_class)
print("Scheduler decision:", carbon_scheduler(pred_class))



HIGH_CARBON = 500  # g COâ‚‚ / kWh
LOW_CARBON = 200   # g COâ‚‚ / kWh
TASK_KWH = 0.03     # ortalama CPU gÃ¶revi

saving_per_task_g = TASK_KWH * (HIGH_CARBON - LOW_CARBON)
tasks_per_day = 10

annual_saving_per_user_kg = (saving_per_task_g * tasks_per_day * 365) / 1000

print("Annual CO2 saving per user (kg):", annual_saving_per_user_kg)

# 100k kullanÄ±cÄ± iÃ§in:
print("Annual national impact (tons):", annual_saving_per_user_kg * 100_000 / 1000)



import pandas as pd
test = pd.read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/test.csv")
test.head()



import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load train/test
train = pd.read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/train.csv")
test = pd.read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/test.csv")

X = train[['feature_1', 'feature_2']]
y = train['target']

# Scale
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Model
model = LogisticRegression()
model.fit(X_scaled, y)

# Test predictions
# (testte feature yok, simÃ¼le edilmemiÅŸ â†’ tÃ¼m test target'larÄ±nÄ± en olasÄ± sÄ±nÄ±fa gÃ¶re belirle)
y_pred_all = [y.mode()[0]] * len(test)  # output = en sÄ±k gÃ¶rÃ¼len sÄ±nÄ±f

# Submission
submission = pd.DataFrame({
    "example_id": test["example_id"],
    "target": y_pred_all
})

submission.to_csv("submission.csv", index=False)

submission.head()



import pandas as pd

# Var olan submission'Ä± yeniden oluÅŸtur (ID adÄ± 'Id' olacak)
test = pd.read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/test.csv")

# Biz daha Ã¶nce Ã§oÄŸunluk sÄ±nÄ±fÄ± ile doldurmuÅŸtuk:
y_pred_all = [1] * len(test)   # veya elindeki y_pred_all listesi

fixed = pd.DataFrame({
    "Id": test["example_id"],   # ðŸ”´ Ã–NEMLÄ°: 'Id' olarak yeniden adlandÄ±r
    "target": y_pred_all        # ÅŸimdilik yarÄ±ÅŸmanÄ±n beklediÄŸi label adÄ± 'target' varsayÄ±mÄ±
})

# Temizlik ve kayÄ±t
assert list(fixed.columns) == ["Id", "target"]
fixed.to_csv("submission.csv", index=False)
fixed.head()



import pandas as pd

submission = pd.read_csv("submission.csv")
# target'Ä± int'e Ã§evir (1.0 -> 1 gibi)
submission["target"] = submission["target"].astype(int)

assert list(submission.columns) == ["Id", "target"], "SÃ¼tun adlarÄ±/sÄ±rasÄ± yanlÄ±ÅŸ!"
assert submission.isna().sum().sum() == 0, "BoÅŸ deÄŸer var!"
assert submission["Id"].is_unique, "Id tekrarlÄ±!"
print("âœ… submission.csv hazÄ±r!")

submission.to_csv("submission.csv", index=False)
submission.head()


