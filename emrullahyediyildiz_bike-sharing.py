# Daten einlesen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")



# Kaggle Dataset Pfad (funktioniert automatisch auf Kaggle)
DATA_DIR = "/kaggle/input/bike-sharing-demand"


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sampleSubmission.csv"))


# Erste Checks
print("Train-Shape:", train.shape)
print("Test-Shape :", test.shape)
print("Sample-Submission:", sample_sub.shape)


# Kopf anzeigen
train.head()


train.describe().T


train.isnull().sum()


print("NaN:", train["count"].isna().sum())
print("Inf:", np.isinf(train["count"]).sum())



train["datetime"] = pd.to_datetime(train["datetime"])
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["weekday"] = train["datetime"].dt.weekday



plt.figure(figsize=(10,5))
sns.histplot(train["count"], bins=50, kde=True, color="skyblue")
plt.title("Verteilung der Zielvariable: count")
plt.show()


sns.histplot(train["count"].dropna(), bins=50, kde=True, color="skyblue")


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # nervige FutureWarnings ausblenden

plt.figure(figsize=(15,10))

# 1ï¸�Stunden
plt.subplot(3,1,1)
sns.boxplot(x="hour", y="count", data=train, palette="Blues")
plt.title("Bike Rentals nach Stunde (Rush Hours sichtbar)", fontsize=14)

#  Wochentage
plt.subplot(3,1,2)
sns.boxplot(x="weekday", y="count", data=train, palette="Greens")
plt.title("Bike Rentals nach Wochentag (0=Montag ... 6=Sonntag)", fontsize=14)

# Jahreszeiten
plt.subplot(3,1,3)
sns.boxplot(x="season", y="count", data=train, palette="Oranges")
plt.title("Bike Rentals nach Saison (1=Winter, 2=FrÃ¼hling, 3=Sommer, 4=Herbst)", fontsize=14)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Durchschnittliche Anzahl Mieten pro Stunde & Wochentag
pivot = train.pivot_table(
    index="weekday", columns="hour", values="count", aggfunc="mean"
)

plt.figure(figsize=(14,6))
sns.heatmap(pivot, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Durchschnitt count'})
plt.title("Heatmap: Durchschnittliche Bike Rentals (Wochentag Ã— Stunde)", fontsize=14)
plt.xlabel("Stunde des Tages")
plt.ylabel("Wochentag (0=Mo ... 6=So)")
plt.show()



plt.figure(figsize=(8,5))
sns.boxplot(x="weather", y="count", data=train, palette="coolwarm")
plt.title("Bike Rentals nach Wetterlage (1=Klar ... 4=Schneesturm)", fontsize=14)
plt.xlabel("Wetterlage")
plt.ylabel("Fahrradmieten (count)")
plt.show()



fig, axes = plt.subplots(2,2, figsize=(12,8))

sns.scatterplot(x="temp", y="count", data=train, ax=axes[0,0], alpha=0.4)
axes[0,0].set_title("Temp vs Count")

sns.scatterplot(x="atemp", y="count", data=train, ax=axes[0,1], alpha=0.4, color="orange")
axes[0,1].set_title("Atemp vs Count")

sns.scatterplot(x="humidity", y="count", data=train, ax=axes[1,0], alpha=0.4, color="green")
axes[1,0].set_title("Humidity vs Count")

sns.scatterplot(x="windspeed", y="count", data=train, ax=axes[1,1], alpha=0.4, color="red")
axes[1,1].set_title("Windspeed vs Count")

plt.tight_layout()
plt.show()



for df in [train, test]:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday



train["count_log"] = np.log1p(train["count"])


drop_cols = ["datetime", "count", "casual", "registered"]
X = train.drop(columns=drop_cols + ["count_log"])  # Features
y = train["count_log"]  # Ziel (log-transformiert)



num_cols = ["temp", "atemp", "humidity", "windspeed", "hour", "day"]
cat_cols = ["season", "holiday", "workingday", "weather", "year", "month", "weekday"]



from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

num_tf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler())
])
cat_tf = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

pre = ColumnTransformer([
    ("num", num_tf, num_cols),
    ("cat", cat_tf, cat_cols)
])



from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except:
    HAS_XGB = False

models = {
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "GBR": GradientBoostingRegressor(random_state=42)
}
if HAS_XGB:
    models["XGB"] = XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )

# RMSLE-Scoring-Funktion
def rmsle_cv(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # scoring: neg_mean_squared_log_error gibt MSLE â†’ sqrt machen wir manuell
    scores = cross_val_score(
        Pipeline([("pre", pre), ("model", model)]),
        X, y,
        cv=kf,
        scoring="neg_mean_squared_log_error",
        n_jobs=-1
    )
    rmsle_scores = np.sqrt(-scores)
    return rmsle_scores



results = []
for name, model in models.items():
    scores = rmsle_cv(model, X, y)
    results.append((name, scores.mean(), scores.std()))

cv_results = pd.DataFrame(results, columns=["Modell", "RMSLE_Mean", "RMSLE_STD"])\
    .sort_values("RMSLE_Mean")
cv_results



from sklearn.pipeline import Pipeline
import numpy as np

# Bestes Modell
best_model = models["XGB"]

# Pipeline mit Preprocessing + Modell
final_pipe = Pipeline([
    ("pre", pre),
    ("model", best_model)
])

# Training auf gesamten Trainingsdaten
final_pipe.fit(X, y)

# Vorhersagen auf Testdaten
test_features = test.drop(columns=["datetime"], errors="ignore")
preds_log = final_pipe.predict(test_features)

# RÃ¼cktransformation log1p -> Originalwerte
preds = np.expm1(preds_log)


# Submission erstellen
submission = sample_sub.copy()
submission["count"] = preds
submission.to_csv("submission.csv", index=False)

print("âœ… submission.csv gespeichert â€“ bereit zum Hochladen!")
submission.head()


# Alle Feature-Namen aus der Pipeline holen
ohe = final_pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
cat_features = ohe.get_feature_names_out(cat_cols)

all_features = num_cols + list(cat_features)

print("Anzahl Features:", len(all_features))
print("Beispiele:", all_features[:15])  # zum Check



import pandas as pd
import matplotlib.pyplot as plt

# Feature Importances aus XGB
xgb_model = final_pipe.named_steps["model"]
importances = xgb_model.feature_importances_

# DataFrame bauen
imp_df = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values("importance", ascending=False).head(15)

# Plot
plt.figure(figsize=(10,6))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance (Top 15)")
plt.show()





