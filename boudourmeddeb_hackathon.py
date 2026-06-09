# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
DATA_PATH = "/kaggle/input/ashrae-energy-prediction/"



building = pd.read_csv(DATA_PATH + "building_metadata.csv")
weather = pd.read_csv(DATA_PATH + "weather_train.csv")



chunksize = 1_000_000
train_chunks = pd.read_csv(
    DATA_PATH + "train.csv",
    chunksize=chunksize
)

sample_list = []

for chunk in train_chunks:
    # Ã‰lectricitÃ© uniquement
    chunk = chunk[chunk["meter"] == 0]

    # Jointure pour filtrer Office
    chunk = chunk.merge(
        building[["building_id", "primary_use"]],
        on="building_id",
        how="left"
    )
    chunk = chunk[chunk["primary_use"] == "Office"]

    # Ã‰chantillon 1 %
    if len(chunk) > 0:
        chunk = chunk.sample(frac=0.01, random_state=42)
        sample_list.append(chunk)

train = pd.concat(sample_list, ignore_index=True)
print("Dataset aprÃ¨s Ã©chantillonnage :", train.shape)



chunksize = 1_000_000
train_chunks = pd.read_csv(
    DATA_PATH + "train.csv",
    chunksize=chunksize
)

sample_list = []

for chunk in train_chunks:
    # Ã‰lectricitÃ© uniquement
    chunk = chunk[chunk["meter"] == 0]

    # Merge CORRIGÃ‰ avec site_id
    chunk = chunk.merge(
        building[["building_id", "site_id", "primary_use"]],
        on="building_id",
        how="left"
    )

    # Filtrer bureaux
    chunk = chunk[chunk["primary_use"] == "Office"]

    # Ã‰chantillon 1 %
    if len(chunk) > 0:
        chunk = chunk.sample(frac=0.01, random_state=42)
        sample_list.append(chunk)

train = pd.concat(sample_list, ignore_index=True)
print("Colonnes disponibles :", train.columns)



train = train.merge(
    weather,
    on=["site_id", "timestamp"],
    how="left"
)

print("Colonnes aprÃ¨s merge mÃ©tÃ©o :")
print(train.columns)



weather["timestamp"] = pd.to_datetime(weather["timestamp"])
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["hour"] = train["timestamp"].dt.hour
train["weekday"] = train["timestamp"].dt.weekday



# Simulation occupancy
def simulate_occupancy(row):
    if row["weekday"] >= 5:  # week-end
        return 0
    elif 8 <= row["hour"] < 12 or 13 <= row["hour"] < 18:
        return 1
    else:
        return 0

train["occupancy"] = train.apply(simulate_occupancy, axis=1)

# Bruit capteur (10%)
noise = np.random.choice([0, 1], size=len(train), p=[0.1, 0.9])
train["occupancy"] = train["occupancy"] * noise



train["air_temperature"] = train["air_temperature"].fillna(
    train["air_temperature"].mean()
)

train["meter_reading"] = train["meter_reading"].fillna(0)



X = train[
    ["meter_reading", "air_temperature", "hour"]
]



features = ["meter_reading", "air_temperature", "hour"]
X = train[features]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=features,
    index=train.index
)



import warnings
warnings.filterwarnings("ignore", category=UserWarning)


from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

features = ["meter_reading", "air_temperature", "hour"]

# SÃ©lection features
X = train[features]

# Normalisation
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=features,
    index=train.index
)

# ModÃ¨le non supervisÃ©
iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# FIT + PREDICT ENSEMBLE (clÃ© pour Ã©viter le warning)
train["anomaly"] = iso.fit_predict(X_scaled)

# Conversion labels
train["anomaly"] = train["anomaly"].map({1: 0, -1: 1})



train["anomaly"].value_counts(normalize=True)



train["waste"] = (
    (train["anomaly"] == 1) & 
    (train["occupancy"] == 0)
).astype(int)



total_kwh = train["meter_reading"].sum()
wasted_kwh = train.loc[train["waste"] == 1, "meter_reading"].sum()
saving_pct = (wasted_kwh / total_kwh) * 100

print(f"Consommation totale : {total_kwh:.2f} kWh")
print(f"Gaspillage dÃ©tectÃ© : {wasted_kwh:.2f} kWh")
print(f"Ã‰conomie potentielle : {saving_pct:.2f} %")



# -------------------------------
# ğŸ”¹ 1ï¸�âƒ£ Copier la consommation actuelle
# -------------------------------
train["optimized_reading"] = train["meter_reading"].copy()

# -------------------------------
# ğŸ”¹ 2ï¸�âƒ£ RÃ¨gle 1 : Ã©teindre clim/lumiÃ¨re si salle vide
# -------------------------------
train.loc[train["occupancy"]==0, "optimized_reading"] *= 0.5
# On simule quâ€™on Ã©conomise 50% si salle vide

# -------------------------------
# ğŸ”¹ 3ï¸�âƒ£ RÃ¨gle 2 : rÃ©duire consommation si tempÃ©rature confortable
# TempÃ©rature confortable : 20Â°C Ã  24Â°C
# -------------------------------
mask_comfort = (train["air_temperature"] >= 20) & (train["air_temperature"] <= 24)
train.loc[mask_comfort, "optimized_reading"] *= 0.8
# On simule 20% Ã©conomie si tempÃ©rature confortable

# -------------------------------
# ğŸ”¹ 4ï¸�âƒ£ Calcul KPI consommation optimisÃ©e
# -------------------------------
optimized_total = train["optimized_reading"].sum()
optimized_saving = total_kwh - optimized_total
optimized_saving_pct = (optimized_saving / total_kwh) * 100

print(f"Consommation totale avant optimisation : {total_kwh:.2f} kWh")
print(f"Consommation totale aprÃ¨s optimisation : {optimized_total:.2f} kWh")
print(f"Ã‰conomie potentielle : {optimized_saving_pct:.2f} %")

# -------------------------------
# ğŸ”¹ 5ï¸�âƒ£ Visualisation comparaison
# -------------------------------
plt.figure(figsize=(12,5))
plt.plot(train["timestamp"], train["meter_reading"], label="Consommation rÃ©elle")
plt.plot(train["timestamp"], train["optimized_reading"], label="Consommation optimisÃ©e", alpha=0.8)
plt.scatter(
    train.loc[train["waste"]==1, "timestamp"],
    train.loc[train["waste"]==1, "meter_reading"],
    color="red", s=10, label="Gaspillage dÃ©tectÃ©"
)
plt.legend()
plt.title("Optimisation automatique de la consommation Ã©nergÃ©tique")
plt.xlabel("Temps")
plt.ylabel("kWh")
plt.show()




# Sauvegarder le DataFrame prÃ©parÃ© pour le dashboard
train.to_csv("/kaggle/working/train_prepared.csv", index=False)





import joblib

# Sauvegarder
joblib.dump(iso, "energy_model.pkl")

# Recharger
energy_model = joblib.load("energy_model.pkl")


