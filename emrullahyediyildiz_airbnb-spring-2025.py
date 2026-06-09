import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')


#df = pd.read_csv("dataSP25.csv")---> lokal

df = pd.read_csv("/kaggle/input/spring-2025-regression-competition/dataSP25.csv")
test_df = pd.read_csv("/kaggle/input/spring-2025-regression-competition/compSP25.csv")



df.head()


df.info()


df.describe()


# die tabellen lÃ¶schen 
df.drop(columns=['id', 'name', 'host_id', 'host_name'], inplace=True)
df.columns


## Outliner definieren 


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,5))

# Preis
plt.subplot(1,3,1)
sns.boxplot(y=df['price'])
plt.title("Boxplot: Preis")

# Minimum Nights
plt.subplot(1,3,2)
sns.boxplot(y=df['minimum_nights'])
plt.title("Boxplot: MindestnÃ¤chte")

# Availability 365
plt.subplot(1,3,3)
sns.boxplot(y=df['availability_365'])
plt.title("Boxplot: VerfÃ¼gbarkeit (365 Tage)")

plt.tight_layout()
plt.show()




df = df[
    (df['price'] > 0) &
    (df['price'] < 2100) &
    (df['minimum_nights'] > 0) &
    (df['minimum_nights'] < 365) &
    (df['availability_365'] > 0)
]

df.shape


# Neues Feature: Hat eine Bewertung (1) oder nicht (0)
df['recent_reviewed'] = df['last_review'].notna().astype(int)

# Alte Spalte entfernen
df.drop('last_review', axis=1, inplace=True)

# Kontrolle
df['recent_reviewed'].value_counts()



# Nur numerische Spalten fÃ¼r die Korrelation
corr = df.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.show()


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution")

plt.subplot(1,3,2)
sns.histplot(df['minimum_nights'], bins=50, kde=True)
plt.title("Minimum Nights Distribution")

plt.subplot(1,3,3)
sns.histplot(df['availability_365'], bins=50, kde=True)
plt.title("Availability 365 Distribution")

plt.tight_layout()
plt.show()


# Fehlende Werte in reviews_per_month = 0 setzen (bedeutet: keine Bewertung)
df['reviews_per_month'].fillna(0, inplace=True)

# Kontrolle: Gibt es noch fehlende Werte?
df.isna().sum()


# Neues Feature: Preis pro Nacht = Gesamtpreis / MindestnÃ¤chte
df['price_per_night'] = df['price'] / df['minimum_nights']

# Kontrolle
df[['price', 'minimum_nights', 'price_per_night']].head()



# Wichtige Punkte in NYC
poi = {
    "times_square": (40.7580, -73.9855),
    "central_park": (40.7851, -73.9683),
    "jfk_airport": (40.6413, -73.7781),
    "liberty_statue": (40.6892, -74.0445),
    "brooklyn_bridge": (40.7061, -73.9969)
}

# FÃ¼r jeden Punkt Distanz berechnen
for name, coords in poi.items():
    df[f"dist_{name}"] = np.hypot(
        df['latitude'] - coords[0],
        df['longitude'] - coords[1]
    )

# Falls du Lat/Lon nicht mehr brauchst, lÃ¶schen:
df.drop(['latitude','longitude'], axis=1, inplace=True)

# Kontrolle
df[[col for col in df.columns if col.startswith("dist_")]].head()


import seaborn as sns
sns.lmplot(data=df, x='dist_times_square', y='price_per_night', height=6, aspect=1.5)




p99 = df['price_per_night'].quantile(0.99)   # 99%-Grenze
df = df[df['price_per_night'] <= p99]


sns.lmplot(data=df, x='dist_times_square', y='price_per_night', height=6, aspect=1.5)



from sklearn.preprocessing import OneHotEncoder
import pandas as pd

cat_cols = ['room_type', 'neighbourhood_group']
ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

encoded = ohe.fit_transform(df[cat_cols])
enc_cols = ohe.get_feature_names_out(cat_cols)

df = df.drop(columns=cat_cols).join(pd.DataFrame(encoded, columns=enc_cols, index=df.index))



import matplotlib.pyplot as plt
import seaborn as sns

# Numerische Spalten auswÃ¤hlen
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Korrelationsmatrix berechnen
corr = df[numeric_cols].corr()

# Heatmap zeichnen
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korrelationsmatrix (Heatmap)")
plt.tight_layout()
plt.show()



#Preis pro Nacht per_night
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='price_per_night', y='price', alpha=0.4)
plt.title('Price vs. Price per Night')
plt.xlabel('Price per Night')
plt.ylabel('Total Price')
plt.tight_layout()
plt.show()


# Boxplot: Preis pro Nacht nach Manhattan vs. Rest
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='neighbourhood_group_Manhattan', y='price_per_night')
plt.title('Preis pro Nacht: Manhattan vs. andere Bezirke')
plt.xlabel('Ist in Manhattan? (0=Nein, 1=Ja)')
plt.ylabel('Preis pro Nacht')
plt.tight_layout()
plt.show()



# Boxplot: Preis pro Nacht nach Zimmertyp (Private Room vs. andere)
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='room_type_Private room', y='price_per_night')
plt.title('Preis pro Nacht nach Zimmertyp (Private Room)')
plt.xlabel('Private Room? (0=Nein, 1=Ja)')
plt.ylabel('Preis pro Nacht')
plt.tight_layout()
plt.show()



# âœ… Dummy 0/1 in lesbares Label umwandeln
df_plot = df.copy()
df_plot['room_type_cat'] = df_plot['room_type_Private room'].map({0: 'Andere', 1: 'Private room'})

# ğŸ“¦ Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(data=df_plot, x='room_type_cat', y='price_per_night')
plt.title('Preis pro Nacht nach Zimmertyp (Private room vs. andere)')
plt.xlabel('Zimmertyp')
plt.ylabel('Preis pro Nacht')
plt.tight_layout()
plt.show()


# ğŸ”¤ Alle One-Hot-Spalten zu room_type finden
rt_cols = [c for c in df.columns if c.startswith('room_type_')]

# âš“ Basis-Kategorie annehmen (falls beim One-Hot per drop='first' entfernt)
base_category = 'Entire home/apt'  # ggf. anpassen

# ğŸ”� Kategorie aus Dummies zurÃ¼ckbauen
df_plot = df.copy()
df_plot['room_type_cat'] = base_category
for c in rt_cols:
    label = re.sub(r'^room_type_', '', c)
    df_plot.loc[df[c] == 1, 'room_type_cat'] = label

# ğŸ“¦ Boxplot fÃ¼r alle Zimmertypen
plt.figure(figsize=(7,4))
sns.boxplot(data=df_plot, x='room_type_cat', y='price_per_night')
plt.title('Preis pro Nacht nach Zimmertyp')
plt.xlabel('Zimmertyp')
plt.ylabel('Preis pro Nacht')
plt.tight_layout()
plt.show()


# Zeig mir, welche room_type*-Spalten es gibt
[c for c in df.columns if c.startswith('room_type')]



# ğŸ“Œ Features auswÃ¤hlen (nur vorhandene Spalten!)
feature_cols = [
    'room_type_Private room',
    'room_type_Shared room',
    'minimum_nights',
    'neighbourhood_group_Queens',
    'dist_times_square',
    'dist_central_park',
    'dist_jfk_airport',
    'dist_liberty_statue',
    'dist_brooklyn_bridge'
]

# ğŸ”€ Features (X) und Zielvariable (y) definieren
X = df[feature_cols]
y = df['price_per_night']   # Ziel: Preis pro Nacht

# âœ… Kontrolle
X.head()



from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import numpy as np

# ğŸ“Š Trainings-/Test-Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ğŸ¤– XGBoost-Regressionsmodell erstellen
model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

# ğŸ”§ Modell trainieren
model.fit(X_train, y_train)

# ğŸ”® Vorhersagen fÃ¼r Train- und Testdaten
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# ğŸ“ˆ Modellbewertung
print("Train RÂ²:", r2_score(y_train, train_pred))
print("Test  RÂ²:", r2_score(y_test, test_pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred)))



# exakt die Trainingsspalten (zur Sicherheit direkt aus X nehmen)
cols = list(X.columns)

# 1 Zeile mit Nullen, dann gezielt setzen
row = pd.DataFrame({c: [0.0] for c in cols})

# ---- Werte setzen (Beispiel) ----
# Zimmertyp: genau einer aktiv; Entire home/apt = beide 0
row.loc[0, 'room_type_Private room'] = 1   # Private room
row.loc[0, 'room_type_Shared room']  = 0   # nicht Shared

row.loc[0, 'minimum_nights'] = 3
row.loc[0, 'neighbourhood_group_Queens'] = 1   # 1=Queens, sonst 0

# Distanzen (Beispielwerte)
row.loc[0, 'dist_times_square']     = 0.08
row.loc[0, 'dist_central_park']     = 0.12
row.loc[0, 'dist_jfk_airport']      = 0.25
row.loc[0, 'dist_liberty_statue']   = 0.18
row.loc[0, 'dist_brooklyn_bridge']  = 0.10

# Vorhersage
pred = model.predict(row)[0]
print(f"Vorhergesagter Preis pro Nacht: ${pred:.2f}")


import joblib

#modell speichern 
joblib.dump(model, "airbnb_model.pkl")


# ğŸ“‚ Testdaten 

# ID zur spÃ¤teren Submission behalten
test_ids = test_df["id"]

# âœ… Korrekt: price_per_night wird NICHT als Feature benutzt.
#    Unser Ziel (y) war price_per_night â€“ im Test soll es vorhergesagt werden, nicht berechnet.

# ğŸ“� Distanz-Features berechnen (wie im Training)
poi = {
    "times_square": (40.7580, -73.9855),
    "central_park": (40.7851, -73.9683),
    "jfk_airport": (40.6413, -73.7781),
    "liberty_statue": (40.6892, -74.0445),
    "brooklyn_bridge": (40.7061, -73.9969)
}

for name, coords in poi.items():
    test_df[f"dist_{name}"] = np.hypot(
        test_df['latitude'] - coords[0],
        test_df['longitude'] - coords[1]
    )

# Alte Koordinaten-Spalten kÃ¶nnen raus
test_df = test_df.drop(columns=['latitude','longitude'])

# ğŸ�·ï¸� Kategorische Features (wie beim Training)
cat_cols = ['room_type', 'neighbourhood_group']

# One-Hot-Encoding
test_df_encoded = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)

# âš–ï¸� Sicherstellen, dass Features identisch zu Training sind
missing_cols = set(X.columns) - set(test_df_encoded.columns)
for col in missing_cols:
    test_df_encoded[col] = 0
test_df_encoded = test_df_encoded[X.columns]  # gleiche Reihenfolge

# ğŸ”® Vorhersagen
predictions = model.predict(test_df_encoded)




# (Optionaler Debug-Check, lokal nÃ¼tzlich â€“ fÃ¼r Kaggle kann dieser Block entfernt werden)
# print("Min Vorhersage:", predictions.min())
# print("Max Vorhersage:", predictions.max())
# print("Mittelwert:", predictions.mean())
# print("Beispiel-Predictions:", predictions[:10])



# Negative Vorhersagen auf 0 setzen (keine Unterkunft ist < 0 $)
predictions = np.clip(predictions, 0, None)

print("Neue Min:", predictions.min())
print("Neue Max:", predictions.max())



# ğŸ“„ Submission-Datei erstellen (nach Clipping)
submission_df = pd.DataFrame({
    'id': test_ids,
    'price_per_night': predictions
})

# CSV speichern
submission_df.to_csv("submissionSP25.csv", index=False)

# Kontrolle: Ersten EintrÃ¤ge anzeigen
submission_df.head()



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,4))
sns.histplot(submission_df['price_per_night'], bins=50, kde=True)
plt.title("Verteilung der vorhergesagten Preise pro Nacht")
plt.xlabel("Preis pro Nacht ($)")
plt.ylabel("Anzahl Listings")
plt.show()


