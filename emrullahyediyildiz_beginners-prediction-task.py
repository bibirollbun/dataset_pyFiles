import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",100)
 


#  Trainings- und Testdaten einlesen
train_df = pd.read_csv("/kaggle/input/beginners-prediction-task/train.csv")
test_df = pd.read_csv("/kaggle/input/beginners-prediction-task/test.csv")


#  Ersten Ãœberblick verschaffen
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# Spalten anzeigen und null werte allgein INFO
train_df.info()


# Fehlende Werte prÃ¼fen

missing = train_df.isnull().sum()
print("Anzahl fehlender Werte pro Spalte:\n", missing[missing > 0])


#Basisstatistiken anzeigen
train_df.describe()


# Erste Zeilen ansehen
train_df.head()


# Fehlende Werte auffÃ¼llen
# attendance_percentage mit dem Mittelwert ersetzen
train_df['attendance_percentage'].fillna(train_df['attendance_percentage'].mean(), inplace=True)


# Kontrolle: noch fehlende Werte?
train_df.isnull().sum().sum()


# ğŸ—‚ï¸� Kategorische Spalten auflisten
cat_cols = train_df.select_dtypes(include='object').columns

print(f"{len(cat_cols)} kategorische Spalten gefunden:\n")
print(cat_cols.tolist())



# Beispiel: Werteverteilung fÃ¼r 'gender'
print(train_df['gender'].value_counts())

# Schneller Ãœberblick fÃ¼r ALLE kategorischen Spalten
for col in cat_cols:
    print(f"\n{col}:\n{train_df[col].value_counts(normalize=True).head()}")



import re

def clean_category(s):
    """Hilfsfunktion: Leerzeichen, Case, Sonderformen fixen"""
    if pd.isnull(s):
        return "Unknown"
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)  # doppelte Leerzeichen entfernen
    return s

# Mapping-Dictionary fÃ¼r alle relevanten Spalten
maps = {
    "gender": {
        "male": "Male", "m": "Male", 
        "female": "Female", "f": "Female"
    },
    "religion": {
        "islam": "Islam", "muslim": "Islam",
        "christianity": "Christianity", "christian": "Christianity"
    },
    "school_type": {
        "public": "Public", "government": "Public",
        "private": "Private"
    },
    "parental_education_level": {
        "secondary": "Secondary", "secondry": "Secondary", "soconary": "Secondary",
        "tertiary": "Tertiary"
    },
    "extracurricular_activity": {
        "yes": "Yes", "no": "No"
    },
    "learning_disability": {
        "yes": "Yes", "no": "No"
    },
    "tutoring_mentoring_program": {
        "yes": "Yes", "no": "No"
    },
    "parental_involvement_level": {
        "low": "Low", "medium": "Medium", "high": "High"
    },
    "bullying_experience": {
        "yes": "Yes", "no": "No"
    },
    "peer_interaction_level": {
        "low": "Low", "medium": "Medium", "high": "High"
    }
}

# Spalten bereinigen
for col in cat_cols:  
    train_df[col] = train_df[col].apply(clean_category)  # alles klein + trim
    if col in maps:  
        train_df[col] = train_df[col].map(maps[col]).fillna(train_df[col].str.title())
    else:
        train_df[col] = train_df[col].str.title()

# Kontrolle
for col in ["gender","ethnicity","religion","school_type",
            "parental_education_level","extracurricular_activity",
            "learning_disability","school_location","tutoring_mentoring_program",
            "parental_involvement_level","bullying_experience","peer_interaction_level"]:
    print(f"\n{col} unique values:", train_df[col].unique())



# ğŸ—‘ï¸� UnnÃ¶tige Spalten ausschlieÃŸen
drop_cols = ["ID", "name"]

cat_cols = [col for col in train_df.select_dtypes(include='object').columns if col not in drop_cols]

print(f"{len(cat_cols)} kategorische Spalten (ohne ID & name):\n")
print(cat_cols)



import matplotlib.pyplot as plt

# ğŸ�¨ Pie-Charts fÃ¼r alle kategorischen Spalten
for col in cat_cols:
    plt.figure(figsize=(5,5))
    train_df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f"Verteilung: {col}")
    plt.ylabel("")  # y-Achsenlabel ausblenden
    plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# ğŸ�¯ Nur numerische Spalten auswÃ¤hlen
num_cols = train_df.select_dtypes(include=['int64','float64']).columns

# ğŸ”— Korrelationsmatrix berechnen
corr = train_df[num_cols].corr()

# ğŸ–¼ï¸� Heatmap plotten
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Korrelationsmatrix der numerischen Features", fontsize=14)
plt.show()

# Extra: Nur Korrelation mit Zielvariable 'gpa'
corr_target = corr["gpa"].sort_values(ascending=False)
print("Korrelation mit GPA:\n", corr_target)



# ===== 1) Ziel & Feature-Listen =====
target = "gpa"

# Spalten, die NICHT ins Modell sollen (IDs, Namen)
drop_cols = ["ID", "name", "hobby", target]


# Kategorische Spalten (mÃ¼ssen encodiert werden)
cat_cols = [
    "gender","ethnicity","religion","state_of_origin","school_type",
    "parental_education_level","extracurricular_activity","learning_disability",
    "school_location","tutoring_mentoring_program","parental_involvement_level",
    "bullying_experience","peer_interaction_level"
]

# ===== 2) X / y bauen =====
X_raw = train_df.drop(columns=drop_cols)
y = train_df[target]

# ===== 3) One-Hot-Encoding mit pandas =====
# drop_first=True vermeidet Dummy-Falle, sparse=False fÃ¼r DataFrame-RÃ¼ckgabe
X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

print("Feature shape nach OHE:", X.shape)
X.head()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",100)
 
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, ExtraTreeRegressor
#pip install xgboost
from xgboost import XGBRegressor
from sklearn.svm import SVR
 
from sklearn.neural_network import MLPRegressor
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
 
def algo_test(x,y):
        #ausgewÃ¤hlten modelle 
        L=LinearRegression()
        R=Ridge()
        Lass=Lasso()
        E=ElasticNet()
        sgd=SGDRegressor()
        ETR=ExtraTreeRegressor()
        GBR=GradientBoostingRegressor()
        kn=KNeighborsRegressor()
        rkn=RadiusNeighborsRegressor(radius=1.0)
        ada=AdaBoostRegressor()
        dt=DecisionTreeRegressor()
        xgb=XGBRegressor()
        svr=SVR()
        mlp_regressor = MLPRegressor()
 
       
        algos=[L,R,Lass,E,sgd,ETR,GBR,ada,kn,dt,xgb,svr,mlp_regressor]
        algo_names=['Linear','Ridge','Lasso','ElasticNet','SGD','Extra Tree','Gradient Boosting',
                    'KNeighborsRegressor','AdaBoost','Decision Tree','XGBRegressor','SVR','mlp_regressor']
        x=MinMaxScaler().fit_transform(x)
        x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.20,random_state=42)
        r_squared= []
        rmse= []
        mae= []
        #Wir erstellen einen Datenrahmen, um die Fehler- und Genauigkeitsraten tabellarisch darzustellen
        result=pd.DataFrame(columns=['R_Squared','RMSE','MAE'],index=algo_names)

        for algo in algos:
            p=algo.fit(x_train,y_train).predict(x_test)
            r_squared.append(r2_score(y_test,p))
            rmse.append(mean_squared_error(y_test,p)**.5)
            mae.append(mean_absolute_error(y_test,p))

 
        #Die Werte des Genauigkeits- und Fehlerraten werden in die Tabelle mit dem Namen Ergebnis eingefÃ¼gt
        result.R_Squared=r_squared
        result.RMSE=rmse
        result.MAE=mae
       #Es sortiert die erstellte Ergebnistabelle nach der Genauigkeitsrate (r2_score) und gibt zurÃ¼ck
        rtable=result.sort_values('R_Squared',ascending=False)
        return rtable


rtable = algo_test(X.values, y.values)
rtable


# Testdaten laden
test_df = pd.read_csv("/kaggle/input/beginners-prediction-task/test.csv")

# Gleiche Vorverarbeitung wie oben
X_test_raw = test_df.drop(columns=["ID", "name"])
X_test = pd.get_dummies(X_test_raw, columns=cat_cols, drop_first=True)

# Spalten mit Train-Features ausrichten (fehlende Spalten fÃ¼llen, extra droppen)
X_aligned, X_test_aligned = X.align(X_test, join="left", axis=1, fill_value=0)

# Beispiel: starkes Modell wÃ¤hlen (z. B. XGBRegressor)
best_model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_aligned, y)

test_pred = best_model.predict(X_test_aligned)





# Submission-Datei bauen (Competition-Format prÃ¼fen!)
sub = pd.DataFrame({
    "ID": test_df["ID"],
    "gpa": test_pred
})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 1) Daten vorbereiten (X, y stammen von deinem OHE-Step) ===
X_mat = X.values  # Feature-Matrix (nach get_dummies, ohne hobby)
y_vec = y.values  # Ziel: gpa

# Optional: Skalieren (nicht nÃ¶tig fÃ¼r Trees, aber konsistent zu deiner Funktion)
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y_vec, test_size=0.2, random_state=42)
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# === 2) Bestes Modell trainieren ===
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train_sc, y_train)
y_pred = gbr.predict(X_test_sc)

# === 3) Metriken ===
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"RÂ²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

# === 4) Parity Plot (Predicted vs Actual) ===
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")  # Diagonale = perfekt
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("Predicted vs Actual (Parity Plot)")
plt.tight_layout()
plt.show()

# === 5) Residuen: Histogramm ===
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=20, kde=True)
plt.xlabel("Residual (Actual - Predicted)")
plt.title("Residual Distribution")
plt.tight_layout()
plt.show()

# === 6) Residuen vs. Vorhersage (HomoskedastizitÃ¤t prÃ¼fen) ===
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted GPA")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()



# Feature Importance fÃ¼r GradientBoosting
importances = gbr.feature_importances_
feat_names = X.columns

imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
imp_df = imp_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=imp_df.head(15), palette="viridis")
plt.title("Top 15 Feature Importances (Gradient Boosting)")
plt.tight_layout()
plt.show()

imp_df.head(15)





