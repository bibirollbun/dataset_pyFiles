!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import missingno as msno


# Set visualization styles for consistency and clarity
sns.set(style="darkgrid", context="talk")
plt.rcParams["figure.figsize"] = (12,8)


# Train Daten zum trainieren der ML-Modelle
path="/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
df_train=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")

# Spaltenbeschreibungen
path_description="/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv"
df_description=pd.read_csv(path_description)

# Test Daten fÃ¼r die Competition
path_test="/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
df_test=pd.read_csv(path_test)

# # -------------- ID-Werte aus df_test speichern --------------

ids = df_test['ID']

# Speichern des Original-DataFrames
df_train_original = df_train.copy()


# Quick overview of shapes
print("Train data shape:", df_train.shape)
print("Test data shape:", df_test.shape)


# -----------------------------
# DATASET OVERVIEW & MISSING VALUES
# -----------------------------
print("\n--- Train Data Info ---")
df_train.info()

print("\n--- Train Data Description ---")
print(df_train.describe(include='all').T)

# List out unique values for a sample of columns (e.g., categorical ones)
sample_cat_cols = df_train.select_dtypes(include="object").columns.tolist()
for col in sample_cat_cols:
    print(f"Unique values in {col}: {df_train[col].unique()[:5]} ...")

# Missing value analysis: Count of missing values per column
missing_counts = df_train.isnull().sum().sort_values(ascending=False)
missing_perc = 100 * df_train.isnull().sum() / len(df_train)
missing_df = pd.concat([missing_counts, missing_perc], axis=1, keys=["MissingCount", "MissingPercent"])
print("\nMissing Values per Column:")
print(missing_df[missing_df.MissingCount > 0].head(15))


import matplotlib.pyplot as plt

# Histogramm fÃ¼r die EFS-Zeit
plt.hist(df_train.loc[df_train.efs==1,"efs_time"], bins=100, label="efs=1, Ereignis vorhanden")
plt.hist(df_train.loc[df_train.efs==0,"efs_time"], bins=100, label="efs=0, Vielleicht Ereignis")

# Achsenbeschriftungen und Titel
plt.xlabel("Beobachtungszeit, efs_time")
plt.ylabel("Dichte")
plt.title("Beobachtungszeiten. Entweder Zeit bis zum Ereignis oder Zeit ohne Ereignis.")

# Legende anzeigen
plt.legend()
plt.show()



# Visual 1: Bar plot for missing values
plt.figure(figsize=(14,6))
missing_df = missing_df[missing_df.MissingCount > 0].sort_values("MissingCount", ascending=False)
sns.barplot(x=missing_df.index, y=missing_df["MissingCount"], palette="viridis")
plt.xticks(rotation=70, ha='right')
plt.title("Missing Values per Column")
plt.ylabel("Count of Missing Values")
plt.xlabel("Columns")
plt.tight_layout()
plt.show()


# Visual 2: Missing value heatmap using missingno
msno.heatmap(df_train, figsize=(12,6), fontsize=12)
plt.title("Missing Value Heatmap")
plt.show()


# -----------------------------
# NUMERICAL DISTRIBUTION & STATISTICAL ANALYSIS
# -----------------------------
# Identify numerical columns (exclude target variables and IDs)
num_cols = df_train.select_dtypes(include=["int64", "int32", "float64", "float32"]).columns.tolist()
exclude = ["ID", "efs", "efs_time"]
num_cols = [c for c in num_cols if c not in exclude]

# Display histograms and KDE plots for a few key numerical variables.
for col in num_cols[:5]:  # limiting to first 5 for brevity; extend as needed
    plt.figure()
    sns.histplot(df_train[col].dropna(), kde=True, bins=30, color="steelblue")
    plt.title(f"Distribution and KDE of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# Compare distributions between efs=1 and efs=0 for a key variable 
if "age_at_hct" in df_train.columns:
    plt.figure()
    sns.histplot(data=df_train, x="age_at_hct", hue="efs", kde=True, bins=30, palette="Set2")
    plt.title("Age at HCT Distribution by Event-Free Survival (efs)")
    plt.xlabel("Age at HCT")
    plt.ylabel("Count")
    plt.show()


# Boxplots to detect outliers in numerical features (showing a few examples)
for col in num_cols[:5]:
    plt.figure()
    sns.boxplot(x=df_train[col], color="lightgreen")
    plt.title(f"Boxplot of {col} (Outlier Detection)")
    plt.xlabel(col)
    plt.show()


# -----------------------------
# CORRELATION & FEATURE RELATIONSHIPS
# -----------------------------

# Compute correlation matrix for numerical variables (including target variables)
corr_vars = df_train[num_cols + ["efs", "efs_time"]].corr()

# Create a mask to hide the upper triangle of the matrix (optional, for better visualization)
mask = np.triu(np.ones_like(corr_vars, dtype=bool))

# Increase figure size for better readability
plt.figure(figsize=(16,12))

# Improved Heatmap
sns.heatmap(
    corr_vars, 
    annot=True,            # Display correlation values inside the cells
    fmt=".2f",             # Format to 2 decimal places
    cmap="coolwarm",       # Colormap for better contrast (positive/negative)
    square=True,           # Keep the cells square
    mask=mask,             # Apply mask to hide the upper triangle
    linewidths=0.5,        # Add lines between cells for clarity
    annot_kws={"size": 8},  # Adjust annotation font size
    cbar_kws={"shrink": 0.8} # Shrink color bar size
)

# Rotate axis labels to avoid overlapping
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)

# Add plot title
plt.title("Correlation Heatmap: Numerical Features & Survival Targets", fontsize=14)

# Display the heatmap
plt.show()


# Identify top correlated features with 'efs' and 'efs_time'
corr_with_efs = corr_vars["efs"].drop(["efs"])
corr_with_efs_time = corr_vars["efs_time"].drop(["efs_time"])
print("\nTop features correlated with efs:")
print(corr_with_efs.sort_values(ascending=False).head(5))
print("\nTop features correlated with efs_time:")
print(corr_with_efs_time.sort_values(ascending=False).head(5))

# Pairplot of the top 3 features correlated with efs (if enough features exist)
top_features = corr_with_efs.abs().sort_values(ascending=False).head(3).index.tolist() + ["efs"]
sns.pairplot(df_train[top_features], hue="efs", palette="Set1")
plt.suptitle("Pairplot of Top Features Correlated with efs", y=1.02)
plt.show()


# -----------------------------
# CATEGORICAL FEATURE ANALYSIS & efs RELATIONSHIPS
# -----------------------------
# Identify categorical columns
cat_cols = df_train.select_dtypes(include="object").columns.tolist()
print("\nCategorical columns:", cat_cols)

# For demonstration, letâ€™s check if key columns exist (e.g., graft_type, race) and plot distributions
key_cat_features = []
for key in ["graft_type", "race", "race_group"]:
    if key in df_train.columns:
        key_cat_features.append(key)

# Plot the distribution of efs across these categorical variables
for col in key_cat_features:
    plt.figure()
    order = df_train[col].value_counts().index
    sns.countplot(data=df_train, x=col, hue="efs", order=order, palette="pastel")
    plt.title(f"Count Plot of {col} by efs")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=75)
    plt.legend(title="efs")
    plt.show()


def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
df_train["y"] = transform_survival_probability(df_train, time_col='efs_time', event_col='efs')


#Data Cleaning
RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in df_train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")



# Fehlende Alter-Angaben berechnen

#df_train['age_at_hct'] = df_train['age_at_hct'].fillna(df_train.groupby('prim_disease_hct')['age_at_hct'].transform('mean'))
#df_test['age_at_hct'] = df_test['age_at_hct'].fillna(df_test.groupby('prim_disease_hct')['age_at_hct'].transform('mean'))


# -------------- Zeilen mit vielen NaN-Werten bestimmen  -------------- 


# Verschiedene Schwellenwerte fÃ¼r NaN-Anteile (z. B. 10%, 30%, 50%, 70%)

#thresholds = [0.2, 0.3,0.35,0.4,0.45, 0.5,0.64]  # Anteil an fehlenden Werten


# ÃœberprÃ¼fung der Zeilen, die Ã¼ber jedem Schwellenwert liegen

#for t in thresholds:
    #threshold_value = t * df_train.shape[1]  # Berechne absolute Anzahl fehlender Werte
    #count = (df_train.isna().sum(axis=1) > threshold_value).sum()
    #print(f"Schwellenwert {int(t * 100)}%: {count} Zeilen haben mehr als {int(t * 100)}% NaN-Werte")



# -------------- Zeilen mit vielen NaN-Werten lÃ¶schen  -------------- 


#threshold_value = 0.5 * df_train.shape[1]# Berechne absolute Anzahl fehlender Werte / Wert kann angepasst werden um zu Ã¼berprÃ¼fen, ob geringere Werte besser sind fÃ¼r die ML-Modelle

#before_cleaning = df_train.shape[0]

#df_train = df_train[df_train.isna().sum(axis=1) <= threshold_value]

#after_cleaning = df_train.shape[0]


# Ausgabe der Anzahl der gelÃ¶schten Zeilen

#print(f"Anzahl der gelÃ¶schten Zeilen: {before_cleaning - after_cleaning}")


CATS = []
for c in FEATURES:
    if df_train[c].dtype=="object":
        CATS.append(c)
        df_train[c] = df_train[c].fillna("NAN")
        df_test[c] = df_test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([df_train,df_test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        #combined[c] = combined[c].astype("category")

        
     # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
df_train = combined.iloc[:len(df_train)].copy()
df_test = combined.iloc[len(df_train):].reset_index(drop=True).copy()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb


FOLDS = 10  # Anzahl der Folds fÃ¼r Cross-Validation
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(df_train))  # OOF-Vorhersagen (Out-of-Fold)
pred_xgb = np.zeros(len(df_test))  # Testvorhersagen

for i, (train_index, val_index) in enumerate(kf.split(df_train)):
    print(f"### Fold {i+1}")
    
    # Trainings- und Validierungsdaten fÃ¼r diesen Fold
    x_train, y_train = df_train.iloc[train_index][FEATURES], df_train.iloc[train_index]["y"]
    x_valid, y_valid = df_train.iloc[val_index][FEATURES], df_train.iloc[val_index]["y"]
    x_test = df_test[FEATURES]  # Testdaten (nicht geÃ¤ndert)

    # XGBoost-Modell
    model_xgb = xgb.XGBRegressor(
        max_depth=6,  # Tiefe der BÃ¤ume
        n_estimators=2000,  # Anzahl der BÃ¤ume
        learning_rate=0.02,  # Lernrate
        colsample_bytree=0.8,  # ZufÃ¤llige Auswahl von Features pro Baum
        subsample=0.8,  # ZufÃ¤llige Auswahl von Trainingsdaten pro Baum
        min_child_weight=50,  # Mindestanzahl der Daten in einem Blatt
        early_stopping_rounds=25,  # FrÃ¼hes Stoppen, wenn der Fehler sich nicht mehr verbessert
    )

    # Modell trainieren
    model_xgb.fit(
        x_train, y_train, 
        eval_set=[(x_valid, y_valid)], 
        verbose=500  # Alle 500 Schritte Ausgabe
    )

    # OOF-Vorhersagen
    oof_xgb[val_index] = model_xgb.predict(x_valid)
    
    # Testvorhersagen
    pred_xgb += model_xgb.predict(x_test)

# Durchschnitt der Testvorhersagen Ã¼ber alle Folds
pred_xgb /= FOLDS

# Speichern des Modells nach dem letzten Fold
final_model_xgb = model_xgb


from metric import score

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# Mejora del grÃ¡fico de Feature Importance - XGBoost
import matplotlib.pyplot as plt
import numpy as np

# Calcular la importancia de las caracterÃ­sticas
xgb_importance = final_model_xgb.feature_importances_

# Ordenar por importancia
sorted_idx_xgb = np.argsort(xgb_importance)[::-1]  # De mayor a menor
sorted_features_xgb = np.array(FEATURES)[sorted_idx_xgb]
sorted_xgb_importance = xgb_importance[sorted_idx_xgb]

# Limitar a las 20 caracterÃ­sticas mÃ¡s importantes (opcional)
top_n = 20
sorted_features_xgb = sorted_features_xgb[:top_n]
sorted_xgb_importance = sorted_xgb_importance[:top_n]

# Crear el grÃ¡fico
plt.figure(figsize=(12, 8))  # Aumentar el tamaÃ±o para mayor claridad
bars = plt.barh(sorted_features_xgb, sorted_xgb_importance, color='steelblue')
plt.title('ğŸ”� Feature Importance - XGBoost', fontsize=16)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Mostrar las caracterÃ­sticas mÃ¡s importantes arriba

# AÃ±adir valores al final de las barras
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()


from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# Permutation Importance Plot  XGBoost
plt.figure(figsize=(12, 8))  # Aumentar tamaÃ±o para mejor legibilidad

# Calcular Permutation Importance
perm_importance_xgb = permutation_importance(final_model_xgb, x_train, y_train, 
                                             n_repeats=10, random_state=42, n_jobs=-1)

# Ordenar la Permutation Importance en orden descendente
sorted_idx_perm_xgb = np.argsort(perm_importance_xgb.importances_mean)[::-1]

# Ordenar las caracterÃ­sticas y sus valores
sorted_features_perm_xgb = np.array(FEATURES)[sorted_idx_perm_xgb]
sorted_perm_importance_xgb = perm_importance_xgb.importances_mean[sorted_idx_perm_xgb]

# Limitar a las 20 caracterÃ­sticas mÃ¡s importantes (opcional)
top_n = 20
sorted_features_perm_xgb = sorted_features_perm_xgb[:top_n]
sorted_perm_importance_xgb = sorted_perm_importance_xgb[:top_n]

# Crear el grÃ¡fico
bars = plt.barh(sorted_features_perm_xgb, sorted_perm_importance_xgb, color='coral')
plt.title('Permutation Importance - XGBoost', fontsize=16)
plt.xlabel('Permutation Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # CaracterÃ­sticas mÃ¡s importantes arriba

# AÃ±adir valores al final de las barras
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb


FOLDS = 10  # Anzahl der Folds fÃ¼r Cross-Validation
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_catboost = np.zeros(len(df_train))  # OOF-Vorhersagen (Out-of-Fold)
pred_catboost = np.zeros(len(df_test))  # Testvorhersagen

for i, (train_index, val_index) in enumerate(kf.split(df_train)):
    print(f"### Fold {i+1}")
    
    # Trainings- und Validierungsdaten fÃ¼r diesen Fold
    x_train, y_train = df_train.iloc[train_index][FEATURES], df_train.iloc[train_index]["y"]
    x_valid, y_valid = df_train.iloc[val_index][FEATURES], df_train.iloc[val_index]["y"]
    x_test = df_test[FEATURES]  # Testdaten (nicht geÃ¤ndert)
    
    # CatBoost-Regressor Modell
    model_catboost = CatBoostRegressor(  
        learning_rate=0.1,    
        grow_policy='Lossguide',
        #early_stopping_rounds=25,
        verbose = 500
    )

    # Modell trainieren
    model_catboost.fit(
        x_train, y_train, 
        eval_set=(x_valid, y_valid)
    )

    # OOF-Vorhersagen (Out-of-Fold Vorhersagen)
    oof_catboost[val_index] = model_catboost.predict(x_valid)
    
    # Testvorhersagen
    pred_catboost += model_catboost.predict(x_test)

# Durchschnitt der Testvorhersagen Ã¼ber alle Folds
pred_catboost /= FOLDS

# Speichern des Modells nach dem letzten Fold
final_model_catboost = model_catboost


from metric import score

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_catboost
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)


# Mejora del grÃ¡fico de Feature Importance - CatBoost
import matplotlib.pyplot as plt
import numpy as np

# Calcular la importancia de las caracterÃ­sticas
catboost_importance = final_model_catboost.get_feature_importance()

# Ordenar por importancia
sorted_idx_catboost = np.argsort(catboost_importance)[::-1]  # De mayor a menor
sorted_features_catboost = np.array(FEATURES)[sorted_idx_catboost]
sorted_catboost_importance = catboost_importance[sorted_idx_catboost]

# Limitar a las 20 caracterÃ­sticas mÃ¡s importantes (opcional)
top_n = 20
sorted_features_catboost = sorted_features_catboost[:top_n]
sorted_catboost_importance = sorted_catboost_importance[:top_n]

# Crear el grÃ¡fico
plt.figure(figsize=(12, 8))  # Aumentar el tamaÃ±o para mayor claridad
bars = plt.barh(sorted_features_catboost, sorted_catboost_importance, color='steelblue')
plt.title('ğŸ”� Feature Importance - CatBoost', fontsize=16)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Mostrar las caracterÃ­sticas mÃ¡s importantes arriba

# AÃ±adir valores al final de las barras
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()






from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# Permutation Importance Plot - CatBoost
plt.figure(figsize=(12, 8))  # Aumentar tamaÃ±o para mejor legibilidad

# Calcular Permutation Importance
perm_importance_catboost = permutation_importance(final_model_catboost, x_train, y_train, 
                                                  n_repeats=10, random_state=42, n_jobs=-1)

# Ordenar la Permutation Importance en orden descendente
sorted_idx_perm_catboost = np.argsort(perm_importance_catboost.importances_mean)[::-1]

# Ordenar las caracterÃ­sticas y sus valores
sorted_features_perm_catboost = np.array(FEATURES)[sorted_idx_perm_catboost]
sorted_perm_importance_catboost = perm_importance_catboost.importances_mean[sorted_idx_perm_catboost]

# Limitar a las 20 caracterÃ­sticas mÃ¡s importantes (opcional)
top_n = 20
sorted_features_perm_catboost = sorted_features_perm_catboost[:top_n]
sorted_perm_importance_catboost = sorted_perm_importance_catboost[:top_n]

# Crear el grÃ¡fico
bars = plt.barh(sorted_features_perm_catboost, sorted_perm_importance_catboost, color='coral')
plt.title('Permutation Importance - CatBoost', fontsize=16)
plt.xlabel('Permutation Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # CaracterÃ­sticas mÃ¡s importantes arriba

# AÃ±adir valores al final de las barras
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()






import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from scipy.stats import rankdata

# FOLD SETTINGS
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Initialize arrays to store out-of-fold and test predictions
oof_lgb = np.zeros(len(df_train))
pred_lgb = np.zeros(len(df_test))

# Loop through each fold
for i, (train_index, test_index) in enumerate(kf.split(df_train)):

    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)

    # Train / Validation Split
    x_train = df_train.loc[train_index, FEATURES].copy()
    y_train = df_train.loc[train_index, 'y']    
    x_valid = df_train.loc[test_index, FEATURES].copy()
    y_valid = df_train.loc[test_index, 'y']
    x_test = df_test[FEATURES].copy()

    # Initialize the LGBMRegressor model
    model_lgb = LGBMRegressor(
        max_depth=3, 
        colsample_bytree=0.4,  
        n_estimators=2500, 
        learning_rate=0.02, 
        objective="regression",  # Regression problem
        verbose=-1,
    )

    # Fit the model without early_stopping_rounds
    model_lgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
    )

    # INFER OOF predictions (Out of Fold)
    oof_lgb[test_index] = model_lgb.predict(x_valid)

    # INFER TEST predictions
    pred_lgb += model_lgb.predict(x_test)

# Compute average test predictions for submission
pred_lgb /= FOLDS





from metric import score

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_lgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for LightGBM KaplanMeier =",m)


import matplotlib.pyplot as plt
import numpy as np

# Berechnen der Feature Importance fÃ¼r das LGBM-Modell
lgb_importance = model_lgb.feature_importances_

# Sortieren nach Wichtigkeit
sorted_idx_lgb = np.argsort(lgb_importance)[::-1]  # Von groÃŸ nach klein
sorted_features_lgb = np.array(FEATURES)[sorted_idx_lgb]
sorted_lgb_importance = lgb_importance[sorted_idx_lgb]

# Begrenzen auf die 20 wichtigsten Features (optional)
top_n = 20
sorted_features_lgb = sorted_features_lgb[:top_n]
sorted_lgb_importance = sorted_lgb_importance[:top_n]

# Erstellen des Diagramms
plt.figure(figsize=(12, 8))  # VergrÃ¶ÃŸern fÃ¼r bessere Lesbarkeit
bars = plt.barh(sorted_features_lgb, sorted_lgb_importance, color='steelblue')
plt.title('ğŸ”� Feature Importance - LightGBM', fontsize=16)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Die wichtigsten Features oben anzeigen

# Werte am Ende der Balken hinzufÃ¼gen
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()



from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# Permutation Importance fÃ¼r LightGBM
plt.figure(figsize=(12, 8))  # VergrÃ¶ÃŸern fÃ¼r bessere Lesbarkeit

# Berechnen der Permutation Importance
perm_importance_lgb = permutation_importance(model_lgb, x_train, y_train, 
                                             n_repeats=5, random_state=42, n_jobs=-1)

# Sortieren der Permutation Importance in absteigender Reihenfolge
sorted_idx_perm_lgb = np.argsort(perm_importance_lgb.importances_mean)[::-1]

# Sortieren der Features und ihrer Werte
sorted_features_perm_lgb = np.array(FEATURES)[sorted_idx_perm_lgb]
sorted_perm_importance_lgb = perm_importance_lgb.importances_mean[sorted_idx_perm_lgb]

# Begrenzen auf die 20 wichtigsten Features (optional)
top_n = 20
sorted_features_perm_lgb = sorted_features_perm_lgb[:top_n]
sorted_perm_importance_lgb = sorted_perm_importance_lgb[:top_n]

# Erstellen des Diagramms
bars = plt.barh(sorted_features_perm_lgb, sorted_perm_importance_lgb, color='coral')
plt.title('Permutation Importance - LightGBM', fontsize=16)
plt.xlabel('Permutation Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Die wichtigsten Features oben anzeigen

# Werte am Ende der Balken hinzufÃ¼gen
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()




df_train["efs_time2"] = df_train.efs_time.copy()
df_train.loc[df_train.efs==0,"efs_time2"] *= -1


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_cox = np.zeros(len(df_train))
pred_xgb_cox = np.zeros(len(df_test))

for i, (train_index, test_index) in enumerate(kf.split(df_train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = df_train.loc[train_index,FEATURES].copy()
    y_train = df_train.loc[train_index,"efs_time2"]    
    x_valid = df_train.loc[test_index,FEATURES].copy()
    y_valid = df_train.loc[test_index,"efs_time2"]
    x_test = df_test[FEATURES].copy()

    model_xgb_cox = XGBRegressor(
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=2000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        objective='survival:cox',
        eval_metric='cox-nloglik',
    )
    model_xgb_cox.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500  
    )
    
    # INFER OOF
    oof_xgb_cox[test_index] = model_xgb_cox.predict(x_valid)
    # INFER TEST
    pred_xgb_cox += model_xgb_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_cox /= FOLDS


from metric import score

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_xgb_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost Cox KaplanMeier =",m)


import matplotlib.pyplot as plt
import numpy as np

# Feature Importance fÃ¼r XGBoost Cox
plt.figure(figsize=(12, 8))  # VergrÃ¶ÃŸern fÃ¼r bessere Lesbarkeit

# Berechnen der Feature Importance
xgb_importance_cox = model_xgb_cox.get_booster().get_score(importance_type='weight')

# Sortieren der Features nach Wichtigkeit
sorted_idx_xgb_cox = np.argsort(list(xgb_importance_cox.values()))[::-1]
sorted_features_xgb_cox = np.array(list(xgb_importance_cox.keys()))[sorted_idx_xgb_cox]
sorted_xgb_importance_cox = np.array(list(xgb_importance_cox.values()))[sorted_idx_xgb_cox]

# Begrenzen auf die 20 wichtigsten Features (optional)
top_n = 20
sorted_features_xgb_cox = sorted_features_xgb_cox[:top_n]
sorted_xgb_importance_cox = sorted_xgb_importance_cox[:top_n]

# Erstellen des Diagramms
bars = plt.barh(sorted_features_xgb_cox, sorted_xgb_importance_cox, color='steelblue')
plt.title('Feature Importance - XGBoost Cox', fontsize=16)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Die wichtigsten Features oben anzeigen

# Werte am Ende der Balken hinzufÃ¼gen
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()



from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# Permutation Importance fÃ¼r XGBoost Cox
plt.figure(figsize=(12, 8))  # VergrÃ¶ÃŸern fÃ¼r bessere Lesbarkeit

# Berechnen der Permutation Importance
perm_importance_xgb_cox = permutation_importance(model_xgb_cox, x_train, y_train, 
                                                 n_repeats=10, random_state=42, n_jobs=-1)

# Sortieren der Permutation Importance in absteigender Reihenfolge
sorted_idx_perm_xgb_cox = np.argsort(perm_importance_xgb_cox.importances_mean)[::-1]

# Sortieren der Features und deren Werte
sorted_features_perm_xgb_cox = np.array(FEATURES)[sorted_idx_perm_xgb_cox]
sorted_perm_importance_xgb_cox = perm_importance_xgb_cox.importances_mean[sorted_idx_perm_xgb_cox]

# Begrenzen auf die 20 wichtigsten Features (optional)
top_n = 20
sorted_features_perm_xgb_cox = sorted_features_perm_xgb_cox[:top_n]
sorted_perm_importance_xgb_cox = sorted_perm_importance_xgb_cox[:top_n]

# Erstellen des Diagramms
bars = plt.barh(sorted_features_perm_xgb_cox, sorted_perm_importance_xgb_cox, color='coral')
plt.title('Permutation Importance - XGBoost Cox', fontsize=16)
plt.xlabel('Permutation Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Die wichtigsten Features oben anzeigen

# Werte am Ende der Balken hinzufÃ¼gen
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()



FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat_cox = np.zeros(len(df_train))
pred_cat_cox = np.zeros(len(df_test))

for i, (train_index, test_index) in enumerate(kf.split(df_train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = df_train.loc[train_index,FEATURES].copy()
    y_train = df_train.loc[train_index,"efs_time2"]    
    x_valid = df_train.loc[test_index,FEATURES].copy()
    y_valid = df_train.loc[test_index,"efs_time2"]
    x_test = df_test[FEATURES].copy()

    model_cat_cox = CatBoostRegressor(
        loss_function="Cox",
        #task_type="GPU",   
        iterations=400,     
        learning_rate=0.1,  
        grow_policy='Lossguide',
        use_best_model=False,
    )
    model_cat_cox.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=100)
    
    # INFER OOF
    oof_cat_cox[test_index] = model_cat_cox.predict(x_valid)
    # INFER TEST
    pred_cat_cox += model_cat_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat_cox /= FOLDS


from metric import score

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_cat_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost Cox KaplanMeier =",m)


import matplotlib.pyplot as plt
import numpy as np

# Feature Importance fÃ¼r CatBoost Cox
catboost_importance = model_cat_cox.get_feature_importance()

# Sortieren nach Wichtigkeit
sorted_idx_catboost = np.argsort(catboost_importance)[::-1]  # Von hÃ¶chster zu niedrigster Wichtigkeit
sorted_features_catboost = np.array(FEATURES)[sorted_idx_catboost]
sorted_catboost_importance = catboost_importance[sorted_idx_catboost]

# Begrenzen auf die 20 wichtigsten Features (optional)
top_n = 20
sorted_features_catboost = sorted_features_catboost[:top_n]
sorted_catboost_importance = sorted_catboost_importance[:top_n]

# Erstellen des Diagramms
plt.figure(figsize=(12, 8))  # VergrÃ¶ÃŸern fÃ¼r bessere Lesbarkeit
bars = plt.barh(sorted_features_catboost, sorted_catboost_importance, color='steelblue')
plt.title('ğŸ”� Feature Importance - CatBoost Cox', fontsize=16)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Die wichtigsten Features oben anzeigen

# Werte am Ende der Balken hinzufÃ¼gen
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()



from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# Permutation Importance fÃ¼r CatBoost Cox
plt.figure(figsize=(12, 8))  # VergrÃ¶ÃŸern fÃ¼r bessere Lesbarkeit

# Berechnen der Permutation Importance
perm_importance_catboost_cox = permutation_importance(model_cat_cox, x_train, y_train, 
                                                      n_repeats=10, random_state=42, n_jobs=-1)

# Sortieren der Permutation Importance in absteigender Reihenfolge
sorted_idx_perm_catboost_cox = np.argsort(perm_importance_catboost_cox.importances_mean)[::-1]

# Sortieren der Features und deren Werte
sorted_features_perm_catboost_cox = np.array(FEATURES)[sorted_idx_perm_catboost_cox]
sorted_perm_importance_catboost_cox = perm_importance_catboost_cox.importances_mean[sorted_idx_perm_catboost_cox]

# Begrenzen auf die 20 wichtigsten Features (optional)
top_n = 20
sorted_features_perm_catboost_cox = sorted_features_perm_catboost_cox[:top_n]
sorted_perm_importance_catboost_cox = sorted_perm_importance_catboost_cox[:top_n]

# Erstellen des Diagramms
bars = plt.barh(sorted_features_perm_catboost_cox, sorted_perm_importance_catboost_cox, color='coral')
plt.title('Permutation Importance - CatBoost Cox', fontsize=16)
plt.xlabel('Permutation Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Die wichtigsten Features oben anzeigen

# Werte am Ende der Balken hinzufÃ¼gen
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()



from scipy.stats import rankdata

# Vorhersagen der 5 Modelle
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()  # Wahre Werte aus den Trainingsdaten
y_pred = df_train[["ID"]].copy()  # Nur IDs, spÃ¤ter mit Vorhersagen gefÃ¼llt

# Kombinieren der Ranglisten der Modelle (Division bleibt bestehen, wie gewÃ¼nscht)
y_pred["prediction"] = rankdata(oof_xgb) + rankdata(oof_catboost) + rankdata(oof_lgb)/ + rankdata(oof_xgb_cox) + rankdata(oof_cat_cox)

# Berechnung des Cross-Validation Scores fÃ¼r das Ensemble-Modell
m = score(y_true.copy(), y_pred.copy(), "ID")

# Ausgabe des Gesamtscores des Ensemble-Modells
print(f"\nOverall CV for Ensemble =",m)





import matplotlib.pyplot as plt
import numpy as np

# Liste der Modelle im Ensemble
ensemble_models = [final_model_xgb, final_model_catboost, model_lgb, model_xgb_cox, model_cat_cox]

# Berechnen der Feature Importance fÃ¼r alle Modelle im Ensemble
feature_importance_ensemble = []
for model in ensemble_models:
    if hasattr(model, 'feature_importances_'):
        feature_importance_ensemble.append(model.feature_importances_)

# Mittelwert der Feature Importances Ã¼ber alle Modelle berechnen
feature_importance_ensemble_mean = np.mean(feature_importance_ensemble, axis=0)

# Sortieren der Feature Importance in absteigender Reihenfolge
sorted_idx_fi_ensemble = np.argsort(feature_importance_ensemble_mean)[::-1]

# Sortieren der Features und deren Werte
sorted_features_fi_ensemble = np.array(FEATURES)[sorted_idx_fi_ensemble]
sorted_fi_ensemble = feature_importance_ensemble_mean[sorted_idx_fi_ensemble]

# Begrenzen auf die 20 wichtigsten Features (optional)
top_n = 20
sorted_features_fi_ensemble = sorted_features_fi_ensemble[:top_n]
sorted_fi_ensemble = sorted_fi_ensemble[:top_n]

# Erstellen des Diagramms
plt.figure(figsize=(12, 8))  # VergrÃ¶ÃŸern fÃ¼r bessere Lesbarkeit
bars = plt.barh(sorted_features_fi_ensemble, sorted_fi_ensemble, color='steelblue')
plt.title('Feature Importance - Ensemble Modell', fontsize=16)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Die wichtigsten Features oben anzeigen

# Werte am Ende der Balken hinzufÃ¼gen
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Liste der PI-Werte fÃ¼r jedes Modell (Die Werte wurden bereits berechnet und gespeichert)
pi_values = {
    'XGBoost': perm_importance_xgb.importances_mean,
    'CatBoost': perm_importance_catboost.importances_mean,
    'LightGBM': perm_importance_lgb.importances_mean,
    'XGBoost Cox': perm_importance_xgb_cox.importances_mean,
    'CatBoost Cox': perm_importance_catboost_cox.importances_mean
}

# Berechnen der mittleren Permutation Importance (PI) Ã¼ber alle Modelle
# Zuerst alle PI-Werte in einer Liste zusammenfassen
permutation_importance_ensemble = np.array(list(pi_values.values()))

# Mittelwert Ã¼ber alle Modelle berechnen
permutation_importance_ensemble_mean = np.mean(permutation_importance_ensemble, axis=0)

# Sortieren der Permutation Importance in absteigender Reihenfolge
sorted_idx_pi_ensemble = np.argsort(permutation_importance_ensemble_mean)[::-1]

# Sortieren der Features und deren Werte
sorted_features_pi_ensemble = np.array(FEATURES)[sorted_idx_pi_ensemble]
sorted_pi_ensemble = permutation_importance_ensemble_mean[sorted_idx_pi_ensemble]

# Begrenzen auf die 20 wichtigsten Features (optional)
top_n = 20
sorted_features_pi_ensemble = sorted_features_pi_ensemble[:top_n]
sorted_pi_ensemble = sorted_pi_ensemble[:top_n]

# Erstellen des Diagramms
plt.figure(figsize=(12, 8))  # VergrÃ¶ÃŸern fÃ¼r bessere Lesbarkeit
bars = plt.barh(sorted_features_pi_ensemble, sorted_pi_ensemble, color='coral')
plt.title('Permutation Importance - Ensemble Modell', fontsize=16)
plt.xlabel('Permutation Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()  # Die wichtigsten Features oben anzeigen

# Werte am Ende der Balken hinzufÃ¼gen
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()



y_pred = df_test[["ID"]].copy()
y_pred["prediction"] = rankdata(pred_xgb) + rankdata(pred_catboost) + rankdata(pred_lgb)/ + rankdata(pred_xgb_cox) + rankdata(pred_cat_cox)

# Save submission
y_pred.to_csv('submission.csv', index=False)


import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

# Set the plot style
sns.set(style="whitegrid", context="talk")

# Ensure 'race_group' has no null values
df_plot = df_train.copy()
df_plot['race_group'] = df_plot['race_group'].fillna('Unknown')

# Filter data up to 90 days
df_plot = df_plot[df_plot['efs_time'] <= 90]

# Create the figure
plt.figure(figsize=(12, 8))
kmf = KaplanMeierFitter()

# Custom color palette
custom_palette = ['#FF5733', '#99EDCC', '#043927', '#E36588', '#9A275A', '#00205B']

# Iterate over each race group (sorted for consistency)
unique_groups = sorted(df_plot['race_group'].unique())
for idx, group in enumerate(unique_groups):
    mask = df_plot['race_group'] == group
    kmf.fit(
        durations=df_plot.loc[mask, 'efs_time'],
        event_observed=df_plot.loc[mask, 'efs'],
        label=str(group)
    )
    kmf.plot(ci_show=False, linewidth=2, color=custom_palette[idx % len(custom_palette)])

# Customize the plot
plt.title("Survival Curves by Race Group ", fontsize=22, fontweight="bold", color="#8B0000")
plt.xlabel("Time", fontsize=16, fontweight="bold")
plt.ylabel("Survival Probability", fontsize=16, fontweight="bold")
plt.xlim(0, 90)

# Set custom ticks for x and y axes (strictly visual)
plt.xticks([0, 50, 90], [0, 50, 90])

# Display y-axis ticks as percentages
plt.yticks([0.4, 0.7, 1.0], ["40%", "70%", "100%"])

# Make the legend appear a bit lower
plt.legend(
    title="Race Group", 
    fontsize=17, 
    title_fontsize=16, 
    loc='upper right', 
    bbox_to_anchor=(1, 0.9)
)

# Remove grid lines
plt.grid(False)

# Remove plot borders (spines)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import pandas as pd

# Set the plot style
sns.set(style="whitegrid", context="talk")

# Ensure 'age_at_hct' has no null values
df_plot = df_train.copy()
df_plot['age_at_hct'] = df_plot['age_at_hct'].fillna(df_plot['age_at_hct'].median())

# Filter data up to 90 days
df_plot = df_plot[df_plot['efs_time'] <= 90]

# Create age groups
bins = [0, 18, 30, 45, 60, 75, 100]
labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
df_plot['age_group'] = pd.cut(df_plot['age_at_hct'], bins=bins, labels=labels, right=False)

# Create the figure
plt.figure(figsize=(12, 8))
kmf = KaplanMeierFitter()

# Custom color palette
custom_palette = ['#FF5733', '#99EDCC', '#043927', '#E36588', '#9A275A', '#00205B']

# Iterate over each age group (sorted for consistency)
unique_groups = sorted(df_plot['age_group'].dropna().unique())
for idx, group in enumerate(unique_groups):
    mask = df_plot['age_group'] == group
    if mask.sum() > 0:  # Ensure there are samples in the group
        kmf.fit(
            durations=df_plot.loc[mask, 'efs_time'],
            event_observed=df_plot.loc[mask, 'efs'],
            label=str(group)
        )
        kmf.plot(ci_show=False, linewidth=2, color=custom_palette[idx % len(custom_palette)])

# Customize the plot
plt.title("Survival Curves by Age Group ", fontsize=22, fontweight="bold", color="#8B0000")
plt.xlabel("Time ", fontsize=16, fontweight="bold")
plt.ylabel("Survival Probability", fontsize=16, fontweight="bold")
plt.xlim(0, 90)

# Custom tick labels (strictly visual)
plt.xticks([0, 50, 90], [0, 50, 90])
plt.yticks([0.4, 0.7, 1.0], ["40%", "70%", "100%"])

# Adjust legend placement
plt.legend(
    title="Age Group",
    fontsize=17,
    title_fontsize=16,
    loc='upper right',
    bbox_to_anchor=(1, 0.9)
)

# Remove grid lines
plt.grid(False)

# Remove plot borders (spines)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()




