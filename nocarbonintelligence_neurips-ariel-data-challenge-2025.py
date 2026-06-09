import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration des chemins (Ã  adapter selon votre environnement Kaggle ou local) ---
ROOT = "/kaggle/input/ariel-data-challenge-2025"

# --- 1. Chargement des Fichiers ClÃ©s ---

print("## ğŸš€ Chargement et Inspection des Fichiers ClÃ©s (MÃ©tadonnÃ©es)\n")

# Charger les mÃ©tadonnÃ©es de base
try:
    df_train = pd.read_csv(os.path.join(ROOT, "train.csv"))
    df_star_info = pd.read_csv(os.path.join(ROOT, "train_star_info.csv"))
    df_wavelengths = pd.read_csv(os.path.join(ROOT, "wavelengths.csv"))
    df_adc_info = pd.read_csv(os.path.join(ROOT, "adc_info.csv"))
except FileNotFoundError as e:
    print(f"Erreur: Fichier non trouvÃ©. Assurez-vous que ROOT est correct. DÃ©tails: {e}")
    exit()

# --- 2. CORRECTION et Inspection des Longueurs d'Onde ---

# Correction: Transposer le DataFrame et nommer la colonne 'wavelength'
df_wavelengths_corrected = df_wavelengths.T.reset_index(drop=True).rename(columns={0: 'wavelength'})
# S'assurer que les valeurs sont numÃ©riques (float)
df_wavelengths_corrected['wavelength'] = pd.to_numeric(df_wavelengths_corrected['wavelength'])

wavelengths = df_wavelengths_corrected['wavelength'].values
num_wl = len(wavelengths)

print("### ğŸŒˆ 1. wavelengths.csv (Grille de Longueur d'Onde - CorrigÃ©)")
print(f"Dimensions corrigÃ©es : {df_wavelengths_corrected.shape}")
print(f"Nombre de points spectraux : {num_wl}")
print(f"Plage de longueurs d'onde : {wavelengths.min():.4f} Ã  {wavelengths.max():.4f} Âµm")
display(df_wavelengths_corrected.head())

# --- 3. PrÃ©paration des DonnÃ©es Spectrales (Spectraux) ---

# VÃ©rifier si df_train a le bon nombre de colonnes spectrales (N_colonnes = 1 + num_wl)
expected_cols = num_wl + 1 
if df_train.shape[1] != expected_cols:
    print(f"\n[ATTENTION] train.csv a {df_train.shape[1]} colonnes mais {num_wl} points spectraux sont attendus (+ planet_id).")

# Extraire les donnÃ©es spectrales (colonnes wl_1 Ã  wl_283)
spectral_data = df_train.iloc[:, 1:] 
print(f"\nDonnÃ©es Spectrales: {spectral_data.shape} (N_planÃ¨tes, N_longueurs_onde)")


# --- 4. STATISTIQUES DESCRIPTIVES DU SPECTRE ---

print("\n### ğŸ“Š 2. Statistiques sur la Profondeur de Transit (RpÂ²/RsÂ²)")

# Calculer la profondeur de transit moyenne et l'Ã©cart-type pour chaque planÃ¨te
df_star_info['mean_depth'] = spectral_data.mean(axis=1)
df_star_info['std_depth'] = spectral_data.std(axis=1)

print("Distribution de la Profondeur de Transit (Moyenne sur toutes les longueurs d'onde):")
display(df_star_info[['mean_depth', 'std_depth']].describe())

# Calculer la profondeur de transit moyenne par longueur d'onde
mean_spectrum = spectral_data.mean(axis=0).values

plt.figure(figsize=(12, 5))
plt.plot(wavelengths, mean_spectrum, color='purple', linewidth=2)
plt.title('Spectre Moyen de Profondeur de Transit sur le Jeu d\'EntraÃ®nement')
plt.xlabel('Longueur d\'Onde (Âµm)')
plt.ylabel('Profondeur de Transit Moyenne (RpÂ²/RsÂ²)')
plt.axvline(x=0.8, color='gray', linestyle='--', label='SÃ©paration FGS1 / AIRS-CH0 (approx)')
plt.axvline(x=1.95, color='gray', linestyle='--', )
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


# --- 5. ANALYSE DE CORRÃ‰LATION ---

print("\n### ğŸ“‰ 3. Analyse des CorrÃ©lations entre ParamÃ¨tres Physiques et Profondeur de Transit")

# Fusionner les stats spectrales avec les infos Ã©toile/planÃ¨te
df_merged = df_star_info.copy()

# Calculer la matrice de corrÃ©lation
correlation_matrix = df_merged[['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i', 'mean_depth', 'std_depth']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Matrice de CorrÃ©lation (ParamÃ¨tres Physiques vs Profondeur de Transit Moyenne/Ã‰cart-type)')
plt.show()

# --- 6. Inspection des ParamÃ¨tres ADC ---

print("\n### ğŸ”§ 4. adc_info.csv (Correction du Signal Brut)")
print(f"Dimensions : {df_adc_info.shape}")
print("Contenu :")
display(df_adc_info)
print("Note : 'gain' et 'offset' sont utilisÃ©s pour convertir les uint16 bruts en float64 de gamme dynamique complÃ¨te.")


import numpy as np
import matplotlib.pyplot as plt

print("### ğŸŒŒ 5. Signal Physique et CalibrÃ© par Instrument")

# ParamÃ¨tres principaux du systÃ¨me
planet_ids = df_train['planet_id']

# --- Fonction de calibration ADC ---
def adc_calibrate(raw_signal, gain, offset):
    """
    Convertit un signal brut ADC en valeurs physiques.
    """
    return raw_signal * gain + offset

# RÃ©cupÃ©rer les gains et offsets
fgs1_gain = df_adc_info['FGS1_adc_gain'].values[0]
fgs1_offset = df_adc_info['FGS1_adc_offset'].values[0]
airs_gain = df_adc_info['AIRS-CH0_adc_gain'].values[0]
airs_offset = df_adc_info['AIRS-CH0_adc_offset'].values[0]

# --- DÃ©terminisme 1 : Segmentation Spectrale ---
# IMPORTANT : L'indexation ci-dessous est une HYPOTHÃˆSE (1-60 et 61-283). 
# Pour un dÃ©terminisme parfait, vous DEVEZ utiliser 'wavelengths.csv' pour trouver les vrais indices 
# de colonnes (wl_i) pour la couverture spectrale de chaque instrument.
fgs1_cols = [f'wl_{i}' for i in range(1, 61)]    # Bande FGS1 (Visible/Proche-IR)
airs_cols = [f'wl_{i}' for i in range(61, 284)]  # Bande AIRS-CH0 (IR)

# Moyenne spectrale BRUTE par planÃ¨te (signal ADC)
signal_fgs1_brut = df_train[fgs1_cols].mean(axis=1)
signal_airs_brut = df_train[airs_cols].mean(axis=1)

# --- Calibration ADC ---
signal_fgs1 = adc_calibrate(signal_fgs1_brut, fgs1_gain, fgs1_offset)
signal_airs = adc_calibrate(signal_airs_brut, airs_gain, airs_offset)

# --- Contrainte Physique (Clipping) ---
# Limite infÃ©rieure Ã  0, limite supÃ©rieure typique Ã  0.1 (10%)
signal_fgs1_calib = np.clip(signal_fgs1, 0, 0.1) 
signal_airs_calib = np.clip(signal_airs, 0, 0.1)

# -------------------------------------------------------------
# --- DÃ©terminisme 2 : Ajout du Bruit (Simulation de RÃ©alisme) ---
# Ceci simule l'incertitude de mesure (souvent appelÃ©e "noise floor")

# Niveau de bruit attendu (Sigma) pour chaque instrument (en RpÂ²/RsÂ²)
NOISE_FGS1 = 5e-5 # ~50 ppm
NOISE_AIRS = 1e-4 # ~100 ppm 

# GÃ©nÃ©rer un bruit gaussien alÃ©atoire centrÃ© sur zÃ©ro
noise_fgs1_sim = np.random.normal(loc=0, scale=NOISE_FGS1, size=len(signal_fgs1_calib))
noise_airs_sim = np.random.normal(loc=0, scale=NOISE_AIRS, size=len(signal_airs_calib))

# Appliquer le bruit au signal calibrÃ©
signal_fgs1_final = signal_fgs1_calib + noise_fgs1_sim
signal_airs_final = signal_airs_calib + noise_airs_sim

# Remplacer les signaux clippÃ©s originaux par les signaux finaux (avec bruit)
signal_fgs1 = signal_fgs1_final
signal_airs = signal_airs_final
# -------------------------------------------------------------


# --- Visualisation (utilise maintenant les signaux avec bruit) ---
plt.figure(figsize=(12, 5))
plt.plot(planet_ids, signal_fgs1, color='orange', label='FGS1 (Visible)', alpha=0.7, marker='.', linestyle='') # Ajout d'un marqueur
plt.plot(planet_ids, signal_airs, color='purple', label='AIRS-CH0 (IR)', alpha=0.7, marker='.', linestyle='')
plt.title("Signal Physique de Transit par PlanÃ¨te (CalibrÃ© ADC + Bruit SimulÃ©)")
plt.xlabel("ID PlanÃ¨te")
plt.ylabel("Profondeur de Transit (RpÂ²/RsÂ²)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()


import pandas as pd
import numpy as np
from IPython.display import display

# --- DÃ‰TERMINISME DE PRÃ‰PARATION ---
# 1. DÃ©finir les donnÃ©es spectrales brutes (wl_1 Ã  wl_283) si ce n'est pas fait
spectral_cols = [col for col in df_train.columns if col.startswith('wl_')]
if not spectral_cols:
    # Cas de fallback si wl_i sont les colonnes aprÃ¨s planet_id
    spectral_cols = df_train.columns[1:] 
spectral_data = df_train[spectral_cols]

# --- 8. Construction du Dataset Final pour ModÃ©lisation ---

print("### ğŸ—‚ï¸� 6. PrÃ©paration du Dataset pour ML / ModÃ©lisation Probabiliste")

# Copier les infos Ã©toiles / planÃ¨tes (Contrainte: utilisation des donnÃ©es 'star_info')
df_dataset = df_star_info.copy()

# Ajouter les signaux simulÃ©s calibrÃ©s (Contrainte: utilisation des rÃ©sultats de la cellule 5)
df_dataset['signal_fgs1'] = signal_fgs1
df_dataset['signal_airs'] = signal_airs

# --- DÃ©terminisme AvancÃ© : Rapport de Signaux (Feature Puissante) ---
# Le rapport des signaux est souvent une meilleure feature que les signaux bruts.
# Ajout d'un epsilon pour Ã©viter la division par zÃ©ro si signal_fgs1 est nul (bien que peu probable avec le bruit)
EPSILON_RATIO = 1e-12 
df_dataset['ratio_airs_fgs1'] = df_dataset['signal_airs'] / (df_dataset['signal_fgs1'] + EPSILON_RATIO)

# Ajouter des features dÃ©rivÃ©es du spectre (BasÃ© sur la contrainte de la variabilitÃ© spectrale)
df_dataset['std_depth'] = spectral_data.std(axis=1) # Ã‰cart-type du signal spectral (forme)
df_dataset['max_depth'] = spectral_data.max(axis=1) # Profondeur maximale
df_dataset['min_depth'] = spectral_data.min(axis=1) # Profondeur minimale

# Afficher un aperÃ§u du dataset final
print(f"Dimensions du dataset final : {df_dataset.shape}")
display(df_dataset.head())

# VÃ©rification des colonnes
print("\nColonnes disponibles :")
print(df_dataset.columns.tolist())

# Optionnel : sauvegarde du dataset pour usage ML / BayÃ©sien
# df_dataset.to_csv("/kaggle/working/dataset_final.csv", index=False)


import pandas as pd
import numpy as np
from IPython.display import display

# --- DÃ‰TERMINISME DE PRÃ‰PARATION ---
# Assurez-vous que ces variables sont bien dÃ©finies Ã  partir des Ã©tapes prÃ©cÃ©dentes :
# 1. DÃ©finir les donnÃ©es spectrales brutes (wl_1 Ã  wl_283)
spectral_cols = [col for col in df_train.columns if col.startswith('wl_')]
spectral_data = df_train[spectral_cols]

# 2. RÃ©cupÃ©rer les valeurs de longueur d'onde (en microns)
# Assumer que 'wavelengths' est une liste ou un tableau NumPy des valeurs WL
# Par exemple: wavelengths = df_wl['wavelength'].values
# Pour cet exemple, je dÃ©finis une liste simple basÃ©e sur le nombre de colonnes:
try:
    # Utiliser les wavelengths provenant de l'import (si elles existent dÃ©jÃ )
    len(wavelengths)
except NameError:
    # Si 'wavelengths' n'est pas dÃ©finie, crÃ©er un tableau simple pour l'exemple
    print("ATTENTION : 'wavelengths' non dÃ©fini. CrÃ©ation d'un tableau placeholder.")
    wavelengths = np.linspace(0.5, 5.0, len(spectral_cols))


print("### ğŸ�›ï¸� 7. Features Spectrales par Longueur d'Onde")

# CrÃ©er un DataFrame pour stocker les features spectrales
df_features = df_train[['planet_id']].copy()

# --- Contrainte : Normalisation par Z-score (pour standardiser la forme du spectre) ---
# Normalisation globale (moins sensible) ou par colonne (Z-score pour chaque longueur d'onde)
spectral_normalized = (spectral_data - spectral_data.mean()) / spectral_data.std()

# Ajouter chaque longueur d'onde comme feature
for idx, wl in enumerate(wavelengths):
    # CrÃ©er le nom de colonne formatÃ© (ex: wl_0.500um)
    col_name = f"wl_{wl:.3f}um"
    
    # RÃ©cupÃ©rer la colonne normalisÃ©e (la numÃ©rotation des colonnes pandas est basÃ©e sur l'index)
    df_features[col_name] = spectral_normalized.iloc[:, idx]

# VÃ©rification des dimensions et aperÃ§u
print(f"Dimensions du dataset features : {df_features.shape}")
display(df_features.head())

# --- Fusion des datasets ---
# Fusionner les features physiques/dÃ©terministes (df_dataset) avec les features spectrales (df_features)
df_ml_ready = df_dataset.merge(df_features, on='planet_id')

print(f"\nDimensions du dataset final ML-ready : {df_ml_ready.shape}")
display(df_ml_ready.head())


import numpy as np
from IPython.display import display
# df_ml_ready est le rÃ©sultat de la cellule prÃ©cÃ©dente

# --- 10. PrÃ©paration des Targets Î¼ et Ïƒ par Instrument ---

print("### ğŸ�¯ 8. Targets Probabilistes DÃ©terministes et Robustes (Î¼ et Ïƒ)")

# Cible Î¼ : moyenne des signaux calibrÃ©s par instrument
df_ml_ready['mu_fgs1'] = df_ml_ready['signal_fgs1']
df_ml_ready['mu_airs'] = df_ml_ready['signal_airs']

# Î¼ globale (moyenne FGS1 + AIRS)
df_ml_ready['mu'] = df_ml_ready[['mu_fgs1', 'mu_airs']].mean(axis=1)

# Cible Ïƒ : Ã©cart-type combinant variabilitÃ© physique + bruit stochastique minimal
# Ce niveau de bruit est une contrainte de modÃ©lisation (noise floor)
noise_level = 1e-4  # bruit stochastique minimal (~100 ppm)
epsilon = 1e-8      # sÃ©curitÃ© pour Ã©viter sigma = 0

# Ïƒ pour FGS1
df_ml_ready['sigma_fgs1'] = np.sqrt(
    # Ã‰cart par rapport Ã  la moyenne globale (variabilitÃ© inter-instrument)
    (df_ml_ready['signal_fgs1'] - df_ml_ready['mu'])**2 + 
    # VariabilitÃ© du spectre (variabilitÃ© intrinsÃ¨que)
    df_ml_ready['std_depth']**2
) + noise_level + epsilon

# Ïƒ pour AIRS
df_ml_ready['sigma_airs'] = np.sqrt(
    # Ã‰cart par rapport Ã  la moyenne globale (variabilitÃ© inter-instrument)
    (df_ml_ready['signal_airs'] - df_ml_ready['mu'])**2 + 
    # VariabilitÃ© du spectre (variabilitÃ© intrinsÃ¨que)
    df_ml_ready['std_depth']**2
) + noise_level + epsilon

# VÃ©rification rapide
print("AperÃ§u des targets Î¼ et Ïƒ par instrument :")
display(df_ml_ready[['planet_id', 'mu_fgs1', 'sigma_fgs1', 'mu_airs', 'sigma_airs', 'mu']].head())

# Optionnel : sauvegarde pour usage ML / BayÃ©sien ou soumission
# df_ml_ready.to_csv("/kaggle/working/dataset_ml_bayes_robust.csv", index=False)

print("\nâœ… Targets probabilistes prÃ©parÃ©es avec sÃ©paration instrument, variabilitÃ© physique et bruit stochastique.")


from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

print("### ğŸ¤– RÃ©gression Multi-Output pour Î¼ et Ïƒ par Wavelength")

# --- PrÃ©parer les features ---
feature_cols = [col for col in df_ml_ready.columns if 'wl_' in col] + ['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']
X = df_ml_ready[feature_cols].values

# --- PrÃ©parer les targets Î¼ et Ïƒ par wavelength ---
wl_cols = [col for col in df_train.columns if col.startswith('wl_')]

# Î¼_target : utiliser directement les valeurs wl_i comme Î¼ (ou moyenne calibrÃ©e si dispo)
y_mu = df_train[wl_cols].values  # shape = (n_planets, 283)

# Ïƒ_target : gÃ©nÃ©rer une estimation du sigma par wavelength
# Approche simple : Ã©cart par rapport Ã  Î¼ + std_depth + noise minimal
std_depth = df_ml_ready['std_depth'].values.reshape(-1, 1)  # shape = (n_planets, 1)
noise_floor = 1e-4  # ~100ppm
y_sigma = np.sqrt((y_mu - y_mu.mean(axis=1).reshape(-1, 1))**2 + std_depth**2) + noise_floor

print(f"Shape Î¼ : {y_mu.shape}, Shape Ïƒ : {y_sigma.shape}")

# --- Split train/validation ---
X_train, X_val, y_train_mu, y_val_mu, y_train_sigma, y_val_sigma = train_test_split(
    X, y_mu, y_sigma, test_size=0.2, random_state=42
)

# -----------------------------------------------------------------
## ModÃ¨le Multi-Output pour Î¼
# -----------------------------------------------------------------
model_mu = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42)
)
print("\nEntraÃ®nement du modÃ¨le Î¼ multi-output...")
model_mu.fit(X_train, y_train_mu)

# PrÃ©dictions sur validation
y_pred_mu = model_mu.predict(X_val)
print(f"MSE global sur Î¼ : {mean_squared_error(y_val_mu, y_pred_mu):.6f}")

# -----------------------------------------------------------------
## ModÃ¨le Multi-Output pour Ïƒ
# -----------------------------------------------------------------
model_sigma = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42)
)
print("\nEntraÃ®nement du modÃ¨le Ïƒ multi-output...")
model_sigma.fit(X_train, y_train_sigma)

# PrÃ©dictions sur validation
y_pred_sigma = model_sigma.predict(X_val)

# Contrainte : Ïƒ doit rester positif
y_pred_sigma = np.clip(y_pred_sigma, 1e-6, None) 
print(f"MSE global sur Ïƒ : {mean_squared_error(y_val_sigma, y_pred_sigma):.6f}")
print(f"Ïƒ prÃ©dit moyen : {y_pred_sigma.mean():.6f}")

print("\nâœ… ModÃ¨les Î¼ et Ïƒ multi-output entraÃ®nÃ©s pour toutes les wavelengths")


import os
import numpy as np
import pandas as pd

print("### ğŸš€ InfÃ©rence Multi-Output et Submission OptimisÃ©e")

# ------------------------------------------------------------
# ğŸ”¹ 1) Charger le vrai nombre de longueurs d'onde (safe)
# ------------------------------------------------------------
df_wavelengths = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/wavelengths.csv")
NUM_SPECTRA = df_wavelengths.shape[1]
print(f"Nombre de longueurs d'onde dÃ©tectÃ© : {NUM_SPECTRA}")

# ------------------------------------------------------------
# ğŸ”¹ 2) Planet list
# ------------------------------------------------------------
planet_ids_test = [
    f for f in os.listdir("/kaggle/input/ariel-data-challenge-2025/test")
    if os.path.isdir(os.path.join("/kaggle/input/ariel-data-challenge-2025/test", f))
]
n_planets = len(planet_ids_test)

# Allocate matrix dynamically
spectra_matrix = np.zeros((n_planets, NUM_SPECTRA))

Rs_vec, Ms_vec, Ts_vec, Mp_vec, P_vec, sma_vec, i_vec = [], [], [], [], [], [], []

# ------------------------------------------------------------
# ğŸ”¹ 3) Lecture + calibration signaux
# ------------------------------------------------------------
for idx, pid in enumerate(planet_ids_test):
    planet_path = os.path.join("/kaggle/input/ariel-data-challenge-2025/test", pid)

    # --- FGS1 ---
    fgs1_files = [
        f for f in os.listdir(planet_path)
        if f.startswith("FGS1_signal_") and f.endswith(".parquet")
    ]
    if fgs1_files:
        fgs1_signal = (
            pd.concat([pd.read_parquet(os.path.join(planet_path, f)) for f in fgs1_files], axis=0)
            .mean(axis=0)
            .values
        )
    else:
        fgs1_signal = np.zeros(60)  # FGS1 est toujours 60 bins

    # --- AIRS ---
    airs_files = [
        f for f in os.listdir(planet_path)
        if f.startswith("AIRS-CH0_signal_") and f.endswith(".parquet")
    ]
    if airs_files:
        airs_signal = (
            pd.concat([pd.read_parquet(os.path.join(planet_path, f)) for f in airs_files], axis=0)
            .mean(axis=0)
            .values
        )
    else:
        airs_signal = np.zeros(NUM_SPECTRA - 60)

    # --- Calibration ADC ---
    fgs1_signal_calib = np.clip(
        fgs1_signal * df_adc_info['FGS1_adc_gain'].values[0] +
        df_adc_info['FGS1_adc_offset'].values[0],
        0, 0.1
    )

    airs_signal_calib = np.clip(
        airs_signal * df_adc_info['AIRS-CH0_adc_gain'].values[0] +
        df_adc_info['AIRS-CH0_adc_offset'].values[0],
        0, 0.1
    )

    # Assemblage spectral complet
    spectra_matrix[idx, :60] = fgs1_signal_calib
    spectra_matrix[idx, 60:] = airs_signal_calib

    # --- Lecture paramÃ¨tres physiques ---
    try:
        star_info_row = df_test_star_info.loc[
            df_test_star_info["planet_id"] == int(pid)
        ].iloc[0]
    except:
        star_info_row = {"Rs": 1.0, "Ms": 1.0, "Ts": 5500.0,
                         "Mp": 1.0, "P": 1.0, "sma": 1.0, "i": 90.0}

    Rs_vec.append(star_info_row["Rs"])
    Ms_vec.append(star_info_row["Ms"])
    Ts_vec.append(star_info_row["Ts"])
    Mp_vec.append(star_info_row["Mp"])
    P_vec.append(star_info_row["P"])
    sma_vec.append(star_info_row["sma"])
    i_vec.append(star_info_row["i"])

# ------------------------------------------------------------
# ğŸ”¹ 4) Normalisation spectrale (basÃ©e sur train)
# ------------------------------------------------------------
spectra_matrix_norm = (
    spectra_matrix - spectral_data.mean().values
) / spectral_data.std().values

# ------------------------------------------------------------
# ğŸ”¹ 5) Construction matrice X_test
# ------------------------------------------------------------
X_test = np.hstack([
    spectra_matrix_norm,
    np.array([Rs_vec, Ms_vec, Ts_vec, Mp_vec, P_vec, sma_vec, i_vec]).T
])

# ------------------------------------------------------------
# ğŸ”¹ 6) InfÃ©rence multi-output
# ------------------------------------------------------------
mu_preds = model_mu.predict(X_test.astype(np.float64))
sigma_preds = np.clip(
    model_sigma.predict(X_test.astype(np.float64)),
    1e-6, None
)

# ------------------------------------------------------------
# ğŸ”¹ 7) Construire submission
# ------------------------------------------------------------
columns = (
    ["planet_id"]
    + [f"wl_{i+1}" for i in range(NUM_SPECTRA)]
    + [f"err_{i+1}" for i in range(NUM_SPECTRA)]
)

df_submission = pd.DataFrame(
    np.hstack([np.array(planet_ids_test).reshape(-1, 1), mu_preds, sigma_preds]),
    columns=columns
)

# ------------------------------------------------------------
# ğŸ”¹ 8) Sauvegarde
# ------------------------------------------------------------
submission_path = "/kaggle/working/submission.csv"
df_submission.to_csv(submission_path, index=False)

print(f"\nâœ… Submission prÃªte : {submission_path}")
df_submission.head()


