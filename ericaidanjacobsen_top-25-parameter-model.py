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

import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

# Definiere die Spalten, die als Features genutzt werden sollen
top_feature_cols = ['conditioning_intensity', 'year_hct', 'age_at_hct',
                    'sex_match', 'donor_age', 'prim_disease_hct', 'gvhd_proph', 
                    'comorbidity_score', 'karnofsky_score', 'cyto_score_detail', 
                    'dri_score', 'cmv_status', 'race_group', 'in_vivo_tcd', 'hla_match_drb1_high', 
                    'tbi_status', 'cardiac', 'cyto_score', 'hla_nmdp_6', 'mrd_hct', 'hla_match_dqb1_high', 
                    'hla_match_a_low', 'pulm_severe', 'psych_disturb', 'hla_match_c_high', 'ID']

def clean_data(df, is_train=True):
    """
    Bereinigt den DataFrame:
    - Ersetzt fehlende Werte (NaN) mit sinnvollen Standardwerten.
    - Entfernt problematische Spaltennamen.
    - Wandelt kategorische Spalten in numerische um.
    """
    df.fillna(0, inplace=True)
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).astype('category').cat.codes
    return df

def main():

    # Trainings- und Testdaten einlesen
    train_path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
    test_path = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Daten bereinigen und auf die benötigten Spalten beschränken
    df_train = clean_data(df_train, is_train=True)
    df_train = df_train[top_feature_cols + ['efs', 'efs_time']]
    
    df_test = clean_data(df_test, is_train=False)
    df_test = df_test[top_feature_cols]

    # Features und Zielvariable definieren
    X_train = df_train.drop(columns=['efs', 'efs_time'], errors='ignore')
    y_train = df_train['efs']
    X_test = df_test

    # Nur numerische Spalten verwenden
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # XGBRegressor initialisieren (ohne vorab feste Parameter)
    xgb_model = XGBRegressor(random_state=42)

    # Parameter Grid für das Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    # GridSearchCV initialisieren
    grid_search = GridSearchCV(estimator=xgb_model,
                               param_grid=param_grid,
                               scoring='neg_mean_squared_error',  # Negatives MSE als Scoring-Metrik
                               cv=3,                            # 3-fache Cross-Validation
                               verbose=1,
                               n_jobs=-1)

    # Hyperparameter Tuning durchführen
    grid_search.fit(X_train, y_train)

    # Bestes Modell verwenden
    best_model = grid_search.best_estimator_

    # Vorhersagen für Testdaten mithilfe des besten Modells vornehmen
    print("Erstelle Vorhersagen für Testdaten...")
    risk_scores = best_model.predict(X_test)
    
    # Überprüfen, ob die ID-Spalte vorhanden ist
    if 'ID' in X_test.columns:
        results = pd.DataFrame({'ID': X_test['ID'], 'prediction': risk_scores})
    else:
        results = pd.DataFrame({'prediction': risk_scores})
    
    # Ergebnisse abspeichern (dieser Teil kann aktiviert werden, wenn das Speichern gewünscht ist)
    results.to_csv(os.path.join("/kaggle/working/", "submission.csv"), index=False)
    print("Ergebnisse wurden in results.csv gespeichert.")

if __name__ == "__main__":
    main()





