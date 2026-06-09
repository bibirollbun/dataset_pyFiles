import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import catboost as cb
import itertools
import joblib



# Load the Data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


print(df_train.shape)
print(df_train.isna().sum())
df_train.head()


print(df_test.shape)
print(df_test.isna().sum())
df_test.head()


sample_sub.head()


print((df_train == np.inf).sum().sum())  # Anzahl der positiven inf-Werte
print((df_train == -np.inf).sum().sum()) # Anzahl der negativen inf-Werte


numerical_variables = ['winddirection', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall'


# Daten bereinigen
df_test['winddirection'] = df_test['winddirection'].fillna(df_test["winddirection"].mean())



import requests, json

def get_hongkong_weather_data(year):

    url = F'https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_{year}.xml'
    response = requests.get(url)
    data = json.loads(response.text)
    cols = [
        'Day', 'Mean Pressure (hPa)', 'Max Air Temp (deg. C)', 
        'Mean Air Temp (deg. C)', 'Min Air Temp (deg. C)', 'Mean Dew Point (deg. C)', 
        'Mean Relative Humidity (%)', 'Mean Amount of Cloud (%)', 'Total Rainfall (mm)',
        'Total Bright Sunshine (hours)', 'Prevailing Wind Direction (degrees)', 
        'Mean Wind Speed (km/h)'
    ]
    df = pd.json_normalize(
        data['stn']['data'],
        record_path='dayData',
        record_prefix='data.'
    ).apply(lambda x: x.str.strip())
    df.columns = cols
    df = df.loc[~df['Day'].isin(['Mean/Total', 'Normal'])] # removing monthly summary rows
    # adjusting dtypes to match the "original" dataset
    for c in df.columns:
        if c=='Day' or '%' in c:
            df[c] = df[c].astype(int)
        elif c=='Total Rainfall (mm)':
            pass
        else:
            df[c] = df[c].str.replace(r'[^\d.-]', '', regex=True) # removing footnote marks
            df[c] = pd.to_numeric(df[c], errors='coerce') # try to convert to float, or to nan if fails

    return df

original = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')

def adjust_to_original(df):
    df.columns = original.columns
    df['rainfall'] = np.where(df['rainfall']=='0.0', 'no', 'yes') # treating "Trace" as "yes"
    return df

cutoff = 31+29+31+30+31+30+31

df = adjust_to_original(get_hongkong_weather_data(2016))
for i in range(cutoff):
    for c in original.columns:
        x = original.iloc[i][c]
        y = df.iloc[i][c]
        if not ((pd.isna(x) and pd.isna(y)) or x == y):
            print(F'row {i}: original-{c}={x} real-{c}={y}')

df = adjust_to_original(get_hongkong_weather_data(2015))
for i in range(cutoff,366):
    for c in original.columns:
        x = original.iloc[i][c]
        y = df.iloc[i-1][c]
        if not ((pd.isna(x) and pd.isna(y)) or x == y):
            print(F'row {i}: original-{c}={x} real-{c}={y}')


# Plot heatmap to show correlation between features
plt.figure(figsize=(12, 8))
sns.heatmap(df_train.corr(), annot=True, cmap='viridis')
plt.show()


import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# KDE plot for Feature-Target Relationship
plt.figure(figsize=(14, 10))
for i, col in enumerate(numerical_variables, 1):
    plt.subplot(3, 4, i)
    sns.kdeplot(df_train[col][df_train['rainfall'] == 1], color='red', label='Rainfall: 1')
    sns.kdeplot(df_train[col][df_train['rainfall'] == 0], color='blue', label='Rainfall: 0')
    plt.title(f'Distribution of {col} by Rainfall')
    plt.legend()
plt.tight_layout()
plt.show()


# Daten nach Regen filtern  
regen_daten = df_train[df_train["rainfall"] > 0]  # Alle Daten mit Regen  
kein_regen_daten = df_train[df_train["rainfall"] == 0]  # Alle Daten ohne Regen  

# Erstelle eine Abbildung mit zwei Unterdiagrammen (Windrosen-Diagramme)  
fig, achsen = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))  

# Erstes Windrosen-Diagramm (mit Regen)  
achse1 = achsen[0]
achse1.set_theta_direction(-1)  # Drehrichtung gegen den Uhrzeigersinn  
achse1.set_theta_offset(np.pi / 2.0)  # Norden (0°) nach oben setzen  
balken1 = achse1.bar(
    np.deg2rad(regen_daten["winddirection"]),  # Windrichtung in Radiant umrechnen  
    regen_daten["windspeed"],  # Windgeschwindigkeit  
    width=np.pi/8,  # Balkenbreite  
    bottom=0.0,  
    color="b"  # Blaue Balken für Windgeschwindigkeit bei Regen  
)
achse1.set_title("Windgeschwindigkeit und -richtung mit Regen")  

# Zweites Windrosen-Diagramm (ohne Regen)  
achse2 = achsen[1]
achse2.set_theta_direction(-1)  
achse2.set_theta_offset(np.pi / 2.0)  
balken2 = achse2.bar(
    np.deg2rad(kein_regen_daten["winddirection"]),  
    kein_regen_daten["windspeed"],  
    width=np.pi/8,  
    bottom=0.0,  
    color="r"  # Rote Balken für Windgeschwindigkeit ohne Regen  
)
achse2.set_title("Windgeschwindigkeit und -richtung ohne Regen")  

plt.tight_layout()  # Layout anpassen  
plt.show()


# Wenn sowohl die Luftfeuchtigkeit als auch die Bewölkung hoch sind, ist es wahrscheinlicher, dass Niederschlag auftritt.
# Dieses Feature hilft, den Zusammenhang zu quantifizieren und so besser vorherzusagen, wann Regen wahrscheinlich ist.
#df_train["humidity_cloud_interaction"] = df_train["humidity"] * df_train["cloud"]
#df_test["humidity_cloud_interaction"] = df_test["humidity"] * df_test["cloud"]

# Diese Interaktion untersucht den Zusammenhang zwischen der Windgeschwindigkeit und der Luftfeuchtigkeit. Ein stärkerer Wind kann mehr Feuchtigkeit in die Atmosphäre transportieren.
#df_train["wind_humidity_interaction"] = df_train["windspeed"] * df_train["humidity"]
#df_test["wind_humidity_interaction"] = df_test["windspeed"] * df_test["humidity"]

# Dieses Feature misst die Wechselwirkung zwischen der Bewölkung und dem Sonnenschein.
# Ein höherer Anteil an Sonnenschein führt zu einer geringeren effektiven Bewölkung, was den Einfluss der Bewölkung auf den Niederschlag verringern könnte.
#df_train["cloud_sun_interaction"] = df_train["cloud"] * (1 - df_train["sunshine"] / 10)
#df_test["cloud_sun_interaction"] = df_test["cloud"] * (1 - df_test["sunshine"] / 10)

# Temperaturbereich (Max-Min) als Indikator für Wetterstabilität
# Ein großer Unterschied kann auf instabile Wetterbedingungen hinweisen.
#df_train["temp_range"] = df_train["maxtemp"] - df_train["mintemp"]
#df_test["temp_range"] = df_test["maxtemp"] - df_test["mintemp"]

# Taupunkt-Differenz zur Temperatur als Maß für Luftfeuchtigkeit
# Ein kleiner Unterschied deutet auf hohe Luftfeuchtigkeit hin, was Niederschlag begünstigen kann.
#df_train["dew_temp_diff"] = df_train["temparature"] - df_train["dewpoint"]
#df_test["dew_temp_diff"] = df_test["temparature"] - df_test["dewpoint"]

# Wechselwirkung zwischen Bewölkung und Taupunkt
# Hohe Bewölkung kombiniert mit einem hohen Taupunkt kann auf Regen hinweisen.
#df_train["cloud_dew_interaction"] = df_train["cloud"] * df_train["dewpoint"]
#df_test["cloud_dew_interaction"] = df_test["cloud"] * df_test["dewpoint"]

# Ein „dunkler Tag“ mit wenig Sonnenschein und starker Bewölkung deutet darauf hin, dass es wahrscheinlich regnen wird, da diese Bedingungen oft mit Niederschlägen verbunden sind.
#df_train["dark_day"] = ((df_train["sunshine"] < 2) & (df_train["cloud"] > 80)).astype(int)
#df_test["dark_day"] = ((df_test["sunshine"] < 2) & (df_test["cloud"] > 80)).astype(int)

# Ein niedriges Sättigungsdefizit kann auf eine hohe Wahrscheinlichkeit für Kondensation und damit für Niederschlag hindeuten. 
# Dieses Feature kann helfen, zu erkennen, wann die Bedingungen für Regen besonders günstig sind
#df_train["saturation_deficit"] = 100 - df_train["humidity"]
#df_test["saturation_deficit"] = 100 - df_test["humidity"]



# Entferne die Zielvariable 'rainfall' aus der Feature-Liste
#features = [col for col in df_train.columns if col != 'rainfall']

# Erstelle Interaktionsfeatures für jedes Paar von Features
#for col1, col2 in itertools.combinations(features, 2):
   # interaction_name = f"{col1}_{col2}_interaction"
   # df_train[interaction_name] = df_train[col1] * df_train[col2]
   # df_test[interaction_name] = df_test[col1] * df_test[col2]

#print("Variante 1: Interaktionen hinzugefügt, Originalspalten bleiben erhalten")


exclude_cols = ["rainfall", "id"]  # Liste der zu entfernenden Spalten
features = [col for col in df_train.columns if col not in exclude_cols]

# Erstelle Interaktionsfeatures für jedes Paar von Features
for col1, col2 in itertools.combinations(features, 2):
    interaction_name = f"{col1}_{col2}_interaction"
    df_train[interaction_name] = df_train[col1] * df_train[col2]
    df_test[interaction_name] = df_test[col1] * df_test[col2]



print("Variante 2: Interaktionen hinzugefügt, Originalspalten entfernt")


# Feature Engineering für df_train und df_test

# Temperaturdurchschnitt der letzten 3 Tage (ohne NaN für die ersten zwei Tage)
df_train['temperature_last3days'] = df_train['temparature'].rolling(window=3, min_periods=1).mean()
df_test['temperature_last3days'] = df_test['temparature'].rolling(window=3, min_periods=1).mean()

# Luftfeuchtigkeitsdurchschnitt der letzten 3 Tage (ohne NaN für die ersten zwei Tage)
df_train['humidity_last3days'] = df_train['humidity'].rolling(window=3, min_periods=1).mean()
df_test['humidity_last3days'] = df_test['humidity'].rolling(window=3, min_periods=1).mean()

# Luftdruckdurchschnitt der letzten 3 Tage (ohne NaN für die ersten zwei Tage)
df_train['pressure_last3days'] = df_train['pressure'].rolling(window=3, min_periods=1).mean()
df_test['pressure_last3days'] = df_test['pressure'].rolling(window=3, min_periods=1).mean()

# Windgeschwindigkeitsdurchschnitt der letzten 3 Tage (ohne NaN für die ersten zwei Tage)
df_train['windspeed_last3days'] = df_train['windspeed'].rolling(window=3, min_periods=1).mean()
df_test['windspeed_last3days'] = df_test['windspeed'].rolling(window=3, min_periods=1).mean()

# Bewölkungsdurchschnitt der letzten 3 Tage (ohne NaN für die ersten zwei Tage)
df_train['cloud_last3days'] = df_train['cloud'].rolling(window=3, min_periods=1).mean()
df_test['cloud_last3days'] = df_test['cloud'].rolling(window=3, min_periods=1).mean()

# Sonnenscheindurchschnitt der letzten 3 Tage (ohne NaN für die ersten zwei Tage)
df_train['sunshine_last3days'] = df_train['sunshine'].rolling(window=3, min_periods=1).mean()
df_test['sunshine_last3days'] = df_test['sunshine'].rolling(window=3, min_periods=1).mean()

# Temperaturdifferenz zum vorherigen Tag 
df_train["temp_diff"] = df_train["temparature"] - df_train["temparature"].shift(1)
df_test["temp_diff"] = df_test["temparature"] - df_test["temparature"].shift(1)
# Ersetze NaN in der ersten Zeile mit 0 oder einem anderen Wert
df_train["temp_diff"].iloc[0] = 0  # Oder df_train["temp_diff"].iloc[0] = df_train["temparature"].iloc[0]
df_test["temp_diff"].iloc[0] = 0  # Oder df_test["temp_diff"].iloc[0] = df_test["temparature"].iloc[0]

# Windgeschwindigkeitsänderung zum Vortag
# Plötzliche Windänderungen können auf Wetterumschwünge hindeuten.
df_train["wind_diff"] = df_train["windspeed"] - df_train["windspeed"].shift(1)
df_test["wind_diff"] = df_test["windspeed"] - df_test["windspeed"].shift(1)
# Ersetze NaN in der ersten Zeile mit 0 oder einem anderen Wert
df_train["wind_diff"].iloc[0] = 0  
df_test["wind_diff"].iloc[0] = 0  

# Luftdruckänderung zum Vortag
# Ein abrupter Abfall im Luftdruck ist oft mit bevorstehendem Regen verbunden.
df_train["pressure_diff"] = df_train["pressure"] - df_train["pressure"].shift(1)
df_test["pressure_diff"] = df_test["pressure"] - df_test["pressure"].shift(1)
# Ersetze NaN in der ersten Zeile mit 0 oder einem anderen Wert
df_train["pressure_diff"].iloc[0] = 0  
df_test["pressure_diff"].iloc[0] = 0  

# Gleitender Durchschnitt für den Taupunkt über 3 Tage zur Erkennung von Trends
df_train["dewpoint_last3days"] = df_train["dewpoint"].rolling(window=3, min_periods=1).mean()
df_test["dewpoint_last3days"] = df_test["dewpoint"].rolling(window=3, min_periods=1).mean()
# Ersetze NaN-Werte der ersten zwei Tage mit dem vorhandenen Wert
df_train["dewpoint_last3days"].fillna(df_train["dewpoint"], inplace=True)
df_test["dewpoint_last3days"].fillna(df_test["dewpoint"], inplace=True)







# Die Umrechnung der Windrichtung in x- und y-Komponenten (unter Verwendung der Cosinus- und Sinusfunktionen) hilft dabei,
# die Windrichtung in ein numerisches Format zu bringen, das leichter zu interpretieren und in mathematischen Modellen zu verwenden ist.
df_train["wind_x"] = np.cos(np.deg2rad(df_train["winddirection"]))
df_train["wind_y"] = np.sin(np.deg2rad(df_train["winddirection"]))
df_test["wind_x"] = np.cos(np.deg2rad(df_test["winddirection"]))
df_test["wind_y"] = np.sin(np.deg2rad(df_test["winddirection"]))

# Dieses Feature klassifiziert den Tag des Jahres in eine Jahreszeit (Frühling, Sommer, Herbst, Winter) basierend auf dem Tag des Jahres.
def classify_season(day):
    if 1 <= day <= 90:
        return 'Spring'
    elif 91 <= day <= 181:
        return 'Summer'
    elif 182 <= day <= 273:
        return 'Fall'
    else:
        return 'Winter'

# Feature für df_train
df_train['season'] = df_train['day'].apply(classify_season)

# Feature für df_test
df_test['season'] = df_test['day'].apply(classify_season)

# One-Hot Encoding für df_train
df_train = pd.get_dummies(df_train, columns=['season'], drop_first=True)

# One-Hot Encoding für df_test
df_test = pd.get_dummies(df_test, columns=['season'], drop_first=True)



# Entferne die Originalspalten
df_train.drop(columns=features, inplace=True)
df_test.drop(columns=features, inplace=True)


from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

# Daten vorbereiten
X = df_train.drop(columns=[target_variable, 'id'])
y = df_train[target_variable]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import catboost as cb

# Splitte die Daten in X (Features) und y (Zielvariable)
X = df_train.drop(columns=[target_variable, 'id'])
y = df_train[target_variable]

# Modelle und Parameter für GridSearchCV
param_grid_catboost = {
    'depth': [4, 6],
    'learning_rate': [0.01, 0.1],
    'iterations': [100, 300]
}

param_grid_logreg = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_and_select_features(model, param_grid, model_name):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        feature_importances = np.abs(best_model.coef_[0])
    else:
        feature_importances = None
    
    if feature_importances is not None:
        sorted_idx = np.argsort(feature_importances)[::-1]
        sorted_features = X.columns[sorted_idx]
        
        best_score = grid_search.best_score_
        best_n = len(X.columns)
        
        for n in range(2, len(X.columns), 2):
            top_n_features = sorted_features[:n]
            X_top_n = X_scaled[:, sorted_idx[:n]]
            
            scores = []
            for train_idx, val_idx in kf.split(X_top_n, y):
                best_model.fit(X_top_n[train_idx], y.iloc[train_idx])
                y_pred_prob = best_model.predict_proba(X_top_n[val_idx])[:, 1]
                scores.append(roc_auc_score(y.iloc[val_idx], y_pred_prob))
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_n = n
        
        print(f"Best number of features for {model_name}: {best_n}")
        top_features = sorted_features[:best_n]

        # **Wichtig: Das Modell wird neu mit den Top-N-Features trainiert**
        X_train_final = X[top_features]
        scaler_final = StandardScaler().fit(X_train_final)
        X_train_final_scaled = scaler_final.transform(X_train_final)
        
        best_model.fit(X_train_final_scaled, y)
        
        return best_model, top_features, scaler_final

    return best_model, X.columns, None

best_catboost, top_features_catboost, scaler_catboost = train_and_select_features(
    cb.CatBoostClassifier(random_state=42, verbose=0), param_grid_catboost, "CatBoost"
)

best_logreg, top_features_logreg, scaler_logreg = train_and_select_features(
    LogisticRegression(random_state=42), param_grid_logreg, "Logistic Regression"
)

def create_submission(best_model, top_features, scaler, model_name):
    X_test_scaled = scaler.transform(df_test[top_features])
    y_test_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    submission = pd.DataFrame({'id': df_test['id'], 'rainfall': y_test_pred_prob})
    submission.to_csv(f'submission_{model_name}.csv', index=False)
    print(f"Submission für {model_name} wurde als 'submission_{model_name}.csv' gespeichert.")

create_submission(best_catboost, top_features_catboost, scaler_catboost, "CatBoost")
create_submission(best_logreg, top_features_logreg, scaler_logreg, "LogisticRegression")



df_train.shape







