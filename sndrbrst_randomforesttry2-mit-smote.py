#import and general settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# settings so that pandas shows all columns and rows, because per default, it 
# shows only some if the dataset is large
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df = pd.read_csv('../input/child-mind-institute-problematic-internet-use/train.csv')
test_data = pd.read_csv('../input/child-mind-institute-problematic-internet-use/test.csv')

# drop the PCAT-columns from the dataset, because the target value (sii) is calculated
# with them and in the test dataset, we won't have them to predict sii

columns_to_drop = [
    'PCIAT-Season',
    'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03', 
    'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 
    'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 
    'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12',
    'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15', 
    'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 
    'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total'
]
df = df.drop(columns=columns_to_drop)


# merge the columns PAQ_A-PAQ_A_Total and PAQ_C-PAQ_C_Total into one column called PAQ_Total
def calculate_paq_total(row):
    a = row['PAQ_A-PAQ_A_Total']
    c = row['PAQ_C-PAQ_C_Total']
    
    if pd.notnull(a) and pd.notnull(c): 
        return (a + c) / 2 # if both values are there, calculate average
    elif pd.notnull(a):
        return a
    elif pd.notnull(c):
        return c
    else: 
        return None

# create a new column and fill it with help of the created function
df['PAQ_Total'] = df.apply(calculate_paq_total, axis=1)
# drop the old two columns
df = df.drop(columns=['PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total'])


# merge the columns Fitness_Endurance-Time_Mins and Fitness_Endurance-Time_Sec, because
# seconds alone have no meaning

# create new column: Fitness_Endurance-Time
df['Fitness_Endurance-Time'] = df['Fitness_Endurance-Time_Mins'] + (df['Fitness_Endurance-Time_Sec'] / 60)

# drop old columns
df = df.drop(columns=['Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec', 'BIA-Season', 'Fitness_Endurance-Season', 'PAQ_A-Season'])


categorical_columns_expressedInNumbers = ['Basic_Demos-Sex', 'PreInt_EduHx-computerinternet_hoursday', 'FGC-FGC_TL_Zone', 'FGC-FGC_CU_Zone', 
                               'FGC-FGC_PU_Zone', 'FGC-FGC_SRR_Zone', 'FGC-FGC_SRL_Zone', 'BIA-BIA_Activity_Level_num', 
                              'BIA-BIA_Frame_num', 'FGC-FGC_GSD_Zone','FGC-FGC_GSND_Zone']

# Konvertiere die Spalten in den Typ 'category'
for col in categorical_columns_expressedInNumbers:
    df[col] = df[col].astype('category')

categorical_columns_expressedInText = df.select_dtypes(include="object").columns

numerical_discrete_columns = ['Basic_Demos-Age', 'CGAS-CGAS_Score', 'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                     'Fitness_Endurance-Max_Stage', 'FGC-FGC_CU', 'FGC-FGC_PU',
                                'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T']
numerical_continuous_columns = ['Physical-BMI', 'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
                       'FGC-FGC_GSND', 'FGC-FGC_GSD', 'FGC-FGC_SRL', 'FGC-FGC_SRR', 'FGC-FGC_TL',
                       'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW',
                       'BIA-BIA_FFM', 'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_ICW', 'BIA-BIA_LDM',
                       'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW', 'Fitness_Endurance-Time',
                       'PAQ_Total']

categorical_columns_expressedInText = list(categorical_columns_expressedInText)
categorical_columns_expressedInNumbers = list(categorical_columns_expressedInNumbers)

if "id" in categorical_columns_expressedInText:
    categorical_columns_expressedInText.remove("id")

# Combine categorical columns
all_categorical_columns = categorical_columns_expressedInNumbers + categorical_columns_expressedInText

numerical_discrete_columns = list(numerical_discrete_columns)
numerical_continuous_columns = list(numerical_continuous_columns)

all_numerical_columns = numerical_discrete_columns + numerical_continuous_columns


def whisker(col):
    q1,q3=np.percentile(col, [25, 75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw, uw


def replace_negatives_with_median(df, columns):
    for col in columns:
        median_value = df[col].median() 
        df[col] = np.where(df[col] <= 0, median_value, df[col])
    return df


# def replace_negatives_with_lower_whisker(df, columns):
#     for col in columns:
#         lw = whisker(df[col])[0] 
#         df[col] = np.where(df[col] <= 0, lw, df[col])
#     return df


columns_with_negative_values = ['Physical-BMI', 'Physical-Weight', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
                               'BIA-BIA_BMR', 'BIA-BIA_FMI', 'BIA-BIA_Fat']
df = replace_negatives_with_median(df, columns_with_negative_values)


def dropRows(df):
    df = df.drop(df[df['BIA-BIA_FFM'] > 6000].index)
    df = df.drop(df[df['BIA-BIA_BMC'] > 100].index)
    #reset index if you want to remain it consecutive
    df = df.reset_index(drop=True)
    return df

df = dropRows(df)


# Missing Value Treatment Attempt 1: Impute with median and mode
df_i_w_m_a_m = df.copy()
# Für numerische Spalten
for i in all_numerical_columns:
    df_i_w_m_a_m[i] = df_i_w_m_a_m[i].fillna(df_i_w_m_a_m[i].median()) 

# Für kategorische Spalten, Werte mit dem Modus auffüllen
for i in all_categorical_columns:
    df_i_w_m_a_m[i] = df_i_w_m_a_m[i].fillna(df_i_w_m_a_m[i].mode()[0])

# Check if values were filled
df_i_w_m_a_m.info()


df_i_w_m_a_m = pd.get_dummies(df_i_w_m_a_m, columns=['Basic_Demos-Sex', 'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'FGC-Season','PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season'], drop_first=True)
df_i_w_m_a_m.head()


from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd

# Gelabelte Daten (sii ist bekannt)
labeled_data = df_i_w_m_a_m[df_i_w_m_a_m['sii'].notna()]

# Unlabelte Daten (sii ist unbekannt)
unlabeled_data = df_i_w_m_a_m[df_i_w_m_a_m['sii'].isna()]

# Gelabelte Daten
ids_labeled = labeled_data['id']  # IDs der gelabelten Daten
X_labeled = labeled_data.drop(['sii', 'id'], axis=1)  # Features ohne 'id' und 'sii'
y_labeled = labeled_data['sii']  # Zielvariable

# Ungelabelte Daten
ids_unlabeled = unlabeled_data['id']  # IDs der ungelabelten Daten
X_unlabeled = unlabeled_data.drop(['sii', 'id'], axis=1)  # Features ohne 'id'
y_unlabeled = pd.Series([-1] * len(X_unlabeled), index=X_unlabeled.index)  # Zielvariable als -1

# NaN-Werte im unlabeled Datensatz überprüfen
print("NaN-Werte in X_unlabeled vor Behandlung:", X_unlabeled.isnull().sum().sum())

# Gelabelte Daten indexieren
X_labeled = X_labeled.reset_index(drop=True)
y_labeled = y_labeled.reset_index(drop=True)

# Aufteilen der gelabelten Daten in Training und Test
X_train, X_test, y_train, y_test = train_test_split(
    X_labeled, y_labeled, test_size=0.2, random_state=42, shuffle=True
)

# SMOTE auf die Trainingsdaten anwenden
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# NaN-Werte nach SMOTE prüfen
print("NaN-Werte in X_train_resampled:", pd.DataFrame(X_train_resampled).isnull().sum().sum())

# Kombinieren der resampleten Trainingsdaten mit den ungelabelten Daten
X_combined = pd.concat([pd.DataFrame(X_train_resampled), X_unlabeled])
y_combined = pd.concat([pd.Series(y_train_resampled), y_unlabeled])

# NaN-Werte durch Imputation ersetzen
imputer = SimpleImputer(strategy="mean")  # Alternativen: "median", "most_frequent"
X_combined = pd.DataFrame(imputer.fit_transform(X_combined), columns=X_combined.columns)

# NaN-Werte erneut prüfen
print("NaN-Werte in X_combined nach Imputation:", X_combined.isnull().sum().sum())

# Basis-Klassifikator
base_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# SelfTrainingClassifier
self_training_model = SelfTrainingClassifier(base_model)

# Training des Modells
self_training_model.fit(X_combined, y_combined)

# Initialisiere eine Variable, um die Anzahl der neuen hochsicheren Labels zu verfolgen
new_high_conf_count = 1  # Startwert > 0, um die Schleife zu starten

# Schleife: Wiederholen, solange neue hochsichere Labels gefunden werden
while new_high_conf_count > 0:
    # Vorhersagen und Wahrscheinlichkeiten für ungelabelte Daten
    predictions = self_training_model.predict(X_unlabeled)
    probabilities = self_training_model.predict_proba(X_unlabeled)

    # Sicherheitsschwelle festlegen (z. B. 0.9)
    confidence_threshold = 0.9
    high_confidence_indices = (probabilities.max(axis=1) >= confidence_threshold)

    # Hochsichere Vorhersagen und ihre Features extrahieren
    X_high_conf = X_unlabeled[high_confidence_indices].dropna()  # Entfernen von NaN-Werten
    y_high_conf = predictions[high_confidence_indices]

    # Zähle die Anzahl der neuen hochsicheren Labels
    new_high_conf_count = len(X_high_conf)
    print(f"Neue hochsichere Vorhersagen in dieser Iteration: {new_high_conf_count}")
    print(f"Anzahl der nicht hoch sicheren Vorhersagen: {len(probabilities) - new_high_conf_count}")

    # Wenn keine neuen hochsicheren Labels mehr vorhanden sind, abbrechen
    if new_high_conf_count == 0:
        break

    # Hochsichere Daten zu gelabelten Daten hinzufügen
    X_train_resampled = pd.concat([pd.DataFrame(X_train_resampled), X_high_conf])
    y_train_resampled = pd.concat([pd.Series(y_train_resampled), pd.Series(y_high_conf, index=X_high_conf.index)])

    # Entfernen der hochsicheren Daten aus den ungelabelten Daten
    X_unlabeled = X_unlabeled.drop(index=X_high_conf.index).dropna()

    # Zielvariable für verbleibende ungelabelte Daten aktualisieren (-1 bleibt für diese erhalten)
    y_unlabeled = pd.Series([-1] * len(X_unlabeled), index=X_unlabeled.index)

    # Kombinieren der gelabelten und verbleibenden ungelabelten Daten
    X_combined = pd.concat([pd.DataFrame(X_train_resampled), X_unlabeled])

    # NaN-Werte in X_combined entfernen
    imputer = SimpleImputer(strategy="mean")
    X_combined = pd.DataFrame(imputer.fit_transform(X_combined), columns=X_combined.columns)

    # Zielvariable anpassen
    y_combined = pd.concat([pd.Series(y_train_resampled), y_unlabeled])

    # Modell erneut trainieren
    self_training_model.fit(X_combined, y_combined)


print("Training abgeschlossen. Keine weiteren hochsicheren Vorhersagen verfügbar.")

# Vorhersagen auf den Testdaten
y_pred = self_training_model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))






from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split

# Gelabelte Daten (sii ist bekannt)
labeled_data = df_i_w_m_a_m[df_i_w_m_a_m['sii'].notna()]

# Unlabelte Daten (sii ist unbekannt)
unlabeled_data = df_i_w_m_a_m[df_i_w_m_a_m['sii'].isna()]

# Gelabelte Daten
ids_labeled = labeled_data['id']  # IDs der gelabelten Daten
X_labeled = labeled_data.drop(['sii', 'id'], axis=1)  # Features ohne 'id' und 'sii'
y_labeled = labeled_data['sii']  # Zielvariable



# Ungelabelte Daten
ids_unlabeled = unlabeled_data['id']  # IDs der ungelabelten Daten
X_unlabeled = unlabeled_data.drop(['sii', 'id'], axis=1)  # Features ohne 'id'
y_unlabeled = pd.Series([-1] * len(X_unlabeled), index=X_unlabeled.index)  # Zielvariable als -1

X_labeled = X_labeled.reset_index(drop=True)
y_labeled = y_labeled.reset_index(drop=True)

# Aufteilen der gelabelten Daten in Training und Test
X_train, X_test, y_train, y_test = train_test_split(
    X_labeled, y_labeled, test_size=0.2, random_state=42, shuffle=True
)

# Kombinieren der gelabelten und ungelabelten Daten
X_combined = pd.concat([X_train, X_unlabeled])
y_combined = pd.concat([y_train, y_unlabeled])

print("Gibt es Überschneidungen zwischen Training und Test?")
print(X_train.index.intersection(X_test.index))  # Sollte leer sein: Index([])





# Basis-Klassifikator
base_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# SelfTrainingClassifier
self_training_model = SelfTrainingClassifier(base_model)



# Training des Modells
self_training_model.fit(X_combined, y_combined)


# Initialisiere eine Variable, um die Anzahl der neuen hochsicheren Labels zu verfolgen
new_high_conf_count = 1  # Startwert > 0, um die Schleife zu starten

# Schleife: Wiederholen, solange neue hochsichere Labels gefunden werden
while new_high_conf_count > 0:
    # Vorhersagen und Wahrscheinlichkeiten für ungelabelte Daten
    predictions = self_training_model.predict(X_unlabeled)
    probabilities = self_training_model.predict_proba(X_unlabeled)

    # Sicherheitsschwelle festlegen (z. B. 0.9)
    confidence_threshold = 0.9
    high_confidence_indices = (probabilities.max(axis=1) >= confidence_threshold)

    # Hochsichere Vorhersagen und ihre Features extrahieren
    X_high_conf = X_unlabeled[high_confidence_indices]
    y_high_conf = predictions[high_confidence_indices]

    # Zähle die Anzahl der neuen hochsicheren Labels
    new_high_conf_count = len(X_high_conf)
    print(f"Neue hochsichere Vorhersagen in dieser Iteration: {new_high_conf_count}")
    print(f"Anzahl der nicht hoch sicheren Vorhersagen: {len(probabilities)-new_high_conf_count}")

    # Wenn keine neuen hochsicheren Labels mehr vorhanden sind, abbrechen
    if new_high_conf_count == 0:
        break

    # Hochsichere Daten zu gelabelten Daten hinzufügen
    X_labeled = pd.concat([X_train, X_high_conf])
    y_labeled = pd.concat([y_train, pd.Series(y_high_conf, index=X_high_conf.index)])

    # Entfernen der hochsicheren Daten aus den ungelabelten Daten
    X_unlabeled = X_unlabeled.drop(index=X_high_conf.index)

    # Zielvariable für verbleibende ungelabelte Daten aktualisieren (-1 bleibt für diese erhalten)
    y_unlabeled = pd.Series([-1] * len(X_unlabeled), index=X_unlabeled.index)

    # Kombinieren der gelabelten und verbleibenden ungelabelten Daten
    X_combined = pd.concat([X_labeled, X_unlabeled])
    y_combined = pd.concat([y_labeled, y_unlabeled])

    # Modell erneut trainieren
    self_training_model.fit(X_combined, y_combined)

print("Training abgeschlossen. Keine weiteren hochsicheren Vorhersagen verfügbar.")



from sklearn.metrics import confusion_matrix, classification_report

# Vorhersagen auf den Testdaten
y_pred = self_training_model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# # Vorhersagen und Wahrscheinlichkeiten für ungelabelte Daten
# predictions = self_training_model.predict(X_unlabeled)
# probabilities = self_training_model.predict_proba(X_unlabeled)

# # Sicherheitsschwelle festlegen (z. B. 0.9)
# confidence_threshold = 0.9
# high_confidence_indices = (probabilities.max(axis=1) >= confidence_threshold)
# num_high_confidence = high_confidence_indices.sum()
# print(f"Anzahl der hochsicheren Vorhersagen: {num_high_confidence}")
# print(f"Anzahl der nicht hoch sicheren Vorhersagen: {len(probabilities)}")

# # Hochsichere Vorhersagen und ihre Features extrahieren
# X_high_conf = X_unlabeled[high_confidence_indices]
# y_high_conf = predictions[high_confidence_indices]



# # Hochsichere Daten zu gelabelten Daten hinzufügen
# X_labeled = pd.concat([X_labeled, X_high_conf])
# y_labeled = pd.concat([y_labeled, pd.Series(y_high_conf, index=X_high_conf.index)])

# # Entfernen der hochsicheren Daten aus den ungelabelten Daten
# X_unlabeled = X_unlabeled.drop(index=X_high_conf.index)

# # Zielvariable für verbleibende ungelabelte Daten aktualisieren (-1 bleibt für diese erhalten)
# y_unlabeled = pd.Series([-1] * len(X_unlabeled), index=X_unlabeled.index)

# # Kombinieren der gelabelten und verbleibenden ungelabelten Daten
# X_combined = pd.concat([X_labeled, X_unlabeled])
# y_combined = pd.concat([y_labeled, y_unlabeled])

# # Modell erneut trainieren
# self_training_model.fit(X_combined, y_combined)



# IDs der Testdaten extrahieren
ids_test = test_data['id']

# Testdaten vorbereiten (entfernen der 'id'-Spalte)
X_test = test_data.drop(['id'], axis=1)

# do necessary transformations, e.g. one-hot encoding

X_test['PAQ_Total'] = X_test.apply(calculate_paq_total, axis=1)
# drop the old two columns
X_test = X_test.drop(columns=['PAQ_A-PAQ_A_Total', 'PAQ_C-PAQ_C_Total'])

X_test['Fitness_Endurance-Time'] = X_test['Fitness_Endurance-Time_Mins'] + (X_test['Fitness_Endurance-Time_Sec'] / 60)

# drop old columns

X_test = X_test.drop(columns=['Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec', 'BIA-Season', 'Fitness_Endurance-Season', 'PAQ_A-Season'])


# Missing Value Treatment Attempt 1: Impute with median and mode
X_test = X_test.copy()
# Für numerische Spalten
for i in all_numerical_columns:
    X_test[i] = X_test[i].fillna(X_test[i].median()) 

# Für kategorische Spalten, Werte mit dem Modus auffüllen
for i in all_categorical_columns:
    X_test[i] = X_test[i].fillna(X_test[i].mode()[0])


X_test = pd.get_dummies(X_test, columns=['Basic_Demos-Sex', 'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'FGC-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season'], drop_first=True)
X_test.head()





# Vorhersagen für die Testdaten
final_predictions = self_training_model.predict(X_test)



# Ergebnisse in einem DataFrame speichern
submission = pd.DataFrame({
    'id': ids_test,  # IDs der Testdaten
    'sii': final_predictions  # Vorhergesagte sii-Werte
})

# Konvertieren der sii-Werte in ganze Zahlen
submission['sii'] = submission['sii'].astype(int)

# Ergebnisse als CSV speichern
submission.to_csv('submission.csv', index=False)



import os

# Absoluten Pfad der Datei anzeigen
print(os.path.abspath('submission.csv'))


