

# Standard libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.lines import Line2D

# Scipy & stats
from scipy.stats import chi2_contingency, mode
from scipy.optimize import linear_sum_assignment

# Sklearn - preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin

# Sklearn - models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

# Sklearn - feature selection
from sklearn.feature_selection import SelectFromModel, RFECV



# Sklearn - metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    cohen_kappa_score, roc_curve, auc, make_scorer
)

# Visualization helpers
from kneed import KneeLocator
import umap

# Sklearn - pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN



def load_training_data():
    return pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')

def load_test_data():
    return pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')



training_data = load_training_data()
training_data.describe()



# voglio filtrare per range ma tenere i nan

condition = (training_data['CGAS-CGAS_Score'] >= 0) & (training_data['CGAS-CGAS_Score'] <= 100)
training_data = training_data[condition | training_data['CGAS-CGAS_Score'].isna()]

condition = training_data['Physical-Weight'] > 0
training_data = training_data[condition | training_data['Physical-Weight'].isna()]

# Remove outliers for 'Fitness_Endurance-Max_Stage' but keep NaN
condition = training_data['Fitness_Endurance-Max_Stage'] > 0
training_data = training_data[condition | training_data['Fitness_Endurance-Max_Stage'].isna()]

# Remove outliers for 'Fitness_Endurance-Time_Mins' but keep NaN
condition = training_data['Fitness_Endurance-Time_Mins'] > 0
training_data = training_data[condition | training_data['Fitness_Endurance-Time_Mins'].isna()]

training_data["BIA-BIA_BMC"] = np.where(training_data["BIA-BIA_BMC"] <= 0, np.nan, training_data["BIA-BIA_BMC"])
training_data["BIA-BIA_BMC"] = np.where(training_data["BIA-BIA_BMC"] > 10, np.nan, training_data["BIA-BIA_BMC"])
# Remove outliers for 'BIA-BIA_DEE' but keep NaN
condition = training_data['BIA-BIA_DEE'] < 17311
training_data = training_data[condition | training_data['BIA-BIA_DEE'].isna()]
# Remove highly implausible values

# Remove implausible body-fat
training_data["BIA-BIA_Fat"] = np.where(training_data["BIA-BIA_Fat"] < 5, np.nan, training_data["BIA-BIA_Fat"])
training_data["BIA-BIA_Fat"] = np.where(training_data["BIA-BIA_Fat"] > 60, np.nan, training_data["BIA-BIA_Fat"])
# Basal Metabolic Rate
training_data["BIA-BIA_BMR"] = np.where(training_data["BIA-BIA_BMR"] > 4000, np.nan, training_data["BIA-BIA_BMR"])
# Daily Energy Expenditure
training_data["BIA-BIA_DEE"] = np.where(training_data["BIA-BIA_DEE"] > 8000, np.nan, training_data["BIA-BIA_DEE"])
# Fat Free Mass Index
training_data["BIA-BIA_FFM"] = np.where(training_data["BIA-BIA_FFM"] <= 0, np.nan, training_data["BIA-BIA_FFM"])
training_data["BIA-BIA_FFM"] = np.where(training_data["BIA-BIA_FFM"] > 300, np.nan, training_data["BIA-BIA_FFM"])
# Fat Mass Index
training_data["BIA-BIA_FMI"] = np.where(training_data["BIA-BIA_FMI"] < 0, np.nan, training_data["BIA-BIA_FMI"])
# Extra Cellular Water
training_data["BIA-BIA_ECW"] = np.where(training_data["BIA-BIA_ECW"] > 100, np.nan, training_data["BIA-BIA_ECW"])
# Intra Cellular Water
training_data["BIA-BIA_ICW"] = np.where(training_data["BIA-BIA_ICW"] > 100, np.nan, training_data["BIA-BIA_ICW"])
# Lean Dry Mass
training_data["BIA-BIA_LDM"] = np.where(training_data["BIA-BIA_LDM"] > 100, np.nan, training_data["BIA-BIA_LDM"])
# Lean Soft Tissue
training_data["BIA-BIA_LST"] = np.where(training_data["BIA-BIA_LST"] > 300, np.nan, training_data["BIA-BIA_LST"])
# Skeletal Muscle Mass
training_data["BIA-BIA_SMM"] = np.where(training_data["BIA-BIA_SMM"] > 300, np.nan, training_data["BIA-BIA_SMM"])
# Total Body Water
training_data["BIA-BIA_TBW"] = np.where(training_data["BIA-BIA_TBW"] > 300, np.nan, training_data["BIA-BIA_TBW"])



# Conteggio delle righe totali
totale_righe = len(training_data)

# Conteggio delle righe con almeno un valore nullo
righe_nulle = training_data.isnull().any(axis=1).sum()

# Conteggio delle righe completamente non nulle
righe_non_nulle = totale_righe - righe_nulle

# Creazione dell'istogramma
plt.bar(["Con almeno un valore nullo", "Non Nulle", "Totali"], [righe_nulle, righe_non_nulle, totale_righe], color=['red', 'green', 'blue'])
plt.xlabel("Dataset")
plt.ylabel("Conteggio")
plt.title("Conteggio di Righe Nulle, Non Nulle e Totali")
plt.show()


null_counts = training_data.isnull().sum()
print(" \nCount total NaN at each column in a DataFrame : \n\n",null_counts)
print(" \nRate total NaN at each column in a DataFrame : \n\n",null_counts / len(training_data))
plt.figure(figsize=(18, 6))
plt.bar(null_counts.index, null_counts.values/len(training_data), color='teal')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Null Value Count', fontsize=12)
plt.title('Count of Null Values by Feature', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.tight_layout()
plt.show()


totale_sii_missing = training_data['sii'].isna().sum()
totale_missing_target = training_data[['sii', 'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04',
                                       'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08',
                                       'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12',
                                       'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16',
                                       'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20','PCIAT-PCIAT_Total']].isna().all(axis=1).sum()
# Creazione dell'istogramma
plt.bar(["Sii NAN", "Sii+ colonne dipendenti"], [totale_sii_missing, totale_missing_target], color=['red', 'green'])
plt.xlabel("Dataset")
plt.title("Conteggio di Righe Sii Nulli, Sii + risposte alle domande")
plt.ylabel("Conteggio")

plt.show()



# recupero tutti i sii missing
# conto quante risposte sono missing

# Conta i valori mancanti per ciascuna colonna

answ_columns = [
    'sii',
    'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04',
    'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08',
    'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12',
    'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16',
    'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20',
    'PCIAT-PCIAT_Total'
]


missing_counts = training_data[answ_columns].isna().sum()

plt.figure(figsize=(12,6))
missing_counts.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title("Valori mancanti per colonna", fontsize=14)
plt.xlabel("Colonne", fontsize=12)
plt.ylabel("Numero di missing", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()




training_data = training_data[training_data['sii'].notna()]


training_data['PAQ_TOTAL'] = training_data['PAQ_A-PAQ_A_Total'].fillna(training_data['PAQ_C-PAQ_C_Total'])
training_data.drop(inplace=True,axis=1,columns="PAQ_A-PAQ_A_Total")
training_data.drop(inplace=True,axis=1,columns="PAQ_C-PAQ_C_Total")


thresh = 0.49 * len(training_data)
training_data.dropna(thresh = thresh, axis = 1, inplace = True)
null_counts = training_data.isnull().sum()
print(" \nCount total NaN at each column in a DataFrame : \n\n",null_counts)
print(" \nRate total NaN at each column in a DataFrame : \n\n",null_counts / len(training_data))
plt.figure(figsize=(18, 6))
plt.bar(null_counts.index, null_counts.values/len(training_data), color='teal')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Null Value Count', fontsize=12)
plt.title('Count of Null Values by Feature', fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.tight_layout()
plt.show()


num_cols = len(training_data.columns)

# Definiamo le soglie da analizzare (dal 10% al 100%)
soglie = np.arange(0.1, 1.1, 0.1)

# Lista per salvare il numero di righe per ogni soglia
righe_per_soglia = []

# Calcola il numero di righe che superano ogni soglia di NaN
for soglia in soglie:
    num_nan = int(soglia * num_cols)
    righe_con_nan = (training_data.isna().sum(axis=1) >= num_nan).sum()
    righe_per_soglia.append(righe_con_nan)

plt.figure(figsize=(10, 5))
plt.bar([f"{int(s*100)}%" for s in soglie], righe_per_soglia, color="red")

# Aggiunta etichette e titolo
plt.xlabel("Percentuale di valori NaN nella riga")
plt.ylabel("Conteggio delle righe")
plt.title("Distribuzione del numero di righe per proporzione di valori mancanti")
plt.xticks(rotation=45)
plt.show()


training_data.drop(columns=['Physical-Height','Physical-Weight'],inplace=True,axis=1,errors='ignore')

def select_bmi(row):
    if pd.notna(row['BIA-BIA_BMI']):
        return row['BIA-BIA_BMI']
    else:
        return row['Physical-BMI']

training_data['BMI'] = training_data.apply(select_bmi, axis=1)
training_data.drop(columns=['BIA-BIA_BMI','Physical-BMI'],inplace=True,axis=1,errors='ignore')


target_columns = ['PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total','sii']

categorical_columns = ['Basic_Demos-Enroll_Season',
                       'CGAS-Season',
                       'Physical-Season',
                       'Fitness_Endurance-Season',
                       'FGC-Season',
                       'BIA-Season',
                       'PAQ_A-Season',
                       'PAQ_C-Season',
                       'PCIAT-Season',
                       'PreInt_EduHx-Season',
                       'Basic_Demos-Enroll_Season_Summer',
                       'Basic_Demos-Enroll_Season_Winter', 'CGAS-Season_Spring',
                       'CGAS-Season_Summer', 'CGAS-Season_Winter', 'Physical-Season_Spring',
                       'Physical-Season_Summer', 'Physical-Season_Winter',
                       'Fitness_Endurance-Season_Spring', 'Fitness_Endurance-Season_Summer',
                       'Fitness_Endurance-Season_Winter', 'FGC-Season_Spring',
                       'FGC-Season_Summer', 'FGC-Season_Winter', 'BIA-Season_Spring',
                       'BIA-Season_Summer', 'BIA-Season_Winter', 'PAQ_A-Season_Spring',
                       'PAQ_A-Season_Summer', 'PAQ_A-Season_Winter', 'PAQ_C-Season_Spring',
                       'PAQ_C-Season_Summer', 'PAQ_C-Season_Winter', 'PCIAT-Season_Spring',
                       'PCIAT-Season_Summer', 'PCIAT-Season_Winter','Basic_Demos-Enroll_Season_Spring',
                       'SDS-Season','PreInt_EduHx-Season_Spring', 'PreInt_EduHx-Season_Summer',
                       'PreInt_EduHx-Season_Winter', 'SDS-Season_Spring', 'SDS-Season_Summer',
                       'SDS-Season_Winter','Basic_Demos-Enroll_Season_Spring','Basic_Demos-Sex','FGC-FGC_PU_Zone','FGC-FGC_SRL_Zone','FGC-FGC_CU_Zone','sii','PAQ_C-Season_Fall' ,'FGC-FGC_TL_Zone' , 'FGC-FGC_SRR_Zone']


def split_X_Y(dataset: pd.DataFrame, target_column: str, test_size: float = 0.25, random_state: int = 42):
    X = dataset.drop(columns=[target_column])
    Y = dataset[target_column]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y
    )

    return X_train, X_test, Y_train, Y_test


X_train,X_test,Y_train,Y_test = split_X_Y(training_data,target_column="sii")
# remove id from train

X_train = X_train.drop(columns="id")
X_test = X_test.drop(columns="id")



plt.hist(training_data['sii'])
plt.title("Sii Distribuition")


season_columns = [col for col in X_train.columns if "Season" in col in col]

# Inizializza lista per salvare i risultati
results = []

# Calcola chi-quadro per ciascuna variabile stagionale
for col in season_columns:
    try:
        table = pd.crosstab(X_train[col], Y_train)
        chi2, p, _, _ = chi2_contingency(table)
        results.append({'column': col, 'p_value': p})
    except Exception as e:
        results.append({'column': col, 'p_value': None, 'error': str(e)})

# Crea il DataFrame dei risultati
chi2_df = pd.DataFrame(results).sort_values(by='p_value')
chi2_df



X_train_encoded = pd.get_dummies(X_train, columns=['PAQ_C-Season'])
X_test_encoded = pd.get_dummies(X_test, columns=['PAQ_C-Season'])
#rimuovo tutte le altre

X_train_encoded = X_train_encoded.drop(columns=[col["column"] for col in results if col["p_value"]> 0.02])
X_test_encoded = X_test_encoded.drop(columns=[col["column"] for col in results if col["p_value"]> 0.02])
X_train_encoded.describe()


def fill_data(dataset: pd.DataFrame, categorical_columns=None, method="median",iterative_imputer=RandomForestRegressor()):
    """
    Imputa i valori mancanti in un DataFrame, separando dati categorici e continui.
    """

    categorical_columns = [col for col in dataset.columns if col in categorical_columns]

    if categorical_columns is None:
        categorical_columns = []
    else:
        categorical_columns = [col for col in dataset.columns if col in categorical_columns]


    continuous_columns = [col for col in dataset.columns if col not in categorical_columns]

    filled_data = dataset.copy()

    imputers = {}

    # Imputazione categorica con SimpleImputer
    if categorical_columns:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        filled_data[categorical_columns] = cat_imputer.fit_transform(filled_data[categorical_columns])
        imputers['categorical'] = cat_imputer

    # Imputazione continua
    if method in ["mean"]:
        cont_imputer = SimpleImputer(strategy=method)
    else:
        #cont_imputer = IterativeImputer(estimator=iterative_imputer, max_iter=10, random_state=0)
        None
        
    if continuous_columns:
        filled_data[continuous_columns] = cont_imputer.fit_transform(filled_data[continuous_columns])
        imputers['continuous'] = cont_imputer

    return filled_data, imputers


def apply_imputers(X_test: pd.DataFrame, imputers: dict, categorical_columns: list[str]) -> pd.DataFrame:
    """
    Applica imputers già fittati a X_test.
    """
    filled_test = X_test.copy()

    categorical_columns = [col for col in X_test.columns if col in categorical_columns]
    continuous_columns = [col for col in X_test.columns if col not in categorical_columns]

    if 'categorical' in imputers and categorical_columns:
        filled_test[categorical_columns] = imputers['categorical'].transform(filled_test[categorical_columns])

    if 'continuous' in imputers and continuous_columns:
        filled_test[continuous_columns] = imputers['continuous'].transform(filled_test[continuous_columns])

    return filled_test


X_train_not_imputed = X_train_encoded.dropna()
Y_train_not_imputed = Y_train.loc[X_train_not_imputed.index]

X_train_imputed_mean,imputer_mean =  fill_data(dataset=X_train_encoded,categorical_columns=categorical_columns,method="mean")
#X_train_imputed_c,imputer_iterative =  fill_data(dataset=X_train_encoded,categorical_columns=categorical_columns,method="iterative")


X_test_not_imputed = X_test_encoded.dropna()
Y_test_not_imputed = Y_test.loc[X_test_not_imputed.index]

X_test_imputed_mean =  apply_imputers(X_test=X_test_encoded,imputers=imputer_mean,categorical_columns=categorical_columns)
#X_test_iterative =  apply_imputers(X_test=X_train_encoded,imputers=imputer_iterative,categorical_columns=categorical_columns)



def standardizer(dataset):
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    return dataset,scaler

def apply_standardizer(standardizer:StandardScaler,X_test):
    return standardizer.transform(X_test)

X_train_not_imputed_std, standardizer_default = standardizer(X_train_not_imputed)
X_train_imputed_mean_std, standardizer_mean = standardizer(X_train_imputed_mean)
#X_train_imputed_c_std, standardizer_c = standardizer(X_train_imputed_c)

X_test_not_imputed_std = apply_standardizer(standardizer_default,X_test_not_imputed)
X_test_imputed_mean_std = apply_standardizer(standardizer=standardizer_mean,X_test=X_test_imputed_mean)
#X_test_imputed_c_std = apply_standardizer(standardizer=standardizer_c, X_test_iterative)


smoteenn = SMOTEENN(random_state=42)

X_train_not_imputed_resampled, Y_train_not_imputed_resampled = smoteenn.fit_resample(X_train_not_imputed_std, Y_train_not_imputed)

X_train_imputed_mean_resampled, Y_train_mean_resampled = smoteenn.fit_resample(X_train_imputed_mean_std, Y_train)
#X_train_imputed_c_resampled, Y_train_c_resampled = smoteenn.fit_resample(X_train_imputed_c_std, Y_train)




def load_clean_training_data():
    training_data = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
    condition = (training_data['CGAS-CGAS_Score'] >= 0) & (training_data['CGAS-CGAS_Score'] <= 100)
    training_data = training_data[condition | training_data['CGAS-CGAS_Score'].isna()]

    condition = training_data['Physical-Weight'] > 0
    training_data = training_data[condition | training_data['Physical-Weight'].isna()]

    # Remove outliers for 'Fitness_Endurance-Max_Stage' but keep NaN
    condition = training_data['Fitness_Endurance-Max_Stage'] > 0
    training_data = training_data[condition | training_data['Fitness_Endurance-Max_Stage'].isna()]

    # Remove outliers for 'Fitness_Endurance-Time_Mins' but keep NaN
    condition = training_data['Fitness_Endurance-Time_Mins'] > 0
    training_data = training_data[condition | training_data['Fitness_Endurance-Time_Mins'].isna()]

    training_data["BIA-BIA_BMC"] = np.where(training_data["BIA-BIA_BMC"] <= 0, np.nan, training_data["BIA-BIA_BMC"])
    training_data["BIA-BIA_BMC"] = np.where(training_data["BIA-BIA_BMC"] > 10, np.nan, training_data["BIA-BIA_BMC"])
    # Remove outliers for 'BIA-BIA_DEE' but keep NaN
    condition = training_data['BIA-BIA_DEE'] < 17311
    training_data = training_data[condition | training_data['BIA-BIA_DEE'].isna()]
    # Remove highly implausible values

    # Remove implausible body-fat
    training_data["BIA-BIA_Fat"] = np.where(training_data["BIA-BIA_Fat"] < 5, np.nan, training_data["BIA-BIA_Fat"])
    training_data["BIA-BIA_Fat"] = np.where(training_data["BIA-BIA_Fat"] > 60, np.nan, training_data["BIA-BIA_Fat"])
    # Basal Metabolic Rate
    training_data["BIA-BIA_BMR"] = np.where(training_data["BIA-BIA_BMR"] > 4000, np.nan, training_data["BIA-BIA_BMR"])
    # Daily Energy Expenditure
    training_data["BIA-BIA_DEE"] = np.where(training_data["BIA-BIA_DEE"] > 8000, np.nan, training_data["BIA-BIA_DEE"])
    # Fat Free Mass Index
    training_data["BIA-BIA_FFM"] = np.where(training_data["BIA-BIA_FFM"] <= 0, np.nan, training_data["BIA-BIA_FFM"])
    training_data["BIA-BIA_FFM"] = np.where(training_data["BIA-BIA_FFM"] > 300, np.nan, training_data["BIA-BIA_FFM"])
    # Fat Mass Index
    training_data["BIA-BIA_FMI"] = np.where(training_data["BIA-BIA_FMI"] < 0, np.nan, training_data["BIA-BIA_FMI"])
    # Extra Cellular Water
    training_data["BIA-BIA_ECW"] = np.where(training_data["BIA-BIA_ECW"] > 100, np.nan, training_data["BIA-BIA_ECW"])
    # Intra Cellular Water
    training_data["BIA-BIA_ICW"] = np.where(training_data["BIA-BIA_ICW"] > 100, np.nan, training_data["BIA-BIA_ICW"])
    # Lean Dry Mass
    training_data["BIA-BIA_LDM"] = np.where(training_data["BIA-BIA_LDM"] > 100, np.nan, training_data["BIA-BIA_LDM"])
    # Lean Soft Tissue
    training_data["BIA-BIA_LST"] = np.where(training_data["BIA-BIA_LST"] > 300, np.nan, training_data["BIA-BIA_LST"])
    # Skeletal Muscle Mass
    training_data["BIA-BIA_SMM"] = np.where(training_data["BIA-BIA_SMM"] > 300, np.nan, training_data["BIA-BIA_SMM"])
    # Total Body Water
    training_data["BIA-BIA_TBW"] = np.where(training_data["BIA-BIA_TBW"] > 300, np.nan, training_data["BIA-BIA_TBW"])

    training_data = training_data[training_data['sii'].notna()]

    training_data['PAQ_TOTAL'] = training_data['PAQ_A-PAQ_A_Total'].fillna(training_data['PAQ_C-PAQ_C_Total'])
    training_data.drop(inplace=True,axis=1,columns="PAQ_A-PAQ_A_Total")
    training_data.drop(inplace=True,axis=1,columns="PAQ_C-PAQ_C_Total")

    thresh = 0.49 * len(training_data)
    training_data.dropna(thresh = thresh, axis = 1, inplace = True)
    training_data.drop(columns=['Physical-Height','Physical-Weight'],inplace=True,axis=1,errors='ignore')

    def select_bmi(row):
        if pd.notna(row['BIA-BIA_BMI']):
            return row['BIA-BIA_BMI']
        else:
            return row['Physical-BMI']

    training_data['BMI'] = training_data.apply(select_bmi, axis=1)
    training_data.drop(columns=['BIA-BIA_BMI','Physical-BMI'],inplace=True,axis=1,errors='ignore')

    training_data = training_data.drop(columns=["SDS-Season","FGC-Season","Basic_Demos-Enroll_Season","Physical-Season","PreInt_EduHx-Season","BIA-Season",
                                                    "PCIAT-Season","CGAS-Season"],errors='ignore')

    training_data =  training_data.drop(columns=["SDS-Season","FGC-Season","Basic_Demos-Enroll_Season","Physical-Season","PreInt_EduHx-Season","BIA-Season",
                                                   "PCIAT-Season","CGAS-Season"],errors='ignore')

    training_data = pd.get_dummies(training_data, columns=['PAQ_C-Season'])

    return training_data


def load_clean_test_data():
    test_data_ = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')

    test_data_['PAQ_TOTAL'] = test_data_['PAQ_A-PAQ_A_Total'].fillna(test_data_['PAQ_C-PAQ_C_Total'])
    test_data_.drop(inplace=True,axis=1,columns="PAQ_A-PAQ_A_Total")
    test_data_.drop(inplace=True,axis=1,columns="PAQ_C-PAQ_C_Total")
    test_data_.drop(columns=['Physical-Height','Physical-Weight'],inplace=True,axis=1,errors='ignore')

    def select_bmi(row):

        if pd.notna(row['BIA-BIA_BMI']):
            return row['BIA-BIA_BMI']
        else:
            return row['Physical-BMI']

    test_data_['BMI'] = test_data_.apply(select_bmi, axis=1)
    test_data_.drop(columns=['BIA-BIA_BMI','Physical-BMI'],inplace=True,axis=1,errors='ignore')

    test_data_ = test_data_.drop(columns=["SDS-Season","FGC-Season","Basic_Demos-Enroll_Season","Physical-Season","PreInt_EduHx-Season","BIA-Season",
                                                "PCIAT-Season","CGAS-Season"],errors='ignore')

    test_data_ = pd.get_dummies(test_data_, columns=['PAQ_C-Season'])

    return test_data_

target_columns = ['PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total','sii']

categorical_columns = ['Basic_Demos-Enroll_Season',
                       'CGAS-Season',
                       'Physical-Season',
                       'Fitness_Endurance-Season',
                       'FGC-Season',
                       'BIA-Season',
                       'PAQ_A-Season',
                       'PAQ_C-Season',
                       'PCIAT-Season',
                       'PreInt_EduHx-Season',
                       'Basic_Demos-Enroll_Season_Summer',
                       'Basic_Demos-Enroll_Season_Winter', 'CGAS-Season_Spring',
                       'CGAS-Season_Summer', 'CGAS-Season_Winter', 'Physical-Season_Spring',
                       'Physical-Season_Summer', 'Physical-Season_Winter',
                       'Fitness_Endurance-Season_Spring', 'Fitness_Endurance-Season_Summer',
                       'Fitness_Endurance-Season_Winter', 'FGC-Season_Spring',
                       'FGC-Season_Summer', 'FGC-Season_Winter', 'BIA-Season_Spring',
                       'BIA-Season_Summer', 'BIA-Season_Winter', 'PAQ_A-Season_Spring',
                       'PAQ_A-Season_Summer', 'PAQ_A-Season_Winter', 'PAQ_C-Season_Spring',
                       'PAQ_C-Season_Summer', 'PAQ_C-Season_Winter', 'PCIAT-Season_Spring',
                       'PCIAT-Season_Summer', 'PCIAT-Season_Winter','Basic_Demos-Enroll_Season_Spring',
                       'SDS-Season','PreInt_EduHx-Season_Spring', 'PreInt_EduHx-Season_Summer',
                       'PreInt_EduHx-Season_Winter', 'SDS-Season_Spring', 'SDS-Season_Summer',
                       'SDS-Season_Winter','Basic_Demos-Enroll_Season_Spring','Basic_Demos-Sex','FGC-FGC_PU_Zone','FGC-FGC_SRL_Zone','FGC-FGC_CU_Zone']


leaky_features = [
    'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03', 'PCIAT-PCIAT_04',
    'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07', 'PCIAT-PCIAT_08',
    'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11', 'PCIAT-PCIAT_12',
    'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15', 'PCIAT-PCIAT_16',
    'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19', 'PCIAT-PCIAT_20',
    'PCIAT-PCIAT_Total'
]



class ModelComparisonPipeline:
    def __init__(self, model_sklearn, model_external,
                 param_grid_sklearn=None, param_grid_external=None,
                 external_search_type="random", n_iter_external=30,
                 n_splits=5,hyper_cv = 5,rf_cv = 5):
        self.model_sklearn = model_sklearn
        self.model_external = model_external
        self.param_grid_sklearn = param_grid_sklearn
        self.param_grid_external = param_grid_external
        self.external_search_type = external_search_type
        self.n_iter_external = n_iter_external
        self.n_splits = n_splits
        self.hyper_cv = hyper_cv
        self.rf_cv = rf_cv
        self.use_resampling = True
        self.results = defaultdict(lambda: defaultdict(list))
        self.external_group = []
        self.sklearn_group = []
        self.feature_columns_sklearn = []
        self.feature_columns_external = []
        

    def apply_RFECV(self,sklearn_model,external_model,X,y,**kwargs):
        
        sklearn_pipeline = self._make_pipeline(sklearn_model)
        external_pipeline = self._make_pipeline(external_model)
        
        cv_for_rfe = StratifiedKFold(n_splits=self.rf_cv, shuffle=True, random_state=42)
        
        selector_sklearn = RFECV(
            estimator=sklearn_pipeline,
            step=1,
            scoring=make_scorer(cohen_kappa_score, weights='quadratic'),
            cv=cv_for_rfe,
            min_features_to_select=1,
            n_jobs=-1,
            importance_getter='named_steps.model.feature_importances_'
        )

        selector_external = RFECV(
            estimator=external_pipeline,
            step=1,
            scoring=make_scorer(cohen_kappa_score, weights='quadratic'),
            cv=cv_for_rfe,
            min_features_to_select=1,
            n_jobs=-1,
            importance_getter='named_steps.model.feature_importances_'
        )


        imputer = SimpleImputer(strategy="mean") 
        X_imputed = imputer.fit_transform(X)
        
        selector_sklearn.fit(X_imputed,y)
        selector_external.fit(X_imputed,y)

        return X.columns[selector_sklearn.support_],X.columns[selector_external.support_]


    def start_pipeline(self, df,
                       imputation_strategy='mean',
                       use_resampling=True,
                       use_RFE=True,**kwargs):
        """
            Executes training/evaluation with Stratified K-Fold CV.
        """
        X = df.drop(columns=['sii'])
        y = df['sii']

        all_columns = X.columns

        X_sklearn = X
        X_external = X
        if use_RFE:
            subset_sklearn, subset_external = self.apply_RFECV(sklearn_model = self.model_sklearn,external_model=self.model_sklearn,X= X,y= y)
            X_sklearn = X[subset_sklearn]
            X_external = X[subset_external]

            print(f"SUBSET {subset_sklearn}")

        self.feature_columns_sklearn = X_sklearn.columns
        self.feature_columns_external = X_external.columns

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        self.use_resampling = use_resampling


        for fold, (train_idx, test_idx) in enumerate(skf.split(X_sklearn, y)):

            print(f"\n===== Fold {fold+1}/{self.n_splits} =====")

            X_train, X_test = X_sklearn.iloc[train_idx], X_sklearn.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Tune hyperparameters and train models
            model_sklearn = self._train_and_predict(
                X_train, y_train,
                X_test, y_test,
                self.feature_columns_sklearn,
                use_RFE=use_RFE,
                model=self.model_sklearn,
                imputation_strategy=imputation_strategy
            )
            # Store trained models for later majority vote
            self.sklearn_group.append(model_sklearn)



        for fold, (train_idx, test_idx) in enumerate(skf.split(X_external, y)):
            print(f"\n===== Fold {fold+1}/{self.n_splits} =====")

            X_train, X_test = X_external.iloc[train_idx], X_external.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Tune hyperparameters and train models
            model_external= self._train_and_predict(
                X_train, y_train,
                X_test, y_test,
                self.feature_columns_external,
                use_RFE=use_RFE,
                model=self.model_external,
                imputation_strategy=imputation_strategy
            )


            self.external_group.append(model_external)

        print("\nCross-validation complete.")
        self.print_summary()


    def _make_pipeline(self, model, imputer=None):
        imputer = SimpleImputer(strategy="mean") 
        return ImbPipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("smoteenn", SMOTEENN(random_state=42) if self.use_resampling else "passthrough"),
            ("model", model)
        ])

    def _finalize_results(self, model, df, y_test, columns_name):
        """
            Evaluate a trained model (or pipeline) on test data.
            Stores accuracy, F1-score, predictions, and performs feature importance & error analysis.
        """
        # Predictions
        y_pred = model.predict(df)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        kappa_score = cohen_kappa_score(y_test, y_pred, weights='quadratic')

        model_name = type(model.named_steps['model']).__name__

        self.results[model_name]['accuracy'].append(acc)
        self.results[model_name]['f1_score'].append(f1)
        self.results[model_name]['kappa_score'].append(kappa_score)

        self.results[model_name]['preds'].append(y_pred)
        self.results[model_name]['true'].append(y_test)
        self.results[model_name]['estimators'].append(model)

        print(f"{model_name}: Accuracy={acc:.4f}, F1-weighted={f1:.4f}")
        print(f"Test Set Cohen Kappa (Quadratic): {kappa_score:.4f}")

        self.compute_feature_importance(model, columns_name)
        self.analyze_errors(model, df, y_test)

    def analyze_errors(self, model, X_test, y_test):
        """
            Analyzes predictions: prints metrics, confusion matrix, and error counts per class.
        """
        print("\nAnalyzing Prediction Errors...")

        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap='Blues')
        plt.title(f"{type(model.named_steps['model']).__name__} - Confusion Matrix")
        plt.show()

        class_labels = np.unique(y_test)
        errors_mask = (y_pred != y_test)
        print(f"\nErrors per class ({type(model.named_steps['model']).__name__}):")
        for cls in class_labels:
            errors = np.sum(errors_mask & (y_test == cls))
            print(f"Class {cls}: {errors} errors")
        
        
    
    def compute_feature_importance(self, model, feature_columns_used):
        """
            Computes feature importance for a trained model.
            Works with pipeline-wrapped models (RandomForest, XGBoost, etc.).
        """
        print("\nComputing Feature Importance...")

        final_model = model.named_steps['model']

        feat_imp = pd.DataFrame({
            'Feature': self.feature_columns_sklearn,
            'Importance': final_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        feat_imp.plot.bar(x='Feature', y='Importance', figsize=(10, 5), legend=False, title="Feature Importance")
        plt.title(f'{model.named_steps["model"]} - Feature importance')
        plt.show()
        return feat_imp


    def print_summary(self):
        """
        Print cross-validation summary across all folds and all models.
        Displays mean + std for Accuracy and F1-score.
        """
        print("\n" + "="*20 + " CROSS-VALIDATION SUMMARY " + "="*20)
        for model_name, metrics in self.results.items():
            mean_acc = np.mean(metrics['accuracy'])

            std_acc = np.std(metrics['accuracy'])
            mean_f1 = np.mean(metrics['f1_score'])
            std_f1 = np.std(metrics['f1_score'])

            kappa_score_mean = np.mean(metrics['kappa_score'])
            kappa_score_std = np.std(metrics['kappa_score'])

            print(f"\nModel: {model_name}")
            print(f"  Avg. Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  Avg. F1-score : {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"  Avg. Kappa-Score : {kappa_score_mean:.4f} ± {kappa_score_std:.4f}")

        print("="*60)

    def _train_and_predict(self, X_train, y_train,
                           X_test, y_test,
                           feature_columns,
                           use_RFE=True,
                           model = None,
                           imputation_strategy='mean'):
        """
        Train and evaluate both sklearn and external models on a fold.
        Handles hyperparameter tuning, optional RFECV, and evaluation.
        Returns the trained models.
        """

        print("Tuning and evaluating models...")

        # ------------------------
        best_pipe = self._tune_hyperparameters(
            clone(model),
            self.param_grid_sklearn,
            X_train, y_train,
            imputation_strategy=imputation_strategy
        )


        self._finalize_results(model=best_pipe,
                                   df=X_test, y_test=y_test,
                                   columns_name=feature_columns)

        return best_pipe


    def _tune_hyperparameters(self, model, params, X_train, y_train,
                             **kwargs):
        pipeline = self._make_pipeline(model, kwargs.get("imputer"))

        ## check if params are passed without mode__ since
        # pipeline is passed

       
        if params:
            params = {
                (k if k.startswith("model__") else f"model__{k}"): v
                for k, v in params.items()
            }
            
        if not params:
            pipeline.fit(X_train, y_train)
            return pipeline

        if self.external_search_type == "random":
            best_pipe = RandomizedSearchCV(
                pipeline, param_distributions=params,
                n_iter=self.n_iter_external, cv=StratifiedKFold(n_splits=self.hyper_cv, shuffle=True, random_state=42),
                scoring="f1_weighted", n_jobs=-1, random_state=42
            )
        else:
            best_pipe = GridSearchCV(
                pipeline, param_grid=params,
                cv=StratifiedKFold(n_splits=self.hyper_cv, shuffle=True, random_state=42), scoring="f1_weighted", n_jobs=-1
            )

        best_pipe.fit(X_train, y_train)
        return best_pipe.best_estimator_

    # --------------------------------------------------------

    def _predict(self,X, method="majority", model_type="external"):
        """
        Get predictions from the stored models (sklearn or external).
        Supports majority voting across folds.
        """
        models = self.external_group if model_type == "external" else self.sklearn_group
        feature_subset = self.feature_columns_external if model_type == "external" else self.feature_columns_sklearn

        if not models:
            raise ValueError(f"No models found in group '{model_type}'. Did you run start_pipeline()?")

        print(feature_subset)

        preds = np.array([model.predict(X[feature_subset]) for model in models])

        if method == "majority":
            majority_preds, _ = mode(preds, axis=0, keepdims=False)
            return majority_preds
        else:
            raise ValueError(f"Unknown method: {method}")

    def predict(self, X_test, y_test,model_type = "external"):
        """
        Analyzes predictions: prints metrics, confusion matrix, and error counts per class.
        Works with pipeline-wrapped models.
        """
        print("\nAnalyzing Prediction Errors...")

        y_pred = self._predict(X=X_test,model_type=model_type,method="majority")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap='Blues')
        plt.title(f"{model_type} - Confusion Matrix")
        plt.show()

        class_labels = np.unique(y_test)
        errors_mask = (y_pred != y_test)
        print(f"\nErrors per class ({model_type}):")
        for cls in class_labels:
            errors = np.sum(errors_mask & (y_test == cls))
            print(f"Class {cls}: {errors} errors")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        kappa_score = cohen_kappa_score(y_test, y_pred, weights='quadratic')

        print(f'Accuracy: {acc}')
        print(f'F1-weighted :{f1}')
        print(f'Kappa-Score :{kappa_score}')

        return y_pred



target = 'sii'
identifier = 'id'

# --- Main Script ---

training_data = load_clean_training_data()
training_data = training_data.drop(columns=[identifier] + leaky_features)

param_grid_rf = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__ccp_alpha": [0.0, 0.001, 0.01],
    "model__criterion": ['gini','entropy']
}


xgb_params = {
       'model__n_estimators': [50, 100, 200],
        'model__max_depth': [3, 5, 7],
        "model__gamma": [0, 0.1, 0.2],
        'model__lambda' : [1,5,10,20],
        'model__eval_metric' : ['rmsle']
}



lr = RandomForestClassifier(random_state=42)


xgb = XGBClassifier(random_state=42, eval_metric='logloss')

pipeline_RE = ModelComparisonPipeline(
    model_sklearn=lr,
    model_external=xgb,
    param_grid_sklearn=param_grid_rf,
    param_grid_external=xgb_params,
    external_search_type="random",
    n_iter_external=10
)

pipeline_RE.start_pipeline(df = training_data, use_resampling=True,use_RFE = False)




lr = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42, eval_metric='logloss')

pipeline_RFE_RE = ModelComparisonPipeline(
    model_sklearn=lr,
    model_external=xgb,
    param_grid_sklearn=param_grid_rf,
    param_grid_external=xgb_params,
    external_search_type="random",
    n_iter_external=10

)


pipeline_RFE_RE.start_pipeline(df = training_data, use_resampling=True,use_RFE = True)



lr = RandomForestClassifier(random_state=42,class_weight= 'balanced_subsample')
xgb = XGBClassifier(random_state=42, eval_metric='logloss')

pipeline_no_RE = ModelComparisonPipeline(
    model_sklearn=lr,
    model_external=xgb,
    param_grid_sklearn=param_grid_rf,
    param_grid_external=xgb_params,
    external_search_type="random",
    n_iter_external=10
)

pipeline_no_RE.start_pipeline(df = training_data, use_resampling=False,use_RFE = False )


lr = RandomForestClassifier(random_state=42,class_weight= 'balanced_subsample')
xgb = XGBClassifier(random_state=42, eval_metric='logloss')

pipeline_no_RE_RFE = ModelComparisonPipeline(
    model_sklearn=lr,
    model_external=xgb,
    param_grid_sklearn=param_grid_rf,
    param_grid_external=xgb_params,
    external_search_type="random",
    n_iter_external=10
)

pipeline_no_RE_RFE.start_pipeline(df = training_data, use_resampling=False,use_RFE = True )



test_data = load_clean_test_data()
X_test = test_data
Y_test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')['sii']


pred = pipeline_RFE_RE.predict(X_test = X_test, y_test = Y_test,model_type = "sklearn")


pred2 = pipeline_RE.predict(X_test = X_test, y_test = Y_test,model_type = "sklearn")


pred3 = pipeline_no_RE.predict(X_test = X_test, y_test = Y_test,model_type = "sklearn")


pred4 = pipeline_no_RE_RFE.predict(X_test = X_test, y_test = Y_test,model_type = "sklearn")


pred5 = pipeline_RE.predict(X_test = X_test, y_test = Y_test)



pred6 =pipeline_RFE_RE.predict(X_test = X_test, y_test = Y_test,model_type = "external")


pred7 = pipeline_no_RE.predict(X_test = X_test, y_test = Y_test)


pred8 = pipeline_no_RE_RFE.predict(X_test = X_test, y_test = Y_test)


results = pd.DataFrame({
    'id': pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')['id'],
    'sii': pred7
})

results.to_csv('submission.csv', index=False)
results






class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add features from unsupervised models:
    - KMeans clusters
    - IsolationForest anomaly scores
    - UMAP projections
    """
    def __init__(self, max_cluster=4, contamination=0.01, n_neighbors=15, min_dist=0.1, random_state=42,target_class = [3],iso_min_f1 = 0.6):
        self.max_cluster = max_cluster
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.iso_min_f1 = iso_min_f1
        self.target_class = target_class

    def fit(self, X, y=None):
        k_values = range(1,10)
        errors = []

        for n_c in k_values:
            kmeans_ = KMeans(n_clusters=n_c, random_state=self.random_state, n_init=10)
            kmeans_.fit(X)

            errors += [kmeans_.inertia_]
            
        # pick the right k_value
        kneedle = KneeLocator(
            x=k_values,
            y=errors,
            curve="convex",
            direction="decreasing"
        )

        optimal_k = kneedle.elbow
        print(f"The optimal number of clusters (elbow point) is: {optimal_k}")

        self.kmeans_ =  KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(X)

        self.iso_ = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        self.iso_.fit(X)

        self.include_iso_feature_ = False
        
        if y is not None:
            y_bin = y.isin(self.target_class).astype(int)
            iso_preds = (self.iso_.predict(X) == -1).astype(int)
            prec = precision_score(y_bin, iso_preds, zero_division=0)
            rec = recall_score(y_bin, iso_preds)
            f1 = f1_score(y_bin, iso_preds, zero_division=0)
            print(f"IsolationForest validation - Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
            cm = confusion_matrix(y_bin, iso_preds)
            print("Confusion matrix ISO-Forest (rows=verità, cols=preds):\n", cm)
                

        self.umap_ = umap.UMAP(n_components=2, n_neighbors=self.n_neighbors, min_dist=self.min_dist, random_state=self.random_state)
        self.umap_.fit(X)

        return self

    def transform(self, X):
        X_transformed = pd.DataFrame(X).copy()

        X_transformed['_km_cluster'] = self.kmeans_.predict(X)

        iso_preds = self.iso_.predict(X)
        X_transformed['_is_anomaly'] = (iso_preds == -1).astype(int)

        umap_proj = self.umap_.transform(X)
        X_transformed['_umap1'] = umap_proj[:, 0]
        X_transformed['_umap2'] = umap_proj[:, 1]

        
        return X_transformed.values

    
    def plot_proj(X,y):
        # Impostazioni generali
        plt.figure(figsize=(12, 5))
        
        # -------------------------
        # 1) Plot UMAP con colori delle classi reali
        # -------------------------
        plt.subplot(1, 3, 1)
        if "_umap1" in X.columns and "_umap2" in X.columns:
            # Colori basati sulle etichette originali (y_train)
            scatter = plt.scatter(
                X["_umap1"], X["_umap2"],
                c=y.astype(int), cmap="tab10", alpha=0.7
            )
            plt.title("UMAP projection colored by KMeans clusters")
            plt.xlabel("_umap1")
            plt.ylabel("_umap2")
            plt.legend(*scatter.legend_elements(), title="Clusters")
        else:
            print("UMAP projection not found")
        
        # -------------------------
        # 2) Evidenziare anomalie
        # -------------------------
        plt.subplot(1, 3, 2)
        if "_umap1" in X.columns and "_umap2" in X.columns:
            colors = X["_is_anomaly"].map({0: "grey", 1: "red"})
            plt .scatter(
                X["_umap1"], X["_umap2"],
                c=colors, alpha=0.7
            )
            plt.title("UMAP projection with anomalies")
            plt.xlabel("_umap1")
            plt.ylabel("_umap2")
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='grey', markersize=8),
                               Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=12)]
            plt.legend(handles=legend_elements)
        else:
            print("UMAP projection not found")
        
        # -------------------------
        # 3) SII
        # -------------------------

        if y is not None:
            plt.subplot(1, 3, 3)
            colors = y.map({0: "blue", 1: "red",2:"green",3:"orange"})
            plt.scatter(
                X["_umap1"], X["_umap2"],
                    c=colors, alpha=0.7
                )
            plt.title("Sii projection")
            plt.xlabel("_umap1")
            plt.ylabel("_umap2")
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='Class:0', markerfacecolor='blue', markersize=8),
                            Line2D([0], [0], marker='o', color='w', label='Class:1', markerfacecolor='red', markersize=9),
                              Line2D([0], [0], marker='o', color='w', label='Class:2', markerfacecolor='green', markersize=10),
                              Line2D([0], [0], marker='o', color='w', label='Class:3', markerfacecolor='orange', markersize=11)]
            plt.legend(handles=legend_elements)
                
        plt.tight_layout()
        plt.show()


training_data = load_clean_training_data()
training_data = training_data.drop(columns=[identifier] + leaky_features)

y_train = training_data['sii']
X_train = training_data.drop(columns='sii')

feature_used = X_train.columns.tolist()

fe = FeatureEngineer(random_state=42, target_class=[3], iso_min_f1=0.001)

imputer = SimpleImputer(strategy="mean")

scaler = StandardScaler()
scaler.fit_transform(X_train)

X_train = pd.DataFrame(imputer.fit_transform(X_train),columns=X_train.columns)

fe.fit(X_train, y_train)

X_transformed = fe.transform(X_train)

X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=list(X_train.columns) + ['_km_cluster','_is_anomaly','_umap1','_umap2']
)





 # Impostazioni generali
plt.figure(figsize=(12, 5))
        
# -------------------------
# 1) Plot UMAP con colori delle classi reali
# -------------------------
plt.subplot(2, 3, 1)
scatter = plt.scatter(
    X_transformed_df["_umap1"], X_transformed_df["_umap2"],
    c=y_train.astype(int), cmap="tab10", alpha=0.7
)
plt.title("UMAP projection colored by KMeans clusters")
plt.xlabel("_umap1")
plt.ylabel("_umap2")
plt.legend(*scatter.legend_elements(), title="Clusters")
        
# -------------------------
# 2) Evidenziare anomalie
# -------------------------
plt.subplot(2, 3, 2)
colors = X_transformed_df["_is_anomaly"].map({0: "grey", 1: "red"})
plt .scatter(
    X_transformed_df["_umap1"], X_transformed_df["_umap2"],
    c=colors, alpha=0.7
)
plt.title("UMAP projection with anomalies")
plt.xlabel("_umap1")
plt.ylabel("_umap2")
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='grey', markersize=8),
                        Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=12)]
plt.legend(handles=legend_elements)

# -------------------------
# 3) SII
# -------------------------

plt.subplot(2, 3, 3)
colors = y_train.map({0: "blue", 1: "red",2:"green",3:"orange"})
plt.scatter(
            X_transformed_df["_umap1"], X_transformed_df["_umap2"],
                    c=colors, alpha=0.7
                )
plt.title("Sii projection")
plt.xlabel("_umap1")
plt.ylabel("_umap2")
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Class:0', markerfacecolor='blue', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Class:1', markerfacecolor='red', markersize=9),
                      Line2D([0], [0], marker='o', color='w', label='Class:2', markerfacecolor='green', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='Class:3', markerfacecolor='orange', markersize=15)]
plt.legend(handles=legend_elements)
                
plt.tight_layout()
plt.show()


# -------------------------
# 4) SII
# -------------------------

plt.subplot(2, 1, 1)
colors = y_train.map({0: "grey", 1: "grey",2:"green",3:"orange"})
plt.scatter(
            X_transformed_df["_umap1"], X_transformed_df["_umap2"],
                    c=colors, alpha=0.7
                )
plt.title("Sii projection")
plt.xlabel("_umap1")
plt.ylabel("_umap2")
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Class:0', markerfacecolor='blue', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Class:1', markerfacecolor='red', markersize=9),
                      Line2D([0], [0], marker='o', color='w', label='Class:2', markerfacecolor='green', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='Class:3', markerfacecolor='orange', markersize=15)]
plt.legend(handles=legend_elements)
                
plt.tight_layout()
plt.show()



from sklearn.metrics import rand_score

# cluster predetti
clusters = X_transformed_df['_km_cluster']

# quanto i cluster coincidono con le classi reali

ari = rand_score(y_train, clusters)
print(f"Rand Index (cluster vs class): {ari:.3f}")


corr = X_transformed_df[['_km_cluster','_umap2','_umap1','_is_anomaly']].copy()
corr['sii'] = y_train

sns.heatmap(corr.corr())


results = pd.DataFrame({
    'id': pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')['id'],
    'sii': pred7
})

results.to_csv('submission.csv', index=False)
results

