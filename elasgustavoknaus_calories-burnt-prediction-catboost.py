import math
import numpy as np 
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split


class RMSLE(object):
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            val = max(approxes[index], 0)
            der1 = math.log1p(targets[index]) - math.log1p(max(0, approxes[index]))
            der2 = -1 / (max(0, approxes[index]) + 1)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result
    
    
class RMSLE_val(object):
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((math.log1p(max(0, approx[i])) - math.log1p(max(0, target[i])))**2)

        return error_sum, weight_sum


df_datas = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col = "id")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col = "id")


df_train


df_datas


# Opción 1: reasignar el DataFrame
df_datas = df_datas.rename(columns={"Gender": "Sex"})

# Opción 2: modificar en sitio (sin crear copia)
df_datas.rename(columns={"Gender": "Sex"}, inplace=True)
df_datas.drop("User_ID", axis = 1, inplace = True)


df_full = pd.concat([df_train, df_datas])
# df_full.drop("Gender",inplace = True)
df_full


df_full.isna().sum()


to_split = df_full.shape[0]

df_full_TT = pd.concat([df_full, df_test], axis = 0)
df_full_TT.drop("Calories", axis = 1, inplace = True)


df_full_TT



def preprocessFE(df):
    # Codificación binaria de sexo
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    
    # Grupo etario ordinal (0=niño, ..., 6=anciano)
    bins_age = [0, 12, 17, 29, 44, 59, 74, np.inf]
    labels_age = list(range(len(bins_age) - 1))  # [0, 1, 2, 3, 4, 5, 6]
    df['grupo_edad'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=True).astype(int)
    
    # Cálculo de IMC
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    
    # Clasificación ordinal del IMC (0=bajo peso, ..., 4=obesidad II+)
    bins_imc = [0, 18.5, 24.9, 29.9, 34.9, np.inf]
    labels_imc = list(range(len(bins_imc) - 1))  # [0,1,2,3,4]
    df['BMI_class'] = pd.cut(df['BMI'], bins=bins_imc, labels=labels_imc, right=True).astype(int)
    
    # Taquicardia relativa (frecuencia cardíaca > 100 bpm)
    df['high_heart_rate'] = (df['Heart_Rate'] > 100).astype(int)
    
    # Clasificación ordinal de nivel de actividad (0=ligera, 1=moderada, 2=intensa)
    def clasificar_actividad(duracion):
        if duracion < 10:
            return 0
        elif duracion < 20:
            return 1
        else:
            return 2
    df['actividad_nivel'] = df['Duration'].apply(clasificar_actividad).astype(int)
    
    return df




df_full_TTP = preprocessFE(df_full_TT)
df_full_TTP


df_full_TTP.describe()


print(f"""df_base: {df_datas.shape[0]};
df_play: {df_train.shape[0]}, df_play_test: {df_test.shape[0]}""")


df_train


df_full_TTP


# Variables categóricas (ordinales, binarias)
categorical_cols = [
    "Sex",
    "grupo_edad",
    "BMI_class",
    "high_heart_rate",
    "actividad_nivel"
]

# Variables continuas (numéricas reales)
continuous_cols = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "BMI"
]



df_full_TTP[continuous_cols].hist(figsize=(12, 8))




def agregar_stats_agrupadas(df, group_cols, stats_cols):
    # Calculamos estadísticas por grupo
    grouped = df.groupby(group_cols)[stats_cols].agg(['mean', 'std', 'min', 'max', 'median']).reset_index()
    
    # Los nombres de columnas quedan como tuplas, las "aplanamos"
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Unimos estas stats al df original (merge por las columnas de grupo)
    df_merged = df.merge(grouped, how='left', left_on=group_cols, right_on=group_cols)
    
    return df_merged

# Variables
group_columns = ['Sex', 'grupo_edad']
stats_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# Aplicamos función
df_con_stats = agregar_stats_agrupadas(df_full_TTP, group_columns, stats_columns)
df_con_stats



to_split


TRAIN = df_con_stats.iloc[:to_split]
TEST = df_con_stats.iloc[to_split:]



TRAIN


Y = df_full.pop("Calories")


Y.hist()


sns.heatmap(TRAIN[continuous_cols].corr())


X_train, X_val, y_train, y_val = train_test_split(
    TRAIN, Y, test_size=0.25, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
eval_pool = Pool(X_val, y_val, cat_features=categorical_cols)
clf = CatBoostRegressor(iterations=90, learning_rate=0.2, early_stopping_rounds=200, loss_function=RMSLE(), eval_metric=RMSLE_val())
clf.fit(train_pool, eval_set=eval_pool)



preds = clf.predict(TEST)
# df_play_test[["id", "cost"]].to_csv("submission_3.csv", index=False)
preds


sub = pd.DataFrame({"id": df_test.index, "Calories": preds})
sub.to_csv("submission.csv", index=False)


