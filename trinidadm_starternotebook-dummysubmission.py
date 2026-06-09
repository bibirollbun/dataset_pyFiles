COMP_DIR = "/kaggle/input/hackaton-udesa-ort"
TRAIN = f"{COMP_DIR}/train.csv"
TEST  = f"{COMP_DIR}/test.csv"
SAMPLE_SUB = f"{COMP_DIR}/sample_submission.csv"


import pandas as pd

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)
sample = pd.read_csv(SAMPLE_SUB)


# visualizamos (primeras filas)
train.head()


# algunas estadÃ­sticas bÃ¡sicas
train.describe()


# Definimos el target y el ID Ãºnico (importante para la submission)
ID_COL = "track_id"
TARGET = "is_hit"


# Generamos el dummy submission para entrar a la competencia
dummy = test[[ID_COL]].copy()
dummy[TARGET] = 0  # predicciones 0 o 1 son vÃ¡lidas para hacer un submission
dummy.to_csv("/kaggle/working/submission.csv", index=False)
print("Dummy submission created:", dummy.head(), sep="\n")

