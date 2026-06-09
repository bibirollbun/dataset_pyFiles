import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_train_raw = pd.read_csv('/kaggle/input/nabla-case-gjensidige-2-0/train.csv')
df_test = pd.read_csv('/kaggle/input/nabla-case-gjensidige-2-0/test.csv')


numeric_variables = ["Age", "Annual Income", "Number of Dependents", "Health Score", "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration"]
categorical_variables = ["Gender", "Marital Status", "Education Level", "Occupation", "Location", "Policy Type", "Policy Start Date", "Customer Feedback", "Smoking Status", "Exercise Frequency", "Property Type"]


# MULIGHET FOR Ã… FORBEDRE DATAEN FOR ANALYSE HER

# HER FJERNER VI ALLER RADER MED NA-VERDIER
df_train = df_train_raw.dropna()

# DENNE LINJEN FYLLER INN ALLE RADER MED NA-VERDIER FOR EN GITT VARIABEL [EKSEMPELVIS "Gender"] TIL GJENNOMSNITTLIG VERDI FOR RESTEN AV RADENE
# df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean())

# DENNE LINJEN FYLLER INN ALLE RADER MED NA-VERDIER FOR EN GITT VARIABEL [EKSEMPELVIS "Gender"] MED DEN VERDIEN SOM FOREKOMMER MEST
# df_train["Gender"] = df_train["Gender"].fillna(df_train["Gender"].mode()[0])

# Skriver ut informasjo om dataen
print(
    f"Antall rader i df_train_raw: {len(df_train_raw):,}\n"
    f"Antall rader i df_train: {len(df_train):,}\n"
    f"Andel rader som forsvant fra df_train_raw til df_train i forberedelsene er: "
    f"{(len(df_train_raw) - len(df_train)) / len(df_train_raw):.2%}"
)


# DENNE MÃ… ENDRES FOR Ã… BESTEME VARIABLER SOM SKAL VISUALISERES OG OPPLÃ˜SNIGNEN - RESTEN KAN BEHOLDES SOM DEN ER HVIS MAN IKKE Ã˜NSKER ANDRE TYPER PLOT
vizualisation_var_hist = "Premium Amount"
num_bins = 50

# Visualisering av den valgte variabelen
plt.figure(figsize=(15,5))
plt.hist(df_train[vizualisation_var_hist], bins=num_bins, color='indianred', edgecolor='black')
plt.title(f'Histogram av kolonnen "{vizualisation_var_hist}"')
plt.xlabel('Verdi')
plt.ylabel('Antall')
plt.show()


# DENNE MÃ… ENDRES FOR Ã… BESTEME VARIABLER SOM SKAL VISUALISERES - RESTEN KAN BEHOLDES SOM DEN ER HVIS MAN IKKE Ã˜NSKER ANDRE TYPER PLOT
vizualisation_var_box = "Education Level"

# Visualisering av den valgte variabelen
import plotly.express as px
fig = px.box(df_train, x=vizualisation_var_box, y='Premium Amount',
             title=f'"Premium Amount" fordelt etter "{vizualisation_var_box}"',
             labels={'Policy Type': 'Polise-type', 'Premium Amount': 'PremiebelÃ¸p (NOK)'})
fig.show()


# Importerer biblioteket
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# Variablene skal ligge i en liste med strenger eks: ['Age', ..., 'Annual Income'] 
# DENNE MÃ… ENDRES FOR Ã… BESTEME VARIABLER SOM SKAL MED I ANALYSEN - RESTEN KAN BEHOLDES SOM DEN ER 
LM_var = ['Age']

# Definerer forklaringsvariablene X og responsvariabelen y for treningsdataen [HER BRUKES "ONE-HOT ENCODING" FOR ALLE KATEGORISKE VARIABLER, DENNE KAN ENDRES DERSOM MAN Ã˜NSKER NOE ANNET] 
X_train = pd.get_dummies(df_train[LM_var], drop_first=True)
X_train = sm.add_constant(X_train)
X_train = X_train.astype(float)

y_train = df_train['Premium Amount']

# Presiserer modellvalg for den lineÃ¦re modellen
model = sm.OLS(y_train, X_train)
lm = model.fit()


# Definerer forklaringsvariablene X for testdataen
X_test = pd.get_dummies(df_test[LM_var], drop_first=True)
X_test = sm.add_constant(X_test)
X_test = X_test.astype(float)

# Skriver ut resulatet av analysen
print(lm.summary())

predictions_LM = lm.predict(X_test) 


# Importerer biblioteket
import statsmodels.api as sm


# Variablene skal ligge i en liste med strenger eks: ['Age', ..., 'Annual Income'] 
# DENNE MÃ… ENDRES FOR Ã… BESTEME VARIABLER SOM SKAL MED I ANALYSEN - RESTEN KAN BEHOLDES SOM DEN ER 
GLM_var = ['Age']

# Definerer forklaringsvariablene X og responsvariabelen y for treningsdataen [HER BRUKES "ONE-HOT ENCODING" FOR ALLE KATEGORISKE VARIABLER, DENNE KAN ENDRES DERSOM MAN Ã˜NSKER NOE ANNET] 
X_train = pd.get_dummies(df_train[GLM_var], drop_first=True)
X_train = sm.add_constant(X_train)
X_train = X_train.astype(float)

y_train = df_train['Premium Amount']

# Presiserer modellvalg for GLM-modellen [HER ER DET MULIG Ã… ENDRE PÃ… FAMILY OG LINK FOR Ã… FÃ… ULIKE FORDELINGER OG LINKFUNSJONER I GLM-MODELLEN] 
model = sm.GLM(y_train, X_train, family=sm.families.Gaussian(link=sm.families.links.Identity()))
glm_fit = model.fit()

# Definerer forklaringsvariablene X for testdataen
X_test = pd.get_dummies(df_test[GLM_var], drop_first=True)
X_test = sm.add_constant(X_test)
X_test = X_test.astype(float)

# Skriver ut resulatet av analysen
print(glm_fit.summary())

# Predikerer basert pÃ¥ forklaringsvariablene fra 
predictions_GLM = glm_fit.predict(X_test)


# Importerer biblioteket
import os
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd


# Variablene skal ligge i en liste med strenger, f.eks.: ['Age', ..., 'Annual Income']
# DENNE MÃ… ENDRES FOR Ã… BESTEMME HVILKE VARIABLER SOM SKAL MED I ANALYSEN
CB_var = ['Age']

# PRESISER HVILKE VARIABLER SOM SKAL VÃ†RE KATEGORISKE
categorical_cols = []

# Definerer forklaringsvariablene X og responsvariabelen y for treningsdataen og testdataen
X_train = df_train[CB_var]
y_train = df_train["Premium Amount"]

X_test = df_test[CB_var]

# Lager CatBoost Pool med kategoriske variabler
train_pool = Pool(X_train, y_train)

# Initialiserer og trener CatBoost-modellen
model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=200,
)
model.fit(train_pool)

# Prediksjon pÃ¥ testdata
predictions_CB = model.predict(X_test)


# KOMMERNTER INN RIKTIG MODELL
predictions = np.zeros(len(df_test))
# predictions = predictions_GLM
# predictions = predictions_LM
# predictions = predictions_CB

# Lager submission-fil
submission = pd.DataFrame({
    'ID': df_test['ID'],
    'prediction': predictions_CB
})
submission.to_csv('submission.csv', index=False)

# Skriver ut submission-filen
print(submission)

