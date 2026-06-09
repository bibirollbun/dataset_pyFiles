import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Charger les donnees
data1 = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
data2 = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

# Supprimer les lignes avec des valeurs manquantes
data1 = data1.dropna()
data2 = data2.dropna()

# Sélectionner les colonnes 'id' comme variable d'entrée et 'num_sold' comme variable cible
x = data2['id'].values.reshape(-1, 1)
y = data2['num_sold'].values.reshape(-1, 1)

# Instancier et entraîner le modèle de régression linéaire
regression = LinearRegression()
regression.fit(x, y)

# Prédire 'num_sold' pour les 'id' dans le jeu de test
x_test = data1['id'].values.reshape(-1, 1)
y_pred = regression.predict(x_test)

#retrouver les coefficients et l ordonnee a l origine de l equation y=ax+b
print(f'Coefficient a (pente) : {regression.coef_[0]}')
print(f'Ordonnée à l origine b : {regression.intercept_}')
#ajouter  sur le graphe la droite 
ordonne = np.linspace(0, 15, 1000).reshape(-1,1)
plt.scatter(x,y)
plt.plot(ordonne, regression.coef_[0]*ordonne + regression.intercept_, color='red')
plt.xlabel('id')
plt.ylabel('Sold')
#plt.show()

# Créer un DataFrame avec les prédictions
submission = pd.DataFrame({
    'id': data1['id'],
    'num_sold': y_pred.flatten()  # Flatten pour convertir en tableau 1D
})

# Sauvegarder le DataFrame dans un fichier CSV
submission.to_csv('submission.csv', index=False)

# Afficher les premières lignes du fichier de soumission
print(submission.head())

