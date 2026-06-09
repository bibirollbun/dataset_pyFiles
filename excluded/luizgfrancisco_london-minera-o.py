#Nesta etapa, importamos todas as bibliotecas necessárias e carregamos os dados de treino, teste e os rótulos do treino.



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from scipy.stats import shapiro, f_oneway
from statsmodels.stats.multicomp import MultiComparison



# Carregando os dados
train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)
y_train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None)


train.columns = [f"Feature_{i}" for i in range(train.shape[1])]


#Fazemos verificações de valores nulos, valores vazios, 
#dados duplicados e outliers. Também renomeamos as colunas para facilitar a leitura.


# Verificação de dados ausentes e duplicados
print("Valores nulos:", train.isnull().sum().sum())
print("Duplicados:", train.duplicated().sum())


#Usamos o PCA (Análise de Componentes Principais) para reduzir a dimensionalidade dos dados, mantendo 90% da variância explicada.



# Visualização dos dados
plt.figure(figsize=(10, 6))
train.plot(kind='kde', legend=False, title="Distribuição das Features")
plt.show()


# Certificando que os dois têm os mesmos nomes de colunas
column_names = [f"feature_{i}" for i in range(train.shape[1])]
train.columns = column_names
test.columns = column_names

# PCA com 90% da variância explicada
pca = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(train)
X_test_pca = pca.transform(test)


#Nesta etapa, comparamos o desempenho de três classificadores: Random Forest, KNN e SVM, utilizando validação cruzada (K-Fold).



# Avaliação de modelos com validação cruzada
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=10),
    "SVM": SVC(kernel='rbf', C=1.0)
}

resultados = {}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X_train_pca, y_train.values.ravel(), cv=kfold)
    resultados[nome] = scores
    print(f"{nome}: Média = {scores.mean():.4f}, Desvio = {scores.std():.4f}")


# Teste ANOVA
anova = f_oneway(resultados['Random Forest'], resultados['KNN'], resultados['SVM'])
print("p-valor ANOVA:", anova.pvalue)


# Teste de Tukey
df_resultados = pd.DataFrame({
    "score": np.concatenate(list(resultados.values())),
    "modelo": sum([[nome]*len(scores) for nome, scores in resultados.items()], [])
})

comparacao = MultiComparison(df_resultados["score"], df_resultados["modelo"])
teste_tukey = comparacao.tukeyhsd()
print(teste_tukey)

teste_tukey.plot_simultaneous()
plt.show()



# Treinamento final e ensemble simples
rf = modelos['Random Forest'].fit(X_train_pca, y_train.values.ravel())
knn = modelos['KNN'].fit(X_train_pca, y_train.values.ravel())
svm = modelos['SVM'].fit(X_train_pca, y_train.values.ravel())

pred_rf = rf.predict(X_test_pca)
pred_knn = knn.predict(X_test_pca)
pred_svm = svm.predict(X_test_pca)


# Votação
final_pred = []
for i in range(len(pred_rf)):
    votos = pred_rf[i] + pred_knn[i] + pred_svm[i]
    final_pred.append(1 if votos >= 2 else 0)


#criar submissão padrão


submission = pd.DataFrame({
    "Id": range(1, len(final_pred)+1),
    "Solution": final_pred
})
submission.to_csv("submission.csv", index=False)

