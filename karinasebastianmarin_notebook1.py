import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')


quantitative_features = ['age_at_hct', 'comorbidity_score', 'hla_match_drb1_high', 'hla_match_drb1_low']
target = 'efs'


X_quantitative = train_data[quantitative_features]
X_quantitative = X_quantitative.fillna(X_quantitative.mean())
y = train_data[target]


X_train_quant, X_test_quant, y_train_quant, y_test_quant = train_test_split(X_quantitative, y, test_size=0.2, random_state=42)


print("\nFase 1: Clasificador J48 (Árbol de decisión)")
j48_model = DecisionTreeClassifier(random_state=42)
j48_model.fit(X_train_quant, y_train_quant)
j48_predictions = j48_model.predict(X_test_quant)
print(classification_report(y_test_quant, j48_predictions))


y_pred_j48 = j48_model.predict(X_test_quant)


print("\nMatriz de confusión - J48:")
print(confusion_matrix(y_test_quant, y_pred_j48))
print("\nReporte de clasificación - J48:")
print(classification_report(y_test_quant, y_pred_j48))
print("\nPrecisión - J48:", accuracy_score(y_test_quant, y_pred_j48))


positive_cases = X_test_quant[y_pred_j48 == 1]


qualitative_features = ['race_group', 'ethnicity', 'donor_related', 'graft_type']
X_qualitative = train_data.loc[positive_cases.index, qualitative_features]
X_qualitative = pd.get_dummies(X_qualitative, drop_first=True)


scaler = StandardScaler()
X_qualitative_normalized = scaler.fit_transform(X_qualitative)


X_train_qual, X_test_qual, y_train_qual, y_test_qual = train_test_split(X_qualitative_normalized, y.loc[positive_cases.index], test_size=0.2, random_state=42)


print("\nFase 2: Clasificador k-Vecinos Más Cercanos (k-NN)")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_qual, y_train_qual)


y_pred_knn = knn_model.predict(X_test_qual)


print("\nMatriz de confusión - k-NN:")
print(confusion_matrix(y_test_qual, y_pred_knn))
print("\nReporte de clasificación - k-NN:")
print(classification_report(y_test_qual, y_pred_knn))
print("\nPrecisión - k-NN:", accuracy_score(y_test_qual, y_pred_knn))


importances = j48_model.feature_importances_
feature_names = quantitative_features
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title('Importancia de las características (J48)')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.show()

