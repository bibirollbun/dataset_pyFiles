import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')

# ğŸ”§ List of numerical features in the original dataset
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']

# ğŸ”� Count unique values for each numerical feature
unique_counts = {col: df[col].nunique() for col in numerical_features}

# ğŸ“Š Convert to DataFrame for plotting
cardinality_df = pd.DataFrame.from_dict(unique_counts, orient='index', columns=['Unique Values'])
cardinality_df = cardinality_df.sort_values('Unique Values', ascending=False)

# ğŸ”¥ Plot
plt.figure(figsize=(8,5))
sns.barplot(x=cardinality_df.index, y=cardinality_df['Unique Values'], palette='viridis')
plt.title('Number of Unique Values per Numerical Feature')
plt.ylabel('Unique Values')
plt.xlabel('Feature')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# ğŸ”� Display the exact numbers
cardinality_df.style.background_gradient(cmap='Blues')



from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

df_categorical = df.copy()


y = df_categorical['Fertilizer Name']
df_categorical = df_categorical.drop(columns=['id', 'Fertilizer Name'])
df_categorical = df_categorical.astype(str)

encoder_cat = OrdinalEncoder()
X_encoded = encoder_cat.fit_transform(df_categorical)

encoder_target = LabelEncoder()
y_encoded = encoder_target.fit_transform(y)

print(f'X shape: {X_encoded.shape}')
print(f'y shape: {y_encoded.shape}')

pd.DataFrame(X_encoded, columns=df_categorical.columns).head()



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# ğŸ”§ FunÃ§Ã£o MAP@3
def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    actual: list of true labels
    predicted: array of predicted label arrays (top k predictions)
    """
    score = 0.0
    for a, p in zip(actual, predicted):
        if a in p:
            rank = np.where(p == a)[0][0]
            score += 1.0 / (rank + 1)
        else:
            score += 0.0
    return score / len(actual)

# ğŸ”¥ Encoding do target (caso nÃ£o tenha sido feito antes)
encoder_target = LabelEncoder()
y_encoded = encoder_target.fit_transform(y)

# ğŸ”¥ Instanciar o modelo
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(encoder_target.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# ğŸš€ Treinar
model.fit(X_encoded, y_encoded)

# ğŸ’¾ Salvar modelo e encoders
model.save_model('xgboost_model_categorical.json')
joblib.dump(encoder_cat, 'encoder_cat.pkl')
joblib.dump(encoder_target, 'encoder_target.pkl')
print("âœ… Modelo e encoder do target salvos!")

# ğŸ”® PrediÃ§Ã£o simples para accuracy
y_pred = model.predict(X_encoded)

# ğŸ”¥ PrediÃ§Ã£o de probabilidades para MAP@3
y_proba = model.predict_proba(X_encoded)

# ğŸ�¯ Pegar top 3 classes com maior probabilidade
top_3 = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]

# âœ… Calcular MAP@3
map3 = mapk(y_encoded, top_3, k=3)

# âœ… AvaliaÃ§Ã£o com accuracy (referÃªncia)
acc = accuracy_score(y_encoded, y_pred)

print(f'âœ… Accuracy: {acc:.4f}')
print(f'âœ… MAP@3: {map3:.4f}')

# ğŸ“œ Classification report
print(classification_report(y_encoded, y_pred, target_names=encoder_target.classes_))

# ğŸ”¥ Confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_encoded, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder_target.classes_, yticklabels=encoder_target.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()



# ğŸ“¦ Importar bibliotecas necessÃ¡rias
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

# ğŸ”¥ Carregar o modelo salvo
model = XGBClassifier()
model.load_model('xgboost_model_categorical.json')

# ğŸ”¥ Carregar os encoders salvos
encoder_cat = joblib.load('encoder_cat.pkl')
encoder_target = joblib.load('encoder_target.pkl')

# ğŸš€ Carregar o dataset de teste
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


# ğŸš© Salvar o ID para o arquivo de submissÃ£o
ids = test_df['id']

# ğŸ”¥ Preparar as features do teste (remover 'id')
X_test = test_df.drop(columns=['id'])
X_test = X_test.astype(str)  # ğŸ”¥ Tratar todas as colunas como categÃ³ricas

# ğŸ”§ Aplicar encoding nas features
X_test_encoded = encoder_cat.transform(X_test)

# ğŸ”® Fazer prediÃ§Ãµes de probabilidades
y_proba = model.predict_proba(X_test_encoded)

# ğŸ�¯ Obter top 3 classes com maior probabilidade
top_3 = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]

# ğŸ”¥ Corrigir o problema do inverse_transform em array 2D
top_3_flat = top_3.flatten()  # Achata o array 2D para 1D
top_3_labels_flat = encoder_target.inverse_transform(top_3_flat)  # Faz inverse transform
top_3_labels = top_3_labels_flat.reshape(top_3.shape)  # Volta para shape (n amostras, 3)

# ğŸ”§ Formatar as prediÃ§Ãµes como string separada por espaÃ§o
predictions = [' '.join(row) for row in top_3_labels]

# ğŸ“œ Gerar o dataframe de submissÃ£o
submission = pd.DataFrame({
    'id': ids,
    'Fertilizer Name': predictions
})

# ğŸ’¾ Salvar como CSV
submission.to_csv('submission.csv', index=False)

print("âœ… Arquivo 'submission.csv' salvo e pronto para submissÃ£o no Kaggle.")


