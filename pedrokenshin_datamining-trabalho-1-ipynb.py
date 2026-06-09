# importando bibliotecas
import os
import sys
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Lendo arquivos
df = pd.read_csv(r'/kaggle/input/playground-series-s4e2/train.csv')
test_df = pd.read_csv(r'/kaggle/input/playground-series-s4e2/test.csv')


# Conferindo o dataframe de treino
df


# Conferindo o dataframe de teste
test_df


# conferindo o tamanho do dataframe
df.shape


# verificando se há valores nulos
df.isnull().sum()


# verificando o tipo dado das colunas
df.info()


# Fazendo uma descrição estatística do dataframe
df.describe()


# Criando um gráfico de barras para cada coluna categórica
def plotagem(columnname, title=None):
    """
    Plota um gráfico de barras horizontal para uma coluna categórica do DataFrame df.

    Parâmetros:
    - columnname: str. Nome da coluna a ser plotada.
    - title: str ou None. Título customizado do gráfico .
    """
    # Verifica se a coluna existe
    if columnname not in df.columns:
        raise ValueError(f"Coluna '{columnname}' não encontrada no DataFrame.")

    # Contagem de valores
    value_counts = df[columnname].value_counts()
    labels = value_counts.index.tolist()

    # Cria a figura
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df,
        y=columnname,
        hue=columnname,
        order=labels,
        palette='deep',
        ax=ax
    )

    # Adiciona os valores ao lado das barras
    max_count = value_counts.max()
    for i, v in enumerate(value_counts):
        ax.text(v + max_count * 0.01, i, str(v), va='center', fontsize=10)

    # Título
    ax.set_title(title if title else columnname, fontsize=14, fontweight='normal')

    # Limpeza visual
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks([])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.yticks(rotation=45) # rotacionando os rótulos do eixo x
    plt.show()



# Chamando a função para plotagem
plotagem("NObeyesdad", title="Tipo de Obesidade")


# Chamando a função para plotagem
plotagem("family_history_with_overweight", title="Histórico Familiar com Sobrepeso")


# Função modificada para plotar gráfico de pizza
def pizza(columnname, title=None):
    """
    Plota um gráfico de pizza para a coluna categórica de gênero do DataFrame.

    Parâmetros:
    - columnname: str. Nome da coluna a ser plotada.
    - title: str ou None. Título do gráfico (opcional).
    """
    # Verifica se a coluna existe
    if columnname not in df.columns:
        raise ValueError(f"Coluna '{columnname}' não encontrada no DataFrame.")

    # Contagem de valores
    value_counts = df[columnname].value_counts()
    labels = value_counts.index.tolist()

    # Cria o gráfico de pizza
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(
        value_counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('deep'),
        wedgeprops=dict(edgecolor='w')
    )

    # Título
    ax.set_title(title if title else columnname, fontsize=14, fontweight='bold')

    # Limpeza visual
    plt.axis('equal')  # Para garantir que o gráfico seja um círculo
    plt.tight_layout()
    plt.show()


# Chamando a função para plotagem
pizza('Gender', title='Distribuição de Gênero')


# Chamando a função para plotagem
plotagem('FAVC', title='Frequência de Consumo Calórico')


# Chamando a função para plotagem
plotagem("CAEC", title="Consumo de Alimentos Entre Refeições")


# Chamando a função para plotagem
plotagem("SMOKE", title="Quantidade de Fumantes")


# Chamando a função para plotagem
plotagem("SCC", title="Monitoram Consumo de Calorias")


# Chamando a função para plotagem
plotagem("CALC", title="Consumo de Álcool")


# Chamando a função para plotagem
plotagem("MTRANS", title="Uso de Transporte")


# Criando um gráfico de dispersão
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Weight', y='Height', hue='NObeyesdad', palette='deep')

# Título
plt.title('Dispersão: Peso vs Altura por Tipo de Obesidade')

# Eixos
plt.xlabel('Peso (kg)')
plt.ylabel('Altura (m)')

# Visualização
plt.legend(loc='best')
plt.grid(True)
plt.show()


# Criando a coluna IMC (Índice de Massa Corporal)
df['IMC'] = df['Weight'] / (df['Height'] ** 2)
test_df['IMC'] = test_df['Weight'] / (test_df['Height'] ** 2)

# Plotando o gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='IMC', hue='NObeyesdad', palette='deep')

# Título
plt.title('Dispersão: IMC vs Idade por Tipo de Obesidade')

# Visualização
plt.xlabel('Idade (anos)')
plt.ylabel('IMC (kg/m²)')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# Função que plota o boxplot de uma coluna numérica vs tipo de obesidade
def boxplot(df, column):
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x='NObeyesdad',
        y=column,
        hue='NObeyesdad',
        palette='deep',
        dodge=False
    )
    # Visualização do boxplot
    plt.grid(True)
    plt.legend([],[], frameon=False)  # Remove a legenda
    plt.title(f'Boxplot de {column} por categoria de obesidade', fontsize=14)
    plt.xlabel('Categoria de obesidade')
    plt.ylabel(column)
    plt.xticks(rotation=30)
    plt.tight_layout()



# Chamando a função para plotagem da Idade
boxplot(df, 'Age')


# Chamando a função para plotagem da Altura
boxplot(df, 'Height')


# Chamando a função para plotagem do Peso
boxplot(df, 'Weight')


# Chamando a função para plotagem do IMC
boxplot(df, 'IMC')


# Chamando a função para plotagem do Consumo diário de água
boxplot(df, 'CH2O')


# Chamando a função para plotagem Tempo de uso de tecnologia
boxplot(df, 'TUE')


#Calculando média do IMC por categoria de obesidade
media_imc = df.groupby('NObeyesdad')['IMC'].mean().sort_values()

# Escolhendo uma paleta de cores para cada categoria
colors = sns.color_palette("deep", n_colors=len(media_imc))

# Plotagem da média de IMC com anotação dos valores
plt.figure(figsize=(10, 6))
bars = plt.bar(media_imc.index, media_imc.values, color=colors)

# Adiciona valor da média em cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.3,
        f'{height:.2f}',
        ha='center',
        va='bottom',
        fontsize=10,
        color='black',
    )

# Configurações do gráfico
plt.title('Média de IMC por Tipo de Obesidade')
plt.xlabel('Tipo de Obesidade')
plt.ylabel('Média de IMC')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Ordenando categorias de obesidade pelo valor médio de IMC
order = df.groupby('NObeyesdad')['IMC'].mean().sort_values().index

# Plotagem do violin horizontal
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=df,
    x='IMC',
    y='NObeyesdad',
    hue='NObeyesdad',
    order=order,
    palette='deep',
    cut=0
)

# Configurações do gráfico
plt.grid(True)
plt.title('Distribuição de IMC por Tipo de Obesidade', fontsize=14, fontweight='normal')
plt.xlabel('IMC')
plt.ylabel('Tipo de Obesidade')
plt.tight_layout()
plt.yticks(rotation=45, ha='right')
plt.show()


#Calculando média do Idade por categoria de obesidade
media = df.groupby('NObeyesdad')['Age'].mean().sort_values()

# Escolhendo uma paleta de cores para cada categoria
colors = sns.color_palette("deep", n_colors=len(media))

# Plotagem da média com anotação dos valores
plt.figure(figsize=(10, 8))
bars = plt.bar(media.index, media.values, color=colors)

# Adiciona valor da média em cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.3,
        f'{height:.2f}',
        ha='center',
        va='bottom',
        fontsize=10
    )

# Configurações do gráfico

plt.title('Média de Idade por Tipo de Obesidade')
plt.xlabel('Tipo de Obesidade')
plt.ylabel('Média de Idade')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Ordenar categorias de obesidade pelo valor médio de BMI
order = df.groupby('NObeyesdad')['Age'].mean().sort_values().index

# Plot do violin plot horizontal
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=df,
    x='Age',
    y='NObeyesdad',
    hue='NObeyesdad',
    order=order,
    palette='deep',
    cut=0
)

# Configurações do gráfico
plt.grid(True)
plt.title('Distribuição de Idade por Tipo de Obesidade', fontsize=14, fontweight='normal')
plt.xlabel('Idade')
plt.ylabel('Tipo de Obesidade')
plt.tight_layout()
plt.yticks(rotation=45, ha='right')
plt.show()


# Relatório estatístico do IMC por categoria de obesidade
df.groupby('NObeyesdad')['IMC'].describe().reset_index().style.background_gradient()


# Criando uma tabela de contingência
cross_tab = pd.crosstab(df['NObeyesdad'], df['MTRANS'])

# Plotando a tabela de contingência como um heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(
    cross_tab,
    annot=True,       # mostra o número dentro de cada célula
    cmap='Blues',     # paleta de azuis
    fmt='d',          # formato inteiro para os números
    cbar=False        # esconde a barra de cor lateral
)

# Adicionando título e rótulos
plt.title('NObeyesdad and MTRANS')
plt.xlabel('')
plt.ylabel('')
plt.show()



# Ordenando as categorias de obesidade
obesity_order = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

# Figura e eixo
plt.figure(figsize=(15, 6))
ax = sns.countplot(
    x='Gender',
    hue='NObeyesdad',
    data=df,
    palette='deep',
    dodge=True,
    hue_order=obesity_order
)

# Título e estilo
plt.title('Distribuição de Tipos de Obesidade por Gênero', fontsize=16, fontweight='normal')
sns.despine(left=True, bottom=False)
plt.xlabel('')
plt.ylabel('Quantidade', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks([])

# Legenda
ax.legend(title='Tipo de Obesidade', title_fontsize=12, fontsize=10, bbox_to_anchor=(0.05, 1), loc='upper left')

# Anotação das barras
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f'{int(height)}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='center',
            xytext=(0, 6),
            textcoords='offset points',
            fontsize=9,
            color='black'
        )

plt.tight_layout()
plt.show()


# importando bibliotecas para pre processamento
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB, CategoricalNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report

from warnings import filterwarnings
filterwarnings('ignore')


# Definindo x e y
X = df.drop(columns=[
    'id','NObeyesdad'
])
y = df['NObeyesdad']

# Colunas numéricas e categóricas
numeric_cols     = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()


# Cria o transformador
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])


# Pré-processamento

# Para GaussianNB
preprocessor_gnb = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Para CategoricalNB
preprocessor_cnb = ColumnTransformer([
    # discretiza em 5 faixas ordinal, gerando valores 0–4
    ('num', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile'), numeric_cols),
    # gera apenas 0 e 1
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])


# Pipelines

# Pipeline com GaussianNB
pipe_gnb = Pipeline([
    ('preproc', preprocessor_gnb),
    ('clf',     GaussianNB())
])

# Pipeline com CategoricalNB
pipe_cnb = Pipeline([
    ('preproc', preprocessor_cnb),
    ('clf',     CategoricalNB())
])



# Separação em treino
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# Treinamento e Avaliação

# Treinar e avaliar GaussianNB
pipe_gnb.fit(X_train, y_train)
y_pred_gnb = pipe_gnb.predict(X_test)
print("GaussianNB Accuracy:", accuracy_score(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb))

# Treinar e avaliar CategoricalNB
pipe_cnb.fit(X_train, y_train)
y_pred_cnb = pipe_cnb.predict(X_test)
print("CategoricalNB Accuracy:", accuracy_score(y_test, y_pred_cnb))
print(classification_report(y_test, y_pred_cnb))


# Validação Cruzada para avaliar o desempenho médio (com 5 folds)
scores = cross_val_score(pipe_gnb, X, y, cv=5, scoring='accuracy')
print("CV GaussianNB:", scores, "→ mean =", scores.mean())


# Pré-processamento: OneHot para categóricas, mantém numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'  # mantém colunas numéricas sem alterações
)


# Pipeline com RandomForest
pipe_rf = Pipeline([
    ('preproc', preprocessor),
    ('clf', RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1))
])


# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# Treinamento
pipe_rf.fit(X_train, y_train)

# Avaliação no conjunto de teste
y_pred = pipe_rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Validação cruzada para avaliar o desempenho do modelo (com 5 folds)
cv_scores = cross_val_score(pipe_rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("Cross-Validation Accuracy: ", cv_scores)
print("Mean CV Accuracy: {:.3f} ± {:.3f}".format(cv_scores.mean(), cv_scores.std()))


# Pré‑processamento
#    - Escala numérica em categorias
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),        numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='drop'
)


# Pipeline com SVM
pipe_svm = Pipeline([
    ('preproc', preprocessor),
    ('clf',     SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])


# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# Treinamento
pipe_svm.fit(X_train, y_train)

# Avaliação no conjunto de teste
y_pred = pipe_svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Validação cruzada para avaliar o desempenho do modelo (com 5 folds)
cv_scores = cross_val_score(pipe_svm, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("Cross-Validation Accuracy:", cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


# Pré-processamento
numeric_proc = StandardScaler()
cat_proc = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_proc, numeric_cols),
        ('cat', cat_proc, categorical_cols)
    ],
    remainder='drop'
)


# Pipeline com k-Nearest Neighbors
pipe_knn = Pipeline([
    ('preproc', preprocessor),
    ('clf', KNeighborsClassifier(n_neighbors=5))  # você pode ajustar o k conforme desejado
])


# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# Treinar e avaliar
pipe_knn.fit(X_train, y_train)
y_pred_knn = pipe_knn.predict(X_test)

print("K-Nearest Neighbors Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


# Validação cruzada para avaliar o desempenho do modelo
param_grid = {'clf__n_neighbors': [3, 5, 7, 9]}
grid_knn = GridSearchCV(pipe_knn, param_grid, cv=5, scoring='accuracy')
grid_knn.fit(X_train, y_train)

print("Melhor valor de k:", grid_knn.best_params_)
print("Melhor acurácia de validação cruzada:", grid_knn.best_score_)


# Pré-processamento
numeric_proc = StandardScaler()
cat_proc = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_proc, numeric_cols),
        ('cat', cat_proc, categorical_cols)
    ],
    remainder='drop'
)


# Pipeline com MLPClassifier
pipe_mlp = Pipeline([
    ('preproc', preprocessor),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(100,),  # uma camada com 100 neurônios
        max_iter=500,               # número de iterações
        random_state=42,
        verbose=False               # se quiser, mude para True para ver o processo de treinamento
    ))
])


# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# Treinar e avaliar
pipe_mlp.fit(X_train, y_train)
y_pred_mlp = pipe_mlp.predict(X_test)

print("MLPClassifier Accuracy:", accuracy_score(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))


# Criando um arquivo de submissão
submission = pd.DataFrame({
    'id': test_df['id'],
    'NObeyesdad': None  # Inicialmente vazio
})

# Preparar X_test exatamente como no treino
X_test = test_df.drop(columns=[
    'id',
])

# Optei pelo Random Forest por ter o melhor resultado
preds = pipe_rf.predict(X_test)

# Preencher a coluna alvo no DataFrame de submissão
submission['NObeyesdad'] = preds

# Salvar o arquivo de submissão no formato exigido
submission.to_csv('submission.csv', index=False)
print("Arquivo submission.csv gerado com sucesso!")
# printando o arquivo de submissão
submission.head()

