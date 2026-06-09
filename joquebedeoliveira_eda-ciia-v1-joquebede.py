import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

# Configurações
plt.style.use('ggplot')
pd.set_option('display.max_columns', 100)

# Carregar dados
train_labels = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
print("Dimensões do dataset:", train_labels.shape)


print("\nInformações das colunas:")
print(train_labels.info())

print("\nValores únicos por coluna:")
for col in train_labels.columns:
    print(f"{col}: {train_labels[col].nunique()} valores únicos")

print("\nDistribuição da coluna 'type':")
print(train_labels['type'].value_counts(normalize=True))

# Visualização
plt.figure(figsize=(10,6))
train_labels['type'].value_counts().plot(kind='bar')
plt.title('Distribuição dos Tipos de Menção')
plt.xticks(rotation=45)
plt.show()


# Criar grafo de relações
G = nx.Graph()

for _, row in train_labels.iterrows():
    G.add_node(row['article_id'], type='article')
    G.add_node(row['dataset_id'], type='dataset')
    G.add_edge(row['article_id'], row['dataset_id'], mention_type=row['type'])

# Métricas básicas
print("\nEstatísticas da Rede:")
print(f"- Artigos: {sum(1 for n, d in G.nodes(data=True) if d['type']=='article')}")
print(f"- Datasets: {sum(1 for n, d in G.nodes(data=True) if d['type']=='dataset')}")
print(f"- Conexões: {G.number_of_edges()}")

# Centralidade
degree_centrality = nx.degree_centrality(G)
top_datasets = sorted(
    [n for n, d in G.nodes(data=True) if d['type'] == 'dataset'],
    key=lambda x: degree_centrality[x], 
    reverse=True
)[:5]

print("\nTop 5 datasets mais mencionados:")
for ds in top_datasets:
    print(f"{ds}: {G.degree(ds)} menções")


# Configuração do TF-IDF para análise de caracteres
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))

# Amostra de 1000 dataset_ids para análise (ou usar todos se for menos que 1000)
sample_size = min(1000, len(train_labels))
tfidf_matrix = tfidf.fit_transform(train_labels['dataset_id'].sample(sample_size, random_state=42))

# Cálculo de frequência dos n-grams
ngrams_freq = pd.Series(
    dict(zip(tfidf.get_feature_names_out(), tfidf_matrix.sum(axis=0).A1))  # Parêntese fechado aqui
)  # Parêntese adicional fechado aqui

# Visualização dos 20 n-grams mais comuns
plt.figure(figsize=(12, 8))
ngrams_freq.sort_values(ascending=False).head(20).plot(kind='barh', color='teal')
plt.title('20 Padrões de Caracteres Mais Frequentes em Dataset IDs', pad=20)
plt.xlabel('Frequência')
plt.ylabel('N-gram de Caracteres')
plt.grid(axis='x', alpha=0.3)
plt.show()

# Análise adicional: tamanho médio dos IDs
print("\nEstatísticas de Tamanho dos Dataset IDs:")
train_labels['id_length'] = train_labels['dataset_id'].str.len()
print(train_labels['id_length'].describe())

# Distribuição de comprimentos
plt.figure(figsize=(10, 6))
sns.histplot(train_labels['id_length'], bins=30, kde=True)
plt.title('Distribuição de Comprimento dos Dataset IDs')
plt.xlabel('Número de Caracteres')
plt.ylabel('Frequência')
plt.show()


# Tentativa de extrair ano (adaptar)
train_labels['year'] = train_labels['article_id'].str.extract(r'(\d{4})')

if not train_labels['year'].isnull().all():
    plt.figure(figsize=(12,6))
    train_labels.groupby('year').size().plot(kind='bar')
    plt.title('Menções por Ano')
    plt.show()
    
    # Evolução dos tipos
    if 'type' in train_labels.columns:
        pd.crosstab(train_labels['year'], train_labels['type']).plot(kind='area')
        plt.title('Evolução dos Tipos de Menção')
        plt.show()
else:
    print("\nNão foi possível extrair informação temporal dos IDs")


from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Preparar dados para clustering
article_degrees = pd.Series({n: G.degree(n) 
                           for n, d in G.nodes(data=True) 
                           if d['type'] == 'article'})

plt.figure(figsize=(10,6))
sns.histplot(article_degrees, bins=50)
plt.title('Distribuição de Menções por Artigo')
plt.show()

# Exemplo de clustering (simplificado)
if len(article_degrees) > 100:
    kmeans = KMeans(n_clusters=3).fit(article_degrees.values.reshape(-1,1))
    print("\nGrupos de artigos por frequência de menções:")
    print(pd.Series(kmeans.labels_).value_counts())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Preparar dados de exemplo (substituir por features reais)
X = pd.DataFrame({
    'article_degree': [G.degree(a) for a in train_labels['article_id']],
    'dataset_degree': [G.degree(d) for d in train_labels['dataset_id']],
    'type_encoded': pd.factorize(train_labels['type'])[0]  # Encoding simples
})

y = train_labels['type']

# Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Modelo baseline
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)

# Avaliação
predictions = baseline_model.predict(X_test)
print(classification_report(y_test, predictions))

# Feature importance
plt.figure(figsize=(10, 6))
pd.Series(baseline_model.feature_importances_, index=X.columns).sort_values().plot(kind='barh')
plt.title('Importância das Features no Modelo Baseline')
plt.show()


print("\n╔════════════════════════════════════════════╗")
print("║             INSIGHTS APRIMORADOS          ║")
print("╚════════════════════════════════════════════╝")

# 1. Análise de distribuição com visualização
type_dist = train_labels['type'].value_counts(normalize=True)
print("\n1. Distribuição de Tipos de Menção (com análise de desbalanceamento):")
plt.figure(figsize=(10,6))
bars = plt.bar(type_dist.index, type_dist.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Distribuição de Tipos de Menção', pad=20)
plt.xlabel('Tipo de Menção')
plt.ylabel('Proporção')
plt.xticks(rotation=45)

# Adicionar rótulos de porcentagem
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1%}', ha='center', va='bottom')

plt.show()

# 2. Análise de datasets com métricas de rede
print("\n2. Top 5 Datasets Mais Mencionados (com métricas de centralidade):")
top_datasets_analysis = []
for ds in top_datasets[:5]:
    degree = G.degree(ds)
    centrality = nx.degree_centrality(G).get(ds, 0)
    betweenness = nx.betweenness_centrality(G).get(ds, 0)
    top_datasets_analysis.append({
        'Dataset': ds,
        'Grau': degree,
        'Centralidade': f"{centrality:.4f}",
        'Betweenness': f"{betweenness:.4f}",
        'Artigos Únicos': len(set(train_labels[train_labels['dataset_id'] == ds]['article_id']))
    })

display(pd.DataFrame(top_datasets_analysis).style.highlight_max(color='lightgreen'))

