# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# Dados sobre as florestas
dados = {
    'nome_floresta': ['Taiga (Floresta Boreal)', 'Floresta AmazÃ´nica'],
    'localizacao': [
        'HemisfÃ©rio Norte (RÃºssia, CanadÃ¡, Alasca, EscandinÃ¡via)',
        'AmÃ©rica do Sul (Brasil, Peru, ColÃ´mbia, etc.)'
    ],
    'tamanho_km2': [12000000, 5500000],
    'populacao': [
        'Comunidades indÃ­genas e pequenas cidades',
        'MilhÃµes de espÃ©cies de plantas, animais e microrganismos'
    ],
    'importancia_climatica': [
        'Maior reservatÃ³rio de carbono terrestre',
        'Filtro de carbono, regulador do ciclo da Ã¡gua'
    ]
}

# Cria um DataFrame do pandas a partir do dicionÃ¡rio de dados
df_florestas = pd.DataFrame(dados)

# Exibe o DataFrame
print("DataFrame com informaÃ§Ãµes das maiores florestas:")
print(df_florestas)

# Exemplo de como acessar uma informaÃ§Ã£o especÃ­fica
print("\n---")
print(f"O tamanho da Taiga Ã© de aproximadamente {df_florestas.loc[0, 'tamanho_km2']} kmÂ².")
print(f"A Floresta AmazÃ´nica Ã© vital para o: {df_florestas.loc[1, 'importancia_climatica']}.")


import pandas as pd

# Dados sobre as maiores florestas do mundo
dados_florestas = {
    'Nome': ['Taiga (Floresta Boreal)', 'Floresta AmazÃ´nica'],
    'LocalizaÃ§Ã£o': [
        'HemisfÃ©rio Norte (RÃºssia, CanadÃ¡, Alasca, EscandinÃ¡via, etc.)',
        'AmÃ©rica do Sul (Brasil, Peru, ColÃ´mbia, etc.)'
    ],
    'Tamanho (kmÂ²)': [12000000, 5500000],
    'PopulaÃ§Ã£o/Biodiversidade': [
        'Habitada por comunidades indÃ­genas e pequenas cidades',
        'MilhÃµes de espÃ©cies de plantas, animais e microrganismos'
    ],
    'ImportÃ¢ncia ClimÃ¡tica': [
        'Maior reservatÃ³rio de carbono terrestre',
        'Filtro de carbono e regulador do ciclo da Ã¡gua'
    ]
}

# Cria um DataFrame do pandas a partir dos dados
df = pd.DataFrame(dados_florestas)

# Exibe o DataFrame
print("Tabela com as maiores florestas do mundo:")
print(df)


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/marcio/Downloads/meu-projeto-chave.json'


import matplotlib.pyplot as plt
import pandas as pd

# Dados das florestas
data = {
    'Nome': ['Taiga', 'Floresta AmazÃ´nica'],
    'Tamanho (milhÃµes de kmÂ²)': [12.0, 5.5],
    'Cor': ['green', 'darkgreen']
}

df = pd.DataFrame(data)

# Criar o grÃ¡fico de barras
plt.figure(figsize=(10, 6))
plt.bar(df['Nome'], df['Tamanho (milhÃµes de kmÂ²)'], color=df['Cor'])

# Adicionar rÃ³tulos e tÃ­tulo
plt.xlabel('Nome da Floresta')
plt.ylabel('Tamanho (milhÃµes de kmÂ²)')
plt.title('Comparativo de Tamanho das Maiores Florestas do Mundo')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar o valor de cada barra no topo
for i, tamanho in enumerate(df['Tamanho (milhÃµes de kmÂ²)']):
    plt.text(i, tamanho + 0.1, f'{tamanho}M kmÂ²', ha='center', va='bottom')

plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# Coordenadas geogrÃ¡ficas aproximadas das florestas
data = {
    'nome': ['Taiga', 'Floresta AmazÃ´nica'],
    'tamanho_km2': [12000000, 5500000],
    'latitude': [60, -5],
    'longitude': [100, -60]
}

df_florestas = pd.DataFrame(data)

# Cria um GeoDataFrame com as coordenadas
gdf_florestas = gpd.GeoDataFrame(
    df_florestas, 
    geometry=gpd.points_from_xy(df_florestas.longitude, df_florestas.latitude)
)

# Carrega os dados do mapa-mÃºndi do GeoPandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Cria o grÃ¡fico
fig, ax = plt.subplots(figsize=(15, 10))

# Plota o mapa-mÃºndi
world.plot(ax=ax, color='lightgray', edgecolor='black')

# Normaliza o tamanho dos pontos para o grÃ¡fico
# O tamanho serÃ¡ proporcional ao tamanho real da floresta
tamanhos_norm = [t / 10000 for t in gdf_florestas['tamanho_km2']]

# Plota os pontos das florestas no mapa
gdf_florestas.plot(
    ax=ax, 
    markersize=tamanhos_norm, 
    color='green', 
    alpha=0.6, 
    label='Floresta'
)

# Adiciona anotaÃ§Ãµes com os nomes das florestas
for x, y, label in zip(gdf_florestas.geometry.x, gdf_florestas.geometry.y, gdf_florestas.nome):
    ax.text(x, y, label, fontsize=12, ha='right')

# ConfiguraÃ§Ãµes do grÃ¡fico
plt.title('LocalizaÃ§Ã£o e Tamanho Relativo das Maiores Florestas do Mundo')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


gdf_florestas


import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# Dados das florestas com coordenadas geogrÃ¡ficas aproximadas
data = {
    'nome': ['Taiga (Floresta Boreal)', 'Floresta AmazÃ´nica'],
    'tamanho_km2': [12000000, 5500000],
    'latitude': [60, -5],
    'longitude': [100, -60]
}

# Cria um DataFrame para organizar os dados
df_florestas = pd.DataFrame(data)

# Converte o DataFrame para um GeoDataFrame, que Ã© especializado em dados geoespaciais
gdf_florestas = gpd.GeoDataFrame(
    df_florestas,
    geometry=gpd.points_from_xy(df_florestas.longitude, df_florestas.latitude)
)

# Carrega os dados do mapa-mÃºndi, que jÃ¡ vÃªm com o GeoPandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Configura o grÃ¡fico e plota o mapa-mÃºndi
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color='lightgray', edgecolor='black')

# Normaliza o tamanho dos pontos para que sejam proporcionais ao tamanho real das florestas
# O fator de 10000 foi escolhido para que os pontos fiquem visÃ­veis no mapa
tamanhos_norm = [t / 10000 for t in gdf_florestas['tamanho_km2']]

# Plota os pontos das florestas no mapa
gdf_florestas.plot(
    ax=ax,
    markersize=tamanhos_norm,
    color='green',
    alpha=0.6,
    label='Floresta'
)

# Adiciona o nome de cada floresta no mapa
for x, y, label in zip(gdf_florestas.geometry.x, gdf_florestas.geometry.y, gdf_florestas.nome):
    ax.text(x, y, label, fontsize=12, ha='right')

# ConfiguraÃ§Ãµes finais do grÃ¡fico
plt.title('LocalizaÃ§Ã£o e Tamanho Relativo das Maiores Florestas do Mundo')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()





# Kaggle jÃ¡ tem o pacote instalado, mas se estiver fora do Kaggle:
!pip install google-cloud-bigquery

from google.cloud import bigquery
import pandas as pd

# Cria o cliente do BigQuery
client = bigquery.Client()


import pandas as pd

data = [
    {
        "nome": "Taiga",
        "localizacao": "RÃºssia, CanadÃ¡, Alasca, Noruega, FinlÃ¢ndia",
        "clima": "SubÃ¡rtico, invernos longos e frios, verÃµes curtos",
        "flora": ["Pinheiros", "Abetos", "Ciprestes", "Cedros", "BÃ©tulas"],
        "fauna": ["Ursos", "Lobos", "Alces", "Tigre-siberiano", "Aves migratÃ³rias"],
        "ameacas": ["ExploraÃ§Ã£o de madeira", "MudanÃ§as climÃ¡ticas", "CaÃ§a", "MineraÃ§Ã£o"]
    },
    {
        "nome": "AmazÃ´nica",
        "localizacao": "Brasil, Peru, ColÃ´mbia, Venezuela, BolÃ­via, Equador",
        "clima": "Equatorial Ãºmido, alta pluviosidade, quente o ano todo",
        "flora": ["Castanheira", "Seringueira", "Mogno", "Pau-rosa", "OrquÃ­deas"],
        "fauna": ["OnÃ§a-pintada", "Boto-cor-de-rosa", "Arara-azul", "PreguiÃ§a", "TamanduÃ¡-bandeira"],
        "ameacas": ["Desmatamento", "Queimadas", "MineraÃ§Ã£o ilegal", "ConstruÃ§Ã£o de hidrelÃ©tricas"]
    }
]

df = pd.DataFrame(data)
df.head()


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 1. Carregar os dados que vocÃª forneceu, adicionando as coordenadas (latitude e longitude)
data = {
    'bioma': ['Taiga', 'AmazÃ´nica'],
    'localizacao': ['RÃºssia, CanadÃ¡, Alasca, Noruega, FinlÃ¢ndia', 'Brasil, Peru, ColÃ´mbia, Venezuela, BolÃ­via, Equador'],
    'clima': ['SubÃ¡rtico, invernos longos e frios, verÃµes curtos', 'Equatorial Ãºmido, alta pluviosidade, quente o ano todo'],
    'latitude': [60, -3],  # Coordenadas aproximadas para a Taiga e a AmazÃ´nia
    'longitude': [100, -60],
    'cor': ['#006400', '#228B22'] # Cor para cada ponto no mapa
}
df = pd.DataFrame(data)

# 2. Converter o DataFrame para um GeoDataFrame (formato geoespacial)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326"
)

# 3. Carregar um mapa-mÃºndi
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# 4. Configurar e plotar o mapa
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.plot(ax=ax, color='lightgray', edgecolor='black')

# 5. Plotar os biomas sobre o mapa
gdf.plot(
    ax=ax,
    marker='o',
    color=gdf['cor'],
    markersize=250,
    edgecolor='black',
    alpha=0.7,
    legend=True
)

# 6. Adicionar os nomes dos biomas ao lado dos pontos
for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf['bioma']):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=12,
        fontweight='bold',
    )

# 7. Adicionar tÃ­tulo e legendas
ax.set_title("Biomas Mundiais", fontsize=20, pad=15)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.show()


SELECT
  nome,
  ST_AREA(geometria) AS area_m2,
  ST_CENTROID(geometria) AS centro
FROM
  `seu_projeto.seu_dataset.florestas_geoespaciais`


# Substitua 'seu-id-do-projeto' pelo ID real do seu projeto.
# Por exemplo: 'meu-projeto-biomas-12345'
project_id = 'seu-id-do-projeto'
# Substitua 'seu-id-do-dataset' pelo ID real do seu dataset.
dataset_id = 'seu-id-do-dataset'
table_id = 'florestas_geoespaciais'

query = f"""
    SELECT
        nome,
        ST_ASGEOJSON(geometria) as geojson,
        localizacao,
        clima,
        flora,
        fauna,
        ameacas
    FROM
        `{project_id}.{dataset_id}.{table_id}`
"""


import folium
from google.cloud import bigquery
import os

# 1. AutenticaÃ§Ã£o e inicializaÃ§Ã£o do cliente BigQuery
# Lembre-se de definir a variÃ¡vel de ambiente ou usar o caminho correto
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'caminho/para/seu/arquivo-chave.json'
client = bigquery.Client()

# 2. Defina o ID da tabela e a query SQL
project_id = 'seu_projeto'
dataset_id = 'seu_dataset'
table_id = 'florestas_geoespaciais'

query = f"""
    SELECT
        nome,
        ST_ASGEOJSON(geometria) as geojson,
        localizacao,
        clima,
        flora,
        fauna,
        ameacas
    FROM
        `{project_id}.{dataset_id}.{table_id}`
"""

# 3. Executar a query e converter o resultado
rows = client.query(query).result()

# Criar uma lista de features GeoJSON a partir dos resultados
geojson_features = []
for row in rows:
    # `row.geojson` jÃ¡ Ã© uma string GeoJSON
    feature = {
        "type": "Feature",
        "geometry": eval(row.geojson),  # Usa eval para converter a string em um dicionÃ¡rio Python
        "properties": {
            "Nome": row.nome,
            "LocalizaÃ§Ã£o": row.localizacao,
            "Clima": row.clima,
            "Flora": ', '.join(row.flora),  # Juntar a lista em uma string
            "Fauna": ', '.join(row.fauna),
            "AmeaÃ§as": ', '.join(row.ameacas)
        }
    }
    geojson_features.append(feature)

geojson_data = {
    "type": "FeatureCollection",
    "features": geojson_features
}

# 4. Criar o mapa com o Folium
m = folium.Map(location=[0, 0], zoom_start=2)

# 5. Adicionar os polÃ­gonos dos biomas ao mapa
folium.GeoJson(
    geojson_data,
    name='Biomas',
    style_function=lambda x: {'fillColor': 'green', 'color': 'black', 'weight': 1.5, 'fillOpacity': 0.5},
    tooltip=folium.GeoJsonTooltip(
        fields=['Nome', 'LocalizaÃ§Ã£o', 'Clima', 'Flora', 'Fauna', 'AmeaÃ§as'],
        aliases=['Nome:', 'LocalizaÃ§Ã£o:', 'Clima:', 'Flora:', 'Fauna:', 'AmeaÃ§as:'],
        localize=True
    )
).add_to(m)

# 6. Salvar o mapa em um arquivo HTML
m.save('mapa_biomas.html')
print("Mapa salvo em 'mapa_biomas.html'. Abra o arquivo no seu navegador.")


CREATE OR REPLACE TABLE seu_projeto.seu_dataset.florestas_geoespaciais (
  nome STRING,
  localizacao STRING,
  clima STRING,
  flora ARRAY<STRING>,
  fauna ARRAY<STRING>,
  ameacas ARRAY<STRING>,
  geometria GEOGRAPHY
);


## ğŸ�� Etapa 2: CÃ³digo Python para Inserir os Dado


from google.cloud import bigquery

# Cria o cliente
client = bigquery.Client()

# Define os dados com geometria aproximada (polÃ­gonos simplificados)
rows = [
    {
        "nome": "Taiga",
        "localizacao": "RÃºssia, CanadÃ¡, Alasca, Noruega, FinlÃ¢ndia",
        "clima": "SubÃ¡rtico, invernos longos e frios, verÃµes curtos",
        "flora": ["Pinheiros", "Abetos", "Ciprestes", "Cedros", "BÃ©tulas"],
        "fauna": ["Ursos", "Lobos", "Alces", "Tigre-siberiano", "Aves migratÃ³rias"],
        "ameacas": ["ExploraÃ§Ã£o de madeira", "MudanÃ§as climÃ¡ticas", "CaÃ§a", "MineraÃ§Ã£o"],
        "geometria": "POLYGON((60 60, 60 -60, -60 -60, -60 60, 60 60))"  # Ã�rea boreal aproximada
    },
    {
        "nome": "AmazÃ´nica",
        "localizacao": "Brasil, Peru, ColÃ´mbia, Venezuela, BolÃ­via, Equador",
        "clima": "Equatorial Ãºmido, alta pluviosidade, quente o ano todo",
        "flora": ["Castanheira", "Seringueira", "Mogno", "Pau-rosa", "OrquÃ­deas"],
        "fauna": ["OnÃ§a-pintada", "Boto-cor-de-rosa", "Arara-azul", "PreguiÃ§a", "TamanduÃ¡-bandeira"],
        "ameacas": ["Desmatamento", "Queimadas", "MineraÃ§Ã£o ilegal", "ConstruÃ§Ã£o de hidrelÃ©tricas"],
        "geometria": "POLYGON((-80 -5, -80 5, -50 5, -50 -5, -80 -5))"  # RegiÃ£o amazÃ´nica aproximada
    }
]

# Define o ID da tabela
table_id = "seu_projeto.seu_dataset.florestas_geoespaciais"

# Insere os dados
errors = client.insert_rows_json(table_id, rows)
if errors == []:
    print("Dados inseridos com sucesso!")
else:
    print("Erros ao inserir:", errors)


SELECT
  nome,
  ST_AREA(geometria) AS area_m2,
  ST_CENTROID(geometria) AS centro
FROM
  `seu_projeto.seu_dataset.florestas_geoespaciais`

