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
import matplotlib.pyplot as plt
import plotly.express as px
import re


file_path = "/kaggle/input/bigquery-ai-hackathon/survey.txt"

def process_survey(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # ExtraÃ§Ã£o de experiÃªncia
    bigquery_exp = re.findall(r"Team Member \d+: (\d+)", content.split("BigQuery AI:")[1].split("Google Cloud:")[0])
    gcloud_exp = re.findall(r"Team Member \d+: (\d+)", content.split("Google Cloud:")[1].split("Feedback:")[0])
    bigquery_exp = list(map(int, bigquery_exp))
    gcloud_exp = list(map(int, gcloud_exp))

    # ExtraÃ§Ã£o de feedback
    feedback_match = re.search(r"Feedback.*?\n\n(.*)", content, re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else "Nenhum feedback encontrado."

    return bigquery_exp, gcloud_exp, feedback

bigquery_exp, gcloud_exp, feedback = process_survey(file_path)


df_exp = pd.DataFrame({
    "BigQuery AI": bigquery_exp,
    "Google Cloud": gcloud_exp
})

fig = px.box(df_exp, title="DistribuiÃ§Ã£o de ExperiÃªncia da Equipe (meses)")
fig.show()


# SimulaÃ§Ã£o de embeddings (substitua por ML.GENERATE_EMBEDDING no BigQuery)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([feedback])

print("âœ… Embedding gerado para anÃ¡lise semÃ¢ntica.")


sql_query = """
SELECT
  IA.FORECAST(
    MODEL => 'linear',
    TABLE => (
      SELECT
        year,
        global_water_availability_liters
      FROM
        `public_dataset.global_water_data`
    ),
    TARGET_COLUMN => 'global_water_availability_liters',
    HORIZON => 30
  ) AS forecast
"""

# Se estiver usando uma biblioteca para executar a query, a chamada seria algo como:
# df = client.query(sql_query).to_dataframe()


from google.cloud import bigquery

# Inicialize o cliente do BigQuery
# Certifique-se de que sua autenticaÃ§Ã£o com o Google Cloud esteja configurada
client = bigquery.Client()

# Defina a consulta SQL como uma string de vÃ¡rias linhas
# Usamos as aspas triplas (""") para isso
sql_query = """
WITH water_data AS (
  SELECT 'Brasil' AS country, 8233 AS freshwater_reserves_km3, 215313498 AS population UNION ALL
  SELECT 'RÃºssia' AS country, 4508 AS freshwater_reserves_km3, 145975300 AS population UNION ALL
  SELECT 'CanadÃ¡' AS country, 2902 AS freshwater_reserves_km3, 38246108 AS population
)
SELECT
  country,
  freshwater_reserves_km3,
  population
FROM
  water_data
ORDER BY
  freshwater_reserves_km3 DESC;
"""

# Execute a consulta e armazene o resultado em um DataFrame do Pandas
# `to_dataframe()` transforma o resultado da consulta em um formato fÃ¡cil de manipular
df = client.query(sql_query).to_dataframe()

# Imprima o DataFrame resultante
print(df)


import pandas as pd
import plotly.express as px
import plotly.io as pio

# Configura o renderizador do Plotly para funcionar no Kaggle Notebooks
pio.renderers.default = 'iframe'

# 1. Preparar os dados para o mapa
# Este DataFrame simula os dados que vocÃª obteria do BigQuery
# com informaÃ§Ãµes sobre as maiores reservas de Ã¡gua potÃ¡vel no mundo.
data = {
    'country': ['Brasil', 'RÃºssia', 'CanadÃ¡', 'China', 'Ã�ndia', 'ColÃ´mbia', 'IndonÃ©sia', 'Estados Unidos', 'Myanmar', 'Congo (RDC)'],
    'freshwater_reserves_km3': [8233, 4508, 2902, 2800, 1897, 1860, 1690, 1530, 1070, 1000],
    'population': [215313498, 145975300, 38246108, 1412000000, 1380000000, 51000000, 275000000, 331000000, 54000000, 95000000],
    'lat': [-14.2350, 61.5240, 56.1304, 35.8617, 20.5937, 4.5709, -0.7893, 37.0902, 21.9139, -4.0383],
    'lon': [-51.9253, 105.3188, -106.3468, 104.1954, 78.9629, -74.2973, 113.9213, -95.7129, 95.9562, 21.7587],
    'continent': ['AmÃ©rica do Sul', 'Ã�sia', 'AmÃ©rica do Norte', 'Ã�sia', 'Ã�sia', 'AmÃ©rica do Sul', 'Ã�sia', 'AmÃ©rica do Norte', 'Ã�sia', 'Ã�frica']
}
df_reserva = pd.DataFrame(data)

# 2. Criar o mapa interativo usando Plotly Express
# O tamanho do ponto serÃ¡ proporcional Ã s reservas de Ã¡gua potÃ¡vel.
# A cor do ponto serÃ¡ determinada pelo continente.
# As informaÃ§Ãµes detalhadas aparecerÃ£o ao passar o mouse.
fig = px.scatter_geo(
    df_reserva,
    lat='lat',          # Coluna para a latitude
    lon='lon',          # Coluna para a longitude
    hover_name='country', # Nome do paÃ­s ao passar o mouse
    size='freshwater_reserves_km3', # Tamanho do ponto proporcional Ã s reservas
    color='continent',  # Cor do ponto baseada no continente
    projection='natural earth', # ProjeÃ§Ã£o do mapa-mÃºndi
    title='Maiores Reservas de Ã�gua PotÃ¡vel do Mundo' # TÃ­tulo do mapa
)

# 3. Personalizar a informaÃ§Ã£o exibida ao passar o mouse (tooltip)
# Isso permite mostrar as reservas em kmÂ³, populaÃ§Ã£o e continente de forma clara.
fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br><br>" +
                  "Reservas: %{marker.size} kmÂ³<br>" +
                  "PopulaÃ§Ã£o: %{customdata[0]:,}<br>" +
                  "Continente: %{customdata[1]}"
)

# Adiciona os dados de populaÃ§Ã£o e continente para serem exibidos no tooltip
fig.update_traces(customdata=df_reserva[['population', 'continent']])

# 4. Exibir o mapa
# No Kaggle, Ã© necessÃ¡rio usar 'Save & Run All (Commit)' para ver o mapa renderizado.
fig.show()




import pandas as pd
import plotly.express as px
import plotly.io as pio

# Configura o renderizador do Plotly para funcionar no Kaggle Notebooks
pio.renderers.default = 'iframe'

# 1. Preparar os dados para o mapa
# Este DataFrame simula os dados que vocÃª obteria do BigQuery
# com informaÃ§Ãµes sobre as maiores reservas de Ã¡gua potÃ¡vel no mundo.
data = {
    'country': ['Brasil', 'RÃºssia', 'CanadÃ¡', 'China', 'Ã�ndia', 'ColÃ´mbia', 'IndonÃ©sia', 'Estados Unidos', 'Myanmar', 'Congo (RDC)'],
    'freshwater_reserves_km3': [8233, 4508, 2902, 2800, 1897, 1860, 1690, 1530, 1070, 1000],
    'population': [215313498, 145975300, 38246108, 1412000000, 1380000000, 51000000, 275000000, 331000000, 54000000, 95000000],
    'lat': [-14.2350, 61.5240, 56.1304, 35.8617, 20.5937, 4.5709, -0.7893, 37.0902, 21.9139, -4.0383],
    'lon': [-51.9253, 105.3188, -106.3468, 104.1954, 78.9629, -74.2973, 113.9213, -95.7129, 95.9562, 21.7587],
    'continent': ['AmÃ©rica do Sul', 'Ã�sia', 'AmÃ©rica do Norte', 'Ã�sia', 'Ã�sia', 'AmÃ©rica do Sul', 'Ã�sia', 'AmÃ©rica do Norte', 'Ã�sia', 'Ã�frica']
}
df_reserva = pd.DataFrame(data)

# 2. Criar o mapa interativo usando Plotly Express
# O tamanho do ponto serÃ¡ proporcional Ã s reservas de Ã¡gua potÃ¡vel.
# A cor do ponto serÃ¡ determinada pelo continente.
# As informaÃ§Ãµes detalhadas aparecerÃ£o ao passar o mouse.
fig = px.scatter_geo(
    df_reserva,
    lat='lat',          # Coluna para a latitude
    lon='lon',          # Coluna para a longitude
    hover_name='country', # Nome do paÃ­s ao passar o mouse
    size='freshwater_reserves_km3', # Tamanho do ponto proporcional Ã s reservas
    color='continent',  # Cor do ponto baseada no continente
    projection='natural earth', # ProjeÃ§Ã£o do mapa-mÃºndi
    title='Maiores Reservas de Ã�gua PotÃ¡vel do Mundo' # TÃ­tulo do mapa
)

# 3. Personalizar a informaÃ§Ã£o exibida ao passar o mouse (tooltip)
# Isso permite mostrar as reservas em kmÂ³, populaÃ§Ã£o e continente de forma clara.
fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br><br>" +
                  "Reservas: %{marker.size} kmÂ³<br>" +
                  "PopulaÃ§Ã£o: %{customdata[0]:,}<br>" +
                  "Continente: %{customdata[1]}"
)

# Adiciona os dados de populaÃ§Ã£o e continente para serem exibidos no tooltip
fig.update_traces(customdata=df_reserva[['population', 'continent']])

# 4. Exibir o mapa
# No Kaggle, Ã© necessÃ¡rio usar 'Save & Run All (Commit)' para ver o mapa renderizado.
fig.show()



import pandas as pd
import plotly.express as px

# 1. Preparar os dados para o mapa
# Este DataFrame simula os dados que vocÃª obteria do BigQuery
# com informaÃ§Ãµes sobre as maiores reservas de Ã¡gua potÃ¡vel.
data = {
    'country': ['Brasil', 'RÃºssia', 'CanadÃ¡', 'China', 'Ã�ndia', 'ColÃ´mbia', 'IndonÃ©sia', 'Estados Unidos'],
    'freshwater_reserves_km3': [8233, 4508, 2902, 2800, 1897, 1860, 1690, 1530],
    'population': [215313498, 145975300, 38246108, 1412000000, 1380000000, 51000000, 275000000, 331000000],
    'lat': [-14.2350, 61.5240, 56.1304, 35.8617, 20.5937, 4.5709, -0.7893, 37.0902],
    'lon': [-51.9253, 105.3188, -106.3468, 104.1954, 78.9629, -74.2973, 113.9213, -95.7129],
    'continent': ['AmÃ©rica do Sul', 'Ã�sia', 'AmÃ©rica do Norte', 'Ã�sia', 'Ã�sia', 'AmÃ©rica do Sul', 'Ã�sia', 'AmÃ©rica do Norte']
}
df_reserva = pd.DataFrame(data)

# 2. Criar o mapa interativo
# Usamos px.scatter_geo para plotar os pontos nos paÃ­ses
fig = px.scatter_geo(
    df_reserva,
    lat='lat',
    lon='lon',
    hover_name='country',
    size='freshwater_reserves_km3',
    color='continent',
    projection='natural earth',
    title='Maiores Reservas de Ã�gua PotÃ¡vel do Mundo'
)

# 3. Personalizar a informaÃ§Ã£o exibida ao passar o mouse (tooltip)
fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br><br>" +
                  "Reservas: %{marker.size} kmÂ³<br>" +
                  "PopulaÃ§Ã£o: %{customdata[0]:,}<br>" +
                  "Continente: %{customdata[1]}"
)
# Adicionar dados de populaÃ§Ã£o e continente para o tooltip
fig.update_traces(customdata=df_reserva[['population', 'continent']])

# 4. Exibir o mapa
fig.show()


import pandas as pd
import plotly.express as px

# 1. Preparar os dados para o mapa
# Este DataFrame simula os dados que vocÃª obteria do BigQuery
# com informaÃ§Ãµes sobre as maiores reservas de Ã¡gua potÃ¡vel.
data = {
    'country': ['Brasil', 'RÃºssia', 'CanadÃ¡', 'China', 'Ã�ndia', 'ColÃ´mbia', 'IndonÃ©sia', 'Estados Unidos'],
    'freshwater_reserves_km3': [8233, 4508, 2902, 2800, 1897, 1860, 1690, 1530],
    'population': [215313498, 145975300, 38246108, 1412000000, 1380000000, 51000000, 275000000, 331000000],
    'lat': [-14.2350, 61.5240, 56.1304, 35.8617, 20.5937, 4.5709, -0.7893, 37.0902],
    'lon': [-51.9253, 105.3188, -106.3468, 104.1954, 78.9629, -74.2973, 113.9213, -95.7129],
    'continent': ['AmÃ©rica do Sul', 'Ã�sia', 'AmÃ©rica do Norte', 'Ã�sia', 'Ã�sia', 'AmÃ©rica do Sul', 'Ã�sia', 'AmÃ©rica do Norte']
}
df_reserva = pd.DataFrame(data)

# 2. Criar o mapa interativo
# Usamos px.scatter_geo para plotar os pontos nos paÃ­ses
fig = px.scatter_geo(
    df_reserva,
    lat='lat',
    lon='lon',
    hover_name='country',
    size='freshwater_reserves_km3',
    color='continent',
    projection='natural earth',
    title='Maiores Reservas de Ã�gua PotÃ¡vel do Mundo'
)

# 3. Personalizar a informaÃ§Ã£o exibida ao passar o mouse (tooltip)
fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br><br>" +
                  "Reservas: %{marker.size} kmÂ³<br>" +
                  "PopulaÃ§Ã£o: %{customdata[0]:,}<br>" +
                  "Continente: %{customdata[1]}"
)
# Adicionar dados de populaÃ§Ã£o e continente para o tooltip
fig.update_traces(customdata=df_reserva[['population', 'continent']])

# 4. Exibir o mapa
fig.show()


# SimulaÃ§Ã£o de dados de previsÃ£o
years = list(range(2025, 2055))
availability = [100 - i*2 for i in range(30)]  # queda linear simulada

df_forecast = pd.DataFrame({
    "Ano": years,
    "Disponibilidade de Ã�gua (%)": availability
})

plt.figure(figsize=(10, 5))
plt.plot(df_forecast["Ano"], df_forecast["Disponibilidade de Ã�gua (%)"], marker='o')
plt.title("PrevisÃ£o de Disponibilidade Global de Ã�gua PotÃ¡vel")
plt.xlabel("Ano")
plt.ylabel("Disponibilidade (%)")
plt.grid(True)
plt.show()

