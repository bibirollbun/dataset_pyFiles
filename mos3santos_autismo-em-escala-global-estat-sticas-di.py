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
import numpy as np
import matplotlib.pyplot as plt

# Criar anos de projeÃ§Ã£o
anos = np.arange(2025, 2076)

# Simular crescimento de diagnÃ³sticos (% da populaÃ§Ã£o)
diagnosticos = np.linspace(1.5, 4.0, len(anos))  # Crescimento gradual

# Simular inclusÃ£o social (Ã­ndice de 0 a 100)
inclusao_social = np.linspace(40, 95, len(anos))  # Aumento da aceitaÃ§Ã£o

# Simular uso de IA em terapias (Ã­ndice de 0 a 100)
uso_ia = np.linspace(10, 90, len(anos))  # Crescimento acelerado

# Criar DataFrame
df = pd.DataFrame({
    'Ano': anos,
    'Diagnosticos (%)': diagnosticos,
    'Inclusao Social (%)': inclusao_social,
    'Uso de IA (%)': uso_ia
})

# Visualizar os dados
df.head()


!pip install google-cloud-bigquery


!pip install pandas numpy matplotlib google-cloud-bigquery


!export GOOGLE_APPLICATION_CREDENTIALS="caminho/para/sua-chave.json"


import pandas as pd
import numpy as np

# Simular projeÃ§Ãµes de 2025 a 2075
anos = np.arange(2025, 2076)
diagnosticos = np.linspace(1.5, 4.0, len(anos))  # % da populaÃ§Ã£o
inclusao_social = np.linspace(40, 95, len(anos))  # Ã­ndice de aceitaÃ§Ã£o
uso_ia = np.linspace(10, 90, len(anos))  # uso de IA em terapias

# Criar DataFrame
df = pd.DataFrame({
    'Ano': anos,
    'Diagnosticos_percentual': diagnosticos,
    'Inclusao_social_indice': inclusao_social,
    'Uso_IA_terapias_indice': uso_ia
})


from google.cloud import bigquery

# Inicializar cliente
client = bigquery.Client()



# Definir ID da tabela (substitua pelos seus dados)
table_id = "seu_projeto.autismo_dataset.projecoes_50_anos"


'''Enviar DataFrame para o BigQuery'
'job = client.load_table_from_dataframe(df, table_id)
'job.result()  # Aguarda finalizaÃ§Ã£o'''



'print("âœ… Dados enviados com sucesso para o BigQuery!")'


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['Ano'], df['Diagnosticos_percentual'], label='DiagnÃ³sticos (%)', color='blue')
plt.plot(df['Ano'], df['Inclusao_social_indice'], label='InclusÃ£o Social (%)', color='green')
plt.plot(df['Ano'], df['Uso_IA_terapias_indice'], label='Uso de IA (%)', color='purple')
plt.title('ProjeÃ§Ãµes sobre Autismo para os PrÃ³ximos 50 Anos')
plt.xlabel('Ano')
plt.ylabel('Percentual / Ã�ndice')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# ğŸ”¢ Dados fictÃ­cios para projeÃ§Ã£o
anos = list(range(2025, 2076, 5))
diagnosticos = [1, 3, 5, 8, 12, 16, 20, 23, 24, 25, 25]
inclusao_social = [5, 10, 18, 27, 35, 45, 55, 60, 65, 68, 70]
uso_ia = [0, 5, 12, 20, 30, 45, 60, 70, 75, 78, 80]

# ğŸ“Š Criando o DataFrame
df = pd.DataFrame({
    'Ano': anos,
    'Diagnosticos (%)': diagnosticos,
    'Inclusao Social (%)': inclusao_social,
    'Uso de IA (%)': uso_ia
})

# ğŸ�¨ Criando o grÃ¡fico
plt.figure(figsize=(12, 6))
plt.plot(df['Ano'], df['Diagnosticos (%)'], label='DiagnÃ³sticos (%)', color='blue', marker='o')
plt.plot(df['Ano'], df['Inclusao Social (%)'], label='InclusÃ£o Social (%)', color='green', marker='s')
plt.plot(df['Ano'], df['Uso de IA (%)'], label='Uso de IA (%)', color='purple', marker='^')

plt.title('ğŸ“ˆ ProjeÃ§Ãµes sobre Autismo para os PrÃ³ximos 50 Anos', fontsize=14)
plt.xlabel('Ano')
plt.ylabel('Percentual / Ã�ndice')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# ğŸ“‹ Lista de direitos
direitos = [
    'Atendimento prioritÃ¡rio',
    'EducaÃ§Ã£o inclusiva',
    'Carteira CIPTEA',
    'BenefÃ­cio BPC/LOAS',
    'IsenÃ§Ã£o de impostos',
    'Transporte gratuito',
    'ReduÃ§Ã£o de jornada (pais)'
]

# ğŸ“Š Estimativa fictÃ­cia de beneficiÃ¡rios (em milhares)
beneficiarios = [80, 70, 60, 50, 40, 30, 20]

# ğŸ�¨ Cores para cada barra
cores = ['#4B8BBE', '#306998', '#FFE873', '#FFD43B', '#646464', '#9C27B0', '#00BFA5']

# ğŸ“ˆ Criando o grÃ¡fico
plt.figure(figsize=(12, 6))
bars = plt.barh(direitos, beneficiarios, color=cores)

# ğŸ�·ï¸� Adicionando valores nas barras
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2,
             f'{width} mil', va='center', fontsize=10)

plt.title('âš–ï¸� Direitos das Pessoas com Autismo - Lei 12.764/2012', fontsize=14)
plt.xlabel('Estimativa de BeneficiÃ¡rios (milhares)')
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.show()


import matplotlib.pyplot as plt

# ğŸŒ� PaÃ­ses e estimativas de pessoas com autismo (em milhÃµes)
paises = ['Estados Unidos ğŸ‡ºğŸ‡¸', 'Dinamarca ğŸ‡©ğŸ‡°', 'CanadÃ¡ ğŸ‡¨ğŸ‡¦', 'AustrÃ¡lia ğŸ‡¦ğŸ‡º', 'Brasil ğŸ‡§ğŸ‡·']
estimativas_milhoes = [5.0, 0.3, 0.5, 0.4, 2.0]

# ğŸ�¨ Cores personalizadas
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# ğŸ“ˆ Criando o grÃ¡fico
plt.figure(figsize=(10, 6))
bars = plt.bar(paises, estimativas_milhoes, color=cores)

# ğŸ�·ï¸� Adicionando valores nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.1,
             f'{height}M', ha='center', va='bottom', fontsize=10)

# ğŸ§  Adicionando anotaÃ§Ã£o sobre prevalÃªncia nos EUA
plt.annotate('1 em cada 36 crianÃ§as nos EUA',
             xy=('Estados Unidos ğŸ‡ºğŸ‡¸', 5.0), xytext=(1, 5.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10)

plt.title('ğŸŒ� EstatÃ­sticas Globais sobre o Autismo', fontsize=14)
plt.ylabel('Estimativa de Pessoas com Autismo (milhÃµes)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


import pandas as pd

# Carregar arquivo
df = pd.read_csv('/kaggle/input/bigquery-ai-hackathon/survey.txt', sep='\t')

# Visualizar primeiras linhas
print(df.head())


import pandas as pd
import plotly.express as px

# SimulaÃ§Ã£o de dados com latitude e longitude
data = {
    'Country': ['United States', 'Brazil', 'Canada', 'Australia', 'Denmark'],
    'Autism_Prevalence': [280, 180, 230, 220, 250],  # por 10.000 habitantes
    'Latitude': [37.0902, -14.2350, 56.1304, -25.2744, 56.2639],
    'Longitude': [-95.7129, -51.9253, -106.3468, 133.7751, 9.5018]
}

df = pd.DataFrame(data)

# Plotar mapa com pontos geogrÃ¡ficos
fig = px.scatter_geo(df,
                     lat='Latitude',
                     lon='Longitude',
                     text='Country',
                     size='Autism_Prevalence',
                     color='Autism_Prevalence',
                     projection='natural earth',
                     title='PrevalÃªncia de Autismo por PaÃ­s (por 10.000 habitantes)',
                     color_continuous_scale='Viridis')

fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
fig.show()


# Leitura do arquivo survey.txt
df_survey = pd.read_csv('/kaggle/input/bigquery-ai-hackathon/survey.txt', sep='\t')

# Exibir colunas disponÃ­veis
print(df_survey.columns)

# Exibir amostra dos dados
print(df_survey.head())


import pandas as pd
import plotly.express as px

# Dados simulados com informaÃ§Ãµes adicionais
data = {
    'PaÃ­s': ['Estados Unidos', 'Brasil', 'CanadÃ¡', 'AustrÃ¡lia', 'Dinamarca'],
    'CÃ³digo': ['USA', 'BRA', 'CAN', 'AUS', 'DNK'],
    'PrevalÃªncia por 10.000 hab.': [280, 180, 230, 220, 250],
    'PopulaÃ§Ã£o Estimada com Autismo': ['9 milhÃµes', '2 milhÃµes', '850 mil', '600 mil', '300 mil'],
    'Latitude': [37.0902, -14.2350, 56.1304, -25.2744, 56.2639],
    'Longitude': [-95.7129, -51.9253, -106.3468, 133.7751, 9.5018]
}

df = pd.DataFrame(data)

# Criar mapa interativo
fig = px.scatter_geo(
    df,
    lat='Latitude',
    lon='Longitude',
    hover_name='PaÃ­s',
    hover_data={
        'PrevalÃªncia por 10.000 hab.': True,
        'PopulaÃ§Ã£o Estimada com Autismo': True,
        'Latitude': False,
        'Longitude': False
    },
    size='PrevalÃªncia por 10.000 hab.',
    color='PrevalÃªncia por 10.000 hab.',
    color_continuous_scale='Plasma',
    projection='natural earth',
    title='ğŸŒ� PrevalÃªncia de Autismo por PaÃ­s (Estimativa Global)'
)

# Estilizar marcadores
fig.update_traces(marker=dict(line=dict(width=1, color='black')))

# Atualizar layout com legenda explicativa
fig.update_layout(
    legend_title_text='PrevalÃªncia por 10.000 habitantes',
    geo=dict(
        showland=True,
        landcolor='rgb(217, 217, 217)',
        showcountries=True,
        countrycolor='gray'
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

fig.show()


import pandas as pd
import plotly.express as px

# Exemplo de dados fictÃ­cios
data = {
    'Country': ['United States', 'Brazil', 'Canada', 'Australia', 'Denmark'],
    'Autism_Prevalence': [2.8, 1.0, 1.5, 1.7, 2.0]  # Percentual estimado da populaÃ§Ã£o
}

df = pd.DataFrame(data)

# Plotar mapa
fig = px.choropleth(df,
                    locations='Country',
                    locationmode='country names',
                    color='Autism_Prevalence',
                    color_continuous_scale='Blues',
                    title='PrevalÃªncia de Autismo por PaÃ­s (%)')
fig.show()

