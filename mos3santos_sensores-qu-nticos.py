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


pip install qiskit qiskit-aer google-cloud-bigquery


file_path = '/kaggle/input/bigquery-ai-hackathon/survey.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
        print("ConteÃºdo do arquivo (string inteira):")
        print(file_content)
except FileNotFoundError:
    print(f"Erro: O arquivo nÃ£o foi encontrado em {file_path}")
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")


import pandas as pd
import matplotlib.pyplot as plt
import folium

# SimulaÃ§Ã£o de dados extraÃ­dos do survey.txt
data = {
    'Team Member': ['Alice', 'Bob', 'Carlos', 'Diana', 'Eva', 'FÃ¡bio'],
    'Country': ['USA', 'USA', 'Brazil', 'Germany', 'Brazil', 'India'],
    'BigQuery_AI_Experience': [12, 8, 15, 10, 20, 5],
    'Google_Cloud_Experience': [18, 10, 24, 12, 30, 8]
}

df = pd.DataFrame(data)

# Agrupando por paÃ­s e calculando mÃ©dias
country_stats = df.groupby('Country')[['BigQuery_AI_Experience', 'Google_Cloud_Experience']].mean().sort_values(by='BigQuery_AI_Experience', ascending=False)

# ğŸ“Š GrÃ¡fico de barras
country_stats.plot(kind='bar', figsize=(10,6), title='MÃ©dia de ExperiÃªncia por PaÃ­s')
plt.ylabel('Meses de ExperiÃªncia')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ğŸŒ� Mapa interativo com folium
# Coordenadas aproximadas dos paÃ­ses
country_coords = {
    'USA': [37.0902, -95.7129],
    'Brazil': [-14.2350, -51.9253],
    'Germany': [51.1657, 10.4515],
    'India': [20.5937, 78.9629]
}

# Criando mapa
m = folium.Map(location=[0, 0], zoom_start=2)

for country, row in country_stats.iterrows():
    lat, lon = country_coords[country]
    popup_text = f"{country}<br>BigQuery AI: {row['BigQuery_AI_Experience']:.1f} meses<br>Google Cloud: {row['Google_Cloud_Experience']:.1f} meses"
    folium.CircleMarker(location=[lat, lon],
                        radius=row['BigQuery_AI_Experience'] / 2,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        popup=popup_text).add_to(m)

# Exibir mapa
m.save("mapa_experiencia.html")


!pip install -q kaleido


df.to_csv("dados_sensores_quanticos.csv", index=False)


df


pip install plotly --upgrade


pip install kaleido==0.2.1


!pip install -U kaleido


import plotly.express as px

# Dados de exemplo
dados = {
    "Categoria": ["A", "B", "C", "D"],
    "Valor": [10, 15, 7, 12]
}

# Criar o grÃ¡fico de barras
fig_bar = px.bar(dados, x="Categoria", y="Valor", title="GrÃ¡fico de Barras")

# Exportar para HTML
fig_bar.write_html("grafico_barras.html")


import pandas as pd
import plotly.express as px
import plotly.io as pio # Import plotly.io

# ğŸ”¬ Dados estruturados
data = {
    'Ã�rea': [
        'Medicina e SaÃºde', 'Medicina e SaÃºde', 'Medicina e SaÃºde',
        'EspaÃ§o e NavegaÃ§Ã£o', 'EspaÃ§o e NavegaÃ§Ã£o', 'EspaÃ§o e NavegaÃ§Ã£o',
        'Defesa e SeguranÃ§a', 'Defesa e SeguranÃ§a', 'Defesa e SeguranÃ§a',
        'Clima e Meio Ambiente', 'Clima e Meio Ambiente', 'Clima e Meio Ambiente'
    ],
    'PaÃ­s': [
        'EUA', 'Alemanha', 'JapÃ£o',
        'EUA', 'China', 'FranÃ§a',
        'EUA', 'Israel', 'Reino Unido',
        'Alemanha', 'Brasil', 'CanadÃ¡'
    ],
    'Latitude': [
        37.0902, 51.1657, 36.2048,
        37.0902, 35.8617, 46.2276,
        37.0902, 31.0461, 55.3781,
        51.1657, -14.2350, 56.1304
    ],
    'Longitude': [
        -95.7129, 10.4515, 138.2529,
        -95.7129, 104.1954, 2.2137,
        -95.7129, 34.8516, -3.4360,
        10.4515, -51.9253, -106.3468
    ]
}

df = pd.DataFrame(data)

# ğŸ“Š GrÃ¡fico de barras interativo
fig_bar = px.bar(df, x='PaÃ­s', color='Ã�rea',
                 title='PaÃ­ses por Ã�rea de AplicaÃ§Ã£o de Sensores QuÃ¢nticos',
                 labels={'PaÃ­s': 'PaÃ­ses', 'Ã�rea': 'Ã�rea de AplicaÃ§Ã£o'},
                 height=500)
fig_bar.show()

# ğŸ¥§ GrÃ¡fico de pizza por Ã¡rea
area_counts = df['Ã�rea'].value_counts().reset_index()
area_counts.columns = ['Ã�rea', 'Quantidade']
fig_pie = px.pie(area_counts, names='Ã�rea', values='Quantidade',
                 title='DistribuiÃ§Ã£o das Ã�reas de AplicaÃ§Ã£o')
fig_pie.show()

# ğŸŒ� GrÃ¡fico de dispersÃ£o geogrÃ¡fica
fig_geo = px.scatter_geo(df,
                         lat='Latitude', lon='Longitude',
                         text='PaÃ­s',
                         color='Ã�rea',
                         title='DistribuiÃ§Ã£o GeogrÃ¡fica das Ã�reas de Sensores QuÃ¢nticos',
                         projection='natural earth')
fig_geo.update_traces(marker=dict(size=10))


# Exibir mapa -
# Use write_html() to save the figure to an HTML file
pio.write_html(fig_geo, 'mapa_experiencia.html')
fig_geo.show()



import folium
import pandas as pd

# ğŸ”¬ Dados estruturados com descriÃ§Ãµes
data = {
    'Ã�rea': [
        'Medicina e SaÃºde', 'Medicina e SaÃºde', 'Medicina e SaÃºde',
        'EspaÃ§o e NavegaÃ§Ã£o', 'EspaÃ§o e NavegaÃ§Ã£o', 'EspaÃ§o e NavegaÃ§Ã£o',
        'Defesa e SeguranÃ§a', 'Defesa e SeguranÃ§a', 'Defesa e SeguranÃ§a',
        'Clima e Meio Ambiente', 'Clima e Meio Ambiente', 'Clima e Meio Ambiente'
    ],
    'PaÃ­s': [
        'EUA', 'Alemanha', 'JapÃ£o',
        'EUA', 'China', 'FranÃ§a',
        'EUA', 'Israel', 'Reino Unido',
        'Alemanha', 'Brasil', 'CanadÃ¡'
    ],
    'Latitude': [
        37.0902, 51.1657, 36.2048,
        37.0902, 35.8617, 46.2276,
        37.0902, 31.0461, 55.3781,
        51.1657, -14.2350, 56.1304
    ],
    'Longitude': [
        -95.7129, 10.4515, 138.2529,
        -95.7129, 104.1954, 2.2137,
        -95.7129, 34.8516, -3.4360,
        10.4515, -51.9253, -106.3468
    ],
    'Cor': [
        '#e74c3c', '#e74c3c', '#e74c3c',         # vermelho suave
        '#3498db', '#3498db', '#3498db',         # azul claro
        '#2ecc71', '#2ecc71', '#2ecc71',         # verde vibrante
        '#f39c12', '#f39c12', '#f39c12'          # laranja dourado
    ],
    'DescriÃ§Ã£o': [
        'AplicaÃ§Ãµes mÃ©dicas como ressonÃ¢ncia magnÃ©tica e monitoramento neural.',
        'AplicaÃ§Ãµes mÃ©dicas como ressonÃ¢ncia magnÃ©tica e monitoramento neural.',
        'AplicaÃ§Ãµes mÃ©dicas como ressonÃ¢ncia magnÃ©tica e monitoramento neural.',
        'Sensores gravitacionais e interferÃ´metros para navegaÃ§Ã£o espacial.',
        'Sensores gravitacionais e interferÃ´metros para navegaÃ§Ã£o espacial.',
        'Sensores gravitacionais e interferÃ´metros para navegaÃ§Ã£o espacial.',
        'Sensores magnÃ©ticos ultra-sensÃ­veis e relÃ³gios quÃ¢nticos militares.',
        'Sensores magnÃ©ticos ultra-sensÃ­veis e relÃ³gios quÃ¢nticos militares.',
        'Sensores magnÃ©ticos ultra-sensÃ­veis e relÃ³gios quÃ¢nticos militares.',
        'Monitoramento atmosfÃ©rico e detecÃ§Ã£o de radiaÃ§Ã£o ambiental.',
        'Monitoramento atmosfÃ©rico e detecÃ§Ã£o de radiaÃ§Ã£o ambiental.',
        'Monitoramento atmosfÃ©rico e detecÃ§Ã£o de radiaÃ§Ã£o ambiental.'
    ]
}

df = pd.DataFrame(data)

# ğŸŒ� Criar mapa base
m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')

# Adicionar marcadores com popups explicativas
for _, row in df.iterrows():
    popup_text = f"""
    <b>PaÃ­s:</b> {row['PaÃ­s']}<br>
    <b>Ã�rea de AplicaÃ§Ã£o:</b> {row['Ã�rea']}<br>
    <b>DescriÃ§Ã£o:</b> {row['DescriÃ§Ã£o']}
    """
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=7,
        color=row['Cor'],
        fill=True,
        fill_color=row['Cor'],
        fill_opacity=0.8,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(m)

# ğŸ“˜ Legenda fixa com cores atualizadas
legend_html = '''
<div style="position: fixed; 
     bottom: 30px; left: 30px; width: 270px; height: 160px; 
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; padding: 10px;">
<b>Legenda - Ã�reas de AplicaÃ§Ã£o da FÃ­sica QuÃ¢ntica</b><br>
<i class="fa fa-circle" style="color:#e74c3c"></i> Medicina e SaÃºde<br>
<i class="fa fa-circle" style="color:#3498db"></i> EspaÃ§o e NavegaÃ§Ã£o<br>
<i class="fa fa-circle" style="color:#2ecc71"></i> Defesa e SeguranÃ§a<br>
<i class="fa fa-circle" style="color:#f39c12"></i> Clima e Meio Ambiente
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# ğŸ’¾ Salvar mapa
m.save("mapa_fisica_quantica_explicativo.html")
m


import folium
from folium.plugins import MarkerCluster

# ğŸŒ� Coordenadas dos paÃ­ses lÃ­deres
country_coords = {
    'EUA': [37.0902, -95.7129],
    'Alemanha': [51.1657, 10.4515],
    'JapÃ£o': [36.2048, 138.2529],
    'China': [35.8617, 104.1954],
    'FranÃ§a': [46.2276, 2.2137],
    'Israel': [31.0461, 34.8516],
    'Reino Unido': [55.3781, -3.4360],
    'Brasil': [-14.2350, -51.9253],
    'CanadÃ¡': [56.1304, -106.3468]
}

# ğŸ§  Ã�reas de aplicaÃ§Ã£o e paÃ­ses envolvidos
areas = {
    'Medicina e SaÃºde': {
        'descriÃ§Ã£o': 'RessonÃ¢ncia magnÃ©tica, Monitoramento biomÃ©dico',
        'cor': 'red',
        'paÃ­ses': ['EUA', 'Alemanha', 'JapÃ£o']
    },
    'EspaÃ§o e NavegaÃ§Ã£o': {
        'descriÃ§Ã£o': 'InterferÃ´metros atÃ´micos, Sensores gravitacionais',
        'cor': 'blue',
        'paÃ­ses': ['EUA', 'China', 'FranÃ§a']
    },
    'Defesa e SeguranÃ§a': {
        'descriÃ§Ã£o': 'Sensores magnÃ©ticos, RelÃ³gios quÃ¢nticos',
        'cor': 'green',
        'paÃ­ses': ['EUA', 'Israel', 'Reino Unido']
    },
    'Clima e Meio Ambiente': {
        'descriÃ§Ã£o': 'Monitoramento atmosfÃ©rico, Sensores de radiaÃ§Ã£o',
        'cor': 'orange',
        'paÃ­ses': ['Alemanha', 'Brasil', 'CanadÃ¡']
    }
}

# ğŸ—ºï¸� Criar mapa base
m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')

# Agrupar marcadores
marker_cluster = MarkerCluster().add_to(m)

# Adicionar marcadores por Ã¡rea
for area, info in areas.items():
    for pais in info['paÃ­ses']:
        lat, lon = country_coords[pais]
        popup_text = f"<b>{pais}</b><br><b>Ã�rea:</b> {area}<br><b>AplicaÃ§Ãµes:</b> {info['descriÃ§Ã£o']}"
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=info['cor'], icon='info-sign')
        ).add_to(marker_cluster)

# ğŸ’¡ Legenda simulada com marcadores fixos
legend_html = '''
<div style="position: fixed; 
     bottom: 30px; left: 30px; width: 250px; height: 150px; 
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; padding: 10px;">
<b>Legenda - Ã�reas de AplicaÃ§Ã£o</b><br>
<i class="fa fa-map-marker fa-2x" style="color:red"></i> Medicina e SaÃºde<br>
<i class="fa fa-map-marker fa-2x" style="color:blue"></i> EspaÃ§o e NavegaÃ§Ã£o<br>
<i class="fa fa-map-marker fa-2x" style="color:green"></i> Defesa e SeguranÃ§a<br>
<i class="fa fa-map-marker fa-2x" style="color:orange"></i> Clima e Meio Ambiente
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# ğŸ’¾ Salvar mapa
m.save("mapa_sensores_quanticos_interativo.html")
m


import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi


simulator = AerSimulator()


# Create circuit
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
circ.measure_all()

# Transpile for simulator
simulator = AerSimulator()
circ = transpile(circ, simulator)

# Run and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)
plot_histogram(counts, title='Bell-State counts')


# Run and get memory
result = simulator.run(circ, shots=10, memory=True).result()
memory = result.get_memory(circ)
print(memory)


# Increase shots to reduce sampling variance
shots = 10000



# Statevector simulation method
sim_statevector = AerSimulator(method='statevector')
job_statevector = sim_statevector.run(circ, shots=shots)
counts_statevector = job_statevector.result().get_counts(0)

# Stabilizer simulation method
sim_stabilizer = AerSimulator(method='stabilizer')
job_stabilizer = sim_stabilizer.run(circ, shots=shots)
counts_stabilizer = job_stabilizer.result().get_counts(0)


# Extended Stabilizer method
sim_extstabilizer = AerSimulator(method='extended_stabilizer')
job_extstabilizer = sim_extstabilizer.run(circ, shots=shots)
counts_extstabilizer = job_extstabilizer.result().get_counts(0)

# Density Matrix simulation method
sim_density = AerSimulator(method='density_matrix')
job_density = sim_density.run(circ, shots=shots)
counts_density = job_density.result().get_counts(0)

# Matrix Product State simulation method
sim_mps = AerSimulator(method='matrix_product_state')
job_mps = sim_mps.run(circ, shots=shots)
counts_mps = job_mps.result().get_counts(0)


plot_histogram([ counts_statevector,counts_stabilizer ,counts_extstabilizer, counts_density, counts_mps],
               title='Counts for different simulation methods',
               legend=[ 'statevector',
                       'density_matrix','stabilizer','extended_stabilizer', 'matrix_product_state'])


# automatic
sim_automatic = AerSimulator(method='automatic')
job_automatic = sim_automatic.run(circ, shots=shots)
counts_automatic = job_automatic.result().get_counts(0)

plot_histogram([counts_automatic], title='Counts for automatic simulation method',legend=[ 'automatic'])


from qiskit_aer import AerError

# Initialize a GPU backend
# Note that the cloud instance for tutorials does not have a GPU
# so this will raise an exception.
try:
    simulator_gpu = AerSimulator(method='statevector', device='GPU')

except AerError as e:
    print(e)


from qiskit_aer import AerError

# Initialize a GPU backend
# Note that the cloud instance for tutorials does not have a GPU
# so this will raise an exception.
try:
    simulator_gpu = AerSimulator(method='tensor_network', device='GPU')

except AerError as e:
    print(e)


simulator = AerSimulator(method='statevector')
simulator.set_options(precision='single')

# Run and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)
print(counts)


# Construct quantum circuit without measure
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
circ.save_statevector()

# Transpile for simulator
simulator = AerSimulator(method='statevector')
circ = transpile(circ, simulator)

# Run and get statevector
result = simulator.run(circ).result()
statevector = result.get_statevector(circ)
plot_state_city(statevector, title='Bell state')


# Construct quantum circuit without measure
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
circ.save_unitary()

# Transpile for simulator
simulator = AerSimulator(method = 'unitary')
circ = transpile(circ, simulator)

# Run and get unitary
result = simulator.run(circ).result()
unitary = result.get_unitary(circ)
print("Circuit unitary:\n", np.asarray(unitary).round(5))


# Construct quantum circuit without measure
steps = 5
circ = QuantumCircuit(1)
for i in range(steps):
    circ.save_statevector(label=f'psi_{i}')
    circ.rx(i * np.pi / steps, 0)
circ.save_statevector(label=f'psi_{steps}')

# Transpile for simulator
simulator = AerSimulator(method= 'automatic')
circ = transpile(circ, simulator)

# Run and get saved data
result = simulator.run(circ).result()
data = result.data(0)
data


# Generate a random statevector
num_qubits = 2
psi = qi.random_statevector(2 ** num_qubits, seed=100)

# Set initial state to generated statevector
circ = QuantumCircuit(num_qubits)
circ.set_statevector(psi)
circ.save_state()

# Transpile for simulator
simulator = AerSimulator(method='statevector')
circ = transpile(circ, simulator)

# Run and get saved data
result = simulator.run(circ).result()
result.data(0)


# Use initilize instruction to set initial state
circ = QuantumCircuit(num_qubits)
circ.initialize(psi, range(num_qubits))
circ.save_state()

# Transpile for simulator
simulator = AerSimulator(method= 'statevector')
circ = transpile(circ, simulator)

# Run and get result data
result = simulator.run(circ).result()
result.data(0)


num_qubits = 2
rho = qi.random_density_matrix(2 ** num_qubits, seed=100)
circ = QuantumCircuit(num_qubits)
circ.set_density_matrix(rho)
circ.save_state()

# Transpile for simulator
simulator = AerSimulator(method='density_matrix')
circ = transpile(circ, simulator)

# Run and get saved data
result = simulator.run(circ).result()
result.data(0)


# Generate a random Clifford C
num_qubits = 2
stab = qi.random_clifford(num_qubits, seed=100)

# Set initial state to stabilizer state C|0>
circ = QuantumCircuit(num_qubits)
circ.set_stabilizer(stab)
circ.save_state()

# Transpile for simulator
simulator = AerSimulator(method= "stabilizer")
circ = transpile(circ, simulator)

# Run and get saved data
result = simulator.run(circ).result()
result.data(0)


# Generate a random unitary
num_qubits = 2
unitary = qi.random_unitary(2 ** num_qubits, seed=100)

# Set initial state to unitary
circ = QuantumCircuit(num_qubits)
circ.set_unitary(unitary)
circ.save_state()

# Transpile for simulator
simulator = AerSimulator(method='unitary')
circ = transpile(circ, simulator)

# Run and get saved data
result = simulator.run(circ).result()
result.data(0)


import qiskit
qiskit.__version__


# Construct quantum circuit
circ = QuantumCircuit(3, 3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure([0, 1, 2], [0, 1, 2])

sim_ideal = AerSimulator()

# Execute and get counts
result = sim_ideal.run(transpile(circ, sim_ideal)).result()
counts = result.get_counts(0)
plot_histogram(counts, title='Ideal counts for 3-qubit GHZ state')

