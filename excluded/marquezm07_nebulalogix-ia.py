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


# ==============================
# ğŸš€ Proyecto: BigQuery AI Hackathon
# ==============================

# 1. LibrerÃ­as necesarias
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
import matplotlib.pyplot as plt

# ==============================
# 2. Lectura de archivos locales (survey.txt)
# ==============================
input_path = "/kaggle/input/bigquery-ai-hackathon/survey.txt"

with open(input_path, "r", encoding="utf-8") as file:
    survey_content = file.read()

print("ğŸ“„ Contenido del archivo survey.txt:\n")
print(survey_content[:500])  # Mostrar primeros 500 caracteres

# Convertir el archivo en una lista estructurada (ejemplo simple)
survey_lines = survey_content.split("\n")
survey_df = pd.DataFrame({"linea": survey_lines})
print("\nâœ… Survey cargado en DataFrame:")
print(survey_df.head())

# ==============================
# 3. ConexiÃ³n a BigQuery
# ==============================
# NOTA: En Kaggle ya hay credenciales de Google Cloud disponibles.
#       Si ejecutas fuera de Kaggle, necesitas cargar tu JSON de credenciales.

client = bigquery.Client()

# ==============================
# 4. Ejemplo de consulta a un dataset pÃºblico
# ==============================
# Usaremos el dataset pÃºblico de Google Analytics (ejemplo del marketplace BigQuery)
query = """
SELECT
  fullVisitorId,
  visitNumber,
  date,
  totals.pageviews AS pageviews,
  trafficSource.medium AS medium
FROM
  `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20170101' AND '20170107'
LIMIT 10
"""

query_job = client.query(query)  # Ejecutar consulta
results = query_job.to_dataframe()

print("\nâœ… Resultados de BigQuery (muestra):")
print(results.head())

# ==============================
# 5. VisualizaciÃ³n de datos
# ==============================
plt.figure(figsize=(8,5))
results['pageviews'].hist(bins=10, edgecolor="black")
plt.title("DistribuciÃ³n de Pageviews (muestra 2017-01-01 a 2017-01-07)")
plt.xlabel("Pageviews")
plt.ylabel("Frecuencia")
plt.show()

# ==============================
# 6. IA (Ejemplo simple de procesamiento de texto)
# ==============================
# Como ejemplo: anÃ¡lisis de feedback del survey usando conteo de palabras clave
feedback_lines = [line for line in survey_lines if "feedback" in line.lower()]
feedback_df = pd.DataFrame({"feedback": feedback_lines})

print("\nğŸ“� Feedback detectado en survey:")
print(feedback_df)


