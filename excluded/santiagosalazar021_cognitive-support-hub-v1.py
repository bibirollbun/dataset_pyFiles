# Celda 0: InstalaciÃ³n de Dependencias
# ------------------------------------
# Ejecuta esta celda primero para asegurarte de que todas las librerÃ­as necesarias
# estÃ©n instaladas en el entorno del notebook.

!pip install --quiet google-cloud-bigquery-storage pyarrow
print("âœ… Dependencias de BigQuery Storage instaladas.")


# -*- coding: utf-8 -*-
import os
import pandas as pd
# --- Â¡LA CORRECCIÃ“N CLAVE! ---
# Importamos pandas_gbq ANTES de intentar configurar sus opciones.
import pandas_gbq
from google.cloud import bigquery
from google.oauth2 import service_account
from kaggle_secrets import UserSecretsClient

# --- CONFIGURACIÃ“N DEL PROYECTO --
PROJECT_ID = "hackathon-cognitive-hub"
LOCATION = "us-east1"
DATASET_ID = "support_tickets_ai"

# --- AutenticaciÃ³n No Interactiva ---
# Acceder al secreto que guardamos en Kaggle
secrets = UserSecretsClient()
gcp_credentials_json = secrets.get_secret("GCP_CREDENTIALS")

# Crear las credenciales a partir del JSON
credentials = service_account.Credentials.from_service_account_info(
    eval(gcp_credentials_json) # eval() convierte el string del secreto a un diccionario
)

# Configurar un cliente de BigQuery con estas credenciales
client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

# --- Usamos el mÃ©todo moderno para configurar pandas-gbq ---
# Ahora que pandas_gbq estÃ¡ importado, estas opciones funcionarÃ¡n.
pandas_gbq.context.project = PROJECT_ID
pandas_gbq.context.credentials = credentials

print(f"âœ… AutenticaciÃ³n con Service Account exitosa.")
print(f"âœ… Proyecto de Google Cloud configurado en: {PROJECT_ID}")

# --- ConfiguraciÃ³n de Pandas y BigQuery Magic ---
%load_ext google.cloud.bigquery
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("âœ… Opciones de visualizaciÃ³n y extensiÃ³n de BigQuery cargadas.")
print("ğŸš€ Â¡El entorno estÃ¡ listo para las demos!")


# --- Demo #1: Query Definition ---

generative_analysis_sql = """
-- Paso 1: Preparar los prompts para la IA.
WITH Prompts AS (
  SELECT
    t.Ticket_ID,
    CONCAT(
      'Analiza esta descripciÃ³n de ticket. Proporciona un resumen de una frase, el sentimiento del cliente (Positivo, Negativo, Neutral) y una categorÃ­a probable (Ej: FacturaciÃ³n, Problema TÃ©cnico, ConfiguraciÃ³n). Formatea la salida como un objeto JSON con las claves "resumen", "sentimiento" y "categoria". DescripciÃ³n del ticket: ',
      REPLACE(t.Ticket_Description, '{product_purchased}', t.Product_Purchased)
    ) AS prompt
  FROM
    `hackathon-cognitive-hub.support_tickets_ai.tickets` AS t
  WHERE
    Ticket_ID <= 10 -- Limitamos a 10 para una demo rÃ¡pida
),

-- Paso 2: Llamar al modelo de IA y limpiar la salida JSON.
ModelResults AS (
  SELECT
    Ticket_ID,
    -- Usamos una expresiÃ³n regular para extraer el JSON puro de la respuesta.
    REGEXP_EXTRACT(ml_generate_text_llm_result, r'{[\s\S]*}') AS cleaned_json_string
  FROM
    ML.GENERATE_TEXT(
      MODEL `hackathon-cognitive-hub.support_tickets_ai.gemini_vision_model`,
      TABLE Prompts,
      STRUCT(
        0.1 AS temperature,
        250 AS max_output_tokens,
        TRUE AS flatten_json_output
      )
    )
)

-- Paso Final: Unir los resultados de la IA con los datos originales.
SELECT
  t.Ticket_ID,
  JSON_VALUE(mr.cleaned_json_string, '$.resumen') AS ticket_summary,
  JSON_VALUE(mr.cleaned_json_string, '$.sentimiento') AS ticket_sentiment,
  JSON_VALUE(mr.cleaned_json_string, '$.categoria') AS ticket_category,
  REPLACE(t.Ticket_Description, '{product_purchased}', t.Product_Purchased) AS Ticket_Description
FROM
  `hackathon-cognitive-hub.support_tickets_ai.tickets` AS t
JOIN
  ModelResults AS mr ON t.Ticket_ID = mr.Ticket_ID
ORDER BY
  t.Ticket_ID;
"""

print("âœ… Consulta SQL para la Demo #1 definida.")


# --- Demo #1: Execution and Display ---
generative_analysis_df = pandas_gbq.read_gbq(
    generative_analysis_sql,
    project_id=PROJECT_ID,
    credentials=credentials
)

print("âœ… Â¡AnÃ¡lisis generativo completado!")
print("Mostrando los tickets enriquecidos con insights de IA:")
display(generative_analysis_df.head())


# --- Demo #2: DefiniciÃ³n de la "Query Maestra"---

# Usamos r""" para crear una "raw string" y evitar problemas con caracteres especiales.
multimodal_analysis_sql = r"""
-- Definimos el lÃ­mite de filas a procesar. Â¡Puedes cambiar este nÃºmero!
DECLARE process_limit INT64 DEFAULT 10;

-- Paso 1: Identificar los archivos adjuntos para cada ticket
WITH TicketFiles AS (
  SELECT
    t.Ticket_ID,
    uf.uri AS file_uri
  FROM
    `hackathon-cognitive-hub.support_tickets_ai.tickets` AS t
  LEFT JOIN
    `hackathon-cognitive-hub.support_tickets_ai.unstructured_files` AS uf
  ON
    STARTS_WITH(uf.uri, CONCAT('gs://cognitive-hub-dataset-santiago/dataset/', CAST(t.Ticket_ID AS STRING), '_'))
  WHERE t.Ticket_ID <= process_limit
),

-- Paso 2: Preparar los prompts para el anÃ¡lisis de texto
TextPrompts AS (
  SELECT
    t.Ticket_ID,
    CONCAT('Analiza esta descripciÃ³n de ticket. Proporciona un resumen de una frase, el sentimiento del cliente (Positivo, Negativo, Neutral) y una categorÃ­a probable (Ej: FacturaciÃ³n, Problema TÃ©cnico, ConfiguraciÃ³n). Formatea la salida como un objeto JSON con las claves "resumen", "sentimiento" y "categoria". DescripciÃ³n del ticket: ',
           REPLACE(t.Ticket_Description, '{product_purchased}', t.Product_Purchased)
    ) AS prompt
  FROM `hackathon-cognitive-hub.support_tickets_ai.tickets` AS t
  WHERE t.Ticket_ID <= process_limit
),

-- Paso 3: Preparar los prompts para el anÃ¡lisis de archivos
MultimodalPrompts AS (
  SELECT
    Ticket_ID,
    CONCAT('Describe el contenido de este archivo (imagen o documento) en una frase para un agente de soporte: ', file_uri) AS prompt
  FROM TicketFiles
  WHERE file_uri IS NOT NULL
),

-- Paso 4: Ejecutar la IA sobre las descripciones de texto
GenerativeAnalysis AS (
  SELECT
    Ticket_ID,
    TRIM(REPLACE(JSON_VALUE(ml_generate_text_result, '$.candidates[0].content.parts[0].text'), '```json', ''), ' \n```') AS cleaned_json_string
  FROM ML.GENERATE_TEXT(MODEL `hackathon-cognitive-hub.support_tickets_ai.gemini_vision_model`, TABLE TextPrompts, STRUCT(0.1 AS temperature, 200 AS max_output_tokens))
),

-- Paso 5: Ejecutar la IA sobre los archivos adjuntos
MultimodalAnalysis AS (
  SELECT
    Ticket_ID,
    TRIM(JSON_VALUE(ml_generate_text_result, '$.candidates[0].content.parts[0].text')) AS file_summary
  FROM ML.GENERATE_TEXT(MODEL `hackathon-cognitive-hub.support_tickets_ai.gemini_vision_model`, TABLE MultimodalPrompts, STRUCT(0.2 AS temperature, 100 AS max_output_tokens))
),

-- Paso 6: Agregar resÃºmenes de archivos
AggregatedMultimodal AS (
  SELECT
    Ticket_ID,
    STRING_AGG(file_summary, '; ') as unstructured_file_summary
  FROM MultimodalAnalysis
  GROUP BY Ticket_ID
)

-- Paso Final: Unir todos los resultados
SELECT
  t.Ticket_ID,
  REPLACE(t.Ticket_Description, '{product_purchased}', t.Product_Purchased) AS Ticket_Description,
  JSON_VALUE(ga.cleaned_json_string, '$.resumen') AS ticket_summary,
  JSON_VALUE(ga.cleaned_json_string, '$.sentimiento') AS ticket_sentiment,
  JSON_VALUE(ga.cleaned_json_string, '$.categoria') AS ticket_category,
  ama.unstructured_file_summary
FROM
  `hackathon-cognitive-hub.support_tickets_ai.tickets` AS t
LEFT JOIN
  GenerativeAnalysis AS ga ON t.Ticket_ID = ga.Ticket_ID
LEFT JOIN
  AggregatedMultimodal AS ama ON t.Ticket_ID = ama.Ticket_ID
WHERE
  t.Ticket_ID <= process_limit
ORDER BY
  t.Ticket_ID;
"""

print("âœ… Consulta SQL para la Demo #2 (Multimodal) definida.")


# --- Demo #2: EjecuciÃ³n y VisualizaciÃ³n ---
multimodal_analysis_df = pandas_gbq.read_gbq(
    multimodal_analysis_sql,
    project_id=PROJECT_ID,
    credentials=credentials
)

print("âœ… Â¡AnÃ¡lisis multimodal de 360 grados completado!")
print("Mostrando la vista unificada de los tickets y sus archivos adjuntos:")
display(multimodal_analysis_df.head())


# --- Demo #3: DefiniciÃ³n de la Consulta para Crear Embeddings ---

ticket_embeddings_sql = r"""
-- Esta consulta crea los embeddings usando las descripciones personalizadas.

WITH Embeddings AS (
  SELECT
    Ticket_ID,
    ml_generate_embedding_result
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `hackathon-cognitive-hub.support_tickets_ai.embedding_model`,
      (
        SELECT
          Ticket_ID,
          REPLACE(Ticket_Description, '{product_purchased}', Product_Purchased) AS content
        FROM
          `hackathon-cognitive-hub.support_tickets_ai.tickets`
        LIMIT 100
      )
    )
)

-- Unimos los embeddings con los datos originales.
SELECT
  t.Ticket_ID,
  REPLACE(t.Ticket_Description, '{product_purchased}', t.Product_Purchased) AS Ticket_Description,
  e.ml_generate_embedding_result
FROM
  `hackathon-cognitive-hub.support_tickets_ai.tickets` AS t
JOIN
  Embeddings AS e ON t.Ticket_ID = e.Ticket_ID;
"""

print("âœ… Consulta SQL para crear la base de conocimiento semÃ¡ntica definida.")


# --- Demo #3: EjecuciÃ³n y Guardado de la Tabla de Embeddings ---

# Ejecutamos la consulta para generar los embeddings
ticket_embeddings_df = pandas_gbq.read_gbq(
    ticket_embeddings_sql,
    project_id=PROJECT_ID,
    credentials=credentials
)

# Guardamos el resultado como una tabla permanente en BigQuery
table_ref = f"{DATASET_ID}.ticket_embeddings"
pandas_gbq.to_gbq(ticket_embeddings_df,
                  destination_table=table_ref,
                  project_id=PROJECT_ID,
                  if_exists='replace')

print(f"âœ… Â¡Base de conocimiento semÃ¡ntica creada con Ã©xito en la tabla 'ticket_embeddings'!")
print(f"Se generaron embeddings para {len(ticket_embeddings_df)} tickets.")
display(ticket_embeddings_df.head())


# --- Demo #3: DefiniciÃ³n de la Consulta de BÃºsqueda Vectorial ---

vector_search_sql = r"""
-- Definimos la descripciÃ³n del nuevo ticket que acaba de llegar.
DECLARE new_ticket_description STRING DEFAULT "My Photoshop is not charging.";

-- Usamos VECTOR_SEARCH para encontrar los tickets mÃ¡s parecidos.
SELECT
  base.Ticket_ID,
  base.Ticket_Description,
  distance
FROM
  VECTOR_SEARCH(
    TABLE `hackathon-cognitive-hub.support_tickets_ai.ticket_embeddings`,
    'ml_generate_embedding_result',
    (
      SELECT ml_generate_embedding_result
      FROM ML.GENERATE_EMBEDDING(
        MODEL `hackathon-cognitive-hub.support_tickets_ai.embedding_model`,
        (SELECT new_ticket_description AS content)
      )
    ),
    top_k => 5,
    distance_type => 'COSINE'
  );
"""
print("âœ… Consulta SQL para la bÃºsqueda semÃ¡ntica definida.")


# --- Demo #3: EjecuciÃ³n y VisualizaciÃ³n de la BÃºsqueda ---
similar_tickets_df = pandas_gbq.read_gbq(
    vector_search_sql,
    project_id=PROJECT_ID,
    credentials=credentials
)

print("âœ… Â¡BÃºsqueda semÃ¡ntica completada!")
print(f"Simulando un nuevo ticket: 'My Photoshop is not charging.'")
print("\nLos 5 tickets mÃ¡s similares encontrados en nuestra base de conocimiento son:")
display(similar_tickets_df)

