CREATE OR REPLACE MODEL
  `outorgas-469923.Outorgas.text_embeddings001`
REMOTE WITH CONNECTION
  `outorgas-469923.southamerica-east1.vertex-outorgas-1263547511`
OPTIONS (
  endpoint = 'gemini-embedding-001'
);

CREATE TABLE `outorgas-469923.Outorgas.cultura_embedding` AS
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `outorgas-469923.Outorgas.text_embeddings001`,
  (
    SELECT INT_CD ,UPPER(LTRIM(RTRIM(CULTURA_IRRIGADA))) as content 
    FROM `outorgas-469923.Outorgas.outorgas`
    WHERE CULTURA_IRRIGADA IS NOT NULL
  )
);



# Step 1: Generate an embedding for search query.
DECLARE search_query_embedding ARRAY<FLOAT64>;
SET search_query_embedding = (
  SELECT ml_generate_embedding_result
  FROM ML.GENERATE_EMBEDDING(
    MODEL `outorgas-469923.Outorgas.text_embeddings001`,
    (SELECT 'BANANA' AS content)
  )
);

# Step 2: Use ML.DISTANCE to find the closest matches.
SELECT
  content,
  ML.DISTANCE(ml_generate_embedding_result, search_query_embedding, 'COSINE') AS distance
FROM
  `outorgas-469923.Outorgas.cultura_embedding`
ORDER BY
  distance ASC


import pandas as pd
df = pd.read_csv("/kaggle/input/bq-results/bq-results-20250914-015320-1757814816662.csv")


df.head()


df = df[df["distance"] != 0.0]


df.head()


# Get value counts as a DataFrame
v = df['distance'].value_counts().reset_index()
v.columns = ['distance', 'count']

# Join back on distance
j = df.merge(v, on='distance', how='left')
j = j.drop_duplicates("distance")


display(j)




