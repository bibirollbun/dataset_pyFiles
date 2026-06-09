# GCP Authentication (Hackathon-safe)
# Judges: environment should already be authenticated.
# The code will auto-detect the project from GOOGLE_CLOUD_PROJECT.
# Fallback project_id can be set manually below if needed.

from google.cloud import bigquery
import vertexai
import os

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-hackathon-project-id")
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
client = bigquery.Client(project=PROJECT_ID)

print("âœ… Vertex AI + BigQuery ready (project:", PROJECT_ID, ")")



query = f"""
CREATE SCHEMA IF NOT EXISTS `{project_id}.world_mood`
OPTIONS(
  location="US",
  default_table_expiration_days=60
);
"""
client.query(query).result()
print("Dataset created.")


# Step 1: Ingest raw GDELT events (last 60 days)

gdelt_query = f"""
CREATE OR REPLACE TABLE `{project_id}.world_mood.gdelt_events_raw`
PARTITION BY event_date
CLUSTER BY country
AS
SELECT
  PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS event_date,
  COALESCE(
    ActionGeo_CountryCode,
    Actor1Geo_CountryCode,
    Actor2Geo_CountryCode
  ) AS country,
  COALESCE(
    ActionGeo_ADM1Code,
    Actor1Geo_ADM1Code,
    Actor2Geo_ADM1Code
  ) AS admin1,
  COALESCE(ActionGeo_Lat, Actor1Geo_Lat, Actor2Geo_Lat) AS lat,
  COALESCE(ActionGeo_Long, Actor1Geo_Long, Actor2Geo_Long) AS lon,
  EventCode,
  EventBaseCode,
  EventRootCode,
  AvgTone AS tone,
  SOURCEURL AS url
FROM `gdelt-bq.gdeltv2.events`
WHERE SQLDATE >= CAST(FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY)) AS INT64);

"""

client.query(gdelt_query).result()
print("gdelt Dataset created.")


# Step 2: Enrich with GKG (themes, persons, orgs) â€” keep only last 7 days for cost

enrich_query = f"""
CREATE OR REPLACE TABLE `{project_id}.world_mood.gdelt_events_enriched`
PARTITION BY event_date
CLUSTER BY country AS
WITH
  base AS (
    -- Use the columns already in your materialized raw table (no SQLDATE here)
    SELECT
      event_date,
      country,
      admin1,
      lat,
      lon,
      EventCode,
      EventBaseCode,
      EventRootCode,
      tone,
      url
    FROM `{project_id}.world_mood.gdelt_events_raw`
    WHERE event_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)   -- keep it cheap
      AND country IS NOT NULL
      AND url IS NOT NULL
  ),
  gkg AS (
    SELECT
      -- GKG DATE is yyyymmddHHMMSS; take first 8 for a DATE if you ever need it
      PARSE_DATE('%Y%m%d', SUBSTR(CAST(date AS STRING), 1, 8)) AS gkg_date,
      DocumentIdentifier AS url,
      V2Themes,
      V2Persons,
      V2Organizations
    FROM `gdelt-bq.gdeltv2.gkg`
    WHERE date >= CAST(FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)) AS INT64) * 1000000  -- match 15-digit format
  ),
  joined AS (
    SELECT
      b.*,
      g.V2Themes,
      g.V2Persons,
      g.V2Organizations
    FROM base b
    LEFT JOIN gkg g
      USING (url)
  )
SELECT
  event_date, country, admin1, lat, lon,
  EventCode, EventBaseCode, EventRootCode, tone, url,
  -- Split semicolon-delimited fields into arrays
  ARRAY(
    SELECT TRIM(x) FROM UNNEST(SPLIT(COALESCE(V2Themes, ''), ';')) x
    WHERE TRIM(x) != '' LIMIT 50
  ) AS themes,
  ARRAY(
    SELECT TRIM(x) FROM UNNEST(SPLIT(COALESCE(V2Persons, ''), ';')) x
    WHERE TRIM(x) != '' LIMIT 30
  ) AS persons,
  ARRAY(
    SELECT TRIM(x) FROM UNNEST(SPLIT(COALESCE(V2Organizations, ''), ';')) x
    WHERE TRIM(x) != '' LIMIT 30
  ) AS orgs
FROM joined;
"""

client.query(enrich_query).result()
print("âœ… gdelt_events_enriched created (7-day, with themes/persons/orgs)")




query = f"""
CREATE OR REPLACE TABLE `{project_id}.world_mood.daily_country_topics`
PARTITION BY event_date
CLUSTER BY country AS
WITH country_day AS (
  SELECT
    event_date,
    country,
    COUNT(*) AS headline_count,
    AVG(tone) AS avg_tone,
    MIN(tone) AS min_tone,
    MAX(tone) AS max_tone,
    ARRAY_AGG(DISTINCT EventRootCode IGNORE NULLS LIMIT 20) AS top_event_types,
    ARRAY_AGG(DISTINCT url IGNORE NULLS LIMIT 30) AS sample_urls
  FROM `{project_id}.world_mood.gdelt_events_raw`
  WHERE country IS NOT NULL
    AND event_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)  -- limit window
  GROUP BY event_date, country
)
SELECT
  event_date,
  country,
  headline_count,
  avg_tone,
  min_tone,
  max_tone,
  top_event_types,
  sample_urls,
  CONCAT(
    'Country: ', country, ' | Date: ', CAST(event_date AS STRING), ' | ',
    'Headlines: ', CAST(headline_count AS STRING), ' | ',
    'Tone avg/min/max: ', CAST(ROUND(avg_tone,2) AS STRING), '/', 
                          CAST(ROUND(min_tone,2) AS STRING), '/', 
                          CAST(ROUND(max_tone,2) AS STRING), ' | ',
    'Top Event Codes: ', ARRAY_TO_STRING(top_event_types, ', '), ' | ',
    'Sample URLs: ', ARRAY_TO_STRING(sample_urls, ' | ')
  ) AS topic_doc
FROM country_day;

"""

client.query(query).result()
print('Daily country topics created')


# Create a remote connection for the embedding model.
query = f"""
CREATE OR REPLACE MODEL `{project_id}.world_mood.embedding_model`
  REMOTE WITH CONNECTION `{location}.kaggle-connection`
  OPTIONS (endpoint = 'gemini-embedding-001');
"""
client.query(query).result()
print("embedding model created.")




query = f"""
CREATE OR REPLACE MODEL `{project_id}.world_mood.gen_model`
  REMOTE WITH CONNECTION `{location}.kaggle-connection`
  OPTIONS (endpoint = 'gemini-2.0-flash-001');
"""
client.query(query).result()
print("gen model created.")



# Generate embeddings for daily country topics

emb_query = f"""
CREATE OR REPLACE TABLE `{project_id}.world_mood.news_embeddings` AS
SELECT
  t.event_date,
  t.country,
  t.content AS topic_doc,                     
  t.ml_generate_embedding_result AS embedding,
  CONCAT(t.country, '-', CAST(t.event_date AS STRING)) AS id
FROM ML.GENERATE_EMBEDDING(
  MODEL `{project_id}.world_mood.embedding_model`,
  (
    SELECT
      event_date,
      country,
      SUBSTR(topic_doc, 1, 3000) AS content                    
    FROM `{project_id}.world_mood.daily_country_topics`
  )
) AS t;


"""
client.query(emb_query).result()
print("news embeddings created.")


# Create vector search helper functions

# Function 1: semantic search by free text query
query = f"""
CREATE OR REPLACE TABLE FUNCTION `{project_id}.world_mood.fn_similar_to_text`(
  query_text STRING
)
RETURNS TABLE<
  event_date DATE,
  country STRING,
  topic_doc STRING,
  distance FLOAT64,
  id STRING
>
AS (
  SELECT
    vs.base.event_date     AS event_date,
    vs.base.country        AS country,
    vs.base.topic_doc      AS topic_doc,
    vs.distance            AS distance,
    vs.base.id             AS id
  FROM VECTOR_SEARCH(
    TABLE `{project_id}.world_mood.news_embeddings`,
    'embedding',
    (SELECT ml_generate_embedding_result AS embedding
     FROM ML.GENERATE_EMBEDDING(
       MODEL `{project_id}.world_mood.embedding_model`,
       (SELECT query_text AS content)
     )),
    'embedding',
    top_k => 10,
    distance_type => 'COSINE'
  ) AS vs
);

"""
client.query(query).result()


# Function 2: find similar past days for a given country + date
query = f"""
CREATE OR REPLACE TABLE FUNCTION `{project_id}.world_mood.fn_similar_to_day`(
  country_code STRING,
  day DATE
)
RETURNS TABLE<
  event_date DATE,
  country STRING,
  topic_doc STRING,
  distance FLOAT64,
  id STRING
>
AS (
  WITH anchor AS (
    SELECT embedding
    FROM `{project_id}.world_mood.news_embeddings`
    WHERE country = country_code AND event_date = day
    LIMIT 1
  )
  SELECT
    vs.base.event_date     AS event_date,
    vs.base.country        AS country,
    vs.base.topic_doc      AS topic_doc,
    vs.distance            AS distance,
    vs.base.id             AS id
  FROM VECTOR_SEARCH(
    TABLE `{project_id}.world_mood.news_embeddings`,
    'embedding',
    (SELECT embedding FROM anchor),
    'embedding',
    top_k => 10,
    distance_type => 'COSINE'
  ) AS vs
  WHERE vs.base.event_date <> day      -- exclude the same day
);


"""
client.query(query).result()


# Demo 1: semantic search with free text query

query_text = "wildfires in Spain"

sql = f"""
SELECT event_date, country, topic_doc, distance
FROM `{project_id}.world_mood.fn_similar_to_text`("{query_text}")
ORDER BY distance ASC
LIMIT 5
"""

df = client.query(sql).to_dataframe()

# Add readable snippet
df["snippet"] = df["topic_doc"].str[:120] + "..."

display(df[["event_date", "country", "snippet", "distance"]])



# Demo 2: find past similar days for Spain on 2025-09-03

sql = f"""
SELECT *
FROM `{project_id}.world_mood.fn_similar_to_day`('ES', DATE '2025-09-03')
ORDER BY distance ASC
LIMIT 5
"""

df = client.query(sql).to_dataframe()

# Add readable snippet
df["snippet"] = df["topic_doc"].str[:120] + "..."

display(df[["event_date", "country", "snippet", "distance"]])



# Create table of top entities (themes + people)

top_query = f"""
CREATE OR REPLACE TABLE `{project_id}.world_mood.daily_top_entities`
PARTITION BY event_date
CLUSTER BY country AS
WITH
-- explode arrays
e AS (
  SELECT
    event_date,
    country,
    LOWER(SPLIT(t, ',')[OFFSET(0)]) AS theme,    -- keep name only
    LOWER(SPLIT(p, ',')[OFFSET(0)]) AS person
  FROM `{project_id}.world_mood.gdelt_events_enriched`,
  UNNEST(themes) t,
  UNNEST(persons) p
  WHERE event_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
),
-- simple noise filter for persons
clean_people AS (
  SELECT *
  FROM e
  WHERE person IS NOT NULL
    AND LENGTH(person) >= 3
    AND person NOT IN ('facebook','twitter','instagram','linkedin','whatsapp','youtube')
),
theme_counts AS (
  SELECT event_date, country, theme, COUNT(*) AS c
  FROM e
  WHERE theme IS NOT NULL
  GROUP BY 1,2,3
),
person_counts AS (
  SELECT event_date, country, person, COUNT(*) AS c
  FROM clean_people
  GROUP BY 1,2,3
),
top_themes AS (
  SELECT event_date, country,
         ARRAY_AGG(STRUCT(theme, c) ORDER BY c DESC LIMIT 10) AS top_themes
  FROM theme_counts
  GROUP BY 1,2
),
top_people AS (
  SELECT event_date, country,
         ARRAY_AGG(STRUCT(person, c) ORDER BY c DESC LIMIT 10) AS top_people
  FROM person_counts
  GROUP BY 1,2
)
SELECT
  t.event_date,
  t.country,
  top_themes,
  top_people
FROM top_themes t
LEFT JOIN top_people p
USING (event_date, country);
"""
client.query(top_query).result()


TOP_N = 80
today_query = f"""
CREATE OR REPLACE TABLE `{project_id}.world_mood.today_for_analogs` AS
WITH latest AS (
  SELECT MAX(event_date) AS d
  FROM `{project_id}.world_mood.daily_country_topics`
)
SELECT
  dct.event_date,
  dct.country
FROM `{project_id}.world_mood.daily_country_topics` dct
JOIN latest ON dct.event_date = latest.d
ORDER BY dct.headline_count DESC
LIMIT {TOP_N};
"""

client.query(today_query).result()
print("âœ… today_for_analogs table created")



ANALOG_SNIP_CHARS = 400
ANALOG_TOPK = 5 
rows = client.query(f"SELECT country, event_date FROM `{project_id}.world_mood.today_for_analogs`").result()
all_results = []

for row in rows:
    country = row.country
    event_date = row.event_date

    # Run fn_similar_to_day
    sim_query = f"""
    SELECT
      '{event_date}' AS event_date,
      '{country}' AS country,
      event_date AS past_date,
      SUBSTR(REGEXP_REPLACE(topic_doc, r'https?://\\S+', ''), 1, {ANALOG_SNIP_CHARS}) AS snippet,
      distance
    FROM `{project_id}.world_mood.fn_similar_to_day`('{country}', DATE '{event_date}')
    WHERE event_date < DATE '{event_date}'
    ORDER BY distance ASC
    LIMIT {ANALOG_TOPK}
    """
    sim_rows = client.query(sim_query).result()
    for r in sim_rows:
        all_results.append((r.event_date, r.country, r.past_date, r.snippet, r.distance))




# Build the dataframe
df = pd.DataFrame(all_results, columns=["event_date", "country", "past_date", "snippet", "distance"])

# âœ… Fix: convert to datetime.date
df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
df["past_date"] = pd.to_datetime(df["past_date"]).dt.date

# BigQuery table ID
table_id = f"{project_id}.world_mood.daily_analogs_flat"

# Set the schema
job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("event_date", "DATE"),
        bigquery.SchemaField("country", "STRING"),
        bigquery.SchemaField("past_date", "DATE"),
        bigquery.SchemaField("snippet", "STRING"),
        bigquery.SchemaField("distance", "FLOAT"),
    ],
    write_disposition="WRITE_TRUNCATE",
)

# Load the data
job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
job.result()

print("âœ… daily_analogs_flat loaded")





final_query = f"""
CREATE OR REPLACE TABLE `{project_id}.world_mood.daily_analogs`
PARTITION BY event_date
CLUSTER BY country AS
WITH ranked AS (
  SELECT
    event_date,
    country,
    ARRAY_AGG(
      STRUCT(past_date, snippet, distance)
      ORDER BY distance ASC
      LIMIT {ANALOG_TOPK}
    ) AS analogs
  FROM `{project_id}.world_mood.daily_analogs_flat`
  GROUP BY event_date, country
)
SELECT
  event_date,
  country,
  analogs,
  ARRAY_TO_STRING(
    ARRAY(
      SELECT
        CONCAT(CAST(a.past_date AS STRING), ': ', a.snippet)
      FROM UNNEST(analogs) AS a
      ORDER BY a.distance ASC
    ),
    '\\n- '
  ) AS analogs_txt
FROM ranked;
"""

job = client.query(final_query)
job.result()
print("âœ… daily_analogs created")



CONTEXT_CHARS = 700
MAX_TOKENS = 300
TEMP = 0.2

brief_query = f"""
CREATE OR REPLACE TABLE `{project_id}.world_mood.daily_briefings`
PARTITION BY event_date AS

WITH latest AS (
  SELECT MAX(event_date) AS d
  FROM `{project_id}.world_mood.daily_country_topics`
),

to_summarize AS (
  SELECT
    dct.event_date,
    dct.country,
    dct.headline_count,
    SUBSTR(REGEXP_REPLACE(dct.topic_doc, r'https?://\\S+', ''), 1, {CONTEXT_CHARS}) AS ctx,
    -- Top entities
    ARRAY_TO_STRING(ARRAY(SELECT t.theme  FROM UNNEST(ent.top_themes) t), ', ') AS themes_top,
    ARRAY_TO_STRING(ARRAY(SELECT p.person FROM UNNEST(ent.top_people) p), ', ') AS people_top,
    analogs.analogs_txt
  FROM `{project_id}.world_mood.daily_country_topics` dct
  JOIN latest ON dct.event_date = latest.d
  LEFT JOIN `{project_id}.world_mood.daily_top_entities` ent
    ON dct.event_date = ent.event_date AND dct.country = ent.country
  LEFT JOIN `{project_id}.world_mood.daily_analogs` analogs
    ON dct.event_date = analogs.event_date AND dct.country = analogs.country
  ORDER BY dct.headline_count DESC
  LIMIT {TOP_N}
),

normalized AS (
  SELECT
    event_date,
    country,
    headline_count,
    ctx,
    COALESCE(analogs_txt, 'None') AS analogs_txt,
    INITCAP(
      TRIM(
        REGEXP_REPLACE(
          REGEXP_REPLACE(
            REGEXP_REPLACE(
              REGEXP_REPLACE(
                REGEXP_REPLACE(
                  LOWER(COALESCE(themes_top,'')),
                  r'(?i)uspec_', ''
                ),
                r'(?i)wb_\\d+_', ''
              ),
              r'(?i)tax_fncact_', 'tax '
            ),
            r'(?i)crisislex_[^, ]+', 'crisis response'
          ),
          r'_', ' '
        )
      )
    ) AS themes_nice,
    INITCAP(
      TRIM(
        REGEXP_REPLACE(COALESCE(people_top,''), r'(?i)\\bLos Angeles\\b,?\\s*', '')
      )
    ) AS people_nice
  FROM to_summarize
),

llm AS (
  SELECT
    s.event_date,
    s.country,
    t.ml_generate_text_llm_result AS llm_text,
    TO_JSON_STRING(t) AS llm_raw
  FROM normalized s
  JOIN ML.GENERATE_TEXT(
    MODEL `{project_id}.world_mood.gen_model`,
    (
      SELECT
        event_date,
        country,
        CONCAT(
          'You are a news analyst. Country is an ISO code.', CHR(10),
          'Fill in this EXACT template with â‰¤90 words, no intro/outro:', CHR(10),
          '[What happened] ', CHR(10),
          '[Key drivers] ', CHR(10),
          '[Impact] ', CHR(10),
          '[Watch next] ', CHR(10),
          'Rules:', CHR(10),
          '- Only summarize events for ISO code ', country, '.', CHR(10),
          '- Use at least TWO items from "Top themes (readable)" and at least ONE name from "Top people".', CHR(10),
          '- Do NOT output raw taxonomy tokens (e.g., uspec_politics_general1 / crisislex_*). Use natural English phrases only.', CHR(10),
          '- Do NOT mention event codes. Avoid boilerplate like "diverse/broad range of events".', CHR(10),
          '- Include at least one concrete number if available (e.g., headline count, tone).', CHR(10),
          '- ALWAYS include all 4 sections: What happened, Key drivers, Impact, and Watch next.', CHR(10),
          '- Do NOT stop before completing the full template.', CHR(10),
          '- Limit each section to 1â€“2 sentences. Avoid speculation. Be concise and specific.', CHR(10),


          'Top themes (readable): ', COALESCE(themes_nice, 'none'), CHR(10),
          'Top people: ', COALESCE(people_nice, 'none'), CHR(10),
          'Context: ', ctx, CHR(10),
          'Relevant past events to consider (analog history):', CHR(10),
          analogs_txt
        ) AS prompt
      FROM normalized
    ),
    STRUCT({TEMP} AS temperature, {MAX_TOKENS} AS max_output_tokens, TRUE AS flatten_json_output)
  ) AS t
  ON s.event_date = t.event_date AND s.country = t.country
),

fallback AS (
  SELECT
    event_date,
    country,
    llm_text,
    SAFE.PARSE_JSON(llm_raw) AS j
  FROM llm
),

final AS (
  SELECT
    event_date,
    country,
    COALESCE(
      llm_text,
      JSON_VALUE(j, '$.ml_generate_text_result.candidates[0].content.parts[0].text'),
      JSON_VALUE(j, '$.ml_generate_text_result.candidates[0].content[0].text'),
      JSON_VALUE(j, '$.predictions[0].content'),
      JSON_VALUE(j, '$.text')
    ) AS briefing_text
  FROM fallback
)

SELECT *
FROM final
WHERE briefing_text IS NOT NULL;
"""

job = client.query(brief_query)
job.result()
print("âœ… daily_briefings created (with analog enrichment)")



moodmap_query = f"""
CREATE OR REPLACE TABLE `{project_id}.world_mood.daily_moodmap`
PARTITION BY event_date AS

WITH
latest_date AS (
  SELECT MAX(event_date) AS d
  FROM `{project_id}.world_mood.daily_briefings`
),

joined_data AS (
  SELECT
    b.event_date,
    b.country,
    b.briefing_text,
    g.avg_tone,
    ent.top_themes
  FROM `{project_id}.world_mood.daily_briefings` b
  JOIN latest_date l ON b.event_date = l.d
  LEFT JOIN `{project_id}.world_mood.daily_country_topics` g
    ON b.event_date = g.event_date AND b.country = g.country
  LEFT JOIN `{project_id}.world_mood.daily_top_entities` ent
    ON b.event_date = ent.event_date AND b.country = ent.country
),

llm_outputs AS (
  SELECT
    s.event_date,
    s.country,
    s.avg_tone,
    s.top_themes,
    s.briefing_text,
    t.ml_generate_text_llm_result AS llm_text,
    TO_JSON_STRING(t) AS llm_raw
  FROM joined_data s
  JOIN ML.GENERATE_TEXT(
    MODEL `{project_id}.world_mood.gen_model`,
    (
      SELECT
        event_date,
        country,
        CONCAT(
          'You are a sentiment analysis model. Read the news briefing and respond with a single number between -1 (very negative) and 1 (very positive).\\n',
          'Briefing:\\n',
          briefing_text, '\\n',
          'Sentiment score:'
        ) AS prompt
      FROM joined_data
    ),
    STRUCT({TEMP} AS temperature, {MAX_TOKENS} AS max_output_tokens, TRUE AS flatten_json_output)
  ) AS t
  ON s.event_date = t.event_date AND s.country = t.country
),

parsed_scores AS (
  SELECT
    event_date,
    country,
    avg_tone,
    top_themes,
    briefing_text,
    SAFE_CAST(REGEXP_EXTRACT(llm_text, r"-?\\d+\\.\\d+") AS FLOAT64) AS sentiment_score
  FROM llm_outputs
),

blended_scores AS (
  SELECT
    event_date,
    country,
    ROUND((
      SAFE_CAST(sentiment_score AS FLOAT64) +
      SAFE_CAST(avg_tone AS FLOAT64) / 10
    ) / 2, 4) AS mood_score,
    top_themes,
    briefing_text
  FROM parsed_scores
),

-- ğŸ†• Add short summary for visualization hovers
short_summary AS (
  SELECT
    s.event_date,
    s.country,
    s.mood_score,
    s.top_themes,
    s.briefing_text,
    t.ml_generate_text_llm_result AS summary_ref
  FROM blended_scores s
  JOIN ML.GENERATE_TEXT(
    MODEL `{project_id}.world_mood.gen_model`,
    (
      SELECT
        event_date,
        country,
        CONCAT(
  'Summarize todayâ€™s news mood for the ISO country code "', country, '" (treat this as a COUNTRY, not a US state).',
  ' Write ONE sentence (â‰¤25 words).',
  ' Do not use labels like [Impact] or [What happened].',
  ' Keep it concise, neutral, and hover-friendly.'
)
 AS prompt
      FROM blended_scores
    ),
    STRUCT({TEMP} AS temperature, 60 AS max_output_tokens, TRUE AS flatten_json_output)
  ) AS t
  ON s.event_date = t.event_date AND s.country = t.country
)

SELECT *
FROM short_summary
WHERE mood_score IS NOT NULL;
"""

job = client.query(moodmap_query)
job.result()
print("âœ… daily_moodmap created (with normalized tone, sentiment score, and short hover summary)")



# Data Peek
df = client.query(f"""
  SELECT event_date, country, mood_score, top_themes, summary_ref
  FROM `{project_id}.world_mood.daily_moodmap`
  WHERE event_date = (SELECT MAX(event_date) FROM `{project_id}.world_mood.daily_moodmap`)
  ORDER BY mood_score DESC
  LIMIT 10
""").to_dataframe()
df.head(10)



!pip install pycountry



# Query daily_moodmap for the latest day
latest_sql = f"""
SELECT *
FROM `{project_id}.world_mood.daily_moodmap`
WHERE event_date = (SELECT MAX(event_date) FROM `{project_id}.world_mood.daily_moodmap`)
"""
today_df = client.query(latest_sql).to_dataframe()



import pycountry

def iso2_to_iso3(iso2):
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except:
        return None

today_df["iso3"] = today_df["country"].apply(iso2_to_iso3)



import plotly.express as px

fig = px.choropleth(
    today_df,
    locations="iso3",
    locationmode="ISO-3",
    color="mood_score",
    color_continuous_scale=px.colors.diverging.RdYlGn,
    range_color=(-1, 1),
    title="ğŸŒ� Global News MoodMap"
)

# Custom hover: clean, structured, but full summary
fig.update_traces(
    hovertemplate=(
        "<b>%{hovertext}</b><br>" +                # Country name
        "Mood Score: <b>%{z:.2f}</b><br><br>" +    # Numeric score, bold
        "<b>Summary:</b><br>%{customdata[0]}"      # Full summary (line break before it)
    ),
    hovertext=today_df["country"],                 # human-readable name
    customdata=today_df[["summary_ref"]]           # pass full summary
)

# Prettier layout
fig.update_layout(
    title=dict(
        text="ğŸŒ� Global News MoodMap<br><sup>Data from GDELT + BigQuery AI</sup>",
        x=0.5,
        xanchor="center"
    ),
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="LightGray",
        projection_type="natural earth",
        showcountries=True,
        countrycolor="white"
    ),
    coloraxis_colorbar=dict(
        title="Mood",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["ğŸ˜¡ Very Negative", "ğŸ˜Ÿ Negative", "ğŸ˜� Neutral", "ğŸ™‚ Positive", "ğŸ˜ƒ Very Positive"],
        len=0.75
    ),
    margin=dict(l=0, r=0, t=60, b=0)
)

fig.show()





