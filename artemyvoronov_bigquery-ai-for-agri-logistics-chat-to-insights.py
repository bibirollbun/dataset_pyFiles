import numpy as np
import pandas as pd 

rawmessages = pd.read_csv("/kaggle/input/commodity-trade-chat-messages/messages.csv", parse_dates=["timestamp"])


messages = rawmessages.drop_duplicates(subset="text", keep="first")

messages = messages[messages["timestamp"] > '2025-08-01'].tail(1000)


import re

RX = {
    "Sell": re.compile(r"""
        (прода(?:м|ём|ем|ётся|ется|ет|ёт|ются|жа|ю|дим|ст)|реализу(?:ю|ем)|предлагаем|реализуе|продаже|предложите цену)
    """, re.I | re.U | re.X),

    "Buy": re.compile(r"""
        (куп(лю|им)|покупка|(|за|по|с|вы)куп(аю|аем|ает|ка|ит)|принима(?:ю|ем)|продайте|закуп)
    """, re.I | re.U | re.X),

    "Trucking": re.compile(r"""
        (перевозк|рейс|плечо|→|подач[аи]|хартия|выгрузк|сцепки|(руб|р|₽)(\\|\/|\.|\s)(тн|т)|грузят|весы|транспорт|сельхозник|
           машин[аы]|тягач|авто|тонар|полуприцеп|грузовик|трал|самосвал|маниту|кун|по\sнорме|норма|
           ищу\s+машин|нужн[ао]\s+машин|погрузк\S+|расстояние|по\s+полной|харти\S+|\-\>)
    """, re.I | re.U | re.X),

    "Spam": re.compile(r"""
        (заработа[йть]|биткоин|crypto|ставк[аи]\s+на\s+спорт|казино|
         подписывайся|бесплатн(?:о|ые)|скачать\s+без\s+регистрации|халтур|\bтемка\b|мутк[аиу]|варик|легкий\sдоход|поднимать|челов|на\sчас|кредит|опыт|подработк|обуч|зара​боток|шабашк|сотрудник|проект|ссылк|комaнд|обучение|онлайн|помощник|выиграт|баз[аы] данных|справочник)
    """, re.I | re.U | re.X),
}

PRIORITY = ["Sell", "Buy", "Trucking", "Spam"]

def classify_regex(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    for label in PRIORITY:
        if RX[label].search(t):
            return label
    return 'Unknown'  

messages["type"] = None
messages["type"] = messages["type"].where(messages["type"].notna(), messages["text"].apply(classify_regex))

trucking_messages = messages[messages["type"]=='Trucking']
print(f'Found {len(trucking_messages)} trucking messages')
print(trucking_messages["text"].head(20))


print(trucking_messages["text"][533257])


from google.cloud import bigquery

from google.colab import auth 
auth.authenticate_user()

client = bigquery.Client(project="cptmax")


table_id = "cptmax.Messages.messages"

job = client.load_table_from_dataframe(trucking_messages, table_id,job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
job.result()  

print("The data has been loaded to ", table_id)


query = """
CREATE OR REPLACE TABLE `cptmax.Messages.message_chunks` AS
SELECT
  timestamp,
  text as input_text,
  JSON_VALUE(ml_generate_text_result, "$.candidates[0].content.parts[0].text") AS chunk_text
FROM ML.GENERATE_TEXT(
  MODEL `cptmax.Messages.grainmessages`,
  (
    SELECT
      CONCAT('''Split the input Russian logistics chat message into minimal coherent CHUNKS, each describing ONE truck load (one commodity/origin/destination/volume/price set).
Return PLAIN TEXT ONLY in this exact format (no quotes, no JSON, no code fences):
chunk1<<<CHUNK>>>chunk2<<<CHUNK>>>chunk3
Rules:
- Do NOT add a separator at the beginning or end.
- Do NOT use backslashes. If a backslash is present in source text, replace it with "/".
- Exclude headers/signatures/contacts that are not part of a load.
- If input is too long, process only the first four loads
Text:
''', text) AS prompt,
      timestamp, text
    FROM `cptmax.Messages.messages`
  ),
  STRUCT(8192 AS max_output_tokens)
);
"""
job = client.query(query)
job.result() 
print("Chunks are marked")


query = """
CREATE OR REPLACE TABLE `cptmax.Messages.truck_load_message` AS
SELECT
  timestamp,                                           
  TRIM(part) AS text                  
FROM `cptmax.Messages.message_chunks`,
UNNEST(
  SPLIT(
    REGEXP_REPLACE(chunk_text, r'\s*<<<CHUNK>>>\s*', '<<<CHUNK>>>'),
    '<<<CHUNK>>>'
  )
) AS part WITH OFFSET AS off
WHERE TRIM(part) <> '';
"""
job = client.query(query)
job.result() 
print("A table with single truck load messages is created")


query = """
CREATE OR REPLACE TABLE `cptmax.Messages.truck_loads` AS
SELECT
  text,
  timestamp,
  commodity,
  commodity_en,
  volume,
  origin,
  origin_en,
  destination,
  destination_en,
  distance_km,
  rate_per_ton,
  notes,
  notes_en
FROM AI.GENERATE_TABLE(
  MODEL `cptmax.Messages.grainmessages`,
  (
    SELECT
      STRUCT(
        '''
        Extract exactly ONE truck load from the text and RETURN ONLY FLAT FIELDS (no JSON, no markdown, no comments).
        
        Fields to output:
        - commodity (ru). 4-класс, 3-класс means Пшеница
        - commodity_en (English)
        - volume (numeric, tons)
        - origin (ru), origin_en
        - destination (ru), destination_en
        - distance_km (numeric if present)
        - rate_per_ton (numeric)
        - notes (ru)
        - notes_en (English translation of notes)

        Notes translation rules:
        - Translate concisely into clear English.
        - Keep measurement units and numeric values intact.
        - Do not translate personal names or phone numbers.
        - If no notes exist, leave both notes and notes_en null.
        ''' AS prompt,
        ARRAY<STRING>[] AS fewshots,
        text AS message
      ) AS prompt,
      text,
      timestamp
    FROM `cptmax.Messages.truck_load_message`
  ),
  STRUCT(
    '''commodity STRING,
     commodity_en STRING,
     volume FLOAT64,
     origin STRING,
     origin_en STRING,
     destination STRING,
     destination_en STRING,
     distance_km FLOAT64,
     rate_per_ton FLOAT64,
     notes STRING,
     notes_en STRING''' AS output_schema,
    8192 AS max_output_tokens
  )
);

"""
job = client.query(query)
job.result() 
print("Structural data has been extracted")


query = """
SELECT
  timestamp,
  commodity,
  commodity_en,
  volume,
  origin,
  origin_en,
  destination,
  destination_en,
  distance_km,
  rate_per_ton,
  notes,
  notes_en
FROM `cptmax.Messages.truck_loads`
"""

loads = client.query(query).to_dataframe()

print(loads.head())


loads["commodity_en"] = loads["commodity_en"].str.lower()
loads.loc[loads["rate_per_ton"].notna() & (loads["rate_per_ton"] < 10), "rate_per_ton"] *= 1000


import matplotlib.pyplot as plt

counts = loads["commodity_en"].value_counts().head(20)

counts.plot(kind="bar", figsize=(8, 5))
plt.title("Number of loads per commodity")
plt.xlabel("Commodity")
plt.ylabel("Count")
plt.show()


import pandas as pd 

mask = (loads["commodity_en"].str.lower() == "wheat") & (loads["distance_km"].between(100, 200))  & (loads["rate_per_ton"] < 10000) # If the rate is greater than 4000, it's probably a sale rather than transportation.
tmp = loads.loc[mask].copy()
tmp["distance_km"] = pd.to_numeric(tmp["distance_km"], errors="coerce")
tmp["rate_per_ton"]       = pd.to_numeric(tmp["rate_per_ton"], errors="coerce")
subset = tmp.dropna(subset=["distance_km", "rate_per_ton"]).copy()

bins = [100, 120, 140, 160, 180, 200]
subset["distance_bin"] = pd.cut(subset["distance_km"], bins=bins)

subset.boxplot(column="rate_per_ton", by="distance_bin", figsize=(8, 6))
plt.title("Wheat transportation rates")
plt.suptitle("") 
plt.xlabel("Distance, km")
plt.ylabel("Rate per metric tone")
plt.show()


import folium

locations = {
    "Краснодарский край, ст. Новощербиновская": (46.476207, 38.647841),  
    "Ростовская обл, Октябрьский р-н, п. Персиановский, с. Кирилловка": (47.515206, 40.107630), 
    "Ростовская обл,Октябрьский р-н, п.Персиановский, с. Кирилловка": (47.515206, 40.107630), 
    "Краснодарский край, с. Кирилловка": (44.768020, 37.726475),  
    "Тамбовская область, Мучкапский район": (51.847190, 42.468834), 
    "Успеновка, Бердянский район": (47.0602593,36.5778106),
    "Виноградный (Ставропольский кр, Новоалександровск рн)": (45.480761, 41.281310),
    "c)": None,
    "Новая Ляда (Тамбовская обл)": (52.712158, 41.669504),  
    "Подгорное (Саратовская обл, Романовский рн)": (51.666641, 42.807778), 
    "Подлесное (Ставропольский кр, Труновский рн)": (45.751584, 42.109104),  
    "Курск": (51.7120849,36.0995613),
    "Краснодарский край , ст. Новощербиновская": (46.476207, 38.647841), 
    "Краснодарский край, Ейский р-н, п. Пролетарский": (46.473571, 38.426279),
    "Пензенская область, Наровчатский район, село Вьюнки": (53.856130, 43.535170),
    "Тамбовская область, Токарёвский район, деревня Чичерино": (52.035590, 41.349791),
    "Ставропольский край, Ипатовский район, Советское Руно": (45.715656, 43.229632),
    "Вьюнки, Пезенская область": (53.856130, 43.535170),
    "Пезенская область Вьюнки": (53.856130, 43.535170),
    "Тамбовская область Сампурский рн д.Марьевка": (52.686974, 41.328379),
    "Славянск на Кубани": (45.239960, 38.146727),
    "Камышеватская Ейский р-н (Краснодарский край)": (46.412892, 37.950287),
    "Тамбовская область Сампурски": (52.401513, 41.659899),
    "Тамбовская область Николина Балка": (52.738009, 41.478849),
    "Николина Балка (Ставропольский край": (45.464224, 42.870250),
    "Николина Балка (Ставропольский край)": (45.464224, 42.870250),
    "г. Светлоград, Ставропольский край": (45.330319, 42.852426),
    "п. Советский, Тимашевский район": (45.541125, 38.785578),
    "с. Бараники, Ростовская область": (46.480175, 41.913584),
    "Ставропольский край,Александровский район, х. Всадник": (44.601284, 43.303905),
    "Ставропольский край, Георгиевский район, п. Новоульяновский": (44.452991, 43.407721),
    "Солдато-Александровское": (44.263790, 43.753306),
    "Ставропольский край, Георгиевский район, п. Падинский": (44.387569, 43.285783),
    "Ставропольский край, Ипатовский район, п. Винодельненский": (45.817521, 43.046392),
    "с.Опытное": (43.629777, 44.136711),
    "с.Орловка": (43.985680, 43.770214),
    "село Орловка": (43.985680, 43.770214),
    "ст.Советская": (44.029370, 44.048146),
    "станица Советская": (44.029370, 44.048146),
    "станица Советская (Ставропольский Край)": (44.029370, 44.048146),
    "Расшеватка, Новоалександровск": (45.573781, 41.034871),  
    "Расшеватская, Ставропольский край": (45.5748000, 41.0349600),  
    "Инжавино": (52.324638, 42.481298),
    "Протасово (Ржаксинский р-н)": (52.132336, 41.736339),
    "Протасово (Ржаксинский р-н), ТАМБОВСКАЯ ОБЛАСТЬ": (52.132336, 41.736339),
    "Тамбовская область, Петровский район, село Петровское": (52.628509, 40.246992),
    "Тамбовская область, Ржаксинский район, село Протасово": (52.132336, 41.736339),
    "Тамбовская область Сампурский рн д.Марьевка НПК": (52.223414, 41.469113),
    "с.Старомарьевка (Р-Н Грачёвский Ставропольский край)": (45.098551, 42.211081),
    "Ставропольский край, Георгиевский район, с. Новозаведенное": (44.263765, 43.630692),
    "Ростовская обл,Октябрьский р-н, п.Персиановский": (47.515206, 40.107630),  
    "с. Дмитриевское (Красногвардейский район, Ставропольский край)": (45.807947, 41.890780),
    "село Гофицкое (Петровский муниципальный округ, Ставропольский край)": (45.081155, 43.040254)
}

mask = (loads["commodity_en"].str.lower() == "peas") 
peasLoads = loads.loc[mask].copy()

peasLoads["origin_coords"] = peasLoads["origin"].map(locations)
peasLoads[["lat", "lon"]] = pd.DataFrame(peasLoads["origin_coords"].tolist(), index=peasLoads.index)

m = folium.Map(
    location=[peasLoads["lat"].mean(), peasLoads["lon"].mean()],
    zoom_start=6
)

for _, row in peasLoads.dropna(subset=["lat", "lon"]).iterrows():
    popup_text = f"{row.get('commodity_en','')} {row.get('volume','')} t<br>{row['origin_en']} -> {row.get('destination_en','')}<br>{row.get('distance_km','')} km<br>{row.get('rate_per_ton','')} RUR per MT"
    popup = folium.Popup(popup_text, max_width=700, min_width=300)
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=popup,
        tooltip=row["origin_en"]
    ).add_to(m)

display(m)


query = """
CREATE OR REPLACE TABLE `cptmax.Messages.series_daily` AS
WITH base AS (
  SELECT
    DATE(timestamp) AS date,
    LOWER(commodity_en) AS commodity_en,
    rate_per_ton,
    distance_km
  FROM `cptmax.Messages.truck_loads`
  WHERE distance_km BETWEEN 100 AND 120 AND rate_per_ton BETWEEN 500 AND 1000
)
SELECT
  date,
  commodity_en,
  APPROX_QUANTILES(rate_per_ton, 100)[OFFSET(50)] AS price_med 
FROM base
GROUP BY date, commodity_en;
"""
job = client.query(query)
job.result() 
print("cptmax.Messages.series_daily has been created")




query = """
CREATE OR REPLACE TABLE `cptmax.Messages.rate_predictions` AS
SELECT *
FROM AI.FORECAST(
  TABLE `cptmax.Messages.series_daily`,
  data_col      => 'price_med',
  timestamp_col => 'date',
  id_cols       => ['commodity_en'],
  horizon       => 3,        
  confidence_level => 0.8    
);
"""
job = client.query(query)
job.result() 
print("cptmax.Messages.rate_predictions has been created")


query = """
SELECT
  date,
  commodity_en,
  price_med,
  NULL AS forecast_value,
  NULL AS lo,
  NULL AS hi
FROM `cptmax.Messages.series_daily`
WHERE commodity_en='wheat'

UNION ALL

SELECT
  DATE(forecast_timestamp) AS date,
  commodity_en,
  NULL AS price_med,
  ROUND(forecast_value),
  ROUND(prediction_interval_lower_bound) AS lo,
  ROUND(prediction_interval_upper_bound) AS hi
FROM `cptmax.Messages.rate_predictions`
WHERE commodity_en='wheat'
ORDER BY date
"""

wheatRate = client.query(query).to_dataframe()
print(wheatRate)


import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import HTML, display

pio.renderers.default = "notebook_connected"

hist = wheatRate[wheatRate['price_med'].notnull()]
pred = wheatRate[wheatRate['forecast_value'].notnull()]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hist['date'], y=hist['price_med'],
    mode='lines+markers',
    name='History',
    line=dict(color='black')
))

fig.add_trace(go.Scatter(
    x=pred['date'], y=pred['forecast_value'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=list(pred['date']) + list(pred['date'][::-1]),
    y=list(pred['hi']) + list(pred['lo'][::-1]),
    fill='toself',
    fillcolor='rgba(0, 0, 255, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name="80% interval"
))

fig.update_layout(
    title="Wheat transportation rate forecast for 100-200 km",
    xaxis_title="Date",
    yaxis_title="Rate per tone",
    template="plotly_white"
)

last_hist = hist.iloc[-1]
first_pred = pred.iloc[0]
fig.add_trace(go.Scatter(
    x=[last_hist['date'], first_pred['date']],
    y=[last_hist['price_med'], first_pred['forecast_value']],
    mode='lines',
    line=dict(color='blue', dash='dot'),
    showlegend=False
))

fig.show()

html = pio.to_html(fig, include_plotlyjs="inline", full_html=False)
display(HTML(html))

