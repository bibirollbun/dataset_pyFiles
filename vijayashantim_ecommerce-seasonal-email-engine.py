import warnings
warnings.filterwarnings('ignore')
%load_ext google.cloud.bigquery


import os
import bigframes.pandas as bpd
import bigframes.ml as bf_ml
import bigframes.ml.llm as bf_llm
from google.cloud import bigquery
from google.cloud.bigquery import magics
import warnings
warnings.filterwarnings('ignore')

PROJECT_ID = "" # Replace with your actual project ID
magics.context.project = PROJECT_ID
bpd.options.bigquery.project = PROJECT_ID


%%bigquery
CREATE OR REPLACE TABLE
  purchases.products AS
SELECT  *
FROM
  bigquery-public-data.thelook_ecommerce.products
WHERE
  LENGTH(TRIM(brand)) > 0
  AND LENGTH(TRIM(category)) > 0
  AND name IS NOT NULL
  AND department IS NOT NULL
  AND category <> 'Intimates';

  
CREATE OR REPLACE TABLE
  purchases.user_product_details AS
WITH
  items AS (
  SELECT
    item.order_id,
    item.user_id,
    item.product_id,
    item.status,
    item.sale_price,
    orders.created_at,
    item.delivered_at,
    orders.num_of_item
  FROM
    `bigquery-public-data`.`thelook_ecommerce`.`order_items` item
  JOIN
    `bigquery-public-data`.`thelook_ecommerce`.`orders` orders
  USING
    (order_id,
      user_id)
  WHERE
    user_id NOT IN (
    SELECT
      user_id
    FROM
      `bigquery-public-data`.`thelook_ecommerce`.`order_items`
    GROUP BY
      product_id,
      user_id
    HAVING
      COUNT(inventory_item_id) > 1 ) )
SELECT
  items.order_id,
  items.product_id,
  items.user_id,
  TRIM(user.email) email,
  TRIM(product.name) AS product,
  items.status,
  items.sale_price,
  items.created_at,
  items.delivered_at,
  TRIM(product.brand) brand,
  TRIM(product.category) category,
  TRIM(product.department) department,
  CONCAT(user.first_name, ' ', user.last_name) user,
  user.gender,
  CONCAT(user.city, ', ', user.country) location,
  items.num_of_item
FROM
  `items`
JOIN
  `bigquery-public-data.thelook_ecommerce.users` AS user
ON
  items.user_id = user.id
JOIN
  `bigquery-public-data.thelook_ecommerce.products` AS product
ON
  items.product_id = product.id
WHERE
  user.city IS NOT NULL
  AND user.first_name IS NOT NULL
  AND user.last_name IS NOT NULL
  AND user.gender IS NOT NULL
  AND user.email IS NOT NULL;


%%bigquery
CREATE OR REPLACE TABLE
  `purchases`.`user_product_success` AS
SELECT
  order_id,
  product_id,
  user_id,
  email,
  product,
  status,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', created_at) AS created_at,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', delivered_at) AS delivered_at,
  category,
  user,
  gender,
  location,
  num_of_item,
  '' AS user_season,
  '' AS product_season,
  '' AS seasonal_relation
FROM 
  `purchases`.`user_product_details`
WHERE
  status = 'Complete'
  AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND created_at < CURRENT_TIMESTAMP()
QUALIFY
  # to refine the data sent to ML for further processing (not necessary if processing time is less)
  COUNT(order_id) OVER (PARTITION BY user_id) > 4;


%%bigquery
CREATE OR REPLACE MODEL
  `purchases.llm_season_model`
REMOTE WITH CONNECTION `projects/---` # your project external vertex AI connect_id
OPTIONS (endpoint = 'gemini-2.5-pro');


%%bigquery
UPDATE
  `purchases`.`user_product_success` AS t1
  SET
  t1.user_season = t2.user_season
FROM (
  SELECT
    location,
    JSON_EXTRACT_SCALAR( ml_generate_text_result, '$.candidates[0].content.parts[0].text') AS user_season
  FROM
    ML.GENERATE_TEXT( MODEL `purchases`.`llm_season_model`,
      (
      SELECT
        location,
        CONCAT( 'determine current season based on the current context (user location: ', location, ', date: ', CAST(CURRENT_DATE() AS STRING), ') ', 'Instructions: check if city or country is mispelled and then correct it. The season should be one of the following: Summer, Autumn, Winter, Spring, All, NA. Respond with only a single word and no special characters or formatting') AS prompt,
      FROM
        (select distinct location from `purchases`.`user_product_success`)))) AS t2
WHERE
  t1.location = t2.location


%%bigquery
UPDATE
  `purchases`.`user_product_success` AS t
SET
  t.product_season = s.product_season
FROM (
  SELECT
    product_id,
    JSON_EXTRACT_SCALAR( ml_generate_text_result, '$.candidates[0].content.parts[0].text') AS product_season
  FROM
    ML.GENERATE_TEXT( MODEL `purchases`.`llm_season_model`,
      (
      SELECT
        CONCAT( 'Find suitable season for the product based on Product Name: ', product, '. Product Category: ', category, 'Instruction: The season should be one of the following: Summer(hot, capris, swim), Autumn(leather jackets, cardigans, long-sleeve), Winter(wool, heavy denim, sweaters), Spring(coats, light jackets, cardigans, blazers), All(used in all seasons). No special characters or formatting. If NULL values use Product Category to find season or Use All') AS prompt,
        product_id,
      FROM
        (select distinct product_id, product, category from `purchases`.`user_product_success` )) )) AS s
WHERE
 t.product_id = s.product_id;


%%bigquery
UPDATE
  `purchases`.`user_product_success` AS t1
  SET
  t1.seasonal_relation = t2.seasonal_relation
FROM (
  SELECT
    user_id,
    product_id,
    CASE
      WHEN (user_season = product_season OR product_season ='All') THEN 'current-season'
      WHEN (user_season ='Winter'AND product_season = 'Summer') OR (user_season ='Summer' AND product_season = 'Winter')
    OR (user_season ='Autumn' AND product_season = 'Spring') OR (user_season ='Spring' AND product_season = 'Autumn') THEN 'counter-season'
      WHEN (user_season = 'Autumn' AND product_season = 'Winter') OR (user_season = 'Winter' AND product_season = 'Summer') OR (user_season = 'Summer' AND product_season = 'Spring') OR (user_season = 'Spring' AND product_season = 'Autumn') THEN 'next-season'
      ELSE 'previous-season'
    END AS seasonal_relation
  FROM
    `purchases`.`user_product_success`) t2 
  WHERE
    t1.user_id = t2.user_id
    AND t1.product_id = t2.product_id


%%bigquery
CREATE OR REPLACE MODEL `purchases.embedding_model`
REMOTE WITH CONNECTION `projects/---` # your project external vertex AI connect_id
OPTIONS (ENDPOINT = 'text-embedding-005');


%%bigquery
CREATE OR REPLACE TABLE
  `purchases.product_embeddings` AS
SELECT
      id as product_id,
      name as product_name,
      brand,
      category,
      department,
      retail_price,
      content,
      ml_generate_embedding_result AS embedding
FROM
  ML.GENERATE_EMBEDDING( MODEL `purchases.embedding_model`,
    (
    SELECT
      id,
      name,
      brand,
      category,
      department,
      retail_price,
      LOWER(concat (name,
          ', ',
          category,
          ', ',
          brand,
          ', $',
          CAST(ROUND(retail_price) AS STRING))) AS content
    FROM
      purchases.products
  ),
    STRUCT( TRUE AS flatten_json_output ) )


%%bigquery
CREATE OR REPLACE VECTOR INDEX `product_embeddings_index`
ON
  `purchases.product_embeddings`(embedding) OPTIONS( index_type = 'IVF',
    distance_type = 'COSINE',
    ivf_options = '{"num_lists": 10}' );


%%bigquery
# Check coverage status for completion of indexing
SELECT
  table_name,
  index_name,
  index_status,
  coverage_percentage
FROM
  `purchases.INFORMATION_SCHEMA.VECTOR_INDEXES`;


%%bigquery
# Get the query embedding into `purchases.user_product_embedding` for similarity search
CREATE OR REPLACE TABLE
  `purchases.user_product_query` AS
WITH
  user AS (
  SELECT
    user_id,
    user,
    product_id,
    user_season,
    product_season,
    seasonal_relation
  FROM
    `purchases`.`user_product_success`
  QUALIFY
    ROW_NUMBER() OVER (PARTITION BY user ORDER BY order_id) = 1 )
SELECT
  user.*,
  embed.embedding AS query_product_embedding
FROM
  user
JOIN
  purchases.product_embeddings AS embed
USING (product_id)
ORDER BY user_id


%%bigquery
# Vector similarity search on products for user with seasonal relationship
CREATE OR REPLACE TABLE
  `purchases`.`user_seasonal_recommendation` AS
SELECT
  query.user_id,
  query.user,
  query.product_id AS query_productid,
  base.brand,
  base.product_id,
  base.product_name,
  base.category,
  base.department,
  ROUND(base.retail_price) price,
  distance,
  query.user_season,
  query.product_season,
  query.seasonal_relation
FROM
  VECTOR_SEARCH( TABLE `purchases`.`product_embeddings`,
    'embedding',
    (
    SELECT
      user_id,
      user,
      product_id,
      user_season,
      product_season,
      seasonal_relation,
      query_product_embedding
    FROM
      `purchases.user_product_query`),
    top_k => 5,
    distance_type => 'COSINE')
WHERE
  base.product_id <> query.product_id
ORDER BY
  query.product_id,
  distance DESC;


%%bigquery recommend
with user as (
      select t.user_id,
          t.query_productid,
          t.user,
          t.user_season,
          t.product_season,
          t.seasonal_relation,
          TO_JSON_STRING( ARRAY_AGG(STRUCT(product_id,
          product_name,
          brand,
          price))) AS product_details
    from 
    `purchases`.`user_seasonal_recommendation` t
    GROUP BY 1, 2, 3, 4, 5, 6
    QUALIFY
  ROW_NUMBER() OVER (PARTITION BY t.user_id ORDER BY t.query_productid) = 1
)
SELECT
  *,
  CONCAT(
    CASE seasonal_relation
      WHEN 'current-season' THEN 'promoting this in-season product, lapurchases styles.'
      WHEN 'next-season' THEN 'products early-season promotions, ahead purchase.'
      WHEN 'counter-season' THEN 'counter season products, clearance or significant discount.'
      ELSE 'limited stock products.'
  END
    ) AS part_of_prompt
FROM user t
 QUALIFY
  ROW_NUMBER() OVER (PARTITION BY t.seasonal_relation ORDER BY t.user_id) = 1


print(recommend)


llm_model = bf_llm.GeminiTextGenerator(model_name="gemini-2.0-flash-lite-001")


Temperature = 0.5


import json
def create_prompt(user, product_list, part_prompt) -> str:
    return f"""
    You are a professional email template designer for an e-commerce brand. 
    Generate a responsive HTML email template for {part_prompt}.
    The email should have a clean, modern design with inline CSS. 
    create an personalised marketing email.
    Email to {user}, 
    It must include a header with a small logo with url (https://placehold.net/7.png), impressive motivation text email banner with color background.
    Body should be qqually aligned same size rectangular section side by side for four products in the list, a product image section, a product name, 
    a brief product description, a prominent button for the call-to-action, 
    and a footer with social media links. 
    Take brand name,the product name and price for 4 products from {product_list}."
    Then centered CTA button text should be "Shop Now." 
    Ensure the design is mobile-responsive and renders correctly across major email clients like Gmail and Outlook.
    Should be equal size rectangular section.
    Do not include any conversational text and explanations.
    """


for index, row in recommend.iterrows():
    prompt_text= create_prompt(row['user'], json.dumps(row['product_details']),row['part_of_prompt'])
    prompt_df = bpd.DataFrame([prompt_text], columns=['input'])
    recommendations = llm_model.predict(prompt_df,  temperature=Temperature)
    html_output = recommendations['ml_generate_text_llm_result'].iloc[0]
    file_name = f"generated_template_{row['user_id']}-{row['query_productid']}.html"
    file_path = os.path.join(os.getcwd(), file_name)
    try:
        with open(file_path, 'w') as f:
            f.write(html_output)
            print(f"Successfully saved HTML content to: {file_path}")
    except IOError as e:
        print(f"Error writing to file: {e}")

