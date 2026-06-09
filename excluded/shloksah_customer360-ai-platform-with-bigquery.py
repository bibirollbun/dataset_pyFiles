!pip install --upgrade google-cloud-bigquery
from google.colab import auth
auth.authenticate_user()
import pandas as pd
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
# Initialize client
client = bigquery.Client(project="sunny-hope-315311")  # replace with your GCP project ID


query = """
CREATE OR REPLACE TABLE `my_database.sample_client_ids` AS
WITH fraud_ids AS (
  SELECT DISTINCT A.CLIENT_ID
  FROM `my_database.transactions_data` a left join `my_database.train_fraud_labels` b on CAST(a.id AS NUMERIC) = CAST(b.id AS NUMERIC)
  WHERE fraud_label is TRUE
),
nonfraud_ids AS (
  SELECT DISTINCT A.CLIENT_ID
  FROM `my_database.transactions_data` a left join `my_database.train_fraud_labels` b on CAST(a.id AS NUMERIC) = CAST(b.id AS NUMERIC)
  WHERE fraud_label is FALSE
  AND A.CLIENT_ID NOT IN (SELECT CLIENT_ID FROM fraud_ids) -- exclude overlap
),
fraud_sample AS (
  SELECT CLIENT_ID
  FROM fraud_ids
  ORDER BY RAND()
  LIMIT 100
),
nonfraud_sample AS (
  SELECT CLIENT_ID
  FROM nonfraud_ids
  ORDER BY RAND()
  LIMIT 100
),
combine_sample as
(SELECT CLIENT_ID FROM fraud_sample
UNION ALL
SELECT CLIENT_ID FROM nonfraud_sample)
SELECT CLIENT_ID, ROW_NUMBER() OVER (ORDER BY CLIENT_ID) AS CLIENT_ID_PROXY FROM combine_sample;
"""

client.query(query).result()
print("✅ Table created successfully")


df_samples = client.query("SELECT * FROM my_database.sample_client_ids").to_dataframe()
print(df_samples.shape)
df_samples.head()


query = """
CREATE OR REPLACE TABLE `my_database.transactions_data_sampled` AS
(SELECT a.* except(id), a.id as transaction_id, case when b.fraud_label then 1 else 0 end as fraud_label
FROM `my_database.transactions_data` a left join `my_database.train_fraud_labels` b on CAST(a.id AS NUMERIC) = CAST(b.id AS NUMERIC)
where a.client_id in (select client_id from `my_database.sample_client_ids`));
"""

client.query(query).result()
print("✅ Table created successfully")


df_transactions = client.query("select * from `my_database.transactions`").to_dataframe()
print(df_transactions.shape)
df_transactions.head()


df_mcc = pd.read_csv("https://raw.githubusercontent.com/greggles/mcc-codes/main/mcc_codes.csv")
df_transactions_w_mcc = pd.merge(df_transactions, df_mcc, on='mcc', how='left')
df_transactions_w_mcc


# Count the occurrences of each irs_description
irs_description_counts = df_transactions_w_mcc['irs_description'].value_counts().reset_index()
irs_description_counts.columns = ['irs_description', 'count']

# Select the top N descriptions for clarity (optional)
top_n = 10
irs_description_counts = irs_description_counts.head(top_n)

# Create a bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='irs_description', data=irs_description_counts, palette='viridis')
plt.title(f'Top {top_n} IRS Description Counts')
plt.xlabel('Count')
plt.ylabel('IRS Description')
plt.tight_layout()
plt.show()


query = """CREATE OR REPLACE TABLE `my_database.complaints_sampled` AS
WITH COMPLAINTS_SAMPLING AS
(
  SELECT *, ROW_NUMBER() OVER(PARTITION BY PRODUCT ORDER BY RAND()) AS RNO FROM `bigquery-public-data.cfpb_complaints.complaint_database`
  where consumer_complaint_narrative is not null
)
SELECT A.* EXCEPT(client_id_proxy),B.client_id
FROM(
      SELECT * EXCEPT(RNO), ROW_NUMBER() OVER (ORDER BY rand()) AS client_id_proxy
      FROM COMPLAINTS_SAMPLING WHERE RNO <=7
    ) a
    inner join `my_database.sample_client_ids` b
    on CAST(a.CLIENT_ID_PROXY AS NUMERIC) = CAST(b.CLIENT_ID_PROXY AS NUMERIC);

"""

client.query(query).result()
print("✅ Table created successfully")


df_complaints = client.query("select * from my_database.complaints_tagging").to_dataframe()
print(df_complaints.shape)
df_complaints.head()


query = """
CREATE OR REPLACE TABLE `my_database.users_data_sampled` AS
(SELECT * except(id), id as client_id FROM `my_database.users_data`
where id in (select client_id from `my_database.sample_client_ids`));
"""

client.query(query).result()
print("✅ Table created successfully")


df_users = client.query("select * from my_database.users_data_sampled").to_dataframe()
print(df_users.shape)
df_users


query = """
-- Cards data
CREATE OR REPLACE TABLE `my_database.card_data_sampled` AS
(SELECT *, concat(card_brand,' - ',card_type) as product_type FROM `my_database.cards_data`
where client_id in (select client_id from `my_database.sample_client_ids`));
"""

client.query(query).result()
print("✅ Table created successfully")


query = """
CREATE OR REPLACE TABLE `my_database.complaints_tagging` as
select *,
  AI.GENERATE(('''Tag the following customer complaints from this list (choose one) -
                  ['Fraud & Scam','Customer Service','Credit Reporting','Account Issues','Debt Collection','Payment Issues','Fees & Charges','Managing Payment','Others']
                  Product:''', coalesce(product,''),
                  'Issue:', coalesce(issue,''),
                  'Customer Complaint', coalesce(consumer_complaint_narrative,''),
                  'Company Response', coalesce(company_public_response,''),
                '''
                  Validation: Make sure to return only the tag from the list and nothing else
                '''),
              connection_id => 'us.my_vertex_connection',
              endpoint => 'gemini-2.5-flash',
              output_schema => 'tag STRING').tag
  from `my_database.complaints_sampled` order by complaint_id;

"""

client.query(query).result()
print("✅ Table created successfully")


df_client_customers = client.query("select * from `my_database.complaints_tagging`").to_dataframe()
print(df_client_customers.shape)
df_client_customers


from matplotlib import pyplot as plt
import seaborn as sns
# Sort the data by count in descending order
complaints_tag_counts = df_client_customers.groupby('tag').size().reset_index(name='count').sort_values(by='count', ascending=False)


# Create a bar chart with color intensity based on count
plt.figure(figsize=(10, 8))
sns.barplot(x='count', y='tag', data=complaints_tag_counts, palette='Blues_r')
plt.title('Complaint Tag Counts')
plt.xlabel('Count')
plt.ylabel('Tag')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.show()


query = """
-- AI Forecast
CREATE OR REPLACE TABLE `my_database.timeseries_forecast_spend` as
SELECT *
FROM
  AI.FORECAST(
    (
      SELECT
        TIMESTAMP_TRUNC(date, month) as transaction_month,
      client_id,
      AVG(amount) AS avg_spend,
      FROM `my_database.transactions`
      GROUP BY 1,2
    ),
    horizon => 6,
    confidence_level => 0.95,
    timestamp_col => 'transaction_month',
    data_col => 'avg_spend',
    id_cols => ['client_id']);

CREATE OR REPLACE TABLE `my_database.timeseries_forecast_avg_spend` as
SELECT client_id, avg(forecast_value) as avg_spend_forecast
FROM `my_database.timeseries_forecast_spend`
GROUP BY 1;
"""

client.query(query).result()
print("✅ Table created successfully")



df_spend_f = client.query("select * from `my_database.timeseries_forecast_spend`").to_dataframe()
print(df_spend_f.shape)
df_spend_f.head(5)


# Filter df_spend_f for client_id=338
df_spend_f_338 = df_spend_f[df_spend_f['client_id'] == 338].copy()

# Convert 'forecast_timestamp' to datetime for plotting
df_spend_f_338['forecast_timestamp'] = pd.to_datetime(df_spend_f_338['forecast_timestamp'])

# Group df_transactions_w_mcc by month and client_id and calculate average spend
df_transactions_monthly_spend = df_transactions_w_mcc.groupby(['client_id', pd.Grouper(key='date', freq='M')])['amount'].mean().reset_index()
df_transactions_monthly_spend.rename(columns={'date': 'forecast_timestamp', 'amount': 'forecast_value'}, inplace=True)

# Filter df_transactions_monthly_spend for client_id=338
df_transactions_monthly_spend_338 = df_transactions_monthly_spend[df_transactions_monthly_spend['client_id'] == 338].copy()

# Concatenate the two dataframes
df_combined_spend_338 = pd.concat([df_transactions_monthly_spend_338, df_spend_f_338[['forecast_timestamp', 'forecast_value']]], ignore_index=True)

# Sort by date for plotting
df_combined_spend_338 = df_combined_spend_338.sort_values('forecast_timestamp')

# Plot the monthly average spend
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_combined_spend_338, x='forecast_timestamp', y='forecast_value')
plt.title('Monthly Average Spend for Client ID 338')
plt.xlabel('Date')
plt.ylabel('Average Spend')
plt.grid(True)
plt.show()


query = """
-----Builidng Customer Features
CREATE OR REPLACE TABLE `my_database.customer_features` AS
SELECT
  u.client_id,
  u.current_age as age,
  u.credit_score,
  u.total_debt,
  u.yearly_income,
  CASE WHEN u.gender = 'M' THEN 1 ELSE 0 END AS gender_m,
  COUNT(t.transaction_id) AS txn_count,
  round(AVG(t.amount),0) AS avg_spend,
  round(SUM(t.amount),0) AS total_spend,
  COUNT(DISTINCT c.product_type) AS unique_products,
  round(AVG(CASE WHEN t.fraud_label=1 THEN 1 ELSE 0 END),1) AS fraud_ratio,
  round(AVG(credit_score),0) as avg_cr,
  MAX(DATE_DIFF(DATE('2020-12-31'), PARSE_DATE('%m/%Y', acct_open_date), YEAR)) as years_since_open
FROM `my_database.users_data_sampled` u
LEFT JOIN `my_database.transactions_data_sampled` t ON u.client_id = t.client_id
LEFT JOIN `my_database.card_data_sampled` c ON t.client_id = c.client_id
GROUP BY 1,2,3,4,5,6
order by 1;

--- Customer Clusters
CREATE OR REPLACE MODEL `my_database.customer_clusters`
OPTIONS(
  model_type='kmeans',
  num_trials=5,                          -- how many models to train
  max_parallel_trials=2,                 -- run trials in parallel
  hparam_tuning_algorithm='RANDOM_SEARCH',
  NUM_CLUSTERS = HPARAM_RANGE(3,10),     -- search k between 3 and 10
  standardize_features=TRUE
) AS
SELECT * except(client_id)
FROM `my_database.customer_features`;


-- Assign Cluster to Customer
CREATE OR REPLACE TABLE `my_database.customer_segments` AS
SELECT
client_id,
CENTROID_ID as cluster_id
FROM ML.PREDICT(
  MODEL `my_database.customer_clusters`,
  TABLE `my_database.customer_features`
);



--Generate Product Recommendations
CREATE OR REPLACE TABLE `my_database.customer_product_recommendation` AS
WITH cluster_products AS (
  SELECT

    cs.cluster_id,
    c.product_type,
    COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY cs.cluster_id) AS product_share
  FROM `my_database.customer_segments` cs
  JOIN `my_database.card_data_sampled` c
  ON cs.client_id = c.client_id
  GROUP BY 1,2
),
current_user_products AS (
  SELECT DISTINCT client_id, product_type
  FROM `my_database.card_data_sampled`
)
SELECT * except(RNO) FROM
(SELECT
  cs.client_id,
  cp.product_type,
  round(cp.product_share*100,2) AS recommendation_strength,
  row_number() over(partition by cs.client_id order by cp.product_share desc) as rno
FROM `my_database.customer_segments` cs
LEFT JOIN cluster_products cp
  ON cs.cluster_id = cp.cluster_id
LEFT JOIN current_user_products up
  ON cs.client_id = up.client_id AND cp.product_type = up.product_type
WHERE up.product_type IS NULL -- To remove current product that the customer already has
ORDER BY cs.client_id, recommendation_strength DESC) T
WHERE RNO = 1;


"""

client.query(query).result()
print("✅ Table created successfully")



df_prod_recom = client.query("select * from `my_database.customer_product_recommendation`").to_dataframe()
print(df_prod_recom.shape)
df_prod_recom.head(5)


df_client_features = client.query("select * from `my_database.customer_features`").to_dataframe()
print(df_client_features.shape)

df_client_clusters = client.query("select * from `my_database.customer_segments`").to_dataframe()
print(df_client_clusters.shape)

df_client_features_w_clusters = pd.merge(df_client_features, df_client_clusters, on='client_id', how='left')
print(df_client_features_w_clusters.shape)
df_client_features_w_clusters


# Select numerical features for PCA
features = ['age', 'credit_score', 'total_debt', 'yearly_income', 'txn_count', 'avg_spend', 'total_spend', 'unique_products', 'fraud_ratio', 'avg_cr', 'years_since_open']
df_pca_features = df_client_features_w_clusters[features]

# Handle potential NaN or inf values (optional, depending on data)
df_pca_features = df_pca_features.replace([np.inf, -np.inf], np.nan).dropna()


# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_pca_features)

# Apply PCA with 3 components
pca = PCA(n_components=3)
principal_components = pca.fit_transform(scaled_features)

# Create a new DataFrame with the principal components and cluster_id
df_pca = pd.DataFrame(data = principal_components, columns = ['principal_component_1', 'principal_component_2', 'principal_component_3'])

# Add the cluster_id back to the PCA dataframe, ensuring alignment with the features used for PCA
df_pca['cluster_id'] = df_client_features_w_clusters.loc[df_pca_features.index, 'cluster_id'].astype(str).values # Convert cluster_id to string for discrete colors


# Plot the interactive 3D scatter plot using Plotly
fig = px.scatter_3d(df_pca,
                    x='principal_component_1',
                    y='principal_component_2',
                    z='principal_component_3',
                    color='cluster_id',
                    title='Customer Clusters after PCA (Interactive 3D)')
fig.show()


query = """
CREATE OR REPLACE TABLE `my_database.customer360` AS
WITH transactions_summary AS (
      SELECT client_id,
      count(transaction_id) AS txn_count,
      avg(amount) as avg_txn_amt,
      SUM(amount) AS total_spend,
      AVG(fraud_label) AS fraud_score
      FROM `my_database.transactions`
      group by 1
)
SELECT
*,
  AI.GENERATE(
    CONCAT(
      "You are a customer success AI agent at a Bank. Based on this customer profile:\n",
      "Avg transaction amount: ", CAST(coalesce(t.avg_txn_amt,0) AS STRING), "\n",
      "Avg predicted future transaction: ", CAST(coalesce(tf.avg_spend_forecast) AS STRING), "\n",
      "Chance of fraud: ", CAST(coalesce(t.fraud_score) AS STRING), "\n",
      "What customer complained about: ", coalesce(cs.tag,"None"), "\n",
      "Recommended New Product based on clustering: ", coalesce(p.recommended_product,"None"), "\n",
      "Suggest the next best action in 1 sentence for increasing customer engagement and improving overall satisfaction."
    ),
    connection_id => 'us.my_vertex_connection',
    endpoint => 'gemini-2.5-flash',
    output_schema => 'recommendation STRING').recommendation,

FROM `my_database.sample_client_ids` s
LEFT JOIN transactions_summary t USING (client_id)
LEFT JOIN `my_database.complaints_tagging` cs USING (client_id)
LEFT JOIN `my_database.timeseries_forecast_avg_spend` tf USING (client_id)
LEFT JOIN `my_database.customer_product_recommendation` p USING (client_id);
"""

client.query(query).result()
print("✅ Table created successfully")



df_c360 = client.query("select * from my_database.customer360").to_dataframe()
print(df_c360.shape)
df_c360.head(5)


pd.set_option('display.max_colwidth', None)
display(df_c360[['client_id','recommendation']])

