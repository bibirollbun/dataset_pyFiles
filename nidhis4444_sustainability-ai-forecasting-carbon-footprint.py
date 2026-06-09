# ---------------------------------------------------------------
# Install required BigQuery dependencies
# ---------------------------------------------------------------

!pip install --quiet "protobuf==3.20.3" google-cloud-bigquery db-dtypes google-cloud-bigquery-storage > /dev/null 2>&1
print(">>> Dependencies installed (quiet mode)")


# ---------------------------------------------------------------
# Import essential libraries 
# ---------------------------------------------------------------

import warnings
import json

# Data & Cloud
from google.cloud import bigquery
from google.oauth2 import service_account
from kaggle_secrets import UserSecretsClient

# Display & Visualization
from IPython.display import Markdown
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Suppress only known, non-critical warnings
warnings.filterwarnings("ignore", message="Unable to determine type for field")

# Checkpoint
print(">>> Packages imported successfully.")
print(">>> "f"matplotlib {plt.matplotlib.__version__}, pandas {pd.__version__}, seaborn {sns.__version__}")


# ---------------------------------------------------------------
# Setup: Secrets, Credentials, BigQuery Client & Helper Functions
# ---------------------------------------------------------------

# 1. Load secret JSON as string from Kaggle secrets
user_secrets = UserSecretsClient()
keyfile_dict = json.loads(user_secrets.get_secret("GCP_KEY"))

# 2. Create credentials object
credentials = service_account.Credentials.from_service_account_info(keyfile_dict)

print(">>> Credentials setup successfully.")


# ===============================================================
# Global Dataset Configuration
# ===============================================================

# 1. Dataset name
DATASET = "carbon_demo"

# 2. Project ID (update this with your own GCP project ID)
# Example: PROJECT_ID = "my-gcp-project"
PROJECT_ID = keyfile_dict["project_id"]

# 3. Initialize BigQuery client
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

# 4. Connection name for AI functions (from Kaggle secrets)
connection_name = user_secrets.get_secret("BIGFRAMES_CONNECTION")

print(">>> BigQuery client initialized.")
print(">>> "f"Using project: {PROJECT_ID}, dataset: {DATASET}")


# ---------------------------------------------------------------
# Helper functions for running SQL queries
# ---------------------------------------------------------------
def bq(query: str, params: dict = None):
    """
    Run a BigQuery SQL query and return a Pandas DataFrame.
    
    Args:
        query (str): SQL query string
        params (dict, optional): Named query parameters {name: value}
    
    Returns:
        pd.DataFrame: Query results
    """
    job_config = None
    if params:
        # Build query parameter config if params provided
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(name, "STRING", value)
                for name, value in params.items()
            ]
        )

    # Execute query (using REST API client for Kaggle compatibility)
    return client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=False)

print(">>> SQL query helper ready.")


# ---------------------------------------------------------------
# Helper function: Run AI.GENERATE and print extracted text
# ---------------------------------------------------------------
def run_ai_generate(prompt: str):
    """
    Executes AI.GENERATE with a given prompt and prints
    the AI-generated response as Markdown text.
    """
    query = f"""
    SELECT AI.GENERATE(
      @prompt,
      connection_id => "projects/{PROJECT_ID}/locations/us/connections/{connection_name}"
    ) AS insights
    """
    df = bq(query, params={"prompt": prompt})

    # Extract AI-generated text
    ai_text = df["insights"][0]["full_response"]["candidates"][0]["content"]["parts"][0]["text"]

    print(">>> AI Insights:\n")
    display(Markdown(ai_text))

print(">>> AI.GENERATE query helper ready.")


# ---------------------------------------------------------------
# 1. Preview services (lookup table for services metadata)
# ---------------------------------------------------------------
df_services = bq(f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET}.services`
""")
print(">>> Services preview loaded. Rows:", len(df_services))
df_services


# ---------------------------------------------------------------
# 2. Preview projects (lookup table for project metadata)
# ---------------------------------------------------------------
df_projects = bq(f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET}.projects`
""")
print(">>> Projects preview loaded. Rows:", len(df_projects))
df_projects


# ---------------------------------------------------------------
# 3. Preview emission_factors (lookup table for regions & carbon intensity)
# ---------------------------------------------------------------
df_regions = bq(f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET}.emission_factors`
""")
print(">>> Emission_factors preview loaded. Rows:", len(df_regions))
df_regions


# ---------------------------------------------------------------
# 4. Preview daily_usage (raw fact table)
# ---------------------------------------------------------------
df_daily_usage = bq(f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET}.daily_usage`
""")
print(">>> Daily_usage preview loaded. Rows:", len(df_daily_usage))
df_daily_usage.head()


# ---------------------------------------------------------------
# 5. Preview daily_emissions (core fact table)
# ---------------------------------------------------------------
df_daily_emissions = bq(f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
""")
print(">>> Daily_emissions preview loaded. Rows:", len(df_daily_emissions))
df_daily_emissions.head()


# ---------------------------------------------------------------
# Daily usage time series for top 3 services
# ---------------------------------------------------------------
df_top_service_trends = bq(f"""
    WITH top_services AS (
      SELECT service
      FROM `{PROJECT_ID}.{DATASET}.daily_usage`
      GROUP BY service
      ORDER BY SUM(usage_amount) DESC
      LIMIT 3
    )
    SELECT 
      date,
      service,
      CAST(SUM(usage_amount) AS INT64) AS total_usage
    FROM `{PROJECT_ID}.{DATASET}.daily_usage`
    WHERE service IN (SELECT service FROM top_services)
    GROUP BY date, service
    ORDER BY date, service
""")

print(">>> Daily usage trends for top 3 services loaded.")

# Pivot for plotting
pivot_top = df_top_service_trends.pivot(index="date", columns="service", values="total_usage").fillna(0)

# Plot
pivot_top.plot(figsize=(14,6), linewidth=1.5, marker="o")
plt.title("Daily Usage Trends for Top 3 Services")
plt.xlabel("Date")
plt.ylabel("Total Usage")
plt.legend(title="Service", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.show()


# ---------------------------------------------------------------
# Last 7 days daily usage by service
# ---------------------------------------------------------------
df_recent_trends = bq(f"""
    SELECT
      date,
      service,
      CAST(SUM(usage_amount) AS INT64) AS total_usage
    FROM `{PROJECT_ID}.{DATASET}.daily_usage`
    WHERE date >= DATE_SUB((SELECT MAX(date) FROM `{PROJECT_ID}.{DATASET}.daily_usage`), INTERVAL 6 DAY)
    GROUP BY date, service
    ORDER BY date, service
""")

print(">>> Recent 7-day daily usage by service loaded.")
pivot_recent = df_recent_trends.pivot(index="date", columns="service", values="total_usage").fillna(0)


# ---------------------------------------------------------------
# Define enterprise vs consumer service groups
# ---------------------------------------------------------------
enterprise_services = ["compute_engine", "bigquery", "gke", "vertex_ai"]
consumer_services   = ["cloud_loadbalancer", "pubsub", "cloud_run", "genai_api", "cloud_functions", "cloud_storage"]

# Enterprise subset
pivot_enterprise = pivot_recent[enterprise_services]
# Consumer subset
pivot_consumer = pivot_recent[consumer_services]

# ---------------------------------------------------------------
# Plot enterprise services
# ---------------------------------------------------------------
pivot_enterprise.plot(figsize=(14,6), linewidth=1.5, marker="o")
plt.title("Last 7 Days - Enterprise Services (Compute, BigQuery, GKE, Vertex AI)")
plt.xlabel("Date")
plt.ylabel("Total Usage")
plt.legend(title="Service", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.show()

# ---------------------------------------------------------------
# Plot consumer services
# ---------------------------------------------------------------
pivot_consumer.plot(figsize=(14,6), linewidth=1.5, marker="o")
plt.title("Last 7 Days - Consumer Services (LoadBalancer, Pub/Sub, Cloud Run, GenAI, Functions, Storage)")
plt.xlabel("Date")
plt.ylabel("Total Usage")
plt.legend(title="Service", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.show()


# ---------------------------------------------------------------
# Build text prompt summarizing the recent 7-day usage data
# ---------------------------------------------------------------
service_prompt = (
    "Here is the last 7 days of daily usage (raw units) by service: "
    + "; ".join(
        f"{row['date']}:{row['service']}={row['total_usage']}"
        for _, row in df_recent_trends.iterrows()
    )
    + ". For reference: Enterprise Services include [Compute Engine, BigQuery, GKE, Vertex AI]; "
      "Consumer Services include [Load Balancer, Pub/Sub, Cloud Run, GenAI API, Cloud Functions, Cloud Storage]. "
      "Provide insights on: "
      "(1) Which services are most dominant in the last week, "
      "(2) How Enterprise vs Consumer services behave differently on weekdays vs weekends, "
      "(3) What these patterns could imply for energy usage and emissions."
)

print(">>> Service prompt built for AI.GENERATE.")
print(service_prompt[:500] + " ...")  # preview only the first 500 chars

run_ai_generate(service_prompt)


# ---------------------------------------------------------------
# 1. Load total emissions directly from v_emissions_total
# ---------------------------------------------------------------
df_total_history = bq(f"""
  SELECT date, emissions_kgco2
  FROM `{PROJECT_ID}.{DATASET}.v_emissions_total`
  ORDER BY date
""")

print(">>> Historical total emissions loaded from v_emissions_total. Rows:", len(df_daily_emissions))
print(df_total_history.head())

# ---------------------------------------------------------------
# 2. Plot
# ---------------------------------------------------------------
plt.figure(figsize=(14,6))

plt.plot(
    df_total_history["date"],
    df_total_history["emissions_kgco2"],
    label="Total Historical Emissions",
    color="blue"
)

# Formatting
plt.title("Total Historical Emissions Over Time", fontsize=14, weight="bold")
plt.xlabel("Date")
plt.ylabel("Emissions (kgCOâ‚‚)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# Forecast Total Emissions (next 30 days)
# ---------------------------------------------------------------
df_forecast_total = bq(f"""
  SELECT *
  FROM AI.FORECAST(
    TABLE `{PROJECT_ID}.{DATASET}.v_emissions_total`,
    timestamp_col => 'date',
    data_col      => 'emissions_kgco2',
    horizon       => 30,
    confidence_level => 0.9
  )
""")

print(">>> Forecast generated for total emissions (next 30 days).")
df_forecast_total.head()

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
plt.figure(figsize=(14,6))

# Historical emissions
plt.plot(
    df_total_history["date"], 
    df_total_history["emissions_kgco2"], 
    label="Historical", 
    color="blue"
)

# Forecast values
plt.plot(
    df_forecast_total["forecast_timestamp"], 
    df_forecast_total["forecast_value"], 
    label="Forecast", 
    color="orange", 
    linestyle="--", 
    marker="o"
)

# Confidence intervals
plt.fill_between(
    df_forecast_total["forecast_timestamp"],
    df_forecast_total["prediction_interval_lower_bound"],
    df_forecast_total["prediction_interval_upper_bound"],
    color="orange",
    alpha=0.2,
    label="90% Confidence Interval"
)

# Formatting
plt.title("Historical + Forecast of Total Emissions (Next 30 Days)", fontsize=14, weight="bold")
plt.xlabel("Date")
plt.ylabel("Emissions (kgCOâ‚‚)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# Find top project by total emissions
# ---------------------------------------------------------------
df_top_project = bq(f"""
    SELECT 
      project_id,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    GROUP BY project_id
    ORDER BY total_emissions DESC
    LIMIT 1
""")

top_project = df_top_project["project_id"].iloc[0]
print(f">>> Top project by emissions: {top_project}")


# ---------------------------------------------------------------
# MoM emissions by region for top project
# ---------------------------------------------------------------
df_proj_region_mom = bq(f"""
    SELECT
      FORMAT_DATE('%Y-%m', date) AS month_str,
      region,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    WHERE project_id = '{top_project}'
    GROUP BY month_str, region
    ORDER BY month_str, region
""")

# Pivot for plotting
pivot_region = df_proj_region_mom.pivot(index="month_str", columns="region", values="total_emissions").fillna(0)

pivot_region.plot(figsize=(14,6), linewidth=2, marker="o")
plt.title(f"MoM Emissions by Region for {top_project}")
plt.ylabel("Emissions (kgCOâ‚‚)")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# ---------------------------------------------------------------
# MoM emissions by service for top project
# ---------------------------------------------------------------
df_proj_service_mom = bq(f"""
    SELECT
      FORMAT_DATE('%Y-%m', date) AS month_str,
      service,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    WHERE project_id = '{top_project}'
    GROUP BY month_str, service
    ORDER BY month_str, service
""")

# Pivot for plotting
pivot_service = df_proj_service_mom.pivot(index="month_str", columns="service", values="total_emissions").fillna(0)

pivot_service.plot(figsize=(14,6), linewidth=2, marker="o")
plt.title(f"MoM Emissions by Service for {top_project}")
plt.ylabel("Emissions (kgCOâ‚‚)")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.legend(title="Service", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# ---------------------------------------------------------------
# AI Insights for top project (services + regions)
# ---------------------------------------------------------------
proj_prompt = (
    f"Here is the month-on-month emissions data (kgCOâ‚‚) for project {top_project}. "
    "By service: " +
    "; ".join(
        f"{row['month_str']}:{row['service']}={row['total_emissions']:.0f}"
        for _, row in df_proj_service_mom.iterrows()
    ) +
    ". And by region: " +
    "; ".join(
        f"{row['month_str']}:{row['region']}={row['total_emissions']:.0f}"
        for _, row in df_proj_region_mom.iterrows()
    ) +
    ". Provide insights: (1) Which services and regions are the main contributors, "
    "(2) Do emissions show growth or reduction, (3) Are there seasonal or regional patterns?"
)

run_ai_generate(proj_prompt)


# ---------------------------------------------------------------
# MoM emissions by service Ã— region
# ---------------------------------------------------------------
df_hist_service_region = bq(f"""
    SELECT
      FORMAT_DATE('%Y-%m', date) AS month_str,
      service,
      region,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    GROUP BY month_str, service, region
    ORDER BY month_str, service, region
""")

print(">>> Combined historical table (service Ã— region) ready.")
df_hist_service_region.head()


# ---------------------------------------------------------------
# Plot a HeatMap
# ---------------------------------------------------------------
# Pivot raw emissions into service Ã— region
pivot_raw = df_hist_service_region.pivot_table(
    index="service", 
    columns="region", 
    values="total_emissions", 
    aggfunc="sum", 
    fill_value=0
)

# Normalize each service row (0â€“1 scale across regions)
pivot_norm = pivot_raw.div(pivot_raw.max(axis=1), axis=0)

plt.figure(figsize=(14,8))
sns.heatmap(pivot_norm, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Normalized Emissions (0â€“1)'})
plt.title("Normalized Emissions by Service Ã— Region")
plt.ylabel("Service")
plt.xlabel("Region")
plt.show()


# ---------------------------------------------------------------
# Build prompt from historical service Ã— region emissions
# ---------------------------------------------------------------
service_region_prompt = (
    "Here is the month-on-month emissions data (kgCOâ‚‚) grouped by service and region: "
    + "; ".join(
        f"{row['month_str']}:{row['service']}:{row['region']}={int(row['total_emissions'])}"
        for _, row in df_hist_service_region.iterrows()
    )
    + ". Provide insights on: "
      "(1) Which regions dominate emissions for each service, "
      "(2) Which services show clear regional patterns, "
      "(3) Any seasonal trends visible across months, "
      "(4) Overall interpretation of service-region interaction, "
      "(5) Energy-mix implications (renewable-heavy vs coal-heavy regions), "
      "(6) Any anomalies or unexpected service-region combinations."
)

run_ai_generate(service_region_prompt)


# ---------------------------------------------------------------
# MoM emissions by region (all projects)
# ---------------------------------------------------------------
df_region_mom = bq(f"""
    SELECT
      FORMAT_DATE('%Y-%m', date) AS month_str,
      region,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    GROUP BY month_str, region
    ORDER BY month_str, region
""")


# ---------------------------------------------------------------
# Plot 
# ---------------------------------------------------------------
pivot_region_mom = df_region_mom.pivot(index="month_str", columns="region", values="total_emissions").fillna(0)

pivot_region_mom.plot(figsize=(14,6), linewidth=2, marker="o")
plt.title("MoM Emissions by Region")
plt.ylabel("Emissions (kgCOâ‚‚)")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# ---------------------------------------------------------------
# AI Insights on Regional Emissions
# ---------------------------------------------------------------
region_prompt = (
    "Here is the emissions data (kgCOâ‚‚) by region: " +
    "; ".join(
        f"{row['month_str']}:{row['region']}={row['total_emissions']:.0f}"
        for _, row in df_region_mom.iterrows()
    ) +
    ". And here are the emission factors (gCOâ‚‚ per kWh): " +
    "; ".join(
        f"{row['region']}={row['g_co2_per_kwh']}"
        for _, row in df_regions.iterrows()
    ) +
    ". Provide insights: (1) Which regions emit more due to higher carbon intensity and which are greener, "
    "(2) What regional energy policies might explain this, (3) Are there any region specific seasonality changes seen maybe due to festivals?"
)

run_ai_generate(region_prompt)


# ---------------------------------------------------------------
# Forecast emissions by region (next 30 days, all projects)
# ---------------------------------------------------------------
df_forecast_region = bq(f"""
SELECT *
FROM AI.FORECAST(
  (SELECT date, region, SUM(emissions_kgco2) AS emissions_kgco2
   FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
   GROUP BY date, region
   ORDER BY date, region),
  timestamp_col    => 'date',
  data_col         => 'emissions_kgco2',
  id_cols          => ['region'],
  horizon          => 30,
  confidence_level => 0.9
)
""")

print(">>> Forecast generated for regional emissions (next 30 days).")
df_forecast_region.head()


# ---------------------------------------------------------------
# Load historical emissions (daily emissions)
# ---------------------------------------------------------------
df_history_region = bq(f"""
  SELECT date, region, SUM(emissions_kgco2) AS emissions_kgco2
  FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
  GROUP BY date, region
  ORDER BY date, region
""")

# ---------------------------------------------------------------
# Prepare forecast data
# ---------------------------------------------------------------
df_forecast_region.rename(columns={
    "forecast_timestamp": "date",
    "forecast_value": "emissions_kgco2"
}, inplace=True)

df_forecast_region["date"] = pd.to_datetime(df_forecast_region["date"])
df_history_region["date"] = pd.to_datetime(df_history_region["date"])

# ---------------------------------------------------------------
# Combine historical + forecast data
# ---------------------------------------------------------------
df_history_region["type"] = "history"
df_forecast_region["type"] = "forecast"

df_combined = pd.concat([df_history_region, df_forecast_region], ignore_index=True)

print(">>> Combined historical + forecast dataset ready.")
df_combined.head()

# ---------------------------------------------------------------
# Plot historical + forecast emissions by region
# ---------------------------------------------------------------
plt.figure(figsize=(14,7))

for region in df_combined["region"].unique():
    df_r = df_combined[df_combined["region"] == region]
    
    # Plot history
    plt.plot(
        df_r[df_r["type"]=="history"]["date"],
        df_r[df_r["type"]=="history"]["emissions_kgco2"],
        label=f"{region} (history)",
        linewidth=2
    )
    
    # Plot forecast
    plt.plot(
        df_r[df_r["type"]=="forecast"]["date"],
        df_r[df_r["type"]=="forecast"]["emissions_kgco2"],
        linestyle="--",
        label=f"{region} (forecast)"
    )

plt.title("Historical + Forecasted Emissions by Region (next 30 days)")
plt.ylabel("Emissions (kgCOâ‚‚)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# ---------------------------------------------------------------
# Coal-heavy vs Renewable-heavy regions: Historical + Forecast
# ---------------------------------------------------------------
coal_region = "asia-south1"       # Coal-heavy
renew_region = "europe-north1"   # Renewable-heavy

plt.figure(figsize=(12,6))

for region, style in [(coal_region, "red"), (renew_region, "green")]:
    df_r = df_combined[df_combined["region"] == region]

    # Plot history
    plt.plot(
        df_r[df_r["type"]=="history"]["date"],
        df_r[df_r["type"]=="history"]["emissions_kgco2"],
        label=f"{region} (history)",
        linewidth=2,
        color=style
    )

    # Plot forecast
    plt.plot(
        df_r[df_r["type"]=="forecast"]["date"],
        df_r[df_r["type"]=="forecast"]["emissions_kgco2"],
        linestyle="--",
        linewidth=2,
        color=style,
        label=f"{region} (forecast)"
    )

plt.title("Coal-Heavy vs Renewable-Heavy Region Emissions Forecast (Next 30 Days)")
plt.ylabel("Emissions (kgCOâ‚‚)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# ---------------------------------------------------------------
# AI Insights on Regional Forecasted Emissions
# ---------------------------------------------------------------
forecast_region_prompt = (
    "Here is the historical+forecasted emissions (kgCOâ‚‚) by region: " +
    "; ".join(
        f"{row['date']}:{row['region']}={row['emissions_kgco2']:.0f}"
        for _, row in df_combined.iterrows()
    ) +
    ". Provide insights: (1) Which regions are expected to emit the most, "
    "(2) Which are greener, (3) Are there upward or downward trends, "
    "(4) How do regional emission factors (gCOâ‚‚/kWh) explain these results,"
    "(5) Are there any interesting insights seen for Coal-Heavy(asia-south1) vs Renewable-Heavy(europe-north1) Region Emissions?"
)

run_ai_generate(forecast_region_prompt)


# ---------------------------------------------------------------
# Total emissions by service 
# ---------------------------------------------------------------
df_service_total = bq(f"""
    SELECT
      service,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    GROUP BY service
    ORDER BY total_emissions DESC
""")

print(">>> Loaded total emissions by service")

# ---------------------------------------------------------------
# Plot pie chart
# ---------------------------------------------------------------
plt.figure(figsize=(8,8))

# Threshold: hide labels below 5%
threshold = 0.05  
total = df_service_total["total_emissions"].sum()

def label_func(pct, allvals):
    fraction = pct/100
    if fraction < threshold:
        return ""  # hide small ones
    else:
        return f"{pct:.1f}%"  # show normal label

wedges, texts, autotexts = plt.pie(
    df_service_total["total_emissions"],
    autopct=lambda pct: label_func(pct, df_service_total["total_emissions"]),
    startangle=140,
    counterclock=False
)

# Add legend with service names
plt.legend(
    wedges,
    df_service_total["service"],
    title="Services",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)

plt.title("Distribution of Total Emissions by Service")
plt.show()


# ---------------------------------------------------------------
# Month-on-month emissions grouped by service
# ---------------------------------------------------------------
df_mom_service = bq(f"""
SELECT
  EXTRACT(YEAR FROM date) AS year,
  EXTRACT(MONTH FROM date) AS month,
  service,
  ROUND(SUM(emissions_kgco2), 2) AS total_emissions
FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
GROUP BY year, month, service
ORDER BY year, month, total_emissions DESC;
""")

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------

# Create "YYYY-MM" month string for x-axis labels
df_mom_service['month_str'] = df_mom_service['year'].astype(str) + "-" + df_mom_service['month'].astype(str)
pivot_df = df_mom_service.pivot(index="month_str", columns="service", values="total_emissions").fillna(0)

pivot_df.plot(kind="bar", stacked=True, figsize=(12,6))

plt.title("Month-on-Month Emissions by Service")
plt.ylabel("kgCOâ‚‚")
plt.xlabel("Month")
plt.xticks(rotation=0, ha="right") 
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()                   

plt.show()


#---------------------------------------------------------------
# AI Insights on Month-on-Month Service Emissions
# ---------------------------------------------------------------
service_prompt = (
    "Here is the month-on-month emissions data (kgCOâ‚‚) by service: " +
    "; ".join(
        f"{row['month_str']}:{row['service']}={row['total_emissions']}"
        for _, row in df_mom_service.iterrows()
    ) +
    ". Provide insights: (1) Which services are consistently high emitters and what could be the reason, "
    "(2) Which show seasonal patterns, (3) Which services show improvement over time,"
    "(4) What could be possible drivers behind these changes?"
)

run_ai_generate(service_prompt)


# ---------------------------------------------------------------
# Forecast emissions by service (next 30 days)
# ---------------------------------------------------------------
df_forecast_service = bq(f"""
SELECT
  forecast_timestamp AS date,
  service,
  forecast_value AS emissions_kgco2,
  prediction_interval_lower_bound,
  prediction_interval_upper_bound
FROM AI.FORECAST(
  (
    SELECT 
      date, 
      service, 
      SUM(emissions_kgco2) AS emissions_kgco2
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    GROUP BY date, service
    ORDER BY date, service
  ),
  timestamp_col    => 'date',
  data_col         => 'emissions_kgco2',
  id_cols          => ['service'],
  horizon          => 30,
  confidence_level => 0.9
)
""")

print(">>> Forecast generated for service emissions (next 30 days).")
df_forecast_service.head()


# ---------------------------------------------------------------
# Historical emissions grouped by service
# ---------------------------------------------------------------
df_history_service = bq(f"""
  SELECT date, service, SUM(emissions_kgco2) AS emissions_kgco2
  FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
  GROUP BY date, service
  ORDER BY date, service
""")

df_history_service["type"] = "history"
df_forecast_service["type"] = "forecast"

df_combined_service = pd.concat([df_history_service, df_forecast_service], ignore_index=True)

# ---------------------------------------------------------------
# Plot clean historical + forecast emissions by service
# ---------------------------------------------------------------
plt.figure(figsize=(14,7))

for service in df_combined_service["service"].unique():
    df_s = df_combined_service[df_combined_service["service"] == service]

    # History
    plt.plot(
        df_s[df_s["type"]=="history"]["date"],
        df_s[df_s["type"]=="history"]["emissions_kgco2"],
        linewidth=2,
        label=f"{service} (history)"
    )

    # Forecast
    plt.plot(
        df_s[df_s["type"]=="forecast"]["date"],
        df_s[df_s["type"]=="forecast"]["emissions_kgco2"],
        linestyle="--",
        linewidth=2,
        label=f"{service} (forecast)"
    )

    # Confidence intervals
    plt.fill_between(
        df_s[df_s["type"]=="forecast"]["date"],
        df_s[df_s["type"]=="forecast"]["prediction_interval_lower_bound"],
        df_s[df_s["type"]=="forecast"]["prediction_interval_upper_bound"],
        alpha=0.15
    )

plt.title("Historical + Forecasted Emissions by Service (next 30 days)")
plt.ylabel("Emissions (kgCOâ‚‚)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# Load forecasted emissions by region (next 30 days, daily)
# ---------------------------------------------------------------
df_forecast_region = bq(f"""
    SELECT
      region,
      forecast_timestamp,
      forecast_value,
      confidence_level,
      prediction_interval_lower_bound,
      prediction_interval_upper_bound,
      ai_forecast_status
    FROM `{PROJECT_ID}.{DATASET}.forecast_by_region`
""")

# Define region groups
asia_regions = ["asia-south1", "asia-northeast1"]
europe_regions = ["europe-north1", "europe-west1"]

# Base case
df_base = df_forecast_region.copy()
df_base["scenario"] = "Base"

# Scenario A: Asia +20%
df_asia = df_forecast_region.copy()
df_asia.loc[df_asia["region"].isin(asia_regions), "forecast_value"] *= 1.2
df_asia["scenario"] = "Asia +20%"

# Scenario B: Europe +20%
df_europe = df_forecast_region.copy()
df_europe.loc[df_europe["region"].isin(europe_regions), "forecast_value"] *= 1.2
df_europe["scenario"] = "Europe +20%"

# Combine all
df_scenarios = pd.concat([df_base, df_asia, df_europe])

# ---------------------------------------------------------------
# Plot aggregated emissions per scenario
# ---------------------------------------------------------------
plt.figure(figsize=(12,6))

for scenario, df_s in df_scenarios.groupby("scenario"):
    daily_totals = df_s.groupby("forecast_timestamp")["forecast_value"].sum().reset_index()
    
    if scenario == "Base":
        # Plot base line
        plt.plot(daily_totals["forecast_timestamp"], daily_totals["forecast_value"],
                 label=scenario, linewidth=2, color="blue")
        
        # Add confidence interval shading (aggregated across regions)
        ci = df_s.groupby("forecast_timestamp").agg({
            "prediction_interval_lower_bound": "sum",
            "prediction_interval_upper_bound": "sum"
        }).reset_index()
        
        plt.fill_between(ci["forecast_timestamp"],
                         ci["prediction_interval_lower_bound"],
                         ci["prediction_interval_upper_bound"],
                         color="blue", alpha=0.2, label="Base CI")
    else:
        # Other scenarios
        plt.plot(daily_totals["forecast_timestamp"], daily_totals["forecast_value"],
                 label=scenario, linewidth=2, linestyle="--")

plt.title("Scenario Forecasts: What if Asia vs Europe grows +20%?")
plt.ylabel("Total Emissions (kgCOâ‚‚)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# ---------------------------------------------------------------
# AI Narration: Regional Growth Scenario
# ---------------------------------------------------------------
df_summary = (
    df_scenarios.groupby(["scenario"])["forecast_value"]
    .sum()
    .reset_index()
)

scenario_data_text = "; ".join(
    f"{row['scenario']}={int(row['forecast_value'])} kgCOâ‚‚"
    for _, row in df_summary.iterrows()
)

print(">>> Data string for prompt:\n", scenario_data_text)

scenario_prompt = (
    f"Here is the what-if forecast emissions data: {scenario_data_text}. "
    "Provide insights: (1) Which scenario has the highest emissions, "
    "(2) How Asia vs Europe differs due to their energy mix (coal-heavy vs renewable-heavy), "
    "(3) Key sustainability takeaways in simple, clear language."
)

run_ai_generate(scenario_prompt)


# Define coal-heavy and renewable-heavy regions
coal_regions = ["asia-south1"]
renew_regions = ["europe-north1"]

# Pick forecast data only
df_forecast_only = df_combined[df_combined["type"] == "forecast"].copy()

# Baseline emissions (no rebalancing)
baseline = df_forecast_only.groupby("date")["emissions_kgco2"].sum().reset_index()
baseline.rename(columns={"emissions_kgco2": "baseline_emissions"}, inplace=True)

# Scenario: Shift 20% of emissions from coal -> renewables
shift_percent = 0.2

df_scenario = df_forecast_only.copy()

# Reduce coal-heavy emissions
df_scenario.loc[df_scenario["region"].isin(coal_regions), "emissions_kgco2"] *= (1 - shift_percent)

# Increase renewable-heavy emissions
df_scenario.loc[df_scenario["region"].isin(renew_regions), "emissions_kgco2"] *= (1 + shift_percent)

# Rebalanced emissions
scenario = df_scenario.groupby("date")["emissions_kgco2"].sum().reset_index()
scenario.rename(columns={"emissions_kgco2": "rebalanced_emissions"}, inplace=True)

# Merge baseline vs scenario
df_compare = baseline.merge(scenario, on="date")
df_compare["delta"] = df_compare["baseline_emissions"] - df_compare["rebalanced_emissions"]

print(">>> Rebalancing scenario prepared.")
df_compare.head()

# ---------------------------------------------------------------
# Plot emissions baseline vs rebalanced
# ---------------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(df_compare["date"], df_compare["baseline_emissions"], label="Baseline", linewidth=2, color="red")
plt.plot(df_compare["date"], df_compare["rebalanced_emissions"], label="Rebalanced (20% shift)", linewidth=2, color="green")

plt.title("Region Rebalancing Scenario: Baseline vs Rebalanced Emissions")
plt.ylabel("Emissions (kgCOâ‚‚)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# ---------------------------------------------------------------
# Plot delta (emission savings)
# ---------------------------------------------------------------
plt.figure(figsize=(12,4))
plt.bar(df_compare["date"], df_compare["delta"], color="blue", alpha=0.6)
plt.axhline(0, color="black", linewidth=1)
plt.title("Emission Savings from Rebalancing (Baseline â€“ Rebalanced)")
plt.ylabel("Î” Emissions (kgCOâ‚‚)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.show()


# ---------------------------------------------------------------
# AI Narration: Region Rebalancing Scenario
# ---------------------------------------------------------------
rebalance_summary = "; ".join(
    f"{row['date'].strftime('%Y-%m-%d')}: baseline={row['baseline_emissions']:.2f}, rebalanced={row['rebalanced_emissions']:.2f}, delta={row['delta']:.2f}"
    for _, row in df_compare.iterrows()
)

rebalance_prompt = (
    "Here is a region rebalancing scenario where 20% of workloads are shifted "
    "from coal-heavy regions (asia-south1) to renewable-heavy regions (europe-north1). "
    "Data per day (kgCOâ‚‚): " + rebalance_summary +
    ". Provide insights: (1) How much emission reduction is achieved overall, "
    "(2) How consistent are the daily savings, "
    "(3) What does this imply for regional workload placement,"
    "(4) What are the pros and cons of achieving this?"
)

run_ai_generate(rebalance_prompt)


# Define policy regions (e.g., stricter carbon tax / renewable mandate regions)
policy_regions = ["europe-north1", "europe-west1"]   # renewable-heavy regions with policy push
policy_reduction = 0.15  # 15% reduction in emissions intensity

# Pick forecast data only (next 30 days)
df_forecast_only = df_combined[df_combined["type"] == "forecast"].copy()

# Baseline: no policy changes
baseline = df_forecast_only.groupby("date")["emissions_kgco2"].sum().reset_index()
baseline.rename(columns={"emissions_kgco2": "baseline_emissions"}, inplace=True)

# Apply policy: reduce emissions in targeted regions
df_policy = df_forecast_only.copy()
df_policy.loc[df_policy["region"].isin(policy_regions), "emissions_kgco2"] *= (1 - policy_reduction)

# Scenario: emissions after policy
scenario = df_policy.groupby("date")["emissions_kgco2"].sum().reset_index()
scenario.rename(columns={"emissions_kgco2": "policy_emissions"}, inplace=True)

# Merge baseline vs policy
df_compare_policy = baseline.merge(scenario, on="date")
df_compare_policy["delta"] = df_compare_policy["baseline_emissions"] - df_compare_policy["policy_emissions"]

print(">>> Policy impact scenario prepared.")
df_compare_policy.head()

# ---------------------------------------------------------------
# Plot baseline vs policy emissions
# ---------------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(df_compare_policy["date"], df_compare_policy["baseline_emissions"], label="Baseline", linewidth=2, color="red")
plt.plot(df_compare_policy["date"], df_compare_policy["policy_emissions"], label="Policy Impact (15% reduction in EU regions)", linewidth=2, color="green")

plt.title("Policy Impact Scenario: Baseline vs Policy-Adjusted Emissions")
plt.ylabel("Emissions (kgCOâ‚‚)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# ---------------------------------------------------------------
# Plot daily savings (Î” emissions)
# ---------------------------------------------------------------
plt.figure(figsize=(12,4))
plt.bar(df_compare_policy["date"], df_compare_policy["delta"], color="blue", alpha=0.6)
plt.axhline(0, color="black", linewidth=1)
plt.title("Emission Savings from Policy Impact (Baseline â€“ Policy)")
plt.ylabel("Î” Emissions (kgCOâ‚‚)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.show()


# ---------------------------------------------------------------
# AI Narration: Policy Impact Scenario
# ---------------------------------------------------------------

# Build prompt with summary data
policy_prompt = (
    "We simulated a policy impact scenario where carbon taxes or renewable mandates reduce emissions "
    "in specific regions (e.g., Europe North, Europe West). "
    "Here is the comparison between baseline and policy-adjusted emissions:\n\n" +
    "\n".join(
        f"{row['date'].strftime('%Y-%m-%d')}: Baseline={row['baseline_emissions']:.2f}, "
        f"Policy={row['policy_emissions']:.2f}, Î”={row['delta']:.2f}"
        for _, row in df_compare_policy.head(10).iterrows()  # include first 10 rows for context
    ) +
    "\n\nPlease explain:\n"
    "1. How effective the policy was in reducing emissions.\n"
    "2. Which regions likely drove these savings.\n"
    "3. Broader implications of carbon taxes/renewable mandates on cloud emissions."
)

run_ai_generate(policy_prompt)


# ---------------------------------------------------------------
# Sustainability Index: Service Ã— Region
# ---------------------------------------------------------------

# Step 1: Emission Intensity (kgCOâ‚‚ / usage unit)
df_intensity = bq(f"""
    SELECT 
      e.service,
      e.region,
      SAFE_DIVIDE(SUM(e.emissions_kgco2), SUM(u.usage_amount)) AS intensity
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions` e
    JOIN `{PROJECT_ID}.{DATASET}.daily_usage` u
      USING (date, project_id, service, region)
    GROUP BY e.service, e.region
""")

# Step 2: Trend (last 30d vs previous 30d)
df_trend = bq(f"""
    WITH base AS (
      SELECT 
        service,
        region,
        date,
        SUM(emissions_kgco2) AS emissions
      FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
      GROUP BY service, region, date
    )
    SELECT
      service,
      region,
      CASE 
        WHEN AVG(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN emissions END)
           > AVG(CASE WHEN date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY) 
                               AND DATE_SUB(CURRENT_DATE(), INTERVAL 31 DAY) 
                      THEN emissions END)
        THEN 'â†‘'
        ELSE 'â†“'
      END AS trend
    FROM base
    GROUP BY service, region
""")

# Step 3: Forecast Risk (based on confidence interval width)
df_forecast_risk = bq(f"""
    SELECT
      service,
      region,
      AVG(prediction_interval_upper_bound - prediction_interval_lower_bound) AS avg_band
    FROM `{PROJECT_ID}.{DATASET}.forecast_by_service_region`
    GROUP BY service, region
""")

# Categorize into Low/Medium/High
df_forecast_risk["risk_level"] = pd.cut(
    df_forecast_risk["avg_band"],
    bins=[-float("inf"), 1000, 5000, float("inf")],
    labels=["Low", "Medium", "High"]
)

# Step 4: Merge into Sustainability Index
df_sustainability = (
    df_intensity
    .merge(df_trend, on=["service","region"], how="left")
    .merge(df_forecast_risk[["service","region","risk_level"]], on=["service","region"], how="left")
)

print(">>> Sustainability Index prepared. Rows: ", len(df_sustainability))
df_sustainability.head()


# ---------------------------------------------------------------
# Visualize as heatmap (intensity)
# ---------------------------------------------------------------
pivot_idx = df_sustainability.pivot(index="service", columns="region", values="intensity")

plt.figure(figsize=(14,8))
sns.heatmap(pivot_idx, cmap="YlOrRd", annot=True, fmt=".2f", cbar_kws={'label': 'kgCOâ‚‚ per unit usage'})
plt.title("Sustainability Index â€“ Emission Intensity (Service Ã— Region)")
plt.ylabel("Service")
plt.xlabel("Region")
plt.show()


# ---------------------------------------------------------------
# Generative AI Narrative for Sustainability Index
# ---------------------------------------------------------------

# Build prompt with summarized data
summary_text = "; ".join(
    f"{row['service']} in {row['region']} â†’ intensity {row['intensity']}, trend {row['trend']}, risk {row['risk_level']}"
    for _, row in df_sustainability.iterrows()  
)

sustainability_prompt = (
    "You are a sustainability analyst. Here is the sustainability index of cloud services across regions:\n"
    f"{summary_text}\n\n"
    "Write a narrative analysis that explains:\n"
    "1. Which services and regions are the most carbon intensive and why? â†’ Need urgent optimization.\n"
    "2. Which regions are riskier for workload placement? â†’ Maybe diversify or rebalance.\n"
    "3. Suggest 2â€“3 strategic actions organizations could take to reduce emissions.\n"
)

run_ai_generate(sustainability_prompt)


# ---------------------------------------------------------------
# AI.GENERATE_BOOL â†’ Trend validation
# ---------------------------------------------------------------
query = f"""
WITH quarterly AS (
  SELECT
    region,
    service,
    EXTRACT(YEAR FROM date) AS year,
    EXTRACT(QUARTER FROM date) AS quarter,
    SUM(emissions_kgco2) AS total_emissions
  FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
  GROUP BY region, service, year, quarter
),
flattened AS (
  SELECT
    STRING_AGG(
      FORMAT('Q%d-%d %s/%s=%f',
             year, quarter, region, service, total_emissions),
      ', '
    ) AS text_summary
  FROM quarterly
)
SELECT AI.GENERATE_BOOL(
  CONCAT(
    'Emissions by quarter for each region and service: ', text_summary,
    '. Question: Did emissions increase in the last quarter compared to the previous one?'
  ),
  connection_id => "projects/{PROJECT_ID}/locations/us/connections/{connection_name}"
) AS increased_last_quarter
FROM flattened
"""
df_bool = bq(query)
print(">>> Did emissions increase last quarter?")

# ---------------------------------------------------------------
# Parse AI.GENERATE_BOOL output
# ---------------------------------------------------------------
raw_resp = df_bool["increased_last_quarter"][0]

# Navigate into nested JSON response
answer_text = raw_resp["full_response"]["candidates"][0]["content"]["parts"][0]["text"]

print(">>> AI.GENERATE_BOOL answer:")
print(answer_text)


print(">>> Month-on-Month service emissions summary loaded.")
df_mom_service.head()

# ---------------------------------------------------------------
# Build prompt for AI.GENERATE_INT
# ---------------------------------------------------------------
mom_summary_text = "; ".join(
    f"{row['month_str']}:{row['service']}={int(row['total_emissions'])}"
    for _, row in df_mom_service.iterrows()
)

service_prompt = (
    "From this month-on-month emissions summary, extract the number of months "
    "when vertex_ai emissions have been above 100000 kgCOâ‚‚: " + mom_summary_text
)

query = f"""
SELECT AI.GENERATE_INT(
  @prompt,
  connection_id => "projects/{PROJECT_ID}/locations/us/connections/{connection_name}"
) AS service_count
"""

# Run query
df_service_count = bq(query, params={"prompt": service_prompt})

# Extract AI response
ai_result = df_service_count["service_count"][0]["full_response"]["candidates"][0]["content"]["parts"][0]["text"]

print("AI.GENERATE_INT Output:\n", ai_result)


# ---------------------------------------------------------------
# Query Q1 vs Q2 emissions from BigQuery
# ---------------------------------------------------------------
df_q1_q2 = bq(f"""
    SELECT
      CASE
        WHEN EXTRACT(QUARTER FROM DATE(date)) = 1 THEN 'Q1'
        WHEN EXTRACT(QUARTER FROM DATE(date)) = 2 THEN 'Q2'
      END AS quarter,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    WHERE EXTRACT(QUARTER FROM DATE(date)) IN (1,2)
    GROUP BY quarter
    ORDER BY quarter
""")
print(df_q1_q2)

prompt_q1_q2 = (
    "Compare the following quarterly emissions:\n" +
    f"Q1={df_q1_q2.loc[df_q1_q2['quarter']=='Q1','total_emissions'].values[0]}, " +
    f"Q2={df_q1_q2.loc[df_q1_q2['quarter']=='Q2','total_emissions'].values[0]}.\n"
    "If Q2 is strictly greater than Q1, return only 'True'. "
    "If Q2 is less than or equal to Q1, return only 'False'."
)

query = f"""
SELECT AI.GENERATE_BOOL(
  @prompt,
  connection_id => "projects/{PROJECT_ID}/locations/us/connections/{connection_name}"
) AS result
"""
df_bool_q1_q2 = bq(query, params={"prompt": prompt_q1_q2})
print(df_bool_q1_q2["result"][0]["full_response"]["candidates"][0]["content"]["parts"][0]["text"])


# Pre-aggregate
df_asia = bq(f"""
    SELECT
      FORMAT_DATE('%Y-%m', date) AS month_str,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    WHERE region = 'asia-south1'
    GROUP BY month_str
    ORDER BY month_str
""")

# Prompt
prompt_asia = (
    "Monthly emissions for asia-south1 are: " +
    "; ".join(f"{row['month_str']}={row['total_emissions']}" for _, row in df_asia.iterrows()) +
    ". If ALL months have total emissions strictly greater than 500000, return only 'True'. "
    "Otherwise return 'False'."
)

df_bool_asia = bq(query, params={"prompt": prompt_asia})
print(df_bool_asia["result"][0]["full_response"]["candidates"][0]["content"]["parts"][0]["text"])


# Pre-aggregate total by service Ã— region
df_service_region = bq(f"""
    SELECT
      region,
      service,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    GROUP BY region, service
""")

# Pivot to find per-region leader
leaders = (
    df_service_region.loc[df_service_region.groupby("region")["total_emissions"].idxmax()]
    [["region","service","total_emissions"]]
)
print(leaders)

# Prompt
prompt_storage = (
    "Here are the highest emitting services by region: " +
    "; ".join(f"{row['region']}={row['service']}({row['total_emissions']})" for _, row in leaders.iterrows()) +
    ". If cloud_storage is the top service in more than 3 regions, return only 'True'. "
    "Otherwise return 'False'."
)

df_bool_storage = bq(query, params={"prompt": prompt_storage})
print(df_bool_storage["result"][0]["full_response"]["candidates"][0]["content"]["parts"][0]["text"])


coal_regions = ["asia-south1", "australia-southeast1"]
renewable_regions = ["europe-north1", "europe-west1"]

# Pre-aggregate per region per month
df_compare = bq(f"""
    SELECT
      region,
      FORMAT_DATE('%Y-%m', date) AS month_str,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    WHERE region IN ('asia-south1','australia-southeast1','europe-north1','europe-west1')
    GROUP BY region, month_str
    ORDER BY region, month_str
""")

# Prompt
prompt_compare = (
    "Here are monthly emissions for coal vs renewable regions: " +
    "; ".join(f"{row['region']}:{row['month_str']}={row['total_emissions']}" for _, row in df_compare.iterrows()) +
    ". Coal-heavy regions = asia-south1, australia-southeast1. "
    "Renewable-heavy regions = europe-north1, europe-west1. "
    "If renewable-heavy regions emit LESS than coal-heavy regions for every month, return only 'True'. "
    "Otherwise return 'False'."
)

df_bool_compare = bq(query, params={"prompt": prompt_compare})
print(df_bool_compare["result"][0]["full_response"]["candidates"][0]["content"]["parts"][0]["text"])


# ---------------------------------------------------------------
# Validation: Check coal vs renewable emissions by month
# ---------------------------------------------------------------
df_validate = bq(f"""
    SELECT
      FORMAT_DATE('%Y-%m', date) AS month,
      SUM(CASE WHEN region IN ('asia-south1','australia-southeast1') THEN emissions_kgco2 ELSE 0 END) AS coal_total,
      SUM(CASE WHEN region IN ('europe-north1','europe-west1') THEN emissions_kgco2 ELSE 0 END) AS renewable_total
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    GROUP BY month
    ORDER BY month
""")

# Add a check column
df_validate["renewable_less_than_coal"] = df_validate["renewable_total"] < df_validate["coal_total"]

print(">>> Validation: Renewable vs Coal-heavy monthly emissions")
print(df_validate.head(12))  # preview first 12 months

# Did renewables emit less every month?
all_true = df_validate["renewable_less_than_coal"].all()
print("\n>>> Ground truth check:")
print("Renewables always less than coal? ->", all_true)


prompt_compare = (
    "Here are monthly results of whether renewable-heavy regions emitted less than coal-heavy regions: " +
    ", ".join(str(val) for val in df_validate["renewable_less_than_coal"].tolist()) +
    ". If ALL values are True, return only 'True'. Otherwise return 'False'."
)

query = f"""
SELECT AI.GENERATE_BOOL(
  @prompt,
  connection_id => "projects/{PROJECT_ID}/locations/us/connections/{connection_name}"
) AS result
"""

df_bool_compare = bq(query, params={"prompt": prompt_compare})
answer_text = df_bool_compare["result"][0]["full_response"]["candidates"][0]["content"]["parts"][0]["text"]

print(">>> AI Answer (Renewables always less than coal?):", answer_text)


# Pre-aggregate monthly totals
df_jan_dec = bq(f"""
    SELECT
      EXTRACT(MONTH FROM date) AS month,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    WHERE EXTRACT(MONTH FROM date) IN (1, 12)
    GROUP BY month
    ORDER BY month
""")
print(df_jan_dec)

# Build prompt
prompt_growth = (
    f"Emissions in January={df_jan_dec.loc[df_jan_dec['month']==1,'total_emissions'].values[0]}, "
    f"December={df_jan_dec.loc[df_jan_dec['month']==12,'total_emissions'].values[0]}. "
    "Compute the percentage growth from January to December as ((Dec - Jan)/Jan)*100. "
    "Return only the numeric value."
)

query = f"""
SELECT AI.GENERATE_DOUBLE(
  @prompt,
  connection_id => "projects/{PROJECT_ID}/locations/us/connections/bigframes-default-connection"
) AS pct_growth
"""

df_double_growth = bq(query, params={"prompt": prompt_growth})
print(">>> % Growth Jan â†’ Dec:", df_double_growth["pct_growth"][0]["full_response"]["candidates"][0]["content"]["parts"][0]["text"])


# Pre-aggregate compute_engine emissions for Q3 (Julâ€“Sep)
df_ce_q3 = bq(f"""
    SELECT
      FORMAT_DATE('%Y-%m', date) AS month,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    WHERE service = 'compute_engine' AND EXTRACT(QUARTER FROM date) = 3
    GROUP BY month
    ORDER BY month
""")
print(df_ce_q3)

# Build prompt
prompt_ce_q3 = (
    "Here are compute_engine emissions for Q3:\n" +
    "; ".join(f"{row['month']}={row['total_emissions']}" for _, row in df_ce_q3.iterrows()) +
    ". For each consecutive pair, compute ((curr - prev)/prev)*100. "
    "Return the month-over-month % changes as a comma-separated list of numeric values."
)

df_double_ce_q3 = bq(query, params={"prompt": prompt_ce_q3})
print(">>> MoM % change for compute_engine in Q3:", df_double_ce_q3["pct_growth"][0]["full_response"]["candidates"][0]["content"]["parts"][0]["text"])


# Pre-aggregate consumer vs enterprise categories
consumer_services = ['cloud_loadbalancer','pubsub','cloud_run','genai_api']
enterprise_services = ['compute_engine','bigquery','gke','vertex_ai']

df_consumer_enterprise = bq(f"""
    SELECT
      CASE WHEN service IN ({",".join([f"'{s}'" for s in consumer_services])}) THEN 'consumer'
           WHEN service IN ({",".join([f"'{s}'" for s in enterprise_services])}) THEN 'enterprise'
           ELSE 'other'
      END AS category,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_emissions`
    GROUP BY category
""")
print(df_consumer_enterprise)

# Build prompt
cons_val = df_consumer_enterprise.loc[df_consumer_enterprise['category']=='consumer','total_emissions'].values[0]
ent_val = df_consumer_enterprise.loc[df_consumer_enterprise['category']=='enterprise','total_emissions'].values[0]

prompt_consumer = (
    f"Consumer total={cons_val}, Enterprise total={ent_val}. "
    "Compute ((Consumer - Enterprise)/Enterprise)*100. "
    "Return only the numeric percentage."
)

df_double_consumer = bq(query, params={"prompt": prompt_consumer})
print(">>> Avg % more emissions (Consumer vs Enterprise):", df_double_consumer["pct_growth"][0]["full_response"]["candidates"][0]["content"]["parts"][0]["text"])

