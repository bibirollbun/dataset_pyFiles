# ---------------------------------------------------------------
# Install required BigFrame dependencies
# ---------------------------------------------------------------

!pip install --quiet --upgrade bigframes google-cloud-automl google-cloud-translate google-ai-generativelanguage adjustText tensorflow 2>/dev/null


# ---------------------------------------------------------------
# Import essential libraries 
# ---------------------------------------------------------------
import json

# BigFrames
from bigframes.ml.llm import GeminiTextGenerator
import bigframes.pandas as bpd

# Data & Cloud
from google.cloud import bigquery
from google.oauth2 import service_account
from kaggle_secrets import UserSecretsClient

# Display & Visualization
from adjustText import adjust_text
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress only known, non-critical warnings 
import warnings
from bigframes import exceptions as bfe
warnings.filterwarnings("ignore", category=FutureWarning, module="bigframes")
warnings.filterwarnings("ignore", category=bfe.DefaultIndexWarning)
warnings.filterwarnings("ignore", category=bfe.PreviewWarning)

# Checkpoint
print(">>> Packages imported successfully.")


# ---------------------------------------------------------------
# 1. Load GCP service account from Kaggle secrets
# ---------------------------------------------------------------
user_secrets = UserSecretsClient()
keyfile_dict = json.loads(user_secrets.get_secret("GCP_KEY"))

# ---------------------------------------------------------------
# 2. Build credentials with proper scopes
# ---------------------------------------------------------------
scopes = [
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/cloud-platform"
]
credentials = service_account.Credentials.from_service_account_info(
    keyfile_dict, scopes=scopes
)

# ---------------------------------------------------------------
# 3. Configure BigFrames 
# ---------------------------------------------------------------
PROJECT_ID = keyfile_dict["project_id"]
DATASET = "carbon_demo"

bpd.options.bigquery.project = PROJECT_ID
bpd.options.bigquery.location = "us"     # adjust if your dataset is in another location
bpd.options.bigquery.credentials = credentials

print(f">>> BigFrames initialized for project: {PROJECT_ID}, {DATASET}")

# ---------------------------------------------------------------
# 4. Initialize BigQuery client
# ---------------------------------------------------------------
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

print(">>> BigQuery client initialized.")

# ---------------------------------------------------------------
# 5. Initialize Gemini generator
# ---------------------------------------------------------------
connection_name = user_secrets.get_secret("BIGFRAMES_CONNECTION")
gen = GeminiTextGenerator(connection_name=connection_name)
print(f">>> GeminiTextGenerator initialized with connection: {connection_name}")


# ---------------------------------------------------------------
# Helper function: Run GeminiTextGenerator and print extracted text
# ---------------------------------------------------------------
def run_gemini(prompt: str, df=None, max_rows: int = 20, return_text: bool = False):
    """
    Wrapper for GeminiTextGenerator to generate narrative insights.

    Args:
        prompt (str): The base text prompt.
        df (optional): A pandas or BigFrames DataFrame to append as tabular context.
        max_rows (int): Number of rows to include from the DataFrame (default 20).
        return_text (bool): If True, return the raw string in addition to Markdown display.
    """
    # If DataFrame is provided, format it neatly as text
    if df is not None:
        if hasattr(df, "to_pandas"):  # BigFrames DataFrame
            df = df.to_pandas()
        df_text = df.head(max_rows).to_string(index=False)
        prompt = f"{prompt}\n\nHere is the data:\n{df_text}"

    # Wrap prompt in BigFrames DataFrame
    df_prompt = bpd.DataFrame({"prompt": [prompt]})
    result = gen.predict(df_prompt["prompt"])

    # Extract output
    text = result.to_pandas().iloc[0]["ml_generate_text_llm_result"]

    # Always display Markdown
    display(Markdown(text))

    # Only return raw string if asked
    if return_text:
        return text


df = bpd.read_gbq(f"{PROJECT_ID}.carbon_demo.daily_emissions")
print(df.head())


df_cost_emissions = bpd.read_gbq(
    f"SELECT date, service, region, usage_amount, cost_usd, emissions_kgco2 "
    f"FROM `{PROJECT_ID}.carbon_demo.daily_cost_emissions`"
)
print(df_cost_emissions.head())


# ---------------------------------------------------------------
# Service Efficiency Metrics
# ---------------------------------------------------------------

# Compute carbon efficiency â†’ kgCOâ‚‚ per $1 spent
df_cost_emissions["carbon_efficiency"] = (
    df_cost_emissions["emissions_kgco2"] / df_cost_emissions["cost_usd"]
)

# Aggregate at service level
df_efficiency = (
    df_cost_emissions
    .groupby("service")
    .agg({"carbon_efficiency": "mean"})
    .reset_index()
    .sort_values("carbon_efficiency", ascending=False)
)

print(">>> Top services by carbon intensity (kgCOâ‚‚ per $1):")
print(df_efficiency.head().to_pandas())


# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------

# Convert to Pandas for plotting
df_efficiency_pd = df_efficiency.to_pandas()

plt.figure(figsize=(10,6))
plt.barh(df_efficiency_pd["service"], df_efficiency_pd["carbon_efficiency"], color="skyblue")
plt.xlabel("Carbon Efficiency (kgCOâ‚‚ per $1)")
plt.ylabel("Service")
plt.title("Service-Level Carbon Efficiency")
plt.gca().invert_yaxis()  # so highest appears at top
plt.show()


# ---------------------------------------------------------------
# AI Insights: Service Efficiency
# ---------------------------------------------------------------
efficiency_prompt = """
You are a sustainability-finance analyst. 
Analyze the service-level carbon efficiency (kgCOâ‚‚ per $1 spent).

Provide insights:
1. Which services are the most carbon-intensive per dollar (least efficient)?
2. Which services are relatively greener per dollar spent?
3. What implications could this have for costâ€“carbon optimization?
"""

# Pass DataFrame directly
run_gemini(efficiency_prompt, df=df_efficiency_pd)


# Compute per-unit averages at service level
df_scatter = (
    df_cost_emissions
    .groupby("service")
    .agg({
        "cost_usd": "mean",
        "usage_amount": "mean",
        "emissions_kgco2": "mean"
    })
    .reset_index()
)

# Derive cost/unit and emissions/unit
df_scatter["cost_per_unit"] = df_scatter["cost_usd"] / df_scatter["usage_amount"]
df_scatter["emissions_per_unit"] = df_scatter["emissions_kgco2"] / df_scatter["usage_amount"]

# Convert to Pandas for plotting
df_scatter_pd = df_scatter.to_pandas()

# Scatter plot
from adjustText import adjust_text

plt.figure(figsize=(8,6))
plt.scatter(df_scatter_pd["cost_per_unit"], df_scatter_pd["emissions_per_unit"], s=100, alpha=0.7)

texts = []
for _, row in df_scatter_pd.iterrows():
    texts.append(plt.text(row["cost_per_unit"], row["emissions_per_unit"], row["service"]))

adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

plt.xlabel("Cost per Unit (USD)")
plt.ylabel("Emissions per Unit (kgCOâ‚‚)")
plt.title("Cost vs Emission Scatterplot (Service-Level)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# ---------------------------------------------------------------
# AI Insights
# ---------------------------------------------------------------
scatter_prompt = """
You are a sustainability-finance analyst.
Analyze this scatterplot of services (X = cost per unit, Y = emissions per unit).

Provide insights:
1. Which services are â€œcheap but dirtyâ€� (low cost/unit but high emissions/unit)?
2. Which services are â€œexpensive but greenâ€� (high cost/unit but low emissions/unit)?
3. Which services appear balanced or efficient?
4. What strategic tradeoffs does this scatterplot reveal?
"""

run_gemini(scatter_prompt, df=df_scatter_pd[["service","cost_per_unit","emissions_per_unit"]])


# Reuse scatter DataFrame
df_frontier = df_scatter_pd.copy()

# Function to find Pareto-efficient points
def is_pareto_efficient(costs, emissions):
    """
    Returns a boolean mask for Pareto-efficient points (minimizing both cost & emissions).
    """
    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if is_efficient[i]:
            # Any point strictly cheaper AND cleaner dominates point i
            dominated = (costs < costs[i]) & (emissions < emissions[i])
            if dominated.any():
                is_efficient[i] = False
    return is_efficient

# Apply Pareto check
efficient_mask = is_pareto_efficient(
    df_frontier["cost_per_unit"].values,
    df_frontier["emissions_per_unit"].values
)
df_frontier["pareto_efficient"] = efficient_mask

# Plot
plt.figure(figsize=(9,7))
plt.scatter(
    df_frontier["cost_per_unit"],
    df_frontier["emissions_per_unit"],
    s=80, alpha=0.6, label="All Services"
)

# Highlight Pareto-efficient points
efficient_points = df_frontier[df_frontier["pareto_efficient"]]
plt.scatter(
    efficient_points["cost_per_unit"],
    efficient_points["emissions_per_unit"],
    s=120, color="red", label="Pareto Frontier"
)

# Connect Pareto-efficient points
efficient_sorted = efficient_points.sort_values("cost_per_unit")
plt.plot(
    efficient_sorted["cost_per_unit"],
    efficient_sorted["emissions_per_unit"],
    color="red", linestyle="--", linewidth=1
)

# Adjusted labels
texts = []
for _, row in efficient_sorted.iterrows():
    plt.text(row["cost_per_unit"], row["emissions_per_unit"], row["service"], 
             fontsize=9, weight="bold", color="red")
adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

# Labels & title
plt.xlabel("Cost per Unit (USD)")
plt.ylabel("Emissions per Unit (kgCOâ‚‚)")
plt.title("Pareto Frontier of Services (Cost vs Emissions)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()


# ---------------------------------------------------------------
# AI Insights
# ---------------------------------------------------------------
pareto_prompt = """
You are a sustainability-finance analyst.
Here is a Pareto frontier of services, based on cost per unit (X) and emissions per unit (Y).

Provide insights:
1. Which services lie on the Pareto frontier (efficient tradeoffs)?
2. Which services are dominated (worse in both cost & emissions)?
3. What does the frontier suggest about balancing cost and sustainability?
4. Recommend 2â€“3 strategies for optimizing workload placement.
"""

run_gemini(
    pareto_prompt, 
    df=df_frontier[["service","cost_per_unit","emissions_per_unit","pareto_efficient"]]
)


# ---------------------------------------------------------------
# 6. Plot Carbon Tax Simulation
# ---------------------------------------------------------------

CARBON_TAX_PER_TON = 50  # USD per metric ton
CARBON_TAX_PER_KG = CARBON_TAX_PER_TON / 1000.0  # convert to per kg

# Aggregate service-level costs and emissions
df_service_costs = bpd.read_gbq(
    f"""
    SELECT
      service,
      SUM(cost_usd) AS base_cost_usd,
      SUM(emissions_kgco2) AS total_emissions
    FROM `{PROJECT_ID}.{DATASET}.daily_cost_emissions`
    GROUP BY service
    ORDER BY base_cost_usd DESC
    """
)

# Add synthetic carbon tax
df_service_costs = df_service_costs.assign(
    carbon_tax_usd = df_service_costs["total_emissions"] * CARBON_TAX_PER_KG,
    effective_cost_usd = df_service_costs["base_cost_usd"] + (df_service_costs["total_emissions"] * CARBON_TAX_PER_KG)
)

# Convert for plotting
df_service_costs_pd = df_service_costs.to_pandas()

# Plot before vs after carbon tax
import matplotlib.pyplot as plt

df_service_costs_pd.set_index("service")[["base_cost_usd", "effective_cost_usd"]].plot(
    kind="bar", figsize=(12,6), color=["steelblue", "tomato"]
)

plt.title("What-if Simulation: Service Costs With +$50/Ton COâ‚‚ Carbon Tax", fontsize=14, weight="bold")
plt.ylabel("Total Cost (USD)")
plt.xlabel("Service")
plt.legend(["Base Cost", "With Carbon Tax"])
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# AI Insights - Carbon Tax Simulation
# ---------------------------------------------------------------
carbon_tax_prompt = f"""
You are a sustainability-policy analyst.
A synthetic carbon tax of $50/ton COâ‚‚ was applied.

Here are service-level aggregates:

{df_service_costs_pd.to_string(index=False)}

Provide insights:
1. Which services see the biggest increase in effective cost?
2. Which remain relatively unaffected?
3. What does this imply for workload placement strategies?
4. Suggest 2â€“3 policy or business actions organizations might take.
"""

run_gemini(carbon_tax_prompt)


# ---------------------------------------------------------------
# 7. Sustainabilityâ€“Finance Narratives
# ---------------------------------------------------------------

narratives_prompt = f"""
You are an executive advisor specializing in sustainability finance.

Based on the analysis of services, regions, and carbon tax impacts:
- Service-level data: 
{df_service_costs_pd[['service','base_cost_usd','total_emissions','effective_cost_usd']].to_string(index=False)}

- Regional data (example, replace with your df):
{df_region_summary.to_string(index=False) if 'df_region_summary' in globals() else '...regional summary...'}

Write an **executive-style summary** (3â€“4 short paragraphs) that:
1. Identifies which services/regions are high risk (costly + carbon intensive).
2. Highlights opportunities for service substitution or regional balancing.
3. Explains financial implications of carbon taxes and policy changes.
4. Suggests 2â€“3 strategic actions organizations should consider.
"""

run_gemini(narratives_prompt)


# ---------------------------------------------------------------
# Scenario: New Product Team â€“ Detailed Recommendation
# ---------------------------------------------------------------

scenario_prompt = f"""
You are advising executives on launching a new AI-powered analytics product
targeted at enterprise customers in North America (US-East) and Europe (Finland).

Here is service-level efficiency data (cost vs emissions per unit):
{df_scatter_pd[['service','cost_per_unit','emissions_per_unit']].to_string(index=False)}

Here is regional emissions intensity data (example, replace with actual df if available):
{df_region_summary.to_string(index=False) if 'df_region_summary' in globals() else '...regional summary...'}

Provide an executive-style recommendation that:
1. Suggests which core services (e.g., BigQuery, Cloud Storage, Vertex AI, Compute Engine) 
   should form the backbone of the new product's architecture.
2. Recommends optimal deployment regions for North America and Europe, balancing cost and sustainability.
3. Explains tradeoffs between cheapest vs greenest vs hybrid strategy.
4. Outlines 3â€“4 **actionable steps** (e.g., carbon-aware FinOps policies, workload placement, 
   KPI monitoring) to ensure the new product team is both cost-efficient and carbon-conscious.
"""

run_gemini(scenario_prompt)

