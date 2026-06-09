import pandas as pd
import numpy as np
import polars as pl 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
import gc
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from scipy.stats import skew, pearsonr
from wordcloud import WordCloud

sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
sns.set_palette('husl')

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



articles = pl.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
customers = pl.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')

lazy_transactions = pl.scan_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')


customers.head()


articles.head()


print("DATASET OVERVIEW")

print(f"Articles : {articles.shape}")
print(f"Customers : {customers.shape}")

transactions_schema = lazy_transactions.schema
transactions_count = lazy_transactions.select(pl.len()).collect().item(0, 0)
print(f"Transactions: {transactions_count:,} rows × {len(transactions_schema)} cols")

print("\nARTICLES DATASET:")
print("Column Name".ljust(25) + "Data Type".ljust(15) + "Non-null Count")
print("-" * 55)
for col, dtype in zip(articles.columns, articles.dtypes):
    non_null = articles.select(pl.col(col).is_not_null().sum()).item(0, 0)
    print(f"{col:<25} {str(dtype):<15} {non_null:>10,}")

print("\CUSTOMERS DATASET:")
print("Column Name".ljust(25) + "Data Type".ljust(15) + "Non-null Count")
print("-" * 55)
for col, dtype in zip(customers.columns, customers.dtypes):
    non_null = customers.select(pl.col(col).is_not_null().sum()).item(0, 0)
    print(f"{col:<25} {str(dtype):<15} {non_null:>10,}")

print("\nTRANSACTIONS DATASET:")
print("Column Name".ljust(25) + "Data Type".ljust(15) + "Sample Values")
print("-" * 55)
transactions_sample = lazy_transactions.head(3).collect()
for col, dtype in transactions_schema.items():
    sample_vals = transactions_sample.select(pl.col(col)).to_series().to_list()
    print(f"{col:<25} {str(dtype):<15} {str(sample_vals[:2])}")


print("MISSING VALUES:")

print("ARTICLES:")
articles_missing = (
    articles
    .select([
        (pl.col(col).is_null().sum() / len(articles) * 100).alias(f"{col}_missing_pct")
        for col in articles.columns
    ])
    .unpivot(variable_name="column", value_name="missing_percentage")
    .filter(pl.col("missing_percentage") > 0)
    .sort("missing_percentage", descending=True)
)

if len(articles_missing) > 0:
    print(articles_missing.to_pandas())
else:
    print("No missing values found in articles dataset")

print("CUSTOMERS:")
customers_missing = (
    customers
    .select([
        (pl.col(col).is_null().sum() / len(customers) * 100).alias(f"{col}_missing_pct")
        for col in customers.columns
    ])
    .unpivot(variable_name="column", value_name="missing_percentage")
    .filter(pl.col("missing_percentage") > 0)
    .sort("missing_percentage", descending=True)
)

if len(customers_missing) > 0:
    print(customers_missing.to_pandas())
else:
    print("No missing values found in customers dataset")

print("TRANSACTIONS - Missing Value Summary:")
transactions_missing = (
    lazy_transactions
    .select([
        (pl.col(col).is_null().sum()).alias(f"{col}_nulls")
        for col in lazy_transactions.columns
    ])
    .collect()
    .transpose(include_header=True, header_name="column")
    .rename({"column_0": "null_count"})
    .with_columns([
        (pl.col("null_count") / transactions_count * 100).alias("missing_percentage")
    ])
    .filter(pl.col("missing_percentage") > 0)
    .sort("missing_percentage", descending=True)
)

if len(transactions_missing) > 0:
    print(transactions_missing.to_pandas())
else:
    print("No missing values found in transactions dataset")



print("MEMORY USAGE")

articles_memory = articles.estimated_size("mb")
customers_memory = customers.estimated_size("mb") 

print(f"Articles dataset:    {articles_memory:.1f} MB")
print(f"Customers dataset:   {customers_memory:.1f} MB")
print(f"Transactions dataset: ~{transactions_count * len(transactions_schema) * 8 / 1024 / 1024:.0f} MB (estimated)")
print(f"TOTAL ESTIMATED:     ~{articles_memory + customers_memory + (transactions_count * len(transactions_schema) * 8 / 1024 / 1024):.0f} MB")



print("TEMPORAL COVERAGE:")

date_info = (
    lazy_transactions
    .select([
        pl.col("t_dat").str.to_date(format="%Y-%m-%d").alias("date")
    ])
    .select([
        pl.col("date").min().alias("earliest_date"),
        pl.col("date").max().alias("latest_date"),
        pl.col("date").n_unique().alias("unique_dates")
    ])
    .collect()
)

earliest_date = date_info.item(0, "earliest_date")
latest_date = date_info.item(0, "latest_date")
unique_dates = date_info.item(0, "unique_dates")

print(f"Date Range: {earliest_date} to {latest_date}")
print(f"Total Days: {(latest_date - earliest_date).days + 1} days")
print(f"Unique Dates: {unique_dates:,} days with transactions")
print(f"Coverage: {unique_dates / ((latest_date - earliest_date).days +1 )* 100:.1f}% of total days")

monthly_transactions = (
    lazy_transactions
    .with_columns([
        pl.col("t_dat").str.to_date(format="%Y-%m-%d").dt.strftime("%Y-%m").alias("year_month")
    ])
    .group_by("year_month")
    .agg([
        pl.len().alias("transaction_count")
    ])
    .sort("year_month")
    .collect()
)


print("COVERAGE")

transaction_entities = (
    lazy_transactions
    .select([
        pl.col("customer_id").n_unique().alias("unique_customers_in_transactions"),
        pl.col("article_id").n_unique().alias("unique_articles_in_transactions")
    ])
    .collect()
)

unique_customers_in_trans = transaction_entities.item(0, "unique_customers_in_transactions")
unique_articles_in_trans = transaction_entities.item(0, "unique_articles_in_transactions")

print(f"Customers in dataset:     {customers.height:,}")
print(f"Customers with purchases: {unique_customers_in_trans:,}")
print(f"Articles in catalog:      {articles.height:,}")
print(f"Articles with sales:      {unique_articles_in_trans:,}")

print("COVERAGE")

customer_coverage = (unique_customers_in_trans / customers.height) * 100
customer_no_purchases = customers.height - unique_customers_in_trans

print(f"Active customers:    {unique_customers_in_trans:,} ({customer_coverage:.1f}%)")
print(f"Inactive customers:  {customer_no_purchases:,} ({100-customer_coverage:.1f}%)")

article_coverage = (unique_articles_in_trans / articles.height) * 100
articles_no_sales = articles.height - unique_articles_in_trans

print(f"Sold articles:       {unique_articles_in_trans:,} ({article_coverage:.1f}%)")
print(f"Never sold articles: {articles_no_sales:,} ({100-article_coverage:.1f}%)")


print("INTERACTION MATRIX:")

total_possible_interactions = unique_customers_in_trans * unique_articles_in_trans
actual_interactions = transactions_count
sparsity = (actual_interactions / total_possible_interactions) * 100

print(f"Interaction Matrix Dimensions: {unique_customers_in_trans:,} × {unique_articles_in_trans:,}")
print(f"Possible interactions:         {total_possible_interactions:,}")
print(f"Actual interactions:           {actual_interactions:,}")
print(f"Matrix density:               {sparsity:.6f}%")
print(f"Sparsity:                     {100-sparsity:.4f}%")


customer_purchase_behavior = (
    lazy_transactions
    .group_by('customer_id')
    .agg([
        pl.len().alias('total_purchases'),
        pl.col('article_id').n_unique().alias('unique_articles'),
        pl.col('price').sum().alias('total_spent'),
        pl.col('price').mean().alias('avg_item_price'),
        pl.col('t_dat').min().alias('first_purchase'),
        pl.col('t_dat').max().alias('last_purchase')
    ])
    .with_columns([
        pl.col('first_purchase').str.to_date(format='%Y-%m-%d'),
        pl.col('last_purchase').str.to_date(format='%Y-%m-%d'),
    ])
    .with_columns([
        (pl.col('last_purchase') - pl.col('first_purchase')).dt.total_days().alias('customer_lifespan_days'),
        (pl.col('total_purchases') / ((pl.col('last_purchase') - pl.col('first_purchase')).dt.total_days() + 1)).alias('purchase_frequency'),
        (pl.col('total_spent') / pl.col('total_purchases')).alias('avg_basket_value')
    ])
    .collect()
)

customer_demographics_behavior = (
    customers
    .join(customer_purchase_behavior, on='customer_id', how='inner')
    .filter(pl.col('age').is_not_null())
)

print(f"Analyzing {len(customer_demographics_behavior):,} customers with complete age and purchase data")

# generations
customer_segments = customer_demographics_behavior.with_columns([
    pl.when(pl.col('age') < 25).then(pl.lit('Gen_Z'))
    .when(pl.col('age') < 35).then(pl.lit('Millennial'))  
    .when(pl.col('age') < 45).then(pl.lit('Gen_X_Young'))
    .when(pl.col('age') < 55).then(pl.lit('Gen_X_Mature'))
    .otherwise(pl.lit('Boomer_Plus'))
    .alias('generation_segment'),
    
    # fashion age groups
    pl.when(pl.col('age') < 20).then(pl.lit('Teen'))
    .when(pl.col('age') < 25).then(pl.lit('Young_Adult'))
    .when(pl.col('age') < 35).then(pl.lit('Professional'))
    .when(pl.col('age') < 45).then(pl.lit('Established'))
    .when(pl.col('age') < 55).then(pl.lit('Mature'))
    .otherwise(pl.lit('Senior'))
    .alias('fashion_age_segment'),
    
    # Customer value segments
    pl.when(pl.col('total_spent') >= pl.col('total_spent').quantile(0.9))
    .then(pl.lit('High_Value'))
    .when(pl.col('total_spent') >= pl.col('total_spent').quantile(0.7))
    .then(pl.lit('Medium_High'))
    .when(pl.col('total_spent') >= pl.col('total_spent').quantile(0.3))
    .then(pl.lit('Medium'))
    .otherwise(pl.lit('Low_Value'))
    .alias('value_segment')
])

# Age segment analysis
age_segment_analysis = (
    customer_segments
    .group_by('generation_segment')
    .agg([
        pl.count().alias('customer_count'),
        pl.col('age').mean().alias('avg_age'),
        pl.col('total_purchases').mean().alias('avg_purchases'),
        pl.col('unique_articles').mean().alias('avg_unique_items'),
        pl.col('total_spent').mean().alias('avg_total_spent'),
        pl.col('avg_item_price').mean().alias('avg_item_price'),
        pl.col('purchase_frequency').mean().alias('avg_purchase_frequency'),
        pl.col('customer_lifespan_days').mean().alias('avg_lifespan_days'),
        
        # Club membership penetration by age
        (pl.col('club_member_status') == 'ACTIVE').mean().alias('club_membership_rate'),
        
        # Fashion news engagement
        (pl.col('fashion_news_frequency') == 'Regularly').mean().alias('regular_news_rate')
    ])
    .sort('avg_age')
)

print("GENERATIONAL CUSTOMER ANALYSIS:")
print("Generation".ljust(15) + "Count".ljust(10) + "Avg Age".ljust(10) + "Avg Purchases".ljust(15) + 
      "Avg Spent".ljust(12) + "Club Rate".ljust(12) + "News Rate")
print("-" * 85)

for row in age_segment_analysis.iter_rows():
    gen, count, age, purchases, spent, _, _, _, _, club_rate, news_rate = row
    print(f"{gen:<15} {count:<10,} {age:<10.1f} {purchases:<15.1f} ${spent:<11.0f} {club_rate:<11.1%} {news_rate:.1%}")


customer_segments_pd = customer_segments.to_pandas()

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Customer Age Analysis', fontsize=16, fontweight='bold')

# Age distribution with generations
sns.histplot(data=customer_segments_pd, x='age', bins=50, kde=True, ax=axes[0,0])
axes[0,0].axvline(customer_segments_pd['age'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {customer_segments_pd["age"].mean():.1f}')
axes[0,0].set_title('Customer Age Distribution', fontsize=12)
axes[0,0].legend()

# Purchase behavior by generation
generation_summary = customer_segments_pd.groupby('generation_segment').agg({
    'total_purchases': 'mean',
    'total_spent': 'mean',
    'avg_item_price': 'mean'
}).reset_index()

generation_summary['customer_count'] = customer_segments_pd.groupby('generation_segment').size().values

sns.barplot(data=customer_segments_pd, x='generation_segment', y='total_purchases', ax=axes[0,1])
axes[0,1].set_title('Average Purchases by Generation', fontsize=12)
axes[0,1].tick_params(axis='x', rotation=45)

# Spending patterns by age
sns.scatterplot(data=customer_segments_pd.sample(5000), x='age', y='total_spent', 
                hue='generation_segment', alpha=0.6, ax=axes[0,2])
axes[0,2].set_title('Total Spending vs Age', fontsize=12)
axes[0,2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Club membership by age groups
club_by_age = customer_segments_pd.groupby('fashion_age_segment')['club_member_status'].apply(
    lambda x: (x == 'ACTIVE').mean() * 100
).reset_index()
club_by_age.columns = ['fashion_age_segment', 'membership_rate']

sns.barplot(data=club_by_age, x='fashion_age_segment', y='membership_rate', ax=axes[1,0])
axes[1,0].set_title('Club Membership Rate by Age Group', fontsize=12)
axes[1,0].set_ylabel('Membership Rate (%)')
axes[1,0].tick_params(axis='x', rotation=45)

# Purchase diversity (unique articles) by age
sns.boxplot(data=customer_segments_pd, x='generation_segment', y='unique_articles', ax=axes[1,1])
axes[1,1].set_title('Purchase Diversity by Generation', fontsize=12)
axes[1,1].tick_params(axis='x', rotation=45)

# Fashion news engagement by age
news_by_age = customer_segments_pd.groupby('generation_segment')['fashion_news_frequency'].apply(
    lambda x: (x == 'Regularly').mean() * 100
).reset_index()
news_by_age.columns = ['generation_segment', 'regular_news_rate']

sns.barplot(data=news_by_age, x='generation_segment', y='regular_news_rate', ax=axes[1,2])
axes[1,2].set_title('Fashion News Engagement by Age', fontsize=12)
axes[1,2].set_ylabel('Regular News Rate (%)')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()



customer_evolution = (
    lazy_transactions
    .with_columns([
        pl.col('t_dat').str.to_date(format='%Y-%m-%d').alias('purchase_date')
    ])
    .sort(['customer_id', 'purchase_date'])
    .with_columns([
        pl.int_range(pl.len()).over('customer_id').alias('purchase_order'),
        pl.col('purchase_date').diff().over('customer_id').dt.total_days().alias('days_between')
    ])
    .group_by('customer_id')
    .agg([
        pl.col('purchase_date').min().alias('first_purchase'),
        pl.col('purchase_date').max().alias('last_purchase'),
        pl.col('purchase_date').count().alias('total_purchases'),
        pl.col('price').sum().alias('lifetime_value'),
        pl.col('price').mean().alias('avg_basket_value'),
        pl.col('article_id').n_unique().alias('unique_items'),
        pl.col('days_between').mean().alias('avg_interval'),
        
        # early vs recent behavior (first 3 vs last 3 purchases)
        pl.col('price').head(3).mean().alias('early_spend'),
        pl.col('price').tail(3).mean().alias('recent_spend'),
        pl.col('article_id').head(5).n_unique().alias('early_diversity'),
        pl.col('article_id').tail(5).n_unique().alias('recent_diversity')
    ])
    .with_columns([
        (pl.col('last_purchase') - pl.col('first_purchase')).dt.total_days().alias('lifespan_days'),
        (pl.col('recent_spend') / pl.col('early_spend')).alias('spend_growth'),
        (pl.col('recent_diversity') / pl.col('early_diversity')).alias('diversity_growth')
    ])
    .with_columns([
        pl.when(pl.col('lifespan_days') <= 30).then(pl.lit('New'))
        .when(pl.col('lifespan_days') <= 180).then(pl.lit('Growing'))
        .when(pl.col('lifespan_days') <= 365).then(pl.lit('Mature'))
        .otherwise(pl.lit('Loyal'))
        .alias('lifecycle_stage'),
        
        pl.when(pl.col('spend_growth') > 1.3).then(pl.lit('Increasing'))
        .when(pl.col('spend_growth') < 0.8).then(pl.lit('Decreasing'))
        .otherwise(pl.lit('Stable'))
        .alias('spend_pattern')
    ])
    .filter(pl.col('total_purchases') >= 3)
    .collect()
)

customer_evolution_demo = (
    customer_evolution
    .join(customers.select(['customer_id', 'age', 'club_member_status']), on='customer_id', how='left')
    .filter(pl.col('age').is_not_null())
)

print(f"Analyzing evolution of {len(customer_evolution_demo):,} customers")

lifecycle_analysis = (
    customer_evolution_demo
    .group_by('lifecycle_stage')
    .agg([
        pl.count().alias('count'),
        pl.col('total_purchases').mean().alias('avg_purchases'),
        pl.col('lifetime_value').mean().alias('avg_ltv'),
        pl.col('avg_basket_value').mean().alias('avg_basket'),
        pl.col('unique_items').mean().alias('avg_diversity'),
        pl.col('spend_growth').median().alias('median_spend_growth'),
        (pl.col('spend_pattern') == 'Increasing').mean().alias('increasing_rate')
    ])
)


print("LIFECYCLE STAGE:")

print("Stage".ljust(12) + "Count".ljust(10) + "Avg Purch".ljust(12) + "Avg LTV".ljust(10) + "Spend Growth".ljust(15) + "Increasing %")
print("-" * 75)
for row in lifecycle_analysis.iter_rows():
    stage, count, purchases, ltv, basket, diversity, growth, inc_rate = row
    print(f"{stage:<12} {count:<10,} {purchases:<12.1f} ${ltv:<9.0f} {growth:<15.2f} {inc_rate:.1%}")

del customer_evolution, lifecycle_analysis



# Monthly cohort
monthly_cohorts = (
    lazy_transactions
    .with_columns([
        pl.col('t_dat').str.to_date(format='%Y-%m-%d').alias('purchase_date')
    ])
    .group_by('customer_id')
    .agg([
        pl.col('purchase_date').min().alias('first_purchase'),
        pl.col('purchase_date').count().alias('purchases')
    ])
    .with_columns([
        pl.col('first_purchase').dt.strftime('%Y-%m').alias('cohort_month')
    ])
    .group_by('cohort_month')
    .agg([
        pl.count().alias('cohort_size'),
        pl.col('purchases').mean().alias('avg_purchases')
    ])
    .sort('cohort_month')
    .collect()
)

# Customer spending patterns
spend_evolution = customer_evolution_demo.select([
    'spend_pattern', 'lifecycle_stage', 'spend_growth', 'age'
]).to_pandas()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Customer Evolution Analysis Over Time', fontsize=16, fontweight='bold')

# lifecycle distribution
lifecycle_counts = customer_evolution_demo['lifecycle_stage'].value_counts()
lifecycle_data = lifecycle_counts.to_pandas()
axes[0,0].pie(lifecycle_data['count'], labels=lifecycle_data['lifecycle_stage'], autopct='%1.1f%%')
axes[0,0].set_title('Customer Lifecycle Distribution')

# Spending evolution by lifecycle
sns.boxplot(data=spend_evolution, x='lifecycle_stage', y='spend_growth', ax=axes[0,1])
axes[0,1].set_title('Spending Growth by Lifecycle Stage')
axes[0,1].axhline(y=1, color='red', linestyle='--', alpha=0.7)

# Age vs spending evolution
sns.scatterplot(data=spend_evolution.sample(5000), x='age', y='spend_growth', 
                hue='lifecycle_stage', alpha=0.6, ax=axes[0,2])
axes[0,2].set_title('Age vs Spending Evolution')
axes[0,2].axhline(y=1, color='red', linestyle='--', alpha=0.7)

# Spend patterns by lifecycle
spend_pattern_cross = pd.crosstab(spend_evolution['lifecycle_stage'], 
                                  spend_evolution['spend_pattern'], normalize='index')
spend_pattern_cross.plot(kind='bar', stacked=True, ax=axes[1,0])
axes[1,0].set_title('Spending Patterns by Lifecycle Stage')
axes[1,0].tick_params(axis='x', rotation=45)

# Cohort size over time
cohort_data = monthly_cohorts.to_pandas()
axes[1,1].plot(cohort_data['cohort_month'], cohort_data['cohort_size'], marker='o')
axes[1,1].set_title('Customer Acquisition by Cohort')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].set_ylabel('New Customers')

# Customer value evolution
lifecycle_summary = customer_evolution_demo.group_by('lifecycle_stage').agg([
    pl.col('lifetime_value').mean().alias('avg_ltv')
]).to_pandas()
axes[1,2].bar(lifecycle_summary['lifecycle_stage'], lifecycle_summary['avg_ltv'])
axes[1,2].set_title('Average LTV by Lifecycle Stage')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


cohort_data = monthly_cohorts.to_pandas()

lifecycle_summary = (
    customer_evolution_demo.group_by('lifecycle_stage')
    .agg([
        pl.col('lifetime_value').mean().alias('avg_ltv')
    ])
).to_pandas()

evolution_insights = customer_evolution_demo.select([
    pl.col('spend_growth').mean().alias('avg_spend_evolution'),
    pl.col('diversity_growth').mean().alias('avg_diversity_evolution'),
    (pl.col('spend_pattern') == 'Increasing').mean().alias('customers_increasing_spend'),
    (pl.col('spend_pattern') == 'Decreasing').mean().alias('customers_decreasing_spend'),
    pl.col('lifespan_days').mean().alias('avg_customer_lifespan'),
    pl.col('avg_interval').mean().alias('avg_purchase_interval')
])

print(f"\nKEY METRICS:")
insights = evolution_insights.to_pandas().iloc[0]
print(f"Average spending evolution: {insights['avg_spend_evolution']:.2f}x")
print(f"Customers increasing spend: {insights['customers_increasing_spend']:.1%}")
print(f"Customers decreasing spend: {insights['customers_decreasing_spend']:.1%}")
print(f"Average customer lifespan: {insights['avg_customer_lifespan']:.0f} days")
print(f"Average purchase interval: {insights['avg_purchase_interval']:.0f} days")

del insights, cohort_data, lifecycle_summary, evolution_insights
gc.collect()


print("Aggregating transaction data to get per-customer statistics...")
customer_agg_query = (
    lazy_transactions
    .group_by("customer_id")
    .agg([
        pl.len().alias("n_transactions"),
        pl.col("article_id").n_unique().alias("n_unique_articles"),
        pl.col("price").sum().alias("total_spend"),
        pl.col("t_dat").n_unique().alias("n_purchase_days")
    ])
)
customer_agg_stats = customer_agg_query.collect()

customer_loyalty_data = (
    customers
    .join(customer_agg_stats, on="customer_id", how="left")
    .with_columns(
        pl.col("club_member_status").fill_null("UNKNOWN").alias("club_member_status")
    )
    .drop_nulls(subset=['n_transactions']) # Drop only customers with no transactions
)


customer_loyalty_pd = customer_loyalty_data.to_pandas()

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Impact of Club Membership on Purchasing Behavior', fontsize=18, fontweight='bold')

# Average Total Items Purchased
sns.boxenplot(data=customer_loyalty_pd, x='club_member_status', y='n_transactions', ax=axes[0], palette='viridis')
axes[0].set_title('Total Items Purchased per Customer', fontsize=14)
axes[0].set_xlabel('Club Member Status')
axes[0].set_ylabel('Number of Transactions (Log Scale)')
axes[0].set_yscale('log')

# Average Total Spend
sns.boxenplot(data=customer_loyalty_pd, x='club_member_status', y='total_spend', ax=axes[1], palette='plasma')
axes[1].set_title('Total Spend per Customer', fontsize=14)
axes[1].set_xlabel('Club Member Status')
axes[1].set_ylabel('Total Spend (Log Scale)')
axes[1].set_yscale('log')

# Average Number of Active Shopping Days
sns.boxenplot(data=customer_loyalty_pd, x='club_member_status', y='n_purchase_days', ax=axes[2], palette='magma')
axes[2].set_title('Distinct Shopping Days per Customer', fontsize=14)
axes[2].set_xlabel('Club Member Status')
axes[2].set_ylabel('Number of Days (Log Scale)')
axes[2].set_yscale('log')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Clean up memory ---
del customer_agg_stats, customer_loyalty_data, customer_loyalty_pd
gc.collect()


print("CUSTOMER TRANSACTION METRICS")

customer_metrics = (
    lazy_transactions
    .with_columns([
        pl.col('t_dat').str.strptime(pl.Date, format='%Y-%m-%d'),
        pl.col('price').cast(pl.Float64)
    ])
    .group_by('customer_id')
    .agg([
        # transaction frequency metrics
        pl.count().alias('total_transactions'),
        pl.col('price').sum().alias('total_spent'),
        pl.col('price').mean().alias('avg_transaction_value'),
        pl.col('price').std().alias('std_transaction_value'),
        pl.col('article_id').n_unique().alias('unique_items_purchased'),
        
        # temporal metrics
        pl.col('t_dat').min().alias('first_purchase_date'),
        pl.col('t_dat').max().alias('last_purchase_date'),
        pl.col('t_dat').n_unique().alias('purchase_days'),
        
        # behavioral metrics
        pl.col('sales_channel_id').mode().first().alias('preferred_channel'),
        pl.col('article_id').count().alias('total_items')
    ])
    .with_columns([
        (pl.col('last_purchase_date') - pl.col('first_purchase_date')).dt.total_days().alias('customer_lifetime_days'),
        (pl.col('total_spent') / pl.col('total_transactions')).alias('avg_basket_size'),
        (pl.col('total_transactions') / pl.col('purchase_days')).alias('purchase_frequency'),
        (pl.col('unique_items_purchased') / pl.col('total_transactions')).alias('item_variety_ratio')
    ])
    .collect()
)

print(f"Calculated metrics for {len(customer_metrics):,} customers")

customer_metrics = customer_metrics.with_columns([
    pl.col('std_transaction_value').fill_null(0),
    pl.col('customer_lifetime_days').fill_null(0),
    pl.col('purchase_frequency').fill_null(pl.col('total_transactions')),
])

print("\nCUSTOMER VALUE TIERS:")

spending_50 = customer_metrics.select(pl.col('total_spent')).to_series().quantile(0.5)
spending_80 = customer_metrics.select(pl.col('total_spent')).to_series().quantile(0.8)
spending_95 = customer_metrics.select(pl.col('total_spent')).to_series().quantile(0.95)

frequency_50 = customer_metrics.select(pl.col('total_transactions')).to_series().quantile(0.5)
frequency_80 = customer_metrics.select(pl.col('total_transactions')).to_series().quantile(0.8)
frequency_95 = customer_metrics.select(pl.col('total_transactions')).to_series().quantile(0.95)

recency_date = customer_metrics.select(pl.col('last_purchase_date')).to_series().max()

print(f"Spending thresholds: 50%=${spending_50:.2f}, 80%=${spending_80:.2f}, 95%=${spending_95:.2f}")
print(f"Frequency thresholds: 50%={frequency_50:.0f}, 80%={frequency_80:.0f}, 95%={frequency_95:.0f} transactions")


customer_metrics.head()


customer_full = (
    customer_metrics
    .join(customers, on='customer_id', how='left')
    .with_columns([
        pl.col('age').fill_null(pl.col('age').median()),
        pl.col('club_member_status').fill_null('UNKNOWN'),
        pl.col('fashion_news_frequency').fill_null('NONE'),
                
        # recency score (days since last purchase)
        (pl.date(2020, 9, 22) - pl.col('last_purchase_date')).dt.total_days().alias('days_since_last_purchase'),
        
        # engagement scores
        (pl.col('total_transactions') * pl.col('avg_transaction_value')).alias('engagement_score'),
        (pl.col('unique_items_purchased') / pl.col('total_transactions')).alias('exploration_ratio'),
        
        # behavior indicators
        pl.when(pl.col('preferred_channel') == 1).then(pl.lit(1)).otherwise(pl.lit(0)).alias('online_shopper'),
        pl.when(pl.col('preferred_channel') == 2).then(pl.lit(1)).otherwise(pl.lit(0)).alias('store_shopper'),
    ])
)

print(f"Customer dataset created with {len(customer_full):,} customers and {customer_full.width} features")


key_metrics = ['total_spent', 'total_transactions', 'avg_transaction_value', 'customer_lifetime_days', 'unique_items_purchased']

df_analysis = customer_full.select(key_metrics+['days_since_last_purchase']).to_pandas()

desc_stats = df_analysis.describe()

desc_stats.round(2)


for metric in key_metrics:
    metric_skewness = skew(df_analysis[metric])
    print(f"{metric}: {metric_skewness:.3f} {'(highly skewed)' if abs(metric_skewness) > 2 else '(moderately skewed)' if abs(metric_skewness) > 0.5 else '(normal)'}")


correlation_matrix = df_analysis[key_metrics].corr()
print("Correlation Matrix:")
correlation_matrix.round(3)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Customer Behavior Metrics Distribution Analysis', fontsize=16, fontweight='bold')

for i, metric in enumerate(key_metrics):
    row, col = i // 3, i % 3
    
    data = df_analysis[metric]
    if skew(data) > 2:
        data = np.log1p(data)  # log(1+x) to handle zeros
        title = f'Log({metric})'
    else:
        title = metric
    
    axes[row, col].hist(data, bins=50, alpha=0.7, edgecolor='black')
    axes[row, col].set_title(title, fontweight='bold')
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].grid(True, alpha=0.3)

# Plot correlation heatmap in the last subplot
axes[1, 2].clear()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
axes[1, 2].set_title('Correlation Matrix', fontweight='bold')

plt.tight_layout()
plt.show()



correlation_matrix.round(3)


current_categories = articles['product_group_name'].value_counts().head(10)
print("Top product_group_name categories:")
print(current_categories.to_pandas())

detailed_categories = articles['product_type_name'].value_counts().head(20)
print("\nTop product_type_name categories:")  
print(detailed_categories.to_pandas())

print("\nProduct types within 'Garment Upper body':")
upper_body_types = (
    articles
    .filter(pl.col('product_group_name') == 'Garment Upper body')
    ['product_type_name'].value_counts().head(10)
)
print(upper_body_types.to_pandas())


hierarchy_cols = [col for col in articles.columns if any(keyword in col.lower() 
                  for keyword in ['department', 'category', 'group', 'type', 'section'])]

hierarchy_analysis = {}
for col in hierarchy_cols:
    unique_count = articles.select(pl.col(col).n_unique()).item()
    hierarchy_analysis[col] = unique_count
    print(f"{col}: {unique_count} unique values")

main_hierarchy = articles.select([
    'product_type_name',
    'product_group_name', 
    'graphical_appearance_name',
    'colour_group_name',
    'department_name',
    'index_name',
    'index_group_name',
    'section_name',
    'garment_group_name'
]).to_pandas()

print("TOP PRODUCT CATEGORIES:")
for col in ['department_name', 'product_group_name', 'product_type_name']:
    print(f"\n{col.upper()}:")
    if col in main_hierarchy.columns:
        top_categories = main_hierarchy[col].value_counts().head(10)
        for category, count in top_categories.items():
            print(f"  {category}: {count:,} products ({count/len(main_hierarchy)*100:.1f}%)")


hierarchy_levels = ['index_group_name', 'garment_group_name', 'product_group_name', 'product_type_name']

hierarchy_data = (
    articles
    .group_by(['index_group_name', 'garment_group_name', 'product_group_name', 'product_type_name'])
    .agg(pl.count().alias('product_count'))
    .sort('product_count', descending=True)
    .to_pandas()
)

print(f"Total hierarchy combinations: {len(hierarchy_data)}")

fig_treemap_corrected = px.treemap(
    hierarchy_data.head(100),
    path=['index_group_name', 'garment_group_name', 'product_group_name', 'product_type_name'],
    values='product_count',
    title='H&M Product Hierarchy - Corrected Treemap (Index Group → Garment Group → Product Group → Product Type)',
    color='product_count',
    color_continuous_scale='Viridis',
    height=700
)

fig_treemap_corrected.update_traces(textinfo="label+value")
fig_treemap_corrected.show()


articles_pd = articles.to_pandas()

# Top 15 Product Groups
plt.figure(figsize=(12, 8))
sns.countplot(y='product_group_name', data=articles_pd, order=articles_pd['product_group_name'].value_counts().head(15).index, palette='mako')
plt.title('Top 15 Most Common Product Groups in Catalog', fontsize=16)
plt.xlabel('Number of Articles')
plt.ylabel('Product Group')
plt.show()

# --- Clean up memory ---
del articles_pd
gc.collect()


articles.head()


## COLOR & MATERIAL // ARTICLES

color_columns = [col for col in articles.columns if 'colour' in col.lower()]

print('color related columns in the articles dataset:\n')
print(color_columns)

for col in color_columns: 
    unique_colors = articles.select(pl.col(col).n_unique()).item()
    print(f"Column: {col} // Unique Color Count: {unique_colors}")

# color distribution

color_distribution = (
    articles
    .group_by('colour_group_name')
    .agg([
        pl.count().alias('product_count'),
        pl.col('product_group_name').n_unique().alias('product_groups_count'),
        pl.col('department_name').n_unique().alias('departments_count')
    ])
    .sort('product_count', descending=True)
    .to_pandas()
)

print(f"\nTotal Unique Colors: {len(color_distribution)}")
print("\nTOP 15 COLORS BY PRODUCT COUNT:")
top_colors = color_distribution.head(15)
for _, row in top_colors.iterrows():
    color = row['colour_group_name']
    count = row['product_count']
    pct = count / len(articles) * 100
    groups = row['product_groups_count']
    depts = row['departments_count']
    print(f"  {color}: {count:,} products ({pct:.1f}%) | {groups} groups | {depts} departments")



articles.head()


feature_list = ['product_group_name','garment_group_name', 'index_group_name', 'index_name', 'section_name',
                'colour_group_name', 'perceived_colour_value_name', 'graphical_apperance_name'
               ]

for feature in feature_list:
    if feature in articles.columns:
        unique_count = articles.select(pl.col(feature).n_unique()).item()
        print(f"{feature} : {unique_count} unique values")


# color-category combinations

color_category_matrix = (
    articles
    .group_by(['index_name', 'perceived_colour_value_name'])
    .agg(pl.count().alias('combination_count'))
    .filter(pl.col('combination_count')>10)
    .sort('combination_count', descending=True)
    .head(25)
    .to_pandas()
)

color_category_matrix


attribute_idf = {}

product_group = articles.select('section_name').to_pandas()['section_name'].value_counts()

for group, count in product_group.head(10).items():
    frequency_count = count/len(articles)
    idf_score = np.log(len(articles)/count) # measure how rare a feature is across a collection.
    attribute_idf[group] = idf_score

attribute_idf



# perceived_colour_value_name for material hints
if 'perceived_colour_value_name' in articles.columns:
    perceived_colors = articles.select('perceived_colour_value_name').to_pandas()['perceived_colour_value_name'].value_counts()
    print("TOP PERCEIVED COLOR VALUES (may contain material info):")
    print(perceived_colors.head(20))

if 'detail_desc' in articles.columns:
    print(f"\nDETAIL DESCRIPTIONS:")
    detail_desc_sample = articles.select('detail_desc').filter(pl.col('detail_desc').is_not_null()).head(10).to_pandas()
    print("Sample detail descriptions:")
    for desc in detail_desc_sample['detail_desc']:
        print(f"  - {desc}")


weekly_sales_query = (
    lazy_transactions
    .with_columns(
        pl.col("t_dat").str.to_date(format="%Y-%m-%d") 
    )
    .with_columns(
        pl.col("t_dat").dt.truncate("1w").alias("week") 
    )
    .group_by("week")
    .agg(
        pl.count().alias("n_transactions")
    )
    .sort("week")
)

print("Calculating weekly sales for the entire dataset...")
weekly_sales = weekly_sales_query.collect()

weekly_sales_pd = weekly_sales.to_pandas()

fig = px.line(weekly_sales_pd, x='week', y='n_transactions', title='Weekly Sales Transactions (Full Dataset)')
fig.show()

del weekly_sales_pd, weekly_sales
gc.collect()


article_first_sale_query = (
    lazy_transactions
    .with_columns(pl.col("t_dat").str.to_date(format="%Y-%m-%d"))
    .group_by("article_id")
    .agg(
        pl.col("t_dat").min().alias("launch_date")
    )
)
article_first_sale = article_first_sale_query.collect()

monthly_sales_query = (
    lazy_transactions
    .with_columns(pl.col("t_dat").str.to_date(format="%Y-%m-%d"))
    .join(articles.lazy(), on='article_id', how='left') 
    .with_columns(
        pl.col("t_dat").dt.month().alias("month") 
    )
    .group_by(["month", "product_group_name"])
    .agg([
        pl.len().alias("n_transactions"),
        pl.col("article_id").n_unique().alias("n_unique_articles_in_month") 
    ])
)
monthly_sales = monthly_sales_query.collect()
print("Monthly sales calculation complete.")


# We are now normalizing by the number of unique articles sold in that period.
# This gives us a "sales per active product", which handles product maturity.
# A new product can only contribute to sales in the months it's active.
monthly_sales = monthly_sales.with_columns(
    (pl.col("n_transactions") / pl.col("n_unique_articles_in_month")).alias("sales_per_active_article")
)

monthly_sales_pd = monthly_sales.to_pandas()

top_12_groups = articles['product_group_name'].value_counts().get_column('product_group_name').to_list()
monthly_sales_top_pd = monthly_sales_pd[monthly_sales_pd['product_group_name'].isin(top_12_groups)]

pivot_table = monthly_sales_top_pd.pivot_table(
    index='product_group_name', 
    columns='month', 
    values='sales_per_active_article',
    fill_value=0
)

pivot_table_normalized = pivot_table.div(pivot_table.sum(axis=1), axis=0)

plt.figure(figsize=(16, 10))
sns.heatmap(pivot_table_normalized, cmap='YlGnBu', annot=False)
plt.title('Seasonal Demand: Normalized Sales Intensity per Product Group', fontsize=18)
plt.xlabel('Month of the Year')
plt.ylabel('Product Group')
plt.show()

# --- Clean up memory ---
del article_first_sale, monthly_sales, monthly_sales_pd, monthly_sales_top_pd, pivot_table, pivot_table_normalized
gc.collect()


# Long Tail Analysis (Maturity-aware) ---

print("Calculating article launch dates and total sales from full dataset...")
article_sales_and_launch_query = (
    lazy_transactions
    .with_columns(pl.col("t_dat").str.to_date(format="%Y-%m-%d"))
    .group_by("article_id")
    .agg([
        pl.len().alias("n_sales"),
        pl.col("t_dat").min().alias("launch_date")
    ])
)

article_stats = article_sales_and_launch_query.collect()

dataset_end_date = article_stats.select(pl.col("launch_date").max()).item()
article_stats = article_stats.with_columns(
    ((pl.lit(dataset_end_date) - pl.col("launch_date")).dt.total_days() / 7 + 1)
    .cast(pl.Int32)
    .alias("article_age_weeks")
)


article_stats = article_stats.with_columns(
    (pl.col("n_sales") / pl.col("article_age_weeks")).alias("sales_per_week")
)

article_stats_sorted = article_stats.sort('n_sales', descending=True)

sales_numpy = article_stats_sorted['n_sales'].to_numpy()

cumulative_sales_perc_numpy = 100 * sales_numpy.cumsum() / sales_numpy.sum()
cumulative_articles_perc_numpy = 100 * (np.arange(1, len(sales_numpy) + 1)) / len(sales_numpy)

article_stats_final = article_stats_sorted.with_columns([
    pl.Series("cumulative_sales_perc", cumulative_sales_perc_numpy),
    pl.Series("cumulative_articles_perc", cumulative_articles_perc_numpy)
])


article_sales_pd = article_stats_final.to_pandas()

fig = px.line(
    article_sales_pd, 
    x='cumulative_articles_perc', 
    y='cumulative_sales_perc',
    title='The Long Tail: Sales Concentration (Full Dataset)',
    labels={'cumulative_articles_perc': '% of Top Articles (Ranked by Raw Sales)', 'cumulative_sales_perc': '% of Total Sales'}
)
fig.add_hline(y=80, line_dash="dash", annotation_text="80% of Sales", annotation_position="bottom right")
fig.add_vline(x=20, line_dash="dash", annotation_text="20% of Articles")
fig.show()

# --- Final clean up ---
del article_stats, article_stats_sorted, article_stats_final, article_sales_pd
gc.collect()


# Customer Purchase Frequency
customer_purchases_query = (
    lazy_transactions
    .group_by('customer_id')
    .agg(
        pl.len().alias('n_purchases')
    )
)

print("\nCalculating purchase frequency for all customers...")
customer_purchases = customer_purchases_query.collect()
print("Calculation complete.")

# Convert to pandas for plotting
customer_purchases_pd = customer_purchases.to_pandas()

plt.figure(figsize=(12, 6))
sns.histplot(customer_purchases_pd['n_purchases'], bins=100, log_scale=(False, True))
plt.title('Distribution of Number of Purchases per Customer (Full Dataset, Log Scale)')
plt.xlabel('Total Number of Articles Purchased')
plt.ylabel('Number of Customers (Log Scale)')
plt.show()

print(customer_purchases_pd['n_purchases'].describe())

# --- Clean up memory ---
del customer_purchases_pd, customer_purchases
gc.collect()


print("Calculating article first appearance dates and total sales...")
article_stats_query = (
    lazy_transactions
    .with_columns(pl.col("t_dat").str.to_date(format="%Y-%m-%d"))
    .group_by("article_id")
    .agg([
        pl.len().alias("n_sales"),
        pl.col("t_dat").min().alias("first_seen_date")
    ])
)
article_stats = article_stats_query.collect()
print("Calculation complete.")

# how long the product has been available in the observed transaction window.
dataset_end_date = article_stats.select(pl.col("first_seen_date").max()).item()
article_stats = article_stats.with_columns(
    ((pl.lit(dataset_end_date) - pl.col("first_seen_date")).dt.total_days() / 7 + 1)
    .cast(pl.Int32)
    .alias("article_age_weeks_in_data")
)
print("Article age calculation complete.")


# sales velocity (sales per week since first seen)
article_stats = article_stats.with_columns(
    (pl.col("n_sales") / pl.col("article_age_weeks_in_data")).alias("sales_per_week")
)


# sales velocity based on article age
age_labels = ["New (0-4 wks)", "1-3 Months", "3-6 Months", "6-12 Months", "1-2 Years", ">2 Years"]
article_stats = article_stats.with_columns(
    pl.when(pl.col("article_age_weeks_in_data") <= 4).then(pl.lit(age_labels[0]))
    .when(pl.col("article_age_weeks_in_data") <= 13).then(pl.lit(age_labels[1]))
    .when(pl.col("article_age_weeks_in_data") <= 26).then(pl.lit(age_labels[2]))
    .when(pl.col("article_age_weeks_in_data") <= 52).then(pl.lit(age_labels[3]))
    .when(pl.col("article_age_weeks_in_data") <= 104).then(pl.lit(age_labels[4]))
    .otherwise(pl.lit(age_labels[5]))
    .alias("age_bin")
)

age_bin_velocity = (
    article_stats
    .group_by("age_bin")
    .agg(pl.col("sales_per_week").mean().alias("avg_sales_velocity"))
)

age_bin_velocity_pd = age_bin_velocity.to_pandas()
age_bin_velocity_pd['age_bin'] = pd.Categorical(age_bin_velocity_pd['age_bin'], categories=age_labels, ordered=True)
age_bin_velocity_pd = age_bin_velocity_pd.sort_values('age_bin')

plt.figure(figsize=(12, 7))
sns.barplot(data=age_bin_velocity_pd, x="age_bin", y="avg_sales_velocity", palette="magma")
plt.title('Average Sales Velocity by Article Age (Time Since First Seen in Data)', fontsize=16)
plt.xlabel('Article Age Since First Appearance')
plt.ylabel('Average Sales per Week')
plt.show()

# --- Clean up memory ---
del article_stats, age_bin_velocity, age_bin_velocity_pd
gc.collect()


print("HALF-LIFE ANALYSIS")

weekly_article_sales = (
    lazy_transactions
    .with_columns(pl.col("t_dat").str.to_date(format="%Y-%m-%d"))
    .with_columns(pl.col("t_dat").dt.truncate("1w").alias("week"))
    .group_by(["article_id", "week"])
    .agg(pl.len().alias("n_sales"))
    .sort(["article_id", "week"])
    .collect()
)
print(f"Weekly sales calculated for {weekly_article_sales['article_id'].n_unique()} unique articles.")


article_stats = (
    weekly_article_sales
    .group_by('article_id')
    .agg(
        pl.sum('n_sales').alias('total_sales'),
        pl.max('n_sales').alias('peak_sales'),
        pl.count('week').alias('weeks_active'),
        pl.min('week').alias('first_week'),
        pl.max('week').alias('last_week'),
    )
    .filter(
        (pl.col('weeks_active') >= 3) &
        (pl.col('total_sales') >= 10) &
        (((pl.col('last_week') - pl.col('first_week')).dt.total_days() / 7) >= 2)
    )
)

peak_sales_info = (
    weekly_article_sales
    .join(article_stats, on='article_id', how='inner')
    # where sales match the article's peak sales
    .filter(pl.col('n_sales') == pl.col('peak_sales'))
    .group_by('article_id')
    # if multiple weeks have peak sales, take the earliest one
    .agg(pl.min('week').alias('peak_week'))
    # join the peak_week back with the rest of the stats
    .join(article_stats, on='article_id', how='inner')
)

print(f"Valid articles and peak week identified for {len(peak_sales_info)} articles.")

half_life_calculation_base = (
    weekly_article_sales
    .join(peak_sales_info.select(['article_id', 'peak_week', 'total_sales']), on='article_id', how='inner')
)

# Find the week where 50% of post-peak sales are consumed
half_life_data = (
    half_life_calculation_base
    .sort(['article_id', 'week'])
    .to_pandas()
    .assign(
        cumulative_sales=lambda df: df.groupby('article_id')['n_sales'].cumsum()
    )
    .pipe(pl.from_pandas) 
    .with_columns([
        # cumulative sales AT peak week
        pl.when(pl.col('week') == pl.col('peak_week'))
        .then(pl.col('cumulative_sales'))
        .otherwise(None)
        .max().over('article_id').alias('sales_up_to_peak')
    ])
    .with_columns([
        (pl.col('total_sales') - pl.col('sales_up_to_peak')).alias('total_remaining_from_peak')
    ])
    .filter(pl.col('week') > pl.col('peak_week'))
    .with_columns([
        (pl.col('total_sales') - pl.col('cumulative_sales')).alias('remaining_sales')
    ])
    .with_columns([
        pl.when(pl.col('total_remaining_from_peak') > 0)
        .then(1 - (pl.col('remaining_sales') / pl.col('total_remaining_from_peak')))
        .otherwise(1.0)
        .alias('remaining_fraction_consumed')
    ])
    .filter(pl.col('remaining_fraction_consumed') >= 0.5)
    .group_by('article_id')
    .agg([
        pl.col('week').min().alias('half_life_week'),
        pl.col('remaining_fraction_consumed').min().alias('actual_fraction')
    ])
)


print(f"Half-life calculated for {len(half_life_data)} articles.")


final_stats = (
    peak_sales_info
    .join(half_life_data, on='article_id', how='inner')
    .with_columns(
        (((pl.col('half_life_week') - pl.col('peak_week')).dt.total_days() / 7)
         .alias('half_life_weeks')),
        (pl.col('peak_sales') / pl.col('total_sales')).alias('peak_intensity')
    )
    .filter(
        (pl.col('half_life_weeks') >= 0) &
        (pl.col('half_life_weeks') <= 52) # Cap at 1 year
    )
)

print(f"Final dataset created for {len(final_stats)} articles.")

final_stats


final_data_with_groups = (
    final_stats
    .join(
        articles.select(['article_id', 'product_group_name']),
        on='article_id',
        how='left'
    )
    .filter(pl.col('product_group_name').is_not_null())
)

category_counts = final_data_with_groups['product_group_name'].value_counts()
significant_categories = (
    category_counts
    .filter(pl.col('count') >= 50)
    .get_column('product_group_name')
    .to_list()
)

plot_data = final_data_with_groups.filter(
    pl.col('product_group_name').is_in(significant_categories)
)

print(f"Dataset for visualization: {len(plot_data)} articles across {len(significant_categories)} categories.")

plot_order = (
    plot_data
    .group_by('product_group_name')
    .agg(pl.col('half_life_weeks').median())
    .sort('half_life_weeks', descending=True)
    .get_column('product_group_name')
    .to_list()
)

plot_data_pd = plot_data.to_pandas()

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
fig.suptitle('Product Popularity Half-Life Analysis', fontsize=16, fontweight='bold')

sns.boxenplot(
    data=plot_data_pd,
    x='half_life_weeks',
    y='product_group_name',
    order=plot_order,
    palette='viridis',
    ax=ax
)
ax.set_title('Distribution of Popularity Half-Life by Product Group', fontsize=12)
ax.set_xlabel('Half-Life (Weeks after Peak Sales)')
ax.set_ylabel('Product Group')
ax.set_xlim(0, 30)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


print("POPULARITY LIFECYCLE:")

lifecycle_summary = final_stats.select([
    pl.col('half_life_weeks').mean().alias('avg_popularity_duration'),
    pl.col('half_life_weeks').median().alias('median_popularity_duration'),
    pl.col('peak_intensity').mean().alias('avg_peak_concentration'),
    pl.col('weeks_active').mean().alias('avg_total_lifespan'),
    (pl.col('half_life_weeks') / pl.col('weeks_active')).mean().alias('avg_decay_ratio')
])

lifecycle_summary.to_pandas().round(2)


# Popularity patterns
persistence_analysis = final_stats.select([
    (pl.col('half_life_weeks') <= 2).sum().alias('very_short_≤2wks'),
    ((pl.col('half_life_weeks') > 2) & (pl.col('half_life_weeks') <= 6)).sum().alias('short_3-6wks'),
    ((pl.col('half_life_weeks') > 6) & (pl.col('half_life_weeks') <= 12)).sum().alias('medium_7-12wks'),
    ((pl.col('half_life_weeks') > 12) & (pl.col('half_life_weeks') <= 20)).sum().alias('long_13-20wks'),
    (pl.col('half_life_weeks') > 20).sum().alias('very_long_>20wks'),
    pl.count().alias('total_products')
]).with_columns([
    (pl.col('very_short_≤2wks') / pl.col('total_products') * 100).alias('very_short_pct'),
    (pl.col('short_3-6wks') / pl.col('total_products') * 100).alias('short_pct'),
    (pl.col('medium_7-12wks') / pl.col('total_products') * 100).alias('medium_pct'),
    (pl.col('long_13-20wks') / pl.col('total_products') * 100).alias('long_pct'),
    (pl.col('very_long_>20wks') / pl.col('total_products') * 100).alias('very_long_pct')
])

print("\nPOPULARITY PERSISTENCE PATTERNS:")
persistence_analysis.to_pandas().round(1)


print("CATEGORY-SPECIFIC LIFECYCLE:")

category_lifecycle = (
    final_stats
    .join(articles.select(['article_id', 'product_group_name']), on='article_id', how='left')
    .filter(pl.col('product_group_name').is_not_null())
    .group_by('product_group_name')
    .agg([
        pl.count().alias('product_count'),
        pl.col('half_life_weeks').mean().alias('avg_half_life'),
        pl.col('half_life_weeks').median().alias('median_half_life'),
        pl.col('half_life_weeks').std().alias('std_half_life'),
        pl.col('peak_intensity').mean().alias('avg_peak_intensity'),
        pl.col('weeks_active').mean().alias('avg_total_lifespan'),
        pl.col('total_sales').mean().alias('avg_volume'),
        
        (pl.col('half_life_weeks') <= 4).mean().alias('fast_fashion_pct'),
        (pl.col('half_life_weeks').is_between(5, 12)).mean().alias('seasonal_pct'),
        (pl.col('half_life_weeks') > 12).mean().alias('staple_pct'),
        
        (pl.col('half_life_weeks') / pl.col('weeks_active')).mean().alias('decay_efficiency'),
        pl.col('total_sales').sum().alias('total_category_volume')
    ])
    .filter(pl.col('product_count') >= 100)  # significant categories
    .sort('median_half_life', descending=True)
)

print("CATEGORY LIFECYCLE COMPARISON:")
category_lifecycle.to_pandas().round(2)



print("PRODUCT LIFECYCLE ARCHETYPES:")

archetypes = final_stats.with_columns([
    pl.when(
        (pl.col('half_life_weeks') <= 3) & (pl.col('peak_intensity') > 0.25)
    ).then(pl.lit('Viral_Trends'))
    .when(
        (pl.col('half_life_weeks') <= 6) & (pl.col('peak_intensity') <= 0.25)
    ).then(pl.lit('Fast_Fashion'))
    .when(
        (pl.col('half_life_weeks').is_between(7, 15)) & (pl.col('peak_intensity') <= 0.20)
    ).then(pl.lit('Seasonal_Core'))
    .when(
        (pl.col('half_life_weeks') > 15) & (pl.col('peak_intensity') <= 0.15)
    ).then(pl.lit('Evergreen_Basics'))
    .otherwise(pl.lit('Mixed_Pattern'))
    .alias('lifecycle_archetype')
])

archetype_summary = archetypes.group_by('lifecycle_archetype').agg([
    pl.count().alias('count'),
    pl.col('half_life_weeks').mean().alias('avg_half_life'),
    pl.col('peak_intensity').mean().alias('avg_peak_intensity'),
    pl.col('total_sales').mean().alias('avg_volume'),
    pl.col('weeks_active').mean().alias('avg_lifespan')
]).with_columns([
    (pl.col('count') / archetypes.height * 100).alias('percentage')
])

print("LIFECYCLE ARCHETYPES:")
archetype_summary.to_pandas().round(2)


half_life_features = final_stats.select([
    'article_id',
    'half_life_weeks',
    'peak_intensity', 
    'total_sales',
    'peak_sales',
    'weeks_active',
    'first_week',
    'last_week',
    'actual_fraction',
    
    (pl.col('half_life_weeks') / pl.col('weeks_active')).alias('lifecycle_efficiency_ratio'),
    (pl.col('total_sales') / pl.col('weeks_active')).alias('sales_velocity'),
    (pl.col('peak_sales') / pl.col('total_sales')).alias('peak_sales_ratio'),
        
    # TREND VELOCITY LABELING
    pl.when(pl.col('half_life_weeks') <= 2)
    .then(pl.lit('ultra_fast'))
    .when(pl.col('half_life_weeks') <= 4)  
    .then(pl.lit('fast_fashion'))         # core H&M model
    .when(pl.col('half_life_weeks') <= 8)
    .then(pl.lit('seasonal_trend'))       # season-driven items
    .when(pl.col('half_life_weeks') <= 16)
    .then(pl.lit('wardrobe_staple'))      # mid-season basics
    .otherwise(pl.lit('timeless_basic'))  # year-round essentials
    .alias('trend_velocity_category')

])



half_life_features.head()


# These are articles that were NOT in our half-life analysis.
mature_article_ids = final_stats.get_column("article_id").to_list()
all_article_ids = articles.get_column("article_id").to_list()
new_article_ids = set(all_article_ids) - set(mature_article_ids)

print(f"\nNumber of 'new' or 'low-interaction' articles: {len(new_article_ids)}")

