# Cell 1: Import Libraries and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("ğŸ“Š Customer Satisfaction Analysis - Phase 2")
print("Libraries imported successfully!")


# Cell 2: Data Loading
print("=== DATA LOADING ===")

# Load dataset

train_file_path = "/kaggle/input/train-dataset/train_dataset.csv"

df = pd.read_csv(train_file_path)

# Basic info
print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {len(df):,}")

# Clean data
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Check satisfaction distribution
satisfaction_dist = df['satisfaction'].value_counts()
print(f"\nSatisfaction Distribution:")
for category, count in satisfaction_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  {category}: {count:,} ({percentage:.1f}%)")

# Display first few rows
df.head()


# Cell 3: Research Questions Definition
print("=== RESEARCH QUESTIONS ===")

research_questions = [
    "RQ1: How does customer demographics (age, gender, loyalty) affect satisfaction?",
    "RQ2: What is the impact of travel class and purpose on customer satisfaction?",
    "RQ3: Which service quality factors are most strongly correlated with satisfaction?",
    "RQ4: How do flight delays affect customer satisfaction across different segments?",
    "RQ5: What patterns exist between flight distance and service satisfaction?",
    "RQ6: How do loyal vs disloyal customers differ in their satisfaction patterns?"
]

for i, question in enumerate(research_questions, 1):
    print(f"{i}. {question}")



# Cell 4: ANALYSIS 1 - Demographic Analysis
print("=== ANALYSIS 1: DEMOGRAPHIC FACTORS ===\n")

# Create subplots for demographic analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Demographic Analysis of Customer Satisfaction', fontsize=16, fontweight='bold')

# 1. Gender Analysis
if 'Gender' in df.columns:
    gender_data = pd.crosstab(df['Gender'], df['satisfaction'], normalize='index') * 100
    gender_data.plot(kind='bar', ax=axes[0,0], color=['#ff7f7f', '#7fbf7f'])
    axes[0,0].set_title('Satisfaction by Gender (%)', fontweight='bold')
    axes[0,0].set_xlabel('Gender')
    axes[0,0].set_ylabel('Percentage')
    axes[0,0].legend(title='Satisfaction')
    axes[0,0].tick_params(axis='x', rotation=45)

# 2. Age Group Analysis
if 'Age' in df.columns:
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 65, 100],
                            labels=['18-25', '26-35', '36-50', '51-65', '65+'])
    age_satisfaction = df.groupby('Age_Group')['satisfaction'].apply(lambda x: (x == 'satisfied').mean() * 100)

    age_satisfaction.plot(kind='bar', ax=axes[0,1], color='skyblue')
    axes[0,1].set_title('Satisfaction Rate by Age Group (%)', fontweight='bold')
    axes[0,1].set_xlabel('Age Group')
    axes[0,1].set_ylabel('Satisfaction Rate (%)')
    axes[0,1].tick_params(axis='x', rotation=45)

# 3. Customer Type Analysis
if 'Customer Type' in df.columns:
    loyalty_data = pd.crosstab(df['Customer Type'], df['satisfaction'], normalize='index') * 100
    loyalty_data.plot(kind='bar', ax=axes[1,0], color=['#ffb347', '#87ceeb'])
    axes[1,0].set_title('Satisfaction by Customer Loyalty (%)', fontweight='bold')
    axes[1,0].set_xlabel('Customer Type')
    axes[1,0].set_ylabel('Percentage')
    axes[1,0].legend(title='Satisfaction')
    axes[1,0].tick_params(axis='x', rotation=45)

# 4. Age Distribution
df['Age'].hist(bins=20, ax=axes[1,1], color='lightcoral', alpha=0.7)
axes[1,1].set_title('Age Distribution of Customers', fontweight='bold')
axes[1,1].set_xlabel('Age')
axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Print statistical insights
print("\nğŸ“Š DEMOGRAPHIC INSIGHTS:")
if 'Gender' in df.columns:
    male_sat = df[df['Gender'] == 'Male']['satisfaction'].apply(lambda x: x == 'satisfied').mean()
    female_sat = df[df['Gender'] == 'Female']['satisfaction'].apply(lambda x: x == 'satisfied').mean()
    print(f"â€¢ Male satisfaction: {male_sat:.1%}")
    print(f"â€¢ Female satisfaction: {female_sat:.1%}")

if 'Customer Type' in df.columns:
    loyal_sat = df[df['Customer Type'] == 'Loyal Customer']['satisfaction'].apply(lambda x: x == 'satisfied').mean()
    disloyal_sat = df[df['Customer Type'] == 'disloyal Customer']['satisfaction'].apply(lambda x: x == 'satisfied').mean()
    print(f"â€¢ Loyal customer satisfaction: {loyal_sat:.1%}")
    print(f"â€¢ Disloyal customer satisfaction: {disloyal_sat:.1%}")


# Cell 5: ANALYSIS 2 - Travel Pattern Analysis
print("=== ANALYSIS 2: TRAVEL PATTERNS ===\n")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Travel Pattern Analysis', fontsize=16, fontweight='bold')

# 1. Travel Class Analysis
if 'Class' in df.columns:
    class_data = pd.crosstab(df['Class'], df['satisfaction'], normalize='index') * 100
    class_data.plot(kind='bar', ax=axes[0,0], color=['#ff9999', '#66b3ff'])
    axes[0,0].set_title('Satisfaction by Travel Class (%)', fontweight='bold')
    axes[0,0].set_xlabel('Class')
    axes[0,0].set_ylabel('Percentage')
    axes[0,0].legend(title='Satisfaction')
    axes[0,0].tick_params(axis='x', rotation=45)

# 2. Travel Type Analysis
if 'Type of Travel' in df.columns:
    travel_data = pd.crosstab(df['Type of Travel'], df['satisfaction'], normalize='index') * 100
    travel_data.plot(kind='bar', ax=axes[0,1], color=['#ffcc99', '#99ffcc'])
    axes[0,1].set_title('Satisfaction by Travel Purpose (%)', fontweight='bold')
    axes[0,1].set_xlabel('Travel Type')
    axes[0,1].set_ylabel('Percentage')
    axes[0,1].legend(title='Satisfaction')
    axes[0,1].tick_params(axis='x', rotation=45)

# 3. Flight Distance Distribution
if 'Flight Distance' in df.columns:
    df['Flight Distance'].hist(bins=30, ax=axes[1,0], color='gold', alpha=0.7)
    axes[1,0].set_title('Flight Distance Distribution', fontweight='bold')
    axes[1,0].set_xlabel('Flight Distance (miles)')
    axes[1,0].set_ylabel('Frequency')

# 4. Distance vs Satisfaction
if 'Flight Distance' in df.columns:
    # Create distance categories
    df['Distance_Category'] = pd.cut(df['Flight Distance'],
                                   bins=[0, 500, 1500, 3000, float('inf')],
                                   labels=['Short\n(<500)', 'Medium\n(500-1500)', 'Long\n(1500-3000)', 'Ultra Long\n(>3000)'])

    distance_satisfaction = df.groupby('Distance_Category')['satisfaction'].apply(lambda x: (x == 'satisfied').mean() * 100)
    distance_satisfaction.plot(kind='bar', ax=axes[1,1], color='mediumseagreen')
    axes[1,1].set_title('Satisfaction Rate by Flight Distance (%)', fontweight='bold')
    axes[1,1].set_xlabel('Distance Category')
    axes[1,1].set_ylabel('Satisfaction Rate (%)')
    axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Print insights
print("\nâœˆï¸� TRAVEL PATTERN INSIGHTS:")
if 'Class' in df.columns:
    class_satisfaction = df.groupby('Class')['satisfaction'].apply(lambda x: (x == 'satisfied').mean())
    for tclass, rate in class_satisfaction.items():
        print(f"â€¢ {tclass} class satisfaction: {rate:.1%}")


print("=== ANALYSIS 3: SERVICE QUALITY ANALYSIS ===\n")

# Service rating columns
service_columns = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]

existing_service_cols = [col for col in service_columns if col in df.columns]
print(f"Found {len(existing_service_cols)} service quality columns")

if existing_service_cols:
    # Calculate correlations
    satisfaction_binary = (df['satisfaction'] == 'satisfied').astype(int)
    correlations = {}

    for service in existing_service_cols:
        if df[service].dtype in ['int64', 'float64']:
            correlation, p_value = pearsonr(df[service].fillna(df[service].mean()), satisfaction_binary)
            correlations[service] = correlation

    # Create correlation plot
    plt.figure(figsize=(12, 8))

    # Sort correlations
    sorted_correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

    services = list(sorted_correlations.keys())[:10]  # Top 10
    corr_values = [sorted_correlations[s] for s in services]

    # Create horizontal bar plot
    colors = ['green' if x > 0 else 'red' for x in corr_values]
    bars = plt.barh(range(len(services)), corr_values, color=colors, alpha=0.7)

    plt.yticks(range(len(services)), [s.replace(' ', '\n') for s in services])
    plt.xlabel('Correlation with Satisfaction')
    plt.title('Top 10 Service Factors - Correlation with Customer Satisfaction', fontweight='bold', fontsize=14)
    plt.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, corr_values)):
        plt.text(value + 0.01 if value > 0 else value - 0.01, i, f'{value:.3f}',
                va='center', ha='left' if value > 0 else 'right')

    plt.tight_layout()
    plt.show()

    # Print top correlations
    print("\nğŸŒŸ TOP SERVICE SATISFACTION DRIVERS:")
    for i, (service, corr) in enumerate(list(sorted_correlations.items())[:5], 1):
        print(f"{i}. {service}: r = {corr:.3f}")


# Cell 7: ANALYSIS 4 - Delay Impact Analysis
print("=== ANALYSIS 4: FLIGHT DELAY IMPACT ===\n")

delay_columns = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
existing_delay_cols = [col for col in delay_columns if col in df.columns]

if existing_delay_cols:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Flight Delay Impact on Customer Satisfaction', fontsize=16, fontweight='bold')

    for idx, delay_col in enumerate(existing_delay_cols):
        # Create delay categories
        df[f'{delay_col}_Category'] = pd.cut(df[delay_col].fillna(0),
                                           bins=[-1, 0, 15, 60, float('inf')],
                                           labels=['No Delay', 'Short\n(1-15 min)', 'Medium\n(16-60 min)', 'Long\n(>60 min)'])

        # Satisfaction by delay category
        delay_satisfaction = df.groupby(f'{delay_col}_Category')['satisfaction'].apply(lambda x: (x == 'satisfied').mean() * 100)

        # Plot satisfaction rates
        delay_satisfaction.plot(kind='bar', ax=axes[0, idx], color=['green', 'yellow', 'orange', 'red'])
        axes[0, idx].set_title(f'Satisfaction by {delay_col}', fontweight='bold')
        axes[0, idx].set_ylabel('Satisfaction Rate (%)')
        axes[0, idx].tick_params(axis='x', rotation=45)

        # Plot delay distribution
        df[delay_col].hist(bins=30, ax=axes[1, idx], alpha=0.7, color='lightblue')
        axes[1, idx].set_title(f'{delay_col} Distribution', fontweight='bold')
        axes[1, idx].set_xlabel('Delay (minutes)')
        axes[1, idx].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Print delay impact insights
    print("\nâ�° DELAY IMPACT INSIGHTS:")
    for delay_col in existing_delay_cols:
        delay_satisfaction = df.groupby(f'{delay_col}_Category')['satisfaction'].apply(lambda x: (x == 'satisfied').mean())
        no_delay_rate = delay_satisfaction.get('No Delay', 0)
        long_delay_rate = delay_satisfaction.get('Long\n(>60 min)', 0)
        if no_delay_rate > 0 and long_delay_rate > 0:
            impact = no_delay_rate - long_delay_rate
            print(f"â€¢ {delay_col}: {impact:.1%} satisfaction decrease with long delays")


# Cell 8: ANALYSIS 5 - Comprehensive Dashboard
print("=== ANALYSIS 5: SATISFACTION DASHBOARD ===\n")

# Create a comprehensive dashboard
fig = plt.figure(figsize=(20, 16))

# 1. Overall satisfaction pie chart
ax1 = plt.subplot(3, 3, 1)
satisfaction_counts = df['satisfaction'].value_counts()
colors = ['lightgreen', 'lightcoral']
wedges, texts, autotexts = ax1.pie(satisfaction_counts.values, labels=satisfaction_counts.index,
                                  autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Overall Satisfaction Distribution', fontweight='bold')

# 2. Satisfaction by multiple factors
if 'Class' in df.columns and 'Type of Travel' in df.columns:
    ax2 = plt.subplot(3, 3, 2)
    cross_tab = pd.crosstab(df['Class'], df['Type of Travel'], df['satisfaction'], aggfunc='count')
    if 'satisfied' in cross_tab.columns:
        sns.heatmap(cross_tab['satisfied'], annot=True, fmt='d', cmap='YlOrRd', ax=ax2)
    ax2.set_title('Satisfied Customers: Class vs Travel Type', fontweight='bold')

# 3. Age vs Satisfaction scatter
if 'Age' in df.columns:
    ax3 = plt.subplot(3, 3, 3)
    satisfied = df[df['satisfaction'] == 'satisfied']
    dissatisfied = df[df['satisfaction'] != 'satisfied']

    ax3.scatter(satisfied['Age'], np.random.normal(1, 0.1, len(satisfied)),
               alpha=0.5, color='green', label='Satisfied')
    ax3.scatter(dissatisfied['Age'], np.random.normal(0, 0.1, len(dissatisfied)),
               alpha=0.5, color='red', label='Dissatisfied')
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Satisfaction')
    ax3.set_title('Age vs Satisfaction Pattern', fontweight='bold')
    ax3.legend()

# 4. Service ratings heatmap
if existing_service_cols:
    ax4 = plt.subplot(3, 3, (4, 6))
    service_means = df.groupby('satisfaction')[existing_service_cols[:8]].mean()
    sns.heatmap(service_means, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4)
    ax4.set_title('Average Service Ratings by Satisfaction Level', fontweight='bold')

# 5. Flight distance impact
if 'Flight Distance' in df.columns:
    ax5 = plt.subplot(3, 3, 7)
    for satisfaction_level in df['satisfaction'].unique():
        subset = df[df['satisfaction'] == satisfaction_level]['Flight Distance']
        ax5.hist(subset, alpha=0.6, label=satisfaction_level, bins=20)
    ax5.set_xlabel('Flight Distance')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Flight Distance Distribution by Satisfaction', fontweight='bold')
    ax5.legend()

# 6. Customer type analysis
if 'Customer Type' in df.columns:
    ax6 = plt.subplot(3, 3, 8)
    customer_satisfaction = df.groupby(['Customer Type', 'satisfaction']).size().unstack(fill_value=0)
    customer_satisfaction.plot(kind='bar', stacked=True, ax=ax6, color=['lightcoral', 'lightgreen'])
    ax6.set_title('Customer Type vs Satisfaction', fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)

# 7. Summary statistics
ax7 = plt.subplot(3, 3, 9)
ax7.axis('off')
summary_text = f"""
SUMMARY STATISTICS
==================
Total Records: {len(df):,}
Overall Satisfaction: {(df['satisfaction'] == 'satisfied').mean():.1%}

Customer Demographics:
- Average Age: {df['Age'].mean():.1f} years
- Gender Split: {df['Gender'].value_counts().to_dict()}

Travel Patterns:
- Business Travel: {(df['Type of Travel'] == 'Business travel').mean():.1%}
- Business Class: {(df['Class'] == 'Business').mean():.1%}

Service Quality:
- Avg Service Rating: {df[existing_service_cols].mean().mean():.2f}/5
"""
ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()


# Cell 9: Key Insights & Recommendations
print("=== KEY INSIGHTS & BUSINESS RECOMMENDATIONS ===\n")

print("ğŸ�¯ MAJOR FINDINGS:")

# Calculate key metrics
overall_satisfaction = (df['satisfaction'] == 'satisfied').mean()
print(f"1. Overall satisfaction rate: {overall_satisfaction:.1%}")

if 'Customer Type' in df.columns:
    loyal_vs_disloyal = df.groupby('Customer Type')['satisfaction'].apply(lambda x: (x == 'satisfied').mean())
    print(f"2. Loyal customers: {loyal_vs_disloyal.get('Loyal Customer', 0):.1%} vs Disloyal: {loyal_vs_disloyal.get('disloyal Customer', 0):.1%}")

if 'Class' in df.columns:
    class_satisfaction = df.groupby('Class')['satisfaction'].apply(lambda x: (x == 'satisfied').mean())
    best_class = class_satisfaction.idxmax()
    worst_class = class_satisfaction.idxmin()
    print(f"3. Best performing class: {best_class} ({class_satisfaction[best_class]:.1%})")
    print(f"4. Needs improvement: {worst_class} class ({class_satisfaction[worst_class]:.1%})")

if existing_service_cols and correlations:
    top_service = max(correlations.items(), key=lambda x: abs(x[1]))
    print(f"5. Most important service factor: {top_service[0]} (r = {top_service[1]:.3f})")

print("\nğŸ“‹ STRATEGIC RECOMMENDATIONS:")
recommendations = [
    "Focus on improving service quality factors with highest correlation to satisfaction",
    "Develop targeted strategies for underperforming customer segments",
    "Implement proactive delay management and communication protocols",
    "Consider class-specific service enhancements based on satisfaction gaps",
    "Create loyalty program incentives addressing specific pain points",
    "Establish real-time satisfaction monitoring for immediate service recovery"
]

for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

print(f"\nâœ… Analysis completed successfully!")
print(f"Dataset: {len(df):,} records | Variables: {len(df.columns)} | Research Questions: 6")
print("Ready for academic submission and business decision-making.")

