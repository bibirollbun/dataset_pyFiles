# Cell 1: Setup and Library Installation
!pip install plotly seaborn scipy numpy pandas matplotlib -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, ttest_ind
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

print("ğŸ�¯ CUSTOMER SATISFACTION ANALYSIS FRAMEWORK")
print("=" * 50)
print("âœ… Libraries imported successfully!")
print("ğŸ“Š Ready for data analysis")



# Cell 2: Data Loading and Initial Exploration
print("\n=== PHASE 1: DATA EXPLORATION ===")

# Upload your CSV file to Google Colab and update the path
# For Colab: upload file and use the filename directly
df = pd.read_csv('/kaggle/input/initial/train_dataset.csv')  # Replace with your filename

# Clean unnecessary columns
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print(f"ğŸ“‹ Dataset Overview:")
print(f"   â€¢ Shape: {df.shape}")
print(f"   â€¢ Total Records: {len(df):,}")
print(f"   â€¢ Features: {len(df.columns)}")

# Show column information
print(f"\nğŸ“Š Column Information:")
print(df.dtypes)

# Check for missing values
print(f"\nğŸ”� Missing Values:")
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    print(missing_data[missing_data > 0])
else:
    print("   âœ… No missing values found")

# Display first few rows
print(f"\nğŸ‘€ Sample Data:")
df.head()


# Cell 3: Research Questions Framework
print("\n=== RESEARCH QUESTIONS FRAMEWORK ===")

research_objectives = {
    "RQ1": "How do demographic factors influence customer satisfaction levels?",
    "RQ2": "What is the relationship between travel patterns and satisfaction?",
    "RQ3": "Which service dimensions drive customer satisfaction most significantly?",
    "RQ4": "How do operational factors (delays, distance) impact customer experience?",
    "RQ5": "What are the satisfaction differences between customer segments?",
    "RQ6": "Can we identify key satisfaction predictors for business strategy?"
}

print("ğŸ�¯ RESEARCH OBJECTIVES:")
for rq, objective in research_objectives.items():
    print(f"   {rq}: {objective}")

# Check target variable distribution
print(f"\nğŸ“ˆ Target Variable Analysis:")
satisfaction_dist = df['satisfaction'].value_counts()
for category, count in satisfaction_dist.items():
    percentage = (count / len(df)) * 100
    print(f"   â€¢ {category}: {count:,} ({percentage:.1f}%)")



# Cell 4: ANALYSIS 1 - Demographic Satisfaction Patterns
print("\n=== ANALYSIS 1: DEMOGRAPHIC SATISFACTION PATTERNS ===")

# Create comprehensive demographic analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Customer Demographics vs Satisfaction Analysis', fontsize=16, fontweight='bold')

# 1. Gender Analysis
if 'Gender' in df.columns:
    gender_crosstab = pd.crosstab(df['Gender'], df['satisfaction'], normalize='index') * 100
    gender_crosstab.plot(kind='bar', ax=axes[0,0], color=['#FF6B6B', '#4ECDC4'], rot=0)
    axes[0,0].set_title('Satisfaction Rate by Gender (%)', fontweight='bold')
    axes[0,0].set_xlabel('Gender')
    axes[0,0].set_ylabel('Percentage')
    axes[0,0].legend(title='Satisfaction', bbox_to_anchor=(1.05, 1))

# 2. Age Distribution Analysis
if 'Age' in df.columns:
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'],
                            bins=[0, 30, 45, 60, 100],
                            labels=['Young (â‰¤30)', 'Middle (31-45)', 'Mature (46-60)', 'Senior (60+)'])

    age_satisfaction = df.groupby('Age_Group')['satisfaction'].apply(lambda x: (x == 'satisfied').mean() * 100)
    age_satisfaction.plot(kind='bar', ax=axes[0,1], color='lightblue', rot=45)
    axes[0,1].set_title('Satisfaction Rate by Age Group (%)', fontweight='bold')
    axes[0,1].set_xlabel('Age Group')
    axes[0,1].set_ylabel('Satisfaction Rate (%)')

# 3. Customer Loyalty Analysis
if 'Customer Type' in df.columns:
    loyalty_crosstab = pd.crosstab(df['Customer Type'], df['satisfaction'], normalize='index') * 100
    loyalty_crosstab.plot(kind='bar', ax=axes[0,2], color=['#FFB347', '#87CEEB'], rot=45)
    axes[0,2].set_title('Satisfaction by Customer Loyalty (%)', fontweight='bold')
    axes[0,2].set_xlabel('Customer Type')
    axes[0,2].set_ylabel('Percentage')
    axes[0,2].legend(title='Satisfaction')

# 4. Age Distribution Histogram
df['Age'].hist(bins=25, ax=axes[1,0], color='coral', alpha=0.7, edgecolor='black')
axes[1,0].axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Age"].mean():.1f}')
axes[1,0].set_title('Customer Age Distribution', fontweight='bold')
axes[1,0].set_xlabel('Age (years)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].legend()

# 5. Gender vs Age Satisfaction Heatmap
if 'Gender' in df.columns and 'Age_Group' in df.columns:
    heatmap_data = df.groupby(['Gender', 'Age_Group'])['satisfaction'].apply(lambda x: (x == 'satisfied').mean()).unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1,1])
    axes[1,1].set_title('Satisfaction Rate: Gender vs Age Group', fontweight='bold')

# 6. Customer Type Distribution
if 'Customer Type' in df.columns:
    customer_counts = df['Customer Type'].value_counts()
    axes[1,2].pie(customer_counts.values, labels=customer_counts.index, autopct='%1.1f%%',
                  colors=['lightgreen', 'lightcoral'], startangle=90)
    axes[1,2].set_title('Customer Type Distribution', fontweight='bold')

plt.tight_layout()
plt.show()

# Statistical Analysis
print("\nğŸ“Š DEMOGRAPHIC INSIGHTS:")
if 'Gender' in df.columns:
    male_satisfaction = df[df['Gender'] == 'Male']['satisfaction'].apply(lambda x: x == 'satisfied').mean()
    female_satisfaction = df[df['Gender'] == 'Female']['satisfaction'].apply(lambda x: x == 'satisfied').mean()
    print(f"   â€¢ Male satisfaction rate: {male_satisfaction:.1%}")
    print(f"   â€¢ Female satisfaction rate: {female_satisfaction:.1%}")
    print(f"   â€¢ Gender satisfaction gap: {abs(male_satisfaction - female_satisfaction):.1%}")

if 'Customer Type' in df.columns:
    loyal_satisfaction = df[df['Customer Type'] == 'Loyal Customer']['satisfaction'].apply(lambda x: x == 'satisfied').mean()
    disloyal_satisfaction = df[df['Customer Type'] == 'disloyal Customer']['satisfaction'].apply(lambda x: x == 'satisfied').mean()
    print(f"   â€¢ Loyal customer satisfaction: {loyal_satisfaction:.1%}")
    print(f"   â€¢ Disloyal customer satisfaction: {disloyal_satisfaction:.1%}")
    print(f"   â€¢ Loyalty impact: +{loyal_satisfaction - disloyal_satisfaction:.1%}")


# Cell 5: ANALYSIS 2 - Travel Behavior and Satisfaction
print("\n=== ANALYSIS 2: TRAVEL BEHAVIOR ANALYSIS ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Travel Patterns and Customer Satisfaction', fontsize=16, fontweight='bold')

# 1. Travel Class Analysis
if 'Class' in df.columns:
    class_satisfaction = pd.crosstab(df['Class'], df['satisfaction'], normalize='index') * 100
    class_satisfaction.plot(kind='bar', ax=axes[0,0], color=['#FF9999', '#66B3FF'], rot=45)
    axes[0,0].set_title('Satisfaction by Travel Class (%)', fontweight='bold')
    axes[0,0].set_xlabel('Travel Class')
    axes[0,0].set_ylabel('Percentage')
    axes[0,0].legend(title='Satisfaction')

# 2. Travel Purpose Analysis
if 'Type of Travel' in df.columns:
    travel_satisfaction = pd.crosstab(df['Type of Travel'], df['satisfaction'], normalize='index') * 100
    travel_satisfaction.plot(kind='bar', ax=axes[0,1], color=['#FFCC99', '#99FFCC'], rot=45)
    axes[0,1].set_title('Satisfaction by Travel Purpose (%)', fontweight='bold')
    axes[0,1].set_xlabel('Travel Type')
    axes[0,1].set_ylabel('Percentage')
    axes[0,1].legend(title='Satisfaction')

# 3. Flight Distance Categories
if 'Flight Distance' in df.columns:
    # Create distance categories
    df['Distance_Category'] = pd.cut(df['Flight Distance'],
                                   bins=[0, 750, 2000, 4000, float('inf')],
                                   labels=['Short\n(<750mi)', 'Medium\n(750-2000mi)',
                                          'Long\n(2000-4000mi)', 'Ultra Long\n(>4000mi)'])

    distance_satisfaction = df.groupby('Distance_Category')['satisfaction'].apply(lambda x: (x == 'satisfied').mean() * 100)
    distance_satisfaction.plot(kind='bar', ax=axes[0,2], color='mediumseagreen', rot=45)
    axes[0,2].set_title('Satisfaction by Flight Distance (%)', fontweight='bold')
    axes[0,2].set_xlabel('Distance Category')
    axes[0,2].set_ylabel('Satisfaction Rate (%)')

# 4. Flight Distance Distribution
if 'Flight Distance' in df.columns:
    df['Flight Distance'].hist(bins=35, ax=axes[1,0], color='gold', alpha=0.7, edgecolor='black')
    axes[1,0].axvline(df['Flight Distance'].mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {df["Flight Distance"].mean():.0f} miles')
    axes[1,0].set_title('Flight Distance Distribution', fontweight='bold')
    axes[1,0].set_xlabel('Distance (miles)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()

# 5. Class vs Travel Type Heatmap
if 'Class' in df.columns and 'Type of Travel' in df.columns:
    class_travel_satisfaction = df.groupby(['Class', 'Type of Travel'])['satisfaction'].apply(
        lambda x: (x == 'satisfied').mean()).unstack(fill_value=0)
    sns.heatmap(class_travel_satisfaction, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1,1])
    axes[1,1].set_title('Satisfaction: Class vs Travel Purpose', fontweight='bold')

# 6. Distance vs Class Boxplot
if 'Flight Distance' in df.columns and 'Class' in df.columns:
    df.boxplot(column='Flight Distance', by='Class', ax=axes[1,2])
    axes[1,2].set_title('Flight Distance by Travel Class', fontweight='bold')
    axes[1,2].set_xlabel('Travel Class')
    axes[1,2].set_ylabel('Flight Distance (miles)')

plt.tight_layout()
plt.show()

# Travel Pattern Insights
print("\nâœˆï¸� TRAVEL PATTERN INSIGHTS:")
if 'Class' in df.columns:
    class_satisfaction_rates = df.groupby('Class')['satisfaction'].apply(lambda x: (x == 'satisfied').mean())
    print("   Class Satisfaction Rates:")
    for travel_class, rate in class_satisfaction_rates.items():
        print(f"     â€¢ {travel_class}: {rate:.1%}")

if 'Type of Travel' in df.columns:
    travel_type_rates = df.groupby('Type of Travel')['satisfaction'].apply(lambda x: (x == 'satisfied').mean())
    print("   Travel Purpose Satisfaction:")
    for travel_type, rate in travel_type_rates.items():
        print(f"     â€¢ {travel_type}: {rate:.1%}")



# Cell 6: ANALYSIS 3 - Service Quality Deep Dive
print("\n=== ANALYSIS 3: SERVICE QUALITY ANALYSIS ===")

# Identify service rating columns
service_columns = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]

existing_services = [col for col in service_columns if col in df.columns]
print(f"ğŸ“Š Analyzing {len(existing_services)} service quality dimensions")

if existing_services:
    # Calculate correlations with satisfaction
    satisfaction_binary = (df['satisfaction'] == 'satisfied').astype(int)
    service_correlations = {}

    for service in existing_services:
        if df[service].dtype in ['int64', 'float64']:
            # Handle missing values
            service_data = df[service].fillna(df[service].mean())
            correlation, p_value = pearsonr(service_data, satisfaction_binary)
            service_correlations[service] = {
                'correlation': correlation,
                'p_value': p_value,
                'mean_satisfied': df[df['satisfaction'] == 'satisfied'][service].mean(),
                'mean_dissatisfied': df[df['satisfaction'] != 'satisfied'][service].mean()
            }

    # Create comprehensive service analysis
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Service Quality Impact Analysis', fontsize=16, fontweight='bold')

    # 1. Top Correlation Factors
    sorted_correlations = dict(sorted(service_correlations.items(),
                                    key=lambda x: abs(x[1]['correlation']), reverse=True))

    top_services = list(sorted_correlations.keys())[:10]
    correlation_values = [sorted_correlations[s]['correlation'] for s in top_services]

    colors = ['darkgreen' if x > 0.3 else 'green' if x > 0.2 else 'orange' if x > 0.1 else 'red' for x in correlation_values]
    bars = axes[0,0].barh(range(len(top_services)), correlation_values, color=colors, alpha=0.8)

    axes[0,0].set_yticks(range(len(top_services)))
    axes[0,0].set_yticklabels([s.replace(' ', '\n').replace('/', '/\n') for s in top_services])
    axes[0,0].set_xlabel('Correlation with Satisfaction')
    axes[0,0].set_title('Top 10 Service Satisfaction Drivers', fontweight='bold')
    axes[0,0].grid(axis='x', alpha=0.3)

    # Add correlation values on bars
    for i, (bar, value) in enumerate(zip(bars, correlation_values)):
        axes[0,0].text(value + 0.005 if value > 0 else value - 0.005, i, f'{value:.3f}',
                       va='center', ha='left' if value > 0 else 'right', fontweight='bold')

    # 2. Service Rating Distribution
    service_means = df[existing_services[:8]].mean().sort_values(ascending=False)
    service_means.plot(kind='bar', ax=axes[0,1], color='skyblue', alpha=0.8)
    axes[0,1].set_title('Average Service Ratings (Top 8)', fontweight='bold')
    axes[0,1].set_ylabel('Average Rating')
    axes[0,1].set_xticklabels([s.replace(' ', '\n') for s in service_means.index], rotation=45)
    axes[0,1].grid(axis='y', alpha=0.3)

    # 3. Satisfaction Level Service Comparison
    service_by_satisfaction = df.groupby('satisfaction')[existing_services[:6]].mean()

    x = np.arange(len(existing_services[:6]))
    width = 0.35

    satisfied_means = service_by_satisfaction.loc['satisfied'] if 'satisfied' in service_by_satisfaction.index else [0]*6
    dissatisfied_means = service_by_satisfaction.loc['neutral or dissatisfied'] if 'neutral or dissatisfied' in service_by_satisfaction.index else [0]*6

    axes[1,0].bar(x - width/2, satisfied_means, width, label='Satisfied', color='lightgreen', alpha=0.8)
    axes[1,0].bar(x + width/2, dissatisfied_means, width, label='Dissatisfied', color='lightcoral', alpha=0.8)

    axes[1,0].set_xlabel('Service Dimensions')
    axes[1,0].set_ylabel('Average Rating')
    axes[1,0].set_title('Service Ratings: Satisfied vs Dissatisfied', fontweight='bold')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([s.replace(' ', '\n') for s in existing_services[:6]], rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)

    # 4. Service Quality Heatmap
    if len(existing_services) >= 6:
        service_satisfaction_matrix = df.groupby('satisfaction')[existing_services[:6]].mean()
        sns.heatmap(service_satisfaction_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   ax=axes[1,1], cbar_kws={'label': 'Average Rating'})
        axes[1,1].set_title('Service Quality Heatmap by Satisfaction', fontweight='bold')
        axes[1,1].set_ylabel('Satisfaction Level')

    plt.tight_layout()
    plt.show()

    # Print detailed service insights
    print("\nğŸŒŸ SERVICE QUALITY INSIGHTS:")
    print("   Top 5 Satisfaction Drivers:")
    for i, (service, data) in enumerate(list(sorted_correlations.items())[:5], 1):
        correlation = data['correlation']
        strength = "Strong" if abs(correlation) > 0.3 else "Moderate" if abs(correlation) > 0.2 else "Weak"
        print(f"     {i}. {service}")
        print(f"        â€¢ Correlation: {correlation:.3f} ({strength})")
        print(f"        â€¢ Satisfied customers rate: {data['mean_satisfied']:.2f}/5")
        print(f"        â€¢ Dissatisfied customers rate: {data['mean_dissatisfied']:.2f}/5")


# Cell 7: ANALYSIS 4 - Operational Factors Impact
print("\n=== ANALYSIS 4: OPERATIONAL FACTORS ANALYSIS ===")

# Analyze delays and operational factors
delay_columns = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
existing_delays = [col for col in delay_columns if col in df.columns]

if existing_delays:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Operational Factors Impact on Customer Satisfaction', fontsize=16, fontweight='bold')

    for idx, delay_col in enumerate(existing_delays):
        # Create delay categories
        df[f'{delay_col}_Category'] = pd.cut(df[delay_col].fillna(0),
                                           bins=[-1, 0, 15, 45, float('inf')],
                                           labels=['On Time', 'Short Delay\n(1-15min)',
                                                  'Medium Delay\n(16-45min)', 'Long Delay\n(>45min)'])

        # Satisfaction by delay category
        delay_satisfaction = df.groupby(f'{delay_col}_Category')['satisfaction'].apply(
            lambda x: (x == 'satisfied').mean() * 100)

        # Plot satisfaction rates
        colors = ['darkgreen', 'yellow', 'orange', 'red']
        delay_satisfaction.plot(kind='bar', ax=axes[0, idx], color=colors, alpha=0.8)
        axes[0, idx].set_title(f'Satisfaction by {delay_col.replace(" in Minutes", "")}', fontweight='bold')
        axes[0, idx].set_ylabel('Satisfaction Rate (%)')
        axes[0, idx].tick_params(axis='x', rotation=45)
        axes[0, idx].grid(axis='y', alpha=0.3)

        # Add percentage labels on bars
        for i, v in enumerate(delay_satisfaction.values):
            axes[0, idx].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Plot delay distribution
        delay_data = df[delay_col].fillna(0)
        delay_data[delay_data <= 300].hist(bins=30, ax=axes[1, idx], alpha=0.7,
                                          color='lightblue', edgecolor='black')
        axes[1, idx].axvline(delay_data.mean(), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {delay_data.mean():.1f}min')
        axes[1, idx].set_title(f'{delay_col.replace(" in Minutes", "")} Distribution', fontweight='bold')
        axes[1, idx].set_xlabel('Delay (minutes)')
        axes[1, idx].set_ylabel('Frequency')
        axes[1, idx].legend()

    # Combined delay impact analysis
    if len(existing_delays) == 2:
        # Calculate total delay
        df['Total_Delay'] = (df[existing_delays[0]].fillna(0) + df[existing_delays[1]].fillna(0))
        df['Total_Delay_Category'] = pd.cut(df['Total_Delay'],
                                          bins=[-1, 0, 30, 90, float('inf')],
                                          labels=['No Delay', 'Minor\n(<30min)',
                                                 'Moderate\n(30-90min)', 'Severe\n(>90min)'])

        total_delay_satisfaction = df.groupby('Total_Delay_Category')['satisfaction'].apply(
            lambda x: (x == 'satisfied').mean() * 100)

        colors = ['darkgreen', 'yellowgreen', 'orange', 'darkred']
        total_delay_satisfaction.plot(kind='bar', ax=axes[0, 2], color=colors, alpha=0.8)
        axes[0, 2].set_title('Satisfaction by Total Delay Impact', fontweight='bold')
        axes[0, 2].set_ylabel('Satisfaction Rate (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(axis='y', alpha=0.3)

        # Add percentage labels
        for i, v in enumerate(total_delay_satisfaction.values):
            axes[0, 2].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Total delay distribution
        total_delay_filtered = df['Total_Delay'][df['Total_Delay'] <= 400]
        total_delay_filtered.hist(bins=25, ax=axes[1, 2], alpha=0.7,
                                 color='lightcoral', edgecolor='black')
        axes[1, 2].axvline(df['Total_Delay'].mean(), color='darkred', linestyle='--', linewidth=2,
                          label=f'Mean: {df["Total_Delay"].mean():.1f}min')
        axes[1, 2].set_title('Total Delay Distribution', fontweight='bold')
        axes[1, 2].set_xlabel('Total Delay (minutes)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()

    plt.tight_layout()
    plt.show()

    # Print delay impact insights
    print("\nâ�° OPERATIONAL IMPACT INSIGHTS:")
    for delay_col in existing_delays:
        delay_categories = df[f'{delay_col}_Category'].unique()
        delay_satisfaction = df.groupby(f'{delay_col}_Category')['satisfaction'].apply(lambda x: (x == 'satisfied').mean())

        on_time_rate = delay_satisfaction.get('On Time', 0)
        long_delay_rate = delay_satisfaction.get('Long Delay\n(>45min)', 0)

        if on_time_rate > 0 and long_delay_rate > 0:
            impact = on_time_rate - long_delay_rate
            print(f"   â€¢ {delay_col}: {impact:.1%} satisfaction drop with long delays")

        print(f"     - On-time satisfaction: {on_time_rate:.1%}")
        print(f"     - Long delay satisfaction: {long_delay_rate:.1%}")




