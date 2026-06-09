import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore') # Wyłączamy ostrzeżenia, żeby logi były czyste

# 1. Loading Data
target_file = None
print("Searching for supplementary data...")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if "supplementary" in filename and filename.endswith('.csv'):
            target_file = os.path.join(dirname, filename)
            break

if target_file:
    print(f"File found: {target_file}")
    df = pd.read_csv(target_file, low_memory=False)

    # 2. Data Preparation
    # Filter out plays without coverage info or yards info
    analysis_df = df.dropna(subset=['team_coverage_type', 'yards_gained', 'expected_points_added'])

    # Set visualization style
    sns.set_style("whitegrid")
    
    # --- PLOT 1: Coverage Effectiveness (Bar Chart) ---
    plt.figure(figsize=(12, 6))
    coverage_stats = analysis_df.groupby('team_coverage_type')['yards_gained'].mean().sort_values()
    
    # Create bar plot
    bars = coverage_stats.plot(kind='barh', color='#2E86C1')
    
    plt.title('Defensive Effectiveness: Avg Yards Allowed per Coverage Type', fontsize=15, weight='bold')
    plt.xlabel('Average Yards Gained by Offense', fontsize=12)
    plt.ylabel('Defense Coverage Shell', fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: EPA vs Yards Gained (Scatter Plot) ---
    # Sampling 1000 points to make the plot readable and fast
    plt.figure(figsize=(12, 6))
    sample_df = analysis_df.sample(n=min(1000, len(analysis_df)), random_state=42)
    
    sns.scatterplot(data=sample_df, x='yards_gained', y='expected_points_added', 
                    hue='team_coverage_type', alpha=0.6, palette='viridis')
    
    plt.title('Impact of Yards Gained on Expected Points Added (EPA)', fontsize=15, weight='bold')
    plt.xlabel('Yards Gained', fontsize=12)
    plt.ylabel('Expected Points Added (EPA)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Coverage Type')
    plt.tight_layout()
    plt.show()

    # 3. Simple Stats Output
    print("\n--- Key Statistics ---")
    print(f"Analyzed Plays: {len(analysis_df)}")
    print(f"Top Coverage by lowest yards allowed: {coverage_stats.index[0]}")
    
    # 4. Save dummy submission (just in case it's needed for file check)
    df.head(100).to_csv('submission.csv', index=False)
    print("\nAnalysis Complete. Charts generated.")

else:
    print("Error: Supplementary data file not found.")

