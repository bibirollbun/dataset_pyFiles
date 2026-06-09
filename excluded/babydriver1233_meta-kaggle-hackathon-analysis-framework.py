import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import kagglehub
import re
from wordcloud import WordCloud
import os


def load_meta_kaggle_data():
    """Load datasets with detailed file discovery"""
    print("â�³ Downloading datasets...")
    try:
        meta_kaggle_path = Path(kagglehub.dataset_download("kaggle/meta-kaggle"))
        meta_code_path = Path(kagglehub.dataset_download("kaggle/meta-kaggle-code"))
        print("âœ… Datasets downloaded successfully")
        
        # Print available files for debugging
        print("\nğŸ“‚ Files in Meta Kaggle:")
        print([f.name for f in meta_kaggle_path.glob("*.csv")][:10])
        print("\nğŸ“‚ Files in Meta Kaggle Code:")
        print([f.name for f in meta_code_path.glob("*")][:10])
        
        return meta_kaggle_path, meta_code_path
    except Exception as e:
        print(f"â�Œ Download failed: {str(e)}")
        return None, None

def safe_read_csv(path):
    """Robust CSV reading with memory management"""
    if path is None or not path.exists():
        print(f"âš ï¸� File not available: {path.name if path else 'None'}")
        return None
    
    try:
        # Read in chunks for memory efficiency
        chunks = []
        for chunk in pd.read_csv(path, low_memory=False, chunksize=100000):
            chunks.append(chunk)
        df = pd.concat(chunks)
        print(f"âœ… Loaded {path.name} ({len(df):,} rows)")
        return df
    except Exception as e:
        print(f"â�Œ Failed to load {path.name}: {str(e)}")
        return None


print("\nğŸ”� Loading Meta Kaggle Data...")
meta_kaggle_path, meta_code_path = load_meta_kaggle_data()

# Define EXACT file names we want to load
exact_files = {
    'competitions': 'Competitions.csv',
    'users': 'Users.csv',
    'teams': 'Teams.csv',
    'submissions': 'Submissions.csv',
    'kernels': 'Kernels.csv',
    'tags': 'Tags.csv',
    'competition_tags': 'CompetitionTags.csv',
    'kernel_versions': 'KernelVersions.csv'
}

# Load all datasets by exact filename
loaded_data = {}
for name, filename in exact_files.items():
    if 'kernel' in name and meta_code_path:
        path = meta_code_path / filename
    else:
        path = meta_kaggle_path / filename
    
    loaded_data[name] = safe_read_csv(path)

# Unpack for easier access
competitions = loaded_data['competitions']
users = loaded_data['users']
teams = loaded_data['teams']
submissions = loaded_data['submissions']
kernels = loaded_data['kernels']
tags = loaded_data['tags']
competition_tags = loaded_data['competition_tags']
kernel_versions = loaded_data['kernel_versions']


def preprocess_data():
    """Clean and prepare the loaded data"""
    print("\nğŸ§¹ Preprocessing data...")
    
    # Process competitions data
    if competitions is not None:
        print("Competitions columns:", competitions.columns.tolist())
        if 'EnabledDate' in competitions.columns:
            competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate'], errors='coerce')
            competitions['Year'] = competitions['EnabledDate'].dt.year
        if 'RewardQuantity' in competitions.columns:
            competitions['PrizeAmount'] = competitions['RewardQuantity'].fillna(0)
    
    # Process users data
    if users is not None:
        if 'RegisterDate' in users.columns:
            users['RegisterDate'] = pd.to_datetime(users['RegisterDate'], errors='coerce')
    
    # Process teams data - use TeamLeaderId as user identifier
    if teams is not None:
        print("Teams columns:", teams.columns.tolist())
        if 'TeamLeaderId' in teams.columns:
            teams['UserId'] = teams['TeamLeaderId']
    
    # Process kernels data
    if kernels is not None:
        if 'CreationDate' in kernels.columns:
            kernels['CreationDate'] = pd.to_datetime(kernels['CreationDate'], errors='coerce')
            kernels['Year'] = kernels['CreationDate'].dt.year
    
    print("âœ… Data preprocessing complete")

preprocess_data()


def analyze_competitions():
    """Analyze competition trends"""
    if competitions is None:
        print("â�Œ No competition data available")
        return
    
    print("\nğŸ�† Competition Analysis")
    
    # Plot competitions per year
    if 'Year' in competitions:
        plt.figure(figsize=(14, 7))
        competitions['Year'].value_counts().sort_index().plot(kind='bar', color='steelblue')
        plt.title('Kaggle Competitions per Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Competitions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("âš ï¸� No year data available for competitions")
    
    # Plot prize money trends
    if 'PrizeAmount' in competitions:
        yearly_prizes = competitions.groupby('Year')['PrizeAmount'].sum()
        plt.figure(figsize=(14, 7))
        yearly_prizes.plot(kind='bar', color='gold')
        plt.title('Total Prize Money by Year (USD)')
        plt.ylabel('Prize Money (USD)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    elif 'RewardQuantity' in competitions:
        print("â„¹ï¸� Prize amounts available in RewardQuantity column")
    else:
        print("âš ï¸� No prize data available")

def analyze_notebooks():
    """Analyze notebook trends"""
    if kernels is None:
        print("â�Œ No kernel data available")
        return
    
    print("\nğŸ“Š Notebook Analysis")
    
    # Notebook creation over time
    if 'Year' in kernels:
        notebook_counts = kernels['Year'].value_counts().sort_index()
        plt.figure(figsize=(14, 7))
        notebook_counts.plot(kind='line', marker='o', color='green')
        plt.title('Kaggle Notebooks Created by Year')
        plt.ylabel('Number of Notebooks')
        plt.tight_layout()
        plt.show()
    
    # Popular languages
    if 'CurrentLanguage' in kernels:
        langs = kernels['CurrentLanguage'].value_counts().head(10)
        plt.figure(figsize=(12, 6))
        langs.plot(kind='barh', color='blue')
        plt.title('Top 10 Notebook Languages')
        plt.tight_layout()
        plt.show()

def analyze_users():
    """Analyze user participation using TeamLeaderId"""
    if users is None or teams is None:
        print("â�Œ Missing user or team data")
        return
    
    print("\nğŸ‘¥ User Analysis")
    
    if 'UserId' not in teams.columns:
        print("âš ï¸� No user identifier found in teams data")
        return
        
    user_activity = teams['UserId'].value_counts().reset_index()
    user_activity.columns = ['UserId', 'CompetitionsJoined']
    
    # Merge with user data
    user_activity = user_activity.merge(users, left_on='UserId', right_on='Id')
    
    # Plot distribution
    plt.figure(figsize=(14, 7))
    sns.histplot(user_activity['CompetitionsJoined'], bins=50, kde=True)
    plt.title('Distribution of Competitions per User (Team Leaders)')
    plt.xlabel('Number of Competitions Joined')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()
    
    # Show top users
    username_col = 'UserName' if 'UserName' in user_activity.columns else 'DisplayName'
    if username_col in user_activity.columns:
        print("\nTop 10 Most Active Users:")
        print(user_activity[[username_col, 'CompetitionsJoined']]
              .sort_values('CompetitionsJoined', ascending=False)
              .head(10))
    else:
        print("â„¹ï¸� Could not find username column to display top users")

def analyze_code():
    """Analyze code patterns"""
    if kernel_versions is None:
        print("â�Œ No kernel versions data available")
        return
    
    print("\nğŸ’» Code Analysis")
    
    try:
        # Sample code for library analysis
        sample = kernel_versions.dropna(subset=['ScriptText']).sample(1000, random_state=1)
        combined_code = ' '.join(sample['ScriptText'].astype(str))
        
        # Find import statements
        imports = re.findall(r'(?:from\s+(\w+)|(?:import\s+(\w+))', combined_code)
        libs = [imp[0] or imp[1] for imp in imports if any(imp)]
        lib_counts = pd.Series(libs).value_counts().head(15)
        
        # Plot
        plt.figure(figsize=(12, 8))
        lib_counts.plot(kind='barh', color='purple')
        plt.title('Top 15 Imported Python Libraries')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"â�Œ Code analysis failed: {str(e)}")


print("\nğŸ“Š Running analysis...")
analyze_competitions()
analyze_notebooks()
analyze_users()
analyze_code()

print("\nğŸ�‰ Analysis complete!")
print("\nğŸ”‘ Key Insights:")
print("- Competition trends over time")
print("- Notebook creation and language popularity")
print("- User participation patterns (team leaders)")
print("- Common Python libraries in Kaggle code")

# Save all figures
os.makedirs('output', exist_ok=True)
for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(f'output/figure_{i}.png', dpi=300, bbox_inches='tight')
print("\nğŸ’¾ Visualizations saved to /output directory")

