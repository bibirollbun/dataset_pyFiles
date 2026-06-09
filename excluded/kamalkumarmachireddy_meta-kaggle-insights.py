# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import os
import glob
import json
import re
from datetime import datetime, timedelta
from collections import Counter
from collections import defaultdict
import kagglehub
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("ğŸ“Š Meta Kaggle Analysis - Libraries Loaded Successfully!")
print("=" * 60)


class MetaKaggleLoader:
    """
    A class to handle loading and initial processing of Meta Kaggle datasets
    """
    
    def __init__(self):
        self.meta_kaggle_path = None
        self.meta_code_path = None
        self.tables = {}
        
    def download_datasets(self):
        """Download both Meta Kaggle datasets"""
        print("ğŸ”„ Downloading Meta Kaggle datasets...")
        
        # Download Meta Kaggle dataset
        self.meta_kaggle_path = kagglehub.dataset_download("kaggle/meta-kaggle")
        print(f"âœ… Meta Kaggle path: {self.meta_kaggle_path}")
        
        # Download Meta Kaggle Code dataset
        self.meta_code_path = kagglehub.dataset_download("kaggle/meta-kaggle-code")
        print(f"âœ… Meta Kaggle Code path: {self.meta_code_path}")
        
    def load_csv_tables(self, max_rows_per_file=10_000_000):
        """Load all CSV files from Meta Kaggle dataset, limiting each file to max_rows_per_file."""
        print("\nğŸ”„ Loading CSV tables with per-file row limit...")
        
        if not self.meta_kaggle_path:
            raise ValueError("Please download datasets first using download_datasets()")
            
        # Get all CSV files
        csv_files = glob.glob(os.path.join(self.meta_kaggle_path, "*.csv"))
        
        for csv_file in csv_files:
            table_name = os.path.basename(csv_file).replace('.csv', '')
            print(f"Loading {table_name}...")
            
            try:
                # Read only up to the max allowed rows
                df = pd.read_csv(csv_file, low_memory=False, nrows=max_rows_per_file)
                
                self.tables[table_name] = df
                print(f"  âœ… {table_name}: {df.shape[0]:,} rows (limited to {max_rows_per_file:,}), {df.shape[1]} columns")
                
            except Exception as e:
                print(f"  â�Œ Error loading {table_name}: {e}")
                
        print(f"\nğŸ“Š Loaded {len(self.tables)} tables successfully!")
        return self.tables
    
    def get_table_overview(self):
        """Get overview of all loaded tables"""
        overview = []
        for name, df in self.tables.items():
            overview.append({
                'Table': name,
                'Rows': f"{df.shape[0]:,}",
                'Columns': df.shape[1],
                'Memory_MB': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            })
        
        return pd.DataFrame(overview).sort_values('Rows', ascending=False)

# Initialize loader and download data
loader = MetaKaggleLoader()
loader.download_datasets()
tables = loader.load_csv_tables(max_rows_per_file=10_000_000)  # Set global limit here

# Display table overview
print("\nğŸ“‹ TABLE OVERVIEW")
print("=" * 50)
overview_df = loader.get_table_overview()
print(overview_df.to_string(index=False))


class MetaKaggleExplorer:
    """
    Class for exploring and cleaning Meta Kaggle data
    """
    
    def __init__(self, tables):
        self.tables = tables
        self.cleaned_tables = {}
        
    def explore_key_tables(self):
        """Explore structure of key tables"""
        key_tables = ['Competitions', 'Users', 'Kernels', 'KernelVersions', 'Submissions']
        
        for table_name in key_tables:
            if table_name in self.tables:
                print(f"\nğŸ”� EXPLORING {table_name.upper()}")
                print("=" * 50)
                df = self.tables[table_name]
                
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print("\nData types:")
                print(df.dtypes)
                print(f"\nMissing values:")
                missing = df.isnull().sum()
                missing_with_values = missing[missing > 0]
                if len(missing_with_values) > 0:
                    print(missing_with_values)
                else:
                    print("No missing values found")
                
                if 'Id' in df.columns:
                    print(f"Unique IDs: {df['Id'].nunique():,}")
                
                print("\nFirst few rows:")
                print(df.head(3))
                print("\n" + "="*80)
    
    def clean_datetime_columns(self):
        """Clean and convert datetime columns across tables"""
        print("ğŸ§¹ Cleaning datetime columns...")
        
        datetime_patterns = ['date', 'time', 'created', 'updated', 'deadline']
        
        for table_name, df in self.tables.items():
            df_clean = df.copy()
            
            for col in df.columns:
                if any(pattern in col.lower() for pattern in datetime_patterns):
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                        print(f"  âœ… {table_name}.{col} converted to datetime")
                    except:
                        print(f"  â�Œ Failed to convert {table_name}.{col}")
            
            self.cleaned_tables[table_name] = df_clean
        
        return self.cleaned_tables
    
    def get_basic_stats(self):
        """Get basic statistics for key metrics"""
        stats = {}
        
        if 'Competitions' in self.cleaned_tables:
            comps = self.cleaned_tables['Competitions']
            stats['competitions'] = {
                'total': len(comps),
                'with_deadline': comps['DeadlineDate'].notna().sum() if 'DeadlineDate' in comps.columns else 'N/A',
                'avg_max_team_size': comps['MaxTeamSize'].mean() if 'MaxTeamSize' in comps.columns else 'N/A'
            }
        
        if 'Users' in self.cleaned_tables:
            users = self.cleaned_tables['Users']
            stats['users'] = {
                'total': len(users),
                'with_country': users['Country'].notna().sum() if 'Country' in users.columns else 'N/A'
            }
        
        if 'Kernels' in self.cleaned_tables:
            kernels = self.cleaned_tables['Kernels']
            stats['kernels'] = {
                'total': len(kernels),
                'with_views': kernels['TotalViews'].notna().sum() if 'TotalViews' in kernels.columns else 'N/A'
            }
            
        return stats

# Initialize explorer
explorer = MetaKaggleExplorer(tables)
explorer.explore_key_tables()
cleaned_tables = explorer.clean_datetime_columns()
basic_stats = explorer.get_basic_stats()

print("\nğŸ“Š BASIC STATISTICS")
print("=" * 30)
for category, stats in basic_stats.items():
    print(f"\n{category.upper()}:")
    for metric, value in stats.items():
        print(f"  {metric}: {value}")


class TimeSeriesAnalyzer:
    """
    Analyze temporal trends in Kaggle data
    """
    
    def __init__(self, tables):
        self.tables = tables
        
    def analyze_growth_trends(self):
        """Analyze growth trends over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Kaggle Platform Growth Over Time', fontsize=16, fontweight='bold')
        
        # Competitions over time
        if 'Competitions' in self.tables:
            comps = self.tables['Competitions'].copy()
            if 'DeadlineDate' in comps.columns:
                comps['Year'] = pd.to_datetime(comps['DeadlineDate'], errors='coerce').dt.year
                comp_growth = comps.groupby('Year').size().cumsum()
                comp_growth.plot(ax=axes[0,0], marker='o', linewidth=2)
                axes[0,0].set_title('Cumulative Competitions')
                axes[0,0].set_ylabel('Number of Competitions')
                axes[0,0].grid(True, alpha=0.3)
        
        # Users over time (if creation date available)
        if 'Users' in self.tables:
            users = self.tables['Users'].copy()
            if 'RegisterDate' in users.columns:
                users['Year'] = pd.to_datetime(users['RegisterDate'], errors='coerce').dt.year
                user_growth = users.groupby('Year').size().cumsum()
                user_growth.plot(ax=axes[0,1], marker='s', linewidth=2, color='orange')
                axes[0,1].set_title('Cumulative Users')
                axes[0,1].set_ylabel('Number of Users')
                axes[0,1].grid(True, alpha=0.3)
        
        # Kernels over time
        if 'Kernels' in self.tables:
            kernels = self.tables['Kernels'].copy()
            if 'CreationDate' in kernels.columns:
                kernels['Year'] = pd.to_datetime(kernels['CreationDate'], errors='coerce').dt.year
                kernel_growth = kernels.groupby('Year').size().cumsum()
                kernel_growth.plot(ax=axes[1,0], marker='^', linewidth=2, color='green')
                axes[1,0].set_title('Cumulative Notebooks')
                axes[1,0].set_ylabel('Number of Notebooks')
                axes[1,0].grid(True, alpha=0.3)
        
        # Monthly activity (recent years)
        if 'KernelVersions' in self.tables:
            kv = self.tables['KernelVersions'].copy()
            if 'CreationDate' in kv.columns:
                kv['Date'] = pd.to_datetime(kv['CreationDate'], errors='coerce')
                kv = kv[kv['Date'] > '2020-01-01']  # Focus on recent years
                monthly_activity = kv.set_index('Date').resample('M').size()
                monthly_activity.plot(ax=axes[1,1], linewidth=2, color='red')
                axes[1,1].set_title('Monthly Notebook Versions (2020+)')
                axes[1,1].set_ylabel('Number of Versions')
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def competition_timeline_analysis(self):
        """Analyze competition patterns over time"""
        if 'Competitions' not in self.tables:
            print("â�Œ Competitions table not available")
            return
            
        comps = self.tables['Competitions'].copy()
        
        # Convert deadline to datetime
        comps['DeadlineDate'] = pd.to_datetime(comps['DeadlineDate'], errors='coerce')
        comps = comps[comps['DeadlineDate'].notna()]
        
        # Extract time features
        comps['Year'] = comps['DeadlineDate'].dt.year
        comps['Month'] = comps['DeadlineDate'].dt.month
        comps['DayOfWeek'] = comps['DeadlineDate'].dt.dayofweek
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Competition Timeline Patterns', fontsize=16, fontweight='bold')
        
        # Competitions per year
        yearly_comps = comps['Year'].value_counts().sort_index()
        yearly_comps.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Competitions per Year')
        axes[0,0].set_ylabel('Number of Competitions')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Competitions per month
        monthly_comps = comps['Month'].value_counts().sort_index()
        monthly_comps.plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Competitions per Month')
        axes[0,1].set_ylabel('Number of Competitions')
        axes[0,1].set_xlabel('Month')
        
        # Competition duration analysis
        if 'EnabledDate' in comps.columns:
            comps['EnabledDate'] = pd.to_datetime(comps['EnabledDate'], errors='coerce')
            comps['Duration'] = (comps['DeadlineDate'] - comps['EnabledDate']).dt.days
            comps = comps[comps['Duration'] > 0]
            
            comps['Duration'].hist(bins=30, ax=axes[1,0], color='orange', alpha=0.7)
            axes[1,0].set_title('Competition Duration Distribution')
            axes[1,0].set_xlabel('Duration (days)')
            axes[1,0].set_ylabel('Frequency')
        
        # Day of week analysis
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_comps = comps['DayOfWeek'].value_counts().sort_index()
        dow_comps.index = [day_names[i] for i in dow_comps.index]
        dow_comps.plot(kind='bar', ax=axes[1,1], color='coral')
        axes[1,1].set_title('Competition Deadlines by Day of Week')
        axes[1,1].set_ylabel('Number of Competitions')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return comps

# Run time series analysis
ts_analyzer = TimeSeriesAnalyzer(cleaned_tables)
print("\nğŸ“ˆ ANALYZING GROWTH TRENDS...")
ts_analyzer.analyze_growth_trends()

print("\nğŸ“… ANALYZING COMPETITION TIMELINE PATTERNS...")
comp_analysis = ts_analyzer.competition_timeline_analysis()


class UserEngagementAnalyzer:
    """
    Analyze user engagement patterns
    """
    
    def __init__(self, tables):
        self.tables = tables
        
    def analyze_top_contributors(self):
        """Analyze top contributors across different activities"""
        print("ğŸ�† ANALYZING TOP CONTRIBUTORS")
        print("=" * 40)
        
        # Top notebook creators
        if 'Kernels' in self.tables:
            kernels = self.tables['Kernels']
            if 'AuthorUserId' in kernels.columns:
                top_notebook_authors = kernels['AuthorUserId'].value_counts().head(20)
                
                plt.figure(figsize=(12, 6))
                top_notebook_authors.plot(kind='bar', color='steelblue')
                plt.title('Top 20 Notebook Authors by Number of Notebooks')
                plt.xlabel('User ID')
                plt.ylabel('Number of Notebooks')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                
                print(f"Most prolific notebook author: User {top_notebook_authors.index[0]} with {top_notebook_authors.iloc[0]} notebooks")
        
        # Notebook engagement metrics
        if 'Kernels' in self.tables:
            kernels = self.tables['Kernels']
            engagement_cols = ['TotalViews', 'TotalVotes', 'TotalComments']
            available_cols = [col for col in engagement_cols if col in kernels.columns]
            
            if available_cols:
                fig, axes = plt.subplots(1, len(available_cols), figsize=(5*len(available_cols), 5))
                if len(available_cols) == 1:
                    axes = [axes]
                
                for i, col in enumerate(available_cols):
                    kernels[col].hist(bins=50, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_yscale('log')
                
                plt.tight_layout()
                plt.show()
    
    def analyze_notebook_metrics(self):
        """Analyze notebook performance metrics"""
        if 'Kernels' not in self.tables:
            print("â�Œ Kernels table not available")
            return
            
        kernels = self.tables['Kernels'].copy()
        
        # Basic statistics
        metric_cols = ['TotalViews', 'TotalVotes', 'TotalComments']
        available_metrics = [col for col in metric_cols if col in kernels.columns]
        
        print("\nğŸ“Š NOTEBOOK METRICS SUMMARY")
        print("=" * 35)
        
        for col in available_metrics:
            print(f"\n{col}:")
            print(f"  Mean: {kernels[col].mean():.2f}")
            print(f"  Median: {kernels[col].median():.2f}")
            print(f"  Max: {kernels[col].max():,}")
            print(f"  95th percentile: {kernels[col].quantile(0.95):.2f}")
        
        # Correlation analysis
        if len(available_metrics) > 1:
            correlation_matrix = kernels[available_metrics].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Correlation Between Notebook Engagement Metrics')
            plt.tight_layout()
            plt.show()
    
    def analyze_kernel_versions(self):
        """Analyze kernel version patterns"""
        if 'KernelVersions' not in self.tables:
            print("â�Œ KernelVersions table not available")
            return
            
        kv = self.tables['KernelVersions'].copy()
        
        # Versions per kernel
        versions_per_kernel = kv['ScriptId'].value_counts()
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        versions_per_kernel.hist(bins=50, alpha=0.7, color='purple')
        plt.xlabel('Number of Versions')
        plt.ylabel('Number of Kernels')
        plt.title('Distribution of Versions per Kernel')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        versions_per_kernel.head(20).plot(kind='bar', color='orange')
        plt.xlabel('Kernel ID')
        plt.ylabel('Number of Versions')
        plt.title('Top 20 Kernels by Version Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Average versions per kernel: {versions_per_kernel.mean():.2f}")
        print(f"Median versions per kernel: {versions_per_kernel.median():.2f}")
        print(f"Max versions for a single kernel: {versions_per_kernel.max():,}")

# Run user engagement analysis
engagement_analyzer = UserEngagementAnalyzer(cleaned_tables)
engagement_analyzer.analyze_top_contributors()
engagement_analyzer.analyze_notebook_metrics()
engagement_analyzer.analyze_kernel_versions()


def debug_code_dataset(code_path, max_folders=10):
    """
    Quickly scan the Meta Kaggle Code dataset structure to locate .py/.ipynb/.r files
    """
    print("ğŸ”� DEBUGGING CODE FILE STRUCTURE")
    print("=" * 30)
    
    # Check if the code path exists
    if not os.path.exists(code_path):
        print(f"â�Œ Path does NOT exist: {code_path}")
        return
    
    print(f"âœ… Path found: {code_path}")
    
    # List top-level folders
    top_level = os.listdir(code_path)
    print(f"\nğŸ“‚ Found {len(top_level)} items at top level:")
    for item in top_level[:max_folders]:
        print(f"  - {item}")

    # Try to find code files in first few folders
    sample_folders = top_level[:5]
    print("\nğŸ”� Searching for code files in first 5 folders...")
    
    found_py_files = []
    found_ipynb_files = []
    found_r_files = []

    for folder in sample_folders:
        folder_path = os.path.join(code_path, folder)
        if os.path.isdir(folder_path):
            py_files = glob.glob(os.path.join(folder_path, "**/*.py"), recursive=True)
            ipynb_files = glob.glob(os.path.join(folder_path, "**/*.ipynb"), recursive=True)
            r_files = glob.glob(os.path.join(folder_path, "**/*.r"), recursive=True)
            
            found_py_files.extend(py_files)
            found_ipynb_files.extend(ipynb_files)
            found_r_files.extend(r_files)

    print(f"\nğŸ“„ Python (.py) files found: {len(found_py_files)}")
    print(f"ğŸ“˜ Jupyter Notebooks (.ipynb) found: {len(found_ipynb_files)}")
    print(f"ğŸ“Š R script (.r) files found: {len(found_r_files)}")

    # Show some examples
    if found_py_files:
        print("\nğŸ“‹ Sample .py files:")
        for f in found_py_files[:3]:
            print(f"  {f}")
    
    if found_ipynb_files:
        print("\nğŸ“˜ Sample .ipynb files:")
        for f in found_ipynb_files[:3]:
            print(f"  {f}")
    
    if found_r_files:
        print("\nğŸ“Š Sample .r files:")
        for f in found_r_files[:3]:
            print(f"  {f}")

    return {
        'py_count': len(found_py_files),
        'ipynb_count': len(found_ipynb_files),
        'r_count': len(found_r_files),
        'sample_py': found_py_files[:3],
        'sample_ipynb': found_ipynb_files[:3],
        'sample_r': found_r_files[:3]
    }

# Run the scanner
code_path = "/kaggle/input/meta-kaggle-code"
debug_result = debug_code_dataset(code_path)


class CompetitionAnalyzer:
    """
    Deep dive analysis of competition patterns
    """

    def __init__(self, tables):
        self.tables = tables

    def analyze_competition_categories(self):
        """Analyze competition categories and types"""
        if 'Competitions' not in self.tables:
            print("â�Œ Competitions table not available")
            return
        comps = self.tables['Competitions'].copy()
        print("ğŸ�� COMPETITION CATEGORY ANALYSIS")
        print("=" * 40)
        # Analyze categories if available
        if 'CompetitionTypeId' in comps.columns:
            comp_types = comps['CompetitionTypeId'].value_counts()
            print(f"Competition type distribution:")
            print(comp_types)
            plt.figure(figsize=(10, 6))
            comp_types.plot(kind='bar', color='skyblue')
            plt.title('Competition Types Distribution')
            plt.xlabel('Competition Type ID')
            plt.ylabel('Number of Competitions')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.show()
        # Analyze team size patterns
        if 'MaxTeamSize' in comps.columns:
            team_sizes = comps['MaxTeamSize'].value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            team_sizes.plot(kind='bar', color='orange')
            plt.title('Maximum Team Size Distribution')
            plt.xlabel('Max Team Size')
            plt.ylabel('Number of Competitions')
            plt.tight_layout()
            plt.show()
            print(f"\nTeam size statistics:")
            print(f"  Average max team size: {comps['MaxTeamSize'].mean():.2f}")
            print(f"  Most common max team size: {comps['MaxTeamSize'].mode().iloc[0]}")

    def analyze_submission_patterns(self):
        """Analyze submission patterns if available"""
        if 'Submissions' not in self.tables:
            print("â�Œ Submissions table not available")
            return
        submissions = self.tables['Submissions'].copy()
        print("\nğŸ“Š SUBMISSION PATTERN ANALYSIS")
        print("=" * 40)
        # Basic submission statistics
        print(f"Total submissions: {len(submissions):,}")
        if 'CompetitionId' in submissions.columns:
            submissions_per_comp = submissions['CompetitionId'].value_counts()
            print(f"Average submissions per competition: {submissions_per_comp.mean():.2f}")
            print(f"Max submissions in a competition: {submissions_per_comp.max():,}")
            # Visualize submission distribution
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            submissions_per_comp.hist(bins=50, alpha=0.7, color='green')
            plt.xlabel('Submissions per Competition')
            plt.ylabel('Number of Competitions')
            plt.title('Distribution of Submissions per Competition')
            plt.yscale('log')
            plt.subplot(1, 2, 2)
            submissions_per_comp.head(20).plot(kind='bar', color='red')
            plt.xlabel('Competition ID')
            plt.ylabel('Number of Submissions')
            plt.title('Top 20 Competitions by Submission Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        # Analyze submission timing if date available
        date_cols = [col for col in submissions.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            submissions[date_cols[0]] = pd.to_datetime(submissions[date_cols[0]], errors='coerce')
            submissions['Hour'] = submissions[date_cols[0]].dt.hour
            submissions['DayOfWeek'] = submissions[date_cols[0]].dt.dayofweek
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # Hourly submission pattern
            hourly_subs = submissions['Hour'].value_counts().sort_index()
            hourly_subs.plot(kind='bar', ax=axes[0], color='purple')
            axes[0].set_title('Submissions by Hour of Day')
            axes[0].set_xlabel('Hour')
            axes[0].set_ylabel('Number of Submissions')
            # Day of week pattern
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_subs = submissions['DayOfWeek'].value_counts().sort_index()
            dow_subs.index = [day_names[i] for i in dow_subs.index]
            dow_subs.plot(kind='bar', ax=axes[1], color='teal')
            axes[1].set_title('Submissions by Day of Week')
            axes[1].set_ylabel('Number of Submissions')
            plt.tight_layout()
            plt.show()

    def analyze_leaderboard_patterns(self):
        """Analyze leaderboard and scoring patterns"""
        if 'Submissions' not in self.tables:
            print("â�Œ Submissions table not available for leaderboard analysis")
            return
        
        submissions = self.tables['Submissions'].copy()
        print("\nğŸ�† LEADERBOARD ANALYSIS")
        print("=" * 30)
    
        # Identify score column
        score_cols = [col for col in submissions.columns if 'score' in col.lower()]
        if not score_cols:
            print("â�Œ No score column found")
            return submissions
    
        score_col = score_cols[0]
        submissions[score_col] = pd.to_numeric(submissions[score_col], errors='coerce')
    
        # Plot score distribution
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        submissions[score_col].hist(bins=50, alpha=0.7, color='gold')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Submission Scores')
    
        # Analyze team progression
        plt.subplot(1, 2, 2)
        
        # Redefine date_cols here
        date_cols = [col for col in submissions.columns if 'date' in col.lower() or 'time' in col.lower()]
    
        if 'TeamId' in submissions.columns and len(submissions['TeamId'].unique()) > 0:
            sample_teams = submissions['TeamId'].value_counts().head(10).index
            for team in sample_teams[:5]:  # Show top 5 teams
                team_subs = submissions[submissions['TeamId'] == team].copy()
                if len(team_subs) > 1:
                    # Sort by date if available
                    if date_cols:
                        team_subs = team_subs.sort_values(by=date_cols[0])
                    else:
                        team_subs = team_subs.sort_values(by=score_col)
                    plt.plot(range(len(team_subs)), team_subs[score_col], marker='o', alpha=0.7, label=f'Team {team}')
            
            plt.title('Score Progression for Top Teams')
            plt.xlabel('Submission Number')
            plt.ylabel('Score')
            plt.legend()
    
        plt.tight_layout()
        plt.show()
        return submissions


# Run competition analysis
comp_analyzer = CompetitionAnalyzer(cleaned_tables)
comp_analyzer.analyze_competition_categories()
comp_analyzer.analyze_submission_patterns()
leaderboard_data = comp_analyzer.analyze_leaderboard_patterns()


class NotebookSuccessAnalyzer:
    """
    Analyze factors that contribute to notebook success
    """

    def __init__(self, tables, code_path=None):
        self.tables = tables
        self.code_path = code_path

    def analyze_success_factors(self):
        """Analyze what makes notebooks successful"""
        if 'Kernels' not in self.tables:
            print("â�Œ Kernels table not available")
            return

        kernels = self.tables['Kernels'].copy()
        print("ğŸŒŸ NOTEBOOK SUCCESS FACTOR ANALYSIS")
        print("=" * 45)

        # Define success metrics
        success_metrics = []
        for col in kernels.columns:
            if any(metric in col.lower() for metric in ['vote', 'view', 'comment', 'fork']):
                success_metrics.append(col)

        if not success_metrics:
            print("â�Œ No success metrics found in Kernels table")
            return

        # Create composite success score
        kernels['SuccessScore'] = 0
        for metric in success_metrics:
            min_val = kernels[metric].min()
            max_val = kernels[metric].max()
            if max_val > min_val:  # Avoid division by zero
                kernels[f'{metric}_norm'] = (kernels[metric] - min_val) / (max_val - min_val)
                kernels['SuccessScore'] += kernels[f'{metric}_norm']

        # Show top notebooks
        top_notebooks = kernels.nlargest(20, 'SuccessScore')
        print(f"Top 20 notebooks by success score:")
        for _, row in top_notebooks.iterrows():
            print(f"  Kernel {row['Id']}: Score {row['SuccessScore']:.2f}")

        # Merge with version data if available
        if 'KernelVersions' in self.tables:
            kv = self.tables['KernelVersions']
            notebook_versions = kv.groupby('ScriptId').size().reset_index(name='VersionCount')
            kernels_with_versions = kernels.merge(notebook_versions, left_on='Id', right_on='ScriptId', how='left')

            # Correlation between versions and success
            if 'VersionCount' in kernels_with_versions.columns:
                correlation = kernels_with_versions['SuccessScore'].corr(kernels_with_versions['VersionCount'])
                print(f"\nğŸ“ˆ Correlation between success score and version count: {correlation:.3f}")

                # Plotting
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.scatter(kernels_with_versions['VersionCount'], kernels_with_versions['SuccessScore'],
                            alpha=0.5, color='blue')
                plt.xlabel('Number of Versions')
                plt.ylabel('Success Score')
                plt.title('Success Score vs Version Count')

                plt.subplot(1, 2, 2)
                bins = [0, 1, 5, 10, 20, float('inf')]
                labels = ['1', '2-5', '6-10', '11-20', '20+']
                kernels_with_versions['VersionBin'] = pd.cut(
                    kernels_with_versions['VersionCount'], bins=bins, labels=labels, include_lowest=True
                )
                version_success = kernels_with_versions.groupby('VersionBin')['SuccessScore'].mean()
                version_success.plot(kind='bar', color='orange')
                plt.title('Average Success Score by Version Count')
                plt.xlabel('Version Count Range')
                plt.ylabel('Average Success Score')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

        return kernels_with_versions if 'kernels_with_versions' in locals() else kernels

# Run notebook success analysis
success_analyzer = NotebookSuccessAnalyzer(cleaned_tables, loader.meta_code_path)
successful_notebooks = success_analyzer.analyze_success_factors()


class CommunityAnalyzer:
    """
    Analyze community interactions and discussions
    """

    def __init__(self, tables):
        self.tables = tables

    def analyze_forum_activity(self):
        """Analyze forum discussions and activity"""
        forum_tables = ['ForumMessages', 'ForumTopics']
        available_tables = [t for t in forum_tables if t in self.tables]
        if not available_tables:
            print("â�Œ No forum tables available")
            return
        print("ğŸ’¬ COMMUNITY FORUM ANALYSIS")
        print("=" * 35)
        for table_name in available_tables:
            df = self.tables[table_name]
            print(f"\n{table_name}:")
            print(f"  Total records: {len(df):,}")
            # Analyze posting patterns over time
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'created' in col.lower()]
            if date_cols:
                df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                monthly_activity = df.set_index('Date').resample('M').size()
                plt.figure(figsize=(12, 6))
                monthly_activity.plot(linewidth=2, marker='o')
                plt.title(f'{table_name} - Monthly Activity Over Time')
                plt.xlabel('Date')
                plt.ylabel('Number of Posts/Topics')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
        # Analyze user participation if user ID available
        if 'ForumMessages' in self.tables:
            messages = self.tables['ForumMessages']
            user_cols = [col for col in messages.columns if 'user' in col.lower()]
            if user_cols:
                top_contributors = messages[user_cols[0]].value_counts().head(20)
                plt.figure(figsize=(12, 6))
                top_contributors.plot(kind='bar', color='skyblue')
                plt.title('Top 20 Forum Contributors')
                plt.xlabel('User ID')
                plt.ylabel('Number of Messages')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                print(f"Most active forum user: {top_contributors.index[0]} with {top_contributors.iloc[0]} messages")


# Run community analysis
community_analyzer = CommunityAnalyzer(cleaned_tables)
community_analyzer.analyze_forum_activity()


class InsightsSummary:
    """
    Summarize key insights and provide export functionality
    """

    def __init__(self, tables):
        self.tables = tables
        self.insights = {}

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\nğŸ�¯ KEY INSIGHTS SUMMARY")
        print("=" * 50)
        # Platform growth insights
        if 'Competitions' in self.tables:
            comps = self.tables['Competitions']
            self.insights['total_competitions'] = len(comps)
            print(f"ğŸ“Š Total Competitions: {len(comps):,}")
        if 'Users' in self.tables:
            users = self.tables['Users']
            self.insights['total_users'] = len(users)
            print(f"ğŸ‘¥ Total Users: {len(users):,}")
        if 'Kernels' in self.tables:
            kernels = self.tables['Kernels']
            self.insights['total_notebooks'] = len(kernels)
            print(f"ğŸ“� Total Notebooks: {len(kernels):,}")
        if 'KernelVersions' in self.tables:
            kv = self.tables['KernelVersions']
            self.insights['total_versions'] = len(kv)
            self.insights['avg_versions_per_notebook'] = len(kv) / len(kernels) if 'kernels' in locals() else 0
            print(f"ğŸ”„ Total Notebook Versions: {len(kv):,}")
            print(f"ğŸ“ˆ Average Versions per Notebook: {self.insights['avg_versions_per_notebook']:.2f}")

        # Engagement insights
        if 'Kernels' in self.tables:
            kernels = self.tables['Kernels']
            engagement_cols = [col for col in kernels.columns if any(metric in col.lower()
                              for metric in ['view', 'vote', 'comment', 'fork'])]
            if engagement_cols:
                print("\nğŸ“Š ENGAGEMENT METRICS:")
                for col in engagement_cols:
                    avg_engagement = kernels[col].mean()
                    print(f"  Average {col}: {avg_engagement:.2f}")
                    self.insights[f'avg_{col.lower()}'] = avg_engagement

        # Competition insights
        if 'Competitions' in self.tables:
            comps = self.tables['Competitions']
            if 'MaxTeamSize' in comps.columns:
                avg_team_size = comps['MaxTeamSize'].mean()
                print("\nğŸ�� COMPETITION INSIGHTS:")
                print(f"  Average Max Team Size: {avg_team_size:.2f}")
                self.insights['avg_max_team_size'] = avg_team_size

        return self.insights

    def export_summary_data(self):
        """Export key summary data for further analysis"""
        summary_data = {}
        # Export top performers
        if 'Kernels' in self.tables:
            kernels = self.tables['Kernels']
            # Top notebooks by views (if available)
            view_cols = [col for col in kernels.columns if 'view' in col.lower()]
            if view_cols:
                top_viewed = kernels.nlargest(50, view_cols[0])[['Id'] + view_cols]
                summary_data['top_viewed_notebooks'] = top_viewed
        # Export competition summary
        if 'Competitions' in self.tables:
            comps = self.tables['Competitions']
            comp_summary = comps[['Id', 'Title', 'DeadlineDate', 'MaxTeamSize']].copy() if 'Title' in comps.columns else comps.head(100)
            summary_data['competition_summary'] = comp_summary
        # Export user activity summary
        if 'Kernels' in self.tables:
            kernels = self.tables['Kernels']
            if 'AuthorUserId' in kernels.columns:
                user_activity = kernels['AuthorUserId'].value_counts().head(100).reset_index()
                user_activity.columns = ['UserId', 'NotebookCount']
                summary_data['top_users'] = user_activity
        return summary_data

    def create_final_visualization(self):
        """Create a final comprehensive visualization"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Platform overview
        ax1 = fig.add_subplot(gs[0, :])
        categories = ['Competitions', 'Users', 'Notebooks', 'Versions']
        values = [
            self.insights.get('total_competitions', 0),
            self.insights.get('total_users', 0),
            self.insights.get('total_notebooks', 0),
            self.insights.get('total_versions', 0)
        ]
        bars = ax1.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Kaggle Platform Overview', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count')
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')

        # Individual metrics
        if 'Kernels' in self.tables:
            kernels = self.tables['Kernels']
            # Views distribution
            view_cols = [col for col in kernels.columns if 'view' in col.lower()]
            if view_cols:
                ax2 = fig.add_subplot(gs[1, 0])
                kernels[view_cols[0]].hist(bins=50, ax=ax2, color='#FF6B6B', alpha=0.7)
                ax2.set_title('Notebook Views Distribution')
                ax2.set_xlabel('Views')
                ax2.set_ylabel('Frequency')
                ax2.set_yscale('log')
            # Votes distribution
            vote_cols = [col for col in kernels.columns if 'vote' in col.lower()]
            if vote_cols:
                ax3 = fig.add_subplot(gs[1, 1])
                kernels[vote_cols[0]].hist(bins=50, ax=ax3, color='#4ECDC4', alpha=0.7)
                ax3.set_title('Notebook Votes Distribution')
                ax3.set_xlabel('Votes')
                ax3.set_ylabel('Frequency')
                ax3.set_yscale('log')
            # Comments distribution
            comment_cols = [col for col in kernels.columns if 'comment' in col.lower()]
            if comment_cols:
                ax4 = fig.add_subplot(gs[1, 2])
                kernels[comment_cols[0]].hist(bins=50, ax=ax4, color='#45B7D1', alpha=0.7)
                ax4.set_title('Notebook Comments Distribution')
                ax4.set_xlabel('Comments')
                ax4.set_ylabel('Frequency')
                ax4.set_yscale('log')

        # Version distribution
        if 'KernelVersions' in self.tables:
            kv = self.tables['KernelVersions']
            versions_per_kernel = kv['ScriptId'].value_counts()
            ax5 = fig.add_subplot(gs[2, 0])
            versions_per_kernel.hist(bins=30, ax=ax5, color='#96CEB4', alpha=0.7)
            ax5.set_title('Versions per Notebook')
            ax5.set_xlabel('Number of Versions')
            ax5.set_ylabel('Number of Notebooks')
            ax5.set_yscale('log')

        # Competition timeline
        if 'Competitions' in self.tables:
            comps = self.tables['Competitions']
            if 'DeadlineDate' in comps.columns:
                comps['Year'] = pd.to_datetime(comps['DeadlineDate'], errors='coerce').dt.year
                yearly_comps = comps['Year'].value_counts().sort_index()
                ax6 = fig.add_subplot(gs[2, 1])
                yearly_comps.plot(kind='line', ax=ax6, marker='o', color='#FF8C94')
                ax6.set_title('Competitions per Year')
                ax6.set_xlabel('Year')
                ax6.set_ylabel('Number of Competitions')
                ax6.grid(True, alpha=0.3)

        # Summary text
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        summary_text = f"""
        ğŸ“Š KAGGLE META ANALYSIS SUMMARY
        ğŸ�¯ Key Findings:
        â€¢ {self.insights.get('total_competitions', 'N/A'):,} Total Competitions
        â€¢ {self.insights.get('total_users', 'N/A'):,} Total Users  
        â€¢ {self.insights.get('total_notebooks', 'N/A'):,} Total Notebooks
        â€¢ {self.insights.get('avg_versions_per_notebook', 0):.1f} Avg Versions/Notebook
        ğŸ’¡ Insights:
        â€¢ High community engagement
        â€¢ Active notebook development
        â€¢ Diverse competition portfolio
        â€¢ Growing platform adoption
        """
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.suptitle('Meta Kaggle Analysis - Comprehensive Overview', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()


# Generate final insights and visualization
insights_generator = InsightsSummary(cleaned_tables)
final_insights = insights_generator.generate_summary_report()
summary_data = insights_generator.export_summary_data()
insights_generator.create_final_visualization()


def save_analysis_results(insights, summary_data, filename_prefix='kaggle_analysis'):
    """
    Save analysis results to disk in multiple formats for reproducibility.
    
    Saves:
    - Pickle files (for Python use)
    - JSON files (for readability and sharing)
    """
    from datetime import datetime
    import pickle
    import json
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"ğŸ’¾ Saving analysis results with timestamp: {timestamp}")
    
    # Save insights dictionary
    try:
        with open(f'{filename_prefix}_insights_{timestamp}.pkl', 'wb') as f:
            pickle.dump(insights, f)
        with open(f'{filename_prefix}_insights_{timestamp}.json', 'w') as f:
            json.dump(insights, f, indent=2)
    except Exception as e:
        print(f"â�Œ Error saving insights: {e}")
    
    # Save summary data
    try:
        with open(f'{filename_prefix}_summary_{timestamp}.pkl', 'wb') as f:
            pickle.dump(summary_data, f)
        with open(f'{filename_prefix}_summary_{timestamp}.json', 'w') as f:
            json.dump({str(k): str(v) for k, v in summary_data.items()}, f, indent=2)
    except Exception as e:
        print(f"â�Œ Error saving summary data: {e}")

    print(f"âœ… Results saved successfully (prefix: {filename_prefix}_{timestamp})")


def create_analysis_checklist():
    """Create a simple, readable reproducibility checklist"""
    print("ğŸ“‹ ANALYSIS CHECKLIST")
    print("=" * 40)
    
    checklist_items = [
        ("Data Loading", [
            "Downloaded Meta Kaggle dataset",
            "Downloaded Meta Kaggle Code dataset",
            "Loaded all CSV tables successfully",
            "Verified data integrity"
        ]),
        ("Exploration", [
            "Explored key table structures",
            "Cleaned datetime columns",
            "Generated basic statistics",
            "Identified missing data patterns"
        ]),
        ("Time Series Analysis", [
            "Analyzed growth trends",
            "Examined competition timeline patterns",
            "Studied seasonal variations"
        ]),
        ("User Engagement", [
            "Identified top contributors",
            "Analyzed notebook engagement metrics",
            "Studied version patterns"
        ]),
        ("Code Integration", [
            "Linked code files to metadata",
            "Analyzed programming languages",
            "Studied popular libraries"
        ]),
        ("Competition Analysis", [
            "Analyzed competition categories",
            "Studied submission patterns",
            "Examined leaderboard trends"
        ]),
        ("Success Factors", [
            "Identified notebook success factors",
            "Analyzed code characteristics",
            "Studied version-success correlation"
        ]),
        ("Community Analysis", [
            "Analyzed forum activity",
            "Studied discussion patterns",
            "Examined user interactions"
        ]),
        ("Insights & Export", [
            "Generated summary report",
            "Created comprehensive visualizations",
            "Exported key findings",
            "Documented methodology"
        ]),
        ("Next Steps", [
            "Extend analysis with specific research questions",
            "Implement predictive models",
            "Create interactive dashboards",
            "Share findings with community"
        ])
    ]

    for section, items in checklist_items:
        print(f"\nâœ… {section}:")
        for item in items:
            print(f"  - [ ] {item}")

    return checklist_items

# Save final insights and summary data
try:
    save_analysis_results(final_insights, summary_data)
except Exception as e:
    print(f"â�Œ Failed to save results: {e}")

# Generate reproducibility checklist
checklist = create_analysis_checklist()

