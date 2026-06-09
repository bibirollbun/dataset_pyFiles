#1.1 Dataset Overview

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# --- Plotting Configuration ---
# Set the visual style for the plots
sns.set(style="whitegrid")

# Set font to display Chinese characters correctly in plots. 
# 'SimHei' is a commonly available font in Kaggle environments.
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'sans-serif']

# Ensure that minus signs are displayed correctly
plt.rcParams['axes.unicode_minus'] = False  

# --- Load the Data ---
# Define the correct path to the CSV file within the Kaggle environment
file_path = '/kaggle/input/h690/h690/jd_sherds_info.csv'

try:
    # Load the dataset into a pandas DataFrame
    df_info = pd.read_csv(file_path)
    
    
    # Display beautiful data preview
    print("ğŸ“Š DATASET PREVIEW - First 5 Rows")
    print("=" * 60)
    
    # Create a styled table for the first 5 rows
    styled_head = df_info.head().style.set_properties(**{
        'background-color': '#f8f9fa',
        'color': '#212529',
        'border': '1px solid #dee2e6',
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#007bff'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('border', '1px solid #0056b3')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #dee2e6'),
            ('padding', '8px')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin', '10px auto'),
            ('width', '100%')
        ]}
    ])
    
    display(styled_head)
    
    print("\n" + "=" * 60)
    print("ğŸ“‹ DATASET INFORMATION SUMMARY")
    print("=" * 60)
    
    # Create a beautiful info summary
    info_data = {
        'Metric': ['Total Rows', 'Total Columns', 'Memory Usage', 'Data Types'],
        'Value': [
            f"{len(df_info):,}",
            f"{len(df_info.columns)}",
            f"{df_info.memory_usage(deep=True).sum() / 1024:.2f} KB",
            f"{len(df_info.dtypes.unique())} unique types"
        ]
    }
    
    info_df = pd.DataFrame(info_data)
    
    styled_info = info_df.style.set_properties(**{
        'background-color': '#f8f9fa',
        'color': '#495057',
        'border': '1px solid #dee2e6',
        'text-align': 'center',
        'font-size': '14px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#28a745'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('border', '1px solid #1e7e34')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #dee2e6'),
            ('padding', '10px'),
            ('font-weight', '500')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin', '10px auto'),
            ('width', '60%')
        ]}
    ])
    
    display(styled_info)
    
    # Column information table
    print("\n" + "=" * 60)
    print("ğŸ”� COLUMN DETAILS")
    print("=" * 60)
    
    column_info = pd.DataFrame({
        'Data Type': df_info.dtypes.astype(str),
        'Non-Null Count': df_info.count(),
        'Null Count': df_info.isnull().sum(),
        'Null Percentage': (df_info.isnull().sum() / len(df_info) * 100).round(2)
    })
    
    styled_columns = column_info.style.set_properties(**{
        'background-color': '#fff3cd',
        'color': '#856404',
        'border': '1px solid #ffeaa7',
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#ffc107'),
            ('color', '#212529'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('border', '1px solid #e0a800')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #ffeaa7'),
            ('padding', '8px')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin', '10px auto'),
            ('width', '100%')
        ]}
    ]).highlight_null(color='#f8d7da')
    
    display(styled_columns)
    
    
except FileNotFoundError:
    print("â�Œ Error: Could not find the file at '{}'".format(file_path))
    print("ğŸ’¡ Please double-check the input data path in your Kaggle Notebook.")
    # In case of an error, create an empty DataFrame to prevent subsequent code from crashing.
    df_info = pd.DataFrame()
    
    # Display error message in a styled format
    error_html = """
    <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border: 1px solid #f5c6cb; border-radius: 5px; margin: 10px 0;">
        <h3 style="margin-top: 0;">âš ï¸� File Not Found Error</h3>
        <p><strong>Path:</strong> {}</p>
        <p><strong>Solution:</strong> Please verify the file path and ensure the dataset is uploaded correctly.</p>
    </div>
    """.format(file_path)
    
    display(HTML(error_html))


#Data Integrity Check

import pandas as pd
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

def data_integrity_check(df):
    """
    Performs data integrity checks with simple, consistent formatting
    """
    if df.empty:
        print("â�Œ No data available for integrity check.")
        return
    
    
    # Check 1: Missing Values Analysis
    print(f"ğŸ“Š Missing Values Analysis:")
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    if total_missing == 0:
        print("   â€¢ No missing values found in the dataset")
    else:
        print(f"   â€¢ Found missing values in {(missing_values > 0).sum()} columns:")
        for col, count in missing_values[missing_values > 0].items():
            percentage = (count / len(df)) * 100
            print(f"     - {col}: {count:,} ({percentage:.1f}%)")
    
    # Check 2: Sherd Side Completeness
    if 'sherd_id' in df.columns and 'image_side' in df.columns:
        sherd_sides = df.groupby('sherd_id')['image_side'].nunique()
        incomplete_sherds = sherd_sides[sherd_sides < 2]
        
        print(f"\nğŸ“Š Sherd Side Completeness:")
        if len(incomplete_sherds) == 0:
            print("   â€¢ All sherd_ids have records for both interior and exterior sides")
        else:
            print(f"   â€¢ Warning: Found {len(incomplete_sherds):,} sherd_ids missing one side")
            print("   â€¢ Examples of incomplete sherds:")
            for sherd_id in incomplete_sherds.head(5).index:
                available_sides = df[df['sherd_id'] == sherd_id]['image_side'].unique()
                print(f"     - {sherd_id}: only has {', '.join(available_sides)}")
    
    # Check 3: Duplicate Records Check
    if 'sherd_id' in df.columns and 'image_side' in df.columns:
        side_counts = df.groupby('sherd_id')['image_side'].count()
        multiple_entries = side_counts[side_counts > 2]
        
        print(f"\nğŸ“Š Duplicate Records Check:")
        if len(multiple_entries) == 0:
            print("   â€¢ Each sherd_id has at most two records (one interior, one exterior)")
        else:
            print(f"   â€¢ Warning: Found {len(multiple_entries):,} sherd_ids with more than two records")
            print("   â€¢ Examples of sherds with multiple records:")
            for sherd_id, count in multiple_entries.head(5).items():
                print(f"     - {sherd_id}: {count} records")
    

# Execute the simple data integrity check
if not df_info.empty:
    data_integrity_check(df_info)
else:
    print("â�Œ No data available for integrity check.")


# Vessel Types Distribution Analysis

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import warnings
from IPython.display import display, HTML
warnings.filterwarnings('ignore')

def analyze_vessel_types(df):
    """
    Analyze vessel type distribution
    """
    if df.empty or 'type' not in df.columns:
        print("â�Œ DataFrame is empty or 'type' column not found.")
        return
    
    
    # Calculate frequency and percentage
    counts = df['type'].value_counts()
    percentages = df['type'].value_counts(normalize=True) * 100
    
    # Create a beautiful summary table
    dist_summary = pd.DataFrame({
        'Category': counts.index,
        'Frequency': counts.values,
        'Percentage (%)': percentages.round(2).values,
        'Cumulative %': percentages.cumsum().round(2).values
    })
    
    # Style the summary table
    styled_summary = dist_summary.style.set_properties(**{
        'background-color': '#f8f9fa',
        'color': '#495057',
        'border': '1px solid #dee2e6',
        'text-align': 'center',
        'font-size': '12px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#17a2b8'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('border', '1px solid #138496'),
            ('padding', '8px')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #dee2e6'),
            ('padding', '6px'),
            ('font-weight', '500')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin', '10px auto'),
            ('width', '90%'),
            ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
        ]}
    ]).format({'Percentage (%)': '{:.2f}%', 'Cumulative %': '{:.2f}%'})
    
    display(styled_summary)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Custom color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#FF9999', '#87CEEB', '#FFB6C1', '#98FB98', '#F0E68C', '#DDA0DD', '#B0E0E6', '#FFA07A', '#20B2AA', '#87CEFA', '#9370DB', '#3CB371']
    
    # Create gradient effect for bars
    bars = ax.barh(range(len(counts)), counts.values, 
                   color=colors[:len(counts)], 
                   height=0.7,
                   edgecolor='white', 
                   linewidth=1.5)
    
    # Set labels and ticks
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=10, fontweight='600')
    ax.set_xlabel('Frequency Count', fontsize=12, fontweight='bold')
    ax.set_title('Vessel Type Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Add value annotations on bars with improved styling
    for i, (bar, count, pct) in enumerate(zip(bars, counts.values, percentages.values)):
        width = bar.get_width()
        # Position text inside bar if bar is long enough, otherwise outside
        if width > max(counts) * 0.1:
            ax.text(width * 0.95, bar.get_y() + bar.get_height()/2, 
                    f'{count:,} ({pct:.1f}%)', 
                    ha='right', va='center', fontweight='bold', fontsize=9,
                    color='white', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        else:
            ax.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{count:,} ({pct:.1f}%)', 
                    ha='left', va='center', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Add subtle background color
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    total_count = len(df)
    unique_categories = len(counts)
    most_common = counts.index[0]
    most_common_pct = percentages.iloc[0]
    
    print(f"ğŸ”� KEY INSIGHTS:")
    print(f"   â€¢ Most frequent category: '{most_common}' ({most_common_pct:.1f}%)")
    print(f"   â€¢ Total unique categories: {unique_categories}")
    print(f"   â€¢ Data coverage: {total_count:,} records analyzed")

# Execute analysis
if not df_info.empty:
    analyze_vessel_types(df_info)
else:
    print("â�Œ No data available for analysis. Please check your data loading process.")


# Parts Distribution Analysis

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import warnings
from IPython.display import display, HTML
warnings.filterwarnings('ignore')

def analyze_parts(df):
    """
    Analyze part distribution
    """
    if df.empty or 'part' not in df.columns:
        print("â�Œ DataFrame is empty or 'part' column not found.")
        return

    
    # Calculate frequency and percentage
    counts = df['part'].value_counts()
    percentages = df['part'].value_counts(normalize=True) * 100
    
    # Create a beautiful summary table
    dist_summary = pd.DataFrame({
        'Category': counts.index,
        'Frequency': counts.values,
        'Percentage (%)': percentages.round(2).values,
        'Cumulative %': percentages.cumsum().round(2).values
    })
    
    # Style the summary table
    styled_summary = dist_summary.style.set_properties(**{
        'background-color': '#f8f9fa',
        'color': '#495057',
        'border': '1px solid #dee2e6',
        'text-align': 'center',
        'font-size': '12px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#28a745'),  # Green theme for parts
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('border', '1px solid #1e7e34'),
            ('padding', '8px')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #dee2e6'),
            ('padding', '6px'),
            ('font-weight', '500')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin', '10px auto'),
            ('width', '90%'),
            ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
        ]}
    ]).format({'Percentage (%)': '{:.2f}%', 'Cumulative %': '{:.2f}%'})
    
    display(styled_summary)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Custom color palette - different from vessel types
    colors = ['#28a745', '#20c997', '#17a2b8', '#6f42c1', '#e83e8c', '#fd7e14', '#ffc107', '#6c757d', '#dc3545', '#007bff', '#198754', '#0dcaf0', '#6610f2', '#d63384', '#fd7e14', '#ffca2c', '#6c757d', '#dc3545', '#0d6efd', '#198754']
    
    # Create gradient effect for bars
    bars = ax.barh(range(len(counts)), counts.values, 
                   color=colors[:len(counts)], 
                   height=0.7,
                   edgecolor='white', 
                   linewidth=1.5)
    
    # Set labels and ticks
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=10, fontweight='600')
    ax.set_xlabel('Frequency Count', fontsize=12, fontweight='bold')
    ax.set_title('Part Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Add value annotations on bars with improved styling
    for i, (bar, count, pct) in enumerate(zip(bars, counts.values, percentages.values)):
        width = bar.get_width()
        # Position text inside bar if bar is long enough, otherwise outside
        if width > max(counts) * 0.1:
            ax.text(width * 0.95, bar.get_y() + bar.get_height()/2, 
                    f'{count:,} ({pct:.1f}%)', 
                    ha='right', va='center', fontweight='bold', fontsize=9,
                    color='white', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        else:
            ax.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{count:,} ({pct:.1f}%)', 
                    ha='left', va='center', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Add subtle background color
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    total_count = len(df)
    unique_categories = len(counts)
    most_common = counts.index[0]
    most_common_pct = percentages.iloc[0]
    
    print(f"ğŸ”� KEY INSIGHTS:")
    print(f"   â€¢ Most frequent category: '{most_common}' ({most_common_pct:.1f}%)")
    print(f"   â€¢ Total unique categories: {unique_categories}")
    print(f"   â€¢ Data coverage: {total_count:,} records analyzed")

# Execute analysis
if not df_info.empty:
    analyze_parts(df_info)
else:
    print("â�Œ No data available for analysis. Please check your data loading process.")


# Units Distribution Analysis

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import warnings
from IPython.display import display, HTML
warnings.filterwarnings('ignore')

def analyze_units(df):
    """
    Analyze unit distribution
    """
    if df.empty or 'unit' not in df.columns:
        print("â�Œ DataFrame is empty or 'unit' column not found.")
        return
    
    
    # Calculate frequency and percentage
    counts = df['unit'].value_counts()
    percentages = df['unit'].value_counts(normalize=True) * 100
    
    # Create a beautiful summary table
    dist_summary = pd.DataFrame({
        'Category': counts.index,
        'Frequency': counts.values,
        'Percentage (%)': percentages.round(2).values,
        'Cumulative %': percentages.cumsum().round(2).values
    })
    
    # Style the summary table
    styled_summary = dist_summary.style.set_properties(**{
        'background-color': '#f8f9fa',
        'color': '#495057',
        'border': '1px solid #dee2e6',
        'text-align': 'center',
        'font-size': '12px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#6f42c1'),  # Purple theme for units
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('border', '1px solid #59359a'),
            ('padding', '8px')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #dee2e6'),
            ('padding', '6px'),
            ('font-weight', '500')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin', '10px auto'),
            ('width', '90%'),
            ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
        ]}
    ]).format({'Percentage (%)': '{:.2f}%', 'Cumulative %': '{:.2f}%'})
    
    display(styled_summary)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Custom color palette - different from previous analyses
    colors = ['#6f42c1', '#e83e8c', '#fd7e14', '#20c997', '#17a2b8', '#ffc107', '#dc3545', '#198754', '#0dcaf0', '#6610f2', '#d63384', '#fd7e14', '#ffca2c', '#6c757d', '#0d6efd', '#198754', '#20c997', '#6f42c1', '#e83e8c', '#fd7e14']
    
    # Create gradient effect for bars
    bars = ax.barh(range(len(counts)), counts.values, 
                   color=colors[:len(counts)], 
                   height=0.7,
                   edgecolor='white', 
                   linewidth=1.5)
    
    # Set labels and ticks
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=10, fontweight='600')
    ax.set_xlabel('Frequency Count', fontsize=12, fontweight='bold')
    ax.set_title('Unit Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Add value annotations on bars with improved styling
    for i, (bar, count, pct) in enumerate(zip(bars, counts.values, percentages.values)):
        width = bar.get_width()
        # Position text inside bar if bar is long enough, otherwise outside
        if width > max(counts) * 0.1:
            ax.text(width * 0.95, bar.get_y() + bar.get_height()/2, 
                    f'{count:,} ({pct:.1f}%)', 
                    ha='right', va='center', fontweight='bold', fontsize=9,
                    color='white', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        else:
            ax.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{count:,} ({pct:.1f}%)', 
                    ha='left', va='center', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Add subtle background color
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    total_count = len(df)
    unique_categories = len(counts)
    most_common = counts.index[0]
    most_common_pct = percentages.iloc[0]
    
    print(f"ğŸ”� KEY INSIGHTS:")
    print(f"   â€¢ Most frequent category: '{most_common}' ({most_common_pct:.1f}%)")
    print(f"   â€¢ Total unique categories: {unique_categories}")
    print(f"   â€¢ Data coverage: {total_count:,} records analyzed")

# Execute analysis
if not df_info.empty:
    analyze_units(df_info)
else:
    print("â�Œ No data available for analysis. Please check your data loading process.")


# Vessel Type Ã— Part Composition Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

def analyze_vessel_part_composition(df, min_frequency=1):
    """
    Analyze vessel type Ã— part composition relationships
    """
    if df.empty:
        print("â�Œ DataFrame is empty.")
        return
    
    print("VESSEL TYPE Ã— PART COMPOSITION ANALYSIS")
    
    # Filter low-frequency categories
    type_counts = df['type'].value_counts()
    frequent_types = type_counts[type_counts >= min_frequency].index
    df_filtered = df[df['type'].isin(frequent_types)]
    
    print(f"ğŸ“Š Analysis Info:")
    print(f"   â€¢ Vessel types analyzed: {len(frequent_types)} (â‰¥{min_frequency} occurrences)")
    print(f"   â€¢ Records included: {len(df_filtered):,} out of {len(df):,} ({len(df_filtered)/len(df)*100:.1f}%)")
    
    # Create cross-tabulation
    crosstab = pd.crosstab(df_filtered['type'], df_filtered['part'])
    
    # Sort by total frequency (most common vessels first)
    vessel_totals = crosstab.sum(axis=1).sort_values(ascending=True)
    crosstab_sorted = crosstab.loc[vessel_totals.index]
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot: Absolute counts (stacked horizontal bar)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', 
             '#98D8C8', '#F7DC6F', '#FF9999', '#87CEEB', '#FFB6C1', '#98FB98']
    
    crosstab_sorted.plot(kind='barh', stacked=True, ax=ax, 
                        color=colors[:len(crosstab_sorted.columns)], 
                        width=0.8)
    
    ax.set_title('Vessel Type Ã— Part Composition', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Fragments', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vessel Type', fontsize=12, fontweight='bold')
    ax.legend(title='Part', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_facecolor('#fafafa')
    
    # Enhanced styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Add total labels
    for i, (idx, row) in enumerate(crosstab_sorted.iterrows()):
        total = row.sum()
        ax.text(total + max(crosstab_sorted.sum(axis=1)) * 0.01, i, 
                f'{total:,}', va='center', ha='left', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics table
    print(f"\nğŸ“‹ COMPOSITION SUMMARY:")
    
    # Create summary with key metrics
    summary_data = []
    for vessel_type in crosstab_sorted.index:
        row_data = crosstab_sorted.loc[vessel_type]
        total_fragments = row_data.sum()
        most_common_part = row_data.idxmax()
        most_common_pct = (row_data.max() / total_fragments) * 100
        part_diversity = (row_data > 0).sum()
        
        summary_data.append({
            'Vessel Type': vessel_type,
            'Total Fragments': total_fragments,
            'Most Common Part': most_common_part,
            'Dominance (%)': f"{most_common_pct:.1f}%",
            'Part Diversity': part_diversity
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Style the summary table
    styled_summary = summary_df.style.set_properties(**{
        'background-color': '#f8f9fa',
        'color': '#495057',
        'border': '1px solid #dee2e6',
        'text-align': 'center',
        'font-size': '11px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#dc3545'),  # Red theme for vessel-part analysis
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('border', '1px solid #c82333'),
            ('padding', '8px')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #dee2e6'),
            ('padding', '6px'),
            ('font-weight', '500')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin', '10px auto'),
            ('width', '90%'),
            ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
        ]}
    ])
    
    display(styled_summary)
    
    # Key insights
    print(f"\nğŸ”� KEY INSIGHTS:")
    if len(summary_df) > 0:
        most_fragments = summary_df.loc[summary_df['Total Fragments'].idxmax()]
        most_diverse = summary_df.loc[summary_df['Part Diversity'].idxmax()]
        
        print(f"   â€¢ Most fragmented vessel: '{most_fragments['Vessel Type']}' ({most_fragments['Total Fragments']:,} fragments)")
        print(f"   â€¢ Most diverse parts: '{most_diverse['Vessel Type']}' ({most_diverse['Part Diversity']} different parts)")
        print(f"   â€¢ Total vessel-part combinations: {len(summary_df)} vessel types analyzed")
        
        # Additional insights
        avg_diversity = summary_df['Part Diversity'].mean()
        total_fragments = summary_df['Total Fragments'].sum()
        print(f"   â€¢ Average part diversity per vessel: {avg_diversity:.1f} parts")
        print(f"   â€¢ Total fragments in analysis: {total_fragments:,}")

# Execute analysis
if not df_info.empty:
    analyze_vessel_part_composition(df_info, min_frequency=1)
else:
    print("â�Œ No data available for analysis. Please check your data loading process.")


# Excavation Unit Ã— Vessel Type Distribution Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

def analyze_unit_vessel_distribution(df, top_n_units=15, min_frequency=1):
    """
    Analyze excavation unit Ã— vessel type distribution relationships
    """
    if df.empty:
        print("â�Œ DataFrame is empty.")
        return
    
    print("EXCAVATION UNIT Ã— VESSEL TYPE DISTRIBUTION")
    
    # Filter to most active units and frequent vessel types
    unit_counts = df['unit'].value_counts()
    type_counts = df['type'].value_counts()
    
    top_units = unit_counts.head(top_n_units).index
    frequent_types = type_counts[type_counts >= min_frequency].index
    
    df_filtered = df[df['unit'].isin(top_units) & df['type'].isin(frequent_types)]
    
    print(f"ğŸ“Š Analysis Info:")
    print(f"   â€¢ Top excavation units: {len(top_units)} out of {df['unit'].nunique()}")
    print(f"   â€¢ Vessel types included: {len(frequent_types)} (â‰¥{min_frequency} occurrences)")
    print(f"   â€¢ Records analyzed: {len(df_filtered):,} out of {len(df):,} ({len(df_filtered)/len(df)*100:.1f}%)")
    
    # Create cross-tabulation
    crosstab = pd.crosstab(df_filtered['unit'], df_filtered['type'])
    
    # Sort units by total activity and types by frequency
    unit_totals = crosstab.sum(axis=1).sort_values(ascending=False)
    type_totals = crosstab.sum(axis=0).sort_values(ascending=False)
    
    crosstab_sorted = crosstab.loc[unit_totals.index, type_totals.index]
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot: Absolute counts heatmap
    sns.heatmap(crosstab_sorted, annot=True, fmt='d', cmap='Blues', 
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Number of Fragments', 'shrink': 0.8})
    
    ax.set_title('Unit Ã— Vessel Type Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Vessel Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Excavation Unit', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Unit specialization analysis
    print(f"\nğŸ“ˆ UNIT SPECIALIZATION ANALYSIS:")
    
    # Calculate percentage by unit (row percentages)
    crosstab_pct = crosstab_sorted.div(crosstab_sorted.sum(axis=1), axis=0) * 100
    
    # Find units with highest concentration in specific vessel types
    specialization_data = []
    for unit in crosstab_sorted.index:
        unit_row = crosstab_pct.loc[unit]
        if unit_row.sum() > 0:  # Avoid empty units
            max_type = unit_row.idxmax()
            max_pct = unit_row.max()
            total_fragments = crosstab_sorted.loc[unit].sum()
            diversity = (crosstab_sorted.loc[unit] > 0).sum()
            
            specialization_data.append({
                'Unit': unit,
                'Specialization': max_type,
                'Concentration (%)': f"{max_pct:.1f}%",
                'Total Fragments': total_fragments,
                'Type Diversity': diversity
            })
    
    spec_df = pd.DataFrame(specialization_data)
    spec_df = spec_df.sort_values('Total Fragments', ascending=False)
    
    # Style the specialization table
    styled_spec = spec_df.style.set_properties(**{
        'background-color': '#f8f9fa',
        'color': '#495057',
        'border': '1px solid #dee2e6',
        'text-align': 'center',
        'font-size': '11px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#007bff'),  # Blue theme for unit-vessel analysis
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('border', '1px solid #0056b3'),
            ('padding', '8px')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #dee2e6'),
            ('padding', '6px'),
            ('font-weight', '500')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('margin', '10px auto'),
            ('width', '90%'),
            ('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
        ]}
    ])
    
    display(styled_spec)
    
    # Additional analysis: Create a summary of unit patterns
    print(f"\nğŸ“Š EXCAVATION PATTERNS:")
    
    # Overall statistics
    total_units_analyzed = len(spec_df)
    avg_fragments_per_unit = spec_df['Total Fragments'].mean()
    avg_diversity_per_unit = spec_df['Type Diversity'].mean()
    
    # Concentration analysis
    high_concentration_units = spec_df[
        spec_df['Concentration (%)'].str.rstrip('%').astype(float) >= 50
    ]
    
    print(f"   â€¢ Units with high specialization (â‰¥50%): {len(high_concentration_units)}/{total_units_analyzed}")
    print(f"   â€¢ Average fragments per unit: {avg_fragments_per_unit:.1f}")
    print(f"   â€¢ Average vessel type diversity: {avg_diversity_per_unit:.1f} types per unit")
    
    # Key insights
    print(f"\nğŸ”� KEY INSIGHTS:")
    if len(spec_df) > 0:
        most_active = spec_df.iloc[0]
        most_specialized = spec_df.loc[spec_df['Concentration (%)'].str.rstrip('%').astype(float).idxmax()]
        most_diverse = spec_df.loc[spec_df['Type Diversity'].idxmax()]
        
        print(f"   â€¢ Most active unit: '{most_active['Unit']}' ({most_active['Total Fragments']} fragments)")
        print(f"   â€¢ Most specialized: '{most_specialized['Unit']}' ({most_specialized['Concentration (%)']} {most_specialized['Specialization']})")
        print(f"   â€¢ Most diverse unit: '{most_diverse['Unit']}' ({most_diverse['Type Diversity']} different types)")
        
        # Pattern insights
        if len(high_concentration_units) > 0:
            print(f"   â€¢ Specialization pattern: {len(high_concentration_units)} units show strong preference for specific vessel types")
        else:
            print(f"   â€¢ Distribution pattern: Units show relatively balanced vessel type distributions")

# Execute analysis
if not df_info.empty:
    analyze_unit_vessel_distribution(df_info, top_n_units=15, min_frequency=1)
else:
    print("â�Œ No data available for analysis. Please check your data loading process.")


#Automated detection of targets

import cv2
import numpy as np
import matplotlib.pyplot as plt
# --- Config ---
# List of image paths to process
image_paths = [
   '/kaggle/input/h690/h690/sherd_images/JD00001_exterior.jpg',
   '/kaggle/input/h690/h690/sherd_images/JD00002_interior.jpg'
]
# Define the rectangular region covering the scale bar and color checker
# Format: (top-left x, top-left y, width, height)
calibration_tools_mask_area = (260, 730, 480, 200)
# --- Loop over each image ---
for image_path in image_paths:
   print(f"\nğŸ“‚ Processing image: {image_path}")
   
   try:
       image = cv2.imread(image_path)
       image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       print("Image loaded successfully.")
   except Exception as e:
       print(f"Failed to load image: {e}")
       continue
   # Convert to grayscale
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   # Apply Gaussian blur
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)
   # Adaptive thresholding to segment the image
   adaptive_thresh = cv2.adaptiveThreshold(
       blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
       cv2.THRESH_BINARY_INV, 51, 9
   )
   # Morphological closing to fill small holes
   kernel = np.ones((5, 5), np.uint8)
   closing = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
   # Mask out the region that contains calibration tools (scale bar + color checker)
   (x, y, w, h) = calibration_tools_mask_area
   cv2.rectangle(closing, (x, y), (x + w, y + h), (0, 0, 0), -1)  # -1 means filled rectangle
   # Find contours in the cleaned-up mask
   contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   print(f"ğŸ”� Found {len(contours)} contours after masking calibration tools.")
   output_image = image_rgb.copy()
   
   if contours:
       # Choose the largest contour, assuming it is the sherd
       sherd_contour = max(contours, key=cv2.contourArea)
       
       # Get bounding box of the sherd
       (sx, sy, sw, sh) = cv2.boundingRect(sherd_contour)
       
       # Draw rectangle and label on the original image
       cv2.rectangle(output_image, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 4)  # Purple box
       cv2.putText(output_image, "Automated Sherd Mark", (sx, sy - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
       print("Successfully marked sherd region.")
       print(f"ğŸ“� Sherd coordinates: (x={sx}, y={sy}, width={sw}, height={sh})")
   else:
       print("No sherd contour detected.")
   
   # Add scale bar and color checker marks
   # Scale Bar
   scale_coords = (275, 780, 465, 88)
   cv2.rectangle(output_image, (scale_coords[0], scale_coords[1]), 
                 (scale_coords[0] + scale_coords[2], scale_coords[1] + scale_coords[3]), (0, 255, 0), 4)
   cv2.putText(output_image, "Scale Bar", (scale_coords[0], scale_coords[1] - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
   print(f"ğŸ“� Scale Bar coordinates: (x={scale_coords[0]}, y={scale_coords[1]}, width={scale_coords[2]}, height={scale_coords[3]})")
   
   # Color Checker
   color_coords = (275, 865, 465, 40)
   cv2.rectangle(output_image, (color_coords[0], color_coords[1]), 
                 (color_coords[0] + color_coords[2], color_coords[1] + color_coords[3]), (255, 0, 0), 4)
   cv2.putText(output_image, "Color Checker", (color_coords[0], color_coords[1] - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
   print(f"ğŸ“� Color Checker coordinates: (x={color_coords[0]}, y={color_coords[1]}, width={color_coords[2]}, height={color_coords[3]})")
   
   # --- Visualization ---
   plt.figure(figsize=(18, 9))
   plt.subplot(1, 3, 1)
   plt.imshow(image_rgb)
   plt.title('1. Original Image')
   plt.axis('off')
   plt.subplot(1, 3, 2)
   plt.imshow(closing, cmap='gray')
   plt.title("2. Segmentation with Calibration Mask")
   plt.axis('off')
   plt.subplot(1, 3, 3)
   plt.imshow(output_image)
   plt.title('3. Detected Sherd')
   plt.axis('off')
   plt.tight_layout()
   plt.show()


# Real-world physical measurement

import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = '/kaggle/input/h690/h690/sherd_images/JD00002_interior.jpg'

# Define the fixed mask area covering the calibration tools
calibration_tools_mask_area = (260, 730, 480, 200)

print(f"\nğŸ“‚ Processing image: {image_path}")

try:
   image = cv2.imread(image_path)
   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   print("Image loaded successfully.")
except Exception as e:
   print(f"Failed to load image: {e}")
   exit()

# --- Image segmentation and sherd detection ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
adaptive_thresh = cv2.adaptiveThreshold(
   blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
   cv2.THRESH_BINARY_INV, 51, 9
)
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Apply masking to exclude calibration tools
(x, y, w, h) = calibration_tools_mask_area
cv2.rectangle(closing, (x, y), (x + w, y + h), (0, 0, 0), -1)

contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"ğŸ”� Found {len(contours)} contours after masking calibration tools.")

output_image = image_rgb.copy()

# --- Real-world physical measurements ---
pixels_per_metric = None

# 1. Compute pixel-to-cm ratio using known scale bar
scale_coords = (275, 780, 465, 88)
scale_bar_pixel_width = scale_coords[2]  
known_width_cm = 6.0  # Known physical length of scale bar in cm
pixels_per_metric = scale_bar_pixel_width / known_width_cm
print(f"ğŸ“� Computed scale: {pixels_per_metric:.2f} pixels/cm")

if contours:
   # Identify the largest contour, assuming it's the sherd
   sherd_contour = max(contours, key=cv2.contourArea)
   
   # 2. Calculate physical dimensions using bounding box
   (sx, sy, sw, sh) = cv2.boundingRect(sherd_contour)
   sherd_width_cm = sw / pixels_per_metric
   sherd_height_cm = sh / pixels_per_metric
   
   # 3. Calculate actual sherd area using contour (not bounding box)
   sherd_area_pixels = cv2.contourArea(sherd_contour)  
   sherd_area_cm2 = sherd_area_pixels / (pixels_per_metric ** 2)
   
   # 4. Calculate bounding box area for comparison
   bbox_area_pixels = sw * sh
   bbox_area_cm2 = bbox_area_pixels / (pixels_per_metric ** 2)
   
   # 5. Calculate fill ratio (actual area vs bounding box area)
   fill_ratio = (sherd_area_pixels / bbox_area_pixels) * 100
   
   print("âœ… Sherd contour successfully identified.")
   print(f"ğŸ“� Sherd pixel coordinates: (x={sx}, y={sy}, width={sw}, height={sh})")
   print("--- Real-World Physical Measurements ---")
   print(f"Bounding Box Width: {sherd_width_cm:.2f} cm")
   print(f"Bounding Box Height: {sherd_height_cm:.2f} cm")
   print(f"Bounding Box Area: {bbox_area_cm2:.2f} cmÂ²")
   print(f"ğŸ�¯ Actual Sherd Area: {sherd_area_cm2:.2f} cmÂ²")
   print(f"ğŸ“Š Fill Ratio: {fill_ratio:.1f}% (sherd area / bounding box area)")
   
   # Draw bounding box
   cv2.rectangle(output_image, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 2)
   
   # Draw actual sherd contour
   cv2.drawContours(output_image, [sherd_contour], -1, (0, 255, 0), 3)
   
   # Add measurement text 
   dimension_text = f"W: {sherd_width_cm:.1f}cm  H: {sherd_height_cm:.1f}cm"
   cv2.putText(output_image, dimension_text, (sx, sy - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
   
   # Add legend
   cv2.putText(output_image, "Green: Actual Sherd Contour", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
   cv2.putText(output_image, "Purple: Bounding Box", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
   
else:
   print("âš ï¸� No sherd contour detected.")

# Add scale bar marking for reference
scale_coords = (275, 780, 465, 88)
cv2.rectangle(output_image, (scale_coords[0], scale_coords[1]), 
             (scale_coords[0] + scale_coords[2], scale_coords[1] + scale_coords[3]), (0, 255, 0), 2)
cv2.putText(output_image, "Scale Bar (6cm)", (scale_coords[0], scale_coords[1] - 5), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# --- Visualization ---
plt.figure(figsize=(15, 10))
plt.imshow(output_image)
plt.title(f'Physical Measurement Results for JD00002_interior.jpg')
plt.axis('off')
plt.tight_layout()
plt.show()


# Standardized color analysis

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def extract_color_patches_auto(color_checker_image):
    """Automatically detect and extract color patches from a color checker"""
    h, w = color_checker_image.shape[:2]
    possible_cols = [12, 13, 14, 15, 16]  # Try different patch counts

    colors = []
    best_cols = 12  # Default

    for cols in possible_cols:
        patch_w = w // cols
        if patch_w > 10:
            best_cols = cols
            break

    print(f"ğŸ�¨ Detected {best_cols} color patches")
    patch_w = w // best_cols

    for j in range(best_cols):
        margin = 2
        x1 = max(0, j * patch_w + margin)
        x2 = min(w, (j + 1) * patch_w - margin)
        y1 = margin
        y2 = h - margin

        if y2 > y1 and x2 > x1:
            patch = color_checker_image[y1:y2, x1:x2]
            avg_color = np.mean(patch, axis=(0, 1))
            colors.append(avg_color)
            print(f"Patch {j+1}: RGB({avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f})")

    return np.array(colors, dtype="float32")

def compute_color_correction_matrix_simple(observed_colors):
    """
    Simplified color correction using white balance assumption
    """
    num_colors = len(observed_colors)
    
    if num_colors >= 3:
        white_patch = observed_colors[-4] if num_colors >= 4 else observed_colors[-1]
        gray_patch = observed_colors[-2] if num_colors >= 2 else observed_colors[0]

        target_white = np.array([240, 240, 240], dtype=np.float32)
        target_gray = np.array([128, 128, 128], dtype=np.float32)

        white_correction = target_white / np.maximum(white_patch, 1.0)
        ccm = np.diag(white_correction)

        print(f"ğŸ�¯ Detected white patch: RGB({white_patch[0]:.0f}, {white_patch[1]:.0f}, {white_patch[2]:.0f})")
        print(f"ğŸ“Š White balance correction: R={white_correction[0]:.3f}, G={white_correction[1]:.3f}, B={white_correction[2]:.3f}")
    else:
        ccm = np.eye(3, dtype=np.float32)
        print("âš ï¸� Insufficient color patches for correction. Using identity matrix.")

    return ccm

def apply_color_correction_matrix(image, ccm):
    """
    Apply the color correction matrix to the entire image
    """
    original_shape = image.shape
    image_flat = image.reshape(-1, 3).astype(np.float32)
    corrected_flat = np.dot(image_flat, ccm.T)
    corrected_flat = np.clip(corrected_flat, 0, 255)
    corrected_image = corrected_flat.reshape(original_shape).astype(np.uint8)
    return corrected_image

# --- Setup and Load Image ---
image_path = '/kaggle/input/h690/h690/sherd_images/JD00002_interior.jpg'

# Define manual ROIs
ROIs = {
    "JD00002_interior": {
        "Pottery Sherd": {"coords": (222, 253, 527, 386)},
        "Color Checker": {"coords": (275, 865, 465, 40)},
    }
}

print(f"\nğŸ“‚ Processing image: {image_path}")
image_name = image_path.split('/')[-1].replace('.jpg', '')

try:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"âœ… Image loaded successfully: {image_path}")
except Exception as e:
    print(f"â�Œ Error loading image: {e}")

# Extract ROIs
current_roi = ROIs["JD00002_interior"]
(x, y, w, h) = current_roi["Color Checker"]["coords"]
observed_checker = image_rgb[y:y+h, x:x+w]
print(f"Color checker area shape: {observed_checker.shape}")

if observed_checker.size == 0:
    print("â�Œ Color checker region is empty. Skipping correction.")
    corrected_image_uint8 = image_rgb.copy()
else:
    observed_colors = extract_color_patches_auto(observed_checker)
    print(f"âœ… Successfully extracted {len(observed_colors)} patches")

    print("ğŸ�¨ Computing color correction matrix...")
    try:
        ccm = compute_color_correction_matrix_simple(observed_colors)
        print(f"CCM matrix:\n{ccm}")

        corrected_image_uint8 = apply_color_correction_matrix(image_rgb, ccm)
    except Exception as e:
        print(f"â�Œ Color correction failed: {e}")
        corrected_image_uint8 = image_rgb.copy()

# Visualize Results
(sx, sy, sw, sh) = current_roi["Pottery Sherd"]["coords"]
original_sherd = image_rgb[sy:sy+sh, sx:sx+sw]
corrected_sherd = corrected_image_uint8[sy:sy+sh, sx:sx+sw]
observed_display = cv2.resize(observed_checker, (600, 100))

plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title(f"1. Original Full Image - {image_name}", fontsize=14)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(corrected_image_uint8)
plt.title(f"2. Color Corrected Full Image - {image_name}", fontsize=14)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(observed_checker)
plt.title("3. Observed Color Checker", fontsize=14)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(original_sherd)
plt.title("4. Sherd - Before Correction", fontsize=14)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(corrected_sherd)
plt.title("5. Sherd - After Correction", fontsize=14)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(observed_display)
plt.title("6. Color Checker (Zoomed)", fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.show()

