# EDA: Make Data Count - Finding Data References in Scientific Literature
# Competition: https://www.kaggle.com/competitions/make-data-count-finding-data-references
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from collections import Counter, defaultdict
import re
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import plotly.io as pio
pio.renderers.default = 'iframe_connected'


# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("ğŸ”¬ EDA: Make Data Count - Data Citation Mining")
print("=" * 70)


# =====================================
# 1. DATA LOADING & INITIAL EXPLORATION
# =====================================

def load_and_explore_data():
    """Load training data and perform initial exploration"""
    print("\nğŸ“Š SECTION 1: DATA LOADING & STRUCTURE")
    print("-" * 50)
    
    # Load training labels
    train_df = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
    
    print(f"ğŸ“‹ Dataset Shape: {train_df.shape}")
    print(f"ğŸ“‹ Memory Usage: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"ğŸ“‹ Columns: {list(train_df.columns)}")
    
    # Basic info
    print(f"\nğŸ”� Data Types:")
    print(train_df.dtypes)
    print(f"\nğŸ”� Missing Values:")
    missing_data = train_df.isnull().sum()
    if missing_data.sum() > 0:
        print(missing_data[missing_data > 0])
    else:
        print("âœ… No missing values found!")
    
    # Sample data
    print(f"\nğŸ“– Sample Data:")
    print(train_df.head(10))
    
    return train_df

def analyze_file_structure():
    """Analyze the file structure and availability"""
    print("\nğŸ“� SECTION 2: FILE STRUCTURE ANALYSIS")
    print("-" * 50)
    
    base_path = Path('/kaggle/input/make-data-count-finding-data-references')
    
    # Count files in each directory
    train_pdf_path = base_path / 'train' / 'PDF'
    train_xml_path = base_path / 'train' / 'XML'
    test_pdf_path = base_path / 'test' / 'PDF'
    test_xml_path = base_path / 'test' / 'XML'
    
    file_counts = {}
    
    for name, path in [('Train PDF', train_pdf_path), ('Train XML', train_xml_path), 
                       ('Test PDF', test_pdf_path), ('Test XML', test_xml_path)]:
        if path.exists():
            files = list(path.glob('*.pdf' if 'PDF' in name else '*.xml'))
            file_counts[name] = len(files)
            print(f"ğŸ“� {name}: {len(files)} files")
            
            # Sample filenames to understand naming pattern
            if files:
                print(f"   Sample files: {[f.name for f in files[:3]]}")
        else:
            file_counts[name] = 0
            print(f"ğŸ“� {name}: Directory not found")
    
    # Visualize file availability
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    train_counts = [file_counts.get('Train PDF', 0), file_counts.get('Train XML', 0)]
    test_counts = [file_counts.get('Test PDF', 0), file_counts.get('Test XML', 0)]
    
    x = ['PDF', 'XML']
    width = 0.35
    
    ax[0].bar([i - width/2 for i in range(len(x))], train_counts, width, label='Available', alpha=0.8)
    ax[0].set_title('Training Files Availability', fontsize=14, fontweight='bold')
    ax[0].set_ylabel('Number of Files')
    ax[0].set_xticks(range(len(x)))
    ax[0].set_xticklabels(x)
    
    ax[1].bar([i - width/2 for i in range(len(x))], test_counts, width, label='Available', alpha=0.8, color='orange')
    ax[1].set_title('Test Files Availability', fontsize=14, fontweight='bold')
    ax[1].set_ylabel('Number of Files')
    ax[1].set_xticks(range(len(x)))
    ax[1].set_xticklabels(x)
    
    plt.tight_layout()
    plt.show()

    
    return file_counts

# Call the data loading & exploration function
train_df = load_and_explore_data()

# Call the file structure analysis function
file_counts = analyze_file_structure()


# =====================================
# 2. CITATION TYPE ANALYSIS
# =====================================

def comprehensive_citation_analysis(df):
    """Comprehensive analysis of citation types"""
    print("\nğŸ�¯ SECTION 3: CITATION TYPE DEEP DIVE")
    print("-" * 50)
    
    # Basic distribution
    type_counts = df['type'].value_counts()
    type_percentages = df['type'].value_counts(normalize=True) * 100
    
    print(f"ğŸ“Š Citation Type Distribution:")
    for cite_type, count in type_counts.items():
        percentage = type_percentages[cite_type]
        print(f"   {cite_type}: {count:,} ({percentage:.1f}%)")
    
    # Calculate imbalance ratio
    imbalance_ratio = type_counts.max() / type_counts.min()
    print(f"âš–ï¸� Class Imbalance Ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 1.5:
        print("âš ï¸� Significant class imbalance detected - consider sampling strategies")
    
    # Interactive visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Citation Type Distribution', 'Citation Type Pie Chart', 
                       'Cumulative Distribution', 'Citation Type by Count'),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Bar chart
    fig.add_trace(
        go.Bar(x=type_counts.index, y=type_counts.values, 
               text=type_counts.values, textposition='auto',
               marker_color=['#FF6B6B', '#4ECDC4']),
        row=1, col=1
    )
    
    # Pie chart
    fig.add_trace(
        go.Pie(labels=type_counts.index, values=type_counts.values,
               marker_colors=['#FF6B6B', '#4ECDC4']),
        row=1, col=2
    )
    
    # Cumulative distribution
    sorted_counts = type_counts.sort_values(ascending=False)
    cumulative = sorted_counts.cumsum()
    fig.add_trace(
        go.Scatter(x=sorted_counts.index, y=cumulative.values,
                  mode='lines+markers', line=dict(width=3)),
        row=2, col=1
    )
    
    # Citation type by frequency
    fig.add_trace(
        go.Bar(x=type_percentages.index, y=type_percentages.values,
               text=[f'{v:.1f}%' for v in type_percentages.values],
               textposition='auto', marker_color=['#FFB347', '#87CEEB']),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, 
                      title_text="Comprehensive Citation Type Analysis")
    fig.show()
    
    return type_counts, type_percentages

# Load the data
train_df = load_and_explore_data()

# Run the citation analysis
type_counts, type_percentages = comprehensive_citation_analysis(train_df)



# =====================================
# 3. DATASET IDENTIFIER ANALYSIS
# =====================================

def advanced_dataset_id_analysis(df):
    """Advanced analysis of dataset identifiers"""
    print("\nğŸ”— SECTION 4: DATASET IDENTIFIER PATTERNS")
    print("-" * 50)
    
    def classify_dataset_id(dataset_id):
        """Classify dataset ID into detailed categories"""
        dataset_id = str(dataset_id)
        
        # DOI patterns
        if 'https://doi.org/' in dataset_id:
            return 'DOI_FULL_HTTPS'
        elif 'http://doi.org/' in dataset_id or 'http://dx.doi.org/' in dataset_id:
            return 'DOI_FULL_HTTP'
        elif dataset_id.startswith('doi:') or dataset_id.startswith('DOI:'):
            return 'DOI_PREFIX'
        elif re.match(r'^10\.\d+/', dataset_id):
            return 'DOI_BARE'
        
        # Repository-specific patterns
        elif dataset_id.startswith('GSE'):
            return 'GEO_GSE'
        elif dataset_id.startswith('GPL'):
            return 'GEO_GPL'
        elif dataset_id.startswith('GSM'):
            return 'GEO_GSM'
        elif 'pdb' in dataset_id.lower() or dataset_id.startswith('PDB'):
            return 'PDB'
        elif dataset_id.startswith('E-'):
            if 'MTAB' in dataset_id:
                return 'ARRAYEXPRESS_MTAB'
            elif 'MEXP' in dataset_id:
                return 'ARRAYEXPRESS_MEXP'
            else:
                return 'ARRAYEXPRESS_OTHER'
        elif dataset_id.startswith('SRR') or dataset_id.startswith('SRA'):
            return 'SRA'
        elif dataset_id.startswith('PRJN') or dataset_id.startswith('PRJE'):
            return 'BIOPROJECT'
        elif re.match(r'^[A-Z]{1,2}\d+', dataset_id):
            return 'GENBANK_ACCESSION'
        else:
            return 'OTHER'
    
    # Apply classification
    df['dataset_category'] = df['dataset_id'].apply(classify_dataset_id)
    
    # Detailed statistics
    category_stats = df['dataset_category'].value_counts()
    print(f"ğŸ“Š Dataset ID Categories ({len(category_stats)} unique types):")
    for category, count in category_stats.items():
        percentage = (count / len(df)) * 100
        print(f"   {category}: {count:,} ({percentage:.1f}%)")
    
    # Length analysis
    df['dataset_id_length'] = df['dataset_id'].astype(str).str.len()
    length_stats = df['dataset_id_length'].describe()
    print(f"\nğŸ“� Dataset ID Length Statistics:")
    print(f"   Mean: {length_stats['mean']:.1f} characters")
    print(f"   Median: {length_stats['50%']:.1f} characters")
    print(f"   Range: {length_stats['min']:.0f} - {length_stats['max']:.0f} characters")
    
    # Citation type by dataset category
    category_citation_crosstab = pd.crosstab(df['dataset_category'], df['type'])
    category_citation_pct = pd.crosstab(df['dataset_category'], df['type'], normalize='index') * 100
    
    print(f"\nğŸ�¯ Citation Type by Dataset Category:")
    print(category_citation_pct.round(1))
    
    # Advanced visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Dataset Categories Distribution', 'ID Length Distribution',
                       'Citation Type by Category', 'Length vs Citation Type'),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "box"}]]
    )
    
    # Category distribution
    fig.add_trace(
        go.Bar(x=category_stats.values, y=category_stats.index, 
               orientation='h', text=category_stats.values, textposition='auto'),
        row=1, col=1
    )
    
    # Length distribution
    fig.add_trace(
        go.Histogram(x=df['dataset_id_length'], nbinsx=30, opacity=0.7),
        row=1, col=2
    )
    
    # Citation type by category (stacked bar)
    categories = category_citation_crosstab.index
    primary_counts = category_citation_crosstab['Primary'].values
    secondary_counts = category_citation_crosstab['Secondary'].values
    
    fig.add_trace(
        go.Bar(y=categories, x=primary_counts, name='Primary', 
               orientation='h', marker_color='#FF6B6B'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(y=categories, x=secondary_counts, name='Secondary',
               orientation='h', marker_color='#4ECDC4'),
        row=2, col=1
    )
    
    # Box plot: Length vs Citation Type
    for cite_type in df['type'].unique():
        subset = df[df['type'] == cite_type]
        fig.add_trace(
            go.Box(y=subset['dataset_id_length'], name=cite_type),
            row=2, col=2
        )
    
    fig.update_layout(height=1000, showlegend=True,
                      title_text="Advanced Dataset Identifier Analysis")
    fig.show()
    
    # Correlation analysis
    print(f"\nğŸ”� Statistical Tests:")
    from scipy.stats import chi2_contingency, ttest_ind
    
    # Chi-square test for category vs citation type
    chi2, p_val, dof, expected = chi2_contingency(category_citation_crosstab)
    print(f"   Chi-square test (Category vs Citation Type): Ï‡Â² = {chi2:.2f}, p = {p_val:.2e}")
    
    # T-test for ID length vs citation type
    primary_lengths = df[df['type'] == 'Primary']['dataset_id_length']
    secondary_lengths = df[df['type'] == 'Secondary']['dataset_id_length']
    t_stat, t_p_val = ttest_ind(primary_lengths, secondary_lengths)
    print(f"   T-test (ID Length vs Citation Type): t = {t_stat:.2f}, p = {t_p_val:.2e}")
    
    return df, category_stats, category_citation_crosstab

# Run dataset ID analysis
train_df, category_stats, category_crosstab = advanced_dataset_id_analysis(train_df)



# =====================================
# 4. ARTICLE-LEVEL ANALYSIS
# =====================================

def comprehensive_article_analysis(df):
    """Comprehensive analysis at article level"""
    print("\nğŸ“„ SECTION 5: ARTICLE-LEVEL ANALYSIS")
    print("-" * 50)
    
    # Article-level aggregations
    article_stats = df.groupby('article_id').agg({
        'dataset_id': 'count',
        'type': lambda x: (x == 'Primary').sum(),
        'dataset_category': lambda x: x.nunique()
    }).rename(columns={
        'dataset_id': 'total_citations',
        'type': 'primary_citations',
        'dataset_category': 'unique_categories'
    })
    
    article_stats['secondary_citations'] = article_stats['total_citations'] - article_stats['primary_citations']
    article_stats['primary_ratio'] = article_stats['primary_citations'] / article_stats['total_citations']
    
    print(f"ğŸ“Š Article-Level Statistics:")
    print(f"   Total Articles: {len(article_stats):,}")
    print(f"   Citations per Article: {article_stats['total_citations'].mean():.2f} Â± {article_stats['total_citations'].std():.2f}")
    print(f"   Primary Citations per Article: {article_stats['primary_citations'].mean():.2f} Â± {article_stats['primary_citations'].std():.2f}")
    print(f"   Unique Categories per Article: {article_stats['unique_categories'].mean():.2f} Â± {article_stats['unique_categories'].std():.2f}")
    
    # Detailed distribution analysis
    citation_distribution = article_stats['total_citations'].value_counts().sort_index()
    print(f"\nğŸ“ˆ Citation Count Distribution:")
    for citations, articles in citation_distribution.head(10).items():
        print(f"   {citations} citation(s): {articles:,} articles ({articles/len(article_stats)*100:.1f}%)")
    
    # High-citation articles
    high_citation_threshold = article_stats['total_citations'].quantile(0.95)
    high_citation_articles = article_stats[article_stats['total_citations'] >= high_citation_threshold]
    print(f"\nğŸ”¥ High-Citation Articles (â‰¥{high_citation_threshold:.0f} citations): {len(high_citation_articles)}")
    
    # Journal analysis (extract from DOI)
    def extract_journal(doi):
        """Extract journal from DOI"""
        try:
            # Pattern for common DOI formats
            if '/' in doi:
                parts = doi.split('/')
                if len(parts) >= 2:
                    journal_part = parts[-2] if parts[-2] != 'doi.org' else parts[-1]
                    return journal_part.split('.')[0] if '.' in journal_part else journal_part
        except:
            pass
        return 'unknown'
    
    df['journal'] = df['article_id'].apply(extract_journal)
    journal_stats = df.groupby('journal').agg({
        'article_id': 'nunique',
        'dataset_id': 'count',
        'type': lambda x: (x == 'Primary').sum() / len(x)
    }).rename(columns={
        'article_id': 'articles',
        'dataset_id': 'citations',
        'type': 'primary_rate'
    }).sort_values('citations', ascending=False)
    
    print(f"\nğŸ“š Top Journals by Citation Count:")
    for journal, stats in journal_stats.head(10).iterrows():
        print(f"   {journal}: {stats['articles']} articles, {stats['citations']} citations, {stats['primary_rate']*100:.1f}% primary")
    
    # Advanced visualizations
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Citations per Article', 'Primary vs Secondary per Article',
                       'Primary Ratio Distribution', 'Categories per Article',
                       'Top Journals', 'Citation Patterns'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "heatmap"}]]
    )
    
    # Citations per article histogram
    fig.add_trace(
        go.Histogram(x=article_stats['total_citations'], nbinsx=50, opacity=0.7,
                    name='Citations per Article'),
        row=1, col=1
    )
    
    # Primary vs Secondary scatter
    fig.add_trace(
        go.Scatter(x=article_stats['primary_citations'], 
                  y=article_stats['secondary_citations'],
                  mode='markers', opacity=0.6, name='Articles'),
        row=1, col=2
    )
    
    # Primary ratio distribution
    fig.add_trace(
        go.Histogram(x=article_stats['primary_ratio'], nbinsx=30, opacity=0.7,
                    name='Primary Ratio'),
        row=2, col=1
    )
    
    # Categories per article
    fig.add_trace(
        go.Histogram(x=article_stats['unique_categories'], nbinsx=20, opacity=0.7,
                    name='Categories per Article'),
        row=2, col=2
    )
    
    # Top journals
    top_journals = journal_stats.head(15)
    fig.add_trace(
        go.Bar(y=top_journals.index, x=top_journals['citations'],
               orientation='h', name='Citations by Journal'),
        row=3, col=1
    )
    
    fig.update_layout(height=1200, showlegend=False,
                      title_text="Comprehensive Article-Level Analysis")
    fig.show()
    
    return article_stats, journal_stats

article_stats, journal_stats = comprehensive_article_analysis(train_df)




from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Dummy initialization example (adjust based on your layout)
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('...', '...', '...', '...', '...', 'Citation Patterns'),
    specs=[[{"type": "histogram"}, {"type": "scatter"}],
           [{"type": "histogram"}, {"type": "histogram"}],
           [{"type": "bar"}, {"type": "heatmap"}]]
)

# Now add heatmap
heat_data = pd.crosstab(article_stats['primary_citations'], article_stats['secondary_citations'])

fig.add_trace(
    go.Heatmap(
        z=heat_data.values,
        x=heat_data.columns.astype(str),
        y=heat_data.index.astype(str),
        colorscale='Viridis',
        colorbar=dict(title='Frequency')
    ),
    row=3, col=2
)

fig.update_layout(height=500)
fig.show()



# =====================================
# 5. TEMPORAL AND PATTERN ANALYSIS
# =====================================

def temporal_and_pattern_analysis(df):
    """Advanced temporal and pattern analysis"""
    print("\nâ�° SECTION 6: TEMPORAL & PATTERN ANALYSIS")
    print("-" * 50)
    
    # Extract year from DOI (approximation)
    def extract_year_from_doi(doi):
        """Extract year from DOI pattern"""
        year_pattern = re.search(r'20\d{2}', str(doi))
        if year_pattern:
            year = int(year_pattern.group())
            if 2000 <= year <= 2024:
                return year
        return None
    
    df['estimated_year'] = df['article_id'].apply(extract_year_from_doi)
    
    # Temporal analysis (where year is available)
    temporal_df = df.dropna(subset=['estimated_year'])
    if len(temporal_df) > 0:
        print(f"ğŸ“… Temporal Analysis ({len(temporal_df)} records with year data):")
        
        yearly_stats = temporal_df.groupby('estimated_year').agg({
            'dataset_id': 'count',
            'type': lambda x: (x == 'Primary').sum() / len(x),
            'dataset_category': lambda x: x.nunique()
        }).rename(columns={
            'dataset_id': 'citations',
            'type': 'primary_rate',
            'dataset_category': 'unique_categories'
        })
        
        print(yearly_stats)
        
        # Temporal visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=yearly_stats.index,
            y=yearly_stats['citations'],
            mode='lines+markers',
            name='Total Citations',
            line=dict(width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=yearly_stats.index,
            y=yearly_stats['primary_rate'] * yearly_stats['citations'].max(),
            mode='lines+markers',
            name='Primary Rate (scaled)',
            yaxis='y2',
            line=dict(width=3, dash='dash')
        ))
        
        fig.update_layout(
            title='Temporal Trends in Data Citations',
            xaxis_title='Year',
            yaxis_title='Citation Count',
            yaxis2=dict(title='Primary Rate', overlaying='y', side='right'),
            height=500
        )
        fig.show()
    
    # Pattern mining: Co-occurrence analysis
    print(f"\nğŸ”� PATTERN MINING:")
    
    # Dataset category co-occurrence within articles
    article_categories = df.groupby('article_id')['dataset_category'].apply(list).reset_index()
    
    # Find common category combinations
    from itertools import combinations
    category_pairs = []
    for categories in article_categories['dataset_category']:
        if len(categories) > 1:
            pairs = list(combinations(set(categories), 2))
            category_pairs.extend(pairs)
    
    pair_counts = Counter(category_pairs)
    print(f"   Top Category Co-occurrences:")
    for pair, count in pair_counts.most_common(10):
        print(f"     {pair[0]} + {pair[1]}: {count} articles")
    
    return temporal_df if len(temporal_df) > 0 else df

temporal_df = temporal_and_pattern_analysis(train_df)



# =====================================
# 6. ADVANCED STATISTICAL ANALYSIS
# =====================================

def advanced_statistical_analysis(df, article_stats):
    """Advanced statistical analysis and hypothesis testing"""
    print("\nğŸ“Š SECTION 7: ADVANCED STATISTICAL ANALYSIS")
    print("-" * 50)
    
    from scipy import stats
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import chi2, mutual_info_classif
    
    # Prepare features for analysis
    feature_df = df.copy()
    
    # Encode categorical variables
    le_category = LabelEncoder()
    feature_df['dataset_category_encoded'] = le_category.fit_transform(feature_df['dataset_category'])
    
    le_journal = LabelEncoder()
    feature_df['journal_encoded'] = le_journal.fit_transform(feature_df['journal'])
    
    # Create binary target
    feature_df['is_primary'] = (feature_df['type'] == 'Primary').astype(int)
    
    # Feature importance analysis
    features_for_analysis = ['dataset_id_length', 'dataset_category_encoded', 'journal_encoded']
    X = feature_df[features_for_analysis]
    y = feature_df['is_primary']
    
    # Chi-square test
    chi2_scores, chi2_pvalues = chi2(X, y)
    
    # Mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    feature_importance_df = pd.DataFrame({
        'feature': features_for_analysis,
        'chi2_score': chi2_scores,
        'chi2_pvalue': chi2_pvalues,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    print(f"ğŸ”¬ Feature Importance Analysis:")
    print(feature_importance_df)
    
    # Correlation analysis
    correlation_features = ['dataset_id_length', 'dataset_category_encoded', 'journal_encoded', 'is_primary']
    correlation_matrix = feature_df[correlation_features].corr()
    
    print(f"\nğŸ”— Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Statistical tests
    print(f"\nğŸ“ˆ Statistical Tests:")
    
    # Test 1: Do different dataset categories have different primary rates?
    category_groups = [group['is_primary'].values for name, group in feature_df.groupby('dataset_category')]
    if len(category_groups) > 2:
        f_stat, f_pvalue = stats.f_oneway(*category_groups)
        print(f"   ANOVA (Category vs Primary Rate): F = {f_stat:.3f}, p = {f_pvalue:.2e}")
    
    # Test 2: Is there a relationship between ID length and citation type?
    primary_lengths = feature_df[feature_df['type'] == 'Primary']['dataset_id_length']
    secondary_lengths = feature_df[feature_df['type'] == 'Secondary']['dataset_id_length']
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(primary_lengths, secondary_lengths, alternative='two-sided')
    print(f"   Mann-Whitney U (Length vs Type): U = {u_stat:.0f}, p = {u_pvalue:.2e}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(primary_lengths) - 1) * primary_lengths.var() + 
                         (len(secondary_lengths) - 1) * secondary_lengths.var()) / 
                        (len(primary_lengths) + len(secondary_lengths) - 2))
    cohens_d = (primary_lengths.mean() - secondary_lengths.mean()) / pooled_std
    print(f"   Effect Size (Cohen's d): {cohens_d:.3f}")
    
    # Visualization of statistical analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Feature Importance', 'Correlation Heatmap',
                       'Length Distribution by Type', 'Category vs Primary Rate'),
        specs=[[{"type": "bar"}, {"type": "heatmap"}],
               [{"type": "histogram"}, {"type": "box"}]]
    )
    
    # Feature importance
    fig.add_trace(
        go.Bar(x=feature_importance_df['feature'], 
               y=feature_importance_df['mutual_info'],
               name='Mutual Information'),
        row=1, col=1
    )
    
    # Correlation heatmap
    fig.add_trace(
        go.Heatmap(z=correlation_matrix.values,
                   x=correlation_matrix.columns,
                   y=correlation_matrix.columns,
                   colorscale='RdBu_r',
                   zmid=0),
        row=1, col=2
    )
    
    # Length distribution
    fig.add_trace(
        go.Histogram(x=primary_lengths, name='Primary', opacity=0.7, nbinsx=30),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=secondary_lengths, name='Secondary', opacity=0.7, nbinsx=30),
        row=2, col=1
    )
    
    # Box plot by category
    for i, category in enumerate(feature_df['dataset_category'].unique()):
        subset = feature_df[feature_df['dataset_category'] == category]
        fig.add_trace(
            go.Box(y=subset['is_primary'], name=category),
            row=2, col=2
        )
    
    fig.update_layout(height=1000, showlegend=True,
                      title_text="Advanced Statistical Analysis")
    fig.show()
    
    return feature_importance_df, correlation_matrix

feature_importance_df, correlation_matrix = advanced_statistical_analysis(train_df, article_stats)


top_articles = article_stats.sort_values("total_citations", ascending=False).head(10).reset_index()
fig = go.Figure(data=[go.Table(
    header=dict(values=list(top_articles.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[top_articles[col] for col in top_articles.columns],
               fill_color='lavender', align='left'))])
fig.update_layout(title="Top 10 Most-Cited Articles")
fig.show()


df_sorted = train_df[['dataset_id', 'dataset_id_length', 'type']].drop_duplicates().sort_values('dataset_id_length', ascending=False)
fig = go.Figure(data=[go.Table(
    header=dict(values=["Dataset ID", "Length", "Type"], fill_color='teal', font=dict(color='white')),
    cells=dict(values=[
        df_sorted['dataset_id'].head(10),
        df_sorted['dataset_id_length'].head(10),
        df_sorted['type'].head(10)
    ])
)])
fig.update_layout(title="ğŸ”¢ Longest Dataset IDs")
fig.show()


rare_categories = train_df.groupby('dataset_category').filter(lambda x: len(x) < 5)
rare_stats = rare_categories.groupby('dataset_category')['type'].value_counts().unstack().fillna(0)
rare_stats['total'] = rare_stats.sum(axis=1)
rare_stats = rare_stats.sort_values('total', ascending=False).head(10)

fig = go.Figure(data=[go.Table(
    header=dict(values=list(rare_stats.reset_index().columns), fill_color='salmon'),
    cells=dict(values=[rare_stats.reset_index()[col] for col in rare_stats.reset_index().columns],
               fill_color='mistyrose'))])
fig.update_layout(title="ğŸŒ± Niche Categories with Citations")
fig.show()



high_primary = article_stats[article_stats['primary_citations'] > 0].copy()
high_primary['primary_ratio'] = high_primary['primary_citations'] / high_primary['total_citations']
high_primary = high_primary.sort_values('primary_ratio', ascending=False).head(10)

fig = go.Figure(data=[go.Table(
    header=dict(values=high_primary.reset_index().columns, fill_color='lightgreen'),
    cells=dict(values=[high_primary.reset_index()[col] for col in high_primary.reset_index().columns],
               fill_color='mintcream'))])
fig.update_layout(title="ğŸ�¯ Articles with Highest Primary Citation Ratio")
fig.show()


