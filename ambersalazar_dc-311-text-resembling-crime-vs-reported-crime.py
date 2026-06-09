# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# DC 311 Violence Analysis: Text Classification vs Reported Crime Trends
# Complete working code with proper error handling and diagnostics

# Step 1: Install Required Packages
!pip install sentence-transformers --quiet
!pip install plotly --quiet
!pip install scikit-learn --quiet
!pip install pandas --quiet
!pip install numpy --quiet
!pip install seaborn --quiet
!pip install matplotlib --quiet

# Step 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Step 3: Load Real DC 311 Data
print("ğŸ“Š Loading DC 311 Service Request Data...")
df_311 = pd.read_csv('/kaggle/input/311-dc-service-request-dataset-for-2025/311_City_Service_Requests_in_2025.csv')

# Display basic information about the dataset
print(f"\nDataset shape: {df_311.shape}")
print("\nColumn names:")
print(df_311.columns.tolist())
print("\nFirst few rows:")
print(df_311.head())

# Analyze columns to find the best text source
print("\nğŸ”� Analyzing columns for text content...")
for col in df_311.columns:
    if 'DETAIL' in col.upper() or 'DESCRIPTION' in col.upper() or 'TEXT' in col.upper():
        non_null_count = df_311[col].notna().sum()
        unique_count = df_311[col].nunique()
        print(f"\n{col}:")
        print(f"  - Non-null values: {non_null_count} ({non_null_count/len(df_311)*100:.1f}%)")
        print(f"  - Unique values: {unique_count}")
        if non_null_count > 0:
            print(f"  - Sample values: {df_311[col].dropna().head(3).tolist()}")

# Smart text column selection - prioritize columns with actual data
text_col = None
text_columns_priority = ['DETAILS', 'SERVICECODEDESCRIPTION', 'SERVICETYPECODEDESCRIPTION', 
                        'SERVICEORDERSTATUS', 'STATUS_CODE']

for col in text_columns_priority:
    if col in df_311.columns:
        non_null_count = df_311[col].notna().sum()
        if non_null_count > len(df_311) * 0.5:  # At least 50% non-null
            text_col = col
            print(f"\nâœ… Selected text column: {text_col} ({non_null_count} non-null values)")
            break

if text_col is None:
    print("\nâ�Œ ERROR: No suitable text column found with sufficient data!")
    print("Exiting analysis...")
    raise ValueError("No suitable text column found")

# For violence detection, we might want to combine multiple text fields
# Check if DETAILS has any data and combine with service descriptions
if 'DETAILS' in df_311.columns:
    details_count = df_311['DETAILS'].notna().sum()
    print(f"\nDETAILS column has {details_count} non-null values")
    if details_count > 1000:  # If we have some details, create combined text
        df_311['combined_text'] = df_311.apply(
            lambda row: ' '.join([
                str(row.get('SERVICECODEDESCRIPTION', '')),
                str(row.get('DETAILS', ''))
            ]).strip(), axis=1
        )
        text_col = 'combined_text'
        print("âœ… Created combined text field from SERVICECODEDESCRIPTION + DETAILS")

# Identify date column
print("\nğŸ“… Analyzing date columns...")
date_columns = [col for col in df_311.columns if 'DATE' in col.upper()]
for col in date_columns:
    try:
        sample_dates = pd.to_datetime(df_311[col].dropna().head(), errors='coerce')
        valid_dates = sample_dates.notna().sum()
        if valid_dates > 0:
            print(f"{col}: {valid_dates} valid dates, range: {sample_dates.min()} to {sample_dates.max()}")
    except:
        print(f"{col}: Could not parse dates")

# Use ADDDATE as primary date column
date_col = 'ADDDATE' if 'ADDDATE' in df_311.columns else date_columns[0] if date_columns else None
print(f"\nâœ… Selected date column: {date_col}")

# Clean the data
print("\nğŸ§¹ Cleaning data...")
df_311_clean = df_311.copy()

# Handle text column
df_311_clean[text_col] = df_311_clean[text_col].fillna('')
df_311_clean = df_311_clean[df_311_clean[text_col].str.strip() != '']
df_311_clean['text'] = df_311_clean[text_col].astype(str)

print(f"Rows with non-empty text: {len(df_311_clean)}")

# Parse dates and remove timezone info
if date_col:
    df_311_clean['date'] = pd.to_datetime(df_311_clean[date_col], errors='coerce')
    df_311_clean['date'] = df_311_clean['date'].dt.tz_localize(None)
    df_311_clean = df_311_clean.dropna(subset=['date'])
    print(f"Rows with valid dates: {len(df_311_clean)}")
else:
    print("âš ï¸� WARNING: No date column found, creating synthetic dates")
    df_311_clean['date'] = pd.date_range(start='2025-01-01', periods=len(df_311_clean), freq='H')

print(f"\nFinal cleaned dataset shape: {len(df_311_clean)} rows")
print(f"Date range: {df_311_clean['date'].min()} to {df_311_clean['date'].max()}")

# Check if we have enough data to continue
if len(df_311_clean) < 100:
    print("\nâ�Œ ERROR: Not enough data after cleaning (less than 100 rows)")
    raise ValueError("Insufficient data for analysis")

# Step 4: Set Up Vector Similarity Search for Violence Classification
print("\nğŸ”� Setting up violence detection model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define violence-related reference texts specific to DC context
# These are calibrated for 311 service requests
violence_references = [
    "violent crime assault weapon gun shooting gunfire shots fired emergency",
    "stabbing knife attack victim injured hurt wounded blood",
    "robbery theft armed dangerous threat threatening menacing fear",
    "gang violence shooting homicide murder killed dead body",
    "domestic violence abuse assault battery hit punch fight",
    "fight brawl altercation physical violence attack aggressive",
    "weapon firearm pistol rifle gun dangerous armed",
    "emergency police help danger unsafe crime in progress",
    "suspicious person activity threat public safety concern"
]

non_violence_references = [
    "pothole road repair infrastructure maintenance street work",
    "trash garbage collection recycling waste removal pickup",
    "noise complaint loud music disturbance quiet hours",
    "parking violation illegal vehicle abandoned car tow",
    "graffiti vandalism property damage cleanup paint removal",
    "street light broken utility repair outage electricity",
    "tree trimming sidewalk repair maintenance landscaping",
    "water leak pipe burst utility issue plumbing",
    "snow removal ice salt plow winter weather",
    "permit license inspection building code compliance"
]

# Encode reference texts
print("Creating reference embeddings...")
violence_embeddings = model.encode(violence_references)
non_violence_embeddings = model.encode(non_violence_references)

# Step 5: Classify 311 Texts
print("\nğŸ¤– Classifying 311 texts for violence content...")
print(f"Total texts to classify: {len(df_311_clean)}")

def classify_violence_batch(texts, model, violence_emb, non_violence_emb, batch_size=100):
    """Classify texts as violent or non-violent using vector similarity"""
    classifications = []
    confidence_scores = []
    violence_scores = []
    
    # Handle empty list
    if len(texts) == 0:
        return [], [], []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        if i % 10000 == 0:
            print(f"  Processing: {i}/{len(texts)} ({i/len(texts)*100:.1f}%)")
        
        batch_texts = texts[i:i+batch_size]
        
        # Encode the batch
        text_embeddings = model.encode(batch_texts, show_progress_bar=False)
        
        for embedding in text_embeddings:
            # Calculate similarities
            violence_sim = cosine_similarity([embedding], violence_emb).max()
            non_violence_sim = cosine_similarity([embedding], non_violence_emb).max()
            
            # Classify based on higher similarity with threshold
            # Adjust threshold for 311 data (typically less violent language)
            is_violent = violence_sim > non_violence_sim and violence_sim > 0.3
            confidence = abs(violence_sim - non_violence_sim)
            
            classifications.append(is_violent)
            confidence_scores.append(confidence)
            violence_scores.append(violence_sim)
    
    return classifications, confidence_scores, violence_scores

# Classify all texts
texts_to_classify = df_311_clean['text'].tolist()

if len(texts_to_classify) > 0:
    predictions, confidences, violence_scores = classify_violence_batch(
        texts_to_classify, 
        model, 
        violence_embeddings, 
        non_violence_embeddings
    )
    
    df_311_clean['predicted_violent'] = predictions
    df_311_clean['confidence'] = confidences
    df_311_clean['violence_score'] = violence_scores
    
    print(f"\nâœ… Classification complete!")
    print(f"Total texts classified as violent: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
    print(f"Average confidence score: {np.mean(confidences):.3f}")
    print(f"Average violence score: {np.mean(violence_scores):.3f}")
else:
    print("\nâ�Œ ERROR: No texts to classify!")
    raise ValueError("No texts available for classification")

# Show sample classifications
print("\nğŸ“‹ Sample Text Analysis:")
print("\nTop 10 highest violence scores:")
top_violent = df_311_clean.nlargest(10, 'violence_score')
for idx, row in top_violent.iterrows():
    print(f"\nText: {row['text'][:150]}...")
    print(f"Violence Score: {row['violence_score']:.3f}, Confidence: {row['confidence']:.3f}")

# Analyze violence by service type
if 'SERVICECODEDESCRIPTION' in df_311_clean.columns:
    print("\nğŸ“Š Violence detection by service type:")
    service_violence = df_311_clean.groupby('SERVICECODEDESCRIPTION').agg({
        'predicted_violent': ['sum', 'count', 'mean'],
        'violence_score': 'mean'
    }).round(3)
    service_violence.columns = ['violent_count', 'total_count', 'violent_pct', 'avg_violence_score']
    service_violence = service_violence.sort_values('violent_count', ascending=False).head(10)
    print(service_violence)

# Step 6: Load or Create Crime Data
print("\nğŸ“Š Checking for crime data...")

# Try to load real DC crime data from common paths
crime_data_loaded = False
crime_paths = [
    '/kaggle/input/dc-crime-data/crime_incidents.csv',
    '/kaggle/input/washington-dc-crime-data/Crime_Incidents_in_*.csv',
    '/kaggle/input/dc-open-data/Crime_Incidents.csv'
]

for path in crime_paths:
    try:
        import glob
        matching_files = glob.glob(path)
        if matching_files:
            df_crime = pd.read_csv(matching_files[0])
            print(f"âœ… Loaded crime data from: {matching_files[0]}")
            print(f"Crime data shape: {df_crime.shape}")
            print(f"Crime data columns: {df_crime.columns.tolist()[:10]}...")
            crime_data_loaded = True
            break
    except:
        continue

if not crime_data_loaded:
    print("\nâš ï¸� WARNING: No real crime data found. Analysis will use simulated crime trends.")
    print("For accurate analysis, please provide DC crime incident data.")
    
    # Create realistic crime data based on DC crime statistics
    date_range = pd.date_range(start=df_311_clean['date'].min(), 
                              end=df_311_clean['date'].max(), 
                              freq='D')
    
    crime_counts = []
    for date in date_range:
        # DC averages about 15-25 violent crimes per day
        base_rate = 20
        
        # Add seasonal variation (higher in summer)
        month = date.month
        seasonal_factor = 1 + 0.3 * np.sin((month - 1) * np.pi / 6)
        
        # Add day of week variation (higher on weekends)
        dow_factor = 1.2 if date.dayofweek in [4, 5, 6] else 1.0
        
        # Add random variation
        daily_crimes = int(base_rate * seasonal_factor * dow_factor * np.random.uniform(0.7, 1.3))
        
        crime_counts.append({
            'date': date,
            'violent_crimes': daily_crimes
        })
    
    df_crime_violent = pd.DataFrame(crime_counts)
    print("Created simulated crime data for demonstration purposes")
else:
    # Process real crime data
    print("\nğŸ”� Processing crime data...")
    
    # Common date column names in crime data
    date_cols = ['REPORT_DAT', 'START_DATE', 'OFFENSE_DATE', 'DATE']
    crime_date_col = None
    for col in date_cols:
        if col in df_crime.columns:
            crime_date_col = col
            break
    
    if crime_date_col:
        df_crime['date'] = pd.to_datetime(df_crime[crime_date_col], errors='coerce')
        
        # Filter for violent crimes
        violent_keywords = ['HOMICIDE', 'ASSAULT', 'ROBBERY', 'SEX', 'WEAPON', 'SHOOTING']
        offense_col = None
        for col in ['OFFENSE', 'OFFENSE_TYPE', 'CRIME_TYPE']:
            if col in df_crime.columns:
                offense_col = col
                break
        
        if offense_col:
            mask = df_crime[offense_col].str.upper().str.contains('|'.join(violent_keywords), na=False)
            df_crime_violent = df_crime[mask].groupby(df_crime['date'].dt.date).size().reset_index(name='violent_crimes')
            df_crime_violent['date'] = pd.to_datetime(df_crime_violent['date'])
            print(f"Found {len(df_crime_violent)} days with violent crime data")
        else:
            # If no offense column, count all crimes
            df_crime_violent = df_crime.groupby(df_crime['date'].dt.date).size().reset_index(name='violent_crimes')
            df_crime_violent['date'] = pd.to_datetime(df_crime_violent['date'])
            print("âš ï¸� No offense type column found, using all crimes")

# Step 7: Aggregate Data for Analysis
print("\nğŸ“Š Aggregating data for analysis...")

# Aggregate 311 violence predictions by date
daily_311 = df_311_clean.groupby(df_311_clean['date'].dt.date).agg({
    'predicted_violent': 'sum',
    'confidence': 'mean',
    'violence_score': 'mean'
}).reset_index()
daily_311.columns = ['date', '311_violent_count', 'avg_confidence', 'avg_violence_score']
daily_311['date'] = pd.to_datetime(daily_311['date'])

# Ensure both dataframes have the same datetime type (no timezone)
daily_311['date'] = pd.to_datetime(daily_311['date']).dt.tz_localize(None)
df_crime_violent['date'] = pd.to_datetime(df_crime_violent['date']).dt.tz_localize(None)

# Merge datasets
df_analysis = pd.merge(daily_311, df_crime_violent, on='date', how='outer').fillna(0)

# Calculate rolling averages
df_analysis['311_7day_avg'] = df_analysis['311_violent_count'].rolling(7, center=True).mean()
df_analysis['crime_7day_avg'] = df_analysis['violent_crimes'].rolling(7, center=True).mean()

# Calculate rolling correlation
df_analysis['rolling_correlation'] = df_analysis['311_violent_count'].rolling(30).corr(df_analysis['violent_crimes'])

print(f"Analysis dataset shape: {df_analysis.shape}")
print(f"Date range: {df_analysis['date'].min()} to {df_analysis['date'].max()}")
print(f"Total 311 violence detections: {df_analysis['311_violent_count'].sum()}")
print(f"Total violent crimes: {df_analysis['violent_crimes'].sum()}")

# Step 8: Create Comprehensive Visualizations
print("\nğŸ“Š Creating visualizations...")

# Only create visualizations if we have data
if len(df_analysis) > 0 and df_analysis['311_violent_count'].sum() > 0:
    
    # 1. Time Series Comparison
    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Violence Trends: 311 Text-Detected vs Reported Crimes', 
                        '30-Day Rolling Correlation')
    )

    # Add traces
    fig1.add_trace(
        go.Scatter(x=df_analysis['date'], y=df_analysis['crime_7day_avg'],
                   name='Reported Violent Crimes (7-day avg)',
                   line=dict(color='red', width=2)),
        row=1, col=1
    )

    fig1.add_trace(
        go.Scatter(x=df_analysis['date'], y=df_analysis['311_7day_avg'],
                   name='311 Text-Detected Violence (7-day avg)',
                   line=dict(color='blue', width=2)),
        row=1, col=1
    )

    fig1.add_trace(
        go.Scatter(x=df_analysis['date'], y=df_analysis['rolling_correlation'],
                   name='30-day Rolling Correlation',
                   line=dict(color='green', width=2)),
        row=2, col=1
    )

    fig1.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig1.update_layout(
        title='DC Violence Analysis: 311 Text Mining vs Reported Crime',
        height=700,
        showlegend=True,
        hovermode='x unified'
    )

    fig1.update_xaxes(title_text='Date', row=2, col=1)
    fig1.update_yaxes(title_text='Count', row=1, col=1)
    fig1.update_yaxes(title_text='Correlation', row=2, col=1)

    fig1.show()

    # 2. Scatter Plot with Regression
    X = df_analysis['311_violent_count'].values.reshape(-1, 1)
    y = df_analysis['violent_crimes'].values

    # Remove zeros for better visualization
    mask = (X.flatten() > 0) & (y > 0)
    X_filtered = X[mask]
    y_filtered = y[mask]

    if len(X_filtered) > 10:
        reg = LinearRegression()
        reg.fit(X_filtered, y_filtered)
        y_pred = reg.predict(X_filtered)
        r2_score = reg.score(X_filtered, y_filtered)
        
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=df_analysis['311_violent_count'],
            y=df_analysis['violent_crimes'],
            mode='markers',
            name='Daily Data',
            marker=dict(
                color=df_analysis['avg_violence_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Violence<br>Score"),
                size=8,
                opacity=0.7
            ),
            text=[f"Date: {d}<br>311 Count: {x}<br>Crimes: {y}" 
                  for d, x, y in zip(df_analysis['date'], df_analysis['311_violent_count'], df_analysis['violent_crimes'])],
            hovertemplate='%{text}<extra></extra>'
        ))

        fig2.add_trace(go.Scatter(
            x=X_filtered.flatten(),
            y=y_pred,
            mode='lines',
            name=f'Regression Line (RÂ² = {r2_score:.3f})',
            line=dict(color='red', width=3)
        ))

        fig2.update_layout(
            title='Correlation: 311 Text-Detected Violence vs Reported Crimes',
            xaxis_title='311 Text-Detected Violence (Daily Count)',
            yaxis_title='Reported Violent Crimes (Daily Count)',
            height=500
        )

        fig2.show()

    # 3. Ward Analysis (if ward column exists)
    if 'WARD' in df_311_clean.columns:
        ward_analysis = df_311_clean.groupby('WARD').agg({
            'predicted_violent': 'sum',
            'confidence': 'mean',
            'violence_score': 'mean'
        }).reset_index()
        ward_analysis = ward_analysis.sort_values('predicted_violent', ascending=False)
        
        fig4 = px.bar(ward_analysis, x='WARD', y='predicted_violent',
                      title='311 Violence-Related Reports by Ward',
                      labels={'predicted_violent': 'Violence-Related Reports'},
                      color='violence_score',
                      color_continuous_scale='Reds')
        fig4.show()

    # 4. Service Type Analysis
    if 'SERVICECODEDESCRIPTION' in df_311_clean.columns:
        violence_by_service = df_311_clean[df_311_clean['predicted_violent']].groupby('SERVICECODEDESCRIPTION').agg({
            'predicted_violent': 'count',
            'violence_score': 'mean'
        }).reset_index()
        violence_by_service.columns = ['SERVICECODEDESCRIPTION', 'count', 'avg_violence_score']
        violence_by_service = violence_by_service.sort_values('count', ascending=False).head(15)
        
        if len(violence_by_service) > 0:
            fig5 = px.bar(violence_by_service, y='SERVICECODEDESCRIPTION', x='count',
                          title='Top 15 Service Types with Violence-Related Content',
                          labels={'count': 'Number of Violence-Related Reports'},
                          color='avg_violence_score',
                          color_continuous_scale='Reds',
                          orientation='h')
            fig5.update_layout(height=600)
            fig5.show()

else:
    print("\nâš ï¸� WARNING: Insufficient violence detections for visualization")
    print("This may indicate:")
    print("1. The violence detection threshold is too high")
    print("2. The text data doesn't contain violence-related content")
    print("3. The text column doesn't have detailed descriptions")

# Step 9: Statistical Analysis
print("\nğŸ“ˆ Statistical Analysis")
print("=" * 60)

if len(df_analysis) > 30 and df_analysis['311_violent_count'].sum() > 0:
    # Overall correlation
    overall_corr = df_analysis['311_violent_count'].corr(df_analysis['violent_crimes'])
    pearson_test = stats.pearsonr(df_analysis['311_violent_count'], df_analysis['violent_crimes'])
    
    print(f"Overall Correlation: {overall_corr:.3f}")
    print(f"Pearson p-value: {pearson_test[1]:.4f}")
    print(f"Statistically significant: {'Yes' if pearson_test[1] < 0.05 else 'No'}")
    
    # Lag correlation analysis
    lag_results = []
    for lag in range(-14, 15):
        if lag < 0 and len(df_analysis) > abs(lag):
            corr = df_analysis['violent_crimes'][:-lag].corr(df_analysis['311_violent_count'][-lag:])
        elif lag > 0 and len(df_analysis) > lag:
            corr = df_analysis['violent_crimes'][lag:].corr(df_analysis['311_violent_count'][:-lag])
        else:
            corr = overall_corr
        lag_results.append({'lag': lag, 'correlation': corr})
    
    lag_df = pd.DataFrame(lag_results)
    best_lag = lag_df.loc[lag_df['correlation'].idxmax()]
    
    print(f"\nLag Analysis:")
    print(f"Best correlation at lag {best_lag['lag']} days: {best_lag['correlation']:.3f}")
else:
    print("Insufficient data for correlation analysis")

# Step 10: Key Insights
print("\nğŸ”� KEY INSIGHTS")
print("=" * 60)

# Detection statistics
violent_pct = (df_311_clean['predicted_violent'].sum() / len(df_311_clean)) * 100
high_conf_violent = df_311_clean[df_311_clean['predicted_violent'] & (df_311_clean['confidence'] > 0.1)].shape[0]

print(f"\n1. VIOLENCE DETECTION:")
print(f"   â€¢ {violent_pct:.1f}% of 311 reports contain violence-related content")
print(f"   â€¢ {high_conf_violent} reports with high-confidence violence detection")
print(f"   â€¢ Average violence score: {df_311_clean['violence_score'].mean():.3f}")

# Service type analysis
if 'SERVICECODEDESCRIPTION' in df_311_clean.columns:
    print(f"\n2. SERVICE TYPES MOST ASSOCIATED WITH VIOLENCE:")
    top_violent_services = df_311_clean[df_311_clean['predicted_violent']].groupby('SERVICECODEDESCRIPTION').size().sort_values(ascending=False).head(5)
    for service, count in top_violent_services.items():
        print(f"   â€¢ {service}: {count} reports")

# Temporal patterns
if df_311_clean['predicted_violent'].sum() > 0:
    violence_by_month = df_311_clean[df_311_clean['predicted_violent']].groupby(df_311_clean['date'].dt.month).size()
    if len(violence_by_month) > 0:
        peak_month = violence_by_month.idxmax()
        low_month = violence_by_month.idxmin()
        
        print(f"\n3. TEMPORAL PATTERNS:")
        print(f"   â€¢ Peak violence reports: Month {peak_month}")
        print(f"   â€¢ Lowest violence reports: Month {low_month}")

# Geographic patterns
if 'WARD' in df_311_clean.columns:
    print(f"\n4. GEOGRAPHIC PATTERNS:")
    ward_violence = df_311_clean[df_311_clean['predicted_violent']].groupby('WARD').size().sort_values(ascending=False).head(3)
    for ward, count in ward_violence.items():
        print(f"   â€¢ Ward {ward}: {count} violence-related reports")

print("\nğŸ’¡ RECOMMENDATIONS")
print("=" * 60)
print("\n1. DATA QUALITY:")
if 'DETAILS' in df_311.columns and df_311['DETAILS'].notna().sum() < 1000:
    print("   â€¢ The DETAILS field is mostly empty - encourage detailed reporting")
    print("   â€¢ Violence detection relies on service codes which may miss context")

print("\n2. IMPLEMENTATION:")
print("   â€¢ Deploy real-time classification for incoming 311 reports")
print("   â€¢ Flag high-confidence violence detections for priority review")
print("   â€¢ Cross-reference with actual police dispatch data")

print("\n3. IMPROVEMENTS:")
print("   â€¢ Train custom model on DC-specific 311 data")
print("   â€¢ Include location data for geographic clustering")
print("   â€¢ Analyze time-of-day patterns")

if not crime_data_loaded:
    print("\n4. DATA REQUIREMENTS:")
    print("   â€¢ This analysis used simulated crime data")
    print("   â€¢ For accurate correlation analysis, provide:")
    print("     - DC crime incident data with dates and offense types")
    print("     - 911 call data for immediate comparison")

print("\nâœ… Analysis Complete!")
print(f"\nProcessed {len(df_311_clean)} 311 reports")
print(f"Detected {df_311_clean['predicted_violent'].sum()} potential violence-related reports")
print(f"Date range: {df_311_clean['date'].min()} to {df_311_clean['date'].max()}")

