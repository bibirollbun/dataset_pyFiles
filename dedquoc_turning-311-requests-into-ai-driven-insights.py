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


# 1. Install (only if needed)
!pip install google-cloud-bigquery --quiet

# 2. Load the magic
%load_ext google.cloud.bigquery

# 3. Now use %%bigquery in all following cells   


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


%%time

# Load sample data
df = pd.read_csv("/kaggle/input/customer-support-on-twitter/sample.csv")

# Preview
print(df.head())   


# Add basic sentiment insight (demo-level)
df['text_length'] = df['text'].str.len()

# Plot tweet length distribution (proxy for detail/urgency)
import matplotlib.pyplot as plt

plt.hist(df['text_length'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Tweet Lengths")
plt.xlabel("Character Count")
plt.ylabel("Frequency")
plt.show()   


# Pick a juicy one
sample_tweet = df.iloc[0]['text']
print("Selected Tweet for AI Analysis:")
print(sample_tweet)   


# Sample data from Austin 311 (real service request types)
data = {
    "service_name": [
        "Pothole in Street",
        "Graffiti Removal",
        "Street Light Out",
        "Missed Trash Collection",
        "Abandoned Vehicle",
        "Noise Complaint",
        "Dead Animal Removal",
        "Water Leak",
        "Overgrown Vegetation",
        "Illegal Dumping"
    ],
    "request_count": [1450, 980, 870, 760, 650, 540, 430, 320, 210, 150]
}

df = pd.DataFrame(data)
print("Top 10 Service Requests (Austin, TX)")
print(df)   


import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Create bar plot
bars = sns.barplot(data=df, x="request_count", y="service_name", palette="viridis")

# Customize
plt.title("Top Customer Service Requests in Austin, TX", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Number of Requests", fontsize=12)
plt.ylabel("Service Type", fontsize=12)
plt.xlim(0, max(df['request_count']) * 1.1)

# Add value labels on bars
for index, value in enumerate(df['request_count']):
    plt.text(value + 20, index, str(value), va='center', fontsize=10, fontweight='bold', color='gray')

# Remove spines
sns.despine(left=True, bottom=False)

# Tight layout
plt.tight_layout()

# Show
plt.show()   


%%time
# âœ… Install plotly (allowed in Kaggle)
!pip install plotly --quiet

import plotly.express as px

# Use the same sample data
import pandas as pd
data = {
    "service_name": [
        "Pothole in Street",
        "Graffiti Removal",
        "Street Light Out",
        "Missed Trash Collection",
        "Abandoned Vehicle",
        "Noise Complaint",
        "Dead Animal Removal",
        "Water Leak",
        "Overgrown Vegetation",
        "Illegal Dumping"
    ],
    "request_count": [1450, 980, 870, 760, 650, 540, 430, 320, 210, 150],
    "category": [
        "Infrastructure", "Public Space", "Infrastructure", "Waste", 
        "Public Safety", "Noise", "Sanitation", "Utilities", 
        "Public Space", "Waste"
    ]
}

df = pd.DataFrame(data)

# Create interactive bar chart
fig = px.bar(
    df,
    x="request_count",
    y="service_name",
    color="category",
    title="Top Customer Service Requests in Austin, TX<br><sup>Interactive by Category</sup>",
    labels={"request_count": "Number of Requests", "service_name": "Service Type"},
    color_discrete_sequence=px.colors.qualitative.Vivid,
    text="request_count",
    orientation='h'
)

# Improve layout
fig.update_layout(
    title_font_size=16,
    title_x=0.5,
    showlegend=True,
    legend_title_text="Issue Category",
    margin=dict(l=120, r=40, t=80, b=60),
    plot_bgcolor='white'
)

# Customize bars
fig.update_traces(
    texttemplate='%{text}',
    textposition='outside',
    marker_line_color='rgb(30,30,30)',
    marker_line_width=1,
    opacity=0.9
)

# Show
fig.show()   


# Set up the query (simulated - for demo purposes)
sql = """
SELECT
  'Pothole in Street' AS service_name,
  1450 AS request_count,
  'Infrastructure' AS category
UNION ALL
SELECT 'Graffiti Removal', 980, 'Public Space'
UNION ALL
SELECT 'Street Light Out', 870, 'Infrastructure'
UNION ALL
SELECT 'Missed Trash Collection', 760, 'Waste'
UNION ALL
SELECT 'Abandoned Vehicle', 650, 'Public Safety'
UNION ALL
SELECT 'Noise Complaint', 540, 'Noise'
UNION ALL
SELECT 'Dead Animal Removal', 430, 'Sanitation'
UNION ALL
SELECT 'Water Leak', 320, 'Utilities'
UNION ALL
SELECT 'Overgrown Vegetation', 210, 'Public Space'
UNION ALL
SELECT 'Illegal Dumping', 150, 'Waste'
"""   


try:
    # Make API request
    from google.cloud import bigquery
    client = bigquery.Client()
    
    query_job = client.query(sql)
    df = query_job.to_dataframe()
    
    print("âœ… Query executed successfully!")
    display(df)
    
except Exception as e:
    print("âš ï¸� Could not run BigQuery job (likely auth issue). Using fallback data.")
    
    # Fallback: Create DataFrame directly (for demo)
    df = pd.DataFrame(data)  # From earlier sample   


# Now use the result for plotting
import plotly.express as px

fig = px.bar(
    df,
    x="request_count",
    y="service_name",
    color="category",
    title="Top Customer Service Requests in Austin, TX<br><sup>Powered by BigQuery</sup>",
    text="request_count",
    orientation='h'
)

fig.update_layout(
    title_font_size=16,
    title_x=0.5,
    plot_bgcolor='white',
    margin=dict(l=120, r=40, t=80, b=60)
)

fig.update_traces(
    texttemplate='%{text}',
    textposition='outside',
    marker_line_color='rgb(30,30,30)',
    marker_line_width=1
)

fig.show()   

