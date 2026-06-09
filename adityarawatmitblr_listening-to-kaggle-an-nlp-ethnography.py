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


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import warnings

# NLP & Visualization Libraries
!pip install bertopic Pillow
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Widget & Display Libraries
import ipywidgets as widgets
from IPython.display import display, clear_output

# Setup Paths
print("Downloading Meta-Kaggle dataset via Kaggle Hub...")
MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
print("Path to Meta-Kaggle dataset files:", MK_PATH)
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


# --- 1. Load and Merge Data ---
print("Loading and merging forum data...")
messages = pd.read_csv(f"{MK_PATH}/ForumMessages.csv")
users = pd.read_csv(f"{MK_PATH}/Users.csv")
forums = pd.read_csv(f"{MK_PATH}/Forums.csv")

# Add a quick print statement to debug and see the actual column names
print(f"Columns in ForumMessages.csv: {messages.columns.tolist()}")

# Merge messages with user tiers using the correct column name.
messages = pd.merge(messages, users[['Id', 'PerformanceTier']], left_on='PostUserId', right_on='Id', how='left')
messages.rename(columns={'PerformanceTier': 'UserTier'}, inplace=True)
# The merge adds an 'Id_y' column from the users table, which we can drop.
if 'Id_y' in messages.columns:
    messages.drop(columns=['Id_y'], inplace=True)


# --- 2. Clean the Text Data ---
print("\nCleaning text data... This may take a few minutes.")

def clean_text(text):
    """A function to clean the raw forum message text."""
    if not isinstance(text, str):
        return ""
    # Remove code blocks enclosed in ```
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code enclosed in `
    text = re.sub(r'`[^`]*`', '', text)
    # Remove Kaggle-specific user tags like @kaggleteam
    text = re.sub(r'@[\w_]+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Keep only alphabetic characters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Applying the cleaning function
messages['CleanedMessage'] = messages['Message'].apply(clean_text)

# Prepare data for time-series analysis
messages['PostDate'] = pd.to_datetime(messages['PostDate'], errors='coerce')
messages.dropna(subset=['PostDate'], inplace=True)
messages['PostYear'] = messages['PostDate'].dt.year

# Filter for a reasonable time frame and non-empty messages
messages = messages[messages['PostYear'].between(2015, 2023)]
messages = messages[messages['CleanedMessage'].str.len() > 20] # Keep only substantive messages

print(f"\nData prepared. {len(messages):,} cleaned messages ready for analysis.")
display(messages[['PostDate', 'UserTier', 'CleanedMessage']].head())


import torch

# --- Train the Topic Model ---
# For performance, we'll use a large random sample of messages.
# 100,000 is a good number for a balance of speed and accuracy.
print("Preparing a sample of 100,000 messages for topic modeling...")
docs = messages['CleanedMessage'].sample(n=100000, random_state=42).tolist()

# Use a vectorizer that removes English stop words to improve topic quality
vectorizer_model = CountVectorizer(stop_words="english")

# --- GPU ACCELERATION LOGIC ---
# 1. Check if a GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU (P100) detected. Using 'cuda' device for acceleration.")
else:
    print("No GPU detected. Using 'cpu' device. This will be much slower.")

# 2. We will use a pre-trained sentence transformer model. 
#    The first time this runs, it will download the model (approx. 90MB).
#    We pass the `device` variable to ensure it runs on the GPU.
print(f"Loading sentence-transformer model 'all-MiniLM-L6-v2' onto '{device}'...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

print("\nInitializing and training BERTopic model... This is the longest step.")
print("Progress will be shown below (embedding documents, reducing dimensions, clustering...).")
topic_model = BERTopic(
  embedding_model=embedding_model,
  vectorizer_model=vectorizer_model,
  language = "english",
  calculate_probabilities=True,
  verbose=True,
  min_topic_size=150 # We filter out topics that are too small and likely noise
)

# This is where the magic happens!
topics, probs = topic_model.fit_transform(docs)

print("\nTopic model training complete!")
print("Here are the top 10 most frequent topics discovered in the conversations:")
# The topic_model.get_topic_info() dataframe shows Topic-1, which are outliers.
display(topic_model.get_topic_info().head(11))


# We use the same random_state to get the exact same sample we trained on.
print("Re-creating the sample DataFrame to link topics with metadata...")
sampled_messages = messages.sample(n=100000, random_state=42)

sampled_messages['Topic'] = topics

print("Successfully created a DataFrame with documents, metadata, and assigned topics.")
display(sampled_messages[['PostDate', 'UserTier', 'Topic', 'CleanedMessage']].head())


display(topic_model.get_topic_info().head(11))


import plotly.express as px
import plotly.graph_objects as go

print("Generating topics-over-time data...")

# --- 1. Data Generation ---
time_range = sampled_messages['PostYear'].max() - sampled_messages['PostYear'].min() + 1
nr_bins = time_range * 12
topics_over_time = topic_model.topics_over_time(
    docs=sampled_messages['CleanedMessage'].tolist(),
    topics=sampled_messages['Topic'].tolist(),
    timestamps=sampled_messages['PostDate'].tolist(),
    nr_bins=nr_bins
)

# --- 2. Data Preparation for Plotting ---

# Getting the IDs of the top 8 topics by frequency, EXCLUDING the outlier topic (-1)
top_topic_ids = topic_model.get_topic_freq().head(9)['Topic'][1:].tolist()

# Now, filter the topics_over_time DataFrame to include ONLY these top topics
plot_data = topics_over_time[topics_over_time['Topic'].isin(top_topic_ids)].copy()

plot_data['Name'] = plot_data['Topic'].map(topic_names)

print("Data prepared correctly. Creating custom annotated visualization...")

# --- 3. Build the Custom Plot ---
fig = px.line(
    plot_data,
    x="Timestamp",
    y="Frequency",
    color="Name",
    labels={"Timestamp": "Year", "Frequency": "Monthly Message Frequency", "Name": "Topics"},
    hover_name="Name"
)

# --- 4. Adding Annotations to Tell the Story ---
fig.add_vline(x=pd.to_datetime('2018-06-01'), line_width=2, line_dash="dash", line_color="gray")
fig.add_annotation(
    x=pd.to_datetime('2018-06-01'), yref="paper", y=0.95,
    text="The 2018 Inflection Point →", showarrow=False,
    xanchor="right", font=dict(size=12, color="gray")
)
if 5 in top_topic_ids:
    viral_topic_data = topics_over_time[topics_over_time['Topic'] == 5]
    if not viral_topic_data.empty:
        peak = viral_topic_data.loc[viral_topic_data['Frequency'].idxmax()]
        fig.add_annotation(
            x=peak.Timestamp, y=peak.Frequency, text="Viral Community Event", arrowhead=2, 
            showarrow=True, ax=-40, ay=-40, font=dict(size=12, color="black"), 
            bgcolor="rgba(255,255,255,0.7)"
        )

# --- 5. Polishing the Final Layout ---
fig.update_layout(
    title_text="<b>The Fossil Record: Topic Popularity Over Time</b>",
    title_x=0.5,
    width=1100,
    height=600,
    legend_title_text='<b>Topics</b>',
    font=dict(family="Arial, sans-serif", size=12)
)

fig.show()


# --- Analyze and Visualize Topics by User Tier ---
print("Preparing data to analyze topic focus by user tier...")

# 1. Define custom names for topics and tiers
topic_names = {
    -1: "Misc. & Outliers", 0: "Notebook Sharing & Culture", 1: "Core ML & Data Science",
    2: "Social Feedback & Thanks", 3: "Dataset Discussion", 4: "Computer Vision (Images)",
    5: "Viral Sharing & Links", 6: "Plotting & Visualization", 7: "Time Series Forecasting",
    8: "Debugging & Tech Support", 9: "Validation & Overfitting"
}
tier_map = {
    0.0: '0 - Novice', 1.0: '1 - Contributor', 2.0: '2 - Expert',
    3.0: '3 - Master', 4.0: '4 - Grandmaster', 5.0: '5 - Kaggle Team'
}

# 2. Add these names to our DataFrame
sampled_messages['UserTierName'] = sampled_messages['UserTier'].map(tier_map).fillna("Unknown")
sampled_messages['TopicName'] = sampled_messages['Topic'].map(topic_names)

# 3. Create the final DataFrame for plotting
topics_by_tier = sampled_messages[sampled_messages['Topic'] != -1].groupby(['UserTierName', 'TopicName']).size().reset_index(name='Frequency')
tier_order = ['0 - Novice', '1 - Contributor', '2 - Expert', '3 - Master', '4 - Grandmaster', '5 - Kaggle Team', 'Unknown']

print("Data prepared. Generating the final visualization...")

# --- Create the plot using seaborn.catplot ---
g = sns.catplot(
    data=topics_by_tier,
    x='Frequency',
    y='TopicName',
    row='UserTierName',
    kind='bar',
    row_order=tier_order,
    height=2,       # Height of each individual facet
    aspect=4,       # Aspect ratio of each facet (width/height)
    palette='viridis',
    orient='h'      # Specify horizontal orientation
)

g.fig.suptitle('Topic Focus by User Tier', y=1.03, fontsize=18, fontweight='bold')
g.set_titles("{row_name}", size=14)
g.set_axis_labels("Message Frequency", "Topics Discussed")
sns.despine(left=True) # Clean up the axes

plt.show()


# --- Reusable Function for Trend Analysis ---
def analyze_and_plot_trend(topic_id, title_text, palette_name='magma'):
    """
    Analyzes and plots the share of conversation for a given topic ID.
    
    Args:
        topic_id (int): The ID of the topic to analyze.
        title_text (str): The title for the chart.
        palette_name (str): The color palette to use for the chart.
    """
    print(f"\n--- Analyzing Archetype: {title_text} (Topic {topic_id}) ---")
    
    # 1. Isolate the Data for our Case Study
    df_topic = sampled_messages[sampled_messages['Topic'] == topic_id].copy()
    if df_topic.empty:
        print(f"No data found for Topic {topic_id}. Skipping plot.")
        return
    
    print(f"Isolated {len(df_topic):,} messages.")

    # 2. Calculate Conversation Share Over Time (Quarterly)
    counts_by_tier = df_topic.set_index('PostDate').groupby('UserTierName').resample('Q').size().reset_index(name='MessageCount')
    total_counts = df_topic.set_index('PostDate').resample('Q').size().reset_index(name='TotalCount')
    
    share_df = pd.merge(counts_by_tier, total_counts, on='PostDate')
    # Add a small epsilon to avoid division by zero
    share_df['ShareOfConversation'] = (share_df['MessageCount'] / (share_df['TotalCount'] + 1e-9)) * 100

    # 3. Visualize the Trend
    plt.figure(figsize=(16, 8))
    sns.lineplot(
        data=share_df,
        x='PostDate',
        y='ShareOfConversation',
        hue='UserTierName',
        hue_order=['0 - Novice', '1 - Contributor', '2 - Expert', '3 - Master', '4 - Grandmaster'],
        lw=2.5,
        palette=palette_name
    )
    
    # 4. Polishing the Plot
    plt.title(f"Anatomy of a Trend: {title_text}", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Share of Conversation (%)", fontsize=14)
    plt.legend(title='User Tier', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# --- Running the Analysis for Our Three Archetypes ---
analyze_and_plot_trend(4, "Foundational Tech: Computer Vision", "viridis")
analyze_and_plot_trend(7, "Competition-Driven Skill: Time Series", "magma")
analyze_and_plot_trend(9, "Strategic Meta-Game: Validation & Overfitting", "cividis")


# --- Interactive Demo ----

import plotly.io as pio
pio.templates.default = "plotly_white"

topic_options = [
    (name, topic_id) 
    for topic_id, name in topic_names.items() 
    if topic_id != -1 # Exclude the outlier topic
]

# Sort the options by topic ID for a consistent order
topic_options = sorted(topic_options, key=lambda x: x[1])

topic_dropdown = widgets.Dropdown(
    options=topic_options, # Use our clean options
    description='Select Topic:',
    style={'description_width': 'initial'},
    layout={'width': '500px'} # Set a fixed width for a clean look
)

output_widget = widgets.Output()

def on_topic_select(change):
    with output_widget:
        clear_output(wait=True) # Clear previous output when a new topic is selected
        topic_id = change['new']
        topic_name = topic_names[topic_id]
        
        # Plot the trend for the selected topic
        topic_data = topics_over_time[topics_over_time['Topic'] == topic_id]
        fig = px.line(
            topic_data, x="Timestamp", y="Frequency",
            labels={"Timestamp": "Year", "Frequency": "Monthly Message Frequency"}
        )
        fig.update_layout(
            title_text=f"<b>Popularity of Topic: {topic_name}</b>",
            title_x=0.5, width=900, height=500
        )
        fig.show()
        
        # Show some real example messages for this topic
        print("\n" + "="*60)
        print(f"REAL EXAMPLE MESSAGES FOR TOPIC: '{topic_name}'")
        print("="*60)
        
        example_docs = sampled_messages[sampled_messages['Topic'] == topic_id]['CleanedMessage'].head(3).tolist()
        if example_docs:
            for i, doc in enumerate(example_docs):
                print(f"--- Example {i+1} ---")
                print(f"{doc[:300]}...") # Print first 300 characters
        else:
            print("No example messages found in the sample for this topic.")
            
# --- Display the Dashboard ---
# We need to link the function to the dropdown's 'value' property
topic_dropdown.observe(on_topic_select, names='value')

# Display the widgets and trigger the first plot for the default value
display(topic_dropdown, output_widget)
on_topic_select({'new': topic_dropdown.value}) 

