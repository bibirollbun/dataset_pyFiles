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


!pip install plotly plotly_express -q


import kagglehub
path = kagglehub.dataset_download("kaggle/meta-kaggle")
print("Path to dataset files:", path)


import os
for dirname, _, filenames in os.walk('/kaggle/input/meta-kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import plotly.express as px
import plotly.io as pio
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# we set default Plotly template for a nicer look
pio.templates.default = "plotly_white"

#Loading the necessary datasets 
try:
    df_kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
    df_kernel_tags = pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')
    df_tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')
    print("Kernels and Tags datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Dataset not found. Please ensure that the file '{e.filename}' is in the correct path.")
    print("Don't forget to add the 'Meta Kaggle' dataset using the 'Add Data' button in the top right of your Kaggle Notebook.")
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    # Exit or handle the error appropriately if data loading fails
    exit()

# Preprocessing the data 
df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])
df_merged_tags = pd.merge(df_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='left')
df_merged_tags.rename(columns={'Name': 'TagName'}, inplace=True)
df_kernel_data = pd.merge(df_kernels, df_merged_tags, left_on='Id', right_on='KernelId', how='inner')

df_kernel_data['YearMonth'] = df_kernel_data['CreationDate'].dt.to_period('M')
tag_popularity_over_time = df_kernel_data.groupby(['YearMonth', 'TagName']).size().reset_index(name='KernelCount')
tag_popularity_over_time['YearMonth'] = tag_popularity_over_time['YearMonth'].dt.to_timestamp()

#Visualzing the Trends with Plotly Express

top_n_tags = df_kernel_data['TagName'].value_counts().head(10).index
df_plot = tag_popularity_over_time[tag_popularity_over_time['TagName'].isin(top_n_tags)]

# Creating the interactive line plot using Plotly Express
fig = px.line(df_plot,
              x='YearMonth',
              y='KernelCount',
              color='TagName',
              title='Evolution of Top Kernel Topic/AI Technique Popularity on Kaggle (by Kernel Count)',
              labels={'YearMonth': 'Date', 'KernelCount': 'Number of Kernels Created', 'TagName': 'Tag Name'},
              hover_data={'YearMonth': '|%Y-%m', 'KernelCount': True, 'TagName': True} # Custom hover info
             )

# Customizing the layout for a cleaner look
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Number of Kernels Created",
    font=dict(size=10),
    legend_title_text='Tag Name',
    hovermode="x unified" # Shows values for all lines at a given x-position when hovering
)

# Making the lines smoother (optional)
fig.update_traces(mode='lines+markers', marker_size=4, line=dict(width=1.5))

# To show the plot in Kaggle notebook or save it as an HTML file
fig.show()


fig.write_html("/kaggle/working/interactive_tag_popularity_over_time.html")

print("\n--- Analysis Complete ---")
print("The interactive plot above shows the trend in the number of kernels created for the top 10 most popular tags over time.")
print("Hover over the lines to see specific values for dates and kernel counts.")


#Specific AI Technique Plot (also interactive)
specific_ai_technique = "Deep Learning"

if specific_ai_technique in df_plot['TagName'].unique():
    fig_single = px.line(df_plot[df_plot['TagName'] == specific_ai_technique],
                         x='YearMonth',
                         y='KernelCount',
                         title=f'Popularity of "{specific_ai_technique.title()}" on Kaggle (by Kernel Count)', # .title() for display
                         labels={'YearMonth': 'Date', 'KernelCount': 'Number of Kernels Created'},
                         hover_data={'YearMonth': '|%Y-%m', 'KernelCount': True}
                        )

    fig_single.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Kernels Created",
        font=dict(size=10)
    )
    fig_single.update_traces(mode='lines+markers', marker_size=6, line=dict(width=2))
    fig.show(renderer="iframe")
    fig_single.write_html(f"/kaggle/working/interactive_{specific_ai_technique.replace(' ', '_')}_popularity.html")

else:
    print(f"\n'{specific_ai_technique}' tag not found in the top N tags or our overall data. Please check the spelling or choose another tag.")


pio.templates.default = "plotly_white" 

# Loading the necessary datasets 
try:
    df_kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
    df_kernel_tags = pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')
    df_tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')
    print("Kernels and Tags datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Dataset not found. Please ensure that the file '{e.filename}' is in the correct path.")
    print("Don't forget to add the 'Meta Kaggle' dataset using the 'Add Data' button in the top right of your Kaggle Notebook.")
    exit() # Exit if essential data is not found
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

#Preprocessing the data
df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])
df_merged_tags = pd.merge(df_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='left')
df_merged_tags.rename(columns={'Name': 'TagName'}, inplace=True)
df_kernel_data = pd.merge(df_kernels, df_merged_tags, left_on='Id', right_on='KernelId', how='inner')

df_kernel_data['YearMonth'] = df_kernel_data['CreationDate'].dt.to_period('M')
tag_popularity_over_time = df_kernel_data.groupby(['YearMonth', 'TagName']).size().reset_index(name='KernelCount')
tag_popularity_over_time['YearMonth'] = tag_popularity_over_time['YearMonth'].dt.to_timestamp()

# Preparing data for Faceting
# Select top N tags for faceting (e.g., top 6 as in your original code)
top_n_for_faceting = df_kernel_data['TagName'].value_counts().head(6).index

# Filtering the data to include only the selected top N tags
df_plot_facet = tag_popularity_over_time[tag_popularity_over_time['TagName'].isin(top_n_for_faceting)].copy()

# Ensuring 'YearMonth' is a datetime type for proper plotting (already done above, but good to re-confirm)
df_plot_facet['YearMonth'] = pd.to_datetime(df_plot_facet['YearMonth'])

#Creating Interactive Facet Plots with Plotly Express ---

fig_facet = px.line(df_plot_facet,
                    x='YearMonth',
                    y='KernelCount',
                    facet_col='TagName', # This creates the small multiples for each TagName
                    facet_col_wrap=3,    # Number of columns before wrapping to the next row
                    title='Kaggle Kernel Tag Popularity Trends (Individual Plots)',
                    labels={'YearMonth': 'Date', 'KernelCount': 'Number of Kernels Created', 'TagName': 'Tag Name'},
                    hover_data={'YearMonth': '|%Y-%m', 'KernelCount': True}, # Custom hover info
                    height=600 # Adjust overall height of the figure
                   )

# Customizing layout for a cleaner look and better readability
fig_facet.update_layout(
    font=dict(size=10),
    title_font_size=18,
    hovermode="x unified", # Shows values for all lines at a given x-position when hovering
    margin=dict(l=40, r=40, t=80, b=40)
)

# Updating traces for line appearance
fig_facet.update_traces(mode='lines+markers', marker_size=4, line=dict(width=1.5))

# Updating x-axis and y-axis titles for each subplot
fig_facet.for_each_xaxis(lambda xaxis: xaxis.update(title_text='')) # Remove individual x-axis titles
fig_facet.for_each_yaxis(lambda yaxis: yaxis.update(title_text='Kernel Count')) # Set y-axis title for each

# Set subplot titles (facet titles)
fig_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1] + " Popularity")) # Clean up facet titles

# Show the plot in Kaggle notebook
fig_facet.show(renderer="iframe")


print("\n--- Plotting Complete ---")
print("Each subplot above shows the interactive popularity trend of a specific Kaggle kernel tag over time.")
print("Hover over the lines to see specific values for dates and kernel counts.")


all_ai_ml_tags_raw = [
    # General AI/ML
    "Artificial Intelligence", "Machine Learning", "Advanced", "Beginner", "Intermediate",
    "Learn", "Research", "AutoML", "Model Comparison", "Model Explainability",
    "Transfer Learning", "Optimization",

    # Core ML Algorithms
    "Classification", "Regression", "Clustering", "Decision Tree", "Logistic Regression",
    "Linear Regression", "K-Means", "Naive Bayes", "Random Forest", "SVM",
    "XGBoost", "LightGBM", "CatBoost",

    # Neural Networks & Deep Learning
    "Deep Learning", "Neural Networks", "DNN", "CNN", "RNN", "LSTM", "Transformer",
    "Transformers", "Multi-head Attention", "Scaled Dot-product Attention", "Attention Dropout",
    "Batch Normalization", "Layer Normalization", "Group Normalization", "Weight Standardization",
    "Gelu", "Leaky ReLU", "Mish", "ReLU", "ReLU6", "Tanh Activation", "Dropout",
    "Residual Block", "Residual Connection", "Dense Block", "Dense Connections",
    "Global Average Pooling", "Max Pooling", "Average Pooling", "Blur Pooling",
    "Anti-alias Downsampling", "Reduction-A", "Reduction-B", "One-shot Aggregation",
    "Selective Kernel", "Split Attention", "Squeeze-and-Excitation Block", "1x1 Convolution",
    "Convolution", "Depthwise Separable Convolution", "Graph Neural Network", "GNN",
    "Adversarial Learning", "GAN", "VAE", "BigGAN", "BigGAN-Deep", "BigBiGAN", "Auxiliary Classifier",

    # Specific DL Architectures/Models
    "Albert", "Amoebanet-A (n=18, f=448)", "Bart", "BERT", "Big Transfer", "Bit", "CenterNet",
    "CondConv", "ConvNeXt", "DeBERTa-v3", "DeiT", "DenseNet", "DistilBERT", "EfficientNet",
    "EfficientNet V2", "EfficientNet V2", "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3",
    "EfficientNet-B4", "EfficientNet-B5", "EfficientNet-B6", "EfficientNet-B7",
    "ELMo", "FCOS", "FNet", "GPT2", "I3D", "ImageNet 2012 Classification", "Inception ResNet v2",
    "Inception v1", "Inception v3", "Inception-V3 Module", "KeywordSpottingNet", "MLPMixer",
    "MobileBERT", "MobileNet", "MobileNet v1", "MobileNet v2", "MobileNet v3", "MobileNetV3",
    "MobileViT", "MoviNet", "Multi-Band MelGAN", "NASNet-A (Mobile)", "NNLM", "RegNetX",
    "RegNetY", "ResNet", "ResNet CIFAR", "ResNet v1 101", "ResNet v1 152", "ResNet v1 50",
    "ResNet v2 101", "ResNet v2 152", "ResNet v2 50", "ResNet101-V2", "ResNet152x4-V2",
    "ResNet19", "ResNet50-V2", "ResNet50x3-V2", "ResNeXt Block", "RetinaNet", "RoBERTa",
    "S3D-G", "SAM", "SegFormer", "Silero-STT", "SSD", "Swin Transformer", "T5",
    "Transformer in Transformer", "Transformer XL", "TSM ResNet50", "UNet", "VGG-Style",
    "ViT", "Wav2Vec2", "Whisper", "XLM-RoBERTa", "YOLO", "YOLOv5", "YOLOv8",

    # Learning Paradigms
    "Reinforcement Learning", "Transfer Learning"
]

# Converting all target tags to lowercase for robust matching (Kaggle tags are often lowercase)
all_ai_ml_tags_lower = [tag.lower() for tag in all_ai_ml_tags_raw]

# Geting unique tags present in the actual df_kernel_data (also lowercased for matching)
actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

#Filter for only the AI/ML tags that *actually exist* in our data 
# We create a set of actual tags for fast lookup
actual_tags_set = set(actual_tags_in_data_lower)

# Creating a list of the *existing* AI/ML tags, preserving original capitalization from df_tags
existing_ai_ml_tags = [
    tag for tag in df_kernel_data['TagName'].unique()
    if tag.lower() in actual_tags_set and tag.lower() in all_ai_ml_tags_lower
]

if not existing_ai_ml_tags:
    print("\nNo AI/ML related tags from your comprehensive list were found in the dataset. Cannot generate plot.")
else:
    print(f"\n--- Found {len(existing_ai_ml_tags)} AI/ML related tags in the dataset. ---")
    print("Here are the first 20 found tags (if more exist):")
    for tag in existing_ai_ml_tags[:20]:
        print(f"- {tag}")
    print("------------------------------------------------------------------")

    df_filtered_ai_ml = df_kernel_data[df_kernel_data['TagName'].isin(existing_ai_ml_tags)].copy()

    if df_filtered_ai_ml.empty:
        print("\nFiltered DataFrame is empty even after identifying existing tags. This might indicate issues with kernel counts for these tags.")
    else:
        df_filtered_ai_ml['YearMonth'] = df_filtered_ai_ml['CreationDate'].dt.to_period('M')
        # We use 'ai_ml_popularity_over_time' directly as px.area works well with long format data
        ai_ml_popularity_over_time = df_filtered_ai_ml.groupby(['YearMonth', 'TagName']).size().reset_index(name='KernelCount')
        ai_ml_popularity_over_time['YearMonth'] = ai_ml_popularity_over_time['YearMonth'].dt.to_timestamp()

        if ai_ml_popularity_over_time.empty or ai_ml_popularity_over_time['KernelCount'].sum() == 0:
            print("\nAggregated AI/ML data is empty or contains no non-zero counts. Cannot generate plot.")
        else:
            # Visualizing the Trends - Stacked Area Plot with Plotly Express 
            fig_area = px.area(ai_ml_popularity_over_time,
                               x='YearMonth',
                               y='KernelCount',
                               color='TagName',
                               title='Evolution of AI/ML Topics and Techniques on Kaggle (Kernel Count)',
                               labels={'YearMonth': 'Date', 'KernelCount': 'Total Kernels Created', 'TagName': 'AI/ML Aspect/Technique'},
                               hover_data={'YearMonth': '|%Y-%m', 'KernelCount': True, 'TagName': True},
                               color_discrete_sequence=px.colors.qualitative.Plotly # Or 'Alphabet', 'Light24' for more colors
                              )

            # Customizing layout for better appearance and legend positioning
            fig_area.update_layout(
                xaxis_title="Date",
                yaxis_title="Total Kernels Created (Stacked)",
                font=dict(size=10),
                title_font_size=18,
                hovermode="x unified", # Shows values for all stacked areas at a given x-position
                legend_title_text='AI/ML Aspect/Technique',
                # Place legend outside the plot area
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    traceorder="reversed" # Optional: to show largest at top of legend
                )
            )

            fig_area.update_xaxes(tickangle=45) # Rotate x-axis labels

            # Show the plot in Kaggle notebook
            
            fig_area.show(renderer="iframe")

            
            fig_area.write_html("/kaggle/working/interactive_ai_ml_stacked_area_plot.html")

            print("\n--- Analysis Complete (AI/ML Stacked Area Plot) ---")
            print("The interactive stacked area plot shows the cumulative trend of kernels for various AI/ML topics and techniques over time,")
            print("highlighting their overall volume and individual proportional contributions.")
            print("Hover over the plot to see specific values for dates and categories.")



general_ai_ml_tags_raw = [
    "Artificial Intelligence", "Machine Learning", "Advanced", "Beginner", "Intermediate",
    "Learn", "Research", "AutoML", "Model Comparison", "Model Explainability",
    "Transfer Learning", "Optimization"
]

# Converting all target tags to lowercase for robust matching
general_ai_ml_tags_lower = [tag.lower() for tag in general_ai_ml_tags_raw]

# Geting unique tags present in the actual df_kernel_data (also lowercased for matching)
actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

# Filtering for only the General AI/ML tags that *actually exist* in our data 
actual_tags_set = set(actual_tags_in_data_lower)

# Creating a list of the *existing* General AI/ML tags, preserving original capitalization from df_kernel_data
existing_general_ai_ml_tags = [
    tag for tag in df_kernel_data['TagName'].unique()
    if tag.lower() in actual_tags_set and tag.lower() in general_ai_ml_tags_lower
]

if not existing_general_ai_ml_tags:
    print("\nNo General AI/ML related tags from your list were found in the dataset. Cannot generate plot.")
else:
    print(f"\n--- Found {len(existing_general_ai_ml_tags)} General AI/ML related tags in the dataset. ---")
    print("Here are the found tags:")
    for tag in existing_general_ai_ml_tags:
        print(f"- {tag}")
    print("------------------------------------------------------------------")

    df_filtered_general_ai_ml = df_kernel_data[df_kernel_data['TagName'].isin(existing_general_ai_ml_tags)].copy()

    if df_filtered_general_ai_ml.empty:
        print("\nFiltered DataFrame for General AI/ML is empty even after identifying existing tags. This might indicate issues with kernel counts for these tags.")
    else:
        # Group by YearMonth and TagName to count kernels
        df_filtered_general_ai_ml['YearMonth'] = df_filtered_general_ai_ml['CreationDate'].dt.to_period('M')
        general_ai_ml_popularity_over_time = df_filtered_general_ai_ml.groupby(['YearMonth', 'TagName']).size().reset_index(name='KernelCount')

        # Converting YearMonth back to timestamp for plotting
        general_ai_ml_popularity_over_time['YearMonth'] = general_ai_ml_popularity_over_time['YearMonth'].dt.to_timestamp()

        if general_ai_ml_popularity_over_time.empty or general_ai_ml_popularity_over_time['KernelCount'].sum() == 0:
            print("\nAggregated General AI/ML data is empty or contains no non-zero counts. Cannot generate plot.")
        else:
            # Visualize the Trends - Stacked Area Plot with Plotly Express
            fig_general_ai_ml_area = px.area(general_ai_ml_popularity_over_time,
                                           x='YearMonth',
                                           y='KernelCount',
                                           color='TagName',
                                           title='Evolution of General AI/ML Topics on Kaggle (Kernel Count)',
                                           labels={'YearMonth': 'Date', 'KernelCount': 'Total Kernels Created', 'TagName': 'General AI/ML Topic'},
                                           hover_data={'YearMonth': '|%Y-%m', 'KernelCount': True, 'TagName': True},
                                           color_discrete_sequence=px.colors.qualitative.Plotly # Or choose another palette like 'T10', 'Set3'
                                          )

            # Customizing layout for better appearance and legend positioning
            fig_general_ai_ml_area.update_layout(
                xaxis_title="Date",
                yaxis_title="Total Kernels Created (Stacked)",
                font=dict(size=10),
                title_font_size=12,
                hovermode="x unified", # Shows values for all stacked areas at a given x-position
                legend_title_text='General AI/ML Topic',
                # Place legend outside the plot area
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    traceorder="reversed" # Optional: to show largest at top of legend
                )
            )

            fig_general_ai_ml_area.update_xaxes(tickangle=45) # Rotate x-axis labels

            # Show the plot in Kaggle notebook
            
            fig_general_ai_ml_area.show(renderer="iframe")

          
            fig_general_ai_ml_area.write_html("/kaggle/working/interactive_general_ai_ml_stacked_area_plot.html")

            print("\n--- Analysis Complete (General AI/ML Stacked Area Plot) ---")
            print("This interactive stacked area plot visualizes the cumulative trend of kernels specifically for 'General AI/ML' topics over time.")
            print("It highlights the overall volume and the proportional contribution of each.")
            print("Hover over the plot to see specific values for dates and categories.")


# Defining General AI/ML related tags
general_ai_ml_tags_raw = [
    "Artificial Intelligence", "Machine Learning", "Advanced", "Beginner", "Intermediate",
    "Learn", "Research", "AutoML", "Model Comparison", "Model Explainability",
    "Transfer Learning", "Optimization"
]

# Converting all target tags to lowercase for robust matching
general_ai_ml_tags_lower = [tag.lower() for tag in general_ai_ml_tags_raw]

# Geting unique tags present in the actual df_kernel_data (also lowercased for matching)
actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

# Filter for only the General AI/ML tags that *actually exist* in our data 
actual_tags_set = set(actual_tags_in_data_lower)

existing_general_ai_ml_tags = [
    tag for tag in df_kernel_data['TagName'].unique()
    if tag.lower() in actual_tags_set and tag.lower() in general_ai_ml_tags_lower
]

if not existing_general_ai_ml_tags:
    print("\nNo 'General AI/ML' tags from the confirmed list were found in the dataset. Cannot generate the plot.")
    print("Please double-check the capitalization or spelling of the tags in your list against df_kernel_data['TagName'].unique().")
else:
    print(f"\n--- Plotting {len(existing_general_ai_ml_tags)} 'General AI/ML' tags. ---")
    print("Tags being plotted:")
    for tag in existing_general_ai_ml_tags:
        print(f"- {tag}")
    print("---------------------------------------------------------------")

    df_filtered_general_ai_ml = df_kernel_data[df_kernel_data['TagName'].isin(existing_general_ai_ml_tags)].copy()

    if df_filtered_general_ai_ml.empty:
        print("\nFiltered DataFrame for 'General AI/ML' is empty. This means no kernels are tagged with these specific tags, or dates are problematic.")
    else:
        df_filtered_general_ai_ml['YearMonth'] = df_filtered_general_ai_ml['CreationDate'].dt.to_period('M')
        df_filtered_general_ai_ml['Year'] = df_filtered_general_ai_ml['CreationDate'].dt.year

        # Creating a 'YearGroup' column for 4-year segments 
        min_year = df_filtered_general_ai_ml['Year'].min()
        max_year = df_filtered_general_ai_ml['Year'].max()

        # Defining bins for 4-year groups
        bins = list(range(min_year, max_year + 5, 4))
        labels = [f"{y}-{y+3}" for y in bins[:-1]]

        df_filtered_general_ai_ml['YearGroup'] = pd.cut(
            df_filtered_general_ai_ml['Year'],
            bins=bins,
            labels=labels[:len(bins)-1],
            right=True,
            include_lowest=True
        )

        df_filtered_general_ai_ml.dropna(subset=['YearGroup'], inplace=True)
        df_filtered_general_ai_ml['YearGroup'] = pd.Categorical(
            df_filtered_general_ai_ml['YearGroup'],
            categories=labels[:len(bins)-1],
            ordered=True
        )

        general_ai_ml_popularity_over_time = df_filtered_general_ai_ml.groupby(['YearGroup', 'YearMonth', 'TagName']).size().reset_index(name='KernelCount')
        general_ai_ml_popularity_over_time['YearMonth'] = general_ai_ml_popularity_over_time['YearMonth'].dt.to_timestamp()

        if general_ai_ml_popularity_over_time.empty or general_ai_ml_popularity_over_time['KernelCount'].sum() == 0:
            print("\nAggregated General AI/ML data for yearly segments is empty or contains no non-zero counts. Cannot generate plot.")
        else:
            #Visualizing the Trends - Line Plot with Facet_Row (Yearly Segments)
            num_year_groups = len(general_ai_ml_popularity_over_time['YearGroup'].unique())
            # Increased height per subplot for better clarity
            facet_height_per_row = 350 # Increased from 300
            total_height = facet_height_per_row * num_year_groups + 150 # Additional space for main title, legend, etc.

            fig_yearly_segments = px.line(general_ai_ml_popularity_over_time,
                                          x='YearMonth',
                                          y='KernelCount',
                                          color='TagName',
                                          facet_row='YearGroup',
                                          title='Evolution of General AI/ML Topics on Kaggle (by Kernel Count) - Yearly Segments',
                                          labels={'YearMonth': 'Date', 'KernelCount': 'Number of Kernels Created', 'TagName': 'AI/ML Topic'},
                                          hover_data={'YearMonth': '|%Y-%m', 'KernelCount': True, 'TagName': True},
                                          height=total_height,
                                          color_discrete_sequence=px.colors.qualitative.Plotly
                                         )

            # Customizing layout for better appearance and readability
            fig_yearly_segments.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Kernels Created",
                font=dict(size=11), # Slightly increased font size
                title_font_size=20, # Increased main title font size
                hovermode="x unified",
                legend_title_text='General AI/ML Topic',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.20,
                    xanchor="right",
                    x=1,
                    font=dict(size=10), # Legend font size
                    bgcolor="rgba(255,255,255,0.6)", # Slightly transparent background for legend
                    bordercolor="LightSteelBlue",
                    borderwidth=1
                ),
                margin=dict(l=60, r=60, t=120, b=100), # More generous margins
                paper_bgcolor='white', # Ensuring background is white for clarity
                plot_bgcolor='white'
            )
            # Adjust spacing between subplots
            fig_yearly_segments.update_traces(mode='lines+markers', marker_size=4, line=dict(width=1.5))

            # Seting subplot titles (facet titles) and remove individual x-axis titles
            fig_yearly_segments.for_each_annotation(lambda a: a.update(text=f"Years: {a.text.split('=')[-1]}", font=dict(size=14))) # Increased facet title font size
            fig_yearly_segments.for_each_xaxis(lambda xaxis: xaxis.update(title_text='', tickangle=45, tickfont=dict(size=10))) # Apply tickangle here and font size
            fig_yearly_segments.for_each_yaxis(lambda yaxis: yaxis.update(title_text='Number of Kernels Created', tickfont=dict(size=10))) # Set y-axis title and font size

            # Show the plot in Kaggle notebook
            fig_yearly_segments.show(renderer="iframe")

            fig_yearly_segments.write_html("/kaggle/working/interactive_general_ai_ml_yearly_segments.html")

            print("\n--- Analysis Complete (General AI/ML Yearly Line Plots) ---")
            print("The interactive line plots above show the trend in the number of kernels created for each 'General AI/ML' topic, broken down into 4-year segments.")
            print("Hover over the lines to see specific values for dates and kernel counts.")


general_ai_ml_tags_raw = [
    "Artificial Intelligence", "Machine Learning", "Advanced", "Beginner", "Intermediate",
    "Learn", "Research", "AutoML", "Model Comparison", "Model Explainability",
    "Transfer Learning", "Optimization"
]

# Converting all target tags to lowercase for robust matching
general_ai_ml_tags_lower = [tag.lower() for tag in general_ai_ml_tags_raw]

# Geting unique tags present in the actual df_kernel_data (also lowercased for matching)
actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

# Filtering for only the General AI/ML tags that *actually exist* in our data 
actual_tags_set = set(actual_tags_in_data_lower)

existing_general_ai_ml_tags = [
    tag for tag in df_kernel_data['TagName'].unique()
    if tag.lower() in actual_tags_set and tag.lower() in general_ai_ml_tags_lower
]

if not existing_general_ai_ml_tags:
    print("\nNo 'General AI/ML' tags from the confirmed list were found in the dataset. Cannot generate the plot.")
    print("Please double-check the capitalization or spelling of the tags in your list against df_kernel_data['TagName'].unique().")
else:
    print(f"\n--- Plotting {len(existing_general_ai_ml_tags)} 'General AI/ML' tags. ---")
    print("Tags being plotted:")
    for tag in existing_general_ai_ml_tags:
        print(f"- {tag}")
    print("---------------------------------------------------------------")

    df_filtered_general_ai_ml = df_kernel_data[df_kernel_data['TagName'].isin(existing_general_ai_ml_tags)].copy()

    if df_filtered_general_ai_ml.empty:
        print("\nFiltered DataFrame for 'General AI/ML' is empty. This means no kernels are tagged with these specific tags, or dates are problematic.")
    else:
        df_filtered_general_ai_ml['Year'] = df_filtered_general_ai_ml['CreationDate'].dt.year

        # Aggregate data by Year and TagName
        yearly_tag_counts = df_filtered_general_ai_ml.groupby(['Year', 'TagName']).size().reset_index(name='KernelCount')

        
        yearly_tag_counts_sorted = yearly_tag_counts.sort_values(by=['Year', 'KernelCount'], ascending=[True, False])


        # Visualizing the Trends - Bar Plot for General AI/ML (Yearly Subplots) 
        num_years = len(yearly_tag_counts_sorted['Year'].unique())
        # Adjusting height per subplot for more clarity and less cramping
        facet_height_per_row = 350 # Each year's plot will have this height
        total_height = facet_height_per_row * num_years + 150 # Add space for main title, legend, margins

        fig_yearly_bars = px.bar(yearly_tag_counts_sorted,
                                 x='TagName',
                                 y='KernelCount',
                                 facet_row='Year', # One row per year
                                 color='TagName', # Color bars by tag name
                                 title='Annual Popularity of General AI/ML Tags on Kaggle',
                                 labels={'TagName': 'AI/ML Tag', 'KernelCount': 'Number of Kernels', 'Year': 'Year'},
                                 hover_data={'TagName': True, 'KernelCount': True, 'Year': True},
                                 height=total_height,
                                 color_discrete_sequence=px.colors.qualitative.Plotly, # Use a qualitative palette for distinct tags
                                 
                                )

        # Customizing layout for better appearance and readability
        fig_yearly_bars.update_layout(
            xaxis_title="AI/ML Tag",
            yaxis_title="Number of Kernels",
            font=dict(size=10),
            title_font_size=14,
            hovermode="x unified", # Hover over any part of the column to see details for all bars in that column
            legend_title_text='AI/ML Tag',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.15,
                xanchor="right",
                x=1,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="LightSteelBlue",
                borderwidth=1
            ),
            margin=dict(l=60, r=60, t=120, b=60), # More generous margins
            paper_bgcolor='white', # Ensure background is white for clarity
            plot_bgcolor='white'
        )

        # Updating x-axis and y-axis settings for each subplot
        fig_yearly_bars.for_each_xaxis(lambda xaxis: xaxis.update(title_text='', tickangle=75, tickfont=dict(size=9))) # Rotate x-axis labels more and reduce font size slightly for tags
        fig_yearly_bars.for_each_yaxis(lambda yaxis: yaxis.update(title_text='Number of Kernels', tickfont=dict(size=9))) # Set y-axis title and font size

        # Updaing facet titles (each year's title)
        fig_yearly_bars.for_each_annotation(lambda a: a.update(text=f"Year: {a.text.split('=')[-1]}", font=dict(size=14))) # Clean up facet titles

        # Show the plot in Kaggle notebook
        fig_yearly_bars.show(renderer="iframe")

    
        fig_yearly_bars.write_html("/kaggle/working/interactive_general_ai_ml_yearly_bar_plots.html")

        print("\n--- Annual Tag Count Summary (General AI/ML) ---")
        print("Here's a detailed breakdown of kernel counts for each tag, year by year:")

        # Create a pivot table for the yearly tag counts
        # Using yearly_tag_counts_sorted to maintain consistency
        pivot_table_counts = yearly_tag_counts_sorted.pivot_table(index='TagName', columns='Year', values='KernelCount', fill_value=0)

        # Sort rows by total count if desired (optional)
        pivot_table_counts['Total'] = pivot_table_counts.sum(axis=1)
        pivot_table_counts = pivot_table_counts.sort_values(by='Total', ascending=False).drop('Total', axis=1)

        print(pivot_table_counts.to_markdown())
        print("\n--- Analysis Complete (Annual Bar Plots & Summary) ---")
        print("The interactive bar plots above visually represent the distribution of General AI/ML tags each year. Below, the table provides the exact kernel counts, giving a precise view of the annual popularity of each tag.")
        print("Hover over any bar to see the exact tag name, year, and kernel count.")


# I define the specific list of skill-related tags I'm interested in ---
skill_level_tags_raw = [
    "Beginner", "Intermediate", "Advanced", "Learn"
]

# I convert all my target tags to lowercase for robust matching, as Kaggle tags are often lowercase.
skill_level_tags_lower = [tag.lower() for tag in skill_level_tags_raw]

# I get unique tags present in my actual df_kernel_data (also lowercased for matching).
actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

# --- 4. Now, I filter for only the skill-related tags that *actually exist* in my data ---
actual_tags_set = set(actual_tags_in_data_lower)

# I create a list of the *existing* skill-related tags, preserving their original capitalization from df_tags.
existing_skill_level_tags = [
    tag for tag in df_kernel_data['TagName'].unique()
    if tag.lower() in actual_tags_set and tag.lower() in skill_level_tags_lower
]

if not existing_skill_level_tags:
    print("\nNone of the specified 'Skill Level and Learning' tags were found in my dataset. I can't generate the plot.")
    print("Please double-check the spelling or capitalization of the tags against df_kernel_data['TagName'].unique().")
else:
    print(f"\n--- Plotting {len(existing_skill_level_tags)} 'Skill Level and Learning' tags. ---")
    print("The tags being plotted are:")
    for tag in existing_skill_level_tags:
        print(f"- {tag}")
    print("---------------------------------------------------------------")

    df_filtered_skill_levels = df_kernel_data[df_kernel_data['TagName'].isin(existing_skill_level_tags)].copy()

    if df_filtered_skill_levels.empty:
        print("\nMy filtered DataFrame for 'Skill Level and Learning' is empty. This might mean there are no kernels associated with these tags, or there are issues with the dates.")
    else:
        df_filtered_skill_levels['YearMonth'] = df_filtered_skill_levels['CreationDate'].dt.to_period('M')
        skill_level_popularity_over_time = df_filtered_skill_levels.groupby(['YearMonth', 'TagName']).size().reset_index(name='KernelCount')

        skill_level_popularity_over_time['YearMonth'] = skill_level_popularity_over_time['YearMonth'].dt.to_timestamp()

        #  Finally, I visualize the trends using an Interactive Line Plot with Plotly Express 
        fig_skill_levels = px.line(skill_level_popularity_over_time,
                                   x='YearMonth',
                                   y='KernelCount',
                                   color='TagName', # Color lines by TagName
                                   title='Evolution of Skill Level and Learning Tags in Kaggle Kernels',
                                   labels={'YearMonth': 'Date', 'KernelCount': 'Number of Kernels Created', 'TagName': 'Skill Level / Learning'},
                                   hover_data={'YearMonth': '|%Y-%m', 'KernelCount': True, 'TagName': True}, # Show data on hover
                                   line_shape="spline", # Smooth the lines for better visual flow
                                   markers=True, # Show markers at data points
                                   height=600, # Set a fixed height for the plot
                                   color_discrete_sequence=px.colors.qualitative.Plotly # Use a good qualitative palette
                                  )

        # Customize layout for better appearance and legend positioning
        fig_skill_levels.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Kernels Created",
            font=dict(size=11), # Slightly increased font size
            title_font_size=16, # Increased main title font size
            hovermode="x unified", # Shows values for all lines at a given x-position
            legend_title_text='Skill Level / Learning',
            legend=dict(
                orientation="h", # Horizontal legend
                yanchor="bottom",
                y=1.02, # Position above the plot
                xanchor="right",
                x=1,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.6)", # Slightly transparent background for legend
                bordercolor="LightSteelBlue",
                borderwidth=1
            ),
            margin=dict(l=60, r=60, t=100, b=60), # Adjust margins for titles/legends
            paper_bgcolor='white', # Ensure background is white for clarity
            plot_bgcolor='white'
        )

        # Update x-axis tick labels rotation
        fig_skill_levels.update_xaxes(tickangle=45, tickfont=dict(size=10))
        fig_skill_levels.update_yaxes(tickfont=dict(size=10))

        # Show the plot in Kaggle notebook
        fig_skill_levels.show(renderer="iframe")

       
        fig_skill_levels.write_html("/kaggle/working/interactive_skill_level_line_plot.html")

        print("\n--- My Analysis is Complete (Skill Level Line Plot) ---")
        print("The interactive line plot above shows the trend in the number of kernels created for each 'Skill Level and Learning' tag over time.")
        print("Hover over the lines to see specific values for dates and kernel counts.")
        print("This helps me identify patterns related to the influx of new users and the progression of existing users within the Kaggle community.")


# Only proceed if dataframes are not empty after loading attempt
if not df_kernels.empty and not df_kernel_tags.empty and not df_tags.empty:
    #  I preprocess my data ---
    # I convert 'CreationDate' to datetime objects for accurate time-series analysis.
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])

    # I merge KernelTags with Tags to get readable tag names, then merge with Kernels for creation dates.
    df_merged_tags = pd.merge(df_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='left')
    df_merged_tags.rename(columns={'Name': 'TagName'}, inplace=True)
    df_kernel_data = pd.merge(df_kernels, df_merged_tags, left_on='Id', right_on='KernelId', how='inner')
    df_kernel_data['TagName'] = df_kernel_data['TagName'].astype(str) # Ensure TagName is string

    #  I define the specific list of modern AI/ML sub-field tags I'm interested in ---
    modern_ai_ml_tags_raw = [
        "AutoML", "Model Explainability", "Transfer Learning"
    ]

    # I convert all my target tags to lowercase for robust matching, as Kaggle tags are often lowercase.
    modern_ai_ml_tags_lower = [tag.lower() for tag in modern_ai_ml_tags_raw]

    # I get unique tags present in my actual df_kernel_data (also lowercased for matching).
    actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

    #  I filter for only the modern AI/ML sub-field tags that *actually exist* in my data ---
    actual_tags_set = set(actual_tags_in_data_lower)

    # I create a list of the *existing* modern AI/ML sub-field tags, preserving their original capitalization from df_tags.
    existing_modern_ai_ml_tags = [
        tag for tag in df_kernel_data['TagName'].unique()
        if tag.lower() in actual_tags_set and tag.lower() in modern_ai_ml_tags_lower
    ]

    if not existing_modern_ai_ml_tags:
        print("\nNone of the specified 'Modern AI/ML Sub-field' tags were found in my dataset. I can't generate the plot.")
        print("Please double-check the spelling or capitalization of the tags against df_kernel_data['TagName'].unique().")
    else:
        print(f"\n--- Plotting {len(existing_modern_ai_ml_tags)} 'Modern AI/ML Sub-field' tags. ---")
        print("The tags being plotted are:")
        for tag in existing_modern_ai_ml_tags:
            print(f"- {tag}")
        print("---------------------------------------------------------------")

        df_filtered_modern_ai_ml = df_kernel_data[df_kernel_data['TagName'].isin(existing_modern_ai_ml_tags)].copy()

        if df_filtered_modern_ai_ml.empty:
            print("\nMy filtered DataFrame for 'Modern AI/ML Sub-fields' is empty. This might mean there are no kernels associated with these tags, or there are issues with the dates.")
        else:
            df_filtered_modern_ai_ml['YearMonth'] = df_filtered_modern_ai_ml['CreationDate'].dt.to_period('M')
            modern_ai_ml_popularity_over_time = df_filtered_modern_ai_ml.groupby(['YearMonth', 'TagName']).size().reset_index(name='KernelCount')

            modern_ai_ml_popularity_over_time['YearMonth'] = modern_ai_ml_popularity_over_time['YearMonth'].dt.to_timestamp()

            # Finally, I visualize the trends using an Interactive Line Plot with Plotly Express ---
            fig_modern_ai_ml = px.line(modern_ai_ml_popularity_over_time,
                                       x='YearMonth',
                                       y='KernelCount',
                                       color='TagName', # Color lines by TagName
                                       title='Evolution of Modern AI/ML Sub-fields in Kaggle Kernels (Kernel Count)',
                                       labels={'YearMonth': 'Date', 'KernelCount': 'Number of Kernels Created', 'TagName': 'AI/ML Sub-field'},
                                       hover_data={'YearMonth': '|%Y-%m', 'KernelCount': True, 'TagName': True}, # Show data on hover
                                       line_shape="spline", # Smooth the lines for better visual flow
                                       markers=True, # Show markers at data points
                                       height=600, # Set a fixed height for the plot
                                       color_discrete_sequence=px.colors.qualitative.Plotly # Use a good qualitative palette
                                      )

            # Customize layout for better appearance and legend positioning
            fig_modern_ai_ml.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Kernels Created",
                font=dict(size=11), # Slightly increased font size
                title_font_size=16, # Increased main title font size
                hovermode="x unified", # Shows values for all lines at a given x-position
                legend_title_text='AI/ML Sub-field',
                legend=dict(
                    orientation="h", # Horizontal legend
                    yanchor="bottom",
                    y=1.02, # Position above the plot
                    xanchor="right",
                    x=1,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.6)", # Slightly transparent background for legend
                    bordercolor="LightSteelBlue",
                    borderwidth=1
                ),
                margin=dict(l=60, r=60, t=100, b=60), # Adjust margins for titles/legends
                paper_bgcolor='white', # Ensure background is white for clarity
                plot_bgcolor='white'
            )

            # Update x-axis tick labels rotation
            fig_modern_ai_ml.update_xaxes(tickangle=45, tickfont=dict(size=10))
            fig_modern_ai_ml.update_yaxes(tickfont=dict(size=10))

            # Show the plot in Kaggle notebook
            fig_modern_ai_ml.show(renderer="iframe")

            
            

            print("\n--- My Analysis is Complete (Modern AI/ML Sub-fields Line Plot) ---")
            print("The interactive line plot above shows the trend in the number of kernels created for 'AutoML', 'Model Explainability', and 'Transfer Learning' over time.")
            print("Hover over the lines to see specific values for dates and kernel counts.")
            print("This helps me identify whether these areas are indeed showing a distinct upward trend, reflecting increasing industry and academic focus.")
else:
    print("\nMy DataFrames are empty after loading. I cannot proceed with the analysis.")


# Only proceed if dataframes are not empty after loading attempt
if not df_kernels.empty and not df_kernel_tags.empty and not df_tags.empty:
    # --- 2. Preprocess the data ---
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])

    # Merge KernelTags with Tags to get readable tag names, then merge with Kernels for creation dates.
    df_merged_tags = pd.merge(df_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='left')
    df_merged_tags.rename(columns={'Name': 'TagName'}, inplace=True)
    df_kernel_data = pd.merge(df_kernels, df_merged_tags, left_on='Id', right_on='KernelId', how='inner')
    df_kernel_data['TagName'] = df_kernel_data['TagName'].astype(str) # Ensure TagName is string

    # Define the specific "Solution Approach" tags you want to track ---
    solution_approach_tags_raw = [
        "Ensembling", "Gradient Boosting", "Neural Network", "Deep Learning",
        "Transfer Learning", "AutoML", "Reinforcement Learning",
        "Computer Vision", "Natural Language Processing",
        "Time Series", "Feature Engineering", "Clustering",
        "Regression", "Classification", "XGBoost", "LightGBM", "CatBoost",
        "Generative Adversarial Network", "Transformer", "LSTM", "RNN", "CNN"
    ]

    # Convert all target tags to lowercase for robust matching.
    solution_approach_tags_lower = [tag.lower() for tag in solution_approach_tags_raw]

    # Get unique tags present in the actual df_kernel_data (also lowercased for matching).
    actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

    # --- 4. Filter for only the solution approach tags that *actually exist* in your data ---
    actual_tags_set = set(actual_tags_in_data_lower)

    existing_solution_approach_tags = [
        tag for tag in df_kernel_data['TagName'].unique()
        if tag.lower() in actual_tags_set and tag.lower() in solution_approach_tags_lower
    ]

    if not existing_solution_approach_tags:
        print("\nNone of the specified 'Solution Approach' tags were found in the dataset. Cannot generate the plot.")
        print("Please review your list of tags or check their spelling against actual tags in df_kernel_data['TagName'].unique().")
    else:
        print(f"\n--- Plotting {len(existing_solution_approach_tags)} 'Solution Approach' tags. ---")
        print("The tags being plotted are:")
        for tag in existing_solution_approach_tags:
            print(f"- {tag}")
        print("---------------------------------------------------------------")

        df_filtered_solution_approaches = df_kernel_data[df_kernel_data['TagName'].isin(existing_solution_approach_tags)].copy()

        if df_filtered_solution_approaches.empty:
            print("\nFiltered DataFrame for 'Solution Approaches' is empty. This means no kernels are associated with these tags or date issues.")
        else:
            # Group by Year and TagName to count occurrences
            df_filtered_solution_approaches['Year'] = df_filtered_solution_approaches['CreationDate'].dt.year
            solution_approach_popularity_over_time = df_filtered_solution_approaches.groupby(['Year', 'TagName']).size().reset_index(name='KernelCount')

            #Visualize the Trends - Line Plot for Solution Approaches with Plotly Express ---
            fig_solution_approaches = px.line(solution_approach_popularity_over_time,
                                              x='Year',
                                              y='KernelCount',
                                              color='TagName', # Color lines by TagName
                                              title='Evolution of Solution Approaches in Kaggle Kernels (Kernel Count)',
                                              labels={'Year': 'Year', 'KernelCount': 'Number of Kernels Tagged', 'TagName': 'Solution Approach'},
                                              hover_data={'Year': True, 'KernelCount': True, 'TagName': True}, # Show data on hover
                                              line_shape="spline", # Smooth the lines for better visual flow
                                              markers=True, # Show markers at data points
                                              height=700, # Set a fixed height for clarity with many lines
                                              color_discrete_sequence=px.colors.qualitative.Plotly # Use a good qualitative palette
                                             )

            # Customize layout for better appearance and legend positioning
            fig_solution_approaches.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of Kernels Tagged",
                font=dict(size=11), # Slightly increased font size
                title_font_size=16, # Increased main title font size
                hovermode="x unified", # Shows values for all lines at a given x-position
                legend_title_text='Solution Approach',
                legend=dict(
                    orientation="h", # Horizontal legend
                    yanchor="bottom",
                    y=1.50, # Position above the plot
                    xanchor="right",
                    x=1,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.6)", # Slightly transparent background for legend
                    bordercolor="LightSteelBlue",
                    borderwidth=1,
                    traceorder="grouped" # Group similar legend items (e.g., if multiple colors per tag)
                ),
                margin=dict(l=60, r=60, t=100, b=100), # Adjust margins for titles/legends
                paper_bgcolor='white', # Ensure background is white for clarity
                plot_bgcolor='white'
            )

            # Update x-axis tick labels rotation and format (for years)
            # Ensuring all years with data are displayed as ticks
            all_years_with_data = sorted(solution_approach_popularity_over_time['Year'].unique())
            fig_solution_approaches.update_xaxes(
                tickvals=all_years_with_data, # Explicitly set tick values to all available years
                tickangle=45,
                tickfont=dict(size=10)
            )
            fig_solution_approaches.update_yaxes(tickfont=dict(size=10))

            # Show the plot in Kaggle notebook
            fig_solution_approaches.show(renderer="iframe")

        

            print("\n--- Analysis Complete (Solution Approaches Line Plot) ---")
            print("The interactive line plot above illustrates the evolution of various solution approaches on Kaggle over time.")
            print("Hover over the lines to see specific values for years and kernel counts. Click on legend items to toggle lines on/off.")
            print("By observing the trends of these tags, you can infer shifts in popular methodologies and techniques.")
else:
    print("\nDataFrames are empty after loading. Cannot proceed with analysis.")


# Only proceed if dataframes are not empty after loading attempt
if not df_kernels.empty and not df_kernel_tags.empty and not df_tags.empty:
    # --- 2. Preprocess the data ---
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])

    df_merged_tags = pd.merge(df_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='left')
    df_merged_tags.rename(columns={'Name': 'TagName'}, inplace=True)
    df_kernel_data = pd.merge(df_kernels, df_merged_tags, left_on='Id', right_on='KernelId', how='inner')

    # --- 3. Filter data for 2022 and 2024 ---
    # Current year is 2024, so we can filter up to now.
    df_2022 = df_kernel_data[df_kernel_data['CreationDate'].dt.year == 2022]
    df_2024 = df_kernel_data[df_kernel_data['CreationDate'].dt.year == 2024]

    if df_2022.empty and df_2024.empty:
        print("\nNo kernel data found for both 2022 and 2024. Cannot perform comparison.")
    else:
        # Count tags for each year ---
        tag_counts_2022 = df_2022['TagName'].value_counts().reset_index()
        tag_counts_2022.columns = ['TagName', 'Count_2022']

        tag_counts_2024 = df_2024['TagName'].value_counts().reset_index()
        tag_counts_2024.columns = ['TagName', 'Count_2024']

        #Merge the counts for comparison ---
        df_comparison = pd.merge(tag_counts_2022, tag_counts_2024, on='TagName', how='outer').fillna(0)

        # Calculate percentage change
        df_comparison['Percentage_Change'] = ((df_comparison['Count_2024'] - df_comparison['Count_2022']) / df_comparison['Count_2022']) * 100
        df_comparison.replace([float('inf'), -float('inf')], pd.NA, inplace=True) # Handle division by zero (new tags in 2024)

        # --- 6. Display results ---
        print("\n--- Top 10 Most Popular Tags in 2024 ---")
        print(tag_counts_2022.head(10).to_string(index=False))

        print("\n--- Top 10 Most Popular Tags in 2024 ---")
        print(tag_counts_2024.head(10).to_string(index=False))

        print("\n--- Tag Popularity Comparison (2022 vs. 2024) - Top 20 Common Tags by 2024 Count ---")
        # Sort by 2024 count to see what's currently popular and how it changed
        df_comparison_sorted = df_comparison.sort_values(by='Count_2024', ascending=False)
        print(df_comparison_sorted.head(20).to_string(index=False))

        print("\n--- Top 10 Tags with Highest Percentage Growth (Min 10 kernels in 2022) ---")
        df_growth = df_comparison_sorted[(df_comparison_sorted['Count_2022'] >= 10) & (df_comparison_sorted['Percentage_Change'] > 0)]
        print(df_growth.sort_values(by='Percentage_Change', ascending=False).head(10).to_string(index=False))

        print("\n--- Top 10 Tags with Highest Percentage Decline (Min 10 kernels in 2022) ---")
        df_decline = df_comparison_sorted[(df_comparison_sorted['Count_2022'] >= 10) & (df_comparison_sorted['Percentage_Change'] < 0)]
        print(df_decline.sort_values(by='Percentage_Change', ascending=True).head(10).to_string(index=False))

        print("\n--- Analysis Complete ---")
        print("The tables above provide a comparative view of kernel tag popularity between 2022 and 2024, showing overall popularity, growth, and decline.")
else:
    print("\nDataFrames are empty after loading. Cannot proceed with analysis.")


# Only proceed if dataframes are not empty after loading attempt
if not df_kernels.empty and not df_kernel_tags.empty and not df_tags.empty:
    # --- 2. Preprocess the data ---
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])

    # Merge Kernels with KernelTags to link kernels to their tags
    # We use 'inner' merge to ensure we only count kernels that actually have tags associated.
    df_kernel_with_tags = pd.merge(df_kernels, df_kernel_tags, left_on='Id', right_on='KernelId', how='inner')

    # Filter data for 2022 and 2025 ---
    # Current year is 2025, so we can filter up to now.
    df_2022_tags = df_kernel_with_tags[df_kernel_with_tags['CreationDate'].dt.year == 2022]
    df_2024_tags = df_kernel_with_tags[df_kernel_with_tags['CreationDate'].dt.year == 2024]

    # Count total tags for each year ---
    total_tags_2022 = len(df_2022_tags)
    total_tags_2024 = len(df_2024_tags)

    #Display results ---
    print("\n--- Total Kernel Tag Counts (2022 vs. 2025) ---")
    print(f"Total number of kernel tags applied in 2022: {total_tags_2022}")
    print(f"Total number of kernel tags applied in 2024 : {total_tags_2024}")

    # Calculate percentage change for the total count
    if total_tags_2022 > 0:
        percentage_change_total = ((total_tags_2024 - total_tags_2022) / total_tags_2022) * 100
        print(f"Percentage change from 2022 to 2024: {percentage_change_total:.2f}%")
    elif total_tags_2024 > 0:
        print("Percentage change: Infinite (No tags in 2022, but tags in 2024)")
    else:
        print("Percentage change: N/A (No tags in either year)")

    print("\n--- Analysis Complete ---")
    print("This shows the overall volume of tagged kernel content created in each year.")
else:
    print("\nMy DataFrames are empty after loading. I cannot proceed with the analysis.")


# Only proceed if dataframes are not empty after loading attempt
if not df_kernels.empty and not df_kernel_tags.empty and not df_tags.empty:
    # Preprocess the data ---
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])

    # Merge Kernels with KernelTags to link kernels to their tags
    # We use 'inner' merge to ensure we only count kernels that actually have tags associated.
    df_kernel_with_tags = pd.merge(df_kernels, df_kernel_tags, left_on='Id', right_on='KernelId', how='inner')

    #Filter data for 2020 and 2022 ---
    # Current year is 2025, so we can filter up to now.
    df_2020_tags = df_kernel_with_tags[df_kernel_with_tags['CreationDate'].dt.year == 2020]
    df_2022_tags = df_kernel_with_tags[df_kernel_with_tags['CreationDate'].dt.year == 2022]

    #  Count total tags for each year ---
    total_tags_2020 = len(df_2020_tags)
    total_tags_2022 = len(df_2022_tags)

    # Display results ---
    print("\n--- Total Kernel Tag Counts (2020 vs. 2022) ---")
    print(f"Total number of kernel tags applied in 2020: {total_tags_2020}")
    print(f"Total number of kernel tags applied in 2022 : {total_tags_2022}")

    # Calculate percentage change for the total count
    if total_tags_2022 > 0:
        percentage_change_total = ((total_tags_2022 - total_tags_2020) / total_tags_2020) * 100
        print(f"Percentage change from 2022 to 2024: {percentage_change_total:.2f}%")
    elif total_tags_20222 > 0:
        print("Percentage change: Infinite (No tags in 2020, but tags in 2022)")
    else:
        print("Percentage change: N/A (No tags in either year)")

    print("\n--- Analysis Complete ---")
    print("This shows the overall volume of tagged kernel content created in each year.")
else:
    print("\nMy DataFrames are empty after loading. I cannot proceed with the analysis.")


import pandas as pd
import plotly.express as px
import plotly.io as pio
import warnings

# Suppress specific FutureWarning messages for cleaner output (if you haven't already)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set default Plotly template for a nicer look
pio.templates.default = "plotly_white"

# Loading the necessary datasets (assuming df_kernels, df_kernel_tags, df_tags are already loaded or handled) ---
try:
    if 'df_kernels' not in locals() and 'df_kernels' not in globals():
        df_kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
        df_kernel_tags = pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')
        df_tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')
        print("Kernels and Tags datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Dataset not found. Please ensure that the file '{e.filename}' is in the correct path.")
    print("Don't forget to add the 'Meta Kaggle' dataset using the 'Add Data' button in the top right of your Kaggle Notebook.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# Only proceed if dataframes are not empty after loading attempt
if not df_kernels.empty and not df_kernel_tags.empty and not df_tags.empty:
    #  Preprocess the data ---
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])

    # Merge KernelTags with Tags to get readable tag names, then merge with Kernels for creation dates.
    df_merged_tags = pd.merge(df_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='left')
    df_merged_tags.rename(columns={'Name': 'TagName'}, inplace=True)
    
    # Ensure TagName column is string type before lowercasing
    df_merged_tags['TagName'] = df_merged_tags['TagName'].astype(str)

    # Recreate df_kernel_data for this section as well, just in case it's not globally available
    # or to ensure it's fresh after previous operations.
    df_kernel_data = pd.merge(df_kernels, df_merged_tags, left_on='Id', right_on='KernelId', how='inner')


    #Find the first kernel creation date for each user
    user_first_kernel_date = df_kernels.groupby('AuthorUserId')['CreationDate'].min().reset_index()
    user_first_kernel_date.rename(columns={'CreationDate': 'FirstKernelDate'}, inplace=True)

    # Merge back to kernels to mark 'first kernels' for each user
    df_kernels_with_first_date = pd.merge(df_kernels, user_first_kernel_date, on='AuthorUserId', how='left')
    df_novice_kernels_raw = df_kernels_with_first_date[df_kernels_with_first_date['CreationDate'] == df_kernels_with_first_date['FirstKernelDate']].copy()

    # Join with tags to get actual tag names for these novice contributions
    df_novice_kernel_tags = pd.merge(df_novice_kernels_raw, df_kernel_tags, left_on='Id', right_on='KernelId', how='inner')
    df_novice_kernel_tags = pd.merge(df_novice_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='inner')
    df_novice_kernel_tags.rename(columns={'Name': 'TagName'}, inplace=True)
    df_novice_kernel_tags['TagName'] = df_novice_kernel_tags['TagName'].astype(str) # Ensure TagName is string

    # Analyze trends of these tags over time (by the year of the user's first kernel)
    df_novice_kernel_tags['FirstKernelYear'] = df_novice_kernel_tags['FirstKernelDate'].dt.year

    # Filter for a relevant time period (e.g., 2015-2024, adjust as per your data range)
    # Get min/max year from the actual data to dynamically set filter range
    min_year_data = df_novice_kernel_tags['FirstKernelYear'].min()
    max_year_data = df_novice_kernel_tags['FirstKernelYear'].max()

    df_novice_kernel_tags_filtered = df_novice_kernel_tags[
        (df_novice_kernel_tags['FirstKernelYear'] >= min_year_data) & # Use dynamic min year
        (df_novice_kernel_tags['FirstKernelYear'] <= max_year_data)    # Use dynamic max year
    ]

    # Calculate top N tags for novice contributions by year
    novice_tag_trends = df_novice_kernel_tags_filtered.groupby(['FirstKernelYear', 'TagName']).size().reset_index(name='Count')

    # Plotting Suggestion ---
    # Example: Plotting trends for 'beginner' and 'exploratory data analysis' tags among novices
    # Filter for specific tags you want to visualize
    tags_to_plot = ['Beginner', 'Exploratory Data Analysis', 'Python'] # Use original capitalization for display
    # Convert tags_to_plot to lowercase for robust matching with data
    tags_to_plot_lower = [tag.lower() for tag in tags_to_plot]

    novice_selected_trends = novice_tag_trends[novice_tag_trends['TagName'].str.lower().isin(tags_to_plot_lower)].copy()

    # If there are no tags to plot after filtering, print a message and exit
    if novice_selected_trends.empty:
        print(f"\nNone of the selected tags ({', '.join(tags_to_plot)}) were found in novice kernel data for the specified years. Cannot generate plot.")
    else:
        # To ensure correct capitalization in the plot legend/tooltip, map back if necessary
        # This step might be redundant if df_novice_kernel_tags already has consistent capitalization from df_tags
        # But it's a good practice if your `tags_to_plot` list has a specific desired capitalization.
        tag_name_map = {tag.lower(): tag for tag in existing_skill_level_tags + existing_modern_ai_ml_tags + tags_to_plot}
        novice_selected_trends['TagName'] = novice_selected_trends['TagName'].str.lower().map(tag_name_map).fillna(novice_selected_trends['TagName'])


        # Create the Plotly Express line plot
        fig_novice_trends = px.line(novice_selected_trends,
                                    x='FirstKernelYear',
                                    y='Count',
                                    color='TagName', # Color lines by TagName
                                    title='Trends in Initial Kernel Tags by Novice Users on Kaggle',
                                    labels={'FirstKernelYear': 'Year of User\'s First Kernel', 'Count': 'Number of Kernels Tagged', 'TagName': 'Tag'},
                                    hover_data={'FirstKernelYear': True, 'Count': True, 'TagName': True}, # Show data on hover
                                    line_shape="spline", # Smooth the lines for better visual flow
                                    markers=True, # Show markers at data points
                                    height=600, # Set a fixed height for clarity
                                    color_discrete_sequence=px.colors.qualitative.Plotly # Use a good qualitative palette
                                   )

        # Customize layout for better appearance and legend positioning
        fig_novice_trends.update_layout(
            xaxis_title="Year of User's First Kernel",
            yaxis_title="Number of Kernels Tagged",
            font=dict(size=11), # Slightly increased font size
            title_font_size=16, # Increased main title font size
            hovermode="x unified", # Shows values for all lines at a given x-position
            legend_title_text='Initial Kernel Tag',
            legend=dict(
                orientation="h", # Horizontal legend
                yanchor="bottom",
                y=1.02, # Position above the plot
                xanchor="right",
                x=1,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.6)", # Slightly transparent background for legend
                bordercolor="LightSteelBlue",
                borderwidth=1
            ),
            margin=dict(l=60, r=60, t=100, b=60), # Adjust margins for titles/legends
            paper_bgcolor='white', # Ensure background is white for clarity
            plot_bgcolor='white'
        )

        # Update x-axis tick labels rotation and ensure all years with data are displayed
        all_years_novice = sorted(novice_selected_trends['FirstKernelYear'].unique())
        fig_novice_trends.update_xaxes(
            tickvals=all_years_novice, # Explicitly set tick values to all available years
            tickangle=45,
            tickfont=dict(size=10)
        )
        fig_novice_trends.update_yaxes(tickfont=dict(size=10))

        # Show the plot in Kaggle notebook
        fig_novice_trends.show(renderer="iframe")

        # If you want to save it as an interactive HTML file
        fig_novice_trends.write_html("/kaggle/working/interactive_novice_initial_kernel_tags_trend.html")

        print("\n--- Analysis Complete (Novice Initial Kernel Tags Trend Plot) ---")
        print("The interactive line plot above shows the trends in the use of specific tags in the very first kernels created by new Kaggle users over time.")
        print("Hover over the lines to see specific values for years and kernel counts. Click on legend items to toggle lines on/off.")
        print("This helps identify the initial focus areas and learning paths of new members joining the Kaggle community.")
else:
    print("\nDataFrames are empty after loading. Cannot proceed with analysis.")


# Only proceed if dataframes are not empty after loading attempt
if not df_kernels.empty and not df_kernel_tags.empty and not df_tags.empty:
    # Preprocess the data ---
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])

    # Merge KernelTags with Tags to get readable tag names, then merge with Kernels for creation dates.
    df_merged_tags = pd.merge(df_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='left')
    df_merged_tags.rename(columns={'Name': 'TagName'}, inplace=True)
    df_merged_tags['TagName'] = df_merged_tags['TagName'].astype(str) # Ensure TagName is string

    df_kernel_data = pd.merge(df_kernels, df_merged_tags, left_on='Id', right_on='KernelId', how='inner')
    df_kernel_data['TagName'] = df_kernel_data['TagName'].astype(str) # Ensure TagName is string

    print("\n--- Analyzing Novice User Contributions ---")

    # Find the first kernel creation date for each user
    user_first_kernel_date = df_kernels.groupby('AuthorUserId')['CreationDate'].min().reset_index()
    user_first_kernel_date.rename(columns={'CreationDate': 'FirstKernelDate'}, inplace=True)

    # Merge back to kernels to identify 'first kernels' for each user
    df_kernels_with_first_date = pd.merge(df_kernels, user_first_kernel_date, on='AuthorUserId', how='left')
    df_novice_kernels_raw = df_kernels_with_first_date[
        df_kernels_with_first_date['CreationDate'] == df_kernels_with_first_date['FirstKernelDate']
    ].copy()

    # Join with tags to get actual tag names for these novice contributions
    df_novice_kernel_tags = pd.merge(
        df_novice_kernels_raw[['Id', 'FirstKernelDate']], # Select only necessary columns
        df_kernel_data[['KernelId', 'TagName']], # Contains all kernel-tag mappings, ensure TagName is string here
        left_on='Id',
        right_on='KernelId',
        how='inner'
    )
    df_novice_kernel_tags['TagName'] = df_novice_kernel_tags['TagName'].astype(str) # Re-confirm TagName as string

    #Analyze trends of these tags over time (by the year of the user's first kernel)
    df_novice_kernel_tags['FirstKernelYear'] = df_novice_kernel_tags['FirstKernelDate'].dt.year

    # Filter for a relevant time period
    start_year = 2015
    # Use the maximum year available in the data for a dynamic end_year
    end_year = df_novice_kernel_tags['FirstKernelYear'].max()
    if pd.isna(end_year): # Handle case where there's no data
        print("\nNo novice kernel data found for any year. Cannot generate plots.")
        exit()
    
    df_novice_kernel_tags_filtered = df_novice_kernel_tags[
        (df_novice_kernel_tags['FirstKernelYear'] >= start_year) &
        (df_novice_kernel_tags['FirstKernelYear'] <= end_year)
    ]

    # Count tag occurrences per year
    novice_tag_trends = df_novice_kernel_tags_filtered.groupby(['FirstKernelYear', 'TagName']).size().reset_index(name='Count')

    # --- Select and Prepare Data for Plotting ---
    key_novice_tags_to_plot = [
        'Beginner', 'Python', 'Exploratory Data Analysis',
        'Data Cleaning', 'Classification', 'Deep Learning',
        'Data Visualization', 'Data Analytics', 'Pandas'
    ]
    # Convert tags_to_plot to lowercase for robust matching with data
    key_novice_tags_to_plot_lower = [tag.lower() for tag in key_novice_tags_to_plot]

    novice_selected_trends = novice_tag_trends[novice_tag_trends['TagName'].str.lower().isin(key_novice_tags_to_plot_lower)].copy()

    if novice_selected_trends.empty:
        print(f"\nNone of the selected tags ({', '.join(key_novice_tags_to_plot)}) were found in novice kernel data for the specified years. Cannot generate plots.")
    else:
        # To create a stacked bar chart (showing proportions) or percentage line plot,
        # it's often useful to normalize counts per year.
        total_novice_kernels_per_year = df_novice_kernels_raw.groupby(df_novice_kernels_raw['FirstKernelDate'].dt.year).size().reset_index(name='TotalKernels')
        total_novice_kernels_per_year.rename(columns={'FirstKernelDate': 'FirstKernelYear'}, inplace=True)
        total_novice_kernels_per_year = total_novice_kernels_per_year[
            (total_novice_kernels_per_year['FirstKernelYear'] >= start_year) &
            (total_novice_kernels_per_year['FirstKernelYear'] <= end_year)
        ]

        novice_selected_trends_normalized = pd.merge(novice_selected_trends, total_novice_kernels_per_year, on='FirstKernelYear', how='left')
        novice_selected_trends_normalized['Proportion'] = novice_selected_trends_normalized['Count'] / novice_selected_trends_normalized['TotalKernels']
        # For better readability in Plotly's default hover, let's also add percentage
        novice_selected_trends_normalized['Proportion_Percent'] = novice_selected_trends_normalized['Proportion'] * 100


        # --- Plotly Plot 1: Line Plot of absolute counts for selected tags ---
        fig_abs_counts = px.line(novice_selected_trends,
                                 x='FirstKernelYear',
                                 y='Count',
                                 color='TagName',
                                 title=f'Absolute Count of Key Initial Kernel Tags by Novice Users ({start_year}-{end_year})',
                                 labels={'FirstKernelYear': 'Year of User\'s First Kernel Creation', 'Count': 'Number of Kernels Tagged', 'TagName': 'Tag Name'},
                                 hover_data={'FirstKernelYear': True, 'Count': True, 'TagName': True},
                                 line_shape="spline",
                                 markers=True,
                                 height=600,
                                 color_discrete_sequence=px.colors.qualitative.Plotly
                                )

        fig_abs_counts.update_layout(
            xaxis_title="Year of User's First Kernel Creation",
            yaxis_title="Number of Kernels Tagged",
            font=dict(size=11),
            title_font_size=16,
            hovermode="x unified",
            legend_title_text='Tag Name',
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10),
                bgcolor="rgba(255,255,255,0.6)", bordercolor="LightSteelBlue", borderwidth=1
            ),
            margin=dict(l=60, r=60, t=100, b=60),
            paper_bgcolor='white', plot_bgcolor='white'
        )
        all_years_abs = sorted(novice_selected_trends['FirstKernelYear'].unique())
        fig_abs_counts.update_xaxes(tickvals=all_years_abs, tickangle=45, tickfont=dict(size=10))
        fig_abs_counts.update_yaxes(tickfont=dict(size=10))

        fig_abs_counts.show(renderer="iframe")
        fig_abs_counts.write_html("/kaggle/working/interactive_novice_initial_kernel_tags_abs_counts.html")


    print("\n--- Novice User Analysis Complete ---")
    print("The interactive plots above illustrate the types of content new Kaggle users are creating as their initial contributions over the years.")
    print("The first plot shows absolute counts, while the second stacked area plot reveals the changing proportions of these tags, allowing you to see shifts in trends among novice contributions.")
    print("Hover over the plots for detailed information on each point.")
else:
    print("\nDataFrames are empty after loading. Cannot proceed with analysis.")


if not df_kernels.empty:
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])
    df_kernels['CreationYear'] = df_kernels['CreationDate'].dt.year

    # Calculate total fork numbers by year
    fork_counts_by_year = df_kernels[df_kernels['ForkParentKernelVersionId'].notna()] \
                                     .groupby('CreationYear').size().reset_index(name='ForkCount')

    # Calculate total kernel counts by year (needed for averaging but not used here)
    total_kernels_by_year = df_kernels.groupby('CreationYear').size().reset_index(name='TotalKernels')

    # Combine for average forks (only ForkCount is sufficient for this chart)
    fork_analysis = pd.merge(fork_counts_by_year, total_kernels_by_year, on='CreationYear', how='left')

    
    min_year_data = fork_analysis['CreationYear'].min()
    max_year_data = fork_analysis['CreationYear'].max()

    fork_analysis_filtered = fork_analysis[
        (fork_analysis['CreationYear'] >= max(2015, min_year_data)) &
        (fork_analysis['CreationYear'] <= max_year_data)
    ]

    if fork_analysis_filtered.empty:
        print("\nFiltered fork analysis data is empty. Charts cannot be generated. Check year range or data availability.")
    else:
        # --- Plotly Graph: Total Kernel Fork Counts ---
        print("\n--- Plotting Total Kernel Fork Numbers ---")
        fig_total_forks = px.line(fork_analysis_filtered,
                                  x='CreationYear',
                                  y='ForkCount',  
                                  title=f'Total Number of Kernel Forks on Kaggle ({max(2015, min_year_data)}-{max_year_data})',
                                  labels={'CreationYear': 'Year', 'ForkCount': 'count'},
                                  hover_data={'CreationYear': True, 'ForkCount': True, 'TotalKernels': True},
                                  line_shape="spline",
                                  markers=True,
                                  height=600,
                                  color_discrete_sequence=px.colors.qualitative.Plotly
                                 )

        fig_total_forks.update_layout(
            xaxis_title="Yeat",
            yaxis_title="Fork count",
            font=dict(size=11),
            title_font_size=16,
            hovermode="x unified",
            margin=dict(l=60, r=60, t=100, b=60),
            paper_bgcolor='white', plot_bgcolor='white'
        )
        all_years_forks = sorted(fork_analysis_filtered['CreationYear'].unique())
        fig_total_forks.update_xaxes(tickvals=all_years_forks, tickangle=45, tickfont=dict(size=10))
        fig_total_forks.update_yaxes(tickfont=dict(size=10))

        fig_total_forks.show(renderer="iframe")
        fig_total_forks.write_html("/kaggle/working/interactive_total_forks_onkaggle.html")
        print("Successfully created Total Kernel Fork Counts chart.")
else:
    print("\nThe df_kernels data frame is empty. Analysis cannot continue.")


if not df_kernels.empty:
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])
    df_kernels['CreationYear'] = df_kernels['CreationDate'].dt.year

    # Calculate total fork numbers by year
    fork_counts_by_year = df_kernels[df_kernels['ForkParentKernelVersionId'].notna()] \
                                     .groupby('CreationYear').size().reset_index(name='ForkCount')

    #Calculate total kernel numbers by year
    total_kernels_by_year = df_kernels.groupby('CreationYear').size().reset_index(name='TotalKernels')

    # Calculate the average number of forks per kernel
    fork_analysis = pd.merge(fork_counts_by_year, total_kernels_by_year, on='CreationYear', how='left')
    fork_analysis['AvgForksPerKernel'] = fork_analysis['ForkCount'] / fork_analysis['TotalKernels']

    # Filter the relevant time period for the plot
    min_year_data = fork_analysis['CreationYear'].min()
    max_year_data = fork_analysis['CreationYear'].max()

    fork_analysis_filtered = fork_analysis[
        (fork_analysis['CreationYear'] >= max(2015, min_year_data)) &
        (fork_analysis['CreationYear'] <= max_year_data)
    ]

    if fork_analysis_filtered.empty:
        print("\nFiltered fork analysis data is empty. Charts cannot be generated. Check year range or data availability.")
    else:
        # Plotting Average Number of Forks per Kernel
        print("\n--- Plotting Average Number of Forks per Kernel ---")
        fig_avg_forks = px.line(fork_analysis_filtered,
                                x='CreationYear',
                                y='AvgForksPerKernel',
                                title=f'Average Number of Forks per Kernel({max(2015, min_year_data)}-{max_year_data})', 
                                labels={'CreationYear': 'Year', 'AvgForksPerKernel': 'Average Number of Forks'}, 
                                hover_data={'CreationYear': True, 'AvgForksPerKernel': ':.2f', 'ForkCount': True, 'TotalKernels': True},
                                line_shape="spline",
                                markers=True,
                                height=600,
                                color_discrete_sequence=px.colors.qualitative.Plotly
                               )

        fig_avg_forks.update_layout(
            xaxis_title="Year",
            yaxis_title="Average Number of Forks",
            font=dict(size=11),
            title_font_size=16,
            hovermode="x unified",
            margin=dict(l=60, r=60, t=100, b=60),
            paper_bgcolor='white', plot_bgcolor='white'
        )
        all_years_forks = sorted(fork_analysis_filtered['CreationYear'].unique())
        fig_avg_forks.update_xaxes(tickvals=all_years_forks, tickangle=45, tickfont=dict(size=10))
        fig_avg_forks.update_yaxes(tickfont=dict(size=10))

        fig_avg_forks.show(renderer="iframe")
        fig_avg_forks.write_html("/kaggle/working/interactive_average_forks_onkaggle.html")
        print("Successfully created Average Number of Forks per Kernel chart.")
else:
    print("\nThe df_kernels data frame is empty. Analysis cannot continue.")


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
print("\n--- Analyzing Characteristics of Most Interacted-With Kernels ---")

# Define parameters for filtering top kernels
start_year = 2015
end_year = 2024 # Current year for your analysis
interaction_metric = 'TotalVotes' # Can also be 'TotalComments' or 'TotalViews'
top_percentage = 0.01 # Top 1% of kernels by interaction metric each year

# Function to get top N percentage of kernels by a given metric per year
def get_top_kernels_by_metric_yearly_percentage(df, metric, percentage_threshold, start_year, end_year):
    df_filtered_years = df[(df['CreationYear'] >= start_year) & (df['CreationYear'] <= end_year)].copy()
    top_kernels_list = []

    for year in sorted(df_filtered_years['CreationYear'].unique()):
        yearly_kernels = df_filtered_years[df_filtered_years['CreationYear'] == year].copy()
        if not yearly_kernels.empty:
            # Calculate the threshold value for the top percentage
            threshold_value = yearly_kernels[metric].quantile(1 - percentage_threshold)
            # Select kernels that meet or exceed this threshold
            top_yearly_kernels = yearly_kernels[yearly_kernels[metric] >= threshold_value]
            top_kernels_list.append(top_yearly_kernels)
            
    return pd.concat(top_kernels_list) if top_kernels_list else pd.DataFrame()

# Get the top kernels based on the defined metric and percentage
df_top_kernels_yearly = get_top_kernels_by_metric_yearly_percentage(
    df_kernels, interaction_metric, top_percentage, start_year, end_year
)

# Extract tags for these top kernels
df_top_kernel_tags = pd.merge(
    df_top_kernels_yearly[['Id', 'CreationYear']], # Relevant columns from top kernels
    df_kernel_data[['KernelId', 'TagName']],        # All kernel-tag mappings
    left_on='Id',
    right_on='KernelId',
    how='inner'
)

# Count tag occurrences in top kernels per year
top_kernel_tag_trends = df_top_kernel_tags.groupby(['CreationYear', 'TagName']).size().reset_index(name='Count')

# --- Prepare data for plotting ---
# First, find the overall top N tags from the 'top_kernel_tag_trends' for better representation
overall_top_tags_in_top_kernels = top_kernel_tag_trends.groupby('TagName')['Count'].sum().nlargest(10).index.tolist()
# Add specific tags if they are not in top N but are crucial for your story
key_tags_for_plot = list(set(overall_top_tags_in_top_kernels + ['deep learning', 'gpu', 'exploratory data analysis', 'classification', 'tabular', 'python']))

top_kernel_selected_trends = top_kernel_tag_trends[top_kernel_tag_trends['TagName'].isin(key_tags_for_plot)]

# Calculate proportions for stacked area plot
total_tags_in_top_kernels_per_year = df_top_kernel_tags.groupby('CreationYear').size().reset_index(name='TotalTags')

top_kernel_selected_trends_normalized = pd.merge(
    top_kernel_selected_trends,
    total_tags_in_top_kernels_per_year,
    on='CreationYear',
    how='left'
)
top_kernel_selected_trends_normalized['Proportion'] = top_kernel_selected_trends_normalized['Count'] / top_kernel_selected_trends_normalized['TotalTags']

# Pivot for stacked area plot
top_kernel_proportions_pivot = top_kernel_selected_trends_normalized.pivot(
    index='CreationYear', columns='TagName', values='Proportion'
).fillna(0)

# Filter for the plotting period
top_kernel_proportions_pivot = top_kernel_proportions_pivot[
    (top_kernel_proportions_pivot.index >= start_year) &
    (top_kernel_proportions_pivot.index <= end_year)
]

# --- Plotting with Plotly ---

if not top_kernel_selected_trends.empty:
    print(f"\n--- Plotting Key Tag Trends in Top {top_percentage*100:.0f}% Kernels ---")

    # Plot 1: Interactive Line Plot of absolute counts for selected top tags
    fig1 = px.line(
        top_kernel_selected_trends,
        x='CreationYear',
        y='Count',
        color='TagName',
        title=f'Absolute Count of Key Tags in Top {top_percentage*100:.0f}% Kernels by {interaction_metric} ({start_year}-{end_year})',
        labels={'CreationYear': 'Year', 'Count': 'Number of Kernels Tagged', 'TagName': 'Tag Name'},
        markers=True,
        hover_data={'CreationYear': True, 'Count': True, 'TagName': True},
        line_shape="spline",
        height=600,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig1.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Kernels Tagged",
        font=dict(size=11),
        title_font_size=16,
        hovermode="x unified", # Unified hover for all lines at a given x
        margin=dict(l=60, r=60, t=100, b=60),
        paper_bgcolor='white', plot_bgcolor='white',
        legend_title_text='Tag Name'
    )
    all_years_top_kernels = sorted(top_kernel_selected_trends['CreationYear'].unique())
    fig1.update_xaxes(
        tickvals=all_years_top_kernels,
        tickangle=45,
        tickfont=dict(size=10),
        range=[start_year - 0.5, end_year + 0.5] # Ensure full year range is visible
    )
    fig1.update_yaxes(tickfont=dict(size=10))

    fig1.show(renderer="iframe")
    fig1.write_html(f"/kaggle/working/interactive_top_kernel_tag_counts_by_{interaction_metric}.html")
    print(f"Successfully created interactive line chart for key tag counts.")


else:
    print("\nFiltered top kernels data is empty. Charts cannot be generated. Check year range, interaction metric, or data availability.")

print("\n--- Most Interacted-With Kernels Analysis Complete ---")
print(f"The interactive plots above illustrate the changing characteristics of the top {top_percentage*100:.0f}% most interacted-with kernels by {interaction_metric} over time.")






meta_kaggle_path = '/kaggle/input/meta-kaggle/'

try:
 
    df_tags = pd.read_csv(f'{meta_kaggle_path}Tags.csv')
    df_kernel_versions = pd.read_csv(f'{meta_kaggle_path}KernelVersions.csv')
    df_kernel_comp_sources = pd.read_csv(f'{meta_kaggle_path}KernelVersionCompetitionSources.csv')
    df_kernel_languages = pd.read_csv(f'{meta_kaggle_path}KernelLanguages.csv') # To infer problem domain via common languages in specific fields

    print("necessary Meta Kaggle CSVs loaded successfully!")

except FileNotFoundError as e:
    print(f"Error loading CSV: {e}. Please ensure the Meta Kaggle dataset is unzipped and located at: {meta_kaggle_path}")
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")



import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Defining the path to our Meta Kaggle dataset
meta_kaggle_path = '/kaggle/input/meta-kaggle/'

#  Load necessary CSVs 
try:
    df_competitions = pd.read_csv(f'{meta_kaggle_path}Competitions.csv')
    df_datasets = pd.read_csv(f'{meta_kaggle_path}Datasets.csv')
    df_dataset_versions = pd.read_csv(f'{meta_kaggle_path}DatasetVersions.csv') # Correctly loaded
    df_competition_tags = pd.read_csv(f'{meta_kaggle_path}CompetitionTags.csv')
    df_dataset_tags = pd.read_csv(f'{meta_kaggle_path}DatasetTags.csv')
    df_tags = pd.read_csv(f'{meta_kaggle_path}Tags.csv') # Make sure this line exists if df_tags is used later

    print("Necessary Meta Kaggle CSVs loaded successfully!")

    # --- Preprocessing ---
    # Convert dates to datetime objects and extract year
    df_competitions['EnabledDate'] = pd.to_datetime(df_competitions['EnabledDate'])
    df_competitions['Year'] = df_competitions['EnabledDate'].dt.year

    df_datasets['CreationDate'] = pd.to_datetime(df_datasets['CreationDate'])
    df_datasets['Year'] = df_datasets['CreationDate'].dt.year

    # Filter data from a reasonable start year to focus on modern AI/ML trends
    start_year = 2015
    end_year = 2024 # Defined for consistent use across plots

    df_competitions_filtered = df_competitions[df_competitions['Year'] >= start_year].copy()
    df_datasets_filtered = df_datasets[df_datasets['Year'] >= start_year].copy()

    # --- Crucial: Process df_dataset_versions for filtering if needed ---
    df_dataset_versions['CreationDate'] = pd.to_datetime(df_dataset_versions['CreationDate'])
    df_dataset_versions['Year'] = df_dataset_versions['CreationDate'].dt.year
    df_dataset_versions_filtered = df_dataset_versions[df_dataset_versions['Year'] >= start_year].copy()

    # --- DEBUGGING TAG MAPPING ---
    print("\n--- Debugging Tag Mapping ---")
    print(f"Shape of df_tags: {df_tags.shape}")
    print(f"Columns of df_tags: {df_tags.columns.tolist()}")

    # Change 'TagName' to 'Name' for df_tags
    if 'Id' not in df_tags.columns or 'Name' not in df_tags.columns:
        raise ValueError("df_tags must contain 'Id' and 'Name' columns.")

    df_tags_map = df_tags.set_index('Id')['Name'].to_dict()
    
    print(f"Size of df_tags_map (unique tags): {len(df_tags_map)}")
    if len(df_tags_map) == 0:
        print("WARNING: df_tags_map is empty. This will cause issues with mapping.")
    else:
        # Print a sample of the map to ensure it's populated
        print(f"Sample of df_tags_map: {list(df_tags_map.items())[:5]}...")

    # --- Apply and check mapping for competition tags ---
    print(f"\nShape of df_competition_tags: {df_competition_tags.shape}")
    print(f"Columns of df_competition_tags: {df_competition_tags.columns.tolist()}")
    if 'TagId' not in df_competition_tags.columns:
        raise ValueError("df_competition_tags must contain 'TagId' column.")

    df_competition_tags['TagName'] = df_competition_tags['TagId'].map(df_tags_map)
    print(f"df_competition_tags['TagName'] head after map:\n{df_competition_tags['TagName'].head()}")
    print(f"df_competition_tags['TagName'] has {df_competition_tags['TagName'].isnull().sum()} NaN values.")
    if df_competition_tags['TagName'].isnull().all():
        print("CRITICAL: All 'TagName' values in df_competition_tags are NaN. Mapping failed completely.")


    # --- Apply and check mapping for dataset tags ---
    print(f"\nShape of df_dataset_tags: {df_dataset_tags.shape}")
    print(f"Columns of df_dataset_tags: {df_dataset_tags.columns.tolist()}")
    if 'TagId' not in df_dataset_tags.columns:
        raise ValueError("df_dataset_tags must contain 'TagId' column.")

    df_dataset_tags['TagName'] = df_dataset_tags['TagId'].map(df_tags_map)
    print(f"df_dataset_tags['TagName'] head after map:\n{df_dataset_tags['TagName'].head()}")
    print(f"df_dataset_tags['TagName'] has {df_dataset_tags['TagName'].isnull().sum()} NaN values.")
    if df_dataset_tags['TagName'].isnull().all():
        print("CRITICAL: All 'TagName' values in df_dataset_tags are NaN. Mapping failed completely.")

    print("\n--- End Debugging Tag Mapping ---")

    # --- Plotting Median Dataset Version Size (MB) by Year ---
    print("\n--- Analyzing Median Dataset Version Size by Year ---")

    if 'TotalCompressedBytes' in df_dataset_versions_filtered.columns:
        # Ensure 'TotalCompressedBytes' is numeric and filter out 0 bytes or very small sizes
        df_dataset_versions_filtered['TotalCompressedBytes'] = pd.to_numeric(df_dataset_versions_filtered['TotalCompressedBytes'], errors='coerce')
        df_dataset_versions_filtered.dropna(subset=['TotalCompressedBytes'], inplace=True)

        # Convert to MB and filter for sizes greater than 0
        df_dataset_versions_filtered['TotalCompressedMB'] = df_dataset_versions_filtered['TotalCompressedBytes'] / (1024**2)
        
        # Calculate median dataset version size by year
        median_dataset_version_size_mb = df_dataset_versions_filtered[
            df_dataset_versions_filtered['TotalCompressedMB'] > 0
        ].groupby('Year')['TotalCompressedMB'].median().reset_index() # .reset_index() is crucial for px.line

        if not median_dataset_version_size_mb.empty:
            # Plotting with Plotly Express
            fig_median_size = px.line(median_dataset_version_size_mb,
                                      x='Year',
                                      y='TotalCompressedMB',
                                      title=f'Median Size of Dataset Versions on Kaggle Over Time (MB) ({start_year}-{end_year})',
                                      labels={'Year': 'Year', 'TotalCompressedMB': 'Median Dataset Version Size (MB)'},
                                      markers=True,
                                      line_shape="spline",
                                      height=600,
                                      color_discrete_sequence=px.colors.qualitative.Plotly,
                                      hover_data={'TotalCompressedMB': ':.2f'} # Format to 2 decimal places on hover
                                     )

            fig_median_size.update_layout(
                xaxis_title="Year",
                yaxis_title="Median Dataset Version Size (MB)",
                font=dict(size=11),
                title_font_size=20,
                hovermode="x unified",
                margin=dict(l=60, r=60, t=100, b=60),
                paper_bgcolor='white', plot_bgcolor='white'
            )
            # Ensure x-axis ticks show all relevant years
            all_years_size_data = sorted(median_dataset_version_size_mb['Year'].unique())
            fig_median_size.update_xaxes(
                tickvals=all_years_size_data,
                tickangle=45,
                tickfont=dict(size=10),
                range=[start_year - 0.5, end_year + 0.5]
            )
            fig_median_size.update_yaxes(tickfont=dict(size=10))

            fig_median_size.show(renderer="iframe")
            fig_median_size.write_html("/kaggle/working/interactive_median_dataset_version_size.html")
            print("Successfully created interactive Median Dataset Version Size (MB) by Year chart.")
        else:
            print("Filtered median dataset version size data is empty. Cannot generate plot.")
    else:
        print("Warning: 'TotalCompressedBytes' column not found in df_dataset_versions_filtered. Cannot plot size trends.")

    print("\n--- Dataset Version Size Analysis Complete ---")

except FileNotFoundError as e:
    print(f"Error loading CSV: {e}. Please ensure the Meta Kaggle dataset is unzipped and located at: {meta_kaggle_path}")
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")


# Suppress specific warnings that might arise from data loading or manipulation if they are expected
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Define the path to  Meta Kaggle dataset
meta_kaggle_path = '/kaggle/input/meta-kaggle/'

#  Load necessary CSVs 
try:
    df_competitions = pd.read_csv(f'{meta_kaggle_path}Competitions.csv')
    df_datasets = pd.read_csv(f'{meta_kaggle_path}Datasets.csv')
    df_dataset_versions = pd.read_csv(f'{meta_kaggle_path}DatasetVersions.csv')
    df_competition_tags = pd.read_csv(f'{meta_kaggle_path}CompetitionTags.csv')
    df_dataset_tags = pd.read_csv(f'{meta_kaggle_path}DatasetTags.csv')
    df_tags = pd.read_csv(f'{meta_kaggle_path}Tags.csv')

    print("Necessary Meta Kaggle CSVs loaded successfully!")

    # --- Preprocessing ---
    # Convert dates to datetime objects and extract year
    df_competitions['EnabledDate'] = pd.to_datetime(df_competitions['EnabledDate'])
    df_competitions['Year'] = df_competitions['EnabledDate'].dt.year

    df_datasets['CreationDate'] = pd.to_datetime(df_datasets['CreationDate'])
    df_datasets['Year'] = df_datasets['CreationDate'].dt.year

    # Filter data from a reasonable start year to focus on modern AI/ML trends
    start_year = 2015
    end_year = 2024 # Current year for our analysis, defined for consistency

    df_competitions_filtered = df_competitions[df_competitions['Year'] >= start_year].copy()
    df_datasets_filtered = df_datasets[df_datasets['Year'] >= start_year].copy()

    # --- Crucial: Process df_dataset_versions for filtering if needed ---
    df_dataset_versions['CreationDate'] = pd.to_datetime(df_dataset_versions['CreationDate'])
    df_dataset_versions['Year'] = df_dataset_versions['CreationDate'].dt.year
    df_dataset_versions_filtered = df_dataset_versions[df_dataset_versions['Year'] >= start_year].copy()

    # --- DEBUGGING TAG MAPPING (from your previous code) ---
    print("\n--- Debugging Tag Mapping ---")
    if 'Id' not in df_tags.columns or 'Name' not in df_tags.columns:
        raise ValueError("df_tags must contain 'Id' and 'Name' columns.")

    df_tags_map = df_tags.set_index('Id')['Name'].to_dict()
    
    print(f"Size of df_tags_map (unique tags): {len(df_tags_map)}")
    if len(df_tags_map) == 0:
        print("WARNING: df_tags_map is empty. This will cause issues with mapping.")
    else:
        print(f"Sample of df_tags_map: {list(df_tags_map.items())[:5]}...")

    # Apply and check mapping for competition tags
    if 'TagId' not in df_competition_tags.columns:
        raise ValueError("df_competition_tags must contain 'TagId' column.")
    df_competition_tags['TagName'] = df_competition_tags['TagId'].map(df_tags_map)
    if df_competition_tags['TagName'].isnull().all():
        print("CRITICAL: All 'TagName' values in df_competition_tags are NaN. Mapping failed completely.")

    # Apply and check mapping for dataset tags
    if 'TagId' not in df_dataset_tags.columns:
        raise ValueError("df_dataset_tags must contain 'TagId' column.")
    df_dataset_tags['TagName'] = df_dataset_tags['TagId'].map(df_tags_map)
    if df_dataset_tags['TagName'].isnull().all():
        print("CRITICAL: All 'TagName' values in df_dataset_tags are NaN. Mapping failed completely.")

    print("\n--- End Debugging Tag Mapping ---")

    # --- 3. Percentage distribution of different data types in datasets (image, text, tabular, time series) ---
    print("\n--- Analyzing Dataset Type Distribution ---")

    # Define categories for data types based on common tags
    data_type_mapping = {
        'computer vision': 'Image/Vision',
        'image': 'Image/Vision',
        'natural language processing': 'Text/NLP',
        'nlp': 'Text/NLP',
        'text': 'Text/NLP',
        'tabular': 'Tabular',
        'time series': 'Time Series',
        'audio': 'Audio',
        'geospatial': 'Geospatial',
        # Add other broad categories as needed.
    }

    # Merge datasets with their tags
    # Ensure df_datasets_filtered is merged, which is already filtered by year
    datasets_with_tags = pd.merge(df_datasets_filtered[['Id', 'Year']], df_dataset_tags, left_on='Id', right_on='DatasetId', how='inner')

    # Map specific tags to broader data types
    datasets_with_tags['BroadDataType'] = datasets_with_tags['TagName'].map(data_type_mapping)
    datasets_with_tags_filtered = datasets_with_tags.dropna(subset=['BroadDataType'])

    # Count occurrences of broad data types per year
    data_type_counts = datasets_with_tags_filtered.groupby(['Year', 'BroadDataType']).size().unstack(fill_value=0)

    # Filter data_type_counts to ensure it's within the desired start_year and end_year
    data_type_counts = data_type_counts[(data_type_counts.index >= start_year) & (data_type_counts.index <= end_year)]


    if not data_type_counts.empty and data_type_counts.sum().sum() > 0: # Check if there's any data to plot
        # Calculate proportions (sum along columns for each year)
        data_type_proportions = data_type_counts.apply(lambda x: x / x.sum(), axis=1)

        print("\nDataset Type Proportions by Year:")
        print(data_type_proportions.tail())

        # Prepare data for Plotly (px.area prefers long format or pivoted wide format)
        # We already have data_type_proportions as a wide format, which px.area can handle
        # Make sure the index (Year) is a column for px.area
        data_type_proportions_plot = data_type_proportions.reset_index()

        fig_data_types = px.area(
            data_type_proportions_plot,
            x='Year',
            y=data_type_proportions_plot.columns[1:], # All columns except 'Year' are data types
            title=f'Proportion of Dataset Types on Kaggle Over Time ({start_year}-{end_year})',
            labels={'value': 'Proportion', 'variable': 'Data Type'},
            height=600,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            line_shape="spline"
        )

        fig_data_types.update_layout(
            xaxis_title="Year",
            yaxis_title="Proportion of Datasets (%)",
            font=dict(size=11),
            title_font_size=20,
            hovermode="x unified", # Unified hover for stacked area
            margin=dict(l=60, r=60, t=100, b=60),
            paper_bgcolor='white', plot_bgcolor='white',
            legend_title_text='Data Type'
        )
        all_years_data_types = sorted(data_type_proportions_plot['Year'].unique())
        fig_data_types.update_xaxes(
            tickvals=all_years_data_types,
            tickangle=45,
            tickfont=dict(size=10),
            range=[start_year - 0.5, end_year + 0.5]
        )
        fig_data_types.update_yaxes(tickfont=dict(size=10), tickformat=".0%", range=[0, 1])

        # Customize hovertemplate for stacked area to show percentage
        fig_data_types.for_each_trace(lambda trace: trace.update(
            hovertemplate='<b>Year</b>: %{x}<br>' +
                          '<b>Data Type</b>: %{fullData.name}<br>' +
                          '<b>Proportion</b>: %{y:.1%}<extra></extra>'
        ))

        fig_data_types.show(renderer="iframe")
        fig_data_types.write_html("/kaggle/working/interactive_dataset_type_distribution.html")
        print("Successfully created interactive stacked area chart for dataset type distribution.")
    else:
        print("No relevant data found for dataset type distribution within the specified years.")

    # For Dataset Sources (corporate, scientific, public):
    print("\n**Note:** Inferring dataset sources (corporate, scientific, public) programmatically from Meta Kaggle CSVs is highly challenging. It would require complex analysis of creator organizations or extensive text mining of dataset metadata.")

except FileNotFoundError as e:
    print(f"Error loading CSV: {e}. Please ensure the Meta Kaggle dataset is unzipped and located at: {meta_kaggle_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings


warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

meta_kaggle_path = '/kaggle/input/meta-kaggle/'

# --- Load necessary CSVs ---
try:
    df_competitions = pd.read_csv(f'{meta_kaggle_path}Competitions.csv')
    df_datasets = pd.read_csv(f'{meta_kaggle_path}Datasets.csv')
    df_dataset_versions = pd.read_csv(f'{meta_kaggle_path}DatasetVersions.csv')
    df_competition_tags = pd.read_csv(f'{meta_kaggle_path}CompetitionTags.csv')
    df_dataset_tags = pd.read_csv(f'{meta_kaggle_path}DatasetTags.csv')
    df_tags = pd.read_csv(f'{meta_kaggle_path}Tags.csv')

    print("Necessary Meta Kaggle CSVs loaded successfully!")

    # --- Preprocessing ---
    # Convert dates to datetime objects and extract year
    df_competitions['EnabledDate'] = pd.to_datetime(df_competitions['EnabledDate'])
    df_competitions['Year'] = df_competitions['EnabledDate'].dt.year

    df_datasets['CreationDate'] = pd.to_datetime(df_datasets['CreationDate'])
    df_datasets['Year'] = df_datasets['CreationDate'].dt.year

    # Filter data from a reasonable start year to focus on modern AI/ML trends
    start_year = 2015
    end_year = 2024 # Current year for our analysis, defined for consistency

    df_competitions_filtered = df_competitions[df_competitions['Year'] >= start_year].copy()
    df_datasets_filtered = df_datasets[df_datasets['Year'] >= start_year].copy()

    # --- Crucial: Process df_dataset_versions for filtering if needed ---
    df_dataset_versions['CreationDate'] = pd.to_datetime(df_dataset_versions['CreationDate'])
    df_dataset_versions['Year'] = df_dataset_versions['CreationDate'].dt.year
    df_dataset_versions_filtered = df_dataset_versions[df_dataset_versions['Year'] >= start_year].copy()

    # --- DEBUGGING TAG MAPPING (from our previous code) ---
    print("\n--- Debugging Tag Mapping ---")
    if 'Id' not in df_tags.columns or 'Name' not in df_tags.columns:
        raise ValueError("df_tags must contain 'Id' and 'Name' columns.")

    df_tags_map = df_tags.set_index('Id')['Name'].to_dict()
    
    print(f"Size of df_tags_map (unique tags): {len(df_tags_map)}")
    if len(df_tags_map) == 0:
        print("WARNING: df_tags_map is empty. This will cause issues with mapping.")
    else:
        print(f"Sample of df_tags_map: {list(df_tags_map.items())[:5]}...")

    # Apply and check mapping for competition tags
    if 'TagId' not in df_competition_tags.columns:
        raise ValueError("df_competition_tags must contain 'TagId' column.")
    df_competition_tags['TagName'] = df_competition_tags['TagId'].map(df_tags_map)
    if df_competition_tags['TagName'].isnull().all():
        print("CRITICAL: All 'TagName' values in df_competition_tags are NaN. Mapping failed completely.")

    # Apply and check mapping for dataset tags
    if 'TagId' not in df_dataset_tags.columns:
        raise ValueError("df_dataset_tags must contain 'TagId' column.")
    df_dataset_tags['TagName'] = df_dataset_tags['TagId'].map(df_tags_map)
    if df_dataset_tags['TagName'].isnull().all():
        print("CRITICAL: All 'TagName' values in df_dataset_tags are NaN. Mapping failed completely.")

    print("\n--- End Debugging Tag Mapping ---")

    print("\n--- Analyzing Industry/Application Area Trends ---")

    # Define key industry/application tags
    industry_tags = {
        'healthcare': 'Healthcare',
        'medical': 'Healthcare',
        'finance': 'Finance',
        'financial': 'Finance',
        'automotive': 'Autonomous Driving/Automotive',
        'autonomous vehicles': 'Autonomous Driving/Automotive',
        'energy': 'Energy',
        'retail': 'Retail',
        'manufacturing': 'Manufacturing',
        'sports': 'Sports',
        'e-commerce': 'E-commerce',
        'education': 'Education'
    }

    # Analyze competition tags for industry trends
    # Ensure df_competitions_filtered is used, which is already filtered by year
    competitions_industry_tags = pd.merge(df_competitions_filtered[['Id', 'Year']],
                                          df_competition_tags,
                                          left_on='Id',
                                          right_on='CompetitionId',
                                          how='inner') # Use inner to ensure valid CompetitionId

    competitions_industry_tags['BroadIndustry'] = competitions_industry_tags['TagName'].map(industry_tags)
    competitions_industry_tags_filtered = competitions_industry_tags.dropna(subset=['BroadIndustry'])

    # Count occurrences of broad industry types per year
    industry_comp_counts = competitions_industry_tags_filtered.groupby(['Year', 'BroadIndustry']).size().unstack(fill_value=0)

    # Filter industry_comp_counts to ensure it's within the desired start_year and end_year
    industry_comp_counts = industry_comp_counts[(industry_comp_counts.index >= start_year) & (industry_comp_counts.index <= end_year)]

    if not industry_comp_counts.empty and industry_comp_counts.sum().sum() > 0: # Check if there's any data to plot
        # Calculate proportions (sum along columns for each year)
        industry_comp_proportions = industry_comp_counts.apply(lambda x: x / x.sum(), axis=1)

        print("\nIndustry/Application Area Proportions in Competitions by Year:")
        print(industry_comp_proportions.tail())

        # Prepare data for Plotly (px.area prefers long format or pivoted wide format)
        industry_comp_proportions_plot = industry_comp_proportions.reset_index()

        fig_industry_trends = px.area(
            industry_comp_proportions_plot,
            x='Year',
            y=industry_comp_proportions_plot.columns[1:], # All columns except 'Year' are industry types
            title=f'Proportion of Industry/Application Areas in Kaggle Competitions Over Time ({start_year}-{end_year})',
            labels={'value': 'Proportion', 'variable': 'Industry Area'},
            height=600,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            line_shape="spline"
        )

        fig_industry_trends.update_layout(
            xaxis_title="Year",
            yaxis_title="Proportion of Competitions (%)",
            font=dict(size=11),
            title_font_size=20,
            hovermode="x unified", # Unified hover for stacked area
            margin=dict(l=60, r=60, t=100, b=60),
            paper_bgcolor='white', plot_bgcolor='white',
            legend_title_text='Industry Area'
        )
        all_years_industry_data = sorted(industry_comp_proportions_plot['Year'].unique())
        fig_industry_trends.update_xaxes(
            tickvals=all_years_industry_data,
            tickangle=45,
            tickfont=dict(size=10),
            range=[start_year - 0.5, end_year + 0.5]
        )
        fig_industry_trends.update_yaxes(tickfont=dict(size=10), tickformat=".0%", range=[0, 1])

        # Customize hovertemplate for stacked area to show percentage
        fig_industry_trends.for_each_trace(lambda trace: trace.update(
            hovertemplate='<b>Year</b>: %{x}<br>' +
                          '<b>Industry Area</b>: %{fullData.name}<br>' +
                          '<b>Proportion</b>: %{y:.1%}<extra></extra>'
        ))

        fig_industry_trends.show(renderer="iframe")
        fig_industry_trends.write_html("/kaggle/working/interactive_industry_trends_competitions.html")
        print("Successfully created interactive stacked area chart for industry/application area trends.")
    else:
        print("No relevant data found for industry/application area distribution within the specified years.")

    print("\n**Note:** You could repeat a similar analysis for df_dataset_tags if you want to see trends in general datasets.")

except FileNotFoundError as e:
    print(f"Error loading CSV: {e}. Please ensure the Meta Kaggle dataset is unzipped and located at: {meta_kaggle_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

