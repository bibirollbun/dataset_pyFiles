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


import kagglehub
path = kagglehub.dataset_download("kaggle/meta-kaggle")
print("Path to dataset files:", path)


import os
for dirname, _, filenames in os.walk('/kaggle/input/meta-kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# --- Loading the necessary datasets ---
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

# --- Preprocessing the data ---
# We're converting 'CreationDate' to datetime objects so we can easily perform time-series analysis.
df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])

# We need to link kernels with their tags. First, we merge KernelTags with Tags to get the easy-to-read tag names.
df_merged_tags = pd.merge(df_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='left')
df_merged_tags.rename(columns={'Name': 'TagName'}, inplace=True) # Renaming 'Name' to 'TagName' helps avoid confusion later.

# Then, we merge this with the Kernels data to get the creation dates for each tagged kernel.
df_kernel_data = pd.merge(df_kernels, df_merged_tags, left_on='Id', right_on='KernelId', how='inner')

# --- Analyzing Popularity Over Time ---

# To track popularity effectively, we're going to count how many times each tag appears over time.
# We'll aggregate by month (or year, if we want smoother trends) for better visualization.
df_kernel_data['YearMonth'] = df_kernel_data['CreationDate'].dt.to_period('M')

# Now, we count the number of kernels for each tag, per month.
tag_popularity_over_time = df_kernel_data.groupby(['YearMonth', 'TagName']).size().reset_index(name='KernelCount')

# We convert 'YearMonth' back to a datetime format, which is better for plotting.
tag_popularity_over_time['YearMonth'] = tag_popularity_over_time['YearMonth'].dt.to_timestamp()

# --- Visualize the Trends ---

# To make our visualizations clear, we'll focus on the top 10 most frequent tags overall.
top_n_tags = df_kernel_data['TagName'].value_counts().head(10).index

plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")

# We'll filter our data to include only these top N tags for better readability in the plot.
df_plot = tag_popularity_over_time[tag_popularity_over_time['TagName'].isin(top_n_tags)]
specific_ai_technique = "deep learning"


# Here, we plot the trends. Each line represents a different tag, showing its kernel count over time.
sns.lineplot(data=df_plot, x='YearMonth', y='KernelCount', hue='TagName', marker='o', markersize=4, lw=1.5)

plt.title('Evolution of Top Kernel Topic/AI Technique Popularity on Kaggle (by Kernel Count)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Kernels Created', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Tag Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/kaggle/working/Analyzing_Popularity_Over_Time.png')
plt.show()

print("\n--- Analysis Complete ---")
print("The plot above shows the trend in the number of kernels created for the top 10 most popular tags over time.")
print("We can easily modify 'top_n_tags' if we want to focus on specific AI techniques or topics of interest.")

specific_ai_technique= "deeplearning"

if specific_ai_technique in df_plot['TagName'].unique():
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_plot[df_plot['TagName'] == specific_ai_technique], x='YearMonth', y='KernelCount', marker='o', lw=2)
    plt.title(f'Popularity of "{specific_ai_technique}" on Kaggle (by Kernel Count)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Kernels Created', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print(f"\n'{specific_ai_technique}' tag not found in the top N tags or our overall data. Please check the spelling or choose another tag.")



top_n_for_faceting = df_kernel_data['TagName'].value_counts().head(6).index

# Filtering the data to include only the selected top N tags
df_plot_facet = tag_popularity_over_time[tag_popularity_over_time['TagName'].isin(top_n_for_faceting)].copy()

# Ensuring 'YearMonth' is a datetime type for proper plotting
df_plot_facet['YearMonth'] = pd.to_datetime(df_plot_facet['YearMonth'])

# Creating a FacetGrid for small multiples
# 'col' specifies the column to create separate plots for (our TagName)
# 'col_wrap' specifies how many columns of plots you want before wrapping to the next row
# 'height' and 'aspect' control the size of each individual subplot
# 'sharey=False' means each subplot will have its own Y-axis scale, which can be useful
# if tag counts vary wildly. Set to 'True' for consistent Y-axis across all plots.
g = sns.FacetGrid(df_plot_facet, col='TagName', col_wrap=3, height=4, aspect=1.2, sharey=False)

# Map a line plot to each facet
g.map(sns.lineplot, 'YearMonth', 'KernelCount', marker='o', lw=1.5, color='blue') # You can customize color

# Set axis labels and titles for each subplot
g.set_axis_labels("Date", "Kernel Count")
g.set_titles(col_template="{col_name} Popularity") # Set titles for each subplot based on TagName

# Rotate X-axis labels for readability
for ax in g.axes.flat:
    ax.tick_params(axis='x', labelrotation=45)

# Add an overall title for the entire figure
plt.suptitle('Kaggle Kernel Tag Popularity Trends (Individual Plots)', fontsize=18, y=1.02)

# Adjust layout to prevent overlaps
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust rect to make space for suptitle
plt.show()

print("\n--- Plotting Complete ---")
print("Each subplot above shows the popularity trend of a specific Kaggle kernel tag over time.")
print("This 'small multiples' approach makes it easier to analyze individual trends without clutter.")


print("\n--- All Unique Kernel Tag Names ---")
unique_tags = df_kernel_data['TagName'].unique()
for tag in sorted(unique_tags): # We'll sort them alphabetically for easier reading
    print(tag)

print(f"\nTotal number of unique tags: {len(unique_tags)}")
print("\n----------------------------------")


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
    "EfficientNet V2", "EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3",
    "EfficientNet-B4", "EfficientNet-B5", "EfficientNet-B6", "EfficientNet-B7", "EfficientNetV2",
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

# Convert all target tags to lowercase for robust matching (Kaggle tags are often lowercase)
all_ai_ml_tags_lower = [tag.lower() for tag in all_ai_ml_tags_raw]

# Get unique tags present in the actual df_kernel_data (also lowercased for matching)
actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

# --- 4. Filter for only the AI/ML tags that *actually exist* in our data ---
# We create a set of actual tags for fast lookup
actual_tags_set = set(actual_tags_in_data_lower)

# Create a list of the *existing* AI/ML tags, preserving original capitalization from df_tags
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
        ai_ml_popularity_over_time = df_filtered_ai_ml.groupby(['YearMonth', 'TagName']).size().reset_index(name='KernelCount')

        ai_ml_popularity_over_time['YearMonth'] = ai_ml_popularity_over_time['YearMonth'].dt.to_timestamp()

        # --- 5. Visualize the Trends - Stacked Area Plot ---
        plt.figure(figsize=(20, 12)) # Larger figure size for potentially many lines/areas
        sns.set_style("whitegrid")

        # Pivot the data for stacked area plot
        df_pivot_ai_ml = ai_ml_popularity_over_time.pivot(index='YearMonth', columns='TagName', values='KernelCount').astype(float).fillna(0)

        if df_pivot_ai_ml.empty or df_pivot_ai_ml.sum().sum() == 0:
            print("\nAfter pivoting, the AI/ML data contains no non-zero numeric values to plot. Check data consistency.")
        else:
            df_pivot_ai_ml.plot.area(stacked=True, figsize=(20, 12), cmap='tab20', alpha=0.8) # Using 'tab20' for more distinct colors

            plt.title('Stacked Area Plot: Evolution of AI/ML Topics and Techniques on Kaggle (Kernel Count)', fontsize=20)
            plt.xlabel('Date', fontsize=15)
            plt.ylabel('Total Kernels Created (Stacked)', fontsize=15)
            plt.xticks(rotation=45, ha='right', fontsize=11)
            plt.yticks(fontsize=11)
            plt.legend(title='AI/ML Aspect/Technique', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for the legend
            plt.savefig('/kaggle/working/ML_related_tags_graph.png')
            plt.show()

        print("\n--- Analysis Complete (AI/ML Stacked Area Plot) ---")
        print("The stacked area plot shows the cumulative trend of kernels for various AI/ML topics and techniques over time,")
        print("highlighting their overall volume and individual proportional contributions.")




# Define the General AI/ML Tags
general_ai_ml_tags_raw = [
    "Artificial Intelligence", "Machine Learning", "Advanced", "Beginner", "Intermediate",
    "Learn", "Research", "AutoML", "Model Comparison", "Model Explainability",
    "Transfer Learning", "Optimization"
]

# Convert all target tags to lowercase for robust matching
general_ai_ml_tags_lower = [tag.lower() for tag in general_ai_ml_tags_raw]

# Get unique tags present in the actual df_kernel_data (also lowercased for matching)
# Ensure 'TagName' column is treated as strings
df_kernel_data['TagName'] = df_kernel_data['TagName'].astype(str)
actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

# --- 2. Filter for only the General AI/ML tags that *actually exist* in our data ---
actual_tags_set = set(actual_tags_in_data_lower)

# Create a list of the *existing* General AI/ML tags, preserving original capitalization from df_kernel_data
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

        # Convert YearMonth back to timestamp for plotting
        general_ai_ml_popularity_over_time['YearMonth'] = general_ai_ml_popularity_over_time['YearMonth'].dt.to_timestamp()

        # --- 3. Visualize the Trends - Stacked Area Plot ---
        plt.figure(figsize=(15, 8)) # Adjust figure size as needed
        sns.set_style("whitegrid")

        # Pivot the data for stacked area plot
        df_pivot_general_ai_ml = general_ai_ml_popularity_over_time.pivot(index='YearMonth', columns='TagName', values='KernelCount').astype(float).fillna(0)

        if df_pivot_general_ai_ml.empty or df_pivot_general_ai_ml.sum().sum() == 0:
            print("\nAfter pivoting, the General AI/ML data contains no non-zero numeric values to plot. Check data consistency.")
        else:
            # Generate the stacked area plot
            df_pivot_general_ai_ml.plot.area(stacked=True, figsize=(15, 8), cmap='tab10', alpha=0.8) # Using 'tab10' for common colors

            plt.title('Stacked Area Plot: Evolution of General AI/ML Topics on Kaggle (Kernel Count)', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Total Kernels Created (Stacked)', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(title='General AI/ML Topic', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for the legend
            plt.savefig('/kaggle/working/General_ML_tags_stacked_area_graph.png')
            plt.show()

        print("\n--- Analysis Complete (General AI/ML Stacked Area Plot) ---")
        print("This stacked area plot visualizes the cumulative trend of kernels specifically for 'General AI/ML' topics over time.")
        print("It highlights the overall volume and the proportional contribution of each")


# I'm creating a list of the *existing* General AI/ML tags, preserving their original capitalization from df_tags.
existing_general_ai_ml_tags = [
    tag for tag in df_kernel_data['TagName'].unique()
    if tag.lower() in actual_tags_set and tag.lower() in general_ai_ml_tags_lower
]

if not existing_general_ai_ml_tags:
    print("\nNo 'General AI/ML' tags from the confirmed list were found in the dataset. I can't generate the plot.")
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
        # I'm adding a 'Year' column here.
        df_filtered_general_ai_ml['Year'] = df_filtered_general_ai_ml['CreationDate'].dt.year

        general_ai_ml_popularity_over_time = df_filtered_general_ai_ml.groupby(['YearMonth', 'TagName', 'Year']).size().reset_index(name='KernelCount')

        general_ai_ml_popularity_over_time['YearMonth'] = general_ai_ml_popularity_over_time['YearMonth'].dt.to_timestamp()

        # --- 5. Visualizing the Trends - Line Plot for General AI/ML (Yearly Subplots) ---

        # I'm finding the minimum and maximum years in the dataset.
        min_year = general_ai_ml_popularity_over_time['Year'].min()
        max_year = general_ai_ml_popularity_over_time['Year'].max()

        # I'm creating groups for every 4 years.
        years_to_plot = range(min_year, max_year + 1, 4) # Starts every 4 years

        # I'm determining the subplot layout.
        num_subplots = len(years_to_plot) # The number of starting points
        # If the last group isn't a full 4 years, I don't need an extra subplot; the range handles this.

        # I'm adjusting the subplot size for better readability.
        fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(20, 8 * num_subplots), sharex=False) # sharex=False so each plot has its own x-axis

        # If there's only one subplot, 'axes' will be an object, not an array.
        if num_subplots == 1:
            axes = [axes] # I'm making it a list so I can use it in the for loop regardless.

        # I'm creating a plot for each 4-year segment.
        for i, start_year in enumerate(years_to_plot):
            end_year = start_year + 3
            if end_year > max_year: # Ensuring the last group doesn't exceed the max year.
                end_year = max_year

            # I'm filtering the data for the current year range.
            df_slice = general_ai_ml_popularity_over_time[
                (general_ai_ml_popularity_over_time['Year'] >= start_year) &
                (general_ai_ml_popularity_over_time['Year'] <= end_year)
            ].copy() # I'm using copy() because I'll be operating on this slice.

            if not df_slice.empty:
                sns.lineplot(data=df_slice, x='YearMonth', y='KernelCount', hue='TagName',
                             marker='o', markersize=4, lw=1.5, ax=axes[i])

                axes[i].set_title(f'Evolution of General AI/ML Topics: {start_year}-{end_year}', fontsize=16)
                axes[i].set_xlabel('Date', fontsize=12)
                axes[i].set_ylabel('Number of Kernels Created', fontsize=12)
                # I'm only rotating the x-axis labels, no horizontal alignment needed here for tick_params.
                axes[i].tick_params(axis='x', rotation=45, labelsize=10)
                axes[i].tick_params(axis='y', labelsize=10)
                axes[i].grid(True, linestyle='--', alpha=0.7)

                # I can place the legend separately for each subplot,
                # or have a common legend for all subplots (as done below).
                axes[i].legend(title='General AI/ML Topic', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        plt.suptitle('Evolution of General AI/ML Topics on Kaggle (by Kernel Count) - Yearly Segments', fontsize=20, y=1.02) # Main title for all subplots
        plt.tight_layout(rect=[0, 0, 0.95, 1.0]) # Adjusting the layout to fit all subplots and the main title.
        # Saving the plot.
        plt.savefig('/kaggle/working/General_AI_ML_Yearly_Trends.png')

        plt.show()

        print("\n--- Analysis Complete (General AI/ML Yearly Line Plots) ---")
        print("The line plots above show the trend in the number of kernels created for each 'General AI/ML' topic, broken down into 4-year segments.")
        print("This allows for a more detailed comparison of their individual popularity trajectories within specific timeframes.")


general_ai_ml_tags_raw = [
    "Artificial Intelligence", "Machine Learning", "Advanced", "Beginner", "Intermediate",
    "Learn", "Research", "AutoML", "Model Comparison", "Model Explainability",
    "Transfer Learning", "Optimization"
]

# Convert all target tags to lowercase for robust matching
general_ai_ml_tags_lower = [tag.lower() for tag in general_ai_ml_tags_raw]

# Ensure 'TagName' column is treated as strings
df_kernel_data['TagName'] = df_kernel_data['TagName'].astype(str)
actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

# --- 2. Filter for only the General AI/ML tags that *actually exist* in our data ---
actual_tags_set = set(actual_tags_in_data_lower)

existing_general_ai_ml_tags = [
    tag for tag in df_kernel_data['TagName'].unique()
    if tag.lower() in actual_tags_set and tag.lower() in general_ai_ml_tags_lower
]

if not existing_general_ai_ml_tags:
    print("\nNo 'General AI/ML' tags from the confirmed list were found in the dataset. I can't generate the plot.")
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

        # Find min/max years to determine plot range
        min_year = yearly_tag_counts['Year'].min()
        max_year = yearly_tag_counts['Year'].max()
        all_years = sorted(yearly_tag_counts['Year'].unique()) # Get only years with data

        # --- Key change for vertical stacking ---
        ncols = 1 # We want a single column of plots
        nrows = len(all_years) # One row for each year with data

        # Adjust figsize for vertical stacking: fixed width, height proportional to number of rows
        # Let's say 12 inches width, and 6 inches height per plot.
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6 * nrows), sharey=True)

        # Handle the case of a single subplot (axes won't be an array then)
        if nrows == 1:
            axes = [axes] # Make it iterable for the loop

        # Loop through each year and create a bar plot
        for i, year in enumerate(all_years): # Iterate through all_years directly
            df_year_slice = yearly_tag_counts[yearly_tag_counts['Year'] == year].sort_values(by='KernelCount', ascending=False)

            if not df_year_slice.empty:
                sns.barplot(data=df_year_slice, x='TagName', y='KernelCount', ax=axes[i], palette='viridis')

                axes[i].set_title(f'Kernel Counts in {year}', fontsize=14)
                axes[i].set_xlabel('AI/ML Tag', fontsize=10)
                axes[i].set_ylabel('Number of Kernels', fontsize=10) # Each plot gets a Y-label now
                axes[i].tick_params(axis='x', rotation=90, labelsize=8) # Rotate for readability
                axes[i].tick_params(axis='y', labelsize=8)
                axes[i].grid(axis='y', linestyle='--', alpha=0.7) # Only Y-axis grid

                # Add count numbers on top of bars
                for p in axes[i].patches:
                    axes[i].annotate(f'{int(p.get_height())}',
                                     (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='center', fontsize=7, color='black', xytext=(0, 5),
                                     textcoords='offset points')
            else:
                # If a year has no data, plot an empty graph with a message
                axes[i].set_title(f'No Data for {year}', fontsize=14)
                axes[i].set_xlabel('')
                axes[i].set_ylabel('')
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].text(0.5, 0.5, 'No kernels found for this year.',
                                 horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes, fontsize=10, color='gray')


        plt.suptitle('Annual Popularity of General AI/ML Tags on Kaggle', fontsize=18, y=1.005) # Adjust suptitle position
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to ensure title fits
        
        # Save the plot
        plt.savefig('/kaggle/working/General_AI_ML_Yearly_Bar_Plots_Vertical.png')
        plt.show()

        print("\n--- Annual Tag Count Summary (General AI/ML) ---")
        print("Here's a detailed breakdown of kernel counts for each tag, year by year:")
        
        # Create a pivot table for the yearly tag counts
        pivot_table_counts = yearly_tag_counts.pivot_table(index='TagName', columns='Year', values='KernelCount', fill_value=0)
        
        # Sort rows by total count if desired (optional)
        pivot_table_counts['Total'] = pivot_table_counts.sum(axis=1)
        pivot_table_counts = pivot_table_counts.sort_values(by='Total', ascending=False).drop('Total', axis=1)

        print(pivot_table_counts.to_markdown())
        print("\n--- Analysis Complete (Annual Bar Plots & Summary) ---")
        print("The bar plots above visually represent the distribution of General AI/ML tags each year. Below, the table provides the exact kernel counts, giving a precise view of the annual popularity of each tag.")




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

        # --- 5. Finally, I visualize the trends using a Line Plot ---
        plt.figure(figsize=(18, 10))
        sns.set_style("whitegrid")

        sns.lineplot(data=skill_level_popularity_over_time, x='YearMonth', y='KernelCount', hue='TagName', marker='o', markersize=4, lw=1.5)

        plt.title('Evolution of Skill Level and Learning Tags in Kaggle Kernels (Kernel Count)', fontsize=18)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Number of Kernels Created', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Skill Level / Learning', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig('/kaggle/working/Skill_Level_Yearly_Bar_Plots_2.png')
        
        plt.show()

        print("\n--- My Analysis is Complete (Skill Level Line Plot) ---")
        print("The line plot above shows the trend in the number of kernels created for each 'Skill Level and Learning' tag over time.")
        print("This helps me identify patterns related to the influx of new users and the progression of existing users within the Kaggle community.")


# Now, I filter for only the skill-related tags that *actually exist* in my data 
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
        df_filtered_skill_levels['Year'] = df_filtered_skill_levels['CreationDate'].dt.year

        # Aggregate data by Year and TagName for bar charts
        yearly_skill_counts = df_filtered_skill_levels.groupby(['Year', 'TagName']).size().reset_index(name='KernelCount')

        # Find min/max years to determine plot range
        min_year = yearly_skill_counts['Year'].min()
        max_year = yearly_skill_counts['Year'].max()
        all_years = range(min_year, max_year + 1)
        
        #  I'll visualize the Annual Trends using Bar Plots (stacked vertically) ---

        # I'm setting up for a single column of plots, so they stack nicely
        ncols = 1
        # I'll have one row per year
        nrows = len(all_years)
        
        # I'm creating the figure and subplots. Each plot will be wide for readability.
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15 * ncols, 8 * nrows), sharey=True)

        # If there's only one subplot, 'axes' isn't an array, so I make it iterable.
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif ncols == 1:
            # If it's a single column with multiple rows, 'axes' is already a 1D array.
            pass
        
        # I'll loop through each year and create a bar plot for it.
        for i, year in enumerate(all_years):
            # Access the correct subplot for the current year.
            current_ax = axes[i] 

            df_year_slice = yearly_skill_counts[yearly_skill_counts['Year'] == year].sort_values(by='KernelCount', ascending=False)

            if not df_year_slice.empty:
                sns.barplot(data=df_year_slice, x='TagName', y='KernelCount', ax=current_ax, palette='coolwarm') # Using 'coolwarm' for a different look
                
                current_ax.set_title(f'Kernel Counts by Skill Level in {year}', fontsize=16)
                current_ax.set_xlabel('Skill Level Tag', fontsize=12)
                current_ax.set_ylabel('Number of Kernels Created', fontsize=12)
                current_ax.tick_params(axis='x', rotation=45, labelsize=10) 
                current_ax.tick_params(axis='y', labelsize=10)
                current_ax.grid(axis='y', linestyle='--', alpha=0.7)

                # I'm adding the count numbers on top of the bars for clarity.
                for p in current_ax.patches:
                    height = p.get_height()
                    if height > 0: # Only show counts for bars with actual values
                        current_ax.annotate(f'{int(height)}', 
                                         (p.get_x() + p.get_width() / 2., height), 
                                         ha='center', va='center', fontsize=8, color='black', xytext=(0, 5), 
                                         textcoords='offset points')
            else:
                # If a year has no data for these tags, I'll display a message.
                current_ax.set_title(f'No Data for {year}', fontsize=16)
                current_ax.set_xlabel('')
                current_ax.set_ylabel('')
                current_ax.set_xticks([])
                current_ax.set_yticks([])
                current_ax.text(0.5, 0.5, 'No kernels found for this year.', 
                                 horizontalalignment='center', verticalalignment='center', transform=current_ax.transAxes, fontsize=12, color='gray')

        # I'm adding a main title for all the plots.
        plt.suptitle('Annual Popularity of Skill Level and Learning Tags on Kaggle', fontsize=20, y=1.005)
        # I'm adjusting the layout to prevent labels from overlapping.
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Saving the plot.
        plt.savefig('/kaggle/working/Skill_Level_Yearly_Bar_Plots_Stacked.png')
        plt.show()

        # I'll print a summary table of the annual tag counts
        print("\n--- Annual Skill Level Tag Count Summary ---")
        print("Here's a detailed breakdown of kernel counts for each skill level tag, year by year:")
        
        # I'm creating a pivot table for easy readability.
        pivot_table_counts = yearly_skill_counts.pivot_table(index='TagName', columns='Year', values='KernelCount', fill_value=0)
        
        # I'll sort the tags by their total count across all years for better insight.
        pivot_table_counts['Total'] = pivot_table_counts.sum(axis=1)
        pivot_table_counts = pivot_table_counts.sort_values(by='Total', ascending=False).drop('Total', axis=1)

        print(pivot_table_counts.to_markdown())
        print("\n--- My Analysis is Complete (Annual Skill Level Bar Plots & Summary) ---")
        print("The bar plots visually show the annual distribution of kernels by skill level, with exact counts on each bar. The table below provides a precise, year-by-year summary, helping me understand the learning trends on Kaggle.")


# Only proceed if dataframes are not empty after loading attempt
if not df_kernels.empty and not df_kernel_tags.empty and not df_tags.empty:
    # --- 2. Next, I preprocess my data ---
    # I convert 'CreationDate' to datetime objects for accurate time-series analysis.
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])

    # I merge KernelTags with Tags to get readable tag names, then merge with Kernels for creation dates.
    df_merged_tags = pd.merge(df_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='left')
    df_merged_tags.rename(columns={'Name': 'TagName'}, inplace=True)
    df_kernel_data = pd.merge(df_kernels, df_merged_tags, left_on='Id', right_on='KernelId', how='inner')

    # --- 3. I define the specific list of modern AI/ML sub-field tags I'm interested in ---
    modern_ai_ml_tags_raw = [
        "AutoML", "Model Explainability", "Transfer Learning"
    ]

    # I convert all my target tags to lowercase for robust matching, as Kaggle tags are often lowercase.
    modern_ai_ml_tags_lower = [tag.lower() for tag in modern_ai_ml_tags_raw]

    # I get unique tags present in my actual df_kernel_data (also lowercased for matching).
    actual_tags_in_data_lower = df_kernel_data['TagName'].str.lower().unique()

    # --- 4. Now, I filter for only the modern AI/ML sub-field tags that *actually exist* in my data ---
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

            # --- 5. Finally, I visualize the trends using a Line Plot ---
            plt.figure(figsize=(18, 10))
            sns.set_style("whitegrid")

            sns.lineplot(data=modern_ai_ml_popularity_over_time, x='YearMonth', y='KernelCount', hue='TagName', marker='o', markersize=4, lw=1.5)

            plt.title('Evolution of Modern AI/ML Sub-fields in Kaggle Kernels (Kernel Count)', fontsize=18)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Number of Kernels Created', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(title='AI/ML Sub-field', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 0.88, 1])
            plt.savefig('/kaggle/working/Modern_AI_Sub-field.png')
            plt.show()

            print("\n--- My Analysis is Complete (Modern AI/ML Sub-fields Line Plot) ---")
            print("The line plot above shows the trend in the number of kernels created for 'AutoML', 'Model Explainability', and 'Transfer Learning' over time.")
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

    # --- 3. Define the specific "Solution Approach" tags you want to track ---
    # This list is crucial. You should expand or modify it based on your hypotheses
    # about what constitutes a "solution approach" in AI/ML.
    # Examples include specific algorithms, modeling paradigms, data processing methods, etc.
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

            # --- 5. Visualize the Trends - Line Plot for Solution Approaches ---
            plt.figure(figsize=(20, 12))
            sns.set_style("whitegrid")

            sns.lineplot(data=solution_approach_popularity_over_time, x='Year', y='KernelCount', hue='TagName', marker='o', markersize=5, lw=2)

            plt.title('Evolution of Solution Approaches in Kaggle Kernels (Kernel Count)', fontsize=20)
            plt.xlabel('Year', fontsize=16)
            plt.ylabel('Number of Kernels Tagged', fontsize=16)
            plt.xticks(solution_approach_popularity_over_time['Year'].unique(), rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(title='Solution Approach', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for the legend
            plt.savefig('/kaggle/Evolution_of_Solution_Approaches.png')
            plt.show()

            print("\n--- Analysis Complete (Solution Approaches Line Plot) ---")
            print("The line plot above illustrates the evolution of various solution approaches on Kaggle over time.")
            print("By observing the trends of these tags, you can infer shifts in popular methodologies and techniques.")
else:
    print("\nDataFrames are empty after loading. Cannot proceed with analysis.")



if not df_kernels.empty and not df_kernel_tags.empty and not df_tags.empty:
    # Preprocess the data
    df_kernels['CreationDate'] = pd.to_datetime(df_kernels['CreationDate'])

    # Merge KernelTags with Tags to get readable tag names, then merge with Kernels for creation dates.
    df_merged_tags = pd.merge(df_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='left')
    df_merged_tags.rename(columns={'Name': 'TagName'}, inplace=True)
    df_kernel_data = pd.merge(df_kernels, df_merged_tags, left_on='Id', right_on='KernelId', how='inner')

  
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

    #Filter for only the solution approach tags that *actually exist* in your data ---
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
            # Add 'Year' column for annual analysis
            df_filtered_solution_approaches['Year'] = df_filtered_solution_approaches['CreationDate'].dt.year

            # Group by Year and TagName to count occurrences
            yearly_solution_counts = df_filtered_solution_approaches.groupby(['Year', 'TagName']).size().reset_index(name='KernelCount')

            # Find min/max years to determine plot range
            min_year = yearly_solution_counts['Year'].min()
            max_year = yearly_solution_counts['Year'].max()
            all_years = range(min_year, max_year + 1)
            
            # --- 5. Visualize the Trends - Bar Plots for Solution Approaches (Stacked Vertically) ---

            # Set up for a single column of plots, so they stack nicely
            ncols = 1
            # One row per year
            nrows = len(all_years)
            
            # Create the figure and subplots. Each plot will be wide for readability.
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15 * ncols, 8 * nrows), sharey=True)

            # If there's only one subplot, 'axes' isn't an array, so make it iterable.
            if nrows == 1 and ncols == 1:
                axes = [axes]
            elif ncols == 1:
                # If it's a single column with multiple rows, 'axes' is already a 1D array.
                pass
            
            # Loop through each year and create a bar plot for it.
            for i, year in enumerate(all_years):
                # Access the correct subplot for the current year.
                current_ax = axes[i] 

                df_year_slice = yearly_solution_counts[yearly_solution_counts['Year'] == year].sort_values(by='KernelCount', ascending=False)

                if not df_year_slice.empty:
                    sns.barplot(data=df_year_slice, x='TagName', y='KernelCount', ax=current_ax, palette='plasma') # Using 'plasma' for a new look
                    
                    current_ax.set_title(f'Kernel Counts by Solution Approach in {year}', fontsize=16)
                    current_ax.set_xlabel('Solution Approach Tag', fontsize=12)
                    current_ax.set_ylabel('Number of Kernels Tagged', fontsize=12)
                    current_ax.tick_params(axis='x', rotation=45, labelsize=10) # Rotate labels for readability
                    current_ax.tick_params(axis='y', labelsize=10)
                    current_ax.grid(axis='y', linestyle='--', alpha=0.7)

                    # Add the count numbers on top of the bars for clarity.
                    for p in current_ax.patches:
                        height = p.get_height()
                        if height > 0: # Only show counts for bars with actual values
                            current_ax.annotate(f'{int(height)}', 
                                             (p.get_x() + p.get_width() / 2., height), 
                                             ha='center', va='center', fontsize=8, color='black', xytext=(0, 5), 
                                             textcoords='offset points')
                else:
                    # If a year has no data for these tags, display a message.
                    current_ax.set_title(f'No Data for {year}', fontsize=16)
                    current_ax.set_xlabel('')
                    current_ax.set_ylabel('')
                    current_ax.set_xticks([])
                    current_ax.set_yticks([])
                    current_ax.text(0.5, 0.5, 'No kernels found for this year.', 
                                     horizontalalignment='center', verticalalignment='center', transform=current_ax.transAxes, fontsize=12, color='gray')

            # Add a main title for all the plots.
            plt.suptitle('Annual Popularity of Solution Approach Tags on Kaggle', fontsize=20, y=1.005)
            # Adjust the layout to prevent labels from overlapping.
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            
            # Saving the plot.
            plt.savefig('/kaggle/working/Solution_Approach_Yearly_Bar_Plots_Stacked.png')
            plt.show()

            # print a summary table of the annual tag counts ---
            print("\n--- Annual Solution Approach Tag Count Summary ---")
            print("Here's a detailed breakdown of kernel counts for each solution approach tag, year by year:")
            
            # Create a pivot table for easy readability.
            pivot_table_counts = yearly_solution_counts.pivot_table(index='TagName', columns='Year', values='KernelCount', fill_value=0)
            
            # Sort the tags by their total count across all years for better insight.
            pivot_table_counts['Total'] = pivot_table_counts.sum(axis=1)
            pivot_table_counts = pivot_table_counts.sort_values(by='Total', ascending=False).drop('Total', axis=1)

            print(pivot_table_counts.to_markdown())
            print("\n--- Analysis Complete (Annual Solution Approach Bar Plots & Summary) ---")
            print("The bar plots visually show the annual distribution of kernels by solution approach, with exact counts on each bar. The table below provides a precise, year-by-year summary, helping me understand the evolution of techniques on Kaggle.")
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
        # --- 4. Count tags for each year ---
        tag_counts_2022 = df_2022['TagName'].value_counts().reset_index()
        tag_counts_2022.columns = ['TagName', 'Count_2022']

        tag_counts_2024 = df_2024['TagName'].value_counts().reset_index()
        tag_counts_2024.columns = ['TagName', 'Count_2024']

        # --- 5. Merge the counts for comparison ---
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

    # --- 3. Filter data for 2022 and 2025 ---
    # Current year is 2025, so we can filter up to now.
    df_2022_tags = df_kernel_with_tags[df_kernel_with_tags['CreationDate'].dt.year == 2022]
    df_2024_tags = df_kernel_with_tags[df_kernel_with_tags['CreationDate'].dt.year == 2024]

    # --- 4. Count total tags for each year ---
    total_tags_2022 = len(df_2022_tags)
    total_tags_2024 = len(df_2024_tags)

    # --- 5. Display results ---
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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress specific FutureWarning messages for cleaner output (if you haven't already)
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# Assuming df_kernels, df_kernel_tags, df_tags are already loaded and preprocessed.
# And df_kernel_data (merged kernels and tags) is available from previous steps.

# 1. Find the first kernel creation date for each user
# FIX: Using 'AuthorUserId' instead of 'AuthorId'
user_first_kernel_date = df_kernels.groupby('AuthorUserId')['CreationDate'].min().reset_index()
user_first_kernel_date.rename(columns={'CreationDate': 'FirstKernelDate'}, inplace=True)

# 2. Merge back to kernels to mark 'first kernels' for each user
df_kernels_with_first_date = pd.merge(df_kernels, user_first_kernel_date, on='AuthorUserId', how='left')
df_novice_kernels_raw = df_kernels_with_first_date[df_kernels_with_first_date['CreationDate'] == df_kernels_with_first_date['FirstKernelDate']].copy()

# 3. Join with tags to get actual tag names for these novice contributions
df_novice_kernel_tags = pd.merge(df_novice_kernels_raw, df_kernel_tags, left_on='Id', right_on='KernelId', how='inner')
df_novice_kernel_tags = pd.merge(df_novice_kernel_tags, df_tags, left_on='TagId', right_on='Id', how='inner')
df_novice_kernel_tags.rename(columns={'Name': 'TagName'}, inplace=True)

# 4. Analyze trends of these tags over time (by the year of the user's first kernel)
df_novice_kernel_tags['FirstKernelYear'] = df_novice_kernel_tags['FirstKernelDate'].dt.year

# Filter for a relevant time period (e.g., 2015-2024, adjust as per your data range)
df_novice_kernel_tags_filtered = df_novice_kernel_tags[
    (df_novice_kernel_tags['FirstKernelYear'] >= 2015) &
    (df_novice_kernel_tags['FirstKernelYear'] <= 2024)
]

# Calculate top N tags for novice contributions by year
# You might want to pick a few relevant tags like 'beginner', 'python', 'exploratory data analysis'
# for plotting to keep the visualization clear.
novice_tag_trends = df_novice_kernel_tags_filtered.groupby(['FirstKernelYear', 'TagName']).size().reset_index(name='Count')

# --- Plotting Suggestion ---
# Example: Plotting trends for 'beginner' and 'exploratory data analysis' tags among novices
# Filter for specific tags you want to visualize
tags_to_plot = ['beginner', 'exploratory data analysis', 'python']
novice_selected_trends = novice_tag_trends[novice_tag_trends['TagName'].isin(tags_to_plot)]

plt.figure(figsize=(12, 7))
sns.lineplot(data=novice_selected_trends, x='FirstKernelYear', y='Count', hue='TagName', marker='o')
plt.title('Trends in Initial Kernel Tags by Novice Users (2015-2024)')
plt.xlabel('Year of User\'s First Kernel')
plt.ylabel('Number of Kernels Tagged')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/kaggle/working/Trends_in_Initial_Kernel_Tags_by_Novice_Users.png')
plt.show()


# ---  Novice User Participation and Initial Contribution Types ---

print("\n--- Analyzing Novice User Contributions ---")

# 1. Find the first kernel creation date for each user
# Use 'AuthorUserId' based on your provided column list
user_first_kernel_date = df_kernels.groupby('AuthorUserId')['CreationDate'].min().reset_index()
user_first_kernel_date.rename(columns={'CreationDate': 'FirstKernelDate'}, inplace=True)

# 2. Merge back to kernels to identify 'first kernels' for each user
# A user's "first kernel" is defined as any kernel created on their 'FirstKernelDate'.
# This handles cases where a user might upload multiple kernels on their very first day.
df_kernels_with_first_date = pd.merge(df_kernels, user_first_kernel_date, on='AuthorUserId', how='left')
df_novice_kernels_raw = df_kernels_with_first_date[
    df_kernels_with_first_date['CreationDate'] == df_kernels_with_first_date['FirstKernelDate']
].copy()

# 3. Join with tags to get actual tag names for these novice contributions
# Use df_kernel_data which already has kernelId linked to tagNames
df_novice_kernel_tags = pd.merge(
    df_novice_kernels_raw[['Id', 'FirstKernelDate']], # Select only necessary columns
    df_kernel_data[['KernelId', 'TagName']], # Contains all kernel-tag mappings
    left_on='Id',
    right_on='KernelId',
    how='inner'
)

# 4. Analyze trends of these tags over time (by the year of the user's first kernel)
df_novice_kernel_tags['FirstKernelYear'] = df_novice_kernel_tags['FirstKernelDate'].dt.year

# Filter for a relevant time period (e.g., from 2015 onwards, or adjust as per your data range)
# Assuming Kaggle became more active for public kernels around 2015-2016
start_year = 2015
end_year = 2024 # Current year for your analysis
df_novice_kernel_tags_filtered = df_novice_kernel_tags[
    (df_novice_kernel_tags['FirstKernelYear'] >= start_year) &
    (df_novice_kernel_tags['FirstKernelYear'] <= end_year)
]

# Count tag occurrences per year
novice_tag_trends = df_novice_kernel_tags_filtered.groupby(['FirstKernelYear', 'TagName']).size().reset_index(name='Count')

# --- Select and Prepare Data for Plotting ---
# To avoid clutter, select a few key tags that represent common initial contributions or shifts.
# You can customize this list based on what you expect or find after an initial look at 'novice_tag_trends.head()'
key_novice_tags_to_plot = [
    'beginner', 'python', 'exploratory data analysis',
    'data cleaning', 'classification', 'deep learning',
    'data visualization', 'data analytics', 'pandas' # Add more if relevant
]

novice_selected_trends = novice_tag_trends[novice_tag_trends['TagName'].isin(key_novice_tags_to_plot)]

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

# --- Plotting ---

# Plot 1: Line Plot of absolute counts for selected tags
plt.figure(figsize=(14, 8))
sns.lineplot(data=novice_selected_trends, x='FirstKernelYear', y='Count', hue='TagName', marker='o', linewidth=2)
plt.title('Absolute Count of Key Initial Kernel Tags by Novice Users (2015-2024)', fontsize=16)
plt.xlabel('Year of User\'s First Kernel Creation', fontsize=12)
plt.ylabel('Number of Kernels Tagged', fontsize=12)
plt.xticks(range(start_year, end_year + 1), rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Tag Name', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/kaggle/working/new_Kaggle_users_as_initial_contributions.png')
plt.show()

# Plot 2: Stacked Area Plot of proportions (better for showing distribution change)
# Pivot for stacked area plot
novice_proportions_pivot = novice_selected_trends_normalized.pivot(index='FirstKernelYear', columns='TagName', values='Proportion').fillna(0)

plt.figure(figsize=(14, 8))
novice_proportions_pivot.plot(kind='area', stacked=True, colormap='viridis', alpha=0.8, figsize=(14, 8))
plt.title('Proportion of Initial Kernel Tags by Novice Users Over Time (2015-2024)', fontsize=16)
plt.xlabel('Year of User\'s First Kernel Creation', fontsize=12)
plt.ylabel('Proportion of First Kernels Tagged (%)', fontsize=12)
plt.xticks(range(start_year, end_year + 1), rotation=45, ha='right')
plt.ylim(0, 1) # Proportions range from 0 to 1
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Tag Name', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/kaggle/working/new_Kaggle_users_as_initial_contributions2.png')
plt.show()

print("\n--- Novice User Analysis Complete ---")
print("The plots above illustrate the types of content new Kaggle users are creating as their initial contributions over the years.")
print("The line plot shows absolute counts, while the stacked area plot reveals the changing proportions of these tags.")


# sub2
# Assuming df_kernels is loaded and 'CreationDate' is datetime

df_kernels['CreationYear'] = df_kernels['CreationDate'].dt.year

# Count total forks per year
fork_counts_by_year = df_kernels[df_kernels['ForkParentKernelVersionId'].notna()] \
                                .groupby('CreationYear').size().reset_index(name='ForkCount')

# Count total unique kernels (potential parents) created per year
# Filter out template kernels or extremely simple ones if they skew the parent count
total_kernels_by_year = df_kernels.groupby('CreationYear').size().reset_index(name='TotalKernels')

# Merge to calculate average forks per kernel
fork_analysis = pd.merge(fork_counts_by_year, total_kernels_by_year, on='CreationYear', how='left')
fork_analysis['AvgForksPerKernel'] = fork_analysis['ForkCount'] / fork_analysis['TotalKernels']

# Filter for a relevant time period for plotting (e.g., 2015-2024)
fork_analysis_filtered = fork_analysis[(fork_analysis['CreationYear'] >= 2015) & (fork_analysis['CreationYear'] <= 2024)]

# --- Plotting Suggestions ---
plt.figure(figsize=(12, 7))
sns.lineplot(data=fork_analysis_filtered, x='CreationYear', y='ForkCount', marker='o')
plt.title('Total Kernel Forks on Kaggle Over Time (2015-2024)')
plt.xlabel('Year')
plt.ylabel('Number of Forks')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/kaggle/working/total_forks_onkaggle.png')
plt.show()

plt.figure(figsize=(12, 7))
sns.lineplot(data=fork_analysis_filtered, x='CreationYear', y='AvgForksPerKernel', marker='o')
plt.title('Average Forks Per Kernel Over Time (2015-2024)')
plt.xlabel('Year')
plt.ylabel('Average Forks')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/kaggle/working/Average_forks_onkaggle.png')
plt.show()


# --- Sub-Question 3: Characteristics of Most Interacted-With Kernels ---

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
# Merge df_top_kernels_yearly with df_kernel_data to get the tags for these specific top kernels
df_top_kernel_tags = pd.merge(
    df_top_kernels_yearly[['Id', 'CreationYear']], # Relevant columns from top kernels
    df_kernel_data[['KernelId', 'TagName']],       # All kernel-tag mappings
    left_on='Id',
    right_on='KernelId',
    how='inner'
)

# Count tag occurrences in top kernels per year
top_kernel_tag_trends = df_top_kernel_tags.groupby(['CreationYear', 'TagName']).size().reset_index(name='Count')

# --- Prepare data for plotting ---
# To avoid clutter, we'll focus on the most popular tags overall within these top kernels,
# or specific tags that represent key trends (e.g., 'deep learning', 'gpu', 'eda', 'python').
# First, find the overall top N tags from the 'top_kernel_tag_trends' for better representation
overall_top_tags_in_top_kernels = top_kernel_tag_trends.groupby('TagName')['Count'].sum().nlargest(10).index.tolist()
# Add specific tags if they are not in top N but are crucial for your story
key_tags_for_plot = list(set(overall_top_tags_in_top_kernels + ['deep learning', 'gpu', 'exploratory data analysis', 'classification', 'tabular', 'python']))


top_kernel_selected_trends = top_kernel_tag_trends[top_kernel_tag_trends['TagName'].isin(key_tags_for_plot)]

# Calculate proportions for stacked area plot
# We need the total number of tags in top kernels for each year to normalize
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

# --- Plotting ---

# Plot 1: Line Plot of absolute counts for selected top tags
plt.figure(figsize=(14, 8))
sns.lineplot(data=top_kernel_selected_trends, x='CreationYear', y='Count', hue='TagName', marker='o', linewidth=2)
plt.title(f'Absolute Count of Key Tags in Top {top_percentage*100:.0f}% Kernels by {interaction_metric} (2015-2024)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Kernels Tagged', fontsize=12)
plt.xticks(range(start_year, end_year + 1), rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Tag Name', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/kaggle/working/characteristics_most_Interacted_With_Kernels.png')
plt.show()

# Plot 2: Stacked Area Plot of proportions
plt.figure(figsize=(14, 8))
top_kernel_proportions_pivot.plot(kind='area', stacked=True, colormap='Spectral', alpha=0.8, figsize=(14, 8))
plt.title(f'Proportion of Key Tags in Top {top_percentage*100:.0f}% Kernels by {interaction_metric} (2015-2024)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Proportion of Tags (%)', fontsize=12)
plt.xticks(range(start_year, end_year + 1), rotation=45, ha='right')
plt.ylim(0, 1) # Proportions range from 0 to 1
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Tag Name', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/kaggle/working/characteristics_most_Interacted_With_Kernels2.png')
plt.show()

print("\n--- Most Interacted-With Kernels Analysis Complete ---")
print(f"The plots above illustrate the changing characteristics of the top {top_percentage*100:.0f}% most interacted-with kernels by {interaction_metric} over time.")


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
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your Meta Kaggle dataset
meta_kaggle_path = '/kaggle/input/meta-kaggle/'

# --- Load necessary CSVs ---
try:
    df_competitions = pd.read_csv(f'{meta_kaggle_path}Competitions.csv')
    df_datasets = pd.read_csv(f'{meta_kaggle_path}Datasets.csv')
    df_competition_tags = pd.read_csv(f'{meta_kaggle_path}CompetitionTags.csv')
    df_dataset_tags = pd.read_csv(f'{meta_kaggle_path}DatasetTags.csv') # Needed for dataset types
    print("necessary Meta Kaggle CSVs loaded successfully!")

    # --- Preprocessing ---
    # Convert dates to datetime objects and extract year
    df_competitions['EnabledDate'] = pd.to_datetime(df_competitions['EnabledDate'])
    df_competitions['Year'] = df_competitions['EnabledDate'].dt.year

    df_datasets['CreationDate'] = pd.to_datetime(df_datasets['CreationDate'])
    df_datasets['Year'] = df_datasets['CreationDate'].dt.year

    # Filter data from a reasonable start year to focus on modern AI/ML trends
    start_year = 2015
    df_competitions_filtered = df_competitions[df_competitions['Year'] >= start_year].copy()
    df_datasets_filtered = df_datasets[df_datasets['Year'] >= start_year].copy()

    # --- DEBUGGING TAG MAPPING ---
    print("\n--- Debugging Tag Mapping ---")
    print(f"Shape of df_tags: {df_tags.shape}")
    print(f"Columns of df_tags: {df_tags.columns.tolist()}")

    # --- FIX IS HERE ---
    # Change 'TagName' to 'Name' for df_tags
    if 'Id' not in df_tags.columns or 'Name' not in df_tags.columns: # Changed 'TagName' to 'Name' here
        raise ValueError("df_tags must contain 'Id' and 'Name' columns.") # Changed 'TagName' to 'Name' here

    df_tags_map = df_tags.set_index('Id')['Name'].to_dict() # Changed 'TagName' to 'Name' here
    # --- END OF FIX ---

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

except FileNotFoundError as e:
    print(f"Error loading CSV: {e}. Please ensure the Meta Kaggle dataset is unzipped and located at: {meta_kaggle_path}")
except Exception as e:
    # This will now print the actual error message like 'TagName' if it's not just a NameError
    print(f"An unexpected error occurred during data loading: {e}")


df_dataset_versions = pd.read_csv(f'{meta_kaggle_path}DatasetVersions.csv')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your Meta Kaggle dataset
meta_kaggle_path = '/kaggle/input/meta-kaggle/'

# --- Load necessary CSVs ---
try:
    df_competitions = pd.read_csv(f'{meta_kaggle_path}Competitions.csv')
    df_datasets = pd.read_csv(f'{meta_kaggle_path}Datasets.csv')
    # ADD THIS LINE: Load df_dataset_versions (This is already correct in your snippet)
    df_dataset_versions = pd.read_csv(f'{meta_kaggle_path}DatasetVersions.csv')
    df_competition_tags = pd.read_csv(f'{meta_kaggle_path}CompetitionTags.csv')
    df_dataset_tags = pd.read_csv(f'{meta_kaggle_path}DatasetTags.csv') # Needed for dataset types
    # Also ensure df_tags is loaded here if it's not done elsewhere
    df_tags = pd.read_csv(f'{meta_kaggle_path}Tags.csv') # Make sure this line exists if df_tags is used later

    print("necessary Meta Kaggle CSVs loaded successfully!")

    # --- Preprocessing ---
    # Convert dates to datetime objects and extract year
    df_competitions['EnabledDate'] = pd.to_datetime(df_competitions['EnabledDate'])
    df_competitions['Year'] = df_competitions['EnabledDate'].dt.year

    df_datasets['CreationDate'] = pd.to_datetime(df_datasets['CreationDate'])
    df_datasets['Year'] = df_datasets['CreationDate'].dt.year

    # Filter data from a reasonable start year to focus on modern AI/ML trends
    start_year = 2015
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

    # --- FIX IS HERE (already correct based on your previous input) ---
    # Change 'TagName' to 'Name' for df_tags
    if 'Id' not in df_tags.columns or 'Name' not in df_tags.columns:
        raise ValueError("df_tags must contain 'Id' and 'Name' columns.")

    df_tags_map = df_tags.set_index('Id')['Name'].to_dict()
    # --- END OF FIX ---

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

    # --- YOUR ORIGINAL CODE SNIPPET (should work now) ---
    # Use 'TotalCompressedBytes' for size analysis
    if 'TotalCompressedBytes' in df_dataset_versions_filtered.columns:
        # Filter out versions with 0 bytes or very small sizes that might skew the median
        # It's good practice to ensure the column is numeric if it's mixed type (DtypeWarning)
        df_dataset_versions_filtered['TotalCompressedBytes'] = pd.to_numeric(df_dataset_versions_filtered['TotalCompressedBytes'], errors='coerce')

        median_dataset_version_size_mb = df_dataset_versions_filtered[
            df_dataset_versions_filtered['TotalCompressedBytes'] > 0
        ].groupby('Year')['TotalCompressedBytes'].median() / (1024**2) # Convert to MB

        print("\nMedian Dataset Version Size (MB) by Year (using TotalCompressedBytes):")
        print(median_dataset_version_size_mb.tail())

        # Plotting Median Dataset Version Size (MB)
        plt.figure(figsize=(12, 6))
        median_dataset_version_size_mb.plot(kind='line', marker='o')
        plt.title('Median Size of Dataset Versions on Kaggle Over Time (MB)')
        plt.xlabel('Year')
        plt.ylabel('Median Dataset Version Size (MB)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('/kaggle/working/Median_Dataset_Version_Size.png')
        plt.show()
    else:
        print("Warning: 'TotalCompressedBytes' column not found in df_dataset_versions_filtered. Cannot plot size trends.")

    # Remove 'TotalFiles' analysis since it's not present
    print("Note: 'TotalFiles' column is not available in df_dataset_versions_filtered, so file count trend cannot be plotted from this source.")

except FileNotFoundError as e:
    print(f"Error loading CSV: {e}. Please ensure the Meta Kaggle dataset is unzipped and located at: {meta_kaggle_path}")
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")


print("Actual columns in df_competitions:", df_competitions.columns.tolist())


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
    # Datasets can have multiple tags, so we'll count them by main type.
}

# Merge datasets with their tags
datasets_with_tags = pd.merge(df_datasets_filtered[['Id', 'Year']], df_dataset_tags, left_on='Id', right_on='DatasetId')

# Map specific tags to broader data types
datasets_with_tags['BroadDataType'] = datasets_with_tags['TagName'].map(data_type_mapping)
datasets_with_tags_filtered = datasets_with_tags.dropna(subset=['BroadDataType'])

# Count occurrences of broad data types per year
data_type_counts = datasets_with_tags_filtered.groupby(['Year', 'BroadDataType']).size().unstack(fill_value=0)
data_type_proportions = data_type_counts.apply(lambda x: x / x.sum(), axis=1)

print("\nDataset Type Proportions by Year:")
print(data_type_proportions.tail())

plt.figure(figsize=(12, 7))
data_type_proportions.plot(kind='area', stacked=True, colormap='Spectral', alpha=0.8, ax=plt.gca())
plt.title('Proportion of Dataset Types on Kaggle Over Time')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/kaggle/working/Analyzing_Dataset_Type_Distribution.png')
plt.show()

# For Dataset Sources (corporate, scientific, public):
# This is very challenging to infer programmatically from Meta Kaggle CSVs.
# It would require analyzing the 'CreatorId' and linking to organizations, or text analysis of dataset descriptions/titles.
print("\n**Note:** Inferring dataset sources (corporate, scientific, public) programmatically from Meta Kaggle CSVs is highly challenging. It would require complex analysis of creator organizations or extensive text mining of dataset metadata.")


# --- 4. Percentage distribution of different industry/application areas ---
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
competitions_industry_tags = pd.merge(df_competitions_filtered[['Id', 'Year']], df_competition_tags, left_on='Id', right_on='CompetitionId')
competitions_industry_tags['BroadIndustry'] = competitions_industry_tags['TagName'].map(industry_tags)
competitions_industry_tags_filtered = competitions_industry_tags.dropna(subset=['BroadIndustry'])

industry_comp_counts = competitions_industry_tags_filtered.groupby(['Year', 'BroadIndustry']).size().unstack(fill_value=0)
industry_comp_proportions = industry_comp_counts.apply(lambda x: x / x.sum(), axis=1)

print("\nIndustry/Application Area Proportions in Competitions by Year:")
print(industry_comp_proportions.tail())

plt.figure(figsize=(12, 7))
industry_comp_proportions.plot(kind='area', stacked=True, colormap='tab20', alpha=0.8, ax=plt.gca())
plt.title('Proportion of Industry/Application Areas in Kaggle Competitions Over Time')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Industry Area', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('/kaggle/working/Analyzing_Industry_trends.png')
plt.show()

# You could repeat a similar analysis for df_dataset_tags if you want to see trends in general datasets.

