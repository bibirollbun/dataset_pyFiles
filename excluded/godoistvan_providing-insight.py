import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train.head()


train.describe().T


test.describe().T


train.info()


test.info()


def remove_outliers(df, column_name):
    """Removes outliers from a specified column of a DataFrame using IQR method.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to remove outliers from.

    Returns:
        A new DataFrame with outliers removed.
    """

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return df_filtered

numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
for col in test[numerical_cols]:
    test = remove_outliers(test, col)



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

for col in numerical_cols:
    plt.figure(figsize=(10, 5))

    sns.kdeplot(train[col].dropna(), label='Train', fill=True)

    sns.kdeplot(test[col].dropna(), label='Test', fill=True)

    plt.title(f'Distribution Comparison for {col}')
    plt.legend()
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

non_numeric_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']


def print_top_words(wordcloud, dataset_name, column_name):
    if wordcloud is None or not wordcloud.words_:
        print(f"\nNo words to display for {dataset_name} - {column_name}")
        return

    total_freq = sum(wordcloud.words_.values())
    if total_freq == 0:
            print(f"\nTotal frequency is zero for {dataset_name} - {column_name}, cannot calculate percentages.")
            return

    word_percentages = {word: (freq / total_freq) * 100
                        for word, freq in wordcloud.words_.items()}

    sorted_words = sorted(word_percentages.items(), key=lambda item: item[1], reverse=True)
    top_20_words = sorted_words[:20]

    print(f"\nTop 20 words in {dataset_name} - {column_name}:")
    for word, percentage in top_20_words:
        print(f"- {word}: {percentage:.2f}%")

for col in non_numeric_cols:
    print(f"--- Processing Column: {col} ---")
    wordcloud_train = None
    wordcloud_test = None

    if col in train.columns and col in test.columns and not train[col].isnull().all() and not test[col].isnull().all():
        try:
            text_train = ' '.join(train[col].dropna().astype(str))
            if not text_train.strip():
                print(f"Skipping Train word cloud for '{col}' due to insufficient text data after cleaning.")
            else:
                print(f"Generating Train word cloud for {col}...")
                wordcloud_train = WordCloud(width=800, height=400, background_color='white').generate(text_train)
                print_top_words(wordcloud_train, "Train", col)

            # --- Process Test Data ---
            text_test = ' '.join(test[col].dropna().astype(str))
            if not text_test.strip():
                print(f"Skipping Test word cloud for '{col}' due to insufficient text data after cleaning.")
            else:
                print(f"\nGenerating Test word cloud for {col}...")
                wordcloud_test = WordCloud(width=800, height=400, background_color='white').generate(text_test)
                print_top_words(wordcloud_test, "Test", col)

            if wordcloud_train or wordcloud_test:
                print(f"\nDisplaying plot for {col}...")
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                fig.suptitle(f'Word Cloud Comparison for: {col}', fontsize=16)

                if wordcloud_train:
                    axes[0].imshow(wordcloud_train, interpolation='bilinear')
                    axes[0].set_title('Train Data')
                else:
                    axes[0].text(0.5, 0.5, 'No data/cloud generated', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
                axes[0].axis('off')

                if wordcloud_test:
                    axes[1].imshow(wordcloud_test, interpolation='bilinear')
                    axes[1].set_title('Test Data')
                else:
                     axes[1].text(0.5, 0.5, 'No data/cloud generated', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
                axes[1].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
            else:
                 print(f"\nSkipping plot display for '{col}' as no word clouds could be generated.")

            print("\n" + "="*30 + "\n")

        except Exception as e:
            print(f"Could not process column '{col}': {e}")
            print("\n" + "="*30 + "\n")
    else:
        print(f"Skipping '{col}' as it's missing in one of the dataframes or contains only null values.")
        print("\n" + "="*30 + "\n")


import pandas as pd
import matplotlib.pyplot as plt
import squarify

non_numeric_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for col in non_numeric_cols:
    print(f"--- Processing Column: {col} ---")

    if col in train.columns and col in test.columns:
        try:
            print(f"Calculating frequencies and percentages for {col}...")
            train_col_data = train[col].dropna()
            test_col_data = test[col].dropna()

            train_counts = train_col_data.value_counts()
            test_counts = test_col_data.value_counts()

            total_train = len(train_col_data)
            total_test = len(test_col_data)

            train_labels = []
            if total_train > 0:
                train_labels = [f"{name}\n({(count / total_train) * 100:.1f}%)"
                                for name, count in train_counts.items()]
            else:
                 train_labels = [f"{name}\n(N/A %)" for name in train_counts.index] # Handle zero total

            test_labels = []
            if total_test > 0:
                test_labels = [f"{name}\n({(count / total_test) * 100:.1f}%)"
                               for name, count in test_counts.items()]
            else:
                 test_labels = [f"{name}\n(N/A %)" for name in test_counts.index]


            if train_counts.empty and test_counts.empty:
                print(f"Skipping '{col}' - No data found after counting.")
                print("\n" + "="*30 + "\n")
                continue

            fig, axes = plt.subplots(1, 2, figsize=(18, 9))
            fig.suptitle(f'Treemap Comparison for: {col}', fontsize=16)
            ax_train = axes[0]
            ax_train.set_title('Train Data')
            ax_train.axis('off')
            if not train_counts.empty:
                try:
                    squarify.plot(sizes=train_counts.values,
                                  label=train_labels,
                                  alpha=0.8,
                                  ax=ax_train,
                                  text_kwargs={'fontsize': 16})
                    ax_train.set_title('Train Data')
                except Exception as e:
                    print(f"Error plotting Train treemap for {col}: {e}")
                    ax_train.text(0.5, 0.5, 'Error plotting treemap', horizontalalignment='center', verticalalignment='center', transform=ax_train.transAxes)
            else:
                ax_train.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax_train.transAxes)

            ax_test = axes[1]
            ax_test.set_title('Test Data')
            ax_test.axis('off')
            if not test_counts.empty:
                 try:
                    squarify.plot(sizes=test_counts.values,
                                label=test_labels,
                                alpha=0.8,
                                ax=ax_test,
                                text_kwargs={'fontsize': 16})
                    ax_test.set_title('Test Data')
                 except Exception as e:
                    print(f"Error plotting Test treemap for {col}: {e}")
                    ax_test.text(0.5, 0.5, 'Error plotting treemap', horizontalalignment='center', verticalalignment='center', transform=ax_test.transAxes)
            else:
                ax_test.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax_test.transAxes)

            print(f"Displaying plot for {col}...")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        except Exception as e:
            print(f"Could not process column '{col}': {e}")

    else:
        print(f"Skipping '{col}' as it's missing in one of the dataframes.")

    print("\n" + "="*30 + "\n")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Generating Correlation Heatmap (Train Set) ---")

# Select numerical columns (excluding 'id')
numerical_cols = train.select_dtypes(include=['float64', 'int64']).drop('id', axis=1, errors='ignore')

# Calculate the correlation matrix
correlation_matrix = numerical_cols.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features (Train Set)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\n" + "="*30 + "\n")


!pip install --upgrade seaborn


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
numerical_col_to_plot = 'Episode_Length_minutes'
categorical_col = 'Genre'
top_n_categories = 10
palette = sns.color_palette("viridis", top_n_categories)
# --- End Configuration ---

print(f"--- Generating Side-by-Side KDE Plots: {numerical_col_to_plot} by {categorical_col} ---")

# Check if necessary columns exist
if (numerical_col_to_plot in train.columns and categorical_col in train.columns and
        numerical_col_to_plot in test.columns and categorical_col in test.columns):

    # --- Find common top N categories across both datasets ---
    train_cat_counts = train[categorical_col].value_counts()
    test_cat_counts = test[categorical_col].value_counts()
    combined_counts = train_cat_counts.add(test_cat_counts, fill_value=0)
    top_categories = combined_counts.nlargest(top_n_categories).index.tolist()
    print(f"Comparing Top {len(top_categories)} common categories: {top_categories}")

    # Filter both dataframes
    train_filtered = train[train[categorical_col].isin(top_categories)].copy()
    test_filtered = test[test[categorical_col].isin(top_categories)].copy()

    # --- Create Subplots (one row per category, two columns for Train/Test) ---
    fig, axes = plt.subplots(top_n_categories, 2, figsize=(10, top_n_categories * 0.9), sharex=True)
    fig.suptitle(f'Distribution of {numerical_col_to_plot} by {categorical_col} (Top {top_n_categories})', fontsize=16, y=1.01)

    # --- Plot KDE for each category ---
    print("Plotting...")
    for i, category in enumerate(top_categories):
        color = palette[i % len(palette)] # Cycle through palette colors

        # Plot Train Data KDE
        ax_train = axes[i, 0]
        train_subset = train_filtered[train_filtered[categorical_col] == category][numerical_col_to_plot].dropna()
        if not train_subset.empty:
            sns.kdeplot(train_subset, ax=ax_train, color=color, fill=True, alpha=0.7, linewidth=1.5)
            ax_train.axhline(y=0, color='black', linewidth=1.0, linestyle='-', zorder=1)

        # Plot Test Data KDE
        ax_test = axes[i, 1]
        test_subset = test_filtered[test_filtered[categorical_col] == category][numerical_col_to_plot].dropna()
        if not test_subset.empty:
            sns.kdeplot(test_subset, ax=ax_test, color=color, fill=True, alpha=0.7, linewidth=1.5)
            ax_test.axhline(y=0, color='black', linewidth=1.0, linestyle='-', zorder=1)


        ax_train.text(-0.05, 0.5, category, transform=ax_train.transAxes,
                      fontsize=10, fontweight='bold', ha='right', va='center')


        # --- Axis Cleanup ---
        for ax in [ax_train, ax_test]:
             ax.set_yticks([])
             ax.set_ylabel('')

             if i < top_n_categories - 1:
                 ax.set_xlabel('')
                 ax.spines['bottom'].set_visible(False)
                 ax.tick_params(axis='x', which='both', bottom=False)
             else:
                 # Ensure bottom plots have label
                  ax.set_xlabel(f"{numerical_col_to_plot.replace('_', ' ')}")

             # Remove side/top spines for cleaner look
             ax.spines['right'].set_visible(False)
             ax.spines['left'].set_visible(False)
             ax.spines['top'].set_visible(False)

        # --- Set Column Titles (only for the top row) ---
        if i == 0:
            ax_train.set_title('Train Data', fontsize=12)
            ax_test.set_title('Test Data', fontsize=12)

    plt.subplots_adjust(hspace=0.3)
    plt.show()

else:
    print("Error: One or more required columns are missing from train or test data.")
    print(f"Required: '{numerical_col_to_plot}', '{categorical_col}' in both datasets.")

print("\n" + "="*30 + "\n")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Generating Pair Plot with Hue (Sampled Train Set) ---")

# Select key numerical columns and a categorical hue
cols_for_pairplot = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                     'Guest_Popularity_percentage', 'Number_of_Ads',
                     'Listening_Time_minutes', 'Episode_Sentiment']

# Check if columns exist
if all(col in train.columns for col in cols_for_pairplot):

    sample_size = 5000
    if len(train) > sample_size:
        print(f"Sampling {sample_size} data points for Pair Plot...")
        train_sample = train.sample(n=sample_size, random_state=42)
    else:
        train_sample = train

    print("Generating plot (this may take a moment)...")
    # Create the pair plot
    sns.pairplot(train_sample[cols_for_pairplot].dropna(),
                 hue='Episode_Sentiment',
                 palette='viridis',
                 diag_kind='kde',
                 plot_kws={'alpha': 0.6, 's': 10},
                 )
    plt.suptitle('Pairwise Relationships by Episode Sentiment (Sampled Train Data)', y=1.02)
    plt.show()

else:
    print(f"One or more required columns missing for Pair Plot: {cols_for_pairplot}")

print("\n" + "="*30 + "\n")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Generating Hexbin Plot: Host Popularity vs Listening Time (Train Set) ---")

# Check if necessary columns exist
if 'Host_Popularity_percentage' in train.columns and 'Listening_Time_minutes' in train.columns:

    # Using jointplot for hexbin with marginal distributions
    sns.jointplot(data=train,
                  x='Host_Popularity_percentage',
                  y='Listening_Time_minutes',
                  kind='hex',
                  cmap='viridis',
                  gridsize=40
                 )
    plt.suptitle('Density of Host Popularity vs. Listening Time (Train Set)', y=1.02)
    plt.tight_layout()
    plt.show()

else:
    print("Required columns ('Host_Popularity_percentage', 'Listening_Time_minutes') not found in train data.")

print("\n" + "="*30 + "\n")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Generating Hexbin Plot: Guest Popularity vs Listening Time (Train Set) ---")

# Check if necessary columns exist
if 'Guest_Popularity_percentage' in train.columns and 'Listening_Time_minutes' in train.columns:

    # Using jointplot for hexbin with marginal distributions
    sns.jointplot(data=train,
                  x='Guest_Popularity_percentage',
                  y='Listening_Time_minutes',
                  kind='hex',
                  cmap='viridis',
                  gridsize=40
                 )
    plt.suptitle('Density of Guest Popularity vs. Listening Time (Train Set)', y=1.02)
    plt.tight_layout()
    plt.show()

else:
    print("Required columns ('Guest_Popularity_percentage', 'Listening_Time_minutes') not found in train data.")

print("\n" + "="*30 + "\n")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


print("--- Generating Comparison Box Plots for Numerical Columns ---")

common_numerical_cols = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads'
]

cols_to_plot = []
for col in common_numerical_cols:
    if col in train.columns and col in test.columns:
        cols_to_plot.append(col)
    else:
        print(f"Warning: Column '{col}' not found in both train and test sets. Skipping.")

for col_name in cols_to_plot:
    print(f"Generating box plot for: {col_name}")

    try:
        train_data = train[[col_name]].copy()
        train_data['dataset'] = 'Train'

        test_data = test[[col_name]].copy()
        test_data['dataset'] = 'Test'

        # Combine the data
        combined_df = pd.concat([train_data, test_data], ignore_index=True)

        # --- Create the plot ---
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(data=combined_df, x='dataset', y=col_name,
                         palette={'Train': 'skyblue', 'Test': 'lightcoral'})

        # --- Customize ---
        plt.title(f'Comparison of {col_name.replace("_", " ")} Distribution', fontsize=14)
        plt.xlabel("Dataset", fontsize=12)
        plt.ylabel(col_name.replace("_", " "), fontsize=12)


        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Could not generate box plot for column '{col_name}': {e}")

    print("-" * 20)

print("--- Plot generation complete ---")

target_col = 'Listening_Time_minutes'
if target_col in train.columns:
     print(f"Generating box plot for train-only column: {target_col}")
     try:
         plt.figure(figsize=(6, 5))
         sns.boxplot(data=train, y=target_col, color='lightgreen')
         plt.title(f'Distribution of {target_col.replace("_", " ")} (Train Set)', fontsize=14)
         plt.ylabel(target_col.replace("_", " "), fontsize=12)
         plt.tight_layout()
         plt.show()
     except Exception as e:
         print(f"Could not generate box plot for column '{target_col}': {e}")


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



print("--- Generating Faceted Scatter Plot (Length vs Listening Time by Genre) ---")
# Sampling because of dataset size
train_sample = train.sample(n=10000, random_state=42) if len(train) > 10000 else train

# Limit to top N genres for readability
top_genres_facet = train_sample['Genre'].value_counts().nlargest(20).index # e.g., Top 6 genres
train_sample_filtered = train_sample[train_sample['Genre'].isin(top_genres_facet)]

if not train_sample_filtered.empty:
    g = sns.relplot(
        data=train_sample_filtered,
        x="Episode_Length_minutes",
        y="Listening_Time_minutes",
        col="Genre",
        col_wrap=3,
        hue="Genre",
        kind="scatter",
        alpha=0.5,
        height=3, aspect=1.2 # Adjust size
    )
    g.fig.suptitle('Episode Length vs. Listening Time across Top Genres (Sampled)', y=1.03)
    plt.show()
else:
    print("Not enough data for selected top genres.")

print("\\n" + "="*30 + "\\n")


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

print("--- Generating Categorical Interaction Plot (Avg Listening Time by Genre & Sentiment) ---")
if 'Genre' in train.columns and 'Episode_Sentiment' in train.columns and 'Listening_Time_minutes' in train.columns:
    top_genres_cat = train['Genre'].value_counts().nlargest(20).index
    train_filtered_cat = train[train['Genre'].isin(top_genres_cat)]

    sns.catplot(
        data=train_filtered_cat,
        x="Genre",
        y="Listening_Time_minutes",
        hue="Episode_Sentiment",
        kind="bar",
        order=top_genres_cat,
        height=5, aspect=1.8,
        errorbar=None #
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Listening Time by Genre and Episode Sentiment')
    plt.tight_layout()
    plt.show()
else:
    print("Required columns missing for categorical interaction plot.")
print("\\n" + "="*30 + "\\n")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
print("--- Generating Time-Based Plot (Avg Listening Time by Day and Time of Day) ---")

# --- Preprocessing ---
try:
    # --- Day of Week ---
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if 'Publication_Day' in train.columns:
        print("Processing Publication_Day...")
        # Ensure it's categorical with the correct order
        train['Day_of_Week'] = pd.Categorical(train['Publication_Day'], categories=days_order, ordered=True)
    else:
         raise ValueError("Column 'Publication_Day' not found.")

    # --- Time of Day (using the existing column) ---
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night'] # Define desired order
    if 'Publication_Time' in train.columns:
        print("Processing Publication_Time (as existing categories)...")
        train['Time_of_Day'] = pd.Categorical(train['Publication_Time'], categories=time_order, ordered=True)

        if train['Time_of_Day'].isnull().any() and not train['Publication_Time'].isnull().all():
             original_nulls = train['Publication_Time'].isnull().sum()
             new_nulls = train['Time_of_Day'].isnull().sum()
             unmapped_count = new_nulls - original_nulls
             if unmapped_count > 0:
                  print(f"Warning: {unmapped_count} entries in 'Publication_Time' did not match expected categories {time_order} and resulted in Null.")
    else:
        raise ValueError("Column 'Publication_Time' not found.")

    # --- Check if preprocessing yielded results ---
    if 'Day_of_Week' not in train.columns or 'Time_of_Day' not in train.columns:
        raise ValueError("Preprocessing failed to create Day_of_Week or Time_of_Day columns.")

    # --- Plotting (remains the same as before) ---
    print("Generating plot...")
    plt.figure(figsize=(12, 6)) # Wider figure for legend
    palette = sns.color_palette("tab10", n_colors=len(time_order))

    # Drop rows where Day_of_Week, Time_of_Day, or the target are NaN before plotting
    plot_data = train.dropna(subset=['Day_of_Week', 'Time_of_Day', 'Listening_Time_minutes'])

    if not plot_data.empty:
        sns.lineplot(
            data=plot_data,
            x='Day_of_Week',
            y='Listening_Time_minutes',
            hue='Time_of_Day',
            hue_order=time_order,
            palette=palette,
            marker='o',
            errorbar=('ci', 99)
        )
        plt.title('Average Listening Time by Publication Day and Time of Day')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Listening Time (minutes)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Time of Day', bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plt.show()
    else:
        print("No data available to plot after filtering NaNs.")


except Exception as e:
    print(f"Could not perform time-based analysis: {e}")

print("\n" + "="*30 + "\n")


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

col_to_compare = 'Episode_Length_minutes'

if col_to_compare in train.columns and col_to_compare in test.columns:
    print(f"--- Generating CDF Plot for {col_to_compare} (Train vs Test) ---")
    plt.figure(figsize=(10, 6))

    sns.ecdfplot(data=train, x=col_to_compare, label='Train', linewidth=2)
    sns.ecdfplot(data=test, x=col_to_compare, label='Test', linewidth=2, linestyle='--')

    plt.title(f'Cumulative Distribution of {col_to_compare.replace("_"," ")}')
    plt.xlabel(col_to_compare.replace("_"," "))
    plt.ylabel('Proportion of Data')
    plt.legend()
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.show()
    print("\\n" + "="*30 + "\\n")
else:
    print(f"Column {col_to_compare} not found in both datasets.")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



print("--- Analyzing Derived Feature: Ads per Minute ---")
if 'Number_of_Ads' in train.columns and 'Episode_Length_minutes' in train.columns and 'Listening_Time_minutes' in train.columns:
    temp_df = train[['Number_of_Ads', 'Episode_Length_minutes', 'Listening_Time_minutes']].dropna().copy()

    temp_df['Ads_per_Minute'] = temp_df['Number_of_Ads'] / temp_df['Episode_Length_minutes'].replace(0, np.nan) # Avoid division by zero
    temp_df = temp_df.dropna(subset=['Ads_per_Minute']) # Remove results from zero-length episodes

    # Cap extreme values if needed for plotting (e.g., ads in very short episodes)
    cap_value = temp_df['Ads_per_Minute'].quantile(0.99)
    temp_df['Ads_per_Minute_Capped'] = temp_df['Ads_per_Minute'].clip(upper=cap_value)


    # Visualize the distribution of the derived feature
    plt.figure(figsize=(10, 5))
    sns.histplot(temp_df['Ads_per_Minute_Capped'], kde=True, bins=50)
    plt.title('Distribution of Ads per Minute (Capped at 99th percentile)')
    plt.xlabel('Ads per Minute')
    plt.show()

    try:
         temp_df['Ads_per_Minute_Bin'] = pd.qcut(temp_df['Ads_per_Minute_Capped'], q=10, labels=False, duplicates='drop')
         plt.figure(figsize=(10, 6))
         sns.lineplot(data=temp_df, x='Ads_per_Minute_Bin', y='Listening_Time_minutes', marker='o', errorbar=('ci', 95))
         plt.title('Average Listening Time vs. Ads per Minute (Binned)')
         plt.xlabel('Ads per Minute (Quantile Bin)')
         plt.ylabel('Average Listening Time (minutes)')
         plt.show()
    except ValueError as e:
         print(f"Could not create bins for Ads_per_Minute, maybe too few unique values? Error: {e}")
    except Exception as e:
         print(f"An error occurred while plotting binned data: {e}")
    finally:
        print("\\n" + "="*30 + "\\n")
else:
    print("Required columns (Number_of_Ads, Episode_Length_minutes) not found.")


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

col_to_check = 'Host_Popularity_percentage'
if col_to_check in train.columns:
    print(f"--- Generating Q-Q Plot for {col_to_check} vs Normal Distribution (Train) ---")
    plt.figure(figsize=(6, 6))
    sm.qqplot(train[col_to_check].dropna(), line='s', fit=True)
    plt.title(f'Q-Q Plot: {col_to_check.replace("_"," ")} vs Normal')
    plt.show()
    print("\\n" + "="*30 + "\\n")

col_to_compare = 'Episode_Length_minutes'
if col_to_compare in train.columns and col_to_compare in test.columns:
    print(f"--- Generating Q-Q Plot for {col_to_compare} (Train vs Test) ---")
    quantiles = np.linspace(0.01, 0.99, 100) # 100 points from 1st to 99th percentile
    train_quantiles = train[col_to_compare].dropna().quantile(quantiles)
    test_quantiles = test[col_to_compare].dropna().quantile(quantiles)

    plt.figure(figsize=(6, 6))
    plt.scatter(train_quantiles, test_quantiles, alpha=0.7)
    limits = [min(train_quantiles.min(), test_quantiles.min()), max(train_quantiles.max(), test_quantiles.max())]
    plt.plot(limits, limits, color='red', linestyle='--')
    plt.xlabel('Train Quantiles')
    plt.ylabel('Test Quantiles')
    plt.title(f'Q-Q Plot: {col_to_compare.replace("_"," ")} (Train vs Test)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    print("\\n" + "="*30 + "\\n")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


print("--- Generating Visualizations for Derived Features ---")


df = train.copy()

print("\n--- Feature 1: Ads per Minute ---")
if 'Number_of_Ads' in df.columns and 'Episode_Length_minutes' in df.columns:
    # Calculate (handle zero length)
    df['Ads_per_Minute'] = df['Number_of_Ads'] / df['Episode_Length_minutes'].replace(0, np.nan)
    df_plot = df[['Ads_per_Minute', 'Listening_Time_minutes']].dropna()

    # Cap extreme values for better scatter plot visibility
    cap_value_apm = df_plot['Ads_per_Minute'].quantile(0.99)
    df_plot['Ads_per_Minute_Capped'] = df_plot['Ads_per_Minute'].clip(upper=cap_value_apm)

    # Visualize relationship with target (Scatter plot with alpha)
    if not df_plot.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x='Ads_per_Minute_Capped', y='Listening_Time_minutes', alpha=0.1, s=10)
        plt.title('Listening Time vs. Ads per Minute (Capped at 99th percentile)')
        plt.xlabel('Ads per Minute (Capped)')
        plt.ylabel('Listening Time (minutes)')
        plt.show()
    else:
        print("No data to plot for Ads per Minute vs Listening Time.")
else:
    print("Required columns missing for Ads per Minute.")

print("\n--- Feature 2: Popularity Ratio (Guest/Host) ---")
if 'Guest_Popularity_percentage' in df.columns and 'Host_Popularity_percentage' in df.columns:
    guest_pop = df['Guest_Popularity_percentage'].fillna(0)
    host_pop = df['Host_Popularity_percentage']
    epsilon = 1e-6
    df['Popularity_Ratio'] = guest_pop / (host_pop + epsilon)

    q_low_pr = df['Popularity_Ratio'].quantile(0.01)
    q_high_pr = df['Popularity_Ratio'].quantile(0.99)
    df['Popularity_Ratio_Capped'] = df['Popularity_Ratio'].clip(lower=q_low_pr, upper=q_high_pr)

    df_plot_pr = df[['Popularity_Ratio_Capped', 'Listening_Time_minutes']].dropna()

    if not df_plot_pr.empty:
        plt.figure(figsize=(10, 5))
        sns.histplot(df_plot_pr['Popularity_Ratio_Capped'], kde=True, bins=50)
        plt.title('Distribution of Popularity Ratio (Guest/Host, Capped 1st-99th percentile)')
        plt.xlabel('Popularity Ratio (Guest/Host)')
        plt.show()

        try:
            df_plot_pr['Popularity_Ratio_Bin'] = pd.qcut(df_plot_pr['Popularity_Ratio_Capped'], q=10, labels=False, duplicates='drop')
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_plot_pr, x='Popularity_Ratio_Bin', y='Listening_Time_minutes', marker='o', errorbar=('ci', 95))
            plt.title('Average Listening Time vs. Popularity Ratio (Binned)')
            plt.xlabel('Popularity Ratio (Quantile Bin)')
            plt.ylabel('Average Listening Time (minutes)')
            plt.show()
        except Exception as e:
            print(f"Could not plot binned Popularity Ratio vs Listening Time: {e}")

    else:
        print("No data to plot for Popularity Ratio.")
else:
    print("Required columns missing for Popularity Ratio.")


print("\n--- Feature 3: Total Popularity ---")
if 'Host_Popularity_percentage' in df.columns and 'Guest_Popularity_percentage' in df.columns:
    # Calculation (fill missing guests with 0)
    df['Total_Popularity'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage'].fillna(0)
    df_plot_tp = df[['Total_Popularity', 'Listening_Time_minutes']].dropna()

    if not df_plot_tp.empty:
        sns.jointplot(data=df_plot_tp, x='Total_Popularity', y='Listening_Time_minutes', kind='hex', cmap='viridis', gridsize=40)
        plt.suptitle('Density of Total Popularity vs. Listening Time', y=1.02)
        plt.tight_layout()
        plt.show()
    else:
        print("No data to plot for Total Popularity.")
else:
    print("Required columns missing for Total Popularity.")


print("\n--- Feature 5: Length Deviation from Genre Mean ---")
if 'Genre' in df.columns and 'Episode_Length_minutes' in df.columns:
    # Calculation
    df['Mean_Length_for_Genre'] = df.groupby('Genre')['Episode_Length_minutes'].transform('mean')
    df['Length_Deviation'] = df['Episode_Length_minutes'] - df['Mean_Length_for_Genre']
    df_plot_ld = df[['Length_Deviation', 'Listening_Time_minutes']].dropna()

    # Visualize distribution of the deviation
    if not df_plot_ld.empty:
        plt.figure(figsize=(10, 5))
        sns.histplot(df_plot_ld['Length_Deviation'], kde=True, bins=50)
        plt.title('Distribution of Episode Length Deviation from Genre Mean')
        plt.xlabel('Length Deviation (minutes)')
        plt.show()

        # Visualize relationship with target (Scatter plot with alpha)
        q_low_ld = df_plot_ld['Length_Deviation'].quantile(0.01)
        q_high_ld = df_plot_ld['Length_Deviation'].quantile(0.99)
        df_plot_ld['Length_Deviation_Capped'] = df_plot_ld['Length_Deviation'].clip(lower=q_low_ld, upper=q_high_ld)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_plot_ld, x='Length_Deviation_Capped', y='Listening_Time_minutes', alpha=0.1, s=10)
        plt.title('Listening Time vs. Length Deviation from Genre Mean (Capped)')
        plt.xlabel('Length Deviation (minutes, Capped 1st-99th percentile)')
        plt.ylabel('Listening Time (minutes)')
        plt.show()
    else:
        print("No data to plot for Length Deviation.")
else:
    print("Required columns 'Genre' or 'Episode_Length_minutes' missing.")


print("\n--- Feature 6: Day-Time Combined Slot ---")
if 'Day_of_Week' in df.columns and 'Time_of_Day' in df.columns:
    # Calculation
    df['Day_Time_Combined'] = df['Day_of_Week'].astype(str) + '-' + df['Time_of_Day'].astype(str)
    df_plot_dtc = df[['Day_Time_Combined', 'Listening_Time_minutes']].dropna()

    # Visualize average listening time per slot (Top N slots)
    if not df_plot_dtc.empty:
        # Find top N most frequent slots
        n_top_slots = 15
        top_slots = df_plot_dtc['Day_Time_Combined'].value_counts().nlargest(n_top_slots).index
        df_plot_dtc_filtered = df_plot_dtc[df_plot_dtc['Day_Time_Combined'].isin(top_slots)]

        # Calculate average listening time per slot
        avg_listening_per_slot = df_plot_dtc_filtered.groupby('Day_Time_Combined')['Listening_Time_minutes'].mean().reset_index()
        avg_listening_per_slot = avg_listening_per_slot.sort_values('Listening_Time_minutes', ascending=False)


        plt.figure(figsize=(12, 7))
        sns.barplot(data=avg_listening_per_slot,
                    x='Day_Time_Combined',
                    y='Listening_Time_minutes',
                    palette='viridis',
                    order = avg_listening_per_slot['Day_Time_Combined'] # Order by avg time
                   )
        plt.title(f'Average Listening Time per Combined Day-Time Slot (Top {n_top_slots} Slots)')
        plt.xlabel('Day - Time Slot')
        plt.ylabel('Average Listening Time (minutes)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No data to plot for Day-Time Combined.")
else:
    print("Required columns 'Day_of_Week' or 'Time_of_Day' missing. Run previous time preprocessing steps first.")


print("\n--- Derived feature visualization complete ---")


import pandas as pd
import numpy as np

print("Loading datasets...")
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
    test_df  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
    print("Datasets loaded successfully.")
    print("\nOriginal Train Info:")
    train_df.info()
    print("\nOriginal Test Info:")
    test_df.info()
except FileNotFoundError:
    print("Error: train.csv or test.csv not found. Please ensure they are in the correct directory.")
    exit() # Or raise an exception

# --- Feature Engineering ---

# 1. Calculate Genre Mean Length from TRAINING data ONLY
print("\nCalculating mean episode length per genre from training data...")
genre_mean_lengths = train_df.groupby('Genre')['Episode_Length_minutes'].mean()
# print(genre_mean_lengths) # Optional: view the means

# 2. Define the feature engineering function
def engineer_features(df, genre_means):
    """Applies derived feature calculations to the dataframe."""
    print(f"\nEngineering features for dataframe with shape {df.shape}...")
    # Make a copy to avoid modifying the original slice if df is one
    df = df.copy()

    # a) Ads per Minute
    print("- Calculating Ads per Minute...")
    # Replace 0 length with NaN before division, then fill resulting NaN Ads_per_Minute with 0
    df['Ads_per_Minute'] = df['Number_of_Ads'] / df['Episode_Length_minutes'].replace(0, np.nan)
    df['Ads_per_Minute'] = df['Ads_per_Minute'].fillna(0) # Fill NaNs resulting from division or original NaNs

    # b) Popularity Ratio (Guest/Host)
    print("- Calculating Popularity Ratio...")
    # Fill missing guest pop with 0, handle host pop = 0 or NaN by adding epsilon
    guest_pop = df['Guest_Popularity_percentage'].fillna(0)
    host_pop = df['Host_Popularity_percentage']
    epsilon = 1e-6
    # Add epsilon only where host_pop is not NaN, avoids issues if host_pop itself is NaN
    df['Popularity_Ratio'] = guest_pop / (host_pop.fillna(0) + epsilon)
    # Handle cases where original host_pop was NaN -> set ratio to NaN or 0? Let's use 0.
    df.loc[host_pop.isnull(), 'Popularity_Ratio'] = 0


    # c) Total Popularity (Host + Guest)
    print("- Calculating Total Popularity...")
    # Fill missing guest pop with 0
    df['Total_Popularity'] = df['Host_Popularity_percentage'].fillna(0) + df['Guest_Popularity_percentage'].fillna(0)

    # d) Is Weekend
    print("- Calculating Is Weekend...")
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday'])

    # e) Length Deviation from Genre Mean (Using pre-calculated train means)
    print("- Calculating Length Deviation from Genre Mean...")
    df['Length_Deviation'] = df['Episode_Length_minutes'] - df['Genre'].map(genre_means)
    # Fill NaNs: could be due to missing original length OR genre not seen in train set
    df['Length_Deviation'] = df['Length_Deviation'].fillna(0)

    # f) Day of Week (Ordered Categorical)
    print("- Processing Day of Week...")
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df['Day_of_Week'] = pd.Categorical(df['Publication_Day'], categories=days_order, ordered=True)

    # g) Time of Day (Ordered Categorical - from existing column)
    print("- Processing Time of Day...")
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    # Convert the existing column to an ordered categorical type
    df['Time_of_Day'] = pd.Categorical(df['Publication_Time'], categories=time_order, ordered=True)
    # Check for values not mapping correctly (optional, good practice)
    if df['Time_of_Day'].isnull().any() and not df['Publication_Time'].isnull().all():
         original_nulls = df['Publication_Time'].isnull().sum()
         new_nulls = df['Time_of_Day'].isnull().sum()
         unmapped_count = new_nulls - original_nulls
         if unmapped_count > 0:
              print(f"  Warning: {unmapped_count} entries in 'Publication_Time' did not map to expected categories {time_order}.")


    # h) Day-Time Combined
    print("- Calculating Day-Time Combined...")
    # Combine the categorical representations
    df['Day_Time_Combined'] = df['Day_of_Week'].astype(str) + '-' + df['Time_of_Day'].astype(str)
    # Handle potential 'NaN-NaN' if source columns had NaNs not caught earlier
    df['Day_Time_Combined'] = df['Day_Time_Combined'].replace('nan-nan', np.nan)


    print("Feature engineering complete.")
    return df

# 3. Apply the function to both DataFrames
train_df_processed = engineer_features(train_df, genre_mean_lengths)
test_df_processed = engineer_features(test_df, genre_mean_lengths)

# --- Verify Results ---
print("\n--- Processed Train Info: ---")
train_df_processed.info()

print("\n--- Processed Test Info: ---")
test_df_processed.info()

print("\n--- First 5 rows of Processed Train Data: ---")
print(train_df_processed.head())

print("\n--- First 5 rows of Processed Test Data: ---")
print(test_df_processed.head())


print(f' Missing train data {train_df_processed.isnull().sum()}')
print(f' Missing test data {test_df_processed.isnull().sum()}')
print(f' Duplicated train data {train_df_processed.duplicated().sum()}')
print(f' Duplicated test data {test_df_processed.duplicated().sum()}')


import pandas as pd
import numpy as np

print("--- Handling Missing Data ---")

# --- Impute Episode_Length_minutes ---
print("Imputing Episode_Length_minutes with median...")
# Calculate median ONLY from train set
median_length = train_df_processed['Episode_Length_minutes'].median()
print(f"Train median Episode_Length_minutes: {median_length:.2f}")
# Fill NaNs in both train and test
train_df_processed['Episode_Length_minutes'] = train_df_processed['Episode_Length_minutes'].fillna(median_length)
test_df_processed['Episode_Length_minutes'] = test_df_processed['Episode_Length_minutes'].fillna(median_length)

# --- Impute Guest_Popularity_percentage (and create indicator) ---
print("Creating Guest_Popularity_Missing indicator and imputing with 0...")
# Create indicator column BEFORE imputing
train_df_processed['Guest_Popularity_Missing'] = train_df_processed['Guest_Popularity_percentage'].isnull().astype(int)
test_df_processed['Guest_Popularity_Missing'] = test_df_processed['Guest_Popularity_percentage'].isnull().astype(int)
# Impute with 0
train_df_processed['Guest_Popularity_percentage'] = train_df_processed['Guest_Popularity_percentage'].fillna(0)
test_df_processed['Guest_Popularity_percentage'] = test_df_processed['Guest_Popularity_percentage'].fillna(0)

# --- Impute Number_of_Ads ---
print("Imputing Number_of_Ads with median...")
# Calculate median ONLY from train set
median_ads = train_df_processed['Number_of_Ads'].median()
print(f"Train median Number_of_Ads: {median_ads:.2f}")
# Fill NaNs in both train and test (though test has none, it's good practice)
train_df_processed['Number_of_Ads'] = train_df_processed['Number_of_Ads'].fillna(median_ads)
test_df_processed['Number_of_Ads'] = test_df_processed['Number_of_Ads'].fillna(median_ads)


# --- Verify Missing Values ---
print("\n--- Missing Data After Imputation ---")
print("\nMissing train data:")
print(train_df_processed.isnull().sum())
print("\nMissing test data:")
print(test_df_processed.isnull().sum())

# Check info for new indicator column
print("\n--- Processed Train Info (Post-Imputation): ---")
train_df_processed.info()


import numpy as np

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

common_numerical_cols = train_df_processed.select_dtypes(include=np.number).columns.intersection(test_df_processed.columns)

for col in common_numerical_cols:
    train_outliers = detect_outliers(train_df_processed, col)
    test_outliers = detect_outliers(test_df_processed, col)
    print(f"Column: {col}")
    print(f"  Train Outliers: {len(train_outliers)} ({len(train_outliers)/len(train_df_processed)*100:.2f}%)")
    print(f"  Test Outliers: {len(test_outliers)} ({len(test_outliers)/len(test_df_processed)*100:.2f}%)")


train_df_processed.info()



import pandas as pd
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

categorical_cols_train = train_df_processed.select_dtypes(include=['category', 'object']).columns
categorical_cols_test = test_df_processed.select_dtypes(include=['category', 'object']).columns

common_categorical_cols = categorical_cols_train.intersection(categorical_cols_test)

for col in common_categorical_cols:
    unique_values = pd.concat([train_df_processed[col], test_df_processed[col]]).unique()
    le.fit(unique_values)
    train_df_processed[col] = le.transform(train_df_processed[col])
    test_df_processed[col] = le.transform(test_df_processed[col])



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
X=train_df_processed.drop('Listening_Time_minutes',axis=1)
y=train_df_processed['Listening_Time_minutes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error

# Initialize models
models = {
    'CatBoost': CatBoostRegressor(verbose=0),
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = rmse
    print(f"{name} RMSE: {rmse}")

# Rank models by RMSE
ranked_models = sorted(results.items(), key=lambda x: x[1])

print("\nModel Ranking (by RMSE):")
for name, rmse in ranked_models:
    print(f"{name}: {rmse}")


