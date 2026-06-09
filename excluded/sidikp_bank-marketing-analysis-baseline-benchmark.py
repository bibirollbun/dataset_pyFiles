import pandas as pd, os, matplotlib.pyplot as plt, seaborn as sns, numpy as np, math
from matplotlib.patches import Patch

import warnings
warnings.filterwarnings("ignore")


# Example: load CSV
analysis_df = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')
print(analysis_df.head(2))


# Prediction data
data_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
data_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


data_train.replace([np.inf, -np.inf], np.nan, inplace=True)
data_test.replace([np.inf, -np.inf], np.nan, inplace=True)
analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)


def basic_info(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print('-'*25, '\nData Type of Dataframe')
    print('-'*25)
    display(data.dtypes)
    print('-'*25, '\nAll Column Descriptive')
    print('-'*25)
    display(data.describe())
    print('-'*25, '\nNumber Unique Value')
    print('-'*25)
    display(data.nunique())
    print('-'*25, '\nColumn Counts for every value')
    print('-'*25)
    for col in data.columns:
        if data[col].dtype == 'object':
            display(data[col].value_counts())
        else:
            print(f"{col} not object datatype")
    print('-'*25, '\nCount of Null')
    print('-'*25)
    display(data.isnull().sum())


print("="*25, '\n Data Train')
display(data_train.head(2))
display(data_train.shape)

print("="*25,  '\n Data Test')
display(data_test.head(2))
display(data_test.shape)


basic_info(data_train)


basic_info(data_test)


def compare_distributions_grid(df_train, df_test, sample_size=10000):
    # Mark source
    df_train = df_train.copy()
    df_train['is_test'] = False
    df_test = df_test.copy()
    df_test['is_test'] = True

    # Combine
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # Numeric columns only
    numeric_cols = df_all.select_dtypes(include=["int", "float"]).columns.tolist()

    # Optional sampling for speed
    if len(df_all) > sample_size:
        df_all = df_all.sample(sample_size, random_state=42)

    # Setup grid layout: 3 columns, rows = ceil(n / 3)
    n_cols = 3
    n_rows = math.ceil(len(numeric_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    # Plot each column
    for i, col in enumerate(numeric_cols):
        sns.histplot(
            data=df_all,
            x=col,
            hue="is_test",
            kde=True,
            stat="density",
            common_norm=False,
            ax=axes[i],
            element="step"
        )
        axes[i].set_title(f"{col} Distribution")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")

    # Turn off unused subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

compare_distributions_grid(data_train, data_test)


def compare_categorical_distributions(df_train, df_test, sample_size=10000):
    # Copy and label source
    df_train = df_train.copy()
    df_train["is_test"] = "Train"
    df_test = df_test.copy()
    df_test["is_test"] = "Test"

    # Concatenate
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # Reduce size if needed
    if len(df_all) > sample_size:
        df_all = df_all.sample(sample_size, random_state=42)

    # Get categorical columns (excluding is_test/target)
    cat_cols = df_all.select_dtypes(include=["object", "category", "bool"]).columns
    cat_cols = [col for col in cat_cols if col != "is_test" and col != "y"]

    # Layout: 3 columns, dynamic rows
    n_cols = 3
    n_rows = math.ceil(len(cat_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        n_unique = df_all[col].nunique()

        if n_unique > 4:
            sns.countplot(
                data=df_all,
                y=col,
                hue="is_test",
                ax=axes[i],
                order=df_all[col].value_counts().index
            )
            axes[i].set_title(f"{col} Distribution (horizontal)")
            axes[i].set_ylabel(col)
            axes[i].set_xlabel("Count")
        else:
            sns.countplot(
                data=df_all,
                x=col,
                hue="is_test",
                ax=axes[i],
                order=df_all[col].value_counts().index
            )
            axes[i].set_title(f"{col} Distribution")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis='x', rotation=30)

    # Hide unused subplots
    for j in range(len(cat_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

compare_categorical_distributions(data_train, data_test)


sns.set(style="whitegrid")
palette = {"yes": "#bea6a1", "no": "#d62728"}
handles = [
    Patch(color=palette['no'], label="No"),
    Patch(color=palette['yes'], label="Yes")
]

def visualize_age_vs_subscription(df):    
    # Setup 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # KDE Histogram by 'y'
    sns.histplot(
        data=df,
        x='age',
        hue='y',
        kde=True,
        stat='density',
        common_norm=False,
        bins=30,
        element='step',
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title('Age Distribution by Subscription')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Density')
    axes[0].legend(handles=handles, title='Subscribed', labels=['No', 'Yes'], loc='upper right')

    # Violin plot
    sns.violinplot(
        data=df,
        x='y',
        y='age',
        inner='quartile',
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title('Age vs Subscription (Violin Plot)')
    axes[1].set_xlabel('Subscribed')
    axes[1].set_ylabel('Age')

    # Final layout
    plt.tight_layout()
    plt.show()

visualize_age_vs_subscription(analysis_df)


def visualize_job_vs_subscription(df):

    # --- Prepare proportional data ---
    prop_df = (
        df.groupby(["job", "y"])
          .size()
          .reset_index(name="count")
    )

    total_per_job = prop_df.groupby("job")["count"].transform("sum")
    prop_df["percentage"] = prop_df["count"] / total_per_job

    # Sort jobs by subscription rate
    job_order = (
        prop_df[prop_df["y"] == "yes"]
        .sort_values("percentage", ascending=False)["job"]
        .tolist()
    )

    # --- Setup figure ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Countplot (grouped bar)
    sns.countplot(
        data=df,
        x="job",
        hue="y",
        palette=palette,
        order=job_order,
        ax=axes[0]
    )
    axes[0].set_title("Subscription Count by Job")
    axes[0].set_xlabel("Job")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title="Subscribed")

    # 2. Percentage barplot
    sns.barplot(
        data=prop_df[prop_df["y"] == "yes"],
        x="job",
        y="percentage",
        palette="Reds",
        order=job_order,
        ax=axes[1]
    )
    axes[1].set_title("Subscription Rate by Job")
    axes[1].set_xlabel("Job")
    axes[1].set_ylabel("Subscription Rate (%)")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    plt.tight_layout()
    plt.show()

visualize_job_vs_subscription(analysis_df)


def visualize_marital_status_vs_subscription(df):
    # Ensure you're working with correct data subset
    df_plot = df[df["y"].notna()]

    # Set figure and axes for side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.3)

    # LEFT PLOT â€“ Countplot with hue (absolute comparison)
    sns.countplot(
        data=df_plot,
        x="marital",
        hue="y",
        order=df_plot["marital"].value_counts().index,
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title("Deposit Subscription by Marital Status")
    axes[0].set_xlabel("Marital Status")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Subscribed", labels=["No", "Yes"])
    axes[0].tick_params(axis='x', rotation=30)

    # RIGHT PLOT â€“ Percentage breakdown (relative comparison)
    total = df_plot.groupby("marital").size()
    count = df_plot.groupby(["marital", "y"]).size()
    marital_group = (count / total).rename("proportion").reset_index()

    print(marital_group)
    
    sns.barplot(
        data=marital_group,
        x="marital",
        y="proportion",
        hue="y",
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title("Proportion of Subscriptions by Marital Status (Percentage)")
    axes[1].set_xlabel("Marital Status")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(handles=handles, title="Subscribed")
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()

visualize_marital_status_vs_subscription(analysis_df)


def visualize_education_vs_subscription(df):
    # Ensure you're working with correct data subset
    df_plot = df[df["y"].notna()]

    # Set figure and axes for side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.3)

    # LEFT PLOT â€“ Countplot with hue (absolute comparison)
    sns.countplot(
        data=df_plot,
        x="education",
        hue="y",
        order=df_plot["education"].value_counts().index,
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title("Deposit Subscription by Education Status")
    axes[0].set_xlabel("Education Status")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Subscribed", labels=["No", "Yes"])
    axes[0].tick_params(axis='x', rotation=30)

    # RIGHT PLOT â€“ Percentage breakdown (relative comparison)
    total = df_plot.groupby("education").size()
    count = df_plot.groupby(["education", "y"]).size()
    education_group = (count / total).rename("proportion").reset_index()

    sns.barplot(
        data=education_group,
        x="education",
        y="proportion",
        hue="y",
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title("Proportion of Subscriptions by Education Status (Percentage)")
    axes[1].set_xlabel("Education Status")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(handles=handles, title="Subscribed")
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()

visualize_education_vs_subscription(analysis_df)


data_train.head(2)


def visualize_credit_default_vs_subscription(df):
    # Ensure you're working with correct data subset
    df_plot = df[df["y"].notna()]

    # Set figure and axes for side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.3)

    # LEFT PLOT â€“ Countplot with hue (absolute comparison)
    sns.countplot(
        data=df_plot,
        x="default",
        hue="y",
        order=df_plot["default"].value_counts().index,
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title("Deposit Subscription by Default Credit Status")
    axes[0].set_xlabel("Has Previously Credit Status")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Subscribed", labels=["No", "Yes"])
    axes[0].tick_params(axis='x', rotation=30)

    # RIGHT PLOT â€“ Percentage breakdown (relative comparison)
    total = df_plot.groupby("default").size()
    count = df_plot.groupby(["default", "y"]).size()
    default_group = (count / total).rename("proportion").reset_index()

    sns.barplot(
        data=default_group,
        x="default",
        y="proportion",
        hue="y",
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title("Proportion of Subscriptions by Default Credit Status (Percentage)")
    axes[1].set_xlabel("Has Previously Credit Status")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(handles=handles, title="Subscribed")
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()

visualize_credit_default_vs_subscription(analysis_df)


def visualize_balance_vs_subscription(df):    
    # Setup 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # KDE Histogram by 'y'
    sns.histplot(
        data=df[(df['balance'] <= 6000) & (df['balance'] >= -2000)],
        x='balance',
        hue='y',
        kde=True,
        stat='density',
        common_norm=False,
        bins=30,
        element='step',
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title('balance Distribution by Subscription')
    axes[0].set_xlabel('balance')
    axes[0].set_ylabel('Density')
    axes[0].legend(handles=handles, title='Subscribed', labels=['No', 'Yes'], loc='upper right')

    # Violin plot
    sns.violinplot(
        data=df[(df['balance'] <= 7000) & (df['balance'] >= -2000)],
        x='y',
        y='balance',
        inner='quartile',
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title('balance vs Subscription (Violin Plot)')
    axes[1].set_xlabel('Subscribed')
    axes[1].set_ylabel('balance')

    # Final layout
    plt.tight_layout()
    plt.show()

visualize_balance_vs_subscription(analysis_df)


def visualize_housing_loan_vs_subscription(df):
    # Ensure you're working with correct data subset
    df_plot = df[df["y"].notna()]

    # Set figure and axes for side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.3)

    # LEFT PLOT â€“ Countplot with hue (absolute comparison)
    sns.countplot(
        data=df_plot,
        x="housing",
        hue="y",
        order=df_plot["housing"].value_counts().index,
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title("Deposit Subscription by Housing Status")
    axes[0].set_xlabel("Housing Status")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Subscribed", labels=["No", "Yes"])
    axes[0].tick_params(axis='x', rotation=30)

    # RIGHT PLOT â€“ Percentage breakdown (relative comparison)
    total = df_plot.groupby("housing").size()
    count = df_plot.groupby(["housing", "y"]).size()
    housing_group = (count / total).rename("proportion").reset_index()
    
    sns.barplot(
        data=housing_group,
        x="housing",
        y="proportion",
        hue="y",
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title("Proportion of Subscriptions by Housing Status (Percentage)")
    axes[1].set_xlabel("Housing Status")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(handles=handles, title="Subscribed")
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()

visualize_housing_loan_vs_subscription(analysis_df)


def visualize_contact_preference(df):
    # Setup
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute count by contact method and subscription
    sns.countplot(
        data=df,
        x='contact',
        hue='y',
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title('Contact Method by Subscription Status')
    axes[0].set_xlabel('Contact Method')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Subscribed', labels=['No', 'Yes'])
    axes[0].tick_params(axis='x', rotation=20)

    # Percentage of subscribers within each contact method
    total = df.groupby("contact").size()
    count = df.groupby(["contact", "y"]).size()
    contact_group = (count / total).rename("proportion").reset_index()

    sns.barplot(
        data=contact_group,
        x='contact',
        y='proportion',
        hue='y',
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title('Proportion of Subscriptions by Contact Method')
    axes[1].set_xlabel('Contact Method')
    axes[1].set_ylabel('Proportion')
    axes[1].legend(handles=handles, title='Subscribed', labels=['No', 'Yes'])
    axes[1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.show()

visualize_contact_preference(analysis_df)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_recent_contact_conversion(df):
    # Copy and classify contacts
    df = df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
    # Absolute count by contact method and subscription
    sns.countplot(
        data=df,
        x='day',
        hue='y',
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title('Recency of Last Contact without campaign by Subscription Status')
    axes[0].set_xlabel('Recency Contact Without Campaign')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Subscribed', labels=['No', 'Yes'])
    axes[0].tick_params(axis='x', rotation=20)
    
    df['recent_contact'] = df['day'].apply(lambda x: '<7 days' if x < 7 else '>=7 days or never')

    # Group and normalize to get proportions
    total = df.groupby("recent_contact").size()
    count = df.groupby(["recent_contact", "y"]).size()
    recent_contact_group = (count / total).rename("proportion").reset_index()
    
    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=recent_contact_group,
        x='recent_contact',
        y='proportion',
        hue='y',
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title('Conversion Rate Based on Recency of Last Contact Without campaign (<7 Days)')
    axes[1].set_xlabel('Recency of Last Contact Without campaign')
    axes[1].set_ylabel('Proportion of Conversion')
    axes[1].legend(handles=handles, title='Subscribed', labels=['No', 'Yes'])
    
    plt.tight_layout()
    plt.show()

visualize_recent_contact_conversion(analysis_df)


def visualize_seconds_vs_subscription(df): 
    df = df[df['duration'] <= 2000]
       
    # Setup 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # KDE Histogram by 'y'
    sns.histplot(
        data=df,
        x='duration',
        hue='y',
        kde=True,
        stat='density',
        common_norm=False,
        bins=30,
        element='step',
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title('Duration Distribution by Subscription')
    axes[0].set_xlabel('Duration')
    axes[0].set_ylabel('Density')
    axes[0].legend(handles=handles, title='Subscribed', labels=['No', 'Yes'], loc='upper right')

    # Violin plot
    sns.violinplot(
        data=df,
        x='y',
        y='duration',
        inner='quartile',
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title('Duration vs Subscription (Violin Plot)')
    axes[1].set_xlabel('Subscribed')
    axes[1].set_ylabel('Duration')

    # Final layout
    plt.tight_layout()
    plt.show()

visualize_seconds_vs_subscription(analysis_df)


def visualize_spam_call_effect(df):
    df = df[df['campaign'] <= 10] 
    # Setup 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Proportion visualization
    total = df.groupby("campaign").size()
    count = df.groupby(["campaign", "y"]).size()
    campaign_contact_group = (count / total).rename("proportion").reset_index()

    sns.barplot(
        data=campaign_contact_group,
        x='campaign',
        y='proportion',
        hue='y',
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title('Proportion of Subscriptions by Campaign Call Frequency')
    axes[0].set_xlabel('Number of Calls During Campaign')
    axes[0].set_ylabel('Proportion')
    axes[0].legend(handles=handles, title='Subscribed', labels=['No', 'Yes'])
    axes[0].tick_params(axis='x', rotation=20)
    
    df['spammy_contact'] = df['campaign'].apply(lambda x: 'Mote than 3 calls' if x > 3 else 'Lower than 3 calls')
    
    total = df.groupby("spammy_contact").size()
    count = df.groupby(["spammy_contact", "y"]).size()
    campaign_contact_group = (count / total).rename("proportion").reset_index()

    # Plot
    sns.barplot(
        data=campaign_contact_group,
        x='spammy_contact',
        y='proportion',
        hue='y',
        palette=palette,
        axes=axes[1]
    )
    axes[1].set_title('Effect of Campaign Call Frequency on Subscription Rate')
    axes[1].set_xlabel('Number of Calls During Campaign')
    axes[1].set_ylabel('Proportion of Subscription Outcome')
    axes[1].legend(handles=handles, title='Subscribed', labels=['No', 'Yes'])

    # Final layout
    plt.tight_layout()
    plt.show()

visualize_spam_call_effect(analysis_df)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_recent_contact_conversion(df):
    df = df.copy()

    df_2 = df[df['pdays'] <= 30] 
    # Setup 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Proportion visualization
    total = df_2.groupby("pdays").size()
    count = df_2.groupby(["pdays", "y"]).size()
    pdays_group = (count / total).rename("proportion").reset_index()
    
    sns.barplot(
        data=pdays_group,
        x='pdays',
        y='proportion',
        hue='y',
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_title('Proportion of Subscriptions by Last Contact Campaign')
    axes[0].set_xlabel('Last Contact Recency Campaign')
    axes[0].set_ylabel('Proportion')
    axes[0].legend(handles=handles, title='Subscribed', labels=['No', 'Yes'])
    axes[0].tick_params(axis='x', rotation=20)

    # Categorize recency
    df_2['recent_contact'] = df['pdays'].apply(
        lambda x: '<14 days' if x < 14 else 'â‰¥14 days or never'
    )

    # # Group & normalize by contact recency
    # grouped = (
    #     df.groupby(['recent_contact', 'y'])
    #     .size()
    #     .groupby(level=0)
    
    #     .apply(lambda x: x / x.sum())
    #     .reset_index(name='proportion')
    # )

    total = df_2.groupby("recent_contact").size()
    count = df_2.groupby(["recent_contact", "y"]).size()
    pdays_group = (count / total).rename("proportion").reset_index()
    
    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=pdays_group,
        x='recent_contact',
        y='proportion',
        hue='y',
        palette=palette,
        ax=axes[1]
    )
    axes[1].set_title('Conversion Rate Based on Recency of Last Contact Campaign')
    axes[1].set_xlabel('Last Contact Recency Campaign')
    axes[1].set_ylabel('Proportion of Subscription Outcome')
    axes[1].legend(handles=handles, title='Subscribed', labels=['No', 'Yes'])
    
    plt.tight_layout()
    plt.show()


visualize_recent_contact_conversion(analysis_df)


def plot_success_vs_failure_by_campaign(df):
    df = df.copy()

    # Filter: meaningful campaign contacts
    df_campaign = df[(df['campaign'] - df['previous']) >= 1]

    # Keep only success/failure outcomes
    df_campaign = df_campaign[df_campaign['poutcome'].isin(['success', 'failure'])]

    # Only successful subscriptions
    df_success = df_campaign[df_campaign['y'] == 'yes']

    # Count number of successes for each outcome type
    success_counts = df_success['poutcome'].value_counts().reindex(['success', 'failure'], fill_value=0)

    # Color palette mapping
    palette = {"success": "#bea6a1", "failure": "#d62728"}

    # Generate color list based on index order
    colors = [palette[outcome] for outcome in success_counts.index]

    # Plot pie chart
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(
        success_counts,
        labels=success_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        colors=colors,
        textprops={'fontsize': 12, 'color': 'black'}
    )

    # Improve label aesthetics
    # for text in texts:
    #     text.set_fontweight('bold')
    #     text.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.3'))

    for autotext in autotexts:
        # autotext.set_fontweight('bold')
        autotext.set_color('white')
        autotext.set_bbox(dict(facecolor='black', edgecolor='none', alpha=0.5, boxstyle='round,pad=0.3'))

    plt.title('Conversions Status by Campaign Outcome', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


plot_success_vs_failure_by_campaign(analysis_df)

