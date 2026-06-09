from IPython.display import display, HTML

display(HTML("""
<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/ABUALHUSSEIN/Kaggle-Loan-Payback-Prediction/refs/heads/main/Predicting-Loan-Payback.png" width="1000">
</div>
"""))


# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configure settings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', None)


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


# Quick look
train.head()



print("Shape of dataset:", train.shape)


print("\n--- Data Info ---")
train.info()



# --- Define Target and Identifier Columns from your dataset ---
TARGET = 'loan_paid_back'
IDENTIFIER = 'id'

# --- Separate columns into different types ---
# Get a list of all columns from the training data
all_cols = train.columns.tolist()

# Identify categorical features (dtype='object')
categorical_features = train.select_dtypes(include=['object']).columns.tolist()

# Identify numerical features (any number type)
# We exclude the identifier and target columns from this list
numerical_features = train.select_dtypes(include=np.number).columns.tolist()
if IDENTIFIER in numerical_features:
    numerical_features.remove(IDENTIFIER)
if TARGET in numerical_features:
    numerical_features.remove(TARGET)

print(f"--- Feature Type Summary ---")
print(f"Identifier Column: {IDENTIFIER}")
print(f"Target Column: {TARGET}")
print(f"Total Features (excluding id and target): {len(numerical_features) + len(categorical_features)}")
print(f"Numerical Features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical Features ({len(categorical_features)}): {categorical_features}\n")


# --- Create the Data Dictionary DataFrame ---
def create_data_dictionary(df):
    """Creates a metadata summary of the dataframe."""
    metadata = []
    for col in df.columns:
        # Determine the role of the column
        if col == IDENTIFIER:
            role = 'Identifier'
        elif col == TARGET:
            role = 'Target'
        else:
            role = 'Feature'
            
        # Determine the type
        if col in categorical_features:
            col_type = 'Categorical'
        elif col in numerical_features:
            col_type = 'Numerical'
        else:
            # This handles 'id' and 'loan_paid_back'
            col_type = 'Identifier/Target'
            
        # Get metadata
        missing_count = df[col].isnull().sum()
        missing_percent = round((missing_count / len(df)) * 100, 2)
        unique_count = df[col].nunique()
        
        metadata.append({
            'Feature Name': col,
            'Role': role,
            'Type': col_type,
            'Missing Values': missing_count,
            'Missing (%)': missing_percent,
            'Unique Values': unique_count
        })
        
    meta_df = pd.DataFrame(metadata)
    return meta_df.set_index('Feature Name')

# Generate and display the data dictionary for the training set
data_dictionary = create_data_dictionary(train)

print("--- Data Dictionary (Train Set) ---")
display(data_dictionary)


#Duplicate rows
print("\n--- Check for Duplicates ---")
print(f"Number of duplicate rows: {train.duplicated().sum()}")


print("\n--- Summary Statistics (Numerical Features) ---")
display(train.describe().T)
    


# --- 1. Analyze Target Variable Distribution ---
target_counts = train['loan_paid_back'].value_counts()
target_perc = train['loan_paid_back'].value_counts(normalize=True) * 100

print("--- Target Variable Distribution ---")
print(target_counts)
print(f"\nPercentage of loans paid back (1.0): {target_perc[1.0]:.2f}%")
print(f"Percentage of loans not paid back (0.0): {target_perc[0.0]:.2f}%")



import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Prepare the data ---
target_counts = train[TARGET].value_counts()
labels = target_counts.index.map({1.0: 'Paid Back', 0.0: 'Defaulted'})
values = target_counts.values
colors = ['#2ecc71', '#e74c3c'] # Define colors to use them consistently

# --- 2. Create the subplot figure ---
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Absolute Counts', 'Class Proportions'),
                    specs=[[{'type': 'bar'}, {'type': 'pie'}]])

# --- 3. Add the Bar Chart ---
fig.add_trace(go.Bar(
    x=labels,
    y=values,
    text=[f'{v:,}' for v in values],
    textposition='auto',
    marker_color=colors,
    showlegend=False  # <-- IMPORTANT: This bar chart will NOT create legend items
), row=1, col=1)

# --- 4. Add the Donut Chart ---
fig.add_trace(go.Pie(
    labels=labels,
    values=values,
    hole=0.4,
    marker_colors=colors,
    textinfo='percent', # <-- Changed to avoid repeating labels
    name='Loan Status' # <-- Gives the legend a group name
), row=1, col=2)

# --- 5. Customize the layout with a VERTICAL legend ---
fig.update_layout(
    title_text='Distribution of Loan Payback Status',
    title_x=0.5,
    height=500,
    yaxis_title='Number of Loans',
    showlegend=True, # <-- Turn the legend ON
    legend_title_text='Loan Status',
    legend=dict(
        orientation="v", # 'v' for vertical
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.02
    )
)

fig.show()

# --- Optional: Print the exact values for reference ---
print("--- Target Variable Distribution ---")
class_0_count = target_counts.get(0, 0)
class_1_count = target_counts.get(1, 0)
total = class_0_count + class_1_count
print(f"Number of loans that defaulted (Class 0): {class_0_count} ({class_0_count/total:.2%})")
print(f"Number of loans that were paid back (Class 1): {class_1_count} ({class_1_count/total:.2%})")



# --- Feature Distribution Analysis ---

# Select numeric features (excluding ID if present)
num_features = train.select_dtypes(include=[np.number]).columns.tolist()
# Columns to exclude
exclude_cols = ["id","loan_paid_back"]
num_features = [col for col in num_features if col not in exclude_cols]

# Define grid size automatically (rows & cols)

n_features = len(num_features)
n_cols = 2
n_rows = int(np.ceil(n_features / n_cols))

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

# Plot each feature
for i, col in enumerate(num_features):
    sns.histplot(train[col], bins=30, kde=True, ax=axes[i], color="skyblue")
    axes[i].set_title(f"Distribution of {col}", fontsize=12, weight="bold")
    axes[i].set_xlabel("")
    axes[i].grid(axis="y", linestyle="--", alpha=0.6)

# Remove empty subplots (if any)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Feature Distributions", fontsize=16, weight="bold", y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


print("--- Checking Levels of Categorical Features ---")
for col in categorical_features:
    num_levels = train[col].nunique()  # Get the number of unique levels
    levels = train[col].unique()      # Get the actual unique levels
    
    print(f"\nFeature: '{col}'")
    print(f"  Number of unique levels: {num_levels}")
    print(f"  Levels: {levels}")
    print("-" * 30)


# --- Visualize Distributions of Categorical Features ---
print("\n--- Distributions of Categorical Features ---")
fig, axes = plt.subplots(nrows=len(categorical_features), ncols=1, figsize=(12, 30))

for i, col in enumerate(categorical_features):
    ax = axes[i]
    sns.countplot(y=col, data=train, order=train[col].value_counts().index, ax=ax, palette='viridis')
    ax.set_title(f'Distribution of {col}', fontsize=14)
    ax.set_xlabel('Count')
    ax.set_ylabel('')
    # Add percentage annotations
    total = len(train[col])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.tight_layout()
plt.show()


# --- 1. Import Plotly for Interactive Visualizations ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the target variable name for convenience
TARGET = 'loan_paid_back'

# --- 2. Create a Reusable Plotting Function ---
def plot_categorical_bivariate(train, col_name, target_name):
    """
    Generates an interactive bar plot showing both the distribution 
    and the target mean (repayment rate) for a categorical feature.
    """
    # Calculate counts and repayment rates
    counts = train[col_name].value_counts()
    repayment_rate = train.groupby(col_name)[target_name].mean() * 100 # as percentage
    
    # Combine into a single DataFrame for plotting
    plot_df = pd.DataFrame({
        'Count': counts,
        'Repayment Rate (%)': repayment_rate
    })
    
    # Sort by repayment rate for a more insightful plot
    plot_df = plot_df.sort_values('Repayment Rate (%)', ascending=True)

    # Create the plot
    fig = go.Figure(go.Bar(
        y=plot_df.index,
        x=plot_df['Count'],
        orientation='h',
        marker=dict(
            color=plot_df['Repayment Rate (%)'],
            colorscale='Viridis', # A colorblind-friendly scale
            showscale=True,
            colorbar=dict(title='Repayment Rate %')
        ),
        text=[f'{count:,} loans' for count in plot_df['Count']],
        textposition='outside'
    ))

    fig.update_layout(
        title=f'<b>Distribution and Repayment Rate for {col_name}</b>',
        height=500 if len(plot_df) < 15 else 800, # Adjust height for grade_subgrade
        plot_bgcolor='#f6f5f5',
        paper_bgcolor='#f6f5f5',
        xaxis_title='Number of Loans',
        yaxis_title=col_name,
        yaxis=dict(tickfont=dict(size=12)),
        title_font_size=18,
        showlegend=False
    )
    
    fig.show()

# --- 3. Generate a plot for each categorical feature ---
for col in categorical_features:
    plot_categorical_bivariate(train, col, TARGET)


# --- 1. Set Pandas display options for better readability ---
pd.set_option('display.max_rows', 100) # Show more rows for grade_subgrade

# --- 2. Loop through categorical features and perform groupby analysis ---
print("--- Groupby Aggregations ---")

for cat_col in categorical_features:
    # Group by the categorical column and calculate aggregations for numerical columns
    grouped_analysis = train.groupby(cat_col)[numerical_features].agg(['mean', 'median', 'count'])
    
    # Style the output for better interpretation
    # Using a background gradient to highlight high/low values
    styled_df = grouped_analysis.style.background_gradient(cmap='viridis').set_caption(
        f"<b>Aggregations for Numerical Features by {cat_col}</b>"
    )
    
    print(f"\n\nAnalysis by: {cat_col}")
    display(styled_df)



# --- 1. Define a reusable function for interaction heatmaps ---
def plot_interaction_heatmap(df, feature1, feature2, target, colorscale='Viridis'):
    """
    Generates an interactive heatmap showing the default rate for the
    interaction between two categorical features.
    """
    # Calculate the default rate (1 - repayment_rate) * 100
    risk_matrix = pd.crosstab(
        index=df[feature1],
        columns=df[feature2],
        values=(1 - df[target]),  # This is the default rate
        aggfunc='mean'
    )
    # Convert to percentage
    risk_matrix *= 100
    
    # Generate the heatmap
    fig = go.Figure(go.Heatmap(
        z=risk_matrix.values,
        x=risk_matrix.columns,
        y=risk_matrix.index,
        colorscale=colorscale,
        text=risk_matrix.values.round(2),
        texttemplate='%{text}%',
        textfont=dict(size=12, color='white')
    ))

    fig.update_layout(
        title=f'<b>Default Rate (%) by {feature1} & {feature2}</b>',
        height=500,
        width=800,
        plot_bgcolor='#f6f5f5',
        paper_bgcolor='#f6f5f5',
        xaxis_title=feature2,
        yaxis_title=feature1,
        title_font_size=18
    )
    
    fig.show()

# --- 2. Generate heatmaps for interesting feature pairs ---

# Pair 1: Employment Status vs. Marital Status (as you suggested)
plot_interaction_heatmap(train, 'employment_status', 'marital_status', TARGET)

# Pair 2: Education Level vs. Employment Status
plot_interaction_heatmap(train, 'education_level', 'employment_status', TARGET)

# Pair 3: Loan Purpose vs. Education Level
plot_interaction_heatmap(train, 'loan_purpose', 'education_level', TARGET, colorscale='Plasma')


# --- 1. Define the Numerical Columns ---
# It's good practice to define this list explicitly.
# We will use the original numerical features for this analysis.
numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                    'loan_amount', 'interest_rate']
TARGET = 'loan_paid_back' # Ensure target variable is defined

# --- 2. Calculate the Correlation Matrix ---
# We select only the numerical columns and the target from our training dataframe.
corr_matrix = train[numerical_cols + [TARGET]].corr()

# --- 3. Create the Interactive Heatmap ---
import plotly.graph_objects as go

# Define a color palette for the plot
colors = ['#2E4053', '#85C1E9', '#FAD7A0'] # Dark Blue, Light Blue, Light Orange

fig = go.Figure(go.Heatmap(
    z=corr_matrix.values, 
    x=corr_matrix.columns, 
    y=corr_matrix.columns,
    colorscale=[[0, colors[0]], [0.5, colors[2]], [1, colors[1]]], # Custom colorscale
    zmin=-1, zmax=1, # Explicitly set the range of the color scale
    text=corr_matrix.values.round(2), 
    texttemplate='%{text}', 
    textfont=dict(size=12, color='white')
))

fig.update_layout(
    title='<b>Feature Correlation Matrix</b>', 
    height=600, 
    width=700,
    plot_bgcolor='#f6f5f5', 
    paper_bgcolor='#f6f5f5'
)

fig.show()

