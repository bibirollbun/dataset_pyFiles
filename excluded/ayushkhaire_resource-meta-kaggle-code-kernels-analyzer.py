import pandas as pd
import numpy as np
from IPython.display import display , HTML , Markdown
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import gc

import warnings
warnings.filterwarnings('ignore')
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")


# load the originals 
kernels_df = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')
about_meta_kaggle_code_df = pd.read_csv('/kaggle/input/meta-kaggle-codemetadata-csv/meta-kaggle-code-notebooks-metadata.csv')
users_df = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')


def get_version_id_from_kaggle_ipynb_path(ipynb_path):
    if pd.notna(ipynb_path):
        return float(ipynb_path.strip('/').split('/')[-1])
    return np.nan


# print(get_version_id_from_kaggle_ipynb_path('/0164/580/164580511'))  # 164580511
# print(get_version_id_from_kaggle_ipynb_path('/0000/277/000277152'))  # 277152


# now prepare the kernels df
about_meta_kaggle_code_df['version_id'] = about_meta_kaggle_code_df['kaggle_code_location'].apply(get_version_id_from_kaggle_ipynb_path)

# now try to do 
# kernel_path_in_meta_kaggle_code in kernel df match to kaggle_code_location in about_meta_kaggle_code_df

merged_kernels_df = pd.merge(
    kernels_df,
    about_meta_kaggle_code_df,
    left_on='CurrentKernelVersionId',
    right_on='version_id',
    how='inner'  # or 'left' if you want to keep all from kernels_df
)
merged_kernels_df = pd.merge(
    merged_kernels_df,
    users_df,
    left_on='AuthorUserId',
    right_on='Id',
    how='inner'  # or 'left' if you want to keep all from kernels_df
)

# look for my notebook
# https://www.kaggle.com/code/ayushkhaire/indian-forts-data-collection?scriptVersionId=164580511
# version id : 164580511
display(HTML(merged_kernels_df[merged_kernels_df['CurrentKernelVersionId'] == 164580511].to_html()))


def fetch_day_month_year(date):
    try:
        parts = str(date).split('/')
        if len(parts) != 3:
            return None, None, None
        return parts[1], parts[0], parts[2]  # Day, Month, Year
    except:
        return None, None, None

# Apply and expand into separate columns
merged_kernels_df[['MadePublicDateDay', 'MadePublicDateMonth', 'MadePublicDateYear']] = \
    merged_kernels_df['MadePublicDate'].apply(fetch_day_month_year).apply(pd.Series)


columns_and_desc_match_sum = {
    'Total notebook views on Kaggle': 'TotalViews',
    'Total notebook comments on Kaggle': 'TotalComments',
    'Total notebook upvotes on Kaggle': 'TotalVotes',

    'Total notebook cells written by the community': 'total_cells',
    'Total code cells written by the community': 'total_code_cells',
    'Total markdown cells written by the community': 'total_markdown_cells',
    'Total collapsed code cells in notebooks': 'total_collapsed_code_cells_in_notebook',
    'Total hidden input cells in notebooks': 'total_hidden_input_code_cells_in_notebook',

    'Total lines written in all notebooks': 'total_lines_in_notebook',
    'Total words written in all notebooks': 'total_words_in_notebook',
    'Total letters written in all notebooks': 'total_letters_in_notebook',

    'Total lines of code written in code cells': 'total_lines_in_notebook_code',
    'Total words written in code cells': 'total_words_in_notebook_code',
    'Total letters written in code cells': 'total_letters_in_notebook_code',

    'Total lines written in markdown cells': 'total_lines_in_notebook_markdown',
    'Total words written in markdown cells': 'total_words_in_notebook_markdown',
    'Total letters written in markdown cells': 'total_letters_in_notebook_markdown',

    'total_collapsed_code_cells_in_notebook':'total_collapsed_code_cells_in_notebook',
    'total_hidden_input_code_cells_in_notebook':'total_hidden_input_code_cells_in_notebook',
    'total_executed_cells_in_notebook':'total_executed_cells_in_notebook',
    'total_output_cells_in_notebook':'total_output_cells_in_notebook', 
    
    'total_packages_in_notebook_code' : 'total_packages_in_notebook_code',
    'total_functions_used_in_notebook_code' :'total_functions_used_in_notebook_code',
    'total_attributes_in_notebook_code':'total_attributes_in_notebook_code',
    'total_methods_used_in_notebook_code':'total_methods_used_in_notebook_code',
}

columns_and_desc_match_count = {
    'Total kernels available by kagglers': 'CurrentKernelVersionId',
    'Total medals awarded on Kaggle': 'Medal',
}


merged_kernels_month_year_df = merged_kernels_df.groupby(
    ['MadePublicDateMonth', 'MadePublicDateYear']
)[list(columns_and_desc_match_sum.values())].sum().reset_index()
merged_kernels_month_year_df = merged_kernels_month_year_df.sort_values(by=['MadePublicDateYear', 'MadePublicDateMonth'])
merged_kernels_month_year_df['time'] = merged_kernels_month_year_df['MadePublicDateMonth'] + '-' + merged_kernels_month_year_df['MadePublicDateYear']
merged_kernels_month_year_df.head(3)


for key , value in columns_and_desc_match_sum.items():
    total = merged_kernels_df[value].sum()
    print(f'{key} : {total}')
for key , value in columns_and_desc_match_count.items():
    total = merged_kernels_df[value].sum()
    print(f'{key} : {total}')


def seaborn_multiline_plot(df, x, y_columns, title="Seaborn Line Plot (Dark Theme)", width=14, height=7):
    """
    Plots multiple line plots from a DataFrame using Seaborn with a fully dark theme.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x (str): The column name for x-axis.
        y_columns (list of str): The column names to plot on y-axis.
        title (str): Plot title.
        width (int): Width of the plot (default: 14).
        height (int): Height of the plot (default: 7).
    """
    # Set dark theme manually
    plt.style.use('dark_background')  # Dark background for matplotlib
    sns.set_theme(style="dark")       # Seaborn dark style (affects gridlines)

    plt.figure(figsize=(width, height))
    
    ax = plt.gca()
    ax.set_facecolor("#0f0f0f")  # Very dark axes background
    
    for y in tqdm(y_columns, desc='Plotting the plot'):
        sns.lineplot(data=df, x=x, y=y, label=y, marker="o")

    plt.title(title, fontsize=16, color='white')
    plt.xlabel(x, fontsize=12, color='white')
    plt.ylabel("Values", fontsize=12, color='white')
    
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    
    legend = plt.legend(title="Metrics", facecolor='#1a1a1a', edgecolor='white', labelcolor='white', title_fontsize=12)
    legend.get_title().set_color('white')
    
    plt.grid(True, color='gray', alpha=0.2)
    plt.tight_layout()
    
    plt.gcf().patch.set_facecolor('#000000')  # Full figure background

    plt.show()

def seaborn_horizontal_bar_plot(series, title="Horizontal Bar Plot (Dark Theme)", width=10, height=15):
    """
    Plots a horizontal bar plot from a pandas Series using Seaborn with a fully dark theme.

    Parameters:
        series (pd.Series): The input Series with index as categories and values as counts.
        title (str): Plot title.
        width (int): Width of the plot (default: 14).
        height (int): Height of the plot (default: 7).
    """
    # Set dark theme manually
    plt.style.use('dark_background')  # Dark background for matplotlib
    sns.set_theme(style="dark")       # Seaborn dark style (affects gridlines)

    plt.figure(figsize=(width, height))
    
    ax = plt.gca()
    ax.set_facecolor("#0f0f0f")  # Very dark axes background

    sns.barplot(x=series.values, y=series.index, orient='h')

    plt.title(title, fontsize=16, color='white')
    plt.xlabel("Usage Count", fontsize=12, color='white')
    plt.ylabel("Component", fontsize=12, color='white')

    plt.xticks(color='white')
    plt.yticks(color='white')

    plt.grid(True, color='gray', alpha=0.2)
    plt.tight_layout()
    
    plt.gcf().patch.set_facecolor('#000000')  # Full figure background

    plt.show()

seaborn_multiline_plot(df = merged_kernels_month_year_df.iloc[::3] , x = 'time', y_columns = [
    'TotalComments', 
    'TotalVotes',
],title = "General progress of kaggle over time",height = 10,width = 35)

seaborn_multiline_plot(df = merged_kernels_month_year_df.iloc[::3] , x = 'time', y_columns = [
    'total_cells', 
    'total_code_cells',
    'total_markdown_cells',
], title = " </> code cells created by kagglers over time </>",height = 13,width = 35)

seaborn_multiline_plot(df = merged_kernels_month_year_df.iloc[::3] , x = 'time', y_columns = [
    'total_lines_in_notebook',
    'total_words_in_notebook',
    'total_letters_in_notebook',
    'total_lines_in_notebook_code', 
    'total_words_in_notebook_code',
    'total_letters_in_notebook_code', 
    'total_packages_in_notebook_code',
    'total_functions_used_in_notebook_code',
    'total_attributes_in_notebook_code',
    'total_methods_used_in_notebook_code',
    'total_lines_in_notebook_markdown',
    'total_words_in_notebook_markdown',
    'total_letters_in_notebook_markdown'
],title = " </> code written by kagglers over time </>",height = 25,width = 35)


user_code_lines_df = (
    merged_kernels_df
    .groupby('UserName')['total_lines_in_notebook']
    .sum()
    .reset_index(name='total_lines_in_notebook')
    .sort_values(by='total_lines_in_notebook', ascending=False)
)

top_users_df = user_code_lines_df.head(50)

# Reuse your horizontal bar plot function
# Select top 50 users
top_users_series = (
    user_code_lines_df
    .set_index('UserName')['total_lines_in_notebook']
    .head(50)
)

# Call your plot function
seaborn_horizontal_bar_plot(
    top_users_series,
    title="Top Kagglers by Total Lines of Code Written",
    width=12,
    height=18
)


country_code_lines_df = (
    merged_kernels_df
    .groupby('Country')['total_lines_in_notebook']
    .sum()
    .reset_index(name='total_lines_in_notebook')
    .sort_values(by='total_lines_in_notebook', ascending=False)
)

top_countries_df = country_code_lines_df.head(50)

# Reuse your horizontal bar plot function
# Select top 50 users
top_countries_series = (
    top_countries_df
    .set_index('Country')['total_lines_in_notebook']
    .head(50)
)

# Call your plot function
seaborn_horizontal_bar_plot(
    top_countries_series,
    title="Top Countries by Total Lines of Code Written",
    width=12,
    height=18
)


code_in_versions = {
    'version':[],
    'type':[],
    'code_component':[]
}

for vrs , pkg, fnc, mtd, attr in tqdm(zip(
    merged_kernels_df['version_id'],
    merged_kernels_df['listed_string_of_packages_in_notebook_code'],
    merged_kernels_df['listed_string_of_functions_used_in_notebook_code'],
    merged_kernels_df['listed_string_of_methods_used_in_notebook_code'],
    merged_kernels_df['listed_string_of_attributes_in_notebook_code']
),
total=len(merged_kernels_df)
):
    if pd.notna(pkg):
        for p in pkg.split(','):
            p = p.strip()
            code_in_versions['version'].append(vrs)
            code_in_versions['type'].append('package')
            code_in_versions['code_component'].append(p)

    if pd.notna(fnc):
        for f in fnc.split(','):
            f = f.strip()
            code_in_versions['version'].append(vrs)
            code_in_versions['type'].append('function')
            code_in_versions['code_component'].append(f)

    if pd.notna(mtd):
        for m in mtd.split(','):
            m = m.strip()
            code_in_versions['version'].append(vrs)
            code_in_versions['type'].append('method')
            code_in_versions['code_component'].append(m)

    if pd.notna(attr):
        for a in attr.split(','):
            a = a.strip()
            code_in_versions['version'].append(vrs)
            code_in_versions['type'].append('attribute')
            code_in_versions['code_component'].append(a)

code_in_versions_df = pd.DataFrame(code_in_versions)

# confirm with my notebook
code_in_versions_df[code_in_versions_df['version'] == 164580511 ].sample(5)


# list all dfs now
%whos DataFrame

# clean all code now
del about_meta_kaggle_code_df
del kernels_df
del merged_kernels_month_year_df
del users_df

gc.collect()


# consider these columns from kernels df 
new_kernels_df = merged_kernels_df[[ 'TotalViews','MadePublicDate','CurrentUrlSlug','CurrentKernelVersionId','UserName', 'PerformanceTier',
       'Country']]
new_kernels_code_df = pd.merge(
    new_kernels_df,
    code_in_versions_df,
    left_on='CurrentKernelVersionId',
    right_on='version',
    how='inner' 
)

# confirm with my notebook
new_kernels_code_df[new_kernels_code_df['version'] == 164580511 ].sample(3)


%whos DataFrame


del code_in_versions_df
del country_code_lines_df
del merged_kernels_df
del new_kernels_df
del top_countries_df
del top_users_df
del user_code_lines_df
gc.collect()
%whos DataFrame


new_kernels_code_df.columns


# get the most used packages , functions , methods and attributes
total_packages_used_by_kagglers = new_kernels_code_df[new_kernels_code_df['type'] == 'package' ]['code_component'].nunique()
total_functions_used_by_kagglers = new_kernels_code_df[new_kernels_code_df['type'] == 'function' ]['code_component'].nunique()
total_methods_used_by_kagglers = new_kernels_code_df[new_kernels_code_df['type'] == 'method' ]['code_component'].nunique()
total_attributes_used_by_kagglers = new_kernels_code_df[new_kernels_code_df['type'] == 'attribute' ]['code_component'].nunique()
print("Total packages used by kagglers : ",total_packages_used_by_kagglers)
print("Total functions used by kagglers : ",total_functions_used_by_kagglers)
print("Total methods used by kagglers : ",total_methods_used_by_kagglers)
print("Total attributes used by kagglers : ",total_attributes_used_by_kagglers)


# Packages
packages_used = new_kernels_code_df[new_kernels_code_df['type'] == 'package']['code_component'].value_counts().sort_values(ascending=False)

# Functions
functions_used = new_kernels_code_df[new_kernels_code_df['type'] == 'function']['code_component'].value_counts().sort_values(ascending=False)

# Methods
methods_used = new_kernels_code_df[new_kernels_code_df['type'] == 'method']['code_component'].value_counts().sort_values(ascending=False)

# Attributes
attributes_used = new_kernels_code_df[new_kernels_code_df['type'] == 'attribute']['code_component'].value_counts().sort_values(ascending=False)

# print("Top 10 Packages Used:\n")
# display(packages_used.head(10))
# print("\nTop 10 Functions Used:\n")
# display(functions_used.head(10))
# print("\nTop 10 Methods Used:\n")
# display(methods_used.head(10))
# print("\nTop 10 Attributes Used:\n")
# display(attributes_used.head(10))


# Top 20 packages
seaborn_horizontal_bar_plot(packages_used.head(50), title="Top 50 Packages Used by Kagglers")

# Top 20 functions
seaborn_horizontal_bar_plot(functions_used.head(50), title="Top 50 Functions Used by Kagglers")

# Top 20 methods
seaborn_horizontal_bar_plot(methods_used.head(50), title="Top 50 Methods Used by Kagglers")

# Top 20 attributes
seaborn_horizontal_bar_plot(attributes_used.head(50), title="Top 50 Attributes Used by Kagglers")


first_uses_by_kagglers = new_kernels_code_df.sort_values('version').drop_duplicates(subset=['code_component', 'type'], keep='first')
# first_uses_by_kagglers


# Sort helper: Make mapping from component to rank
package_rank = {pkg: i for i, pkg in enumerate(packages_used.head(100).index)}
function_rank = {f: i for i, f in enumerate(functions_used.head(100).index)}
method_rank = {m: i for i, m in enumerate(methods_used.head(100).index)}
attribute_rank = {a: i for i, a in enumerate(attributes_used.head(100).index)}

# Filter and sort top 100 packages
top_100_packages_usage = first_uses_by_kagglers[
    (first_uses_by_kagglers['type'] == 'package') &
    (first_uses_by_kagglers['code_component'].isin(package_rank))
].copy()
top_100_packages_usage['rank'] = top_100_packages_usage['code_component'].map(package_rank)
top_100_packages_usage['link'] = 'https://kaggle.com/code/' + top_100_packages_usage['UserName'] + "/" + top_100_packages_usage['CurrentUrlSlug'] + '/'
top_100_packages_usage = top_100_packages_usage.sort_values('rank')

# Filter and sort top 100 functions
top_100_functions_usage = first_uses_by_kagglers[
    (first_uses_by_kagglers['type'] == 'function') &
    (first_uses_by_kagglers['code_component'].isin(function_rank))
].copy()
top_100_functions_usage['rank'] = top_100_functions_usage['code_component'].map(function_rank)
top_100_functions_usage['link'] = 'https://kaggle.com/code/' + top_100_functions_usage['UserName'] + "/" + top_100_functions_usage['CurrentUrlSlug'] + '/'
top_100_functions_usage = top_100_functions_usage.sort_values('rank')

# Filter and sort top 100 methods
top_100_methods_usage = first_uses_by_kagglers[
    (first_uses_by_kagglers['type'] == 'method') &
    (first_uses_by_kagglers['code_component'].isin(method_rank))
].copy()
top_100_methods_usage['rank'] = top_100_methods_usage['code_component'].map(method_rank)
top_100_methods_usage['link'] = 'https://kaggle.com/code/' + top_100_methods_usage['UserName'] + "/" + top_100_methods_usage['CurrentUrlSlug'] + '/'
top_100_methods_usage = top_100_methods_usage.sort_values('rank')

# Filter and sort top 100 attributes
top_100_attributes_usage = first_uses_by_kagglers[
    (first_uses_by_kagglers['type'] == 'attribute') &
    (first_uses_by_kagglers['code_component'].isin(attribute_rank))
].copy()
top_100_attributes_usage['rank'] = top_100_attributes_usage['code_component'].map(attribute_rank)
top_100_attributes_usage['link'] = 'https://kaggle.com/code/' + top_100_attributes_usage['UserName'] + "/" + top_100_attributes_usage['CurrentUrlSlug'] + '/'
top_100_attributes_usage = top_100_attributes_usage.sort_values('rank')

display(Markdown("## Kagglers who used top 100 packages for first time"))
display(HTML(top_100_packages_usage[['UserName','code_component','link']].to_html(index=False)))

display(Markdown("## Kagglers who used top 100 functions for first time"))
display(HTML(top_100_functions_usage[['UserName','code_component','link']].to_html(index=False)))

display(Markdown("## Kagglers who used top 100 methods for first time"))
display(HTML(top_100_methods_usage[['UserName','code_component','link']].to_html(index=False)))

display(Markdown("## Kagglers who used top 100 attributes for first time"))
display(HTML(top_100_attributes_usage[['UserName','code_component','link']].to_html(index=False)))


del top_100_packages_usage
del top_100_functions_usage
del top_100_methods_usage
del top_100_attributes_usage
gc.collect()
%whos DataFrame


def extract_month_year(date_str):
    if isinstance(date_str, str) and "/" in date_str:
        parts = date_str.split("/")
        return f"{parts[0]}-{parts[2]}"
    return None  # or "Unknown"

the_new_times_list = []
for d in tqdm(new_kernels_code_df['MadePublicDate']):
    the_new_times_list.append(extract_month_year(d))

new_kernels_code_df['publish-time'] = the_new_times_list   



def generate_overtime_usage_plot(df, usage_type, top_n=20, step=2):
    # Filter by type (function/method/attribute)
    # Filter by type (function/method/attribute)
    filtered_df = df[df['type'] == usage_type].copy()

    # Convert publish-time to datetime format
    filtered_df['publish-time'] = pd.to_datetime(filtered_df['publish-time'], format='%m-%Y')

    # Sort by datetime
    filtered_df = filtered_df.sort_values(by='publish-time')

    # Group by publish-time and code_component
    overtime_usage = (
        filtered_df
        .groupby(['publish-time', 'code_component'])
        .size()
        .reset_index(name='count')
    )

    # Top N components (based on total usage across all time)
    top_components = (
        overtime_usage.groupby('code_component')['count'].sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    # Filter to top components only
    top_overtime = overtime_usage[overtime_usage['code_component'].isin(top_components)]

    # Pivot to wide format
    usage_df = top_overtime.pivot_table(
        index='publish-time',
        columns='code_component',
        values='count',
        fill_value=0
    ).reset_index()

    usage_df.columns.name = None  # remove pivot name

    # Prepare list of columns for y
    y_columns = [col for col in usage_df.columns if col != 'publish-time']

    # Display markdown title
    display(Markdown(f"# ğŸ“ˆ Top {top_n} `{usage_type}`s used by Kagglers over time"))

    # Plot every `step` rows to avoid clutter
    seaborn_multiline_plot(
        df=usage_df.iloc[::step],
        x='publish-time',
        y_columns=y_columns,
        title=f"ğŸ“ˆ </> Top {top_n} `{usage_type}`s used over time",
        height=15,
        width=25
    )

# Generate plots for each type
generate_overtime_usage_plot(new_kernels_code_df, 'package')
generate_overtime_usage_plot(new_kernels_code_df, 'function')
generate_overtime_usage_plot(new_kernels_code_df, 'method')
generate_overtime_usage_plot(new_kernels_code_df, 'attribute')


%whos DataFrame


popular_ai_packages = [
    'transformers', 'diffusers', 'sentence_transformers', 'accelerate', 'peft',
    'bitsandbytes', 'auto_gptq', 'openai', 'llama_index', 'trl'
]

popular_ai_functions = [
    'from_pretrained', 'generate', 'tokenize', 'encode_plus', 'pipeline',
    'get_scheduler', 'get_cosine_schedule_with_warmup', 'image_grid',
    'prepare_model_for_kbit_training', 'prepare_prompt'
]

popular_ai_methods = [
    'generate', 'to', 'eval', 'half', 'compile', 'push_to_hub',
    'prepare_inputs_for_generation', 'quantize', 'backward', 'train'
]

popular_ai_classes = [
    'AutoModelForCausalLM', 'AutoTokenizer', 'DiffusionPipeline',
    'StableDiffusionPipeline', 'LlamaTokenizer', 'PeftModel',
    'Trainer', 'CLIPModel', 'AutoProcessor', 'AutoImageProcessor'
]

popular_ai_attributes = [
    'logits', 'attentions', 'hidden_states', 'config',
    'pixel_values', 'text_input_ids', 'generation_config',
    'model_args', 'quantization_config', 'input_features'
]

def generate_ai_component_usage_plot(df, ai_components, usage_type='package', step=2):
    """
    Plots usage of selected AI-related packages/functions/methods/attributes over time.
    
    Args:
        df (pd.DataFrame): DataFrame with code usage.
        ai_components (list): List of code components to visualize.
        usage_type (str): One of ['package', 'function', 'method', 'attribute'].
        step (int): Row step size for downsampling the plot.
    """
    # Filter by usage type and AI-related components
    filtered_df = df[
        (df['type'] == usage_type) &
        (df['code_component'].isin(ai_components))
    ]

    # Convert publish-time to datetime format
    filtered_df['publish-time'] = pd.to_datetime(filtered_df['publish-time'], format='%m-%Y')

    # Sort by datetime
    filtered_df = filtered_df.sort_values(by='publish-time')


    if filtered_df.empty:
        display(Markdown(f"### âš ï¸� No `{usage_type}` entries found for given AI components."))
        return

    # Group by publish-time and component
    overtime_usage = (
        filtered_df
        .groupby(['publish-time', 'code_component'])
        .size()
        .reset_index(name='count')
    )

    # Pivot to wide format
    usage_df = overtime_usage.pivot_table(
        index='publish-time',
        columns='code_component',
        values='count',
        fill_value=0
    ).reset_index()

    usage_df.columns.name = None  # remove pivot name

    # Prepare list of columns for y
    y_columns = [col for col in usage_df.columns if col != 'publish-time']

    # Display markdown title
    display(Markdown(f"# ğŸ¤– Usage of AI-related `{usage_type}`s over time"))

    # Plot every `step` rows to reduce clutter
    seaborn_multiline_plot(
        df=usage_df.iloc[::step],
        x='publish-time',
        y_columns=y_columns,
        title=f"ğŸ¤– </> Usage of AI-related `{usage_type}`s over time",
        height=15,
        width=35
    )


generate_ai_component_usage_plot(new_kernels_code_df, popular_ai_packages, 'package')
generate_ai_component_usage_plot(new_kernels_code_df, popular_ai_functions, 'function')
generate_ai_component_usage_plot(new_kernels_code_df, popular_ai_methods, 'method')
generate_ai_component_usage_plot(new_kernels_code_df, popular_ai_classes, 'function')
generate_ai_component_usage_plot(new_kernels_code_df, popular_ai_attributes, 'attribute')


org_ai_packages = [
    'transformers', 'datasets', 'accelerate', 'jax', 't5',
    'openai', 'kaggle', 'fairseq', 'dm-haiku', 'gpt-neox'
]

generate_ai_component_usage_plot(new_kernels_code_df, org_ai_packages, usage_type='package')


# Create rank maps for popular AI components
ai_package_rank = {pkg: i for i, pkg in enumerate(popular_ai_packages)}
ai_function_rank = {f: i for i, f in enumerate(popular_ai_functions)}
ai_method_rank = {m: i for i, m in enumerate(popular_ai_methods)}
ai_class_rank = {c: i for i, c in enumerate(popular_ai_classes)}
org_package_rank = {pkg: i for i, pkg in enumerate(org_ai_packages)}

# Packages
ai_packages_usage = first_uses_by_kagglers[
    (first_uses_by_kagglers['type'] == 'package') &
    (first_uses_by_kagglers['code_component'].isin(ai_package_rank))
].copy()
ai_packages_usage['rank'] = ai_packages_usage['code_component'].map(ai_package_rank)
ai_packages_usage['link'] = (
    'https://kaggle.com/code/' +
    ai_packages_usage['UserName'] + "/" +
    ai_packages_usage['CurrentUrlSlug'] + '/'
)
ai_packages_usage = ai_packages_usage.sort_values('rank')

ai_functions_usage = first_uses_by_kagglers[
    (first_uses_by_kagglers['type'] == 'function') &
    (first_uses_by_kagglers['code_component'].isin(ai_function_rank))
].copy()
ai_functions_usage['rank'] = ai_functions_usage['code_component'].map(ai_function_rank)
ai_functions_usage['link'] = (
    'https://kaggle.com/code/' +
    ai_functions_usage['UserName'] + "/" +
    ai_functions_usage['CurrentUrlSlug'] + '/'
)
ai_functions_usage = ai_functions_usage.sort_values('rank')

ai_methods_usage = first_uses_by_kagglers[
    (first_uses_by_kagglers['type'] == 'method') &
    (first_uses_by_kagglers['code_component'].isin(ai_method_rank))
].copy()
ai_methods_usage['rank'] = ai_methods_usage['code_component'].map(ai_method_rank)
ai_methods_usage['link'] = (
    'https://kaggle.com/code/' +
    ai_methods_usage['UserName'] + "/" +
    ai_methods_usage['CurrentUrlSlug'] + '/'
)
ai_methods_usage = ai_methods_usage.sort_values('rank')

ai_classes_usage = first_uses_by_kagglers[
    (first_uses_by_kagglers['type'] == 'function') &  # class instead of attribute
    (first_uses_by_kagglers['code_component'].isin(ai_class_rank))
].copy()
ai_classes_usage['rank'] = ai_classes_usage['code_component'].map(ai_class_rank)
ai_classes_usage['link'] = (
    'https://kaggle.com/code/' +
    ai_classes_usage['UserName'] + "/" +
    ai_classes_usage['CurrentUrlSlug'] + '/'
)
ai_classes_usage = ai_classes_usage.sort_values('rank')

org_packages_usage = first_uses_by_kagglers[
    (first_uses_by_kagglers['type'] == 'package') &
    (first_uses_by_kagglers['code_component'].isin(org_package_rank))
].copy()

org_packages_usage['rank'] = org_packages_usage['code_component'].map(org_package_rank)
org_packages_usage['link'] = (
    'https://kaggle.com/code/' +
    org_packages_usage['UserName'] + "/" +
    org_packages_usage['CurrentUrlSlug'] + '/'
)

org_packages_usage = org_packages_usage.sort_values('rank')

display(Markdown("## ğŸ¤– Kagglers who used top AI packages for the first time"))
display(HTML(ai_packages_usage[['UserName', 'code_component', 'link']].to_html(index=False)))

display(Markdown("## ğŸ§  Kagglers who used top AI functions for the first time"))
display(HTML(ai_functions_usage[['UserName', 'code_component', 'link']].to_html(index=False)))

display(Markdown("## âš™ï¸� Kagglers who used top AI methods for the first time"))
display(HTML(ai_methods_usage[['UserName', 'code_component', 'link']].to_html(index=False)))

display(Markdown("## ğŸ�—ï¸� Kagglers who used top AI classes for the first time"))
display(HTML(ai_classes_usage[['UserName', 'code_component', 'link']].to_html(index=False)))

display(Markdown("## ğŸ�¢ Kagglers who used organization-based AI packages for the first time"))
display(HTML(org_packages_usage[['UserName', 'code_component', 'link']].to_html(index=False)))


%whos DataFrame

del ai_classes_usage
del ai_functions_usage
del ai_methods_usage
del ai_packages_usage
del org_packages_usage
gc.collect()

%whos DataFrame

