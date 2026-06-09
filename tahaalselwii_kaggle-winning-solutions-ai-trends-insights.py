import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from collections import defaultdict
import re
# Load and clean the data
competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
competitions = competitions.dropna(subset=['HostSegmentTitle'])

# Count competition types
segment_counts = competitions['HostSegmentTitle'].value_counts()

# Create formatted legend labels: e.g., "Community (9063)"
legend_labels = [f"{name} ({count})" for name, count in zip(segment_counts.index, segment_counts)]

# Plot
plt.figure(figsize=(8, 8))
colors = plt.get_cmap('tab20').colors

# Draw pie chart without internal labels
wedges, _ = plt.pie(
    segment_counts,
    labels=None,
    colors=colors,
    startangle=140
)

# Add legend with names and counts
plt.legend(
    wedges,
    legend_labels,
    title='Competition Type',
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)

plt.title('Number of Kaggle Competitions by Type', fontsize=14)
plt.tight_layout()
plt.show()


def generate_kaggle_summary(year: int, file_path: str ):
    """
    Generate and display a styled HTML summary of Kaggle winning solutions for a given year.

    This function reads a CSV file containing information about Kaggle competition solutions,
    filters the entries by competition launch year, and renders a well-formatted and interactive
    HTML table that groups solutions by competition. Each solution title is rendered as a clickable link.

    Parameters:
    -----------
    year : int
        The year of competition launch to filter the dataset by.

    file_path : str
        Path to the CSV file containing the Kaggle winning solutions.
        
    Output:
    -------
    Displays an interactive HTML table directly inside a Jupyter Notebook.
    The output includes:
    - Competition name and index
    - Launch date and type
    - Competition overview
    - Table of solutions with clickable titles, summaries, and technologies used
    """

    # ================================
    # ğŸ“Œ 1. Load and Filter Data
    # ================================
    df = pd.read_csv(file_path)

    df['Competition Launch Date'] = pd.to_datetime(df['Competition Launch Date'], errors='coerce')
    df = df[df['Competition Launch Date'].dt.year == year]

    required_cols = {'Solution Link', 'Solution Title', 'Competition Name'}
    if not required_cols.issubset(df.columns):
        raise ValueError("âš ï¸� Required columns are missing from the file.")

    # ==========================================
    # ğŸ”— 2. Convert Solution Titles to Clickable Links
    # ==========================================
    df['Solution Title'] = df.apply(
        lambda x: f'<a href="{x["Solution Link"]}" target="_blank"><b>{x["Solution Title"]}</b></a>'
        if pd.notna(x["Solution Link"]) else x["Solution Title"],
        axis=1
    )

    # ================================
    # ğŸ“Š 3. Select Columns to Display
    # ================================
    display_cols = ['Solution Title', 'Solution Summary', 'Technologies Used']

    # ================================
    # ğŸ�¨ 4. CSS Styling for the Table
    # ================================
    css = """
    <style>
      .scroll-container {
        max-height: 600px;
        overflow-y: auto;
        overflow-x: auto;
        display: block;
        border: 1px solid #ccc;
        margin-top: 10px;
      }
      table {
        border-collapse: collapse;
        width: 100%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        direction: ltr;
      }
      thead th {
        background-color: white;
        color: black;
        padding: 10px;
        text-align: left !important;
        font-weight: bold;
        border-bottom: 2px solid #ddd;
        position: sticky;
        top: 0;
        z-index: 2;
      }
      tbody td {
        border: 1px solid #ddd;
        padding: 8px;
        vertical-align: top;
        text-align: left !important;
        word-wrap: break-word;
      }
      tbody td a {
        color: #0074cc;
        text-decoration: none;
        font-weight: bold;
        display: block;
      }
      tbody tr:nth-child(even) {
        background-color: #f9f9f9;
      }
      tbody tr:hover {
        background-color: #f1f1f1;
      }
      .competition-row td {
        background-color: #20BEFF;
        color: white;
        font-weight: bold;
        font-size: 16px;
        padding: 10px;
        border: none;
      }
      .launch-date-row td {
        background-color: #f5f5f5;
        color: #555;
        font-size: 12px;
        padding: 6px;
        border: none;
        font-style: italic;
      }
      .competition-overview td {
        background-color: #e8f4fc;
        color: #000;
        font-style: italic;
        padding: 8px;
        border: none;
      }
    </style>
    """

    # ================================
    # ğŸ§± 5. Build the HTML Content
    # ================================
    html = f"""
    <h2><strong>Per-Competition Winning Solutions - {year}</strong></h2>
    <div class="scroll-container">
    <table>
      <thead>
        <tr>
          <th>Solution Title</th>
          <th>Solution Summary</th>
          <th>Technologies Used</th>
        </tr>
      </thead>
      <tbody>
    """

    for idx, (comp_name, group) in enumerate(df.groupby('Competition Name', sort=False), start=1):
        launch_date = group['Competition Launch Date'].iloc[0]
        launch_str = launch_date.strftime('%d %B %Y â€“ %H:%M') if pd.notna(launch_date) else "Unknown Date"
        comp_type = group['Competition Type'].iloc[0] if 'Competition Type' in group.columns and pd.notna(group['Competition Type'].iloc[0]) else "Unknown Type"

        html += f'<tr class="competition-row"><td colspan="3">{idx}. {comp_name}</td></tr>'
        html += f'<tr class="launch-date-row">'
        html += f'<td colspan="2">Launched on: {launch_str}</td>'
        html += f'<td style="text-align: right; font-weight: bold;">{comp_type}</td>'
        html += '</tr>'

        if 'Competition Overview' in df.columns:
            comp_overview = group['Competition Overview'].iloc[0]
            if pd.notna(comp_overview) and comp_overview.strip():
                html += f'<tr class="competition-overview"><td colspan="3">{comp_overview}</td></tr>'

        for _, row in group.iterrows():
            html += "<tr>"
            for col in display_cols:
                val = row[col] if pd.notna(row[col]) else ""
                html += f"<td>{val}</td>"
            html += "</tr>"

    html += """
      </tbody>
    </table>
    </div>
    """

    # ================================
    # ğŸ“¤ 6. Display the Final Output
    # ================================
    display(HTML(css + html))


def plot_keyword_frequencies_vertical(keywords, csv_files, chart_title="", x_label="Technology", bar_color='orange'):
    """
    Counts the frequency of specified keywords from a list of CSV files (with a 'Technologies Used' column)
    and plots the results as a vertical bar chart.

    Parameters:
    - keywords: List of keyword strings to search for.
    - csv_files: List of CSV file paths.
    - chart_title: Title of the plot.
    - x_label: Label for the X-axis.
    - bar_color: Color of the bars in the plot.

    """
    if not keywords:
        print("âš ï¸� The provided keyword list is empty.")
        return pd.DataFrame(columns=["Keyword", "Count"])

    technology_texts = []

    # Read and collect all technology texts from CSV files
    for file in csv_files:
        df = pd.read_csv(file)
        if "Technologies Used" in df.columns:
            technology_texts += df["Technologies Used"].dropna().tolist()
        else:
            print(f"âš ï¸� File '{file}' does not contain 'Technologies Used' column.")

    keyword_counts = defaultdict(int)

    # Count occurrences of each keyword
    for text in technology_texts:
        text_lower = str(text).lower()
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower in text_lower:
                keyword_counts[keyword] += 1

    # Convert counts to DataFrame
    count_df = pd.DataFrame(list(keyword_counts.items()), columns=["Keyword", "Count"])
    count_df = count_df.sort_values(by="Count", ascending=False).reset_index(drop=True)

    # Plot all keywords
    plt.figure(figsize=(10, 6))
    plt.bar(count_df["Keyword"], count_df["Count"], color=bar_color)
    plt.title(chart_title)
    plt.xlabel(x_label)   
    plt.ylabel("Number of winning solutions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_keyword_trends_by_year(keyword_list, csv_files, title="Keyword Trends Over Years", line_colors=None):
    """
    Analyze and visualize the yearly trend of specific keywords in 'Technologies Used' across multiple CSV files.

    Parameters:
    - keyword_list: List of keywords to search for.
    - csv_files: List of CSV file paths containing 'Technologies Used' and 'Competition Launch Date'.
    - title: Plot title.
    - line_colors: Optional dictionary mapping each keyword to a specific color.
    
    """

    # Initialize structure to hold yearly keyword counts
    yearly_keyword_counts = defaultdict(lambda: defaultdict(int))

    for file in csv_files:
        df = pd.read_csv(file)

        if "Technologies Used" not in df.columns or "Competition Launch Date" not in df.columns:
            print(f"âš ï¸� File '{file}' is missing required columns.")
            continue

        for _, row in df.iterrows():
            tech_text = str(row["Technologies Used"]).lower()
            year = pd.to_datetime(row["Competition Launch Date"], errors='coerce').year

            if pd.isna(year):
                continue

            for keyword in keyword_list:
                keyword_lower = keyword.lower().strip()
                if keyword_lower in tech_text:
                    yearly_keyword_counts[year][keyword] += 1

    # Convert to DataFrame
    df_counts = pd.DataFrame(yearly_keyword_counts).T.fillna(0).astype(int)
    df_counts = df_counts[keyword_list]  # keep only the selected keywords
    df_counts = df_counts.sort_index()   # sort by year

    # Plotting
    plt.figure(figsize=(10, 6))
    for keyword in keyword_list:
        if keyword in df_counts.columns:
            color = line_colors.get(keyword) if line_colors else None
            plt.plot(df_counts.index, df_counts[keyword], marker='o', label=keyword, color=color)

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Number of winning solutions")
    plt.xticks(df_counts.index)
    plt.legend(title="Keyword")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_keyword_frequencies_horizontal(keywords, csv_files, title="Keyword Frequency", ylabel="Method name"):
    """
    Counts the frequency of specified keywords from multiple CSV files (each containing a 'Technologies Used' column)
    and displays the result using a horizontal bar chart.

    Parameters:
    - keywords: List of keywords to count and plot.
    - csv_files: List of CSV file paths to extract technologies from.
    - title: Title of the plot.
    - ylabel: Label for the y-axis.

    """
    if not keywords:
        print("âš ï¸� The provided keyword list is empty.")
        return pd.DataFrame(columns=["Keyword", "Count"])

    # Collect all tech texts from CSVs
    tech_texts = []
    for file in csv_files:
        df = pd.read_csv(file)
        if "Technologies Used" in df.columns:
            tech_texts += df["Technologies Used"].dropna().tolist()
        else:
            print(f"âš ï¸� File '{file}' does not contain 'Technologies Used' column.")

    # Count keyword occurrences
    keyword_counts = defaultdict(int)
    for text in tech_texts:
        text_lower = str(text).lower()
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower in text_lower:
                keyword_counts[keyword] += 1

    # Convert to DataFrame
    df = pd.DataFrame(list(keyword_counts.items()), columns=["Keyword", "Count"])
    df = df.sort_values(by="Count", ascending=True)  # ascending for horizontal bar

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(df["Keyword"], df["Count"], color=plt.cm.Paired(range(len(df))))
    plt.title(title)
    plt.xlabel("Number of winning solutions")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


# Custom file name
generate_kaggle_summary(2025, '/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2025.csv')


# Custom file name
generate_kaggle_summary(2024, '/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2024.csv')


# Custom file name
generate_kaggle_summary(2023, '/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2023.csv')


# Custom file name
generate_kaggle_summary(2022, '/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2022.csv')


# Custom file name
generate_kaggle_summary(2021, '/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2021.csv')


# Custom file name
generate_kaggle_summary(2020, '/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2020.csv')


# Custom file name
generate_kaggle_summary(2019, '/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2019.csv')


# CSV files
csv_files = [
    "/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2020.csv",
    "/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2021.csv",
    "/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2022.csv",
    "/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2023.csv",
    "/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2024.csv",
    "/kaggle/input/kaggle-winning-solutions-digest/Kaggle_Winning_Solutions_2025.csv"
]


# Define a list of keywords to plot
transformer_keywords = ["RoBERTa", "DeBERTa", "MaxViT", "Swin"]

plot_keyword_frequencies_vertical(
    keywords=transformer_keywords,
    csv_files = csv_files,
    chart_title="Usage of Transformers in Winning Kaggle Solutions (2020-2025)",
    x_label="Model",
    bar_color="skyblue"
)


# Define a list of keywords to plot
transformers_list = ["RoBERTa", "DeBERTa", "MaxViT", "Swin"]

line_colors = {
    "RoBERTa": "blue",
    "DeBERTa": "green",
    "MaxViT": "red",
    "Swin": "orange"
}

plot_keyword_trends_by_year(transformers_list, 
                           csv_files,
                           title="Yearly Trends of Transformers (2020â€“2025)",
                           line_colors=line_colors)


# Define a list of keywords to plot
LLMs =  ['Gemma', 'Qwen', 'DeepSeek','LLaMA']

plot_keyword_frequencies_vertical(
    keywords=LLMs,
    csv_files = csv_files,
    x_label="Model",
    chart_title="Usage of LLMs in Winning Kaggle Solutions (2020-2025)",
    bar_color="#800080"
)


# Define a list of keywords to plot
LLMs =  ['Gemma', 'Qwen', 'DeepSeek','LLaMA']

line_colors = {
    "Gemma": "blue",
    "Qwen": "green",
    "DeepSeek": "red",
    "LLaMA": "orange"
}

plot_keyword_trends_by_year(LLMs, 
                            csv_files, 
                            title="Yearly Trends of LLMs (2020â€“2025)",
                            line_colors=line_colors)


# Define a list of keywords to plot
Gradient_Boosting =  ['LightGBM', 'XGBoost', 'CatBoost']

plot_keyword_frequencies_vertical(
    keywords=Gradient_Boosting,
    csv_files = csv_files,
    x_label="Algorithm",
    chart_title="Usage of Gradient Boosting Algorithms in Winning Kaggle Solutions (2020-2025)",
    bar_color="magenta"
)


line_colors = {
    "LightGBM": "blue",
    "XGBoost": "red",
    "CatBoost": "orange"
}

Gradient_Boosting =  ['LightGBM', 'XGBoost', 'CatBoost']
plot_keyword_trends_by_year(Gradient_Boosting, csv_files, 
                                       title="Yearly Trends of Gradient Boosting Algorithms (2020â€“2025)",
                                       line_colors=line_colors)


# Define a list of data augmentation techniques to visualize
augmentation_methods = [
        'Albumentations',' TTA', 'MixUp', 'CutMix', 'MLQA', 
        'SpecAugment', 'RandAugment', 'Cutout'
]

plot_keyword_frequencies_horizontal(
    keywords=augmentation_methods,
    csv_files = csv_files,
    title="Common Data Augmentation Techniques Used in winning Solutions (2020â€“2025)",
    ylabel = "Technique"
)  


# Define a list of loss functions to visualize
Loss_Functions = [
        'BCE', 'BCEWithLogitsLoss', 'Focal', 'ArcFace', 'CrossEntropy', 'Dice'
]

plot_keyword_frequencies_horizontal(
    keywords = Loss_Functions,
    csv_files = csv_files,
    title = "Common Loss Functions Used in winning Solutions (2020â€“2025)",
    ylabel = "Function"
)


# Define a list of optimizers to visualize
Optimizers = ['AdamW', 'SWA', 'SGD', 'Bayesian Optimization']

plot_keyword_frequencies_horizontal(
    keywords = Optimizers,
    csv_files = csv_files,
    title = "Common Optimizers Used in winning Solutions (2020â€“2025)",
    ylabel = "Optimizer"
)


# Define a list of Tools and Libraries to visualize
Tools_Libraries = [
         'PyTorch', 'OpenCV', 'Tensorflow', 'timm', 'vLLM', 'Polars', 
        'scikit-learn', 'Keras', 'mmdetection', 'MONAI', 'OpenVINO', 'NumPy', 
        'ONNX', 'TensorRT', 'Pandas', 'jax', 'fastai', 'mlxtend'
        ]

plot_keyword_frequencies_horizontal(
    keywords = Tools_Libraries,
    csv_files = csv_files,
    title = "Common Libraries and Frameworks Used in winning Solutions (2020â€“2025)",
    ylabel = "Tool"
)

