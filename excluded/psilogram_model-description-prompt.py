# --------------------------
# Required libraries
# --------------------------
import pandas as pd  # For data manipulation and analysis
from openai import OpenAI  # OpenAI client library (requires installation via pip install openai)
from kaggle_secrets import UserSecretsClient  # To securely retrieve stored secrets like API keys
import os  # Provides functions for interacting with the operating system
import re  # Regular expression operations for pattern matching
import json  # For JSON encoding and decoding
from openai import OpenAI  # Importing OpenAI library again (likely redundant but maintains original logic)

# Initialize a client to access user secrets stored in Kaggle
user_secrets = UserSecretsClient()

# Retrieve the OpenAI API key securely from Kaggle's secrets storage
openai_key = UserSecretsClient().get_secret("Openai")



DATA_FOLDER = "/kaggle/input/amazon-grid-analysis"  # Path to the input dataset folder

def list_featimp_files(folder):
    """
    List and sort feature importance files in descending order by scale.

    Args:
        folder (str): Directory containing the feature importance CSV files.

    Returns:
        list of tuples: Each tuple contains (scale, filename).
    """
    return sorted(
        [(int(m.group(1)), f) for f in os.listdir(folder)
         if (m := re.search(r'featimp_(\d+)\.0\.csv$', f))],  # Extract numeric scale from filenames
        reverse=True  # Sort in descending order of scale
    )

def collect_unique_features(files, data_folder):
    """
    Collect a list of unique features and their descriptions from CSV files.

    Args:
        files (list): List of tuples (scale, filename).
        data_folder (str): Directory where the files are located.

    Returns:
        list: List of dictionaries with unique feature and description pairs.
    """
    seen = set()
    features = []
    for _, fname in files:
        df = pd.read_csv(os.path.join(data_folder, fname))  # Read feature importance CSV
        for _, row in df.iterrows():
            key = (row['feature'], row['description'])
            if key not in seen:
                seen.add(key)
                features.append({
                    "feature": row['feature'],
                    "current_description": row['description']
                })
    return features

def request_refined_descriptions(raw_features, client, model="gpt-4.1"):
    """
    Use an LLM to generate improved descriptions for terrain features.

    Args:
        raw_features (list): List of dictionaries with raw features and descriptions.
        client (OpenAI): Initialized OpenAI client.
        model (str): OpenAI model to use (default is "gpt-4.1").

    Returns:
        list: List of dictionaries with improved descriptions.

    Raises:
        ValueError: If the LLM response cannot be parsed as JSON.
    """
    prompt = f"""
You are a geoarchaeologist and terrain modeling expert.

Below is a list of terrain features with short descriptions, used to predict promising archaeological sites.

**Task:**
- For each feature, rewrite the description to be richer, more precise, and mention relevant landforms (plateaus, terraces, rivers, floodplains, etc).
- Keep each description under 30 words.
- Return ONLY valid JSON: a list of {{"feature": "...", "improved_description": "..."}} objects.

Features:
{json.dumps(raw_features, indent=2)}
"""

    # Send prompt to OpenAI chat API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a geoarchaeologist. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message.content.strip()

    try:
        # Handle optional code block formatting
        if response.startswith("```json"):
            response = response.split("```json")[-1].split("```")[0].strip()
        elif response.startswith("```"):
            response = response.strip("```").strip()
        return json.loads(response)  # Parse JSON output from model
    except Exception as e:
        raise ValueError("Failed to parse GPT JSON response.") from e

def build_trend_data(files, feature_desc_map, data_folder, top_n=None, normalize=True):
    """
    Build trend data showing the importance of features across spatial scales.

    Args:
        files (list): List of tuples (scale, filename).
        feature_desc_map (dict): Map of feature names to improved descriptions.
        data_folder (str): Folder where CSV files are stored.
        top_n (int, optional): Number of top features to include per scale. If None, include all.
        normalize (bool): Whether to normalize importance values to 0-100 range.

    Returns:
        list: Trend data containing feature importance info per scale.
    """
    all_importances = []

    # First pass: collect all importance values to calculate normalization range
    for scale, fname in files:
        df = pd.read_csv(os.path.join(data_folder, fname))
        all_importances.extend(df['importance_mean'].values)

    min_imp = min(all_importances)
    max_imp = max(all_importances)
    range_imp = max_imp - min_imp if max_imp != min_imp else 1  # Prevent division by zero

    trend = []
    for scale, fname in files:
        df = pd.read_csv(os.path.join(data_folder, fname))
        if normalize:
            # Normalize importance values to a 0-100 scale
            df['importance_mean'] = 100 * (df['importance_mean'] - min_imp) / range_imp

        if top_n is not None:
            # Select top N features by importance
            df = df.sort_values("importance_mean", ascending=False).head(top_n)

        records = []
        for _, row in df.iterrows():
            records.append({
                "feature": row['feature'],
                "description": feature_desc_map.get(row['feature'], row['description']),
                "importance": row['importance_mean'],
                "mean_all": row['mean_all_cells'],
                "mean_top5pct": row['mean_top_5pct'],
                "mean_top10pct": row['mean_top_10pct']
            })

        trend.append({
            "cell_size_m": scale,  # Spatial resolution of the analysis cell
            "top_features": records
        })

    return trend




def generate_summary_prompt(trend_json, top_n):
    """
    Create a detailed prompt for the LLM to generate a geoarchaeological summary.

    Args:
        trend_json (list): Multi-scale feature importance and statistics data.
        top_n (int): Number of top features per scale included in the summary.

    Returns:
        str: A formatted prompt string for input to the language model.
    """
    return f"""
You are an expert geoarchaeologist and senior field advisor.

Below is multi-scale data from a one-class SVM model predicting promising areas for undiscovered archaeological sites in the Amazon Basin. Each resolution includes:
- The TOP {top_n} most important features
- Improved description
- Importance score
- Mean value across all cells vs. top 5% and 10% best-scoring cells

Note:
Each finer resolution keeps only high-score areas from the previous level, zooming in on the most promising terrain.

**Multi-resolution results:**  
{json.dumps(trend_json, indent=2)}

**Write a concise analytical summary (≤500 words):**
- Explain what each feature reveals about terrain (note that delta features describe how the cell differs from its neighbors)
- Describe how importance shifts with scale
- Relate these to landforms: plateaus, terraces, ridges, rivers, floodplains, etc.
- Avoid filler. Be precise and geoarchaeologically grounded.
- Finish with a summary of terrain types the model is favoring.
- Write in a formal, academic register.  
  • Avoid evaluative or promotional adjectives such as “prime”, “ideal”, “preferred”, “excellent”, “high-potential”.  
  • Favour evidence-based verbs (“indicates”, “suggests”, “is consistent with”) and hedging where appropriate.  
- Format the code with markup commands for a clean presentation
"""

def generate_multiscale_archaeo_summary(api_key, model="gpt-4.1", top_n=5, data_folder=DATA_FOLDER):
    """
    Generates a summary and multi-scale analysis of terrain feature importance
    using OpenAI's GPT model.

    Args:
        api_key (str): OpenAI API key for authentication.
        model (str): The OpenAI model to use (default is "gpt-4.1").
        top_n (int): Number of top features to include per resolution scale.
        data_folder (str): Path to the folder containing input feature importance data.

    Returns:
        tuple: (LLM-generated summary string, list of multi-scale trend data).
    """
    # Initialize the OpenAI client with the provided API key
    client = OpenAI(api_key=api_key)

    # Load and sort feature importance files
    files = list_featimp_files(data_folder)

    # Extract unique features with current descriptions
    raw_features = collect_unique_features(files, data_folder)

    # Request improved feature descriptions from the language model
    improved = request_refined_descriptions(raw_features, client, model=model)

    # Map each feature to its refined description
    desc_map = {d["feature"]: d["improved_description"] for d in improved}

    # Build trend data (importance and summary stats) across all scales
    multiscale_data = build_trend_data(files, desc_map, data_folder, top_n=top_n)

    # Generate a rich summary prompt based on the multiscale data
    summary_prompt = generate_summary_prompt(multiscale_data, top_n)

    # Send the prompt to the model to generate the analytical summary
    final_response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert geoarchaeologist and senior data scientist. "
                    "Always analyze numeric trends carefully, link them to real landforms, "
                    "and keep your summary precise and clear."
                )
            },
            {"role": "user", "content": summary_prompt}
        ]
    )

    # Return the summary content and the data used to generate it
    return final_response.choices[0].message.content, multiscale_data




# Step 1: Load and sort feature importance files from the data folder
files = list_featimp_files(DATA_FOLDER)

# Step 2: Collect all unique terrain features and their current descriptions
raw_features = collect_unique_features(files, DATA_FOLDER)

# Step 3: Initialize the OpenAI client with the provided API key
client = OpenAI(api_key=openai_key)

# Step 4: Send the features to the model to get richer, domain-specific descriptions
improved = request_refined_descriptions(raw_features, client)

# Step 5: Create a lookup dictionary mapping each feature to its improved description
desc_map = {d["feature"]: d["improved_description"] for d in improved}

# Step 6: Generate trend data across spatial resolutions using the improved descriptions
multiscale_data = build_trend_data(files, desc_map, DATA_FOLDER, top_n=10)

# Step 7: Generate a detailed prompt to request a multi-scale analytical summary
summary_prompt = generate_summary_prompt(multiscale_data, top_n=10)

# Step 8: Request a detailed, geoarchaeologically grounded summary from the model
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a geoarchaeologist and senior data scientist."},
        {"role": "user", "content": summary_prompt}
    ]
)

# Step 9: Extract the summary text from the model response
summary_text = response.choices[0].message.content

# Step 10: Save the summary to a local file
with open("OpenAI_model_description.txt", "w") as file:
    file.write(summary_text)

# Step 11: Print the summary to the console
print(summary_text)




import pandas as pd
import matplotlib.pyplot as plt

def plot_normalized_feature_trends(trend_data, top_n=5):
    """
    Plot normalized feature importance trends across spatial resolutions.

    This visualization highlights how the relative importance of top features
    changes across cell sizes (resolutions) used in the model.

    Args:
        trend_data (list): List of dictionaries with feature importance across scales.
        top_n (int): Number of top features to track from both coarsest and finest resolutions.
    """
    # Flatten the trend data into a tabular list
    rows = []
    for entry in trend_data:
        scale = entry['cell_size_m']  # spatial resolution
        for f in entry['top_features']:
            rows.append({
                'cell_size_m': scale,
                'feature': f['feature'],
                'importance': f['importance']
            })

    # Convert the flattened list into a DataFrame
    df = pd.DataFrame(rows)

    # Normalize importance within each resolution (0 to 1 scale)
    df['importance_norm'] = df.groupby('cell_size_m')['importance'].transform(
        lambda x: x / x.max() if x.max() > 0 else x
    )

    # Get a sorted list of resolutions, from coarsest to finest
    resolutions = sorted(df['cell_size_m'].unique(), reverse=True)
    resolution_indices = {res: i for i, res in enumerate(resolutions)}

    # Identify top features at both coarsest and finest scales
    coarse_top = df[df['cell_size_m'] == resolutions[0]].nlargest(top_n, 'importance')
    fine_top = df[df['cell_size_m'] == resolutions[-1]].nlargest(top_n, 'importance')
    selected_features = pd.concat([coarse_top, fine_top])['feature'].unique()

    # Filter the data to keep only the selected features
    filtered = df[df['feature'].isin(selected_features)]

    # Pivot the data: rows = resolution, columns = feature, values = normalized importance
    pivot_df = (
        filtered
        .pivot(index='cell_size_m', columns='feature', values='importance_norm')
        .reindex(resolutions)  # Ensure all resolutions are in correct order
        .fillna(0)              # Fill missing values with 0
    )

    # Plot using evenly spaced x-axis values
    x_vals = list(range(len(resolutions)))
    x_labels = [str(r) for r in resolutions]

    plt.figure(figsize=(15, 8))
    for feature in pivot_df.columns:
        y_vals = pivot_df[feature].values
        plt.plot(x_vals, y_vals, marker='o', label=feature)

    # Plot aesthetics and labels
    plt.title(f"Normalized Feature Importance Across Resolutions\n(top {top_n} at coarsest and finest)")
    plt.xlabel("Resolution (Cell Size in meters)")
    plt.ylabel("Normalized Importance")
    plt.xticks(ticks=x_vals, labels=x_labels)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save plot to file
    plt.savefig('Feature_importance_trends.png')

    # Show plot
    plt.show()

# Build full-resolution trend data without filtering by top N features
multiscale_data_full = build_trend_data(files, desc_map, DATA_FOLDER, top_n=None)

# Generate the plot showing how top features behave across scales
plot_normalized_feature_trends(multiscale_data_full, top_n=5)


