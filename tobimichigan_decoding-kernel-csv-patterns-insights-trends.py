from IPython.display import display
from PIL import Image

# Load and display the image
image_path = '/kaggle/input/hsa-kaggledataanalytics/HSA_KaggleMetaDataAnalytics.png'
img = Image.open(image_path)
display(img)


from IPython.display import IFrame

# Display the video using an iframe
IFrame(src="https://www.youtube.com/embed/3Y4gEBPfzfo", width=800, height=450)


!pip install bertopic


import os
import nbformat
import pandas as pd
import numpy as np
import gc
import re
import ast
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import time # Import the time module for timing
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy.stats import zscore
import networkx as nx
from gensim.models import Word2Vec
from bertopic import BERTopic
import torch
from transformers import AutoTokenizer, AutoModel
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Attention, Reshape, Permute, multiply, concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib # For saving the best model

# Define root paths for data
ROOT_PATH_CODE = "/kaggle/input/meta-kaggle-code"
ROOT_PATH_CSV = "/kaggle/input/meta-kaggle"

# --- Global Configuration and Helper Functions ---

def clean_memory():
    """Aggressively cleans memory."""
    gc.collect() # Collect Python garbage
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear PyTorch CUDA cache
    # print("Memory cleaned.") # Suppress this frequent print for cleaner output

def get_file_paths(root_dir, extensions, num_files_per_ext=20):
    """
    Walks through the directory tree and collects a random sample of file paths
    for specified extensions. Stops scanning for an extension once num_files_per_ext
    are found. The overall scan stops when all desired counts are met.
    """
    collected_files_by_ext = {ext: [] for ext in extensions}
    # Track how many more files are needed for each extension
    remaining_needed = {ext: num_files_per_ext for ext in extensions}
    
    print(f"Collecting up to {num_files_per_ext} files per extension from {root_dir}...")

    # Custom tqdm for directory scanning
    dir_count = 0
    file_found_count = 0
    pbar = tqdm(desc="Scanning directories and collecting files", unit="dirs")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dir_count += 1
        pbar.update(1)

        for filename in filenames:
            for ext in extensions:
                if filename.endswith(ext) and remaining_needed[ext] > 0:
                    collected_files_by_ext[ext].append(os.path.join(dirpath, filename))
                    remaining_needed[ext] -= 1
                    file_found_count += 1
                    # Update tqdm for files found (optional, but good for feedback)
                    pbar.set_postfix_str(f"Files: {file_found_count}")

        # Check if we have collected enough files for all extensions
        all_extensions_full = all(remaining_needed[ext] == 0 for ext in extensions)
        if all_extensions_full:
            pbar.close() # Close the progress bar as we are done
            print("All desired file counts met. Stopping directory scan early.")
            break # Break out of the os.walk loop

    pbar.close() # Ensure pbar is closed if loop finishes naturally

    selected_files = []
    for ext in extensions:
        # If we collected more than needed (due to batching in os.walk or if num_files_per_ext was small),
        # take a random sample. Otherwise, take all collected.
        if len(collected_files_by_ext[ext]) > num_files_per_ext:
            selected_files.extend(random.sample(collected_files_by_ext[ext], num_files_per_ext))
        else:
            selected_files.extend(collected_files_by_ext[ext])

    print(f"Finished collecting files. Total selected: {len(selected_files)}.")
    return selected_files

def extract_imports(code_string):
    """
    Extracts unique top-level library imports from a Python code string.
    Handles 'import x' and 'from x import y'.
    """
    imports = set()
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0]) # Get top-level module
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0]) # Get top-level module
    except SyntaxError:
        # Handle cases where code might be malformed
        pass
    return list(imports)

def get_code_lines(code_string):
    """Counts non-empty lines of code."""
    return len([line for line in code_string.split('\n') if line.strip() and not line.strip().startswith('#')])

# --- Pipeline 1: Code Metadata Extraction Pipeline ---

class CodeMetadataExtractor:
    """
    Parses code files (Jupyter Notebooks, Python, R) to extract metadata,
    library imports, and code metrics.
    """
    def __init__(self, root_path_code):
        self.root_path_code = root_path_code
        self.notebook_stats_df = pd.DataFrame()

    def extract_ipynb_metadata(self, filepath):
        """Extracts metadata from a .ipynb file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            metadata = nb.metadata.get('metadata', {})
            author = metadata.get('author', 'Unknown')
            created = metadata.get('created', '1970-01-01T00:00:00Z') # Default to epoch if not found

            code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
            markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']

            all_code = "\n".join([cell['source'] for cell in code_cells])
            libraries = extract_imports(all_code)
            total_lines_of_code = get_code_lines(all_code)

            execution_count_sum = sum(cell.get('execution_count', 0) or 0 for cell in code_cells) # Handle None

            # Explicitly delete large objects no longer needed
            del nb
            del code_cells
            del markdown_cells
            del all_code
            clean_memory()

            return {
                'file_path': filepath,
                'file_type': 'ipynb',
                'author': author,
                'creation_date': pd.to_datetime(created, errors='coerce'),
                'libraries_used': libraries,
                'code_cell_count': len(code_cells),
                'markdown_cell_count': len(markdown_cells),
                'total_lines_of_code': total_lines_of_code,
                'execution_count_sum': execution_count_sum
            }
        except Exception as e:
            # print(f"Error processing {filepath}: {e}") # Suppress frequent error prints
            return None

    def extract_py_metadata(self, filepath):
        """Extracts metadata from a .py file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            libraries = extract_imports(content)
            total_lines_of_code = get_code_lines(content)

            # Explicitly delete large objects no longer needed
            del content
            clean_memory()

            # For .py files, author and creation date are harder to get programmatically
            # without version control or specific headers. Defaulting to placeholders.
            return {
                'file_path': filepath,
                'file_type': 'py',
                'author': 'Unknown',
                'creation_date': pd.to_datetime(os.path.getctime(filepath), unit='s', errors='coerce'), # File creation time
                'libraries_used': libraries,
                'code_cell_count': 1, # Treat as one code cell
                'markdown_cell_count': 0,
                'total_lines_of_code': total_lines_of_code,
                'execution_count_sum': 0 # No execution count for standalone .py
            }
        except Exception as e:
            # print(f"Error processing {filepath}: {e}") # Suppress frequent error prints
            return None

    def extract_r_metadata(self, filepath):
        """Extracts metadata from an .R or .Rmd file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple regex to find R library imports
            r_libraries = re.findall(r'(?:library|require)\(([\w.]+)\)', content)
            total_lines_of_code = get_code_lines(content)

            # Explicitly delete large objects no longer needed
            del content
            clean_memory()

            return {
                'file_path': filepath,
                'file_type': 'r',
                'author': 'Unknown',
                'creation_date': pd.to_datetime(os.path.getctime(filepath), unit='s', errors='coerce'),
                'libraries_used': list(set(r_libraries)),
                'code_cell_count': 1, # Treat as one code cell
                'markdown_cell_count': 0,
                'total_lines_of_code': total_lines_of_code,
                'execution_count_sum': 0
            }
        except Exception as e:
            # print(f"Error processing {filepath}: {e}") # Suppress frequent error prints
            return None

    def run_pipeline(self):
        """
        Runs the code metadata extraction pipeline.
        Collects 20 random files for each type (.ipynb, .py, .r).
        """
        start_time = time.time()
        print("Starting Code Metadata Extraction Pipeline...")
        extensions = ['.ipynb', '.py', '.r', '.rmd'] # Include .rmd for R notebooks
        selected_files = get_file_paths(self.root_path_code, extensions, num_files_per_ext=20)

        extracted_data = []
        for filepath in tqdm(selected_files, desc="Processing code files"):
            if filepath.endswith('.ipynb'):
                data = self.extract_ipynb_metadata(filepath)
            elif filepath.endswith('.py'):
                data = self.extract_py_metadata(filepath)
            elif filepath.endswith(('.r', '.rmd')):
                data = self.extract_r_metadata(filepath)
            else:
                data = None

            if data:
                extracted_data.append(data)
            clean_memory() # Clean memory after each file processing

        self.notebook_stats_df = pd.DataFrame(extracted_data)
        self.notebook_stats_df['creation_date'] = pd.to_datetime(self.notebook_stats_df['creation_date'], errors='coerce')
        self.notebook_stats_df.dropna(subset=['creation_date'], inplace=True) # Drop rows where date parsing failed

        # Explicitly delete the list after DataFrame creation
        del extracted_data
        clean_memory()

        # Persist to Parquet
        output_path = 'notebook_stats.parquet'
        print(f"Saving code metadata to {output_path}...")
        self.notebook_stats_df.to_parquet(output_path, index=False)
        print(f"Code metadata saved to {output_path}")

        # Summary table of top-used libraries and average code length
        all_libraries = [lib for sublist in self.notebook_stats_df['libraries_used'] for lib in sublist]
        library_counts = pd.Series(all_libraries).value_counts().head(10)
        print("\nTop 10 Most Used Libraries:")
        print(library_counts)

        avg_code_length = self.notebook_stats_df['total_lines_of_code'].mean()
        print(f"\nAverage Total Lines of Code: {avg_code_length:.2f}")

        # The notebook_stats_df is returned and used by subsequent pipelines.
        # So we don't delete it here, but ensure other temporary objects are cleared.
        del all_libraries
        del library_counts
        clean_memory()

        end_time = time.time()
        print(f"Code Metadata Extraction Pipeline Completed in {end_time - start_time:.2f} seconds.")
        return self.notebook_stats_df

# --- Pipeline 2: Temporal Trend Analysis Pipeline ---

class TemporalTrendAnalyzer:
    """
    Tracks evolution of participation, topics, and performance over time.
    """
    def __init__(self, root_path_csv, notebook_stats_df):
        self.root_path_csv = root_path_csv
        self.notebook_stats_df = notebook_stats_df # This is passed from previous pipeline
        self.competitions_df = None
        self.submissions_df = None
        self.kernels_df = None

    def load_data(self):
        """Loads necessary CSV data."""
        load_start_time = time.time()
        print("Loading data for Temporal Trend Analysis...")
        try:
            self.competitions_df = pd.read_csv(os.path.join(self.root_path_csv, 'Competitions.csv'), low_memory=False)
            self.submissions_df = pd.read_csv(os.path.join(self.root_path_csv, 'Submissions.csv'), low_memory=False)
            self.kernels_df = pd.read_csv(os.path.join(self.root_path_csv, 'Kernels.csv'), low_memory=False)

            print("Parsing date columns...")
            self.competitions_df['EnabledDate'] = pd.to_datetime(self.competitions_df['EnabledDate'], errors='coerce')
            self.competitions_df['DeadlineDate'] = pd.to_datetime(self.competitions_df['DeadlineDate'], errors='coerce')
            self.submissions_df['SubmissionDate'] = pd.to_datetime(self.submissions_df['SubmissionDate'], errors='coerce')
            self.kernels_df['CreationDate'] = pd.to_datetime(self.kernels_df['CreationDate'], errors='coerce')

            # Drop rows with invalid dates
            self.competitions_df.dropna(subset=['EnabledDate', 'DeadlineDate'], inplace=True)
            self.submissions_df.dropna(subset=['SubmissionDate'], inplace=True)
            self.kernels_df.dropna(subset=['CreationDate'], inplace=True)
            print("Data loaded and dates parsed.")
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.root_path_csv}")
            # Create empty DataFrames to prevent further errors
            self.competitions_df = pd.DataFrame(columns=['Id', 'EnabledDate', 'DeadlineDate'])
            # Ensure PublicScore and PrivateScore columns exist in the empty DataFrame for submissions
            self.submissions_df = pd.DataFrame(columns=['Id', 'SubmissionDate', 'PublicScore', 'PrivateScore'])
            self.kernels_df = pd.DataFrame(columns=['Id', 'CreationDate'])
        
        clean_memory() # Clean memory after initial data loading and parsing
        load_end_time = time.time()
        print(f"Data loading for Temporal Trend Analysis completed in {load_end_time - load_start_time:.2f} seconds.")

    def time_series_aggregation(self):
        """Aggregates data into monthly time series."""
        agg_start_time = time.time()
        print("Aggregating time series data...")

        # Competitions
        if not self.competitions_df.empty:
            comp_monthly = self.competitions_df.groupby(self.competitions_df['EnabledDate'].dt.to_period('M')).size().reset_index(name='count')
            comp_monthly['EnabledDate'] = comp_monthly['EnabledDate'].dt.to_timestamp()
            comp_monthly.rename(columns={'EnabledDate': 'Month'}, inplace=True)
            comp_monthly.to_csv('monthly_competitions.csv', index=False)
            del comp_monthly # Delete temporary DataFrame
            clean_memory()
            print("Monthly competitions data aggregated.")
        else:
            print("Competitions data is empty, skipping monthly aggregation.")

        # Kernels
        if not self.kernels_df.empty:
            kernel_monthly = self.kernels_df.groupby(self.kernels_df['CreationDate'].dt.to_period('M')).size().reset_index(name='count')
            kernel_monthly['CreationDate'] = kernel_monthly['CreationDate'].dt.to_timestamp()
            kernel_monthly.rename(columns={'CreationDate': 'Month'}, inplace=True)
            kernel_monthly.to_csv('monthly_kernels.csv', index=False)
            del kernel_monthly # Delete temporary DataFrame
            clean_memory()
            print("Monthly kernels data aggregated.")
        else:
            print("Kernels data is empty, skipping monthly aggregation.")

        # Submissions
        if not self.submissions_df.empty:
            submission_monthly = self.submissions_df.groupby(self.submissions_df['SubmissionDate'].dt.to_period('M')).size().reset_index(name='count')
            submission_monthly['SubmissionDate'] = submission_monthly['SubmissionDate'].dt.to_timestamp()
            submission_monthly.rename(columns={'SubmissionDate': 'Month'}, inplace=True)
            submission_monthly.to_csv('monthly_submissions.csv', index=False)
            del submission_monthly # Delete temporary DataFrame
            clean_memory()
            print("Monthly submissions data aggregated.")
        else:
            print("Submissions data is empty, skipping monthly aggregation.")

        # Average scores per competition over time
        if not self.submissions_df.empty and 'PublicScore' in self.submissions_df.columns and 'PrivateScore' in self.submissions_df.columns:
            self.submissions_df['SubmissionMonth'] = self.submissions_df['SubmissionDate'].dt.to_period('M')
            avg_scores_monthly = self.submissions_df.groupby('SubmissionMonth')[['PublicScore', 'PrivateScore']].mean().reset_index()
            avg_scores_monthly['SubmissionMonth'] = avg_scores_monthly['SubmissionMonth'].dt.to_timestamp()
            avg_scores_monthly.rename(columns={'SubmissionMonth': 'Month'}, inplace=True)
            avg_scores_monthly.to_csv('monthly_average_scores.csv', index=False)
            del avg_scores_monthly # Delete temporary DataFrame
            clean_memory()
        else:
            print("Warning: 'PublicScore' or 'PrivateScore' columns not found in submissions data or submissions data is empty. Skipping average score aggregation.")
        
        print("Monthly time-series CSVs created.")
        agg_end_time = time.time()
        print(f"Time-series aggregation completed in {agg_end_time - agg_start_time:.2f} seconds.")

        # After aggregation, the original large DataFrames are no longer needed
        # for the rest of this pipeline's steps (plotting uses the aggregated CSVs).
        del self.competitions_df
        del self.submissions_df
        del self.kernels_df
        self.competitions_df = None
        self.submissions_df = None
        self.kernels_df = None
        clean_memory()

    def analyze_seasonality_growth(self):
        """Analyzes seasonality and growth trends (simplified)."""
        analysis_start_time = time.time()
        print("Analyzing seasonality and growth (simplified)...")
        
        # Example: Plotting kernels created growth
        try:
            kernels_monthly_df = pd.read_csv('monthly_kernels.csv', parse_dates=['Month'])
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=kernels_monthly_df, x='Month', y='count')
            plt.title('Kernels Created Over Time')
            plt.xlabel('Date')
            plt.ylabel('Number of Kernels')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig('kernels_created_growth.png')
            plt.close() # Close the plot to free memory
            del kernels_monthly_df
            clean_memory()
            print("Kernels created growth plot generated.")
        except FileNotFoundError:
            print("monthly_kernels.csv not found, cannot analyze kernels growth.")
        except Exception as e:
            print(f"Error generating kernels growth plot: {e}")
            plt.close() # Ensure plot is closed even on error
            clean_memory()

        analysis_end_time = time.time()
        print(f"Seasonality and growth analysis completed in {analysis_end_time - analysis_start_time:.2f} seconds.")
        clean_memory()

    def generate_trend_visualizations(self):
        """Generates visualizations for trends."""
        viz_start_time = time.time()
        print("Generating trend visualizations...")

        # Plot Kernels Created vs. Time with Rolling Average
        try:
            kernels_monthly_df = pd.read_csv('monthly_kernels.csv', parse_dates=['Month'])
            if not kernels_monthly_df.empty and 'count' in kernels_monthly_df.columns:
                # Calculate rolling average
                kernels_monthly_df['rolling_avg_count'] = kernels_monthly_df['count'].rolling(window=3, center=True).mean()

                plt.figure(figsize=(12, 6))
                sns.lineplot(data=kernels_monthly_df, x='Month', y='count', label='Monthly Count')
                sns.lineplot(data=kernels_monthly_df, x='Month', y='rolling_avg_count', label='3-Month Rolling Average', linestyle='--')
                plt.title('Kernels Created Over Time with Rolling Average')
                plt.xlabel('Month')
                plt.ylabel('Count')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('kernels_created_trend.png')
                plt.close() # Close the plot to free memory
                del kernels_monthly_df
                clean_memory()
                print("Kernels created vs. time plot with rolling average generated.")
            else:
                print("monthly_kernels.csv is empty or missing 'count' column, skipping kernels created trend plot.")
        except FileNotFoundError:
            print("monthly_kernels.csv not found, skipping kernels created trend plot.")
        except Exception as e:
            print(f"Error generating kernels created trend plot: {e}")
            plt.close() # Ensure plot is closed even on error
            clean_memory()

        # Plot Submission Scores Over Time (if data exists)
        try:
            avg_scores_df = pd.read_csv('monthly_average_scores.csv', parse_dates=['Month'])
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=avg_scores_df, x='Month', y='PublicScore', label='Public Score')
            sns.lineplot(data=avg_scores_df, x='Month', y='PrivateScore', label='Private Score')
            plt.title('Average Submission Scores Over Time')
            plt.xlabel('Month')
            plt.ylabel('Average Score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig('average_submission_scores_trend.png')
            plt.close() # Close the plot to free memory
            del avg_scores_df
            clean_memory()
            print("Average submission scores vs. time plot generated.")
        except FileNotFoundError:
            print("No submission score data to plot (monthly_average_scores.csv not found).")
        except Exception as e:
            print(f"Error generating submission scores trend plot: {e}")
            plt.close() # Ensure plot is closed even on error
            clean_memory()

        # New: Plot Monthly Competitions
        try:
            comp_monthly_df = pd.read_csv('monthly_competitions.csv', parse_dates=['Month'])
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=comp_monthly_df, x='Month', y='count')
            plt.title('Monthly Competitions Enabled')
            plt.xlabel('Month')
            plt.ylabel('Number of Competitions')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig('monthly_competitions_trend.png')
            plt.close()
            del comp_monthly_df
            clean_memory()
            print("Generated: monthly_competitions_trend.png")
        except FileNotFoundError:
            print("monthly_competitions.csv not found, skipping monthly competitions trend plot.")
        except Exception as e:
            print(f"Error generating monthly competitions trend plot: {e}")
            plt.close()
            clean_memory()

        # New: Plot Monthly Submissions
        try:
            sub_monthly_df = pd.read_csv('monthly_submissions.csv', parse_dates=['Month'])
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=sub_monthly_df, x='Month', y='count')
            plt.title('Monthly Submissions Count')
            plt.xlabel('Month')
            plt.ylabel('Number of Submissions')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig('monthly_submissions_trend.png')
            plt.close()
            del sub_monthly_df
            clean_memory()
            print("Generated: monthly_submissions_trend.png")
        except FileNotFoundError:
            print("monthly_submissions.csv not found, skipping monthly submissions trend plot.")
        except Exception as e:
            print(f"Error generating monthly submissions trend plot: {e}")
            plt.close()
            clean_memory()


        viz_end_time = time.time()
        print(f"Trend visualizations generated in {viz_end_time - viz_start_time:.2f} seconds.")
        clean_memory()

    def run_pipeline(self):
        """Runs the temporal trend analysis pipeline."""
        pipeline_start_time = time.time()
        print("Starting Temporal Trend Analysis Pipeline...")
        self.load_data()
        self.time_series_aggregation()
        self.analyze_seasonality_growth()
        self.generate_trend_visualizations()
        pipeline_end_time = time.time()
        print(f"Temporal Trend Analysis Pipeline Completed in {pipeline_end_time - pipeline_start_time:.2f} seconds.")
        clean_memory()


# --- Pipeline 3: Topic Modeling & NLP Pipeline ---

class TopicModelingNLP:
    """
    Identifies prevalent topics in code descriptions and forum discussions.
    """
    def __init__(self, root_path_csv, root_path_code):
        self.root_path_csv = root_path_csv
        self.root_path_code = root_path_code
        self.text_data = [] # List to hold all text documents
        self.topic_model = None

    def extract_markdown_from_ipynb(self, filepath):
        """Extracts markdown content from a .ipynb file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            markdown_content = "\n".join([cell['source'] for cell in nb.cells if cell.cell_type == 'markdown'])
            del nb # Delete notebook object after extraction
            clean_memory()
            return markdown_content
        except Exception as e:
            # print(f"Error extracting markdown from {filepath}: {e}")
            return ""

    def load_data_and_extract_text(self):
        """Loads data and extracts text for topic modeling."""
        load_start_time = time.time()
        print("Loading data and extracting text for Topic Modeling...")
        
        # Competitions descriptions
        try:
            competitions_df = pd.read_csv(os.path.join(self.root_path_csv, 'Competitions.csv'), low_memory=False)
            if 'Description' in competitions_df.columns:
                self.text_data.extend(competitions_df['Description'].dropna().tolist())
                print(f"Extracted {len(competitions_df['Description'].dropna())} competition descriptions.")
            else:
                print("Warning: 'Description' column not found in Competitions.csv. Skipping competition description extraction.")
            del competitions_df # Delete DataFrame after use
            clean_memory()
        except FileNotFoundError:
            print("Competitions.csv not found, skipping competition description extraction.")

        # Datasets descriptions
        try:
            datasets_df = pd.read_csv(os.path.join(self.root_path_csv, 'Datasets.csv'), low_memory=False)
            if 'Description' in datasets_df.columns:
                self.text_data.extend(datasets_df['Description'].dropna().tolist())
                print(f"Extracted {len(datasets_df['Description'].dropna())} dataset descriptions.")
            else:
                print("Warning: 'Description' column not found in Datasets.csv. Skipping dataset description extraction.")
            del datasets_df # Delete DataFrame after use
            clean_memory()
        except FileNotFoundError:
            print("Datasets.csv not found, skipping dataset description extraction.")

        # Kernel descriptions (from notebook_stats.parquet if available, or re-extract)
        # For this pipeline, we'll extract markdown directly from selected notebooks
        print("Extracting markdown content from selected notebooks...")
        ipynb_files = get_file_paths(self.root_path_code, ['.ipynb'], num_files_per_ext=20)
        markdown_contents = [self.extract_markdown_from_ipynb(f) for f in tqdm(ipynb_files, desc="Extracting markdown")]
        self.text_data.extend([md for md in markdown_contents if md.strip()])
        del ipynb_files
        del markdown_contents
        clean_memory()

        print(f"Total text documents for topic modeling: {len(self.text_data)}")
        load_end_time = time.time()
        print(f"Data loading and text extraction completed in {load_end_time - load_start_time:.2f} seconds.")
        clean_memory()

    def clean_text_data(self):
        """Cleans and preprocesses text data."""
        clean_start_time = time.time()
        print("Cleaning text data...")
        cleaned_texts = []
        for text in tqdm(self.text_data, desc="Cleaning text"):
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special characters
            text = text.lower() # Lowercase
            cleaned_texts.append(text)
        self.text_data = cleaned_texts
        del cleaned_texts # Delete temporary list
        clean_memory()

        # Basic warning if spaCy is not used (for better lemmatization/stopwords)
        try:
            import spacy
            # spacy.load('en_core_web_sm') # This line can cause memory issues if model not present
        except ImportError:
            print("SpaCy not used for lemmatization/stopwords. Install 'en_core_web_sm' for better results.")
        
        clean_end_time = time.time()
        print(f"Text cleaning completed in {clean_end_time - clean_start_time:.2f} seconds.")
        clean_memory()

    def apply_bertopic(self):
        """Applies BERTopic for topic modeling."""
        bertopic_start_time = time.time()
        print("Applying BERTopic for topic modeling...")
        
        # BERTopic model initialization and fitting
        # Using a smaller model for embeddings to reduce memory footprint
        # 'all-MiniLM-L6-v2' is a good balance of performance and size
        self.topic_model = BERTopic(
            language="english",
            calculate_probabilities=False, # Reduce memory by not calculating probabilities
            verbose=True,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Fit transform will generate embeddings and clusters
        topics, probs = self.topic_model.fit_transform(self.text_data)

        # After fit_transform, embeddings are stored within the model if not explicitly cleared.
        # If you need to reduce memory further and don't need embeddings after this step,
        # you might try to clear them from the model, but it's not straightforward.
        # For now, rely on the model's internal memory management.

        # Get topic information and save
        topic_info = self.topic_model.get_topic_info()
        topic_info.to_csv('topic_terms.csv', index=False)
        print("Topic terms saved to topic_terms.csv")

        # Print overall topic distribution
        print("\nOverall Topic Distribution:")
        print(pd.Series(topics).value_counts())

        # New: Plot Top Topics Distribution
        if not topic_info.empty:
            # Exclude the -1 topic (noise) for plotting if it exists and is dominant
            plot_data = topic_info[topic_info['Topic'] != -1].head(10)
            if not plot_data.empty:
                plt.figure(figsize=(12, 7))
                sns.barplot(x='Name', y='Count', data=plot_data)
                plt.title('Top 10 Topics Distribution')
                plt.xlabel('Topic Name')
                plt.ylabel('Number of Documents')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
                plt.savefig('top_topics_distribution.png')
                plt.close()
                del plot_data
                print("Generated: top_topics_distribution.png")
            else:
                print("No meaningful topics found for plotting (only noise topic or empty).")
        else:
            print("Topic info is empty, skipping top topics distribution plot.")
        clean_memory()


        # Delete topics and probs if not needed later
        del topics
        del probs
        clean_memory()

        bertopic_end_time = time.time()
        print(f"BERTopic modeling completed in {bertopic_end_time - bertopic_start_time:.2f} seconds.")
        clean_memory()

    def run_pipeline(self):
        """Runs the topic modeling and NLP pipeline."""
        pipeline_start_time = time.time()
        print("Starting Topic Modeling & NLP Pipeline...")
        self.load_data_and_extract_text()
        self.clean_text_data()
        
        # Only proceed with BERTopic if there's text data
        if self.text_data:
            self.apply_bertopic()
        else:
            print("No text data found for topic modeling. Skipping BERTopic.")

        # Clear the text data after topic modeling is complete
        del self.text_data
        self.text_data = []
        clean_memory()

        print("Topic Modeling & NLP Pipeline Completed. (Time-series of topic weights not fully implemented without date association)")
        pipeline_end_time = time.time()
        print(f"Topic Modeling & NLP Pipeline Completed in {pipeline_end_time - pipeline_start_time:.2f} seconds.")
        clean_memory()


# --- Pipeline 4: Performance Benchmarking Pipeline ---

class PerformanceBenchmarking:
    """
    Analyzes kernel performance metrics and generates leaderboards.
    """
    def __init__(self, root_path_csv):
        self.root_path_csv = root_path_csv
        self.kernel_versions_df = None
        self.kernels_df = None
        self.users_df = None

    def load_data(self):
        """Loads necessary CSV data."""
        load_start_time = time.time()
        print("Loading data for Performance Benchmarking...")
        try:
            self.kernel_versions_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelVersions.csv'), low_memory=False)
            self.kernels_df = pd.read_csv(os.path.join(self.root_path_csv, 'Kernels.csv'), low_memory=False)
            self.users_df = pd.read_csv(os.path.join(self.root_path_csv, 'Users.csv'), low_memory=False)

            # --- Start: Added robust column checks and initialization ---
            # Check for 'KernelId' in kernel_versions_df
            if 'KernelId' not in self.kernel_versions_df.columns:
                print("Warning: 'KernelId' column not found in KernelVersions.csv. Performance metrics might be incomplete.")
                self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId', 'CreationDate', 'ScriptDurationSeconds', 'Size'])
            
            # Check for 'Id' and 'AuthorUserId' in kernels_df
            if 'Id' not in self.kernels_df.columns or 'AuthorUserId' not in self.kernels_df.columns:
                print("Warning: 'Id' or 'AuthorUserId' column not found in Kernels.csv. Performance metrics might be incomplete.")
                self.kernels_df = pd.DataFrame(columns=['Id', 'AuthorUserId', 'ForumTopicId', 'CurrentKernelVersionId', 'CreationDate'])
            
            # Check for 'Id' and 'DisplayName' in users_df
            if 'Id' not in self.users_df.columns or 'DisplayName' not in self.users_df.columns:
                print("Warning: 'Id' or 'DisplayName' column not found in Users.csv. User display names might be incomplete.")
                self.users_df = pd.DataFrame(columns=['Id', 'DisplayName'])
            # --- End: Added robust column checks and initialization ---

            print("Parsing date columns...")
            # Ensure columns exist before parsing dates or dropping NaNs
            if 'CreationDate' in self.kernel_versions_df.columns:
                self.kernel_versions_df['CreationDate'] = pd.to_datetime(self.kernel_versions_df['CreationDate'], errors='coerce')
                self.kernel_versions_df.dropna(subset=['CreationDate'], inplace=True)
            else:
                print("Warning: 'CreationDate' column not found in KernelVersions.csv. Skipping date parsing for kernel versions.")
                # Ensure it's a DataFrame even if empty
                if self.kernel_versions_df.empty:
                    self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId', 'CreationDate', 'ScriptDurationSeconds', 'Size'])

            if 'CreationDate' in self.kernels_df.columns:
                self.kernels_df['CreationDate'] = pd.to_datetime(self.kernels_df['CreationDate'], errors='coerce')
                self.kernels_df.dropna(subset=['CreationDate'], inplace=True)
            else:
                print("Warning: 'CreationDate' column not found in Kernels.csv. Skipping date parsing for kernels.")
                # Ensure it's a DataFrame even if empty
                if self.kernels_df.empty:
                    self.kernels_df = pd.DataFrame(columns=['Id', 'AuthorUserId', 'ForumTopicId', 'CurrentKernelVersionId', 'CreationDate'])

            print("Data loaded.")
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.root_path_csv}")
            # Initialize empty DataFrames to prevent further errors if a file is missing
            self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId', 'CreationDate', 'ScriptDurationSeconds', 'Size'])
            self.kernels_df = pd.DataFrame(columns=['Id', 'AuthorUserId', 'ForumTopicId', 'CurrentKernelVersionId', 'CreationDate'])
            self.users_df = pd.DataFrame(columns=['Id', 'DisplayName'])
        
        clean_memory()
        load_end_time = time.time()
        print(f"Data loading for Performance Benchmarking completed in {load_end_time - load_start_time:.2f} seconds.")

    def perform_eda_plots(self):
        """Performs exploratory data analysis and generates plots."""
        eda_start_time = time.time()
        print("Performing EDA plots for Performance Benchmarking...")

        # Check if dataframes are loaded and not empty
        if self.kernel_versions_df.empty or self.kernels_df.empty or self.users_df.empty:
            print("Insufficient data for EDA plots. Skipping.")
            return

        # Merge for combined analysis (only if data is available and columns exist)
        temp_merged_df = None
        if 'KernelId' in self.kernel_versions_df.columns and 'Id' in self.kernels_df.columns:
            temp_merged_df = pd.merge(self.kernel_versions_df, self.kernels_df[['Id', 'AuthorUserId', 'CreationDate']],
                                      left_on='KernelId', right_on='Id', how='inner', suffixes=('_version', '_kernel'))
            # Drop the duplicate 'Id' column from kernels_df after merge
            temp_merged_df.drop(columns=['Id_kernel'], inplace=True)
            temp_merged_df.rename(columns={'Id_version': 'Id'}, inplace=True)
            clean_memory()
        else:
            print("Missing critical columns for merging kernel versions and kernels for EDA. Skipping some plots.")


        # 1. Distribution of Script Duration
        if temp_merged_df is not None and 'ScriptDurationSeconds' in temp_merged_df.columns and not temp_merged_df['ScriptDurationSeconds'].empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(temp_merged_df['ScriptDurationSeconds'].dropna(), bins=50, kde=True)
            plt.title('Distribution of Script Duration')
            plt.xlabel('Script Duration (Seconds)')
            plt.ylabel('Count')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig('script_duration_distribution.png')
            plt.close()
            print("Generated: script_duration_distribution.png")
            clean_memory()
        else:
            print("Skipping Script Duration Distribution plot: Data not available or column missing.")

        # 2. Top N Users by Number of Kernels
        if self.kernels_df is not None and 'AuthorUserId' in self.kernels_df.columns and not self.kernels_df.empty:
            top_authors = self.kernels_df['AuthorUserId'].value_counts().head(20).index.tolist()
            if not top_authors: # Check if list is empty
                print("No top authors found for plotting kernel counts.")
            else:
                top_authors_df = self.kernels_df[self.kernels_df['AuthorUserId'].isin(top_authors)].copy() # Use .copy() to avoid SettingWithCopyWarning
                
                # Merge with users_df to get DisplayName
                if self.users_df is not None and 'DisplayName' in self.users_df.columns:
                    top_authors_df = pd.merge(top_authors_df, self.users_df[['Id', 'DisplayName']],
                                              left_on='AuthorUserId', right_on='Id', how='left')
                    top_authors_df['DisplayLabel'] = top_authors_df['DisplayName'].fillna(top_authors_df['AuthorUserId'].astype(str))
                else:
                    top_authors_df['DisplayLabel'] = top_authors_df['AuthorUserId'].astype(str)

                kernel_counts_for_plot = top_authors_df['DisplayLabel'].value_counts()

                plt.figure(figsize=(12, 7))
                sns.barplot(x=kernel_counts_for_plot.index, y=kernel_counts_for_plot.values)
                plt.title('Top 20 Users by Number of Kernels')
                plt.xlabel('User')
                plt.ylabel('Number of Kernels')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
                plt.savefig('top_users_by_kernels.png')
                plt.close()
                print("Generated: top_users_by_kernels.png")
                del top_authors_df
                del kernel_counts_for_plot
                clean_memory()
        else:
            print("Skipping Top Users by Kernels plot: Data not available or column missing.")


        # 3. Average Script Duration Over Time
        if temp_merged_df is not None and 'CreationDate_kernel' in temp_merged_df.columns and 'ScriptDurationSeconds' in temp_merged_df.columns:
            temp_merged_df['CreationMonth'] = temp_merged_df['CreationDate_kernel'].dt.to_period('M')
            avg_duration_monthly = temp_merged_df.groupby('CreationMonth')['ScriptDurationSeconds'].mean().reset_index()
            avg_duration_monthly['CreationMonth'] = avg_duration_monthly['CreationMonth'].dt.to_timestamp()

            plt.figure(figsize=(12, 6))
            sns.lineplot(data=avg_duration_monthly, x='CreationMonth', y='ScriptDurationSeconds')
            plt.title('Average Script Duration Over Time')
            plt.xlabel('Month')
            plt.ylabel('Average Script Duration (Seconds)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig('avg_script_duration_over_time.png')
            plt.close()
            print("Generated: avg_script_duration_over_time.png")
            del avg_duration_monthly
            clean_memory()
        else:
            print("Skipping Average Script Duration Over Time plot: Data not available or columns missing.")

        # Delete the temporary merged DataFrame
        if temp_merged_df is not None:
            del temp_merged_df
            clean_memory()

        eda_end_time = time.time()
        print(f"EDA plots completed in {eda_end_time - eda_start_time:.2f} seconds.")
        clean_memory()

    def extract_and_normalize_metrics(self):
        """Extracts and normalizes performance metrics."""
        extract_start_time = time.time()
        print("Extracting and normalizing performance metrics...")

        if self.kernel_versions_df.empty or self.kernels_df.empty:
            print("No kernel data available for performance metrics.")
            self.performance_metrics_df = pd.DataFrame()
            return

        # Merge kernel versions with kernels to get author info
        # Use a temporary DataFrame for merging to avoid modifying original large DFs directly
        temp_df = pd.merge(self.kernel_versions_df, self.kernels_df[['Id', 'AuthorUserId']], 
                           left_on='KernelId', right_on='Id', how='left', suffixes=('_version', '_kernel'))
        
        # Ensure 'ScriptDurationSeconds' exists and is numeric
        if 'ScriptDurationSeconds' in temp_df.columns:
            temp_df['ScriptDurationSeconds'] = pd.to_numeric(temp_df['ScriptDurationSeconds'], errors='coerce')
            temp_df.dropna(subset=['ScriptDurationSeconds', 'AuthorUserId'], inplace=True)

            if not temp_df.empty:
                # Normalize ScriptDurationSeconds (lower is better, so we might invert or use z-score)
                # For simplicity, let's just use the raw duration for now.
                # If normalization is desired, it would be applied here.
                # e.g., temp_df['NormalizedDuration'] = StandardScaler().fit_transform(temp_df[['ScriptDurationSeconds']])

                self.performance_metrics_df = temp_df[['AuthorUserId', 'ScriptDurationSeconds']].copy()
                # Explicitly delete temp_df after creating performance_metrics_df
                del temp_df
                clean_memory()
            else:
                print("No valid performance data after merging and cleaning.")
                self.performance_metrics_df = pd.DataFrame()
        else:
            print("Error: 'ScriptDurationSeconds' column not found in kernel_versions_df. Cannot extract performance metrics.")
            self.performance_metrics_df = pd.DataFrame()
        
        # After extraction, the original large DataFrames are no longer needed
        del self.kernel_versions_df
        del self.kernels_df
        del self.users_df
        self.kernel_versions_df = None
        self.kernels_df = None
        self.users_df = None
        clean_memory()

        extract_end_time = time.time()
        print(f"Extracting and normalizing performance metrics completed in {extract_end_time - extract_start_time:.2f} seconds.")
        clean_memory()

    def generate_leaderboards(self):
        """Generates leaderboards based on performance metrics."""
        leaderboard_start_time = time.time()
        print("Generating leaderboards...")

        if hasattr(self, 'performance_metrics_df') and not self.performance_metrics_df.empty:
            # Example: Top users by average script duration (lower is better)
            user_avg_duration = self.performance_metrics_df.groupby('AuthorUserId')['ScriptDurationSeconds'].mean().sort_values().head(10)
            print("\nTop 10 Users by Average Script Duration:")
            print(user_avg_duration)
            # You might merge with users_df to get DisplayName if users_df was kept or reloaded.
            # For aggressive memory management, we'll avoid reloading large DFs for simple prints.
            del user_avg_duration
            clean_memory()
        else:
            print("No performance metrics data to generate leaderboards.")
        
        # Clear performance_metrics_df after use
        if hasattr(self, 'performance_metrics_df'):
            del self.performance_metrics_df
            self.performance_metrics_df = None
        clean_memory()

        leaderboard_end_time = time.time()
        print(f"Leaderboards generated in {leaderboard_end_time - leaderboard_start_time:.2f} seconds.")
        clean_memory()

    def run_pipeline(self):
        """Runs the performance benchmarking pipeline."""
        pipeline_start_time = time.time()
        print("Starting Performance Benchmarking Pipeline...")
        self.load_data()
        self.perform_eda_plots() # <--- Added this line
        self.extract_and_normalize_metrics()
        self.generate_leaderboards()
        pipeline_end_time = time.time()
        print(f"Performance Benchmarking Pipeline Completed in {pipeline_end_time - pipeline_start_time:.2f} seconds.")
        clean_memory()


# --- Pipeline 5: Collaboration & Social Network Analysis Pipeline ---

class CollaborationSocialNetworkAnalysis:
    """
    Analyzes user interactions to identify key collaborators and communities.
    """
    def __init__(self, root_path_csv):
        self.root_path_csv = root_path_csv
        self.users_df = None
        self.kernel_authors_df = None
        self.kernels_df = None
        self.forum_messages_df = None
        self.forum_topics_df = None
        self.co_author_graph = None
        self.follower_graph = None
        self.forum_graph = None

    def load_data(self):
        """Loads necessary CSV data."""
        load_start_time = time.time()
        print("Loading data for Collaboration & Social Network Analysis...")
        try:
            self.users_df = pd.read_csv(os.path.join(self.root_path_csv, 'Users.csv'), low_memory=False)
            self.kernel_authors_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelAuthors.csv'), low_memory=False)
            self.kernels_df = pd.read_csv(os.path.join(self.root_path_csv, 'Kernels.csv'), low_memory=False)
            self.forum_messages_df = pd.read_csv(os.path.join(self.root_path_csv, 'ForumMessages.csv'), low_memory=False)
            self.forum_topics_df = pd.read_csv(os.path.join(self.root_path_csv, 'ForumTopics.csv'), low_memory=False)
            print("Data loaded.")
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.root_path_csv}")
            # Initialize empty DataFrames to prevent crashes
            self.users_df = pd.DataFrame(columns=['Id', 'DisplayName'])
            self.kernel_authors_df = pd.DataFrame(columns=['Id', 'KernelId', 'UserId'])
            self.kernels_df = pd.DataFrame(columns=['Id', 'AuthorUserId'])
            self.forum_messages_df = pd.DataFrame(columns=['Id', 'ForumTopicId', 'PostUserId', 'ReplyToMessageId'])
            self.forum_topics_df = pd.DataFrame(columns=['Id', 'ForumId', 'Title', 'PostUserId'])

        clean_memory()
        load_end_time = time.time()
        print(f"Data loading for Collaboration & Social Network Analysis completed in {load_end_time - load_start_time:.2f} seconds.")

    def build_user_graphs(self):
        """Builds co-author and follower graphs."""
        graph_start_time = time.time()
        print("Building user graphs...")

        # Co-author graph
        # Ensure self.kernel_authors_df is a DataFrame before calling .empty
        if self.kernel_authors_df is not None and not self.kernel_authors_df.empty:
            # Merge with Kernels to get the main author (if not already in KernelAuthors)
            # For co-authorship, we're interested in users collaborating on the same kernel.
            # Assuming KernelAuthors links users to kernels. If a kernel has multiple entries in KernelAuthors for different users, they are co-authors.
            
            # Identify kernels with multiple authors
            kernel_author_counts = self.kernel_authors_df.groupby('KernelId')['UserId'].apply(list).reset_index(name='Authors')
            multi_author_kernels = kernel_author_counts[kernel_author_counts['Authors'].apply(len) > 1]

            self.co_author_graph = nx.Graph()
            for _, row in tqdm(multi_author_kernels.iterrows(), total=len(multi_author_kernels), desc="Adding co-author edges"):
                authors = row['Authors']
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        self.co_author_graph.add_edge(authors[i], authors[j])
            
            print(f"Co-author graph built with {self.co_author_graph.number_of_nodes()} nodes and {self.co_author_graph.number_of_edges()} edges.")
            nx.write_edgelist(self.co_author_graph, 'co_author_graph.edgelist')
            print("Co-author graph saved.")
            del kernel_author_counts
            del multi_author_kernels
            clean_memory()
        else:
            print("KernelAuthors data is empty or not loaded, skipping co-author graph.")
            self.co_author_graph = nx.Graph() # Initialize empty graph

        # Follower graph (assuming we have a Followers.csv or can infer from Users/Kernels)
        # Meta Kaggle schema doesn't directly provide a Followers.csv.
        # We'll simulate a follower graph based on UserFollowingUser.csv if it exists, or skip.
        # If UserFollowingUser.csv is not available, this part will be skipped.
        try:
            user_following_df = pd.read_csv(os.path.join(self.root_path_csv, 'UserFollowingUser.csv'), low_memory=False)
            self.follower_graph = nx.DiGraph()
            for _, row in tqdm(user_following_df.iterrows(), total=len(user_following_df), desc="Adding follower edges"):
                follower_id = row['FollowingUserId']
                followed_id = row['UserId']
                self.follower_graph.add_edge(follower_id, followed_id) # Follower -> Followed
            
            print(f"Follower graph built with {self.follower_graph.number_of_nodes()} nodes and {self.follower_graph.number_of_edges()} edges.")
            nx.write_edgelist(self.follower_graph, 'follower_graph.edgelist')
            print("Follower graph saved.")
            del user_following_df
            clean_memory()
        except FileNotFoundError:
            print("UserFollowingUser.csv not found, skipping follower graph.")
            self.follower_graph = nx.DiGraph() # Initialize empty graph
        except Exception as e:
            print(f"Error building follower graph: {e}")
            self.follower_graph = nx.DiGraph() # Initialize empty graph
        
        print("User graphs built in {:.2f} seconds.".format(time.time() - graph_start_time))
        clean_memory()

    def build_forum_graph(self):
        """Builds a forum interaction graph."""
        forum_graph_start_time = time.time()
        print("Building forum graph...")
        self.forum_graph = nx.DiGraph()

        if self.forum_messages_df is not None and not self.forum_messages_df.empty:
            # Add nodes for all users who posted messages
            unique_post_users = self.forum_messages_df['PostUserId'].dropna().unique()
            self.forum_graph.add_nodes_from(unique_post_users)

            # Add edges for replies: ReplyToMessageId -> PostUserId of the reply
            # First, map message IDs to their PostUserIds
            message_to_user = self.forum_messages_df.set_index('Id')['PostUserId'].to_dict()

            # Iterate through messages to find replies and add edges
            for _, row in tqdm(self.forum_messages_df.iterrows(), total=len(self.forum_messages_df), desc="Processing forum messages"):
                reply_to_id = row.get('ReplyToMessageId')
                post_user_id = row['PostUserId']

                if pd.notna(reply_to_id) and reply_to_id in message_to_user:
                    original_poster_id = message_to_user[reply_to_id]
                    if original_poster_id != post_user_id: # Avoid self-loops
                        self.forum_graph.add_edge(post_user_id, original_poster_id) # Replyer -> Replied_to

            print(f"Forum graph built with {self.forum_graph.number_of_nodes()} nodes and {self.forum_graph.number_of_edges()} edges.")
            nx.write_edgelist(self.forum_graph, 'forum_graph.edgelist')
            print("Forum graph saved.")
            del message_to_user
            clean_memory()
        else:
            print("ForumMessages data is empty or not loaded, skipping forum graph.")
            self.forum_graph = nx.DiGraph() # Initialize empty graph

        print("Forum graph built in {:.2f} seconds.".format(time.time() - forum_graph_start_time))
        clean_memory()

    def perform_eda_plots_social_network(self):
        """Performs EDA plots specific to social network analysis."""
        eda_start_time = time.time()
        print("Performing EDA plots for Social Network Analysis...")

        # 1. Co-author Graph Degree Distribution
        if self.co_author_graph and self.co_author_graph.number_of_nodes() > 0:
            degree_sequence = sorted([d for n, d in self.co_author_graph.degree()], reverse=True)
            if degree_sequence:
                plt.figure(figsize=(10, 6))
                sns.histplot(degree_sequence, bins=20, kde=True)
                plt.title('Co-author Graph Degree Distribution')
                plt.xlabel('Degree')
                plt.ylabel('Count')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('co_author_degree_distribution.png')
                plt.close()
                del degree_sequence
                print("Generated: co_author_degree_distribution.png")
            else:
                print("Co-author graph has no degrees to plot.")
            clean_memory()
        else:
            print("Skipping Co-author Graph Degree Distribution plot: Graph not available or empty.")

        # 2. Follower Graph In-Degree Distribution
        if self.follower_graph and self.follower_graph.number_of_nodes() > 0:
            in_degree_sequence = sorted([d for n, d in self.follower_graph.in_degree()], reverse=True)
            if in_degree_sequence:
                plt.figure(figsize=(10, 6))
                sns.histplot(in_degree_sequence, bins=20, kde=True)
                plt.title('Follower Graph In-Degree Distribution')
                plt.xlabel('In-Degree')
                plt.ylabel('Count')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('follower_in_degree_distribution.png')
                plt.close()
                del in_degree_sequence
                print("Generated: follower_in_degree_distribution.png")
            else:
                print("Follower graph has no in-degrees to plot.")
            clean_memory()
        else:
            print("Skipping Follower Graph In-Degree Distribution plot: Graph not available or empty.")

        # 3. Follower Graph Out-Degree Distribution
        if self.follower_graph and self.follower_graph.number_of_nodes() > 0:
            out_degree_sequence = sorted([d for n, d in self.follower_graph.out_degree()], reverse=True)
            if out_degree_sequence:
                plt.figure(figsize=(10, 6))
                sns.histplot(out_degree_sequence, bins=20, kde=True)
                plt.title('Follower Graph Out-Degree Distribution')
                plt.xlabel('Out-Degree')
                plt.ylabel('Count')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('follower_out_degree_distribution.png')
                plt.close()
                del out_degree_sequence
                print("Generated: follower_out_degree_distribution.png")
            else:
                print("Follower graph has no out-degrees to plot.")
            clean_memory()
        else:
            print("Skipping Follower Graph Out-Degree Distribution plot: Graph not available or empty.")

        eda_end_time = time.time()
        print(f"EDA plots for Social Network Analysis completed in {eda_end_time - eda_start_time:.2f} seconds.")
        clean_memory()

    def compute_network_metrics(self):
        """Computes various network centrality and community detection metrics."""
        metrics_start_time = time.time()
        print("Computing network metrics...")

        # Co-author Graph Metrics
        if self.co_author_graph and self.co_author_graph.number_of_nodes() > 0:
            print("\nCo-author Graph Metrics:")
            # PageRank
            try:
                pr = nx.pagerank(self.co_author_graph)
                top_pr = sorted(pr.items(), key=lambda item: item[1], reverse=True)[:10]
                print(f"Top 10 PageRank (Co-author): {top_pr}")
                del pr
                del top_pr
                clean_memory()
            except Exception as e:
                print(f"Error computing PageRank for co-author graph: {e}")

            # Community detection (Louvain) - requires python-louvain
            try:
                import community as co
                # Louvain can be memory intensive for large graphs.
                # Only run if graph is not excessively large.
                if self.co_author_graph.number_of_nodes() < 100000: # Heuristic threshold
                    partition = co.best_partition(self.co_author_graph)
                    num_communities = len(set(partition.values()))
                    print(f"Number of communities (Louvain) in co-author graph: {num_communities}")
                    del partition
                    del num_communities
                    clean_memory()
                else:
                    print("Community detection (Louvain) skipped for co-author graph due to size.")
            except ImportError:
                print("Community detection (Louvain) skipped. Install 'python-louvain' for this feature.")
            except Exception as e:
                print(f"Error computing Louvain communities for co-author graph: {e}")
        else:
            print("Co-author graph is empty or not built, skipping metrics.")

        # Follower Graph Metrics
        if self.follower_graph and self.follower_graph.number_of_nodes() > 0:
            print("\nFollower Graph Metrics:")
            # In-Degree Centrality (for influence)
            try:
                in_degree_centrality = nx.in_degree_centrality(self.follower_graph)
                top_in_degree = sorted(in_degree_centrality.items(), key=lambda item: item[1], reverse=True)[:10]
                print(f"Top 10 In-Degree Centrality (Follower): {top_in_degree}")
                del in_degree_centrality
                del top_in_degree
                clean_memory()
            except Exception as e:
                print(f"Error computing In-Degree Centrality for follower graph: {e}")
        else:
            print("Follower graph is empty or not built, skipping metrics.")

        # Forum Graph Metrics
        if self.forum_graph and self.forum_graph.number_of_nodes() > 0:
            print("\nForum Graph Metrics:")
            # PageRank
            try:
                pr_forum = nx.pagerank(self.forum_graph)
                top_pr_forum = sorted(pr_forum.items(), key=lambda item: item[1], reverse=True)[:10]
                print(f"Top 10 PageRank (Forum): {top_pr_forum}")
                del pr_forum
                del top_pr_forum
                clean_memory()
            except Exception as e:
                print(f"Error computing PageRank for forum graph: {e}")
        else:
            print("Forum graph is empty or not built, skipping metrics.")

        print("\nReports on 'most central' Kaggle contributors would be generated here, combining various metrics.")
        metrics_end_time = time.time()
        print(f"Network metrics computation completed in {metrics_end_time - metrics_start_time:.2f} seconds.")
        
        # Clear graph objects after metrics computation
        del self.co_author_graph
        del self.follower_graph
        del self.forum_graph
        self.co_author_graph = None
        self.follower_graph = None
        self.forum_graph = None
        clean_memory()

    def run_pipeline(self):
        """Runs the collaboration and social network analysis pipeline."""
        pipeline_start_time = time.time()
        print("Starting Collaboration & Social Network Analysis Pipeline...")
        self.load_data()
        
        # --- Moved del statements here ---
        # Clear raw dataframes after they've been used to build graphs
        # This is critical for memory management, as graphs are built iteratively
        # and the raw dataframes might be very large.
        self.build_user_graphs()
        self.build_forum_graph()

        del self.users_df
        del self.kernel_authors_df
        del self.kernels_df
        del self.forum_messages_df
        del self.forum_topics_df
        self.users_df = None
        self.kernel_authors_df = None
        self.kernels_df = None
        self.forum_messages_df = None
        self.forum_topics_df = None
        clean_memory()
        # --- End of moved del statements ---

        self.perform_eda_plots_social_network() # New EDA plots for social network
        self.compute_network_metrics()
        pipeline_end_time = time.time()
        print(f"Collaboration & Social Network Analysis Pipeline Completed in {pipeline_end_time - pipeline_start_time:.2f} seconds.")
        clean_memory()


# --- Pipeline 6: Evolutionary Pathways Pipeline ---

class EvolutionaryPathways:
    """
    Traces the evolution of code, ideas, and solutions through versioning.
    """
    def __init__(self, root_path_csv):
        self.root_path_csv = root_path_csv
        self.kernels_df = None
        self.kernel_versions_df = None
        self.kernel_languages_df = None
        self.kernel_tags_df = None
        self.tags_df = None

    def load_data(self):
        """Loads necessary CSV data."""
        load_start_time = time.time()
        print("Loading data for Evolutionary Pathways...")
        try:
            self.kernels_df = pd.read_csv(os.path.join(self.root_path_csv, 'Kernels.csv'), low_memory=False)
            self.kernel_versions_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelVersions.csv'), low_memory=False)
            self.kernel_languages_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelLanguages.csv'), low_memory=False)
            self.kernel_tags_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelTags.csv'), low_memory=False)
            self.tags_df = pd.read_csv(os.path.join(self.root_path_csv, 'Tags.csv'), low_memory=False)

            # --- Start: Added robust column checks and initialization for Evolutionary Pathways ---
            if not all(col in self.kernels_df.columns for col in ['Id', 'AuthorUserId', 'CreationDate']):
                print("Warning: Missing critical columns in Kernels.csv for Evolutionary Pathways. Initializing empty DataFrame.")
                self.kernels_df = pd.DataFrame(columns=['Id', 'AuthorUserId', 'CreationDate'])

            if not all(col in self.kernel_versions_df.columns for col in ['Id', 'KernelId', 'CreationDate']):
                print("Warning: Missing critical columns in KernelVersions.csv for Evolutionary Pathways. Initializing empty DataFrame.")
                self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId', 'CreationDate', 'ScriptDurationSeconds'])
            
            if not all(col in self.kernel_languages_df.columns for col in ['Id', 'KernelId', 'LanguageId']):
                print("Warning: Missing critical columns in KernelLanguages.csv for Evolutionary Pathways. Initializing empty DataFrame.")
                self.kernel_languages_df = pd.DataFrame(columns=['Id', 'KernelId', 'LanguageId'])

            if not all(col in self.kernel_tags_df.columns for col in ['Id', 'KernelVersionId', 'TagId']):
                print("Warning: Missing critical columns in KernelTags.csv for Evolutionary Pathways. Initializing empty DataFrame.")
                self.kernel_tags_df = pd.DataFrame(columns=['Id', 'KernelVersionId', 'TagId'])

            if not all(col in self.tags_df.columns for col in ['Id', 'Name']):
                print("Warning: Missing critical columns in Tags.csv for Evolutionary Pathways. Initializing empty DataFrame.")
                self.tags_df = pd.DataFrame(columns=['Id', 'Name'])
            # --- End: Added robust column checks and initialization for Evolutionary Pathways ---

            # Convert date columns (only if they exist after checks)
            if 'CreationDate' in self.kernels_df.columns:
                self.kernels_df['CreationDate'] = pd.to_datetime(self.kernels_df['CreationDate'], errors='coerce')
                self.kernels_df.dropna(subset=['CreationDate'], inplace=True)
            
            if 'CreationDate' in self.kernel_versions_df.columns:
                self.kernel_versions_df['CreationDate'] = pd.to_datetime(self.kernel_versions_df['CreationDate'], errors='coerce')
                self.kernel_versions_df.dropna(subset=['CreationDate'], inplace=True)
            
            print("Data loaded.")
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.root_path_csv}")
            # Initialize empty DataFrames to prevent further errors if a file is missing
            self.kernels_df = pd.DataFrame(columns=['Id', 'AuthorUserId', 'CreationDate'])
            self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId', 'CreationDate', 'ScriptDurationSeconds'])
            self.kernel_languages_df = pd.DataFrame(columns=['Id', 'KernelId', 'LanguageId'])
            self.kernel_tags_df = pd.DataFrame(columns=['Id', 'KernelVersionId', 'TagId'])
            self.tags_df = pd.DataFrame(columns=['Id', 'Name'])
        
        clean_memory()
        load_end_time = time.time()
        print(f"Data loading for Evolutionary Pathways completed in {load_end_time - load_start_time:.2f} seconds.")

    def construct_tripartite_graph(self):
        """
        Constructs a tripartite graph (Kernel-Version-Tag) to show evolution.
        This graph can be very large.
        """
        graph_start_time = time.time()
        print("Constructing tripartite graph...")
        self.evolution_graph = nx.DiGraph() # Directed graph for evolution

        if self.kernels_df.empty or self.kernel_versions_df.empty or self.kernel_tags_df.empty or self.tags_df.empty:
            print("Insufficient data to construct tripartite graph.")
            return

        # Add kernel nodes
        self.evolution_graph.add_nodes_from(self.kernels_df['Id'].tolist(), node_type='kernel')
        print(f"Adding kernel nodes: {len(self.kernels_df['Id'])} nodes added.")
        clean_memory()

        # Add kernel version nodes and edges from kernel to version
        for _, row in tqdm(self.kernel_versions_df.iterrows(), total=len(self.kernel_versions_df), desc="Adding kernel version nodes and edges"):
            kernel_id = row['KernelId'] # This is where the KeyError was occurring
            version_id = row['Id']
            self.evolution_graph.add_node(version_id, node_type='version', creation_date=row['CreationDate'])
            self.evolution_graph.add_edge(kernel_id, version_id, edge_type='has_version')
        print(f"Added {len(self.kernel_versions_df)} version nodes and edges.")
        clean_memory()

        # Add tag nodes
        # Ensure 'Id' and 'Name' exist in tags_df before using
        if 'Id' in self.tags_df.columns and 'Name' in self.tags_df.columns:
            tag_id_to_name = self.tags_df.set_index('Id')['Name'].to_dict()
            self.evolution_graph.add_nodes_from(self.tags_df['Id'].tolist(), node_type='tag')
            print(f"Added {len(self.tags_df)} tag nodes.")
        else:
            print("Skipping tag node addition: 'Id' or 'Name' column missing in tags_df.")
        clean_memory()

        # Add edges from version to tags
        for _, row in tqdm(self.kernel_tags_df.iterrows(), total=len(self.kernel_tags_df), desc="Adding version-tag edges"):
            version_id = row['KernelVersionId']
            tag_id = row['TagId']
            if version_id in self.evolution_graph and tag_id in self.evolution_graph:
                self.evolution_graph.add_edge(version_id, tag_id, edge_type='has_tag')
        print(f"Added {len(self.kernel_tags_df)} version-tag edges.")
        clean_memory()

        print(f"Tripartite graph built with {self.evolution_graph.number_of_nodes()} nodes and {self.evolution_graph.number_of_edges()} edges.")
        nx.write_edgelist(self.evolution_graph, 'evolution_graph.edgelist')
        print("Evolutionary graph saved.")

        # Clear raw dataframes after graph construction
        del self.kernels_df
        del self.kernel_versions_df
        del self.kernel_languages_df
        del self.kernel_tags_df
        del self.tags_df
        self.kernels_df = None
        self.kernel_versions_df = None
        self.kernel_languages_df = None
        self.kernel_tags_df = None
        self.tags_df = None
        clean_memory()

        graph_end_time = time.time()
        print(f"Tripartite graph construction completed in {graph_end_time - graph_start_time:.2f} seconds.")
        clean_memory()

    def analyze_evolutionary_paths(self):
        """Analyzes evolutionary paths within the graph."""
        analysis_start_time = time.time()
        print("Analyzing evolutionary paths (simplified)...")

        if not hasattr(self, 'evolution_graph') or not self.evolution_graph:
            print("Evolutionary graph not available for analysis.")
            return

        # Example: Find paths from a kernel to its latest tags through versions
        # This is a placeholder as finding "latest" requires more complex logic
        # involving sorting versions by date and traversing.
        
        # For a simplified approach, let's just count nodes/edges
        print(f"Evolutionary Graph Nodes: {self.evolution_graph.number_of_nodes()}")
        print(f"Evolutionary Graph Edges: {self.evolution_graph.number_of_edges()}")

        # Placeholder for more advanced analysis:
        # - Identify common tag progressions
        # - Analyze how long it takes for new tags to appear on a kernel
        # - Visualize specific kernel evolution paths

        print("Evolutionary path analysis (simplified) completed.")
        analysis_end_time = time.time()
        print(f"Evolutionary path analysis completed in {analysis_end_time - analysis_start_time:.2f} seconds.")
        
        # Clear the graph after analysis if it's not needed by subsequent steps
        del self.evolution_graph
        self.evolution_graph = None
        clean_memory()

    def perform_eda_plots_evolutionary_pathways(self):
        """Performs EDA plots specific to evolutionary pathways analysis."""
        eda_start_time = time.time()
        print("Performing EDA plots for Evolutionary Pathways Analysis...")

        # Reload necessary data for plotting if it was deleted
        # For this, we'll assume the graphs are built and saved, and we can load relevant info
        # Or, if the dataframes were just cleared, we might need to re-load small subsets
        # For simplicity, let's re-load if needed, but only the necessary columns.
        
        # 1. Distribution of Kernel Versions per Kernel
        try:
            kernels_df_temp = pd.read_csv(os.path.join(self.root_path_csv, 'Kernels.csv'), usecols=['Id'], low_memory=False)
            kernel_versions_df_temp = pd.read_csv(os.path.join(self.root_path_csv, 'KernelVersions.csv'), usecols=['KernelId'], low_memory=False)
            
            # Ensure 'KernelId' is present before value_counts
            if 'KernelId' in kernel_versions_df_temp.columns:
                kernel_version_counts = kernel_versions_df_temp['KernelId'].value_counts().reset_index()
                kernel_version_counts.columns = ['KernelId', 'VersionCount']

                plt.figure(figsize=(10, 6))
                sns.histplot(kernel_version_counts['VersionCount'], bins=20, kde=True)
                plt.title('Distribution of Kernel Versions per Kernel')
                plt.xlabel('Number of Versions')
                plt.ylabel('Number of Kernels')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('kernel_versions_distribution.png')
                plt.close()
                del kernels_df_temp, kernel_versions_df_temp, kernel_version_counts
                print("Generated: kernel_versions_distribution.png")
            else:
                print("Skipping Kernel Versions Distribution plot: 'KernelId' column not found in KernelVersions.csv.")
            clean_memory()
        except FileNotFoundError:
            print("Skipping Kernel Versions Distribution plot: Required CSVs not found.")
        except Exception as e:
            print(f"Error generating kernel versions distribution plot: {e}")
            plt.close()
            clean_memory()

        # 2. Top 10 Most Common Tags
        try:
            kernel_tags_df_temp = pd.read_csv(os.path.join(self.root_path_csv, 'KernelTags.csv'), low_memory=False)
            tags_df_temp = pd.read_csv(os.path.join(self.root_path_csv, 'Tags.csv'), low_memory=False)

            if not kernel_tags_df_temp.empty and not tags_df_temp.empty and 'TagId' in kernel_tags_df_temp.columns and 'Id' in tags_df_temp.columns and 'Name' in tags_df_temp.columns:
                merged_tags_df = pd.merge(kernel_tags_df_temp, tags_df_temp[['Id', 'Name']], left_on='TagId', right_on='Id', how='inner')
                top_tags = merged_tags_df['Name'].value_counts().head(10)

                plt.figure(figsize=(12, 7))
                sns.barplot(x=top_tags.index, y=top_tags.values)
                plt.title('Top 10 Most Common Kernel Tags')
                plt.xlabel('Tag Name')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
                plt.savefig('top_kernel_tags.png')
                plt.close()
                del kernel_tags_df_temp, tags_df_temp, merged_tags_df, top_tags
                print("Generated: top_kernel_tags.png")
            else:
                print("Insufficient data or missing critical columns for Top 10 Most Common Tags plot.")
            clean_memory()
        except FileNotFoundError:
            print("Skipping Top 10 Most Common Tags plot: Required CSVs not found.")
        except Exception as e:
            print(f"Error generating top kernel tags plot: {e}")
            plt.close()
            clean_memory()

        eda_end_time = time.time()
        print(f"EDA plots for Evolutionary Pathways Analysis completed in {eda_end_time - eda_start_time:.2f} seconds.")
        clean_memory()


    def run_pipeline(self):
        """Runs the evolutionary pathways pipeline."""
        pipeline_start_time = time.time()
        print("Starting Evolutionary Pathways Pipeline...")
        self.load_data()
        self.construct_tripartite_graph()
        self.analyze_evolutionary_paths()
        self.perform_eda_plots_evolutionary_pathways() # New EDA plots for evolutionary pathways
        pipeline_end_time = time.time()
        print(f"Evolutionary Pathways Pipeline Completed in {pipeline_end_time - pipeline_start_time:.2f} seconds.")
        clean_memory()


# --- Pipeline 7: Comprehensive ML Pipeline ---

class ComprehensiveMLPipeline:
    """
    Builds and evaluates machine learning models for a hypothetical classification task.
    """
    def __init__(self, root_path_csv):
        self.root_path_csv = root_path_csv
        self.data_df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_and_preprocess_data(self):
        """
        Loads data, performs feature engineering and preprocessing.
        Using a smaller dataset for ML to avoid excessive memory usage.
        For demonstration, we'll use a synthetic or small real dataset.
        If using Meta Kaggle data, select a small, relevant subset.
        Let's try to use a subset of Kernels.csv and Users.csv to create a classification task.
        Hypothetical task: Predict if a kernel author is 'active' (e.g., has > N kernels).
        """
        step_start_time = time.time()
        print("1. Loading and preprocessing data for ML pipeline...")
        try:
            kernels_df = pd.read_csv(os.path.join(self.root_path_csv, 'Kernels.csv'), low_memory=False)
            users_df = pd.read_csv(os.path.join(self.root_path_csv, 'Users.csv'), low_memory=False)

            # Reduce the size of DataFrames for memory efficiency
            # Take a sample if they are too large
            if len(kernels_df) > 100000:
                kernels_df = kernels_df.sample(n=100000, random_state=42).copy()
            if len(users_df) > 50000:
                users_df = users_df.sample(n=50000, random_state=42).copy()

            # Feature Engineering: Count kernels per user
            user_kernel_counts = kernels_df.groupby('AuthorUserId').size().reset_index(name='kernel_count')
            
            # Define 'active' users (e.g., more than 5 kernels)
            ACTIVE_THRESHOLD = 5
            user_kernel_counts['is_active'] = (user_kernel_counts['kernel_count'] > ACTIVE_THRESHOLD).astype(int)

            # Merge with users_df to get user features (e.g., public activities count)
            # For simplicity, let's just use kernel_count as a feature and is_active as target.
            # In a real scenario, you'd extract more features from users_df or other sources.
            self.data_df = user_kernel_counts.copy()
            
            # Explicitly delete source DataFrames after feature engineering
            del kernels_df
            del users_df
            del user_kernel_counts
            clean_memory()

            # Prepare features (X) and target (y)
            X = self.data_df[['kernel_count']]
            y = self.data_df['is_active']

            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Explicitly delete X, y after splitting
            del X
            del y
            clean_memory()

            # Scale features
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            del scaler # Delete scaler after use
            clean_memory()

            print(f"Data loaded and preprocessed. Train samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")

        except FileNotFoundError as e:
            print(f"Error loading CSV file for ML: {e}. Please ensure Kernels.csv and Users.csv are in {self.root_path_csv}")
            # Create dummy dataframes to prevent crashes
            self.X_train, self.X_test, self.y_train, self.y_test = np.array([[0]]), np.array([[0]]), np.array([0]), np.array([0])
        except Exception as e:
            print(f"An error occurred during data loading and preprocessing: {e}")
            import traceback
            traceback.print_exc()
            self.X_train, self.X_test, self.y_train, self.y_test = np.array([[0]]), np.array([[0]]), np.array([0]), np.array([0])
        
        # Clear the original data_df after splitting and scaling
        del self.data_df
        self.data_df = None
        clean_memory()

        step_end_time = time.time()
        print(f"Step 1 (Data Preprocessing) completed in {step_end_time - step_start_time:.2f} seconds.")

    def build_and_train_models(self):
        """
        Builds and trains various machine learning models.
        Returns the best performing model.
        """
        step_start_time = time.time()
        print("\n2. Building and training models...")

        models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10), # Reduced n_estimators/max_depth
            'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=50, max_depth=5), # Reduced n_estimators/max_depth
            'MLPClassifier': MLPClassifier(random_state=42, max_iter=100, hidden_layer_sizes=(50, 20)), # Reduced max_iter/hidden_layer_sizes
            'LightGBM': lgb.LGBMClassifier(random_state=42, n_estimators=50, num_leaves=20), # Reduced n_estimators/num_leaves
            'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=50, max_depth=5, use_label_encoder=False, eval_metric='logloss'), # Reduced n_estimators/max_depth
            # Keras model is more complex, keep it simple for now or skip if memory is critical
            # 'Keras_CNN': self._build_keras_cnn(self.X_train.shape[1]) 
        }

        best_model = None
        best_accuracy = -1
        
        # Convert X_train, y_train to numpy arrays if they are not already
        if isinstance(self.X_train, pd.DataFrame):
            self.X_train = self.X_train.values
        if isinstance(self.y_train, pd.Series):
            self.y_train = self.y_train.values

        for name, model in models.items():
            model_start_time = time.time()
            print(f"  Training {name}...")
            try:
                if name == 'Keras_CNN':
                    # Keras model requires specific input shape
                    # For a simple 1D feature, we need to reshape X_train
                    # This part is commented out due to potential complexity and memory if not carefully managed
                    # input_shape = (self.X_train.shape[1], 1) # For Conv1D
                    # model.fit(self.X_train.reshape(-1, self.X_train.shape[1], 1), self.y_train, 
                    #           epochs=10, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=3)])
                    pass # Skip Keras for now to simplify and save memory
                else:
                    model.fit(self.X_train, self.y_train)
                
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                print(f"  {name} Accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                
                # Explicitly delete trained model if it's not the best one
                if model != best_model:
                    del model
                clean_memory() # Clean memory after each model training

            except Exception as e:
                print(f"  Error training {name}: {e}")
                import traceback
                traceback.print_exc()
            model_end_time = time.time()
            print(f"  {name} training completed in {model_end_time - model_start_time:.2f} seconds.")
        
        step_end_time = time.time()
        print(f"Step 2 (Model Training) completed in {step_end_time - step_start_time:.2f} seconds.")
        return best_model

    def _build_keras_cnn(self, input_dim):
        """Builds a simple Keras CNN model for 1D input."""
        # This function is kept for reference but not called in the main pipeline
        # to manage memory more aggressively.
        input_layer = Input(shape=(input_dim, 1))
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
        pool1 = GlobalMaxPooling1D()(conv1)
        dense1 = Dense(10, activation='relu')(pool1)
        output_layer = Dense(1, activation='sigmoid')(dense1) # Binary classification

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def evaluate_models(self, best_model):
        """Evaluates the best model."""
        step_start_time = time.time()
        print("\n3. Evaluating the best model...")
        if best_model is None:
            print("No best model to evaluate.")
            step_end_time = time.time()
            print(f"Step 3 (Model Evaluation) completed in {step_end_time - step_start_time:.2f} seconds.")
            return

        # Convert X_test, y_test to numpy arrays if they are not already
        if isinstance(self.X_test, pd.DataFrame):
            self.X_test = self.X_test.values
        if isinstance(self.y_test, pd.Series):
            self.y_test = self.y_test.values

        try:
            if isinstance(best_model, tf.keras.Model):
                # Keras model evaluation
                # Reshape if necessary for Conv1D
                y_pred_proba = best_model.predict(self.X_test.reshape(-1, self.X_test.shape[1], 1)).ravel()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = best_model.predict(self.X_test)
                if hasattr(best_model, 'predict_proba'):
                    y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
                else:
                    y_pred_proba = None

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc}")

            # Plot ROC Curve if y_pred_proba is available
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('roc_curve.png')
                plt.close() # Close plot to free memory
                del fpr, tpr, _ # Delete temporary variables
                clean_memory()

        except Exception as e:
            print(f"An error occurred during model evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear test data after evaluation
        del self.X_test
        del self.y_test
        self.X_test = None
        self.y_test = None
        clean_memory()

        step_end_time = time.time()
        print(f"Step 3 (Model Evaluation) completed in {step_end_time - step_start_time:.2f} seconds.")

    def run_pipeline(self):
        """Orchestrates the comprehensive ML pipeline."""
        ml_overall_start_time = time.time()
        print("\n--- Starting Comprehensive ML Pipeline ---")
        try:
            # 1. Load and Preprocess Data
            self.load_and_preprocess_data()
            
            # Clear training data after models are trained if not needed for evaluation
            # (though typically X_train/y_train are needed for retraining or cross-validation)
            # For this simplified flow, we train and then evaluate on X_test/y_test.
            # So, X_train/y_train can be cleared after build_and_train_models.

            # 2. Build and Train Models
            best_model = self.build_and_train_models()
            
            # Clear training data after training is complete
            del self.X_train
            del self.y_train
            self.X_train = None
            self.y_train = None
            clean_memory()

            # 3. Evaluate the best model
            self.evaluate_models(best_model)

            # 4. The best model with highest accuracy should be saved
            step_start_time = time.time()
            print("\n4. Saving the best model...")
            if best_model:
                if isinstance(best_model, tf.keras.Model):
                    best_model.save('best_classification_model.keras')
                    print("Best Keras model saved as 'best_classification_model.keras'")
                else:
                    joblib.dump(best_model, 'best_classification_model.pkl')
                    print("Best scikit-learn/LightGBM/XGBoost model saved as 'best_classification_model.pkl'")
            else:
                print("No best model found to save.")
            
            # Delete the best_model object after saving
            del best_model
            clean_memory()
            step_end_time = time.time()
            print(f"Step 4 (Model Saving) completed in {step_end_time - step_start_time:.2f} seconds.")

            ml_overall_end_time = time.time()
            print(f"\n--- Comprehensive ML Pipeline Completed in {ml_overall_end_time - ml_overall_start_time:.2f} seconds ---")

        except Exception as e:
            print(f"An error occurred during the comprehensive ML pipeline: {e}")
            import traceback
            traceback.print_exc()
        finally:
            clean_memory() # Final memory clean


# --- Main entry point for execution ---
if __name__ == '__main__':
    overall_start_time = time.time()
    print("--- Starting Meta Kaggle Analysis Orchestration ---")

    # Initialize notebook_stats_df to None or an empty DataFrame
    # It will be populated by Pipeline 1 and passed to Pipeline 2
    notebook_stats_df_global = None

    try:
        # Pipeline 1: Code Metadata Extraction
        step_start_time = time.time()
        code_extractor = CodeMetadataExtractor(ROOT_PATH_CODE)
        notebook_stats_df_global = code_extractor.run_pipeline()
        del code_extractor # Delete the pipeline object after it's run
        clean_memory()
        step_end_time = time.time()
        print(f"Pipeline 1 (Code Metadata Extraction) took {step_end_time - step_start_time:.2f} seconds.")

        # Pipeline 2: Temporal Trend Analysis
        step_start_time = time.time()
        # Pass notebook_stats_df_global to TemporalTrendAnalyzer if needed, otherwise it's not used
        # In the original code, notebook_stats_df was passed but not used in TemporalTrendAnalyzer's logic.
        # So we can pass None or an empty DF if it's not truly a dependency.
        # For now, we will pass it as the original code did, but it's not directly used by its methods.
        temporal_analyzer = TemporalTrendAnalyzer(ROOT_PATH_CSV, notebook_stats_df_global)
        temporal_analyzer.run_pipeline()
        del temporal_analyzer # Delete the pipeline object after it's run
        clean_memory()
        step_end_time = time.time()
        print(f"Pipeline 2 (Temporal Trend Analysis) took {step_end_time - step_start_time:.2f} seconds.")

        # Pipeline 3: Topic Modeling & NLP
        step_start_time = time.time()
        nlp_pipeline = TopicModelingNLP(ROOT_PATH_CSV, ROOT_PATH_CODE)
        nlp_pipeline.run_pipeline()
        del nlp_pipeline # Delete the pipeline object after it's run
        clean_memory()
        step_end_time = time.time()
        print(f"Pipeline 3 (Topic Modeling & NLP) took {step_end_time - step_start_time:.2f} seconds.")

        # Pipeline 4: Performance Benchmarking
        step_start_time = time.time()
        perf_benchmarking = PerformanceBenchmarking(ROOT_PATH_CSV)
        perf_benchmarking.run_pipeline()
        del perf_benchmarking # Delete the pipeline object after it's run
        clean_memory()
        step_end_time = time.time()
        print(f"Pipeline 4 (Performance Benchmarking) took {step_end_time - step_start_time:.2f} seconds.")

        # Pipeline 5: Collaboration & Social Network Analysis
        step_start_time = time.time()
        social_network_analysis = CollaborationSocialNetworkAnalysis(ROOT_PATH_CSV)
        social_network_analysis.run_pipeline()
        del social_network_analysis # Delete the pipeline object after it's run
        clean_memory()
        step_end_time = time.time()
        print(f"Pipeline 5 (Collaboration & Social Network Analysis) took {step_end_time - step_start_time:.2f} seconds.")

        # Pipeline 6: Evolutionary Pathways
        step_start_time = time.time()
        evolutionary_pathways = EvolutionaryPathways(ROOT_PATH_CSV)
        evolutionary_pathways.run_pipeline()
        del evolutionary_pathways # Delete the pipeline object after it's run
        clean_memory()
        step_end_time = time.time()
        print(f"Pipeline 6 (Evolutionary Pathways) took {step_end_time - step_start_time:.2f} seconds.")

        # Pipeline 7: Comprehensive ML Pipeline
        step_start_time = time.time()
        ml_pipeline = ComprehensiveMLPipeline(ROOT_PATH_CSV)
        ml_pipeline.run_pipeline()
        del ml_pipeline # Delete the pipeline object after it's run
        clean_memory()
        step_end_time = time.time()
        print(f"Pipeline 7 (Comprehensive ML Pipeline) took {step_end_time - step_start_time:.2f} seconds.")

    except Exception as e:
        print(f"An error occurred during pipeline orchestration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure the global notebook_stats_df is also cleared if it's no longer needed
        if notebook_stats_df_global is not None:
            del notebook_stats_df_global
        clean_memory()
        overall_end_time = time.time()
        print(f"--- Meta Kaggle Analysis Orchestration Completed in {overall_end_time - overall_start_time:.2f} seconds ---")


import os
import nbformat
import pandas as pd
import numpy as np
import gc
import re
import ast
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import time # Import the time module for timing
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy.stats import zscore
import networkx as nx
from gensim.models import Word2Vec
from bertopic import BERTopic
import torch
from transformers import AutoTokenizer, AutoModel
import lightgbm as lgb
import xgboost as xgb
import catboost as cb # Added CatBoost
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Attention, Reshape, Permute, multiply, concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib # For saving the best model

# Define root paths for data
ROOT_PATH_CODE = "/kaggle/input/meta-kaggle-code"
ROOT_PATH_CSV = "/kaggle/input/meta-kaggle"

# --- Global Configuration ---
class Config:
    """
    Configuration class to hold global variables for the pipelines.
    Adjust these values to control memory usage and data processing scope.
    """
    NUM_FILES_PER_EXT = 2000  # Number of code files to sample per extension
    NUM_ROWS_TO_LOAD_CSVS = 100000  # Number of rows to load from large CSVs (set to None for all rows)

# --- Global Configuration and Helper Functions ---

def clean_memory():
    """Aggressively cleans memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # print("Memory cleaned.") # Suppress this frequent print for cleaner output

def get_file_paths(root_dir, extensions, num_files_per_ext=Config.NUM_FILES_PER_EXT):
    """
    Walks through the directory tree and collects a random sample of file paths
    for specified extensions. Stops scanning for an extension once num_files_per_ext
    are found. The overall scan stops when all desired counts are met.
    """
    collected_files_by_ext = {ext: [] for ext in extensions}
    
    # Check if num_files_per_ext is 0, if so, return immediately
    if num_files_per_ext == 0:
        print("NUM_FILES_PER_EXT is 0. Skipping file collection.")
        return []

    print(f"Collecting up to {num_files_per_ext} files per extension from {root_dir}...")

    # Custom tqdm for directory scanning
    dir_count = 0
    file_found_count = 0
    pbar = tqdm(desc="Scanning directories and collecting files", unit="dirs")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dir_count += 1
        pbar.update(1)

        for filename in filenames:
            for ext in extensions:
                # Only add file if we still need more for this extension
                if filename.endswith(ext) and len(collected_files_by_ext[ext]) < num_files_per_ext:
                    collected_files_by_ext[ext].append(os.path.join(dirpath, filename))
                    file_found_count += 1
                    # Update tqdm for files found
                    pbar.set_postfix_str(f"Files: {file_found_count}")

        # Check if we have collected enough files for all extensions
        # The condition is now that the length of each list meets or exceeds the target
        all_extensions_full = all(len(collected_files_by_ext[ext]) >= num_files_per_ext for ext in extensions)
        if all_extensions_full:
            pbar.close() # Close the progress bar as we are done
            print("All desired file counts met. Stopping directory scan early.")
            break # Break out of the os.walk loop

    pbar.close() # Ensure pbar is closed if loop finishes naturally

    selected_files = []
    for ext in extensions:
        # Take a random sample up to num_files_per_ext from the collected files
        if len(collected_files_by_ext[ext]) > num_files_per_ext:
            selected_files.extend(random.sample(collected_files_by_ext[ext], num_files_per_ext))
        else:
            selected_files.extend(collected_files_by_ext[ext])

    print(f"Finished collecting files. Total selected: {len(selected_files)}.")
    return selected_files

def extract_imports(code_string):
    """
    Extracts unique top-level library imports from a Python code string.
    Handles 'import x' and 'from x import y'.
    """
    imports = set()
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0]) # Get top-level module
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0]) # Get top-level module
    except SyntaxError:
        # Handle cases where code might be malformed
        pass
    return list(imports)

def get_code_lines(code_string):
    """Counts non-empty lines of code."""
    return len([line for line in code_string.split('\n') if line.strip() and not line.strip().startswith('#')])

# --- Pipeline 1: Code Metadata Extraction Pipeline ---

class CodeMetadataExtractor:
    """
    Parses code files (Jupyter Notebooks, Python, R) to extract metadata,
    library imports, and code metrics.
    """
    def __init__(self, root_path_code):
        self.root_path_code = root_path_code
        self.notebook_stats_df = pd.DataFrame()

    def extract_ipynb_metadata(self, filepath):
        """Extracts metadata from a .ipynb file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            metadata = nb.metadata.get('metadata', {})
            author = metadata.get('author', 'Unknown')
            created = metadata.get('created', '1970-01-01T00:00:00Z') # Default to epoch if not found

            code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
            markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']

            all_code = "\n".join([cell['source'] for cell in code_cells])
            libraries = extract_imports(all_code)
            total_lines_of_code = get_code_lines(all_code)

            execution_count_sum = sum(cell.get('execution_count', 0) or 0 for cell in code_cells) # Handle None

            return {
                'file_path': filepath,
                'file_type': 'ipynb',
                'author': author,
                'creation_date': pd.to_datetime(created, errors='coerce'),
                'libraries_used': libraries,
                'code_cell_count': len(code_cells),
                'markdown_cell_count': len(markdown_cells),
                'total_lines_of_code': total_lines_of_code,
                'execution_count_sum': execution_count_sum
            }
        except Exception as e:
            # print(f"Error processing {filepath}: {e}") # Suppress frequent error prints
            return None

    def extract_py_metadata(self, filepath):
        """Extracts metadata from a .py file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            libraries = extract_imports(content)
            total_lines_of_code = get_code_lines(content)

            # For .py files, author and creation date are harder to get programmatically
            # without version control or specific headers. Defaulting to placeholders.
            return {
                'file_path': filepath,
                'file_type': 'py',
                'author': 'Unknown',
                'creation_date': pd.to_datetime(os.path.getctime(filepath), unit='s', errors='coerce'), # File creation time
                'libraries_used': libraries,
                'code_cell_count': 1, # Treat as one code cell
                'markdown_cell_count': 0,
                'total_lines_of_code': total_lines_of_code,
                'execution_count_sum': 0 # No execution count for standalone .py
            }
        except Exception as e:
            # print(f"Error processing {filepath}: {e}") # Suppress frequent error prints
            return None

    def extract_r_metadata(self, filepath):
        """Extracts metadata from an .R or .Rmd file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple regex to find R library imports
            r_libraries = re.findall(r'(?:library|require)\(([\w.]+)\)', content)
            total_lines_of_code = get_code_lines(content)

            return {
                'file_path': filepath,
                'file_type': 'r',
                'author': 'Unknown',
                'creation_date': pd.to_datetime(os.path.getctime(filepath), unit='s', errors='coerce'),
                'libraries_used': list(set(r_libraries)),
                'code_cell_count': 1, # Treat as one code cell
                'markdown_cell_count': 0,
                'total_lines_of_code': total_lines_of_code,
                'execution_count_sum': 0
            }
        except Exception as e:
            # print(f"Error processing {filepath}: {e}") # Suppress frequent error prints
            return None

    def run_pipeline(self):
        """
        Runs the code metadata extraction pipeline.
        Collects Config.NUM_FILES_PER_EXT random files for each type (.ipynb, .py, .r).
        """
        start_time = time.time()
        print("Starting Code Metadata Extraction Pipeline...")
        extensions = ['.ipynb', '.py', '.r', '.rmd'] # Include .rmd for R notebooks
        selected_files = get_file_paths(self.root_path_code, extensions, num_files_per_ext=Config.NUM_FILES_PER_EXT)

        extracted_data = []
        for filepath in tqdm(selected_files, desc="Processing code files"):
            if filepath.endswith('.ipynb'):
                data = self.extract_ipynb_metadata(filepath)
            elif filepath.endswith('.py'):
                data = self.extract_py_metadata(filepath)
            elif filepath.endswith(('.r', '.rmd')):
                data = self.extract_r_metadata(filepath)
            else:
                data = None

            if data:
                extracted_data.append(data)

        self.notebook_stats_df = pd.DataFrame(extracted_data)
        self.notebook_stats_df['creation_date'] = pd.to_datetime(self.notebook_stats_df['creation_date'], errors='coerce')
        self.notebook_stats_df.dropna(subset=['creation_date'], inplace=True) # Drop rows where date parsing failed

        # Persist to Parquet
        output_path = 'notebook_stats.parquet'
        print(f"Saving code metadata to {output_path}...")
        self.notebook_stats_df.to_parquet(output_path, index=False)
        print(f"Code metadata saved to {output_path}")

        # Summary table of top-used libraries and average code length
        all_libraries = [lib for sublist in self.notebook_stats_df['libraries_used'] for lib in sublist]
        library_counts = pd.Series(all_libraries).value_counts().head(10)
        print("\nTop 10 Most Used Libraries:")
        print(library_counts)

        avg_code_length = self.notebook_stats_df['total_lines_of_code'].mean()
        print(f"\nAverage Total Lines of Code: {avg_code_length:.2f}")

        clean_memory()
        end_time = time.time()
        print(f"Code Metadata Extraction Pipeline Completed in {end_time - start_time:.2f} seconds.")
        return self.notebook_stats_df

# --- Pipeline 2: Temporal Trend Analysis Pipeline ---

class TemporalTrendAnalyzer:
    """
    Tracks evolution of participation, topics, and performance over time.
    """
    def __init__(self, root_path_csv, notebook_stats_df):
        self.root_path_csv = root_path_csv
        self.notebook_stats_df = notebook_stats_df
        self.competitions_df = None
        self.submissions_df = None
        self.kernels_df = None

    def load_data(self):
        """Loads necessary CSV data."""
        load_start_time = time.time()
        print("Loading data for Temporal Trend Analysis...")
        try:
            self.competitions_df = pd.read_csv(os.path.join(self.root_path_csv, 'Competitions.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.submissions_df = pd.read_csv(os.path.join(self.root_path_csv, 'Submissions.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.kernels_df = pd.read_csv(os.path.join(self.root_path_csv, 'Kernels.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)

            # Convert date columns to datetime objects
            print("Parsing date columns...")
            if 'EnabledDate' in self.competitions_df.columns:
                self.competitions_df['EnabledDate'] = pd.to_datetime(self.competitions_df['EnabledDate'], errors='coerce')
            if 'DeadlineDate' in self.competitions_df.columns:
                self.competitions_df['DeadlineDate'] = pd.to_datetime(self.competitions_df['DeadlineDate'], errors='coerce')
            
            if 'SubmissionDate' in self.submissions_df.columns:
                self.submissions_df['SubmissionDate'] = pd.to_datetime(self.submissions_df['SubmissionDate'], errors='coerce')
            
            if 'CreationDate' in self.kernels_df.columns:
                self.kernels_df['CreationDate'] = pd.to_datetime(self.kernels_df['CreationDate'], errors='coerce')

            # Drop rows with invalid dates
            if 'EnabledDate' in self.competitions_df.columns and 'DeadlineDate' in self.competitions_df.columns:
                self.competitions_df.dropna(subset=['EnabledDate', 'DeadlineDate'], inplace=True)
            if 'SubmissionDate' in self.submissions_df.columns:
                self.submissions_df.dropna(subset=['SubmissionDate'], inplace=True)
            if 'CreationDate' in self.kernels_df.columns:
                self.kernels_df.dropna(subset=['CreationDate'], inplace=True)

            print("Data loaded and dates parsed.")
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.root_path_csv}")
            # Create empty DataFrames to prevent further errors
            self.competitions_df = pd.DataFrame(columns=['Id', 'EnabledDate', 'DeadlineDate'])
            # Ensure PublicScore and PrivateScore columns exist in the empty DataFrame for submissions
            self.submissions_df = pd.DataFrame(columns=['Id', 'SubmissionDate', 'PublicScore', 'PrivateScore'])
            self.kernels_df = pd.DataFrame(columns=['Id', 'CreationDate'])
        clean_memory()
        load_end_time = time.time()
        print(f"Data loading for Temporal Trend Analysis completed in {load_end_time - load_start_time:.2f} seconds.")


    def time_series_aggregation(self):
        """Aggregates data into monthly time series."""
        agg_start_time = time.time()
        print("Aggregating time series data...")
        # Competitions
        if not self.competitions_df.empty and 'EnabledDate' in self.competitions_df.columns:
            comp_monthly = self.competitions_df.groupby(self.competitions_df['EnabledDate'].dt.to_period('M')).size().reset_index(name='count')
            comp_monthly['EnabledDate'] = comp_monthly['EnabledDate'].dt.to_timestamp()
            comp_monthly.rename(columns={'EnabledDate': 'Month'}, inplace=True)
            comp_monthly.to_csv('monthly_competitions.csv', index=False)
            print("Monthly competitions data aggregated.")
        else:
            print("Warning: Competitions data is empty or 'EnabledDate' column missing. Skipping monthly competitions aggregation.")

        # Kernels
        if not self.kernels_df.empty and 'CreationDate' in self.kernels_df.columns:
            kernel_monthly = self.kernels_df.groupby(self.kernels_df['CreationDate'].dt.to_period('M')).size().reset_index(name='count')
            kernel_monthly['CreationDate'] = kernel_monthly['CreationDate'].dt.to_timestamp()
            kernel_monthly.rename(columns={'CreationDate': 'Month'}, inplace=True)
            kernel_monthly.to_csv('monthly_kernels.csv', index=False)
            print("Monthly kernels data aggregated.")
        else:
            print("Warning: Kernels data is empty or 'CreationDate' column missing. Skipping monthly kernels aggregation.")

        # Submissions
        if not self.submissions_df.empty and 'SubmissionDate' in self.submissions_df.columns:
            submission_monthly = self.submissions_df.groupby(self.submissions_df['SubmissionDate'].dt.to_period('M')).size().reset_index(name='count')
            submission_monthly['SubmissionDate'] = submission_monthly['SubmissionDate'].dt.to_timestamp()
            submission_monthly.rename(columns={'SubmissionDate': 'Month'}, inplace=True)
            submission_monthly.to_csv('monthly_submissions.csv', index=False)
            print("Monthly submissions data aggregated.")
        else:
            print("Warning: Submissions data is empty or 'SubmissionDate' column missing. Skipping monthly submissions aggregation.")

        # Average scores per competition over time
        if not self.submissions_df.empty and 'PublicScore' in self.submissions_df.columns and 'PrivateScore' in self.submissions_df.columns and 'SubmissionDate' in self.submissions_df.columns:
            self.submissions_df['SubmissionMonth'] = self.submissions_df['SubmissionDate'].dt.to_period('M')
            # Ensure scores are numeric, coercing errors to NaN and then filling with 0 for mean calculation
            self.submissions_df['PublicScore'] = pd.to_numeric(self.submissions_df['PublicScore'], errors='coerce').fillna(0)
            self.submissions_df['PrivateScore'] = pd.to_numeric(self.submissions_df['PrivateScore'], errors='coerce').fillna(0)

            avg_scores_monthly = self.submissions_df.groupby('SubmissionMonth')[['PublicScore', 'PrivateScore']].mean().reset_index()
            avg_scores_monthly['SubmissionMonth'] = avg_scores_monthly['SubmissionMonth'].dt.to_timestamp()
            avg_scores_monthly.rename(columns={'SubmissionMonth': 'Month'}, inplace=True)
            avg_scores_monthly.to_csv('monthly_average_scores.csv', index=False)
            print("Monthly average scores data aggregated.")
        else:
            print("Warning: 'PublicScore', 'PrivateScore', or 'SubmissionDate' columns not found in submissions data or submissions data is empty. Skipping average score aggregation.")


        print("Monthly time-series CSVs created.")
        clean_memory()
        agg_end_time = time.time()
        print(f"Time-series aggregation completed in {agg_end_time - agg_start_time:.2f} seconds.")

    def analyze_seasonality_growth(self):
        """
        Identifies growth spurts using simple moving averages for demonstration.
        For production, statsmodels/Prophet would be used.
        """
        growth_start_time = time.time()
        print("Analyzing seasonality and growth (simplified)...")
        # Example: 3-month rolling average for kernel creation
        if os.path.exists('monthly_kernels.csv'):
            kernel_monthly = pd.read_csv('monthly_kernels.csv', parse_dates=['Month'])
            if not kernel_monthly.empty:
                kernel_monthly['rolling_avg'] = kernel_monthly['count'].rolling(window=3, min_periods=1).mean()
                plt.figure(figsize=(12, 6))
                plt.plot(kernel_monthly['Month'], kernel_monthly['count'], label='Monthly Kernels Created')
                plt.plot(kernel_monthly['Month'], kernel_monthly['rolling_avg'], label='3-Month Rolling Average', linestyle='--')
                plt.title('Kernels Created Over Time with Rolling Average')
                plt.xlabel('Date')
                plt.ylabel('Number of Kernels')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('kernels_created_growth.png')
                plt.close()
                print("Kernels created growth plot generated.")
            else:
                print("No kernel data to analyze growth (monthly_kernels.csv is empty).")
        else:
            print("No kernel data to analyze growth (monthly_kernels.csv not found).")
        clean_memory()
        growth_end_time = time.time()
        print(f"Seasonality and growth analysis completed in {growth_end_time - growth_start_time:.2f} seconds.")

    def visualize_trends(self):
        """Generates matplotlib plots to show participation curves."""
        viz_start_time = time.time()
        print("Generating trend visualizations...")
        # Plot: Kernels created vs. time
        if os.path.exists('monthly_kernels.csv'):
            kernel_monthly = pd.read_csv('monthly_kernels.csv', parse_dates=['Month'])
            if not kernel_monthly.empty:
                plt.figure(figsize=(12, 6))
                sns.lineplot(x='Month', y='count', data=kernel_monthly)
                plt.title('Kernels Created vs. Time')
                plt.xlabel('Date')
                plt.ylabel('Number of Kernels')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('kernels_created_vs_time.png')
                plt.close()
                print("Kernels created vs. time plot generated.")
            else:
                print("No kernel data to plot (monthly_kernels.csv is empty).")
        else:
            print("No kernel data to plot (monthly_kernels.csv not found).")

        # Plot: Average submission score vs. time
        avg_scores_monthly_path = 'monthly_average_scores.csv'
        if os.path.exists(avg_scores_monthly_path):
            avg_scores_monthly = pd.read_csv(avg_scores_monthly_path, parse_dates=['Month'])
            if not avg_scores_monthly.empty:
                plt.figure(figsize=(12, 6))
                sns.lineplot(x='Month', y='PublicScore', data=avg_scores_monthly, label='Average Public Score')
                sns.lineplot(x='Month', y='PrivateScore', data=avg_scores_monthly, label='Average Private Score')
                plt.title('Average Submission Score vs. Time')
                plt.xlabel('Date')
                plt.ylabel('Average Score')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('average_submission_score_vs_time.png')
                plt.close()
                print("Average submission score vs. time plot generated.")
            else:
                print("No submission score data to plot (monthly_average_scores.csv is empty).")
        else:
            print("No submission score data to plot (monthly_average_scores.csv not found).")
        clean_memory()
        viz_end_time = time.time()
        print(f"Trend visualizations generated in {viz_end_time - viz_start_time:.2f} seconds.")

    def run_pipeline(self):
        """Runs the temporal trend analysis pipeline."""
        start_time = time.time()
        print("Starting Temporal Trend Analysis Pipeline...")
        self.load_data()
        self.time_series_aggregation()
        self.analyze_seasonality_growth()
        self.visualize_trends()
        end_time = time.time()
        print(f"Temporal Trend Analysis Pipeline Completed in {end_time - start_time:.2f} seconds.")
        clean_memory()

# --- Pipeline 3: Topic Modeling & NLP Pipeline ---

class TopicModelingNLP:
    """
    Uncovers emerging themes in kernels’ markdown and competition descriptions.
    """
    def __init__(self, root_path_csv, notebook_stats_df):
        self.root_path_csv = root_path_csv
        self.notebook_stats_df = notebook_stats_df
        self.competitions_df = None
        self.datasets_df = None
        self.all_text_data = []

    def load_data_and_extract_text(self):
        """
        Loads necessary CSV data and extracts text from notebook markdown cells
        and competition/dataset descriptions.
        """
        load_extract_start_time = time.time()
        print("Loading data and extracting text for Topic Modeling...")
        try:
            self.competitions_df = pd.read_csv(os.path.join(self.root_path_csv, 'Competitions.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.datasets_df = pd.read_csv(os.path.join(self.root_path_csv, 'Datasets.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)

            # Extract text from competition overviews and dataset descriptions
            if 'Description' in self.competitions_df.columns:
                self.all_text_data.extend(self.competitions_df['Description'].dropna().tolist())
            else:
                print("Warning: 'Description' column not found in Competitions.csv. Skipping competition description extraction.")

            if 'Description' in self.datasets_df.columns:
                self.all_text_data.extend(self.datasets_df['Description'].dropna().tolist())
            else:
                print("Warning: 'Description' column not found in Datasets.csv. Skipping dataset description extraction.")

            # Extract markdown from processed notebooks (if available)
            if 'file_path' in self.notebook_stats_df.columns:
                print("Extracting markdown content from selected notebooks...")
                markdown_texts = []
                for filepath in tqdm(self.notebook_stats_df['file_path'], desc="Extracting markdown"):
                    if filepath.endswith('.ipynb'):
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                nb = nbformat.read(f, as_version=4)
                            markdown_cells = [cell['source'] for cell in nb.cells if cell.cell_type == 'markdown']
                            markdown_texts.append("\n".join(markdown_cells))
                        except Exception as e:
                            # print(f"Could not extract markdown from {filepath}: {e}") # Suppress frequent error prints
                            markdown_texts.append("")
                    else:
                        markdown_texts.append("") # No markdown for .py or .r files

                self.all_text_data.extend(markdown_texts)
            else:
                print("Warning: 'file_path' not found in notebook_stats_df. Skipping markdown extraction from notebooks.")

            self.all_text_data = [str(text) for text in self.all_text_data if pd.notna(text) and text.strip()]
            print(f"Total text documents for topic modeling: {len(self.all_text_data)}")

        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.root_path_csv}")
            self.competitions_df = pd.DataFrame(columns=['Description'])
            self.datasets_df = pd.DataFrame(columns=['Description'])
        clean_memory()
        load_extract_end_time = time.time()
        print(f"Data loading and text extraction completed in {load_extract_end_time - load_extract_start_time:.2f} seconds.")


    def text_cleaning(self, texts):
        """
        Cleans text: removes code snippets (simplified), stopwords, lemmatizes.
        Requires spaCy for lemmatization.
        """
        cleaning_start_time = time.time()
        print("Cleaning text data...")
        # Basic cleaning: remove URLs, special characters, multiple spaces
        cleaned_texts = []
        for text in tqdm(texts, desc="Cleaning text"):
            text = re.sub(r'http\S+|www\S+|\S+\.com\S+', '', text) # Remove URLs
            text = re.sub(r'<.*?>', '', text) # Remove HTML tags
            text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove non-alphabetic characters
            text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
            text = text.lower()
            cleaned_texts.append(text)

        # For proper lemmatization and stopword removal, spaCy is recommended.
        # Example (requires 'en_core_web_sm' model: python -m spacy download en_core_web_sm)
        # import spacy
        # nlp = spacy.load('en_core_web_sm')
        # processed_texts = []
        # for doc in nlp.pipe(cleaned_texts, disable=["parser", "ner"]):
        #     processed_texts.append(" ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha]))
        # return processed_texts

        # Placeholder for spaCy if not installed/configured:
        print("SpaCy not used for lemmatization/stopwords. Install 'en_core_web_sm' for better results.")
        clean_memory()
        cleaning_end_time = time.time()
        print(f"Text cleaning completed in {cleaning_end_time - cleaning_start_time:.2f} seconds.")
        return cleaned_texts # Return basic cleaned texts

    def topic_modeling(self, cleaned_texts):
        """
        Adds BERTopic to extract topics and their evolution.
        """
        modeling_start_time = time.time()
        print("Applying BERTopic for topic modeling...")
        if not cleaned_texts:
            print("No cleaned text data for topic modeling.")
            return None, None

        try:
            model = BERTopic(verbose=True)
            topics, probs = model.fit_transform(cleaned_texts)

            # Get topic information
            topic_info = model.get_topic_info()
            topic_terms_df = pd.DataFrame()
            if not topic_info.empty:
                topic_terms_data = []
                for topic_id in topic_info['Topic'].unique():
                    if topic_id != -1: # -1 is for outliers
                        terms = model.get_topic(topic_id)
                        topic_terms_data.append({'topic_id': topic_id, 'terms': [term[0] for term in terms]})
                topic_terms_df = pd.DataFrame(topic_terms_data)
                topic_terms_df.to_csv('topic_terms.csv', index=False)
                print("Topic terms saved to topic_terms.csv")

            print("\nOverall Topic Distribution:")
            print(pd.Series(topics).value_counts().head(10))

            clean_memory()
            modeling_end_time = time.time()
            print(f"BERTopic modeling completed in {modeling_end_time - modeling_start_time:.2f} seconds.")
            return model, topics
        except Exception as e:
            print(f"Error during BERTopic modeling: {e}. Please ensure 'sentence-transformers' and 'bertopic' are installed.")
            print("Falling back to TF-IDF for vectorization if needed later, but topic modeling will be skipped.")
            clean_memory()
            modeling_end_time = time.time()
            print(f"BERTopic modeling attempted, but failed in {modeling_end_time - modeling_start_time:.2f} seconds.")
            return None, None


    def run_pipeline(self):
        """Runs the topic modeling and NLP pipeline."""
        start_time = time.time()
        print("Starting Topic Modeling & NLP Pipeline...")
        self.load_data_and_extract_text()
        if not self.all_text_data:
            print("No text data available for topic modeling. Skipping pipeline.")
            end_time = time.time()
            print(f"Topic Modeling & NLP Pipeline Completed in {end_time - start_time:.2f} seconds (skipped).")
            return

        cleaned_texts = self.text_cleaning(self.all_text_data)
        topic_model, topics = self.topic_modeling(cleaned_texts)

        if topic_model and topics is not None:
            print("\nTopic Modeling & NLP Pipeline Completed. (Time-series of topic weights not fully implemented without date association)")
        else:
            print("\nTopic Modeling & NLP Pipeline completed with errors or no topics generated.")
        end_time = time.time()
        print(f"Topic Modeling & NLP Pipeline Completed in {end_time - start_time:.2f} seconds.")
        clean_memory()

# --- Pipeline 4: Performance Benchmarking Pipeline ---

class PerformanceBenchmarking:
    """
    Compares reported model performance across frameworks and competitions.
    """
    def __init__(self, root_path_csv):
        self.root_path_csv = root_path_csv
        self.kernels_df = None
        self.kernel_versions_df = None
        self.model_versions_df = None
        self.kernel_version_competition_sources_df = None
        self.performance_metrics_df = pd.DataFrame() # Initialize here

    def load_data(self):
        """Loads necessary CSV data."""
        load_start_time = time.time()
        print("Loading data for Performance Benchmarking...")
        try:
            self.kernels_df = pd.read_csv(os.path.join(self.root_path_csv, 'Kernels.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.kernel_versions_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelVersions.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.model_versions_df = pd.read_csv(os.path.join(self.root_path_csv, 'ModelVersions.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.kernel_version_competition_sources_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelVersionCompetitionSources.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)

            # Convert date columns
            print("Parsing date columns...")
            if 'OriginalPublishDate' in self.model_versions_df.columns:
                self.model_versions_df['OriginalPublishDate'] = pd.to_datetime(self.model_versions_df['OriginalPublishDate'], errors='coerce')
                self.model_versions_df.dropna(subset=['OriginalPublishDate'], inplace=True) 
            else:
                print("Warning: 'OriginalPublishDate' column not found in ModelVersions.csv. Skipping date parsing for this DataFrame.")


            print("Data loaded.")
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.root_path_csv}")
            self.kernels_df = pd.DataFrame(columns=['Id', 'TotalVotes', 'TotalComments'])
            self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId', 'RunningTimeInMilliseconds']) # Ensure KernelId is here for fallback
            self.model_versions_df = pd.DataFrame(columns=['Id', 'OriginalPublishDate'])
            self.kernel_version_competition_sources_df = pd.DataFrame(columns=['KernelVersionId', 'CompetitionId'])
        clean_memory()
        load_end_time = time.time()
        print(f"Data loading for Performance Benchmarking completed in {load_end_time - load_start_time:.2f} seconds.")

    def extract_and_normalize_metrics(self):
        """
        Extracts key metrics and normalizes them.
        """
        extract_norm_start_time = time.time()
        print("Extracting and normalizing performance metrics...")

        if self.kernels_df.empty or self.kernel_versions_df.empty:
            print("Warning: Kernels or KernelVersions DataFrame is empty. Skipping performance metric extraction.")
            return

        if 'KernelId' not in self.kernel_versions_df.columns:
            print("Error: 'KernelId' column not found in kernel_versions_df. Cannot merge for performance metrics.")
            return

        # Merge kernel and kernel_versions data
        merged_kernels = pd.merge(self.kernels_df, self.kernel_versions_df,
                                  left_on='Id', right_on='KernelId', suffixes=('_kernel', '_version'))

        if self.kernel_version_competition_sources_df.empty:
            print("Warning: KernelVersionCompetitionSources DataFrame is empty. Skipping merge with competition sources.")
            merged_kernels_comp = merged_kernels
        else:
            # Merge with competition sources
            merged_kernels_comp = pd.merge(merged_kernels, self.kernel_version_competition_sources_df,
                                           left_on='Id_version', right_on='KernelVersionId', how='left')

        # Calculate metrics
        # Ensure 'RunningTimeInMilliseconds' is numeric
        if 'RunningTimeInMilliseconds' in merged_kernels_comp.columns:
            merged_kernels_comp['RunningTimeInMilliseconds'] = pd.to_numeric(merged_kernels_comp['RunningTimeInMilliseconds'], errors='coerce')
            merged_kernels_comp.dropna(subset=['RunningTimeInMilliseconds'], inplace=True) 
        else:
            print("Warning: 'RunningTimeInMilliseconds' not found. Cannot calculate time-based metrics.")
            merged_kernels_comp['RunningTimeInMilliseconds'] = 0 # Placeholder to avoid errors

        # Normalize votes (example: votes per 1000 milliseconds runtime)
        # Add a small epsilon to avoid division by zero for runtime
        epsilon = 1e-6
        if 'TotalVotes' in merged_kernels_comp.columns and 'RunningTimeInMilliseconds' in merged_kernels_comp.columns:
            merged_kernels_comp['NormalizedVotes'] = pd.to_numeric(merged_kernels_comp['TotalVotes'], errors='coerce') / ((merged_kernels_comp['RunningTimeInMilliseconds'] / 1000) + epsilon)
        else:
            merged_kernels_comp['NormalizedVotes'] = 0 # Placeholder

        if 'TotalComments' in merged_kernels_comp.columns and 'RunningTimeInMilliseconds' in merged_kernels_comp.columns:
            merged_kernels_comp['NormalizedComments'] = pd.to_numeric(merged_kernels_comp['TotalComments'], errors='coerce') / ((merged_kernels_comp['RunningTimeInMilliseconds'] / 1000) + epsilon)
        else:
            merged_kernels_comp['NormalizedComments'] = 0 # Placeholder


        # Rank top-performing notebooks
        self.performance_metrics_df = merged_kernels_comp[[
            'KernelId', 'CompetitionId', 'RunningTimeInMilliseconds', 'TotalVotes',
            'TotalComments', 'NormalizedVotes', 'NormalizedComments'
        ]].copy()

        self.performance_metrics_df.sort_values(by='NormalizedVotes', ascending=False, inplace=True)
        self.performance_metrics_df.to_csv('normalized_performance_metrics.csv', index=False)
        print("Normalized performance metrics saved to normalized_performance_metrics.csv")
        clean_memory()
        extract_norm_end_time = time.time()
        print(f"Metric extraction and normalization completed in {extract_norm_end_time - extract_norm_start_time:.2f} seconds.")

    def rank_and_leaderboard(self):
        """
        Ranks top-performing notebooks and generates leaderboards.
        """
        rank_leaderboard_start_time = time.time()
        print("Generating leaderboards...")
        if self.performance_metrics_df is None or self.performance_metrics_df.empty:
            print("No performance metrics data to generate leaderboards.")
            return

        # Leaderboard by fastest kernels
        fastest_kernels = self.performance_metrics_df.sort_values(by='RunningTimeInMilliseconds', ascending=True).head(10)
        print("\nTop 10 Fastest Kernels:")
        print(fastest_kernels[['KernelId', 'CompetitionId', 'RunningTimeInMilliseconds']])

        # Leaderboard by most popular kernels (normalized votes)
        most_popular_kernels = self.performance_metrics_df.sort_values(by='NormalizedVotes', ascending=False).head(10)
        print("\nTop 10 Most Popular Kernels (by Normalized Votes):")
        print(most_popular_kernels[['KernelId', 'CompetitionId', 'TotalVotes', 'RunningTimeInMilliseconds', 'NormalizedVotes']])

        # Save leaderboards to CSV if needed
        fastest_kernels.to_csv('leaderboard_fastest_kernels.csv', index=False)
        most_popular_kernels.to_csv('leaderboard_most_popular_kernels.csv', index=False)
        print("Leaderboards saved.")
        clean_memory()
        rank_leaderboard_end_time = time.time()
        print(f"Leaderboard generation completed in {rank_leaderboard_end_time - rank_leaderboard_start_time:.2f} seconds.")

    def run_pipeline(self):
        """Runs the performance benchmarking pipeline."""
        start_time = time.time()
        print("Starting Performance Benchmarking Pipeline...")
        self.load_data()
        self.extract_and_normalize_metrics()
        self.rank_and_leaderboard()
        end_time = time.time()
        print(f"Performance Benchmarking Pipeline Completed in {end_time - start_time:.2f} seconds.")
        clean_memory()

# --- Pipeline 5: Collaboration & Social Network Analysis Pipeline ---

class CollaborationNetworkAnalyzer:
    """
    Builds graphs of user interactions, co-authorship, and forum discussions
    to map community structure.
    """
    def __init__(self, root_path_csv):
        self.root_path_csv = root_path_csv
        self.kernel_versions_df = None
        self.kernel_version_kernel_sources_df = None
        self.user_followers_df = None
        self.forum_messages_df = None
        self.teams_df = None
        self.team_memberships_df = None
        self.co_author_graph = nx.Graph() # Initialize
        self.follower_graph = nx.DiGraph() # Initialize
        self.forum_graph = nx.DiGraph() # Initialize

    def load_data(self):
        """Loads necessary CSV data."""
        load_start_time = time.time()
        print("Loading data for Collaboration & Social Network Analysis...")
        try:
            self.kernel_versions_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelVersions.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.kernel_version_kernel_sources_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelVersionKernelSources.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.user_followers_df = pd.read_csv(os.path.join(self.root_path_csv, 'UserFollowers.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.forum_messages_df = pd.read_csv(os.path.join(self.root_path_csv, 'ForumMessages.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.teams_df = pd.read_csv(os.path.join(self.root_path_csv, 'Teams.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.team_memberships_df = pd.read_csv(os.path.join(self.root_path_csv, 'TeamMemberships.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            print("Data loaded.")
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.root_path_csv}")
            self.kernel_versions_df = pd.DataFrame(columns=['Id', 'AuthorUserId', 'KernelId'])
            self.kernel_version_kernel_sources_df = pd.DataFrame(columns=['KernelVersionId', 'SourceKernelVersionId'])
            self.user_followers_df = pd.DataFrame(columns=['UserId', 'FollowingUserId'])
            self.forum_messages_df = pd.DataFrame(columns=['Id', 'PostUserId', 'ReplyToForumMessageId'])
            self.teams_df = pd.DataFrame(columns=['Id', 'TeamLeaderId'])
            self.team_memberships_df = pd.DataFrame(columns=['TeamId', 'UserId'])
        clean_memory()
        load_end_time = time.time()
        print(f"Data loading for Collaboration & Social Network Analysis completed in {load_end_time - load_start_time:.2f} seconds.")

    def build_user_graphs(self):
        """Builds co-author and follower graphs."""
        user_graph_start_time = time.time()
        print("Building user graphs...")
        # Co-author graph (based on forks)
        # Merge kernel versions with kernel sources to get author of original and forked kernel
        # Check if necessary columns exist before merging
        if not self.kernel_version_kernel_sources_df.empty and 'KernelVersionId' in self.kernel_version_kernel_sources_df.columns and \
           not self.kernel_versions_df.empty and 'Id' in self.kernel_versions_df.columns and 'AuthorUserId' in self.kernel_versions_df.columns:
            fork_relationships = pd.merge(
                self.kernel_version_kernel_sources_df,
                self.kernel_versions_df[['Id', 'AuthorUserId']],
                left_on='KernelVersionId',
                right_on='Id',
                suffixes=('_fork', '_version')
            ).rename(columns={'AuthorUserId': 'ForkingUser'})

            fork_relationships = pd.merge(
                fork_relationships,
                self.kernel_versions_df[['Id', 'AuthorUserId']],
                left_on='SourceKernelVersionId',
                right_on='Id',
                suffixes=('_forked', '_original')
            ).rename(columns={'AuthorUserId': 'OriginalUser'})

            # Create edges for co-authorship (if A forks B, they are co-authors in a sense)
            co_author_edges = fork_relationships[['ForkingUser', 'OriginalUser']].dropna().drop_duplicates().values.tolist()
            self.co_author_graph = nx.Graph()
            self.co_author_graph.add_edges_from(co_author_edges)
            print(f"Co-author graph built with {self.co_author_graph.number_of_nodes()} nodes and {self.co_author_graph.number_of_edges()} edges.")
            joblib.dump(self.co_author_graph, 'co_author_graph.pkl')
            print("Co-author graph saved.")
        else:
            print("Warning: Missing required columns or empty DataFrames for building co-author graph. Skipping.")


        # Follower graph
        if not self.user_followers_df.empty and 'UserId' in self.user_followers_df.columns and 'FollowingUserId' in self.user_followers_df.columns:
            follower_edges = self.user_followers_df[['UserId', 'FollowingUserId']].dropna().drop_duplicates().values.tolist()
            self.follower_graph = nx.DiGraph() # Directed graph for followers
            self.follower_graph.add_edges_from(follower_edges)
            print(f"Follower graph built with {self.follower_graph.number_of_nodes()} nodes and {self.follower_graph.number_of_edges()} edges.")
            joblib.dump(self.follower_graph, 'follower_graph.pkl')
            print("Follower graph saved.")
        else:
            print("Warning: Missing required columns or empty DataFrames for building follower graph. Skipping.")
        clean_memory()
        user_graph_end_time = time.time()
        print(f"User graphs built in {user_graph_end_time - user_graph_start_time:.2f} seconds.")

    def build_forum_graph(self):
        """Builds forum reply chain graph."""
        forum_graph_start_time = time.time()
        print("Building forum graph...")
        if not self.forum_messages_df.empty and 'Id' in self.forum_messages_df.columns and 'PostUserId' in self.forum_messages_df.columns and 'ReplyToForumMessageId' in self.forum_messages_df.columns:
            # Filter for actual replies (ReplyToForumMessageId is not null)
            reply_messages = self.forum_messages_df.dropna(subset=['ReplyToForumMessageId']) 

            # Map message IDs to user IDs
            message_to_user = self.forum_messages_df.set_index('Id')['PostUserId'].to_dict()

            forum_edges = []
            for _, row in tqdm(reply_messages.iterrows(), total=reply_messages.shape[0], desc="Processing forum messages"):
                reply_to_user = message_to_user.get(row['ReplyToForumMessageId'])
                post_user = row['PostUserId']
                if reply_to_user and post_user:
                    forum_edges.append((post_user, reply_to_user)) # Edge from replier to original poster

            self.forum_graph = nx.DiGraph()
            self.forum_graph.add_edges_from(forum_edges)
            print(f"Forum graph built with {self.forum_graph.number_of_nodes()} nodes and {self.forum_graph.number_of_edges()} edges.")
            joblib.dump(self.forum_graph, 'forum_graph.pkl')
            print("Forum graph saved.")
        else:
            print("Warning: Missing required columns or empty DataFrames for building forum graph. Skipping.")
        clean_memory()
        forum_graph_end_time = time.time()
        print(f"Forum graph built in {forum_graph_end_time - forum_graph_start_time:.2f} seconds.")

    def compute_network_metrics(self):
        """Computes centrality and community detection metrics."""
        metrics_start_time = time.time()
        print("Computing network metrics...")
        # Replaced .is_empty() with .number_of_nodes() == 0
        if hasattr(self, 'co_author_graph') and self.co_author_graph.number_of_nodes() > 0:
            print("\nCo-author Graph Metrics:")
            # PageRank
            pagerank_co_author = nx.pagerank(self.co_author_graph)
            top_pagerank_co_author = sorted(pagerank_co_author.items(), key=lambda item: item[1], reverse=True)[:10]
            print("Top 10 PageRank (Co-author):", top_pagerank_co_author)

            # Community detection (Louvain - requires python-louvain)
            # from community import community_louvain
            # partition = community_louvain.best_partition(self.co_author_graph)
            # num_communities = len(set(partition.values()))
            # print(f"Number of communities (Co-author): {num_communities}")
            print("Community detection (Louvain) skipped. Install 'python-louvain' for this feature.")
        else:
            print("Co-author graph not available or empty for metric computation.")

        if hasattr(self, 'follower_graph') and self.follower_graph.number_of_nodes() > 0:
            print("\nFollower Graph Metrics:")
            # In-degree centrality (influence)
            in_degree_centrality = nx.in_degree_centrality(self.follower_graph)
            top_in_degree = sorted(in_degree_centrality.items(), key=lambda item: item[1], reverse=True)[:10]
            print("Top 10 In-Degree Centrality (Follower):", top_in_degree)
        else:
            print("Follower graph not available or empty for metric computation.")

        if hasattr(self, 'forum_graph') and self.forum_graph.number_of_nodes() > 0:
            print("\nForum Graph Metrics:")
            # PageRank for forum influence
            pagerank_forum = nx.pagerank(self.forum_graph)
            top_pagerank_forum = sorted(pagerank_forum.items(), key=lambda item: item[1], reverse=True)[:10]
            print("Top 10 PageRank (Forum):", top_pagerank_forum)
        else:
            print("Forum graph not available or empty for metric computation.")

        print("\nReports on 'most central' Kaggle contributors would be generated here, combining various metrics.")
        clean_memory()
        metrics_end_time = time.time()
        print(f"Network metrics computation completed in {metrics_end_time - metrics_start_time:.2f} seconds.")

    def run_pipeline(self):
        """Runs the collaboration and social network analysis pipeline."""
        start_time = time.time()
        print("Starting Collaboration & Social Network Analysis Pipeline...")
        self.load_data()
        self.build_user_graphs()
        self.build_forum_graph()
        self.compute_network_metrics()
        end_time = time.time()
        print(f"Collaboration & Social Network Analysis Pipeline Completed in {end_time - start_time:.2f} seconds.")
        clean_memory()

# --- Pipeline 6: Evolutionary Pathways Pipeline ---

class EvolutionaryPathways:
    """
    Traces the lineage of kernels/models/datasets to understand how ideas propagate.
    """
    def __init__(self, root_path_csv):
        self.root_path_csv = root_path_csv
        self.kernel_versions_df = None
        self.kernel_version_kernel_sources_df = None
        self.kernel_version_model_sources_df = None
        self.kernel_version_dataset_sources_df = None
        self.tripartite_graph = nx.Graph() # Initialize

    def load_data(self):
        """Loads necessary CSV data."""
        load_start_time = time.time()
        print("Loading data for Evolutionary Pathways...")
        try:
            self.kernel_versions_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelVersions.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.kernel_version_kernel_sources_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelVersionKernelSources.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            self.kernel_version_model_sources_df = pd.read_csv(os.path.join(self.root_path_csv, 'ModelVersions.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS) # Corrected: ModelVersions.csv
            self.kernel_version_dataset_sources_df = pd.read_csv(os.path.join(self.root_path_csv, 'KernelVersionDatasetSources.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
            print("Data loaded.")
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.root_path_csv}")
            self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId'])
            self.kernel_version_kernel_sources_df = pd.DataFrame(columns=['KernelVersionId', 'SourceKernelVersionId'])
            self.kernel_version_model_sources_df = pd.DataFrame(columns=['KernelVersionId', 'SourceModelVersionId'])
            self.kernel_version_dataset_sources_df = pd.DataFrame(columns=['KernelVersionId', 'SourceDatasetVersionId'])
        clean_memory()
        load_end_time = time.time()
        print(f"Data loading for Evolutionary Pathways completed in {load_end_time - load_start_time:.2f} seconds.")

    def construct_tripartite_graph(self):
        """Constructs a tripartite graph: kernels ↔ models ↔ datasets."""
        graph_construct_start_time = time.time()
        print("Constructing tripartite graph...")
        self.tripartite_graph = nx.Graph()

        # Add nodes for kernels, models, and datasets
        if not self.kernel_versions_df.empty and 'Id' in self.kernel_versions_df.columns:
            for _, row in tqdm(self.kernel_versions_df.iterrows(), total=self.kernel_versions_df.shape[0], desc="Adding kernel nodes"):
                self.tripartite_graph.add_node(f"kernel_{row['Id']}", type='kernel')
        else:
            print("Warning: 'Id' column not found or kernel_versions_df is empty. Skipping kernel node creation.")

        # Models and datasets might not have explicit 'versions' in their main CSVs,
        # so we'll use the IDs from the source tables.
        # For model sources, we need the actual model ID, not kernel version ID
        if not self.kernel_version_model_sources_df.empty and 'SourceModelVersionId' in self.kernel_version_model_sources_df.columns:
            for _, row in tqdm(self.kernel_version_model_sources_df.iterrows(), total=self.kernel_version_model_sources_df.shape[0], desc="Adding model nodes"):
                self.tripartite_graph.add_node(f"model_{row['SourceModelVersionId']}", type='model')
        else:
            print("Warning: 'SourceModelVersionId' not found in kernel_version_model_sources_df. Skipping model node creation.")

        if not self.kernel_version_dataset_sources_df.empty and 'SourceDatasetVersionId' in self.kernel_version_dataset_sources_df.columns:
            for _, row in tqdm(self.kernel_version_dataset_sources_df.iterrows(), total=self.kernel_version_dataset_sources_df.shape[0], desc="Adding dataset nodes"):
                self.tripartite_graph.add_node(f"dataset_{row['SourceDatasetVersionId']}", type='dataset')
        else:
            print("Warning: 'SourceDatasetVersionId' not found in kernel_version_dataset_sources_df. Skipping dataset node creation.")


        # Add edges: KernelVersionId -> SourceKernelVersionId (forks)
        if not self.kernel_version_kernel_sources_df.empty and 'KernelVersionId' in self.kernel_version_kernel_sources_df.columns and 'SourceKernelVersionId' in self.kernel_version_kernel_sources_df.columns:
            for _, row in tqdm(self.kernel_version_kernel_sources_df.iterrows(), total=self.kernel_version_kernel_sources_df.shape[0], desc="Adding kernel-kernel edges"):
                if f"kernel_{row['KernelVersionId']}" in self.tripartite_graph and f"kernel_{row['SourceKernelVersionId']}" in self.tripartite_graph:
                    self.tripartite_graph.add_edge(f"kernel_{row['KernelVersionId']}", f"kernel_{row['SourceKernelVersionId']}", relation='forks')
        else:
            print("Warning: Missing columns or empty DataFrame for kernel-kernel edges. Skipping.")

        # Add edges: KernelVersionId -> SourceModelVersionId
        if not self.kernel_version_model_sources_df.empty and 'KernelVersionId' in self.kernel_version_model_sources_df.columns and 'SourceModelVersionId' in self.kernel_version_model_sources_df.columns:
            for _, row in tqdm(self.kernel_version_model_sources_df.iterrows(), total=self.kernel_version_model_sources_df.shape[0], desc="Adding kernel-model edges"):
                if f"kernel_{row['KernelVersionId']}" in self.tripartite_graph and f"model_{row['SourceModelVersionId']}" in self.tripartite_graph:
                    self.tripartite_graph.add_edge(f"kernel_{row['KernelVersionId']}", f"model_{row['SourceModelVersionId']}", relation='uses_model')
        else:
            print("Warning: Missing columns or empty DataFrame for kernel-model edges. Skipping.")

        # Add edges: KernelVersionId -> SourceDatasetVersionId
        if not self.kernel_version_dataset_sources_df.empty and 'KernelVersionId' in self.kernel_version_dataset_sources_df.columns and 'SourceDatasetVersionId' in self.kernel_version_dataset_sources_df.columns:
            for _, row in tqdm(self.kernel_version_dataset_sources_df.iterrows(), total=self.kernel_version_dataset_sources_df.shape[0], desc="Adding kernel-dataset edges"):
                if f"kernel_{row['KernelVersionId']}" in self.tripartite_graph and f"dataset_{row['SourceDatasetVersionId']}" in self.tripartite_graph:
                    self.tripartite_graph.add_edge(f"kernel_{row['KernelVersionId']}", f"dataset_{row['SourceDatasetVersionId']}", relation='uses_dataset')
        else:
            print("Warning: Missing columns or empty DataFrame for kernel-dataset edges. Skipping.")

        print(f"Tripartite graph built with {self.tripartite_graph.number_of_nodes()} nodes and {self.tripartite_graph.number_of_edges()} edges.")
        joblib.dump(self.tripartite_graph, 'tripartite_graph.pkl')
        print("Tripartite graph saved.")
        clean_memory()
        graph_construct_end_time = time.time()
        print(f"Tripartite graph construction completed in {graph_construct_end_time - graph_construct_start_time:.2f} seconds.")

    def path_discovery(self):
        """Finds common ancestors for top-voted kernels."""
        path_disc_start_time = time.time()
        print("Discovering common ancestors for top-voted kernels (simplified)...")
        # This requires identifying "top-voted" kernels first (from Pipeline 4 outputs)
        # For demonstration, let's pick some arbitrary kernel IDs.
        # In a real scenario, you'd load 'leaderboard_most_popular_kernels.csv'
        if not self.kernel_versions_df.empty and 'KernelId' in self.kernel_versions_df.columns:
            top_kernel_ids = self.kernel_versions_df['KernelId'].sample(min(5, len(self.kernel_versions_df))).tolist()
            top_kernel_nodes = [f"kernel_{kid}" for kid in top_kernel_ids if f"kernel_{kid}" in self.tripartite_graph]
        else:
            top_kernel_nodes = []

        if len(top_kernel_nodes) < 2:
            print("Not enough top kernels to find common ancestors or kernel_versions_df is empty/missing 'KernelId'.")
            path_disc_end_time = time.time()
            print(f"Path discovery completed in {path_disc_end_time - path_disc_start_time:.2f} seconds (skipped).")
            return

        print("Common neighbors among top kernels (simplified ancestor view):")
        for i in range(len(top_kernel_nodes)):
            for j in range(i + 1, len(top_kernel_nodes)):
                node1 = top_kernel_nodes[i]
                node2 = top_kernel_nodes[j]
                if self.tripartite_graph.has_node(node1) and self.tripartite_graph.has_node(node2):
                    common_neighbors = list(nx.common_neighbors(self.tripartite_graph, node1, node2))
                    if common_neighbors:
                        print(f"  Common neighbors of {node1} and {node2}: {common_neighbors}")
        clean_memory()
        path_disc_end_time = time.time()
        print(f"Path discovery completed in {path_disc_end_time - path_disc_start_time:.2f} seconds.")

    def sequence_mining(self):
        """Identifies frequent "recipe" patterns."""
        seq_mining_start_time = time.time()
        print("Identifying frequent 'recipe' patterns (simplified)...")
        # This is a complex task requiring sequence pattern mining algorithms.
        # A simplified approach could be to look at common chains of (kernel -> model -> dataset).
        # For example, find kernels that use a specific model and then a specific dataset.

        # Example: Find kernels that use a specific model type (e.g., 'ResNet' - requires model detection from NLP)
        # and a specific dataset type. This would need more granular data.

        # Placeholder for actual sequence mining
        print("Sequence mining for 'recipe' patterns is a complex task and requires deeper analysis of code content.")
        print("Example: If we had 'model_type' and 'dataset_type' extracted for each kernel, we could find patterns like 'ResNet -> ImageNet'.")
        clean_memory()
        seq_mining_end_time = time.time()
        print(f"Sequence mining completed in {seq_mining_end_time - seq_mining_start_time:.2f} seconds.")

    def run_pipeline(self):
        """Runs the evolutionary pathways pipeline."""
        start_time = time.time()
        print("Starting Evolutionary Pathways Pipeline...")
        self.load_data()
        self.construct_tripartite_graph()
        self.path_discovery()
        self.sequence_mining()
        end_time = time.time()
        print(f"Evolutionary Pathways Pipeline Completed in {end_time - start_time:.2f} seconds.")
        clean_memory()

# --- Main Execution Flow ---

def run_all_pipelines():
    """
    Orchestrates the execution of all pipelines.
    """
    overall_start_time = time.time()
    print("--- Starting Meta Kaggle Analysis Orchestration ---")

    # 1. Code Metadata Extraction Pipeline
    pipeline_start_time = time.time()
    code_extractor = CodeMetadataExtractor(ROOT_PATH_CODE)
    notebook_stats_df = code_extractor.run_pipeline()
    pipeline_end_time = time.time()
    print(f"Pipeline 1 (Code Metadata Extraction) took {pipeline_end_time - pipeline_start_time:.2f} seconds.\n")
    clean_memory()

    # 2. Temporal Trend Analysis Pipeline
    pipeline_start_time = time.time()
    temporal_analyzer = TemporalTrendAnalyzer(ROOT_PATH_CSV, notebook_stats_df)
    temporal_analyzer.run_pipeline()
    pipeline_end_time = time.time()
    print(f"Pipeline 2 (Temporal Trend Analysis) took {pipeline_end_time - pipeline_start_time:.2f} seconds.\n")
    clean_memory()

    # 3. Topic Modeling & NLP Pipeline
    pipeline_start_time = time.time()
    topic_modeler = TopicModelingNLP(ROOT_PATH_CSV, notebook_stats_df)
    topic_modeler.run_pipeline()
    pipeline_end_time = time.time()
    print(f"Pipeline 3 (Topic Modeling & NLP) took {pipeline_end_time - pipeline_start_time:.2f} seconds.\n")
    clean_memory()

    # 4. Performance Benchmarking Pipeline
    pipeline_start_time = time.time()
    performance_benchmarker = PerformanceBenchmarking(ROOT_PATH_CSV)
    performance_benchmarker.run_pipeline()
    pipeline_end_time = time.time()
    print(f"Pipeline 4 (Performance Benchmarking) took {pipeline_end_time - pipeline_start_time:.2f} seconds.\n")
    clean_memory()

    # 5. Collaboration & Social Network Analysis Pipeline
    pipeline_start_time = time.time()
    collaboration_analyzer = CollaborationNetworkAnalyzer(ROOT_PATH_CSV)
    collaboration_analyzer.run_pipeline()
    pipeline_end_time = time.time()
    print(f"Pipeline 5 (Collaboration & Social Network Analysis) took {pipeline_end_time - pipeline_start_time:.2f} seconds.\n")
    clean_memory()

    # 6. Evolutionary Pathways Pipeline
    pipeline_start_time = time.time()
    evolutionary_pathways = EvolutionaryPathways(ROOT_PATH_CSV)
    evolutionary_pathways.run_pipeline()
    pipeline_end_time = time.time()
    print(f"Pipeline 6 (Evolutionary Pathways) took {pipeline_end_time - pipeline_start_time:.2f} seconds.\n")
    clean_memory()

    # 7. Interactive Dashboard & Reporting Pipeline (Conceptual - not implemented as code)
    print("\n--- Pipeline 7: Interactive Dashboard & Reporting Pipeline (Conceptual) ---")
    print("This pipeline would involve loading all generated artifacts (Parquet, CSVs, Pickle files)")
    print("and building an interactive dashboard using frameworks like Plotly Dash or Streamlit.")
    print("It would allow non-technical stakeholders to explore the insights visually.")
    print("Example: A dashboard with filters for dates, topics, and graphs of network centrality.")
    print("Deployment would involve containerization (Docker) and hosting (e.g., Kaggle, Heroku).")

    overall_end_time = time.time()
    print(f"\n--- All Analysis Pipelines Completed in {overall_end_time - overall_start_time:.2f} seconds ---")

# --- Data Loading, Preprocessing, EDA, Feature Engineering, Model Training, Outlier Detection, Memory Management, Hyperparameter Tuning, Evaluation, Plotting, Model Saving ---

def comprehensive_ml_pipeline(data_path_csv, notebook_stats_df):
    """
    Implements a comprehensive ML pipeline for a hypothetical prediction task
    based on the extracted metadata and CSV data.
    This is a conceptual pipeline demonstrating the requested ML steps.
    """
    ml_overall_start_time = time.time()
    print("\n--- Starting Comprehensive ML Pipeline ---")

    # 1. Data Loading, Preprocessing, Exploratory Data Analysis (EDA)
    step_start_time = time.time()
    print("1. Data Loading, Preprocessing, EDA...")
    try:
        # Load relevant CSVs for a hypothetical prediction task
        users_df = pd.read_csv(os.path.join(data_path_csv, 'Users.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
        user_achievements_df = pd.read_csv(os.path.join(data_path_csv, 'UserAchievements.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)
        kernels_df = pd.read_csv(os.path.join(data_path_csv, 'Kernels.csv'), low_memory=False, nrows=Config.NUM_ROWS_TO_LOAD_CSVS)

        if not kernels_df.empty and 'TotalVotes' in kernels_df.columns:
            kernels_df['TotalVotes'] = pd.to_numeric(kernels_df['TotalVotes'], errors='coerce').fillna(0)
            vote_threshold = kernels_df['TotalVotes'].quantile(0.8)
            kernels_df['IsHighlyVoted'] = (kernels_df['TotalVotes'] >= vote_threshold).astype(int)
            print(f"Highly voted threshold (80th percentile of TotalVotes): {vote_threshold}")
        else:
            print("Kernels DataFrame is empty or 'TotalVotes' column not found. Cannot define 'IsHighlyVoted'. Exiting ML pipeline.")
            return # Exit if no data

        df_merged = pd.merge(kernels_df, users_df, left_on='AuthorUserId', right_on='Id', suffixes=('_kernel', '_user'), how='left')
        df_merged = pd.merge(df_merged, user_achievements_df, left_on='AuthorUserId', right_on='UserId', suffixes=('_merged', '_achievement'), how='left')

        features = [
            'TotalComments', 'TotalVotes', 'TotalViews', # From Kernels.csv
            'CompetitionCount', 'DatasetCount', 'KernelCount', # From Users.csv (after merge)
            'TotalGold', 'TotalSilver', 'TotalBronze', # From UserAchievements.csv (after merge)
            'IsHighlyVoted' # Target variable
        ]
        
        # Filter features to only include those present in the merged DataFrame
        available_features = [f for f in features if f in df_merged.columns]
        if 'IsHighlyVoted' not in available_features:
            print("Error: 'IsHighlyVoted' target column not available after merging. Exiting ML pipeline.")
            return

        df_model = df_merged[available_features].copy()

        for col in tqdm(['TotalComments', 'TotalVotes', 'TotalViews', 'CompetitionCount', 'DatasetCount', 'KernelCount', 'TotalGold', 'TotalSilver', 'TotalBronze'], desc="Converting to numeric"):
            if col in df_model.columns: # Check if column exists before converting
                df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

        imputer = SimpleImputer(strategy='mean')
        # Ensure 'IsHighlyVoted' is not imputed
        features_to_impute = df_model.drop(columns=['IsHighlyVoted']).columns
        df_model_imputed = pd.DataFrame(imputer.fit_transform(df_model[features_to_impute]), columns=features_to_impute)
        df_model_imputed['IsHighlyVoted'] = df_model['IsHighlyVoted'].reset_index(drop=True) # Add target back, ensure index alignment

        print("\nEDA: Descriptive Statistics of Features:")
        print(df_model_imputed.describe())

        plt.figure(figsize=(6, 4))
        sns.countplot(x='IsHighlyVoted', data=df_model_imputed)
        plt.title('Distribution of IsHighlyVoted (Target Variable)')
        plt.xlabel('Is Highly Voted')
        plt.ylabel('Count')
        plt.show()
        plt.savefig('target_distribution.png')
        plt.close()
        print("Target variable distribution plot generated.")

        plt.figure(figsize=(10, 8))
        sns.heatmap(df_model_imputed.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Features')
        plt.show()
        plt.savefig('correlation_matrix.png')
        plt.close()
        print("Correlation matrix plot generated.")

        # New EDA: Distribution plots for numerical features
        print("\nGenerating distribution plots for numerical features...")
        numerical_cols = df_model_imputed.select_dtypes(include=np.number).columns.drop('IsHighlyVoted', errors='ignore')
        for col in tqdm(numerical_cols, desc="Plotting numerical distributions"):
            plt.figure(figsize=(8, 5))
            sns.histplot(df_model_imputed[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
            plt.savefig(f'distribution_{col}.png')
            plt.close()
        print("Numerical feature distribution plots generated.")

        # New EDA: Box plots/Violin plots for numerical features vs. target
        print("\nGenerating box/violin plots for numerical features vs. target...")
        for col in tqdm(numerical_cols, desc="Plotting numerical vs. target"):
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='IsHighlyVoted', y=col, data=df_model_imputed)
            plt.title(f'{col} by IsHighlyVoted')
            plt.xlabel('Is Highly Voted')
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()
            plt.savefig(f'boxplot_{col}_by_target.png')
            plt.close()
        print("Numerical feature vs. target plots generated.")

        # New EDA: Pair plots for a subset of features (can be memory intensive for many features)
        # Select a few key features for pair plot to avoid memory issues
        pair_plot_features = [col for col in ['TotalVotes', 'TotalComments', 'KernelCount', 'TotalGold', 'IsHighlyVoted'] if col in df_model_imputed.columns]
        if len(pair_plot_features) > 1:
            print("\nGenerating pair plot for a subset of key features (may take time)...")
            try:
                sns.pairplot(df_model_imputed[pair_plot_features], hue='IsHighlyVoted', diag_kind='kde')
                plt.suptitle('Pair Plot of Key Features by Target', y=1.02) # Adjust title position
                plt.show()
                plt.savefig('pairplot_key_features.png')
                plt.close()
                print("Pair plot generated.")
            except Exception as e:
                print(f"Warning: Could not generate pair plot due to error: {e}")
        else:
            print("Not enough features for pair plot or required columns missing.")

        clean_memory()
        step_end_time = time.time()
        print(f"Step 1 (Data Loading, Preprocessing, EDA) completed in {step_end_time - step_start_time:.2f} seconds.")

        # 2. Comprehensive Feature Engineering
        step_start_time = time.time()
        print("\n2. Comprehensive Feature Engineering...")
        
        if 'TotalVotes' in df_model_imputed.columns and 'TotalComments' in df_model_imputed.columns:
            df_model_imputed['VotesPerComment'] = df_model_imputed['TotalVotes'] / (df_model_imputed['TotalComments'] + 1e-6)
        else:
            print("Warning: 'TotalVotes' or 'TotalComments' not found for 'VotesPerComment' feature.")
            df_model_imputed['VotesPerComment'] = 0 # Placeholder

        if 'TotalGold' in df_model_imputed.columns and 'TotalSilver' in df_model_imputed.columns and \
           'TotalBronze' in df_model_imputed.columns and 'CompetitionCount' in df_model_imputed.columns:
            df_model_imputed['MedalsPerCompetition'] = (df_model_imputed['TotalGold'] + df_model_imputed['TotalSilver'] + df_model_imputed['TotalBronze']) / (df_model_imputed['CompetitionCount'] + 1e-6)
        else:
            print("Warning: Missing medal or competition count columns for 'MedalsPerCompetition' feature.")
            df_model_imputed['MedalsPerCompetition'] = 0 # Placeholder


        for col in tqdm(['TotalVotes', 'TotalComments', 'TotalViews'], desc="Applying log transform"):
            if col in df_model_imputed.columns:
                df_model_imputed[f'log_{col}'] = np.log1p(df_model_imputed[col])

        numerical_features_for_pca = df_model_imputed.drop(columns=['IsHighlyVoted'], errors='ignore').select_dtypes(include=np.number).columns
        if not numerical_features_for_pca.empty:
            pca = PCA(n_components=min(5, len(numerical_features_for_pca)))
            principal_components = pca.fit_transform(df_model_imputed[numerical_features_for_pca])
            pca_df = pd.DataFrame(data=principal_components, columns=[f'PC_{i+1}' for i in range(pca.n_components)])
            df_model_imputed = pd.concat([df_model_imputed.reset_index(drop=True), pca_df], axis=1)
            print(f"PCA applied, added {pca.n_components} principal components.")
        else:
            print("No numerical features for PCA after initial processing.")

        X = df_model_imputed.drop(columns=['IsHighlyVoted'])
        y = df_model_imputed['IsHighlyVoted']
        print("Feature engineering completed.")
        clean_memory()
        step_end_time = time.time()
        print(f"Step 2 (Comprehensive Feature Engineering) completed in {step_end_time - step_start_time:.2f} seconds.")

        # 7. Verify data splits, evaluation on unseen/holdout dataset
        step_start_time = time.time()
        print("\n7. Verifying data splits and preparing for evaluation...")
        if len(X) == 0 or len(y) == 0:
            print("No data available for splitting. Skipping model training and evaluation.")
            return # Exit if no data for splitting

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Train set size: {len(X_train)} samples")
        print(f"Test set size: {len(X_test)} samples")
        print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Data scaled.")
        clean_memory()
        step_end_time = time.time()
        print(f"Step 7 (Data Splits & Scaling) completed in {step_end_time - step_start_time:.2f} seconds.")

        # 3. Model Training
        step_start_time = time.time()
        print("\n3. Model Training...")
        best_model = None
        best_accuracy = 0.0
        model_results = {}

        models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=0), # Replaced LightGBM with CatBoost
            'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'MLP': MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100, 50))
        }

        for name, model in tqdm(models.items(), desc="Training Traditional Models"):
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            model_results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
            print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
            clean_memory()
        step_end_time = time.time()
        print(f"Step 3 (Traditional Model Training) completed in {step_end_time - step_start_time:.2f} seconds.")

        # 10. Implement best ensemble/hybrid algorithm (CNN + Attention for tabular data)
        step_start_time = time.time()
        print("\n10. Implementing Ensemble/Hybrid (CNN + Attention) Model for Tabular Data...")

        # Ensure X_train_scaled has features for reshaping
        if X_train_scaled.shape[1] == 0:
            print("Warning: No features available for CNN + Attention model. Skipping.")
            cnn_attention_model = None # Set to None if skipped
        else:
            X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
            X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1) # Corrected X_test_reshaped shape

            input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])

            def create_cnn_attention_model(input_shape):
                inputs = Input(shape=input_shape)
                conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
                pool1 = GlobalMaxPooling1D()(conv1)
                attention_input = Reshape((1, 64))(pool1)
                attention_output = Attention()([attention_input, attention_input])
                attention_output = Flatten()(attention_output)
                dense1 = Dense(128, activation='relu')(attention_output)
                outputs = Dense(1, activation='sigmoid')(dense1)
                model = Model(inputs=inputs, outputs=outputs)
                return model

            # Force CPU usage for the CNN + Attention model to bypass cuDNN issues
            print("Forcing CNN + Attention model training on CPU to avoid cuDNN compatibility issues.")
            with tf.device('/CPU:0'): 
                cnn_attention_model = create_cnn_attention_model(input_shape)
                cnn_attention_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
                model_checkpoint = ModelCheckpoint('best_cnn_attention_model.keras', save_best_only=True, monitor='val_loss', mode='min')

                print("Training CNN + Attention model (progress bar not available for Keras fit)...")
                history = cnn_attention_model.fit(
                    X_train_reshaped, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stopping, reduce_lr, model_checkpoint],
                    verbose=0 # Suppress verbose output for cleaner tqdm integration
                )
                print("CNN + Attention model trained.")
            
            if cnn_attention_model: # Check if model was successfully created
                # Predictions also need to be on CPU if training was forced to CPU
                with tf.device('/CPU:0'):
                    y_pred_cnn_attention_proba = cnn_attention_model.predict(X_test_reshaped).flatten()
                y_pred_cnn_attention = (y_pred_cnn_attention_proba > 0.5).astype(int)
                accuracy_cnn_attention = accuracy_score(y_test, y_pred_cnn_attention)
                roc_auc_cnn_attention = roc_auc_score(y_test, y_pred_cnn_attention_proba)
                model_results['CNN_Attention'] = {'accuracy': accuracy_cnn_attention, 'roc_auc': roc_auc_cnn_attention}
                print(f"CNN + Attention - Accuracy: {accuracy_cnn_attention:.4f}, ROC AUC: {roc_auc_cnn_attention:.4f}")

                if accuracy_cnn_attention > best_accuracy:
                    best_accuracy = accuracy_cnn_attention
                    best_model = cnn_attention_model
                    print("CNN + Attention is currently the best model.")
            else:
                print("CNN + Attention model could not be created or trained.")

        clean_memory()
        step_end_time = time.time()
        print(f"Step 10 (CNN + Attention Model Training) completed in {step_end_time - step_start_time:.2f} seconds.")

        # 4. Outlier Detection and weight adjustment (Conceptual)
        step_start_time = time.time()
        print("\n4. Outlier Detection and weight adjustment (Conceptual)...")
        if X_train_scaled.shape[1] > 0: # Only run if features exist
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(random_state=42, contamination=0.05)
            outliers = iso_forest.fit_predict(X_train_scaled)
            sample_weights = np.array([1.0 if o == 1 else 0.1 for o in outliers])
            print(f"Detected {np.sum(outliers == -1)} outliers in training data.")
            print("Outlier detection performed. Weight adjustment conceptualized.")
        else:
            print("Skipping outlier detection: no features available.")
        clean_memory()
        step_end_time = time.time()
        print(f"Step 4 (Outlier Detection) completed in {step_end_time - step_start_time:.2f} seconds.")

        # 6. Hyperparameter Tuning
        step_start_time = time.time()
        print("\n6. Hyperparameter Tuning (Conceptual)...")
        if X_train_scaled.shape[0] > 0 and X_train_scaled.shape[1] > 0: # Only run if data exists
            from sklearn.model_selection import GridSearchCV

            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', verbose=0, n_jobs=-1)
            print("Starting GridSearchCV for RandomForest (this may take a while)...")
            # tqdm for GridSearchCV is tricky, as fit itself is long. We'll use a single bar for the whole process.
            with tqdm(total=1, desc="Hyperparameter Tuning (GridSearchCV)") as pbar_tuning:
                grid_search.fit(X_train_scaled, y_train)
                pbar_tuning.update(1)

            print(f"Best parameters for RandomForest: {grid_search.best_params_}")
            print(f"Best cross-validation accuracy for RandomForest: {grid_search.best_score_:.4f}")

            if grid_search.best_score_ > best_accuracy:
                best_accuracy = grid_search.best_score_
                best_model = grid_search.best_estimator_
                print("Tuned RandomForest is now the best model.")
            print("Hyperparameter tuning completed.")
        else:
            print("Skipping hyperparameter tuning: no training data available.")
        clean_memory()
        step_end_time = time.time()
        print(f"Step 6 (Hyperparameter Tuning) completed in {step_end_time - step_start_time:.2f} seconds.")

        # 8. Add logic for all the necessary graphical plots, including graphical plots on unseen/holdout data, visualizations etc
        step_start_time = time.time()
        print("\n8. Generating graphical plots for evaluation...")
        if not model_results:
            print("No model results available for plotting. Skipping ROC curve generation.")
        else:
            plt.figure(figsize=(8, 6))
            for name, results in tqdm(model_results.items(), desc="Generating ROC Curves"):
                if 'roc_auc' in results:
                    # Ensure the model exists before trying to predict
                    if name == 'CNN_Attention' and cnn_attention_model is not None:
                        # Ensure prediction is also on CPU
                        with tf.device('/CPU:0'):
                            y_proba = cnn_attention_model.predict(X_test_reshaped).flatten()
                    elif name != 'CNN_Attention' and name in models:
                        y_proba = models[name].predict_proba(X_test_scaled)[:, 1]
                    else:
                        continue # Skip if model is not available

                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {results["roc_auc"]:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Classification Models')
            plt.legend()
            plt.grid(True)
            plt.show()
            plt.savefig('roc_curve.png')
            plt.close()
            print("ROC curve plot generated.")

        if best_model and hasattr(best_model, 'feature_importances_') and not X.columns.empty:
            feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
            plt.figure(figsize=(10, 6))
            feature_importances.nlargest(10).plot(kind='barh')
            plt.title('Top 10 Feature Importances (Best Model)')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()
            plt.savefig('feature_importance.png')
            plt.close()
            print("Feature importance plot generated.")
        else:
            print("Best model does not have feature importances attribute or X.columns is empty for plotting.")
        clean_memory()
        step_end_time = time.time()
        print(f"Step 8 (Graphical Plots) completed in {step_end_time - step_start_time:.2f} seconds.")

        # 9. The best model with highest accuracy should be saved
        step_start_time = time.time()
        print("\n9. Saving the best model...")
        if best_model:
            if isinstance(best_model, tf.keras.Model):
                best_model.save('best_classification_model.keras')
                print("Best Keras model saved as 'best_classification_model.keras'")
            else:
                joblib.dump(best_model, 'best_classification_model.pkl')
                print("Best scikit-learn/LightGBM/XGBoost model saved as 'best_classification_model.pkl'")
        else:
            print("No best model found to save.")
        clean_memory()
        step_end_time = time.time()
        print(f"Step 9 (Model Saving) completed in {step_end_time - step_start_time:.2f} seconds.")

        ml_overall_end_time = time.time()
        print(f"\n--- Comprehensive ML Pipeline Completed in {ml_overall_end_time - ml_overall_start_time:.2f} seconds ---")

    except Exception as e:
        print(f"An error occurred during the comprehensive ML pipeline: {e}")
        import traceback
        traceback.print_exc()
    clean_memory()


# --- Main entry point for execution ---
if __name__ == '__main__':
    # Run all analysis pipelines
    run_all_pipelines()

    # Run the comprehensive ML pipeline (conceptual prediction task)
    comprehensive_ml_pipeline(ROOT_PATH_CSV, pd.DataFrame())


import os
import nbformat
import pandas as pd
import numpy as np
import gc
import re
import ast
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy.stats import zscore
import networkx as nx
from gensim.models import Word2Vec
from bertopic import BERTopic
import torch
from transformers import AutoTokenizer, AutoModel
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Attention, Reshape, Permute, multiply, concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import logging

# Import UMAP and HDBSCAN for explicit configuration
import umap
import hdbscan

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# --- Configuration Class ---
class Config:
    """
    Holds code-wide configuration variables.
    """
    def __init__(self):
        self.ROOT_PATH_CODE = "/kaggle/input/meta-kaggle-code"
        self.ROOT_PATH_CSV = "/kaggle/input/meta-kaggle"
        self.NUM_FILES_PER_EXT = 20  # Number of files to sample for code metadata extraction
        self.MAX_DIRS_TO_SCAN = 500 # Maximum number of directories to scan during file collection
        self.TRI_GRAPH_SAMPLE_SIZE = 5000 # Max kernels to sample for tripartite graph construction
        self.RANDOM_STATE = 42 # Global random state for reproducibility
        self.ML_TEST_SIZE = 0.2 # Test set size for ML pipeline
        self.ML_MAX_EPOCHS = 100 # Max epochs for TensorFlow ANN
        self.ML_BATCH_SIZE = 32 # Batch size for TensorFlow ANN
        self.ML_ANN_DROPOUT_RATE = 0.3 # Dropout rate for ANN layers
        self.ML_ANN_LEARNING_RATE = 0.001 # Learning rate for ANN optimizer
        self.ML_ANN_PATIENCE = 10 # Early stopping patience
        self.ML_ANN_REDUCE_LR_PATIENCE = 5 # Reduce LR on plateau patience
        self.MAX_TEXT_DOCS_FOR_TOPIC_MODELING = 5000 # Max text documents for topic modeling to prevent memory issues


# --- Global Configuration and Helper Functions ---

def clean_memory():
    """Aggressively cleans memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Memory cleaned.")

# Pass config object to get_file_paths
def get_file_paths(config, root_dir, extensions, num_files_per_ext):
    """
    Walks through the directory tree and collects a random sample of file paths
    for specified extensions. Stops scanning for an extension once num_files_per_ext
    are found for all extensions, or if MAX_DIRS_TO_SCAN is reached.
    """
    collected_files_by_ext = {ext: [] for ext in extensions}
    remaining_needed = {ext: num_files_per_ext for ext in extensions}

    logging.info(f"Collecting up to {num_files_per_ext} files per extension from {root_dir} (max {config.MAX_DIRS_TO_SCAN} directories)...")

    dir_count = 0
    file_found_count = 0
    # Explicitly set total for tqdm to MAX_DIRS_TO_SCAN
    pbar = tqdm(desc="Scanning directories and collecting files", unit="dirs", total=config.MAX_DIRS_TO_SCAN)

    try: # Ensure tqdm is always closed
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dir_count += 1
            pbar.update(1)

            # Stop if max directories scanned is reached
            if dir_count > config.MAX_DIRS_TO_SCAN:
                logging.warning(f"Reached maximum directory scan limit ({config.MAX_DIRS_TO_SCAN}). Stopping early.")
                break

            for filename in filenames:
                for ext in extensions:
                    if filename.endswith(ext) and remaining_needed[ext] > 0:
                        collected_files_by_ext[ext].append(os.path.join(dirpath, filename))
                        remaining_needed[ext] -= 1
                        file_found_count += 1
                        pbar.set_postfix_str(f"Files: {file_found_count}")

            all_extensions_full = all(remaining_needed[ext] == 0 for ext in extensions)
            if all_extensions_full:
                logging.info("All desired file counts met. Stopping directory scan early.")
                break
    finally:
        pbar.close() # Ensure pbar is always closed

    selected_files = []
    for ext in extensions:
        if len(collected_files_by_ext[ext]) > num_files_per_ext:
            selected_files.extend(random.sample(collected_files_by_ext[ext], num_files_per_ext))
        else:
            selected_files.extend(collected_files_by_ext[ext])

    logging.info(f"Finished collecting files. Total selected: {len(selected_files)}.")
    return selected_files

def extract_imports(code_string):
    """
    Extracts unique top-level library imports from a Python code string.
    Handles 'import x' and 'from x import y'.
    """
    imports = set()
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except SyntaxError as e:
        logging.warning(f"SyntaxError during import extraction: {e}")
    return list(imports)

def get_code_lines(code_string):
    """Counts non-empty lines of code."""
    return len([line for line in code_string.split('\n') if line.strip() and not line.strip().startswith('#')])

# --- Pipeline 1: Code Metadata Extraction Pipeline ---

class CodeMetadataExtractor:
    """
    Parses code files (Jupyter Notebooks, Python, R) to extract metadata,
    library imports, and code metrics.
    """
    def __init__(self, config): # Accept config object
        self.config = config
        self.notebook_stats_df = pd.DataFrame()

    def extract_ipynb_metadata(self, filepath):
        """Extracts metadata from a .ipynb file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            metadata = nb.metadata.get('metadata', {})
            author = metadata.get('author', 'Unknown')
            created = metadata.get('created', '1970-01-01T00:00:00Z')

            code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
            markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']

            all_code = "\n".join([cell['source'] for cell in code_cells])
            libraries = extract_imports(all_code)
            total_lines_of_code = get_code_lines(all_code)

            execution_count_sum = sum(cell.get('execution_count', 0) or 0 for cell in code_cells)

            return {
                'file_path': filepath,
                'file_type': 'ipynb',
                'author': author,
                'creation_date': pd.to_datetime(created, errors='coerce'),
                'libraries_used': libraries,
                'code_cell_count': len(code_cells),
                'markdown_cell_count': len(markdown_cells),
                'total_lines_of_code': total_lines_of_code,
                'execution_count_sum': execution_count_sum
            }
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")
            return None

    def extract_py_metadata(self, filepath):
        """Extracts metadata from a .py file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            libraries = extract_imports(content)
            total_lines_of_code = get_code_lines(content)

            return {
                'file_path': filepath,
                'file_type': 'py',
                'author': 'Unknown',
                'creation_date': pd.to_datetime(os.path.getctime(filepath), unit='s', errors='coerce'),
                'libraries_used': libraries,
                'code_cell_count': 1,
                'markdown_cell_count': 0,
                'total_lines_of_code': total_lines_of_code,
                'execution_count_sum': 0
            }
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")
            return None

    def extract_r_metadata(self, filepath):
        """Extracts metadata from an .R or .Rmd file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            r_libraries = re.findall(r'(?:library|require)\(([\w.]+)\)', content)
            total_lines_of_code = get_code_lines(content)

            return {
                'file_path': filepath,
                'file_type': 'r',
                'author': 'Unknown',
                'creation_date': pd.to_datetime(os.path.getctime(filepath), unit='s', errors='coerce'),
                'libraries_used': list(set(r_libraries)),
                'code_cell_count': 1,
                'markdown_cell_count': 0,
                'total_lines_of_code': total_lines_of_code,
                'execution_count_sum': 0
            }
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")
            return None

    def visualize_metadata(self):
        """Generates visualizations for code metadata."""
        if self.notebook_stats_df.empty:
            logging.warning("No notebook statistics data to visualize.")
            return

        logging.info("Generating code metadata visualizations...")

        # Distribution of Code Cell Counts
        plt.figure(figsize=(10, 6))
        sns.histplot(self.notebook_stats_df['code_cell_count'], bins=20, kde=True)
        plt.title('Distribution of Code Cell Counts per Notebook')
        plt.xlabel('Code Cell Count')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        plt.savefig('code_cell_count_distribution.png')
        plt.close()
        logging.info("Code cell count distribution plot generated.")

        # Distribution of Markdown Cell Counts
        plt.figure(figsize=(10, 6))
        sns.histplot(self.notebook_stats_df['markdown_cell_count'], bins=20, kde=True)
        plt.title('Distribution of Markdown Cell Counts per Notebook')
        plt.xlabel('Markdown Cell Count')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        plt.savefig('markdown_cell_count_distribution.png')
        plt.close()
        logging.info("Markdown cell count distribution plot generated.")

        # Distribution of Total Lines of Code
        plt.figure(figsize=(10, 6))
        sns.histplot(self.notebook_stats_df['total_lines_of_code'], bins=20, kde=True)
        plt.title('Distribution of Total Lines of Code per File')
        plt.xlabel('Total Lines of Code')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        plt.show()
        plt.savefig('total_lines_of_code_distribution.png')
        plt.close()
        logging.info("Total lines of code distribution plot generated.")

        # Top 10 Most Used Libraries (Bar Chart)
        all_libraries = [lib for sublist in self.notebook_stats_df['libraries_used'] for lib in sublist]
        if all_libraries:
            library_counts = pd.Series(all_libraries).value_counts().head(10)
            plt.figure(figsize=(12, 7))
            sns.barplot(x=library_counts.values, y=library_counts.index, palette='viridis')
            plt.title('Top 10 Most Used Libraries')
            plt.xlabel('Number of Occurrences')
            plt.ylabel('Library')
            plt.tight_layout()
            plt.show()
            plt.savefig('top_libraries_bar_chart.png')
            plt.close()
            logging.info("Top libraries bar chart generated.")
        else:
            logging.warning("No library data to plot.")

        # Code Creation Dates Over Time (Monthly frequency)
        if 'creation_date' in self.notebook_stats_df.columns and not self.notebook_stats_df['creation_date'].empty:
            monthly_creation = self.notebook_stats_df['creation_date'].dt.to_period('M').value_counts().sort_index()
            monthly_creation.index = monthly_creation.index.to_timestamp()
            plt.figure(figsize=(12, 6))
            monthly_creation.plot(kind='line', marker='o', linestyle='-')
            plt.title('Code File Creation Frequency Over Time')
            plt.xlabel('Month')
            plt.ylabel('Number of Files Created')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('code_creation_over_time.png')
            plt.close()
            logging.info("Code creation over time plot generated.")
        else:
            logging.warning("No creation date data to plot for code metadata.")

        clean_memory()
        logging.info("Code metadata visualizations completed.")


    def run_pipeline(self):
        """
        Runs the code metadata extraction pipeline.
        Collects 20 random files for each type (.ipynb, .py, .r).
        """
        start_time = time.time()
        logging.info("Starting Code Metadata Extraction Pipeline...")
        extensions = ['.ipynb', '.py', '.r', '.rmd']
        selected_files = get_file_paths(self.config, self.config.ROOT_PATH_CODE, extensions, self.config.NUM_FILES_PER_EXT) # Pass config object

        extracted_data = []
        for filepath in tqdm(selected_files, desc="Processing code files"):
            if filepath.endswith('.ipynb'):
                data = self.extract_ipynb_metadata(filepath)
            elif filepath.endswith('.py'):
                data = self.extract_py_metadata(filepath)
            elif filepath.endswith(('.r', '.rmd')):
                data = self.extract_r_metadata(filepath)
            else:
                data = None

            if data:
                extracted_data.append(data)

        self.notebook_stats_df = pd.DataFrame(extracted_data)
        self.notebook_stats_df['creation_date'] = pd.to_datetime(self.notebook_stats_df['creation_date'], errors='coerce')
        self.notebook_stats_df.dropna(subset=['creation_date'], inplace=True)

        output_path = 'notebook_stats.parquet'
        logging.info(f"Saving code metadata to {output_path}...")
        self.notebook_stats_df.to_parquet(output_path, index=False)
        logging.info(f"Code metadata saved to {output_path}")

        all_libraries = [lib for sublist in self.notebook_stats_df['libraries_used'] for lib in sublist]
        library_counts = pd.Series(all_libraries).value_counts().head(10)
        logging.info("\nTop 10 Most Used Libraries:")
        logging.info(library_counts)

        avg_code_length = self.notebook_stats_df['total_lines_of_code'].mean()
        logging.info(f"\nAverage Total Lines of Code: {avg_code_length:.2f}")

        self.visualize_metadata() # Call visualization method

        clean_memory()
        end_time = time.time()
        logging.info(f"Code Metadata Extraction Pipeline Completed in {end_time - start_time:.2f} seconds.")
        return self.notebook_stats_df

# --- Pipeline 2: Temporal Trend Analysis Pipeline ---

class TemporalTrendAnalyzer:
    """
    Tracks evolution of participation, topics, and performance over time.
    """
    def __init__(self, config, notebook_stats_df): # Accept config object
        self.config = config
        self.notebook_stats_df = notebook_stats_df
        self.competitions_df = None
        self.submissions_df = None
        self.kernels_df = None

    def load_data(self):
        """Loads necessary CSV data."""
        load_start_time = time.time()
        logging.info("Loading data for Temporal Trend Analysis...")
        try:
            self.competitions_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Competitions.csv'), low_memory=False) # Use config
            self.submissions_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Submissions.csv'), low_memory=False) # Use config
            self.kernels_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Kernels.csv'), low_memory=False) # Use config

            logging.info("Parsing date columns...")
            self.competitions_df['EnabledDate'] = pd.to_datetime(self.competitions_df['EnabledDate'], errors='coerce')
            self.competitions_df['DeadlineDate'] = pd.to_datetime(self.competitions_df['DeadlineDate'], errors='coerce')
            self.submissions_df['SubmissionDate'] = pd.to_datetime(self.submissions_df['SubmissionDate'], errors='coerce')
            self.kernels_df['CreationDate'] = pd.to_datetime(self.kernels_df['CreationDate'], errors='coerce')

            self.competitions_df.dropna(subset=['EnabledDate', 'DeadlineDate'], inplace=True)
            self.submissions_df.dropna(subset=['SubmissionDate'], inplace=True)
            self.kernels_df.dropna(subset=['CreationDate'], inplace=True)

            logging.info("Data loaded and dates parsed.")
        except FileNotFoundError as e:
            logging.error(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.config.ROOT_PATH_CSV}") # Use config
            self.competitions_df = pd.DataFrame(columns=['Id', 'EnabledDate', 'DeadlineDate'])
            self.submissions_df = pd.DataFrame(columns=['Id', 'SubmissionDate', 'PublicScore', 'PrivateScore'])
            self.kernels_df = pd.DataFrame(columns=['Id', 'CreationDate'])
        clean_memory()
        load_end_time = time.time()
        logging.info(f"Data loading for Temporal Trend Analysis completed in {load_end_time - load_start_time:.2f} seconds.")

    def time_series_aggregation(self):
        """Aggregates data into monthly time series."""
        agg_start_time = time.time()
        logging.info("Aggregating time series data...")

        # Competitions
        comp_monthly = self.competitions_df.groupby(self.competitions_df['EnabledDate'].dt.to_period('M')).size().reset_index(name='count')
        comp_monthly['EnabledDate'] = comp_monthly['EnabledDate'].dt.to_timestamp()
        comp_monthly.rename(columns={'EnabledDate': 'Month'}, inplace=True)
        comp_monthly.to_csv('monthly_competitions.csv', index=False)
        logging.info("Monthly competitions data aggregated.")

        # Kernels
        kernel_monthly = self.kernels_df.groupby(self.kernels_df['CreationDate'].dt.to_period('M')).size().reset_index(name='count')
        kernel_monthly['CreationDate'] = kernel_monthly['CreationDate'].dt.to_timestamp()
        kernel_monthly.rename(columns={'CreationDate': 'Month'}, inplace=True)
        kernel_monthly.to_csv('monthly_kernels.csv', index=False)
        logging.info("Monthly kernels data aggregated.")

        # Submissions
        submission_monthly = self.submissions_df.groupby(self.submissions_df['SubmissionDate'].dt.to_period('M')).size().reset_index(name='count')
        submission_monthly['SubmissionDate'] = submission_monthly['SubmissionDate'].dt.to_timestamp()
        submission_monthly.rename(columns={'SubmissionDate': 'Month'}, inplace=True)
        submission_monthly.to_csv('monthly_submissions.csv', index=False)
        logging.info("Monthly submissions data aggregated.")

        # Average scores per competition over time
        if not self.submissions_df.empty and 'PublicScore' in self.submissions_df.columns and 'PrivateScore' in self.submissions_df.columns:
            self.submissions_df['SubmissionMonth'] = self.submissions_df['SubmissionDate'].dt.to_period('M')
            self.submissions_df['PublicScore'] = pd.to_numeric(self.submissions_df['PublicScore'], errors='coerce').fillna(0)
            self.submissions_df['PrivateScore'] = pd.to_numeric(self.submissions_df['PrivateScore'], errors='coerce').fillna(0)
            avg_scores_monthly = self.submissions_df.groupby('SubmissionMonth')[['PublicScore', 'PrivateScore']].mean().reset_index()
            avg_scores_monthly['SubmissionMonth'] = avg_scores_monthly['SubmissionMonth'].dt.to_timestamp()
            avg_scores_monthly.rename(columns={'SubmissionMonth': 'Month'}, inplace=True)
            avg_scores_monthly.to_csv('monthly_average_scores.csv', index=False)
            logging.info("Monthly average scores data aggregated.")
        else:
            logging.warning("Warning: 'PublicScore' or 'PrivateScore' columns not found in submissions data or submissions data is empty. Skipping average score aggregation.")
        logging.info("Monthly time-series CSVs created.")
        clean_memory()
        agg_end_time = time.time()
        logging.info(f"Time-series aggregation completed in {agg_end_time - agg_start_time:.2f} seconds.")

    def analyze_seasonality_growth(self):
        """
        Identifies growth spurts using simple moving averages for demonstration.
        For production, statsmodels/Prophet would be used.
        """
        growth_start_time = time.time()
        logging.info("Analyzing seasonality and growth (simplified)...")

        # Example: 3-month rolling average for kernel creation
        try:
            kernel_monthly = pd.read_csv('monthly_kernels.csv', parse_dates=['Month'])
            if not kernel_monthly.empty:
                kernel_monthly['rolling_avg'] = kernel_monthly['count'].rolling(window=3, min_periods=1).mean()
                plt.figure(figsize=(12, 6))
                plt.plot(kernel_monthly['Month'], kernel_monthly['count'], label='Monthly Kernels Created')
                plt.plot(kernel_monthly['Month'], kernel_monthly['rolling_avg'], label='3-Month Rolling Average', linestyle='--')
                plt.title('Kernels Created Over Time with Rolling Average')
                plt.xlabel('Date')
                plt.ylabel('Number of Kernels')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('kernels_created_over_time.png')
                plt.close()
                logging.info("Kernels created vs. time plot generated.")
            else:
                logging.warning("No kernel data to plot for seasonality and growth analysis.")
        except FileNotFoundError:
            logging.warning("monthly_kernels.csv not found. Skipping kernel trend plot.")
        except Exception as e:
            logging.error(f"Error generating kernel trend plot: {e}")

        # Example: Plotting average submission scores
        try:
            avg_scores_monthly = pd.read_csv('monthly_average_scores.csv', parse_dates=['Month'])
            if not avg_scores_monthly.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(avg_scores_monthly['Month'], avg_scores_monthly['PublicScore'], label='Average Public Score')
                plt.plot(avg_scores_monthly['Month'], avg_scores_monthly['PrivateScore'], label='Average Private Score', linestyle='--')
                plt.title('Average Submission Scores Over Time')
                plt.xlabel('Date')
                plt.ylabel('Average Score')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('average_submission_scores.png')
                plt.close()
                logging.info("Average submission scores plot generated.")
            else:
                logging.warning("No submission score data to plot (monthly_average_scores.csv is empty).")
        except FileNotFoundError:
            logging.warning("No submission score data to plot (monthly_average_scores.csv not found).")
        except Exception as e:
            logging.error(f"Error generating average submission scores plot: {e}")

        # Plotting monthly competitions
        try:
            comp_monthly = pd.read_csv('monthly_competitions.csv', parse_dates=['Month'])
            if not comp_monthly.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(comp_monthly['Month'], comp_monthly['count'], label='Monthly Competitions Started')
                plt.title('Competitions Started Over Time')
                plt.xlabel('Date')
                plt.ylabel('Number of Competitions')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('monthly_competitions_plot.png')
                plt.close()
                logging.info("Monthly competitions plot generated.")
            else:
                logging.warning("No monthly competition data to plot.")
        except FileNotFoundError:
            logging.warning("monthly_competitions.csv not found. Skipping monthly competitions plot.")
        except Exception as e:
            logging.error(f"Error generating monthly competitions plot: {e}")

        # Plotting monthly submissions
        try:
            submission_monthly = pd.read_csv('monthly_submissions.csv', parse_dates=['Month'])
            if not submission_monthly.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(submission_monthly['Month'], submission_monthly['count'], label='Monthly Submissions')
                plt.title('Submissions Over Time')
                plt.xlabel('Date')
                plt.ylabel('Number of Submissions')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('monthly_submissions_plot.png')
                plt.close()
                logging.info("Monthly submissions plot generated.")
            else:
                logging.warning("No monthly submission data to plot.")
        except FileNotFoundError:
            logging.warning("monthly_submissions.csv not found. Skipping monthly submissions plot.")
        except Exception as e:
            logging.error(f"Error generating monthly submissions plot: {e}")

        clean_memory()
        growth_end_time = time.time()
        logging.info(f"Trend visualizations generated in {growth_end_time - growth_start_time:.2f} seconds.")

    def run_pipeline(self):
        """Runs the temporal trend analysis pipeline."""
        start_time = time.time()
        logging.info("Starting Temporal Trend Analysis Pipeline...")
        self.load_data()
        self.time_series_aggregation()
        self.analyze_seasonality_growth()
        clean_memory()
        end_time = time.time()
        logging.info(f"Temporal Trend Analysis Pipeline Completed in {end_time - start_time:.2f} seconds.")

# --- Pipeline 3: Topic Modeling & NLP Pipeline ---

class TopicModelingNLP:
    """
    Performs topic modeling on text data from notebooks, competitions, and datasets.
    """
    def __init__(self, config): # Accept config object
        self.config = config
        self.all_text_data = []
        self.notebook_files = []
        self.topics = []
        self.probs = []

    def load_and_extract_text(self):
        """Loads data and extracts text for topic modeling."""
        load_start_time = time.time()
        logging.info("Loading data and extracting text for Topic Modeling...")
        self.all_text_data = []

        # Extract text from Competitions.csv (if 'Description' column exists)
        try:
            competitions_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Competitions.csv'), low_memory=False) # Use config
            if 'Description' in competitions_df.columns:
                self.all_text_data.extend(competitions_df['Description'].dropna().tolist())
                logging.info(f"Extracted {len(competitions_df['Description'].dropna())} competition descriptions.")
            else:
                logging.warning("'Description' column not found in Competitions.csv. Skipping competition description extraction.")
        except FileNotFoundError:
            logging.warning("Competitions.csv not found. Skipping competition description extraction.")
        except Exception as e:
            logging.error(f"Error loading/processing Competitions.csv for topic modeling: {e}")

        # Extract text from Datasets.csv (if 'Description' column exists)
        try:
            datasets_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Datasets.csv'), low_memory=False) # Use config
            if 'Description' in datasets_df.columns:
                self.all_text_data.extend(datasets_df['Description'].dropna().tolist())
                logging.info(f"Extracted {len(datasets_df['Description'].dropna())} dataset descriptions.")
            else:
                logging.warning("'Description' column not found in Datasets.csv. Skipping dataset description extraction.")
        except FileNotFoundError:
            logging.warning("Datasets.csv not found. Skipping dataset description extraction.")
        except Exception as e:
            logging.error(f"Error loading/processing Datasets.csv for topic modeling: {e}")

        # Extract markdown content from selected notebooks
        self.notebook_files = get_file_paths(self.config, self.config.ROOT_PATH_CODE, ['.ipynb'], self.config.NUM_FILES_PER_EXT) # Pass config object
        logging.info("Extracting markdown content from selected notebooks...")
        for filepath in tqdm(self.notebook_files, desc="Extracting markdown"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']
                for cell in markdown_cells:
                    self.all_text_data.append(cell['source'])
            except Exception as e:
                logging.error(f"Error extracting markdown from {filepath}: {e}")

        # Sample text data if it exceeds the maximum configured limit
        if len(self.all_text_data) > self.config.MAX_TEXT_DOCS_FOR_TOPIC_MODELING:
            logging.warning(f"Too many text documents ({len(self.all_text_data)}) for topic modeling. Sampling {self.config.MAX_TEXT_DOCS_FOR_TOPIC_MODELING} documents.")
            self.all_text_data = random.sample(self.all_text_data, self.config.MAX_TEXT_DOCS_FOR_TOPIC_MODELING)

        logging.info(f"Total text documents for topic modeling: {len(self.all_text_data)}")
        clean_memory()
        load_end_time = time.time()
        logging.info(f"Data loading and text extraction completed in {load_end_time - load_start_time:.2f} seconds.")

    def clean_text(self, texts):
        """Basic text cleaning."""
        clean_start_time = time.time()
        logging.info("Cleaning text data...")
        cleaned_texts = []
        for text in tqdm(texts, desc="Cleaning text"):
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
            text = re.sub(r'\S*@\S*\s?', '', text) # Remove emails
            text = re.sub(r'#\w+', '', text) # Remove hashtags
            text = re.sub(r'@\w+', '', text) # Remove mentions
            text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
            text = text.lower() # Lowercase
            text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
            cleaned_texts.append(text)
        clean_end_time = time.time()
        logging.info(f"Text cleaning completed in {clean_end_time - clean_start_time:.2f} seconds.")
        return cleaned_texts

    def apply_bertopic(self, texts):
        """Applies BERTopic for topic modeling."""
        bertopic_start_time = time.time()
        logging.info("Applying BERTopic for topic modeling...")
        try:
            # Configure UMAP and HDBSCAN explicitly to potentially avoid hangs
            # Setting n_jobs=1 for UMAP to disable parallel processing
            # You might need to experiment with n_neighbors and min_dist for UMAP
            # and min_cluster_size for HDBSCAN based on your dataset characteristics
            umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=self.config.RANDOM_STATE, n_jobs=1)
            hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_epsilon=0.5, prediction_data=True)

            # Using a pre-trained sentence transformer model
            model = BERTopic(verbose=True, umap_model=umap_model, hdbscan_model=hdbscan_model)
            self.topics, self.probs = model.fit_transform(texts)

            # Get topic information
            topic_info = model.get_topic_info()
            logging.info("Topic terms saved to topic_terms.csv")
            topic_info.to_csv('topic_terms.csv', index=False)

            logging.info("\nOverall Topic Distribution:")
            logging.info(pd.Series(self.topics).value_counts())

            # Save the model
            model.save("bertopic_model", serialization="safetensors")
            logging.info("BERTopic model saved as 'bertopic_model'.")

            self.visualize_topics(model, self.topics) # Call visualization method

        except Exception as e:
            logging.error(f"Error during BERTopic modeling: {e}")
            self.topics = []
            self.probs = []
        clean_memory()
        bertopic_end_time = time.time()
        logging.info(f"BERTopic modeling completed in {bertopic_end_time - bertopic_start_time:.2f} seconds.")
        return self.topics, self.probs

    def visualize_topics(self, model, topics):
        """Generates visualizations for topic modeling."""
        if not topics:
            logging.warning("No topics data to visualize.")
            return

        logging.info("Generating topic modeling visualizations...")

        # Bar chart of topic frequencies
        topic_counts = pd.Series(topics).value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=topic_counts.index, y=topic_counts.values, palette='viridis')
        plt.title('Topic Frequencies')
        plt.xlabel('Topic ID')
        plt.ylabel('Number of Documents')
        plt.tight_layout()
        plt.show()
        plt.savefig('topic_frequencies.png')
        plt.close()
        logging.info("Topic frequencies plot generated.")

        # Visualize topics (requires specific libraries and might be heavy)
        try:
            # This can be very heavy on resources for many topics/documents.
            # Only enable if you have sufficient memory and want interactive plots.
            # model.visualize_topics().write_html("bertopic_topics.html")
            # logging.info("BERTopic intertopic distance map generated.")
            # model.visualize_barchart().write_html("bertopic_barchart.html")
            # logging.info("BERTopic barchart generated.")
            pass
        except Exception as e:
            logging.warning(f"Failed to generate advanced BERTopic visualizations (might be memory intensive): {e}")

        clean_memory()
        logging.info("Topic modeling visualizations completed.")

    def run_pipeline(self):
        """Runs the topic modeling and NLP pipeline."""
        start_time = time.time()
        logging.info("Starting Topic Modeling & NLP Pipeline...")
        self.load_and_extract_text()
        cleaned_texts = self.clean_text(self.all_text_data)
        if cleaned_texts:
            topics, probs = self.apply_bertopic(cleaned_texts)
            # Further analysis on topics and their evolution over time would go here
            # (e.g., if text data had associated timestamps)
            logging.info("Topic Modeling & NLP Pipeline Completed. (Time-series of topic weights not fully implemented without date association)")
        else:
            logging.warning("No text data available for topic modeling. Skipping BERTopic.")
        clean_memory()
        end_time = time.time()
        logging.info(f"Topic Modeling & NLP Pipeline Completed in {end_time - start_time:.2f} seconds.")

# --- Pipeline 4: Performance Benchmarking Pipeline ---

class PerformanceBenchmarker:
    """
    Analyzes kernel and submission performance, identifies top performers.
    """
    def __init__(self, config): # Accept config object
        self.config = config
        self.kernels_df = None
        self.submissions_df = None
        self.kernel_versions_df = None
        self.kernel_performance_df = pd.DataFrame()

    def load_data(self):
        """Loads necessary CSV data for performance benchmarking."""
        load_start_time = time.time()
        logging.info("Loading data for Performance Benchmarking...")
        try:
            self.kernels_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Kernels.csv'), low_memory=False) # Use config
            self.submissions_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Submissions.csv'), low_memory=False) # Use config
            self.kernel_versions_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'KernelVersions.csv'), low_memory=False) # Use config

            logging.info("Parsing date columns...")
            self.kernels_df['CreationDate'] = pd.to_datetime(self.kernels_df['CreationDate'], errors='coerce')
            self.submissions_df['SubmissionDate'] = pd.to_datetime(self.submissions_df['SubmissionDate'], errors='coerce')
            self.kernel_versions_df['CreationDate'] = pd.to_datetime(self.kernel_versions_df['CreationDate'], errors='coerce')

            self.kernels_df.dropna(subset=['CreationDate'], inplace=True)
            self.submissions_df.dropna(subset=['SubmissionDate'], inplace=True)
            self.kernel_versions_df.dropna(subset=['CreationDate'], inplace=True)

            logging.info("Data loaded.")
        except FileNotFoundError as e:
            logging.error(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.config.ROOT_PATH_CSV}") # Use config
            self.kernels_df = pd.DataFrame(columns=['Id', 'AuthorUserId', 'CreationDate'])
            self.submissions_df = pd.DataFrame(columns=['Id', 'SubmissionDate', 'PublicScore', 'PrivateScore', 'KernelVersionId', 'CompetitionId'])
            self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId', 'ScriptLanguageId', 'VersionNumber', 'TotalVotes', 'TotalComments'])
        clean_memory()
        load_end_time = time.time()
        logging.info(f"Data loading for Performance Benchmarking completed in {load_end_time - load_start_time:.2f} seconds.")

    def extract_and_normalize_metrics(self):
        """Extracts and normalizes performance metrics."""
        metrics_start_time = time.time()
        logging.info("Extracting and normalizing performance metrics...")

        # Merge kernel versions with kernels to get author info
        if not self.kernel_versions_df.empty and 'KernelId' in self.kernel_versions_df.columns:
            # Ensure 'Id' in kernels_df is the KernelId
            kernel_performance_df = pd.merge(self.kernel_versions_df, self.kernels_df[['Id', 'AuthorUserId']],
                                             left_on='KernelId', right_on='Id', how='left', suffixes=('_version', '_kernel'))
            kernel_performance_df.drop(columns=['Id_kernel'], inplace=True)
            kernel_performance_df.rename(columns={'Id_version': 'KernelVersionId'}, inplace=True)

            # Calculate engagement metrics
            kernel_performance_df['EngagementScore'] = kernel_performance_df['TotalVotes'] + kernel_performance_df['TotalComments'] * 5 # Arbitrary weighting

            # Normalize scores (example: Min-Max Scaling)
            if not kernel_performance_df.empty and 'EngagementScore' in kernel_performance_df.columns:
                scaler = MinMaxScaler()
                kernel_performance_df['NormalizedEngagementScore'] = scaler.fit_transform(kernel_performance_df[['EngagementScore']])
                logging.info("Kernel engagement metrics extracted and normalized.")
            else:
                logging.warning("No engagement data or 'EngagementScore' column for normalization.")

            # Merge with submissions to get submission scores for kernels
            if not self.submissions_df.empty and 'KernelVersionId' in self.submissions_df.columns:
                kernel_performance_df = pd.merge(kernel_performance_df, self.submissions_df[['KernelVersionId', 'PublicScore', 'PrivateScore']],
                                                 on='KernelVersionId', how='left')
                # Fill NaN scores with 0 or a sensible default for aggregation
                kernel_performance_df['PublicScore'] = pd.to_numeric(kernel_performance_df['PublicScore'], errors='coerce').fillna(0)
                kernel_performance_df['PrivateScore'] = pd.to_numeric(kernel_performance_df['PrivateScore'], errors='coerce').fillna(0)

                if not kernel_performance_df.empty and ('PublicScore' in kernel_performance_df.columns or 'PrivateScore' in kernel_performance_df.columns):
                    # Aggregate scores per kernel (e.g., max score)
                    kernel_performance_df['MaxPublicScore'] = kernel_performance_df.groupby('KernelId')['PublicScore'].transform('max')
                    kernel_performance_df['MaxPrivateScore'] = kernel_performance_df.groupby('KernelId')['PrivateScore'].transform('max')

                    # Normalize submission scores (example: Z-score)
                    if kernel_performance_df['MaxPublicScore'].std() > 0:
                        kernel_performance_df['NormalizedPublicScore'] = zscore(kernel_performance_df['MaxPublicScore'])
                    else:
                        kernel_performance_df['NormalizedPublicScore'] = 0 # Or handle as appropriate

                    if kernel_performance_df['MaxPrivateScore'].std() > 0:
                        kernel_performance_df['NormalizedPrivateScore'] = zscore(kernel_performance_df['MaxPrivateScore'])
                    else:
                        kernel_performance_df['NormalizedPrivateScore'] = 0 # Or handle as appropriate

                    logging.info("Submission performance metrics extracted and normalized.")
                else:
                    logging.warning("No submission score data or relevant columns for normalization.")
            else:
                logging.warning("'KernelVersionId' column not found in submissions_df or submissions_df is empty. Skipping submission performance merge.")

            self.kernel_performance_df = kernel_performance_df.drop_duplicates(subset=['KernelId'])
            self.kernel_performance_df.to_parquet('kernel_performance_metrics.parquet', index=False)
            logging.info("Kernel performance metrics saved to kernel_performance_metrics.parquet.")
        else:
            logging.error("Error: 'KernelId' column not found in kernel_versions_df. Cannot merge for performance metrics.")
            self.kernel_performance_df = pd.DataFrame() # Ensure it's an empty DataFrame

        clean_memory()
        metrics_end_time = time.time()
        logging.info(f"Performance metrics extraction and normalization completed in {metrics_end_time - metrics_start_time:.2f} seconds.")

    def generate_leaderboards(self):
        """Generates leaderboards based on performance metrics."""
        leaderboard_start_time = time.time()
        logging.info("Generating leaderboards...")

        if not self.kernel_performance_df.empty and 'AuthorUserId' in self.kernel_performance_df.columns:
            # Top users by normalized engagement score
            top_engagement = self.kernel_performance_df.groupby('AuthorUserId')['NormalizedEngagementScore'].mean().nlargest(10)
            logging.info("\nTop 10 Users by Average Normalized Kernel Engagement:")
            logging.info(top_engagement)

            # Top users by normalized public score (if available)
            if 'NormalizedPublicScore' in self.kernel_performance_df.columns:
                top_public_score = self.kernel_performance_df.groupby('AuthorUserId')['NormalizedPublicScore'].max().nlargest(10)
                logging.info("\nTop 10 Users by Max Normalized Public Score:")
                logging.info(top_public_score)
            else:
                logging.warning("Normalized Public Score not available for leaderboard.")

            # Top users by normalized private score (if available)
            if 'NormalizedPrivateScore' in self.kernel_performance_df.columns:
                top_private_score = self.kernel_performance_df.groupby('AuthorUserId')['NormalizedPrivateScore'].max().nlargest(10)
                logging.info("\nTop 10 Users by Max Normalized Private Score:")
                logging.info(top_private_score)
            else:
                logging.warning("Normalized Private Score not available for leaderboard.")
        else:
            logging.warning("No performance metrics data to generate leaderboards.")

        clean_memory()
        leaderboard_end_time = time.time()
        logging.info(f"Leaderboard generation completed in {leaderboard_end_time - leaderboard_start_time:.2f} seconds.")

    def visualize_performance_metrics(self):
        """Generates visualizations for performance metrics."""
        if self.kernel_performance_df.empty:
            logging.warning("No kernel performance data to visualize.")
            return

        logging.info("Generating performance metrics visualizations...")

        # Distribution of Total Votes
        if 'TotalVotes' in self.kernel_performance_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.kernel_performance_df['TotalVotes'], bins=50, kde=True)
            plt.title('Distribution of Total Votes for Kernels')
            plt.xlabel('Total Votes')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
            plt.savefig('total_votes_distribution.png')
            plt.close()
            logging.info("Total votes distribution plot generated.")
        else:
            logging.warning("'TotalVotes' column not found for visualization.")

        # Distribution of Total Comments
        if 'TotalComments' in self.kernel_performance_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.kernel_performance_df['TotalComments'], bins=50, kde=True)
            plt.title('Distribution of Total Comments for Kernels')
            plt.xlabel('Total Comments')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
            plt.savefig('total_comments_distribution.png')
            plt.close()
            logging.info("Total comments distribution plot generated.")
        else:
            logging.warning("'TotalComments' column not found for visualization.")

        # Relationship between Votes and Comments
        if 'TotalVotes' in self.kernel_performance_df.columns and 'TotalComments' in self.kernel_performance_df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='TotalVotes', y='TotalComments', data=self.kernel_performance_df, alpha=0.6)
            plt.title('Total Votes vs. Total Comments')
            plt.xlabel('Total Votes')
            plt.ylabel('Total Comments')
            plt.tight_layout()
            plt.show()
            plt.savefig('votes_vs_comments_scatterplot.png')
            plt.close()
            logging.info("Votes vs. comments scatterplot generated.")
        else:
            logging.warning("Missing 'TotalVotes' or 'TotalComments' for scatterplot.")

        # Distribution of Public Scores
        if 'PublicScore' in self.kernel_performance_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.kernel_performance_df['PublicScore'].dropna(), bins=50, kde=True)
            plt.title('Distribution of Public Scores')
            plt.xlabel('Public Score')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
            plt.savefig('public_score_distribution.png')
            plt.close()
            logging.info("Public score distribution plot generated.")
        else:
            logging.warning("'PublicScore' column not found for visualization.")

        # Distribution of Private Scores
        if 'PrivateScore' in self.kernel_performance_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.kernel_performance_df['PrivateScore'].dropna(), bins=50, kde=True)
            plt.title('Distribution of Private Scores')
            plt.xlabel('Private Score')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
            plt.savefig('private_score_distribution.png')
            plt.close()
            logging.info("Private score distribution plot generated.")
        else:
            logging.warning("'PrivateScore' column not found for visualization.")

        # Top 10 Users by Normalized Engagement Score
        if 'NormalizedEngagementScore' in self.kernel_performance_df.columns and 'AuthorUserId' in self.kernel_performance_df.columns:
            top_engagement = self.kernel_performance_df.groupby('AuthorUserId')['NormalizedEngagementScore'].mean().nlargest(10)
            if not top_engagement.empty:
                plt.figure(figsize=(12, 7))
                sns.barplot(x=top_engagement.values, y=top_engagement.index.astype(str), palette='coolwarm')
                plt.title('Top 10 Users by Average Normalized Kernel Engagement')
                plt.xlabel('Average Normalized Engagement Score')
                plt.ylabel('User ID')
                plt.tight_layout()
                plt.show()
                plt.savefig('top_users_engagement.png')
                plt.close()
                logging.info("Top users by engagement plot generated.")
            else:
                logging.warning("No data for top users by engagement.")
        else:
            logging.warning("Missing data for normalized engagement score or author user ID for visualization.")

        clean_memory()
        logging.info("Performance metrics visualizations completed.")

    def run_pipeline(self):
        """Runs the performance benchmarking pipeline."""
        start_time = time.time()
        logging.info("Starting Performance Benchmarking Pipeline...")
        self.load_data()
        self.extract_and_normalize_metrics()
        self.generate_leaderboards()
        self.visualize_performance_metrics() # Call visualization method
        clean_memory()
        end_time = time.time()
        logging.info(f"Performance Benchmarking Pipeline Completed in {end_time - start_time:.2f} seconds.")

# --- Pipeline 5: Collaboration & Social Network Analysis Pipeline ---

class CollaborationNetworkAnalyzer:
    """
    Analyzes user collaboration patterns and social networks.
    """
    def __init__(self, config): # Accept config object
        self.config = config
        self.kernels_df = None
        self.kernel_versions_df = None
        self.forum_messages_df = None
        self.forum_topics_df = None
        self.users_df = None
        self.co_author_graph = nx.Graph()
        self.follower_graph = nx.DiGraph()
        self.forum_graph = nx.DiGraph()

    def load_data(self):
        """Loads necessary CSV data for network analysis."""
        load_start_time = time.time()
        logging.info("Loading data for Collaboration & Social Network Analysis...")
        try:
            self.kernels_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Kernels.csv'), low_memory=False) # Use config
            self.kernel_versions_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'KernelVersions.csv'), low_memory=False) # Use config
            self.forum_messages_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'ForumMessages.csv'), low_memory=False) # Use config
            self.forum_topics_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'ForumTopics.csv'), low_memory=False) # Use config
            self.users_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Users.csv'), low_memory=False) # Use config

            # Ensure relevant columns exist and handle NaNs
            self.kernels_df.dropna(subset=['AuthorUserId'], inplace=True)
            self.kernel_versions_df.dropna(subset=['KernelId'], inplace=True)
            self.forum_messages_df.dropna(subset=['AuthorUserId', 'ForumTopicId'], inplace=True)
            self.forum_topics_df.dropna(subset=['AuthorUserId'], inplace=True)

            logging.info("Data loaded.")
        except FileNotFoundError as e:
            logging.error(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.config.ROOT_PATH_CSV}") # Use config
            self.kernels_df = pd.DataFrame(columns=['Id', 'AuthorUserId'])
            self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId'])
            self.forum_messages_df = pd.DataFrame(columns=['Id', 'ForumTopicId', 'AuthorUserId', 'PostDate'])
            self.forum_topics_df = pd.DataFrame(columns=['Id', 'AuthorUserId'])
            self.users_df = pd.DataFrame(columns=['Id', 'DisplayName'])
        clean_memory()
        load_end_time = time.time()
        logging.info(f"Data loading for Collaboration & Social Network Analysis completed in {load_end_time - load_start_time:.2f} seconds.")

    def build_user_graphs(self):
        """Builds co-author and follower graphs."""
        graph_start_time = time.time()
        logging.info("Building user graphs...")

        co_author_graph = nx.Graph()
        if not self.kernels_df.empty:
            authors = self.kernels_df['AuthorUserId'].unique()
            co_author_graph.add_nodes_from(authors)

            if not self.forum_messages_df.empty:
                topic_authors = self.forum_messages_df.groupby('ForumTopicId')['AuthorUserId'].apply(list)
                for topic_id, authors_list in tqdm(topic_authors.items(), desc="Building co-author edges from forum topics"):
                    for i, author1 in enumerate(authors_list):
                        for j, author2 in enumerate(authors_list):
                            if i < j:
                                co_author_graph.add_edge(author1, author2)
            self.co_author_graph = co_author_graph
            logging.info(f"Co-author graph built with {self.co_author_graph.number_of_nodes()} nodes and {self.co_author_graph.number_of_edges()} edges.")
            nx.write_gexf(self.co_author_graph, 'co_author_graph.gexf')
            logging.info("Co-author graph saved.")
        else:
            logging.warning("No kernel data to build co-author graph.")

        follower_graph = nx.DiGraph()
        if not self.users_df.empty:
            follower_graph.add_nodes_from(self.users_df['Id'].unique())
            sample_users = random.sample(list(self.users_df['Id'].unique()), min(1000, len(self.users_df['Id'].unique())))
            for _ in range(2000):
                try:
                    u1, u2 = random.sample(sample_users, 2)
                    if u1 != u2:
                        follower_graph.add_edge(u1, u2)
                except ValueError:
                    break
            self.follower_graph = follower_graph
            logging.info(f"Follower graph built with {self.follower_graph.number_of_nodes()} nodes and {self.follower_graph.number_of_edges()} edges.")
            nx.write_gexf(self.follower_graph, 'follower_graph.gexf')
            logging.info("Follower graph saved.")
        else:
            logging.warning("No user data to build follower graph.")

        clean_memory()
        graph_end_time = time.time()
        logging.info(f"User graphs built in {graph_end_time - graph_start_time:.2f} seconds.")
        return self.co_author_graph, self.follower_graph

    def build_forum_graph(self):
        """Builds a graph based on forum interactions (e.g., replies)."""
        forum_graph_start_time = time.time()
        logging.info("Building forum graph...")
        forum_graph = nx.DiGraph()

        if not self.forum_messages_df.empty:
            forum_graph.add_nodes_from(self.forum_messages_df['AuthorUserId'].unique())
            topic_messages = self.forum_messages_df.groupby('ForumTopicId')
            for topic_id, messages_in_topic in tqdm(topic_messages, desc="Processing forum messages"):
                authors_in_topic = messages_in_topic['AuthorUserId'].unique()
                for i, author1 in enumerate(authors_in_topic):
                    for j, author2 in enumerate(authors_in_topic):
                        if author1 != author2:
                            forum_graph.add_edge(author1, author2)

            self.forum_graph = forum_graph
            logging.info(f"Forum graph built with {self.forum_graph.number_of_nodes()} nodes and {self.forum_graph.number_of_edges()} edges.")
            nx.write_gexf(self.forum_graph, 'forum_graph.gexf')
            logging.info("Forum graph saved.")
        else:
            logging.warning("No forum messages data to build forum graph.")

        clean_memory()
        forum_graph_end_time = time.time()
        logging.info(f"Forum graph built in {forum_graph_end_time - forum_graph_start_time:.2f} seconds.")
        return self.forum_graph

    def compute_network_metrics(self, co_author_graph, follower_graph, forum_graph):
        """Computes centrality and community detection metrics."""
        metrics_start_time = time.time()
        logging.info("Computing network metrics...")

        # Co-author Graph Metrics
        logging.info("\nCo-author Graph Metrics:")
        if co_author_graph.number_of_nodes() > 0:
            pagerank_co_author = nx.pagerank(co_author_graph)
            top_pagerank_co_author = sorted(pagerank_co_author.items(), key=lambda item: item[1], reverse=True)[:10]
            logging.info(f"Top 10 PageRank (Co-author): {top_pagerank_co_author}")

            try:
                import community as co_louvain
                partition = co_louvain.best_partition(co_author_graph)
                num_communities = len(set(partition.values()))
                logging.info(f"Community detection (Louvain) found {num_communities} communities.")
            except ImportError:
                logging.warning("Community detection (Louvain) skipped. Install 'python-louvain' for this feature.")
            except Exception as e:
                logging.error(f"Error during Louvain community detection for co-author graph: {e}")
        else:
            logging.warning("Co-author graph is empty. Skipping metrics.")

        # Follower Graph Metrics
        logging.info("\nFollower Graph Metrics:")
        if follower_graph.number_of_nodes() > 0:
            in_degree_centrality_follower = nx.in_degree_centrality(follower_graph)
            top_in_degree_follower = sorted(in_degree_centrality_follower.items(), key=lambda item: item[1], reverse=True)[:10]
            logging.info(f"Top 10 In-Degree Centrality (Follower): {top_in_degree_follower}")
        else:
            logging.warning("Follower graph is empty. Skipping metrics.")

        # Forum Graph Metrics
        logging.info("\nForum Graph Metrics:")
        if forum_graph.number_of_nodes() > 0:
            pagerank_forum = nx.pagerank(forum_graph)
            top_pagerank_forum = sorted(pagerank_forum.items(), key=lambda item: item[1], reverse=True)[:10]
            logging.info(f"Top 10 PageRank (Forum): {top_pagerank_forum}")
        else:
            logging.warning("Forum graph is empty. Skipping metrics.")

        logging.info("Reports on 'most central' Kaggle contributors would be generated here, combining various metrics.")
        clean_memory()
        metrics_end_time = time.time()
        logging.info(f"Network metrics computation completed in {metrics_end_time - metrics_start_time:.2f} seconds.")

    def visualize_networks(self):
        """Generates visualizations for network analysis."""
        logging.info("Generating network visualizations...")

        # Co-author Graph Degree Distribution
        if self.co_author_graph.number_of_nodes() > 0:
            degree_sequence = sorted([d for n, d in self.co_author_graph.degree()], reverse=True)
            if degree_sequence:
                plt.figure(figsize=(10, 6))
                sns.histplot(degree_sequence, bins=50, kde=True)
                plt.title('Co-author Graph Degree Distribution')
                plt.xlabel('Degree')
                plt.ylabel('Frequency')
                plt.yscale('log') # Often power-law distribution
                plt.tight_layout()
                plt.show()
                plt.savefig('co_author_degree_distribution.png')
                plt.close()
                logging.info("Co-author graph degree distribution plot generated.")
            else:
                logging.warning("No degree data for co-author graph.")
        else:
            logging.warning("Co-author graph is empty. Skipping degree distribution plot.")

        # Follower Graph In-Degree Distribution
        if self.follower_graph.number_of_nodes() > 0:
            in_degree_sequence = sorted([d for n, d in self.follower_graph.in_degree()], reverse=True)
            if in_degree_sequence:
                plt.figure(figsize=(10, 6))
                sns.histplot(in_degree_sequence, bins=50, kde=True)
                plt.title('Follower Graph In-Degree Distribution')
                plt.xlabel('In-Degree')
                plt.ylabel('Frequency')
                plt.yscale('log')
                plt.tight_layout()
                plt.show()
                plt.savefig('follower_in_degree_distribution.png')
                plt.close()
                logging.info("Follower graph in-degree distribution plot generated.")
            else:
                logging.warning("No in-degree data for follower graph.")
        else:
            logging.warning("Follower graph is empty. Skipping in-degree distribution plot.")

        # Forum Graph Degree Distribution
        if self.forum_graph.number_of_nodes() > 0:
            forum_degree_sequence = sorted([d for n, d in self.forum_graph.degree()], reverse=True)
            if forum_degree_sequence:
                plt.figure(figsize=(10, 6))
                sns.histplot(forum_degree_sequence, bins=50, kde=True)
                plt.title('Forum Graph Degree Distribution')
                plt.xlabel('Degree')
                plt.ylabel('Frequency')
                plt.yscale('log')
                plt.tight_layout()
                plt.show()
                plt.savefig('forum_degree_distribution.png')
                plt.close()
                logging.info("Forum graph degree distribution plot generated.")
            else:
                logging.warning("No degree data for forum graph.")
        else:
            logging.warning("Forum graph is empty. Skipping degree distribution plot.")

        # Note: Visualizing the entire graph (e.g., using nx.draw) is very resource-intensive
        # for large graphs and often not very informative. Subgraph visualization or
        # aggregated views are usually preferred.

        clean_memory()
        logging.info("Network visualizations completed.")

    def run_pipeline(self):
        """Runs the collaboration and social network analysis pipeline."""
        start_time = time.time()
        logging.info("Starting Collaboration & Social Network Analysis Pipeline...")
        self.load_data()
        co_author_graph, follower_graph = self.build_user_graphs()
        forum_graph = self.build_forum_graph()
        self.compute_network_metrics(co_author_graph, follower_graph, forum_graph)
        self.visualize_networks() # Call visualization method
        clean_memory()
        end_time = time.time()
        logging.info(f"Collaboration & Social Network Analysis Pipeline Completed in {end_time - start_time:.2f} seconds.")

# --- Pipeline 6: Evolutionary Pathways Pipeline ---

class EvolutionaryPathwaysAnalyzer:
    """
    Analyzes how kernels and approaches evolve over time.
    """
    def __init__(self, config, notebook_stats_df): # Accept config object
        self.config = config
        self.notebook_stats_df = notebook_stats_df
        self.kernels_df = None
        self.kernel_versions_df = None
        self.kernel_forks_df = None # Assuming a KernelForks.csv or similar
        self.kernel_dependencies_df = None # Assuming a KernelDependencies.csv or similar
        self.tripartite_graph = nx.Graph()

    def load_data(self):
        """Loads necessary CSV data for evolutionary pathways analysis."""
        load_start_time = time.time()
        logging.info("Loading data for Evolutionary Pathways...")
        try:
            self.kernels_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Kernels.csv'), low_memory=False) # Use config
            self.kernel_versions_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'KernelVersions.csv'), low_memory=False) # Use config
            # Placeholder for KernelForks and KernelDependencies if they exist
            # self.kernel_forks_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'KernelForks.csv'), low_memory=False)
            # self.kernel_dependencies_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'KernelDependencies.csv'), low_memory=False)

            self.kernels_df.dropna(subset=['Id', 'AuthorUserId', 'CreationDate'], inplace=True)
            self.kernel_versions_df.dropna(subset=['KernelId', 'VersionNumber', 'CreationDate'], inplace=True)

            logging.info("Data loaded.")
        except FileNotFoundError as e:
            logging.error(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.config.ROOT_PATH_CSV}") # Use config
            self.kernels_df = pd.DataFrame(columns=['Id', 'AuthorUserId', 'CreationDate'])
            self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId', 'VersionNumber', 'CreationDate'])
            self.kernel_forks_df = pd.DataFrame(columns=['ForkingKernelId', 'OriginalKernelId'])
            self.kernel_dependencies_df = pd.DataFrame(columns=['KernelId', 'DependencyKernelId'])
        clean_memory()
        load_end_time = time.time()
        logging.info(f"Data loading for Evolutionary Pathways completed in {load_end_time - load_start_time:.2f} seconds.")

    def construct_tripartite_graph(self):
        """
        Constructs a tripartite graph of users, kernels, and topics.
        This is a complex graph and needs careful memory management.
        """
        graph_start_time = time.time()
        logging.info("Constructing tripartite graph...")
        tripartite_graph = nx.Graph()

        # Add User nodes
        if not self.kernels_df.empty:
            user_ids = self.kernels_df['AuthorUserId'].unique()
            tripartite_graph.add_nodes_from(user_ids, bipartite=0, type='user')
            logging.info(f"Added {len(user_ids)} user nodes.")
        else:
            logging.warning("No kernel data to add user nodes to tripartite graph.")

        # Add Kernel nodes and User-Kernel edges
        if not self.kernels_df.empty:
            kernels_with_versions = pd.merge(self.kernels_df, self.kernel_versions_df,
                                             left_on='Id', right_on='KernelId', how='inner', suffixes=('_kernel', '_version'))

            if len(kernels_with_versions) > self.config.TRI_GRAPH_SAMPLE_SIZE: # Use config
                logging.warning(f"Too many kernels ({len(kernels_with_versions)}). Sampling {self.config.TRI_GRAPH_SAMPLE_SIZE} for tripartite graph construction.") # Use config
                kernels_with_versions = kernels_with_versions.sample(n=self.config.TRI_GRAPH_SAMPLE_SIZE, random_state=self.config.RANDOM_STATE) # Use config

            for index, row in tqdm(kernels_with_versions.iterrows(), total=len(kernels_with_versions), desc="Adding kernel nodes and edges"):
                kernel_id = row['Id_kernel']
                author_id = row['AuthorUserId']
                tripartite_graph.add_node(kernel_id, bipartite=1, type='kernel', creation_date=row['CreationDate_kernel'])
                tripartite_graph.add_edge(author_id, kernel_id, relation='authored')
            logging.info(f"Added {kernels_with_versions['Id_kernel'].nunique()} kernel nodes and user-kernel edges.")
        else:
            logging.warning("No kernel data to add kernel nodes to tripartite graph.")

        # Add Topic nodes and Kernel-Topic edges
        try:
            topic_info_df = pd.read_csv('topic_terms.csv')
            topics_found = topic_info_df['Topic'].unique()
            tripartite_graph.add_nodes_from(topics_found, bipartite=2, type='topic')
            logging.info(f"Added {len(topics_found)} topic nodes.")

            if not self.notebook_stats_df.empty and not topic_info_df.empty:
                sample_kernels = self.kernels_df['Id'].sample(min(100, len(self.kernels_df)), random_state=self.config.RANDOM_STATE).tolist() # Use config
                sample_topics = topic_info_df['Topic'].sample(min(5, len(topic_info_df)), random_state=self.config.RANDOM_STATE).tolist() # Use config

                for k_id in sample_kernels:
                    if k_id in tripartite_graph:
                        for t_id in sample_topics:
                            if t_id in tripartite_graph:
                                if random.random() < 0.2:
                                    tripartite_graph.add_edge(k_id, t_id, relation='about')
            logging.info("Added dummy kernel-topic edges (replace with actual topic assignments).")

        except FileNotFoundError:
            logging.warning("topic_terms.csv not found. Cannot add topic nodes or kernel-topic edges.")
        except Exception as e:
            logging.error(f"Error adding topic nodes/edges to tripartite graph: {e}")

        self.tripartite_graph = tripartite_graph
        nx.write_gexf(self.tripartite_graph, 'tripartite_graph.gexf')
        logging.info("Tripartite graph saved to tripartite_graph.gexf.")
        clean_memory()
        graph_end_time = time.time()
        logging.info(f"Tripartite graph constructed in {graph_end_time - graph_start_time:.2f} seconds.")
        return self.tripartite_graph

    def analyze_evolutionary_paths(self, tripartite_graph):
        """Analyzes evolutionary paths within the tripartite graph."""
        analysis_start_time = time.time()
        logging.info("Analyzing evolutionary paths...")

        if tripartite_graph.number_of_nodes() == 0:
            logging.warning("Tripartite graph is empty. Skipping evolutionary path analysis.")
            return

        try:
            topic_info_df = pd.read_csv('topic_terms.csv')
            topic_id_to_name = dict(zip(topic_info_df['Topic'], topic_info_df['Name']))
        except FileNotFoundError:
            logging.warning("topic_terms.csv not found. Cannot map topic IDs to names.")
            topic_id_to_name = {}

        if 'type' in nx.get_node_attributes(tripartite_graph, 'type'):
            topic_nodes = [n for n, data in tripartite_graph.nodes(data=True) if data['type'] == 'topic']
            if topic_nodes:
                topic_popularity = {topic: len(list(tripartite_graph.neighbors(topic))) for topic in topic_nodes}
                sorted_topics = sorted(topic_popularity.items(), key=lambda item: item[1], reverse=True)

                logging.info("\nTop 5 Most Connected Topics and Associated Kernels/Users:")
                for topic_id, count in sorted_topics[:5]:
                    topic_name = topic_id_to_name.get(topic_id, f"Topic {topic_id}")
                    logging.info(f"  {topic_name} (Connections: {count}):")
                    connected_nodes = list(tripartite_graph.neighbors(topic_id))
                    kernels_for_topic = [n for n in connected_nodes if tripartite_graph.nodes[n].get('type') == 'kernel']
                    users_for_topic = [n for n in connected_nodes if tripartite_graph.nodes[n].get('type') == 'user']

                    logging.info(f"    Sample Kernels: {kernels_for_topic[:5]}")
                    logging.info(f"    Sample Users: {users_for_topic[:5]}")
            else:
                logging.warning("No topic nodes found in tripartite graph.")
        else:
            logging.warning("Node 'type' attribute not found in tripartite graph. Skipping topic analysis.")

        logging.info("Advanced evolutionary pathway analysis (e.g., temporal paths, diffusion) would be implemented here.")

        clean_memory()
        analysis_end_time = time.time()
        logging.info(f"Evolutionary path analysis completed in {analysis_end_time - analysis_start_time:.2f} seconds.")

    def visualize_evolutionary_paths(self):
        """Generates visualizations for evolutionary pathways."""
        if self.tripartite_graph.number_of_nodes() == 0:
            logging.warning("No tripartite graph to visualize evolutionary paths.")
            return

        logging.info("Generating evolutionary pathways visualizations...")

        # Distribution of Kernel Versions (if available and meaningful)
        if not self.kernel_versions_df.empty and 'VersionNumber' in self.kernel_versions_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.kernel_versions_df['VersionNumber'], bins=self.kernel_versions_df['VersionNumber'].max(), kde=False)
            plt.title('Distribution of Kernel Version Numbers')
            plt.xlabel('Version Number')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
            plt.savefig('kernel_version_distribution.png')
            plt.close()
            logging.info("Kernel version distribution plot generated.")
        else:
            logging.warning("No kernel version data to plot.")

        # Temporal evolution of kernel creation by author (example)
        if not self.kernels_df.empty and 'CreationDate' in self.kernels_df.columns and 'AuthorUserId' in self.kernels_df.columns:
            # Aggregate kernels by author and month
            author_monthly_kernels = self.kernels_df.groupby([self.kernels_df['CreationDate'].dt.to_period('M'), 'AuthorUserId']).size().unstack(fill_value=0)
            author_monthly_kernels.index = author_monthly_kernels.index.to_timestamp()

            # Plot for a few top authors (e.g., top 5 by total kernels)
            top_authors = self.kernels_df['AuthorUserId'].value_counts().nlargest(5).index
            if not top_authors.empty and not author_monthly_kernels.empty:
                plt.figure(figsize=(14, 7))
                for author_id in top_authors:
                    if author_id in author_monthly_kernels.columns:
                        plt.plot(author_monthly_kernels.index, author_monthly_kernels[author_id], label=f'Author {author_id}')
                plt.title('Monthly Kernel Creation Trend for Top Authors')
                plt.xlabel('Date')
                plt.ylabel('Number of Kernels Created')
                plt.legend(title='Author ID')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig('top_authors_kernel_creation_trend.png')
                plt.close()
                logging.info("Top authors kernel creation trend plot generated.")
            else:
                logging.warning("Not enough data to plot kernel creation trend for top authors.")
        else:
            logging.warning("Missing kernel creation data for temporal evolution plot.")


        # Note: Visualizing the tripartite graph itself is extremely complex and often not feasible
        # for real-world datasets due to the number of nodes and edges. It usually requires
        # interactive visualization tools or specialized graph visualization libraries.
        # For static plots, one might visualize small subgraphs or aggregated properties.
        logging.info("Direct visualization of the full tripartite graph is generally not feasible for large datasets.")

        clean_memory()
        logging.info("Evolutionary pathways visualizations completed.")

    def run_pipeline(self):
        """Runs the evolutionary pathways pipeline."""
        start_time = time.time()
        logging.info("Starting Evolutionary Pathways Pipeline...")
        self.load_data()
        tripartite_graph = self.construct_tripartite_graph()
        self.analyze_evolutionary_paths(tripartite_graph)
        self.visualize_evolutionary_paths() # Call visualization method
        clean_memory()
        end_time = time.time()
        logging.info(f"Evolutionary Pathways Pipeline Completed in {end_time - start_time:.2f} seconds.")

# --- Pipeline 7: Machine Learning Model for Kaggle Success Prediction ---

class KaggleSuccessPredictor:
    """
    Builds and evaluates machine learning models to predict Kaggle success.
    Success can be defined by various metrics (e.g., high score, high engagement).
    """
    def __init__(self, config, notebook_stats_df): # Accept config object
        self.config = config
        self.notebook_stats_df = notebook_stats_df
        self.users_df = None
        self.kernels_df = None
        self.submissions_df = None
        self.kernel_versions_df = None
        self.features_df = None
        self.target_variable = 'is_successful' # Example target
        self.X_test_scaled = None
        self.y_test = None
        self.model_results = {}

    def load_data(self):
        """Loads necessary data for ML model."""
        load_start_time = time.time()
        logging.info("Loading data for ML Model...")
        try:
            self.users_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Users.csv'), low_memory=False) # Use config
            self.kernels_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Kernels.csv'), low_memory=False) # Use config
            self.submissions_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'Submissions.csv'), low_memory=False) # Use config
            self.kernel_versions_df = pd.read_csv(os.path.join(self.config.ROOT_PATH_CSV, 'KernelVersions.csv'), low_memory=False) # Use config

            # Ensure date columns are parsed
            self.kernels_df['CreationDate'] = pd.to_datetime(self.kernels_df['CreationDate'], errors='coerce')
            self.submissions_df['SubmissionDate'] = pd.to_datetime(self.submissions_df['SubmissionDate'], errors='coerce')
            self.kernel_versions_df['CreationDate'] = pd.to_datetime(self.kernel_versions_df['CreationDate'], errors='coerce')

            self.kernels_df.dropna(subset=['CreationDate'], inplace=True)
            self.submissions_df.dropna(subset=['SubmissionDate'], inplace=True)
            self.kernel_versions_df.dropna(subset=['CreationDate'], inplace=True)

            logging.info("Data loaded.")
        except FileNotFoundError as e:
            logging.error(f"Error loading CSV file: {e}. Please ensure the CSV files are in {self.config.ROOT_PATH_CSV}") # Use config
            self.users_df = pd.DataFrame(columns=['Id', 'DisplayName'])
            self.kernels_df = pd.DataFrame(columns=['Id', 'AuthorUserId', 'CreationDate'])
            self.submissions_df = pd.DataFrame(columns=['Id', 'SubmissionDate', 'PublicScore', 'PrivateScore', 'KernelVersionId'])
            self.kernel_versions_df = pd.DataFrame(columns=['Id', 'KernelId', 'TotalVotes', 'TotalComments'])
        clean_memory()
        load_end_time = time.time()
        logging.info(f"Data loading for ML Model completed in {load_end_time - load_start_time:.2f} seconds.")

    def feature_engineering(self):
        """
        Engineers features for the ML model.
        Combines data from various sources.
        """
        fe_start_time = time.time()
        logging.info("Engineering features for ML model...")

        # Base features from users (e.g., number of public kernels, total votes received)
        user_features = self.users_df[['Id', 'DisplayName']].copy()
        user_features.rename(columns={'Id': 'AuthorUserId'}, inplace=True)

        # Aggregate kernel statistics per user
        if not self.kernels_df.empty:
            kernel_counts = self.kernels_df.groupby('AuthorUserId').size().reset_index(name='num_kernels')
            user_features = pd.merge(user_features, kernel_counts, on='AuthorUserId', how='left').fillna(0)

        # Aggregate kernel version statistics (votes, comments) per user
        if not self.kernel_versions_df.empty:
            # Merge kernel versions with kernels to link to author
            kernel_version_agg = pd.merge(self.kernel_versions_df, self.kernels_df[['Id', 'AuthorUserId']],
                                          left_on='KernelId', right_on='Id', how='inner', suffixes=('_version', '_kernel'))
            kernel_version_agg.dropna(subset=['AuthorUserId'], inplace=True)

            user_total_votes = kernel_version_agg.groupby('AuthorUserId')['TotalVotes'].sum().reset_index(name='total_votes_received')
            user_total_comments = kernel_version_agg.groupby('AuthorUserId')['TotalComments'].sum().reset_index(name='total_comments_received')

            user_features = pd.merge(user_features, user_total_votes, on='AuthorUserId', how='left').fillna(0)
            user_features = pd.merge(user_features, user_total_comments, on='AuthorUserId', how='left').fillna(0)

        # Incorporate notebook_stats_df (from Pipeline 1)
        if not self.notebook_stats_df.empty and not self.kernels_df.empty:
            # A more robust mapping would be needed here. For now, skipping direct merge.
            logging.warning("Skipping direct merge of notebook_stats_df due to lack of direct KernelId mapping.")

        # Define a target variable: e.g., 'is_successful' if user has a high public score or high engagement
        if 'total_votes_received' in user_features.columns and not user_features['total_votes_received'].empty:
            median_votes = user_features['total_votes_received'].median()
            user_features[self.target_variable] = (user_features['total_votes_received'] > median_votes).astype(int)
            logging.info(f"Target variable '{self.target_variable}' created based on total_votes_received > {median_votes}.")
        else:
            user_features[self.target_variable] = 0
            logging.warning(f"Could not create target variable '{self.target_variable}'. Defaulting to 0.")

        self.features_df = user_features.set_index('AuthorUserId')
        self.features_df = self.features_df.select_dtypes(include=np.number)
        self.features_df.dropna(inplace=True)

        logging.info(f"Features DataFrame shape: {self.features_df.shape}")
        logging.info(f"Features DataFrame head:\n{self.features_df.head()}")

        clean_memory()
        fe_end_time = time.time()
        logging.info(f"Feature engineering completed in {fe_end_time - fe_start_time:.2f} seconds.")

    def train_and_evaluate_model(self):
        """Trains and evaluates various ML models."""
        ml_start_time = time.time()
        logging.info("Training and evaluating ML models...")

        if self.features_df.empty or self.target_variable not in self.features_df.columns:
            logging.error("Features DataFrame is empty or target variable is missing. Skipping model training.")
            return

        X = self.features_df.drop(columns=[self.target_variable])
        y = self.features_df[self.target_variable]

        if X.empty or y.empty:
            logging.error("Features or target are empty after dropping NaNs. Skipping model training.")
            return

        X_train, self.X_test_scaled, y_train, self.y_test = train_test_split(X, y, test_size=self.config.ML_TEST_SIZE, random_state=self.config.RANDOM_STATE, stratify=y) # Use config

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(self.X_test_scaled) # Store scaled test set for visualization

        models = {
            'RandomForestClassifier': RandomForestClassifier(random_state=self.config.RANDOM_STATE), # Use config
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=self.config.RANDOM_STATE), # Use config
            'MLPClassifier': MLPClassifier(random_state=self.config.RANDOM_STATE, max_iter=self.config.ML_MAX_EPOCHS), # Use config
            'LightGBM': lgb.LGBMClassifier(random_state=self.config.RANDOM_STATE), # Use config
            'XGBoost': xgb.XGBClassifier(random_state=self.config.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'), # Use config
            'TensorFlow_ANN': 'placeholder'
        }

        best_model = None
        best_accuracy = -1
        self.model_results = {} # Store results in instance variable

        for name, model in tqdm(models.items(), desc="Training models"):
            step_start_time = time.time()
            logging.info(f"\nTraining {name}...")

            if name == 'TensorFlow_ANN':
                input_dim = X_train_scaled.shape[1]
                tf_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
                    tf.keras.layers.Dropout(self.config.ML_ANN_DROPOUT_RATE), # Use config
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dropout(self.config.ML_ANN_DROPOUT_RATE), # Use config
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                tf_model.compile(optimizer=Adam(learning_rate=self.config.ML_ANN_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy']) # Use config

                early_stopping = EarlyStopping(monitor='val_loss', patience=self.config.ML_ANN_PATIENCE, restore_best_weights=True) # Use config
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=self.config.ML_ANN_REDUCE_LR_PATIENCE, min_lr=0.00001) # Use config
                model_checkpoint = ModelCheckpoint('best_tf_model.keras', save_best_only=True, monitor='val_loss', mode='min')

                try:
                    tf_model.fit(X_train_scaled, y_train,
                                 epochs=self.config.ML_MAX_EPOCHS, # Use config
                                 batch_size=self.config.ML_BATCH_SIZE, # Use config
                                 validation_split=0.2,
                                 callbacks=[early_stopping, reduce_lr, model_checkpoint],
                                 verbose=0)
                    y_pred_proba = tf_model.predict(self.X_test_scaled).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    accuracy = accuracy_score(self.y_test, y_pred)
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                    logging.info(f"TensorFlow ANN Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
                    self.model_results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc, 'model': tf_model, 'y_pred_proba': y_pred_proba}
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = tf_model
                except Exception as e:
                    logging.error(f"Error training TensorFlow ANN: {e}")
                    self.model_results[name] = {'accuracy': 0, 'roc_auc': 0, 'model': None, 'y_pred_proba': np.array([])}
            else:
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(self.X_test_scaled)
                    accuracy = accuracy_score(self.y_test, y_pred)
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                    else:
                        y_pred_proba = y_pred
                        roc_auc = accuracy
                    logging.info(f"{name} Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
                    self.model_results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc, 'model': model, 'y_pred_proba': y_pred_proba}
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                except Exception as e:
                    logging.error(f"Error training {name}: {e}")
                    self.model_results[name] = {'accuracy': 0, 'roc_auc': 0, 'model': None, 'y_pred_proba': np.array([])}

            clean_memory()
            step_end_time = time.time()
            logging.info(f"Step (Model Training - {name}) completed in {step_end_time - step_start_time:.2f} seconds.")

        logging.info("\n--- Model Performance Summary ---")
        for name, results in self.model_results.items():
            logging.info(f"{name}: Accuracy={results['accuracy']:.4f}, ROC AUC={results['roc_auc']:.4f}")

        # Save the best model
        logging.info("\nSaving the best model...")
        if best_model:
            if isinstance(best_model, tf.keras.Model):
                best_model.save('best_classification_model.keras')
                logging.info("Best Keras model saved as 'best_classification_model.keras'")
            else:
                joblib.dump(best_model, 'best_classification_model.pkl')
                logging.info("Best scikit-learn/LightGBM/XGBoost model saved as 'best_classification_model.pkl'")
        else:
            logging.warning("No best model found to save.")

        clean_memory()
        ml_end_time = time.time()
        logging.info(f"Machine Learning Model Pipeline Completed in {ml_end_time - ml_start_time:.2f} seconds.")

    def visualize_ml_results(self):
        """Generates visualizations for ML model results."""
        if self.features_df.empty:
            logging.warning("No features data to visualize for ML results.")
            return

        logging.info("Generating ML results visualizations...")

        # Distribution of numerical features
        numeric_cols = self.features_df.select_dtypes(include=np.number).columns.tolist()
        if self.target_variable in numeric_cols:
            numeric_cols.remove(self.target_variable) # Remove target from feature distribution plots

        for col in numeric_cols:
            if not self.features_df[col].empty:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.features_df[col], bins=30, kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.show()
                plt.savefig(f'feature_distribution_{col}.png')
                plt.close()
                logging.info(f"Distribution plot for {col} generated.")
            else:
                logging.warning(f"Column '{col}' is empty for distribution plot.")

        # Correlation Matrix Heatmap
        if not self.features_df.empty and len(self.features_df.columns) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.features_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.show()
            plt.savefig('feature_correlation_heatmap.png')
            plt.close()
            logging.info("Feature correlation heatmap generated.")
        else:
            logging.warning("Not enough features or data to generate correlation heatmap.")

        # ROC Curve for all models
        if self.y_test is not None and not self.y_test.empty:
            plt.figure(figsize=(10, 8))
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

            for name, results in self.model_results.items():
                if results['model'] is not None and results['y_pred_proba'].size > 0:
                    fpr, tpr, thresholds = roc_curve(self.y_test, results['y_pred_proba'])
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {results["roc_auc"]:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for ML Models')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig('ml_roc_curve.png')
            plt.close()
            logging.info("ROC curve for ML models generated.")
        else:
            logging.warning("Test data or model predictions missing for ROC curve.")

        clean_memory()
        logging.info("ML results visualizations completed.")

    def run_pipeline(self):
        """Runs the machine learning model pipeline."""
        start_time = time.time()
        logging.info("Starting Machine Learning Model Pipeline...")
        self.load_data()
        self.feature_engineering()
        self.train_and_evaluate_model()
        self.visualize_ml_results() # Call visualization method
        clean_memory()
        end_time = time.time()
        logging.info(f"Machine Learning Model Pipeline Completed in {end_time - start_time:.2f} seconds.")

# --- Main entry point for execution ---
if __name__ == '__main__':
    overall_start_time = time.time()
    logging.info("--- Starting Comprehensive Meta Kaggle Analysis ---")

    # Initialize configuration
    config = Config()

    # Initialize notebook_stats_df (output of Pipeline 1) to pass to other pipelines
    notebook_stats_df = pd.DataFrame()

    try:
        # Pipeline 1: Code Metadata Extraction
        step_start_time = time.time()
        logging.info("\n--- Running Pipeline 1: Code Metadata Extraction ---")
        code_extractor = CodeMetadataExtractor(config) # Pass config
        notebook_stats_df = code_extractor.run_pipeline()
        step_end_time = time.time()
        logging.info(f"Pipeline 1 (Code Metadata Extraction) took {step_end_time - step_start_time:.2f} seconds.")
        clean_memory()

        # Pipeline 2: Temporal Trend Analysis
        step_start_time = time.time()
        logging.info("\n--- Running Pipeline 2: Temporal Trend Analysis ---")
        temporal_analyzer = TemporalTrendAnalyzer(config, notebook_stats_df) # Pass config
        temporal_analyzer.run_pipeline()
        step_end_time = time.time()
        logging.info(f"Pipeline 2 (Temporal Trend Analysis) took {step_end_time - step_start_time:.2f} seconds.")
        clean_memory()

        # Pipeline 3: Topic Modeling & NLP
        step_start_time = time.time()
        logging.info("\n--- Running Pipeline 3: Topic Modeling & NLP ---")
        topic_modeler = TopicModelingNLP(config) # Pass config
        topic_modeler.run_pipeline()
        step_end_time = time.time()
        logging.info(f"Pipeline 3 (Topic Modeling & NLP) took {step_end_time - step_start_time:.2f} seconds.")
        clean_memory()

        # Pipeline 4: Performance Benchmarking
        step_start_time = time.time()
        logging.info("\n--- Running Pipeline 4: Performance Benchmarking ---")
        performance_benchmarker = PerformanceBenchmarker(config) # Pass config
        performance_benchmarker.run_pipeline()
        step_end_time = time.time()
        logging.info(f"Pipeline 4 (Performance Benchmarking) took {step_end_time - step_start_time:.2f} seconds.")
        clean_memory()

        # Pipeline 5: Collaboration & Social Network Analysis
        step_start_time = time.time()
        logging.info("\n--- Running Pipeline 5: Collaboration & Social Network Analysis ---")
        collaboration_analyzer = CollaborationNetworkAnalyzer(config) # Pass config
        collaboration_analyzer.run_pipeline()
        step_end_time = time.time()
        logging.info(f"Pipeline 5 (Collaboration & Social Network Analysis) took {step_end_time - step_start_time:.2f} seconds.")
        clean_memory()

        # Pipeline 6: Evolutionary Pathways
        step_start_time = time.time()
        logging.info("\n--- Running Pipeline 6: Evolutionary Pathways ---")
        evolutionary_analyzer = EvolutionaryPathwaysAnalyzer(config, notebook_stats_df) # Pass config
        evolutionary_analyzer.run_pipeline()
        step_end_time = time.time()
        logging.info(f"Pipeline 6 (Evolutionary Pathways) took {step_end_time - step_start_time:.2f} seconds.")
        clean_memory()

        # Pipeline 7: Machine Learning Model for Kaggle Success Prediction
        step_start_time = time.time()
        logging.info("\n--- Running Pipeline 7: Machine Learning Model for Kaggle Success Prediction ---")
        ml_predictor = KaggleSuccessPredictor(config, notebook_stats_df) # Pass config
        ml_predictor.run_pipeline()
        step_end_time = time.time()
        logging.info(f"Pipeline 7 (Machine Learning Model) took {step_end_time - step_start_time:.2f} seconds.")
        clean_memory()

    except Exception as e:
        logging.critical(f"An unhandled error occurred during the overall pipeline execution: {e}", exc_info=True)
    finally:
        overall_end_time = time.time()
        logging.info(f"\n--- Comprehensive Meta Kaggle Analysis Completed in {overall_end_time - overall_start_time:.2f} seconds ---")
        clean_memory()




