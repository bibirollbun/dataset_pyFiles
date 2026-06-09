import kagglehub
import os
import nbformat
import pandas as pd
import sklearn.linear_model
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast

from datetime import datetime
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy import array
from kagglehub import KaggleDatasetAdapter


meta_path = kagglehub.dataset_download("kaggle/meta-kaggle/versions/1772")
print("Meta Kaggle:", meta_path)
code_path = kagglehub.dataset_download("kaggle/meta-kaggle-code/versions/109")
print("Meta kaggle code:", code_path)


print("\n========= Meta Kaggle Code Chunks =========")
c = 0
for file in os.listdir(code_path):
    print(f'ğŸ—ƒï¸� {file}')
    c += 1
    if c == 10:
        print('ğŸ—ƒï¸�' + ' ...')
        break


kernel_versions_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle/versions/1772",
    "KernelVersions.csv",
)
kernel_versions_df.head()


kernel_versions_df['Imports'] = [[] for _ in range(len(kernel_versions_df))]
kernel_versions_df['MethodCalls'] = [[] for _ in range(len(kernel_versions_df))]
if kernel_versions_df.index.name != 'Id':
    kernel_versions_df.set_index('Id', inplace=True)
kernel_versions_df.head()


DATASET_PATH = '/kaggle/input/kaggles-most-used-packages-and-method-calls/meta-kaggle-code-packages-methods.csv'
SAVE_DATASET = False
MAX_FILES = 1_000_000_000
PYTHON_IMPORT_REGEX = re.compile(r'(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))')
PYTHON_METHOD_REGEX = re.compile(r'(?<!def\s)(?:\.|\b)([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
R_IMPORT_REGEX = re.compile(r'(?:library|require)\((?:[\'"]?)([a-zA-Z0-9_.]+)(?:[\'"]?)\)')

def extract_imports(code_text, language):
    if language == 'Python':
        matches = PYTHON_IMPORT_REGEX.findall(code_text)
        return list(set(filter(None, [m[0] or m[1] for m in matches])))
    elif language == 'R':
        return list(set(R_IMPORT_REGEX.findall(code_text)))
    else:
        return []
    
def extract_python_methods(code_text):
    method_candidates = PYTHON_METHOD_REGEX.findall(code_text)
    ignore_list = {
        'if', 'for', 'while', 'print', 'len', 'int', 'float', 'str',
        'range', 'list', 'dict', 'set', 'map', 'filter', 'open', 'input',
        'get', 'set', 'append', 'pop', 'join', 'split', 'format', 'type'
    }
    filtered = [m + '()' for m in method_candidates if m not in ignore_list]
    return list(set(filtered))


if Path(DATASET_PATH).exists():
    kernel_versions_df = pd.read_csv(DATASET_PATH, compression='zip', storage_options=None, encoding='utf-8') 
else:
    SAVE_DATASET = True
    valid_extensions = {'.py', '.ipynb', '.r', '.rmd'}
    file_counter = 0
    file_infos = []

    def process_file(file_path, ext, kernel_id):
        try:
            if ext == '.ipynb':
                with open(file_path, 'r', encoding='utf-8') as f:
                    nb_content = nbformat.read(f, as_version=4)
                content = "\n".join(
                    cell.source for cell in nb_content.cells 
                    if cell.cell_type == "code" and cell.source
                )
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            lang = 'Python' if ext in ['.py', '.ipynb'] else 'R'
            imports = extract_imports(content, lang)

            method_calls = []
            if lang == 'Python':
                method_calls = extract_python_methods(content)

            return (kernel_id, imports, method_calls)
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=512) as executor:
        futures = []
        for dirpath, _, filenames in os.walk(code_path):
            for filename in filenames:
                if file_counter >= MAX_FILES:
                    break
                _, ext = os.path.splitext(filename)
                if ext.lower() in valid_extensions:
                    try:
                        kernel_id = int(filename.replace(ext, ''))
                        file_path = os.path.join(dirpath, filename)
                        futures.append(executor.submit(process_file, file_path, ext, kernel_id))
                        file_counter += 1
                        if file_counter % 10000 == 0:
                            print(f"Queued {file_counter} files...")
                    except ValueError:
                        continue
            if file_counter >= MAX_FILES:
                break

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                kernel_id, imports, method_calls = result
                if kernel_id in kernel_versions_df.index:
                    kernel_versions_df.at[kernel_id, 'Imports'] = imports
                    kernel_versions_df.at[kernel_id, 'MethodCalls'] = method_calls
            if i % 10000 == 0:
                print(f"Processed {i} files...")


found_kernels_df = kernel_versions_df[kernel_versions_df['Imports'].str.len() > 0]

# After reloading the dataset, the arrays are often seen as strings, so cast them literally.
for col in ['Imports', 'MethodCalls']:
    if col in found_kernels_df.columns:
        found_kernels_df[col] = found_kernels_df[col].dropna().apply(ast.literal_eval) 
        
print(len(found_kernels_df))
found_kernels_df.head()


if SAVE_DATASET:
    found_kernels_df.to_csv(DATASET_PATH, index=False, compression=dict(method='zip', archive_name='meta-kaggle-code-packages-methods.csv'))
    print(f'Saved {len(found_kernels_df)} rows to {os.path.abspath(DATASET_PATH)}')


R_list = [1, 5, 12, 13, 15, 16]
python_list = [2, 8, 9, 14]


found_kernels_df['CreationDate'] = pd.to_datetime(found_kernels_df['CreationDate'], errors='coerce')
r_df = found_kernels_df[found_kernels_df['ScriptLanguageId'].isin(R_list)].copy()
py_df = found_kernels_df[found_kernels_df['ScriptLanguageId'].isin(python_list)].copy()

# -------------------- IMPORT FREQUENCY --------------------

r_imports = [pkg for sublist in r_df['Imports'].dropna() for pkg in sublist]
py_imports = [pkg for sublist in py_df['Imports'].dropna() for pkg in sublist]

r_counts = Counter(r_imports)
py_counts = Counter(py_imports)
total_counts = r_counts + py_counts

r_top = pd.DataFrame(r_counts.most_common(20), columns=["Package", "Count"])
py_top = pd.DataFrame(py_counts.most_common(20), columns=["Package", "Count"])
total_top = pd.DataFrame(total_counts.most_common(20), columns=["Package", "Count"])

def plot_top(df, title):
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, y="Package", x="Count", palette="viridis")
    plt.title(title)
    plt.xlabel("Import Count")
    plt.tight_layout()
    plt.show()

plot_top(py_top, "Top 20 Most Imported Python Packages")
plot_top(r_top, "Top 20 Most Imported R Packages")
plot_top(total_top, "Top 20 Most Imported Packages Overall")

# -------------------- METHOD CALL FREQUENCY --------------------

py_methodcalls = [m for sublist in py_df['MethodCalls'].dropna() for m in sublist]
py_method_counts = Counter(py_methodcalls)
py_method_top = pd.DataFrame(py_method_counts.most_common(50), columns=["Method", "Count"])

def plot_top_methods(df, title):
    plt.figure(figsize=(12,10))
    sns.barplot(data=df, y="Method", x="Count", palette="magma")
    plt.title(title)
    plt.xlabel("Method Call Count")
    plt.tight_layout()
    plt.show()

plot_top_methods(py_method_top, "Top 50 Most Used Python Method Calls")



# Count number of imports per kernel over time (per quarter)
def extract_time_series(df, name):
    df = df.copy()
    df = df.dropna(subset=["CreationDate", "Imports"])
    df["ImportCount"] = df["Imports"].apply(len)
    df["Quarter"] = df["CreationDate"].dt.to_period("Q").dt.to_timestamp()
    time_series = df.groupby("Quarter")["ImportCount"].sum()
    return time_series.rename(name)

py_series = extract_time_series(py_df, "Python")
r_series = extract_time_series(r_df, "R")

full_index = pd.date_range(start=min(py_series.index.min(), r_series.index.min()),
                           end=max(py_series.index.max(), r_series.index.max()),
                           freq='QS')  # QS = Quarter Start

py_series = py_series.reindex(full_index, fill_value=0)
r_series = r_series.reindex(full_index, fill_value=0)

time_df = pd.concat([py_series, r_series], axis=1)
time_df.plot(figsize=(12, 6), marker="o")
plt.title("Total Number of Imports Over Time (Quarterly)")
plt.ylabel("Total Imports")
plt.xlabel("Quarter")
plt.grid(True)
plt.tight_layout()
plt.show()



py_df = py_df.dropna(subset=["CreationDate", "Imports"]).copy()
r_df = r_df.dropna(subset=["CreationDate", "Imports"]).copy()
py_df["Quarter"] = py_df["CreationDate"].dt.to_period("Q").dt.to_timestamp()
r_df["Quarter"] = r_df["CreationDate"].dt.to_period("Q").dt.to_timestamp()

# Get top 5 packages overall
py_all_imports = [pkg for sublist in py_df["Imports"] for pkg in sublist]
r_all_imports = [pkg for sublist in r_df["Imports"] for pkg in sublist]
py_top5 = [pkg for pkg, _ in Counter(py_all_imports).most_common(10)]
r_top5 = [pkg for pkg, _ in Counter(r_all_imports).most_common(10)]

# Get per-quarter counts and smooth
def get_package_time_series(df, top_packages):
    quarter_index = pd.date_range(start=df["Quarter"].min(), end=df["Quarter"].max(), freq='QS')
    package_data = {pkg: [] for pkg in top_packages}
    
    for quarter in quarter_index:
        quarter_df = df[df["Quarter"] == quarter]
        imports = [pkg for sublist in quarter_df["Imports"] for pkg in sublist]
        counts = Counter(imports)
        for pkg in top_packages:
            package_data[pkg].append(counts.get(pkg, 0))
    
    df_out = pd.DataFrame(package_data, index=quarter_index)
    return df_out.rolling(window=2, min_periods=1).mean()

py_package_df = get_package_time_series(py_df, py_top5)
r_package_df = get_package_time_series(r_df, r_top5)

# Combine to find global min/max with actual data
combined_df = pd.concat([py_package_df, r_package_df], axis=1)
nonzero_mask = (combined_df > 0).any(axis=1)
trimmed_index = combined_df.index[nonzero_mask]

x_start = trimmed_index.min()
x_end = trimmed_index.max()

plt.figure(figsize=(14, 7))

# Plot Python packages with solid lines
for col in py_package_df.columns:
    plt.plot(py_package_df.index, py_package_df[col], label=f"Python: {col}", linestyle='-')

# Plot R packages with dashed lines
for col in r_package_df.columns:
    plt.plot(r_package_df.index, r_package_df[col], label=f"R: {col}", linestyle='--')

plt.title("Top 10 Python and R Package Imports per Quarter (Smoothed)")
plt.xlabel("Quarter")
plt.ylabel("Import Count (Smoothed)")
plt.grid(True)
plt.legend(title="Package")
plt.xlim(x_start, x_end)
plt.tight_layout()
plt.show()



# Distribution of number of imports per kernel
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

sns.histplot(py_df['Imports'].dropna().apply(len), bins=30, ax=axes[0], color='skyblue')
axes[0].set_title("Python Kernel Import Count Distribution")
axes[0].set_xlabel("Number of Imports")

sns.histplot(r_df['Imports'].dropna().apply(len), bins=30, ax=axes[1], color='salmon')
axes[1].set_title("R Kernel Import Count Distribution")
axes[1].set_xlabel("Number of Imports")

plt.suptitle("Distribution of Number of Packages Imported per Kernel")
plt.tight_layout()
plt.show()



# Track top 10 packages from total counts over time
top_5_packages = [pkg for pkg, _ in total_counts.most_common(10)]

records = []

for _, row in found_kernels_df.dropna(subset=["CreationDate", "Imports"]).iterrows():
    date = pd.to_datetime(row["CreationDate"]).to_period("Q").to_timestamp()
    for pkg in row["Imports"]:
        if pkg in top_5_packages:
            records.append((date, pkg))

trend_df = pd.DataFrame(records, columns=["Date", "Package"])
trend_counts = trend_df.groupby(["Date", "Package"]).size().unstack(fill_value=0)

trend_counts.plot(figsize=(12,6))
plt.title("Top 10 Imported Packages Over Time")
plt.ylabel("Import Count")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()




from collections import Counter

category_keywords = {
    "Visualization": [
        "matplotlib", "seaborn", "plotly", "bokeh", "altair", "ggplot2", "lattice", "shiny", "dash", "leaflet",
        "folium", "geopandas", "highcharter", "holoviews", "pygal", "vega", "vegalite", "pyecharts", "plotnine",
        "networkx", "ggraph", "graphviz", "dygraphs", "echarts4r", "canvasxpress", "heatmaply", "ggvis", "ggmap",
        "base", "corrplot", "matplotlib-inline", "ggbiplot", "cowplot", "ggpubr", "sjPlot", "sciplot", "ggthemes",
        "paletteer", "colorspace", "ggtext", "gganimate", "rbokeh", "patchwork", "plotrix", "maps", "tmap",
        "leafem", "mapview", "cartopy", "mplfinance", "missingno", "datatableplot"
    ],
    "Training": [
        "xgboost", "lightgbm", "catboost", "sklearn", "tensorflow", "keras", "pytorch", "fastai", "jax", "h2o",
        "spark", "caret", "tidymodels", "mlr3", "randomforest", "mxnet", "cntk", "mlpack", "theano", "torch",
        "glmnet", "rpart", "gbm", "nnet", "kernlab", "xlearn", "dl4j", "neuralnet", "ranger", "mljar", "tpot",
        "auto-sklearn", "flaml", "pycaret", "bigml", "scikit-multilearn", "bayes_opt", "optuna", "ray.tune",
        "hyperopt", "shap", "lime", "skorch", "deepchem", "cvxpy", "pymc3", "prophet", "paddlepaddle", "detectron2",
        "segmentation_models", "autokeras", "albumentations", "keras_tuner", "torchvision"
    ],
    "Data Science": [
        "pandas", "polars", "numpy", "scipy", "dask", "dplyr", "data.table", "tidyr", "readr", "lubridate",
        "pyspark", "statsmodels", "stringr", "nltk", "opencv", "text2vec", "tm", "quanteda", "arrow", "janitor",
        "skimr", "forcats", "magrittr", "reshape2", "tibble", "haven", "zoo", "xts", "feather", "snowballstemmer",
        "textclean", "textblob", "langdetect", "langid", "textstat", "dateutil", "fuzzywuzzy", "pyjanitor",
        "pandarallel", "datatable", "spacy", "wordcloud", "ftfy", "re", "regex", "beautifulsoup4", "lxml",
        "html5lib", "requests", "jsonlite", "httr", "tidytext"
    ],
    "ML/LLM Models": [
        "randomforestclassifier", "xgbclassifier", "transformers", "bert", "llama", "llama2", "gpt", "gpt2",
        "gpt3", "gpt4", "t5", "roberta", "distilbert", "electra", "albert", "bart", "deberta", "bloom", "falcon",
        "mistral", "claude", "gemma", "vicuna", "sagemaker", "openllm", "autogluon", "huggingface", "knn", "svm",
        "logisticregression", "linearregression", "naivebayes", "kmeans", "decisiontree", "gradientboosting",
        "sgdclassifier", "ridge", "lasso", "stackingclassifier", "baggingclassifier", "qda", "lda",
        "isolationforest", "dbscan", "mean_shift", "onevsrestclassifier", "xglm", "flan", "openchat", "command-r",
        "wizardcoder", "orcamini", "phi2", "mixtral", "marcod", "moondream", "baichuan", "chatglm"
    ]
}


flat_imports = [pkg.lower() for imports in found_kernels_df["Imports"].dropna() for pkg in imports]
all_counts = Counter(flat_imports)

category_counts = {}
for category, keywords in category_keywords.items():
    category_counts[category] = {pkg: all_counts[pkg] for pkg in keywords if pkg in all_counts}

category_counts



for category, counts in category_counts.items():
    if not counts:
        continue
    plt.figure(figsize=(14, 10))
    sns.barplot(x=list(counts.values()), y=list(counts.keys()))
    plt.xscale('log')
    plt.title(f"Most Used {category} Packages (Log Scale)")
    plt.xlabel("Log(Count)")
    plt.ylabel("Package")
    plt.tight_layout()
    plt.show()



compare_packages = ["torch", "tensorflow"]

records = []
for _, row in found_kernels_df.dropna(subset=["CreationDate", "Imports"]).iterrows():
    date = pd.to_datetime(row["CreationDate"]).to_period("Q").to_timestamp()
    for pkg in row["Imports"]:
        if pkg.lower() in compare_packages:
            records.append((date, pkg.lower()))

trend_df = pd.DataFrame(records, columns=["Date", "Package"])
trend_counts = trend_df.groupby(["Date", "Package"]).size().unstack(fill_value=0)

trend_counts.plot(figsize=(12, 6))
plt.title("Imports Over Time: Torch vs TensorFlow")
plt.ylabel("Import Count")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()




