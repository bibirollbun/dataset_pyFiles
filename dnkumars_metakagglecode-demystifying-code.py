import polars as pl
from IPython.display import IFrame
import kagglehub
import os
import plotly.express as px
import polars as pl
from pathlib import Path
import json
import glob
import os
import codecs


MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")
print("âœ… Downloaded Meta-Kaggle data.")
print("ğŸ“‚ MK_PATH =", MK_PATH)
print("ğŸ“‚ MKC_PATH =", MKC_PATH)


Kernels = pl.read_csv("/kaggle/input/meta-kaggle/Kernels.csv")
print(Kernels.columns)
print(Kernels.shape)
Kernels.head()


KernelVersions = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv")
print(KernelVersions.columns)
print(KernelVersions.shape)
KernelVersions.head()


kernels = pl.read_csv(f"{MK_PATH}/Kernels.csv")
kernels = kernels.with_columns(pl.col("Id").cast(pl.Int64))
kernels = kernels.with_columns(pl.col("Id").alias("KernelId"))

versions = pl.read_csv(f"{MK_PATH}/KernelVersions.csv")
versions = versions.with_columns(pl.col("Id").cast(pl.Int64))
versions = versions.with_columns(pl.col("ScriptId").cast(pl.Int64))


def get_version_by_id(version_id: int) -> pl.DataFrame:
    return versions.filter(pl.col("Id") == version_id)

def get_kernel_by_id(kernel_id: int) -> pl.DataFrame:
    return kernels.filter(pl.col("KernelId") == kernel_id)


def id_to_path(file_id: int) -> str:
    padded_id_str = str(file_id).zfill(10)
    prefix = f"{MKC_PATH}/{padded_id_str[0:4]}/{padded_id_str[4:7]}/{file_id}.*"
    matching_paths = glob.glob(prefix)
    return matching_paths[0] if len(matching_paths) == 1 else ""

def path_to_id(file_path: str) -> int | None:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return int(base_name) if base_name.isdigit() else None

def get_file_extension(file_path: str) -> str:
    return os.path.splitext(file_path)[1]

def get_ipynb_source(ipynb_path: str) -> list[list[str]]:
    with codecs.open(ipynb_path, 'r', encoding='utf-8') as f:
        raw_source_content = f.read()
    json_data = json.loads(raw_source_content)
    return [cell.get('source', []) for cell in json_data.get('cells', []) if cell.get('cell_type') == 'code']

def get_source_code(file_path: str) -> list[list[str]]:
    ext = get_file_extension(file_path)
    if ext == ".ipynb":
        return get_ipynb_source(file_path)
    return [Path(file_path).read_text().splitlines()]

def get_source_code_by_id(file_id: int) -> list[list[str]] | None:
    file_path = id_to_path(file_id)
    return get_source_code(file_path) if file_path else None

def get_first_n_files(path, n=10):
    found = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            found.append(filename)
            if len(found) >= n:
                return found
    return found


kernel_1000 = KernelVersions.filter(pl.col("Id") == 1000)
kernel_1000


id = 1000
source = get_source_code_by_id(id)

file_path_for_id = id_to_path(id)
print(f"\nPath for ID {id}: {file_path_for_id}")
print(f"ID from path: {path_to_id(file_path_for_id)}")
print(f"File extension: {get_file_extension(file_path_for_id)}")
if source and source[0]:
    print("First 10 lines:\n" + "\n".join(source[0][:10]))

# Get version and kernel metadata
version_info = get_version_by_id(id)
if not version_info.is_empty():
    print("\nVersion Info:")
    print(version_info)
    script_id = version_info.select("ScriptId")[0, 0]
    kernel_info = get_kernel_by_id(script_id)
    print("\nKernel Info:")
    print(kernel_info)


script_lang_counts = versions.select([
    pl.col("ScriptLanguageId")
]).group_by("ScriptLanguageId").len().sort("len", descending=True)

script_lang_counts.to_pandas()


KernelLanguages = pl.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')
KernelLanguages.to_pandas()


import polars as pl
MK_PATH = Path('/kaggle/input/meta-kaggle')
kv_csv = MK_PATH / 'KernelVersions.csv'
versions = pl.read_csv(kv_csv)
versions = versions.with_columns([
    pl.col("Id").cast(pl.Int64, strict=False),
    pl.col("ScriptId").cast(pl.Int64, strict=False),
    pl.col("CreationDate").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M:%S", strict=False).alias("ParsedCreationDate")
])
yearly_counts = (versions
    .filter(pl.col("ScriptLanguageId") == 8)
    .group_by(pl.col("ParsedCreationDate").dt.year().alias("Year"))
    .agg(Count=pl.col("Id").count())
    .sort("Year")
)
yearly_counts.to_pandas()


ipynb_file = "/kaggle/input/mkc-language-path-list/MKC_Language_list/ipynb_file_list.txt"
py_file = "/kaggle/input/mkc-language-path-list/MKC_Language_list/py_file_list.txt"
r_file = "/kaggle/input/mkc-language-path-list/MKC_Language_list/r_file_list.txt"
rmd_file = "/kaggle/input/mkc-language-path-list/MKC_Language_list/rmd_file_list.txt"
def read_file_list(filepath):
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]
ipynb_paths = read_file_list(ipynb_file)
py_paths = read_file_list(py_file)
r_paths = read_file_list(r_file)
rmd_paths = read_file_list(rmd_file)
print("IPYNB files:", ipynb_paths[:5])
print("Python files:", py_paths[:5])
print("R files:", r_paths[:5])
print("Rmd files:", rmd_paths[:5])


get_source_code('/kaggle/input/meta-kaggle-code/0111/437/111437241.ipynb')


path_to_id('/kaggle/input/meta-kaggle-code/0111/437/111437241.ipynb')


get_version_by_id(111437241)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
import os
import json
from collections import Counter
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)


notebook_list_path = "/kaggle/input/mkc-language-path-list/MKC_Language_list/ipynb_file_list.txt"
with open(notebook_list_path, "r") as f:
    notebook_paths = [line.strip() for line in f if line.strip()]

def extract_libraries(cell_content):
    libraries = []
    for cell in cell_content:
        if not isinstance(cell, list):
            lines = cell.split('\n')
        else:
            cell = '\n'.join(cell)
            lines = cell.split('\n')
        import_statements = [l.strip() for l in lines if l.startswith('import ') or l.startswith('from ')]
        library_names = [l.split()[1] for l in import_statements if len(l.split()) > 1]
        libraries.extend(library_names)
    return list(set(libraries))


collected_libraries = []
processed_files = []
markdown_counts = []
code_counts = []

total_files = min(100_000,len(notebook_paths))

for i in tqdm(range(total_files)):
    file_path = notebook_paths[i]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
            notebook_cells = notebook_data['cells']
            markdown_count = len([c for c in notebook_cells if c['cell_type'] == 'markdown'])
            code_count = len([c for c in notebook_cells if c['cell_type'] == 'code'])
            markdown_counts.append(markdown_count)
            code_counts.append(code_count)
            code_cell_content = [c['source'] for c in notebook_cells if c['cell_type'] == 'code']
            libraries = extract_libraries(code_cell_content)
            collected_libraries.extend(libraries)
            processed_files.append(file_path)
    except:
        continue


print(f"Successfully processed {len(processed_files)}, missed {total_files - len(processed_files)}")
library_freq = dict(Counter(collected_libraries))
data_frame = pd.DataFrame({'library': list(library_freq.keys()), 'frequency': list(library_freq.values())})
data_frame['primary_library'] = data_frame['library'].map(lambda x: x.split('.')[0])
data_frame['secondary_libraries'] = data_frame['library'].map(lambda x: '.'.join(x.split('.')[1:]))
top_libraries = data_frame.sort_values(by='frequency', ascending=False).iloc[:15]
fig_top = px.bar(top_libraries, x='library', y='frequency', title='Top 15 Python Libraries in Notebooks')
def get_sub_library_counts(df, library='torch'):
    subset_df = df[df['primary_library'] == library]
    subset_df = subset_df[subset_df['secondary_libraries'].str.len() > 0]
    return subset_df.sort_values(by='frequency', ascending=False).iloc[:10]
for lib, color in zip(['torch', 'sklearn', 'tensorflow', 'keras'], ['#eb4123', 'green', 'orange', 'brown']):
    fig = px.bar(get_sub_library_counts(data_frame, lib), x='secondary_libraries', y='frequency',
                 color_discrete_sequence=[color],
                 title=f'{lib} Sublibraries')
    iplot(fig)


fig_top.write_html("mkcipynb.html", include_plotlyjs="cdn")
display(IFrame("mkcipynb.html", width=1200, height=700))


total_markdown = sum(markdown_counts)
total_code = sum(code_counts)
avg_markdown = total_markdown / len(markdown_counts)
avg_code = total_code / len(code_counts)
cell_ratio = total_markdown / total_code
print(f'Average markdown cells: {avg_markdown:.2f}')
print(f'Average code cells: {avg_code:.2f}')
print(f'Markdown-to-code ratio: {cell_ratio:.3f}')


r_file_list = "/kaggle/input/mkc-language-path-list/MKC_Language_list/r_file_list.txt"
with open(r_file_list, "r") as f:
    r_file_paths = [line.strip() for line in f if line.strip()]

def extract_r_libraries(script_lines):
    libraries = []
    for line in script_lines:
        line = line.strip()
        if line.startswith("library(") or line.startswith("require("):
            parts = line.split("(")
            if len(parts) > 1:
                lib = parts[1].split(")")[0].strip().replace('"', '').replace("'", '')
                if lib:
                    libraries.append(lib)
        elif "library(" in line or "require(" in line:
            if "library(" in line:
                lib = line.split("library(")[1].split(")")[0].strip().replace('"', '').replace("'", '')
            elif "require(" in line:
                lib = line.split("require(")[1].split(")")[0].strip().replace('"', '').replace("'", '')
            if lib:
                libraries.append(lib)
    return libraries


collected_r_libraries = []
processed_r_files = []

total_r_files = len(r_file_paths)

for i in tqdm(range(total_r_files)):
    file_path = r_file_paths[i]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            script_lines = f.readlines()
            extracted_libs = extract_r_libraries(script_lines)
            collected_r_libraries.extend(extracted_libs)
            processed_r_files.append(file_path)
    except:
        continue

print(f"Successfully processed {len(processed_r_files)}, missed {total_r_files - len(processed_r_files)}")


library_counts = dict(Counter(collected_r_libraries))
r_data_frame = pd.DataFrame({'library': list(library_counts.keys()), 'frequency': list(library_counts.values())})
r_data_frame = r_data_frame.sort_values(by='frequency', ascending=False)

top_r_libraries = r_data_frame.head(15)
fig_r_top = px.bar(top_r_libraries, x='library', y='frequency', title='Top 15 R Libraries Used in .r Files')


fig_r_top.write_html("mkcr.html", include_plotlyjs="cdn")
display(IFrame("mkcr.html", width=1200, height=700))


python_file_list = "/kaggle/input/mkc-language-path-list/MKC_Language_list/py_file_list.txt"
with open(python_file_list, "r") as f:
    python_paths = [line.strip() for line in f if line.strip()]

def extract_python_libraries(script_lines):
    libraries = []
    for line in script_lines:
        line = line.strip()
        if line.startswith('import ') or line.startswith('from '):
            tokens = line.split()
            if len(tokens) >= 2:
                lib = tokens[1].split('.')[0]
                if lib.isidentifier():
                    libraries.append(lib)
    return libraries

collected_python_libraries = []
processed_python_files = []

total_files = min(100_000,len(python_paths))


for i in tqdm(range(total_files)):
    file_path = python_paths[i]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            script_lines = f.readlines()
            extracted_libs = extract_python_libraries(script_lines)
            collected_python_libraries.extend(extracted_libs)
            processed_python_files.append(file_path)
    except:
        continue

print(f"Successfully processed {len(processed_python_files)}, missed {total_files - len(processed_python_files)}")


library_frequencies = dict(Counter(collected_python_libraries))
python_data_frame = pd.DataFrame({'library': list(library_frequencies.keys()), 'frequency': list(library_frequencies.values())})
python_data_frame = python_data_frame.sort_values(by='frequency', ascending=False)

top_python_libraries = python_data_frame.head(15)
fig_python_top = px.bar(top_python_libraries, x='library', y='frequency', title='Top 15 Python Libraries Used in .py Files')


fig_python_top.write_html("mkcpy.html", include_plotlyjs="cdn")
display(IFrame("mkcpy.html", width=1200, height=700))

