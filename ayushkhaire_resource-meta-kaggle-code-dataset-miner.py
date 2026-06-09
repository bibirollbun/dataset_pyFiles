import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import ast
import re
from IPython.display import display , HTML


def extract_imports(line):
    line = line.strip()

    # Remove inline comments
    line = re.sub(r"#.*", "", line).strip()

    # Handle multiline bracketed imports as single-line strings
    line = line.replace("\n", " ").replace("\r", " ")

    packages = set()
    functions = set()

    if line.startswith("import "):
        modules = line[7:].split(",")
        for module in modules:
            module = module.strip()
            if " as " in module:
                mod_name = module.split(" as ")[0].strip()
                packages.add(mod_name)
            else:
                packages.add(module)

    elif line.startswith("from "):
        try:
            parts = line.split("import")
            module_part = parts[0][5:].strip()  # after "from"
            imported_part = parts[1].strip()

            # Remove brackets and split
            imported_part = re.sub(r"[()\[\]{}]", "", imported_part)
            items = [item.strip() for item in imported_part.split(",")]
            for item in items:
                if " as " in item:
                    func = item.split(" as ")[0].strip()
                    functions.add(func)
                else:
                    functions.add(item)

            packages.add(module_part)
        except Exception as e:
            pass  # optionally log this

    return list(packages), list(functions)

def extract_methods_and_attributes(code):
    methods = set()
    attributes = set()

    class MethodAttributeVisitor(ast.NodeVisitor):
        def visit_Attribute(self, node):
            # Attribute access like df.head or np.array
            if isinstance(node.ctx, ast.Load):
                attr_name = node.attr
                if isinstance(node.parent, ast.Call) and node.parent.func == node:
                    methods.add(attr_name)
                else:
                    attributes.add(attr_name)
            self.generic_visit(node)

    # Attach parent references for method detection
    def add_parents(node, parent=None):
        for child in ast.iter_child_nodes(node):
            child.parent = node
            add_parents(child, child)

    try:
        tree = ast.parse(code)
        add_parents(tree)
        MethodAttributeVisitor().visit(tree)
    except SyntaxError:
        pass  # optionally log

    return sorted(methods), sorted(attributes)

def analyze_python_code_block(source, is_code_string=False):
    total_packages = set()
    total_functions = set()
    total_methods = set()
    total_attributes = set()

    if not is_code_string:
        with open(source, 'r') as f:
            content = f.read()
    else:
        content = source.strip()

    lines = content.splitlines()

    for line in lines:
        packages, functions = extract_imports(line)
        total_packages.update(packages)
        total_functions.update(functions)

    methods, attributes = extract_methods_and_attributes(content)
    total_methods.update(methods)
    total_attributes.update(attributes)

    return {
        "packages_count": len(total_packages),
        "packages": sorted(total_packages),

        "functions_count": len(total_functions),
        "functions": sorted(total_functions),

        "methods_count": len(total_methods),
        "methods": sorted(total_methods),

        "attributes_count": len(total_attributes),
        "attributes": sorted(total_attributes),

        "lines": len(lines),
        "letters": len(content),
        "words": len(content.split())
    }

code = '''
from numpy import array, linspace
import pandas as pd
df = pd.read_csv("file.csv")
print(df.head())
print(df.columns)
np.array([1, 2, 3])
'''
result = analyze_python_code_block(code, is_code_string=True)
print(result)


# linux command to find all the existing files
# ! find /kaggle/input/meta-kaggle-code -type f


# analyze the notebook cells

def analyze_notebook_code_cell(cell):
    try:
        if cell['cell_type'] != 'code':
            return None
        # Join multi-line cell source into a string
        code = ''.join(cell['source'])
        # Analyze the code using your existing logic
        cell_analysis = analyze_python_code_block(code, is_code_string=True)
        # Safely get metadata
        metadata = cell.get('metadata', {})
        cell_collasp = metadata.get('collapsed', False)
        cell_hidd_inp = metadata.get('_kg_hide-input', False)
        cell_exec_cnt = cell.get('execution_count', None)
        cell_opt_lng = len(cell.get('outputs', []))
        return {
            'cell_analysis': cell_analysis,
            'cell_collasp': cell_collasp,
            'cell_hidd_inp': cell_hidd_inp,
            'cell_exec_cnt': cell_exec_cnt,
            'cell_opt_lng': cell_opt_lng
        }
    except:
        pass

def analyze_notebook_markdown_cell(cell):
    if cell['cell_type'] != 'markdown':
        return None
    source = ''.join(cell['source'])  
    no_of_letters = len(source)
    no_of_lines = len(source.split('\n'))
    no_of_words = len(source.split())
    return {
        'letters': no_of_letters,
        'lines': no_of_lines,
        'words': no_of_words
    }


def analyze_the_notebook(notebook_path):
    notebook_meta_data = {
        # General
        'kaggle_code_location': None,
        'notebook_version': None,
        'language_name': None,
        'language_version': None,
        'total_cells': 0,
        'total_code_cells': 0,
        'total_markdown_cells': 0,
        'total_lines_in_notebook': 0,
        'total_words_in_notebook': 0,
        'total_letters_in_notebook': 0,

        # Code-related
        'total_lines_in_notebook_code': 0,
        'total_words_in_notebook_code': 0,
        'total_letters_in_notebook_code': 0,
        'total_packages_in_notebook_code': 0,
        'total_functions_used_in_notebook_code': 0,
        'total_attributes_in_notebook_code': 0,
        'total_methods_used_in_notebook_code': 0,
        'listed_string_of_packages_in_notebook_code': '',
        'listed_string_of_functions_used_in_notebook_code': '',
        'listed_string_of_attributes_in_notebook_code': '',
        'listed_string_of_methods_used_in_notebook_code': '',
        'total_collapsed_code_cells_in_notebook': 0,
        'total_hidden_input_code_cells_in_notebook': 0,
        'total_executed_cells_in_notebook': 0,
        'highest_executed_cell_no_in_notebook': 0,
        'total_output_cells_in_notebook': 0,

        # Markdown-related
        'total_lines_in_notebook_markdown': 0,
        'total_words_in_notebook_markdown': 0,
        'total_letters_in_notebook_markdown': 0,
    }

    with open(notebook_path) as f:
        content_dict = json.load(f)

    # Metadata
    notebook_meta_data['kaggle_code_location'] = notebook_path.replace('/kaggle/input/meta-kaggle-code', '').replace('.ipynb', '')
    notebook_meta_data['notebook_version'] = f"{content_dict.get('nbformat', '?')}.{content_dict.get('nbformat_minor', '?')}"
    lang_info = content_dict.get("metadata", {}).get("language_info", {})
    notebook_meta_data['language_name'] = lang_info.get('name')
    notebook_meta_data['language_version'] = lang_info.get('version')

    # Aggregators
    all_packages = set()
    all_functions = set()
    all_methods = set()
    all_attributes = set()

    for cell in content_dict.get("cells", []):
        notebook_meta_data['total_cells'] += 1

        if cell['cell_type'] == 'code':
            notebook_meta_data['total_code_cells'] += 1
            result = analyze_notebook_code_cell(cell)
            
            if result:
                ca = result['cell_analysis']
                notebook_meta_data['total_lines_in_notebook_code'] += ca['lines']
                notebook_meta_data['total_words_in_notebook_code'] += ca['words']
                notebook_meta_data['total_letters_in_notebook_code'] += ca['letters']
                all_packages.update(ca['packages'])
                all_functions.update(ca['functions'])
                all_methods.update(ca['methods'])
                all_attributes.update(ca['attributes'])

                if result['cell_collasp']:
                    notebook_meta_data['total_collapsed_code_cells_in_notebook'] += 1
                if result['cell_hidd_inp']:
                    notebook_meta_data['total_hidden_input_code_cells_in_notebook'] += 1
                if result['cell_exec_cnt'] is not None:
                    notebook_meta_data['total_executed_cells_in_notebook'] += 1
                    notebook_meta_data['highest_executed_cell_no_in_notebook'] = max(
                        notebook_meta_data['highest_executed_cell_no_in_notebook'],
                        result['cell_exec_cnt']
                    )
                notebook_meta_data['total_output_cells_in_notebook'] += result['cell_opt_lng']

        elif cell['cell_type'] == 'markdown':
            notebook_meta_data['total_markdown_cells'] += 1
            result = analyze_notebook_markdown_cell(cell)
            if result:
                notebook_meta_data['total_lines_in_notebook_markdown'] += result['lines']
                notebook_meta_data['total_words_in_notebook_markdown'] += result['words']
                notebook_meta_data['total_letters_in_notebook_markdown'] += result['letters']

    # Combined totals
    notebook_meta_data['total_lines_in_notebook'] = (
        notebook_meta_data['total_lines_in_notebook_code'] +
        notebook_meta_data['total_lines_in_notebook_markdown']
    )
    notebook_meta_data['total_words_in_notebook'] = (
        notebook_meta_data['total_words_in_notebook_code'] +
        notebook_meta_data['total_words_in_notebook_markdown']
    )
    notebook_meta_data['total_letters_in_notebook'] = (
        notebook_meta_data['total_letters_in_notebook_code'] +
        notebook_meta_data['total_letters_in_notebook_markdown']
    )

    # Final code stats
    notebook_meta_data['total_packages_in_notebook_code'] = len(all_packages)
    notebook_meta_data['total_functions_used_in_notebook_code'] = len(all_functions)
    notebook_meta_data['total_methods_used_in_notebook_code'] = len(all_methods)
    notebook_meta_data['total_attributes_in_notebook_code'] = len(all_attributes)

    notebook_meta_data['listed_string_of_packages_in_notebook_code'] = ', '.join(sorted(all_packages))
    notebook_meta_data['listed_string_of_functions_used_in_notebook_code'] = ', '.join(sorted(all_functions))
    notebook_meta_data['listed_string_of_methods_used_in_notebook_code'] = ', '.join(sorted(all_methods))
    notebook_meta_data['listed_string_of_attributes_in_notebook_code'] = ', '.join(sorted(all_attributes))

    return notebook_meta_data


result = analyze_the_notebook('/kaggle/input/meta-kaggle-code/0014/000/14000070.ipynb')
# print(result)


# analyze parent directory

def analyze_the_parent_directory(parent_dir_path):
    # Create an empty dictionary where each value is a list
    notebook_meta_data = {
        'kaggle_code_location': [],
        'notebook_version': [],
        'language_name': [],
        'language_version': [],
        'total_cells': [],
        'total_code_cells': [],
        'total_markdown_cells': [],
        'total_lines_in_notebook': [],
        'total_words_in_notebook': [],
        'total_letters_in_notebook': [],
        'total_lines_in_notebook_code': [],
        'total_words_in_notebook_code': [],
        'total_letters_in_notebook_code': [],
        'total_packages_in_notebook_code': [],
        'total_functions_used_in_notebook_code': [],
        'listed_string_of_packages_in_notebook_code': [],
        'listed_string_of_functions_used_in_notebook_code': [],
        'total_attributes_in_notebook_code': [],
        'total_methods_used_in_notebook_code': [],
        'listed_string_of_attributes_in_notebook_code': [],
        'listed_string_of_methods_used_in_notebook_code': [],
        'total_collapsed_code_cells_in_notebook': [],
        'total_hidden_input_code_cells_in_notebook': [],
        'total_executed_cells_in_notebook': [],
        'highest_executed_cell_no_in_notebook': [],
        'total_output_cells_in_notebook': [],
        'total_lines_in_notebook_markdown': [],
        'total_words_in_notebook_markdown': [],
        'total_letters_in_notebook_markdown': [],
    }

    # List all files in the parent directory
    total_files_in_the_directory = os.listdir(parent_dir_path)

    # Process each notebook
    for file in total_files_in_the_directory:
        notebook_path = os.path.join(parent_dir_path, file)
        if notebook_path.endswith('.ipynb'):
            try:
                result = analyze_the_notebook(notebook_path)
                for key in notebook_meta_data:
                    notebook_meta_data[key].append(result.get(key))
            except Exception as e:
                print(f"Error analyzing {notebook_path}: {e}")

    notebook_meta_data_df = pd.DataFrame(notebook_meta_data)
    return notebook_meta_data_df

# display(HTML(analyze_the_parent_directory('/kaggle/input/meta-kaggle-code/0011/029').to_html()))


# analyze the sub - directory

def analyze_sub_directory(sub_directory_path):
    sub_dir_dataframe = pd.DataFrame()
    total_folders_in_the_directory = os.listdir(sub_directory_path)
    for dirc in total_folders_in_the_directory:
        parent_path = sub_directory_path + '/' + dirc
        all_files_data_in_parent = analyze_the_parent_directory(parent_path)
        sub_dir_dataframe = pd.concat([sub_dir_dataframe,all_files_data_in_parent])
    return sub_dir_dataframe

# analyze_sub_directory('/kaggle/input/meta-kaggle-code/0011/')
# (32348,25)

# just for making sure 
# Walk through the directory tree and find .ipynb files
# ipynb_files = []
# for root, dirs, files in os.walk(parent_dir):
#     for fname in files:
#         if fname.endswith('.ipynb'):
#             full_path = os.path.join(root, fname)
#             ipynb_files.append(full_path)

# print(f"Found {len(ipynb_files)} notebook files:\n")
# Found 32438 notebook files:


# display(HTML(analyze_sub_directory('/kaggle/input/meta-kaggle-code/0011/').to_html()))


# finally , raid the root !

# def analyze_root_directory(root_directory_path):
#     root_dir_dataframe = pd.DataFrame()
#     total_folders_in_the_directory = os.listdir(root_directory_path)
#     for sub_dirc in tqdm(total_folders_in_the_directory,desc = f'analyzing ROOT directory {root_directory_path}'):
#         sub_dirc_path = root_directory_path + '/' + sub_dirc
#         all_sub_data = analyze_sub_directory(sub_dirc_path)
#         root_dir_dataframe = pd.concat([root_dir_dataframe,all_sub_data])
#     return root_dir_dataframe

# -------------------------------------

# linear compute do not work
# also it utilize cpu 20% most of time , max

# --------------------------------------


def assign_sub_dirs_in_pool(sub_dir_pool, no_threads):
    return [group.tolist() for group in np.array_split(sub_dir_pool, no_threads)]


sub_dir_pool = os.listdir('/kaggle/input/meta-kaggle-code')
no_threads = 40 # launnch 40 process

sub_dir_pool_groups = assign_sub_dirs_in_pool(sub_dir_pool, no_threads)
# print(f"Total groups formed: {len(sub_dir_pool_groups)}")
# Total groups formed: 20

def get_df_for_group_and_merge(the_group):
    group_df = pd.DataFrame()
    for sub_dirc in the_group :
        sub_dirc_path = '/kaggle/input/meta-kaggle-code' + '/' + sub_dirc
        all_sub_data = analyze_sub_directory(sub_dirc_path)
        group_df = pd.concat([group_df,all_sub_data])
    return group_df

# get_df_for_group_and_merge(
#     sub_dir_pool_groups[0] # first group
# )

# launch the threads

with ProcessPoolExecutor(max_workers=no_threads) as executor:
    results = list(executor.map(get_df_for_group_and_merge, sub_dir_pool_groups))


# Merge all group DataFrames
root_dir_dataframe = pd.concat(results, ignore_index=True)


# display(HTML(root_dir_dataframe.sample(10).to_html()))


root_dir_dataframe.to_csv('meta-kaggle-code-notebooks-metadata.csv',index = False)

