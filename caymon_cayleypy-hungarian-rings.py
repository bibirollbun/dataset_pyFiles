from typing import Any, List, Dict
import pandas as pd
from ast import literal_eval
from sympy.combinatorics import Permutation

import time
import joblib
import itertools
import math
from pathlib import Path
import csv
import json


!pip install git+https://github.com/cayleypy/cayleypy -q
!pip install numba


import torch
from cayleypy import CayleyGraph, prepare_graph
from cayleypy.algo.bfs_numpy import bfs_numpy
import numpy as np
import gc


from typing import Any, List
from cayleypy.puzzles.hungarian_rings import hungarian_rings_permutations
from cayleypy.puzzles.puzzles import Puzzles


print(4, 1, 4, 1)
print(hungarian_rings_permutations(left_size=4, left_index=1, right_size=4, right_index=1))
print(6, 2, 5, 1)
print(hungarian_rings_permutations(6, 2, 5, 1))


puzzle_info = pd.read_csv('/kaggle/input/santa-2023/puzzle_info.csv')
puzzle_info = puzzle_info[puzzle_info['puzzle_type'].str.contains("wreath_")]
puzzle_info['allowed_moves'] = puzzle_info['allowed_moves'].apply(literal_eval)

wreath_moves = {}
for _, row in puzzle_info.iterrows():
    wreath_moves[row["puzzle_type"]] = [row["allowed_moves"]["l"], row["allowed_moves"]["r"]]

wreath_moves.keys()

wreath_parameters = {
    'wreath_6/6': (6, 2, 6, 3),
    'wreath_7/7': (7, 2, 7, 3),
    'wreath_12/12': (12, 3, 12, 4),
    'wreath_21/21': (21, 6, 21, 7),
    'wreath_33/33': (33, 9, 33, 10),
    'wreath_100/100': (100, 25, 100, 26),
}

def test_wreath(name, output, expected):
    print(name)
    # print("L", output[0]) 
    assert all(x == y for x, y in zip(output[0], expected[0]))
    # print("R", output[1])
    assert all(x == y for x, y in zip(output[1], expected[1]))

for name, parameters in wreath_parameters.items():
    permutations = hungarian_rings_permutations(*parameters)
    test_wreath(name, permutations, wreath_moves[name])


graph_def = Puzzles.hungarian_rings(4, 1, 5, 2)
graph = CayleyGraph(graph_def)
bfs_numpy(graph)


def calculate_layer_sizes_numpy(left_size: int, left_index: int, right_size: int, right_index: int):
    graph_def = Puzzles.hungarian_rings(left_size, left_index, right_size, right_index)
    graph = CayleyGraph(graph_def)
    curr = time.time()
    layer_sizes = bfs_numpy(graph)
    return layer_sizes, round(time.time() - curr, 2)


def read_dump(file: Path):
    result = {}    
    if file.exists():
        with file.open(mode='r', encoding='utf-8') as csvfile:
            for key, value in csv.reader(csvfile):
                parameters = tuple(map(int, key.split(",")))
                result[parameters] = json.loads(value)
    return result


def write_dump(file: Path, data: Dict):
    to_write = {}
    for source_key, value in data.items():
        if source_key[0] > source_key[2] or (source_key[0] == source_key[2] and source_key[1] > source_key[3]):
            to_write[(source_key[2], source_key[3], source_key[0], source_key[1])] = value
        else:
            to_write[source_key] = value   
            
    if dump_path.exists():
        dump_path.unlink()
    with file.open("w", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for k, v in to_write.items():
            writer.writerow((','.join([str(x) for x in k]), v))
    

github_data_path = Path('/usr/local/lib/python3.11/dist-packages/cayleypy/data/hungarian_rings_growth.csv')
github_data = read_dump(github_data_path)
dump_path = Path('/kaggle/working/hungar_dict_cpu.csv')
dump_data = {}
# if dump_path.exists():
#     # dump_path.unlink()
#     dump_data = read_dump(dump_path)
calculation_result = github_data | dump_data


def get_calculation_result(left_size: int, left_index: int, right_size: int, right_index: int):
    parameters = (left_size, left_index, right_size, right_index)
    clone = (right_size, right_index, left_size, left_index)
    if parameters in calculation_result:
        return calculation_result[parameters]
    elif clone in calculation_result:
        return calculation_result[clone]
    else:
        return None


def get_layer_sizes(left_size: int, left_index: int, right_size: int, right_index: int, verbose: bool=False):    
    parameters = (left_size, left_index, right_size, right_index)
    layer_sizes = get_calculation_result(*parameters)
    # print(parameters)
    if layer_sizes is None:
        layer_sizes, seconds = calculate_layer_sizes_numpy(*parameters)
        if verbose:
            print("params", parameters, "time:", seconds, "max_layer:", np.max(layer_sizes), "sum:", sum(layer_sizes),  "diameter:", len(layer_sizes))
        calculation_result[parameters] = layer_sizes
    else:
        if verbose:
            print("params", parameters, "max_layer:", np.max(layer_sizes), "sum:", sum(layer_sizes), "diameter:", len(layer_sizes))
    return layer_sizes


max_right_size = 6
max_left_size = 6
min_ring_size = 3

#with clones
for left_ring_size in range(min_ring_size, max_left_size + 1):
    for right_ring_size in range(min_ring_size, max_right_size + 1):
        for left_index in range(0, left_ring_size//2 + 1):
            if left_index == 0:
                parameters = (left_ring_size, 0, right_ring_size, 0)
                if parameters not in calculation_result:
                    layer_sizes, seconds = calculate_layer_sizes_numpy(*parameters)
                    calculation_result[parameters] = layer_sizes
            else:
                for right_index in range(1, right_ring_size//2 + 1 ): 
                    parameters = (left_ring_size, left_index, right_ring_size, right_index)
                    if parameters not in calculation_result:
                        layer_sizes, seconds = calculate_layer_sizes_numpy(*parameters)
                        calculation_result[parameters] = layer_sizes



variants = [
    (range(7, 10), 3), # 12
    (range(7, 10), 4),
    (range(7, 10), 5) # 10
]

verbose = False

for left_ring_size_variants, right_ring_size in variants:
    for left_ring_size in left_ring_size_variants:
        for left_index in range(1, left_ring_size//2 + 1):
            for right_index in range(1, right_ring_size//2 + 1): 
                parameters = (left_ring_size, left_index, right_ring_size, right_index)
                layer_sizes = get_layer_sizes(*parameters, verbose=verbose)
                dump_data[parameters] = layer_sizes
                write_dump(dump_path, dump_data)

calculation_result = calculation_result | dump_data


import matplotlib.pyplot as plt
for ring_size in range(4, 7):
    for index in range(1, ring_size//2 + 1):
        key = (ring_size, index, ring_size, index)
        layer_sizes = calculation_result[key]
        if ring_size % 2 == 0 and index == ring_size/2:
            plt.plot(np.log(layer_sizes), linestyle="--", label=f"ring_size={ring_size} index={index}")
        else:
            plt.plot(np.log(layer_sizes), label=f"ring_size={ring_size} index={index}")

print("Symmetrical rings")
for parameters in [(4, 1, 4, 1), (4, 2, 4, 2), (6, 1, 6, 1), (6, 2, 6, 2), (6, 3, 6, 3)]:
    layer_sizes = calculation_result[parameters]
    print("params", parameters, "max_layer:", np.max(layer_sizes), "diameter:", len(layer_sizes))

plt.legend(fontsize=7)
plt.grid()
plt.show()


groups = {
    (5, 1, 4, 2) : [(5, 1, 4, 1)],
    (6, 3, 6, 1) : [(6, 1, 6, 1), (6, 2, 6, 1)],
}

for main, similar in groups.items():
    layer_sizes = calculation_result[main]
    plt.plot(np.log(layer_sizes), label=f"params={main}")

    for item in similar:
        layer_sizes = calculation_result[item]
        plt.plot(np.log(layer_sizes), linestyle="--", label=f"params={item}")

plt.legend(fontsize=7)
plt.grid()
plt.show()


import matplotlib.pyplot as plt
from cayleypy.puzzles.hungarian_rings import hungarian_rings_generators
from cayleypy.cayley_graph_def import CayleyGraphDef

def calculate_layer_sizes_numpy_without_checks(left_size: int, left_index: int, right_size: int, right_index: int):
    generators, generator_names = hungarian_rings_generators(left_size, left_index, right_size, right_index)
    n = len(generators[0])
    name = f"hungarian_rings-{left_size}-{left_index}-{right_size}-{right_index}"
    graph_def = CayleyGraphDef.create(
        generators, central_state=list(range(n)), generator_names=generator_names, name=name
    )
    graph = CayleyGraph(graph_def)
    curr = time.time()
    layer_sizes = bfs_numpy(graph)
    return layer_sizes, round(time.time() - curr, 2)

twins = {
    (4, 1, 3, 1) : [(4, 3, 3, 1),(4, 1, 3, 2)],
    (6, 1, 5, 2) : [(6, 5, 5, 2), (5, 2, 6, 1)],
    (6, 1, 6, 2) : [(6, 5, 6, 2), (6, 1, 6, 4), (6, 2, 6, 1)]
}

for main, similar in twins.items():
    layer_sizes = calculation_result[main]
    plt.plot(np.log(layer_sizes), label=f"params={main}")

    for twin in similar:
        if twin in calculation_result:
            layer_sizes = calculation_result[twin]
        else:
            calculate_layer_sizes_numpy_without_checks(*twin)
        plt.plot(np.log(layer_sizes), linestyle="--", label=f"params={twin}")

plt.legend(fontsize=7)
plt.grid()
plt.show()


# for main, similar in twins.items():
#     for s_params in similar:
#         assert all(x == y for x, y in zip(calculation_result[main], calculation_result[s_params]))


import matplotlib.pyplot as plt

left_ring_size = 9
right_ring_size = 5
right_index = 1

for left_index in range(1, left_ring_size//2 + 1):
    parameters = (left_ring_size, left_index, right_ring_size, right_index)
    layer_sizes = calculation_result[parameters]
    plt.plot(np.log(layer_sizes), label=f"params={parameters}")

right_index = 2
for left_index in range(1, left_ring_size//2 + 1):
    parameters = (left_ring_size, left_index, right_ring_size, right_index)
    layer_sizes = calculation_result[parameters]
    plt.plot(np.log(layer_sizes), linestyle="--", label=f"params={parameters}")

plt.legend(fontsize=7)
plt.grid()
plt.show()


groups = list(itertools.product([7, 8], [3, 4, 5])) + list(itertools.product([9], [3, 4, 5])) + [(6, 6)]
for left_ring_size, right_ring_size in groups:
    min_d = 0
    min_indexes = []
    max_d = 0
    max_indexes = []
    for left_index in range(1, (left_ring_size + 1)//2):
        for right_index in range(1, (right_ring_size + 1)//2): 
            parameters = (left_ring_size, left_index, right_ring_size, right_index)
            layer_sizes = calculation_result[parameters]
            d = len(layer_sizes)
            if min_d == 0 or d < min_d:
                min_d = d
                min_indexes = [(left_index, right_index)]
            elif d == min_d:
                min_indexes.append((left_index, right_index))

            if max_d < d:
                max_d = d
                max_indexes = [(left_index, right_index)]
            elif max_d == d:
                max_indexes.append((left_index, right_index))
    print(left_ring_size, right_ring_size, "min_indexes:", min_indexes, "max_indexes", max_indexes)   


right_ring_size = 5
verbose = False
right_index = 1
left_index = 1

for left_ring_size in range(7, 10):
    parameters = (left_ring_size, left_index, right_ring_size, right_index)
    get_layer_sizes(*parameters, verbose=verbose)
    with dump_path.open("wb") as f:
        joblib.dump(calculation_result, f)


import matplotlib.pyplot as plt

right_ring_size = 5
right_index = 1
left_index = 1

params_list = []
n_list = []
max_layers = []
sum_list = []
diameters = []

for left_ring_size in range(4, 10):
    left_index = 1
    parameters = (left_ring_size, left_index, right_ring_size, right_index)
    layer_sizes = calculation_result[parameters]
    
    params_list.append(parameters)
    n_list.append(left_ring_size + right_ring_size - 2)
    max_layers.append(np.max(layer_sizes))
    sum_list.append(sum(layer_sizes))
    diameters.append(len(layer_sizes))
    # print("params", parameters, "max_layer:", np.max(layer_sizes), "diameter:", len(layer_sizes))
    
    plt.plot(np.log(layer_sizes), label=f"{parameters}")

    left_index = 2
    parameters = (left_ring_size, left_index, right_ring_size, right_index)
    layer_sizes = calculation_result[parameters]
    plt.plot(np.log(layer_sizes), linestyle="--", alpha=0.5, label=f"{parameters}")

df = pd.DataFrame({'params': params_list, "n": n_list, 'max_layer': max_layers, 'sum':sum_list, 'diameters': diameters} )
df["n!"] = df["n"].apply(lambda x: math.factorial(x))
df["n!/2"] = df["n"].apply(lambda x: math.factorial(x)//2)
print(df.to_string(index=False))

plt.legend(fontsize=10, loc=2)
plt.xlabel("Distance")
plt.ylabel("Number of States(log scale)")
plt.grid()
plt.savefig('hungarian_rings_fixed_size_growth.png')
plt.show()



verbose = False
param_list = []
for ring_size in range(4, 8):
    left_index = ring_size//3
    right_index = left_index + 1
    parameters = (ring_size, left_index, ring_size, right_index)
    get_layer_sizes(*parameters, verbose=verbose)
    param_list.append(parameters)
    with dump_path.open("wb") as f:
        joblib.dump(calculation_result, f)


import matplotlib.pyplot as plt

params_list = []
n_list = []
max_layers = []
sum_list = []
diameters = []

for parameters in param_list:
    layer_sizes = calculation_result[parameters]
    
    params_list.append(parameters)
    n_list.append(parameters[0] + parameters[2] - 2)
    max_layers.append(np.max(layer_sizes))
    sum_list.append(sum(layer_sizes))
    diameters.append(len(layer_sizes))
    
    plt.plot(np.log(layer_sizes), label=f"{parameters}")

df = pd.DataFrame({'params': params_list, 'n': n_list, 'max_layer': max_layers, 'sum':sum_list, 'diameter': diameters} )
df["n!"] = df["n"].apply(lambda x: math.factorial(x))
df["n!/2"] = df["n"].apply(lambda x: math.factorial(x)//2)
print(df.to_string(index=False))

plt.legend(fontsize=10, loc=2)
plt.xlabel("Distance")
plt.ylabel("Number of States(log scale)")
plt.grid()
plt.savefig('hungarian_rings_santa_growth.png')
plt.show()



import networkx as nx
np.set_printoptions(threshold=np.inf)

params_list = [(3, 1, 3, 1), (3, 1, 4, 1), (4, 1, 4, 1), (4, 1, 5, 1), (5, 1, 5, 1)]
for_drawing = [(3, 1, 3, 1), (3, 1, 4, 1), (4, 1, 4, 1)]
for_clipping = [(3, 1, 4, 1), (4, 1, 4, 1)]
for_eigenvalues = [(3, 1, 3, 1), (3, 1, 4, 1), (4, 1, 4, 1), (4, 1, 5, 1)]

for parameters in params_list:
    n = parameters[0] + parameters[2] - 2
    print(f"\n{'-' * 30}\nParameters: {parameters}...")
    graph_def = Puzzles.hungarian_rings(*parameters)
    graph = CayleyGraph(graph_def)
    bfs_result = graph.bfs(return_all_edges=True, return_all_hashes=True)

    adj_matrix = bfs_result.adjacency_matrix()

    # Вычисление собственных значений
    eigenvalues = np.linalg.eigvalsh(adj_matrix)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]  # Сначала сортируем
    
    if parameters in for_eigenvalues:
        eigenvalues_rounded = np.round(eigenvalues_sorted, 2)

        # Подсчет кратностей
        unique_vals, counts = np.unique(eigenvalues_rounded, return_counts=True)
        multiplicities = {val: counts[i] for i, val in enumerate(unique_vals)}

        print(f"Eigenvalues with multiplicities (n = {n}):")
        eigenvals_str = "[" +", ".join([f"{x}^{{{y}}}" for x,y in zip(unique_vals, counts)]) + "]"
        print(eigenvals_str)
    
    plt.figure(figsize=(8, 5))
    plt.hist(eigenvalues_sorted, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Spectrum of Cayley Graph for n = {n} {parameters}")
    plt.xlabel("Eigenvalues")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    param_str = "".join([str(p) for p in list(parameters)])
    plt.savefig(f'hungarian_rings_spectrum_{n}_{param_str}.png')
    plt.show()

    
    if parameters in for_drawing:
        print(f"Drawing graph for n = {n}...")
        G = nx.from_numpy_array(adj_matrix)
            
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        nx.draw(G, pos, node_size=30, with_labels=False, alpha=0.8)
        plt.title(f"Cayley Graph for n = {n} {parameters}")
        plt.savefig(f'hungarian_rings_graph_{n}.png')
        plt.show()

    if parameters in for_clipping:
        bfs_result = graph.bfs(return_all_edges=True, return_all_hashes=True, max_diameter=3)
    
        adj_matrix = bfs_result.adjacency_matrix()        
        G = nx.from_numpy_array(adj_matrix)
            
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, k=0.15, iterations=320)
        nx.draw(G, pos, node_size=30, with_labels=False, alpha=0.8)
        plt.title(f"Firs layers of Cayley Graph for n = {n} {parameters}")
        plt.show()




