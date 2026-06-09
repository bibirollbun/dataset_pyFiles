# I convert the tree to ete3 as I am more familiar with it's structure

import sys
!{sys.executable} -m pip install ete3
!{sys.executable} -m pip install fathomnet
from ete3 import Tree

import numpy as np
import pandas as pd

from tqdm import tqdm
from ete3 import Tree
from fathomnet.api import worms


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Get list of accepted categories

import json
with open('/kaggle/input/fathomnet-2025/dataset_train.json', 'r') as f:
    data = json.load(f)
classes = [x['name'] for x in data['categories']]
classes


def recursive_child_snatcher(anc):
    # Recursively gets a list of children and ranks from a fathomnet ancestor. 
    children = [x.name for x in anc.children]
    childrens_ranks = [x.rank for x in anc.children]
    
    assert len(children) == 1 # bifurcating trees not implemented
    if len(anc.children[0].children) > 0:
        childrens_children, childrens_childrens_ranks = recursive_child_snatcher(anc.children[0])
        return children + childrens_children, childrens_ranks + childrens_childrens_ranks
    else:
        return children, childrens_ranks

# convert to an ete3 Tree (This is personal preference as I have worked with them before)
tree = Tree()
already_added = ['']
for label in tqdm(classes):
    if label in already_added:
        continue

        
    anc = worms.get_ancestors(label)
    children, ranks = recursive_child_snatcher(anc)
    children = [''] + children
    ranks = [''] + ranks
    for i in range(len(children)-1):
        parent_name, child_name = children[i:i+2]
        parent_rank, child_rank = ranks[i:i+2]
        if child_name in already_added:
            continue

        parent_node = [node for node in tree.traverse() if node.name == parent_name][0]
        parent_node.rank = parent_rank
        child = Tree(name=child_name)
        child.rank = child_rank
        parent_node.add_child(child)
        already_added += [child_name]
print(tree)


# set distances to 0 for ranks not included in loss calculation
for node in tree.traverse():
    if node.name in classes:
        continue
    accepted_ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    if node.rank not in accepted_ranks:
        node.dist = 0




# write tree in newick format including distances
tree.write(outfile="tree.nh",format=3)


# make distance matrix
def tree_to_distance_matrix(tree, labels):
    n = len(labels)
    labels = sorted(labels)

    # Create a blank distance matrix
    dist_matrix = np.zeros((n, n))

    # Fill the matrix with pairwise distances
    for i, name1 in enumerate(labels):
        node1 = [node for node in tree.traverse() if node.name == name1][0]
        for j, name2 in enumerate(labels):
            if i <= j:

                d = node1.get_distance(str(name2))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d  # symmetric

    df = pd.DataFrame(dist_matrix, index=labels, columns=labels)

    return df
df = tree_to_distance_matrix(tree, classes)
df




