import json
import pandas as pd
import numpy as np


base_path='/kaggle/input/arc-prize-2025/'

# Loading JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


training_challenges   = load_json(base_path +'arc-agi_training_challenges.json')
training_solutions    = load_json(base_path +'arc-agi_training_solutions.json')

evaluation_challenges = load_json(base_path +'arc-agi_evaluation_challenges.json')
evaluation_solutions  = load_json(base_path +'arc-agi_evaluation_solutions.json')

test_challenges       = load_json(base_path +'arc-agi_test_challenges.json')


#Compare the Input Matrix dimension vs the output Matrix (all pairs of examples)
def eq_dim_matrix(challenge,pairs):
    eq_count = 0
    x =0
    for pair in range(pairs):
        if len(challenge[pair]['input']) == len(challenge[pair]['output']):
            if len(challenge[pair]['input'][0]) == len(challenge[pair]['output'][0]):
                eq_count = eq_count + 1
            else:
                eq_count = eq_count - 1
        else:
            eq_count = -1

    if eq_count == pairs:
        x = 1
    return x

# Validate if one the Matrix is a square one
def squared_matrix(matrix):
    x = 0
    if len(matrix) == len(matrix[0]):
        x = 1
    return x

#Validate if both Matrix are squared (Input and output of all pairs of examples)
def squared_matrix_full(challenge,pairs):
    sq_in, sq_out, sq_both, x, y  = 0,0,0,0,0
    for pair in range(pairs):
        x = x + squared_matrix(challenge[pair]['input'])
        y = y + squared_matrix(challenge[pair]['output'])
    if x == pairs:
        sq_in = 1
    if y == pairs:
        sq_out = 1
    if sq_in + sq_out == 2:
        sq_both = 1
    return sq_in, sq_out, sq_both


id = []
pairs = []
eq_dim = []
sq_in = []
sq_out = []
sq_both = []
cols_in = []
rows_in = []
cols_out = []
rows_out = []

for challenge in training_challenges:
    id.append(challenge)
    pairs.append(len(training_challenges[challenge]['train']))
    eq_dim.append(eq_dim_matrix(training_challenges[challenge]['train'],len(training_challenges[challenge]['train'])))
    x, y, z = squared_matrix_full(training_challenges[challenge]['train'],len(training_challenges[challenge]['train']))
    sq_in.append(x)
    sq_out.append(y)
    sq_both.append(z)
    rows_in.append(len(training_challenges[challenge]['train'][0]['input']))
    cols_in.append(len(training_challenges[challenge]['train'][0]['input'][0]))
    rows_out.append(len(training_challenges[challenge]['train'][0]['output']))
    cols_out.append(len(training_challenges[challenge]['train'][0]['output'][0]))


meta_train = pd.DataFrame({'id':id,'pairs':pairs,'eq_dim':eq_dim, 'sq_in':sq_in, 'sq_out':sq_out, 'sq_both':sq_both, 'cols_in':cols_in, 'rows_in':rows_in, 'cols_out':cols_out, 'rows_out':rows_out})


#Metadata example
meta_train.head()


# adding more dimensionality epecs
meta_train['area_in'] = meta_train['cols_in']* meta_train['rows_in']
meta_train['area_out'] = meta_train['cols_out']* meta_train['rows_out']
meta_train['increase'] = meta_train['area_out']/meta_train['area_in']


# Showing the new columns
meta_train.head()


#  Filters to create the finals datasets

#filtering just equals dim ones
eq_dim_train = meta_train[meta_train['eq_dim']==1]
#filtering the examples with square input and output
sq_both_train = meta_train[meta_train['sq_both']==1]
#filtering the examples that are not square input and output
non_squared_train = meta_train[meta_train['sq_both']==0]
#filtering the examples that are not equals dim input and output
diff_train = meta_train[meta_train['eq_dim']==0]


#  Creating the finals datasets

#filtering examples that are equal dimension and squared matrix (input and output)
equals_squared_train = eq_dim_train[eq_dim_train['sq_both']==1]
#filtering examples that are equal dimension but not squared matrix (input and output)
equals_non_squared_train = eq_dim_train[eq_dim_train['sq_both']==0]
#filtering the examples with square matrix but not equal dimension (input and output)
diff_squared_train = sq_both_train[sq_both_train['eq_dim']==0]
#filtering the examples that are not square and not equal dimension (input and output)
diff_non_squared_train = non_squared_train[non_squared_train['eq_dim']==0]


equals_squared_train.head()


equals_non_squared_train.head()


diff_squared_train.head()


diff_non_squared_train.head()


print("eq n sq: ", equals_squared_train.shape[0])
print("eq non sq: ", equals_non_squared_train.shape[0])
print("diff n sq: ", diff_squared_train.shape[0])
print("diff non sq: ", diff_non_squared_train.shape[0])
print("\nTotal: ", equals_squared_train.shape[0] + equals_non_squared_train.shape[0] + diff_squared_train.shape[0] + diff_non_squared_train.shape[0])


#All the metadata
meta_train.to_csv('meta_train_2025.csv',index=False)


#exporting by specs to work indepently
equals_squared_train.to_csv('equals_squared_train_2025.csv',index=False)
equals_non_squared_train.to_csv('equals_non_squared_train_2025.csv',index=False)
diff_squared_train.to_csv('diff_squared_train_2025.csv',index=False)
diff_non_squared_train.to_csv('diff_non_squared_train_2025.csv',index=False)

