# For each block of code you should add a comment explaining what is going on.
# Once you have run the entire worksheet you should use GA to optimize a task of your choosing (using appropriate data that you find seperatly). 


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(606 + 60)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')
data

# This reads the data from family_data.csv into a dataframe, and then displays the first and last 5 rows.



matrix = data[['choice_0', 'choice_1', 'choice_2', 'choice_3', 'choice_4',
       'choice_5', 'choice_6', 'choice_7', 'choice_8', 'choice_9']].to_numpy()

# This block converts the respective columns and their associated rows into a 2 dimensional matrix
# of size (5000, 10) [(rows, columns)]. .to_numpy strictly converts to a 2darray every time. If 
# we needed a 3darray, then we would have to use the .reshape() function after .to_numpy().


submission = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv')
submission

# Like before, this block of code reads data from sample_submission.csv into a dataframe, submission.
# Then, it displays the first and last 5 rows.


best = pd.read_csv("/kaggle/input/sub1-csv/submission.csv")
best = best['assigned_day'].to_list()

# Reads in data from sub1.csv. Then it makes it so only the column of "assigned_day" is selected, and the .to_list()
# turns the series of data into a python list.


chromosome = [0 for i in range(500000)] # This creates a list with length 500000, consisting of all 0's.
for i in range(5000): 
    chromosome[i*100+best[i]-1] = 1
    
population = []
population.append(chromosome)


# This whole block of code "encodes" the families currently best assigned day. Every 100 slots in the chromosone is
# dedicated to one family. The position within that 100 slot is determined by the expression best[i] - 1, and which 100
# slot is that specific family's is determiend by i*100. In doing this, only one value out of the 100 slots for each family
# is assigned a value of 1, indicating that was chosen as that family's best day to attend. This is all based on the 
# currently best known solution, read in from best.


#https://www.kaggle.com/xhlulu/santa-s-2019-4x-faster-cost-function
family_size_dict = data[['n_people']].to_dict()['n_people'] # Creates a dictionary where the key is a family's index, and the value is the amount of people in that family.

cols = [f'choice_{i}' for i in range(10)] # creates a list where each value is a string corresponding to a choice. Ex: choice_0, choice_1, ..., choice_9
choice_dict = data[cols].T.to_dict()

# data[cols] selects only the columns from data which contain the choice_# for the families.
# .T is a transpose which switches the rows and columns. Now the rows are the choices and the columns are the family indexes.
# .to_dict converts this into a dictionary. The key is the family ID/index, and the value is a dictionary containing their preferred choices.

N_DAYS = 100 # The amount of days the families are to be schedules across.
MAX_OCCUPANCY = 300 # This and the line below define constraints for the cost function.
MIN_OCCUPANCY = 125

# from 100 to 1
days = list(range(N_DAYS,0,-1)) # Creates a list of days starting from 100 and going to 1.

family_size_ls = list(family_size_dict.values()) # creates a list in order of the family indexes which contains the size of each family.
choice_dict_num = [{vv:i for i, vv in enumerate(di.values())} for di in choice_dict.values()]

# choice_dict_num is meant to create a list where each value is a dictionary, where the key corresponds
# to a day number choice for a given family, and the value is the pentalty rank for that given day if the family is assigned it.
# Each item in the list is a family, where the index corresponds to the family's index.

# Computer penalities in a list
penalties_dict = {
    n: [
        0,
        50,
        50 + 9 * n,
        100 + 9 * n,
        200 + 9 * n,
        200 + 18 * n,
        300 + 18 * n,
        300 + 36 * n,
        400 + 36 * n,
        500 + 36 * n + 199 * n,
        500 + 36 * n + 398 * n
    ]
    for n in range(max(family_size_dict.values())+1)
} 

# This creates a dictionary, where the key value is the size of a given family. The value is a list,
# where each value of the list consists of the potential penalty cost which is calculated based on choice of day assigned and number of people in family.

# Everything before this point consists of preprocessing the data, setting up dictionaries and lists we can use to more effectively look up certain values.

def cost_function(prediction): # used to determine the total cost for the way we have assigned the families. the goal is the minimize the output of this function.
    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k:0 for k in days}
    
    # Looping over each family; d is the day, n is size of that family, 
    # and choice is their top choices
    for n, d, choice in zip(family_size_ls, prediction, choice_dict_num):
        # add the family member count to the daily occupancy
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d not in choice:
            penalty += penalties_dict[n][-1]
        else:
            penalty += penalties_dict[n][choice[d]]

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for v in daily_occupancy.values():
        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):
            penalty += 100000000 # if we ever go beyond our constraints, we want to add a LOT of penalty since this is very bad.

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty


def convert(chromosome):
    '''
    The goal of this function is to determine what day each family has been assigned
    to attend Santa's workshop. It adds this value to a list, indexes. Each value in indexes
    is represented as the exact day they were assigned (calculated by (i+1)-(i//100)*100).
    The index of the day inside indexes corresponds to the index of the family the day was assigned to.
    '''
    indexes = []
    for i in range(0,500000):
        if chromosome[i] == 1:
            indexes.append((i+1)-(i//100)*100)
    return indexes

def selection(population, selection_size, group_size):
    '''
    population -- the list of current solutions
    selection_size -- the amount of parents we want to choose from this generation
    group_size -- determines how many participants will be competing
    The general purpose of this function is to be a "tournament selection", which aims to give the best
    competitors of each generation the best chance of being parents relative to the other competitors.
    '''
    parents = []
    for i in range(selection_size):
        minimum = 9999999999999999999
        index = -1
        for t in range(group_size):
            chromosome =  np.random.randint(len(population))
            for_test = convert(population[chromosome])
            if cost_function(for_test) < minimum:
                minimum = cost_function(for_test)
                index = chromosome
        parents.append(population[index])
                
    return parents
        
def crossover(p1, p2):
    '''
    This function aims to represent the swapping of traits between two parents.
    '''
    p = [p1[i] for i in range(50000)]
    for i in range(50000, 100000):
        p.append(p2[i])
    for i in range(100000, 150000):
        p.append(p1[i])
    for i in range(150000, 200000):
        p.append(p2[i])
    for i in range(200000, 250000):
        p.append(p1[i])
    for i in range(250000, 300000):
        p.append(p2[i])
    for i in range(300000, 350000):
        p.append(p1[i])
    for i in range(350000, 400000):
        p.append(p2[i])
    for i in range(400000, 450000):
        p.append(p1[i])
    for i in range(450000, 500000):
        p.append(p2[i])
    return p

def mutation(family_matrix, chromosome, desired_rate=10):
    '''
    The aim of this function is to introduce random "mutations", i.e., changes into a given new solution
    created from the crossover.
    '''
    family_number = np.random.randint(5000)
    desired_probability = np.random.randint(100)
    if desired_probability < desired_rate:
        new_day = np.random.randint(100)
    else:
        ind = np.random.randint(10)
        new_day = family_matrix[family_number][ind] - 1
    for i in range(family_number*100, family_number*100+100):
        chromosome[i] = 0
    chromosome[family_number*100+new_day] = 1
    
    return chromosome

def reproduction(family_matrix, population, new_generation_size, mutation_rate, number_of_mutations):
    '''
    This function aims to create a new generation of chromosomes by simulating the reproduction process,
    implementing the three previously made functions.
    '''
    new_generation = []
    for i in range(new_generation_size):
        p1_index = np.random.randint(len(population))
        p2_index = np.random.randint(len(population))
        p = crossover(population[p1_index], population[p2_index])
        mutation_probability = np.random.randint(100)
        if mutation_probability >= mutation_rate:
            mutations_number = np.random.randint(number_of_mutations)
            for m in range(mutations_number):
                p = mutation(family_matrix, p, 10)
            
        new_generation.append(p)
    return new_generation


def epoch_optimal(population):
    '''
    The purpose of this function is to determine which chromosome from a given generation is the 
    fittest (in this context, has the lowest cost.)
    '''
    minimum = 9999999999999999999999999999
    chromosome=-1
    for i in population:
        test = convert(i)
        if cost_function(test)<minimum:
            chromosome = i
            minimum = cost_function(test)
            
    return chromosome, minimum


population = reproduction(matrix, population, 50, 0.25, 50)


best = -1
best_val = 105163.8446075958
for i in range(20):
    print(i)
    population = selection(population, 25, 5)
    population = reproduction(matrix, population, 50, 0.25, 10)
    ind, val = epoch_optimal(population)
    print('Min on epoch: ', str(val))
    if best_val > val:
        best_val = val
        best = ind


sub = convert(best)
submission['assigned_day'] = sub
submission.to_csv('submission.csv', index=False)
submission


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Set random seed for repeatability.
np.random.seed(44)


# Read data into dataframe.
df = pd.read_csv("/kaggle/input/knapsack-problem/knapsack_5_items_new.csv")

# Display the head of items.
df.head()


# Convert data from temp df into new data frame where lists represent items in a knapsack.
items = pd.DataFrame(columns={'Weights','Prices','Capacity','Best picks'})

for i in range(len(df)):
    items = items.append({'Weights':
                            [df['W1'][i],df['W2'][i],df['W3'][i],df['W4'][i],
                                       df['W5'][i]],
                            'Prices':[df['P1'][i],df['P2'][i],df['P3'][i],df['P4'][i],
                                       df['P5'][i]],
                            'Capacity':df['Capacity'][i],
                            'Best picks':[df['BP1'][i],df['BP2'][i],df['BP3'][i],
                                          df['BP4'][i],df['BP5'][i]],},ignore_index=True)



items.head()

'''
Capacity -- represents the maximum weight the knapsack can hold
Best picks -- a list, where each index corresponds to an items matching the index in weights and prices. A 1 denotes this item is included in the knapsack. A 0 denotes it isn't.
Prices -- a list corresponding to the price of each item in the knapsacks.
Weights -- a list where each index represents an items weight. Each item has the potential to be in the knapsack if its deemed a best pick.
'''


items.shape

'''
There are 9885 knapsacks, each of which has the best picks determined for them.
This data will be used to train our model.
'''

