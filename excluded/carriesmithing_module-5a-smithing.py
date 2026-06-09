# For each block of code you should add a comment explaining what is going on.
# Once you have run the entire worksheet you should use GA to optimize a task of your choosing (using appropriate data that you find seperatly). 


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(666)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# what this cell does:
# - import numpy, pandas, and os 
# - use np to create a random seed 
# - gets the directory and filepaths from the current directory then prints them out



data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')
data

# creates dataframe called data from the family_data.csv then prints it


matrix = data[['choice_0', 'choice_1', 'choice_2', 'choice_3', 'choice_4',
       'choice_5', 'choice_6', 'choice_7', 'choice_8', 'choice_9']].to_numpy()

# converts the df "data" to numpy array "matrix"


submission = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv')
submission 

# creates dataframe called submission from the sample_submission.csv then prints it


# there was a private dataset I'm assuming was this one I'm guessing this is similar to submission so I'm subsituting so it can continue
# best = pd.read_csv("../input/local1/sub1.csv")
best = submission['assigned_day'].to_list()


# creates dataframe called best from the sub1.csv and then changes best to be a list of the values from the column assigned_day


chromosome = [0 for i in range(500000)]

for i in range(5000):
    chromosome[i*100+best[i]-1] = 1
    
population = []
population.append(chromosome)

# create list called chromosome of length 500000 with 0 in every spot, then changes some of the values to 1 using the "best" list
# creates empty list population then adds the chromosome list to population


#https://www.kaggle.com/xhlulu/santa-s-2019-4x-faster-cost-function
family_size_dict = data[['n_people']].to_dict()['n_people'] 

cols = [f'choice_{i}' for i in range(10)]
choice_dict = data[cols].T.to_dict()


N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

# from 100 to 1
days = list(range(N_DAYS,0,-1))


family_size_ls = list(family_size_dict.values())
choice_dict_num = [{vv:i for i, vv in enumerate(di.values())} for di in choice_dict.values()]


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

def cost_function(prediction):
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
            penalty += 100000000

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


# creates dictionary from the dataframe "data" column n_people
# creates dictionary from subset of "data" df (columns starting with "choice") transposed
# creates and assigns values to constants: n_days, max_occupancy, min_occupancy
# creates list called "days" starting at N_DAYS to 1 in descending order
# creates list called "family_size_ls" made of the values from family_size_dict
# creates a dict of the values in choice_dict
# creates a dict of penality values
# defines function cost_function - takes in a prediction and returns a penalty value


def convert(chromosome):
    indexes = []
    for i in range(0,500000):
        if chromosome[i] == 1:
            indexes.append((i+1)-(i//100)*100)
    return indexes

def selection(population, selection_size, group_size):

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

# five functions are defined that simulate population over time (aka the steps of the genetic algorithm)


def epoch_optimal(population):
    
    minimum = 9999999999999999999999999999
    chromosome=-1
    for i in population:
        test = convert(i)
        if cost_function(test)<minimum:
            chromosome = i
            minimum = cost_function(test)
            
    return chromosome, minimum

# this function finds the lowest cost (best) chromosome in an given population list, with the minimum cost 


population = reproduction(matrix, population, 50, 0.25, 50)

# simulate reproduction occuring in a population (replace current population with new population from reproduction function)


best = -1
best_val = 105163.8446075958
for i in range(2):
    print(i)
    population = selection(population, 25, 5)
    population = reproduction(matrix, population, 50, 0.25, 10)
    ind, val = epoch_optimal(population)
    print('Min on epoch: ', str(val))
    if best_val > val:
        best_val = val
        best = ind

# run genetic algorithm 20 times 


#orig line: sub = convert(best) but with the current code that best is an int variable so this was erroring I assumed you wanted the index of the "best chromosome" in the population
sub = convert(population[best]) 
submission['assigned_day'] = sub
submission.to_csv('submission.csv', index=False)
submission

# I believe this cell was supposed to create a csv of the best chromosomes


import pandas as pd
cancer = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv') # create df
cancer['diagnosis'] = cancer['diagnosis'].map({'M': 1, 'B': 0}) # replace b and m with 0 and 1
cancer = cancer.drop('Unnamed: 32', axis=1) # remove column full of null values
cancer.head() # show df


# rename columns
cancer_og_names = cancer.columns.to_list()
cancer_new_names = {old_name: f'symptom_{i}' for old_name, i in zip(cancer.columns.to_list(), range(-2, cancer.shape[1])) if old_name not in ['id', 'diagnosis']}
cancer.rename(columns=cancer_new_names, inplace=True)
cancer.columns


matrix = cancer[['symptom_0', 'symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 
               'symptom_5', 'symptom_6', 'symptom_7', 'symptom_8', 'symptom_9', 
               'symptom_10', 'symptom_11', 'symptom_12', 'symptom_13', 'symptom_14', 
               'symptom_15', 'symptom_16', 'symptom_17', 'symptom_18', 'symptom_19', 
               'symptom_20', 'symptom_21', 'symptom_22', 'symptom_23', 'symptom_24', 
               'symptom_25', 'symptom_26', 'symptom_27', 'symptom_28', 'symptom_29']].to_numpy()
matrix


submission = cancer[['id', 'diagnosis']]
submission


best = cancer['diagnosis'].to_list()


chromosome = [0 for i in range(500000)]
for i in range(569):
    chromosome[i*100+best[i]-1] = 1
    
population = []
population.append(chromosome)


#https://www.kaggle.com/xhlulu/santa-s-2019-4x-faster-cost-function
cancer_dict = cancer[['diagnosis']].to_dict()['diagnosis'] 

cols = [f'choice_{i}' for i in range(30)]
choice_dict = data[cols].T.to_dict()


N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

# from 100 to 1
days = list(range(N_DAYS,0,-1))


family_size_ls = list(family_size_dict.values())
choice_dict_num = [{vv:i for i, vv in enumerate(di.values())} for di in choice_dict.values()]


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

def cost_function(prediction):
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
            penalty += 100000000

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


# creates dictionary from the dataframe "data" column n_people
# creates dictionary from subset of "data" df (columns starting with "choice") transposed
# creates and assigns values to constants: n_days, max_occupancy, min_occupancy
# creates list called "days" starting at N_DAYS to 1 in descending order
# creates list called "family_size_ls" made of the values from family_size_dict
# creates a dict of the values in choice_dict
# creates a dict of penality values
# defines function cost_function - takes in a prediction and returns a penalty value


best = -1
best_val = 105163.8446075958
for i in range(20):
    print(i)
    cancer_pop = selection(cancer_pop, 25, 5)
    cancer_pop = reproduction(matrix, cancer_pop, 50, 0.25, 10)
    ind, val = epoch_optimal(cancer_pop)
    print('Min on epoch: ', str(val))
    if best_val > val:
        best_val = val
        best = ind

