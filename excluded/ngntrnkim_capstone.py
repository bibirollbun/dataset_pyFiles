import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import random
from sklearn.model_selection import RandomizedSearchCV


# def function_evaluation(x):
#     # Bukin function N.6
#     # f(x) = 100 * sqrt(abs(x2 - 0.01 * x1^2)) + 0.01 * abs(x1 + 10)
#     fx = 100 * math.sqrt(abs(x[1] - 0.01 * x[0]**2)) + 0.01 * abs(x[0] + 10)
#     return fx
data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', header = 0)
# col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
# data.columns = col_names
data = data[:2000]
label_encoder = preprocessing.LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

# Split the data into features and target variable
X = data.drop('num_sold', axis=1)
y = data['num_sold']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# x = [n_estimators,max_features, max_depth]
result = {}
count = 0
from sklearn.metrics import mean_absolute_error
def function_evaluation(x):
    global result
    if str(x) in result:
        return result[str(x)]
    global count
    count += 1
    # print(x)
    if x[0] == 0 or x[1] == 0 or x[2] == 0:
        INVALID_INPUT = 99999
        return INVALID_INPUT
    else:
        # Train the random forest model
        clf = RandomForestClassifier(n_estimators = abs(x[0]), max_features = abs(x[1]/100), max_depth = abs(x[2]), random_state=42)
        clf.fit(X_train, y_train)

        # Predict the target variable for the test set
        y_pred = clf.predict(X_test)

        # Evaluate the model
        mape = mean_absolute_error(y_test, y_pred)
        # print(y_test[:10], y_pred[:10])
        # print(f'MAPE: {mape}')
        result[str(x)] = mape
        return mape

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def grid_search(domain, split):
    param_dist = {
        'n_estimators': [random.randint(domain[0][0][0], domain[0][0][1]) for _ in range(split)],
        'max_depth': [random.randint(domain[1][0][0], domain[1][0][1]) for _ in range(split)],
        'min_samples_split': [random.randint(domain[2][0][0], domain[2][0][1]) for _ in range(split)],
    }
        # Randomized search
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=10,  # Limit to 10 evaluations
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_absolute_error',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    random_search.fit(X_train, y_train)

    # Results
    print("Best Parameters:", random_search.best_params_)
    print("Best Cross-Validation Accuracy:", random_search.best_score_)

    # Test set evaluation
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mape = mean_absolute_error(y_test, y_pred)
    # print(y_test[:10], y_pred[:10])
    # print(f'MAPE: {mape}')
    print(mape)
    return map



# gridSearch = []
# for i in range (1, 51):
#     gridSearch.append(grid_search(domain, i))


import random 
import math

class Point(object):
    def __init__(self, chromosome_list=None, value_list=None):
        self.fitness = math.inf
        self.individuals = []
        
        if chromosome_list is not None:
            self.individuals = [Individual(chromosome=c) for c in chromosome_list]
        elif value_list is not None:
            self.individuals = [Individual(value=v) for v in value_list]

    def fitness_evaluation(self):
        # Evaluate the fitness of the point
        self.fitness = function_evaluation([individual.value for individual in self.individuals])

class Individual(object):
    def __init__(self, chromosome=None, value=None):
        if chromosome is not None:
            self.chromosome = chromosome
            self.value = self.binary_to_float(chromosome)
        elif value is not None:
            self.value = value
            self.chromosome = self.float_to_binary(value)
        self.fitness = 0

    def float_to_binary(self, num: float) -> str:
        if num < 0:
            sign = 1
            num = -num
        else:
            sign = 0

        # Convert integer part to binary
        integer_part = int(num)
        fractional_part = num - integer_part
        binary_integer = bin(integer_part).replace("0b", "")

        # Convert fractional part to binary
        binary_fraction = []
        while fractional_part:
            fractional_part *= 2
            bit = int(fractional_part)
            if bit == 1:
                fractional_part -= bit
                binary_fraction.append('1')
            else:
                binary_fraction.append('0')
            if len(binary_fraction) > 52:  # Limit the length to prevent infinite loop
                break

        binary_fraction = ''.join(binary_fraction)
        binary_representation = f"{binary_integer}.{binary_fraction}"

        # self.chromosome = binary_representation

        return f"{'-' if sign else ''}{binary_representation}"
    
    def binary_to_float(self, binary: str) -> float:
        sign = -1 if binary[0] == '1' else 1
        binary = binary[1:]
        integer_part, fractional_part = binary.split('.')
        integer_part = int(integer_part, 2) if integer_part != '' else 0
        fractional_part = sum(int(bit) * (2 ** -(i + 1)) for i, bit in enumerate(fractional_part)) if fractional_part != '' else 0
        return sign * (integer_part + fractional_part)

def initialize_population(population_size, domain):
    population = []
    for i in range(population_size):
        value_list = []
        for j in range(len(domain)):
            if domain[j][1] == 'int':
                value_list.append(random.randint(domain[j][0][0], domain[j][0][1]))
            elif domain[j][1] == 'float':
                value_list.append(random.uniform(domain[j][0][0], domain[j][0][1]))
        population.append(Point(value_list=value_list))
    return population

def evaluate_population(population):
    for point in population:
        point.fitness_evaluation()

def selection_fitness_ranking(population, num_parents, type):
    # select the best individuals based on their minimum fitness
    if type == 'min':
        return sorted(population, key=lambda x: x.fitness)[:num_parents]
    # select the best individuals based on their maximum fitness
    elif type == 'max':
        return sorted(population, key=lambda x: x.fitness, reverse=True)[:num_parents]

def selection_tournament(population, num_parents, tournament_size):
    # Tournament selection: Select the best individual from a random subset of the population
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda x: x.fitness)
        parents.append(winner)
    return parents

def selection_roulette(population, num_parents):
    # Roulette wheel selection: Select individuals based on their relative fitness
    total_fitness = sum(point.fitness for point in population)
    parents = []
    for _ in range(num_parents):
        pick = random.uniform(0, total_fitness)
        current = 0
        for point in population:
            current += point.fitness
            if current > pick:
                parents.append(point)
                break
    return parents

def mutation(point, prob_mutation=1): #TODO: One point mutation
    new_individuals = []
    # Check if mutation should occur for each individual
    if random.random() < prob_mutation:
        for i, individual in enumerate(point.individuals):
            chromosome = list(individual.chromosome)  # Convert to list for mutability
            length = len(chromosome)
            # Pick a random index and flip the bit
            index = random.randint(0, length - 1)
            while chromosome[index] == '.':
                index = random.randint(0, length - 1)
            chromosome[index] = '0' if chromosome[index] == '1' else '1'
            # Update the individual's chromosome
            new_individuals.append(''.join(chromosome))  # Update individual with mutated chromosome
        return Point(chromosome_list=new_individuals)  # Return updated point
    else:
        return None

    
def crossover(point1, point2, crossover_probability = 1):
    if random.random() < crossover_probability:
        new_point_1 = []
        new_point_2 = []
        for ind1, ind2 in zip(point1.individuals, point2.individuals):
            # get "." position in ind1, ind2
            dot_position1 = ind1.chromosome.index(".")
            dot_position2 = ind2.chromosome.index(".")
            left = min(dot_position1, dot_position2)
            right = max(dot_position1, dot_position2)
            # get crossover point   
            crossover_point = random.randint(0, len(ind1.chromosome) - 1)
            while crossover_point >= left and crossover_point <= right:
                crossover_point = random.randint(0, len(ind1.chromosome) - 1)
            # create new chromosome
            new_point_1.append(ind1.chromosome[:crossover_point] + ind2.chromosome[crossover_point:])
            new_point_2.append(ind2.chromosome[:crossover_point] + ind1.chromosome[crossover_point:])
        return Point(chromosome_list=new_point_1), Point(chromosome_list=new_point_2)
    else:
        return None
    
def extrema(population, type):
    if type == 'max':
        return max(population, key=lambda x: x.fitness)
    elif type == 'min':
        return min(population, key=lambda x: x.fitness)




# import random
# import numpy as np


population_size = 10
domain =  [[[1,  500], 'int'], [[1, 100], 'int'], [[1, 500], 'int']]
max_generations = 10
crossover_probability = 0.9
mutation_probability = 0.5


population = []
for i in range(population_size):
    value_list = []
    population.append(Point(value_list=[364, 100, 88]))


evaluate_population(population)


fitness = extrema(population, 'min').fitness


fitness


# population = initialize_population(population_size=population_size, domain=domain)
# evaluate_population(population)
# fitness = extrema(population, 'max').fitness


def genetic_algorithm(population, population_size, max_generations, crossover_probability, mutation_probability):
    evaluate_population(population)
    best_res = extrema(population, 'min')
    print(f"Original: {best_res.fitness}")

    result = []
    for generation in range(max_generations):
        new_population = []
        # select 5 best individual in the population to pass on the next generation
        # selection as option
        best_individuals = selection_fitness_ranking(population, 5, 'min')
        new_population.append(best_individuals[0])

        # create mutation of best individuals
        for individual in best_individuals:
            mutation_result = mutation(individual, mutation_probability)
            if mutation_result is not None:
                new_population.append(mutation_result)

        # create other offsprings by crossover
        for i in range(population_size - len(new_population)):
            parent1 = random.choice(best_individuals)
            parent2 = random.choice(best_individuals)
            while parent2 == parent1:
                parent2 = random.choice(best_individuals)
            crossover_result = crossover(parent1, parent2, crossover_probability)
            if crossover_result is not None:
                new_population.extend(crossover_result)

        population = new_population
        # evaluate the fitness of each individual in the population
        evaluate_population(population)
        best_res = extrema(population, 'min')
        # print(best_res.individuals)
        values = [str(individual.value) for individual in best_res.individuals]
        print(values)
        # break
        result.append(best_res.fitness)
        print(f"Generation {generation}: {best_res.fitness}, {count}")
        # for point in population:
        #     values = [str(individual.value) for individual in point.individuals]
        #     print(" ".join(values))
    return result


print("population size = 10, max generation = 50")


population_size = 10
population = []
for i in range(population_size):
    value_list = []
    population.append(Point(value_list=[364, 100, 88]))


full_result = {}
count = 0
for i in range(10):    
    full_result[i] = genetic_algorithm(population, population_size, 50, crossover_probability, mutation_probability)
    print(count)
    count = 0


full_result


df = pd.DataFrame(full_result)
df.to_csv('population10generation50.csv')


full_result_2 = []


print("population size = 10, max generation = 10")


population_size = 50
population = []
for i in range(population_size):
    value_list = []
    population.append(Point(value_list=[364, 100, 88]))


full_result_2 = {}
for i in range(10):
    full_result_2[i] = genetic_algorithm(population, population_size, 50, crossover_probability, mutation_probability)
    print(count)
    count = 0


df = pd.DataFrame(full_result_2)
df.to_csv('population50generation50.csv')


# gridSearch = []


# print('grid search')





# # Number of runs
# num_runs = len(data)

# # Determine the maximum number of generations across all runs
# max_generations = max(len(run) for run in data)

# # Create a list of generation indices
# generations = list(range(max_generations))

# # Plot all runs
# plt.figure(figsize=(8, 5))

# for run in data:
#     if run:  # Only plot non-empty runs
#         plt.plot(range(len(run)), run, marker='o', linestyle='-', alpha=0.6)

# plt.xlabel("Generation")
# plt.ylabel("Score")
# plt.title("Score Evolution Over Generations (Multiple Runs)")
# plt.grid(True)
# plt.show()

