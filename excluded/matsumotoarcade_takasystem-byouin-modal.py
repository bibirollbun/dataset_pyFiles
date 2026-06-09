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


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances

# Genetic Algorithm imports
import random
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import seaborn as sns

# --- Constants and Parameters ---
POPULATION_SIZE = 50
NUM_GENERATIONS = 20
MUTATION_RATE = 0.1

# --- Ambulância simulation parameters ---
BASE_SPEED = 60  # km/h
MAX_SPEED = 100  # km/h
GRAVE_PROBLEM_AVOIDANCE_IMPACT = 0.8  # Reduction if a severe issue avoided

# --- Constants ---
TARGET_EFFICIENCY = 100  # Target efficiency for calculating K
FIXED_COST = 500

def calculate_lref_d0(df):
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

# --- Load Data ---
# Simulate generic data instead of loading from files
num_sequences = 10
data = {
    'target_id': [f'seq_{i}' for i in range(num_sequences)],
    'sequence': ['AUCG' * (i + 1) for i in range(num_sequences)], # Different sequence lengths
    'description': ['Generic sequence' for _ in range(num_sequences)]
}

test_df = pd.DataFrame(data)
test_df = calculate_lref_d0(test_df)

# --- GENETIC ALGORITHM FOR AMBULANCE PARAMETERS ---
def create_individual():
    """Creates a random individual (ambulance parameters)."""
    return {
        'speed_factor': random.uniform(0.8, 1.2), # Factor for speed adjustment
        'route_preference': random.uniform(0.0, 1.0)  # Preference for faster vs. shorter route
    }

def calculate_efficiency(individual, distance, patients, time, grave_problem_avoided, fixed_cost):
    """Calculates the efficiency based on the provided equation."""
    # Adjust parameters based on the individual
    effective_speed = BASE_SPEED * individual['speed_factor']
    if effective_speed > MAX_SPEED:
        effective_speed = MAX_SPEED

    # Route adjustment based on 'route_preference' - Simplified here
    adjusted_distance = distance * (1 - individual['route_preference'] * 0.1)  # Shorter route preference reduces distance
    adjusted_time = time * (1 + individual['route_preference'] * 0.1)  # Shorter route preference might increase time slightly

    # Calculate the penalization
    penalization_simulation = GRAVE_PROBLEM_AVOIDANCE_IMPACT if grave_problem_avoided else 0.0

    # Calculate efficiency
    efficiency = ((patients / adjusted_time) / (adjusted_distance * (1 + penalization_simulation) + fixed_cost))

    return efficiency

def evaluate_fitness(individual, distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost):
    """Evaluates the fitness of an individual over a set of scenarios."""
    total_efficiency = 0
    num_scenarios = len(distances)

    for i in range(num_scenarios):
        distance = distances[i]
        patients = patients_list[i]
        time = times_list[i]
        grave_problem_avoided = grave_problems_avoided_list[i]

        efficiency = calculate_efficiency(individual, distance, patients, time, grave_problem_avoided, fixed_cost)
        total_efficiency += efficiency

    # Return the average efficiency
    return total_efficiency / num_scenarios

def crossover(parent1, parent2):
    """Performs crossover between two parents."""
    child = {}
    for key in parent1.keys():
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(individual):
    """Mutates an individual."""
    for key in individual.keys():
        if random.random() < MUTATION_RATE:
            # Adjust the mutation based on the key. For example, we should use a small mutation on factors and larger on route selection.
            if key == 'speed_factor':
                individual[key] += random.uniform(-0.1, 0.1) # Small adjustments
                individual[key] = max(0.8, min(1.2, individual[key]))  # Clamp value.
            else: # route_preference.
                individual[key] += random.uniform(-0.2, 0.2)  # Larger adjustment
                individual[key] = max(0.0, min(1.0, individual[key])) # Clamp value.
    return individual

def genetic_algorithm(distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost):
    """Runs the genetic algorithm."""
    # 1. Initialization
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    fitness_history = []
    best_individual_history = []

    # 2. Evolution
    for generation in range(NUM_GENERATIONS):
        # a. Evaluate Fitness
        fitness_scores = [evaluate_fitness(individual, distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost) for individual in population]

        # Check if fitness_scores is empty
        if not fitness_scores:
            print("Warning: All fitness scores are zero. Stopping the GA.")
            return create_individual(), fitness_history, best_individual_history  # Return a default individual

        # Find best individual only if fitness_scores is not empty
        if fitness_scores:
            best_individual_index = np.argmax(fitness_scores)
            best_individual = population[best_individual_index]
            best_fitness = fitness_scores[best_individual_index]
            fitness_history.append(best_fitness)
            best_individual_history.append(best_individual)
        else:
            # If fitness_scores is empty, append a default value
            best_individual = create_individual()
            best_fitness = 0.0  # Set to a default value
            fitness_history.append(best_fitness)
            best_individual_history.append(best_individual)

        # b. Selection (Tournament Selection)
        selected_indices = random.choices(range(POPULATION_SIZE), weights=fitness_scores, k=POPULATION_SIZE)
        selected_population = [population[i] for i in selected_indices]

        # c. Crossover
        offspring = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1] if i+1 < POPULATION_SIZE else selected_population[i]
            child = crossover(parent1, parent2)
            offspring.append(child)

        # d. Mutation
        mutated_offspring = [mutate(child) for child in offspring]

        # Replace the old population with the new offspring
        population = mutated_offspring

        print(f"Generation {generation}: Best individual = {best_individual}, Fitness = {best_fitness}")

    # 3. Return the Best Individual
    if fitness_scores:
        best_individual_index = np.argmax(fitness_scores)
        return population[best_individual_index], fitness_history, best_individual_history
    else:
        return create_individual(), fitness_history, best_individual_history

# --- Simulation Scenarios ---
# Here, we define different scenarios for the ambulance to operate in. These scenarios include
# the distance traveled, the number of patients attended, the time taken, and whether a grave
# problem was avoided. This allows us to test the ambulance in different situations.
distances = [100, 150, 200, 120]  # Distances in km
patients_list = [5, 8, 6, 7]  # Number of patients
times_list = [2, 3, 2.5, 2.2]  # Times in hours
grave_problems_avoided_list = [True, False, True, False]  # Whether a grave problem was avoided

# --- Run Genetic Algorithm ---
try:
    best_params, fitness_history, best_individual_history = genetic_algorithm(distances, patients_list, times_list, grave_problems_avoided_list, FIXED_COST)
    print("\nBest Ambulance Parameters:", best_params)
except ValueError as e:
    print(f"Error during genetic algorithm execution: {e}")
    best_params = create_individual()  # Use a default individual
    fitness_history = []
    best_individual_history = []


# --- Calculate K factor ---
# To calculate the K factor, let's assume an 'average' scenario:
avg_distance = np.mean(distances)
avg_patients = np.mean(patients_list)
avg_time = np.mean(times_list)
avg_grave_problem_avoided = any(grave_problems_avoided_list) # Assume there's at least one.

# Calculate efficiency with the best parameters
avg_efficiency_with_best_params = calculate_efficiency(
    best_params, avg_distance, avg_patients, avg_time, avg_grave_problem_avoided, FIXED_COST
)

# Calculate K based on the target efficiency
K = TARGET_EFFICIENCY / avg_efficiency_with_best_params

print("Calculated K factor:", K)

# --- PLOT FITNESS HISTORY ---
if fitness_history:
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Average Efficiency)")
    plt.title("Genetic Algorithm: Fitness Over Generations")
    plt.grid(True)
    plt.show()
else:
    print("No fitness history to plot (GA might have failed).")

# --- DYNAMIC SCENARIO VISUALIZATION ---
def visualize_scenario(scenario_index):
    distance = distances[scenario_index]
    patients = patients_list[scenario_index]
    time = times_list[scenario_index]
    grave_problem_avoided = grave_problems_avoided_list[scenario_index]

    efficiency = calculate_efficiency(best_params, distance, patients, time, grave_problem_avoided, FIXED_COST)
    scaled_efficiency = efficiency * K

    print(f"Scenario {scenario_index + 1}:")
    print(f"  Distance: {distance} km")
    print(f"  Patients: {patients}")
    print(f"  Time: {time} hours")
    print(f"  Grave problem avoided: {grave_problem_avoided}")

    print(f"\nBest Parameters:")
    print(f"  Speed Factor: {best_params['speed_factor']:.2f}")
    print(f"  Route Preference: {best_params['route_preference']:.2f}")

    print(f"\nEfficiency (Unscaled): {efficiency:.4f}")
    print(f"Efficiency (Scaled - K={K:.2f}): {scaled_efficiency:.2f}")

    # Create a bar chart for efficiency
    plt.figure(figsize=(6, 4))
    plt.bar(['Efficiency'], [scaled_efficiency], color='green')
    plt.ylabel('Efficiency (Scaled)')
    plt.title(f'Scenario {scenario_index + 1} Efficiency')
    plt.ylim(0, 1.2 * TARGET_EFFICIENCY)  # Set y-axis limit slightly above target
    plt.show()

# Create an interactive widget for scenario selection
scenario_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=len(distances) - 1,
    step=1,
    description='Scenario:',
    continuous_update=False
)

interactive_plot = interactive(visualize_scenario, scenario_index=scenario_slider)
display(interactive_plot)

# --- TABLE OF BEST INDIVIDUALS OVER GENERATIONS ---
if best_individual_history:
    best_individuals_df = pd.DataFrame(best_individual_history)
    best_individuals_df['Generation'] = range(len(best_individuals_df))
    best_individuals_df = best_individuals_df[['Generation', 'speed_factor', 'route_preference']]  # Reorder columns

    print("\nTable: Best Individuals Over Generations")
    display(HTML(best_individuals_df.to_html(index=False)))
else:
     print("No best individuals history to display (GA might have failed).")

# --- ANALYTICAL EQUATION DISPLAY ---
print("\nAnalytical Equation:")
equation_text = """
Efficiency = K * [ (Patients / Adjusted Time) / (Adjusted Distance * (1 + Penalization Simulação) + Custo Fixo) ]
Where:
  Adjusted Distance = Distance * (1 - Route Preference * 0.1)
  Adjusted Time = Time * (1 + Route Preference * 0.1)
  Penalization Simulação = 0.8 if Grave Problem Avoided else 0.0
  K = Scaling Factor (to bring efficiency to a meaningful scale)
"""
print(equation_text)

# --- PREDICTION CODE (Using the optimized parameters from the GA) ---
# Remaining structure prediction code is unchanged and executed (as per user request).  If
# this code caused errors or was not desired, you would comment it out here.
num_structures = 2
np.random.seed(42)
helix_radius = 5.0
helix_pitch = 3.0
z_scaling_factor = 1.0

def generate_helical_structure(sequence_length, structure_number, phase_shift=0):
    angles = np.linspace(0, 4 * np.pi, sequence_length) + phase_shift
    random_offsets = np.random.rand(sequence_length) * 1.0
    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  + z_scaling_factor * structure_number
    return x_coords, y_coords, z_coords

all_structure_data = []
results_data = []

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Structures of RNA Sequences")

for index, row in test_df.iterrows():  # Use the simulated test_df
    target_id = row['target_id']
    sequence = row['sequence']
    sequence_length = len(sequence)

    x_coords1, y_coords1, z_coords1 = generate_helical_structure(sequence_length, 1)
    x_coords2, y_coords2, z_coords2 = generate_helical_structure(sequence_length, 2, phase_shift=np.pi)

    ax.plot(x_coords1, y_coords1, z_coords1, c='red', label=f"{target_id} Helix 1")
    ax.plot(x_coords2, y_coords2, z_coords2, c='blue', label=f"{target_id} Helix 2")

    all_coords = np.column_stack((np.concatenate([x_coords1, x_coords2]),
                                   np.concatenate([y_coords1, y_coords2]),
                                   np.concatenate([z_coords1, z_coords2])))

    distance_matrix = pairwise_distances(all_coords)
    avg_distance = np.mean(distance_matrix)

    results_data.append([target_id, sequence_length, avg_distance])

    for residue_index in range(sequence_length):
        resname = sequence[residue_index]
        resid = residue_index + 1
        structure_id = f"{target_id}_1"
        all_structure_data.append([structure_id, resname, resid, x_coords1[residue_index], y_coords1[residue_index], z_coords1[residue_index]])
    for residue_index in range(sequence_length):
        resname = sequence[residue_index]
        resid = residue_index + 1
        structure_id = f"{target_id}_2"
        all_structure_data.append([structure_id, resname, resid, x_coords2[residue_index], y_coords2[residue_index], z_coords2[residue_index]])

ax.legend()
plt.show()

results_df = pd.DataFrame(results_data, columns=['target_id', 'sequence_length', 'average_distance'])

plt.figure(figsize=(10, 6))
plt.bar(results_df['target_id'], results_df['average_distance'])
plt.xlabel("Target ID")
plt.ylabel("Average Distance (Singularity Measure)")
plt.title("Comparison of Singularity Measures")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

col_names = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']

submission_df = pd.DataFrame(all_structure_data, columns=col_names)

print("Code execution complete.  No submission file created.")


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances

# Genetic Algorithm imports
import random
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import seaborn as sns
import folium

# --- Constants and Parameters ---
POPULATION_SIZE = 50
NUM_GENERATIONS = 20
MUTATION_RATE = 0.1

# --- Ambulância simulation parameters ---
BASE_SPEED = 60  # km/h
MAX_SPEED = 100  # km/h
GRAVE_PROBLEM_AVOIDANCE_IMPACT = 0.8  # Reduction if a severe issue avoided

# --- Constants ---
TARGET_EFFICIENCY = 100  # Target efficiency for calculating K
FIXED_COST = 500

# --- Generic patient data ---
# Function to generate random patient coordinates and severity
def generate_patient_data(num_patients):
    patient_locations = []
    for i in range(num_patients):
        latitude = random.uniform(-90, 90)  # Simulate latitudes
        longitude = random.uniform(-180, 180)  # Simulate longitudes
        severity = random.randint(1, 10)  # Severity from 1 to 10
        patient_locations.append((latitude, longitude, severity))
    return patient_locations

# --- Functions (unchanged from previous version) ---
def calculate_lref_d0(df):
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

def create_individual():
    """Creates a random individual (ambulance parameters)."""
    return {
        'speed_factor': random.uniform(0.8, 1.2), # Factor for speed adjustment
        'route_preference': random.uniform(0.0, 1.0)  # Preference for faster vs. shorter route
    }

def calculate_efficiency(individual, distance, patients, time, grave_problem_avoided, fixed_cost):
    """Calculates the efficiency based on the provided equation."""
    # Adjust parameters based on the individual
    effective_speed = BASE_SPEED * individual['speed_factor']
    if effective_speed > MAX_SPEED:
        effective_speed = MAX_SPEED

    # Route adjustment based on 'route_preference' - Simplified here
    adjusted_distance = distance * (1 - individual['route_preference'] * 0.1)  # Shorter route preference reduces distance
    adjusted_time = time * (1 + individual['route_preference'] * 0.1)  # Shorter route preference might increase time slightly

    # Calculate the penalization
    penalization_simulation = GRAVE_PROBLEM_AVOIDANCE_IMPACT if grave_problem_avoided else 0.0

    # Calculate efficiency
    efficiency = ((patients / adjusted_time) / (adjusted_distance * (1 + penalization_simulation) + fixed_cost))

    return efficiency

def evaluate_fitness(individual, distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost):
    """Evaluates the fitness of an individual over a set of scenarios."""
    total_efficiency = 0
    num_scenarios = len(distances)

    for i in range(num_scenarios):
        distance = distances[i]
        patients = patients_list[i]
        time = times_list[i]
        grave_problem_avoided = grave_problems_avoided_list[i]

        efficiency = calculate_efficiency(individual, distance, patients, time, grave_problem_avoided, fixed_cost)
        total_efficiency += efficiency

    # Return the average efficiency
    return total_efficiency / num_scenarios

def crossover(parent1, parent2):
    """Performs crossover between two parents."""
    child = {}
    for key in parent1.keys():
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(individual):
    """Mutates an individual."""
    for key in individual.keys():
        if random.random() < MUTATION_RATE:
            # Adjust the mutation based on the key. For example, we should use a small mutation on factors and larger on route selection.
            if key == 'speed_factor':
                individual[key] += random.uniform(-0.1, 0.1) # Small adjustments
                individual[key] = max(0.8, min(1.2, individual[key]))  # Clamp value.
            else: # route_preference.
                individual[key] += random.uniform(-0.2, 0.2)  # Larger adjustment
                individual[key] = max(0.0, min(1.0, individual[key])) # Clamp value.
    return individual

def genetic_algorithm(distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost):
    """Runs the genetic algorithm."""
    # 1. Initialization
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    fitness_history = []
    best_individual_history = []

    # 2. Evolution
    for generation in range(NUM_GENERATIONS):
        # a. Evaluate Fitness
        fitness_scores = [evaluate_fitness(individual, distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost) for individual in population]

        # Check if fitness_scores is empty
        if not fitness_scores:
            print("Warning: All fitness scores are zero. Stopping the GA.")
            return create_individual(), fitness_history, best_individual_history  # Return a default individual

        # Find best individual only if fitness_scores is not empty
        if fitness_scores:
            best_individual_index = np.argmax(fitness_scores)
            best_individual = population[best_individual_index]
            best_fitness = fitness_scores[best_individual_index]
            fitness_history.append(best_fitness)
            best_individual_history.append(best_individual)
        else:
            # If fitness_scores is empty, append a default value
            best_individual = create_individual()
            best_fitness = 0.0  # Set to a default value
            fitness_history.append(best_fitness)
            best_individual_history.append(best_individual)

        # b. Selection (Tournament Selection)
        selected_indices = random.choices(range(POPULATION_SIZE), weights=fitness_scores, k=POPULATION_SIZE)
        selected_population = [population[i] for i in selected_indices]

        # c. Crossover
        offspring = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1] if i+1 < POPULATION_SIZE else selected_population[i]
            child = crossover(parent1, parent2)
            offspring.append(child)

        # d. Mutation
        mutated_offspring = [mutate(child) for child in offspring]

        # Replace the old population with the new offspring
        population = mutated_offspring

        print(f"Generation {generation}: Best individual = {best_individual}, Fitness = {best_fitness}")

    # 3. Return the Best Individual
    if fitness_scores:
        best_individual_index = np.argmax(fitness_scores)
        return population[best_individual_index], fitness_history, best_individual_history
    else:
        return create_individual(), fitness_history, best_individual_history

# --- MAIN SCRIPT ---

# --- Simulated Data and Load/Create it
# Simulate generic data instead of loading from files
num_sequences = 10
data = {
    'target_id': [f'seq_{i}' for i in range(num_sequences)],
    'sequence': ['AUCG' * (i + 1) for i in range(num_sequences)], # Different sequence lengths
    'description': ['Generic sequence' for _ in range(num_sequences)]
}
test_df = pd.DataFrame(data)
test_df = calculate_lref_d0(test_df)

# --- Create Ambulance Scenario Data: Number and Severity of Cases
num_patients_per_scenario = [random.randint(3, 10) for _ in range(4)] # Random number of patients per scenario
patient_data_per_scenario = [generate_patient_data(num_patients) for num_patients in num_patients_per_scenario]

# --- Assign Scenario - Data - Patients, Time to Solve the Grave Problem or not:
# Make a table and define parameters for each Scenario
scenarios = [
    {"distance": 100, "grave_problem_avoided": True, "time": 2},
    {"distance": 150, "grave_problem_avoided": False, "time": 3},
    {"distance": 200, "grave_problem_avoided": True, "time": 2.5},
    {"distance": 120, "grave_problem_avoided": False, "time": 2.2}
]
distances = [scenario["distance"] for scenario in scenarios]
grave_problems_avoided_list = [scenario["grave_problem_avoided"] for scenario in scenarios]
times_list = [scenario["time"] for scenario in scenarios]
patients_list = num_patients_per_scenario

# --- Run Genetic Algorithm ---
try:
    best_params, fitness_history, best_individual_history = genetic_algorithm(distances, patients_list, times_list, grave_problems_avoided_list, FIXED_COST)
    print("\nBest Ambulance Parameters:", best_params)
except ValueError as e:
    print(f"Error during genetic algorithm execution: {e}")
    best_params = create_individual()  # Use a default individual
    fitness_history = []
    best_individual_history = []

# --- Calculate K factor ---
# To calculate the K factor, let's assume an 'average' scenario:
avg_distance = np.mean(distances)
avg_patients = np.mean(patients_list)
avg_time = np.mean(times_list)
avg_grave_problem_avoided = any(grave_problems_avoided_list) # Assume there's at least one.

# Calculate efficiency with the best parameters
avg_efficiency_with_best_params = calculate_efficiency(
    best_params, avg_distance, avg_patients, avg_time, avg_grave_problem_avoided, FIXED_COST
)

# Calculate K based on the target efficiency
K = TARGET_EFFICIENCY / avg_efficiency_with_best_params

print("Calculated K factor:", K)

# --- Create base map (centered on the mean location) ---
all_latitudes = [loc[0] for scenario_data in patient_data_per_scenario for loc in scenario_data]
all_longitudes = [loc[1] for scenario_data in patient_data_per_scenario for loc in scenario_data]

if all_latitudes and all_longitudes:  # Only create map if patient data exists
    mean_latitude = sum(all_latitudes) / len(all_latitudes)
    mean_longitude = sum(all_longitudes) / len(all_longitudes)

    m = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=6)

    # Add markers for each patient location for each scenario with color corresponding to the simulation
    colors = ['red', 'blue', 'green', 'purple']  # Colors for different scenarios

    for i, patient_data in enumerate(patient_data_per_scenario):
        color = colors[i % len(colors)]  # Get a unique color per scenario
        for latitude, longitude, severity in patient_data:
            folium.CircleMarker(
                location=[latitude, longitude],
                radius=severity * 2, # Circle size corresponding to severity
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.4,
                popup=f'Severity: {severity} - Scenario: {i+1}' # Tooltip with scenario
            ).add_to(m)

    # Display the map
    print("\nPatient Location Map:")
    display(m)
else:
    print("No patient data available to display on the map.")

# --- Plot Fitness History ---
if fitness_history:
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Average Efficiency)")
    plt.title("Genetic Algorithm: Fitness Over Generations")
    plt.grid(True)
    plt.show()
else:
    print("No fitness history to plot (GA might have failed).")

# --- Create Simulation Table ---
scenario_data = []
for i in range(len(scenarios)):
    scenario = scenarios[i]
    efficiency = calculate_efficiency(best_params, distances[i], patients_list[i], times_list[i], grave_problems_avoided_list[i], FIXED_COST)
    scaled_efficiency = efficiency * K
    scenario_data.append({
        "Scenario": i + 1,
        "Distance (km)": scenario["distance"],
        "Patients": patients_list[i],
        "Time (hours)": scenario["time"],
        "Grave Problem Avoided": scenario["grave_problem_avoided"],
        "Unscaled Efficiency": efficiency,
        "Scaled Efficiency": scaled_efficiency
    })
scenario_df = pd.DataFrame(scenario_data)

print("\nSimulation Scenario Table:")
display(HTML(scenario_df.to_html(index=False)))

# --- ANALYTICAL EQUATION DISPLAY ---
print("\nAnalytical Equation:")
equation_text = """
Efficiency = K * [ (Patients / Adjusted Time) / (Adjusted Distance * (1 + Penalization Simulação) + Custo Fixo) ]
Where:
  Adjusted Distance = Distance * (1 - Route Preference * 0.1)
  Adjusted Time = Time * (1 + Route Preference * 0.1)
  Penalization Simulação = 0.8 if Grave Problem Avoided else 0.0
  K = Scaling Factor (to bring efficiency to a meaningful scale)
"""
print(equation_text)

# --- PREDICTION CODE (Using the optimized parameters from the GA) ---
# Remaining structure prediction code is unchanged and executed (as per user request).  If
# this code caused errors or was not desired, you would comment it out here.
num_structures = 2
np.random.seed(42)
helix_radius = 5.0
helix_pitch = 3.0
z_scaling_factor = 1.0

def generate_helical_structure(sequence_length, structure_number, phase_shift=0):
    angles = np.linspace(0, 4 * np.pi, sequence_length) + phase_shift
    random_offsets = np.random.rand(sequence_length) * 1.0
    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  + z_scaling_factor * structure_number
    return x_coords, y_coords, z_coords

all_structure_data = []
results_data = []

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Structures of RNA Sequences")

for index, row in test_df.iterrows():  # Use the simulated test_df
    target_id = row['target_id']
    sequence = row['sequence']
    sequence_length = len(sequence)

    x_coords1, y_coords1, z_coords1 = generate_helical_structure(sequence_length, 1)
    x_coords2, y_coords2, z_coords2 = generate_helical_structure(sequence_length, 2, phase_shift=np.pi)

    ax.plot(x_coords1, y_coords1, z_coords1, c='red', label=f"{target_id} Helix 1")
    ax.plot(x_coords2, y_coords2, z_coords2, c='blue', label=f"{target_id} Helix 2")

    all_coords = np.column_stack((np.concatenate([x_coords1, x_coords2]),
                                   np.concatenate([y_coords1, y_coords2]),
                                   np.concatenate([z_coords1, z_coords2])))

    distance_matrix = pairwise_distances(all_coords)
    avg_distance = np.mean(distance_matrix)

    results_data.append([target_id, sequence_length, avg_distance])

    for residue_index in range(sequence_length):
        resname = sequence[residue_index]
        resid = residue_index + 1
        structure_id = f"{target_id}_1"
        all_structure_data.append([structure_id, resname, resid, x_coords1[residue_index], y_coords1[residue_index], z_coords1[residue_index]])
    for residue_index in range(sequence_length):
        resname = sequence[residue_index]
        resid = residue_index + 1
        structure_id = f"{target_id}_2"
        all_structure_data.append([structure_id, resname, resid, x_coords2[residue_index], y_coords2[residue_index], z_coords2[residue_index]])

ax.legend()
plt.show()

results_df = pd.DataFrame(results_data, columns=['target_id', 'sequence_length', 'average_distance'])

plt.figure(figsize=(10, 6))
plt.bar(results_df['target_id'], results_df['average_distance'])
plt.xlabel("Target ID")
plt.ylabel("Average Distance (Singularity Measure)")
plt.title("Comparison of Singularity Measures")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

col_names = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']

submission_df = pd.DataFrame(all_structure_data, columns=col_names)

print("Code execution complete.  No submission file created.")

# --- Explanation of Simulation ---
print("\nExplanation of Simulation:")
explanation_text = """
This simulation uses a Genetic Algorithm to optimize the parameters of an ambulance service. The goal is to maximize efficiency, which is defined as the number of patients served per unit of time, divided by the total cost of operation.

The parameters that are optimized by the Genetic Algorithm are:
  - Speed Factor: A multiplier for the ambulance's base speed. This represents the ambulance's ability to drive faster.
  - Route Preference: A value between 0 and 1 that represents the ambulance's preference for taking a faster route vs. a shorter route.

The scenarios that are simulated represent different situations the ambulance might encounter. These scenarios vary in terms of:
  - Distance to patients.
  - Number of patients.
  - Time to solve a grave problem or not.
  - Whether a grave problem, such as a traffic jam, was avoided.

The Genetic Algorithm works by creating a population of random individuals. Each individual represents a set of ambulance parameters. The fitness of each individual is evaluated by simulating the ambulance service's operation with those parameters. The individuals with the highest fitness are selected to reproduce and create the next generation. This process is repeated for a number of generations, until the population converges on a set of parameters that maximizes efficiency.

The map is a visualization of patients locations. The simulation table shows the parameters of the ambulance service that were found to be optimal, as well as the efficiency that was achieved with those parameters.
"""
print(explanation_text)


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances

# Genetic Algorithm imports
import random
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import seaborn as sns
import folium

# --- Constants and Parameters ---
POPULATION_SIZE = 50
NUM_GENERATIONS = 20
MUTATION_RATE = 0.1

# --- Ambulância simulation parameters ---
BASE_SPEED = 60  # km/h
MAX_SPEED = 100  # km/h
GRAVE_PROBLEM_AVOIDANCE_IMPACT = 0.8  # Reduction if a severe issue avoided

# --- Constants ---
TARGET_EFFICIENCY = 100  # Target efficiency for calculating K
FIXED_COST = 500

# --- Hydrogen Propulsion Constants ---
HYDROGEN_EFFICIENCY_FACTOR = 2  # Multiplier for the hydrogen propulsion calculation

# --- Generic patient data ---
# Function to generate random patient coordinates and severity
def generate_patient_data(num_patients):
    patient_locations = []
    for i in range(num_patients):
        latitude = random.uniform(-90, 90)  # Simulate latitudes
        longitude = random.uniform(-180, 180)  # Simulate longitudes
        severity = random.randint(1, 10)  # Severity from 1 to 10
        patient_locations.append((latitude, longitude, severity))
    return patient_locations

# --- Functions (unchanged from previous version) ---
def calculate_lref_d0(df):
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

def create_individual():
    """Creates a random individual (ambulance parameters)."""
    return {
        'speed_factor': random.uniform(0.8, 1.2), # Factor for speed adjustment
        'route_preference': random.uniform(0.0, 1.0),  # Preference for faster vs. shorter route
        'hydrogen_efficiency': random.uniform(0.8, 1.2),  # Factor for hydrogen propulsion efficiency
        'hydrogen_station_distance_preference': random.uniform(0.1,100.0) # The R parameter, for distance to the hydrogen station
    }

def calculate_hydrogen_propulsion(speed, R,individual):
    """Calculates hydrogen propulsion efficiency."""
    x = speed  # use speed of ambulance as x for efficiency optimization calculation
    eltze = 10 # Simulate the fixed value for variable eltze

    hydrogen_propulsion = HYDROGEN_EFFICIENCY_FACTOR * (2 * (3 * (x**2)) + ((-21**9) + 3) / (R**2)) / (eltze + 36)
    return hydrogen_propulsion

def calculate_efficiency(individual, distance, patients, time, grave_problem_avoided, fixed_cost):
    """Calculates the efficiency, incorporating hydrogen propulsion."""
    # Adjust parameters based on the individual
    effective_speed = BASE_SPEED * individual['speed_factor']
    if effective_speed > MAX_SPEED:
        effective_speed = MAX_SPEED

    # Route adjustment based on 'route_preference' - Simplified here
    adjusted_distance = distance * (1 - individual['route_preference'] * 0.1)  # Shorter route preference reduces distance
    adjusted_time = time * (1 + individual['route_preference'] * 0.1)  # Shorter route preference might increase time slightly

    # Calculate the penalization
    penalization_simulation = GRAVE_PROBLEM_AVOIDANCE_IMPACT if grave_problem_avoided else 0.0

    # Calculate hydrogen propulsion
    hydrogen_propulsion = calculate_hydrogen_propulsion(effective_speed, individual['hydrogen_station_distance_preference'],individual) * individual['hydrogen_efficiency']

    # Incorporate hydrogen propulsion into efficiency calculation (Example: Additively)
    efficiency = ((patients / adjusted_time) / (adjusted_distance * (1 + penalization_simulation) + fixed_cost)) + hydrogen_propulsion

    return efficiency

def evaluate_fitness(individual, distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost):
    """Evaluates the fitness of an individual over a set of scenarios."""
    total_efficiency = 0
    num_scenarios = len(distances)

    for i in range(num_scenarios):
        distance = distances[i]
        patients = patients_list[i]
        time = times_list[i]
        grave_problem_avoided = grave_problems_avoided_list[i]

        efficiency = calculate_efficiency(individual, distance, patients, time, grave_problem_avoided, fixed_cost)
        total_efficiency += efficiency

    # Return the average efficiency
    return total_efficiency / num_scenarios

def crossover(parent1, parent2):
    """Performs crossover between two parents."""
    child = {}
    for key in parent1.keys():
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(individual):
    """Mutates an individual."""
    for key in individual.keys():
        if random.random() < MUTATION_RATE:
            # Adjust the mutation based on the key. For example, we should use a small mutation on factors and larger on route selection.
            if key == 'speed_factor':
                individual[key] += random.uniform(-0.1, 0.1) # Small adjustments
                individual[key] = max(0.8, min(1.2, individual[key]))  # Clamp value.
            elif key == 'route_preference':
                individual[key] += random.uniform(-0.2, 0.2)  # Larger adjustment
                individual[key] = max(0.0, min(1.0, individual[key])) # Clamp value.
            elif key == 'hydrogen_efficiency':
                individual[key] += random.uniform(-0.1, 0.1) # Small adjustments
                individual[key] = max(0.8, min(1.2, individual[key]))
            else: # hydrogen_station_distance_preference, R parameter in propulsion calculation
                individual[key] += random.uniform(-10, 10) # Adjust R for the hydrogen propulsion calculation
                individual[key] = max(0.1, min(100.0, individual[key]))  #R must be positive.
    return individual

def genetic_algorithm(distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost):
    """Runs the genetic algorithm."""
    # 1. Initialization
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    fitness_history = []
    best_individual_history = []

    # 2. Evolution
    for generation in range(NUM_GENERATIONS):
        # a. Evaluate Fitness
        fitness_scores = [evaluate_fitness(individual, distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost) for individual in population]

        # Check if fitness_scores is empty
        if not fitness_scores:
            print("Warning: All fitness scores are zero. Stopping the GA.")
            return create_individual(), fitness_history, best_individual_history  # Return a default individual

        # Find best individual only if fitness_scores is not empty
        if fitness_scores:
            best_individual_index = np.argmax(fitness_scores)
            best_individual = population[best_individual_index]
            best_fitness = fitness_scores[best_individual_index]
            fitness_history.append(best_fitness)
            best_individual_history.append(best_individual)
        else:
            # If fitness_scores is empty, append a default value
            best_individual = create_individual()
            best_fitness = 0.0  # Set to a default value
            fitness_history.append(best_fitness)
            best_individual_history.append(best_individual)

        # b. Selection (Tournament Selection)
        selected_indices = random.choices(range(POPULATION_SIZE), weights=fitness_scores, k=POPULATION_SIZE)
        selected_population = [population[i] for i in selected_indices]

        # c. Crossover
        offspring = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1] if i+1 < POPULATION_SIZE else selected_population[i]
            child = crossover(parent1, parent2)
            offspring.append(child)

        # d. Mutation
        mutated_offspring = [mutate(child) for child in offspring]

        # Replace the old population with the new offspring
        population = mutated_offspring

        print(f"Generation {generation}: Best individual = {best_individual}, Fitness = {best_fitness}")

    # 3. Return the Best Individual
    if fitness_scores:
        best_individual_index = np.argmax(fitness_scores)
        return population[best_individual_index], fitness_history, best_individual_history
    else:
        return create_individual(), fitness_history, best_individual_history

# --- MAIN SCRIPT ---

# --- Simulated Data and Load/Create it
# Simulate generic data instead of loading from files
num_sequences = 10
data = {
    'target_id': [f'seq_{i}' for i in range(num_sequences)],
    'sequence': ['AUCG' * (i + 1) for i in range(num_sequences)], # Different sequence lengths
    'description': ['Generic sequence' for _ in range(num_sequences)]
}
test_df = pd.DataFrame(data)
test_df = calculate_lref_d0(test_df)

# --- Create Ambulance Scenario Data: Number and Severity of Cases
num_patients_per_scenario = [random.randint(3, 10) for _ in range(4)] # Random number of patients per scenario
patient_data_per_scenario = [generate_patient_data(num_patients) for num_patients in num_patients_per_scenario]

# --- Assign Scenario - Data - Patients, Time to Solve the Grave Problem or not:
# Make a table and define parameters for each Scenario
scenarios = [
    {"distance": 100, "grave_problem_avoided": True, "time": 2},
    {"distance": 150, "grave_problem_avoided": False, "time": 3},
    {"distance": 200, "grave_problem_avoided": True, "time": 2.5},
    {"distance": 120, "grave_problem_avoided": False, "time": 2.2}
]
distances = [scenario["distance"] for scenario in scenarios]
grave_problems_avoided_list = [scenario["grave_problem_avoided"] for scenario in scenarios]
times_list = [scenario["time"] for scenario in scenarios]
patients_list = num_patients_per_scenario

# --- Run Genetic Algorithm ---
try:
    best_params, fitness_history, best_individual_history = genetic_algorithm(distances, patients_list, times_list, grave_problems_avoided_list, FIXED_COST)
    print("\nBest Ambulance Parameters:", best_params)
except ValueError as e:
    print(f"Error during genetic algorithm execution: {e}")
    best_params = create_individual()  # Use a default individual
    fitness_history = []
    best_individual_history = []

# --- Calculate K factor ---
# To calculate the K factor, let's assume an 'average' scenario:
avg_distance = np.mean(distances)
avg_patients = np.mean(patients_list)
avg_time = np.mean(times_list)
avg_grave_problem_avoided = any(grave_problems_avoided_list) # Assume there's at least one.

# Calculate efficiency with the best parameters
avg_efficiency_with_best_params = calculate_efficiency(
    best_params, avg_distance, avg_patients, avg_time, avg_grave_problem_avoided, FIXED_COST
)

# Calculate K based on the target efficiency
K = TARGET_EFFICIENCY / avg_efficiency_with_best_params

print("Calculated K factor:", K)

# --- Create base map (centered on the world) ---
m = folium.Map(location=[0, 0], zoom_start=2)  # Center on 0,0 and zoom out

# Add markers for each patient location for each scenario with color corresponding to the simulation
colors = ['red', 'blue', 'green', 'purple']  # Colors for different scenarios

for i, patient_data in enumerate(patient_data_per_scenario):
    color = colors[i % len(colors)]  # Get a unique color per scenario
    for latitude, longitude, severity in patient_data:
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=severity * 2, # Circle size corresponding to severity
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.4,
            popup=f'Severity: {severity} - Scenario: {i+1}' # Tooltip with scenario
        ).add_to(m)

# Display the map
print("\nPatient Location Map:")
display(m)

# --- Plot Fitness History ---
if fitness_history:
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Average Efficiency)")
    plt.title("Genetic Algorithm: Fitness Over Generations")
    plt.grid(True)
    plt.show()
else:
    print("No fitness history to plot (GA might have failed).")

# --- Create Simulation Table ---
scenario_data = []
for i in range(len(scenarios)):
    scenario = scenarios[i]
    efficiency = calculate_efficiency(best_params, distances[i], patients_list[i], times_list[i], grave_problems_avoided_list[i], FIXED_COST)
    scaled_efficiency = efficiency * K
    scenario_data.append({
        "Scenario": i + 1,
        "Distance (km)": scenario["distance"],
        "Patients": patients_list[i],
        "Time (hours)": scenario["time"],
        "Grave Problem Avoided": scenario["grave_problem_avoided"],
        "Unscaled Efficiency": efficiency,
        "Scaled Efficiency": scaled_efficiency
    })
scenario_df = pd.DataFrame(scenario_data)

print("\nSimulation Scenario Table:")
display(HTML(scenario_df.to_html(index=False)))

# --- ANALYTICAL EQUATION DISPLAY ---
print("\nAnalytical Equation:")
equation_text = """
Overall Efficiency = Scaled Service Efficiency + Hydrogen Propulsion Efficiency
Where:

Service Efficiency = K * [ (Patients / Adjusted Time) / (Adjusted Distance * (1 + Penalization Simulação) + Custo Fixo) ]
  Adjusted Distance = Distance * (1 - Route Preference * 0.1)
  Adjusted Time = Time * (1 + Route Preference * 0.1)
  Penalization Simulação = 0.8 if Grave Problem Avoided else 0.0
  K = Scaling Factor (to bring efficiency to a meaningful scale)

Hydrogen Propulsion Efficiency = HYDROGEN_EFFICIENCY_FACTOR * (2 * (3 * (x**2)) + ((-21**9) + 3) / (R**2)) / (eltze + 36)
  Where:
    x = Speed of the ambulance
    R = Individual value for distance to station
    eltze = a fixed parameter (10, by default in this simulation)
"""
print(equation_text)

# --- PREDICTION CODE (Using the optimized parameters from the GA) ---
# Remaining structure prediction code is unchanged and executed (as per user request).  If
# this code caused errors or was not desired, you would comment it out here.
num_structures = 2
np.random.seed(42)
helix_radius = 5.0
helix_pitch = 3.0
z_scaling_factor = 1.0

def generate_helical_structure(sequence_length, structure_number, phase_shift=0):
    angles = np.linspace(0, 4 * np.pi, sequence_length) + phase_shift
    random_offsets = np.random.rand(sequence_length) * 1.0
    x_coords = helix_radius * np.cos(angles) + random_offsets
    y_coords = helix_radius * np.sin(angles) + random_offsets
    z_coords = helix_pitch * angles  + z_scaling_factor * structure_number
    return x_coords, y_coords, z_coords

all_structure_data = []
results_data = []

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Structures of RNA Sequences")

for index, row in test_df.iterrows():  # Use the simulated test_df
    target_id = row['target_id']
    sequence = row['sequence']
    sequence_length = len(sequence)

    x_coords1, y_coords1, z_coords1 = generate_helical_structure(sequence_length, 1)
    x_coords2, y_coords2, z_coords2 = generate_helical_structure(sequence_length, 2, phase_shift=np.pi)

    ax.plot(x_coords1, y_coords1, z_coords1, c='red', label=f"{target_id} Helix 1")
    ax.plot(x_coords2, y_coords2, z_coords2, c='blue', label=f"{target_id} Helix 2")

    all_coords = np.column_stack((np.concatenate([x_coords1, x_coords2]),
                                   np.concatenate([y_coords1, y_coords2]),
                                   np.concatenate([z_coords1, z_coords2])))

    distance_matrix = pairwise_distances(all_coords)
    avg_distance = np.mean(distance_matrix)

    results_data.append([target_id, sequence_length, avg_distance])

    for residue_index in range(sequence_length):
        resname = sequence[residue_index]
        resid = residue_index + 1
        structure_id = f"{target_id}_1"
        all_structure_data.append([structure_id, resname, resid, x_coords1[residue_index], y_coords1[residue_index], z_coords1[residue_index]])
    for residue_index in range(sequence_length):
        resname = sequence[residue_index]
        resid = residue_index + 1
        structure_id = f"{target_id}_2"
        all_structure_data.append([structure_id, resname, resid, x_coords2[residue_index], y_coords2[residue_index], z_coords2[residue_index]])

ax.legend()
plt.show()

results_df = pd.DataFrame(results_data, columns=['target_id', 'sequence_length', 'average_distance'])

plt.figure(figsize=(10, 6))
plt.bar(results_df['target_id'], results_df['average_distance'])
plt.xlabel("Target ID")
plt.ylabel("Average Distance (Singularity Measure)")
plt.title("Comparison of Singularity Measures")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

col_names = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']

submission_df = pd.DataFrame(all_structure_data, columns=col_names)

print("Code execution complete.  No submission file created.")


import pandas as pd
import math
import os
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances

# Genetic Algorithm imports
import random
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import seaborn as sns
import folium

# --- Constants and Parameters ---
POPULATION_SIZE = 50
NUM_GENERATIONS = 20
MUTATION_RATE = 0.1

# --- Ambulância simulation parameters ---
BASE_SPEED = 60  # km/h
MAX_SPEED = 100  # km/h
GRAVE_PROBLEM_AVOIDANCE_IMPACT = 0.8  # Reduction if a severe issue avoided

# --- Constants ---
TARGET_EFFICIENCY = 100  # Target efficiency for calculating K
FIXED_COST = 500

# --- Hydrogen Propulsion Constants ---
HYDROGEN_EFFICIENCY_FACTOR = 2  # Multiplier for the hydrogen propulsion calculation

# --- Generic patient data ---
# Function to generate random patient coordinates and severity
def generate_patient_data(num_patients):
    patient_locations = []
    for i in range(num_patients):
        latitude = random.uniform(-90, 90)  # Simulate latitudes
        longitude = random.uniform(-180, 180)  # Simulate longitudes
        severity = random.randint(1, 10)  # Severity from 1 to 10
        patient_locations.append((latitude, longitude, severity))
    return patient_locations

# --- Functions (unchanged from previous version) ---
def calculate_lref_d0(df):
    df['Lref'] = df['sequence'].apply(len)
    df['d0'] = df['Lref'].apply(lambda lref: 0.6 * math.sqrt(lref - 0.5) - 2.5)
    return df

def create_individual():
    """Creates a random individual (ambulance parameters)."""
    return {
        'speed_factor': random.uniform(0.8, 1.2), # Factor for speed adjustment
        'route_preference': random.uniform(0.0, 1.0),  # Preference for faster vs. shorter route
        'hydrogen_efficiency': random.uniform(0.8, 1.2),  # Factor for hydrogen propulsion efficiency
        'hydrogen_station_distance_preference': random.uniform(0.1,100.0) # The R parameter, for distance to the hydrogen station
    }

def calculate_hydrogen_propulsion(speed, R,individual):
    """Calculates hydrogen propulsion efficiency."""
    x = speed  # use speed of ambulance as x for efficiency optimization calculation
    eltze = 10 # Simulate the fixed value for variable eltze

    hydrogen_propulsion = HYDROGEN_EFFICIENCY_FACTOR * (2 * (3 * (x**2)) + ((-21**9) + 3) / (R**2)) / (eltze + 36)
    return hydrogen_propulsion

def calculate_efficiency(individual, distance, patients, time, grave_problem_avoided, fixed_cost):
    """Calculates the efficiency, incorporating hydrogen propulsion."""
    # Adjust parameters based on the individual
    effective_speed = BASE_SPEED * individual['speed_factor']
    if effective_speed > MAX_SPEED:
        effective_speed = MAX_SPEED

    # Route adjustment based on 'route_preference' - Simplified here
    adjusted_distance = distance * (1 - individual['route_preference'] * 0.1)  # Shorter route preference reduces distance
    adjusted_time = time * (1 + individual['route_preference'] * 0.1)  # Shorter route preference might increase time slightly

    # Calculate the penalization
    penalization_simulation = GRAVE_PROBLEM_AVOIDANCE_IMPACT if grave_problem_avoided else 0.0

    # Calculate hydrogen propulsion
    hydrogen_propulsion = calculate_hydrogen_propulsion(effective_speed, individual['hydrogen_station_distance_preference'],individual) * individual['hydrogen_efficiency']

    # Incorporate hydrogen propulsion into efficiency calculation (Example: Additively)
    efficiency = ((patients / adjusted_time) / (adjusted_distance * (1 + penalization_simulation) + fixed_cost)) + hydrogen_propulsion

    return efficiency

def evaluate_fitness(individual, distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost):
    """Evaluates the fitness of an individual over a set of scenarios."""
    total_efficiency = 0
    num_scenarios = len(distances)

    for i in range(num_scenarios):
        distance = distances[i]
        patients = patients_list[i]
        time = times_list[i]
        grave_problem_avoided = grave_problems_avoided_list[i]

        efficiency = calculate_efficiency(individual, distance, patients, time, grave_problem_avoided, fixed_cost)
        total_efficiency += efficiency

    # Return the average efficiency
    return total_efficiency / num_scenarios

def crossover(parent1, parent2):
    """Performs crossover between two parents."""
    child = {}
    for key in parent1.keys():
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(individual):
    """Mutates an individual."""
    for key in individual.keys():
        if random.random() < MUTATION_RATE:
            # Adjust the mutation based on the key. For example, we should use a small mutation on factors and larger on route selection.
            if key == 'speed_factor':
                individual[key] += random.uniform(-0.1, 0.1) # Small adjustments
                individual[key] = max(0.8, min(1.2, individual[key]))  # Clamp value.
            elif key == 'route_preference':
                individual[key] += random.uniform(-0.2, 0.2)  # Larger adjustment
                individual[key] = max(0.0, min(1.0, individual[key])) # Clamp value.
            elif key == 'hydrogen_efficiency':
                individual[key] += random.uniform(-0.1, 0.1) # Small adjustments
                individual[key] = max(0.8, min(1.2, individual[key]))
            else: # hydrogen_station_distance_preference, R parameter in propulsion calculation
                individual[key] += random.uniform(-10, 10) # Adjust R for the hydrogen propulsion calculation
                individual[key] = max(0.1, min(100.0, individual[key]))  #R must be positive.
    return individual

def genetic_algorithm(distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost):
    """Runs the genetic algorithm."""
    # 1. Initialization
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    fitness_history = []
    best_individual_history = []

    # 2. Evolution
    for generation in range(NUM_GENERATIONS):
        # a. Evaluate Fitness
        fitness_scores = [evaluate_fitness(individual, distances, patients_list, times_list, grave_problems_avoided_list, fixed_cost) for individual in population]

        # Check if fitness_scores is empty
        if not fitness_scores:
            print("Warning: All fitness scores are zero. Stopping the GA.")
            return create_individual(), fitness_history, best_individual_history  # Return a default individual

        # Find best individual only if fitness_scores is not empty
        if fitness_scores:
            best_individual_index = np.argmax(fitness_scores)
            best_individual = population[best_individual_index]
            best_fitness = fitness_scores[best_individual_index]
            fitness_history.append(best_fitness)
            best_individual_history.append(best_individual)
        else:
            # If fitness_scores is empty, append a default value
            best_individual = create_individual()
            best_fitness = 0.0  # Set to a default value
            fitness_history.append(best_fitness)
            best_individual_history.append(best_individual)

        # b. Selection (Tournament Selection)
        selected_indices = random.choices(range(POPULATION_SIZE), weights=fitness_scores, k=POPULATION_SIZE)
        selected_population = [population[i] for i in selected_indices]

        # c. Crossover
        offspring = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1] if i+1 < POPULATION_SIZE else selected_population[i]
            child = crossover(parent1, parent2)
            offspring.append(child)

        # d. Mutation
        mutated_offspring = [mutate(child) for child in offspring]

        # Replace the old population with the new offspring
        population = mutated_offspring

        print(f"Generation {generation}: Best individual = {best_individual}, Fitness = {best_fitness}")

    # 3. Return the Best Individual
    if fitness_scores:
        best_individual_index = np.argmax(fitness_scores)
        return population[best_individual_index], fitness_history, best_individual_history
    else:
        return create_individual(), fitness_history, best_individual_history

# --- MAIN SCRIPT ---

# --- Simulated Data and Load/Create it
# Simulate generic data instead of loading from files
num_sequences = 10
data = {
    'target_id': [f'seq_{i}' for i in range(num_sequences)],
    'sequence': ['AUCG' * (i + 1) for i in range(num_sequences)], # Different sequence lengths
    'description': ['Generic sequence' for _ in range(num_sequences)]
}
test_df = pd.DataFrame(data)
test_df = calculate_lref_d0(test_df)

# --- Create Ambulance Scenario Data: Number and Severity of Cases
num_patients_per_scenario = [random.randint(3, 10) for _ in range(4)] # Random number of patients per scenario
patient_data_per_scenario = [generate_patient_data(num_patients) for num_patients in num_patients_per_scenario]

# --- Assign Scenario - Data - Patients, Time to Solve the Grave Problem or not:
# Make a table and define parameters for each Scenario
scenarios = [
    {"distance": 100, "grave_problem_avoided": True, "time": 2},
    {"distance": 150, "grave_problem_avoided": False, "time": 3},
    {"distance": 200, "grave_problem_avoided": True, "time": 2.5},
    {"distance": 120, "grave_problem_avoided": False, "time": 2.2}
]
distances = [scenario["distance"] for scenario in scenarios]
grave_problems_avoided_list = [scenario["grave_problem_avoided"] for scenario in scenarios]
times_list = [scenario["time"] for scenario in scenarios]
patients_list = num_patients_per_scenario

# --- Run Genetic Algorithm ---
try:
    best_params, fitness_history, best_individual_history = genetic_algorithm(distances, patients_list, times_list, grave_problems_avoided_list, FIXED_COST)
    print("\nBest Ambulance Parameters:", best_params)
except ValueError as e:
    print(f"Error during genetic algorithm execution: {e}")
    best_params = create_individual()  # Use a default individual
    fitness_history = []
    best_individual_history = []

# --- Calculate K factor ---
# To calculate the K factor, let's assume an 'average' scenario:
avg_distance = np.mean(distances)
avg_patients = np.mean(patients_list)
avg_time = np.mean(times_list)
avg_grave_problem_avoided = any(grave_problems_avoided_list) # Assume there's at least one.

# Calculate efficiency with the best parameters
avg_efficiency_with_best_params = calculate_efficiency(
    best_params, avg_distance, avg_patients, avg_time, avg_grave_problem_avoided, FIXED_COST
)

# Calculate K based on the target efficiency
K = TARGET_EFFICIENCY / avg_efficiency_with_best_params

print("Calculated K factor:", K)

# --- Prepare Data for Output File ---
output_data = {
    "best_params": best_params,
    "fitness_history": fitness_history,
    "distances": distances,
    "patients_list": patients_list,
    "times_list": times_list,
    "grave_problems_avoided_list": grave_problems_avoided_list,
    "K": K,
    "best_RNA" : test_df.to_dict(),
    "best_test" : test_df.to_dict(),
    "ex" : patient_data_per_scenario
}

# --- Output File Creation ---
output_filename = "ambulance_simulation_data.world"
output_path = os.path.join("/kaggle/working", output_filename)

# --- Write the world file
# Open the file for writing
with open(output_path, 'w') as f:
    # Write header
    f.write("// Ambulance Simulation Data\n\n")

    # Write each of the data
    f.write("// Best Parameters:\n")
    for k, v in output_data["best_params"].items():
        f.write(f"//   {k}: {v}\n")
    f.write("\n")

    f.write("// Fitness History:\n")
    f.write(f"//   {str(output_data['fitness_history'])}\n")
    f.write("\n")

    f.write("// Distances:\n")
    f.write(f"//   {str(output_data['distances'])}\n")
    f.write("\n")

    f.write("// Patients List:\n")
    f.write(f"//   {str(output_data['patients_list'])}\n")
    f.write("\n")

    f.write("// Times List:\n")
    f.write(f"//   {str(output_data['times_list'])}\n")
    f.write("\n")

    f.write("// Grave Problems Avoided List:\n")
    f.write(f"//   {str(output_data['grave_problems_avoided_list'])}\n")
    f.write("\n")

    f.write("// K Value:\n")
    f.write(f"//   {str(output_data['K'])}\n")
    f.write("\n")

    f.write("// Best RNA Data:\n")
    f.write(f"//   {str(output_data['best_RNA'])}\n")
    f.write("\n")

    f.write("// Best Test Data:\n")
    f.write(f"//   {str(output_data['best_test'])}\n")
    f.write("\n")

    f.write("// Patient Data per Scenario:\n")
    f.write(f"//   {str(output_data['ex'])}\n")
    f.write("\n")

# --- Verify File Creation ---
if os.path.exists(output_path):
    print(f"Successfully created the file: {output_path}")
else:
    print("Failed to create the file.")

# --- Clear any leftover data ---
all_structure_data = []
results_data = []

