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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
import random
import math  # Import math for sqrt
import folium
from IPython.display import display, HTML
from sklearn.metrics import pairwise_distances
from mpl_toolkits.mplot3d import Axes3D

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

# --- Report Generation: SBIR Application Framework ---
report = """
# SBIR Phase I Proposal: Integrated Urban Resilience System

## 1. Executive Summary

This SBIR proposal outlines a novel integrated system designed to enhance urban resilience in the face of climate change and increasing urban demands. Our proposed system combines advanced water treatment technology, sustainable energy solutions, and optimized emergency response capabilities, all driven by data-driven insights and innovative algorithms.  The core components are:

*   **Rainwater Harvesting and Ozone Treatment System ("Totem"):**  A decentralized water treatment station that captures rainwater, treats it using ozone and hydrogen, and provides clean water for local use.
*   **Hydrogen-Powered Ambulance Fleet:** An optimized fleet of ambulances utilizing hydrogen propulsion for cleaner and more efficient emergency response.
*   **Data-Driven Optimization Platform:** A platform that leverages real-time data and advanced algorithms to optimize ambulance dispatch, water treatment operations, and energy distribution.

## 2. Problem Statement

Urban areas face increasing challenges due to:

*   **Water Scarcity:** Climate change and population growth exacerbate water scarcity issues. Traditional water treatment facilities are often centralized and vulnerable to disruptions.
*   **Air Pollution:**  Fossil fuel-powered vehicles contribute to air pollution, especially during emergency response scenarios.
*   **Inefficient Emergency Response:**  Traditional ambulance dispatch systems may not be optimized for real-time conditions, leading to delays and increased costs.

## 3. Proposed Solution

Our integrated system addresses these challenges through:

*   **Decentralized Water Treatment:** The "Totem" provides a sustainable and resilient water source, reducing reliance on centralized systems.
*   **Clean Energy Transportation:**  Hydrogen-powered ambulances eliminate tailpipe emissions, improving air quality and reducing carbon footprint.
*   **Optimized Resource Allocation:**  Our data-driven platform optimizes ambulance dispatch, water treatment operations, and energy distribution based on real-time conditions and predictive analytics.

## 4. Technical Approach

### 4.1. Rainwater Harvesting and Ozone Treatment System ("Totem")

*   **Design and Engineering:** Develop a modular and scalable "Totem" design that can be adapted to various urban environments.
*   **Ozone and Hydrogen Treatment:** Implement an advanced ozone and hydrogen treatment process to remove contaminants and pathogens from rainwater.
*   **System Integration:** Integrate the water treatment system with a rainwater harvesting system and a water storage tank.

### 4.2. Hydrogen-Powered Ambulance Fleet

*   **Ambulance Conversion:**  Convert existing ambulances to utilize hydrogen fuel cell technology.
*   **Hydrogen Refueling Infrastructure:**  Establish a hydrogen refueling station near the central ambulance dispatch location.
*   **Performance Optimization:**  Optimize the ambulance performance for emergency response scenarios using data-driven insights.

### 4.3. Data-Driven Optimization Platform

*   **Data Acquisition:**  Collect real-time data from various sources, including weather sensors, traffic sensors, patient data, and energy consumption data.
*   **Algorithm Development:**  Develop advanced algorithms to optimize ambulance dispatch, water treatment operations, and energy distribution.
*   **Platform Integration:**  Integrate the data-driven platform with the "Totem" and the hydrogen-powered ambulance fleet.

## 5. Innovation

This project introduces several key innovations:

*   **Integrated Approach:**  Combines water treatment, sustainable energy, and emergency response into a single, cohesive system.
*   **Data-Driven Optimization:**  Leverages real-time data and advanced algorithms to optimize resource allocation and improve system performance.
*   **Modular Design:**  The "Totem" is designed to be modular and scalable, allowing for adaptation to various urban environments.

## 6. Commercial Potential

The proposed system has significant commercial potential:

*   **Urban Resilience Market:**  There is a growing market for solutions that enhance urban resilience in the face of climate change and increasing urban demands.
*   **Government Funding:**  Government agencies are increasingly investing in sustainable infrastructure projects.
*   **Licensing Opportunities:**  The technology developed in this project can be licensed to city governments, water utilities, and emergency response agencies.

## 7. Team

Our team comprises experts in:

*   **Water Treatment Engineering**
*   **Sustainable Energy Technology**
*   **Data Science and Algorithm Development**
*   **Urban Planning and Emergency Response**

## 8. Budget

The Phase I budget will be allocated to:

*   **System Design and Engineering**
*   **Algorithm Development and Data Analysis**
*   **Market Research and Commercialization Planning**

## 9. Timeline

The Phase I activities will be completed within [Insert Timeline - e.g., 6 months].

## 10. Expected Outcomes

The expected outcomes of this Phase I project are:

*   **Detailed system design and engineering specifications.**
*   **Functional prototype of the data-driven optimization platform.**
*   **Market research report and commercialization plan.**
*   **Feasibility assessment of the hydrogen-powered ambulance fleet.**

## 11. Figures and Tables

[Insert Figures and Tables from the code output, demonstrating results]

*   **Figure 1: Patient Location Map**
*   **Figure 2: Fitness History Plot**
*   **Table 1: Simulation Scenario Table**
*   **Analytical Equation:**
    """ + equation_text + """
*   **3D Structures of RNA Sequences"**
*   **Comparison of Singularity Measures"**

## 12. Conclusions

This integrated system offers a promising approach to enhance urban resilience and address critical challenges related to water scarcity, air pollution, and emergency response.  The proposed project has the potential to generate significant economic, social, and environmental benefits for urban communities.

"""

print(report)
print("Complete")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# --- Business Model: Integrated Urban Cooling and Emergency Response ---

# Value Proposition: Reduce heat island effects, improve water quality, and enhance emergency response efficiency, contributing to increased productivity and a healthier urban environment, ultimately boosting the US GDP.

# Target Customers: City governments, municipalities, urban development agencies.

# Revenue Streams:
# 1.  Initial System Installation and Integration: Fees for installing the integrated "Totem" water treatment and cooling system.
# 2.  Maintenance and Service Contracts: Recurring revenue for maintenance, monitoring, and upgrades of the system.
# 3.  Data Analytics Subscriptions: Fees for providing data-driven insights on system performance, environmental impact, and optimized resource allocation.
# 4.  Carbon Credit Generation and Sales: Revenue from generating and selling carbon credits due to reduced energy consumption and emissions.
# 5.  Ambulance Service Fee optimization : Revenue from optimizing ambulance service costs and fees

# Cost Structure:
# 1.  System Manufacturing and Installation Costs: Costs associated with producing and installing the "Totem" and hydrogen ambulance systems.
# 2.  Operating and Maintenance Costs: Ongoing costs for system maintenance, energy consumption, and personnel.
# 3.  Data Acquisition and Processing Costs: Expenses for collecting, storing, and processing data.
# 4.  Research and Development Costs: Investment in continuous innovation and improvement of the system.
# 5.  Marketing and Sales Costs: Expenses for promoting and selling the system.

# --- Simulated Data: Heat Island Effect Reduction ---

# Assumptions:
# 1.  "Totem" coverage: The fraction of the urban area covered by "Totem" cooling systems.
# 2.  Base temperature: Average baseline temperature of the city during summer months (degrees Celsius).
# 3.  Cooling effect: Average temperature reduction per "Totem" unit installed (degrees Celsius).
# 4.  Area affected by temperature changes.
# 5.  Area affected by water treatiment

# Data
city_area = 100  # square kilometers
totem_coverage_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # 10%, 20%, 30%, 40%, 50% coverage
base_temperature = 35  # degrees Celsius
cooling_effect_per_totem = 0.5 # degrees Celsius, for each 10% of coverage
treatment_effectiveness = 0.2 # Factor for water treatiment.
population = 1000000  # Number of people living in the area
productivity_increase_per_degree = 0.01  # 1% increase in productivity per degree reduced
health_cost_reduction_per_degree = 0.005 # 0.5% reduction in health costs per degree reduced
water_treatment_improvement_multiplier = 0.1 # Factor for water treatiment.

# Calculate Temperature Reduction and Economic Impact
temperature_reduction = [coverage * cooling_effect_per_totem * (1 + treatment_effectiveness * water_treatment_improvement_multiplier) for coverage in totem_coverage_levels] # Calculate a better temperature reduction due to the water treatiment
productivity_increase = [reduction * productivity_increase_per_degree for reduction in temperature_reduction] # Calculate the productivity based on the temperature reduction
health_cost_reduction = [reduction * health_cost_reduction_per_degree for reduction in temperature_reduction] # Calculate the cost reduction based on the temperature reduction

# Create a DataFrame
df_cooling = pd.DataFrame({
    'Totem Coverage (%)': [coverage * 100 for coverage in totem_coverage_levels],
    'Temperature Reduction (C)': temperature_reduction,
    'Productivity Increase (%)': productivity_increase,
    'Health Cost Reduction (%)': health_cost_reduction
})

# Display Table
print("\nHeat Island Effect Reduction and Economic Impact:")
display(HTML(df_cooling.to_html(index=False)))

# --- Simulated Data: Ambulance Efficiency Improvement ---

# Assumptions:
# 1. Base Ambulance Efficiency: A baseline measure for ambulance operations per year.
# 2. Efficiency Increase per Improvement: The increase achieved with the integrated system.
# 3. Value of Improved Emergency Response: The monetary savings associated with the improved emergency response.

# Data
base_ambulance_efficiency = 1000  # Calls handled per year
efficiency_increase_levels = [0.05, 0.1, 0.15, 0.2, 0.25]  # 5%, 10%, 15%, 20%, 25% efficiency increase
value_per_emergency_call = 500  # Dollars, in terms of lives saved and averted economic damages

# Calculate the benefits of increased efficiency
efficiency_increase = [increase * base_ambulance_efficiency for increase in efficiency_increase_levels] # Calculate a total improvemente in call number due to increase
economic_benefit = [increase * value_per_emergency_call for increase in efficiency_increase]  # Calculate how much save in the area by increasing the efficiency

# Create a DataFrame
df_ambulance = pd.DataFrame({
    'Efficiency Increase (%)': [increase * 100 for increase in efficiency_increase_levels],
    'Calls Handle Increase': efficiency_increase,
    'Economic Benefit ($)': economic_benefit
})

# Display Table
print("\nAmbulance Efficiency Improvement and Economic Benefit:")
display(HTML(df_ambulance.to_html(index=False)))

# --- GDP Impact Analysis ---

# Assumptions:
# 1. Population: The population of the city that benefits from the integrated system.
# 2. Workforce Participation Rate: the percentage of people in a city that are actively working
# 3. Average Salary: Average yearly salary per worker (dollars).
# 4. Health cost per person: Average yearly spend on health
# 5. Fraction: Area or city part to test

# Parameters
workforce_participation_rate = 0.6 # Fraction of pop actively employed.
average_salary = 60000 # USD/Year
health_cost_per_person = 2000 # USD/Year
fraction = 0.1 # Part of the area to test

# Economic Impact of Heat Reduction
base_gdp = population * workforce_participation_rate * average_salary # Calculate how much the base area produce
productivity_value_increases = [base_gdp * increase * fraction for increase in productivity_increase] # Calculate how much increase per temp reduction
health_saving_increases = [health_cost_per_person * population * increase * fraction for increase in health_cost_reduction] # Calculate how much save per health reduction

# Economic Impact of Ambulances
economic_ambulance_benefit = [benefit * fraction for benefit in economic_benefit]

# Create dataFrame
data = []

# Store parameters for print at the last table
parameters = {
    'city_area' : city_area,
    'base_temperature': base_temperature,
    'cooling_effect_per_totem': cooling_effect_per_totem,
    'treatment_effectiveness' : treatment_effectiveness,
    'population' : population,
    'productivity_increase_per_degree' : productivity_increase_per_degree,
    'health_cost_reduction_per_degree' : health_cost_reduction_per_degree,
    'water_treatment_improvement_multiplier' : water_treatment_improvement_multiplier,
    'base_ambulance_efficiency': base_ambulance_efficiency,
    'value_per_emergency_call' : value_per_emergency_call,
    'workforce_participation_rate' : workforce_participation_rate,
    'average_salary' : average_salary,
    'health_cost_per_person' : health_cost_per_person,
    'fraction' : fraction
}

for i in range(len(totem_coverage_levels)):
    data.append([
        totem_coverage_levels[i] * 100,
        temperature_reduction[i],
        productivity_value_increases[i],
        health_saving_increases[i],
        efficiency_increase_levels[i] * 100,
        economic_ambulance_benefit[i],
        productivity_value_increases[i] + economic_ambulance_benefit[i]
    ])

# Create Data Frame to store values of improvement
df_gdp_impact = pd.DataFrame(data, columns=[
    'Totem Coverage (%)',
    'Temperature Reduction (°C)',
    'GDP Increase (USD)',
    'Health Cost Savings (USD)',
    'Ambulance Efficiency Increase (%)',
    'Economic Benefit (USD)',
    'Total Economic Impact (USD)'
])

# Print data frame
print("GDP Impact Analysis (per year):")
display(HTML(df_gdp_impact.to_html(index=False)))

# Plot economic increases in area
plt.figure(figsize=(10, 6))
plt.plot(df_gdp_impact['Totem Coverage (%)'], df_gdp_impact['GDP Increase (USD)'], marker='o', label='GDP Increase (Heat)')
plt.plot(df_gdp_impact['Totem Coverage (%)'], df_gdp_impact['Economic Benefit (USD)'], marker='o', label='Economic Benefit (Ambulance)')
plt.xlabel("Totem Coverage (%)")
plt.ylabel("Economic Value (USD)")
plt.title("Economic Value by Improve Area, Ambulance, Heat")
plt.legend()
plt.grid(True)
plt.show()

# Plot GDP and Health savings
plt.figure(figsize=(10, 6))
plt.plot(df_gdp_impact['Totem Coverage (%)'], df_gdp_impact['GDP Increase (USD)'], marker='o', label='GDP Increase')
plt.plot(df_gdp_impact['Totem Coverage (%)'], df_gdp_impact['Health Cost Savings (USD)'], marker='o', label='Health Cost Savings')
plt.xlabel("Totem Coverage (%)")
plt.ylabel("Economic Value (USD)")
plt.title("Economic Value by Improve Area, Health and GDP")
plt.legend()
plt.grid(True)
plt.show()

# Print Simulation Parameters
print("\nSimulation Parameters")
display(HTML(pd.DataFrame(parameters, index = ['value']).to_html()))

# --- Business Applications and SBIR Pathways ---

report = """
# Business Applications and SBIR Pathways

## 1. Water Scarcity Mitigation

*   **Business Application:** Providing decentralized water solutions to drought-stricken regions, reducing reliance on centralized water systems and ensuring water availability during peak demand periods.
*   **SBIR Pathway:** Develop a scalable and cost-effective "Totem" design that can be rapidly deployed in water-scarce communities.
    * **Relevant Subtopic:** Data Informatics & Green House Gas Monitoring
    * **Relevance:** Water scarcity is often exacerbated by climate change, which also impacts greenhouse gas emissions. This system contributes to monitoring and mitigating these effects.

## 2. Urban Heat Island Reduction

*   **Business Application:** Implementing urban cooling infrastructure to reduce heat stress, improve air quality, and lower energy consumption in densely populated areas.
*   **SBIR Pathway:** Optimize the "Totem" cooling system for maximum efficiency and minimal energy consumption, exploring innovative materials and design strategies.
    * **Relevant Subtopic:** Decision Support Tools Leveraging NASA Earth Science Data
    * **Relevance:** Using NASA Earth science data to monitor urban heat island effects and optimize the placement and operation of "Totem" cooling systems.

## 3. Emergency Response Optimization

*   **Business Application:** Enhancing ambulance services through data-driven dispatch and hydrogen propulsion, improving response times and reducing emissions during critical emergencies.
*   **SBIR Pathway:** Develop an AI-powered dispatch system that predicts emergency hotspots and optimizes ambulance routes in real-time, utilizing data from various sources.
    * **Relevant Subtopic:** Nontraditional Aviation Operations for Wildfire Response (Adaptable concept)
    * **Relevance:** The core principles of efficient routing and rapid response apply to both emergency medical services and wildfire response scenarios. The SBIR could focus on adapting the routing algorithm for wildfire-specific challenges.

## 4. Sustainable Urban Development

*   **Business Application:** Integrating the "Totem" and hydrogen ambulance systems into new urban development projects, creating self-sufficient and environmentally friendly communities.
*   **SBIR Pathway:** Develop a comprehensive urban planning tool that incorporates the "Totem" and hydrogen ambulance systems, optimizing resource allocation and minimizing environmental impact.
    * **Relevant Subtopic:** All three subtopics are relevant as sustainable urban development encompasses data informatics, decision support, and innovative transportation.
    * **Relevance:** This overarching application ties together all the individual components, demonstrating the holistic impact of the integrated system on urban sustainability.

The SBIR pathways will focus on specific aspects of the integrated system, allowing for targeted research and development efforts.
"""
print("Business Applications and SBIR Pathways:")
print(report)

