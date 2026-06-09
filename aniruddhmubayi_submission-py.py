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
# Import your model or any other necessary libraries

# Load the test data
test_data = pd.read_csv('path_to_test_data.csv')

# Preprocess the data (if needed)
# e.g., test_data = preprocess(test_data)

# Load your trained model (if applicable)
# from joblib import load
# model = load('your_model.joblib')

# Make predictions
# predictions = model.predict(test_data)

# Create a DataFrame for submission
submission = pd.DataFrame({
    'Id': test_data['Id'],  # Replace 'Id' with the actual ID column name
    'Prediction': predictions  # Replace 'Prediction' with the actual prediction column name
})

# Save the submission file
submission.to_csv('submission.csv', index=False)


import random

class Agent:
    def __init__(self, name):
        self.name = name
        self.resources = 100  # Starting resources
        self.opponent_resources = 100  # Opponent's resources
        self.history = []  # To keep track of actions

    def gather_resources(self):
        # Simple resource gathering strategy
        gathered = random.randint(5, 15)  # Randomly gather between 5 and 15 resources
        self.resources += gathered
        self.history.append(('gather', gathered))
        print(f"{self.name} gathered {gathered} resources. Total: {self.resources}")

    def allocate_resources(self):
        # Simple allocation strategy
        if self.resources > 50:
            allocated = random.randint(10, 30)  # Randomly allocate between 10 and 30 resources
            self.resources -= allocated
            print(f"{self.name} allocated {allocated} resources. Remaining: {self.resources}")
            return allocated
        return 0

    def analyze_opponent(self):
        # Basic analysis of opponent's resources
        if self.opponent_resources < 50:
            print(f"{self.name} notices opponent is low on resources.")
            return True  # Opportunity to attack
        return False

    def take_action(self):
        # Decide whether to gather or allocate resources
        if self.analyze_opponent():
            allocated = self.allocate_resources()
            if allocated > 0:
                print(f"{self.name} attacks with {allocated} resources!")
        else:
            self.gather_resources()

    def update_opponent_resources(self, amount):
        self.opponent_resources -= amount
        print(f"{self.name} updated opponent's resources. Opponent's total: {self.opponent_resources}")

def simulate_game(agent1, agent2, rounds=10):
    for round in range(rounds):
        print(f"\n--- Round {round + 1} ---")
        agent1.take_action()
        agent2.take_action()

        # Simulate resource exchange
        if random.random() < 0.5:  # Random chance for agent1 to affect agent2
            agent1.update_opponent_resources(random.randint(5, 20))
        if random.random() < 0.5:  # Random chance for agent2 to affect agent1
            agent2.update_opponent_resources(random.randint(5, 20))

# Create two agents
agent1 = Agent("Explorer A")
agent2 = Agent("Explorer B")

# Simulate the game
simulate_game(agent1, agent2)

