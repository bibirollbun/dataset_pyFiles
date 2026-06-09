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


import cv2 


import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SimpleDataLoader:
    def __init__(self, base_path="/kaggle/input/arc-prize-2025"):
        self.base_path = Path(base_path)
        self.files = {
            'train_challenges': self.base_path / 'arc-agi_training_challenges.json',
            'train_solutions': self.base_path / 'arc-agi_training_solutions.json',
            'test_challenges': self.base_path / 'arc-agi_test_challenges.json'
        }
        
    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def load_data(self):
        data = {}
        for name, path in self.files.items():
            data[name] = self.load_json(path)
        return data['train_challenges'], data['train_solutions'], data['test_challenges']

class SimpleProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.grid_size = 30
        self.num_colors = 10
        
    def normalize_grid(self, grid):
        grid = np.array(grid)
        # Simple resize to target size
        if grid.size == 0:
            return np.zeros((self.grid_size, self.grid_size))
        return cv2.resize(grid, (self.grid_size, self.grid_size), interpolation=cv2.INTER_NEAREST) / (self.num_colors - 1)
    
    def extract_features(self, grid):
        # Basic features only
        return [
            np.mean(grid),
            np.std(grid),
            np.sum(grid > 0)
        ]
    
    def prepare_data(self, challenges, solutions=None):
        X, y = [], []
        for task_id in challenges:
            task = challenges[task_id]
            for example in task['train']:
                if solutions and task_id in solutions:
                    input_norm = self.normalize_grid(example['input'])
                    output_norm = self.normalize_grid(solutions[task_id][0])  # First solution only
                    X.append(self.extract_features(input_norm))
                    y.append(output_norm)
        return np.array(X), np.array(y)

class SimpleModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(30*30, activation='sigmoid')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return tf.reshape(self.output_layer(x), (-1, 30, 30))

class SimpleARCSolver:
    def __init__(self):
        self.loader = SimpleDataLoader()
        self.processor = SimpleProcessor()
        self.model = SimpleModel()
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
    
    def train(self, epochs=20):
        challenges, solutions, _ = self.loader.load_data()
        X, y = self.processor.prepare_data(challenges, solutions)
        X = self.processor.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
    
    def solve_task(self, task):
        input_norm = self.processor.normalize_grid(task['input'])
        features = self.processor.extract_features(input_norm)
        features = self.processor.scaler.transform([features])
        prediction = self.model.predict(features)[0]
        return (prediction * 9).round().astype(int).tolist()  # Convert to 0-9 values
    
    def generate_submission(self, output_file="submission.json"):
        _, _, test_challenges = self.loader.load_data()
        submission = {}
        for task_id in test_challenges:
            task = test_challenges[task_id]
            solutions = []
            for test_input in task['test']:
                solution = self.solve_task(test_input)
                solutions.append({"attempt_1": solution, "attempt_2": solution})  # Same attempt twice
            submission[task_id] = solutions
        with open(output_file, 'w') as f:
            json.dump(submission, f)

def main():
    solver = SimpleARCSolver()
    solver.train()
    solver.generate_submission()
    print("Done!")

if __name__ == "__main__":
    main()


from IPython.display import FileLink

# Replace 'submission.json' with your file path if different
FileLink(r'submission.json')


!zip submission.zip submission.json


import shutil

# Move submission.zip to the output directory
shutil.move("submission.zip", "/kaggle/working/submission.zip")




