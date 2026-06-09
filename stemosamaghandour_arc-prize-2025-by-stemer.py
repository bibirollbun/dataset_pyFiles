# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import matplotlib as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


        data = pd.read_excel('/kaggle/input/egx-jul-2025/egx_jul_2025.xlsx', parse_dates=['Highest Price Date'], index_col='Highest Price Date')


print(data.columns)


import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, sobel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, concatenate

class ARCReasoner:
    def __init__(self, lookback_window=30, forecast_window=5, num_features=4):
        self.lookback_window = lookback_window
        self.forecast_window = forecast_window
        self.num_features = num_features
        self._build_model()
    
    def _build_model(self):
        """Build the neural network model"""
        # Chart data input
        chart_input = Input(shape=(self.lookback_window, self.num_features, 3), name='chart_input')
        
        # Numerical data input
        num_input = Input(shape=(self.num_features,), name='num_input')
        
        # Process chart data
        x = Conv2D(32, (3, 3), activation='relu')(chart_input)
        x = Flatten()(x)
        
        # Combine with numerical data
        merged = concatenate([x, num_input])
        
        # Output layer
        output = Dense(1)(merged)
        
        # Create and compile model
        self.model = Model(inputs=[chart_input, num_input], outputs=output)
        self.model.compile(optimizer='adam', loss='mse')
        print(f"Model built for {self.num_features} features")
    
    def _apply_transformations(self, window):
        """Apply transformations to window data"""
        try:
            if isinstance(window, pd.DataFrame):
                window = window.values.astype(np.float32)
            
            window = np.nan_to_num(window)
            smoothed = gaussian_filter(window, sigma=1)
            sobel_v = sobel(window, axis=0)
            sobel_h = sobel(window, axis=1)
            edges = np.sqrt(sobel_v**2 + sobel_h**2)
            
            return np.stack([window, smoothed, edges], axis=-1)
        except Exception as e:
            print(f"Transformation failed: {e}")
            return None
    
    def _prepare_data(self, stock_data):
        """Prepare training data"""
        # Select numeric data
        numeric_data = stock_data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] < self.num_features:
            raise ValueError(f"Need at least {self.num_features} numeric features")
        
        # Use first num_features if too many
        if numeric_data.shape[1] > self.num_features:
            numeric_data = numeric_data.iloc[:, :self.num_features]
        
        numeric_data = numeric_data.ffill().bfill()
        
        chart_images = []
        numerical_data = []
        targets = []
        
        for i in range(self.lookback_window, len(numeric_data)-self.forecast_window):
            window = numeric_data.iloc[i-self.lookback_window:i]
            target = numeric_data.iloc[i:i+self.forecast_window].iloc[:, 0].mean()  # Using first column as target
            
            transformed = self._apply_transformations(window)
            if transformed is None:
                continue
                
            try:
                chart_images.append(transformed.reshape(
                    self.lookback_window,
                    self.num_features,
                    3
                ))
                numerical_data.append(window.mean().values)
                targets.append(target)
            except Exception as e:
                print(f"Window {i} processing failed: {e}")
                continue
        
        if not chart_images:
            raise ValueError("No valid windows created")
            
        return (np.array(chart_images), np.array(numerical_data)), np.array(targets)
    
    def train(self, stock_data, epochs=50, batch_size=32):
        """Train the model"""
        (X_chart, X_num), y = self._prepare_data(stock_data)
        
        print(f"Training on {len(X_chart)} samples")
        print(f"Shapes - X_chart: {X_chart.shape}, X_num: {X_num.shape}, y: {y.shape}")
        
        self.model.fit(
            [X_chart, X_num],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
    
    def predict(self, recent_data):
        """Make predictions"""
        (X_chart, X_num), _ = self._prepare_data(recent_data)
        return self.model.predict([X_chart, X_num])


#Debugging Steps:
#Check Your Data:


print("Numeric columns:", data.select_dtypes(include=[np.number]).columns)
print("Number of features:", len(data.select_dtypes(include=[np.number]).columns))


import pandas as pd
import numpy as np

def clean_data(df):
    """Clean data while preserving datetime columns"""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Identify numeric and datetime columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns
    
    # Process numeric columns
    if not numeric_cols.empty:
        # Replace infinities with NaN
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Clip extreme values only for numeric columns
        df[numeric_cols] = df[numeric_cols].clip(lower=-1e6, upper=1e6)
    
    # Process datetime columns (if any)
    if not datetime_cols.empty:
        # Forward fill datetime columns
        df[datetime_cols] = df[datetime_cols].ffill()
    
    return df

try:
    # 1. Load data
    data = pd.read_excel('/kaggle/input/egx-jul-2025/egx_jul_2025.xlsx')
    
    # 2. Clean data
    data = clean_data(data)
    
    # 3. Verify cleaning
    print("NaN values after cleaning:", data.isna().sum().sum())
    print("Data types:\n", data.dtypes)
    
    # 4. Continue with analysis
    num_features = len(data.select_dtypes(include=[np.number]).columns)
    print(f"Number of numeric features: {num_features}")
    
    # 5. Initialize and train model
    model = ARCReasoner(
        lookback_window=30,
        forecast_window=5,
        num_features=num_features
    )
    model.train(data)

except Exception as e:
    print(f"Error: {e}")
    raise


#Verify Model Input:


print("Model expects:", model.model.input_shape)



#Test Data Preparation:


(X_chart, X_num), y = model._prepare_data(data)
print("Prepared data shapes:", X_chart.shape, X_num.shape, y.shape)





# File paths for ARC Prize 2025 (as per Kaggle input)
TEST_CHALLENGES_PATH = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
SUBMISSION_OUTPUT_PATH = "/kaggle/working/submission.json"

# ğŸ”� Dummy solver that just copies the input as output
# You should replace this with your real reasoning engine
def identity_solver(input_grid):
    return input_grid

# ğŸ§  Solver for a single task
def solve_task(task_data):
    solutions = []
    for test in task_data.get("test", []):
        inp = test["input"]
        solutions.append({
            "attempt_1": identity_solver(inp),
            "attempt_2": identity_solver(inp)
        })
    return solutions
# ğŸ§¾ Main function to generate a submission
def generate_submission():
    # Load test challenges
    with open(TEST_CHALLENGES_PATH, 'r') as f:
        test_tasks = json.load(f)

    submission = {}
    # Each task_id maps to its test cases
    for task_id, task_data in test_tasks.items():
        submission[task_id] = solve_task(task_data)

    # Save the output to submission.json
    with open(SUBMISSION_OUTPUT_PATH, 'w') as f:
        json.dump(submission, f)
    print(f"âœ… Submission saved to {SUBMISSION_OUTPUT_PATH}")

# ğŸš€ Run
if __name__ == "__main__":
    generate_submission()

# Kaggle paths
DATA_DIR = '/kaggle/input/arc-prize-2025'  # Adjust to actual input name
OUTPUT_FILE = '/kaggle/working/submission.json'


# Dummy solver â€“ replace this with actual logic
def identity_solver(input_grid):
    return input_grid

def solve_task(task):
    solutions = []
    for test_case in task['test']:
        inp = test_case['input']
        attempt1 = identity_solver(inp)
        attempt2 = identity_solver(inp)
        solutions.append({
            "attempt_1": attempt1,
            "attempt_2": attempt2
        })
    return solutions
def load_task(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

""" 
def generate_submission(test_dir):
    submission = {}
    for filename in os.listdir(test_dir):
        if filename.endswith('.json'):
            task_id = filename[:-5]
            path = os.path.join(test_dir, filename)
            task = load_task(path)
            submission[task_id] = solve_task(task)
    return submission
"""
def generate_submission():
    with open(TEST_CHALLENGES_PATH, 'r') as f:
        test_tasks = json.load(f)

    submission = {}
    for task_id, task_data in test_tasks.items():
        submission[task_id] = solve_task(task_data)
    
    return submission
    
def save_submission(submission, output_path):
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    print(f'Saved: {output_path}')


"""  error 
# Execute all
if __name__ == '__main__':
    test_path = os.path.join(DATA_DIR)  # Expected structure: /test/*.json
    submission = generate_submission(test_path)
    save_submission(submission, OUTPUT_FILE)
    """

# Execute all
if __name__ == '__main__':
    submission = generate_submission()  # No path argument needed
    save_submission(submission, SUBMISSION_OUTPUT_PATH)



