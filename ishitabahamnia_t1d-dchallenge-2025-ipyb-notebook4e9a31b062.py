! pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow


import os
import pandas as pd

# Check the competition directory structure
competition_path = '/kaggle/input/type-1-diabetes-t-1-d-d-challenge-2025/'

try:
    # List all files and directories in the competition folder
    competition_contents = os.listdir(competition_path)
    print("Competition directory contents:")
    for item in competition_contents:
        item_path = os.path.join(competition_path, item)
        if os.path.isdir(item_path):
            print(f"ğŸ“� {item}/")
            sub_items = os.listdir(item_path)
            for sub_item in sub_items:
                print(f"   â”œâ”€â”€ {sub_item}")
        else:
            print(f"ğŸ“„ {item}")
            
    # Try to find data files with common patterns
    data_files = []
    for root, dirs, files in os.walk(competition_path):
        for file in files:
            if file.endswith('.csv') or file.endswith('.parquet') or file.endswith('.json'):
                data_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(data_files)} data files:")
    for file in data_files:
        print(f"  - {file}")
        
except FileNotFoundError:
    print("Competition directory not found. Using sample data.")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()


import os
import pandas as pd

# Check the competition directory structure
competition_path = '/kaggle/input/type-1-diabetes-t-1-d-d-challenge-2025/'

try:
    # List all files and directories in the competition folder
    competition_contents = os.listdir(competition_path)
    print("Competition directory contents:")
    for item in competition_contents:
        item_path = os.path.join(competition_path, item)
        if os.path.isdir(item_path):
            print(f"ğŸ“� {item}/")
            sub_items = os.listdir(item_path)
            for sub_item in sub_items:
                print(f"   â”œâ”€â”€ {sub_item}")
        else:
            print(f"ğŸ“„ {item}")
            
    # Try to find data files with common patterns
    data_files = []
    for root, dirs, files in os.walk(competition_path):
        for file in files:
            if file.endswith('.csv') or file.endswith('.parquet') or file.endswith('.json'):
                data_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(data_files)} data files:")
    for file in data_files:
        print(f"  - {file}")
        
except FileNotFoundError:
    print("Competition directory not found. Using sample data.")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()


# List files in current directory
print("Files in current directory:")
print(os.listdir('.'))



import os
import pandas as pd

# List files in current directory
print("Files in current directory:")
print(os.listdir('.'))




import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    # Create sample CGM data (continuous glucose monitoring)
    dates = pd.date_range(start='2024-01-01', periods=144, freq='5min')  # 12 hours of data
    cgm_data = pd.DataFrame({
        'timestamp': dates,
        'glucose_value': np.random.uniform(70, 180, 144)
    })
    cgm_data.to_csv('cgm_data.csv', index=False)
    
    # Create sample insulin data
    insulin_dates = pd.date_range(start='2024-01-01', periods=6, freq='2H')
    insulin_data = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(1, 8, 6),
        'insulin_type': ['rapid'] * 6
    })
    insulin_data.to_csv('insulin_data.csv', index=False)
    
    # Create sample meals data
    meal_dates = pd.date_range(start='2024-01-01', periods=3, freq='4H')
    meals_data = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': [45, 60, 35],
        'meal_type': ['breakfast', 'lunch', 'dinner']
    })
    meals_data.to_csv('meals.csv', index=False)
    
    # Create sample exercise data
    exercise_dates = pd.date_range(start='2024-01-01', periods=2, freq='6H')
    exercise_data = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': [30, 45],
        'intensity': ['moderate', 'vigorous'],
        'exercise_type': ['running', 'cycling']
    })
    exercise_data.to_csv('exercise.csv', index=False)
    
    print("Sample data created successfully!")

# Create sample files
create_sample_data()

# Load the sample data
file_paths = {
    'cgm': 'cgm_data.csv',
    'insulin': 'insulin_data.csv',
    'meals': 'meals.csv',
    'exercise': 'exercise.csv'
}

cgm_df = pd.read_csv(file_paths['cgm'], parse_dates=['timestamp'])
insulin_df = pd.read_csv(file_paths['insulin'], parse_dates=['timestamp'])
meals_df = pd.read_csv(file_paths['meals'], parse_dates=['timestamp'])
exercise_df = pd.read_csv(file_paths['exercise'], parse_dates=['timestamp'])

print("Data loaded successfully!")
print(f"CGM data: {len(cgm_df)} records")
print(f"Insulin data: {len(insulin_df)} records")
print(f"Meals data: {len(meals_df)} records")
print(f"Exercise data: {len(exercise_df)} records")


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample data if files don't exist
def create_sample_data():
    # Sample CGM data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    cgm_data = pd.DataFrame({
        'timestamp': dates,
        'glucose_value': np.random.uniform(70, 180, 100)
    })
    cgm_data.to_csv('cgm_data.csv', index=False)
    
    # Sample insulin data
    insulin_dates = pd.date_range(start='2024-01-01', periods=10, freq='H')
    insulin_data = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(1, 10, 10)
    })
    insulin_data.to_csv('insulin_data.csv', index=False)
    
    # Sample meals data
    meal_dates = pd.date_range(start='2024-01-01', periods=5, freq='4H')
    meals_data = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': np.random.randint(20, 60, 5)
    })
    meals_data.to_csv('meals.csv', index=False)
    
    # Sample exercise data
    exercise_dates = pd.date_range(start='2024-01-01', periods=3, freq='6H')
    exercise_data = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': np.random.randint(15, 60, 3),
        'intensity': ['moderate', 'light', 'vigorous']
    })
    exercise_data.to_csv('exercise.csv', index=False)

# Create sample files if they don't exist
for file in ['cgm_data.csv', 'insulin_data.csv', 'meals.csv', 'exercise.csv']:
    if not os.path.exists(file):
        create_sample_data()
        break

# Now load the data
cgm_df = pd.read_csv('cgm_data.csv', parse_dates=['timestamp'])
insulin_df = pd.read_csv('insulin_data.csv', parse_dates=['timestamp'])
meals_df = pd.read_csv('meals.csv', parse_dates=['timestamp'])
exercise_df = pd.read_csv('exercise.csv', parse_dates=['timestamp'])

print("Data loaded successfully!")
print(f"CGM data shape: {cgm_df.shape}")
print(f"Insulin data shape: {insulin_df.shape}")
print(f"Meals data shape: {meals_df.shape}")
print(f"Exercise data shape: {exercise_df.shape}")


import os
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample data for demonstration purposes."""
    # Generate sample CGM data
    cgm_dates = pd.date_range(start='2024-01-01', periods=144, freq='5min')
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose_value': np.random.uniform(70, 180, len(cgm_dates))
    })

    # Generate sample insulin data
    insulin_dates = pd.date_range(start='2024-01-01', periods=10, freq='h')
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_value': np.random.uniform(1, 5, len(insulin_dates))
    })

    # Generate sample meals data
    meal_dates = pd.date_range(start='2024-01-01', periods=5, freq='4h')
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': np.random.uniform(20, 60, len(meal_dates))
    })

    # Generate sample exercise data
    exercise_dates = pd.date_range(start='2024-01-01', periods=3, freq='6h')
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'intensity': np.random.uniform(0.5, 1.0, len(exercise_dates))
    })

    return cgm_df, insulin_df, meals_df, exercise_df

# ============================================================
# Load Kaggle or Sample Data
# ============================================================
kaggle_input_path = '/kaggle/input/'
if os.path.exists(kaggle_input_path):
    print("Files in Kaggle input directory:")
    competition_folders = os.listdir(kaggle_input_path)
    print(competition_folders)
    
    diabetes_folders = [f for f in competition_folders if 'diabetes' in f.lower() or 'glucose' in f.lower()]
    
    if diabetes_folders:
        competition_path = os.path.join(kaggle_input_path, diabetes_folders[0])
        print(f"\nFiles in {competition_path}:")
        competition_files = os.listdir(competition_path)
        print(competition_files)
        
        try:
            cgm_df = pd.read_csv(os.path.join(competition_path, 'cgm_data.csv'), parse_dates=['timestamp'])
            insulin_df = pd.read_csv(os.path.join(competition_path, 'insulin_data.csv'), parse_dates=['timestamp'])
            meals_df = pd.read_csv(os.path.join(competition_path, 'meals.csv'), parse_dates=['timestamp'])
            exercise_df = pd.read_csv(os.path.join(competition_path, 'exercise.csv'), parse_dates=['timestamp'])
            print("Loaded data from Kaggle competition")
        except FileNotFoundError:
            print("Files not found in competition folder. Creating sample data...")
            cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
    else:
        print("No diabetes-related competitions found. Creating sample data...")
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
else:
    print("Kaggle input directory not found. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()

# ============================================================
# Normalize column names for consistency
# ============================================================
if 'glucose_value' in cgm_df.columns:
    cgm_df = cgm_df.rename(columns={'glucose_value': 'glucose'})
if 'insulin_value' in insulin_df.columns:
    insulin_df = insulin_df.rename(columns={'insulin_value': 'insulin'})
if 'intensity' in exercise_df.columns:
    exercise_df = exercise_df.rename(columns={'intensity': 'exercise'})

print("\nâœ… Data ready for analysis pipeline:")
print("CGM:", cgm_df.head(2))
print("Insulin:", insulin_df.head(2))
print("Meals:", meals_df.head(2))
print("Exercise:", exercise_df.head(2))

# Now you can safely call:
# visualize_pipeline(cgm_df, insulin_df, meals_df, exercise_df)



import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# First, check what files are available in the Kaggle environment
print("Files in current directory:")
actual_files = os.listdir('/kaggle/input')
print(actual_files)

# If no files found in input, check working directory
if not actual_files:
    print("No files found in /kaggle/input, checking working directory:")
    actual_files = os.listdir('.')
    print(actual_files)

# Map the available files to our expected data types
file_mapping = {}
for file in actual_files:
    file_lower = file.lower()
    if 'cgm' in file_lower or 'glucose' in file_lower or 'sensor' in file_lower:
        file_mapping['cgm'] = file
    elif 'insulin' in file_lower:
        file_mapping['insulin'] = file
    elif 'meal' in file_lower or 'food' in file_lower or 'carb' in file_lower:
        file_mapping['meals'] = file
    elif 'exercise' in file_lower or 'activity' in file_lower:
        file_mapping['exercise'] = file

print(f"\nFile mapping: {file_mapping}")

# Load the data if files are found, otherwise create sample data
if file_mapping:
    try:
        if 'cgm' in file_mapping:
            cgm_df = pd.read_csv(file_mapping['cgm'], parse_dates=['timestamp'])
            print(f"Loaded CGM data: {len(cgm_df)} records")
        else:
            print("No CGM file found")
            cgm_df = pd.DataFrame()
            
        if 'insulin' in file_mapping:
            insulin_df = pd.read_csv(file_mapping['insulin'], parse_dates=['timestamp'])
            print(f"Loaded insulin data: {len(insulin_df)} records")
        else:
            print("No insulin file found")
            insulin_df = pd.DataFrame()
            
        if 'meals' in file_mapping:
            meals_df = pd.read_csv(file_mapping['meals'], parse_dates=['timestamp'])
            print(f"Loaded meals data: {len(meals_df)} records")
        else:
            print("No meals file found")
            meals_df = pd.DataFrame()
            
        if 'exercise' in file_mapping:
            exercise_df = pd.read_csv(file_mapping['exercise'], parse_dates=['timestamp'])
            print(f"Loaded exercise data: {len(exercise_df)} records")
        else:
            print("No exercise file found")
            exercise_df = pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Creating sample data instead...")
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
else:
    print("No data files found. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()

# Function to create sample data if no files are found
def create_sample_data():
    """Create sample diabetes data for testing"""
    # Sample CGM data (continuous glucose monitoring)
    dates = pd.date_range(start='2024-01-01', periods=288, freq='5min')  # 24 hours of data
    cgm_data = pd.DataFrame({
        'timestamp': dates,
        'glucose_value': np.random.normal(120, 30, 288).clip(70, 250)  # Normal distribution clipped
    })
    
    # Sample insulin data
    insulin_dates = pd.date_range(start='2024-01-01', periods=6, freq='4H')
    insulin_data = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(2, 10, 6),
        'insulin_type': ['rapid'] * 6
    })
    
    # Sample meals data
    meal_dates = pd.date_range(start='2024-01-01', periods=3, freq='6H')
    meals_data = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': [60, 75, 50],  # carbs in grams
        'meal_type': ['breakfast', 'lunch', 'dinner']
    })
    
    # Sample exercise data
    exercise_dates = pd.date_range(start='2024-01-01', periods=2, freq='8H')
    exercise_data = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': [45, 30],
        'intensity': ['moderate', 'vigorous'],
        'exercise_type': ['running', 'cycling']
    })
    
    print("Sample data created successfully!")
    return cgm_data, insulin_data, meals_data, exercise_data

# Display data overview
print("\n" + "="*50)
print("DATA OVERVIEW")
print("="*50)
print(f"CGM data shape: {cgm_df.shape if not cgm_df.empty else 'No data'}")
print(f"Insulin data shape: {insulin_df.shape if not insulin_df.empty else 'No data'}")
print(f"Meals data shape: {meals_df.shape if not meals_df.empty else 'No data'}")
print(f"Exercise data shape: {exercise_df.shape if not exercise_df.empty else 'No data'}")

# Show first few rows of each dataframe
if not cgm_df.empty:
    print("\nCGM data preview:")
    print(cgm_df.head())
if not insulin_df.empty:
    print("\nInsulin data preview:")
    print(insulin_df.head())
if not meals_df.empty:
    print("\nMeals data preview:")
    print(meals_df.head())
if not exercise_df.empty:
    print("\nExercise data preview:")
    print(exercise_df.head())


import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# First, let's just check what's available
print("Checking available files...")

# Check main input directory
try:
    input_files = os.listdir('/kaggle/input')
    print("Files in /kaggle/input:", input_files)
except:
    print("No /kaggle/input directory")

# Check working directory
working_files = os.listdir('.')
print("Files in working directory:", working_files)

# Create sample data function
def create_sample_data():
    print("Creating sample diabetes data...")
    
    # CGM data
    cgm_dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose': np.random.uniform(70, 180, 100)
    })
    
    # Insulin data
    insulin_dates = pd.date_range('2024-01-01', periods=10, freq='2H')
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'dose': np.random.uniform(1, 5, 10),
        'type': ['rapid'] * 10
    })
    
    # Meals data
    meal_dates = pd.date_range('2024-01-01', periods=5, freq='4H')
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': np.random.randint(20, 60, 5)
    })
    
    # Exercise data
    exercise_dates = pd.date_range('2024-01-01', periods=3, freq='6H')
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration': np.random.randint(15, 60, 3),
        'intensity': ['light', 'moderate', 'vigorous']
    })
    
    return cgm_df, insulin_df, meals_df, exercise_df

# Create sample data (since no files were found)
cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()

print("\nSample data created successfully!")
print(f"CGM data: {len(cgm_df)} records")
print(f"Insulin data: {len(insulin_df)} records")
print(f"Meals data: {len(meals_df)} records")
print(f"Exercise data: {len(exercise_df)} records")

# Show preview
print("\nCGM data preview:")
print(cgm_df.head())


import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to create sample data if no files are found
def create_sample_data():
    """Create sample diabetes data for testing"""
    # Sample CGM data (continuous glucose monitoring)
    dates = pd.date_range(start='2024-01-01', periods=288, freq='5min')  # 24 hours of data
    cgm_data = pd.DataFrame({
        'timestamp': dates,
        'glucose_value': np.random.normal(120, 30, 288).clip(70, 250)  # Normal distribution clipped
    })
    
    # Sample insulin data
    insulin_dates = pd.date_range(start='2024-01-01', periods=6, freq='4H')
    insulin_data = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(2, 10, 6),
        'insulin_type': ['rapid'] * 6
    })
    
    # Sample meals data
    meal_dates = pd.date_range(start='2024-01-01', periods=3, freq='6H')
    meals_data = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': [60, 75, 50],  # carbs in grams
        'meal_type': ['breakfast', 'lunch', 'dinner']
    })
    
    # Sample exercise data
    exercise_dates = pd.date_range(start='2024-01-01', periods=2, freq='8H')
    exercise_data = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': [45, 30],
        'intensity': ['moderate', 'vigorous'],
        'exercise_type': ['running', 'cycling']
    })
    
    print("Sample data created successfully!")
    return cgm_data, insulin_data, meals_data, exercise_data

# First, check what files are available in the Kaggle environment
print("Files in current directory:")
actual_files = os.listdir('/kaggle/input')
print(actual_files)

# Check the competition directory specifically
competition_dirs = ['type-1-diabetes-t-1-d-d-challenge-2025', 't1d-dchallenge-2025-ipyb-notebook4e9a31b062']
all_files = []

for dir_name in competition_dirs:
    try:
        dir_path = f'/kaggle/input/{dir_name}/'
        files_in_dir = os.listdir(dir_path)
        print(f"\nFiles in {dir_path}: {files_in_dir}")
        all_files.extend([f'{dir_path}{file}' for file in files_in_dir])
    except FileNotFoundError:
        print(f"Directory {dir_path} not found")

# If no files found in input, check working directory
if not all_files:
    print("\nNo files found in competition directories, checking working directory:")
    working_files = os.listdir('.')
    print(working_files)
    all_files = working_files

# Map the available files to our expected data types
file_mapping = {}
for file in all_files:
    file_lower = str(file).lower()
    if 'cgm' in file_lower or 'glucose' in file_lower or 'sensor' in file_lower:
        file_mapping['cgm'] = file
    elif 'insulin' in file_lower:
        file_mapping['insulin'] = file
    elif 'meal' in file_lower or 'food' in file_lower or 'carb' in file_lower:
        file_mapping['meals'] = file
    elif 'exercise' in file_lower or 'activity' in file_lower:
        file_mapping['exercise'] = file

print(f"\nFile mapping: {file_mapping}")

# Load the data if files are found, otherwise create sample data
if file_mapping:
    try:
        if 'cgm' in file_mapping:
            cgm_df = pd.read_csv(file_mapping['cgm'], parse_dates=['timestamp'])
            print(f"Loaded CGM data: {len(cgm_df)} records")
        else:
            print("No CGM file found")
            cgm_df = pd.DataFrame()
            
        if 'insulin' in file_mapping:
            insulin_df = pd.read_csv(file_mapping['insulin'], parse_dates=['timestamp'])
            print(f"Loaded insulin data: {len(insulin_df)} records")
        else:
            print("No insulin file found")
            insulin_df = pd.DataFrame()
            
        if 'meals' in file_mapping:
            meals_df = pd.read_csv(file_mapping['meals'], parse_dates=['timestamp'])
            print(f"Loaded meals data: {len(meals_df)} records")
        else:
            print("No meals file found")
            meals_df = pd.DataFrame()
            
        if 'exercise' in file_mapping:
            exercise_df = pd.read_csv(file_mapping['exercise'], parse_dates=['timestamp'])
            print(f"Loaded exercise data: {len(exercise_df)} records")
        else:
            print("No exercise file found")
            exercise_df = pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Creating sample data instead...")
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
else:
    print("No data files found. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()

# Display data overview
print("\n" + "="*50)
print("DATA OVERVIEW")
print("="*50)
print(f"CGM data shape: {cgm_df.shape if not cgm_df.empty else 'No data'}")
print(f"Insulin data shape: {insulin_df.shape if not insulin_df.empty else 'No data'}")
print(f"Meals data shape: {meals_df.shape if not meals_df.empty else 'No data'}")
print(f"Exercise data shape: {exercise_df.shape if not exercise_df.empty else 'No data'}")

# Show first few rows of each dataframe
if not cgm_df.empty:
    print("\nCGM data preview:")
    print(cgm_df.head())
if not insulin_df.empty:
    print("\nInsulin data preview:")
    print(insulin_df.head())
if not meals_df.empty:
    print("\nMeals data preview:")
    print(meals_df.head())
if not exercise_df.empty:
    print("\nExercise data preview:")
    print(exercise_df.head())


import os
import pandas as pd

# Check Kaggle input directory
kaggle_input_path = '/kaggle/input/'
if os.path.exists(kaggle_input_path):
    print("Files in Kaggle input directory:")
    competition_folders = os.listdir(kaggle_input_path)
    print(competition_folders)
    
    # Look for diabetes-related competitions
    diabetes_folders = [f for f in competition_folders if 'diabetes' in f.lower() or 'glucose' in f.lower()]
    
    if diabetes_folders:
        competition_path = os.path.join(kaggle_input_path, diabetes_folders[0])
        print(f"\nFiles in {competition_path}:")
        competition_files = os.listdir(competition_path)
        print(competition_files)
        
        # Try to load files from competition
        try:
            cgm_df = pd.read_csv(os.path.join(competition_path, 'cgm_data.csv'), parse_dates=['timestamp'])
            insulin_df = pd.read_csv(os.path.join(competition_path, 'insulin_data.csv'), parse_dates=['timestamp'])
            meals_df = pd.read_csv(os.path.join(competition_path, 'meals.csv'), parse_dates=['timestamp'])
            exercise_df = pd.read_csv(os.path.join(competition_path, 'exercise.csv'), parse_dates=['timestamp'])
            print("Loaded data from Kaggle competition")
        except FileNotFoundError:
            print("Files not found in competition folder. Creating sample data...")
            cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
    else:
        print("No diabetes-related competitions found. Creating sample data...")
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
else:
    print("Kaggle input directory not found. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()

# Your analysis code here...


import os
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample data for demonstration purposes."""
    # Generate sample CGM data
    cgm_dates = pd.date_range(start='2024-01-01', periods=144, freq='5min')
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose_value': np.random.uniform(70, 180, len(cgm_dates))
    })

    # Generate sample insulin data
    insulin_dates = pd.date_range(start='2024-01-01', periods=10, freq='h')
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_value': np.random.uniform(1, 5, len(insulin_dates))
    })

    # Generate sample meals data
    meal_dates = pd.date_range(start='2024-01-01', periods=5, freq='4h')
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': np.random.uniform(20, 60, len(meal_dates))
    })

    # Generate sample exercise data
    exercise_dates = pd.date_range(start='2024-01-01', periods=3, freq='6h')
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'intensity': np.random.uniform(0.5, 1.0, len(exercise_dates))
    })

    return cgm_df, insulin_df, meals_df, exercise_df

# ============================================================
# Load Kaggle or Sample Data
# ============================================================
kaggle_input_path = '/kaggle/input/'
if os.path.exists(kaggle_input_path):
    print("Files in Kaggle input directory:")
    competition_folders = os.listdir(kaggle_input_path)
    print(competition_folders)
    
    diabetes_folders = [f for f in competition_folders if 'diabetes' in f.lower() or 'glucose' in f.lower()]
    
    if diabetes_folders:
        competition_path = os.path.join(kaggle_input_path, diabetes_folders[0])
        print(f"\nFiles in {competition_path}:")
        competition_files = os.listdir(competition_path)
        print(competition_files)
        
        try:
            cgm_df = pd.read_csv(os.path.join(competition_path, 'cgm_data.csv'), parse_dates=['timestamp'])
            insulin_df = pd.read_csv(os.path.join(competition_path, 'insulin_data.csv'), parse_dates=['timestamp'])
            meals_df = pd.read_csv(os.path.join(competition_path, 'meals.csv'), parse_dates=['timestamp'])
            exercise_df = pd.read_csv(os.path.join(competition_path, 'exercise.csv'), parse_dates=['timestamp'])
            print("Loaded data from Kaggle competition")
        except FileNotFoundError:
            print("Files not found in competition folder. Creating sample data...")
            cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
    else:
        print("No diabetes-related competitions found. Creating sample data...")
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
else:
    print("Kaggle input directory not found. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()

# ============================================================
# Normalize column names for consistency
# ============================================================
if 'glucose_value' in cgm_df.columns:
    cgm_df = cgm_df.rename(columns={'glucose_value': 'glucose'})
if 'insulin_value' in insulin_df.columns:
    insulin_df = insulin_df.rename(columns={'insulin_value': 'insulin'})
if 'intensity' in exercise_df.columns:
    exercise_df = exercise_df.rename(columns={'intensity': 'exercise'})

print("\nâœ… Data ready for analysis pipeline:")
print("CGM:", cgm_df.head(2))
print("Insulin:", insulin_df.head(2))
print("Meals:", meals_df.head(2))
print("Exercise:", exercise_df.head(2))

# Now you can safely call:
# visualize_pipeline(cgm_df, insulin_df, meals_df, exercise_df)



import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to create sample data
def create_sample_data():
    """Create sample diabetes data for testing"""
    print("Creating sample diabetes data...")
    
    # Sample CGM data (continuous glucose monitoring)
    dates = pd.date_range(start='2024-01-01', periods=288, freq='5min')  # 24 hours of data
    cgm_data = pd.DataFrame({
        'timestamp': dates,
        'glucose_value': np.random.normal(120, 30, 288).clip(70, 250)
    })
    
    # Sample insulin data
    insulin_dates = pd.date_range(start='2024-01-01', periods=6, freq='4H')
    insulin_data = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(2, 10, 6),
        'insulin_type': ['rapid'] * 6
    })
    
    # Sample meals data
    meal_dates = pd.date_range(start='2024-01-01', periods=3, freq='6H')
    meals_data = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': [60, 75, 50],
        'meal_type': ['breakfast', 'lunch', 'dinner']
    })
    
    # Sample exercise data
    exercise_dates = pd.date_range(start='2024-01-01', periods=2, freq='8H')
    exercise_data = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': [45, 30],
        'intensity': ['moderate', 'vigorous'],
        'exercise_type': ['running', 'cycling']
    })
    
    print("Sample data created successfully!")
    return cgm_data, insulin_data, meals_data, exercise_data

# Check what files are available
print("Checking available files...")
print("Files in current directory:", os.listdir('.'))

# Try to load data with flexible approach
def load_data_files():
    """Try to load data files with flexible naming"""
    data = {}
    available_files = os.listdir('.')
    
    # Look for CGM data files
    cgm_patterns = ['cgm', 'glucose', 'sensor']
    for pattern in cgm_patterns:
        matching_files = [f for f in available_files if pattern in f.lower() and f.endswith('.csv')]
        if matching_files:
            try:
                data['cgm'] = pd.read_csv(matching_files[0], parse_dates=['timestamp'])
                print(f"Loaded CGM data from: {matching_files[0]}")
                break
            except Exception as e:
                print(f"Error loading {matching_files[0]}: {e}")
    
    # Look for insulin data files
    insulin_patterns = ['insulin', 'bolus', 'basal']
    for pattern in insulin_patterns:
        matching_files = [f for f in available_files if pattern in f.lower() and f.endswith('.csv')]
        if matching_files:
            try:
                data['insulin'] = pd.read_csv(matching_files[0], parse_dates=['timestamp'])
                print(f"Loaded insulin data from: {matching_files[0]}")
                break
            except Exception as e:
                print(f"Error loading {matching_files[0]}: {e}")
    
    # Look for meals data files
    meal_patterns = ['meal', 'food', 'carb', 'nutrition']
    for pattern in meal_patterns:
        matching_files = [f for f in available_files if pattern in f.lower() and f.endswith('.csv')]
        if matching_files:
            try:
                data['meals'] = pd.read_csv(matching_files[0], parse_dates=['timestamp'])
                print(f"Loaded meals data from: {matching_files[0]}")
                break
            except Exception as e:
                print(f"Error loading {matching_files[0]}: {e}")
    
    # Look for exercise data files
    exercise_patterns = ['exercise', 'activity', 'workout']
    for pattern in exercise_patterns:
        matching_files = [f for f in available_files if pattern in f.lower() and f.endswith('.csv')]
        if matching_files:
            try:
                data['exercise'] = pd.read_csv(matching_files[0], parse_dates=['timestamp'])
                print(f"Loaded exercise data from: {matching_files[0]}")
                break
            except Exception as e:
                print(f"Error loading {matching_files[0]}: {e}")
    
    return data

# Try to load existing files
loaded_data = load_data_files()

# Assign loaded data or create sample data
cgm_df = loaded_data.get('cgm', pd.DataFrame())
insulin_df = loaded_data.get('insulin', pd.DataFrame())
meals_df = loaded_data.get('meals', pd.DataFrame())
exercise_df = loaded_data.get('exercise', pd.DataFrame())

# If no data was loaded, create sample data
if cgm_df.empty:
    print("No data files found. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()

# Now perform your analysis
print("\n" + "="*50)
print("DATA ANALYSIS")
print("="*50)

# Initial Inspection
print("CGM Data Shape:", cgm_df.shape)
print("Insulin Data Shape:", insulin_df.shape)
print("Meals Data Shape:", meals_df.shape)
print("Exercise Data Shape:", exercise_df.shape)

print("\nCGM Data Info:")
cgm_df.info()
print("\nFirst 5 rows of CGM:")
print(cgm_df.head())

# Check for missing values
print("\nMissing Values in CGM Data:")
print(cgm_df.isnull().sum())

# Basic statistics
print("\nGlucose Value Statistics:")
print(cgm_df['glucose_value'].describe())

# Additional analysis
print("\n" + "="*50)
print("ADDITIONAL ANALYSIS")
print("="*50)

# Time range analysis
if not cgm_df.empty:
    print(f"Time range: {cgm_df['timestamp'].min()} to {cgm_df['timestamp'].max()}")
    print(f"Total duration: {cgm_df['timestamp'].max() - cgm_df['timestamp'].min()}")

# Glucose distribution
if not cgm_df.empty:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(cgm_df['glucose_value'], bins=30, alpha=0.7, color='blue')
    plt.title('Glucose Value Distribution')
    plt.xlabel('Glucose Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

# Save sample data for future use (optional)
if cgm_df.empty or insulin_df.empty or meals_df.empty or exercise_df.empty:
    print("Saving sample data for future use...")
    cgm_df.to_csv('cgm_data.csv', index=False)
    insulin_df.to_csv('insulin_data.csv', index=False)
    meals_df.to_csv('meals.csv', index=False)
    exercise_df.to_csv('exercise.csv', index=False)
    print("Sample data saved as CSV files")


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Simple function to create sample data
def create_diabetes_data():
    """Create sample diabetes data"""
    print("Creating sample diabetes data...")
    
    # CGM data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    cgm_df = pd.DataFrame({
        'timestamp': dates,
        'glucose_value': np.random.uniform(70, 180, 100)
    })
    
    # Insulin data
    insulin_dates = pd.date_range('2024-01-01', periods=10, freq='2H')
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(1, 5, 10)
    })
    
    # Meals data
    meal_dates = pd.date_range('2024-01-01', periods=5, freq='4H')
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': np.random.randint(20, 60, 5)
    })
    
    # Exercise data
    exercise_dates = pd.date_range('2024-01-01', periods=3, freq='6H')
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': np.random.randint(15, 60, 3)
    })
    
    return cgm_df, insulin_df, meals_df, exercise_df

# Try to load data, if files don't exist, create sample data
try:
    cgm_df = pd.read_csv('cgm_data.csv', parse_dates=['timestamp'])
    insulin_df = pd.read_csv('insulin_data.csv', parse_dates=['timestamp'])
    meals_df = pd.read_csv('meals.csv', parse_dates=['timestamp'])
    exercise_df = pd.read_csv('exercise.csv', parse_dates=['timestamp'])
    print("Loaded existing data files")
except FileNotFoundError:
    print("Data files not found. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_diabetes_data()

# Your analysis code
print("CGM Data Shape:", cgm_df.shape)
print("Insulin Data Shape:", insulin_df.shape)
print("\nCGM Data Info:")
cgm_df.info()
print("\nFirst 5 rows of CGM:")
print(cgm_df.head())
print("\nMissing Values in CGM Data:")
print(cgm_df.isnull().sum())
print("\nGlucose Value Statistics:")
print(cgm_df['glucose_value'].describe())


# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning & Statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                            classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)
import scipy.stats as stats

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Function to create sample diabetes data
def create_sample_diabetes_data():
    """Create realistic sample diabetes data for analysis"""
    print("Creating sample diabetes data...")
    
    # Generate timestamps for 3 days of data
    start_date = '2024-01-01 00:00:00'
    end_date = '2024-01-03 23:55:00'
    
    # CGM Data (Continuous Glucose Monitoring) - every 5 minutes
    cgm_dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Create realistic glucose patterns
    base_glucose = np.random.normal(120, 15, len(cgm_dates))
    
    # Add meal effects (spikes after typical meal times)
    meal_times = [8, 12, 18, 21]  # 8am, 12pm, 6pm, 9pm
    for hour in meal_times:
        meal_mask = (cgm_dates.hour == hour) & (cgm_dates.minute == 0)
        base_glucose[meal_mask] += np.random.normal(30, 8, meal_mask.sum())
    
    # Add night time decrease
    night_mask = (cgm_dates.hour >= 22) | (cgm_dates.hour <= 6)
    base_glucose[night_mask] -= np.random.normal(15, 5, night_mask.sum())
    
    # Add some random noise and clip to realistic range
    glucose_values = base_glucose + np.random.normal(0, 5, len(cgm_dates))
    glucose_values = np.clip(glucose_values, 70, 250)
    
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose_value': glucose_values,
        'patient_id': 'PATIENT_001'
    })
    
    # Insulin Data - 3-4 doses per day
    insulin_dates = []
    for day in range(3):
        # Breakfast dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        # Lunch dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        # Dinner dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        # Bedtime dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 22:00:00'))
    
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(2, 8, len(insulin_dates)),
        'insulin_type': ['rapid'] * len(insulin_dates),
        'patient_id': 'PATIENT_001'
    })
    
    # Meals Data - 3 meals per day + optional snack
    meal_dates = []
    meal_carbs = []
    meal_types = []
    
    for day in range(3):
        # Breakfast
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        meal_carbs.append(np.random.randint(40, 60))
        meal_types.append('breakfast')
        
        # Lunch
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        meal_carbs.append(np.random.randint(50, 80))
        meal_types.append('lunch')
        
        # Dinner
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        meal_carbs.append(np.random.randint(60, 90))
        meal_types.append('dinner')
        
        # Optional snack
        if np.random.random() > 0.5:
            meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 21:00:00'))
            meal_carbs.append(np.random.randint(15, 30))
            meal_types.append('snack')
    
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': meal_carbs,
        'meal_type': meal_types,
        'patient_id': 'PATIENT_001'
    })
    
    # Exercise Data - random exercise sessions
    exercise_dates = []
    exercise_durations = []
    exercise_intensities = []
    
    for day in range(3):
        if np.random.random() > 0.4:  # 60% chance of exercise each day
            hour = np.random.choice([7, 17, 19])  # Morning or evening
            exercise_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} {hour}:00:00'))
            exercise_durations.append(np.random.randint(20, 60))
            exercise_intensities.append(np.random.choice(['light', 'moderate', 'vigorous']))
    
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': exercise_durations,
        'intensity': exercise_intensities,
        'patient_id': 'PATIENT_001'
    })
    
    print("Sample diabetes data created successfully!")
    return cgm_df, insulin_df, meals_df, exercise_df

# Try to load data files with error handling
try:
    cgm_df = pd.read_csv('/content/cgm_data.csv', parse_dates=['timestamp'])
    print("Loaded CGM data")
except FileNotFoundError:
    print("CGM data file not found")
    cgm_df = pd.DataFrame()

try:
    insulin_df = pd.read_csv('/content/insulin_data.csv', parse_dates=['timestamp'])
    print("Loaded insulin data")
except FileNotFoundError:
    print("Insulin data file not found")
    insulin_df = pd.DataFrame()

try:
    meals_df = pd.read_csv('/content/meals.csv', parse_dates=['timestamp'])
    print("Loaded meals data")
except FileNotFoundError:
    print("Meals data file not found")
    meals_df = pd.DataFrame()

try:
    exercise_df = pd.read_csv('/content/exercise.csv', parse_dates=['timestamp'])
    print("Loaded exercise data")
except FileNotFoundError:
    print("Exercise data file not found")
    exercise_df = pd.DataFrame()

# If any dataframe is empty, create sample data
if cgm_df.empty or insulin_df.empty or meals_df.empty or exercise_df.empty:
    print("\nSome data files are missing. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_diabetes_data()
    
    # Save sample data for future use
    cgm_df.to_csv('cgm_data.csv', index=False)
    insulin_df.to_csv('insulin_data.csv', index=False)
    meals_df.to_csv('meals.csv', index=False)
    exercise_df.to_csv('exercise.csv', index=False)
    print("Sample data saved as CSV files")

# Data Overview
print("\n" + "="*60)
print("DATA OVERVIEW")
print("="*60)
print(f"CGM Data: {len(cgm_df)} records")
print(f"Insulin Data: {len(insulin_df)} records")
print(f"Meals Data: {len(meals_df)} records")
print(f"Exercise Data: {len(exercise_df)} records")

print("\nCGM Data Info:")
cgm_df.info()
print("\nFirst 5 rows of CGM Data:")
print(cgm_df.head())

# Basic Statistics
print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print("Glucose Value Statistics:")
print(cgm_df['glucose_value'].describe())

if not insulin_df.empty:
    print("\nInsulin Dose Statistics:")
    print(insulin_df['insulin_dose'].describe())

if not meals_df.empty:
    print("\nMeal Carbs Statistics:")
    print(meals_df['carbs'].describe())

if not exercise_df.empty:
    print("\nExercise Duration Statistics:")
    print(exercise_df['duration_minutes'].describe())

# Missing Values Analysis
print("\n" + "="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)
print("Missing values in CGM Data:")
print(cgm_df.isnull().sum())

# Visualization
print("\n" + "="*60)
print("DATA VISUALIZATION")
print("="*60)

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Glucose distribution
axes[0, 0].hist(cgm_df['glucose_value'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Glucose Value Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Glucose Value (mg/dL)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(alpha=0.3)

# Time series of glucose (first 24 hours)
first_day = cgm_df[cgm_df['timestamp'] <= cgm_df['timestamp'].min() + pd.Timedelta(hours=24)]
axes[0, 1].plot(first_day['timestamp'], first_day['glucose_value'], color='red', linewidth=1.5)
axes[0, 1].set_title('Glucose Levels (First 24 Hours)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Glucose (mg/dL)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(alpha=0.3)

# Insulin dose distribution
if not insulin_df.empty:
    axes[1, 0].hist(insulin_df['insulin_dose'], bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Insulin Dose Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Insulin Dose (units)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'No insulin data', ha='center', va='center', transform=axes[1, 0].transAxes)

# Meal carbs distribution
if not meals_df.empty:
    axes[1, 1].hist(meals_df['carbs'], bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('Meal Carbohydrates Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Carbohydrates (g)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, 'No meal data', ha='center', va='center', transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.show()

# Additional analysis ready for machine learning
print("\nData preparation complete. Ready for machine learning modeling!")
print("\nAvailable for analysis:")
print("- Glucose time series data")
print("- Insulin administration records")
print("- Meal consumption data")
print("- Exercise activity data")


# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning & Statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                            classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)
import scipy.stats as stats

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Function to create sample diabetes data
def create_sample_diabetes_data():
    """Create realistic sample diabetes data for analysis"""
    print("Creating sample diabetes data...")
    
    # Generate timestamps for 3 days of data
    start_date = '2024-01-01 00:00:00'
    end_date = '2024-01-03 23:55:00'
    
    # CGM Data (Continuous Glucose Monitoring) - every 5 minutes
    cgm_dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Create realistic glucose patterns
    base_glucose = np.random.normal(120, 15, len(cgm_dates))
    
    # Add meal effects (spikes after typical meal times)
    meal_times = [8, 12, 18, 21]  # 8am, 12pm, 6pm, 9pm
    for hour in meal_times:
        meal_mask = (cgm_dates.hour == hour) & (cgm_dates.minute == 0)
        base_glucose[meal_mask] += np.random.normal(30, 8, meal_mask.sum())
    
    # Add night time decrease
    night_mask = (cgm_dates.hour >= 22) | (cgm_dates.hour <= 6)
    base_glucose[night_mask] -= np.random.normal(15, 5, night_mask.sum())
    
    # Add some random noise and clip to realistic range
    glucose_values = base_glucose + np.random.normal(0, 5, len(cgm_dates))
    glucose_values = np.clip(glucose_values, 70, 250)
    
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose_value': glucose_values,
        'patient_id': 'PATIENT_001'
    })
    
    # Insulin Data - 3-4 doses per day
    insulin_dates = []
    for day in range(3):
        # Breakfast dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        # Lunch dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        # Dinner dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        # Bedtime dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 22:00:00'))
    
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(2, 8, len(insulin_dates)),
        'insulin_type': ['rapid'] * len(insulin_dates),
        'patient_id': 'PATIENT_001'
    })
    
    # Meals Data - 3 meals per day + optional snack
    meal_dates = []
    meal_carbs = []
    meal_types = []
    
    for day in range(3):
        # Breakfast
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        meal_carbs.append(np.random.randint(40, 60))
        meal_types.append('breakfast')
        
        # Lunch
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        meal_carbs.append(np.random.randint(50, 80))
        meal_types.append('lunch')
        
        # Dinner
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        meal_carbs.append(np.random.randint(60, 90))
        meal_types.append('dinner')
        
        # Optional snack
        if np.random.random() > 0.5:
            meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 21:00:00'))
            meal_carbs.append(np.random.randint(15, 30))
            meal_types.append('snack')
    
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': meal_carbs,
        'meal_type': meal_types,
        'patient_id': 'PATIENT_001'
    })
    
    # Exercise Data - random exercise sessions
    exercise_dates = []
    exercise_durations = []
    exercise_intensities = []
    
    for day in range(3):
        if np.random.random() > 0.4:  # 60% chance of exercise each day
            hour = np.random.choice([7, 17, 19])  # Morning or evening
            exercise_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} {hour}:00:00'))
            exercise_durations.append(np.random.randint(20, 60))
            exercise_intensities.append(np.random.choice(['light', 'moderate', 'vigorous']))
    
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': exercise_durations,
        'intensity': exercise_intensities,
        'patient_id': 'PATIENT_001'
    })
    
    print("Sample diabetes data created successfully!")
    return cgm_df, insulin_df, meals_df, exercise_df

# Try to load data files with error handling
try:
    cgm_df = pd.read_csv('/content/cgm_data.csv', parse_dates=['timestamp'])
    print("Loaded CGM data")
except FileNotFoundError:
    print("CGM data file not found")
    cgm_df = pd.DataFrame()

try:
    insulin_df = pd.read_csv('/content/insulin_data.csv', parse_dates=['timestamp'])
    print("Loaded insulin data")
except FileNotFoundError:
    print("Insulin data file not found")
    insulin_df = pd.DataFrame()

try:
    meals_df = pd.read_csv('/content/meals.csv', parse_dates=['timestamp'])
    print("Loaded meals data")
except FileNotFoundError:
    print("Meals data file not found")
    meals_df = pd.DataFrame()

try:
    exercise_df = pd.read_csv('/content/exercise.csv', parse_dates=['timestamp'])
    print("Loaded exercise data")
except FileNotFoundError:
    print("Exercise data file not found")
    exercise_df = pd.DataFrame()

# If any dataframe is empty, create sample data
if cgm_df.empty or insulin_df.empty or meals_df.empty or exercise_df.empty:
    print("\nSome data files are missing. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_diabetes_data()
    
    # Save sample data for future use
    cgm_df.to_csv('cgm_data.csv', index=False)
    insulin_df.to_csv('insulin_data.csv', index=False)
    meals_df.to_csv('meals.csv', index=False)
    exercise_df.to_csv('exercise.csv', index=False)
    print("Sample data saved as CSV files")

# Data Overview
print("\n" + "="*60)
print("DATA OVERVIEW")
print("="*60)
print(f"CGM Data: {len(cgm_df)} records")
print(f"Insulin Data: {len(insulin_df)} records")
print(f"Meals Data: {len(meals_df)} records")
print(f"Exercise Data: {len(exercise_df)} records")

print("\nCGM Data Info:")
cgm_df.info()
print("\nFirst 5 rows of CGM Data:")
print(cgm_df.head())

# Basic Statistics
print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print("Glucose Value Statistics:")
print(cgm_df['glucose_value'].describe())

if not insulin_df.empty:
    print("\nInsulin Dose Statistics:")
    print(insulin_df['insulin_dose'].describe())

if not meals_df.empty:
    print("\nMeal Carbs Statistics:")
    print(meals_df['carbs'].describe())

if not exercise_df.empty:
    print("\nExercise Duration Statistics:")
    print(exercise_df['duration_minutes'].describe())

# Missing Values Analysis
print("\n" + "="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)
print("Missing values in CGM Data:")
print(cgm_df.isnull().sum())

# Visualization
print("\n" + "="*60)
print("DATA VISUALIZATION")
print("="*60)

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Glucose distribution
axes[0, 0].hist(cgm_df['glucose_value'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Glucose Value Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Glucose Value (mg/dL)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(alpha=0.3)

# Time series of glucose (first 24 hours)
first_day = cgm_df[cgm_df['timestamp'] <= cgm_df['timestamp'].min() + pd.Timedelta(hours=24)]
axes[0, 1].plot(first_day['timestamp'], first_day['glucose_value'], color='red', linewidth=1.5)
axes[0, 1].set_title('Glucose Levels (First 24 Hours)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Glucose (mg/dL)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(alpha=0.3)

# Insulin dose distribution
if not insulin_df.empty:
    axes[1, 0].hist(insulin_df['insulin_dose'], bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Insulin Dose Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Insulin Dose (units)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'No insulin data', ha='center', va='center', transform=axes[1, 0].transAxes)

# Meal carbs distribution
if not meals_df.empty:
    axes[1, 1].hist(meals_df['carbs'], bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('Meal Carbohydrates Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Carbohydrates (g)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, 'No meal data', ha='center', va='center', transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.show()

# Additional analysis ready for machine learning
print("\nData preparation complete. Ready for machine learning modeling!")
print("\nAvailable for analysis:")
print("- Glucose time series data")
print("- Insulin administration records")
print("- Meal consumption data")
print("- Exercise activity data")


import os
import pandas as pd

# Function to generate sample data if competition files are missing
def create_sample_data():
    # Example CGM data
    cgm_dates = pd.date_range(start='2024-01-01', periods=24, freq='h')
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose': [100 + i % 20 for i in range(24)]
    })

    # Example insulin data
    insulin_dates = pd.date_range(start='2024-01-01', periods=10, freq='h')
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_units': [1 + (i % 3) for i in range(10)]
    })

    # Example meals data
    meal_dates = pd.date_range(start='2024-01-01', periods=5, freq='4h')
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': [30, 50, 45, 60, 40]
    })

    # Example exercise data
    exercise_dates = pd.date_range(start='2024-01-01', periods=3, freq='6h')
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'exercise_minutes': [30, 45, 60]
    })

    return cgm_df, insulin_df, meals_df, exercise_df


# Load Kaggle or fallback to sample data
kaggle_input_path = '/kaggle/input/'
if os.path.exists(kaggle_input_path):
    print("Files in Kaggle input directory:")
    competition_folders = os.listdir(kaggle_input_path)
    print(competition_folders)
    
    diabetes_folders = [f for f in competition_folders if 'diabetes' in f.lower() or 'glucose' in f.lower()]
    
    if diabetes_folders:
        competition_path = os.path.join(kaggle_input_path, diabetes_folders[0])
        print(f"\nFiles in {competition_path}:")
        competition_files = os.listdir(competition_path)
        print(competition_files)
        
        try:
            cgm_df = pd.read_csv(os.path.join(competition_path, 'cgm_data.csv'), parse_dates=['timestamp'])
            insulin_df = pd.read_csv(os.path.join(competition_path, 'insulin_data.csv'), parse_dates=['timestamp'])
            meals_df = pd.read_csv(os.path.join(competition_path, 'meals.csv'), parse_dates=['timestamp'])
            exercise_df = pd.read_csv(os.path.join(competition_path, 'exercise.csv'), parse_dates=['timestamp'])
            print("Loaded data from Kaggle competition")
        except FileNotFoundError:
            print("Files not found in competition folder. Creating sample data...")
            cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
    else:
        print("No diabetes-related competitions found. Creating sample data...")
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
else:
    print("Kaggle input directory not found. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()


# ==============================
# âœ… Safe-check statistics block
# ==============================
print("\n--- Dataset Overview ---")

# CGM safe checks
if not cgm_df.empty:
    print(f"CGM records: {len(cgm_df)}")
    if 'glucose' in cgm_df.columns:
        print(f"  Glucose range: {cgm_df['glucose'].min()} - {cgm_df['glucose'].max()}")
    if 'timestamp' in cgm_df.columns:
        print(f"  CGM start: {cgm_df['timestamp'].min()}, end: {cgm_df['timestamp'].max()}")

# Insulin safe checks
if not insulin_df.empty:
    print(f"Insulin records: {len(insulin_df)}")
    if 'insulin_units' in insulin_df.columns:
        print(f"  Avg insulin dose: {insulin_df['insulin_units'].mean():.2f}")

# Meals safe checks
if not meals_df.empty:
    print(f"Meal records: {len(meals_df)}")
    if 'carbs' in meals_df.columns:
        print(f"  Avg carbs per meal: {meals_df['carbs'].mean():.1f} g")

# Exercise safe checks
if not exercise_df.empty:
    print(f"Exercise records: {len(exercise_df)}")
    if 'exercise_minutes' in exercise_df.columns:
        print(f"  Avg exercise: {exercise_df['exercise_minutes'].mean():.1f} mins")


# âœ… Preview first rows safely
print("\n--- Data Previews ---")
print("CGM:\n", cgm_df.head())
print("\nInsulin:\n", insulin_df.head())
print("\nMeals:\n", meals_df.head())
print("\nExercise:\n", exercise_df.head())



import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to generate sample data if competition files are missing
def create_sample_data():
    # Example CGM data
    cgm_dates = pd.date_range(start='2024-01-01', periods=24, freq='h')
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose': [100 + i % 20 for i in range(24)]
    })

    # Example insulin data
    insulin_dates = pd.date_range(start='2024-01-01', periods=10, freq='h')
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_units': [1 + (i % 3) for i in range(10)]
    })

    # Example meals data
    meal_dates = pd.date_range(start='2024-01-01', periods=5, freq='4h')
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': [30, 50, 45, 60, 40]
    })

    # Example exercise data
    exercise_dates = pd.date_range(start='2024-01-01', periods=3, freq='6h')
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'exercise_minutes': [30, 45, 60]
    })

    return cgm_df, insulin_df, meals_df, exercise_df


# Load Kaggle or fallback to sample data
kaggle_input_path = '/kaggle/input/'
if os.path.exists(kaggle_input_path):
    print("Files in Kaggle input directory:")
    competition_folders = os.listdir(kaggle_input_path)
    print(competition_folders)
    
    diabetes_folders = [f for f in competition_folders if 'diabetes' in f.lower() or 'glucose' in f.lower()]
    
    if diabetes_folders:
        competition_path = os.path.join(kaggle_input_path, diabetes_folders[0])
        print(f"\nFiles in {competition_path}:")
        competition_files = os.listdir(competition_path)
        print(competition_files)
        
        try:
            cgm_df = pd.read_csv(os.path.join(competition_path, 'cgm_data.csv'), parse_dates=['timestamp'])
            insulin_df = pd.read_csv(os.path.join(competition_path, 'insulin_data.csv'), parse_dates=['timestamp'])
            meals_df = pd.read_csv(os.path.join(competition_path, 'meals.csv'), parse_dates=['timestamp'])
            exercise_df = pd.read_csv(os.path.join(competition_path, 'exercise.csv'), parse_dates=['timestamp'])
            print("Loaded data from Kaggle competition")
        except FileNotFoundError:
            print("Files not found in competition folder. Creating sample data...")
            cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
    else:
        print("No diabetes-related competitions found. Creating sample data...")
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
else:
    print("Kaggle input directory not found. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()


# ==============================
# âœ… Safe-check statistics block
# ==============================
print("\n--- Dataset Overview ---")

if not cgm_df.empty:
    print(f"CGM records: {len(cgm_df)}")
    if 'glucose' in cgm_df.columns:
        print(f"  Glucose range: {cgm_df['glucose'].min()} - {cgm_df['glucose'].max()}")
    if 'timestamp' in cgm_df.columns:
        print(f"  CGM start: {cgm_df['timestamp'].min()}, end: {cgm_df['timestamp'].max()}")

if not insulin_df.empty and 'insulin_units' in insulin_df.columns:
    print(f"Insulin records: {len(insulin_df)}, avg dose: {insulin_df['insulin_units'].mean():.2f}")

if not meals_df.empty and 'carbs' in meals_df.columns:
    print(f"Meal records: {len(meals_df)}, avg carbs: {meals_df['carbs'].mean():.1f} g")

if not exercise_df.empty and 'exercise_minutes' in exercise_df.columns:
    print(f"Exercise records: {len(exercise_df)}, avg exercise: {exercise_df['exercise_minutes'].mean():.1f} mins")


# ==============================
# âœ… Visualization block
# ==============================
print("\n--- Generating Plots ---")

# CGM line chart
if not cgm_df.empty and {'timestamp', 'glucose'}.issubset(cgm_df.columns):
    plt.figure(figsize=(10, 4))
    plt.plot(cgm_df['timestamp'], cgm_df['glucose'], marker='o')
    plt.title("CGM Glucose Levels Over Time")
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.grid(True)
    plt.show()

# Scatter: insulin vs meals (if timestamps overlap)
if not insulin_df.empty and not meals_df.empty:
    if {'timestamp', 'insulin_units'}.issubset(insulin_df.columns) and {'timestamp', 'carbs'}.issubset(meals_df.columns):
        merged = pd.merge_asof(insulin_df.sort_values('timestamp'),
                               meals_df.sort_values('timestamp'),
                               on='timestamp', direction='nearest', tolerance=pd.Timedelta("1h"))
        if not merged.empty:
            plt.figure(figsize=(6, 6))
            plt.scatter(merged['carbs'], merged['insulin_units'], color='purple')
            plt.title("Meals (Carbs) vs Insulin Units")
            plt.xlabel("Carbs (g)")
            plt.ylabel("Insulin Units")
            plt.grid(True)
            plt.show()

# Exercise vs glucose (nearest join)
if not exercise_df.empty and not cgm_df.empty:
    if {'timestamp', 'exercise_minutes'}.issubset(exercise_df.columns) and {'timestamp', 'glucose'}.issubset(cgm_df.columns):
        merged = pd.merge_asof(exercise_df.sort_values('timestamp'),
                               cgm_df.sort_values('timestamp'),
                               on='timestamp', direction='nearest', tolerance=pd.Timedelta("1h"))
        if not merged.empty:
            plt.figure(figsize=(6, 6))
            plt.scatter(merged['exercise_minutes'], merged['glucose'], color='green')
            plt.title("Exercise Minutes vs Glucose Levels")
            plt.xlabel("Exercise (mins)")
            plt.ylabel("Glucose (mg/dL)")
            plt.grid(True)
            plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Unified Visualization Pipeline
# ============================================================
def visualize_pipeline(cgm_df=None, insulin_df=None, meals_df=None, exercise_df=None):
    """
    Generates end-to-end visualizations:
    - CGM glucose line chart
    - Scatter plot (carbs vs insulin)
    - Correlation heatmap across glucose, insulin, carbs, exercise
    """
    # --- 1. Line Chart: CGM ---
    if cgm_df is not None and "glucose" in cgm_df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(cgm_df["timestamp"], cgm_df["glucose"], marker="o", linestyle="-", color="blue")
        plt.title("CGM Glucose Trend")
        plt.xlabel("Time")
        plt.ylabel("Glucose Level (mg/dL)")
        plt.tight_layout()
        plt.show()
    else:
        print("âš ï¸� Skipping CGM line chart (no glucose data).")

    # --- 2. Scatter Plot: Meals vs Insulin ---
    if meals_df is not None and insulin_df is not None:
        if "carbs" in meals_df.columns and "insulin" in insulin_df.columns:
            merged = pd.merge_asof(
                meals_df.sort_values("timestamp"),
                insulin_df.sort_values("timestamp"),
                on="timestamp",
                direction="nearest"
            )
            plt.figure(figsize=(6, 5))
            plt.scatter(merged["carbs"], merged["insulin"], alpha=0.7, c="green", edgecolors="black")
            plt.title("Meals (Carbs) vs Insulin Dosage")
            plt.xlabel("Carbs (g)")
            plt.ylabel("Insulin (units)")
            plt.tight_layout()
            plt.show()
        else:
            print("âš ï¸� Skipping scatter plot (carbs or insulin column missing).")
    else:
        print("âš ï¸� Skipping scatter plot (meals or insulin data missing).")

    # --- 3. Correlation Heatmap ---
    dfs = []
    if cgm_df is not None and "glucose" in cgm_df.columns:
        dfs.append(cgm_df[["timestamp", "glucose"]])
    if meals_df is not None and "carbs" in meals_df.columns:
        dfs.append(meals_df[["timestamp", "carbs"]])
    if insulin_df is not None and "insulin" in insulin_df.columns:
        dfs.append(insulin_df[["timestamp", "insulin"]])
    if exercise_df is not None and "exercise" in exercise_df.columns:
        dfs.append(exercise_df[["timestamp", "exercise"]])

    if dfs:
        merged_all = dfs[0]
        for d in dfs[1:]:
            merged_all = pd.merge_asof(
                merged_all.sort_values("timestamp"),
                d.sort_values("timestamp"),
                on="timestamp",
                direction="nearest"
            )
        numeric_df = merged_all.select_dtypes(include="number")
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
            plt.title("Correlation Heatmap (Glucose, Carbs, Insulin, Exercise)")
            plt.tight_layout()
            plt.show()
        else:
            print("âš ï¸� Not enough numeric variables for correlation heatmap.")
    else:
        print("âš ï¸� No numeric data available for correlation heatmap.")


# ============================================================
# Example usage after loading data
# ============================================================
# visualize_pipeline(cgm_df, insulin_df, meals_df, exercise_df)



import os
import pandas as pd

# Function to generate sample data if competition files are missing
def create_sample_data():
    # Example CGM data
    cgm_dates = pd.date_range(start='2024-01-01', periods=24, freq='h')
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose': [100 + i % 20 for i in range(24)]
    })

    # Example insulin data
    insulin_dates = pd.date_range(start='2024-01-01', periods=10, freq='h')
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_units': [1 + (i % 3) for i in range(10)]
    })

    # Example meals data
    meal_dates = pd.date_range(start='2024-01-01', periods=5, freq='4h')
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': [30, 50, 45, 60, 40]
    })

    # Example exercise data
    exercise_dates = pd.date_range(start='2024-01-01', periods=3, freq='6h')
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'exercise_minutes': [30, 45, 60]
    })

    return cgm_df, insulin_df, meals_df, exercise_df


# Check Kaggle input directory
kaggle_input_path = '/kaggle/input/'
if os.path.exists(kaggle_input_path):
    print("Files in Kaggle input directory:")
    competition_folders = os.listdir(kaggle_input_path)
    print(competition_folders)
    
    # Look for diabetes-related competitions
    diabetes_folders = [f for f in competition_folders if 'diabetes' in f.lower() or 'glucose' in f.lower()]
    
    if diabetes_folders:
        competition_path = os.path.join(kaggle_input_path, diabetes_folders[0])
        print(f"\nFiles in {competition_path}:")
        competition_files = os.listdir(competition_path)
        print(competition_files)
        
        # Try to load files from competition
        try:
            cgm_df = pd.read_csv(os.path.join(competition_path, 'cgm_data.csv'), parse_dates=['timestamp'])
            insulin_df = pd.read_csv(os.path.join(competition_path, 'insulin_data.csv'), parse_dates=['timestamp'])
            meals_df = pd.read_csv(os.path.join(competition_path, 'meals.csv'), parse_dates=['timestamp'])
            exercise_df = pd.read_csv(os.path.join(competition_path, 'exercise.csv'), parse_dates=['timestamp'])
            print("Loaded data from Kaggle competition")
        except FileNotFoundError:
            print("Files not found in competition folder. Creating sample data...")
            cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
    else:
        print("No diabetes-related competitions found. Creating sample data...")
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
else:
    print("Kaggle input directory not found. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()


# âœ… Quick sanity check
print("\nPreview of generated CGM data:")
print(cgm_df.head())
print("\nPreview of insulin data:")
print(insulin_df.head())



import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# VISUALIZATION PIPELINE (CGM, Meals vs Insulin, Correlation)
# ============================================================
def visualize_pipeline(df):
    """
    Generate end-to-end visualizations for CGM, meals vs insulin,
    and correlation heatmap with safe-checks.
    """

    if df is None or df.empty:
        print("âš ï¸� Dataframe is empty. Skipping all visualizations.")
        return

    # --- 1. Line Chart: CGM over time ---
    if "glucose" in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df.index, df["glucose"], marker="o", linestyle="-", color="blue", label="Glucose")
        plt.title("CGM Glucose Trend")
        plt.xlabel("Index / Time")
        plt.ylabel("Glucose Level (mg/dL)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("âš ï¸� 'glucose' column not found. Skipping CGM line chart.")

    # --- 2. Scatter Plot: Meals (carbs) vs Insulin ---
    if "carbs" in df.columns and "insulin" in df.columns:
        plt.figure(figsize=(6, 5))
        plt.scatter(df["carbs"], df["insulin"], alpha=0.7, c="green", edgecolors="black")
        plt.title("Meals (Carbs) vs Insulin Dosage")
        plt.xlabel("Carbs (g)")
        plt.ylabel("Insulin (units)")
        plt.tight_layout()
        plt.show()
    else:
        print("âš ï¸� 'carbs' or 'insulin' column not found. Skipping scatter plot.")

    # --- 3. Correlation Heatmap ---
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            square=True,
            linewidths=0.5
        )
        plt.title("Correlation Heatmap (Glucose, Carbs, Insulin, Exercise, etc.)")
        plt.tight_layout()
        plt.show()
    else:
        print("âš ï¸� Not enough numeric variables to compute correlations. Skipping heatmap.")


# ============================================================
# Example Usage
# ============================================================
# df = pd.DataFrame({
#     "glucose": [90, 110, 130, 150],
#     "carbs": [30, 45, 60, 80],
#     "insulin": [5, 10, 12, 15],
#     "exercise": [0, 1, 0, 1]
# })
# visualize_pipeline(df)



# ===============================
# FULL DIABETES ANALYTICS PIPELINE + VISUALIZATION
# ===============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    classification_report, roc_auc_score
)

# ----------------------------
# Sample Data Generator
# ----------------------------
def create_sample_data():
    """Create sample data for demonstration purposes."""
    cgm_dates = pd.date_range(start='2024-01-01', periods=144, freq='5min')
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose_value': np.random.uniform(60, 220, len(cgm_dates))  # wider range for anomalies
    })

    insulin_dates = pd.date_range(start='2024-01-01', periods=10, freq='h')
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_value': np.random.uniform(1, 5, len(insulin_dates))
    })

    meal_dates = pd.date_range(start='2024-01-01', periods=5, freq='4h')
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': np.random.uniform(20, 60, len(meal_dates))
    })

    exercise_dates = pd.date_range(start='2024-01-01', periods=3, freq='6h')
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'intensity': np.random.uniform(0.5, 1.0, len(exercise_dates))
    })

    return cgm_df, insulin_df, meals_df, exercise_df

# ----------------------------
# Visualization
# ----------------------------
def plot_glucose_time_series(featured_df, insulin_df, meals_df):
    plt.figure(figsize=(14, 6))

    # Plot glucose values
    plt.plot(featured_df["timestamp"], featured_df["glucose_value"], label="Glucose", color="black", alpha=0.6)

    # Rolling averages
    for window in [6, 12]:  # 30min, 60min if 5-min intervals
        plt.plot(featured_df["timestamp"], 
                 featured_df["glucose_value"].rolling(window=window).mean(),
                 label=f"{window*5} min rolling avg")

    # Overlay insulin (scaled for visibility)
    if "insulin_value" in featured_df.columns:
        plt.plot(featured_df["timestamp"], 
                 featured_df["insulin_value"]*30,  # scale insulin for plotting
                 label="Insulin (scaled)", color="purple", linestyle="--")

    # Meal events
    for _, meal in meals_df.iterrows():
        plt.axvline(meal["timestamp"], color="orange", linestyle=":", alpha=0.8)
    if not meals_df.empty:
        plt.text(meals_df["timestamp"].iloc[0], featured_df["glucose_value"].max(),
                 "Meal events", color="orange", fontsize=9, verticalalignment="bottom")

    # Anomalies
    hyper = featured_df[featured_df["glucose_value"] > 180]
    hypo = featured_df[featured_df["glucose_value"] < 70]
    plt.scatter(hyper["timestamp"], hyper["glucose_value"], color="red", label="Hyperglycemia", zorder=5)
    plt.scatter(hypo["timestamp"], hypo["glucose_value"], color="blue", label="Hypoglycemia", zorder=5)

    # Formatting
    plt.axhline(70, color="blue", linestyle="--", alpha=0.5)
    plt.axhline(180, color="red", linestyle="--", alpha=0.5)
    plt.title("ğŸ“ˆ Glucose Trends with Insulin & Meal Events")
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main Pipeline
# ----------------------------
def run_diabetes_pipeline(cgm_df=None, insulin_df=None, meals_df=None, exercise_df=None, save_path="diabetes_featured_data.csv"):
    print("="*60)
    print(" RUNNING DIABETES ANALYTICS PIPELINE ")
    print("="*60)

    if cgm_df is None or insulin_df is None or meals_df is None or exercise_df is None:
        print("âš ï¸� Missing input data â€” generating sample data...")
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()

    # Align timestamps
    for df in [cgm_df, insulin_df, meals_df, exercise_df]:
        if "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True)

    # Merge datasets
    featured_df = cgm_df.copy()
    for df, col in [(insulin_df, "insulin_value"), (meals_df, "carbs"), (exercise_df, "intensity")]:
        if col in df.columns:
            featured_df = pd.merge_asof(featured_df, df, on="timestamp", direction="backward")
    featured_df.fillna(0, inplace=True)

    # Future labels
    featured_df["glucose_future"] = featured_df["glucose_value"].shift(-12)
    featured_df["hypo_risk_60min"] = (featured_df["glucose_future"] < 70).astype(int)
    featured_df["hyper_risk_60min"] = (featured_df["glucose_future"] > 180).astype(int)
    featured_df.dropna(inplace=True)

    # ----------------------------
    # Visualization
    # ----------------------------
    plot_glucose_time_series(featured_df, insulin_df, meals_df)

    # Save
    featured_df.to_csv(save_path, index=False)
    print(f"\nâœ… Processed dataset saved to {save_path}")

    return featured_df

# ----------------------------
# Run pipeline
# ----------------------------
if __name__ == "__main__":
    featured_df = run_diabetes_pipeline()



# ===============================
# FULL DIABETES ANALYTICS PIPELINE + VISUALIZATION
# ===============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    classification_report, roc_auc_score
)

# ----------------------------
# Sample Data Generator
# ----------------------------
def create_sample_data():
    """Create sample data for demonstration purposes."""
    cgm_dates = pd.date_range(start='2024-01-01', periods=144, freq='5min')
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose_value': np.random.uniform(60, 220, len(cgm_dates))  # wider range for anomalies
    })

    insulin_dates = pd.date_range(start='2024-01-01', periods=10, freq='h')
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_value': np.random.uniform(1, 5, len(insulin_dates))
    })

    meal_dates = pd.date_range(start='2024-01-01', periods=5, freq='4h')
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': np.random.uniform(20, 60, len(meal_dates))
    })

    exercise_dates = pd.date_range(start='2024-01-01', periods=3, freq='6h')
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'intensity': np.random.uniform(0.5, 1.0, len(exercise_dates))
    })

    return cgm_df, insulin_df, meals_df, exercise_df

# ----------------------------
# Visualization
# ----------------------------
def plot_glucose_time_series(featured_df, insulin_df, meals_df):
    plt.figure(figsize=(14, 6))

    # Plot glucose values
    plt.plot(featured_df["timestamp"], featured_df["glucose_value"], label="Glucose", color="black", alpha=0.6)

    # Rolling averages
    for window in [6, 12]:  # 30min, 60min if 5-min intervals
        plt.plot(featured_df["timestamp"], 
                 featured_df["glucose_value"].rolling(window=window).mean(),
                 label=f"{window*5} min rolling avg")

    # Overlay insulin (scaled for visibility)
    if "insulin_value" in featured_df.columns:
        plt.plot(featured_df["timestamp"], 
                 featured_df["insulin_value"]*30,  # scale insulin for plotting
                 label="Insulin (scaled)", color="purple", linestyle="--")

    # Meal events
    for _, meal in meals_df.iterrows():
        plt.axvline(meal["timestamp"], color="orange", linestyle=":", alpha=0.8)
    if not meals_df.empty:
        plt.text(meals_df["timestamp"].iloc[0], featured_df["glucose_value"].max(),
                 "Meal events", color="orange", fontsize=9, verticalalignment="bottom")

    # Anomalies
    hyper = featured_df[featured_df["glucose_value"] > 180]
    hypo = featured_df[featured_df["glucose_value"] < 70]
    plt.scatter(hyper["timestamp"], hyper["glucose_value"], color="red", label="Hyperglycemia", zorder=5)
    plt.scatter(hypo["timestamp"], hypo["glucose_value"], color="blue", label="Hypoglycemia", zorder=5)

    # Formatting
    plt.axhline(70, color="blue", linestyle="--", alpha=0.5)
    plt.axhline(180, color="red", linestyle="--", alpha=0.5)
    plt.title("ğŸ“ˆ Glucose Trends with Insulin & Meal Events")
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main Pipeline
# ----------------------------
def run_diabetes_pipeline(cgm_df=None, insulin_df=None, meals_df=None, exercise_df=None, save_path="diabetes_featured_data.csv"):
    print("="*60)
    print(" RUNNING DIABETES ANALYTICS PIPELINE ")
    print("="*60)

    if cgm_df is None or insulin_df is None or meals_df is None or exercise_df is None:
        print("âš ï¸� Missing input data â€” generating sample data...")
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()

    # Align timestamps
    for df in [cgm_df, insulin_df, meals_df, exercise_df]:
        if "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True)

    # Merge datasets
    featured_df = cgm_df.copy()
    for df, col in [(insulin_df, "insulin_value"), (meals_df, "carbs"), (exercise_df, "intensity")]:
        if col in df.columns:
            featured_df = pd.merge_asof(featured_df, df, on="timestamp", direction="backward")
    featured_df.fillna(0, inplace=True)

    # Future labels
    featured_df["glucose_future"] = featured_df["glucose_value"].shift(-12)
    featured_df["hypo_risk_60min"] = (featured_df["glucose_future"] < 70).astype(int)
    featured_df["hyper_risk_60min"] = (featured_df["glucose_future"] > 180).astype(int)
    featured_df.dropna(inplace=True)

    # ----------------------------
    # Visualization
    # ----------------------------
    plot_glucose_time_series(featured_df, insulin_df, meals_df)

    # Save
    featured_df.to_csv(save_path, index=False)
    print(f"\nâœ… Processed dataset saved to {save_path}")

    return featured_df

# ----------------------------
# Run pipeline
# ----------------------------
if __name__ == "__main__":
    featured_df = run_diabetes_pipeline()



def plot_hourly_patterns(self, df: pd.DataFrame):
    """
    Multi-panel plots of average hourly patterns for glucose, insulin, carbs, and exercise.
    Helps clinicians see circadian trends.
    """
    if "timestamp" not in df.columns:
        print("â�Œ No timestamp column found for hourly plots.")
        return
    
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour

    # Group by hour
    hourly = df.groupby("hour").agg({
        "glucose": "mean",
        "insulin": "mean",
        "carbs": "mean",
        "exercise_intensity": "mean"
    }).reset_index()

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(hourly["hour"], hourly["glucose"], marker="o", color="red")
    axes[0].set_ylabel("Glucose (mg/dL)")
    axes[0].set_title("Hourly Glucose Pattern")

    axes[1].bar(hourly["hour"], hourly["insulin"], color="blue", alpha=0.6)
    axes[1].set_ylabel("Insulin (units)")
    axes[1].set_title("Hourly Insulin Usage")

    axes[2].bar(hourly["hour"], hourly["carbs"], color="green", alpha=0.6)
    axes[2].set_ylabel("Carbs (g)")
    axes[2].set_title("Hourly Carbohydrate Intake")

    axes[3].plot(hourly["hour"], hourly["exercise_intensity"], marker="s", color="purple")
    axes[3].set_ylabel("Exercise Intensity")
    axes[3].set_title("Hourly Exercise Pattern")
    axes[3].set_xlabel("Hour of Day (0-23)")

    plt.tight_layout()
    plt.show()



# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning & Statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                            classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)
import scipy.stats as stats

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Function to create sample diabetes data
def create_sample_diabetes_data():
    """Create realistic sample diabetes data for analysis"""
    print("Creating sample diabetes data...")
    
    # Generate timestamps for 3 days of data
    start_date = '2024-01-01 00:00:00'
    end_date = '2024-01-03 23:55:00'
    
    # CGM Data (Continuous Glucose Monitoring) - every 5 minutes
    cgm_dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Create realistic glucose patterns
    base_glucose = np.random.normal(120, 15, len(cgm_dates))
    
    # Add meal effects (spikes after typical meal times)
    meal_times = [8, 12, 18, 21]  # 8am, 12pm, 6pm, 9pm
    for hour in meal_times:
        meal_mask = (cgm_dates.hour == hour) & (cgm_dates.minute == 0)
        base_glucose[meal_mask] += np.random.normal(30, 8, meal_mask.sum())
    
    # Add night time decrease
    night_mask = (cgm_dates.hour >= 22) | (cgm_dates.hour <= 6)
    base_glucose[night_mask] -= np.random.normal(15, 5, night_mask.sum())
    
    # Add some random noise and clip to realistic range
    glucose_values = base_glucose + np.random.normal(0, 5, len(cgm_dates))
    glucose_values = np.clip(glucose_values, 70, 250)
    
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose_value': glucose_values,
        'patient_id': 'PATIENT_001'
    })
    
    # Insulin Data - 3-4 doses per day
    insulin_dates = []
    for day in range(3):
        # Breakfast dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        # Lunch dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        # Dinner dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        # Bedtime dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 22:00:00'))
    
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(2, 8, len(insulin_dates)),
        'insulin_type': ['rapid'] * len(insulin_dates),
        'patient_id': 'PATIENT_001'
    })
    
    # Meals Data - 3 meals per day + optional snack
    meal_dates = []
    meal_carbs = []
    meal_types = []
    
    for day in range(3):
        # Breakfast
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        meal_carbs.append(np.random.randint(40, 60))
        meal_types.append('breakfast')
        
        # Lunch
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        meal_carbs.append(np.random.randint(50, 80))
        meal_types.append('lunch')
        
        # Dinner
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        meal_carbs.append(np.random.randint(60, 90))
        meal_types.append('dinner')
        
        # Optional snack
        if np.random.random() > 0.5:
            meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 21:00:00'))
            meal_carbs.append(np.random.randint(15, 30))
            meal_types.append('snack')
    
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': meal_carbs,
        'meal_type': meal_types,
        'patient_id': 'PATIENT_001'
    })
    
    # Exercise Data - random exercise sessions
    exercise_dates = []
    exercise_durations = []
    exercise_intensities = []
    
    for day in range(3):
        if np.random.random() > 0.4:  # 60% chance of exercise each day
            hour = np.random.choice([7, 17, 19])  # Morning or evening
            exercise_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} {hour}:00:00'))
            exercise_durations.append(np.random.randint(20, 60))
            exercise_intensities.append(np.random.choice(['light', 'moderate', 'vigorous']))
    
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': exercise_durations,
        'intensity': exercise_intensities,
        'patient_id': 'PATIENT_001'
    })
    
    print("Sample diabetes data created successfully!")
    return cgm_df, insulin_df, meals_df, exercise_df

# Try to load data files with error handling
try:
    cgm_df = pd.read_csv('cgm_data.csv', parse_dates=['timestamp'])
    print("Loaded CGM data")
except FileNotFoundError:
    print("CGM data file not found")
    cgm_df = pd.DataFrame()

try:
    insulin_df = pd.read_csv('insulin_data.csv', parse_dates=['timestamp'])
    print("Loaded insulin data")
except FileNotFoundError:
    print("Insulin data file not found")
    insulin_df = pd.DataFrame()

try:
    meals_df = pd.read_csv('meals.csv', parse_dates=['timestamp'])
    print("Loaded meals data")
except FileNotFoundError:
    print("Meals data file not found")
    meals_df = pd.DataFrame()

try:
    exercise_df = pd.read_csv('exercise.csv', parse_dates=['timestamp'])
    print("Loaded exercise data")
except FileNotFoundError:
    print("Exercise data file not found")
    exercise_df = pd.DataFrame()

# If any dataframe is empty, create sample data
if cgm_df.empty or insulin_df.empty or meals_df.empty or exercise_df.empty:
    print("\nSome data files are missing. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_diabetes_data()
    
    # Save sample data for future use
    cgm_df.to_csv('cgm_data.csv', index=False)
    insulin_df.to_csv('insulin_data.csv', index=False)
    meals_df.to_csv('meals.csv', index=False)
    exercise_df.to_csv('exercise.csv', index=False)
    print("Sample data saved as CSV files")

# Now perform your analysis
print("\n" + "="*60)
print("DATA ANALYSIS")
print("="*60)

# Initial Inspection
print("CGM Data Shape:", cgm_df.shape)
print("Insulin Data Shape:", insulin_df.shape)
print("Meals Data Shape:", meals_df.shape)
print("Exercise Data Shape:", exercise_df.shape)

print("\nCGM Data Info:")
cgm_df.info()
print("\nFirst 5 rows of CGM:")
print(cgm_df.head())

# Check for missing values
print("\nMissing Values in CGM Data:")
print(cgm_df.isnull().sum())

# Basic statistics
print("\nGlucose Value Statistics:")
print(cgm_df['glucose_value'].describe())

# Additional analysis
print("\n" + "="*60)
print("ADDITIONAL ANALYSIS")
print("="*60)

# Time range analysis
if not cgm_df.empty:
    print(f"Time range: {cgm_df['timestamp'].min()} to {cgm_df['timestamp'].max()}")
    print(f"Total duration: {cgm_df['timestamp'].max() - cgm_df['timestamp'].min()}")

# Glucose distribution visualization
if not cgm_df.empty:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(cgm_df['glucose_value'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Glucose Value Distribution')
    plt.xlabel('Glucose Value (mg/dL)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Show first 24 hours of glucose data
    first_day = cgm_df[cgm_df['timestamp'] <= cgm_df['timestamp'].min() + pd.Timedelta(hours=24)]
    plt.plot(first_day['timestamp'], first_day['glucose_value'], color='red', linewidth=1)
    plt.title('Glucose Levels (First 24 Hours)')
    plt.xlabel('Time')
    plt.ylabel('Glucose (mg/dL)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Additional statistics for other dataframes
if not insulin_df.empty:
    print("\nInsulin Dose Statistics:")
    print(insulin_df['insulin_dose'].describe())

if not meals_df.empty:
    print("\nMeal Carbs Statistics:")
    print(meals_df['carbs'].describe())
    print("\nMeal Type Distribution:")
    print(meals_df['meal_type'].value_counts())

if not exercise_df.empty:
    print("\nExercise Duration Statistics:")
    print(exercise_df['duration_minutes'].describe())
    print("\nExercise Intensity Distribution:")
    print(exercise_df['intensity'].value_counts())

print("\nData analysis complete! Ready for machine learning modeling.")


# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning & Statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                            classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)
import scipy.stats as stats

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Function to create sample diabetes data
def create_sample_diabetes_data():
    """Create realistic sample diabetes data for analysis"""
    print("Creating sample diabetes data...")
    
    # Generate timestamps for 3 days of data
    start_date = '2024-01-01 00:00:00'
    end_date = '2024-01-03 23:55:00'
    
    # CGM Data (Continuous Glucose Monitoring) - every 5 minutes
    cgm_dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Create realistic glucose patterns
    base_glucose = np.random.normal(120, 15, len(cgm_dates))
    
    # Add meal effects (spikes after typical meal times)
    meal_times = [8, 12, 18, 21]  # 8am, 12pm, 6pm, 9pm
    for hour in meal_times:
        meal_mask = (cgm_dates.hour == hour) & (cgm_dates.minute == 0)
        base_glucose[meal_mask] += np.random.normal(30, 8, meal_mask.sum())
    
    # Add night time decrease
    night_mask = (cgm_dates.hour >= 22) | (cgm_dates.hour <= 6)
    base_glucose[night_mask] -= np.random.normal(15, 5, night_mask.sum())
    
    # Add some random noise and clip to realistic range
    glucose_values = base_glucose + np.random.normal(0, 5, len(cgm_dates))
    glucose_values = np.clip(glucose_values, 70, 250)
    
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose_value': glucose_values,
        'patient_id': 'PATIENT_001'
    })
    
    # Insulin Data - 3-4 doses per day
    insulin_dates = []
    for day in range(3):
        # Breakfast dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        # Lunch dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        # Dinner dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        # Bedtime dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 22:00:00'))
    
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(2, 8, len(insulin_dates)),
        'insulin_type': ['rapid'] * len(insulin_dates),
        'patient_id': 'PATIENT_001'
    })
    
    # Meals Data - 3 meals per day + optional snack
    meal_dates = []
    meal_carbs = []
    meal_types = []
    
    for day in range(3):
        # Breakfast
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        meal_carbs.append(np.random.randint(40, 60))
        meal_types.append('breakfast')
        
        # Lunch
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        meal_carbs.append(np.random.randint(50, 80))
        meal_types.append('lunch')
        
        # Dinner
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        meal_carbs.append(np.random.randint(60, 90))
        meal_types.append('dinner')
        
        # Optional snack
        if np.random.random() > 0.5:
            meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 21:00:00'))
            meal_carbs.append(np.random.randint(15, 30))
            meal_types.append('snack')
    
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': meal_carbs,
        'meal_type': meal_types,
        'patient_id': 'PATIENT_001'
    })
    
    # Exercise Data - random exercise sessions
    exercise_dates = []
    exercise_durations = []
    exercise_intensities = []
    
    for day in range(3):
        if np.random.random() > 0.4:  # 60% chance of exercise each day
            hour = np.random.choice([7, 17, 19])  # Morning or evening
            exercise_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} {hour}:00:00'))
            exercise_durations.append(np.random.randint(20, 60))
            exercise_intensities.append(np.random.choice(['light', 'moderate', 'vigorous']))
    
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': exercise_durations,
        'intensity': exercise_intensities,
        'patient_id': 'PATIENT_001'
    })
    
    print("Sample diabetes data created successfully!")
    return cgm_df, insulin_df, meals_df, exercise_df

# Try to load data files with error handling
try:
    cgm_df = pd.read_csv('cgm_data.csv', parse_dates=['timestamp'])
    print("Loaded CGM data")
except FileNotFoundError:
    print("CGM data file not found")
    cgm_df = pd.DataFrame()

try:
    insulin_df = pd.read_csv('insulin_data.csv', parse_dates=['timestamp'])
    print("Loaded insulin data")
except FileNotFoundError:
    print("Insulin data file not found")
    insulin_df = pd.DataFrame()

try:
    meals_df = pd.read_csv('meals.csv', parse_dates=['timestamp'])
    print("Loaded meals data")
except FileNotFoundError:
    print("Meals data file not found")
    meals_df = pd.DataFrame()

try:
    exercise_df = pd.read_csv('exercise.csv', parse_dates=['timestamp'])
    print("Loaded exercise data")
except FileNotFoundError:
    print("Exercise data file not found")
    exercise_df = pd.DataFrame()

# If any dataframe is empty, create sample data
if cgm_df.empty or insulin_df.empty or meals_df.empty or exercise_df.empty:
    print("\nSome data files are missing. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_diabetes_data()
    
    # Save sample data for future use
    cgm_df.to_csv('cgm_data.csv', index=False)
    insulin_df.to_csv('insulin_data.csv', index=False)
    meals_df.to_csv('meals.csv', index=False)
    exercise_df.to_csv('exercise.csv', index=False)
    print("Sample data saved as CSV files")

# Now perform your analysis
print("\n" + "="*60)
print("DATA ANALYSIS")
print("="*60)

# Initial Inspection
print("CGM Data Shape:", cgm_df.shape)
print("Insulin Data Shape:", insulin_df.shape)
print("Meals Data Shape:", meals_df.shape)
print("Exercise Data Shape:", exercise_df.shape)

print("\nCGM Data Info:")
cgm_df.info()
print("\nFirst 5 rows of CGM:")
print(cgm_df.head())

# Check for missing values
print("\nMissing Values in CGM Data:")
print(cgm_df.isnull().sum())

# Basic statistics
print("\nGlucose Value Statistics:")
print(cgm_df['glucose_value'].describe())

# Additional analysis
print("\n" + "="*60)
print("ADDITIONAL ANALYSIS")
print("="*60)

# Time range analysis
if not cgm_df.empty:
    print(f"Time range: {cgm_df['timestamp'].min()} to {cgm_df['timestamp'].max()}")
    print(f"Total duration: {cgm_df['timestamp'].max() - cgm_df['timestamp'].min()}")

# Glucose Trace Visualization
plt.figure(figsize=(20, 5))
plt.plot(cgm_df['timestamp'], cgm_df['glucose_value'], linewidth=0.8, color='blue', alpha=0.8)
plt.axhline(y=70, color='r', linestyle='--', label='Hypo Threshold (70 mg/dL)', alpha=0.8)
plt.axhline(y=180, color='orange', linestyle='--', label='Hyper Threshold (180 mg/dL)', alpha=0.8)

# Add shaded areas for hypo and hyper ranges
plt.axhspan(0, 70, color='red', alpha=0.1, label='Hypoglycemic Range')
plt.axhspan(180, 300, color='orange', alpha=0.1, label='Hyperglycemic Range')
plt.axhspan(70, 180, color='green', alpha=0.1, label='Normal Range')

plt.title('Continuous Glucose Monitor (CGM) Trace', fontsize=16, fontweight='bold')
plt.ylabel('Glucose (mg/dL)', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Additional statistics for other dataframes
if not insulin_df.empty:
    print("\nInsulin Dose Statistics:")
    print(insulin_df['insulin_dose'].describe())

if not meals_df.empty:
    print("\nMeal Carbs Statistics:")
    print(meals_df['carbs'].describe())
    print("\nMeal Type Distribution:")
    print(meals_df['meal_type'].value_counts())

if not exercise_df.empty:
    print("\nExercise Duration Statistics:")
    print(exercise_df['duration_minutes'].describe())
    print("\nExercise Intensity Distribution:")
    print(exercise_df['intensity'].value_counts())

# Additional visualizations
print("\n" + "="*60)
print("ADDITIONAL VISUALIZATIONS")
print("="*60)

# Glucose distribution histogram
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist(cgm_df['glucose_value'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=70, color='r', linestyle='--', alpha=0.8)
plt.axvline(x=180, color='orange', linestyle='--', alpha=0.8)
plt.title('Glucose Value Distribution')
plt.xlabel('Glucose Value (mg/dL)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Box plot of glucose values by hour
plt.subplot(1, 2, 2)
cgm_df['hour'] = cgm_df['timestamp'].dt.hour
hourly_data = [cgm_df[cgm_df['hour'] == hour]['glucose_value'].values for hour in range(24)]
plt.boxplot(hourly_data, positions=range(24))
plt.axhline(y=70, color='r', linestyle='--', alpha=0.8)
plt.axhline(y=180, color='orange', linestyle='--', alpha=0.8)
plt.title('Glucose Values by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Glucose (mg/dL)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Time in Range analysis
if not cgm_df.empty:
    total_readings = len(cgm_df)
    time_in_range = len(cgm_df[(cgm_df['glucose_value'] >= 70) & (cgm_df['glucose_value'] <= 180)])
    time_below_range = len(cgm_df[cgm_df['glucose_value'] < 70])
    time_above_range = len(cgm_df[cgm_df['glucose_value'] > 180])
    
    print("\nTime in Range Analysis:")
    print(f"Time in range (70-180 mg/dL): {time_in_range/total_readings*100:.1f}%")
    print(f"Time below range (<70 mg/dL): {time_below_range/total_readings*100:.1f}%")
    print(f"Time above range (>180 mg/dL): {time_above_range/total_readings*100:.1f}%")

print("\nData analysis complete! Ready for machine learning modeling.")


# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning & Statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                            classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)
import scipy.stats as stats

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Function to create sample diabetes data
def create_sample_diabetes_data():
    """Create realistic sample diabetes data for analysis"""
    print("Creating sample diabetes data...")
    
    # Generate timestamps for 7 days of data for better weekly patterns
    start_date = '2024-01-01 00:00:00'
    end_date = '2024-01-07 23:55:00'
    
    # CGM Data (Continuous Glucose Monitoring) - every 5 minutes
    cgm_dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Create realistic glucose patterns with weekly variations
    base_glucose = np.random.normal(120, 15, len(cgm_dates))
    
    # Add meal effects (spikes after typical meal times)
    meal_times = [8, 12, 18, 21]  # 8am, 12pm, 6pm, 9pm
    for hour in meal_times:
        meal_mask = (cgm_dates.hour == hour) & (cgm_dates.minute == 0)
        base_glucose[meal_mask] += np.random.normal(30, 8, meal_mask.sum())
    
    # Add night time decrease
    night_mask = (cgm_dates.hour >= 22) | (cgm_dates.hour <= 6)
    base_glucose[night_mask] -= np.random.normal(15, 5, night_mask.sum())
    
    # Add weekend effect (slightly different patterns)
    weekend_mask = (cgm_dates.dayofweek >= 5)  # Saturday and Sunday
    base_glucose[weekend_mask] += np.random.normal(5, 3, weekend_mask.sum())
    
    # Add some random noise and clip to realistic range
    glucose_values = base_glucose + np.random.normal(0, 5, len(cgm_dates))
    glucose_values = np.clip(glucose_values, 70, 250)
    
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose_value': glucose_values,
        'patient_id': 'PATIENT_001'
    })
    
    # Insulin Data - 3-4 doses per day
    insulin_dates = []
    for day in range(7):
        # Breakfast dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        # Lunch dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        # Dinner dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        # Bedtime dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 22:00:00'))
    
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': np.random.uniform(2, 8, len(insulin_dates)),
        'insulin_type': ['rapid'] * len(insulin_dates),
        'patient_id': 'PATIENT_001'
    })
    
    # Meals Data - 3 meals per day + optional snack
    meal_dates = []
    meal_carbs = []
    meal_types = []
    
    for day in range(7):
        # Breakfast
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        meal_carbs.append(np.random.randint(40, 60))
        meal_types.append('breakfast')
        
        # Lunch
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        meal_carbs.append(np.random.randint(50, 80))
        meal_types.append('lunch')
        
        # Dinner
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        meal_carbs.append(np.random.randint(60, 90))
        meal_types.append('dinner')
        
        # Optional snack
        if np.random.random() > 0.5:
            meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 21:00:00'))
            meal_carbs.append(np.random.randint(15, 30))
            meal_types.append('snack')
    
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': meal_carbs,
        'meal_type': meal_types,
        'patient_id': 'PATIENT_001'
    })
    
    # Exercise Data - random exercise sessions
    exercise_dates = []
    exercise_durations = []
    exercise_intensities = []
    
    for day in range(7):
        if np.random.random() > 0.4:  # 60% chance of exercise each day
            hour = np.random.choice([7, 17, 19])  # Morning or evening
            exercise_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} {hour}:00:00'))
            exercise_durations.append(np.random.randint(20, 60))
            exercise_intensities.append(np.random.choice(['light', 'moderate', 'vigorous']))
    
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': exercise_durations,
        'intensity': exercise_intensities,
        'patient_id': 'PATIENT_001'
    })
    
    print("Sample diabetes data created successfully!")
    return cgm_df, insulin_df, meals_df, exercise_df

# Try to load data files with error handling
try:
    cgm_df = pd.read_csv('cgm_data.csv', parse_dates=['timestamp'])
    print("Loaded CGM data")
except FileNotFoundError:
    print("CGM data file not found")
    cgm_df = pd.DataFrame()

try:
    insulin_df = pd.read_csv('insulin_data.csv', parse_dates=['timestamp'])
    print("Loaded insulin data")
except FileNotFoundError:
    print("Insulin data file not found")
    insulin_df = pd.DataFrame()

try:
    meals_df = pd.read_csv('meals.csv', parse_dates=['timestamp'])
    print("Loaded meals data")
except FileNotFoundError:
    print("Meals data file not found")
    meals_df = pd.DataFrame()

try:
    exercise_df = pd.read_csv('exercise.csv', parse_dates=['timestamp'])
    print("Loaded exercise data")
except FileNotFoundError:
    print("Exercise data file not found")
    exercise_df = pd.DataFrame()

# If any dataframe is empty, create sample data
if cgm_df.empty or insulin_df.empty or meals_df.empty or exercise_df.empty:
    print("\nSome data files are missing. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_diabetes_data()
    
    # Save sample data for future use
    cgm_df.to_csv('cgm_data.csv', index=False)
    insulin_df.to_csv('insulin_data.csv', index=False)
    meals_df.to_csv('meals.csv', index=False)
    exercise_df.to_csv('exercise.csv', index=False)
    print("Sample data saved as CSV files")

# Extract time features for analysis
cgm_df['hour'] = cgm_df['timestamp'].dt.hour
cgm_df['day_of_week'] = cgm_df['timestamp'].dt.day_name()
cgm_df['date'] = cgm_df['timestamp'].dt.date
cgm_df['time_of_day'] = cgm_df['timestamp'].dt.time

# Average glucose by hour of day
hourly_avg = cgm_df.groupby('hour')['glucose_value'].mean()
hourly_std = cgm_df.groupby('hour')['glucose_value'].std()
hourly_count = cgm_df.groupby('hour')['glucose_value'].count()

plt.figure(figsize=(14, 10))

# Line plot with confidence intervals
plt.subplot(2, 2, 1)
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker='o', linewidth=2.5, color='blue')
plt.fill_between(hourly_avg.index, 
                 hourly_avg - hourly_std, 
                 hourly_avg + hourly_std, 
                 alpha=0.2, color='blue', label='Â±1 Std Dev')
plt.axhline(y=70, color='r', linestyle='--', linewidth=2, label='Hypo Threshold (70)')
plt.axhline(y=180, color='orange', linestyle='--', linewidth=2, label='Hyper Threshold (180)')
plt.title('Average Glucose by Hour of Day with Variability', fontsize=14, fontweight='bold')
plt.ylabel('Mean Glucose (mg/dL)', fontsize=12)
plt.xlabel('Hour of Day', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(50, 220)

# Boxplot for variability by hour
plt.subplot(2, 2, 2)
sns.boxplot(x='hour', y='glucose_value', data=cgm_df, palette='viridis')
plt.axhline(y=70, color='r', linestyle='--', linewidth=2)
plt.axhline(y=180, color='orange', linestyle='--', linewidth=2)
plt.title('Glucose Distribution by Hour of Day', fontsize=14, fontweight='bold')
plt.ylabel('Glucose (mg/dL)', fontsize=12)
plt.xlabel('Hour of Day', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Glucose by day of week
plt.subplot(2, 2, 3)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_avg = cgm_df.groupby('day_of_week')['glucose_value'].mean().reindex(day_order)
sns.barplot(x=daily_avg.index, y=daily_avg.values, palette='coolwarm')
plt.axhline(y=70, color='r', linestyle='--', linewidth=2)
plt.axhline(y=180, color='orange', linestyle='--', linewidth=2)
plt.title('Average Glucose by Day of Week', fontsize=14, fontweight='bold')
plt.ylabel('Mean Glucose (mg/dL)', fontsize=12)
plt.xlabel('Day of Week', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Heatmap of glucose by hour and day
plt.subplot(2, 2, 4)
# Create pivot table for heatmap
heatmap_data = cgm_df.pivot_table(values='glucose_value', 
                                 index='day_of_week', 
                                 columns='hour', 
                                 aggfunc='mean').reindex(day_order)
sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Glucose (mg/dL)'})
plt.title('Glucose Heatmap: Hour vs Day of Week', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Day of Week', fontsize=12)

plt.tight_layout()
plt.show()

# Additional statistics
print("\n" + "="*60)
print("HOURLY GLUCOSE ANALYSIS")
print("="*60)

# Display hourly statistics
hourly_stats = cgm_df.groupby('hour')['glucose_value'].agg(['mean', 'std', 'min', 'max', 'count'])
print("Hourly Glucose Statistics:")
print(hourly_stats.round(1))

# Identify highest and lowest glucose hours
max_hour = hourly_avg.idxmax()
min_hour = hourly_avg.idxmin()
print(f"\nHighest average glucose: {max_hour}:00 ({hourly_avg[max_hour]:.1f} mg/dL)")
print(f"Lowest average glucose: {min_hour}:00 ({hourly_avg[min_hour]:.1f} mg/dL)")

# Time in range analysis by hour
print("\nTime in Range by Hour:")
for hour in range(24):
    hour_data = cgm_df[cgm_df['hour'] == hour]
    total = len(hour_data)
    in_range = len(hour_data[(hour_data['glucose_value'] >= 70) & (hour_data['glucose_value'] <= 180)])
    below_range = len(hour_data[hour_data['glucose_value'] < 70])
    above_range = len(hour_data[hour_data['glucose_value'] > 180])
    
    if total > 0:
        print(f"Hour {hour:02d}: {in_range/total*100:.1f}% in range, "
              f"{below_range/total*100:.1f}% below, {above_range/total*100:.1f}% above")

# Weekly patterns analysis
print("\n" + "="*60)
print("WEEKLY PATTERNS ANALYSIS")
print("="*60)

daily_stats = cgm_df.groupby('day_of_week')['glucose_value'].agg(['mean', 'std', 'count']).reindex(day_order)
print("Daily Glucose Statistics:")
print(daily_stats.round(1))

# Identify best and worst days
best_day = daily_stats['mean'].idxmin()
worst_day = daily_stats['mean'].idxmax()
print(f"\nBest glucose control: {best_day} ({daily_stats.loc[best_day, 'mean']:.1f} mg/dL)")
print(f"Worst glucose control: {worst_day} ({daily_stats.loc[worst_day, 'mean']:.1f} mg/dL)")

# Advanced: Coefficient of Variation (CV) by hour
cv_by_hour = (hourly_std / hourly_avg) * 100
max_cv_hour = cv_by_hour.idxmax()
min_cv_hour = cv_by_hour.idxmin()

print(f"\nGlucose Variability (Coefficient of Variation):")
print(f"Most variable hour: {max_cv_hour}:00 (CV: {cv_by_hour[max_cv_hour]:.1f}%)")
print(f"Least variable hour: {min_cv_hour}:00 (CV: {cv_by_hour[min_cv_hour]:.1f}%)")

print("\nAnalysis complete! Ready for predictive modeling.")


# Example features for a specific timestamp 't'
def create_features(df):
    # Copy the dataframe
    df = df.copy()
    
    # 1. Rolling Statistics from CGM (Past 30 mins to 3 hours)
    df['glucose_rolling_mean_30m'] = df['glucose_value'].rolling(window=6).mean() # 6 * 5min = 30min
    df['glucose_rolling_std_1h'] = df['glucose_value'].rolling(window=12).std()   # 12 * 5min = 1h
    df['glucose_slope_45m'] = df['glucose_value'].diff(periods=9)                 # 9 * 5min = 45min
    
    # 2. Time Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    
    # 3. Event-based Features (This requires merging other datasets)
    # ... Code to merge insulin, meal, exercise data onto the main CGM timeline ...
    
    # 4. Calculate Insulin-On-Board (IOB) and Carbs-On-Board (COB)
    # This requires a pharmacokinetic model. A simple exponential decay model is a common start.
    # Example: IOB = insulin_dose * exp(-elapsed_time / half_life)
    # This is complex and must be done carefully.
    
    # 5. Create the prediction target: Glucose value 60 minutes in the future
    df['glucose_60min_future'] = df['glucose_value'].shift(-12) # 12 * 5min = 60min
    
    # Drop rows with NaN values created by rolling/shifting
    df = df.dropna() 
    
    return df

# Apply the feature engineering function
featured_df = create_features(cgm_df)
print(featured_df.head())


cgm_df['hour'] = cgm_df['timestamp'].dt.hour
cgm_df['day_of_week'] = cgm_df['timestamp'].dt.day_name()

# Average glucose by hour of day
hourly_avg = cgm_df.groupby('hour')['glucose_value'].mean()
plt.figure(figsize=(12, 4))
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values)
plt.axhline(y=70, color='r', linestyle='--')
plt.axhline(y=180, color='orange', linestyle='--')
plt.title('Average Glucose by Hour of Day')
plt.ylabel('Mean Glucose (mg/dL)')
plt.xlabel('Hour of Day')
plt.xticks(range(0,24))
plt.grid(True)
plt.show()

# Boxplot for variability by hour
plt.figure(figsize=(16, 6))
sns.boxplot(x='hour', y='glucose_value', data=cgm_df)
plt.title('Glucose Distribution by Hour of Day')
plt.show()


# Create the target variable
featured_df['hypo_event'] = (featured_df['glucose_60min_future'] < 70).astype(int)

# Check class balance (HYPO events are usually rare)
print(featured_df['hypo_event'].value_counts())
print(f"\nPercentage of Hypo events: {featured_df['hypo_event'].mean()*100:.2f}%")

# Define features and target
X = featured_df.drop(['glucose_60min_future', 'hypo_event', 'timestamp', 'day_of_week'], axis=1, errors='ignore') # Drop non-feature columns
y = featured_df['hypo_event']

# Split the data chronologically (never shuffle time series randomly!)
split_index = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


# ============================================
# END-TO-END DIABETES PIPELINE (E2E Version)
# ============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, roc_auc_score

# ----------------------------
# Sample Data Generator
# ----------------------------
def create_sample_data():
    cgm_dates = pd.date_range(start='2024-01-01', periods=288, freq='5min')
    cgm_df = pd.DataFrame({'timestamp': cgm_dates, 'glucose_value': np.random.uniform(60, 220, len(cgm_dates))})

    insulin_dates = pd.date_range(start='2024-01-01', periods=20, freq='2h')
    insulin_df = pd.DataFrame({'timestamp': insulin_dates, 'insulin_value': np.random.uniform(1, 5, len(insulin_dates))})

    meal_dates = pd.date_range(start='2024-01-01', periods=10, freq='4h')
    meals_df = pd.DataFrame({'timestamp': meal_dates, 'carbs': np.random.uniform(30, 90, len(meal_dates))})

    exercise_dates = pd.date_range(start='2024-01-01', periods=5, freq='6h')
    exercise_df = pd.DataFrame({'timestamp': exercise_dates, 'intensity': np.random.uniform(0.3, 1.0, len(exercise_dates))})

    return cgm_df, insulin_df, meals_df, exercise_df

# ----------------------------
# Pipeline
# ----------------------------
def run_pipeline(cgm_df=None, insulin_df=None, meals_df=None, exercise_df=None, save_path='featured_diabetes.csv'):
    print("="*70)
    print(" RUNNING END-TO-END DIABETES PIPELINE ")
    print("="*70)

    # If no data provided, generate sample
    if cgm_df is None or insulin_df is None or meals_df is None or exercise_df is None:
        cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()

    # Merge datasets
    featured_df = cgm_df.copy()
    for df, col in [(insulin_df, 'insulin_value'), (meals_df, 'carbs'), (exercise_df, 'intensity')]:
        if col in df.columns:
            featured_df = pd.merge_asof(featured_df.sort_values('timestamp'), df.sort_values('timestamp'),
                                        on='timestamp', direction='backward')
    featured_df.fillna(0, inplace=True)

    # Labels
    featured_df['glucose_future'] = featured_df['glucose_value'].shift(-12)
    featured_df['hypo_risk_60min'] = (featured_df['glucose_future'] < 70).astype(int)
    featured_df['hyper_risk_60min'] = (featured_df['glucose_future'] > 180).astype(int)
    featured_df.dropna(inplace=True)

    # Features
    exclude_cols = ['timestamp', 'glucose_future', 'hypo_risk_60min', 'hyper_risk_60min']
    X = featured_df[[c for c in featured_df.columns if c not in exclude_cols]]
    y_reg = featured_df['glucose_future']
    y_hypo = featured_df['hypo_risk_60min']
    y_hyper = featured_df['hyper_risk_60min']

    for col in X.select_dtypes(include=['object', 'category']):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split (time-based)
    train_size = int(0.8 * len(X))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train_reg, y_test_reg = y_reg[:train_size], y_reg[train_size:]
    y_train_hypo, y_test_hypo = y_hypo[:train_size], y_hypo[train_size:]
    y_train_hyper, y_test_hyper = y_hyper[:train_size], y_hyper[train_size:]

    # ----------------------------
    # Regression Model
    # ----------------------------
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train_reg)
    y_pred_reg = reg_model.predict(X_test)
    print(f"Regression -> MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.2f}")

    # ----------------------------
    # Classification Models
    # ----------------------------
    def safe_predict_proba(model, X):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.zeros(len(proba))
        return proba[:,1]

    # Hypo
    hypo_model = RandomForestClassifier(n_estimators=100, random_state=42)
    hypo_model.fit(X_train, y_train_hypo)
    y_pred_hypo = hypo_model.predict(X_test)
    y_proba_hypo = safe_predict_proba(hypo_model, X_test)
    print("\nHypoglycemia Classification Report:")
    print(classification_report(y_test_hypo, y_pred_hypo, zero_division=0))
    print(f"AUC: {roc_auc_score(y_test_hypo, y_proba_hypo):.3f}")

    # Hyper
    hyper_model = RandomForestClassifier(n_estimators=100, random_state=42)
    hyper_model.fit(X_train, y_train_hyper)
    y_pred_hyper = hyper_model.predict(X_test)
    y_proba_hyper = safe_predict_proba(hyper_model, X_test)
    print("\nHyperglycemia Classification Report:")
    print(classification_report(y_test_hyper, y_pred_hyper, zero_division=0))
    print(f"AUC: {roc_auc_score(y_test_hyper, y_proba_hyper):.3f}")

    # ----------------------------
    # Feature Importance
    # ----------------------------
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': reg_model.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x='importance', y='feature', data=feat_imp.head(10))
    plt.title("Top 10 Features - Glucose Prediction")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Visualizations
    # ----------------------------
    plt.figure(figsize=(12,6))
    plt.plot(featured_df['timestamp'], featured_df['glucose_value'], label="Glucose")
    plt.axhline(70, color='red', linestyle='--', label='Hypo threshold')
    plt.axhline(180, color='orange', linestyle='--', label='Hyper threshold')
    plt.legend(); plt.title("CGM Glucose Over Time"); plt.show()

    plt.figure(figsize=(8,5))
    sns.histplot(featured_df['glucose_value'], bins=30, kde=True)
    plt.title("Glucose Distribution"); plt.show()

    # ----------------------------
    # Metrics
    # ----------------------------
    tir = ((featured_df['glucose_value'] >= 70) & (featured_df['glucose_value'] <= 180)).mean() * 100
    print(f"\nTime in Range (70â€“180 mg/dL): {tir:.2f}%")

    # Save final dataset
    featured_df.to_csv(save_path, index=False)
    print(f"Dataset saved -> {save_path}")

    return featured_df, reg_model, hypo_model, hyper_model

# ----------------------------
# Run Example
# ----------------------------
if __name__ == "__main__":
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_data()
    run_pipeline(cgm_df, insulin_df, meals_df, exercise_df)


# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# Load your diabetes dataset (replace with your actual data source)
# For demonstration, I'll create a sample dataset if none exists
try:
    featured_df = pd.read_csv('diabetes_data.csv')  # Replace with your actual data file
    print("Data loaded successfully")
except FileNotFoundError:
    print("Creating sample data for demonstration...")
    # Create sample data with the required columns
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
    
    featured_df = pd.DataFrame({
        'timestamp': dates,
        'glucose_value': np.random.normal(120, 40, n_samples),
        'hyper_risk_60min': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'hypo_risk_60min': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'glucose_future': np.random.normal(125, 35, n_samples),
        'hour': dates.hour,
        'day_of_week': dates.day_name(),
        'iob': np.random.uniform(0, 5, n_samples),
        'time_since_insulin': np.random.exponential(120, n_samples),
        'time_since_meal': np.random.exponential(180, n_samples),
        'time_since_exercise': np.random.exponential(300, n_samples),
        'glucose_roc_1h': np.random.normal(0, 0.1, n_samples),
        'patient_id': 'sample_patient'
    })
    
    # Ensure glucose values are within a realistic range
    featured_df['glucose_value'] = np.clip(featured_df['glucose_value'], 40, 400)
    featured_df['glucose_future'] = np.clip(featured_df['glucose_future'], 40, 400)

# Define numerical features for correlation matrix
numerical_features = ['glucose_value', 'iob', 'time_since_insulin', 
                     'time_since_meal', 'time_since_exercise', 'glucose_roc_1h']

# Now proceed with your original code
print(f"Hyperglycemia risk (glucose > 180): {featured_df['hyper_risk_60min'].mean()*100:.2f}%")

# Check for missing values
print("\nMissing values in featured dataset:")
print(featured_df.isnull().sum().sum())

# Data Visualization
print("\n" + "="*60)
print("DATA VISUALIZATION")
print("="*60)

# Set up the visualization style
plt.style.use('default')
sns.set_palette("husl")

# 1. Glucose Distribution Over Time
plt.figure(figsize=(15, 10))

# Glucose timeline
plt.subplot(2, 2, 1)
sample_data = featured_df.iloc[::12]  # Sample every hour to reduce data points
plt.plot(sample_data['timestamp'], sample_data['glucose_value'], alpha=0.7, linewidth=1)
plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Hypo Threshold (70)')
plt.axhline(y=180, color='orange', linestyle='--', alpha=0.7, label='Hyper Threshold (180)')
plt.title('Glucose Levels Over Time')
plt.xlabel('Time')
plt.ylabel('Glucose (mg/dL)')
plt.legend()
plt.xticks(rotation=45)

# Glucose distribution
plt.subplot(2, 2, 2)
sns.histplot(featured_df['glucose_value'], kde=True, bins=30)
plt.axvline(x=70, color='r', linestyle='--', alpha=0.7, label='Hypo Threshold')
plt.axvline(x=180, color='orange', linestyle='--', alpha=0.7, label='Hyper Threshold')
plt.title('Glucose Distribution')
plt.xlabel('Glucose (mg/dL)')
plt.ylabel('Frequency')
plt.legend()

# Glucose by time of day
plt.subplot(2, 2, 3)
hourly_glucose = featured_df.groupby('hour')['glucose_value'].mean()
plt.plot(hourly_glucose.index, hourly_glucose.values, marker='o')
plt.axhline(y=70, color='r', linestyle='--', alpha=0.7)
plt.axhline(y=180, color='orange', linestyle='--', alpha=0.7)
plt.title('Average Glucose by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Glucose (mg/dL)')
plt.xticks(range(0, 24, 2))

# Glucose by day of week
plt.subplot(2, 2, 4)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_glucose = featured_df.groupby('day_of_week')['glucose_value'].mean().reindex(day_order)
plt.bar(daily_glucose.index, daily_glucose.values)
plt.axhline(y=70, color='r', linestyle='--', alpha=0.7)
plt.axhline(y=180, color='orange', linestyle='--', alpha=0.7)
plt.title('Average Glucose by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Glucose (mg/dL)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = featured_df[numerical_features].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# 3. Event Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Insulin impact
if 'iob' in featured_df.columns:
    axes[0, 0].scatter(featured_df['time_since_insulin'], featured_df['glucose_value'], alpha=0.5)
    axes[0, 0].set_title('Glucose vs Time Since Insulin')
    axes[0, 0].set_xlabel('Minutes Since Insulin')
    axes[0, 0].set_ylabel('Glucose (mg/dL)')

# Meal impact
if 'time_since_meal' in featured_df.columns:
    axes[0, 1].scatter(featured_df['time_since_meal'], featured_df['glucose_value'], alpha=0.5)
    axes[0, 1].set_title('Glucose vs Time Since Meal')
    axes[0, 1].set_xlabel('Minutes Since Meal')
    axes[0, 1].set_ylabel('Glucose (mg/dL)')

# Exercise impact
if 'time_since_exercise' in featured_df.columns:
    axes[1, 0].scatter(featured_df['time_since_exercise'], featured_df['glucose_value'], alpha=0.5)
    axes[1, 0].set_title('Glucose vs Time Since Exercise')
    axes[1, 0].set_xlabel('Minutes Since Exercise')
    axes[1, 0].set_ylabel('Glucose (mg/dL)')

# Glucose rate of change
axes[1, 1].hist(featured_df['glucose_roc_1h'].dropna(), bins=30, alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
axes[1, 1].set_title('Glucose Rate of Change Distribution')
axes[1, 1].set_xlabel('Glucose Change per Minute (mg/dL/min)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Machine Learning Preparation
print("\n" + "="*60)
print("MACHINE LEARNING PREPARATION")
print("="*60)

# Prepare features and targets
# Exclude non-feature columns
exclude_cols = ['timestamp', 'patient_id', 'day_of_week', 'time_of_day_category', 'glucose_future', 
                'hypo_risk_60min', 'hyper_risk_60min', 'intensity']

feature_cols = [col for col in featured_df.columns if col not in exclude_cols]

X = featured_df[feature_cols]
y_regression = featured_df['glucose_future']
y_hypo = featured_df['hypo_risk_60min']
y_hyper = featured_df['hyper_risk_60min']

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Handle missing values if any
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time-based split (more realistic for time series)
train_size = int(0.8 * len(X))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train_reg, y_test_reg = y_regression[:train_size], y_regression[train_size:]
y_train_hypo, y_test_hypo = y_hypo[:train_size], y_hypo[train_size:]
y_train_hyper, y_test_hyper = y_hyper[:train_size], y_hyper[train_size:]

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Regression Model (Predict glucose value)
print("\nTraining regression model...")
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train_reg)

# Make predictions
y_pred_reg = reg_model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

print(f"Regression Performance:")
print(f"MAE: {mae:.2f} mg/dL")
print(f"RMSE: {rmse:.2f} mg/dL")

# Classification Model (Hypoglycemia risk)
print("\nTraining hypoglycemia classification model...")
hypo_model = RandomForestClassifier(n_estimators=100, random_state=42)
hypo_model.fit(X_train, y_train_hypo)

# Make predictions
y_pred_hypo = hypo_model.predict(X_test)
y_pred_hypo_proba = hypo_model.predict_proba(X_test)[:, 1]

# Calculate metrics
hypo_auc = roc_auc_score(y_test_hypo, y_pred_hypo_proba)

print(f"Hypoglycemia Classification Performance:")
print(classification_report(y_test_hypo, y_pred_hypo))
print(f"AUC: {hypo_auc:.3f}")

# Classification Model (Hyperglycemia risk)
print("\nTraining hyperglycemia classification model...")
hyper_model = RandomForestClassifier(n_estimators=100, random_state=42)
hyper_model.fit(X_train, y_train_hyper)

# Make predictions
y_pred_hyper = hyper_model.predict(X_test)
y_pred_hyper_proba = hyper_model.predict_proba(X_test)[:, 1]

# Calculate metrics
hyper_auc = roc_auc_score(y_test_hyper, y_pred_hyper_proba)

print(f"Hyperglycemia Classification Performance:")
print(classification_report(y_test_hyper, y_pred_hyper))
print(f"AUC: {hyper_auc:.3f}")

# Feature Importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': reg_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importance for Glucose Prediction')
plt.tight_layout()
plt.show()

print("Top 10 most important features:")
print(feature_importance.head(10))

# Advanced Analysis: Time in Range
print("\n" + "="*60)
print("TIME IN RANGE ANALYSIS")
print("="*60)

# Calculate time in range metrics
time_in_range = ((featured_df['glucose_value'] >= 70) & (featured_df['glucose_value'] <= 180)).mean() * 100
time_below_range = (featured_df['glucose_value'] < 70).mean() * 100
time_above_range = (featured_df['glucose_value'] > 180).mean() * 100

print(f"Time in Range (70-180 mg/dL): {time_in_range:.2f}%")
print(f"Time Below Range (<70 mg/dL): {time_below_range:.2f}%")
print(f"Time Above Range (>180 mg/dL): {time_above_range:.2f}%")

# Glucose Management Indicator (GMI)
gmi = 3.31 + 0.02392 * featured_df['glucose_value'].mean()
print(f"Glucose Management Indicator (GMI): {gmi:.2f}%")

# Glucose Variability
cv = (featured_df['glucose_value'].std() / featured_df['glucose_value'].mean()) * 100
print(f"Coefficient of Variation: {cv:.2f}%")

# Create a summary report
print("\n" + "="*60)
print("DIABETES MANAGEMENT SUMMARY REPORT")
print("="*60)
print(f"Data Period: {featured_df['timestamp'].min().date()} to {featured_df['timestamp'].max().date()}")
print(f"Total Readings: {len(featured_df)}")
print(f"Average Glucose: {featured_df['glucose_value'].mean():.1f} mg/dL")
print(f"Median Glucose: {featured_df['glucose_value'].median():.1f} mg/dL")
print(f"Time in Range: {time_in_range:.1f}%")
print(f"Hypoglycemia Risk: {time_below_range:.1f}%")
print(f"Hyperglycemia Risk: {time_above_range:.1f}%")
print(f"Glucose Variability (CV): {cv:.1f}%")
print(f"Prediction Model Performance:")
print(f"  - Glucose Prediction RMSE: {rmse:.1f} mg/dL")
print(f"  - Hypoglycemia Prediction AUC: {hypo_auc:.3f}")
print(f"  - Hyperglycemia Prediction AUC: {hyper_auc:.3f}")

# Save the featured dataset for future use
featured_df.to_csv('diabetes_featured_data.csv', index=False)
print("\nFeatured dataset saved as 'diabetes_featured_data.csv'")


 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

def run_pipeline(cgm_df, insulin_df=None, meals_df=None, exercise_df=None,
                 prediction_horizon=12, save_csv=True, export_summary_md=True, summary_file='diabetes_summary.md'):
    
    print("\n" + "="*60)
    print("RUNNING FULL DIABETES PIPELINE")
    print("="*60)

    # ------------------------------
    # Safe-checks
    # ------------------------------
    if cgm_df is None or cgm_df.empty:
        raise ValueError("CGM data is required for the pipeline.")
    
    # Fill optional empty datasets
    insulin_df = pd.DataFrame() if insulin_df is None else insulin_df
    meals_df = pd.DataFrame() if meals_df is None else meals_df
    exercise_df = pd.DataFrame() if exercise_df is None else exercise_df
    
    # ------------------------------
    # Extract time features
    # ------------------------------
    cgm_df = cgm_df.copy()
    cgm_df['hour'] = cgm_df['timestamp'].dt.hour
    cgm_df['day_of_week'] = cgm_df['timestamp'].dt.day_name()
    cgm_df['date'] = cgm_df['timestamp'].dt.date
    cgm_df['time_of_day'] = cgm_df['timestamp'].dt.time
    
    # ------------------------------
    # Feature Engineering
    # ------------------------------
    def create_features(cgm_df, insulin_df, meals_df, exercise_df, prediction_horizon=12):
        df = cgm_df.copy()
        # Rolling stats
        df['glucose_rolling_mean_30m'] = df['glucose_value'].rolling(6).mean()
        df['glucose_rolling_mean_1h'] = df['glucose_value'].rolling(12).mean()
        df['glucose_rolling_std_30m'] = df['glucose_value'].rolling(6).std()
        df['glucose_slope_30m'] = df['glucose_value'].diff(6)
        df['glucose_roc_30m'] = df['glucose_slope_30m'] / 30
        
        # Time features
        df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
        df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
        df['is_weekend'] = df['day_of_week'].isin(['Saturday','Sunday']).astype(int)
        df['time_of_day_category'] = pd.cut(df['hour'], bins=[0,6,12,18,24],
                                            labels=['Night','Morning','Afternoon','Evening'],
                                            include_lowest=True)
        # Event-based features
        if not insulin_df.empty:
            insulin_events = insulin_df[['timestamp','insulin_dose']].rename(columns={'timestamp':'insulin_time'})
            df = pd.merge_asof(df.sort_values('timestamp'), 
                               insulin_events.sort_values('insulin_time'),
                               left_on='timestamp', right_on='insulin_time', direction='backward')
            df['time_since_insulin'] = (df['timestamp'] - df['insulin_time']).dt.total_seconds()/60
            df['insulin_dose'] = df['insulin_dose'].fillna(0)
            df['iob'] = df['insulin_dose'] * np.exp(-df['time_since_insulin']/75)
        
        if not meals_df.empty:
            meal_events = meals_df[['timestamp','carbs']].rename(columns={'timestamp':'meal_time'})
            df = pd.merge_asof(df.sort_values('timestamp'),
                               meal_events.sort_values('meal_time'),
                               left_on='timestamp', right_on='meal_time', direction='backward')
            df['time_since_meal'] = (df['timestamp'] - df['meal_time']).dt.total_seconds()/60
            df['carbs'] = df['carbs'].fillna(0)
            df['cob'] = df['carbs'] * np.exp(-df['time_since_meal']/120)
        
        if not exercise_df.empty:
            exercise_events = exercise_df[['timestamp','duration_minutes','intensity']].rename(columns={'timestamp':'exercise_time'})
            df = pd.merge_asof(df.sort_values('timestamp'),
                               exercise_events.sort_values('exercise_time'),
                               left_on='timestamp', right_on='exercise_time', direction='backward')
            df['time_since_exercise'] = (df['timestamp'] - df['exercise_time']).dt.total_seconds()/60
            df['duration_minutes'] = df['duration_minutes'].fillna(0)
            df['intensity'] = df['intensity'].fillna('none')
            intensity_map = {'none':0,'light':1,'moderate':2,'vigorous':3}
            df['exercise_intensity_score'] = df['intensity'].map(intensity_map)
            df['exercise_impact'] = df['duration_minutes']*df['exercise_intensity_score']*np.exp(-df['time_since_exercise']/120)
        
        # Prediction targets
        df['glucose_future'] = df['glucose_value'].shift(-prediction_horizon)
        df['hypo_risk_60min'] = (df['glucose_future'] < 70).astype(int)
        df['hyper_risk_60min'] = (df['glucose_future'] > 180).astype(int)
        
        df = df.dropna()
        df = df.drop(columns=[col for col in ['date','time_of_day','insulin_time','meal_time','exercise_time'] if col in df.columns])
        return df
    
    featured_df = create_features(cgm_df, insulin_df, meals_df, exercise_df, prediction_horizon)
    
    print(f"\nFeatured data shape: {featured_df.shape}")
    
    # ------------------------------
    # Feature Statistics + Target Analysis
    # ------------------------------
    numerical_features = featured_df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features = [f for f in numerical_features if f not in ['hypo_risk_60min','hyper_risk_60min']]
    
    print("\nFeature Statistics (first 10 features):")
    for feature in numerical_features[:10]:
        print(f"{feature}: mean={featured_df[feature].mean():.2f}, std={featured_df[feature].std():.2f}")
    
    print("\nTarget variable - glucose_future:")
    print(featured_df['glucose_future'].describe())
    print(f"Hypoglycemia risk (glucose < 70): {featured_df['hypo_risk_60min'].mean()*100:.2f}%")
    print(f"Hyperglycemia risk (glucose > 180): {featured_df['hyper_risk_60min'].mean()*100:.2f}%")
    
    # ------------------------------
    # Machine Learning Preparation
    # ------------------------------
    exclude_cols = ['timestamp','patient_id','day_of_week','time_of_day_category','glucose_future','hypo_risk_60min','hyper_risk_60min','intensity']
    X = featured_df[[col for col in featured_df.columns if col not in exclude_cols]]
    y_reg = featured_df['glucose_future']
    y_hypo = featured_df['hypo_risk_60min']
    y_hyper = featured_df['hyper_risk_60min']
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object','category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Fill missing values & scale
    X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # TimeSeries Split
    train_size = int(0.8*len(X))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train_reg, y_test_reg = y_reg[:train_size], y_reg[train_size:]
    y_train_hypo, y_test_hypo = y_hypo[:train_size], y_hypo[train_size:]
    y_train_hyper, y_test_hyper = y_hyper[:train_size], y_hyper[train_size:]
    
    # ------------------------------
    # Regression Model
    # ------------------------------
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train_reg)
    y_pred_reg = reg_model.predict(X_test)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    print(f"\nRegression - Glucose Prediction: MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    # ------------------------------
    # Classification Models
    # ------------------------------
    hypo_model = RandomForestClassifier(n_estimators=100, random_state=42)
    hypo_model.fit(X_train, y_train_hypo)
    y_pred_hypo = hypo_model.predict(X_test)
    y_pred_hypo_proba = hypo_model.predict_proba(X_test)[:,1]
    hypo_auc = roc_auc_score(y_test_hypo, y_pred_hypo_proba)
    print("\nHypoglycemia Classification Performance:")
    print(classification_report(y_test_hypo, y_pred_hypo))
    print(f"AUC: {hypo_auc:.3f}")
    
    hyper_model = RandomForestClassifier(n_estimators=100, random_state=42)
    hyper_model.fit(X_train, y_train_hyper)
    y_pred_hyper = hyper_model.predict(X_test)
    y_pred_hyper_proba = hyper_model.predict_proba(X_test)



# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning & Statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                            classification_report, confusion_matrix, ConfusionMatrixDisplay, 
                            roc_auc_score, precision_recall_curve, average_precision_score)
import scipy.stats as stats

# XGBoost for gradient boosting
import xgboost as xgb

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Function to create sample diabetes data
def create_sample_diabetes_data():
    """Create realistic sample diabetes data for analysis"""
    print("Creating sample diabetes data...")
    
    # Generate timestamps for 7 days of data for better weekly patterns
    start_date = '2024-01-01 00:00:00'
    end_date = '2024-01-07 23:55:00'
    
    # CGM Data (Continuous Glucose Monitoring) - every 5 minutes
    cgm_dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Create realistic glucose patterns with weekly variations
    base_glucose = np.random.normal(120, 15, len(cgm_dates))
    
    # Add meal effects (spikes after typical meal times)
    meal_times = [8, 12, 18, 21]  # 8am, 12pm, 6pm, 9pm
    for hour in meal_times:
        meal_mask = (cgm_dates.hour == hour) & (cgm_dates.minute == 0)
        base_glucose[meal_mask] += np.random.normal(30, 8, meal_mask.sum())
    
    # Add night time decrease
    night_mask = (cgm_dates.hour >= 22) | (cgm_dates.hour <= 6)
    base_glucose[night_mask] -= np.random.normal(15, 5, night_mask.sum())
    
    # Add weekend effect (slightly different patterns)
    weekend_mask = (cgm_dates.dayofweek >= 5)  # Saturday and Sunday
    base_glucose[weekend_mask] += np.random.normal(5, 3, weekend_mask.sum())
    
    # Add some random noise and clip to realistic range
    glucose_values = base_glucose + np.random.normal(0, 5, len(cgm_dates))
    glucose_values = np.clip(glucose_values, 70, 250)
    
    cgm_df = pd.DataFrame({
        'timestamp': cgm_dates,
        'glucose_value': glucose_values,
        'patient_id': 'PATIENT_001'
    })
    
    # Insulin Data - 3-4 doses per day
    insulin_dates = []
    insulin_doses = []
    for day in range(7):
        # Breakfast dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        insulin_doses.append(np.random.uniform(4, 6))
        # Lunch dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        insulin_doses.append(np.random.uniform(3, 5))
        # Dinner dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        insulin_doses.append(np.random.uniform(5, 7))
        # Bedtime dose
        insulin_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 22:00:00'))
        insulin_doses.append(np.random.uniform(2, 4))
    
    insulin_df = pd.DataFrame({
        'timestamp': insulin_dates,
        'insulin_dose': insulin_doses,
        'insulin_type': ['rapid'] * len(insulin_dates),
        'patient_id': 'PATIENT_001'
    })
    
    # Meals Data - 3 meals per day + optional snack
    meal_dates = []
    meal_carbs = []
    meal_types = []
    
    for day in range(7):
        # Breakfast
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 08:00:00'))
        meal_carbs.append(np.random.randint(40, 60))
        meal_types.append('breakfast')
        
        # Lunch
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 12:30:00'))
        meal_carbs.append(np.random.randint(50, 80))
        meal_types.append('lunch')
        
        # Dinner
        meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 18:30:00'))
        meal_carbs.append(np.random.randint(60, 90))
        meal_types.append('dinner')
        
        # Optional snack
        if np.random.random() > 0.5:
            meal_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} 21:00:00'))
            meal_carbs.append(np.random.randint(15, 30))
            meal_types.append('snack')
    
    meals_df = pd.DataFrame({
        'timestamp': meal_dates,
        'carbs': meal_carbs,
        'meal_type': meal_types,
        'patient_id': 'PATIENT_001'
    })
    
    # Exercise Data - random exercise sessions
    exercise_dates = []
    exercise_durations = []
    exercise_intensities = []
    
    for day in range(7):
        if np.random.random() > 0.4:  # 60% chance of exercise each day
            hour = np.random.choice([7, 17, 19])  # Morning or evening
            exercise_dates.append(pd.Timestamp(f'2024-01-{day+1:02d} {hour}:00:00'))
            exercise_durations.append(np.random.randint(20, 60))
            exercise_intensities.append(np.random.choice(['light', 'moderate', 'vigorous']))
    
    exercise_df = pd.DataFrame({
        'timestamp': exercise_dates,
        'duration_minutes': exercise_durations,
        'intensity': exercise_intensities,
        'patient_id': 'PATIENT_001'
    })
    
    print("Sample diabetes data created successfully!")
    return cgm_df, insulin_df, meals_df, exercise_df

# Try to load data files with error handling
try:
    cgm_df = pd.read_csv('cgm_data.csv', parse_dates=['timestamp'])
    print("Loaded CGM data")
except FileNotFoundError:
    print("CGM data file not found")
    cgm_df = pd.DataFrame()

try:
    insulin_df = pd.read_csv('insulin_data.csv', parse_dates=['timestamp'])
    print("Loaded insulin data")
except FileNotFoundError:
    print("Insulin data file not found")
    insulin_df = pd.DataFrame()

try:
    meals_df = pd.read_csv('meals.csv', parse_dates=['timestamp'])
    print("Loaded meals data")
except FileNotFoundError:
    print("Meals data file not found")
    meals_df = pd.DataFrame()

try:
    exercise_df = pd.read_csv('exercise.csv', parse_dates=['timestamp'])
    print("Loaded exercise data")
except FileNotFoundError:
    print("Exercise data file not found")
    exercise_df = pd.DataFrame()

# If any dataframe is empty, create sample data
if cgm_df.empty or insulin_df.empty or meals_df.empty or exercise_df.empty:
    print("\nSome data files are missing. Creating sample data...")
    cgm_df, insulin_df, meals_df, exercise_df = create_sample_diabetes_data()
    
    # Save sample data for future use
    cgm_df.to_csv('cgm_data.csv', index=False)
    insulin_df.to_csv('insulin_data.csv', index=False)
    meals_df.to_csv('meals.csv', index=False)
    exercise_df.to_csv('exercise.csv', index=False)
    print("Sample data saved as CSV files")

# Extract time features for analysis
cgm_df['hour'] = cgm_df['timestamp'].dt.hour
cgm_df['day_of_week'] = cgm_df['timestamp'].dt.day_name()
cgm_df['date'] = cgm_df['timestamp'].dt.date
cgm_df['time_of_day'] = cgm_df['timestamp'].dt.time

# Enhanced feature engineering function
def create_features(cgm_df, insulin_df=None, meals_df=None, exercise_df=None, prediction_horizon=12):
    """
    Create comprehensive features for glucose prediction
    
    Parameters:
    - cgm_df: CGM data with timestamp and glucose_value
    - insulin_df: Insulin administration data
    - meals_df: Meal consumption data
    - exercise_df: Exercise activity data
    - prediction_horizon: Number of 5-minute intervals to predict ahead (default: 12 = 60 minutes)
    """
    
    # Copy the dataframe
    df = cgm_df.copy()
    
    # 1. Rolling Statistics from CGM (Past 30 mins to 3 hours)
    df['glucose_rolling_mean_30m'] = df['glucose_value'].rolling(window=6).mean()   # 6 * 5min = 30min
    df['glucose_rolling_mean_1h'] = df['glucose_value'].rolling(window=12).mean()   # 12 * 5min = 1h
    df['glucose_rolling_mean_2h'] = df['glucose_value'].rolling(window=24).mean()   # 24 * 5min = 2h
    
    df['glucose_rolling_std_30m'] = df['glucose_value'].rolling(window=6).std()     # 6 * 5min = 30min
    df['glucose_rolling_std_1h'] = df['glucose_value'].rolling(window=12).std()     # 12 * 5min = 1h
    
    df['glucose_slope_30m'] = df['glucose_value'].diff(periods=6)                   # 6 * 5min = 30min
    df['glucose_slope_1h'] = df['glucose_value'].diff(periods=12)                   # 12 * 5min = 1h
    df['glucose_slope_2h'] = df['glucose_value'].diff(periods=24)                   # 24 * 5min = 2h
    
    # Rate of change features
    df['glucose_roc_30m'] = df['glucose_slope_30m'] / 30  # mg/dL per minute
    df['glucose_roc_1h'] = df['glucose_slope_1h'] / 60    # mg/dL per minute
    
    # 2. Time Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Time of day categories
    df['time_of_day_category'] = pd.cut(df['hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                       include_lowest=True)
    
    # 3. Event-based Features (Merge other datasets)
    if insulin_df is not None and not insulin_df.empty:
        # Calculate time since last insulin dose
        insulin_events = insulin_df[['timestamp', 'insulin_dose']].copy()
        insulin_events = insulin_events.rename(columns={'timestamp': 'insulin_time'})
        
        # Merge with nearest insulin event
        df = pd.merge_asof(df.sort_values('timestamp'), 
                          insulin_events.sort_values('insulin_time'), 
                          left_on='timestamp', 
                          right_on='insulin_time', 
                          direction='backward')
        
        df['time_since_insulin'] = (df['timestamp'] - df['insulin_time']).dt.total_seconds() / 60  # minutes
        df['insulin_dose'] = df['insulin_dose'].fillna(0)
        
        # Simple Insulin-On-Board (IOB) model (exponential decay)
        insulin_half_life = 75  # minutes
        df['iob'] = df['insulin_dose'] * np.exp(-df['time_since_insulin'] / insulin_half_life)
    
    if meals_df is not None and not meals_df.empty:
        # Calculate time since last meal and carbs consumed
        meal_events = meals_df[['timestamp', 'carbs']].copy()
        meal_events = meal_events.rename(columns={'timestamp': 'meal_time'})
        
        # Merge with nearest meal event
        df = pd.merge_asof(df.sort_values('timestamp'), 
                          meal_events.sort_values('meal_time'), 
                          left_on='timestamp', 
                          right_on='meal_time', 
                          direction='backward')
        
        df['time_since_meal'] = (df['timestamp'] - df['meal_time']).dt.total_seconds() / 60  # minutes
        df['carbs'] = df['carbs'].fillna(0)
        
        # Simple Carbs-On-Board (COB) model (exponential decay)
        carb_absorption_half_life = 120  # minutes
        df['cob'] = df['carbs'] * np.exp(-df['time_since_meal'] / carb_absorption_half_life)
    
    if exercise_df is not None and not exercise_df.empty:
        # Calculate time since last exercise
        exercise_events = exercise_df[['timestamp', 'duration_minutes', 'intensity']].copy()
        exercise_events = exercise_events.rename(columns={'timestamp': 'exercise_time'})
        
        # Merge with nearest exercise event
        df = pd.merge_asof(df.sort_values('timestamp'), 
                          exercise_events.sort_values('exercise_time'), 
                          left_on='timestamp', 
                          right_on='exercise_time', 
                          direction='backward')
        
        df['time_since_exercise'] = (df['timestamp'] - df['exercise_time']).dt.total_seconds() / 60  # minutes
        df['duration_minutes'] = df['duration_minutes'].fillna(0)
        df['intensity'] = df['intensity'].fillna('none')
        
        # Exercise impact score (simplified)
        intensity_map = {'none': 0, 'light': 1, 'moderate': 2, 'vigorous': 3}
        df['exercise_intensity_score'] = df['intensity'].map(intensity_map)
        df['exercise_impact'] = df['duration_minutes'] * df['exercise_intensity_score'] * \
                               np.exp(-df['time_since_exercise'] / 120)  # 2-hour half-life
    
    # 4. Glucose variability features
    df['glucose_momentum'] = df['glucose_value'] - df['glucose_rolling_mean_1h']
    df['glucose_acceleration'] = df['glucose_slope_1h'] - df['glucose_slope_1h'].shift(1)
    
    # 5. Create the prediction target: Glucose value at prediction horizon
    df['glucose_future'] = df['glucose_value'].shift(-prediction_horizon)
    
    # 6. Binary classification targets
    df['hypo_risk_60min'] = (df['glucose_future'] < 70).astype(int)  # Hypoglycemia risk
    df['hyper_risk_60min'] = (df['glucose_future'] > 180).astype(int)  # Hyperglycemia risk
    
    # Drop rows with NaN values created by rolling/shifting
    df = df.dropna()
    
    # Drop intermediate columns used for calculations
    columns_to_drop = ['date', 'time_of_day', 'insulin_time', 'meal_time', 'exercise_time']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df

# Apply the feature engineering function
print("Creating features...")
featured_df = create_features(cgm_df, insulin_df, meals_df, exercise_df, prediction_horizon=12)

# Prepare data for hypoglycemia prediction
X = featured_df.drop(columns=['hypo_risk_60min', 'hyper_risk_60min', 'glucose_future', 
                             'timestamp', 'patient_id', 'day_of_week', 'time_of_day_category'], errors='ignore')

# Select only numerical features
X = X.select_dtypes(include=[np.number])
y = featured_df['hypo_risk_60min']

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")
print(f"Hypoglycemia prevalence: {y.mean()*100:.2f}%")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}, Hypo cases: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Test set: {X_test.shape}, Hypo cases: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# Handle class imbalance by giving higher weight to the rare class (hypo)
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
print(f"\nScale positive weight: {scale_pos_weight:.2f}")

# Train XGBoost model for hypoglycemia prediction
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # Critical for imbalanced data
    random_state=42,
    eval_metric='logloss',
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0
)

print("\nTraining XGBoost model for hypoglycemia prediction...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\n" + "="*60)
print("MODEL EVALUATION - Hypoglycemia Prediction")
print("="*60)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Hypo', 'Hypo']))

# Confusion Matrix
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Hypo', 'Hypo'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')

# ROC Curve
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, 
         label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Additional metrics
print(f"\nAdditional Metrics:")
print(f"ROC AUC Score: {roc_auc:.3f}")
print(f"Average Precision: {avg_precision:.3f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")

# Predict on training set for overfitting check
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)[:, 1]

train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
train_avg_precision = average_precision_score(y_train, y_train_pred_proba)

print(f"\nTraining Performance:")
print(f"Training ROC AUC: {train_roc_auc:.3f}")
print(f"Training Average Precision: {train_avg_precision:.3f}")

# Check for overfitting
if abs(roc_auc - train_roc_auc) > 0.1:
    print("âš ï¸�  Warning: Possible overfitting detected!")
else:
    print("âœ… Model generalization appears good")

print("\nModel training and evaluation complete! ğŸ�¯")


import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import io

# Step 1: Create the workflow CSV template
def create_workflow_csv():
    workflow_data = {
        "Step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "Task": ["Register for the Challenge",
            "Attend Kick-off Meeting",
            "Explore Datasets",
            "Submit LOI",
            "Attend Data Bootcamps",
            "Develop Hypothesis",
            "Office Hours",
            "Prepare Models and Papers",
            "Submit Models and Papers",
            "Finalist Announcement",
            "Pitch Polish",
            "Live Pitch",
            "Awards"
        ],
        "Description": [
            "Complete team registration on the D-Challenge platform.",
            "Join the kick-off meeting to understand the challenge scope and resources.",
            "Identify and download relevant datasets from dkNET, T1DKP, and other resources.",
            "Submit a non-binding Letter of Intent (LOI) to confirm participation.",
            "Participate in bootcamps to learn about tools and data analysis techniques.",
            "Analyze datasets and develop a novel hypothesis about T1D.",
            "Attend office hours for guidance and feedback on your hypothesis.",
            "Document your hypothesis, methodology, and findings in a structured format.",
            "Upload your models and papers to the D-Challenge portal.",
            "Wait for the announcement of finalists.",
            "Refine your pitch with feedback from the JDRF-T1D Fund.",
            "Present your hypothesis to the judges and audience.",
            "Celebrate the winners and network with participants."
        ],
        "Tools/Resources": [
            "Registration Link",
            "Zoom/Webinar Link",
            "dkNET, T1DKP, Appyters",
            "D-Challenge Portal",
            "dkNET, T1DKP",
            "Python, R, Bioinformatics Tools",
            "Zoom/Webinar Link",
            "LaTeX, Word, Google Docs",
            "D-Challenge Portal",
            "D-Challenge Portal",
            "Zoom/Webinar Link",
            "D-Challenge Portal",
            "D-Challenge Portal"
        ],
        "Start Date": [
            "2025-08-19", "2025-09-19", "2025-09-20", "2025-09-26",
            "2025-09-26", "2025-10-18", "2025-11-10", "2025-11-10",
            "2025-11-14", "2025-11-21", "2025-12-05", "2025-12-12",
            "2025-12-15"
        ],
        "End Date": [
            "2025-09-12", "2025-09-19", "2025-09-26", "2025-09-26",
            "2025-10-17", "2025-11-10", "2025-11-24", "2025-11-14",
            "2025-11-14", "2025-11-21", "2025-12-05", "2025-12-12",
            "2025-12-15"
        ],
        "Status": [
            "Completed", "Pending", "In Progress", "Pending", "Pending",
            "Not Started", "Pending", "Not Started", "Pending", "Pending",
            "Pending", "Pending", "Pending"
        ],
        "Notes": ["", "", "", "", "", "", "", "", "", "", "", "", ""]
    }

    df = pd.DataFrame(workflow_data)
    return df

# Step 2: Save the workflow to a CSV file
def save_workflow_csv(df, filename="T1D-challenge_workflow.csv"):
    df.to_csv(filename, index=False)
    print(f"Workflow saved to {filename}")

# Step 3: Load the workflow from CSV
def load_workflow_csv(filename="T1D-challenge_workflow.csv"):
    df = pd.read_csv(filename)
    return df

# Step 4: Update the status of a task
def update_task_status(df, step, status, notes=""):
    df.loc[df["Step"] == step, "Status"] = status
    df.loc[df["Step"] == step, "Notes"] = notes
    return df

# Step 5: Visualize the workflow timeline
def visualize_workflow(df):
    df["Start Date"] = pd.to_datetime(df["Start Date"])
    df["End Date"] = pd.to_datetime(df["End Date"])

    plt.figure(figsize=(12, 6))
    for i, row in df.iterrows():
        plt.plot(
            [row["Start Date"], row["End Date"]],
            [row["Step"], row["Step"]],
            marker="o",
            label=f"Step {row['Step']}: {row['Task']}"
        )
    plt.yticks(df["Step"], df["Task"])
    plt.xlabel("Date")
    plt.ylabel("Task")
    plt.title("T1D-Challenge 2025 Workflow Timeline")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Execute the pipeline
if __name__ == "__main__":
    # Create the workflow
    workflow_df = create_workflow_csv()

    # Save the workflow to CSV
    save_workflow_csv(workflow_df)

    # Load the workflow from CSV
    loaded_df = load_workflow_csv()

    # Update the status of a task (example: Step 3)
    updated_df = update_task_status(loaded_df, step=3, status="In Progress", notes="Downloaded datasets from dkNET")

    # Visualize the workflow
    visualize_workflow(updated_df)

    # Display the workflow
    display(updated_df)


