# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import time
import os
import gc

import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime



# Define the paths
OUTPUT_EVALUATION_IMG_PATH = "/kaggle/working/evaluation"
OUTPUT_TRAINING_IMG_PATH = "/kaggle/working/training"

# Create the directories if they don't exist
os.makedirs(OUTPUT_EVALUATION_IMG_PATH, exist_ok=True)
os.makedirs(OUTPUT_TRAINING_IMG_PATH, exist_ok=True)



OUTPUT_EVALUATION_CSV_PATH = "/kaggle/working/arc_evaluation.csv"
OUTPUT_TRAINING_CSV_PATH = "/kaggle/working/arc_training.csv"




class LogWriter:
    def __init__(self, log_path="/kaggle/working/arc_log_file.txt"):
        self.log_path = log_path
        self.log_file = open(self.log_path, "a", encoding="utf-8")

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"[{timestamp}] {message}\n")

    def close(self):
        if self.log_file:
            self.log_file.close()



class ReadAndProcessJson:
    """
    A class to read a JSON file and process its contents.
    This class is specifically designed to handle the ARC challenge dataset,
    reading the data and then visualizing the grids.
    """
    def __init__(self, file_path,OUTPUT_CSV_PATH, OUTPUT_IMG_PATH):
        """
        Initializes the class with the path to the JSON file and
        sets up the visualization parameters.

        Args:
            file_path (str or Path): The path to the ARC challenge JSON file.
        """
        self.logger = LogWriter(Path("/kaggle/working/arc_log_file.txt"))

        self.file_path = Path(file_path)
        self.OUTPUT_CSV_PATH = OUTPUT_CSV_PATH
        self.OUTPUT_IMG_PATH = OUTPUT_IMG_PATH
        # Define a color map for the ARC tasks.
        # The ARC dataset uses numbers 0-9 to represent colors.
        self.cmap = mcolors.ListedColormap([
            '#000000',  # 0: Black
            '#0074D9',  # 1: Blue
            '#FF4136',  # 2: Red
            '#2ECC40',  # 3: Green
            '#FFDC00',  # 4: Yellow
            '#AAAAAA',  # 5: Grey
            '#F012BE',  # 6: Magenta
            '#FF851B',  # 7: Orange
            '#7FDBFF',  # 8: Sky blue
            '#8B0000'   # 9: Dark red
        ])

        # A color normalization object to map the number range to the color map.
        self.norm = mcolors.Normalize(vmin=0, vmax=9)

    def _read_json(self):
        """
        Private method to read and parse the JSON file from the given path.
        Handles file-related errors gracefully.

        Returns:
            dict: The loaded challenges data if successful, otherwise None.
        """
        try:
            with open(self.file_path, 'r') as f:
                challenges = json.load(f)
                self.logger.log(f"[INFO ] Successfully loaded {len(challenges)} challenges.")
                return challenges
        except FileNotFoundError:
            self.logger.log(f"[ERROR ] Error: The file '{self.file_path}' was not found.")
            return None
        except json.JSONDecodeError:
            self.logger.log(f"[ERROR ] Error: Could not decode JSON from the file '{self.file_path}'.")
            return None
        except Exception as e:
            self.logger.log(f"[ERROR ] An unexpected error occurred during file reading: {e}")
            return None

    def _print_grids(self, challenges):
            """
            Private method to iterate through the challenges and print all I/O grids
    
            Args:
                challenges (dict): The dictionary containing ARC challenge data.
            Result:
                save solutions_df to csv file at OUTPUT_CSV_PATH with index=False
            """
            if challenges is None:
                self.logger.log(f"[ERROR ] - ARC Challenge  - No challenges found.")
                return None

            solutions_df = pd.DataFrame(columns=['id','task_type','input','expected'])
            with tqdm(total=len(challenges.items()), dynamic_ncols=True, leave=True) as pbar:
                for task_id, task_data in challenges.items():
                    #print(f"\n--- Processing Challenge: {task_id} ---")
                    
                    if 'train' in task_data and task_data['train']:
                        train_examples = task_data['train']
                        #display(f"ARC Challenge: {task_id}")
        
                        for i, example in enumerate(train_examples):
                            #display(f"Taks ID: {task_id} - GRID SET NUMBER:  {i}")
                            #display(f"INPUT: Type:{type(example['input'])}")# Data:\n{example['input']}")
                            #display(f"OUTPUT: Type:{type(example['output'])}")# Data:\n{example['output']}")
                            # Append the results to the DataFrame
                            msg = f"[OK] Processing Task {task_id} - [PROCESSED]"
                            self.logger.log(msg)
                            pbar.set_description(msg)
                            new_row = pd.DataFrame([{
                                'id': task_id,
                                'task_type': " ", #parsed_data['Classification'],
                                'input': example['input'], 
                                'expected': example['output'],                    
                            }]) 
                            solutions_df = pd.concat([solutions_df, new_row], ignore_index=True)
    
                    else:
                        msg = f"[MISSING] No training data found for challenge {task_id}."
                        self.logger.log(msg)
                        pbar.set_description(msg)
                        
                solutions_df.to_csv(self.OUTPUT_CSV_PATH, index=False)
                self.logger.log(f"[INFO ] - Dataframe saved to csv file at: {self.OUTPUT_CSV_PATH}")
                
    def _draw_images(self, challenges):
        """
        Private method to iterate through the challenges and visualize
        the input and output grids using matplotlib.
    
        Args:
            challenges (dict): The dictionary containing ARC challenge data.
        """
        self.logger.log(f"[INFO ] - ARC Challenge  - Start drawing images.")
        if challenges is None:
            self.logger.log(f"[ERROR ] - ARC Challenge  - No challenges found.")
            return None
    
        for task_id, task_data in challenges.items():
            if 'train' in task_data and task_data['train']:
                train_examples = task_data['train']
                num_examples = len(train_examples)
    
                # Each row contains 4 examples (8 grids: input-output pairs)
                num_rows = (num_examples + 3) // 4
                fig, axes = plt.subplots(num_rows, 8, figsize=(16, 3 * num_rows))
    
                # Ensure axes is always 2D
                if num_rows == 1:
                    axes = [axes]
    
                fig.suptitle(f"ARC Challenge: {task_id}", fontsize=16)
    
                for idx in range(num_rows * 4):
                    row = idx // 4
                    col = (idx % 4) * 2  # 0, 2, 4, 6
    
                    if idx < num_examples:
                        example = train_examples[idx]
                        input_grid = example['input']
                        output_grid = example['output']
                    else:
                        # Create gray canvas for missing examples
                        input_grid = np.full((3, 3), 7)  # Adjust size if needed
                        output_grid = np.full((3, 3), 7)
    
                    # Plot input grid
                    ax_input = axes[row][col]
                    ax_input.imshow(input_grid, cmap=self.cmap, norm=self.norm)
                    ax_input.set_title(f"Example {idx+1} - Input")
                    ax_input.axis('off')
    
                    # Plot output grid
                    ax_output = axes[row][col + 1]
                    ax_output.imshow(output_grid, cmap=self.cmap, norm=self.norm)
                    ax_output.set_title(f"Example {idx+1} - Output")
                    ax_output.axis('off')
    
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
                #
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.OUTPUT_IMG_PATH}/{task_id}_{timestamp}_arc_class_img.png"
                fig.savefig(filename, dpi=300)
                self.logger.log(f"[INFO ] - ARC Challenge figure saved to png file at: {filename}")
                plt.close(fig)

            else:
                msg = f"[PLOT * MISSING] No training data found for challenge {task_id}."
                self.logger.log(msg)
                pbar.set_description(msg)

    

    def process(self):
        """
        Main public method to execute the entire visualization pipeline.
        It orchestrates the data reading and image drawing steps.
        """
        challenges_data = self._read_json()
        self._print_grids(challenges_data)
        self._draw_images(challenges_data)

    def __del__(self):
        self.logger.close()




# --- Main execution block ---

# --- Main execution block ---
# Define the specific JSON files to process.
file_path_1 =  Path("/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json")
file_path_2 = Path("/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json")

display(f"Processing Evaluation file: {file_path_1}")
pipeline = ReadAndProcessJson(file_path_1, OUTPUT_EVALUATION_CSV_PATH, OUTPUT_EVALUATION_IMG_PATH)
try:
    pipeline.process()
finally:
    # Explicitly close the log file handler to ensure resources are released.
    pipeline.logger.close()
    
del pipeline
gc.collect()



display(f"Processing Training file: {file_path_2}")

pipeline = ReadAndProcessJson(file_path_2, OUTPUT_TRAINING_CSV_PATH, OUTPUT_TRAINING_IMG_PATH)
try:
    pipeline.process()
finally:
    # Explicitly close the log file handler to ensure resources are released.
    pipeline.logger.close()
    
del pipeline
gc.collect()

