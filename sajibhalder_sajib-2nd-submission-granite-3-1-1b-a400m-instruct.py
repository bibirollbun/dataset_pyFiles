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


import numpy as np
import pandas as pd

# Parameters
L = 1.0  # Length of the domain
N = 40  # Number of grid points (adjusted for max 10k rows dataset size)
alpha = 0.01  # Thermal conductivity
T_max = 1.0  # Maximum time
dt = 0.001  # Time step
dx = dy = L / N  # Grid spacing

# Discretized spatial domain
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Initialize temperature field
T = np.zeros((N, N))

# Define the source term f(x, y)
def source_term(x, y):
    return 8 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

# Time loop for solving the heat equation
def solve_heat_equation_case_1(T, alpha, dt, dx, dy, T_max, max_rows=10000):
    num_time_steps = int(T_max / dt)
    dataset = []
    
    # Apply boundary conditions
    for i in range(N):
        for j in range(N):
            if i == 0 or i == N-1 or j == 0 or j == N-1:
                T[i, j] = 0  # Boundary conditions set to 0 as per the problem
    
    time_steps_taken = 0  # To ensure dataset doesn't exceed max_rows
    
    # Time-stepping loop
    for t in range(num_time_steps):
        T_new = T.copy()
        
        # Interior points update (finite difference method)
        for i in range(1, N-1):
            for j in range(1, N-1):
                T_new[i, j] = T[i, j] + alpha * dt * (
                    (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                    (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2 +
                    source_term(x[i], y[j])
                )
        
        T = T_new
        
        # Save data for each time step, but limit dataset size
        for i in range(N):
            for j in range(N):
                if len(dataset) < max_rows:
                    dataset.append([x[i], y[j], T[i, j], t*dt])
                if len(dataset) >= max_rows:
                    break
            if len(dataset) >= max_rows:
                break
        if len(dataset) >= max_rows:
            break
    
    return pd.DataFrame(dataset, columns=['x', 'y', 'z', 'Temperature'])

# Solve for Case 1
dataset_case_1 = solve_heat_equation_case_1(T=T, alpha=alpha, dt=dt, dx=dx, dy=dy, T_max=T_max)


# Save as CSV
csv_file_path = "heat_eq_case_1.csv"
dataset_case_1.to_csv(csv_file_path, index=False)


import numpy as np
import pandas as pd

# Parameters
L = 1.0  # Length of the domain
N = 40  # Number of grid points (adjusted for max 10k rows dataset size)
alpha = 0.01  # Thermal conductivity (this doesn't affect Case 2 as equation is homogeneous)
T_max = 1.0  # Maximum time
dt = 0.001  # Time step
dx = dy = L / N  # Grid spacing

# Discretized spatial domain
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Initialize temperature field
T = np.zeros((N, N))

# Boundary conditions for Case 2
def boundary_conditions(x, y):
    # T(x,0) = 0, T(x,1) = sin(πx), T(0,y) = 0, T(1,y) = 0
    if y == 0 or y == 1:
        return 0
    if x == 0 or x == 1:
        return 0
    return np.sin(np.pi * x)  # For y=1, the boundary condition is sin(πx)

# Time loop for solving the heat equation
def solve_heat_equation_case_2(T, alpha, dt, dx, dy, T_max, max_rows=10000):
    num_time_steps = int(T_max / dt)
    dataset = []
    
    # Apply initial and boundary conditions
    for i in range(N):
        for j in range(N):
            T[i, j] = boundary_conditions(x[i], y[j])
    
    time_steps_taken = 0  # To ensure dataset doesn't exceed max_rows
    
    # Time-stepping loop
    for t in range(num_time_steps):
        T_new = T.copy()
        
        # Interior points update (finite difference method)
        for i in range(1, N-1):
            for j in range(1, N-1):
                T_new[i, j] = T[i, j] + alpha * dt * (
                    (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                    (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
                )
        
        T = T_new
        
        # Save data for each time step, but limit dataset size
        for i in range(N):
            for j in range(N):
                if len(dataset) < max_rows:
                    dataset.append([x[i], y[j], T[i, j], t*dt])
                if len(dataset) >= max_rows:
                    break
            if len(dataset) >= max_rows:
                break
        if len(dataset) >= max_rows:
            break
    
    return pd.DataFrame(dataset, columns=['x', 'y', 'z', 'Temperature'])

# Solve for Case 2
dataset_case_2 = solve_heat_equation_case_2(T=T, alpha=alpha, dt=dt, dx=dx, dy=dy, T_max=T_max)


# Save as CSV
csv_file_path = "heat_eq_case_2.csv"
dataset_case_2.to_csv(csv_file_path, index=False)


import numpy as np
import pandas as pd

# Parameters
L = 1.0  # Length of the domain
N = 40  # Number of grid points (adjusted for max 10k rows dataset size)
alpha = 0.01  # Thermal conductivity (not required here as it's a steady-state equation)
T_max = 1.0  # Maximum time
dt = 0.001  # Time step
dx = dy = L / N  # Grid spacing

# Discretized spatial domain
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Initialize temperature field
T = np.zeros((N, N))

# Boundary conditions for Case 3
def boundary_conditions(x, y):
    if x == 0:
        return 0  # T(0, y) = 0
    if x == 1:
        return y * (1 - y)  # T(1, y) = y(1 - y)
    if y == 0 or y == 1:
        return 0  # T(x,0) = 0 and T(x,1) = 0
    return None  # Interior points

# Apply boundary conditions
for i in range(N):
    for j in range(N):
        bc_value = boundary_conditions(x[i], y[j])
        if bc_value is not None:
            T[i, j] = bc_value

# Solve the heat equation using the finite difference method
def solve_heat_equation_case_3(T, alpha, dt, dx, dy, T_max, max_rows=10000):
    num_time_steps = int(T_max / dt)
    dataset = []

    time_steps_taken = 0  # To ensure dataset doesn't exceed max_rows

    for t in range(num_time_steps):
        T_new = T.copy()

        # Interior points update (finite difference method)
        for i in range(1, N-1):
            for j in range(1, N-1):
                T_new[i, j] = T[i, j] + alpha * dt * (
                    (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                    (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
                )

        T = T_new

        # Save data for each time step, but limit dataset size
        for i in range(N):
            for j in range(N):
                if len(dataset) < max_rows:
                    dataset.append([x[i], y[j], T[i, j], t*dt])
                if len(dataset) >= max_rows:
                    break
            if len(dataset) >= max_rows:
                break
        if len(dataset) >= max_rows:
            break

    return pd.DataFrame(dataset, columns=['x', 'y', 'z', 'Temperature'])

# Solve for Case 3
dataset_case_3 = solve_heat_equation_case_3(T=T, alpha=alpha, dt=dt, dx=dx, dy=dy, T_max=T_max)

# Save as CSV
csv_file_path = "heat_eq_case_3.csv"
dataset_case_3.to_csv(csv_file_path, index=False)



import numpy as np
import pandas as pd

# Parameters
L = 1.0  # Length of the domain
N = 40  # Number of grid points (adjusted for max 10k rows dataset size)
alpha = 0.01  # Thermal diffusivity
T_max = 1.0  # Maximum simulation time
dt = 0.001  # Time step
dx = dy = L / N  # Grid spacing

# Discretized spatial domain
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Initialize temperature field with initial condition T(x,y,0) = sin(πx)sin(πy)
T = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Boundary conditions for Case 4
def apply_boundary_conditions(T):
    T[0, :] = 0  # T(0,y,t) = 0
    T[-1, :] = 0  # T(1,y,t) = 0
    T[:, 0] = 0  # T(x,0,t) = 0
    T[:, -1] = 0  # T(x,1,t) = 0
    return T

# Solve the heat equation using the finite difference method
def solve_heat_equation_case_4(T, alpha, dt, dx, dy, T_max, max_rows=10000):
    num_time_steps = int(T_max / dt)
    dataset = []

    for t in range(num_time_steps):
        T_new = T.copy()

        # Interior points update (explicit finite difference method)
        for i in range(1, N-1):
            for j in range(1, N-1):
                T_new[i, j] = T[i, j] + alpha * dt * (
                    (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                    (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
                )

        T = apply_boundary_conditions(T_new)  # Apply boundary conditions

        # Save data for each time step, but limit dataset size
        for i in range(N):
            for j in range(N):
                if len(dataset) < max_rows:
                    dataset.append([x[i], y[j], T[i, j], t * dt])
                if len(dataset) >= max_rows:
                    break
            if len(dataset) >= max_rows:
                break
        if len(dataset) >= max_rows:
            break

    return pd.DataFrame(dataset, columns=['x', 'y', 'z', 'Temperature'])

# Solve for Case 4
dataset_case_4 = solve_heat_equation_case_4(T=T, alpha=alpha, dt=dt, dx=dx, dy=dy, T_max=T_max)

# Save to CSV
csv_file_path = "heat_eq_case_4.csv"
dataset_case_4.to_csv(csv_file_path, index=False)


csv_files = ["/kaggle/working/heat_eq_case_1.csv",
                "/kaggle/working/heat_eq_case_2.csv",
                "/kaggle/working/heat_eq_case_3.csv",
                "/kaggle/working/heat_eq_case_4.csv"]
# Merge all CSV files
final_df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True)

# Save the final merged dataset
final_csv_path = "/kaggle/working/final_heat_eq.csv"
final_df.to_csv(final_csv_path, index=False)

print(f"Final dataset saved: {final_csv_path}")


import pandas as pd

# Load the dataset
data_path = "/kaggle/working/final_heat_eq.csv"  # Update with your dataset name
df = pd.read_csv(data_path)

# Display the first few rows
print(df.head(3))


# Create LLM fine-tuning dataset
df["prompt"] = df.apply(lambda row: 
    f"What is the temperature at point ({row['x']}, {row['y']}, {row['z']})?", axis=1)
df["response"] = df["Temperature"].apply(lambda temp: f"The temperature is {temp}.")

# Save the fine-tuning dataset
df[["prompt", "response"]].to_csv("heat_eq_finetune.csv", index=False)
print("Fine-tuning CSV saved: heat_eq_finetune.csv")


import pandas as pd

# Load the dataset
data_path = "/kaggle/working/heat_eq_finetune.csv"  # Update with your dataset name
df = pd.read_csv(data_path)

# Display the first few rows
print(df.head(3))


!pip install pyvista


import pyvista as pv

# Load the VTK file
vtk_file_path = "/kaggle/input/fine-tuning-lm-physical-interpretation-hackathon/Case1.vtk"
mesh = pv.read(vtk_file_path)

# Print information about the mesh
print(mesh)


import numpy as np

# Convert VTK points to NumPy array
points = np.array(mesh.points)

# Convert cell data to NumPy
cell_data = mesh.cell_data
point_data = mesh.point_data

print("Points:\n", points)
print("Point Data:\n", point_data)


import pyvista as pv
import pandas as pd
import numpy as np


vtk_file_path = "/kaggle/input/fine-tuning-lm-physical-interpretation-hackathon/Case1.vtk"
mesh = pv.read(vtk_file_path)

# Extract Points (x, y, z)
points = np.array(mesh.points)

# Extract Scalars (Point Data)
point_data = {name: mesh.point_data[name] for name in mesh.point_data.keys()}

# Convert to DataFrame
df = pd.DataFrame(points, columns=["x", "y", "z"])

# Add scalar fields
for name, values in point_data.items():
    df[name] = values

# Save as CSV
csv_file_path = "Case1vtk_extracted_data.csv"
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved: {csv_file_path}")


# Load the VTK file
vtk_file_path = "/kaggle/input/fine-tuning-lm-physical-interpretation-hackathon/Case2.vtk"
mesh = pv.read(vtk_file_path)
# Extract Points (x, y, z)
points = np.array(mesh.points)

# Extract Scalars (Point Data)
point_data = {name: mesh.point_data[name] for name in mesh.point_data.keys()}

# Convert to DataFrame
df = pd.DataFrame(points, columns=["x", "y", "z"])

# Add scalar fields
for name, values in point_data.items():
    df[name] = values

# Save as CSV
csv_file_path = "Case2vtk_extracted_data.csv"
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved: {csv_file_path}")


# Load the VTK file
vtk_file_path = "/kaggle/input/fine-tuning-lm-physical-interpretation-hackathon/Case3.vtk"
mesh = pv.read(vtk_file_path)
# Extract Points (x, y, z)
points = np.array(mesh.points)

# Extract Scalars (Point Data)
point_data = {name: mesh.point_data[name] for name in mesh.point_data.keys()}

# Convert to DataFrame
df = pd.DataFrame(points, columns=["x", "y", "z"])

# Add scalar fields
for name, values in point_data.items():
    df[name] = values

# Save as CSV
csv_file_path = "Case3vtk_extracted_data.csv"
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved: {csv_file_path}")


# Load the VTK file
vtk_file_path = "/kaggle/input/fine-tuning-lm-physical-interpretation-hackathon/Case4.vtk"
mesh = pv.read(vtk_file_path)
# Extract Points (x, y, z)
points = np.array(mesh.points)

# Extract Scalars (Point Data)
point_data = {name: mesh.point_data[name] for name in mesh.point_data.keys()}

# Convert to DataFrame
df = pd.DataFrame(points, columns=["x", "y", "z"])

# Add scalar fields
for name, values in point_data.items():
    df[name] = values

# Save as CSV
csv_file_path = "Case4vtk_extracted_data.csv"
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved: {csv_file_path}")


csv_files = ["/kaggle/working/Case1vtk_extracted_data.csv",
                "/kaggle/working/Case2vtk_extracted_data.csv",
                "/kaggle/working/Case3vtk_extracted_data.csv",
                "/kaggle/working/Case4vtk_extracted_data.csv"]
# Merge all CSV files
final_df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True)

# Save the final merged dataset
final_csv_path = "/kaggle/working/final_vtk_dataset.csv"
final_df.to_csv(final_csv_path, index=False)

print(f"Final dataset saved: {final_csv_path}")


import pandas as pd

# Load the dataset
data_path = "/kaggle/working/final_vtk_dataset.csv"  # Update with your dataset name
df = pd.read_csv(data_path)

# Display the first few rows
print(df.head(3))


print(df.tail(3))


# Create LLM fine-tuning dataset
df["prompt"] = df.apply(lambda row: 
    f"What is the temperature at point ({row['x']}, {row['y']}, {row['z']})?", axis=1)
df["response"] = df["Temperature"].apply(lambda temp: f"The temperature is {temp}.")

# Save the fine-tuning dataset
df[["prompt", "response"]].to_csv("heat_equation_finetune.csv", index=False)
print("Fine-tuning CSV saved: heat_equation_finetune.csv")


import pandas as pd

# Load the dataset
data_path = "/kaggle/working/heat_equation_finetune.csv"  # Update with your dataset name
df = pd.read_csv(data_path)

# Display the first few rows
print(df.head(3))


# consider only Case1 VTK file data:
def solve_heat_equation_case1():
    nx, ny = 50, 50  
    dx, dy = 1 / (nx - 1), 1 / (ny - 1) 
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    T = np.zeros((ny, nx))

    T[0, :] = 0  
    T[-1, :] = 0 
    T[:, 0] = 0  
    T[:, -1] = 0 

    def force_function(x, y):
        return 8 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    for _ in range(1000):  
        T_new = T.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                T_new[j, i] = 0.25 * (T[j + 1, i] + T[j - 1, i] + T[j, i + 1] + T[j, i - 1] - dx**2 * force_function(x[i], y[j]))
        T = T_new

    grid = pv.StructuredGrid(X, Y, np.zeros_like(X))
    grid["Temperature"] = T.ravel()
    grid.save("Case1_FDM.vtk")

solve_heat_equation_case1()

grid = pv.read("Case1_FDM.vtk")

# Extract data
points = grid.points
temperature = grid["Temperature"]

data = pd.DataFrame({
    "x": points[:, 0],
    "y": points[:, 1],
    "temperature": temperature
})

data.to_csv("Case1_FDM.csv", index=False)


!pip install transformers==4.46.0 datasets peft bitsandbytes accelerate torch


# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from peft import get_peft_model, TaskType
from peft import LoraConfig, get_peft_model  # LoRA fine-tuning
import torch
import gc
import os
from accelerate import Accelerator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
accelerator = Accelerator(cpu=True)


# Define model path
model_path = "ibm-granite/granite-3.1-1b-a400m-instruct"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to avoid errors


# Load model with 4-bit quantization to save memory
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             device_map="auto", 
                                             load_in_4bit=True)


model.gradient_checkpointing_enable()


# Apply LoRA (Low-Rank Adaptation) to speed up fine-tuning
lora_config = LoraConfig(
    r=4,  # Low-rank adaptation size
    lora_alpha=8,  # taking Scaling factor=8
    target_modules=["q_proj", "v_proj"],  # LoRA applied to attention layers
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM")



model = get_peft_model(model, lora_config)


# solution for case1 data
data = pd.read_csv("Case1_FDM.csv")
dataset = Dataset.from_pandas(data)


# Tokenization function
def preprocess_function(examples):
    examples["temperature"] = [str(temp) for temp in examples["temperature"]]
    tokenized_inputs = tokenizer(examples["temperature"], truncation=True, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

    return tokenized_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)


torch.cuda.empty_cache()
gc.collect()


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Shuffle once and split into 80% train / 20% eval
split_idx = int(0.8 * len(tokenized_dataset))
train_data = tokenized_dataset.shuffle(seed=42).select(range(split_idx))


eval_data = tokenized_dataset.select(range(split_idx, len(tokenized_dataset)))


#  Training arguments optimized for speed
training_args = TrainingArguments(
    output_dir="./granite_finetuned",
    run_name="granite_experiment",
    per_device_train_batch_size=4,  # Increase batch size if GPU allows
    gradient_accumulation_steps=4,  #  Simulates larger batch size
    #per_device_eval_batch_size=8,
    num_train_epochs=3,  # Reduce training time
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,   #  Mixed precision training (Faster training)
    save_strategy="epoch", # Save model every epoch 
    save_steps=500,  # Save every 500 steps instead of 100
    save_total_limit=2,  # Keep only the last 2 checkpoints
    logging_dir="./logs",  
    #logging_steps=10,# Log every 10 steps
)


from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Speeds up inference/training
    bnb_4bit_quant_type="nf4"
)


# Initialize Trainer with optimized settings
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator,
    #eval_dataset=eval_data
)


model, trainer = accelerator.prepare(model, trainer)


#  Train the model (Now much faster!)
trainer.train()

