#Remove conflicting packages from the Kaggle base environment.Â¶
!pip uninstall -qqy kfp jupyterlab libpysal thinc spacy fastai ydata-profiling google-cloud-bigquery google-generativeai
!pip install -qU 'langgraph==0.3.21' 'langchain-google-genai==2.1.2' 'langgraph-prebuilt==0.1.7'


import numpy as np 
import pandas as pd 
import os
import operator
import re # Import regex for parsing
import json
import io
import sys



from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # Import AIMessage
# from langchain_openai import ChatOpenAI # Replace with your desired LLM provider
from langgraph.graph import StateGraph, END
# Kaggle and Google AI
from kaggle_secrets import UserSecretsClient
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types
from google.api_core import retry

# IPython Display
from IPython.display import Markdown, Image, display

# PDF Processing
import pypdf

# ChromaDB
#import chromadb
#from chromadb import Documents, EmbeddingFunction, Embeddings


from typing import TypedDict, Annotated, Optional, Literal, List, Dict, Any
from typing_extensions import TypedDict # 
from langchain_core.messages import BaseMessage 
from contextlib import redirect_stdout

# Langchain and Langgraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # Ensure these are imported


# Pretty Print
from pprint import pprint
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GOOGLE_API_KEY")
train, test, submission = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv'),pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv'),pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


# Using simple dicts for messages initially for clarity, but BaseMessage[] is better practice# Example: messages: Annotated[list[BaseMessage], add_messages]
# Using simple dicts for messages initially for clarity, but BaseMessage[] is better practice# Example: messages: Annotated[list[BaseMessage], add_messages]
# --- State Definition ---
class GraphState(TypedDict):
    """
    ğŸ—‚ï¸� Central state definition for the CALOR-IA multi-agent workflow.
    Reflects using DataFrame variable names instead of file paths.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    supervisor_tasks: Optional[List[str]]
    current_task_description: Optional[str]
    analyst_output: Optional[Dict[str, Any]]
    analyst_llm_output: Optional[Dict[str, Any]]
    scientist_llm_output: Optional[Dict[str, Any]]
    scientist_output: Optional[Dict[str, Any]]
    code_final: Optional[Dict[str, Any]]
    processed_train_name: Optional[str]     
    processed_test_name: Optional[str]
    loop_process: bool 
    loop:  Optional[int]
    error: Optional[str]
    next_agent: Optional[Literal["DataAnalyst",  "DataScientist", "DataAnalystIA","DataScientistIA", "Supervisor", "code_compiler_AI","HumanInterpreter", "__end__"]] 
    interpreter_finished : bool
    final_answer_generated: bool # Added for clarity



CALOR_IA_DATA_ANALYST_SYSINT = (
"system",
"""
ğŸ¤– You are CALOR-IA Data Analyst Bot.
ğŸ�¯ Mission
1ï¸�âƒ£ Execute the data analysis task provided in the current_task_description from the Supervisor.
2ï¸�âƒ£ Write clean, runnable Python ğŸ�� code for data loading, cleaning, exploration, and visualization (using pandas, matplotlib, seaborn, etc.).
3ï¸�âƒ£ Analyze data (potentially loaded from intermediate_data_path if provided) and extract valuable insights.
4ï¸�âƒ£ **Prepare data for the Data Scientist, with a strong emphasis on creating high-impact features that are highly relevant to the task's expected goals and downstream modeling requirements. Actively evaluate engineered features to demonstrate their potential value.** Update intermediate_data_path if new data artifacts are created.
5ï¸�âƒ£ Place your results (code, summary of findings, paths to saved plots or data) into the analyst_output dictionary in the graph state.
6ï¸�âƒ£ Collaborate with CALOR_IA_DATA_SCIENTIST_SYSINT via the Supervisor by providing necessary data artifacts and insights.

ğŸ“� Style & Rules
  * Assure you use inputs train, test, submission = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv'),pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv'),pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')--
- Drop id column to avoid errors
- Do not create data, just prepare the datase and provide ideas to improve dataset
- train, test = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv'),pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
- You must implement data augmentation:
    Practical Workflow Example
    Suppose you have a dataset of various food items (rows), containing both:
    â€“ Categorical features: type of cuisine, dietary restrictions (vegan, gluten-free), meal type (breakfast, lunch, dinner), etc.
    â€“ Numeric features: grams of fat, grams of protein, grams of carbohydrates, total weight.
    â€“ Target: total calories (a continuous numeric value).
    You might:
    (a) Use feature engineering: create macros ratio or density features.
    (b) Train a CTGAN to generate additional synthetic food items after confirming it handles both numeric and categorical columns well.
    (c) Append sampled rows that look realistic to your training data.
    (d) Evaluate your model (e.g., gradient boosted trees, random forest, or a neural network) to see if it outperforms a baseline.
    (e) Monitor overfitting or unrealistic data by analyzing the distribution of the synthesized rows.

return python code do not loose any piece of code 
"""
)

CALOR_IA_DATA_SCIENTIST_SYSINT = (
    "system",
    """
    ğŸ¤– You are CALOR-IA Data Scientist Bot.


      * Assure you use inputs train, test = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv'),pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

    ğŸ�¯ Mission
     You are a data scientist how has to reduce REMSLE error and other functions, also you have to apply k flod validation to avoid overfitting
     You must improve the provided code, that could be implementing a model or make something of the provided pipeline of models and ensamble
     return python code do not loose any piece of code 
""")

CALOR_IA_CODE_COMPILER_SYSINT = (
    "system",
    """
    ğŸ¤– CALOR-IA Code Compiler Bot: Blending code fragments for Kaggle compatibility


    * Assure you use inputs train, test = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv'),pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

    ğŸ�¯ Primary Mission:
    1ï¸�âƒ£ Blend/merge code fragments from different sources into ONE cohesive, runnable script
    !Note separe train['Calories'] as predicted value to avoid conflicts with train and test preprocessing 
    Assure all libraries are imported correctly
    inputs : train, test, submission = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv'),pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv'),pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
    2ï¸�âƒ£ Ensure the code runs flawlessly in Kaggle environment
            
    ğŸ“‹ Technical Requirements:
    
    â€¢ Resolve all variable references between blended code sections
    â€¢ Ensure all imports are at the top of the script
    â€¢ Be carefull with shape handle 
    â€¢ build robust prediction using a blend method of models
    â€¢ Assure ejecution won't run for ever
    â€¢ Fix any syntax errors or logical inconsistencies
    â€¢ Keep all viable modeling code (LightGBM, XGBoost, Random Forest, etc.)
    â€¢ Implement ensemble prediction by combining model outputs
    â€¢ Assure continuity on the logic is correct
    â€¢ Generate a final submission.csv file
    
    
    ğŸš« Common Issues to Fix:
    â€¢ Inconsistent variable names between code segments
    â€¢ Missing function definitions or imports
    â€¢ Incorrect file paths for Kaggle environment
    â€¢ Syntax errors or incomplete code blocks
    
    ğŸ”� Return Format:
    Return ONLY the complete, ready-to-run Python script as a single coherent block.
    
    Let's create Kaggle-ready code that works right out of the box!
    """
)


def supervisor_node(state: GraphState) -> Dict[str, Any]:
    """
    Central decision-making node. Routes tasks to appropriate agents
    or ends the workflow based on the current state.
    Now routes to DataAnalyst_AI AFTER HumanInterpreter.
    """
    print("\n--- SUPERVISOR ---")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    print(f"Supervisor reviewing state. Last message type: {type(last_message).__name__ if last_message else 'None'}")
    if hasattr(last_message, 'content'):
         content_display = str(last_message.content)
         print(f"Supervisor received content: {content_display[:200]}{'...' if len(content_display) > 200 else ''}")

    # --- Retrieve outputs from potential sources ---
    analyst_orig_output = state.get('analyst_output')
    scientist_output_data = state.get('scientist_output')
    analyst_llm_output = state.get('analyst_llm_output')
    scientist_llm_output = state.get('scientist_llm_output')
    code = state.get('code_final')
    loop_process = state.get('loop_process', False)
    interpreter_finished = state.get('interpreter_finished', False)
    final_answer_generated = state.get('final_answer_generated', False)
    initial_train_name = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
    initial_test_name = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
    tasks = state.get('supervisor_tasks', [])
    error_flag = state.get('error')
    loop = state.get('loop')
    print(f"Original Analyst output available: {analyst_orig_output is not None}")
    print(f"Scientist output available: {scientist_output_data is not None}")
    print(f"Interpreter finished signal: {interpreter_finished}")
    print(f"Final answer generated flag: {final_answer_generated}")
    print(f'loop =======> # {loop}')
    next_agent = None
    task = None
    new_message_content = None
    updated_state_for_return = {}

    if loop == 0 and final_answer_generated:
        next_agent = '__end__'
        new_message_content = "final script generated." 
        return {"next_agent": next_agent}

    if not loop_process and final_answer_generated:
        next_agent = "DataAnalyst_AI"
        new_message_content = "another iteration over data analyst."     
        if loop == None:
            loop = 0
        updated_state_for_return['final_answer_generated'] = False
        
        
    if loop_process and not final_answer_generated:
        next_agent = "code_compiler_AI"
        new_message_content = "Workflow complete. Final script and reviews have been generated."
        updated_state_for_return['final_answer_generated'] = True  # Force set this flag

    # --- Decision Logic (Ordered by Priority) ---
    if loop_process:
        print('Analyst reviewed code and provide an improvement')
        loop_process = False
        next_agent = "DataScientist_AI"
        final_answer_generated = False
        new_message_content = "Analyst AI review complete. Routing compiled script to Data Scientist AI for review."
        
    # 1. Check for final answer signal - highest priority
    if final_answer_generated:
        print("Supervisor: Final answer signal received. Ending workflow.")
        next_agent = "DataAnalyst_AI"
        new_message_content = "Workflow complete. lets loop over trough all specialist one more time."
    # 2. Check if Scientist AI has completed review

    
    elif scientist_llm_output is not None and not final_answer_generated:
        print("Supervisor: Scientist AI review complete. Final answer should be ready.")
        # Force end the workflow since we've gone through all steps
        next_agent = "code_compiler_AI"
        new_message_content = "Workflow complete. Final script and reviews have been generated."
        updated_state_for_return['final_answer_generated'] = True  # Force set this flag
    
    # 3. Check if Analyst AI has finished its review - Next step is Scientist AI
    elif analyst_llm_output is not None and scientist_llm_output is None:
        print("Supervisor: Analyst AI review complete. Routing to Scientist AI for review.")
        next_agent = "DataScientist_AI"
        new_message_content = "Analyst AI review complete. Routing compiled script to Data Scientist AI for review."
    
    # 4. Check if Interpreter has finished - Start AI review chain
    elif interpreter_finished and analyst_llm_output is None:
        # Reset the flag to prevent re-triggering
        updated_state_for_return['interpreter_finished'] = False
        print("Supervisor: Interpreter finished compiling script. Routing to Data Analyst AI for review.")
        next_agent = "DataAnalyst_AI"
        new_message_content = "Interpreter step complete. Routing compiled script for AI review."
    
    # 5. Check for error condition
    elif error_flag is not None:
        print(f"Supervisor: Error detected in state: {error_flag}. Routing to Human Interpreter for review.")
        next_agent = "HumanInterpreter"
    
    # 6. Check for Scientist output data - Route to Interpreter
    elif scientist_output_data is not None and not interpreter_finished:
        print("Supervisor: Scientist code generation complete. Routing to HumanInterpreter for compilation.")
        next_agent = "HumanInterpreter"
        new_message_content = "Scientist code generation complete. Compiling final script."
        updated_state_for_return["processed_train_name"] = state.get("processed_train_name")
        updated_state_for_return["processed_test_name"] = state.get("processed_test_name")
        print(f"Supervisor propagating names from state for next step: train='{updated_state_for_return.get('processed_train_name')}', test='{updated_state_for_return.get('processed_test_name')}'")
    
    # 7. Check for original DataAnalyst output - Route to DataScientist
    elif analyst_orig_output is not None and scientist_output_data is None:
        print("Supervisor: Original Data Analyst successfully generated code and data names. Routing to Data Scientist.")
        updated_state_for_return["processed_train_name"] = analyst_orig_output.get('processed_train_name')
        updated_state_for_return["processed_test_name"] = analyst_orig_output.get('processed_test_name')
        print(f"Supervisor propagating names from original analyst output to Scientist: train='{updated_state_for_return.get('processed_train_name')}', test='{updated_state_for_return.get('processed_test_name')}'")
        task = "Build and evaluate LightGBM and XGBoost models using the processed data."
        next_agent = "DataScientist"
    
    # 8. Initial state handling
    elif 'initial_input_message' in locals() or 'initial_input_message' in globals():
        print("Supervisor: Initial human message received. Assigning initial task to Original Data Analyst.")
        next_agent = "DataAnalyst"
        new_message_content = "Okay, starting the data analysis workflow with the Data Analyst."
        task = "Perform initial EDA and preprocessing on train/test data."
    
    # 9. Unexpected state - End workflow
    else:
        print("Supervisor: Unexpected state - no initial message or prior agent output. Ending.")
        error_flag = "Workflow ended in an unexpected state."
        next_agent = "__end__"
        new_message_content = "Workflow terminated due to an unexpected state."

    # --- Prepare Return Dictionary ---
    updated_messages = list(messages)
    if new_message_content is not None:
        if not updated_messages or (hasattr(updated_messages[-1], 'content') and updated_messages[-1].content != new_message_content):
            updated_messages.append(AIMessage(content=str(new_message_content)))

    return_dict = {
        "messages": updated_messages,
        "supervisor_tasks": tasks,
        "current_task_description": task,
        "next_agent": next_agent,
        "error": error_flag,
        "final_answer_generated": updated_state_for_return.get('final_answer_generated', final_answer_generated),
        "interpreter_finished": updated_state_for_return.get('interpreter_finished', interpreter_finished),
        "initial_train_name": initial_train_name,
        "initial_test_name": initial_test_name,
        "loop_process" : loop_process,
        'loop' : loop,
        **updated_state_for_return
    }

    print(f"DEBUG: Supervisor RETURNING next_agent: '{return_dict['next_agent']}' (Type: {type(return_dict['next_agent'])})")
    print(f"DEBUG: Supervisor RETURNING final_answer_generated: {return_dict.get('final_answer_generated')}")
    print(f"DEBUG: Supervisor RETURNING interpreter_finished: {return_dict.get('interpreter_finished')}")

    return return_dict


def data_analyst_node(state: GraphState) -> Dict[str, Any]:
    """Data Analyst Node - Updated for DataFrame variable names."""
    print("\n--- DATA ANALYST ---")
    task = state.get('current_task_description')
    # Assume initial DFs are named 'train_df' and 'test_df' for clarity
    # Adjust these names if your actual initial variables are different (e.g., 'train', 'test')
    initial_train_name = state.get('initial_train_name')# <-- Adjust if your initial var is different
    initial_test_name =  state.get('initial_test_name') # <-- Adjust if your initial var is different

    if not task:
        error_output = {"error": "Data Analyst received no task.", "next_agent": "Supervisor"}
        print("Returning error output:", error_output)
        return error_output

    # Simulate the names of the DataFrames that the processing code will create
    processed_train_name = f"processed_{initial_train_name}" 
    processed_test_name = f"processed_{initial_test_name}"   

    print(f"Analyst expects input DataFrame variable: '{initial_train_name}'")
    print(f"Analyst expects input DataFrame variable: '{initial_test_name}'")
    print(f"Analyst will simulate creating output DataFrames: '{processed_train_name}' and '{processed_test_name}'")
    # inroducing SATYA create_predictor_features
    # --- Simulate Code (Make sure variable names match below) ---
    
    simulated_code =  generated_code = (f"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor # This import might be needed in the generated code

# --- Helper Functions (Copying the ones provided by user) ---
# Assuming these were defined or should be included in the generated code

# Define helper functions here (copying from user's input)
# ... (your create_features, swish, root_mean_squared_error, build_swish_mlp, rmsle functions go here)
def create_features(df):
    '''
    Create advanced features from the input data.
    This function serves as a form of "data augmentation" for tabular data
    by deriving new, potentially more informative features from existing ones.
    '''
    df = df.copy()
    epsilon = 1e-6 # Small constant to avoid division by zero

    # --- Basic Interaction & Ratio Features (>= 6 columns engineered) ---

    # 1. Body Mass Index (BMI): Standard health metric, relates weight and height.
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2 + epsilon)

    # 2. Duration * Heart_Rate: Represents total heartbeats during exercise, proxy for effort.
    df['Duration_HR'] = df['Duration'] * df['Heart_Rate']

    # 3. Duration * Body_Temp: Represents total heat generated during exercise, proxy for intensity/effort.
    df['Duration_Temp'] = df['Duration'] * df['Body_Temp']

    # 4. Weight / Height Ratio: Simpler ratio than BMI, might capture different linear relationships.
    df['Weight_Height_Ratio'] = df['Weight'] / (df['Height'] + epsilon)

    # 5. Heart Rate per Minute: Already captured by Heart_Rate, but could consider rate relative to duration?
    #    Let's create Heart_Rate_per_Duration instead.
    df['Heart_Rate_per_Duration'] = df['Heart_Rate'] / (df['Duration'] + epsilon) # Rate of heartbeats per minute of exercise

    # 6. Body Temperature per Minute: Change in temp per minute of exercise.
    df['Body_Temp_per_Duration'] = df['Body_Temp'] / (df['Duration'] + epsilon)

    # --- More Complex Interaction Features ---

    # 7. Exercise Intensity: Combines HR, Duration, and inversely related to Age (older might have lower max HR).
    df['Exercise_Intensity'] = (df['Heart_Rate'] * df['Duration']) / (df['Age'] + epsilon)

    # 8. Metabolic Factor: Combines BMI, Duration, HR, and inversely related to Age. A complex interaction term.
    df['Metabolic_Factor'] = (df['BMI'] * df['Duration'] * df['Heart_Rate']) / (df['Age'] + epsilon)

    # 9. Age * Duration Interaction: Older individuals exercising for longer might have different calorie burn patterns.
    df['Age_Duration_Interaction'] = df['Age'] * df['Duration']

    # 10. Height * Weight Product: Simple interaction, might capture overall body size effect differently than BMI.
    df['Height_Weight_Product'] = df['Height'] * df['Weight']

    # --- Polynomial Features (Example for Duration) ---
    # Captures non-linear relationship with Duration.
    df['Duration_Sq'] = df['Duration']**2

    # --- Categorical Features (Discretization) ---

    # 11. Age Group: Discretizing Age can help capture non-linear effects or group-specific patterns.
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'], right=False)
    df['Age_Group'] = df['Age_Group'].astype(object).fillna('Unknown')


    # 12. BMI Category: Standard health categories for BMI.
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 35, 40, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III'], right=False)
    df['BMI_Category'] = df['BMI_Category'].astype(object).fillna('Unknown')


    # 13. Heart Rate Zones (Simplified): Based on max HR (approx 220-Age) and exercise intensity.
    df['Max_HR_Estimate'] = 220 - df['Age']
    df['HR_Zone'] = pd.cut(df['Heart_Rate'] / (df['Max_HR_Estimate'] + epsilon),
                           bins=[0, 0.6, 0.7, 0.8, 0.9, 1.1], # Example zones based on % of max HR
                           labels=['<60%', '60-70%', '70-80%', '80-90%', '>90%'],
                           right=False)
    df['HR_Zone'] = df['HR_Zone'].astype(object).fillna('Unknown')
    df = df.drop('Max_HR_Estimate', axis=1) # Drop intermediate column


    # 14. Duration Category: Simple bins for exercise duration.
    df['Duration_Category'] = pd.cut(df['Duration'], bins=[0, 15, 30, 45, 60, 120, 300], labels=['<15min', '15-30min', '30-45min', '45-60min', '1-2hr', '>2hr'], right=False)
    df['Duration_Category'] = df['Duration_Category'].astype(object).fillna('Unknown')


    # Interaction between Sex and Duration/HeartRate/BodyTemp
    df['Sex_Male_Duration'] = df['Duration'] * (df['Sex'] == 'M')
    df['Sex_Female_Duration'] = df['Duration'] * (df['Sex'] == 'F')
    df['Sex_Male_HR'] = df['Heart_Rate'] * (df['Sex'] == 'M')
    df['Sex_Female_HR'] = df['Heart_Rate'] * (df['Sex'] == 'F')


    return df


# --- Load Data (Assuming 'train' and 'test' DataFrames are already loaded) ---
print("Using input DataFrames named '{initial_train_name}' and '{initial_test_name}'...")
#Assuming 'y' and 'submission' also exist
If 'id' and 'Calories' need dropping, it should happen *before* feature engineering on the original DFs.
# Let's assume the initial DataFrames passed in via '{initial_train_name}' and '{initial_test_name}'
# are already prepped (ids dropped, target 'y' separated).
If not, add that step here:
y = {initial_train_name}['Calories'] 
# {initial_train_name} = {initial_train_name}.drop('Calories', axis=1) # Example drop

# Example: If train_df/test_df *contain* id/Calories
# train_df_working = {initial_train_name}.copy()
# test_df_working = {initial_test_name}.copy()
# if 'id' in train_df_working.columns: train_ids = train_df_working['id'] ; train_df_working = train_df_working.drop('id', axis=1)
# if 'Calories' in train_df_working.columns: y = train_df_working['Calories'] ; train_df_working = train_df_working.drop('Calories', axis=1)
# if 'id' in test_df_working.columns: test_ids = test_df_working['id'] ; test_df_working = test_df_working.drop('id', axis=1)
# print(f"Initial train shape: {{train_df_working.shape}}")
# print(f"Initial test shape: {{test_df_working.shape}}")


# --- Feature Engineering ---
print("\nApplying feature engineering...")
# Apply feature engineering to the working copies (or original DFs if pre-cleaned)
# Using the assumed initial variable names from the state.
X_train_fe = create_features({initial_train_name}) 
X_test_fe = create_features({initial_test_name})

print(f"Train shape after feature engineering: {{X_train_fe.shape}}")
print(f"Test shape after feature engineering: {{X_test_fe.shape}}")

# Identify numerical and categorical features AFTER feature engineering
# Ensure handling of potential errors if a feature was expected but not created
all_features = list(X_train_fe.columns)
numerical_features = [col for col in all_features if X_train_fe[col].dtype in ['int64', 'float64']]
categorical_features = [col for col in all_features if X_train_fe[col].dtype == 'object'] # Using object dtype for pandas categorical/string

print(f"Numerical features ({{len(numerical_features)}}): {{numerical_features}}")
print(f"Categorical features ({{len(categorical_features)}}): {{categorical_features}}")

# --- Preprocessing Pipeline (KNN Imputation & Quantile Transformation + OneHot) ---
print("\nSetting up preprocessing pipeline...")

# Check if numerical/categorical features exist before creating pipelines
transformers = []
if numerical_features:
    numerical_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),       # KNN imputation
        ('scaler', QuantileTransformer(output_distribution='normal')) # Robust scaling via QuantileTransformer
    ])
    transformers.append(('num', numerical_pipeline, numerical_features))
else:
    print("No numerical features found. Skipping numerical pipeline.")

if categorical_features:
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)) # One-Hot Encoding
    ])
    transformers.append(('cat', categorical_pipeline, categorical_features))
else:
     print("No categorical features found. Skipping categorical pipeline.")

# Create ColumnTransformer only if there are transformers
if transformers:
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    # Fit and Transform data
    print("Fitting and transforming data with preprocessor...")
    # *** Assign results to the variable names specified by the analyst node ***
    {processed_train_name} = preprocessor.fit_transform(X_train_fe)
    {processed_test_name} = preprocessor.transform(X_test_fe)

    print(f"Processed train shape (variable '{processed_train_name}'): {{{processed_train_name}.shape}}")
    print(f"Processed test shape (variable '{processed_test_name}'): {{{processed_test_name}.shape}}")

else:
    print("No features identified for preprocessing. Processed data will be empty.")
    # Handle case where no features are processed - might need to create empty arrays
    {processed_train_name} = np.empty((X_train_fe.shape[0], 0)) # Create empty numpy array with correct rows
    {processed_test_name} = np.empty((X_test_fe.shape[0], 0))
    print(f"Processed train shape (variable '{processed_train_name}'): {{{processed_train_name}.shape}}")
    print(f"Processed test shape (variable '{processed_test_name}'): {{{processed_test_name}.shape}}")


# The processed data is now available as numpy arrays named '{processed_train_name}' and '{processed_test_name}'
# The target variable 'y' is assumed to also be available from the initial loading step.
# This completes the Analyst's task of preparing the data for the Scientist.

""")


    return {
        "analyst_output": {
            "code": simulated_code,
            "processed_train_name": processed_train_name,
            "processed_test_name": processed_test_name,
        },
        "current_task_description": None,
    }
    return{
        "analyst_output":{
            "code": simulated_code,
            "processed_train_name": processed_train_name,
            "processed_test_name": processed_test_name,
        },
         "current_task_description": None, # Task is complete for Analyst
    }


import os
from typing import Dict, Any

# Assuming GraphState class is defined elsewhere
# Assuming necessary library imports (like pandas, numpy etc.) are at the top level of your script

def data_scientist_node(state: GraphState) -> Dict[str, Any]:
    """
    Data Scientist Node - Generates modeling code (LightGBM, XGBoost)
    using processed data names provided by the DataAnalyst node via the state.
    This version DOES NOT execute the generated code.
    """
    print("\n--- DATA SCIENTIST (Code Generation Only) ---")
    task = state.get('current_task_description')
    if not task:
        print("Scientist: No task received.")
        # Return an error structure within the expected output key
        return {
            "scientist_output": {"error": "Data Scientist received no task."},
            "next_agent": "Supervisor"
            }

    # --- Get processed DataFrame names from the analyst's output in state ---
    # The analyst node puts these names in 'analyst_output' dictionary
    analyst_result = state.get('analyst_output')
    if not analyst_result or not isinstance(analyst_result, dict):
        error_msg = "Scientist Error: Analyst output not found or invalid in state."
        print(error_msg)
        return {
            "scientist_output": {"error": error_msg},
             "next_agent": "Supervisor"
             }


    # Get the variable names *as they will be created* by the analyst's code
    processed_train_name = analyst_result.get('processed_train_name')
    processed_test_name  = analyst_result.get('processed_test_name')
    # Assume the analyst's code also creates a variable named 'target'
    # This name is hardcoded based on the analyst's provided simulated code structur

    if not processed_train_name or not processed_test_name:
        error_msg = "Scientist Error: Processed DataFrame names not found within analyst output in state."
        print(error_msg)
        return {
            "scientist_output": {"error": error_msg},
             "next_agent": "Supervisor"
             }

    print(f"Scientist generating code for task: {task}")
    print(f"Will use processed train DataFrame variable: '{processed_train_name}'")
    print(f"Will use processed test DataFrame variable : '{processed_test_name}'")


    # --- Generate Code String ---
    # Use single braces {} ONLY for variables substituted NOW ({processed_train_name}, etc.)
    # Use double braces {{}} for f-strings intended for LATER execution within the generated code.
    generated_code = (f'''
# --- Imports for the generated modeling script ---
import pandas as pd # Might be needed if submission is loaded/modified
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

# expected_shape should be evaluated within the generated code
expected_shape = ({processed_test_name}.shape[0] if '{processed_test_name}' in locals() else 0)

# Import TensorFlow components if available
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow not available. TensorFlow model will be skipped.")
    tf = None # Set tf to None to check availability later

# --- Helper Functions (Copying the ones from Analyst's generated code) ---
# Ensure these match what the analyst generated, especially the RMSLE and TF functions

def swish(x):
    # Need to handle both numpy (for direct use if needed) and tensorflow
    if tf and isinstance(x, tf.Tensor):
         return x * tf.keras.backend.sigmoid(x)
    else: # Assume numpy or similar
         # This numpy version is just for local testing outside TF context if needed
         # Actual TF model will use the TF version
         return x / (1 + np.exp(-x)) # Sigmoid approximation for numpy

def root_mean_squared_error(y_true, y_pred):
    # Need to handle both numpy and tensorflow
    if tf and isinstance(y_true, tf.Tensor):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    else: # Assume numpy
        return np.sqrt(np.mean(np.square(y_pred - y_true)))


def build_swish_mlp(input_shape):
    if tf is None:
        print("TensorFlow not available, cannot build swish MLP.")
        return None
    inputs = Input(shape=input_shape)
    x = Dense(64, activation=swish)(inputs)
    x = Dense(128, activation=swish)(x)
    x = Dense(64, activation=swish)(x)
    x = Dense(1)(x)
    model = Model(inputs, x)
    # Use RootMeanSquaredError metric for TensorFlow
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# rmsle function from sklearn.metrics
def rmsle(y_true, y_pred):
    # Ensure predictions are non-negative for log calculation
    y_pred = np.maximum(0, y_pred)
    # Ensure inputs are numpy arrays and handle potential shape issues
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Escape braces for f-string inside the generated code
    print(f"{{Debug RMSLE shapes - True: {{y_true.shape}}, Pred: {{y_pred.shape}}}}") # Corrected escaping
    if y_true.shape != y_pred.shape:
        print(f"{{Warning: Shape mismatch between true and predicted values ({{y_true.shape}} vs {{y_pred.shape}}). Cannot compute RMSLE.}}") # Corrected escaping
        return float('inf')
    # Add a small epsilon to avoid log(0)
    return np.sqrt(mean_squared_log_error(np.maximum(0, y_true) + 1e-9, y_pred + 1e-9))


# --- Data Splitting ---
print("\nSplitting data into training and validation sets...")
# Use the processed train data variable name from the analyst node
# Assuming the target 'y' is also available from the initial data loading/prep
# Escape the variables being checked and accessed so they evaluate in the generated code context
if '{processed_train_name}' in locals() and 'y' in locals() and {{processed_train_name}}.shape[0] == y.shape[0]:
    X_train, X_val, y_train, y_val = train_test_split({processed_train_name}, y, test_size=0.2, random_state=42)
    # Escape braces for f-strings inside the generated code
    print(f"{{Train shapes: {{X_train.shape}}, {{y_train.shape}}}}") # CORRECTED
    print(f"{{Validation shapes: {{X_val.shape}}, {{y_val.shape}}}}") # CORRECTED
    data_split_success = True
else:
    # Escape braces for f-strings inside the generated code
    print(f"{{Error: Processed train data ('{processed_train_name}') or target ('y') not found or shapes mismatch. Skipping modeling.}}")
    # Use {{ }} for the outer f-string, and {{variable}} for variables inside the inner f-string
    print(f"{{'{processed_train_name}' in locals(): {{'{processed_train_name}' in locals()}}}}") # CORRECTED
    print(f"{{'y' in locals(): {{'y' in locals()}}}}") # CORRECTED
    # Escape variables/accessors being checked and accessed
    if '{processed_train_name}' in locals(): print(f"{{Shape of '{processed_train_name}': {{locals()['{processed_train_name}'].shape}}}}") # CORRECTED
    if 'y' in locals(): print(f"{{Shape of 'y': {{y.shape}}}}") # CORRECTED
    X_train, X_val, y_train, y_val = np.array([]).reshape(0,-1), np.array([]).reshape(0,-1), np.array([]), np.array([]) # Create empty arrays
    data_split_success = False


if data_split_success and X_train.shape[0] > 0:
    # --- Scaling for Neural Network ---
    # NN often performs better with features scaled to a 0-1 range and log-transformed target
    print("\nScaling features for Neural Network using MinMaxScaler...")
    scaler_nn = MinMaxScaler()
    X_train_scaled = scaler_nn.fit_transform(X_train)
    X_val_scaled = scaler_nn.transform(X_val)
    # Scale the actual test set using the processed test data variable name
    # Escape variable being checked and accessed
    if '{processed_test_name}' in locals():
       X_test_scaled = scaler_nn.transform({{processed_test_name}})
       test_scaling_success = True
    else:
       print(f"{{Error: Processed test data ('{processed_test_name}') not found for scaling.}}")
       X_test_scaled = np.array([]).reshape(0,-1)
       test_scaling_success = False


    print("Log transforming target for NN...")
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    # --- Model Training ---

    # Train LightGBM (using split processed data - numpy arrays)
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=2000, learning_rate=0.03, num_leaves=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=0.1, seed=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_val_pred = lgb_model.predict(X_val)
    # Predict on the processed test data using the variable name
    # Escape variable being checked and accessed
    if '{processed_test_name}' in locals():
       lgb_test_pred = lgb_model.predict({{processed_test_name}})
    else:
       # Escape variable accessor being accessed in the generated code
       lgb_test_pred = np.zeros({{processed_test_name}}.shape[0] if '{processed_test_name}' in locals() else 0) # Predict zeros if test data missing
       print(f"{{Warning: '{processed_test_name}' not available for LightGBM test prediction.}}")
    print("LightGBM training complete.")

    # Train XGBoost (using split processed data - numpy arrays)
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_estimators=2000, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.8, lambda_l1=0.1, lambda_l2=0.1, seed=42, n_jobs=-1, tree_method='hist') # Use hist for faster training
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    xgb_val_pred = xgb_model.predict(X_val)
    # Predict on the processed test data using the variable name
    # Escape variable being checked and accessed
    if '{processed_test_name}' in locals():
       xgb_test_pred = xgb_model.predict({{processed_test_name}})
    else:
        # Escape variable accessor being accessed in the generated code
        xgb_test_pred = np.zeros({{processed_test_name}}.shape[0] if '{processed_test_name}' in locals() else 0)
        print(f"{{Warning: '{processed_test_name}' not available for XGBoost test prediction.}}")
    print("XGBoost training complete.")

    # Train RandomForestRegressor (using split processed data - numpy arrays)
    print("\nTraining RandomForestRegressor...")
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_val_pred = rf_model.predict(X_val)
    # Predict on the processed test data using the variable name
    # Escape variable being checked and accessed
    if '{processed_test_name}' in locals():
        rf_test_pred = rf_model.predict({{processed_test_name}})
    else:
        # Escape variable accessor being accessed in the generated code
        rf_test_pred = np.zeros({{processed_test_name}}.shape[0] if '{processed_test_name}' in locals() else 0)
        print(f"{{Warning: '{processed_test_name}' not available for RandomForest test prediction.}}")
    print("RandomForestRegressor training complete.")

    # Train CatBoostRegressor (using split processed data - numpy arrays)
    print("\nTraining CatBoostRegressor...")
    cat_model = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, random_seed=42, verbose=0, early_stopping_rounds=100, l2_leaf_reg=3, loss_function='RMSE')
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_val_pred = cat_model.predict(X_val)
    # Predict on the processed test data using the variable name
    # Escape variable being checked and accessed
    if '{processed_test_name}' in locals():
        cat_test_pred = cat_model.predict({{processed_test_name}})
    else:
         # Escape variable accessor being accessed in the generated code
         cat_test_pred = np.zeros({{processed_test_name}}.shape[0] if '{processed_test_name}' in locals() else 0)
         print(f"{{Warning: '{processed_test_name}' not available for CatBoost test prediction.}}")
    print("CatBoostRegressor training complete.")


    # Build and train TensorFlow model (using scaled data)
    if tf and test_scaling_success: # Only attempt TF if TF is available and test data was scaled
        print("\nBuilding and training TensorFlow model...")
        if X_train_scaled.shape[0] == 0 or X_train_scaled.shape[1] == 0:
             print("{{Warning: Scaled training data is empty. Skipping TensorFlow model training.}}") # Corrected escaping
             tf_test_pred = np.zeros(X_test_scaled.shape[0])
             tf_val_pred = np.zeros(X_val_scaled.shape[0])
        else:
            tf_model = build_swish_mlp(input_shape=(X_train_scaled.shape[1],))
            if tf_model: # Check if model was built successfully
                early_stopping_tf = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=15, restore_best_weights=True, mode='min', verbose=0)

                print("{{  Training TensorFlow model...}}") # Corrected escaping
                history = tf_model.fit(X_train_scaled, y_train_log,
                                       epochs=200,
                                       batch_size=64,
                                       verbose=0,
                                       validation_data=(X_val_scaled, y_val_log),
                                       callbacks=[early_stopping_tf])
                print("{{  TensorFlow model training complete.}}") # Corrected escaping

                print("{{Making TensorFlow predictions...}}") # Corrected escaping
                tfy_test_pred_log = tf_model.predict(X_test_scaled).flatten()
                tf_test_pred = np.expm1(tfy_test_pred_log)
                # Clip to original target range (need 'y' min/max - assuming 'y' is available and is the original target)
                if 'y' in locals():
                    # Escape accessor y.min() and y.max()
                    tf_test_pred = np.clip(tf_test_pred, {{y}}.min(), {{y}}.max())
                else:
                     # Fallback clipping or warning if y is not available
                     tf_test_pred = np.maximum(0, tf_test_pred) # At least ensure non-negative
                     print("{{Warning: Original target 'y' not available for TensorFlow prediction clipping.}}") # Corrected escaping


                tfy_val_pred_log = tf_model.predict(X_val_scaled).flatten()
                tf_val_pred = np.expm1(tfy_val_pred_log)
                if 'y' in locals():
                    # Escape accessor y.min() and y.max()
                    tf_val_pred = np.clip(tf_val_pred, {{y}}.min(), {{y}}.max())
                else:
                    tf_val_pred = np.maximum(0, tf_val_pred)
                    print("{{Warning: Original target 'y' not available for TensorFlow validation prediction clipping.}}") # Corrected escaping
                print("{{TensorFlow predictions made.}}") # Corrected escaping
            else: # TF model build failed
                tf_test_pred = np.zeros(X_test_scaled.shape[0])
                tf_val_pred = np.zeros(X_val_scaled.shape[0])
                print("{{TensorFlow model could not be built. Skipping TF predictions.}}") # Corrected escaping

    else: # TF not available or test data not scaled
        print("\nSkipping TensorFlow model training and prediction.")
        # Ensure prediction variables exist even if TF is skipped
        # Need the shape of the processed test data to create zeros array
        # Escape variable accessor being accessed in the generated code
        test_data_shape = ({{processed_test_name}}.shape[0] if '{processed_test_name}' in locals() else 0)
        tf_test_pred = np.zeros(test_data_shape)
        tf_val_pred = np.zeros(X_val.shape[0]) # Use X_val shape for validation predictions
        print("{{TensorFlow prediction variables initialized to zeros.}}") # Corrected escaping


    # --- Validation Scores ---
    print(f"\n--- Validation RMSLE Scores ---")
    # Escape accessor y_val.shape
    if {{y_val}}.shape[0] > 0:
        # Check if prediction variables exist and have correct shape before computing scores
        # Escape accessors prediction.shape and y_val.shape
        if 'lgb_val_pred' in locals() and {{lgb_val_pred}}.shape == {{y_val}}.shape:
           # Escape braces for f-string inside the generated code
           print(f"{{LightGBM Validation RMSLE: {{rmsle(y_val, lgb_val_pred):.5f}}}}") # CORRECTED
        else:
           print("{{LightGBM Validation RMSLE: N/A (predictions missing or shape mismatch)}}")

        # Escape accessors prediction.shape and y_val.shape
        if 'xgb_val_pred' in locals() and {{xgb_val_pred}}.shape == {{y_val}}.shape:
           # Escape braces for f-string inside the generated code
           print(f"{{XGBoost Validation RMSLE: {{rmsle(y_val, xgb_val_pred):.5f}}}}") # CORRECTED
        else:
            print("{{XGBoost Validation RMSLE: N/A (predictions missing or shape mismatch)}}")

        # Escape accessors prediction.shape and y_val.shape
        if 'rf_val_pred' in locals() and {{rf_val_pred}}.shape == {{y_val}}.shape:
           # Escape braces for f-string inside the generated code
           print(f"{{RandomForest Validation RMSLE: {{rmsle(y_val, rf_val_pred):.5f}}}}") # CORRECTED
        else:
            print("{{RandomForest Validation RMSLE: N/A (predictions missing or shape mismatch)}}")

        # Escape accessors prediction.shape and y_val.shape
        if 'cat_val_pred' in locals() and {{cat_val_pred}}.shape == {{y_val}}.shape:
            # Escape braces for f-string inside the generated code
            print(f"{{CatBoost Validation RMSLE: {{rmsle(y_val, cat_val_pred):.5f}}}}") # CORRECTED
        else:
             print("{{CatBoost Validation RMSLE: N/A (predictions missing or shape mismatch)}}")

        # Escape accessors prediction.shape and y_val.shape
        if 'tf_val_pred' in locals() and {{tf_val_pred}}.shape == {{y_val}}.shape:
           # Escape braces for f-string inside the generated code
           print(f"{{TensorFlow Validation RMSLE: {{rmsle(y_val, tf_val_pred):.5f}}}}") # CORRECTED
        else:
            print("{{TensorFlow Validation RMSLE: N/A (predictions missing or shape mismatch)}}")

    else:
        print("{{No validation data available to compute scores.}}")
    print(f"------------------------------")

    # --- Ensemble Prediction ---
    print("\nCreating ensemble prediction...")

    # Ensure all test prediction variables exist and have the same expected shape
    # The expected shape is the number of rows in the processed test data


    # List of models and their test predictions
    # Escape prediction variable accessors
    test_preds = {{
        'lgb': lgb_test_pred if 'lgb_test_pred' in locals() and {{lgb_test_pred}}.shape[0] == expected_shape else np.zeros(expected_shape),
        'xgb': xgb_test_pred if 'xgb_test_pred' in locals() and {{xgb_test_pred}}.shape[0] == expected_shape else np.zeros(expected_shape),
        'rf':  rf_test_pred  if 'rf_test_pred'  in locals() and {{rf_test_pred}}.shape[0] == expected_shape else np.zeros(expected_shape),
        'cat': cat_test_pred if 'cat_test_pred' in locals() and {{cat_test_pred}}.shape[0] == expected_shape else np.zeros(expected_shape),
        'tf':  tf_test_pred  if 'tf_test_pred'  in locals() and {{tf_test_pred}}.shape[0] == expected_shape else np.zeros(expected_shape),
    }}
    # Escape prediction variable accessors and y_val shape accessor
    val_preds = {{
        'lgb': lgb_val_pred if 'lgb_val_pred' in locals() and {{lgb_val_pred}}.shape == {{y_val}}.shape else np.zeros({{y_val}}.shape[0]),
        'xgb': xgb_val_pred if 'xgb_val_pred' in locals() and {{xgb_val_pred}}.shape == {{y_val}}.shape else np.zeros({{y_val}}.shape[0]),
        'rf':  rf_val_pred  if 'rf_val_pred'  in locals() and {{rf_val_pred}}.shape == {{y_val}}.shape else np.zeros({{y_val}}.shape[0]),
        'cat': cat_val_pred if 'cat_val_pred' in locals() and {{cat_val_pred}}.shape == {{y_val}}.shape else np.zeros({{y_val}}.shape[0]),
        'tf':  tf_val_pred  if 'tf_val_pred'  in locals() and {{tf_val_pred}}.shape == {{y_val}}.shape else np.zeros({{y_val}}.shape[0]),
    }}

    # Calculate weights based on validation performance (using RMSLE)
    # Avoid division by zero or using infinite RMSLE from errors
    # Escape accessors for val_preds and y_val
    val_rmsles = {{
        'lgb': rmsle({{y_val}}, {{val_preds}}['lgb']) if {{y_val}}.shape[0] > 0 and np.sum(np.abs({{val_preds}}['lgb'])) > 0 else float('inf'),
        'xgb': rmsle({{y_val}}, {{val_preds}}['xgb']) if {{y_val}}.shape[0] > 0 and np.sum(np.abs({{val_preds}}['xgb'])) > 0 else float('inf'),
        'rf':  rmsle({{y_val}}, {{val_preds}}['rf'])  if {{y_val}}.shape[0] > 0 and np.sum(np.abs({{val_preds}}['rf'])) > 0 else float('inf'),
        'cat': rmsle({{y_val}}, {{val_preds}}['cat']) if {{y_val}}.shape[0] > 0 and np.sum(np.abs({{val_preds}}['cat'])) > 0 else float('inf'),
        'tf':  rmsle({{y_val}}, {{val_preds}}['tf'])  if {{y_val}}.shape[0] > 0 and np.sum(np.abs({{val_preds}}['tf'])) > 0 and tf is not None else float('inf'), # Only include TF if available
    }}

    # Calculate inverse RMSLE (higher for better models), handle inf/zero
    # Escape dictionary view accessors
    inv_rmsles = {{k: (1.0 / v if v != 0 and v != float('inf') else 0) for k, v in {{val_rmsles}}.items()}}

    # Normalize to get weights
    # Escape dictionary view accessors
    total_inv = sum({{inv_rmsles}}.values())
    if total_inv > 1e-9: # Avoid division by near zero
        # Escape dictionary view accessors
        weights = {{k: v / total_inv for k, v in {{inv_rmsles}}.items()}}
    else: # If all models failed or had infinite RMSLE, use equal weights or zero weights
        print("{{Warning: Cannot calculate meaningful ensemble weights (all models failed or infinite RMSLE). Using equal weights as fallback.}}") # Corrected escaping
        # Escape dictionary view accessors
        valid_models = [m for m, inv in {{inv_rmsles}}.items() if inv > 0]
        if valid_models:
             equal_weight = 1.0 / len(valid_models)
             # Escape dictionary creation and looping
             weights = {{m: equal_weight for m in valid_models}}
             for m in models: # Ensure all keys are in weights dict
                 if m not in weights: weights[m] = 0
        else:
             print("{{Warning: No models trained successfully. Ensemble prediction will be zero.}}") # Corrected escaping
             # Escape dictionary creation
             weights = {{m: 0 for m in models}}


    # Escape braces for f-string inside the generated code
    print(f"{{Model weights: {{weights}}}}") # CORRECTED

    # Calculate weighted ensemble predictions
    ensemble_predictions = np.zeros(expected_shape)
    for model in models:
        # Escape weights and test_preds dictionary accessors
        ensemble_predictions += {{weights}}[model] * {{test_preds}}[model]


    # Ensure predictions are non-negative (calories can't be negative)
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    # --- Create Submission ---
    # Assume 'submission' DataFrame with an 'id' column is available from initial loading
    print("\nPreparing submission file...")
    # Escape submission DataFrame accessors (.columns, .shape[0]) and expected_shape
    if 'submission' in locals() and 'id' in {{submission}}.columns and {{submission}}.shape[0] == {{expected_shape}}:
        # Escape submission DataFrame item assignment and method call
        {{submission}}['Calories'] = ensemble_predictions
        {{submission}}.to_csv('ensemble_submission.csv', index=False)
        print("{{Submission file created: ensemble_submission.csv}}") # Corrected escaping
    else:
        print("{{Error: Submission DataFrame not found, invalid, or shape mismatch. Cannot create submission file.}}") # Corrected escaping
        print("{{Details:}}") # Corrected escaping
        if 'submission' not in locals(): print("{{'submission' variable not found in generated code locals().}}") # Corrected escaping
        # Escape submission DataFrame column check
        elif 'id' not in {{submission}}.columns: print("{{'submission' DataFrame missing 'id' column.}}") # Corrected escaping
        # Escape submission DataFrame shape and expected_shape comparison within the *inner* f-string.
        # This needs to be escaped using {{ and }} in the outer f-string.
        elif {{submission}}.shape[0] != {{expected_shape}}: print(f"{{Submission shape ({{submission.shape[0]}}) does not match test prediction shape ({{expected_shape}}).}}") # CORRECTED
    # --- CORRECTED LINES END HERE ---


    print("\nProcess complete!") # This print does not contain a variable, so no inner f-string escaping needed

else:
    # Escape variable accessor being accessed in the generated code
    print("\\nSkipping model training, validation, and submission due to data split failure or empty training data.")
    # Ensure ensemble_predictions variable exists for potential return, even if zero
    ensemble_predictions = np.zeros({{processed_test_name}}.shape[0] if '{processed_test_name}' in locals() else 0)

    ''') 
                    
    # --- Code Generation Complete ---
    print("--- Code Generation Complete ---")

    # --- Prepare Results (Code Only) ---
    # Since we are not executing, we only return the generated code.
    # No stdout, error capture, or artifact paths from execution.
    scientist_result = {
        "code": generated_code,
        "processed_train_name": processed_train_name, # Pass the train DataFrame name
        "processed_test_name": processed_test_name,   # Pass the test DataFrame name
    }
    

    # --- Return state update ---
    return {
        "scientist_output": scientist_result,
        "next_agent": "Supervisor",
        "current_task_description": None
    }


def bot_to_human_interpreter(state: GraphState) -> Dict[str, Any]:
    """
    Gathers code from Analyst and Scientist outputs, blends them into a single script,
    adds explanations within the code, and formats it for copy-pasting.
    Returns an update dict with the combined code message and ROUTES TO THE REVIEW NODE.
    Does NOT set final_answer_generated=True.
    """
    print("\n--- BOT TO HUMAN INTERPRETER (Code Compiler) ---")

    # --- Robustly retrieve and validate outputs from the state ---
    # Note: This version assumes it's receiving state after Analyst/Scientist
    analyst_output_raw = state.get('analyst_llm_output')  # Check LLM output first
    scientist_output_raw = state.get('scientist_llm_output')
    
    if not analyst_output_raw:
        analyst_output_raw = state.get('analyst_output')  # Fallback to original if LLM output not found
        
    if not scientist_output_raw:
        scientist_output_raw = state.get('scientist_output')

    # Initialize variables to avoid UnboundLocalError
    analyst_output = {}
    scientist_output = {}
    
    # Ensure they are dictionaries before proceeding
    if analyst_output_raw:
        analyst_output = analyst_output_raw if isinstance(analyst_output_raw, dict) else {}
    if scientist_output_raw:
        scientist_output = scientist_output_raw if isinstance(scientist_output_raw, dict) else {}
    
    # Initialize a list to hold parts of the combined script
    combined_code_parts = []
    
    # Add an introductory comment/header to the script
    combined_code_parts.append("# Combined Data Science and Data Analyst IA's Workflow Script\n")
    combined_code_parts.append("# Generated by Analyst-IA's Agent Team\n\n")

    # Indicate if there were errors
    analyst_error = analyst_output.get('error')
    scientist_error = scientist_output.get('error')
    
    if analyst_error or scientist_error:
        combined_code_parts.append("# !!! NOTE: Errors occurred during the automated workflow stages. Review the error notes below.\n\n")

    combined_code_parts.append("# This script combines preprocessing and modeling steps.\n")
    combined_code_parts.append("# Ensure you have the necessary libraries installed (e.g., pandas, numpy, sklearn, lightgbm, xgboost, matplotlib).\n")
    combined_code_parts.append("# Make sure your initial 'train' and 'test' DataFrames (or equivalent data loading) exist before running.\n\n")

    # --- Section 1: Data Preprocessing (from Data Analyst) ---
    combined_code_parts.append("# --- 1. Data Preprocessing ---\n")
    combined_code_parts.append("# This section handles steps like identifying column types, imputation, scaling, and encoding.\n")
    combined_code_parts.append("# It prepares the raw data into processed DataFrames ready for modeling.\n\n")

    # Get the Analyst's code and error status (safe now due to checks above)
    analyst_code = analyst_output.get('code', '')

    # Safely get expected output names, handling cases where output is missing or incomplete
    processed_train_name = analyst_output.get('processed_train_name', "processed_train_name")  # Use a reasonable default
    processed_test_name = analyst_output.get('processed_test_name', "processed_test_name")    # Use a reasonable default

    if analyst_error:
        combined_code_parts.append(f"# !!! Data Analyst Error during preprocessing execution or code generation.\n")
        combined_code_parts.append(f"# !!! Error Details: {analyst_error}\n\n")
    elif not analyst_code:
        combined_code_parts.append("# Data preprocessing code was not provided by the Analyst.\n\n")
    else:
        # Append the analyst's code
        combined_code_parts.append(analyst_code.strip())
        combined_code_parts.append("\n\n")  # Add spacing after the code block
        # Add comments explaining the *expected output variables*
        combined_code_parts.append("# Expected outputs from this section (as variables in your environment):\n")
        combined_code_parts.append(f"# - {processed_train_name}: Processed training features (potentially including target)\n")
        combined_code_parts.append(f"# - {processed_test_name}: Processed test features\n\n")

    # --- Section 2: Model Training and Evaluation (from Data Scientist) ---
    combined_code_parts.append("# --- 2. Model Training and Evaluation ---\n")
    combined_code_parts.append("# This section uses the processed data to train machine learning models (LightGBM, XGBoost) and evaluate them.\n")
    combined_code_parts.append(f"# It assumes variables like '{processed_train_name}' and '{processed_test_name}' exist from the previous section.\n\n")

    # Get the Scientist's code and error status (safe now due to checks above)
    scientist_code = scientist_output.get('code', '')

    if scientist_error:
        combined_code_parts.append(f"# !!! Data Scientist Error during modeling execution or code generation.\n")
        combined_code_parts.append(f"# !!! Error Details: {scientist_error}\n\n")
        # Optionally include stdout even if there was an error, might contain clues
        if scientist_stdout:
            combined_code_parts.append("# Captured output (may contain error details):\n")
            commented_stdout = "\n".join([f"# {line}" for line in scientist_stdout.strip().split('\n')])
            combined_code_parts.append(commented_stdout + "\n\n")
    elif not scientist_code:
        combined_code_parts.append("# Model training code was not provided by the Scientist.\n\n")
    else:
        # Append the scientist's code
        combined_code_parts.append(scientist_code.strip())
        combined_code_parts.append("\n\n")  # Add spacing after the code block
        # Add comments explaining expected outputs/artifacts
        combined_code_parts.append("# Expected outputs/artifacts from this section:\n")
        combined_code_parts.append("# - Console output showing training progress and validation scores.\n")
        combined_code_parts.append("# - 'lgb_metric_plot.png', 'xgb_metric_plot.png': Plots showing model performance during training.\n")
        combined_code_parts.append("# - 'lgb_test_preds.csv', 'xgb_test_preds.csv': CSV files with predictions on the test set.\n\n")

    # --- Section 3: Summary Notes ---
    combined_code_parts.append("# --- 3. Summary Notes ---\n")
    if analyst_error or scientist_error:
        combined_code_parts.append("# NOTE: Review the error messages and generated code sections carefully.\n")
    else:
        combined_code_parts.append("# NOTE: The automated workflow appears to have completed successfully. Review the generated files and console output.\n")

    
        # Special case: if we have raw strings instead of structured output
    if isinstance(analyst_output_raw, str) or isinstance(scientist_output_raw, str):
        if isinstance(analyst_output_raw, str):
            combined_code_parts.append("# --- Analyst Output ---\n")
            combined_code_parts.append(analyst_output_raw)
            combined_code_parts.append("\n\n")
            
        if isinstance(scientist_output_raw, str):
            combined_code_parts.append("# --- Scientist Output ---\n")
            combined_code_parts.append(scientist_output_raw)
            combined_code_parts.append("\n\n")
            
        final_python_script = "".join(combined_code_parts)
        
        human_message_content = "Here is the complete Python script generated by the workflow:\n\n"
        human_message_content += "```python\n"
        human_message_content += final_python_script
        human_message_content += "```\n"
        print("Interpreter compiled code and generated message for CALOR-IA Analyst review.")
        
        return {

            "analyst_llm_output": analyst_output_raw,
            "scientist_llm_output": scientist_output_raw,
            "messages": state.get("messages", []) + [AIMessage(content=human_message_content)],
            "interpreter_finished":True,
            "next_agent": "supervisor",
        }
    else:
        if analyst_code != None:
            
            combined_code_parts.append(analyst_code)
            combined_code_parts.append("\n# --- End of Analyst Script ---")
            
        if scientist_code != None:
            
            combined_code_parts.append(scientist_code)
            combined_code_parts.append("\n# --- End of Scientist Script ---")
    
    combined_code_parts.append("\n# --- End of Script ---")
    combined_code_parts.append("\n# Copy and paste the entire block above into your environment/notebook to run the workflow!\n")

    # Combine all parts into a single string representing the full script
    final_python_script = "".join(combined_code_parts)

    # Wrap the entire script in a single markdown code block for easy copy-pasting
    human_message_content = "Here is the complete Python script generated by the workflow, combining preprocessing and modeling steps:\n\n"
    human_message_content += "```python\n"  # Start markdown code block
    human_message_content += final_python_script
    human_message_content += "```\n"        # End markdown code block

    print("Interpreter compiled code and generated message for CALOR-IA Analyst review.")

    # Prepare the state update dictionary to return
    return {
        "analyst_output": analyst_output,
        "scientist_output": scientist_output,
        "analyst_llm_output": analyst_output_raw,
        "scientist_llm_output": scientist_output_raw,
        "messages": state.get("messages", []) + [AIMessage(content=human_message_content)],
        "interpreter_finished":True,
        "next_agent": "supervisor",  # Route to the review node
    }


data_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    google_api_key=secret_value_0,
    temperature = 0.3,
    max_output_tokens = 9000,          
)


# --- Define the new CALOR_IA_DATA_ANALYST_NODE ---
def CALOR_IA_DATA_ANALYST_NODE(state: GraphState) -> Dict[str, Any]:
    """
    Reviews the compiled Python script from the interpreter and generates a summary
    or concluding message for the user using an LLM.
    """
    print("\n--- CALOR-IA DATA ANALYST (Post-Compilation Review) ---")
    
    messages = state.get("messages", []) 
    analyst_output = state.get('analyst_output')
    # Try to get compiled script directly from state first
    analyst_code = analyst_output.get("code", "")
    final_answer_generated = state.get('final_answer_generated')

    if final_answer_generated:
        final_code = state.get('code_final')
        prompt_messages = [
        CALOR_IA_DATA_ANALYST_SYSINT,
        HumanMessage(content=f'''
        Assure or provide techniques that may increase the quality of  train, test = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv'),:\n\n```python\n{final_code}...\n```\n\n improve the dataset columns = {train.columns}  
        perform data augmentation or make more similar train data to test data 
        avoid tis kind of error:
        ValueError: could not convert string to float: male processing and apply feature engineering return full python code.
        ''')
        ]
        
        print("CALOR-IA Analyst preprocessing pipeline...")
        try:
            print("CALOR-IA Analyst: LLM response.")
            llm_response = data_llm.invoke(prompt_messages)
            print("CALOR-IA Analyst: LLM invocation successful.")
            summary_message_content = str(llm_response.content)
            print(f"CALOR-IA Analyst IA analyst: {summary_message_content}...")
        except Exception as e:
            print(f"--- CALOR-IA Analyst Runtime Error during LLM call ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            summary_message_content = f"An error occurred while generating the final summary: {e}"
            if not state.get('error'):
                 state['error'] = summary_message_content
        
        # Append the new summary message to the state
        updated_messages = messages + [AIMessage(content=summary_message_content)]
        print(f"CALOR-IA Analyst adding message to state: {summary_message_content}...")
        loop_process = True
        print("CALOR-IA Analyst routing to Supervisor.")
        return {
            "analyst_llm_output": summary_message_content,
            "message" : updated_messages, # Pass the script to other nodes
            "loop_process": loop_process,
            "next_agent": "supervisor"
        }
        
    # If no direct script found, try to extract from messages
    if not analyst_code:
        # Find the last message that might contain the compiled script
        last_message = messages[-1] if messages else None
        
        if last_message and isinstance(last_message, AIMessage):
            # Attempt to extract the markdown code block content
            match = re.search(r"```python\n(.*)```", last_message.content, re.DOTALL)
            if match:
                compiled_script_content = match.group(1).strip()
                print("CALOR-IA Analyst found compiled script in message.")
            else:
                print("CALOR-IA Analyst: Could not find script markdown block in last message.")
                # Try to find messages with the script data in another format
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and "# Combined Data Science and Data Analyst" in msg.content:
                        compiled_script_content = msg.content
                        print("CALOR-IA Analyst found script content in a message without markdown.")
                        break
                
                if not compiled_script_content:
                    compiled_script_content = last_message.content.strip() if last_message.content else "No content found in last message."
        else:
            print("CALOR-IA Analyst: Last message was not an AI message or state was empty.")
            compiled_script_content = "Error: Could not retrieve compiled script content."


    
    # Store the found script for other nodes to use
    state["analyst_code"] = analyst_code
    
    # Prepare prompt for the LLM to summarize the script
    prompt_messages = [
        CALOR_IA_DATA_ANALYST_SYSINT,
        HumanMessage(content=f"provide data audmentation  :\n\n```python\n{analyst_code}...\n```\n\n columns = {train.columns} provide executable python preprocess pipeline for kaggle.")
    ]

    print("CALOR-IA Analyst invoking LLM for summary...")
    try:
        print("CALOR-IA Analyst: LLM response.")
        llm_response = data_llm.invoke(prompt_messages)
        print("CALOR-IA Analyst: LLM invocation successful.")
        summary_message_content = str(llm_response.content)
        print(f"CALOR-IA Analyst IA analyst: {summary_message_content}...")
    except Exception as e:
        print(f"--- CALOR-IA Analyst Runtime Error during LLM call ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        summary_message_content = f"An error occurred while generating the final summary: {e}"
        if not state.get('error'):
             state['error'] = summary_message_content

    # Append the new summary message to the state
    updated_messages = messages + [AIMessage(content=summary_message_content)]
    print(f"CALOR-IA Analyst adding message to state: {summary_message_content[:200]}...")

    print("CALOR-IA Analyst routing to Supervisor.")
    return {
        "analyst_llm_output": summary_message_content,
        "message" : updated_messages, # Pass the script to other nodes
        "next_agent": "Supervisor"
    }


sci_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    google_api_key=secret_value_0,
    temperature = 0.3,
    max_output_tokens = 9000,          
)

# --- Define the new CALOR_IA_DATA_ANALYST_NODE ---
def CALOR_IA_DATA_SCIENTIST_NODE(state: GraphState) -> Dict[str, Any]:
    """
    Reviews the compiled Python script from the interpreter and generates a summary
    or concluding message for the user using an LLM.
    """
    print("\n--- CALOR_IA_DATA_SCIENTIST (Post-Compilation Review) ---")
    
    scientist_output = state.get("scientist_output")
    
    compiled_script_content = scientist_output.get("code")


    loop_process = state.get('loop_process')
    final_answered_genarated = state.get('final_answered_generated')
    analyst_llm_output = state.get('analyst_llm_output')

    
    if not loop_process and final_answered_genarated:
        prompt_message = [
            CALOR_IA_DATA_SCIENTIST_SYSINT,
            HumanMessage(content=f'Assure cross validation is perfecly applied,    Assure optuna is applied and assure it will take just the first 3 enhacement or use a eraly stop method  {analyst_llm_output} avoid this error ValueError: Input array must be 1 dimensional, assure pd.to_csv(submission.csv, infex=False) exist and python script is complete')
        ]
        try:
            print("CALOR-IA Scientist: LLM response.")
            llm_response = sci_llm.invoke(prompt_messages)
            print("CALOR-IA Scientist: LLM invocation successful.")
            summary_message_content = str(llm_response.content)
            print(f"CALOR_IA_DATA_SCIENTIST LLM response content: {summary_message_content}...")
        except Exception as e:
            print(f"--- CALOR_IA_DATA_SCIENTIST Runtime Error during LLM call ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            summary_message_content = f"An error occurred while generating the final summary: {e}"
            if not state.get('error'):
                 state['error'] = summary_message_content

        updated_messages = state.get("messages", []) + [AIMessage(content=summary_message_content)]
        print(f"CALOR_IA_DATA_SCIENTIST adding message to state: {summary_message_content}...")

            
        print("CALOR-IA Scientist routing to Supervisor.")
        return {
            "scientist_llm_output": summary_message_content,
            "loop_process": False,
            'loop': loop,
            "message": updated_messages,  # Pass the script forward
            "next_agent": "Supervisor"
        }
        
    # If not found directly, try to get from messages
    if not compiled_script_content:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if last_message and isinstance(last_message, AIMessage):
            # Attempt to extract the markdown code block content
            match = re.search(r"```python\n(.*)```", last_message.content, re.DOTALL)
            if match:
                compiled_script_content = match.group(1).strip()
                print("CALOR-IA Scientist found compiled script.")
            else:
                print("CALOR-IA Scientist: Could not find script markdown block in last message.")
                # Try to find script in other messages
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and "# Combined Data Science and Data Analyst" in msg.content:
                        compiled_script_content = msg.content
                        print("CALOR-IA Scientist found script content in a message without markdown.")
                        break
                
                if not compiled_script_content:
                    compiled_script_content = last_message.content.strip() if last_message.content else "No content found in last message."
        else:
            print("CALOR-IA Scientist: Last message was not an AI message or state was empty.")
            compiled_script_content = "Error: Could not retrieve compiled script content."

    # Prepare prompt for the LLM to summarize the script
    prompt_messages = [
        CALOR_IA_DATA_SCIENTIST_SYSINT,
        HumanMessage(content=f" Apply cross validation or related technique to reduce metric RMSLE,    Assure optuna is applied and assure it will take just the first 3 enhacement or use a eraly stop method   :\n\n```python\n{compiled_script_content}...\n```\n\n thiis code provided has RMSLE = 0.0583 you must beat it; Please provide executable python preprocess pipeline for kaggle.")
    ]

    print("CALOR_IA_DATA_SCIENTIST...")
    try:
        print("CALOR-IA Scientist: LLM response.")
        llm_response = sci_llm.invoke(prompt_messages)
        print("CALOR-IA Scientist: LLM invocation successful.")
        summary_message_content = str(llm_response.content)
        print(f"CALOR_IA_DATA_SCIENTIST LLM response content: {summary_message_content}...")
    except Exception as e:
        print(f"--- CALOR_IA_DATA_SCIENTIST Runtime Error during LLM call ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        summary_message_content = f"An error occurred while generating the final summary: {e}"
        if not state.get('error'):
             state['error'] = summary_message_content

    # Append the new summary message to the state
    updated_messages = state.get("messages", []) + [AIMessage(content=summary_message_content)]
    print(f"CALOR_IA_DATA_SCIENTIST adding message to state: {summary_message_content}...")

    print("CALOR-IA Scientist routing to Supervisor.")
    return {
        "scientist_llm_output": summary_message_content,
        "message": updated_messages,  # Pass the script forward
        "next_agent": "Supervisor"
    }


code_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    google_api_key=secret_value_0,
    temperature = 0.1,
    max_output_tokens = 12000,          
)


def CODE_COMPILER_IA(state: GraphState) -> Dict[str, Any]:
    """
    Combines the preprocessing script with the model training script and outputs the final result.
    """
    analyst_output = state.get("analyst_llm_output")
    scientist_output = state.get("scientist_llm_output")
    
    print("\n--- CODE_COMPILER_IA (Combining Scripts) ---")
    
    loop_process = state.get('loop_process')
    final_answered_generated = state.get('final_answered_generated')
    loop = state.get('loop')        

    if loop == 0:
        prompt_messages = [
        CALOR_IA_CODE_COMPILER_SYSINT,
        HumanMessage(content=(f"""
        train, test = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv'), pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv'), 
        you have to set up python code ready to run on Kaggle:
        
        {analyst_output}.) if multiple pipelines chose just one, assure categorical and numerical data is separate before processing
        
        review ans set upin order to work with:
       
        {scientist_output} 
        # assure 
   # assure pd.to_csv(submission.csv, infex=False) exist and python script is complete
    """  )
                    )]
                     
        try:
            print("CODE_COMPILER_IA: LLM response.")
            llm_response = code_llm.invoke(prompt_messages)
            print("CODE_COMPILER_IA: LLM invocation successful.")
            summary_message_content = str(llm_response.content)
            print(f"CALOR_IA_DATA_SCIENTIST LLM response content: {summary_message_content}...")
        except Exception as e:
            print(f"--- CALOR_IA_DATA_SCIENTIST Runtime Error during LLM call ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            summary_message_content = f"An error occurred while generating the final summary: {e}"
            
        if not state.get('error'):
            state['error'] = summary_message_content
            
        updated_messages = state.get("messages", []) + [AIMessage(content=summary_message_content)]
        
        return {
                "final_answered_generated": summary_message_content,
                "code_final": updated_messages,
                "loop_process": True,
                'loop': loop,# Pass the script forward
                "next_agent": "supervisor"
                    }                     
        print("CODE_COMPILER_IA: Combined script generated successfully.")
    if loop_process and final_answered_generated:

        prompt_messages = [
        CALOR_IA_CODE_COMPILER_SYSINT,
        HumanMessage(content=(f"""
        
        you have to set up python code ready to run on Kaggle:
        apply all improvements processed on pipelines
        data_analyst pipeline
        assure id columns are always handle as id = 'id'
        Assure no errors on data types
        Assure optuna is applied and assure it will take just the first 3 enhacement or use a eraly stop method 
        review
        
        {analyst_output}.)
        
        review
       
        {scientist_output}
        
   # assure models are enssamble on defined correctly
    # 
    """  )
                    )]
                     
        try:
            print("CODE_COMPILER_IA: LLM response.")
            llm_response = code_llm.invoke(prompt_messages)
            print("CODE_COMPILER_IA: LLM invocation successful.")
            summary_message_content = str(llm_response.content)
            print(f"CALOR_IA_DATA_SCIENTIST LLM response content: {summary_message_content}...")
        except Exception as e:
            print(f"--- CALOR_IA_DATA_SCIENTIST Runtime Error during LLM call ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            summary_message_content = f"An error occurred while generating the final summary: {e}"
            
        if not state.get('error'):
            state['error'] = summary_message_content

        
        
        # Return the complete combined script
        return {
            "messages": messages + [AIMessage(content=summary_message_content)],
            "code_final": summary_message_content,
            "loop_process": False,
            'loop': loop,
            "final_answer_generated": True,
            "next_agent": "Supervisor"
            }
    # Get modeling script - try to find in scientist_output

    
    messages = state.get("messages", [])
    
    # If we couldn't find modeling script in state, search through messages
    if not scientist_output and not analyst_output:
        # Find preprocessing script first (from analyst)
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and "```python" in msg.content and ("CALOR-IA Analyst: LLM response." or "CALOR-IA Scientist: LLM response.") in msg.content:
                match = re.search(r"```python\n(.*?)```", msg.content, re.DOTALL)
                if match:
                    preprocessing_script = match.group(1).strip()
                    print("CODE_COMPILER_IA found preprocessing script in message.")
                    break
    
    # Create the combined script
    prompt_messages = [
        CALOR_IA_CODE_COMPILER_SYSINT,
        HumanMessage(content=(f"""
        
        you have to set up python code ready to run on Kaggle:
        
        data_analyst pipeline
        #assure data analyst process
        {analyst_output}.)
        data)scientist pipeline
        {scientist_output}
    ] # assure other models not loss expected 5 diferent models: 
    # assure random forest and cat bost where add to the pipepline also print to check on their progress of execution as before
    Becareful with output.shape before to blend the predictions and assure predict in the correct scaled dataset
    

    """)
                    )]
    try:
        print("CALOR-IA Scientist: LLM response.")
        llm_response = code_llm.invoke(prompt_messages)
        print("CALOR-IA Scientist: LLM invocation successful.")
        summary_message_content = str(llm_response.content)
        print(f"CALOR_IA_DATA_SCIENTIST LLM response content: {summary_message_content}...")
    except Exception as e:
        print(f"--- CALOR_IA_DATA_SCIENTIST Runtime Error during LLM call ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        summary_message_content = f"An error occurred while generating the final summary: {e}"
        if not state.get('error'):
            state['error'] = summary_message_content

   
                     
    print("CODE_COMPILER_IA: Combined script generated successfully.")

    # Return the complete combined script
    return {
        "messages": messages + [AIMessage(content=summary_message_content)],
        "code_final": summary_message_content,
        "final_answer_generated": True,
        "next_agent": "Supervisor"
    }


workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("DataAnalyst", data_analyst_node)
workflow.add_node("DataAnalyst_AI", CALOR_IA_DATA_ANALYST_NODE)
workflow.add_node("DataSciientist_AI", CALOR_IA_DATA_SCIENTIST_NODE)
workflow.add_node("DataScientist", data_scientist_node)
workflow.add_node("HumanInterpreter", bot_to_human_interpreter) # Use this exact name
workflow.add_node("code_compiler_AI", CODE_COMPILER_IA)
# Define entry point
workflow.set_entry_point("Supervisor")

# --- ADD THESE EDGES ---
# After Analyst runs, go back to Supervisor
workflow.add_edge("DataAnalyst", "Supervisor")

workflow.add_edge("DataScientist", "Supervisor")

workflow.add_edge("HumanInterpreter", "Supervisor")

workflow.add_edge("DataAnalyst_AI", "Supervisor")

workflow.add_edge("DataSciientist_AI", "Supervisor")

workflow.add_edge("code_compiler_AI", "Supervisor")

# --- END OF ADDED EDGES ---

# Define conditional edges FROM SUPERVISOR ONLY
workflow.add_conditional_edges(
    "Supervisor",
    # Function to decide route based on supervisor's decision
    lambda state: state.get("next_agent"),
    # Mapping decision to node name
    {
        "DataAnalyst": "DataAnalyst",
        "DataScientist": "DataScientist",
        "HumanInterpreter": "HumanInterpreter", # Ensure this matches add_node name
        "DataAnalyst_AI": "DataAnalyst_AI",
        "DataScientist_AI": "DataSciientist_AI",
        "code_compiler_AI":"code_compiler_AI",
        "Supervisor": "Supervisor", # Allow looping back if needed (e.g., waiting)
        "__end__": END # Map "__end__" string to the graph's end state
    }
)

# Compile the graph
app = workflow.compile()


import json # Make sure json is imported if you haven't already

# Assuming imports, node definitions, graph setup, and compilation are done above

# --- Run the Graph ---

# Initial state setup (unchanged)
initial_input_message = f""" Start working ğŸ’ª
"""
initial_state = {
    "messages": [HumanMessage(content=initial_input_message)],
    "final_answer_generated": False,
    "current_task_description": "Preprocess and analyze training and test data.", 
}

print("\n--- STARTING WORKFLOW (Using DataFrame Variable Names) ---")

# Keep track of seen message contents to avoid reprinting supervisor messages repeatedly
seen_message_contents = set()


for step, event in enumerate(app.stream(initial_state, {"recursion_limit":40})):
    print(f"\n--- Workflow Step {step + 1} ---")
    for node_name, update in event.items():
        print(f"Processing update from node: {node_name}")

        # Check if the update is valid and process messages
        if update is None:
            print("Node returned None, no state update.")
            continue # Skip processing if the node returned None

        # Process messages added in this update
        if 'messages' in update:
            new_messages_in_update = [
                msg for msg in update['messages']
                if isinstance(msg, (AIMessage, HumanMessage)) and msg.content not in seen_message_contents
            ]
            for msg in new_messages_in_update:
                if isinstance(msg, AIMessage):
                    print(f"ğŸ¤– {node_name} says: {msg.content}")
                elif isinstance(msg, HumanMessage):
                    print(f"ğŸ§‘â€�ğŸ’» User says (via state): {msg.content}") # User messages might reappear if state is passed
                seen_message_contents.add(msg.content)


        # The HumanInterpreter output is already formatted with the code block
        # We don't need special processing here if it's added to messages
        # The loop above processing 'messages' will print it when the HumanInterpreter runs
        # If you wanted to handle HumanInterpreter output *differently* here, you could add:
        # if node_name == 'HumanInterpreter':
        #     # Access the state after the update to get the new message
        #     final_message = update.get('messages', [])[-1] if update.get('messages') else None
        #     if final_message and isinstance(final_message, AIMessage):
        #          print("\n--- FINAL REPORT ---")
        #          print(final_message.content) # Print the full markdown content
        #          print("--- END FINAL REPORT ---")


    print("\n" + "="*50 + "\n")

# After the loop finishes, the graph execution has completed (__end__ reached or limit hit)
print("\n--- Workflow Finished ---")




import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Import TensorFlow components if available
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow not available. TensorFlow model will be skipped.")
    tf = None # Set tf to None to check availability later

# --- Helper Functions ---
def create_features(df):
    '''
    Create advanced features from the input data.
    This function serves as a form of "data augmentation" for tabular data
    by deriving new, potentially more informative features from existing ones.
    '''
    df = df.copy()
    epsilon = 1e-6 # Small constant to avoid division by zero

    # --- Basic Interaction & Ratio Features (>= 6 columns engineered) ---

    # 1. Body Mass Index (BMI): Standard health metric, relates weight and height.
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2 + epsilon)

    # 2. Duration * Heart_Rate: Represents total heartbeats during exercise, proxy for effort.
    df['Duration_HR'] = df['Duration'] * df['Heart_Rate']

    # 3. Duration * Body_Temp: Represents total heat generated during exercise, proxy for intensity/effort.
    df['Duration_Temp'] = df['Duration'] * df['Body_Temp']

    # 4. Weight / Height Ratio: Simpler ratio than BMI, might capture different linear relationships.
    df['Weight_Height_Ratio'] = df['Weight'] / (df['Height'] + epsilon)

    # 5. Heart Rate per Minute: Already captured by Heart_Rate, but could consider rate relative to duration?
    #    Let's create Heart_Rate_per_Duration instead.
    df['Heart_Rate_per_Duration'] = df['Heart_Rate'] / (df['Duration'] + epsilon) # Rate of heartbeats per minute of exercise

    # 6. Body Temperature per Minute: Change in temp per minute of exercise.
    df['Body_Temp_per_Duration'] = df['Body_Temp'] / (df['Duration'] + epsilon)

    # --- More Complex Interaction Features ---

    # 7. Exercise Intensity: Combines HR, Duration, and inversely related to Age (older might have lower max HR).
    df['Exercise_Intensity'] = (df['Heart_Rate'] * df['Duration']) / (df['Age'] + epsilon)

    # 8. Metabolic Factor: Combines BMI, Duration, HR, and inversely related to Age. A complex interaction term.
    df['Metabolic_Factor'] = (df['BMI'] * df['Duration'] * df['Heart_Rate']) / (df['Age'] + epsilon)

    # 9. Age * Duration Interaction: Older individuals exercising for longer might have different calorie burn patterns.
    df['Age_Duration_Interaction'] = df['Age'] * df['Duration']

    # 10. Height * Weight Product: Simple interaction, might capture overall body size effect differently than BMI.
    df['Height_Weight_Product'] = df['Height'] * df['Weight']

    # --- Polynomial Features (Example for Duration) ---
    # Captures non-linear relationship with Duration.
    df['Duration_Sq'] = df['Duration']**2

    # --- Categorical Features (Discretization) ---

    # 11. Age Group: Discretizing Age can help capture non-linear effects or group-specific patterns.
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'], right=False)
    df['Age_Group'] = df['Age_Group'].astype(object).fillna('Unknown')


    # 12. BMI Category: Standard health categories for BMI.
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 35, 40, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III'], right=False)
    df['BMI_Category'] = df['BMI_Category'].astype(object).fillna('Unknown')


    # 13. Heart Rate Zones (Simplified): Based on max HR (approx 220-Age) and exercise intensity.
    df['Max_HR_Estimate'] = 220 - df['Age']
    df['HR_Zone'] = pd.cut(df['Heart_Rate'] / (df['Max_HR_Estimate'] + epsilon),
                           bins=[0, 0.6, 0.7, 0.8, 0.9, 1.1], # Example zones based on % of max HR
                           labels=['<60%', '60-70%', '70-80%', '80-90%', '>90%'],
                           right=False)
    df['HR_Zone'] = df['HR_Zone'].astype(object).fillna('Unknown')
    df = df.drop('Max_HR_Estimate', axis=1) # Drop intermediate column


    # 14. Duration Category: Simple bins for exercise duration.
    df['Duration_Category'] = pd.cut(df['Duration'], bins=[0, 15, 30, 45, 60, 120, 300], labels=['<15min', '15-30min', '30-45min', '45-60min', '1-2hr', '>2hr'], right=False)
    df['Duration_Category'] = df['Duration_Category'].astype(object).fillna('Unknown')


    # Interaction between Sex and Duration/HeartRate/BodyTemp
    df['Sex_Male_Duration'] = df['Duration'] * (df['Sex'] == 'M')
    df['Sex_Female_Duration'] = df['Duration'] * (df['Sex'] == 'F')
    df['Sex_Male_HR'] = df['Heart_Rate'] * (df['Sex'] == 'M')
    df['Sex_Female_HR'] = df['Heart_Rate'] * (df['Sex'] == 'F')


    return df

# TensorFlow specific helper functions
if tf:
    def swish(x):
        return x * tf.keras.backend.sigmoid(x)

    def root_mean_squared_error(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def build_swish_mlp(input_shape):
        inputs = Input(shape=input_shape)
        x = Dense(64, activation=swish)(inputs)
        x = Dense(128, activation=swish)(x)
        x = Dense(64, activation=swish)(x)
        x = Dense(1)(x)
        model = Model(inputs, x)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return model

# rmsle function from sklearn.metrics
def rmsle(y_true, y_pred):
    # Ensure predictions are non-negative for log calculation
    y_pred = np.maximum(0, y_pred)
    # Ensure inputs are numpy arrays and handle potential shape issues
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # print(f"Debug RMSLE shapes - True: {y_true.shape}, Pred: {y_pred.shape}") # Debug print
    if y_true.shape != y_pred.shape:
        print(f"Warning: Shape mismatch between true and predicted values ({y_true.shape} vs {y_pred.shape}). Cannot compute RMSLE.")
        return float('inf')
    # Add a small epsilon to avoid log(0)
    return np.sqrt(mean_squared_log_error(np.maximum(0, y_true) + 1e-9, y_pred + 1e-9))


# --- Load Data ---
print("Loading data...")
try:
    train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Ensure train.csv, test.csv, and sample_submission.csv are in the specified path.")
    # Exit or handle error appropriately in a real script
    # For this example, we'll create dummy data if files aren't found
    print("Creating dummy data for demonstration.")
    data = {
        'id': range(100),
        'Sex': np.random.choice(['M', 'F'], 100),
        'Age': np.random.randint(15, 80, 100),
        'Height': np.random.uniform(150, 190, 100),
        'Weight': np.random.uniform(50, 100, 100),
        'Duration': np.random.uniform(10, 60, 100),
        'Heart_Rate': np.random.uniform(80, 180, 100),
        'Body_Temp': np.random.uniform(37, 40, 100),
        'Calories': np.random.uniform(50, 500, 100)
    }
    train = pd.DataFrame(data)
    test = pd.DataFrame(data).drop('Calories', axis=1).head(50) # Smaller test set
    submission = pd.DataFrame({'id': test['id'], 'Calories': 0})


# --- Separate Target and Drop ID ---
print("Separating target variable and dropping 'id' column...")
y_train_original = train['Calories']
train_ids = train['id'] # Keep train ids if needed later, though not for training
test_ids = test['id'] # Keep test ids for submission

train_df_features = train.drop(['id', 'Calories'], axis=1)
test_df_features = test.drop('id', axis=1)

print(f"Initial train shape (features only): {train_df_features.shape}")
print(f"Initial test shape (features only): {test_df_features.shape}")
print(f"Target variable 'y_train_original' shape: {y_train_original.shape}")


# --- Feature Engineering (Data Augmentation for Tabular Data) ---
print("\nApplying feature engineering...")
X_train_fe = create_features(train_df_features)
X_test_fe = create_features(test_df_features)

print(f"Train shape after feature engineering: {X_train_fe.shape}")
print(f"Test shape after feature engineering: {X_test_fe.shape}")

# Identify numerical and categorical features AFTER feature engineering
# Ensure handling of potential errors if a feature was expected but not created
numerical_features = X_train_fe.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train_fe.select_dtypes(include='object').columns.tolist() # Using object dtype for pandas categorical/string

print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# --- Preprocessing Pipeline (KNN Imputation & Quantile Transformation + OneHot) ---
print("\nSetting up preprocessing pipeline...")

# Check if numerical/categorical features exist before creating pipelines
transformers = []
if numerical_features:
    numerical_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),       # KNN imputation
        ('scaler', QuantileTransformer(output_distribution='normal', n_quantiles=100)) # Robust scaling via QuantileTransformer
    ])
    transformers.append(('num', numerical_pipeline, numerical_features))
    print(f"Added numerical pipeline for {len(numerical_features)} features.")
else:
    print("No numerical features found. Skipping numerical pipeline.")

if categorical_features:
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)) # One-Hot Encoding
    ])
    transformers.append(('cat', categorical_pipeline, categorical_features))
    print(f"Added categorical pipeline for {len(categorical_features)} features.")
else:
     print("No categorical features found. Skipping categorical pipeline.")

# Create ColumnTransformer only if there are transformers
if transformers:
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    # Fit and Transform data
    print("Fitting preprocessor on engineered training data...")
    preprocessor.fit(X_train_fe)

    print("Transforming engineered training data...")
    X_train_processed = preprocessor.transform(X_train_fe)

    print("Transforming engineered test data...")
    X_test_processed = preprocessor.transform(X_test_fe)

    print(f"\nProcessed train shape: {X_train_processed.shape}")
    print(f"Processed test shape: {X_test_processed.shape}")

    # Get feature names after preprocessing (optional, mainly for inspection)
    try:
         processed_feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
         print(f"Warning: Could not get feature names after preprocessing: {e}")
         processed_feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]

    # Convert processed numpy arrays back to DataFrame for easier inspection/use if needed
    # X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)
    # X_test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)

    # print("\nProcessed data (first 5 rows of train):")
    # print(X_train_processed_df.head())

else:
    print("\nNo features identified for preprocessing. Processed data will be empty.")
    # Handle case where no features are processed - create empty arrays/DataFrames
    X_train_processed = np.empty((X_train_fe.shape[0], 0))
    X_test_processed = np.empty((X_test_fe.shape[0], 0))
    # X_train_processed_df = pd.DataFrame(X_train_processed)
    # X_test_processed_df = pd.DataFrame(X_test_processed)
    processed_feature_names = []
    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed test shape: {X_test_processed.shape}")


print("\nData preprocessing complete.")

# --- Data Splitting ---
print("\nSplitting data into training and validation sets...")
# Use the processed train data variable name from the analyst node
# Assuming the target 'y_train_original' is available from the initial data loading/prep
if X_train_processed.shape[0] > 0 and X_train_processed.shape[0] == y_train_original.shape[0]:
    X_train, X_val, y_train, y_val = train_test_split(X_train_processed, y_train_original, test_size=0.2, random_state=42)
    print(f"Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"Validation shapes: {X_val.shape}, {y_val.shape}")
    data_split_success = True
else:
    print(f"Error: Processed train data ('X_train_processed') or target ('y_train_original') not found or shapes mismatch. Skipping modeling.")
    print(f"'X_train_processed' shape: {X_train_processed.shape}")
    print(f"'y_train_original' shape: {y_train_original.shape}")
    X_train, X_val, y_train, y_val = np.array([]).reshape(0,-1), np.array([]).reshape(0,-1), np.array([]), np.array([]) # Create empty arrays
    data_split_success = False


if data_split_success and X_train.shape[0] > 0:
    # --- Scaling for Neural Network ---
    # NN often performs better with features scaled to a 0-1 range and log-transformed target
    print("\nScaling features for Neural Network using MinMaxScaler...")
    scaler_nn = MinMaxScaler()
    X_train_scaled = scaler_nn.fit_transform(X_train)
    X_val_scaled = scaler_nn.transform(X_val)
    # Scale the actual test set using the processed test data variable name
    if X_test_processed.shape[0] > 0:
       X_test_scaled = scaler_nn.transform(X_test_processed)
       test_scaling_success = True
    else:
       print(f"Error: Processed test data ('X_test_processed') not found for scaling.")
       X_test_scaled = np.array([]).reshape(0,-1)
       test_scaling_success = False


    print("Log transforming target for NN...")
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    # --- Model Training ---
    models = ['lgb', 'xgb', 'rf', 'cat', 'tf'] # Define models list for ensemble loop

    # Train LightGBM (using split processed data - numpy arrays)
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=2000, learning_rate=0.03, num_leaves=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=0.1, seed=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_val_pred = lgb_model.predict(X_val)
    # Predict on the processed test data using the variable name
    if X_test_processed.shape[0] > 0:
       lgb_test_pred = lgb_model.predict(X_test_processed)
    else:
       lgb_test_pred = np.zeros(X_test_processed.shape[0]) # Predict zeros if test data missing
       print(f"Warning: 'X_test_processed' not available for LightGBM test prediction.")
    print("LightGBM training complete.")

    # Train XGBoost (using split processed data - numpy arrays)
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_estimators=2000, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.8, lambda_l1=0.1, lambda_l2=0.1, seed=42, n_jobs=-1, tree_method='hist') # Use hist for faster training
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    xgb_val_pred = xgb_model.predict(X_val)
    # Predict on the processed test data using the variable name
    if X_test_processed.shape[0] > 0:
       xgb_test_pred = xgb_model.predict(X_test_processed)
    else:
        xgb_test_pred = np.zeros(X_test_processed.shape[0])
        print(f"Warning: 'X_test_processed' not available for XGBoost test prediction.")
    print("XGBoost training complete.")

    # Train RandomForestRegressor (using split processed data - numpy arrays)
    print("\nTraining RandomForestRegressor...")
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_val_pred = rf_model.predict(X_val)
    # Predict on the processed test data using the variable name
    if X_test_processed.shape[0] > 0:
        rf_test_pred = rf_model.predict(X_test_processed)
    else:
        rf_test_pred = np.zeros(X_test_processed.shape[0])
        print(f"Warning: 'X_test_processed' not available for RandomForest test prediction.")
    print("RandomForestRegressor training complete.")

    # Train CatBoostRegressor (using split processed data - numpy arrays)
    print("\nTraining CatBoostRegressor...")
    cat_model = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, random_seed=42, verbose=0, early_stopping_rounds=100, l2_leaf_reg=3, loss_function='RMSE')
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_val_pred = cat_model.predict(X_val)
    # Predict on the processed test data using the variable name
    if X_test_processed.shape[0] > 0:
        cat_test_pred = cat_model.predict(X_test_processed)
    else:
         cat_test_pred = np.zeros(X_test_processed.shape[0])
         print(f"Warning: 'X_test_processed' not available for CatBoost test prediction.")
    print("CatBoostRegressor training complete.")


    # Build and train TensorFlow model (using scaled data)
    if tf and test_scaling_success and X_train_scaled.shape[0] > 0 and X_train_scaled.shape[1] > 0: # Only attempt TF if TF is available, test data was scaled, and train data is not empty
        print("\nBuilding and training TensorFlow model...")
        tf_model = build_swish_mlp(input_shape=(X_train_scaled.shape[1],))
        if tf_model: # Check if model was built successfully
            early_stopping_tf = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=15, restore_best_weights=True, mode='min', verbose=0)

            print("  Training TensorFlow model...")
            history = tf_model.fit(X_train_scaled, y_train_log,
                                   epochs=200,
                                   batch_size=64,
                                   verbose=0,
                                   validation_data=(X_val_scaled, y_val_log),
                                   callbacks=[early_stopping_tf])
            print("  TensorFlow model training complete.")

            print("Making TensorFlow predictions...")
            tfy_test_pred_log = tf_model.predict(X_test_scaled).flatten()
            tf_test_pred = np.expm1(tfy_test_pred_log)
            # Clip to original target range (need 'y_train_original' min/max)
            tf_test_pred = np.clip(tf_test_pred, y_train_original.min(), y_train_original.max())


            tfy_val_pred_log = tf_model.predict(X_val_scaled).flatten()
            tf_val_pred = np.expm1(tfy_val_pred_log)
            tf_val_pred = np.clip(tf_val_pred, y_train_original.min(), y_train_original.max())
            print("TensorFlow predictions made.")
        else: # TF model build failed
            tf_test_pred = np.zeros(X_test_scaled.shape[0])
            tf_val_pred = np.zeros(X_val_scaled.shape[0])
            print("TensorFlow model could not be built. Skipping TF predictions.")

    else: # TF not available or data issues
        print("\nSkipping TensorFlow model training and prediction.")
        # Ensure prediction variables exist even if TF is skipped
        test_data_shape = X_test_processed.shape[0] if X_test_processed.shape[0] > 0 else test.shape[0] # Fallback shape if processed is empty
        tf_test_pred = np.zeros(test_data_shape)
        tf_val_pred = np.zeros(X_val.shape[0]) # Use X_val shape for validation predictions
        print("TensorFlow prediction variables initialized to zeros.")


    # --- Validation Scores ---
    print(f"\n--- Validation RMSLE Scores ---")
    if y_val.shape[0] > 0:
        # Check if prediction variables exist and have correct shape before computing scores
        if 'lgb_val_pred' in locals() and lgb_val_pred.shape == y_val.shape:
           print(f"LightGBM Validation RMSLE: {rmsle(y_val, lgb_val_pred):.5f}")
        else:
           print("LightGBM Validation RMSLE: N/A (predictions missing or shape mismatch)")

        if 'xgb_val_pred' in locals() and xgb_val_pred.shape == y_val.shape:
           print(f"XGBoost Validation RMSLE: {rmsle(y_val, xgb_val_pred):.5f}")
        else:
            print("XGBoost Validation RMSLE: N/A (predictions missing or shape mismatch)")

        if 'rf_val_pred' in locals() and rf_val_pred.shape == y_val.shape:
           print(f"RandomForest Validation RMSLE: {rmsle(y_val, rf_val_pred):.5f}")
        else:
            print("RandomForest Validation RMSLE: N/A (predictions missing or shape mismatch)")

        if 'cat_val_pred' in locals() and cat_val_pred.shape == y_val.shape:
            print(f"CatBoost Validation RMSLE: {rmsle(y_val, cat_val_pred):.5f}")
        else:
             print("CatBoost Validation RMSLE: N/A (predictions missing or shape mismatch)")

        if 'tf_val_pred' in locals() and tf_val_pred.shape == y_val.shape:
           print(f"TensorFlow Validation RMSLE: {rmsle(y_val, tf_val_pred):.5f}")
        else:
            print("TensorFlow Validation RMSLE: N/A (predictions missing or shape mismatch)")

    else:
        print("No validation data available to compute scores.")
    print(f"------------------------------")

    # --- Ensemble Prediction ---
    print("\nCreating ensemble prediction...")

    # Ensure all test prediction variables exist and have the same expected shape
    # The expected shape is the number of rows in the processed test data
    expected_shape = X_test_processed.shape[0] if X_test_processed.shape[0] > 0 else test.shape[0] # Fallback shape

    # List of models and their test predictions
    test_preds = {
        'lgb': lgb_test_pred if 'lgb_test_pred' in locals() and lgb_test_pred.shape[0] == expected_shape else np.zeros(expected_shape),
        'xgb': xgb_test_pred if 'xgb_test_pred' in locals() and xgb_test_pred.shape[0] == expected_shape else np.zeros(expected_shape),
        'rf':  rf_test_pred  if 'rf_test_pred'  in locals() and rf_test_pred.shape[0] == expected_shape else np.zeros(expected_shape),
        'cat': cat_test_pred if 'cat_test_pred' in locals() and cat_test_pred.shape[0] == expected_shape else np.zeros(expected_shape),
        'tf':  tf_test_pred  if 'tf_test_pred'  in locals() and tf_test_pred.shape[0] == expected_shape else np.zeros(expected_shape),
    }
    val_preds = {
        'lgb': lgb_val_pred if 'lgb_val_pred' in locals() and lgb_val_pred.shape == y_val.shape else np.zeros(y_val.shape[0]),
        'xgb': xgb_val_pred if 'xgb_val_pred' in locals() and xgb_val_pred.shape == y_val.shape else np.zeros(y_val.shape[0]),
        'rf':  rf_val_pred  if 'rf_val_pred'  in locals() and rf_val_pred.shape == y_val.shape else np.zeros(y_val.shape[0]),
        'cat': cat_val_pred if 'cat_val_pred' in locals() and cat_val_pred.shape == y_val.shape else np.zeros(y_val.shape[0]),
        'tf':  tf_val_pred  if 'tf_val_pred'  in locals() and tf_val_pred.shape == y_val.shape else np.zeros(y_val.shape[0]),
    }

    # Calculate weights based on validation performance (using RMSLE)
    # Avoid division by zero or using infinite RMSLE from errors
    val_rmsles = {
        'lgb': rmsle(y_val, val_preds['lgb']) if y_val.shape[0] > 0 and np.sum(np.abs(val_preds['lgb'])) > 0 else float('inf'),
        'xgb': rmsle(y_val, val_preds['xgb']) if y_val.shape[0] > 0 and np.sum(np.abs(val_preds['xgb'])) > 0 else float('inf'),
        'rf':  rmsle(y_val, val_preds['rf'])  if y_val.shape[0] > 0 and np.sum(np.abs(val_preds['rf'])) > 0 else float('inf'),
        'cat': rmsle(y_val, val_preds['cat']) if y_val.shape[0] > 0 and np.sum(np.abs(val_preds['cat'])) > 0 else float('inf'),
        'tf':  rmsle(y_val, val_preds['tf'])  if y_val.shape[0] > 0 and np.sum(np.abs(val_preds['tf'])) > 0 and tf is not None else float('inf'), # Only include TF if available
    }

    # Calculate inverse RMSLE (higher for better models), handle inf/zero
    inv_rmsles = {k: (1.0 / v if v != 0 and v != float('inf') else 0) for k, v in val_rmsles.items()}

    # Normalize to get weights
    total_inv = sum(inv_rmsles.values())
    if total_inv > 1e-9: # Avoid division by near zero
        weights = {k: v / total_inv for k, v in inv_rmsles.items()}
    else: # If all models failed or had infinite RMSLE, use equal weights or zero weights
        print("Warning: Cannot calculate meaningful ensemble weights (all models failed or infinite RMSLE). Using equal weights as fallback.")
        valid_models = [m for m, inv in inv_rmsles.items() if inv > 0]
        if valid_models:
             equal_weight = 1.0 / len(valid_models)
             weights = {m: equal_weight for m in models} # Initialize all weights to 0 first
             for m in valid_models: # Assign equal weight to valid models
                 weights[m] = equal_weight
        else:
             print("Warning: No models trained successfully. Ensemble prediction will be zero.")
             weights = {m: 0 for m in models}


    print(f"Model weights: {weights}")

    # Calculate weighted ensemble predictions
    ensemble_predictions = np.zeros(expected_shape)
    for model in models:
        ensemble_predictions += weights[model] * test_preds[model]


    # Ensure predictions are non-negative (calories can't be negative)
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    # --- Create Submission ---
    # Assume 'submission' DataFrame with an 'id' column is available from initial loading
    print("\nPreparing submission file...")
    if 'submission' in locals() and 'id' in submission.columns and submission.shape[0] == expected_shape:
        submission['Calories'] = ensemble_predictions
        submission.to_csv('submission.csv', index=False)
        print("Submission file created: submission.csv")
    else:
        print("Error: Submission DataFrame not found, invalid, or shape mismatch. Cannot create submission file.")
        print("Details:")
        if 'submission' not in locals(): print("'submission' variable not found.")
        elif 'id' not in submission.columns: print("'submission' DataFrame missing 'id' column.")
        elif submission.shape[0] != expected_shape: print(f"Submission shape ({submission.shape[0]}) does not match test prediction shape ({expected_shape}).")


    print("\nProcess complete!")

else:
    print("\nSkipping model training, validation, and submission due to data split failure or empty training data.")
    # Ensure ensemble_predictions variable exists for potential return, even if zero
    expected_shape = X_test_processed.shape[0] if X_test_processed.shape[0] > 0 else test.shape[0] # Fallback shape
    ensemble_predictions = np.zeros(expected_shape)
    # Create a dummy submission file if possible
    if 'submission' in locals() and 'id' in submission.columns and submission.shape[0] == expected_shape:
         submission['Calories'] = ensemble_predictions
         submission.to_csv('submission.csv', index=False)
         print("Created dummy submission.csv with zero predictions due to processing/modeling failure.")
    else:
         print("Could not create submission file due to missing data or shape mismatch.")




