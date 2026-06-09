#Remove conflicting packages from the Kaggle base environment.Â¶
!pip uninstall -qqy kfp jupyterlab libpysal thinc spacy fastai ydata-profiling google-cloud-bigquery google-generativeai
!pip install -qU 'langgraph==0.3.21' 'langchain-google-genai==2.1.2' 'langgraph-prebuilt==0.1.7'


import numpy as np 
import pandas as pd 
from typing_extensions import TypedDict
from typing import TypedDict, Annotated, Dict, Any
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from kaggle_secrets import UserSecretsClient
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GOOGLE_API_KEY")


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
ids = test['id']


class GraphState(TypedDict):
    """
    ğŸ—‚ï¸� Central state definition for the multi-agent workflow.
    """
    query: Annotated[BaseMessage, add_messages]
    code: str
    code_2: str
    code_3: str
    model1: Any
    model2: Any
    model3: Any
    result: str
    train: pd.DataFrame
    test: pd.DataFrame


class AgentModels: 

    @staticmethod
    def DataAnalyst(state: GraphState):
        
        llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=secret_value_0,
        temperature = 0.1,
        max_output_tokens = 10000,          
        )
        print('Start data analisys')
        prompt = f"""
        You are in data analyst pipeline and preprocess. 
        The user query: {state['query']}
        you can access to two functions 'AgentModels.get_data() and AgentModels.run_code()'
        When you call AgentModels.get_data or AgentModels.run_code, always pass the variable `state` as an argument.
        For example:
        state = AgentModels.get_data(state)
        you will prepare this two dataset in a pipeline before to build predictions
        models, in order to make them give the best results
        
        1. important drop test['id'] and train['id'].
        2. This data has no elements to concatenate
        3. the predicted column will be train['BeatsPerMinute'] this one doesn't exist in test set
        4. Search and handle outliers
        5. train and test must be processed in the same preprocess pipeline
        Write Python code (only code, no explanations)
        """

        response = llm.invoke(prompt)
        code = response.content.strip("```python").strip("```")
        state["code"] = code
        state["train"] = train
        state["test"] = test
        return state
        
    @staticmethod
    def DataScientist(state: GraphState):
        llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=secret_value_0,
        temperature = 0.1,
        max_output_tokens = 10000,
        )
        analist_code = state["code"]
        train = state['train']
        test = state['test']
        print('start data science')
        
        prompt = f"""
        Use joblib and glob to store models and predictions on disk to save RAM. 
        âš¡ Always use hard disk checkpoints instead of keeping everything in memory.
        
        Available functions:
        - AgentModels.get_data(state)  # Returns a dictionary with train and test DataFrames
        - AgentModels.run_code(state)  # Executes code saved in state
        Example:
            state = AgentModels.get_data(state)
        
        âš¡ VERY IMPORTANT:
        - Always pass `state` as an argument when calling AgentModels.get_data or AgentModels.run_code.
        - Avoid errors like:
            'dict' object has no attribute 'X_train_processed'
            'dict' object has no attribute 'train'
            'dict' object has no attribute 'test'
        
        âš¡ Data Info:
        - Target column: train['BeatsPerMinute']
        - Test set does not contain 'BeatsPerMinute' (predict this column).
        - Submissions must include: test['id'] and predicted 'BeatsPerMinute'.
        
        âš¡ Modeling Task:
        - Build **10 different regression models** using different techniques:
            1. Ridge Regression (L2 norm)
            2. Lasso Regression (L1 norm)
            3. Polynomial Features 
            4. Cross-Validated (e.g., KFold CV or ElasticNet)

        - RMSE error obtained by you in XGB and gradiendboost was 26.39 try to make it better
        
        âš¡ Evaluation:
        - Evaluate each model using RMSE on the training data (or via cross-validation).
        - Save each trained model as .pkl using joblib.
       
        âš¡ File Management:
        - Save model checkpoints with joblib:
            joblib.dump(model, "/kaggle/working/model_checkpoint.pkl")
            model = joblib.load("/kaggle/working/model_checkpoint.pkl")
        
        - Save batch predictions:
            batch_preds = model.predict(X_batch)
            pd.DataFrame({'"id": batch_ids, "BeatsPerMinute": batch_preds'}).to_csv(
                "/kaggle/working/preds_batch_1.csv", index=False
            )
    

        âš¡ Pipeline Instructions:
        
        1. Load and preprocess data safely.
        2. Train multiple models and reduce RMSE (consider blending/stacking).
        3. Use joblib checkpoints to save and reload models/predictions to minimize RAM usage.
        4. Save batch predictions to disk and later concatenate for the final submission.
        
        âš¡ Preprocessing:
        Use the following preprocessing pipeline:\n{analist_code}
        
        âš¡ Final Instructions:
        - Return **only valid Python code**.
        - Import all necessary libraries.
        - Ensure reproducibility and memory efficiency.
        """

        response = llm.invoke(prompt)
        code = response.content.strip("```python").strip("```")
        state["code_2"] = code
        return state

    @staticmethod
    def Enhacer(state: GraphState):
        
        llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=secret_value_0,
        temperature = 0.1,
        max_output_tokens = 10000,          
        )
        code_2 = state['code_2']
        print('RMSE reduction started')
        prompt = f"""
        Import all necessary libraries.

        Available functions:
        - AgentModels.get_data(state)  # Returns a dictionary with train and test DataFrames
        - AgentModels.run_code(state)  # Executes code saved in state
        Example:
            state = AgentModels.get_data(state)
            
        Avoid Errors:
        Error in Enhacer code: name 'state' is not defined
        Error in Enhacer code: unterminated string literal (detected at line 291) (<string>, line 291)

        Your task:
        - Optimize RMSE as much as possible.
        - Create final submissions to win a Kaggle competition.
        - Always return valid Python code only (no text).

        âš¡ Constraints:
        - Always save memory by offloading batches/models to disk.
        - Ensure reproducibility and valid submissions.
        - Return only working Python code.
                
        âš¡ Rules for AgentModels:
        - Always call `AgentModels.get_data(state)` and `AgentModels.run_code(state)` using `state` as an argument.

        - Reload batches for final submission:
            import glob
            all_files = glob.glob("/kaggle/working/preds_batch_*.csv")
            final_submission = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
            final_submission.to_csv("/kaggle/working/final_submission.csv", index=False)

        YOUR GOAL IS CREATE THE FINAL SUBMISSION USING WEIGHTS FOR A BLENDED OF SUBMISSION.csv USING GLOB
        to load the models of data scientist node {code_2}
        
        - Example:
            RMSE_reduced_model = pd.DataFrame({'"id": test["id"], "BeatsPerMinute": model_1_predictions'})
            RMSE_reduced_model.to_csv("submission_regretion_model1.csv", index=False)
        """
        response = llm.invoke(prompt)
        code = response.content.strip("```python").strip("```")
        state["code_3"] = code
        return state
        
    @staticmethod
    def get_data(state:GraphState):
        try:
            train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
            test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
            state["train"] = train
            state["test"] = test
            return state
        except Exception as e:
            state["result"] = f"Error loading data: {e}"
            return state
            
    @staticmethod
    def run_code(state:GraphState):
            print('running code')
            local_vars = {
                "train": state.get("train"),
                "test": state.get("test"),
                "state": state, 
                "AgentModels": AgentModels, 
            }
    
            # Run analyst code
            if state.get("code"):
                try:
                    exec(state["code"], globals(), local_vars)
                    state["train"] = local_vars.get("train")
                    state["test"] = local_vars.get("test")
                    state["result"] = "DataAnalyst code executed successfully."
                except Exception as e:
                    state["result"] = f"Error in DataAnalyst code: {e}"
    
            # Run scientist code
            if state.get("code_2"):
                try:
                    exec(state["code_2"], globals(), local_vars)
                    state["result"] = "DataScientist code executed successfully."
                except Exception as e:
                    state["result"] = f"Error in DataScientist code: {e}"
    
            if state.get("code_3"):
                try:
                    exec(state["code_3"], globals(), local_vars)
                    state["result"] = "Enhacer code executed successfully."
                except Exception as e:
                    state["result"] = f"Error in Enhacer code: {e}"
    
            return state


graph = StateGraph(GraphState)

graph.add_node("analyst", AgentModels.DataAnalyst)
graph.add_node("scientist", AgentModels.DataScientist)
graph.add_node("enhacer", AgentModels.Enhacer)
graph.add_node("run", AgentModels.run_code)

graph.set_entry_point("analyst")
graph.add_edge("analyst", "scientist")
graph.add_edge("scientist", "enhacer")
graph.add_edge("enhacer", "run")


app = graph.compile()


result = app.invoke({"query": "Start process"})


print(result['result']) #-> run to check errors




