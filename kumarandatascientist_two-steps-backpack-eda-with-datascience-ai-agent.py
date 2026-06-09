!cp -r /kaggle/input/edadatascienceaiagent/other/default/1/* /kaggle/working

import sys 
sys.path.append('/kaggle/working/EDADatascienceAIAgent.py')

from EDADatascienceAIAgent import EDADatascienceAIAgent



if __name__ == "__main__":
    # Specify the path to your dataset (CSV format)
    filepath = "/kaggle/input/playground-series-s5e2/train.csv"  # Replace with your dataset path
    
    # Instantiate the AI agent
    agent = EDADatascienceAIAgent(filepath)
    
    # Run the agent's methods
    agent.load_data()
    agent.clean_data()
    agent.perform_eda()
    agent.visualize_data()
    #agent.additional_visualizations()

