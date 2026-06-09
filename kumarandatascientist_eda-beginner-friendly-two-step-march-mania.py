!cp -r /kaggle/input/edadatascienceaiagent/other/default/2/* /kaggle/working

import sys 
sys.path.append('/kaggle/working/EDADatascienceAIAgent_v2.py')

from EDADatascienceAIAgent_v2 import EDADatascienceAIAgent_v2



if __name__ == "__main__":
    # Specify the path to your dataset (CSV format)
    filepath = "/kaggle/input/march-machine-learning-mania-2025/MGameCities.csv"  # Replace with your dataset path

    # Instantiate the AI agent
    agent = EDADatascienceAIAgent_v2(filepath)
    
    # Run the agent's methods
    agent.load_data()
    agent.clean_data()
    agent.perform_eda()
    agent.visualize_data()
    
    # Optionally, print available graph options and configurations
    agent.print_graph_options()
    
    # Optional: Pass an optional dictionary of parameters to customize the EDA plots.
    custom_options = {
        'violin': {'figsize': (7, 5), 'color': 'cyan'},
        'kde': {'figsize': (7, 5), 'shade': True, 'color': 'magenta'},
        'joint': {'kind': 'hex'},
        'reg': {'figsize': (7, 5), 'color': 'darkred'},
        # ... add other custom options as desired.
    }
    agent.additional_visualizations(options=custom_options)




if __name__ == "__main__":
    # Specify the path to your dataset (CSV format)
    filepath = "/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv"  # Replace with your dataset path
    
    # Instantiate the AI agent
    agent = EDADatascienceAIAgent(filepath)
    
    # Run the agent's methods
    agent.load_data()
    agent.clean_data()
    agent.perform_eda()
    agent.visualize_data()
    #agent.additional_visualizations()



if __name__ == "__main__":
    # Specify the path to your dataset (CSV format)
    filepath = "/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv"  # Replace with your dataset path

    # Instantiate the AI agent
    agent = EDADatascienceAIAgent_v2(filepath)
    
    # Run the agent's methods
    agent.load_data()
    agent.clean_data()
    agent.perform_eda()
    agent.visualize_data()
    
    # Optionally, print available graph options and configurations
    agent.print_graph_options()
    
    # Optional: Pass an optional dictionary of parameters to customize the EDA plots.
    custom_options = {
        'violin': {'figsize': (7, 5), 'color': 'cyan'},
        'kde': {'figsize': (7, 5), 'shade': True, 'color': 'magenta'},
        'joint': {'kind': 'hex'},
        'reg': {'figsize': (7, 5), 'color': 'darkred'},
        # ... add other custom options as desired.
    }
    agent.additional_visualizations(options=custom_options)




if __name__ == "__main__":
    # Specify the path to your dataset (CSV format)
    filepath = "/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv"  # Replace with your dataset path

    # Instantiate the AI agent
    agent = EDADatascienceAIAgent_v2(filepath)
    
    # Run the agent's methods
    agent.load_data()
    agent.clean_data()
    agent.perform_eda()
    agent.visualize_data()
    
    # Optionally, print available graph options and configurations
    agent.print_graph_options()
    
    # Optional: Pass an optional dictionary of parameters to customize the EDA plots.
    custom_options = {
        'violin': {'figsize': (7, 5), 'color': 'cyan'},
        'kde': {'figsize': (7, 5), 'shade': True, 'color': 'magenta'},
        'joint': {'kind': 'hex'},
        'reg': {'figsize': (7, 5), 'color': 'darkred'},
        # ... add other custom options as desired.
    }
    agent.additional_visualizations(options=custom_options)





