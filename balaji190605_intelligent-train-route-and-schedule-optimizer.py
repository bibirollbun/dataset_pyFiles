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


import os
import heapq
from typing import List, Dict, Tuple, Any

# --- LLM and API Setup (Conceptual) ---
# In a real scenario, this key enables LLM-powered agents (RHA, RA).
os.environ['LLM_API_KEY'] = 'YOUR_GEMINI_API_KEY_HERE' 

# Simulate a Memory Bank (Long term memory)
# This stores user preferences for the Recommendation Agent (RA).
MEMORY_BANK = {
    'user_123': {'dislikes': ['Dadar transfer'], 'preference': 'Fastest time prioritized'}
}

# Minimal Train Network Data (Static Schedule)
# Format: (Origin, Destination, Travel Time in minutes)
TRAIN_NETWORK = [
    ('Kalyan', 'Dadar', 45),
    ('Dadar', 'CSMT', 20),
    ('Kalyan', 'Thane', 15),
    ('Thane', 'CSMT', 50),
    ('Kalyan', 'CSMT', 80),  # Direct Fast Train
]

print("Setup Complete. TRAIN_NETWORK and MEMORY_BANK loaded.")


class AStarRouteOptimizerTool:
    """
    Implements the core A* shortest path algorithm as a Custom Tool.
    This is the core optimization engine used by the OPA.
    """

    def __init__(self, network: List[Tuple]):
        self.graph = self._build_graph(network)

    def _build_graph(self, network: List[Tuple]) -> Dict[str, List[Tuple[str, int]]]:
        """Converts the list of routes into an adjacency list (graph)."""
        graph = {}
        for start, end, time in network:
            if start not in graph:
                graph[start] = []
            graph[start].append((end, time))
        return graph

    def find_best_route(self, start_stn: str, end_stn: str, constraint: str) -> List[Dict]:
        """
        Runs the A* search. Simulates the work of a Loop Agent 
        iterating to find the optimal path while applying penalties.
        """
        # (A* implementation using a priority queue)
        open_list = [(0, 0, start_stn, [start_stn])] 
        g_costs = {start_stn: 0}

        while open_list:
            f_cost, g_cost, current_stn, path = heapq.heappop(open_list)
            
            if current_stn == end_stn:
                # Long term memory check integrated for penalty calculation
                transfer_penalty = 0
                if 'Dadar' in path and 'hate Dadar' in constraint:
                    transfer_penalty = 30 # Apply 30 min penalty if user's constraint is violated
                
                total_cost = g_cost + transfer_penalty
                
                # Context Engineering: Summarize the path before returning
                return [{
                    'path': ' -> '.join(path),
                    'travel_time_min': g_cost,
                    'penalty_min': transfer_penalty,
                    'total_cost_min': total_cost,
                    'transfers': len(path) - 2
                }]

            if current_stn in self.graph:
                for neighbor, time in self.graph[current_stn]:
                    new_g_cost = g_cost + time
                    
                    if new_g_cost < g_costs.get(neighbor, float('inf')):
                        g_costs[neighbor] = new_g_cost
                        new_path = path + [neighbor]
                        
                        # H-cost (Heuristic): Simple 0
                        new_f_cost = new_g_cost 
                        heapq.heappush(open_list, (new_f_cost, new_g_cost, neighbor, new_path))

        return [{"error": "No route found."}]

print("AStarRouteOptimizerTool defined.")


class Agent:
    """A minimal class representing an agent powered by its role."""
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.session_context = {} # Sessions & state management

    def run(self, input_data: Dict) -> Dict:
        """Simulates the agent's action based on its role."""
        print(f"\n[AGENT: {self.name}] Running task: {self.role}...")
        
        if self.name == "RHA":
            # 1. Request Handler Agent (RHA): Agent powered by an LLM (Simulated)
            query = input_data.get('user_query', '')
            self.session_context['query'] = query
            # LLM output simulation:
            self.session_context['origin'] = 'Kalyan'
            self.session_context['destination'] = 'CSMT'
            self.session_context['constraint'] = 'hate Dadar transfer' 
            return {'status': 'Parameters parsed.', 'params': self.session_context}
        
        elif self.name == "DCA":
            # 2. Data Collector Agent (DCA): Parallel Agents / Built-in Tools (Simulated)
            print("  - Executing StaticDBTool lookup (Custom Tool 1).")
            # Built-in Tools Simulation (e.g., Google Search/Real-Time API):
            realtime_status = 'Minor delays on Central Line, estimated +5 minutes travel time.' 
            print(f"  - Executing Built-in Tool (Google Search/API): {realtime_status}")
            
            # Note the use of input_data['params'] for correct data passing.
            return {'status': 'Data collected.', 'schedules': TRAIN_NETWORK, 'realtime': realtime_status, 'params': input_data['params']}

        elif self.name == "OPA":
            # 3. Optimization Agent (OPA): Custom Tool / Context Engineering
            params = input_data['params']
            optimizer = AStarRouteOptimizerTool(input_data['schedules']) 
            
            # Custom Tool Run (A* search / Loop Agent Simulation)
            top_routes = optimizer.find_best_route(params['origin'], params['destination'], params['constraint'])
            
            # Context Engineering: Summarize the complex output
            summary = {
                'top_route': top_routes[0] if top_routes and 'error' not in top_routes[0] else None,
                'realtime_impact': input_data['realtime']
            }
            return {'status': 'Optimization complete.', 'optimized_summary': summary}
            
        elif self.name == "RA":
            # 4. Recommendation Agent (RA): Long term memory / A2A Protocol
            summary = input_data['optimized_summary']
            params = input_data['params']
            user_mem = MEMORY_BANK.get('user_123', {}) # Access Long term memory
            
            final_route = summary['top_route']
            
            # Rationale based on memory check
            reason = ""
            if final_route and final_route.get('penalty_min', 0) > 0:
                reason = f"Route was selected with a higher travel time to **avoid the {user_mem['dislikes'][0]}** transfer, a known preference from your **Memory Bank**."
            else:
                reason = "This route is the quickest available and adheres to your general preference."
            
            # Final LLM Output Generation (Simulated)
            output = f"""
            ***ðŸš‚ Final Recommended Journey (Personalized)***
            
            * **Origin:** {params['origin']} | **Destination:** {params['destination']}
            * **Path:** {final_route['path']}
            * **Total Travel Time (Cost):** {final_route['total_cost_min']} minutes
            * **Transfers:** {final_route['transfers']}
            * **Real-time Status:** {summary['realtime_impact']}

            **Agent Rationale (Based on Memory Check):**
            {reason}
            """
            return {'status': 'Recommendation complete.', 'output': output}

print("Agent class and all specialized agent logic defined.")


def run_sequential_workflow(query: str):
    """
    Orchestrates the Multi-Agent System in a Sequential flow.
    Manages the context transfer, simulating the A2A Protocol.
    """
    
    # 1. Initialization (Agents)
    rha = Agent("RHA", "Request Handler (LLM)")
    dca = Agent("DCA", "Data Collector (Parallel/Tools)")
    opa = Agent("OPA", "Path Optimizer (Custom Tool/Loop)")
    ra = Agent("RA", "Recommendation Generator (Memory)")

    context = {'user_query': query}
    
    # Sequential Step 1: RHA (Starts the process)
    context.update(rha.run(context))
    
    # Sequential Step 2: DCA (Executes parallel data retrieval)
    context.update(dca.run(context))
    
    # Sequential Step 3: OPA (Core optimization, Context Engineering)
    context.update(opa.run(context))
    
    # Sequential Step 4: RA (Final decision, A2A transfer occurs here)
    context.update(ra.run(context))
    
    return context['output']

print("Orchestrator function defined.")


if __name__ == "__main__":
    
    # The enquiry contains the constraint that triggers the Long Term Memory check.
    user_query = "What's the best train from Kalyan to CSMT, but I really hate changing trains at Dadar."
    
    print("="*80)
    print("ðŸš€ INTELLIGENT TRAIN ROUTE OPTIMIZER - KAGGLE CAPSTONE DEMO")
    print(f"USER ENQUIRY: {user_query}")
    print("="*80)
    
    final_recommendation = run_sequential_workflow(user_query)
    
    print("\n\n" + "="*80)
    print("âœ… FINAL AGENT RECOMMENDATION:")
    print("="*80)
    print(final_recommendation)

