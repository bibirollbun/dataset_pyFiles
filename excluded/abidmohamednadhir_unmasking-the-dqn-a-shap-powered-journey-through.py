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


!pip install luxai-s3


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import numpy as np
import pandas as pd
import torch
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class RLAnalyzer:
    def __init__(self, agent, env_cfg, params):
        self.agent = agent
        self.env_cfg = env_cfg
        self.params = params
        self.feature_names = [
            'unit_pos_x', 'unit_pos_y',
            'closest_relic_x', 'closest_relic_y',
            'normalized_energy', 'normalized_step'
        ] + [f'sensor_mask_{i}' for i in range(24 * 24)] + \
           [f'local_tile_type_{i}' for i in range(25)] + \
           [f'local_tile_energy_{i}' for i in range(25)] + \
           ['score', 'wins', 'sensor_range', 'nebula_reduction']
        
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
    def collect_episode_data(self, obs, action, reward):
        """Collect state, action, reward data during an episode"""
        
        
        state = self._process_observation(obs)
        self.state_history.append(state)
        # Only store the action type (first element)
        if isinstance(action, np.ndarray):
            # If it's a multi-unit action array, store first unit's action type
            self.action_history.append(action[0][0])
        else:
            # If it's a single action tuple/list, store the action type
            self.action_history.append(action[0])
        self.reward_history.append(reward)
    
    def _process_observation(self, obs):
        """Convert observation to state representation used by the agent"""
        unit_pos = obs["units"]["position"][self.agent.team_id][0]
        unit_energy = obs["units"]["energy"][self.agent.team_id][0]
        relic_nodes = obs["relic_nodes"]
        step = obs.get("step", 0)
        relic_mask = obs["relic_nodes_mask"]
        sensor_mask = obs["sensor_mask"][self.agent.team_id][0]
        tile_type = obs["map_features"]["tile_type"]
        tile_energy = obs["map_features"]["energy"]
        score = obs["team_points"][self.agent.team_id]
        wins = obs["team_wins"][self.agent.team_id]
        
        return self.agent._state_representation(
            unit_pos, unit_energy, relic_nodes, step, relic_mask,
            sensor_mask, tile_type, tile_energy, score, wins, self.params
        ).cpu().numpy()
    
    def compute_feature_importance(self):
        """Compute feature importance using SHAP values"""
        states = np.array(self.state_history)
        
        # Create a background dataset for SHAP
        background = states[np.random.choice(len(states), min(100, len(states)), replace=False)]
        
        # Create a PyTorch model wrapper for SHAP
        def model_predict(x):
            with torch.no_grad():
                return self.agent.policy_net(torch.FloatTensor(x).to(self.agent.device)).cpu().numpy()
        
        # Initialize SHAP explainer
        explainer = shap.KernelExplainer(model_predict, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(states[:100])  # Limit to 100 samples for computation efficiency
        
        # Create importance summary
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values[0]).mean(0)
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def analyze_state_action_correlation(self):
        """Analyze correlation between state features and actions"""
        states = np.array(self.state_history)
        actions = np.array(self.action_history)  # Now this should be 1D
        
        # Debug information
        print("States shape:", states.shape)
        print("Actions shape:", actions.shape)
        print("First few actions:", actions[:5])
        
        # Create DataFrame
        df = pd.DataFrame(states, columns=self.feature_names)
        df['action'] = actions
        
        # Compute correlations and handle any NaN values
        correlation = df.corr()['action'].fillna(0).sort_values(ascending=False)
        
        return correlation
    
    def analyze_reward_distribution(self):
        """Analyze the distribution of rewards"""
        rewards = np.array(self.reward_history)
        
        stats = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'median': np.median(rewards)
        }
        
        return stats
    
    def plot_feature_importance(self, importance_df, top_n=20):
        """Plot top N most important features"""
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
        plt.title('Top Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_reward_history(self):
        """Plot reward history over time"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.reward_history)
        plt.title('Reward History')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.tight_layout()
        plt.show()
    
    def analyze_action_distribution(self):
        """Analyze the distribution of actions taken"""
        actions = np.array(self.action_history)
        action_counts = pd.Series(actions).value_counts()
        action_names = ['stay', 'up', 'right', 'down', 'left', 'sap']
        
        distribution = pd.Series(
            {action_names[i]: action_counts.get(i, 0) for i in range(6)}
        )
        
        return distribution
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report"""
        # Feature importance
        try:
            importance_df = self.compute_feature_importance()
        except Exception as e:
            print(f"Warning: Feature importance computation failed: {e}")
            importance_df = pd.DataFrame()
        
        # State-action correlation
        try:
            correlation = self.analyze_state_action_correlation()
        except Exception as e:
            print(f"Warning: State-action correlation analysis failed: {e}")
            correlation = pd.Series()
        
        # Reward statistics
        reward_stats = self.analyze_reward_distribution()
        
        # Action distribution
        action_dist = self.analyze_action_distribution()
        
        # Create and return report
        report = {
            'feature_importance': importance_df,
            'state_action_correlation': correlation,
            'reward_statistics': reward_stats,
            'action_distribution': action_dist
        }
        
        return report
    
    def plot_analysis_dashboard(self):
        """Create a comprehensive visualization dashboard"""
        plt.figure(figsize=(20, 15))
        
        # Feature importance plot
        plt.subplot(2, 2, 1)
        importance_df = self.compute_feature_importance()
        sns.barplot(data=importance_df.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance')
        
        # Reward history plot
        plt.subplot(2, 2, 2)
        plt.plot(self.reward_history)
        plt.title('Reward History')
        
        # Action distribution plot
        plt.subplot(2, 2, 3)
        action_dist = self.analyze_action_distribution()
        sns.barplot(x=action_dist.index, y=action_dist.values)
        plt.title('Action Distribution')
        
        # Reward distribution plot
        plt.subplot(2, 2, 4)
        sns.histplot(self.reward_history)
        plt.title('Reward Distribution')
        
        plt.tight_layout()
        plt.show()

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)
    
# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, player: str, env_cfg, params, training=False) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.params = params
        self.training = training
        
        # DQN parameters
        self.state_size = (
                            2 +  # Unit position (x, y)
                            2 +  # Closest relic node (x, y)
                            1 +  # Normalized energy
                            1 +  # Normalized step count
                            24 * 24 +  # Flattened sensor mask (vision power map)
                            25 +  # Localized tile type (5x5 grid flattened)
                            25 +  # Localized tile energy (5x5 grid flattened)
                            1 +   # Game score
                            1 +   # Games won
                            2     # Params (sensor range, nabula vision reduction) 
                        )
        self.action_size = 6  # stay, up, right, down, left, sap
        self.hidden_size = 128
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)
        
        if not training:
            self.load_model()
            self.epsilon = 0.0

    def _get_surrounding_features(self, unit_pos, map_features):
        features = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x = unit_pos[0] + dx
                y = unit_pos[1] + dy
                if 0 <= x < 24 and 0 <= y < 24:
                    features.append(map_features[x][y])
                else:
                    features.append(-1)  # Out of bounds
        return np.array(features)

    def get_local_tile_type(self, tile_type, unit_pos, radius=2):
        """
        Extracts a localized view of the tile type centered on the unit's position.
    
        Args:
            tile_type (np.array): Tile types on the map (24x24).
            unit_pos (np.array): Position of the unit (x, y).
            radius (int): Radius of the localized view (default is 2, resulting in a 5x5 grid).
    
        Returns:
            np.array: Flattened array of localized tile types.
        """
        x, y = unit_pos
        h, w = tile_type.shape
        local_tile_type = tile_type[max(0, x-radius):min(h, x+radius+1),
                                    max(0, y-radius):min(w, y+radius+1)]


        # Pad the array to ensure a fixed size (5x5)
        padded_tile_type = np.pad(local_tile_type, 
                                  ((max(0, radius - x), max(0, x + radius + 1 - h)),
                                   (max(0, radius - y), max(0, y + radius + 1 - w))),
                                  mode='constant', constant_values=-1)
        return padded_tile_type.flatten()

    def get_local_tile_energy(self, tile_energy, unit_pos, radius=2):
        """
        Extracts a localized view of the tile energy centered on the unit's position.
    
        Args:
            tile_energy (np.array): Energy distribution on the map (24x24).
            unit_pos (np.array): Position of the unit (x, y).
            radius (int): Radius of the localized view (default is 2, resulting in a 5x5 grid).
    
        Returns:
            np.array: Flattened array of localized tile energies.
        """
        x, y = unit_pos
        h, w = tile_energy.shape
        local_tile_energy = tile_energy[max(0, x-radius):min(h, x+radius+1),
                                        max(0, y-radius):min(w, y+radius+1)]
        # Pad the array to ensure a fixed size (5x5)
        padded_tile_energy = np.pad(local_tile_energy, 
                                    ((max(0, radius - x), max(0, x + radius + 1 - h)),
                                     (max(0, radius - y), max(0, y + radius + 1 - w))),
                                    mode='constant', constant_values=-1)
        
        return padded_tile_energy.flatten()

    def compute_vision_power_map(self, unit_pos, tile_type, params):
        """
        Computes the vision power map for a team based on the positions of its units.
        Args:
            unit_pos: Position of the unit (x, y).
            tile_type: 2D array representing the tile types on the map.
            params: EnvParams object containing environment parameters.
        Returns:
            vision_power_map: 2D array representing the vision power for each tile.
        """
        #print(f"PARAMS In vision power:{params}")
        
        # Extract parameters from the EnvParams object
        unit_sensor_range = params.unit_sensor_range
        nebula_tile_vision_reduction = params.nebula_tile_vision_reduction
    
        # Initialize the vision power map
        vision_power_map = np.zeros((24, 24))  # Assuming map size is 24x24
    
        # Compute vision power for each tile within the sensor range
        x, y = unit_pos
        for dx in range(-unit_sensor_range, unit_sensor_range + 1):
            for dy in range(-unit_sensor_range, unit_sensor_range + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 24 and 0 <= ny < 24:  # Ensure the tile is within bounds
                    vision_power_map[nx, ny] += 1 + unit_sensor_range - min(abs(dx), abs(dy))
    
        # Apply nebula tile vision reduction
        nebula_mask = (tile_type == 1)  # Mask for nebula tiles (assuming type ID 1)
        vision_power_map[nebula_mask] -= nebula_tile_vision_reduction
    
        return vision_power_map
        
    def _state_representation(self, unit_pos, unit_energy, relic_nodes, step, relic_mask, sensor_mask, tile_type, tile_energy, score, wins, params):
        """
        Generates a state representation for a given unit based on its position, energy,
        the closest relic node, current step, sensor mask, local tile type, and local tile energy.
    
        Args:
            unit_pos (np.array): Position of the unit (x, y).
            unit_energy (int): Energy level of the unit.
            relic_nodes (np.array): Array of relic node positions (n x 2).
            step (int): Current step in the game.
            relic_mask (np.array): Boolean mask indicating which relic nodes are active.
            sensor_mask (np.array): Sensor mask for the unit.
            tile_type (np.array): Tile types on the map (24x24).
            tile_energy (np.array): Energy distribution on the map (24x24).
    
        Returns:
            torch.FloatTensor: State representation as a PyTorch tensor.
            """
        #print(f"PARAMS In state representation:{params}")
        if not relic_mask.any():
            closest_relic = np.array([-1, -1])
        else:
            visible_relics = relic_nodes[relic_mask]
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]

        # Normalize the unit's energy and step count
        normalized_energy = unit_energy / 400.0  # Assuming max energy is 400
        normalized_step = step / 505.0          # Assuming max steps is 505
        
        # Compute vision power map and derive sensor mask
        vision_power_map = self.compute_vision_power_map(unit_pos, tile_type, params)
        sensor_mask = (vision_power_map > 0)
        flattened_sensor_mask = sensor_mask.flatten().astype(float)
        

        # Get localized tile type and tile energy
        local_tile_type = self.get_local_tile_type(tile_type, unit_pos, radius=2)  # Radius of 2 (5x5 grid)
        local_tile_energy = self.get_local_tile_energy(tile_energy, unit_pos, radius=2)  # Radius of 2 (5x5 grid)

        # Convert params to a 1D array
        params_array = np.array([
            params.unit_sensor_range,
            params.nebula_tile_vision_reduction
        ])
        
        # Concatenate all features into a single state vector
        state = np.concatenate([
            unit_pos,                           # Unit's position (x, y)
            closest_relic,                      # Closest relic node (x, y)
            [normalized_energy],                # Normalized energy
            [normalized_step],                  # Normalized step count
            flattened_sensor_mask,              # Flattened sensor mask
            local_tile_type,                    # Localized tile type (5x5 grid flattened)
            local_tile_energy,                  # Localized tile energy (5x5 grid flattened)    
            [score],                            # Game Score
            [wins],                              # Games Won
            params_array
        ])

        return torch.FloatTensor(state).to(self.device)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # Units Mask
        units_mask = np.array(obs["units_mask"][self.team_id])
        # Units
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        # Sensor Mask
        sensor_mask = np.array(obs["sensor_mask"][self.team_id])
        # Map Features
        tile_energy = np.array(obs["map_features"]["energy"])
        tile_type = np.array(obs["map_features"]["tile_type"])
        # Relic Nodes
        relic_nodes = np.array(obs["relic_nodes"])
        # Relic Nodes mask
        relic_mask = np.array(obs["relic_nodes_mask"])
        # Team Points
        self.score = np.array(obs["team_points"][self.team_id])
        # Team Wins
        wins = np.array(obs["team_wins"][self.team_id])
       
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
       
       # if step % 500 == 0:
          #print(f"memory:  {len(self.memory)}")

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_units = np.where(units_mask)[0]
        
        for unit_id in available_units:
            state = self._state_representation(
                unit_positions[unit_id],
                unit_energys[unit_id],
                relic_nodes,
                step,
                relic_mask,
                sensor_mask[unit_id],
                tile_type,
                tile_energy,
                self.score,
                wins,
                self.params
            )

            # action_type = random.randrange(self.action_size)
            self.unit_explore_locations = dict()
            self.relic_node_positions = []
            self.discovered_relic_nodes_ids = set()

            # visible relic nodes
            visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
            # save any new relic nodes that we discover for the rest of the game.
            for id in visible_relic_node_ids:
                if id not in self.discovered_relic_nodes_ids:
                    self.discovered_relic_nodes_ids.add(id)
                    self.relic_node_positions.append(observed_relic_node_positions[id])

            
            if random.random() < self.epsilon and self.training:
                if len(self.relic_node_positions) > 0:
                    nearest_relic_node_position = self.relic_node_positions[0]
                    unit_pos = unit_positions[unit_id]
                    manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])

                    # if close to the relic node we want to move randomly around it and hope to gain points
                    if manhattan_distance <= 4:
                        random_direction = np.random.randint(0, 5)
                        actions[unit_id] = [random_direction, 0, 0]
                    else:
                        # otherwise we want to move towards the relic node
                        actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
                else:
                    #pick a random location on the map for the unit to explore
                    unit_pos = unit_positions[unit_id]
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                    # using the direction_to tool we can generate a direction that makes the unit move to the saved location
                    # note that the first index of each unit's action represents the type of action. See specs for more details
                    actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
            else:
                with torch.no_grad():
                    q_values = self.policy_net(state)
                    action_type = q_values.argmax().item()
                    #print(f"Q-values: {q_values}")
                if action_type == 5:  # Sap action
                    # Find closest enemy unit
                    opp_positions = obs["units"]["position"][self.opp_team_id]
                    opp_mask = obs["units_mask"][self.opp_team_id]
                    valid_targets = []

                    for opp_id, pos in enumerate(opp_positions):
                        if opp_mask[opp_id] and pos[0] != -1:
                            valid_targets.append(pos)

                    if valid_targets:
                        target_pos = valid_targets[0]  # Choose first valid target
                        actions[unit_id] = [5, target_pos[0], target_pos[1]]
                    else:
                        actions[unit_id] = [0, 0, 0]  # Stay if no valid targets
                else:
                    actions[unit_id] = [action_type, 0, 0]

    
        #print(f (Actions: {actions}")
        
        return actions

    def learn(self, step, last_obs, actions, obs, rewards, dones):
        if not self.training or len(self.memory) < self.batch_size:
          return
            
        
        rewards = self.score
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if step % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        #print(f"Loss: {loss.item()} Epsilon: {self.epsilon} Score: {rewards} Step: {step}")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'dqn_model_{self.player}.pth')

    def load_model(self):
        try:
            checkpoint = torch.load(f'dqn_model_{self.player}.pth')
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No trained model found for {self.player}")

##################################################################################
##################################################################################

from luxai_s3.wrappers import LuxAIS3GymEnv
from luxai_s3.params import EnvParams

def evaluate_agents(agent_1_cls, agent_2_cls, seed=42, training=True, games_to_play=3):
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=seed)
    
    env_cfg = info["params"]  
    # Create an instance of EnvParams
    params = EnvParams()

    player_0 = Agent("player_0", env_cfg, params, training=training)
    player_1 = Agent("player_1", env_cfg, params, training=training)

    # Track rewards for each game
    game_rewards = {"player_0": [], "player_1": []}

    # Create analyzers for both agents
    analyzer_0 = RLAnalyzer(player_0, env_cfg, params)
    analyzer_1 = RLAnalyzer(player_1, env_cfg, params)

    for i in range(games_to_play):
        obs, info = env.reset()
        game_done = False
        step = 0
        last_obs = None
        last_actions = None
        # Initialize rewards for the current game
        current_game_rewards = {"player_0": 0, "player_1": 0}
        #print(f"{i}")
        while not game_done:
            
            actions = {}
            
            # Store current observation for learning
            if training:
                last_obs = {
                    "player_0": obs["player_0"].copy(),
                    "player_1": obs["player_1"].copy()
                }

            # Get actions
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

            if training:
                last_actions = actions.copy()

            # Environment step
            obs, rewards ,terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id],
                "player_1": obs["player_1"]["team_points"][player_1.team_id]
            }  
            # Accumulate rewards for the current game
            current_game_rewards["player_0"] += rewards["player_0"]
            current_game_rewards["player_1"] += rewards["player_1"]

            # Collect data for analysis
            analyzer_0.collect_episode_data(obs["player_0"], actions["player_0"], rewards["player_0"])
            analyzer_1.collect_episode_data(obs["player_1"], actions["player_1"], rewards["player_1"])
            
            
            # Store experiences and learn
            if training and last_obs is not None:
                # Store experience for each unit
                for agent in [player_0, player_1]:
                    for unit_id in range(env_cfg["max_units"]):
                        if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            current_state = agent._state_representation(
                                last_obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                                last_obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                                last_obs[agent.player]["relic_nodes"],
                                step,
                                last_obs[agent.player]["relic_nodes_mask"],
                                last_obs[agent.player]["sensor_mask"][agent.team_id][unit_id],
                                last_obs[agent.player]["map_features"]["tile_type"],
                                last_obs[agent.player]["map_features"]["energy"],
                                last_obs[agent.player]["team_points"][agent.team_id],
                                last_obs[agent.player]["team_wins"][agent.team_id],
                                params,

                            )
                            
                            next_state = agent._state_representation(
                                obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                                obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                                obs[agent.player]["relic_nodes"],
                                step + 1,
                                obs[agent.player]["relic_nodes_mask"],
                                last_obs[agent.player]["sensor_mask"][agent.team_id][unit_id],
                                last_obs[agent.player]["map_features"]["tile_type"],
                                last_obs[agent.player]["map_features"]["energy"],
                                last_obs[agent.player]["team_points"][agent.team_id],
                                last_obs[agent.player]["team_wins"][agent.team_id],
                                params,
                            )
                            
                            agent.memory.push(
                                current_state,
                                last_actions[agent.player][unit_id][0],
                                rewards[agent.player],
                                next_state,
                                dones[agent.player]
                            )
                
                # Learn from experiences
                player_0.learn(step, last_obs["player_0"], actions["player_0"], 
                             obs["player_0"], rewards["player_0"], dones["player_0"])
                player_1.learn(step, last_obs["player_1"], actions["player_1"], 
                             obs["player_1"], rewards["player_1"], dones["player_1"])

            if dones["player_0"] or dones["player_1"]:
                game_done = True
                if training:
                    player_0.save_model()
                    player_1.save_model()

            step += 1
        # Log rewards for the current game
        game_rewards["player_0"].append(current_game_rewards["player_0"])
        game_rewards["player_1"].append(current_game_rewards["player_1"])
        print(f"Game {i+1}:")
        print(f"  Player 0 Reward: {current_game_rewards['player_0']}")
        print(f"  Player 1 Reward: {current_game_rewards['player_1']}")

    # After training/evaluation, generate analysis
    report_0 = analyzer_0.generate_analysis_report()
    report_1 = analyzer_1.generate_analysis_report()
    
    # Visualize results
    analyzer_0.plot_analysis_dashboard()
    analyzer_1.plot_analysis_dashboard()
    
    env.close()
    if training:
      player_0.save_model()
      player_1.save_model()

# Call the evaluate_agents function to start training
evaluate_agents(
    agent_1_cls=Agent,  # Use the Agent class for both players
    agent_2_cls=Agent,
    seed=42,            # Set a random seed for reproducibility
    training=True,      # Enable training mode
    games_to_play=10    # Number of games to play for training
)

# Evaluate the trained agents
evaluate_agents(
    agent_1_cls=Agent,
    agent_2_cls=Agent,
    seed=42,
    training=False,     # Disable training mode for evaluation
    games_to_play=5     # Number of games to play for evaluation
)

