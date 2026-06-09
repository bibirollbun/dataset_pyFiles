# verify version
!python --version
!pip install --upgrade luxai-s3
!mkdir agent && cp -r ../input/lux-ai-season-3/* agent/


import sys
sys.path.insert(1, 'agent')


import json
from IPython.display import display, Javascript
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

def render_episode(episode: RecordEpisode) -> None:
    data = json.dumps(episode.serialize_episode_data(), separators=(",", ":"))
    display(Javascript(f"""
var iframe = document.createElement('iframe');
iframe.src = 'https://s3vis.lux-ai.org/#/kaggle';
iframe.width = '100%';
iframe.scrolling = 'no';

iframe.addEventListener('load', event => {{
    event.target.contentWindow.postMessage({data}, 'https://s3vis.lux-ai.org');
}});

new ResizeObserver(entries => {{
    for (const entry of entries) {{
        entry.target.height = `${{Math.round(320 + 0.3 * entry.contentRect.width)}}px`;
    }}
}}).observe(iframe);

element.append(iframe);
    """))

def evaluate_agents(agent_1_cls, agent_2_cls, seed=42, games_to_play=3, replay_save_dir="replays"):
    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True), save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
    )
    obs, info = env.reset(seed=seed)
    for i in range(games_to_play):
        obs, info = env.reset()
        env_cfg = info["params"] # only contains observable game parameters
        player_0 = agent_1_cls("player_0", env_cfg)
        player_1 = agent_2_cls("player_1", env_cfg)
    
        # main game loop
        game_done = False
        step = 0
        print(f"Running game {i}")
        while not game_done:
            actions = dict()
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
            obs, reward, terminated, truncated, info = env.step(actions)
            # info["state"] is the environment state object, you can inspect/play around with it to e.g. print
            # unobservable game data that agents can't see
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1
        render_episode(env)
    env.close() # free up resources and save final replay


from lux.utils import direction_to
import numpy as np
from collections import defaultdict

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        
        # Simple state tracking
        self.relic_values = defaultdict(int)  # Points per relic
        self.unit_targets = {}  # Current target for each unit
        self.claimed_relics = set()  # Relics being targeted
        
    def closest_relic(self, unit_pos, relic_positions, occupied_relics):
        """Find closest unclaimed relic."""
        best_dist = float('inf')
        best_relic = None
        
        for relic_pos in relic_positions:
            relic_tuple = tuple(relic_pos)
            if relic_tuple in occupied_relics:
                continue
                
            dist = abs(unit_pos[0] - relic_pos[0]) + abs(unit_pos[1] - relic_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_relic = relic_pos
                
        return best_relic
        
    def get_move_action(self, unit_pos, target_pos):
        """Get direction to move towards target."""
        if np.array_equal(unit_pos, target_pos):
            return [0, 0, 0]  # Stay if at target
        return [direction_to(unit_pos, target_pos), 0, 0]
        
    def can_pickup_relic(self, unit_pos, relic_positions, relic_mask):
        """Check if unit can pickup a relic."""
        for i, relic_pos in enumerate(relic_positions):
            if relic_mask[i] and np.array_equal(unit_pos, relic_pos):
                return True
        return False

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        
        # Get game state
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energy = np.array(obs["units"]["energy"][self.team_id])
        relic_positions = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        
        # Track which relics are being targeted
        occupied_relics = set()
        
        # Process each unit
        for unit_id in range(len(unit_positions)):
            if not obs["units_mask"][self.team_id][unit_id]:
                continue
                
            unit_pos = unit_positions[unit_id]
            
            # Try to pickup relic if possible
            if unit_energy[unit_id] >= 10 and self.can_pickup_relic(unit_pos, relic_positions, relic_mask):
                actions[unit_id] = [0, 0, 6]  # Pickup action
                continue
                
            # Check if unit needs a new target
            if unit_id not in self.unit_targets or not any(np.array_equal(self.unit_targets[unit_id], relic_pos) 
                                                         for relic_pos in relic_positions):
                # Find closest unclaimed relic
                target = self.closest_relic(unit_pos, relic_positions, occupied_relics)
                if target is not None:
                    self.unit_targets[unit_id] = target
                    occupied_relics.add(tuple(target))
                    
            # Move towards target if we have one
            if unit_id in self.unit_targets:
                actions[unit_id] = self.get_move_action(unit_pos, self.unit_targets[unit_id])
            
            # If no target available, move towards center
            else:
                center = np.array([self.env_cfg["map_width"] // 2, self.env_cfg["map_height"] // 2])
                actions[unit_id] = self.get_move_action(unit_pos, center)
                
        return actions


evaluate_agents(Agent, Agent)


%%writefile agent/agent.py
from lux.utils import direction_to
import numpy as np
from collections import defaultdict

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        
        # Simple state tracking
        self.relic_values = defaultdict(int)  # Points per relic
        self.unit_targets = {}  # Current target for each unit
        self.claimed_relics = set()  # Relics being targeted
        
    def closest_relic(self, unit_pos, relic_positions, occupied_relics):
        """Find closest unclaimed relic."""
        best_dist = float('inf')
        best_relic = None
        
        for relic_pos in relic_positions:
            relic_tuple = tuple(relic_pos)
            if relic_tuple in occupied_relics:
                continue
                
            dist = abs(unit_pos[0] - relic_pos[0]) + abs(unit_pos[1] - relic_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_relic = relic_pos
                
        return best_relic
        
    def get_move_action(self, unit_pos, target_pos):
        """Get direction to move towards target."""
        if np.array_equal(unit_pos, target_pos):
            return [0, 0, 0]  # Stay if at target
        return [direction_to(unit_pos, target_pos), 0, 0]
        
    def can_pickup_relic(self, unit_pos, relic_positions, relic_mask):
        """Check if unit can pickup a relic."""
        for i, relic_pos in enumerate(relic_positions):
            if relic_mask[i] and np.array_equal(unit_pos, relic_pos):
                return True
        return False

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        
        # Get game state
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energy = np.array(obs["units"]["energy"][self.team_id])
        relic_positions = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        
        # Track which relics are being targeted
        occupied_relics = set()
        
        # Process each unit
        for unit_id in range(len(unit_positions)):
            if not obs["units_mask"][self.team_id][unit_id]:
                continue
                
            unit_pos = unit_positions[unit_id]
            
            # Try to pickup relic if possible
            if unit_energy[unit_id] >= 10 and self.can_pickup_relic(unit_pos, relic_positions, relic_mask):
                actions[unit_id] = [0, 0, 6]  # Pickup action
                continue
                
            # Check if unit needs a new target
            if unit_id not in self.unit_targets or not any(np.array_equal(self.unit_targets[unit_id], relic_pos) 
                                                         for relic_pos in relic_positions):
                # Find closest unclaimed relic
                target = self.closest_relic(unit_pos, relic_positions, occupied_relics)
                if target is not None:
                    self.unit_targets[unit_id] = target
                    occupied_relics.add(tuple(target))
                    
            # Move towards target if we have one
            if unit_id in self.unit_targets:
                actions[unit_id] = self.get_move_action(unit_pos, self.unit_targets[unit_id])
            
            # If no target available, move towards center
            else:
                center = np.array([self.env_cfg["map_width"] // 2, self.env_cfg["map_height"] // 2])
                actions[unit_id] = self.get_move_action(unit_pos, center)
                
        return actions


!luxai-s3 agent/main.py agent/main.py --seed 101 -o replay.html


!cd agent && tar -czf submission.tar.gz *
!mv agent/submission.tar.gz .

