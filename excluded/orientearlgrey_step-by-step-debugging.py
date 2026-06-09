! pip install --upgrade luxai-s3
! mkdir base && cp -r ../input/lux-ai-season-3/* base/


import sys
sys.path.insert(1, 'base')

import numpy as np

from luxai_s3.wrappers import LuxAIS3GymEnv
from base.agent import Agent as BaseAgent
from base.lux.utils import direction_to


class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        self.map_tiles = None

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        self.map_tiles = np.array(obs["map_features"]["tile_type"])
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)


        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match
        
        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
            

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                
                # if close to the relic node we want to hover around it and hope to gain points
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        return actions

    def get_map_tiles(self):
        return self.map_tiles


# Initialise environment
env = LuxAIS3GymEnv(numpy_output=True)
obs, info = env.reset()
env_cfg = info["params"]

# Initialise agents
player_0 = Agent("player_0", env_cfg)
player_1 = BaseAgent("player_1", env_cfg)

# Initialise match
step = 0


# Take action 
actions = {}
for agent in [player_0, player_1]:
    actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

obs, rewards ,terminated, truncated, info = env.step(actions)
done = {k: terminated[k] | truncated[k] for k in terminated}

if not (done["player_0"] and done["player_1"]):
    step += 1

print(f'Step: {step}')
# print(obs)


player_0.get_map_tiles()[1][1]


import random


random.choice([1,3])


def pivot_to(ideal_direction: int, legal_move_directions: list) -> int:
    if ideal_direction not in legal_move_directions:
        # top-down illegal?
        if ideal_direction in [1, 3]:
            return random.choice([2,4])
        else:
            return random.choice([1,3])





pivot_to(direction_to(np.array([23,23]), np.array([12,12])), [[0, 1, 4]])




