pip install --upgrade luxai-s3


! mkdir agent
! cp -r /kaggle/input/lux-ai-season-3/lux agent


%%writefile agent/base.py

from enum import IntEnum

class Global:
    # Game-related constants
    SPACE_SIZE = 24
    MAX_UNITS = 16
    RELIC_REWARD_RANGE = 2
    MAX_STEPS_IN_MATCH = 100
    MAX_ENERGY_PER_TILE = 20
    MAX_RELIC_NODES = 6
    LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR = 50
    LAST_MATCH_WHEN_RELIC_CAN_APPEAR = 2

    # Unit movement and sensor constants
    UNIT_MOVE_COST = 1
    UNIT_SAP_COST = 30
    UNIT_SAP_RANGE = 3
    UNIT_SENSOR_RANGE = 2

    # Obstacle movement constants
    OBSTACLE_MOVEMENT_PERIOD = 20
    OBSTACLE_MOVEMENT_DIRECTION = (0, 0)

    # Nebula energy reduction
    NEBULA_ENERGY_REDUCTION = 5

    # Exploration flags
    ALL_RELICS_FOUND = False
    ALL_REWARDS_FOUND = False
    OBSTACLE_MOVEMENT_PERIOD_FOUND = False
    OBSTACLE_MOVEMENT_DIRECTION_FOUND = False

    # Game logs
    REWARD_RESULTS = []
    OBSTACLES_MOVEMENT_STATUS = []

    # Hidden node energy
    HIDDEN_NODE_ENERGY = 0

# Make SPACE_SIZE accessible directly
SPACE_SIZE = Global.SPACE_SIZE

class NodeType(IntEnum):
    unknown = -1
    empty = 0
    nebula = 1
    asteroid = 2

class ActionType(IntEnum):
    center = 0
    up = 1
    right = 2
    down = 3
    left = 4
    sap = 5

    @classmethod
    def from_coordinates(cls, current_position, next_position):
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]
        if dx < 0:
            return ActionType.left
        elif dx > 0:
            return ActionType.right
        elif dy < 0:
            return ActionType.up
        elif dy > 0:
            return ActionType.down
        else:
            return ActionType.center


%%writefile agent/pathfinding.py

import heapq
import numpy as np
from base import SPACE_SIZE, NodeType, Global, ActionType

CARDINAL_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def astar(weights, start, goal):
    def heuristic(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    queue = []
    nodes = np.zeros((*weights.shape, 4), dtype=np.float32)
    nodes[:] = -1

    heapq.heappush(queue, (0, start))
    nodes[start[0], start[1], :] = (*start, 0, heuristic(start, goal))

    while queue:
        f, (x, y) = heapq.heappop(queue)

        if (x, y) == goal:
            return reconstruct_path(nodes, start, goal)

        if f > nodes[x, y, 3]:
            continue

        distance = nodes[x, y, 2]
        for x_, y_ in get_neighbors(x, y):
            cost = weights[y_, x_]
            if cost < 0:
                continue

            new_distance = distance + cost
            if nodes[x_, y_, 2] < 0 or nodes[x_, y_, 2] > new_distance:
                new_f = new_distance + heuristic((x_, y_), goal)
                nodes[x_, y_, :] = x, y, new_distance, new_f
                heapq.heappush(queue, (new_f, (x_, y_)))

    return []

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(x, y):
    for dx, dy in CARDINAL_DIRECTIONS:
        x_ = x + dx
        y_ = y + dy
        if 0 <= x_ < SPACE_SIZE and 0 <= y_ < SPACE_SIZE:
            yield x_, y_

def reconstruct_path(nodes, start, goal):
    p = goal
    path = [p]
    while p != start:
        x = int(nodes[p[0], p[1], 0])
        y = int(nodes[p[0], p[1], 1])
        p = x, y
        path.append(p)
    return path[::-1]

def create_weights(space):
    weights = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
    for node in space:
        if not node.is_walkable:
            weight = -1
        else:
            node_energy = node.energy if node.energy is not None else Global.HIDDEN_NODE_ENERGY
            weight = Global.MAX_ENERGY_PER_TILE + 1 - node_energy
        if node.type == NodeType.nebula:
            weight += Global.NEBULA_ENERGY_REDUCTION
        weights[node.y][node.x] = weight
    return weights

def path_to_actions(path):
    actions = []
    if not path:
        return actions

    last_position = path[0]
    for x, y in path[1:]:
        direction = ActionType.from_coordinates(last_position, (x, y))
        actions.append(direction)
        last_position = (x, y)
    return actions


%%writefile agent/agent.py


import numpy as np  # Add this line at the top of the file
from base import Global, NodeType, ActionType, SPACE_SIZE
from pathfinding import astar, create_weights, path_to_actions

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.energy = None
        self.is_visible = False
        self.is_walkable = True  # Default to walkable unless updated

class Space:
    def __init__(self):
        self._nodes = [[Node(x, y) for x in range(SPACE_SIZE)] for y in range(SPACE_SIZE)]

    def get_node(self, x, y):
        return self._nodes[y][x]

    def update(self, step, obs, team_id):
        # Update the map based on the observation
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]

        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                node = self.get_node(x, y)
                if sensor_mask[x, y]:
                    node.type = NodeType(int(obs_tile_type[x, y]))
                    node.energy = int(obs_energy[x, y])
                    node.is_visible = True
                else:
                    node.is_visible = False

class Ship:
    def __init__(self, unit_id):
        self.unit_id = unit_id
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None

    def clean(self):
        """Reset the ship's state when it becomes inactive."""
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None

class Fleet:
    def __init__(self, team_id):
        self.team_id = team_id
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def update(self, obs, space):
        """Update the fleet based on the observation."""
        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
                ship.action = None
            else:
                ship.clean()  # Reset the ship if it's inactive

class Agent:
    def __init__(self, player, env_cfg):
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.space = Space()
        self.fleet = Fleet(self.team_id)

    def act(self, step, obs, remainingOverageTime):
        # Update the space and fleet based on the observation
        self.space.update(step, obs, self.team_id)
        self.fleet.update(obs, self.space)

        # Assign tasks to ships
        self.find_relics()
        self.harvest()

        # Create actions array
        actions = np.zeros((len(self.fleet.ships), 3), dtype=int)
        for i, ship in enumerate(self.fleet.ships):
            if ship.action is not None:
                actions[i] = ship.action, 0, 0
        return actions

    def find_relics(self):
        # Logic for finding relics
        pass

    def harvest(self):
        # Logic for harvesting energy
        pass


%%writefile agent/main.py
import json
from argparse import Namespace
from agent import Agent
from lux.kit import from_json

agent_dict = {}
agent_prev_obs = {}

def agent_fn(observation, configurations):
    global agent_dict
    obs = observation.obs
    if type(obs) == str:
        obs = json.loads(obs)
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime

    if step == 0:
        agent_dict[player] = Agent(player, configurations["env_cfg"])
    agent = agent_dict[player]
    actions = agent.act(step, from_json(obs), remainingOverageTime)
    return dict(action=actions.tolist())

if __name__ == "__main__":
    def read_input():
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    step = 0
    player_id = 0
    env_cfg = None
    i = 0
    while True:
        inputs = read_input()
        raw_input = json.loads(inputs)
        observation = Namespace(
            **dict(
                step=raw_input["step"],
                obs=raw_input["obs"],
                remainingOverageTime=raw_input["remainingOverageTime"],
                player=raw_input["player"],
                info=raw_input["info"],
            )
        )
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        print(json.dumps(actions))


import subprocess

# Run the Lux AI S3 command
command = ["luxai-s3", "agent/main.py", "agent/main.py", "--output=replay.html"]
subprocess.run(command)


! luxai-s3 agent/main.py agent/main.py --output=replay.html


import IPython
IPython.display.HTML(filename='replay.html')


!cd agent && tar -czf submission.tar.gz *
!mv agent/submission.tar.gz .

