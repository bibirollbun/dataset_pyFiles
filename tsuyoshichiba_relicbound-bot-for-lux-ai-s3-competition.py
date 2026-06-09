! mkdir agent
! cp -r /kaggle/input/lux-ai-season-3/lux agent


%%writefile agent/base.py

from enum import IntEnum


class Global:

    # Game related constants:

    SPACE_SIZE = 24
    MAX_UNITS = 16
    RELIC_REWARD_RANGE = 2
    MAX_STEPS_IN_MATCH = 100
    MAX_ENERGY_PER_TILE = 20
    MAX_RELIC_NODES = 6
    LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR = 50
    LAST_MATCH_WHEN_RELIC_CAN_APPEAR = 2

    # We will find the exact value of these constants during the game
    UNIT_MOVE_COST = 1  # OPTIONS: list(range(1, 6))
    UNIT_SAP_COST = 30  # OPTIONS: list(range(30, 51))
    UNIT_SAP_RANGE = 3  # OPTIONS: list(range(3, 8))
    UNIT_SENSOR_RANGE = 2  # OPTIONS: [1, 2, 3, 4]
    OBSTACLE_MOVEMENT_PERIOD = 20  # OPTIONS: 6.67, 10, 20, 40
    OBSTACLE_MOVEMENT_DIRECTION = (0, 0)  # OPTIONS: [(1, -1), (-1, 1)]

    # We will NOT find the exact value of these constants during the game
    NEBULA_ENERGY_REDUCTION = 5  # OPTIONS: [0, 1, 2, 3, 5, 25]

    # Exploration flags:

    ALL_RELICS_FOUND = False
    ALL_REWARDS_FOUND = False
    OBSTACLE_MOVEMENT_PERIOD_FOUND = False
    OBSTACLE_MOVEMENT_DIRECTION_FOUND = False

    # 共有される reward タイルの情報
    SHARED_REWARD_TILES = set()
    
    # Game logs:

    # REWARD_RESULTS: [{"nodes": Set[Node], "points": int}, ...]
    # A history of reward events, where each entry contains:
    # - "nodes": A set of nodes where our ships were located.
    # - "points": The number of points scored at that location.
    # This data will help identify which nodes yield points.
    REWARD_RESULTS = []

    # obstacles_movement_status: list of bool
    # A history log of obstacle (asteroids and nebulae) movement events.
    # - `True`: The ships' sensors detected a change in the obstacles' positions at this step.
    # - `False`: The sensors did not detect any changes.
    # This information will be used to determine the speed and direction of obstacle movement.
    OBSTACLES_MOVEMENT_STATUS = []

    # Others:

    # The energy on the unknown tiles will be used in the pathfinding
    HIDDEN_NODE_ENERGY = 0


SPACE_SIZE = Global.SPACE_SIZE


class NodeType(IntEnum):
    unknown = -1
    empty = 0
    nebula = 1
    asteroid = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


_DIRECTIONS = [
    (0, 0),  # center
    (0, -1),  # up
    (1, 0),  # right
    (0, 1),  #  down
    (-1, 0),  # left
    (0, 0),  # sap
]


class ActionType(IntEnum):
    center = 0
    up = 1
    right = 2
    down = 3
    left = 4
    sap = 5

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

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

    def to_direction(self):
        return _DIRECTIONS[self]


def get_match_step(step: int) -> int:
    return step % (Global.MAX_STEPS_IN_MATCH + 1)


def get_match_number(step: int) -> int:
    return step // (Global.MAX_STEPS_IN_MATCH + 1)


def warp_int(x):
    if x >= SPACE_SIZE:
        x -= SPACE_SIZE
    elif x < 0:
        x += SPACE_SIZE
    return x


def warp_point(x, y) -> tuple:
    return warp_int(x), warp_int(y)


def get_opposite(x, y) -> tuple:
    # Returns the mirrored point across the diagonal
    return SPACE_SIZE - y - 1, SPACE_SIZE - x - 1


def is_upper_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 >= y


def is_lower_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 <= y


def is_team_sector(team_id, x, y) -> bool:
    return is_upper_sector(x, y) if team_id == 0 else is_lower_sector(x, y)



%%writefile agent/pathfinding.py

import heapq
import numpy as np

from base import SPACE_SIZE, NodeType, Global, ActionType

CARDINAL_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def astar(weights, start, goal):
    # A* algorithm
    # returns the shortest path form start to goal

    min_weight = weights[np.where(weights >= 0)].min()

    def heuristic(p1, p2):
        return min_weight * manhattan_distance(p1, p2)

    queue = []

    # nodes: [x, y, (parent.x, parent.y, distance, f)]
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


def manhattan_distance(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(x, y):
    for dx, dy in CARDINAL_DIRECTIONS:
        x_ = x + dx
        if x_ < 0 or x_ >= SPACE_SIZE:
            continue

        y_ = y + dy
        if y_ < 0 or y_ >= SPACE_SIZE:
            continue

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


def nearby_positions(x, y, distance):
    for x_ in range(max(0, x - distance), min(SPACE_SIZE, x + distance + 1)):
        for y_ in range(max(0, y - distance), min(SPACE_SIZE, y + distance + 1)):
            yield x_, y_


def create_weights(space):
    # create weights for AStar algorithm

    weights = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
    for node in space:

        if not node.is_walkable:
            weight = -1
        else:
            node_energy = node.energy
            if node_energy is None:
                node_energy = Global.HIDDEN_NODE_ENERGY

            # pathfinding can't deal with negative weight
            weight = Global.MAX_ENERGY_PER_TILE + 1 - node_energy

        if node.type == NodeType.nebula:
            weight += Global.NEBULA_ENERGY_REDUCTION

        weights[node.y][node.x] = weight

    return weights


def find_closest_target(start, targets):
    target, min_distance = None, float("inf")
    for t in targets:
        d = manhattan_distance(start, t)
        if d < min_distance:
            target, min_distance = t, d

    return target, min_distance


def estimate_energy_cost(space, path):
    if len(path) <= 1:
        return 0

    energy = 0
    last_position = path[0]
    for x, y in path[1:]:
        node = space.get_node(x, y)
        if node.energy is not None:
            energy -= node.energy
        else:
            energy -= Global.HIDDEN_NODE_ENERGY

        if node.type == NodeType.nebula:
            energy += Global.NEBULA_ENERGY_REDUCTION

        if (x, y) != last_position:
            energy += Global.UNIT_MOVE_COST

    return energy


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


#!/usr/bin/env python
import copy
import numpy as np
from scipy.signal import convolve2d
from sys import stderr  # 標準エラー出力用
from base import (
    Global,
    NodeType,
    ActionType,
    SPACE_SIZE,
    get_match_step,
    warp_point,
    get_opposite,
    is_team_sector,
    get_match_number,
)
from debug import show_map, show_energy_field, show_exploration_map
from pathfinding import (
    astar,
    find_closest_target,
    nearby_positions,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
    manhattan_distance,
)

##############################################
# ヘルパー関数：点が優先探索領域（三角形）内にあるか判定
##############################################
def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def in_priority_triangle(point, start):
    if start == (23, 23):
        A = (23, 23)
        B = (23, 0)
        C = (0, 23)
    elif start == (0, 0):
        A = (0, 0)
        B = (23, 0)
        C = (0, 23)
    else:
        return True
    d1 = sign(point, A, B)
    d2 = sign(point, B, C)
    d3 = sign(point, C, A)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

##############################################
# 基本クラス：Node, Space, Ship, Fleet
##############################################
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown      # タイルの種類（unknown, nebula, asteroidなど）
        self.energy = None                # タイル上のエネルギー
        self.is_visible = False           # センサーで確認できるか否か
        self._relic = False               # 遺跡タイルか否か
        self._reward = False              # Rewardタイルか否か
        self._explored_for_relic = False  
        self._explored_for_reward = True  

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def relic(self):
        return self._relic

    @property
    def reward(self):
        return self._reward

    @property
    def explored_for_relic(self):
        return self._explored_for_relic

    @property
    def explored_for_reward(self):
        return self._explored_for_reward

    def update_relic_status(self, status: bool | None):
        if self._explored_for_relic and self._relic and status is False:
            raise ValueError(f"Can't change relic status for {self}")
        if status is None:
            self._explored_for_relic = False
            return
        self._relic = status
        self._explored_for_relic = True

    def update_reward_status(self, status: bool | None):
        if self._explored_for_reward and self._reward and status is False:
            raise ValueError(f"Can't change reward status for {self}")
        if status is None:
            self._explored_for_reward = False
            return
        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid

    @property
    def coordinates(self) -> tuple:
        return self.x, self.y

    def manhattan_distance(self, other: "Node") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

class Space:
    def __init__(self):
        self._nodes = [[Node(x, y) for x in range(SPACE_SIZE)] for y in range(SPACE_SIZE)]
        self._relic_nodes = set()
        self._reward_nodes = set()

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    @property
    def relic_nodes(self) -> set:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set:
        return self._reward_nodes

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def clear(self):
        for node in self:
            node.is_visible = False

    def update(self, step, obs, team_id, team_reward):
        self.move_obstacles(step)
        self._update_map(obs)
        self._update_relic_map(step, obs, team_id, team_reward)

    def _update_relic_map(self, step, obs, team_id, team_reward):
        for mask, xy in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask and not self.get_node(*xy).relic:
                self._update_relic_status(*xy, status=True)
                for x, y in nearby_positions(*xy, Global.RELIC_REWARD_RANGE):
                    if not self.get_node(x, y).reward:
                        self._update_reward_status(x, y, status=None)
        all_relics_found = True
        all_rewards_found = True
        for node in self:
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(*node.coordinates, status=False)
            if not node.explored_for_relic:
                all_relics_found = False
            if not node.explored_for_reward:
                all_rewards_found = False
        Global.ALL_RELICS_FOUND = all_relics_found
        Global.ALL_REWARDS_FOUND = all_rewards_found
        match = get_match_number(step)
        match_step = get_match_step(step)
        num_relics_th = 2 * min(match, Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1
        if not Global.ALL_RELICS_FOUND:
            if len(self._relic_nodes) >= num_relics_th:
                Global.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=False)
        if not Global.ALL_REWARDS_FOUND:
            if (match_step > Global.LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR or len(self._relic_nodes) >= num_relics_th):
                self._update_reward_status_from_relics_distribution()
                self._update_reward_results(obs, team_id, team_reward)
                self._update_reward_status_from_reward_results()

    def _update_reward_status_from_reward_results(self):
        for result in Global.REWARD_RESULTS:
            unknown_nodes = set()
            known_reward = 0
            for n in result["nodes"]:
                if n.explored_for_reward and not n.reward:
                    continue
                if n.reward:
                    known_reward += 1
                    continue
                unknown_nodes.add(n)
            if not unknown_nodes:
                continue
            reward = result["reward"] - known_reward
            if reward == 0:
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=False)
            elif reward == len(unknown_nodes):
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=True)

    def _update_reward_results(self, obs, team_id, team_reward):
        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                ship_nodes.add(self.get_node(*position))
        Global.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": team_reward})

    def _update_reward_status_from_relics_distribution(self):
        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        for node in self:
            if node.relic or not node.explored_for_relic:
                relic_map[node.y][node.x] = 1
        reward_size = 2 * Global.RELIC_REWARD_RANGE + 1
        reward_map = convolve2d(
            relic_map,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        for node in self:
            if reward_map[node.y][node.x] == 0:
                node.update_reward_status(False)

    def _update_relic_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_relic_status(status)
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_relic_status(status)
        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_reward_status(status)
        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    def _update_map(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]
        obstacles_shifted = False
        energy_nodes_shifted = False
        for node in self:
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]
            if is_visible and not node.is_unknown and node.type.value != obs_tile_type[x, y]:
                obstacles_shifted = True
            if is_visible and node.energy is not None and node.energy != obs_energy[x, y]:
                energy_nodes_shifted = True
        Global.OBSTACLES_MOVEMENT_STATUS.append(obstacles_shifted)
        def clear_map_info():
            for n in self:
                n.type = NodeType.unknown
        if not Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and obstacles_shifted:
            direction = self._find_obstacle_movement_direction(obs)
            if direction:
                Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                Global.OBSTACLE_MOVEMENT_DIRECTION = direction
                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)
            else:
                clear_map_info()
        if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            period = self._find_obstacle_movement_period(Global.OBSTACLES_MOVEMENT_STATUS)
            if period is not None:
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                Global.OBSTACLE_MOVEMENT_PERIOD = period
            if obstacles_shifted:
                clear_map_info()
        if obstacles_shifted and Global.OBSTACLE_MOVEMENT_PERIOD_FOUND and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND:
            clear_map_info()
        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])
            node.is_visible = is_visible
            if is_visible and node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))
                self.get_node(*get_opposite(x, y)).type = node.type
            if is_visible:
                node.energy = int(obs_energy[x, y])
                self.get_node(*get_opposite(x, y)).energy = node.energy
            elif energy_nodes_shifted:
                node.energy = None
            if is_visible:
                if node.reward:
                    Global.SHARED_REWARD_TILES.add(node.coordinates)
                else:
                    Global.SHARED_REWARD_TILES.discard(node.coordinates)

    def _find_obstacle_movement_period(self, obstacles_movement_status):
        if len(obstacles_movement_status) < 81:
            return
        num_movements = sum(obstacles_movement_status)
        if num_movements <= 2:
            return 40
        elif num_movements <= 4:
            return 20
        elif num_movements <= 8:
            return 10
        else:
            return 20 / 3

    def _find_obstacle_movement_direction(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_tile_type = obs["map_features"]["tile_type"]
        suitable_directions = []
        for direction in [(1, -1), (-1, 1)]:
            moved_space = self.move(*direction, inplace=False)
            match = True
            for node in moved_space:
                x, y = node.coordinates
                if sensor_mask[x, y] and not node.is_unknown and obs_tile_type[x, y] != node.type.value:
                    match = False
                    break
            if match:
                suitable_directions.append(direction)
        if len(suitable_directions) == 1:
            return suitable_directions[0]

    def move_obstacles(self, step):
        if (Global.OBSTACLE_MOVEMENT_PERIOD_FOUND and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and Global.OBSTACLE_MOVEMENT_PERIOD > 0):
            speed = 1 / Global.OBSTACLE_MOVEMENT_PERIOD
            if (step - 2) * speed % 1 > (step - 1) * speed % 1:
                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

    def move(self, dx: int, dy: int, *, inplace=False) -> "Space":
        if not inplace:
            new_space = copy.deepcopy(self)
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self

    def clear_exploration_info(self):
        Global.REWARD_RESULTS = []
        Global.ALL_RELICS_FOUND = False
        Global.ALL_REWARDS_FOUND = False
        for node in self:
            if not node.relic:
                self._update_relic_status(node.x, node.y, status=None)

##############################################
# Ship, Fleet クラス
##############################################
class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None
        self.task: str | None = None
        self.target: Node | None = None
        self.action: ActionType | None = None
        self.role: str | None = None
        self.prev_coordinates = None
        self.lock_turns = 0

    def __repr__(self):
        return f"Ship({self.unit_id}, node={self.node.coordinates if self.node else None}, energy={self.energy}, role={self.role})"

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clean(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None
        self.role = None
        self.prev_coordinates = None
        self.lock_turns = 0

class Fleet:
    def __init__(self, team_id):
        self.team_id = team_id
        self.points = 0
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clean()

    def update(self, obs, space: Space):
        self.points = int(obs["team_points"][self.team_id])
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
                ship.clean()

##############################################
# パラメータ推定・マップ管理クラス
##############################################
class MatchParameterEstimator:
    def __init__(self):
        self.estimated_params = {}
        self.observation_history = []

    def update(self, step, obs):
        self.observation_history.append(obs)
        if step > 5 and "nebula_drift_speed" not in self.estimated_params:
            self.estimated_params["nebula_drift_speed"] = 0.15

class MapManager:
    def __init__(self):
        self.space = Space()

    def update(self, step, obs, team_id, team_reward):
        self.space.move_obstacles(step)
        self.space._update_map(obs)
        self.space._update_relic_map(step, obs, team_id, team_reward)

##############################################
# タスク管理クラス
##############################################
class TaskManager:
    ENERGY_THRESHOLD = 50

    def __init__(self, fleet: Fleet, map_manager: MapManager):
        self.fleet = fleet
        self.map_manager = map_manager

    def update_tasks(self, step: int):
        for ship in self.fleet:
            if ship.node is not None and ship.node.reward and ship.node.energy is not None and ship.node.energy > 0:
                if ship.energy >= 5:
                    if ship.lock_turns <= 0:
                        ship.lock_turns = 5
                    else:
                        ship.lock_turns -= 1
                        continue
                else:
                    ship.task = "recharge"
                    ship.target = ship.node
                    ship.action = ActionType.center
                    ship.lock_turns = 0
        for ship in self.fleet:
            if ship.energy < 5 and not (ship.node and ship.node.reward and ship.node.energy and ship.node.energy > 0):
                ship.task = "recharge"
                ship.target = ship.node
                ship.action = ActionType.center
        for ship in self.fleet:
            if ship.task == "recharge" and ship.energy < 40:
                ship.action = ActionType.center
                continue

        self._assign_recharge_tasks()
        self._assign_reward_tile_tasks()
        self._assign_relic_tasks()
        self._assign_harvest_tasks()
        self._assign_exploration_and_target_tasks(step)

    def _assign_recharge_tasks(self):
        space = self.map_manager.space
        for ship in self.fleet:
            if ship.energy < 5:
                continue
            if ship.energy < self.ENERGY_THRESHOLD:
                best_tile, _ = self._find_best_recharge_tile(ship, space)
                if best_tile:
                    path = astar(create_weights(space), ship.coordinates, best_tile.coordinates)
                    if path:
                        ship.task = "recharge"
                        ship.target = best_tile
                        actions = path_to_actions(path)
                        if actions:
                            ship.action = actions[0]

    def _find_best_recharge_tile(self, ship, space):
        best_tile = None
        best_distance = float('inf')
        for node in space:
            if node.is_visible and node.energy is not None and node.energy >= 0.8 * Global.MAX_ENERGY_PER_TILE:
                d = manhattan_distance(ship.coordinates, node.coordinates)
                if d < best_distance:
                    best_distance = d
                    best_tile = node
        return best_tile, best_distance

    def _assign_exploration_and_target_tasks(self, step: int):
        space = self.map_manager.space
        candidate_targets = []
        start_pos = (0,0) if Global.TEAM_ID == 0 else (23,23)
        if not Global.ALL_RELICS_FOUND and not Global.ALL_REWARDS_FOUND:
            for node in space:
                if not node.is_visible:
                    if in_priority_triangle(node.coordinates, start_pos):
                        candidate_targets.append((node.coordinates, 200))
                    else:
                        candidate_targets.append((node.coordinates, 50))
        else:
            if step < 10:
                directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                for ship in self.fleet:
                    if ship.task is None and ship.energy >= Global.UNIT_MOVE_COST:
                        for d in directions:
                            target = warp_point(ship.coordinates[0] + d[0]*3, ship.coordinates[1] + d[1]*3)
                            if not (space.get_node(*target).is_visible and space.get_node(*target).reward):
                                candidate_targets.append((target, 100))
                for node in space:
                    if not node.is_visible:
                        candidate_targets.append((node.coordinates, 80))
            else:
                for node in space:
                    if not node.is_visible:
                        candidate_targets.append((node.coordinates, 100))
                    elif node.is_visible and node.is_walkable:
                        value = node.energy if node.energy is not None else 0
                        if node.relic:
                            value += 100
                        if node.reward:
                            value += 50
                        if value >= 60:
                            candidate_targets.append((node.coordinates, value))
        candidate_targets.sort(key=lambda x: -x[1])
        assigned_targets = set()
        for ship in self.fleet:
            if ship.task is not None or ship.energy < Global.UNIT_MOVE_COST:
                continue
            best_target = None
            best_distance = float('inf')
            for pos, _ in candidate_targets:
                if pos in assigned_targets:
                    continue
                distance = np.linalg.norm(np.array(ship.coordinates) - np.array(pos))
                if distance < best_distance:
                    best_distance = distance
                    best_target = pos
            if best_target is not None:
                path = astar(create_weights(space), ship.coordinates, best_target)
                if path:
                    if step < 10:
                        ship.task = "wide_exploration"
                    else:
                        if space.get_node(*best_target).reward:
                            ship.task = "find_rewards"
                        else:
                            ship.task = "area_control"
                    ship.target = space.get_node(*best_target)
                    actions = path_to_actions(path)
                    if actions:
                        ship.action = actions[0]
                    assigned_targets.add(best_target)
            else:
                fallback_target = None
                for nx, ny in nearby_positions(ship.coordinates[0], ship.coordinates[1], 1):
                    if (nx, ny) != ship.coordinates:
                        fallback_target = (nx, ny)
                        break
                if fallback_target is not None:
                    path = astar(create_weights(space), ship.coordinates, fallback_target)
                    if path:
                        ship.task = "fallback"
                        ship.target = space.get_node(*fallback_target)
                        actions = path_to_actions(path)
                        if actions:
                            ship.action = actions[0]

    def _assign_reward_tile_tasks(self):
        space = self.map_manager.space
        held_reward_tiles = {}
        for ship in self.fleet:
            if ship.task in ("harvest", "find_rewards") and ship.target is not None:
                coord = ship.target.coordinates
                held_reward_tiles.setdefault(coord, []).append(ship)
        for ship in self.fleet:
            if ship.task is None and ship.energy >= Global.UNIT_MOVE_COST and held_reward_tiles:
                best_target = None
                best_distance = float("inf")
                for coord, assigned in held_reward_tiles.items():
                    if len(assigned) < 1:
                        d = manhattan_distance(ship.coordinates, coord)
                        if d < best_distance:
                            best_distance = d
                            best_target = coord
                if best_target is not None:
                    path = astar(create_weights(space), ship.coordinates, best_target)
                    if path:
                        ship.task = "harvest"
                        ship.target = space.get_node(*best_target)
                        actions = path_to_actions(path)
                        if actions:
                            ship.action = actions[0]
                        held_reward_tiles[best_target].append(ship)
        for tile_coord in Global.SHARED_REWARD_TILES:
            if tile_coord not in held_reward_tiles:
                best_ship = None
                best_distance = float("inf")
                for ship in self.fleet:
                    if ship.task is None and ship.energy >= Global.UNIT_MOVE_COST:
                        d = manhattan_distance(ship.coordinates, tile_coord)
                        if d < best_distance:
                            best_distance = d
                            best_ship = ship
                if best_ship:
                    path = astar(create_weights(space), best_ship.coordinates, tile_coord)
                    if path:
                        best_ship.task = "harvest"
                        best_ship.target = space.get_node(*tile_coord)
                        actions = path_to_actions(path)
                        if actions:
                            best_ship.action = actions[0]
                        held_reward_tiles[tile_coord] = [best_ship]

    def _assign_relic_tasks(self):
        space = self.map_manager.space
        if Global.ALL_RELICS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return
        targets = {node.coordinates for node in space if not node.explored_for_relic and is_team_sector(self.fleet.team_id, *node.coordinates)}
        for ship in self.fleet:
            if ship.task is None:
                target, _ = find_closest_target(ship.coordinates, targets)
                if target:
                    path = astar(create_weights(space), ship.coordinates, target)
                    energy = estimate_energy_cost(space, path)
                    actions = path_to_actions(path)
                    if actions and ship.energy >= energy:
                        ship.task = "find_relics"
                        ship.target = space.get_node(*target)
                        ship.action = actions[0]
                        for x, y in path:
                            for xy in nearby_positions(x, y, Global.UNIT_SENSOR_RANGE):
                                targets.discard(xy)

    def _assign_harvest_tasks(self):
        space = self.map_manager.space
        def set_harvest_task(ship, target_node):
            if ship.node == target_node:
                if target_node.energy is not None and target_node.energy > 0:
                    ship.task = "harvest"
                    ship.target = target_node
                    ship.action = ActionType.center
                    return True
                else:
                    ship.task = "harvest"
                    ship.target = target_node
                    ship.action = ActionType.center
                    return True
            path = astar(create_weights(space), ship.coordinates, target_node.coordinates)
            energy = estimate_energy_cost(space, path)
            actions = path_to_actions(path)
            if not actions or ship.energy < energy:
                return False
            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        harvest_assignments = {}
        for ship in self.fleet:
            if ship.task == "harvest" and ship.target is not None:
                key = ship.target.coordinates
                harvest_assignments.setdefault(key, []).append(ship)
        for key, ships in harvest_assignments.items():
            if len(ships) > 1:
                ships.sort(key=lambda s: s.energy, reverse=True)
                for extra in ships[1:]:
                    extra.task = None
                    extra.target = None
                    extra.action = None
        for ship in self.fleet:
            if ship.task == "harvest":
                if ship.target is None:
                    ship.task = None
                    continue
                if ship.node == ship.target and ship.energy >= Global.UNIT_MOVE_COST:
                    if ship.target.energy is not None and ship.target.energy > 0:
                        ship.action = ActionType.center
                    else:
                        ship.action = ActionType.center
                else:
                    set_harvest_task(ship, ship.target)
        available_targets = {n.coordinates for n in space.reward_nodes if n.is_walkable}
        for key in harvest_assignments.keys():
            available_targets.discard(key)
        for ship in self.fleet:
            if ship.task is None and ship.energy >= Global.UNIT_MOVE_COST:
                target, _ = find_closest_target(ship.coordinates, list(available_targets))
                if target and set_harvest_task(ship, space.get_node(*target)):
                    available_targets.discard(target)
                else:
                    ship.task = None
                    ship.target = None

##############################################
# 役割割当（統合）
##############################################
def assign_roles_to_ships(fleet, step):
    center = Global.SPACE_SIZE // 2
    for ship in fleet.ships:
        if ship.node is None:
            continue
        if step < 10:
            x, y = ship.coordinates
            if abs(x - center) + abs(y - center) > Global.SPACE_SIZE // 2:
                ship.role = "scout"
            else:
                ship.role = "miner"
        else:
            ship.role = "miner"

##############################################
# 戦略モジュール：SAP 戦略を含む
##############################################
class StrategyModule:
    def __init__(self, map_manager: MapManager, task_manager: TaskManager, fleet: Fleet, opp_fleet: Fleet):
        self.map_manager = map_manager
        self.task_manager = task_manager
        self.fleet = fleet
        self.opp_fleet = opp_fleet

    def decide(self, step, obs):
        self.task_manager.update_tasks(step)
        self._assign_sap_actions()

    def _vulnerability_score(self, enemy):
        threshold = 50
        if enemy.energy < threshold:
            return (threshold - enemy.energy) / threshold
        return 0

    # 敵ユニットを検出した場合、SAP アクションを試みる
    def _assign_sap_actions(self):
        enemy_targeting = {}
        max_sap_per_tile = 1  # 同一敵ユニットに対して最大1台のSAPを割り当てる
        for enemy in self.opp_fleet.ships:
            if enemy.node is None:
                continue
            target_coord = enemy.node.coordinates
            current_assigned = enemy_targeting.get(target_coord, [])
            if len(current_assigned) >= max_sap_per_tile:
                continue
            for ally in self.fleet.ships:
                if ally.node is None:
                    continue
                # 既に他の重要タスクを抱えているユニットは除外
                if ally.task in ("harvest", "recharge", "sap"):
                    continue
                if ally.energy < Global.UNIT_SAP_COST:
                    continue
                path = astar(create_weights(self.map_manager.space), ally.coordinates, target_coord)
                if not path or len(path) == 0:
                    continue
                distance = manhattan_distance(ally.coordinates, target_coord)
                if distance <= Global.UNIT_SAP_RANGE:
                    ally.task = "sap"
                    ally.target = self.map_manager.space.get_node(*target_coord)
                    ally.action = ActionType.sap
                else:
                    ally.task = "sap"
                    ally.target = self.map_manager.space.get_node(*target_coord)
                    ally.action = path_to_actions(path)[0]
                enemy_targeting.setdefault(target_coord, []).append(ally)
                if len(enemy_targeting[target_coord]) >= max_sap_per_tile:
                    break

##############################################
# Agent クラス（統合版）
##############################################
class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.env_cfg = env_cfg
        self.start_position = (0, 0) if self.team_id == 0 else (23, 23)
        self.map_manager = MapManager()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(1 - self.team_id)
        self.task_manager = TaskManager(self.fleet, self.map_manager)
        self.strategy_module = StrategyModule(self.map_manager, self.task_manager, self.fleet, self.opp_fleet)
        self.param_estimator = MatchParameterEstimator()
        Global.TEAM_ID = self.team_id
        Global.START_POSITION = self.start_position

    def update_roles(self, step):
        assign_roles_to_ships(self.fleet, step)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        match_step = get_match_step(step)
        match_number = get_match_number(step)
        if match_step == 0:
            self.fleet.clear()
            self.opp_fleet.clear()
            self.map_manager.space.move_obstacles(step)
            return self.create_actions_array()
        team_points = int(obs["team_points"][self.team_id])
        reward = max(0, team_points - self.fleet.points)
        self.map_manager.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.map_manager.space)
        self.opp_fleet.update(obs, self.map_manager.space)
        # 敵ユニットの位置を全味方で共有
        Global.ENEMY_POSITIONS = [ship.node.coordinates for ship in self.opp_fleet if ship.node is not None]
        self.param_estimator.update(step, obs)
        self.update_roles(step)
        for ship in self.fleet:
            if ship.node is not None:
                if ship.prev_coordinates is not None and ship.coordinates == ship.prev_coordinates:
                    ship.task = None
                ship.prev_coordinates = ship.coordinates
        self.strategy_module.decide(step, obs)
        return self.create_actions_array()

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)
        for i, ship in enumerate(ships):
            if ship.action is not None:
                actions[i] = ship.action, 0, 0
        return actions



%%writefile agent/debug.py

from sys import stderr
from collections import defaultdict

from base import Global, NodeType


def show_energy_field(space, only_visible=True):
    line = " + " + " ".join([f"{x:>2}" for x in range(Global.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Global.SPACE_SIZE):

        str_row = []

        for x in range(Global.SPACE_SIZE):
            node = space.get_node(x, y)
            if node.energy is None or (only_visible and not node.is_visible):
                str_row.append(" ..")
            else:
                str_row.append(f"{node.energy:>3}")

        str_grid += "".join([f"{y:>2}", *str_row, f" {y:>2}", "\n"])

    str_grid += line
    print(str_grid, file=stderr)


def show_map(space, fleet=None, only_visible=True):
    """
    legend:
        n - nebula
        a - asteroid
        ~ - relic
        _ - reward
        1:H - ships
    """
    ship_signs = (
        [" "] + [str(x) for x in range(1, 10)] + ["A", "B", "C", "D", "E", "F", "H"]
    )

    ships = defaultdict(int)
    if fleet:
        for ship in fleet:
            ships[ship.node.coordinates] += 1

    line = " + " + " ".join([f"{x:>2}" for x in range(Global.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Global.SPACE_SIZE):

        str_row = []

        for x in range(Global.SPACE_SIZE):
            node = space.get_node(x, y)

            if node.type == NodeType.unknown or (only_visible and not node.is_visible):
                str_row.append("..")
                continue

            if node.type == NodeType.nebula:
                s1 = "ñ" if node.relic else "n"
            elif node.type == NodeType.asteroid:
                s1 = "ã" if node.relic else "a"
            else:
                s1 = "~" if node.relic else " "

            if node.reward:
                if s1 == " ":
                    s1 = "_"

            if node.coordinates in ships:
                num_ships = ships[node.coordinates]
                s2 = str(ship_signs[num_ships])
            else:
                s2 = " "

            str_row.append(s1 + s2)

        str_grid += " ".join([f"{y:>2}", *str_row, f"{y:>2}", "\n"])

    str_grid += line
    print(str_grid, file=stderr)


def show_exploration_map(space):
    """
    legend:
        R - relic
        P - reward
    """
    print(
        f"all relics found: {Global.ALL_RELICS_FOUND}, "
        f"all rewards found: {Global.ALL_REWARDS_FOUND}",
        file=stderr,
    )

    line = " + " + " ".join([f"{x:>2}" for x in range(Global.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Global.SPACE_SIZE):

        str_row = []

        for x in range(Global.SPACE_SIZE):
            node = space.get_node(x, y)
            if not node.explored_for_relic:
                s1 = "."
            else:
                s1 = "R" if node.relic else " "

            if not node.explored_for_reward:
                s2 = "."
            else:
                s2 = "P" if node.reward else " "

            str_row.append(s1 + s2)

        str_grid += " ".join([f"{y:>2}", *str_row, f"{y:>2}", "\n"])

    str_grid += line
    print(str_grid, file=stderr)



%%writefile agent/main.py

import json
from argparse import Namespace
from agent import Agent
from lux.kit import from_json

### DO NOT REMOVE THE FOLLOWING CODE ###
# store potentially multiple dictionaries as kaggle imports code directly
agent_dict = dict()
agent_prev_obs = dict()

def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
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
        """
        Reads input from stdin
        """
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
        # send actions to engine
        print(json.dumps(actions))



!pip install --upgrade luxai-s3


!luxai-s3 agent/main.py agent/main.py --output=replay1.html


import IPython # load the HTML replay
IPython.display.HTML(filename='replay1.html')


!cd agent && tar -czf submission.tar.gz *
!mv agent/submission.tar.gz .

