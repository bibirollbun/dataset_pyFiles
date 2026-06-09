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
    UNIT_SENSOR_RANGE = 1  # OPTIONS: [1, 2, 3, 4]
    OBSTACLE_MOVEMENT_PERIOD = 7  # OPTIONS: 7(?), 10, 20, 40
    OBSTACLE_MOVEMENT_DIRECTION = (0, 0)  # OPTIONS: [(1, -1), (-1, 1)]

    # We will NOT find the exact value of these constants during the game
    NEBULA_ENERGY_REDUCTION = 5  # OPTIONS: [0, 1, 2, 3, 5, 25]

    # Exploration flags:

    ALL_RELICS_FOUND = False
    ALL_REWARDS_FOUND = False
    OBSTACLE_MOVEMENT_PERIOD_FOUND = False
    OBSTACLE_MOVEMENT_DIRECTION_FOUND = False

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

import copy
import numpy as np
from sys import stderr
from scipy.signal import convolve2d

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


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.energy = None
        self.is_visible = False

        self._relic = False
        self._reward = False
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

    def update_relic_status(self, status: None | bool):
        if self._explored_for_relic and self._relic and not status:
            raise ValueError(
                f"Can't change the relic status {self._relic}->{status} for {self}"
                ", the tile has already been explored"
            )

        if status is None:
            self._explored_for_relic = False
            return

        self._relic = status
        self._explored_for_relic = True

    def update_reward_status(self, status: None | bool):
        if self._explored_for_reward and self._reward and not status:
            raise ValueError(
                f"Can't change the reward status {self._reward}->{status} for {self}"
                ", the tile has already been explored"
            )

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
    def coordinates(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Node") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


class Space:
    def __init__(self):
        self._nodes: list[list[Node]] = []
        for y in range(SPACE_SIZE):
            row = [Node(x, y) for x in range(SPACE_SIZE)]
            self._nodes.append(row)

        # set of nodes with a relic
        self._relic_nodes: set[Node] = set()

        # set of nodes that provide points
        self._reward_nodes: set[Node] = set()

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def update(self, step, obs, team_id, team_reward):
        self.move_obstacles(step)
        self._update_map(obs)
        self._update_relic_map(step, obs, team_id, team_reward)

    def _update_relic_map(self, step, obs, team_id, team_reward):
        for mask, xy in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask and not self.get_node(*xy).relic:
                # We have found a new relic.
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
                # all relics found, mark all nodes as explored for relics
                Global.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=False)

        if not Global.ALL_REWARDS_FOUND:
            if (
                match_step > Global.LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR
                or len(self._relic_nodes) >= num_relics_th
            ):
                self._update_reward_status_from_relics_distribution()
                self._update_reward_results(obs, team_id, team_reward)
                self._update_reward_status_from_reward_results()

    def _update_reward_status_from_reward_results(self):
        # We will use Global.REWARD_RESULTS to identify which nodes yield points
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
                # all nodes already explored, nothing to do here
                continue

            reward = result["reward"] - known_reward  # reward from unknown_nodes

            if reward == 0:
                # all nodes are empty
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=False)

            elif reward == len(unknown_nodes):
                # all nodes yield points
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=True)

            elif reward > len(unknown_nodes):
                # we shouldn't be here
                print(
                    f"Something wrong with reward result: {result}"
                    ", this result will be ignored.",
                    file=stderr,
                )

    def _update_reward_results(self, obs, team_id, team_reward):
        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                # Only units with non-negative energy can give points
                ship_nodes.add(self.get_node(*position))

        Global.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": team_reward})

    def _update_reward_status_from_relics_distribution(self):
        # Rewards can only occur near relics.
        # Therefore, if there are no relics near the node
        # we can infer that the node does not contain a reward.

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
                # no relics in range RELIC_REWARD_RANGE
                node.update_reward_status(False)

    def _update_relic_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_relic_status(status)

        # relics are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_relic_status(status)

        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)

        # rewards are symmetrical
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

            if (
                is_visible
                and not node.is_unknown
                and node.type.value != obs_tile_type[x, y]
            ):
                obstacles_shifted = True

            if (
                is_visible
                and node.energy is not None
                and node.energy != obs_energy[x, y]
            ):
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
            period = self._find_obstacle_movement_period(
                Global.OBSTACLES_MOVEMENT_STATUS
            )
            if period is not None:
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                Global.OBSTACLE_MOVEMENT_PERIOD = period

            if obstacles_shifted:
                clear_map_info()

        if (
            obstacles_shifted
            and Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
        ):
            # maybe something is wrong
            clear_map_info()

        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])

            node.is_visible = is_visible

            if is_visible and node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))

                # we can also update the node type on the other side of the map
                # because the map is symmetrical
                self.get_node(*get_opposite(x, y)).type = node.type

            if is_visible:
                node.energy = int(obs_energy[x, y])

                # the energy field should be symmetrical
                self.get_node(*get_opposite(x, y)).energy = node.energy

            elif energy_nodes_shifted:
                # The energy field has changed
                # I cannot predict what the new energy field will be like.
                node.energy = None

    @staticmethod
    def _find_obstacle_movement_period(obstacles_movement_status):
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
            return 7

    def _find_obstacle_movement_direction(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_tile_type = obs["map_features"]["tile_type"]

        suitable_directions = []
        for direction in [(1, -1), (-1, 1)]:
            moved_space = self.move(*direction, inplace=False)

            match = True
            for node in moved_space:
                x, y = node.coordinates
                if (
                    sensor_mask[x, y]
                    and not node.is_unknown
                    and obs_tile_type[x, y] != node.type.value
                ):
                    match = False
                    break

            if match:
                suitable_directions.append(direction)

        if len(suitable_directions) == 1:
            return suitable_directions[0]

    def clear(self):
        for node in self:
            node.is_visible = False

    def move_obstacles(self, step):
        if (
            Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            and Global.OBSTACLE_MOVEMENT_PERIOD > 0
            and (step - 1) % Global.OBSTACLE_MOVEMENT_PERIOD == 0
        ):
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


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None

        self.task: str | None = None
        self.target: Node | None = None
        self.action: ActionType | None = None

    def __repr__(self):
        return (
            f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy})"
        )

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clean(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0  # how many points have we scored in this match so far
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


class Agent:

    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        match_step = get_match_step(step)
        match_number = get_match_number(step)

        # print(f"start step={match_step}({step}, {match_number})", file=stderr)

        if match_step == 0:
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.move_obstacles(step)
            if match_number <= Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR:
                self.space.clear_exploration_info()
            return self.create_actions_array()

        points = int(obs["team_points"][self.team_id])

        # how many points did we score in the last step
        reward = max(0, points - self.fleet.points)

        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

        # self.show_explored_map()

        self.find_relics()
        self.find_rewards()
        self.harvest()

        # for ship in self.fleet:
        #     print(ship, ship.task, ship.target, ship.action, file=stderr)

        return self.create_actions_array()

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action is not None:
                actions[i] = ship.action, 0, 0

        return actions

    def find_relics(self):
        if Global.ALL_RELICS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        targets = set()
        for node in self.space:
            if not node.explored_for_relic:
                # We will only find relics in our part of the map
                # because relics are symmetrical.
                if is_team_sector(self.fleet.team_id, *node.coordinates):
                    targets.add(node.coordinates)

        def set_task(ship):
            if ship.task and ship.task != "find_relics":
                return False

            if ship.energy < Global.UNIT_MOVE_COST:
                return False

            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return False

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "find_relics"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

                for x, y in path:
                    for xy in nearby_positions(x, y, Global.UNIT_SENSOR_RANGE):
                        if xy in targets:
                            targets.remove(xy)

                return True

            return False

        for ship in self.fleet:
            if set_task(ship):
                continue

            if ship.task == "find_relics":
                ship.task = None
                ship.target = None

    def find_rewards(self):
        if Global.ALL_REWARDS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        unexplored_relics = self.get_unexplored_relics()

        relic_node_to_ship = {}
        for ship in self.fleet:
            if ship.task == "find_rewards":
                if ship.target is None:
                    ship.task = None
                    continue

                if (
                    ship.target in unexplored_relics
                    and ship.energy > Global.UNIT_MOVE_COST * 5
                ):
                    relic_node_to_ship[ship.target] = ship
                else:
                    ship.task = None
                    ship.target = None

        for relic in unexplored_relics:
            if relic not in relic_node_to_ship:

                # find the closest ship to the relic node
                min_distance, closes_ship = float("inf"), None
                for ship in self.fleet:
                    if ship.task and ship.task != "find_rewards":
                        continue

                    if ship.energy < Global.UNIT_MOVE_COST * 5:
                        continue

                    distance = manhattan_distance(ship.coordinates, relic.coordinates)
                    if distance < min_distance:
                        min_distance, closes_ship = distance, ship

                if closes_ship:
                    relic_node_to_ship[relic] = closes_ship

        def set_task(ship, relic_node, can_pause):
            targets = []
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    targets.append((x, y))

            target, _ = find_closest_target(ship.coordinates, targets)

            if target == ship.coordinates and not can_pause:
                target, _ = find_closest_target(
                    ship.coordinates,
                    [
                        n.coordinates
                        for n in self.space
                        if n.explored_for_reward and n.is_walkable
                    ],
                )

            if not target:
                return

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

        can_pause = True
        for n, s in sorted(
            list(relic_node_to_ship.items()), key=lambda _: _[1].unit_id
        ):
            if set_task(s, n, can_pause):
                if s.target == s.node:
                    # If one ship is stationary, we will move all the other ships.
                    # This will help generate more useful data in Global.REWARD_RESULTS.
                    can_pause = False
            else:
                if s.task == "find_rewards":
                    s.task = None
                    s.target = None

    def get_unexplored_relics(self) -> list[Node]:
        relic_nodes = []
        for relic_node in self.space.relic_nodes:
            if not is_team_sector(self.team_id, *relic_node.coordinates):
                continue

            explored = True
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    explored = False
                    break

            if explored:
                continue

            relic_nodes.append(relic_node)

        return relic_nodes

    def harvest(self):

        def set_task(ship, target_node):
            if ship.node == target_node:
                ship.task = "harvest"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False

            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "harvest":
                if ship.target is None:
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if n.is_walkable and n not in booked_nodes:
                targets.add(n.coordinates)
        if not targets:
            return

        for ship in self.fleet:
            if ship.task:
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                targets.remove(target)
            else:
                ship.task = None
                ship.target = None

    def show_visible_energy_field(self):
        print("Visible energy field:", file=stderr)
        show_energy_field(self.space)

    def show_explored_energy_field(self):
        print("Explored energy field:", file=stderr)
        show_energy_field(self.space, only_visible=False)

    def show_visible_map(self):
        print("Visible map:", file=stderr)
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_explored_map(self):
        print("Explored map:", file=stderr)
        show_map(self.space, self.fleet, only_visible=False)

    def show_exploration_map(self):
        print("Exploration map:", file=stderr)
        show_exploration_map(self.space)


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
                s1 = "単" if node.relic else "n"
            elif node.type == NodeType.asteroid:
                s1 = "達" if node.relic else "a"
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
import os
import random
import numpy as np
from argparse import Namespace
from agent import Agent
from lux.kit import from_json

### DO NOT REMOVE THE FOLLOWING CODE ###
# store potentially multiple dictionaries as kaggle imports code directly
agent_dict = dict()
agent_prev_obs = dict()

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    seed_everything(seed=0)
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


!luxai-s3 agent/main.py agent/main.py --output=replay.html


import IPython # load the HTML replay
IPython.display.HTML(filename='replay.html')


!cd agent && tar -czf submission.tar.gz *
!mv agent/submission.tar.gz .




