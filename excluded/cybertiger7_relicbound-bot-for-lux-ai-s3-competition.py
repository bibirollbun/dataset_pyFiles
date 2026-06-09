! mkdir agent
! cp -r /kaggle/input/lux-ai-season-3/lux agent


%%writefile agent/base.py

# Standard library imports
import json
import heapq
import math
from enum import IntEnum
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from collections import defaultdict, deque
import random

# Third-party imports
import numpy as np

"""
Game constants and core definitions for the Lux AI Season 3 agent.
These definitions establish the fundamental structures and rules of the game.
"""

# Map dimensions
SPACE_SIZE = 24

# =========================================
# Enumerations for game elements
# =========================================

class NodeType(IntEnum):
    """
    Types of tiles that can exist on the map.
    
    Attributes:
        unknown: Tile that hasn't been observed yet
        empty: Empty space that units can move through
        nebula: Special tile that affects vision and energy
        asteroid: Impassable tile that blocks movement
    """
    unknown = -1
    empty = 0
    nebula = 1
    asteroid = 2
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name


class ActionType(IntEnum):
    """
    Types of actions that units can perform.
    
    Attributes:
        center: Stay in place
        up: Move up (decrease y)
        right: Move right (increase x)
        down: Move down (increase y)
        left: Move left (decrease x)
        sap: Perform a sap action against opponents
    """
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
        """
        Determine the action needed to move from current to next position.
        
        Args:
            current_position: (x, y) starting coordinates
            next_position: (x, y) target coordinates
            
        Returns:
            The ActionType needed to move in that direction
        """
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
        """
        Convert action to a direction vector.
        
        Returns:
            (dx, dy) tuple representing the direction of movement
        """
        # Pre-computed direction vectors for each action
        directions = [
            (0, 0),   # center - no movement
            (0, -1),  # up - decrease y
            (1, 0),   # right - increase x
            (0, 1),   # down - increase y
            (-1, 0),  # left - decrease x
            (0, 0),   # sap - no movement
        ]
        return directions[self]


class ShipRole:
    """
    Defines possible roles that can be assigned to ships.
    
    These roles determine ship behavior and priorities:
    - EXPLORER: Focus on discovering the map and finding resources
    - HARVESTER: Focus on collecting rewards from known reward tiles
    - DEFENDER: Protect valuable areas from enemy ships
    - ATTACKER: Actively target and disrupt enemy ships
    """
    EXPLORER = "explorer"
    HARVESTER = "harvester"
    DEFENDER = "defender"
    ATTACKER = "attacker"


class Global:
    """
    Container for global game parameters and state.
    
    Contains both known constants and parameters that will
    be discovered during gameplay.
    """
    # ========== Fixed game constants ==========
    SPACE_SIZE = 24
    MAX_UNITS = 16
    RELIC_REWARD_RANGE = 2
    MAX_STEPS_IN_MATCH = 100
    MAX_ENERGY_PER_TILE = 20
    MAX_RELIC_NODES = 6
    LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR = 50
    LAST_MATCH_WHEN_RELIC_CAN_APPEAR = 2
    
    # ========== Variable parameters (discovered during gameplay) ==========
    # Ship parameters
    UNIT_MOVE_COST = 1               # Energy cost for movement
    UNIT_SAP_COST = 30               # Energy cost for sap action
    UNIT_SAP_RANGE = 3               # Range of sap action
    UNIT_SENSOR_RANGE = 2            # Vision range around ships
    
    # Nebula effects
    NEBULA_ENERGY_REDUCTION = 5      # Energy reduction from nebula tiles
    NEBULA_TILE_VISION_REDUCTION = 1 # Vision reduction from nebula tiles
    
    # Combat parameters
    UNIT_SAP_DROPOFF_FACTOR = 0.5    # Damage reduction per distance
    UNIT_ENERGY_VOID_FACTOR = 0.125  # Energy void field strength factor
    
    # ========== Movement prediction ==========
    OBSTACLE_MOVEMENT_PERIOD = 20           # How often obstacles move
    OBSTACLE_MOVEMENT_DIRECTION = (0, 0)    # Direction of movement
    
    # ========== State tracking flags ==========
    # Exploration progress
    ALL_RELICS_FOUND = False
    ALL_REWARDS_FOUND = False
    
    # Movement detection
    OBSTACLE_MOVEMENT_PERIOD_FOUND = False
    OBSTACLE_MOVEMENT_DIRECTION_FOUND = False
    
    # ========== Historical data ==========
    REWARD_RESULTS = []              # History of rewards collected
    OBSTACLES_MOVEMENT_STATUS = []   # History of obstacle movements
    
    # ========== Miscellaneous ==========
    HIDDEN_NODE_ENERGY = 0           # Default energy for unseen nodes


# =========================================
# Utility functions
# =========================================

def get_match_step(step: int) -> int:
    """
    Convert global step to step within the current match.
    
    Args:
        step: Global step count since start of game
        
    Returns:
        Step within the current match (0-100)
    """
    return step % (Global.MAX_STEPS_IN_MATCH + 1)


def get_match_number(step: int) -> int:
    """
    Calculate which match we're currently in (1-5).
    
    Args:
        step: Global step count since start of game
        
    Returns:
        Current match number (0-indexed)
    """
    return step // (Global.MAX_STEPS_IN_MATCH + 1)


def warp_int(x: int) -> int:
    """
    Wrap a coordinate around the map edges.
    
    Args:
        x: Input coordinate
        
    Returns:
        Coordinate wrapped to stay within map bounds
    """
    if x >= SPACE_SIZE:
        x -= SPACE_SIZE
    elif x < 0:
        x += SPACE_SIZE
    return x


def warp_point(x: int, y: int) -> tuple:
    """
    Wrap both coordinates of a point around map edges.
    
    Args:
        x, y: Input coordinates
        
    Returns:
        (x, y) tuple with both coordinates wrapped
    """
    return warp_int(x), warp_int(y)


def get_opposite(x: int, y: int) -> tuple:
    """
    Get the point symmetric to (x,y) across the map.
    
    Used for symmetry-based map exploration and knowledge sharing.
    
    Args:
        x, y: Original coordinates
        
    Returns:
        (x', y') coordinates of the symmetric point
    """
    return SPACE_SIZE - y - 1, SPACE_SIZE - x - 1


def is_upper_sector(x: int, y: int) -> bool:
    """
    Determine if coordinates are in the upper sector of the map.
    
    Args:
        x, y: Position to check
        
    Returns:
        True if position is in upper sector, False otherwise
    """
    return SPACE_SIZE - x - 1 >= y


def is_lower_sector(x: int, y: int) -> bool:
    """
    Determine if coordinates are in the lower sector of the map.
    
    Args:
        x, y: Position to check
        
    Returns:
        True if position is in lower sector, False otherwise
    """
    return SPACE_SIZE - x - 1 <= y


def is_team_sector(team_id: int, x: int, y: int) -> bool:
    """
    Check if coordinates are in the specified team's sector.
    
    Args:
        team_id: Team identifier (0 or 1)
        x, y: Position to check
        
    Returns:
        True if position is in team's sector, False otherwise
    """
    return is_upper_sector(x, y) if team_id == 0 else is_lower_sector(x, y)


def manhattan_distance(pos1: tuple, pos2: tuple) -> int:
    """
    Calculate Manhattan distance between two points.
    
    Manhattan distance is the sum of the absolute differences of their
    Cartesian coordinates, which represents movement along grid lines.
    
    Args:
        pos1: First position (x1, y1)
        pos2: Second position (x2, y2)
        
    Returns:
        Manhattan distance between the points
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


%%writefile agent/pathfinding.py

from base import *

def create_weights(space, avoid_enemies=False, enemy_positions=None):
    """
    Create a weight map for pathfinding with optional enemy avoidance.
    
    Args:
        space: The game space containing nodes and their properties
        avoid_enemies: Whether to add penalties for enemy-controlled areas
        enemy_positions: List of (x,y) coordinates of enemy ships
        
    Returns:
        2D numpy array of weights where higher values mean higher movement cost
    """
    # Pre-allocate the weight matrix
    weights = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
    
    # Cache commonly accessed values for performance
    max_energy = Global.MAX_ENERGY_PER_TILE + 1
    nebula_penalty = Global.NEBULA_ENERGY_REDUCTION
    hidden_energy = Global.HIDDEN_NODE_ENERGY
    
    # Process nodes in batches for better vectorization
    # First mark all impassable tiles
    for y in range(SPACE_SIZE):
        for x in range(SPACE_SIZE):
            node = space.get_node(x, y)
            
            if not node.is_walkable:
                weights[y, x] = -1  # Impassable
    
    # Then process all walkable tiles
    for y in range(SPACE_SIZE):
        for x in range(SPACE_SIZE):
            if weights[y, x] < 0:  # Skip already marked impassable tiles
                continue
                
            node = space.get_node(x, y)
            
            # Energy contributes inversely to weight (more energy = lower weight)
            node_energy = node.energy if node.energy is not None else hidden_energy
                
            # Base weight calculation
            weight = max_energy - node_energy
            
            # Add nebula penalty if applicable
            if node.type == NodeType.nebula:
                weight += nebula_penalty
                
            weights[y, x] = weight
    
    # Add enemy avoidance penalties if requested
    if avoid_enemies and enemy_positions and len(enemy_positions) > 0:
        # Pre-compute the danger mask for better performance
        danger_mask = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        sap_range = Global.UNIT_SAP_RANGE
        
        for x, y in enemy_positions:
            # Calculate bounds for the danger area to avoid unnecessary iterations
            x_min = max(0, x - sap_range)
            x_max = min(SPACE_SIZE - 1, x + sap_range)
            y_min = max(0, y - sap_range)
            y_max = min(SPACE_SIZE - 1, y + sap_range)
            
            for nx in range(x_min, x_max + 1):
                for ny in range(y_min, y_max + 1):
                    # Check if within sap range square
                    if abs(nx - x) <= sap_range and abs(ny - y) <= sap_range:
                        # Calculate Manhattan distance for penalty scaling
                        distance = abs(nx - x) + abs(ny - y)
                        
                        # Higher penalty closer to enemy, with a minimum value
                        if distance > 0:
                            danger_mask[ny, nx] += 10.0 / distance
                        else:
                            danger_mask[ny, nx] += 20.0  # Direct position has highest penalty
        
        # Apply the danger mask only to traversable tiles
        traversable = (weights >= 0)
        weights = weights + (danger_mask * traversable)
    
    return weights

def astar(weights, start, goal):
    """
    A* pathfinding algorithm optimized for performance.
    
    Args:
        weights: 2D numpy array of movement costs
        start: (x,y) starting coordinates
        goal: (x,y) goal coordinates
        
    Returns:
        List of (x,y) coordinates forming the optimal path, or empty list if no path
    """
    # Early exit for same start and goal
    if start == goal:
        return [start]
        
    # Early exit if goal is impassable
    goal_x, goal_y = goal
    if weights[goal_y, goal_x] < 0:
        return []
    
    # Get valid weights for heuristic calculation
    valid_mask = weights >= 0
    if not np.any(valid_mask):
        return []  # No valid tiles
        
    min_weight = np.min(weights[valid_mask])
    
    # Heuristic function: minimum possible cost * Manhattan distance
    def heuristic(p1, p2):
        return min_weight * manhattan_distance(p1, p2)

    # Priority queue for open set
    open_set = []
    
    # Tracking arrays: one for g-scores, one for parents, one for closed set
    g_scores = np.full((SPACE_SIZE, SPACE_SIZE), np.inf, dtype=np.float32)
    parents = np.full((SPACE_SIZE, SPACE_SIZE, 2), -1, dtype=np.int32)
    closed_set = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
    
    # Initialize start node
    g_scores[start[1], start[0]] = 0
    f_score = heuristic(start, goal)
    heapq.heappush(open_set, (f_score, start))
    
    # Direction vectors for neighbors
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Down, Up, Right, Left
    
    # Main A* loop
    while open_set:
        # Get node with lowest f_score
        _, current = heapq.heappop(open_set)
        current_x, current_y = current
        
        # Skip if already processed
        if closed_set[current_y, current_x]:
            continue
            
        # Mark as processed
        closed_set[current_y, current_x] = True
        
        # Check if we reached the goal
        if current == goal:
            return reconstruct_path(parents, start, goal)
        
        # Get current g-score
        current_g = g_scores[current_y, current_x]
        
        # Process neighbors
        for dx, dy in directions:
            neighbor_x, neighbor_y = current_x + dx, current_y + dy
            
            # Skip out-of-bounds neighbors
            if not (0 <= neighbor_x < SPACE_SIZE and 0 <= neighbor_y < SPACE_SIZE):
                continue
                
            # Skip impassable neighbors
            if weights[neighbor_y, neighbor_x] < 0:
                continue
                
            # Skip already processed neighbors
            if closed_set[neighbor_y, neighbor_x]:
                continue
            
            # Calculate tentative g-score
            tentative_g = current_g + weights[neighbor_y, neighbor_x]
            
            # If this path is better than previous ones
            if tentative_g < g_scores[neighbor_y, neighbor_x]:
                # Update path
                parents[neighbor_y, neighbor_x] = [current_x, current_y]
                g_scores[neighbor_y, neighbor_x] = tentative_g
                
                # Calculate f-score and add to open set
                f_score = tentative_g + heuristic((neighbor_x, neighbor_y), goal)
                heapq.heappush(open_set, (f_score, (neighbor_x, neighbor_y)))
    
    # No path found
    return []

def reconstruct_path(parents, start, goal):
    """
    Reconstruct the path from parents array.
    
    Args:
        parents: 3D array where parents[y, x] gives the parent coordinates of (x,y)
        start: Starting position (x,y)
        goal: Goal position (x,y)
        
    Returns:
        List of coordinates from start to goal
    """
    path = [goal]
    current = goal
    
    while current != start:
        current_x, current_y = current
        parent_x, parent_y = parents[current_y, current_x]
        
        # Check for invalid parent
        if parent_x == -1 or parent_y == -1:
            return []  # Broken path
            
        current = (parent_x, parent_y)
        path.append(current)
    
    # Reverse to get path from start to goal
    return path[::-1]

def find_closest_target(start, targets):
    """
    Find the closest target position to a starting position.
    
    Args:
        start: Starting position (x,y)
        targets: List of target positions [(x,y), ...]
        
    Returns:
        Tuple of (closest_target, distance_to_target) or (None, inf) if no targets
    """
    if not targets:
        return None, float("inf")
    
    # For small number of targets, linear search is fine
    if len(targets) < 50:
        target, min_distance = None, float("inf")
        for t in targets:
            d = manhattan_distance(start, t)
            if d < min_distance:
                target, min_distance = t, d
        return target, min_distance
    
    # For larger sets, use numpy for vectorized operations
    targets_array = np.array(list(targets))
    start_array = np.array(start)
    
    # Calculate Manhattan distances using broadcasting
    distances = np.abs(targets_array - start_array).sum(axis=1)
    
    # Find index of minimum distance
    min_idx = np.argmin(distances)
    
    return tuple(targets_array[min_idx]), distances[min_idx]

def estimate_energy_cost(space, path):
    """
    Estimate the total energy cost/gain of following a path.
    
    Args:
        space: Game space object with node information
        path: List of (x,y) coordinates forming a path
        
    Returns:
        Estimated net energy cost (positive means energy spent, negative means energy gained)
    """
    if len(path) <= 1:
        return 0

    energy = 0
    move_cost = Global.UNIT_MOVE_COST
    nebula_cost = Global.NEBULA_ENERGY_REDUCTION
    hidden_energy = Global.HIDDEN_NODE_ENERGY
    
    # First position doesn't cost movement energy but may provide tile energy
    x, y = path[0]
    node = space.get_node(x, y)
    if node.energy is not None:
        energy -= node.energy
    else:
        energy -= hidden_energy
        
    # Add nebula cost if applicable
    if node.type == NodeType.nebula:
        energy += nebula_cost
    
    # Process remaining positions
    for i in range(1, len(path)):
        x, y = path[i]
        node = space.get_node(x, y)
        
        # Add movement cost
        energy += move_cost
        
        # Subtract energy from tile
        if node.energy is not None:
            energy -= node.energy
        else:
            energy -= hidden_energy
        
        # Add nebula cost if applicable
        if node.type == NodeType.nebula:
            energy += nebula_cost

    return energy

def path_to_actions(path):
    """
    Convert a path to a list of actions.
    
    Args:
        path: List of (x,y) coordinates forming a path
        
    Returns:
        List of ActionType values representing moves to follow the path
    """
    if not path or len(path) <= 1:
        return []

    actions = []
    for i in range(1, len(path)):
        current = path[i-1]
        next_pos = path[i]
        direction = ActionType.from_coordinates(current, next_pos)
        actions.append(direction)

    return actions

class OpponentTracker:
    def __init__(self, history_length=5):
        """
        Initialize the opponent tracker.
        
        Args:
            history_length: Number of past positions to store for each ship
        """
        self.ship_history = {}  # unit_id -> list of past positions
        self.history_length = history_length
        self.predicted_positions = {}  # unit_id -> predicted next position
        self.dangerous_areas_cache = None  # Cached dangerous areas
        self.cache_step = -1  # Step when cache was last updated
        
    def update(self, obs, opp_team_id):
        """
        Update the history of opponent ship positions.
        
        Args:
            obs: Current game observation
            opp_team_id: ID of the opponent team
        """
        # Reset cache
        self.dangerous_areas_cache = None
        
        # Get opponent ship positions
        active_ships = obs["units_mask"][opp_team_id]
        positions = obs["units"]["position"][opp_team_id]
        
        # Update history for each ship
        for unit_id, (active, position) in enumerate(zip(active_ships, positions)):
            if active:
                # Initialize history if needed
                if unit_id not in self.ship_history:
                    self.ship_history[unit_id] = []
                
                # Append current position
                current_pos = tuple(position)
                history = self.ship_history[unit_id]
                
                # Only append if position changed or history is empty
                if not history or current_pos != history[-1]:
                    history.append(current_pos)
                    
                    # Keep history within length limit
                    if len(history) > self.history_length:
                        history.pop(0)
                    
                    # Update prediction when position changes
                    self.predicted_positions[unit_id] = self._predict_single_position(unit_id)
            else:
                # Remove inactive ships
                if unit_id in self.ship_history:
                    del self.ship_history[unit_id]
                if unit_id in self.predicted_positions:
                    del self.predicted_positions[unit_id]
    
    def _predict_single_position(self, unit_id):
        """
        Predict the next position for a single ship based on its movement history.
        
        Args:
            unit_id: ID of the ship
            
        Returns:
            Predicted (x,y) position or None if prediction isn't possible
        """
        history = self.ship_history.get(unit_id, [])
        
        if len(history) < 2:
            return history[-1] if history else None
        
        # Analyze recent movements to find patterns
        recent_movements = []
        for i in range(1, len(history)):
            prev_x, prev_y = history[i-1]
            curr_x, curr_y = history[i]
            recent_movements.append((curr_x - prev_x, curr_y - prev_y))
        
        # Count movement frequencies
        movement_counts = {}
        for m in recent_movements:
            movement_counts[m] = movement_counts.get(m, 0) + 1
        
        if not movement_counts:
            return history[-1]
        
        # Find most common movement
        most_common = max(movement_counts.items(), key=lambda x: x[1])[0]
        
        # Apply to current position
        curr_x, curr_y = history[-1]
        dx, dy = most_common
        new_x = max(0, min(SPACE_SIZE - 1, curr_x + dx))
        new_y = max(0, min(SPACE_SIZE - 1, curr_y + dy))
        
        return new_x, new_y
    
    def predict_next_position(self, unit_id):
        """
        Get the predicted next position for a ship.
        
        Args:
            unit_id: ID of the ship
            
        Returns:
            Predicted (x,y) position or None if prediction isn't possible
        """
        # Use cached prediction if available
        if unit_id in self.predicted_positions:
            return self.predicted_positions[unit_id]
            
        # Otherwise calculate and cache
        prediction = self._predict_single_position(unit_id)
        self.predicted_positions[unit_id] = prediction
        return prediction
        
    def get_dangerous_areas(self, sap_range, current_step=None):
        """
        Identify areas that might be dangerous due to opponent ships.
        
        Args:
            sap_range: Range of ship sap action
            current_step: Current game step for cache invalidation
            
        Returns:
            Set of (x,y) tuples representing dangerous positions
        """
        # Use cached result if available for this step
        if current_step is not None and self.cache_step == current_step and self.dangerous_areas_cache is not None:
            return self.dangerous_areas_cache
        
        dangerous_areas = set()
        
        # Process all opponent ships
        for unit_id, history in self.ship_history.items():
            if not history:
                continue
                
            # Current position and surrounding area
            curr_pos = history[-1]
            dangerous_areas.add(curr_pos)
            
            # Add sap range around current position
            self._add_danger_area(dangerous_areas, curr_pos, sap_range)
            
            # Add predicted position and surrounding area
            next_pos = self.predict_next_position(unit_id)
            if next_pos and next_pos != curr_pos:  # Only if different from current
                dangerous_areas.add(next_pos)
                self._add_danger_area(dangerous_areas, next_pos, sap_range)
        
        # Cache result if step provided
        if current_step is not None:
            self.dangerous_areas_cache = dangerous_areas
            self.cache_step = current_step
            
        return dangerous_areas
    
    def _add_danger_area(self, danger_set, center_pos, sap_range):
        """
        Add an area around a position to the danger set.
        
        Args:
            danger_set: Set to add points to
            center_pos: Center position (x,y)
            sap_range: Range to mark as dangerous
        """
        x, y = center_pos
        
        # Calculate bounds to avoid unnecessary checks
        x_min = max(0, x - sap_range)
        x_max = min(SPACE_SIZE - 1, x + sap_range)
        y_min = max(0, y - sap_range)
        y_max = min(SPACE_SIZE - 1, y + sap_range)
        
        # Add all positions within sap range
        for nx in range(x_min, x_max + 1):
            for ny in range(y_min, y_max + 1):
                if abs(nx - x) <= sap_range and abs(ny - y) <= sap_range:
                    danger_set.add((nx, ny))

class ValueEstimator:
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        """
        Initialize the value estimator for Temporal Difference learning.
        
        Args:
            learning_rate: Rate at which new information updates existing values
            discount_factor: Weight given to future rewards vs immediate rewards
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Value maps for different game aspects
        self.value_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        self.energy_value_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        self.reward_value_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        
        # Visit tracking for exploration bonus
        self.visit_counts = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int32)
        
        # Reward history and stats
        self.reward_history = {}  # (x, y) -> list of rewards
        self.position_stats = {}  # (x, y) -> {'visits': count, 'total_reward': sum, 'avg_reward': mean}
        
    def update(self, old_state, new_state, reward):
        """
        Update value estimates using TD learning.
        
        Args:
            old_state: Previous position (x,y)
            new_state: Current position (x,y)
            reward: Reward received in the transition
        """
        if old_state is None:
            return
            
        old_x, old_y = old_state
        new_x, new_y = new_state
        
        # Update visit counts
        self.visit_counts[old_y, old_x] += 1
        
        # Track rewards
        if old_state not in self.reward_history:
            self.reward_history[old_state] = []
        self.reward_history[old_state].append(reward)
        
        # Update position statistics
        if old_state not in self.position_stats:
            self.position_stats[old_state] = {'visits': 0, 'total_reward': 0.0, 'avg_reward': 0.0}
        stats = self.position_stats[old_state]
        stats['visits'] += 1
        stats['total_reward'] += reward
        stats['avg_reward'] = stats['total_reward'] / stats['visits']
        
        # TD update for main value map
        old_value = self.value_map[old_y, old_x]
        new_value = self.value_map[new_y, new_x]
        
        # V(s) = V(s) + α * (r + γ * V(s') - V(s))
        self.value_map[old_y, old_x] = old_value + self.learning_rate * (
            reward + self.discount_factor * new_value - old_value
        )
        
        # Separate update for reward value map if this was a reward
        if reward > 0:
            self.reward_value_map[old_y, old_x] += self.learning_rate * reward
        
        # Update energy value map based on energy component of reward
        # Assuming energy reward is part of the total reward
        energy_component = reward * 0.3  # Estimate energy portion
        self.energy_value_map[old_y, old_x] += self.learning_rate * energy_component
        
    def estimate_node_value(self, node):
        """
        Estimate the combined value of a node for decision making.
        
        Args:
            node: The node to evaluate
            
        Returns:
            Estimated value combining learned value, exploration bonus, and other factors
        """
        if node is None or not hasattr(node, 'coordinates'):
            return 0.0
            
        x, y = node.coordinates
        
        # Get different value components
        learned_value = self.value_map[y, x]
        reward_value = self.reward_value_map[y, x]
        energy_value = self.energy_value_map[y, x]
        
        # Calculate exploration bonus - higher for less visited nodes
        visit_count = self.visit_counts[y, x]
        exploration_bonus = 3.0 if visit_count == 0 else 1.0 / np.sqrt(visit_count)
        
        # Energy consideration from node itself
        node_energy_value = 0
        if node.energy is not None:
            node_energy_value = node.energy / 20.0  # Normalize to 0-1 range
        
        # Historical reward value from position stats
        historical_reward = 0
        if node.coordinates in self.position_stats:
            historical_reward = self.position_stats[node.coordinates]['avg_reward']
        
        # Combine all factors with appropriate weights
        combined_value = (
            learned_value * 2.0 +           # Learned value (TD estimate)
            exploration_bonus * 0.5 +       # Exploration incentive
            node_energy_value * 0.8 +       # Current energy value
            energy_value * 0.5 +            # Historical energy value
            reward_value * 2.0 +            # Historical reward value
            historical_reward * 3.0         # Direct reward history
        )
        
        return combined_value
        
    def reset_for_new_match(self):
        """Reset exploration tracking for a new match while preserving learned values."""
        # Keep value maps but reset visit counts
        self.visit_counts = np.zeros_like(self.visit_counts)
        
        # We also reset position stats to encourage re-exploration
        self.position_stats = {}


%%writefile agent/agent.py

from base import *
from pathfinding import *

class Node:
    """
    Represents a single location on the game map, tracking its properties and state.
    
    Each node maintains information about its type (empty, nebula, asteroid),
    energy content, visibility, and whether it contains relics or rewards.
    """
    __slots__ = ('x', 'y', 'type', 'energy', 'is_visible', 
                '_relic', '_reward', '_explored_for_relic', '_explored_for_reward',
                'visits', 'total_reward', 'value')
    
    def __init__(self, x, y):
        # Position
        self.x = x
        self.y = y
        
        # Physical properties
        self.type = NodeType.unknown
        self.energy = None
        self.is_visible = False

        # Special features
        self._relic = False
        self._reward = False
        self._explored_for_relic = False
        self._explored_for_reward = True  # Default to true for efficiency
        
        # Value tracking for decision making
        self.visits = 0
        self.total_reward = 0
        self.value = 0

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return hash(self.coordinates)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.x == other.x and self.y == other.y

    @property
    def relic(self):
        """Whether this node contains a relic."""
        return self._relic

    @property
    def reward(self):
        """Whether this node provides reward points."""
        return self._reward

    @property
    def explored_for_relic(self):
        """Whether we've determined if this node has a relic."""
        return self._explored_for_relic

    @property
    def explored_for_reward(self):
        """Whether we've determined if this node gives rewards."""
        return self._explored_for_reward

    def update_relic_status(self, status):
        """
        Update the relic status of this node.
        
        Args:
            status: True if node has relic, False if not, None to mark as unexplored
        """
        # Prevent changing confirmed relics to non-relics
        if self._explored_for_relic and self._relic and status is False:
            return
            
        if status is None:
            self._explored_for_relic = False
            return

        self._relic = status
        self._explored_for_relic = True

    def update_reward_status(self, status):
        """
        Update the reward status of this node.
        
        Args:
            status: True if node gives rewards, False if not, None to mark as unexplored
        """
        # Prevent changing confirmed rewards to non-rewards
        if self._explored_for_reward and self._reward and status is False:
            return
            
        if status is None:
            self._explored_for_reward = False
            return

        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self):
        """Whether the node type is unknown."""
        return self.type == NodeType.unknown

    @property
    def is_walkable(self):
        """Whether units can move onto this node."""
        return self.type != NodeType.asteroid

    @property
    def coordinates(self):
        """The (x, y) coordinates of this node."""
        return (self.x, self.y)

    def manhattan_distance(self, other):
        """Calculate Manhattan distance to another node."""
        if not isinstance(other, Node):
            raise TypeError("Expected Node object")
        return abs(self.x - other.x) + abs(self.y - other.y)
        
    def update_value(self, reward):
        """
        Update node value using incremental average.
        
        Args:
            reward: The reward received at this node
        """
        self.visits += 1
        self.total_reward += reward
        self.value = self.total_reward / self.visits


class Space:
    """
    Represents the game map, maintaining a grid of Nodes and tracking special features.
    
    Handles updating the map based on observations, tracking relics and rewards,
    and predicting obstacle movements.
    """
    def __init__(self):
        # Initialize the grid of nodes
        self._nodes = np.empty((SPACE_SIZE, SPACE_SIZE), dtype=object)
        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                self._nodes[y, x] = Node(x, y)

        # Special node collections
        self._relic_nodes = set()  # Nodes with relics
        self._reward_nodes = set()  # Nodes that provide points
        
        # Value map for positions
        self.value_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=float)
        
        # Caches for performance
        self._node_cache = {}  # (x, y) -> Node
        self._walkable_cache = {}  # (x, y) -> bool
        self._opposite_cache = {}  # (x, y) -> (opposite_x, opposite_y)
        
        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0

    def __repr__(self):
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        """Iterator over all nodes in the space."""
        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                yield self._nodes[y, x]

    @property
    def relic_nodes(self):
        """Set of nodes that contain relics."""
        return self._relic_nodes

    @property
    def reward_nodes(self):
        """Set of nodes that provide reward points."""
        return self._reward_nodes

    def get_node(self, x, y):
        """
        Get the node at coordinates (x, y).
        
        Uses caching for performance when the same node is requested multiple times.
        """
        # Check cache first
        coords = (x, y)
        if coords in self._node_cache:
            self._cache_hits += 1
            return self._node_cache[coords]
            
        # Cache miss - retrieve from grid
        self._cache_misses += 1
        node = self._nodes[y, x]
        
        # Update cache (limited size to prevent memory issues)
        if len(self._node_cache) < 1000:  # Limit cache size
            self._node_cache[coords] = node
            
        return node

    def update(self, step, obs, team_id, team_reward):
        """
        Update the space based on new observations.
        
        Args:
            step: Current game step
            obs: Current observation from the environment
            team_id: ID of our team
            team_reward: Reward received by our team this step
        """
        # Clear caches to prevent staleness
        self._clear_caches()
        
        # Update in the correct order
        self.move_obstacles(step)
        self._update_map(obs)
        self._update_relic_map(step, obs, team_id, team_reward)
        self._update_value_map()

    def _clear_caches(self):
        """Clear all caches to prevent using stale data."""
        self._node_cache = {}
        self._walkable_cache = {}
        # Keep opposite_cache as it doesn't change

    def _update_relic_map(self, step, obs, team_id, team_reward):
        """
        Update knowledge about relics and rewards based on observations.
        
        Args:
            step: Current game step
            obs: Current observation from the environment
            team_id: ID of our team
            team_reward: Reward received by our team this step
        """
        # Process newly discovered relic nodes
        for mask, xy in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask and not self.get_node(*xy).relic:
                # Found a new relic
                self._update_relic_status(*xy, status=True)
                
                # Mark surrounding nodes as potentially containing rewards
                for x, y in self._get_relic_reward_area(*xy):
                    if not self.get_node(x, y).reward:
                        self._update_reward_status(x, y, status=None)

        # Check which nodes have been fully explored
        self._update_exploration_status(obs)
        
        # Update global flags based on exploration progress
        self._update_global_exploration_flags()
        
        # Try to infer additional information in later stages
        match_number = get_match_number(step)
        match_step = get_match_step(step)
        self._infer_additional_information(match_number, match_step)
        
        # Update reward knowledge based on latest observations
        if not Global.ALL_REWARDS_FOUND:
            self._update_reward_results(obs, team_id, team_reward)
            self._update_reward_status_from_reward_results()

    def _get_relic_reward_area(self, relic_x, relic_y):
        """
        Get coordinates of potential reward tiles around a relic.
        
        Args:
            relic_x, relic_y: Coordinates of the relic
            
        Returns:
            Generator of (x, y) coordinates within reward range of the relic
        """
        reward_range = Global.RELIC_REWARD_RANGE
        
        # Calculate bounds to avoid checking out-of-bounds coordinates
        x_min = max(0, relic_x - reward_range)
        x_max = min(SPACE_SIZE - 1, relic_x + reward_range)
        y_min = max(0, relic_y - reward_range)
        y_max = min(SPACE_SIZE - 1, relic_y + reward_range)
        
        # Generate coordinates within Manhattan distance
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if abs(x - relic_x) + abs(y - relic_y) <= reward_range:
                    yield x, y

    def _update_exploration_status(self, obs):
        """Update which nodes have been explored based on visibility."""
        for node in self:
            x, y = node.coordinates
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(x, y, status=False)

    def _update_global_exploration_flags(self):
        """Update global flags indicating exploration progress."""
        all_relics_found = all(node.explored_for_relic for node in self)
        all_rewards_found = all(node.explored_for_reward for node in self)
        
        Global.ALL_RELICS_FOUND = all_relics_found
        Global.ALL_REWARDS_FOUND = all_rewards_found

    def _infer_additional_information(self, match_number, match_step):
        """Infer additional information based on game progress."""
        # Try to infer if all relics have been found based on expected count
        num_expected_relics = 2 * min(match_number, Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1
        
        if not Global.ALL_RELICS_FOUND and len(self._relic_nodes) >= num_expected_relics:
            # We've found all expected relics - mark everything else as explored
            Global.ALL_RELICS_FOUND = True
            for node in self:
                if not node.explored_for_relic:
                    self._update_relic_status(node.x, node.y, status=False)
        
        # After the relic appearance cutoff, infer reward statuses
        if not Global.ALL_REWARDS_FOUND:
            if (match_step > Global.LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR or 
                len(self._relic_nodes) >= num_expected_relics):
                self._update_reward_status_from_relics_distribution()

    def _update_reward_status_from_reward_results(self):
        """Use reward history to infer which nodes yield rewards."""
        for result in Global.REWARD_RESULTS:
            # Sort nodes into categories
            unknown_nodes = set()
            known_reward_count = 0
            
            for node in result["nodes"]:
                if node.explored_for_reward and not node.reward:
                    # Already know this doesn't yield rewards
                    continue
                elif node.reward:
                    # Already know this yields rewards
                    known_reward_count += 1
                else:
                    # Status unknown
                    unknown_nodes.add(node)
            
            if not unknown_nodes:
                # All nodes in this result have known status
                continue
            
            # Determine how many rewards came from unknown nodes
            unknown_rewards = result["reward"] - known_reward_count
            
            # Update node statuses based on observed rewards
            if unknown_rewards == 0:
                # None of the unknown nodes yielded rewards
                for node in unknown_nodes:
                    self._update_reward_status(node.x, node.y, status=False)
            elif unknown_rewards == len(unknown_nodes):
                # All unknown nodes yielded rewards
                for node in unknown_nodes:
                    self._update_reward_status(node.x, node.y, status=True)

    def _update_reward_results(self, obs, team_id, team_reward):
        """Record which nodes ships were on when rewards were received."""
        if team_reward == 0:
            return  # No rewards to track
            
        # Find all active ships with non-negative energy
        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                ship_nodes.add(self.get_node(*position))
        
        # Record this reward event
        Global.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": team_reward})

    def _update_reward_status_from_relics_distribution(self):
        """
        Infer which nodes cannot provide rewards based on distance from relics.
        
        Rewards can only occur near relics, so if there's no relic within the
        reward range, that node cannot provide rewards.
        """
        # For each node, check if there's a relic nearby
        for node in self:
            # Skip nodes with known status
            if node.explored_for_reward:
                continue
                
            has_nearby_relic = False
            for relic_node in self._relic_nodes:
                if node.manhattan_distance(relic_node) <= Global.RELIC_REWARD_RANGE:
                    has_nearby_relic = True
                    break
            
            # If no relics nearby, this node cannot provide rewards
            if not has_nearby_relic:
                node.update_reward_status(False)

    def _update_relic_status(self, x, y, status):
        """
        Update the relic status of a node and its symmetric counterpart.
        
        Args:
            x, y: Coordinates of the node
            status: New relic status (True, False, or None)
        """
        node = self.get_node(x, y)
        node.update_relic_status(status)

        # Update symmetric node (relics are symmetric across the map)
        opp_x, opp_y = self._get_opposite(x, y)
        opp_node = self.get_node(opp_x, opp_y)
        opp_node.update_relic_status(status)

        # Add to relic nodes set if this is a relic
        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)

    def _get_opposite(self, x, y):
        """
        Get the coordinates of the point symmetric to (x, y) across the map.
        
        Args:
            x, y: Original coordinates
            
        Returns:
            (opposite_x, opposite_y) coordinates
        """
        # Check cache first
        coords = (x, y)
        if coords in self._opposite_cache:
            return self._opposite_cache[coords]
        
        # Calculate opposite coordinates
        opposite = (SPACE_SIZE - y - 1, SPACE_SIZE - x - 1)
        
        # Cache result
        self._opposite_cache[coords] = opposite
        
        return opposite

    def _update_reward_status(self, x, y, status):
        """
        Update the reward status of a node and its symmetric counterpart.
        
        Args:
            x, y: Coordinates of the node
            status: New reward status (True, False, or None)
        """
        node = self.get_node(x, y)
        node.update_reward_status(status)

        # Update symmetric node (rewards are symmetric across the map)
        opp_x, opp_y = self._get_opposite(x, y)
        opp_node = self.get_node(opp_x, opp_y)
        opp_node.update_reward_status(status)

        # Add to reward nodes set if this yields rewards
        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    def _update_map(self, obs):
        """
        Update the map based on new observations.
        
        Args:
            obs: Current observation from the environment
        """
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]

        # First pass: detect if obstacles or energy nodes have shifted
        obstacles_shifted, energy_nodes_shifted = self._detect_shifts(obs)
        Global.OBSTACLES_MOVEMENT_STATUS.append(obstacles_shifted)

        # Handle obstacle movement detection and prediction
        self._handle_obstacle_movement(obs, obstacles_shifted)

        # Update node information for all visible tiles
        self._update_visible_nodes(obs, sensor_mask, obs_energy, obs_tile_type, energy_nodes_shifted)

    def _detect_shifts(self, obs):
        """
        Detect if obstacles or energy have shifted since last observation.
        
        Args:
            obs: Current observation from the environment
            
        Returns:
            (obstacles_shifted, energy_nodes_shifted) boolean flags
        """
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]
        
        obstacles_shifted = False
        energy_nodes_shifted = False
        
        for node in self:
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]
            
            if not is_visible:
                continue

            # Check if obstacle type changed
            if not node.is_unknown and node.type.value != obs_tile_type[x, y]:
                obstacles_shifted = True
                
            # Check if energy changed
            if node.energy is not None and node.energy != obs_energy[x, y]:
                energy_nodes_shifted = True
                
            if obstacles_shifted and energy_nodes_shifted:
                break  # No need to check further
        
        return obstacles_shifted, energy_nodes_shifted

    def _handle_obstacle_movement(self, obs, obstacles_shifted):
        """
        Handle detection and prediction of obstacle movement.
        
        Args:
            obs: Current observation from the environment
            obstacles_shifted: Whether obstacles have shifted
        """
        # Function to reset map knowledge when needed
        def clear_map_info():
            for n in self:
                n.type = NodeType.unknown

        # Try to determine obstacle movement direction if not already known
        if not Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and obstacles_shifted:
            direction = self._find_obstacle_movement_direction(obs)
            if direction:
                Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                Global.OBSTACLE_MOVEMENT_DIRECTION = direction
                # Apply the detected movement to our model
                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)
            else:
                # Movement direction unclear, reset knowledge
                clear_map_info()

        # Try to determine obstacle movement period if not already known
        if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            period = self._find_obstacle_movement_period(Global.OBSTACLES_MOVEMENT_STATUS)
            if period is not None:
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                Global.OBSTACLE_MOVEMENT_PERIOD = period

            if obstacles_shifted:
                clear_map_info()

        # Handle case where something unexpected happened
        if (obstacles_shifted and 
            Global.OBSTACLE_MOVEMENT_PERIOD_FOUND and 
            Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND):
            # Our predictions were wrong, reset knowledge
            clear_map_info()

    def _update_visible_nodes(self, obs, sensor_mask, obs_energy, obs_tile_type, energy_nodes_shifted):
        """
        Update node information for all visible tiles.
        
        Args:
            obs: Current observation
            sensor_mask: Boolean mask of visible tiles
            obs_energy: Energy values from observation
            obs_tile_type: Tile types from observation
            energy_nodes_shifted: Whether energy nodes have shifted
        """
        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])
            
            # Update visibility
            node.is_visible = is_visible
            
            if not is_visible:
                # Handle non-visible nodes
                if energy_nodes_shifted:
                    # Energy might have changed, so mark as unknown
                    node.energy = None
                continue
            
            # Update tile type if unknown
            if node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))
                
                # Update symmetric tile
                opp_x, opp_y = self._get_opposite(x, y)
                self.get_node(opp_x, opp_y).type = node.type
            
            # Always update energy for visible tiles
            node.energy = int(obs_energy[x, y])
            
            # Update energy for symmetric tile
            opp_x, opp_y = self._get_opposite(x, y)
            self.get_node(opp_x, opp_y).energy = node.energy

    @staticmethod
    def _find_obstacle_movement_period(obstacles_movement_status):
        """
        Determine the period of obstacle movement from history.
        
        Args:
            obstacles_movement_status: List of boolean values indicating when obstacles moved
            
        Returns:
            Estimated period or None if not enough data
        """
        if len(obstacles_movement_status) < 81:
            return None  # Not enough data

        # Count how many times obstacles have moved
        num_movements = sum(obstacles_movement_status)

        # Determine period based on frequency of movements
        if num_movements <= 2:
            return 40
        elif num_movements <= 4:
            return 20
        elif num_movements <= 8:
            return 10
        else:
            return 20 / 3

    def _find_obstacle_movement_direction(self, obs):
        """
        Determine the direction of obstacle movement.
        
        Args:
            obs: Current observation
            
        Returns:
            Movement direction tuple or None if cannot determine
        """
        sensor_mask = obs["sensor_mask"]
        obs_tile_type = obs["map_features"]["tile_type"]

        # Try each possible direction
        suitable_directions = []
        for direction in [(1, -1), (-1, 1)]:  # Possible movement directions
            # Create a hypothetical space with obstacles moved in this direction
            moved_space = self.move(*direction, inplace=False)
            
            # Check if this movement matches the observed state
            match = True
            for node in moved_space:
                x, y = node.coordinates
                if (sensor_mask[x, y] and 
                    not node.is_unknown and 
                    obs_tile_type[x, y] != node.type.value):
                    match = False
                    break
            
            if match:
                suitable_directions.append(direction)
        
        # If exactly one direction matches, that's our answer
        if len(suitable_directions) == 1:
            return suitable_directions[0]
            
        return None  # Can't determine direction

    def _update_value_map(self):
        """Update the value map based on known rewards and energy."""
        # For optimization, use vectorized operations where possible
        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                node = self.get_node(x, y)
                
                # Calculate base value from energy
                energy_value = 0
                if node.energy is not None:
                    energy_value = node.energy / 20  # Normalize to 0-1 range
                
                # Add high value for reward nodes
                reward_value = 10 if node.reward else 0
                
                # Combine values
                self.value_map[y, x] = reward_value + energy_value

    def clear(self):
        """Reset visibility for all nodes."""
        for node in self:
            node.is_visible = False

    def move_obstacles(self, step):
        """
        Move obstacles according to predicted pattern.
        
        Args:
            step: Current game step
        """
        if (Global.OBSTACLE_MOVEMENT_PERIOD_FOUND and
            Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and
            Global.OBSTACLE_MOVEMENT_PERIOD > 0):
            
            # Calculate if obstacles should move at this step
            speed = 1 / Global.OBSTACLE_MOVEMENT_PERIOD
            if (step - 2) * speed % 1 > (step - 1) * speed % 1:
                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

    def move(self, dx, dy, *, inplace=False):
        """
        Move all obstacles in the space by the given offset.
        
        Args:
            dx, dy: Movement offset
            inplace: Whether to modify this space or create a new one
            
        Returns:
            The moved space (self if inplace, new Space otherwise)
        """
        if not inplace:
            # Create a new space with shifted obstacles
            new_space = Space()
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            # Modify this space in-place
            # Store all types first to avoid overwriting during movement
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self

    def clear_exploration_info(self):
        """Reset exploration information for a new match."""
        Global.REWARD_RESULTS = []
        Global.ALL_RELICS_FOUND = False
        Global.ALL_REWARDS_FOUND = False
        
        # Reset exploration status for all non-relic nodes
        for node in self:
            if not node.relic:
                self._update_relic_status(node.x, node.y, status=None)
    
    def _nearby_positions(self, x, y, distance):
        """
        Get positions within Manhattan distance of a point.
        
        Args:
            x, y: Center coordinates
            distance: Maximum Manhattan distance
            
        Returns:
            Generator of (x, y) coordinates within distance
        """
        # Calculate bounds to avoid checking out-of-bounds coordinates
        x_min = max(0, x - distance)
        x_max = min(SPACE_SIZE - 1, x + distance)
        y_min = max(0, y - distance)
        y_max = min(SPACE_SIZE - 1, y + distance)
        
        # Generate coordinates within Manhattan distance
        for nx in range(x_min, x_max + 1):
            for ny in range(y_min, y_max + 1):
                if abs(nx - x) + abs(ny - y) <= distance:
                    yield nx, ny


class Ship:
    """
    Represents a ship in the game, tracking its state and assigned tasks.
    
    Ships have positions, energy levels, and can be assigned roles and targets.
    They maintain history to support pattern detection.
    """
    __slots__ = ('unit_id', 'energy', 'node', 'previous_node', 
                'task', 'role', 'target', 'action', 'position_history')
    
    def __init__(self, unit_id):
        self.unit_id = unit_id
        self.energy = 0
        self.node = None
        self.previous_node = None
        
        # Task-related attributes
        self.task = None
        self.role = ShipRole.EXPLORER  # Default role
        self.target = None
        self.action = None
        
        # History for pattern detection
        self.position_history = []

    def __repr__(self):
        coords = self.coordinates if self.node else None
        return f"Ship({self.unit_id}, pos={coords}, energy={self.energy}, role={self.role})"

    @property
    def coordinates(self):
        """The ship's current coordinates, or None if not on a node."""
        return self.node.coordinates if self.node else None

    def update(self, node, energy):
        """
        Update the ship with a new position and energy level.
        
        Args:
            node: New node position
            energy: New energy level
        """
        # Store previous position
        self.previous_node = self.node
        
        # Update current state
        self.node = node
        self.energy = energy
        
        # Update position history
        if node and self.coordinates:
            self.position_history.append(self.coordinates)
            # Limit history length to avoid memory issues
            if len(self.position_history) > 10:
                self.position_history.pop(0)

    def clean(self):
        """Reset the ship's state completely."""
        self.energy = 0
        self.node = None
        self.previous_node = None
        self.task = None
        self.target = None
        self.action = None
        self.position_history = []


class Fleet:
    """
    Manages a collection of ships belonging to a team.
    
    Tracks team points and provides iteration over active ships.
    """
    def __init__(self, team_id):
        self.team_id = team_id
        self.points = 0  # Points scored in current match
        
        # Pre-allocate all possible ships
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]
        
        # Track active ships for faster iteration
        self.active_ships = set()

    def __repr__(self):
        return f"Fleet({self.team_id}, active_ships={len(self.active_ships)})"

    def __iter__(self):
        """Iterator over active ships only."""
        for ship_id in self.active_ships:
            yield self.ships[ship_id]

    def clear(self):
        """Reset the fleet for a new match."""
        self.points = 0
        self.active_ships.clear()
        for ship in self.ships:
            ship.clean()

    def update(self, obs, space):
        """
        Update all ships based on new observations.
        
        Args:
            obs: Current observation from the environment
            space: Game space object containing node information
        """
        # Update team points
        self.points = int(obs["team_points"][self.team_id])
        
        # Track which ships are still active
        still_active = set()
        
        # Update each ship
        for ship_id, (active, position, energy) in enumerate(zip(
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        )):
            ship = self.ships[ship_id]
            
            if active:
                # Update ship with new position and energy
                ship.update(space.get_node(*position), int(energy))
                
                # Reset action for new turn
                ship.action = None
                
                # Mark as still active
                still_active.add(ship_id)
            elif ship_id in self.active_ships:
                # Ship was active but is no longer - clean it
                ship.clean()
        
        # Update active ship set
        self.active_ships = still_active


class RoleCoordinator:
    """
    Manages role assignment for ships based on game state and ship capabilities.
    Dynamically balances the fleet composition between explorers, harvesters, 
    defenders, and attackers depending on the phase of the game.
    """
    
    # Predefined role distributions for different game phases
    EARLY_GAME_DISTRIBUTION = {
        ShipRole.EXPLORER: 0.6,   # Heavy focus on exploration
        ShipRole.HARVESTER: 0.3,  # Some resource gathering
        ShipRole.DEFENDER: 0.0,   # No need for defense early
        ShipRole.ATTACKER: 0.1    # Minimal aggression
    }
    
    MID_GAME_DISTRIBUTION = {
        ShipRole.EXPLORER: 0.3,   # Less exploration needed
        ShipRole.HARVESTER: 0.5,  # Higher focus on harvesting
        ShipRole.DEFENDER: 0.1,   # Some defense of resources
        ShipRole.ATTACKER: 0.1    # Some aggression
    }
    
    LATE_GAME_DISTRIBUTION = {
        ShipRole.EXPLORER: 0.1,   # Minimal exploration
        ShipRole.HARVESTER: 0.6,  # Maximum harvesting
        ShipRole.DEFENDER: 0.2,   # Increased defense
        ShipRole.ATTACKER: 0.1    # Some aggression
    }
    
    def __init__(self):
        self.ship_roles = {}  # unit_id -> role
        self.role_counts = {role: 0 for role in ShipRole.__dict__.values() 
                            if isinstance(role, str) and not role.startswith('_')}
        
        # Cache for suitability calculations (unit_id -> {role -> score})
        self.suitability_cache = {}
        self.cache_expiry = {}  # unit_id -> turn when cache expires
        self.CACHE_LIFETIME = 5  # Number of turns to keep suitability scores
        
    def assign_roles(self, fleet, space, match_step, match_number):
        """
        Assign roles to ships based on the current game state.
        
        Args:
            fleet: The fleet of ships to assign roles to
            space: The game space with map information
            match_step: Current step within the match
            match_number: Current match number (1-5)
        """
        # Clear expired cache entries
        self._clear_expired_cache(match_step)
        
        # Reset role counts
        for role in self.role_counts:
            self.role_counts[role] = 0
            
        # Determine target role distribution based on game phase
        target_distribution = self._get_phase_distribution(match_number, match_step)
            
        # Count active ships
        active_ships = list(fleet)
        num_active_ships = len(active_ships)
        
        if num_active_ships == 0:
            return
            
        # Calculate target counts
        target_counts = self._calculate_target_counts(target_distribution, num_active_ships)
        
        # Evaluate ship suitability for roles
        suitability = self._evaluate_all_ships_suitability(active_ships, space, match_step)
        
        # Assign roles to ships
        self._assign_roles_to_ships(active_ships, suitability, target_counts)
        
    def _get_phase_distribution(self, match_number, match_step):
        """Determine which distribution to use based on game phase."""
        if match_number <= 1 and match_step < 50:
            return self.EARLY_GAME_DISTRIBUTION
        elif match_number <= 3:
            return self.MID_GAME_DISTRIBUTION
        else:
            return self.LATE_GAME_DISTRIBUTION
    
    def _calculate_target_counts(self, distribution, num_ships):
        """Calculate how many ships should be assigned to each role."""
        target_counts = {}
        
        # Initial calculation based on percentages
        for role, percentage in distribution.items():
            target_counts[role] = max(1, int(percentage * num_ships))
        
        # Ensure we don't exceed the ship count by reducing from largest roles first
        self._balance_role_counts(target_counts, num_ships)
            
        return target_counts
    
    def _balance_role_counts(self, target_counts, num_ships):
        """Adjust role counts to match available ships."""
        while sum(target_counts.values()) > num_ships:
            # Find role with highest count
            role_to_reduce = max(target_counts.items(), key=lambda x: x[1])[0]
            target_counts[role_to_reduce] -= 1
    
    def _evaluate_all_ships_suitability(self, ships, space, current_step):
        """Evaluate suitability of all ships for all roles."""
        suitability = {}
        
        for ship in ships:
            # Use cached values if available and not expired
            if (ship.unit_id in self.suitability_cache and 
                self.cache_expiry.get(ship.unit_id, 0) >= current_step):
                suitability[ship.unit_id] = self.suitability_cache[ship.unit_id]
            else:
                # Calculate new suitability scores
                suitability[ship.unit_id] = self._evaluate_ship_suitability(ship, space)
                
                # Cache the results
                self.suitability_cache[ship.unit_id] = suitability[ship.unit_id]
                self.cache_expiry[ship.unit_id] = current_step + self.CACHE_LIFETIME
                
        return suitability
    
    def _clear_expired_cache(self, current_step):
        """Remove expired entries from the suitability cache."""
        expired_ids = [unit_id for unit_id, expiry in self.cache_expiry.items() 
                      if expiry < current_step]
        
        for unit_id in expired_ids:
            del self.suitability_cache[unit_id]
            del self.cache_expiry[unit_id]
    
    def _assign_roles_to_ships(self, ships, suitability, target_counts):
        """Assign roles to maximize overall suitability."""
        assigned_ships = set()
        
        # First, maintain existing roles if they're still needed
        self._maintain_existing_roles(ships, assigned_ships, target_counts)
                
        # Then, assign remaining ships
        self._assign_remaining_ships(ships, suitability, assigned_ships, target_counts)
    
    def _maintain_existing_roles(self, ships, assigned_ships, target_counts):
        """Keep ships in their current roles when possible."""
        for ship in ships:
            current_role = self.ship_roles.get(ship.unit_id)
            if current_role and self.role_counts[current_role] < target_counts[current_role]:
                self.ship_roles[ship.unit_id] = current_role
                self.role_counts[current_role] += 1
                assigned_ships.add(ship.unit_id)
                ship.role = current_role
    
    def _assign_remaining_ships(self, ships, suitability, assigned_ships, target_counts):
        """Assign roles to unassigned ships based on suitability."""
        # Process roles in priority order
        for role in [ShipRole.EXPLORER, ShipRole.HARVESTER, ShipRole.DEFENDER, ShipRole.ATTACKER]:
            # Skip if we've already assigned enough ships to this role
            if self.role_counts[role] >= target_counts[role]:
                continue
                
            # Find best ships for this role until target count is reached
            while self.role_counts[role] < target_counts[role]:
                best_ship = self._find_best_ship_for_role(
                    ships, suitability, assigned_ships, role)
                
                if best_ship:
                    self.ship_roles[best_ship.unit_id] = role
                    self.role_counts[role] += 1
                    assigned_ships.add(best_ship.unit_id)
                    best_ship.role = role
                else:
                    break  # No more suitable ships for this role
    
    def _find_best_ship_for_role(self, ships, suitability, assigned_ships, role):
        """Find the most suitable unassigned ship for a given role."""
        best_ship = None
        best_score = -1
        
        for ship in ships:
            if ship.unit_id in assigned_ships:
                continue
                
            score = suitability[ship.unit_id][role]
            if score > best_score:
                best_ship = ship
                best_score = score
                
        return best_ship
        
    def _evaluate_ship_suitability(self, ship, space):
        """Evaluate how suitable a ship is for each role."""
        if not ship.coordinates:
            # Can't evaluate ships without coordinates
            return {role: 0.0 for role in self.role_counts}
            
        suitability = {}
        
        # Pre-calculate common values
        energy_factor = ship.energy / 400  # Normalize energy to 0-1 range
        
        # --- Explorer suitability ---
        unexplored_distance = self._distance_to_nearest_unexplored(ship, space)
        distance_factor = 1 - min(1, unexplored_distance / 20)
        explorer_score = energy_factor * 0.5 + distance_factor * 0.5
        suitability[ShipRole.EXPLORER] = explorer_score
        
        # --- Harvester suitability ---
        reward_distance = self._distance_to_nearest_reward(ship, space)
        reward_distance_factor = 1 - min(1, reward_distance / 10)
        harvester_score = energy_factor * 0.3 + reward_distance_factor * 0.7
        suitability[ShipRole.HARVESTER] = harvester_score
        
        # --- Defender suitability ---
        # Defenders need high energy for sapping and should be near reward nodes
        defender_score = energy_factor * 0.8 + (1 - min(1, reward_distance / 15)) * 0.2
        suitability[ShipRole.DEFENDER] = defender_score
        
        # --- Attacker suitability ---
        # Attackers need high energy for sapping
        attacker_score = energy_factor * 0.8 + 0.2
        suitability[ShipRole.ATTACKER] = attacker_score
        
        return suitability
        
    def _distance_to_nearest_unexplored(self, ship, space):
        """Calculate distance to nearest unexplored node using early stopping."""
        if not ship.coordinates:
            return 20  # Default distance if no coordinates
            
        min_distance = float('inf')
        
        # Start checking nearby areas first for efficiency
        for search_radius in range(1, 21):  # Limit search radius
            found_unexplored = False
            
            for node in space:
                if not node.explored_for_relic or not node.explored_for_reward:
                    distance = manhattan_distance(ship.coordinates, node.coordinates)
                    
                    # Early stopping when we find something within search radius
                    if distance <= search_radius:
                        return distance
                        
                    min_distance = min(min_distance, distance)
                    found_unexplored = True
            
            # If we've found some unexplored nodes but none within radius, we can return
            if found_unexplored:
                break
                
        return min_distance if min_distance != float('inf') else 20
        
    def _distance_to_nearest_reward(self, ship, space):
        """Calculate distance to nearest reward node using early stopping."""
        if not ship.coordinates or not space.reward_nodes:
            return 20  # Default distance if no coordinates or rewards
            
        # Optimization: check closest first with sorted approach
        # But only sort if we have many reward nodes
        if len(space.reward_nodes) > 10:
            sorted_nodes = sorted(
                space.reward_nodes, 
                key=lambda n: manhattan_distance(ship.coordinates, n.coordinates)
            )
            if sorted_nodes:
                return manhattan_distance(ship.coordinates, sorted_nodes[0].coordinates)
        
        # For fewer nodes, just do a linear scan
        min_distance = float('inf')
        for node in space.reward_nodes:
            distance = manhattan_distance(ship.coordinates, node.coordinates)
            min_distance = min(min_distance, distance)
            
        return min_distance if min_distance != float('inf') else 20
        
    def get_role(self, unit_id):
        """Get the assigned role for a ship"""
        return self.ship_roles.get(unit_id, ShipRole.EXPLORER)  # Default to explorer


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        # Initialize game constants from environment config
        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        # Initialize components
        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)
        
        # Enhanced components
        self.role_coordinator = RoleCoordinator()
        self.opponent_tracker = OpponentTracker()
        self.value_estimator = ValueEstimator()
        
        # State tracking
        self.previous_match_number = -1
        self.previous_positions = {}  # unit_id -> previous position

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Generate actions for the current turn based on observations.
        
        Args:
            step: Current game step
            obs: Current observation from the environment
            remainingOverageTime: Remaining overage time
            
        Returns:
            Numpy array of actions in the format expected by the environment
        """
        match_step = get_match_step(step)
        match_number = get_match_number(step)
        
        # Handle new match initialization
        if match_number != self.previous_match_number:
            self.previous_match_number = match_number
            self.value_estimator.reset_for_new_match()
            self.previous_positions = {}
            
        if match_step == 0:
            # Nothing to do at the beginning of the match
            # Clean up from previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.move_obstacles(step)
            if match_number <= Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR:
                self.space.clear_exploration_info()
                
            # Return default actions for all units
            return np.zeros((Global.MAX_UNITS, 3), dtype=np.int32)
        
        # Calculate rewards from previous step
        points = int(obs["team_points"][self.team_id])
        reward = max(0, points - self.fleet.points)
        
        # Update all models with new observations
        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)
        self.opponent_tracker.update(obs, self.opp_team_id)
        
        # Assign roles to ships
        self.role_coordinator.assign_roles(self.fleet, self.space, match_step, match_number)
        
        # Update value estimates based on ship movements
        for ship in self.fleet:
            if ship.unit_id in self.previous_positions:
                old_position = self.previous_positions[ship.unit_id]
                new_position = ship.coordinates
                
                # Update value estimator with reward if this ship was on a reward node
                node_reward = 1.0 if ship.node.reward else 0.0
                energy_reward = ship.node.energy if ship.node.energy is not None else 0.0
                total_reward = node_reward + energy_reward / 20.0
                
                self.value_estimator.update(old_position, new_position, total_reward)
            
            # Store current position for next update
            if ship.coordinates:
                self.previous_positions[ship.unit_id] = ship.coordinates
        
        # Process ships based on their roles
        for ship in self.fleet:
            role = ship.role
            
            if role == ShipRole.EXPLORER:
                self.process_explorer(ship, obs)
            elif role == ShipRole.HARVESTER:
                self.process_harvester(ship, obs)
            elif role == ShipRole.DEFENDER:
                self.process_defender(ship, obs)
            elif role == ShipRole.ATTACKER:
                self.process_attacker(ship, obs)
        
        return self.create_actions_array()
    
    def create_actions_array(self):
        """
        Create the actions array expected by the environment.
        
        Returns:
            Numpy array of shape (MAX_UNITS, 3) containing actions for all units
        """
        # Create a zeros array for all possible units
        actions = np.zeros((Global.MAX_UNITS, 3), dtype=np.int32)
        
        # Fill in actions for active ships
        for ship in self.fleet:
            unit_id = ship.unit_id
            
            if unit_id >= Global.MAX_UNITS:
                continue  # Skip if unit ID is out of bounds
                
            if ship.action is not None:
                # For move actions, the other parameters are 0
                if ship.action != ActionType.sap:
                    actions[unit_id, 0] = int(ship.action)
                else:
                    # For sap actions, we need to specify the target
                    if ship.target is not None and ship.coordinates:
                        tx, ty = ship.target.coordinates
                        sx, sy = ship.coordinates
                        dx, dy = tx - sx, ty - sy
                        
                        # Ensure within sap range
                        if abs(dx) <= Global.UNIT_SAP_RANGE and abs(dy) <= Global.UNIT_SAP_RANGE:
                            actions[unit_id, 0] = int(ship.action)
                            actions[unit_id, 1] = int(dx)
                            actions[unit_id, 2] = int(dy)
                        else:
                            # Fallback to center if target out of range
                            actions[unit_id, 0] = int(ActionType.center)
                    else:
                        # Fallback to center if no target
                        actions[unit_id, 0] = int(ActionType.center)
            else:
                # Default to center action if none specified
                actions[unit_id, 0] = int(ActionType.center)
        
        # Validate the array
        assert isinstance(actions, np.ndarray), "Actions must be a numpy array"
        assert actions.dtype == np.int32, "Actions must have dtype int32"
        assert actions.shape[1] == 3, "Actions must have 3 columns"
        
        return actions
    
    def process_explorer(self, ship, obs):
        """
        Handle exploration tasks for a ship.
        """
        # If everything is explored, switch to harvesting
        if Global.ALL_RELICS_FOUND and Global.ALL_REWARDS_FOUND:
            self.process_harvester(ship, obs)
            return
        
        # Identify exploration targets
        targets = set()
        for node in self.space:
            # Only consider unexplored nodes that are walkable and in our sector
            if ((not node.explored_for_relic or not node.explored_for_reward) 
                and node.is_walkable 
                and is_team_sector(self.team_id, *node.coordinates)):
                targets.add(node.coordinates)
        
        if not targets:
            # No exploration targets, switch to harvesting
            self.process_harvester(ship, obs)
            return
        
        # Find the best target considering distance, value, and safety
        best_target = None
        best_score = float('-inf')
        
        for target in targets:
            target_node = self.space.get_node(*target)
            
            # Distance factor (inverse - closer is better)
            distance = manhattan_distance(ship.coordinates, target_node.coordinates)
            distance_factor = max(1, 20 - distance) / 20  # Scale to 0-1
            
            # Value factor from value estimator
            value_factor = self.value_estimator.estimate_node_value(target_node) / 10  # Scale to 0-1
            
            # Safety factor - avoid enemy danger zones
            danger_areas = self.opponent_tracker.get_dangerous_areas(Global.UNIT_SAP_RANGE)
            safety_factor = 0.5  # Default medium safety
            if target in danger_areas:
                safety_factor = 0.1  # Very dangerous
            elif any(manhattan_distance(target, d) <= 2 for d in danger_areas):
                safety_factor = 0.3  # Somewhat dangerous
            else:
                safety_factor = 1.0  # Safe
            
            # Combine factors with appropriate weights
            score = (
                distance_factor * 2.0 +   # Distance is important
                value_factor * 1.0 +      # Value has some importance
                safety_factor * 1.5       # Safety is quite important
            )
            
            if score > best_score:
                best_score = score
                best_target = target
        
        if not best_target:
            # No valid target found, switch to harvesting
            self.process_harvester(ship, obs)
            return
        
        # Create weight map for pathfinding
        weights = create_weights(
            self.space,
            avoid_enemies=True,
            enemy_positions=[s.coordinates for s in self.opp_fleet]
        )
        
        # Find path to target
        path = astar(weights, ship.coordinates, best_target)
        if not path:
            # No path found, try another target or just rest
            ship.task = "explore"
            ship.target = ship.node
            ship.action = ActionType.center
            return
        
        energy = estimate_energy_cost(self.space, path)
        actions = path_to_actions(path)
        
        if actions and ship.energy >= energy:
            ship.task = "explore"
            ship.target = self.space.get_node(*best_target)
            ship.action = actions[0]
        else:
            # Not enough energy, just rest and collect energy
            ship.task = "explore"
            ship.target = ship.node
            ship.action = ActionType.center
    
    def process_harvester(self, ship, obs):
        """
        Handle harvesting tasks for a ship.
        """
        # Identify reward targets
        reward_nodes = [n for n in self.space.reward_nodes if n.is_walkable]
        
        if not reward_nodes:
            # No known reward nodes, try to find the best energy spot
            energy_targets = []
            for node in self.space:
                if node.is_walkable and node.energy is not None and node.energy > 5:
                    energy_targets.append(node)
            
            if not energy_targets:
                # No good energy spots either, just rest
                ship.task = "rest"
                ship.target = ship.node
                ship.action = ActionType.center
                return
            
            # Select best energy node
            best_node = max(energy_targets, key=lambda n: self.value_estimator.estimate_node_value(n))
            
            # Find path to energy node
            weights = create_weights(
                self.space,
                avoid_enemies=True,
                enemy_positions=[s.coordinates for s in self.opp_fleet]
            )
            
            path = astar(weights, ship.coordinates, best_node.coordinates)
            if not path:
                # No path found, just rest
                ship.task = "rest"
                ship.target = ship.node
                ship.action = ActionType.center
                return
            
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            
            if actions and ship.energy >= energy:
                ship.task = "harvest_energy"
                ship.target = best_node
                ship.action = actions[0]
            else:
                # Not enough energy, just rest
                ship.task = "rest"
                ship.target = ship.node
                ship.action = ActionType.center
            
            return
        
        # Score the reward nodes
        scored_nodes = []
        for node in reward_nodes:
            # Base value from reward and energy
            base_value = 10  # Base value for reward nodes
            if node.energy is not None:
                base_value += node.energy / 4  # Energy adds some value
            
            # Distance factor
            distance = manhattan_distance(ship.coordinates, node.coordinates)
            distance_factor = max(0, 1 - distance/20)  # Further = lower factor
            
            # Danger factor
            dangerous_areas = self.opponent_tracker.get_dangerous_areas(Global.UNIT_SAP_RANGE)
            danger_factor = 1.0  # Default: safe
            if node.coordinates in dangerous_areas:
                danger_factor = 0.2  # Very dangerous
            elif any(manhattan_distance(node.coordinates, d) <= 2 for d in dangerous_areas):
                danger_factor = 0.6  # Somewhat dangerous
            
            # Competition factor - prefer areas with fewer friendly ships
            ships_nearby = 0
            for other_ship in self.fleet:
                if other_ship != ship and manhattan_distance(other_ship.coordinates, node.coordinates) <= 3:
                    ships_nearby += 1
            competition_factor = max(0.2, 1 - ships_nearby * 0.2)  # More ships = lower factor
            
            # Calculate final score
            score = base_value * distance_factor * danger_factor * competition_factor
            
            scored_nodes.append((node, score))
        
        # Select the best node
        if not scored_nodes:
            # No valid nodes, just rest
            ship.task = "rest"
            ship.target = ship.node
            ship.action = ActionType.center
            return
        
        best_node = max(scored_nodes, key=lambda item: item[1])[0]
        
        # Find path to target
        weights = create_weights(
            self.space,
            avoid_enemies=True,
            enemy_positions=[s.coordinates for s in self.opp_fleet]
        )
        
        path = astar(weights, ship.coordinates, best_node.coordinates)
        if not path:
            # No path found, just rest
            ship.task = "rest"
            ship.target = ship.node
            ship.action = ActionType.center
            return
        
        energy = estimate_energy_cost(self.space, path)
        actions = path_to_actions(path)
        
        if actions and ship.energy >= energy:
            ship.task = "harvest"
            ship.target = best_node
            ship.action = actions[0]
        else:
            # Not enough energy, just rest
            ship.task = "rest"
            ship.target = ship.node
            ship.action = ActionType.center
    
    def process_defender(self, ship, obs):
        """
        Handle defense tasks for a ship.
        """
        # Find valuable areas to defend
        valuable_areas = [n for n in self.space.reward_nodes if n.is_walkable]
        if not valuable_areas:
            # Nothing to defend, switch to harvesting
            self.process_harvester(ship, obs)
            return
        
        # Check if there are enemy ships threatening our valuable areas
        threats = self.find_threats_to_valuable_areas(valuable_areas)
        
        if threats:
            # Enemy ships threatening our valuable areas, try to sap them
            target_enemy = self.select_best_sap_target(ship, threats)
            ship.task = "defend"
            ship.target = self.space.get_node(*target_enemy)
            
            # Calculate sap direction
            sx, sy = ship.coordinates
            tx, ty = target_enemy
            dx, dy = tx - sx, ty - sy
            
            # Check if in range for sap
            if abs(dx) <= Global.UNIT_SAP_RANGE and abs(dy) <= Global.UNIT_SAP_RANGE and ship.energy >= Global.UNIT_SAP_COST:
                ship.action = ActionType.sap
            else:
                # Move toward the threat
                weights = create_weights(self.space)
                path = astar(weights, ship.coordinates, target_enemy)
                actions = path_to_actions(path)
                if actions:
                    ship.action = actions[0]
                else:
                    ship.action = ActionType.center
        else:
            # No immediate threats, position near valuable areas
            # Choose the most valuable area to defend
            target_area = max(valuable_areas, key=lambda n: self.value_estimator.estimate_node_value(n))
            ship.task = "position"
            ship.target = target_area
            
            # If already at the target, just stay there
            if ship.node == target_area:
                ship.action = ActionType.center
                return
            
            # Otherwise, move toward the target
            weights = create_weights(self.space)
            path = astar(weights, ship.coordinates, target_area.coordinates)
            actions = path_to_actions(path)
            if actions:
                ship.action = actions[0]
            else:
                ship.action = ActionType.center
    
    def process_attacker(self, ship, obs):
        """
        Handle attack tasks for a ship.
        """
        # Find enemy ships to target
        enemy_positions = []
        for active, position in zip(
            obs["units_mask"][self.opp_team_id],
            obs["units"]["position"][self.opp_team_id]
        ):
            if active:
                enemy_positions.append(tuple(position))
        
        if not enemy_positions:
            # No visible enemies, switch to harvesting
            self.process_harvester(ship, obs)
            return
        
        # Score enemy ships based on several factors
        scored_targets = []
        for pos in enemy_positions:
            enemy_node = self.space.get_node(*pos)
            
            # Higher score for enemies near our reward nodes
            near_reward_bonus = 0
            for reward_node in self.space.reward_nodes:
                if manhattan_distance(pos, reward_node.coordinates) <= 2:
                    near_reward_bonus += 5
            
            # Higher score for enemies with predicted low energy
            # (This would require tracking enemy energy, which we don't have yet)
            energy_score = 0
            
            # Higher score for enemies we can actually reach
            distance = manhattan_distance(ship.coordinates, pos)
            accessibility_score = max(0, 10 - distance)
            
            # Combine scores
            total_score = near_reward_bonus + energy_score + accessibility_score
            scored_targets.append((pos, total_score))
        
        # Select highest-scoring target
        if not scored_targets:
            self.process_harvester(ship, obs)
            return
        
        best_target = max(scored_targets, key=lambda x: x[1])[0]
        ship.task = "attack"
        ship.target = self.space.get_node(*best_target)
        
        # Check if in sap range
        sx, sy = ship.coordinates
        tx, ty = best_target
        dx, dy = tx - sx, ty - sy
        
        if abs(dx) <= Global.UNIT_SAP_RANGE and abs(dy) <= Global.UNIT_SAP_RANGE and ship.energy >= Global.UNIT_SAP_COST:
            # In range for sap
            ship.action = ActionType.sap
        else:
            # Move toward the target
            weights = create_weights(self.space)
            path = astar(weights, ship.coordinates, best_target)
            actions = path_to_actions(path)
            if actions:
                ship.action = actions[0]
            else:
                ship.action = ActionType.center
    
    def find_threats_to_valuable_areas(self, valuable_areas):
        """
        Identify enemy ships that threaten our valuable areas.
        """
        threats = []
        for enemy_unit_id in self.opponent_tracker.ship_history:
            history = self.opponent_tracker.ship_history[enemy_unit_id]
            if not history:
                continue
                
            current_pos = history[-1]
            
            # Check if this enemy is near any valuable area
            for area in valuable_areas:
                distance = manhattan_distance(current_pos, area.coordinates)
                if distance <= 3:  # Threat threshold
                    threats.append(current_pos)
                    break
                
            # Also consider predicted positions
            next_pos = self.opponent_tracker.predict_next_position(enemy_unit_id)
            if next_pos:
                for area in valuable_areas:
                    distance = manhattan_distance(next_pos, area.coordinates)
                    if distance <= 3:  # Threat threshold
                        threats.append(next_pos)
                        break
        
        return threats
    
    def select_best_sap_target(self, ship, enemy_positions):
        """
        Select the best enemy to sap.
        """
        # Score based on:
        # 1. Distance (closer is better)
        # 2. Number of enemies in the area (more is better - AOE advantage)
        
        scored_targets = []
        for pos in enemy_positions:
            # Base score inversely proportional to distance
            distance = manhattan_distance(ship.coordinates, pos)
            distance_score = max(0, 10 - distance)
            
            # Count nearby enemies for potential AOE damage
            nearby_enemies = 0
            for other_pos in enemy_positions:
                if manhattan_distance(pos, other_pos) <= 1:  # Adjacent or same position
                    nearby_enemies += 1
            
            aoe_score = nearby_enemies * 3  # Significant bonus for AOE potential
            
            total_score = distance_score + aoe_score
            scored_targets.append((pos, total_score))
        
        if not scored_targets:
            return enemy_positions[0]  # Fallback to first enemy
            
        return max(scored_targets, key=lambda x: x[1])[0]


%%writefile agent/debug.py

from base import *

def print_summary(agent, step, obs):
    """
    Print a summary of the current game state.
    
    Args:
        agent: The agent instance
        step: Current game step
        obs: Current observation
    """
    match_step = get_match_step(step)
    match_number = get_match_number(step)
    
    print(f"==== Step {step} (Match {match_number}, Step {match_step}) ====")
    print(f"Team: {agent.player}, Points: {agent.fleet.points}")
    
    # Print active ships
    print(f"\nActive Ships: {len(agent.fleet.active_ships)}")
    for ship in agent.fleet:
        print(f"  Ship {ship.unit_id}: pos={ship.coordinates}, energy={ship.energy}, role={ship.role}")
    
    # Print enemy ships
    print(f"\nEnemy Ships: {len(agent.opp_fleet.active_ships)}")
    for ship in agent.opp_fleet:
        print(f"  Ship {ship.unit_id}: pos={ship.coordinates}, energy={ship.energy}")
    
    # Print reward nodes
    print(f"\nReward Nodes: {len(agent.space.reward_nodes)}")
    for i, node in enumerate(agent.space.reward_nodes):
        if i >= 5:  # Limit output for readability
            print(f"  ...and {len(agent.space.reward_nodes) - 5} more")
            break
        print(f"  Node at {node.coordinates}, energy={node.energy}")
    
    # Print exploration status
    print(f"\nExploration Status:")
    print(f"  All Relics Found: {Global.ALL_RELICS_FOUND}")
    print(f"  All Rewards Found: {Global.ALL_REWARDS_FOUND}")
    
    # Print movement prediction
    print(f"\nObstacle Movement:")
    print(f"  Direction Found: {Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND}")
    print(f"  Direction: {Global.OBSTACLE_MOVEMENT_DIRECTION}")
    print(f"  Period Found: {Global.OBSTACLE_MOVEMENT_PERIOD_FOUND}")
    print(f"  Period: {Global.OBSTACLE_MOVEMENT_PERIOD}")
    
    print("\n")

def visualize_map(agent, obs, highlight_coords=None):
    """
    Visualize the current map state in ASCII.
    
    Args:
        agent: The agent instance
        obs: Current observation
        highlight_coords: Optional list of coordinates to highlight
    """
    if highlight_coords is None:
        highlight_coords = []
        
    highlight_set = set(highlight_coords)
    
    # Create a blank grid
    grid = [[' ' for _ in range(SPACE_SIZE)] for _ in range(SPACE_SIZE)]
    
    # Mark obstacles
    for node in agent.space:
        x, y = node.coordinates
        if node.type == NodeType.asteroid:
            grid[y][x] = '#'
        elif node.type == NodeType.nebula:
            grid[y][x] = '~'
            
    # Mark reward nodes
    for node in agent.space.reward_nodes:
        x, y = node.coordinates
        grid[y][x] = '$'
    
    # Mark friendly ships
    for ship in agent.fleet:
        if ship.coordinates:
            x, y = ship.coordinates
            grid[y][x] = 'F'
    
    # Mark enemy ships
    for ship in agent.opp_fleet:
        if ship.coordinates:
            x, y = ship.coordinates
            grid[y][x] = 'E'
    
    # Mark highlights
    for x, y in highlight_coords:
        grid[y][x] = 'X'
    
    # Print the grid
    print("  " + "".join(f"{i%10}" for i in range(SPACE_SIZE)))
    for y in range(SPACE_SIZE):
        print(f"{y%10} " + "".join(grid[y]))
        
def visualize_pathfinding(agent, start, goal, path=None):
    """
    Visualize pathfinding results.
    
    Args:
        agent: The agent instance
        start: Starting coordinates (x, y)
        goal: Goal coordinates (x, y)
        path: Optional path to display
    """
    # Create weight map
    weights = create_weights(agent.space)
    
    # Find path if not provided
    if path is None:
        from agent.pathfinding import astar
        path = astar(weights, start, goal)
    
    # Create a grid for visualization
    grid = [[' ' for _ in range(SPACE_SIZE)] for _ in range(SPACE_SIZE)]
    
    # Fill in the basics
    for y in range(SPACE_SIZE):
        for x in range(SPACE_SIZE):
            node = agent.space.get_node(x, y)
            if not node.is_walkable:
                grid[y][x] = '#'
            elif weights[y, x] > 1.5 * Global.MAX_ENERGY_PER_TILE:
                grid[y][x] = '~'  # High cost
    
    # Mark path
    if path:
        for x, y in path:
            grid[y][x] = '.'
            
    # Mark start and goal
    start_x, start_y = start
    goal_x, goal_y = goal
    grid[start_y][start_x] = 'S'
    grid[goal_y][goal_x] = 'G'
    
    # Print the grid
    print(f"Path from {start} to {goal}:")
    print("  " + "".join(f"{i%10}" for i in range(SPACE_SIZE)))
    for y in range(SPACE_SIZE):
        print(f"{y%10} " + "".join(grid[y]))
    
    # Print path details
    if path:
        from agent.pathfinding import estimate_energy_cost
        energy_cost = estimate_energy_cost(agent.space, path)
        print(f"Path length: {len(path)}, Estimated energy cost: {energy_cost}")
    else:
        print("No path found")


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


!luxai-s3 agent/main.py agent/main.py --output=replay.html


import IPython # load the HTML replay
IPython.display.HTML(filename='replay.html')


!cd agent && tar -czf submission.tar.gz *
!mv agent/submission.tar.gz .




