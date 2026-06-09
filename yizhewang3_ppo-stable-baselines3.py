! pip install stable-baselines3


!mkdir agent && cp -r ../input/lux-ai-season-3/* agent/
import sys
sys.path.insert(1, 'agent')


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


# def warp_int(x):
#     if x >= SPACE_SIZE:
#         x -= SPACE_SIZE
#     elif x < 0:
#         x += SPACE_SIZE
#     return x


# def warp_point(x, y) -> tuple:
#     return warp_int(x), warp_int(y)


def get_opposite(x, y) -> tuple:
    # Returns the mirrored point across the diagonal
    return SPACE_SIZE - y - 1, SPACE_SIZE - x - 1


def is_upper_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 >= y


def is_lower_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 <= y


def is_team_sector(team_id, x, y) -> bool:
    return is_upper_sector(x, y) if team_id == 0 else is_lower_sector(x, y)



%%writefile agent/ppo_game_env.py
import sys
import gym
from gym import spaces
import numpy as np

# 导入 base 中的全局常量和辅助函数
from base import Global, ActionType, SPACE_SIZE, get_opposite

# 定义常量：队伍数、最大单位数、最大遗迹节点数
NUM_TEAMS = 2
MAX_UNITS = Global.MAX_UNITS
MAX_RELIC_NODES = Global.MAX_RELIC_NODES

class PPOGameEnv(gym.Env):
    """
    PPOGameEnv 模拟环境尽可能还原真实比赛环境，并满足以下要求：
    
    1. 观察数据计算修改：
       - 每个己方单位均有自己独立的 sensor mask（由 compute_unit_vision(unit) 计算），
         并由 get_unit_obs(unit) 构造出符合固定格式的局部观察（字典形式）。
       - 返回给代理的全局观察则采用所有己方单位 sensor mask 的联合（逻辑“或”），
         保持比赛返回 obs 的固定格式。
    
    2. 奖励函数优化：
       根据动作更新环境状态，并返回 (observation, reward, done, info)。
        修改后的奖励逻辑：
          1. 每个 unit 单独计算 unit_reward。
          2. 若移动动作导致超出地图或目标 tile 为 Asteroid，则判定为无效，unit_reward -0.2。
          3. Sap 动作：
             - 检查 unit 局部 obs 中 relic_nodes_mask 是否存在 relic；
             - 如果存在，统计 unit 8 邻域内敌方单位数，若数目>=2，则 sap 奖励 = +1.0×敌方单位数，否则扣 -2.0；
             - 若无 relic 可见，则同样扣 -2.0。
          4. 非 sap 动作：
             - 成功移动后，检查该 unit 是否位于任一 relic 配置内的潜力点：
                  * 若首次访问该潜力点，unit_reward +2.0，并标记 visited；
                  * 如果该潜力点尚未兑现 team point，则增加 self.score 1，同时 unit_reward +3.0 并标记为 team_points_space；
                  * 如果已在 team_points_space 上，则每回合奖励 +3.0；
             - 若 unit 位于能量节点（energy == Global.MAX_ENERGY_PER_TILE），unit_reward +0.2；
             - 若 unit 位于 Nebula（tile_type==1），unit_reward -0.2；
             - 如果 unit 移动后与敌方 unit 重合，且对方能量低于己方，则对每个满足条件的敌方 unit 奖励 +1.0。
          5. 全局探索奖励：所有己方单位联合视野中新发现 tile，每个奖励 +0.1。
          6. 每一step结束，奖励 point*0.5的奖励 + 规则*0.5的奖励
    
    3. 敌方单位策略说明：
       - 敌方单位在出生后不主动行动，其位置仅由环境每 20 步整体滚动（右移 1 格）改变，
         属于被动对手。这样设计主要用于初期调试，后续可引入更主动的对抗策略。
    """
    
    def __init__(self):
        super(PPOGameEnv, self).__init__()
        
        # 修改动作空间：每个单位独立决策（动作取值范围为 0~5）
        self.action_space = spaces.MultiDiscrete([len(ActionType)] * MAX_UNITS)
        
        # 观察空间保持不变
        self.observation_space = spaces.Dict({
            "units_position": spaces.Box(
                low=0,
                high=SPACE_SIZE - 1,
                shape=(NUM_TEAMS, MAX_UNITS, 2),
                dtype=np.int32
            ),
            "units_energy": spaces.Box(
                low=0,
                high=400,  # 单位能量上限 400
                shape=(NUM_TEAMS, MAX_UNITS, 1),
                dtype=np.int32
            ),
            "units_mask": spaces.Box(
                low=0,
                high=1,
                shape=(NUM_TEAMS, MAX_UNITS),
                dtype=np.int8
            ),
            "sensor_mask": spaces.Box(
                low=0,
                high=1,
                shape=(SPACE_SIZE, SPACE_SIZE),
                dtype=np.int8
            ),
            "map_features_tile_type": spaces.Box(
                low=-1,
                high=2,
                shape=(SPACE_SIZE, SPACE_SIZE),
                dtype=np.int8
            ),
            "map_features_energy": spaces.Box(
                low=-1,
                high=Global.MAX_ENERGY_PER_TILE,
                shape=(SPACE_SIZE, SPACE_SIZE),
                dtype=np.int8
            ),
            "relic_nodes_mask": spaces.Box(
                low=0,
                high=1,
                shape=(MAX_RELIC_NODES,),
                dtype=np.int8
            ),
            "relic_nodes": spaces.Box(
                low=-1,
                high=SPACE_SIZE - 1,
                shape=(MAX_RELIC_NODES, 2),
                dtype=np.int32
            ),
            "team_points": spaces.Box(
                low=0,
                high=1000,
                shape=(NUM_TEAMS,),
                dtype=np.int32
            ),
            "team_wins": spaces.Box(
                low=0,
                high=1000,
                shape=(NUM_TEAMS,),
                dtype=np.int32
            ),
            "steps": spaces.Box(
                low=0, high=Global.MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
            ),
            "match_steps": spaces.Box(
                low=0, high=Global.MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
            ),
            "remainingOverageTime": spaces.Box(
                low=0, high=1000, shape=(1,), dtype=np.int32
            ),
            "env_cfg_map_width": spaces.Box(
                low=0, high=SPACE_SIZE, shape=(1,), dtype=np.int32
            ),
            "env_cfg_map_height": spaces.Box(
                low=0, high=SPACE_SIZE, shape=(1,), dtype=np.int32
            ),
            "env_cfg_max_steps_in_match": spaces.Box(
                low=0, high=Global.MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32
            ),
            "env_cfg_unit_move_cost": spaces.Box(
                low=0, high=100, shape=(1,), dtype=np.int32
            ),
            "env_cfg_unit_sap_cost": spaces.Box(
                low=0, high=100, shape=(1,), dtype=np.int32
            ),
            "env_cfg_unit_sap_range": spaces.Box(
                low=0, high=100, shape=(1,), dtype=np.int32
            )
        })
        
        self.max_steps = Global.MAX_STEPS_IN_MATCH
        self.current_step = 0

        # 全图状态：地图瓦片、遗迹标记、能量地图
        self.tile_map = None     # -1未知、0空地、1星云、2小行星
        self.relic_map = None    # relic 存在标记，1 表示存在
        self.energy_map = None   # 每个 tile 的能量值
        
        # 单位状态：己方和敌方单位列表，每个单位以字典表示 {"x": int, "y": int, "energy": int}
        self.team_units = []    # 己方
        self.enemy_units = []   # 敌方
        
        # 出生点：己方出生于左上角，敌方出生于右下角
        self.team_spawn = (0, 0)
        self.enemy_spawn = (SPACE_SIZE - 1, SPACE_SIZE - 1)
        
        # 探索记录：全图布尔数组，记录己方联合视野中已见过的 tile（全局只记录一次）
        self.visited = None
        
        # 团队得分（己方得分）
        self.score = 0
        
        # 模拟环境的部分参数（env_cfg）
        self.env_cfg = {
            "map_width": SPACE_SIZE,
            "map_height": SPACE_SIZE,
            "max_steps_in_match": Global.MAX_STEPS_IN_MATCH,
            "unit_move_cost": Global.UNIT_MOVE_COST,
            "unit_sap_cost": Global.UNIT_SAP_COST if hasattr(Global, "UNIT_SAP_COST") else 30,
            "unit_sap_range": Global.UNIT_SAP_RANGE,
        }
        
        # 新增：用于 relic 配置相关奖励
        self.relic_configurations = []   # list of (center_x, center_y, mask(5x5 bool))
        self.potential_visited = None      # 全图记录，shape (SPACE_SIZE, SPACE_SIZE)
        self.team_points_space = None      # 全图记录，哪些格子已经贡献过 team point

        self._init_state()

    def _init_state(self):
        """初始化全图状态、单位和记录"""
        num_tiles = SPACE_SIZE * SPACE_SIZE
        
        # 初始化 tile_map：随机部分设为 Nebula (1) 或 Asteroid (2)
        self.tile_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_nebula = int(num_tiles * 0.1)
        num_asteroid = int(num_tiles * 0.1)
        indices = np.random.choice(num_tiles, num_nebula + num_asteroid, replace=False)
        flat_tiles = self.tile_map.flatten()
        flat_tiles[indices[:num_nebula]] = 1
        flat_tiles[indices[num_nebula:]] = 2
        self.tile_map = flat_tiles.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # 初始化 relic_map：随机选取 3 个位置设置为 1（表示存在 relic）
        self.relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        relic_indices = np.random.choice(num_tiles, 3, replace=False)
        flat_relic = self.relic_map.flatten()
        flat_relic[relic_indices] = 1
        self.relic_map = flat_relic.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # 初始化 energy_map：随机生成 2 个能量节点，值设为 MAX_ENERGY_PER_TILE，其余为 0
        self.energy_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_energy_nodes = 2
        indices_energy = np.random.choice(num_tiles, num_energy_nodes, replace=False)
        flat_energy = self.energy_map.flatten()
        for idx in indices_energy:
            flat_energy[idx] = Global.MAX_ENERGY_PER_TILE
        self.energy_map = flat_energy.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # 初始化己方单位：初始生成 1 个单位，出生于 team_spawn
        self.team_units = []
        spawn_x, spawn_y = self.team_spawn
        self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})
        
        # 初始化敌方单位：初始生成 1 个单位，出生于 enemy_spawn
        self.enemy_units = []
        spawn_x_e, spawn_y_e = self.enemy_spawn
        self.enemy_units.append({"x": spawn_x_e, "y": spawn_y_e, "energy": 100})
        
        # 初始化探索记录：全图大小，取各己方单位联合视野后标记已见区域
        self.visited = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        union_mask = self.get_global_sensor_mask()
        self.visited = union_mask.copy()
        
        # 初始化 team score
        self.score = 0
        
        # 新增：初始化 relic 配置，及潜力点记录
        self.relic_configurations = []
        relic_coords = np.argwhere(self.relic_map == 1)
        for (y, x) in relic_coords:
            # 生成一个 5x5 mask，随机选择 5 个格子为 True（训练的时候选10个吧，避免奖励太过于稀疏）
            mask = np.zeros((5,5), dtype=bool)
            indices = np.random.choice(25, 8, replace=False)
            mask_flat = mask.flatten()
            mask_flat[indices] = True
            mask = mask_flat.reshape((5,5))
            self.relic_configurations.append((x, y, mask))
        self.potential_visited = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        self.team_points_space = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        
        self.current_step = 0

    def compute_unit_vision(self, unit):
        """
        根据传入 unit 的位置计算其独立的 sensor mask，
        计算范围为单位传感器范围（切比雪夫距离），并对 Nebula tile 减少贡献。
        取消环绕，只有在地图内的 tile 才计算。
        返回布尔矩阵 shape (SPACE_SIZE, SPACE_SIZE)。
        """
        sensor_range = Global.UNIT_SENSOR_RANGE
        nebula_reduction = 2
        vision = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        x, y = unit["x"], unit["y"]
        for dy in range(-sensor_range, sensor_range + 1):
            for dx in range(-sensor_range, sensor_range + 1):
                new_x = x + dx
                new_y = y + dy
                if not (0 <= new_x < SPACE_SIZE and 0 <= new_y < SPACE_SIZE):
                    continue
                contrib = sensor_range + 1 - max(abs(dx), abs(dy))
                if self.tile_map[new_y, new_x] == 1:
                    contrib -= nebula_reduction
                vision[new_y, new_x] += contrib
        return vision > 0

    def get_global_sensor_mask(self):
        """
        返回己方所有单位 sensor mask 的联合（逻辑 OR）。
        """
        mask = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        for unit in self.team_units:
            mask |= self.compute_unit_vision(unit)
        return mask

    def get_unit_obs(self, unit):
        """
        根据传入 unit 的独立 sensor mask 构造局部观察字典，
        格式与比赛返回固定 JSON 格式相同。
        仅使用该 unit 自己能看到的区域进行过滤。
        """
        sensor_mask = self.compute_unit_vision(unit)
        map_tile_type = np.where(sensor_mask, self.tile_map, -1)
        map_energy = np.where(sensor_mask, self.energy_map, -1)
        map_features = {"tile_type": map_tile_type, "energy": map_energy}
        sensor_mask_int = sensor_mask.astype(np.int8)
        
        # 构造单位信息，分别对己方与敌方单位过滤（使用该 unit 的 sensor mask）
        units_position = np.full((NUM_TEAMS, MAX_UNITS, 2), -1, dtype=np.int32)
        units_energy = np.full((NUM_TEAMS, MAX_UNITS, 1), -1, dtype=np.int32)
        units_mask = np.zeros((NUM_TEAMS, MAX_UNITS), dtype=np.int8)
        for i, u in enumerate(self.team_units):
            ux, uy = u["x"], u["y"]
            if sensor_mask[uy, ux]:
                units_position[0, i] = np.array([ux, uy])
                units_energy[0, i] = u["energy"]
                units_mask[0, i] = 1
        for i, u in enumerate(self.enemy_units):
            ux, uy = u["x"], u["y"]
            if sensor_mask[uy, ux]:
                units_position[1, i] = np.array([ux, uy])
                units_energy[1, i] = u["energy"]
                units_mask[1, i] = 1
        units = {"position": units_position, "energy": units_energy}
        
        # 构造 relic_nodes 信息：仅显示在 sensor_mask 内的 relic 坐标
        relic_coords = np.argwhere(self.relic_map == 1)
        relic_nodes = np.full((MAX_RELIC_NODES, 2), -1, dtype=np.int32)
        relic_nodes_mask = np.zeros(MAX_RELIC_NODES, dtype=np.int8)
        idx = 0
        for (ry, rx) in relic_coords:
            if idx >= MAX_RELIC_NODES:
                break
            if sensor_mask[ry, rx]:
                relic_nodes[idx] = np.array([rx, ry])
                relic_nodes_mask[idx] = 1
            else:
                relic_nodes[idx] = np.array([-1, -1])
                relic_nodes_mask[idx] = 0
            idx += 1
        
        team_points = np.array([self.score, 0], dtype=np.int32)
        team_wins = np.array([0, 0], dtype=np.int32)
        steps = self.current_step
        match_steps = self.current_step
        
        obs = {
            "units": units,
            "units_mask": units_mask,
            "sensor_mask": sensor_mask_int,
            "map_features": map_features,
            "relic_nodes_mask": relic_nodes_mask,
            "relic_nodes": relic_nodes,
            "team_points": team_points,
            "team_wins": team_wins,
            "steps": steps,
            "match_steps": match_steps
        }
        observation = {
            "obs": obs,
            "remainingOverageTime": 60,
            "player": "player_0",
            "info": {"env_cfg": self.env_cfg}
        }
        return observation

    def get_obs(self):
        """
        返回平铺后的全局观测字典，确保所有键与 observation_space 完全一致。
        """
        sensor_mask = self.get_global_sensor_mask()
        sensor_mask_int = sensor_mask.astype(np.int8)
        
        map_features_tile_type = np.where(sensor_mask, self.tile_map, -1)
        map_features_energy = np.where(sensor_mask, self.energy_map, -1)
        
        units_position = np.full((NUM_TEAMS, MAX_UNITS, 2), -1, dtype=np.int32)
        units_energy = np.full((NUM_TEAMS, MAX_UNITS, 1), -1, dtype=np.int32)
        units_mask = np.zeros((NUM_TEAMS, MAX_UNITS), dtype=np.int8)

        # 己方单位
        for i, unit in enumerate(self.team_units):
            ux, uy = unit["x"], unit["y"]
            if sensor_mask[uy, ux]:
                units_position[0, i] = np.array([ux, uy])
                units_energy[0, i] = unit["energy"]
                units_mask[0, i] = 1
        # 敌方单位
        for i, unit in enumerate(self.enemy_units):
            ux, uy = unit["x"], unit["y"]
            if sensor_mask[uy, ux]:
                units_position[1, i] = np.array([ux, uy])
                units_energy[1, i] = unit["energy"]
                units_mask[1, i] = 1
                
        relic_coords = np.argwhere(self.relic_map == 1)
        relic_nodes = np.full((MAX_RELIC_NODES, 2), -1, dtype=np.int32)
        relic_nodes_mask = np.zeros((MAX_RELIC_NODES,), dtype=np.int8)
        idx = 0
        for (ry, rx) in relic_coords:
            if idx >= MAX_RELIC_NODES:
                break
            if sensor_mask[ry, rx]:
                relic_nodes[idx] = np.array([rx, ry])
                relic_nodes_mask[idx] = 1
            else:
                relic_nodes[idx] = np.array([-1, -1])
                relic_nodes_mask[idx] = 0
            idx += 1
    
        team_points = np.array([self.score, 0], dtype=np.int32)
        team_wins = np.array([0, 0], dtype=np.int32)
        steps = np.array([self.current_step], dtype=np.int32)
        match_steps = np.array([self.current_step], dtype=np.int32)
        remainingOverageTime = np.array([60], dtype=np.int32)
        
        env_cfg_map_width = np.array([self.env_cfg["map_width"]], dtype=np.int32)
        env_cfg_map_height = np.array([self.env_cfg["map_height"]], dtype=np.int32)
        env_cfg_max_steps_in_match = np.array([self.env_cfg["max_steps_in_match"]], dtype=np.int32)
        env_cfg_unit_move_cost = np.array([self.env_cfg["unit_move_cost"]], dtype=np.int32)
        env_cfg_unit_sap_cost = np.array([self.env_cfg["unit_sap_cost"]], dtype=np.int32)
        env_cfg_unit_sap_range = np.array([self.env_cfg["unit_sap_range"]], dtype=np.int32)
        
        flat_obs = {
            "units_position": units_position,
            "units_energy": units_energy,
            "units_mask": units_mask,
            "sensor_mask": sensor_mask_int,
            "map_features_tile_type": map_features_tile_type,
            "map_features_energy": map_features_energy,
            "relic_nodes_mask": relic_nodes_mask,
            "relic_nodes": relic_nodes,
            "team_points": team_points,
            "team_wins": team_wins,
            "steps": steps,
            "match_steps": match_steps,
            "remainingOverageTime": remainingOverageTime,
            "env_cfg_map_width": env_cfg_map_width,
            "env_cfg_map_height": env_cfg_map_height,
            "env_cfg_max_steps_in_match": env_cfg_max_steps_in_match,
            "env_cfg_unit_move_cost": env_cfg_unit_move_cost,
            "env_cfg_unit_sap_cost": env_cfg_unit_sap_cost,
            "env_cfg_unit_sap_range": env_cfg_unit_sap_range
        }
        
        return flat_obs
    
    def reset(self):
        """
        重置环境状态，并返回初始的平铺观测数据。
        """
        self._init_state()
        return self.get_obs()

    def _spawn_unit(self, team):
        """生成新单位：己方或敌方，初始能量 100，出生于各自出生点"""
        if team == 0:
            spawn_x, spawn_y = self.team_spawn
            self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})
        elif team == 1:
            spawn_x, spawn_y = self.enemy_spawn
            self.enemy_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})

    def step(self, actions):
        """
        根据动作更新环境状态，并返回 (observation, reward, done, info)。
        修改后的奖励逻辑：
          1. 每个 unit 单独计算 unit_reward。
          2. 若移动动作导致超出地图或目标 tile 为 Asteroid，则判定为无效，unit_reward -0.2。
          3. Sap 动作：
             - 检查 unit 局部 obs 中 relic_nodes_mask 是否存在 relic；
             - 如果存在，统计 unit 8 邻域内敌方单位数，若数目>=2，则 sap 奖励 = +1.0×敌方单位数，否则扣 -2.0；
             - 若无 relic 可见，则同样扣 -2.0。
          4. 非 sap 动作：
             - 成功移动后，检查该 unit 是否位于任一 relic 配置内的潜力点：
                  * 若首次访问该潜力点，unit_reward +2.0，并标记 visited；
                  * 如果该潜力点尚未兑现 team point，则增加 self.score 1，同时 unit_reward +3.0 并标记为 team_points_space；
                  * 如果已在 team_points_space 上，则每回合奖励 +3.0；
             - 若 unit 位于能量节点（energy == Global.MAX_ENERGY_PER_TILE），unit_reward +0.2；
             - 若 unit 位于 Nebula（tile_type==1），unit_reward -0.2；
             - 如果 unit 移动后与敌方 unit 重合，且对方能量低于己方，则对每个满足条件的敌方 unit 奖励 +1.0。
          5. 全局探索奖励：所有己方单位联合视野中新发现 tile，每个奖励 +0.1。
          6. 每一step结束，奖励 point*0.3的奖励 + 规则*0.7的奖励
          7. 每 3 步生成新单位；每 20 步整体滚动地图和敌方单位位置（滚动时对敌方单位使用边界检查）。
        """
        prev_score = self.score
        
        self.current_step += 1
        total_reward = 0.0

        # 处理每个己方单位
        for idx, unit in enumerate(self.team_units):
            unit_reward = 0.0
            act = actions[idx]
            action_enum = ActionType(act)
            # print(f"Unit {idx} action: {action_enum}",file=sys.stderr)
            
            # 获取该 unit 的局部 obs
            unit_obs = self.get_unit_obs(unit)
            
            # 如果动作为 sap
            if action_enum == ActionType.sap:
                # 检查局部 obs 中是否有 relic 可见
                if np.any(unit_obs["obs"]["relic_nodes_mask"] == 1):
                    # 统计 unit 周围 8 邻域内敌方单位数
                    enemy_count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx = unit["x"] + dx
                            ny = unit["y"] + dy
                            if not (0 <= nx < SPACE_SIZE and 0 <= ny < SPACE_SIZE):
                                continue
                            for enemy in self.enemy_units:
                                if enemy["x"] == nx and enemy["y"] == ny:
                                    enemy_count += 1
                    if enemy_count >= 2:
                        unit_reward += 1.0 * enemy_count
                    else:
                        unit_reward -= 1.0
                else:
                    unit_reward -= 1.0
                # Sap 动作不改变位置
            else:
                # 计算移动方向
                if action_enum in [ActionType.up, ActionType.right, ActionType.down, ActionType.left]:
                    dx, dy = action_enum.to_direction()
                else:
                    dx, dy = (0, 0)
                new_x = unit["x"] + dx
                new_y = unit["y"] + dy
                # 检查边界和障碍
                if not (0 <= new_x < SPACE_SIZE and 0 <= new_y < SPACE_SIZE):
                    new_x, new_y = unit["x"], unit["y"]
                    unit_reward -= 0.2  # 超出边界
                elif self.tile_map[new_y, new_x] == 2:
                    new_x, new_y = unit["x"], unit["y"]
                    unit_reward -= 0.2  # 遇到 Asteroid
                else:
                    # 移动成功
                    unit["x"], unit["y"] = new_x, new_y
                
                # 重新获取移动后的局部 obs
                unit_obs = self.get_unit_obs(unit)
                
                # 检查 relic 配置奖励：遍历所有 relic 配置，判断该 unit 是否位于配置中（计算时考虑边界）
                for (rx, ry, mask) in self.relic_configurations:
                    # relic 配置区域范围：中心 (rx, ry) ±2
                    # 如果 unit 在 [rx-2, rx+2] 和 [ry-2, ry+2] 范围内
                    if rx - 2 <= unit["x"] <= rx + 2 and ry - 2 <= unit["y"] <= ry + 2:
                        # 计算在配置 mask 中的索引
                        ix = unit["x"] - rx + 2
                        iy = unit["y"] - ry + 2
                        # 检查索引是否在 mask 范围内（考虑边界）
                        if 0 <= ix < 5 and 0 <= iy < 5:
                            # 如果该潜力点未被访问，则奖励 +2.0
                            if not mask[iy, ix]:
                                if not self.potential_visited[unit["y"], unit["x"]]:
                                    unit_reward += 1.5
                                    self.potential_visited[unit["y"], unit["x"]] = True
                            # 如果潜力点是真的points_space，则奖励 +3.0
                            else:
                                # 如果该点尚未产生 team point，则增加 team point并奖励 +3.0
                                if not self.team_points_space[unit["y"], unit["x"]]:
                                    self.score += 1
                                    unit_reward += 3.0
                                    self.team_points_space[unit["y"], unit["x"]] = True
                                else:
                                    # 已在 team_points_space 上，每回合奖励 +3.0
                                    self.score += 1
                                    unit_reward += 3.0
                # 能量节点奖励
                if unit_obs["obs"]["map_features"]["energy"][unit["y"], unit["x"]] == Global.MAX_ENERGY_PER_TILE:
                    unit_reward += 0.2
                # Nebula 惩罚
                if unit_obs["obs"]["map_features"]["tile_type"][unit["y"], unit["x"]] == 1:
                    unit_reward -= 0.2
                # 攻击行为：若与敌方单位重合且对方能量低于己方，则对每个敌人奖励 +1.0
                for enemy in self.enemy_units:
                    if enemy["x"] == unit["x"] and enemy["y"] == unit["y"]:
                        if enemy["energy"] < unit["energy"]:
                            unit_reward += 1.0
            total_reward += unit_reward
            # print("################################",file=sys.stderr)
            # print("step:",self.current_step)
            # print("")
            # print(total_reward,file=sys.stderr)

        # 全局探索奖励：利用所有己方单位联合视野中新发现的 tile
        union_mask = self.get_global_sensor_mask()
        new_tiles = union_mask & (~self.visited)
        num_new = np.sum(new_tiles)
        if num_new > 0:
            total_reward += 0.2 * num_new
        self.visited[new_tiles] = True

        # 每 3 步生成新单位（若未达到 MAX_UNITS）
        if self.current_step % 3 == 0:
            if len(self.team_units) < MAX_UNITS:
                self._spawn_unit(team=0)
            if len(self.enemy_units) < MAX_UNITS:
                self._spawn_unit(team=1)

        # 每 20 步整体滚动地图、遗迹和能量图，以及敌方单位位置（右移 1 格，边界检查）
        if self.current_step % 20 == 0:
            # 这里采用 np.roll 保持地图内部数据不变，但对于敌方单位，我们检查边界
            self.tile_map = np.roll(self.tile_map, shift=1, axis=1)
            self.relic_map = np.roll(self.relic_map, shift=1, axis=1)
            self.energy_map = np.roll(self.energy_map, shift=1, axis=1)
            for enemy in self.enemy_units:
                new_ex = enemy["x"] + 1
                if new_ex >= SPACE_SIZE:
                    new_ex = enemy["x"]  # 保持不变
                enemy["x"] = new_ex

        # 在 step 结束时计算 self.score 的增加量
        score_increase = self.score - prev_score
    
        # 将总奖励合并：total_reward * 0.5 + score_increase * 0.5
        final_reward = total_reward * 0.5 + score_increase * 0.5
        # final_reward =score_increase
        
        done = self.current_step >= self.max_steps
        # done = self.current_step >= 200
        info = {"score": self.score, "step": self.current_step}
        return self.get_obs(), final_reward, done, info

    def render(self, mode='human'):
        display = self.tile_map.astype(str).copy()
        for unit in self.team_units:
            display[unit["y"], unit["x"]] = 'A'
        print("Step:", self.current_step)
        print(display)



%%writefile agent/train.py

from stable_baselines3 import PPO
from ppo_game_env import PPOGameEnv

# 创建环境实例
env = PPOGameEnv()

# 使用多层感知机策略初始化 PPO 模型
# model = PPO("MultiInputPolicy", env,learning_rate=0.0005,ent_coef=0.1,vf_coef = 0.3, verbose=1)
model = PPO("MultiInputPolicy", env,learning_rate=0.0005, verbose=1)
# model_1 = PPO("MultiInputPolicy", env,learning_rate=0.0005, verbose=1)



# total_timesteps may need to adjust
# model.learn(total_timesteps=960000)
model.learn(total_timesteps=6000)

# 保存训练好的模型
model.save("/kaggle/working/agent/ppo_game_env_model")

# 测试：加载模型并进行一次模拟
# obs = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()



import os
import sys
import numpy as np
from stable_baselines3 import PPO

def transform_obs(comp_obs, env_cfg=None):
    """
    将比赛引擎返回的 JSON 观测转换为模型训练时使用的平铺观测格式。
    
    比赛环境的观测格式（comp_obs）结构如下：
      {
        "obs": {
            "units": {"position": Array(T, N, 2), "energy": Array(T, N, 1)},
            "units_mask": Array(T, N),
            "sensor_mask": Array(W, H),
            "map_features": {"energy": Array(W, H), "tile_type": Array(W, H)},
            "relic_nodes_mask": Array(R),
            "relic_nodes": Array(R, 2),
            "team_points": Array(T),
            "team_wins": Array(T),
            "steps": int,
            "match_steps": int
        },
        "remainingOverageTime": int,
        "player": str,
        "info": {"env_cfg": dict}
      }
    
    我们需要构造如下平铺字典（与 PPOGameEnv.get_obs() 返回的格式一致）：
      {
        "units_position": (T, N, 2),
        "units_energy": (T, N, 1),
        "units_mask": (T, N),
        "sensor_mask": (W, H),
        "map_features_tile_type": (W, H),
        "map_features_energy": (W, H),
        "relic_nodes_mask": (R,),
        "relic_nodes": (R, 2),
        "team_points": (T,),
        "team_wins": (T,),
        "steps": (1,),
        "match_steps": (1,),
        "remainingOverageTime": (1,),
        "env_cfg_map_width": (1,),
        "env_cfg_map_height": (1,),
        "env_cfg_max_steps_in_match": (1,),
        "env_cfg_unit_move_cost": (1,),
        "env_cfg_unit_sap_cost": (1,),
        "env_cfg_unit_sap_range": (1,)
      }
    """
    # 如果存在 "obs" 键，则取其内部数据，否则直接使用 comp_obs
    if "obs" in comp_obs:
        base_obs = comp_obs["obs"]
    else:
        base_obs = comp_obs


    flat_obs = {}

    # 处理 units 数据
    if "units" in base_obs:
        flat_obs["units_position"] = np.array(base_obs["units"]["position"], dtype=np.int32)
        flat_obs["units_energy"] = np.array(base_obs["units"]["energy"], dtype=np.int32)
        # 如果 units_energy 的 shape 为 (NUM_TEAMS, MAX_UNITS) 则扩展一个维度
        if flat_obs["units_energy"].ndim == 2:
            flat_obs["units_energy"] = np.expand_dims(flat_obs["units_energy"], axis=-1)
    else:
        flat_obs["units_position"] = np.array(base_obs["units_position"], dtype=np.int32)
        flat_obs["units_energy"] = np.array(base_obs["units_energy"], dtype=np.int32)
        if flat_obs["units_energy"].ndim == 2:
            flat_obs["units_energy"] = np.expand_dims(flat_obs["units_energy"], axis=-1)
    
    # 处理 units_mask
    if "units_mask" in base_obs:
        flat_obs["units_mask"] = np.array(base_obs["units_mask"], dtype=np.int8)
    else:
        flat_obs["units_mask"] = np.zeros(flat_obs["units_position"].shape[:2], dtype=np.int8)
    
    # 处理 sensor_mask：若返回的是 3D 数组，则取逻辑 or 得到全局 mask
    sensor_mask_arr = np.array(base_obs["sensor_mask"], dtype=np.int8)
    if sensor_mask_arr.ndim == 3:
        sensor_mask = np.any(sensor_mask_arr, axis=0).astype(np.int8)
    else:
        sensor_mask = sensor_mask_arr
    flat_obs["sensor_mask"] = sensor_mask

    # 处理 map_features（tile_type 与 energy）
    if "map_features" in base_obs:
        mf = base_obs["map_features"]
        flat_obs["map_features_tile_type"] = np.array(mf["tile_type"], dtype=np.int8)
        flat_obs["map_features_energy"] = np.array(mf["energy"], dtype=np.int8)
    else:
        flat_obs["map_features_tile_type"] = np.array(base_obs["map_features_tile_type"], dtype=np.int8)
        flat_obs["map_features_energy"] = np.array(base_obs["map_features_energy"], dtype=np.int8)

    # 处理 relic 节点信息
    if "relic_nodes_mask" in base_obs:
        flat_obs["relic_nodes_mask"] = np.array(base_obs["relic_nodes_mask"], dtype=np.int8)
    else:
        max_relic = env_cfg.get("max_relic_nodes", 6) if env_cfg is not None else 6
        flat_obs["relic_nodes_mask"] = np.zeros((max_relic,), dtype=np.int8)
    if "relic_nodes" in base_obs:
        flat_obs["relic_nodes"] = np.array(base_obs["relic_nodes"], dtype=np.int32)
    else:
        max_relic = env_cfg.get("max_relic_nodes", 6) if env_cfg is not None else 6
        flat_obs["relic_nodes"] = np.full((max_relic, 2), -1, dtype=np.int32)

    # 处理团队得分与胜局
    if "team_points" in base_obs:
        flat_obs["team_points"] = np.array(base_obs["team_points"], dtype=np.int32)
    else:
        flat_obs["team_points"] = np.zeros(2, dtype=np.int32)
    if "team_wins" in base_obs:
        flat_obs["team_wins"] = np.array(base_obs["team_wins"], dtype=np.int32)
    else:
        flat_obs["team_wins"] = np.zeros(2, dtype=np.int32)

    # 处理步数信息
    if "steps" in base_obs:
        flat_obs["steps"] = np.array([base_obs["steps"]], dtype=np.int32)
    else:
        flat_obs["steps"] = np.array([0], dtype=np.int32)
    if "match_steps" in base_obs:
        flat_obs["match_steps"] = np.array([base_obs["match_steps"]], dtype=np.int32)
    else:
        flat_obs["match_steps"] = np.array([0], dtype=np.int32)

    # 注意：不在此处处理 remainingOverageTime，
    # 将在 Agent.act 中利用传入的参数添加

    # 补全环境配置信息
    if env_cfg is not None:
        flat_obs["env_cfg_map_width"] = np.array([env_cfg["map_width"]], dtype=np.int32)
        flat_obs["env_cfg_map_height"] = np.array([env_cfg["map_height"]], dtype=np.int32)
        flat_obs["env_cfg_max_steps_in_match"] = np.array([env_cfg["max_steps_in_match"]], dtype=np.int32)
        flat_obs["env_cfg_unit_move_cost"] = np.array([env_cfg["unit_move_cost"]], dtype=np.int32)
        flat_obs["env_cfg_unit_sap_cost"] = np.array([env_cfg["unit_sap_cost"]], dtype=np.int32)
        flat_obs["env_cfg_unit_sap_range"] = np.array([env_cfg["unit_sap_range"]], dtype=np.int32)
    else:
        flat_obs["env_cfg_map_width"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_map_height"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_max_steps_in_match"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_unit_move_cost"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_unit_sap_cost"] = np.array([0], dtype=np.int32)
        flat_obs["env_cfg_unit_sap_range"] = np.array([0], dtype=np.int32)

    return flat_obs

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        # 如果 env_cfg 中没有 "max_units"，则补上默认值 16
        if "max_units" not in self.env_cfg:
            self.env_cfg["max_units"] = 16

        # 加载训练好的 PPO 模型（请确保模型文件路径正确）
        model_path = os.path.join(os.path.dirname(__file__), "ppo_game_env_model.zip")
        self.model = PPO.load(model_path)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        根据比赛观测与当前步数决定各单位动作。
        输出为形状 (max_units, 3) 的 numpy 数组，每行格式为 [动作类型, delta_x, delta_y]，
        其中非汲取动作时 delta_x 和 delta_y 固定为 0。
        """
        import sys
        # # 当 step 为 11 时打印调试信息
        # if step == 11:
        #     print("DEBUG: Agent.act() 调用参数：", file=sys.stderr)
        #     print("DEBUG: self.player =", self.player, file=sys.stderr)
        #     print("DEBUG: step =", step, file=sys.stderr)
        #     # 打印 obs 的 key 列表，可以查看观测数据的大致结构
        #     print("DEBUG: obs keys =", list(obs.keys()), file=sys.stderr)
        #     print("=============================================================", file=sys.stderr)
        #     print("DEBUG: ob =", obs, file=sys.stderr)
        #     print("DEBUG: remainingOverageTime =", remainingOverageTime, file=sys.stderr)
        #     print("#############################################################", file=sys.stderr)
        
        flat_obs = transform_obs(obs, self.env_cfg)
        # 如果当前 agent 为 player_1，则交换单位信息（确保和训练时候一致，己方视角永远在第一位置）
        if self.player == "player_1":
            flat_obs["units_position"] = flat_obs["units_position"][::-1]
            flat_obs["units_energy"] = flat_obs["units_energy"][::-1]
            flat_obs["units_mask"] = flat_obs["units_mask"][::-1]
            
        # 手动添加 remainingOverageTime（取自传入参数）
        flat_obs["remainingOverageTime"] = np.array([remainingOverageTime], dtype=np.int32)

        # if step == 11:
        #     print("------------------------------------------------------------", file=sys.stderr)
        #     print("DEBUG: flat_obs =", flat_obs, file=sys.stderr)
        #     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", file=sys.stderr)
        # 使用模型预测动作（deterministic 模式）
        action, _ = self.model.predict(flat_obs, deterministic=True)
        # 确保 action 为 numpy 数组，并显式设置为 np.int32 类型
        action = np.array(action, dtype=np.int32)

        max_units = self.env_cfg["max_units"]
        actions = np.zeros((max_units, 3), dtype=np.int32)
        for i, a in enumerate(action):
            actions[i, 0] = int(a)
            actions[i, 1] = 0  # 若为 sap 动作，可在此扩展目标偏移
            actions[i, 2] = 0
        return actions




%%writefile agent/main.py

import json
from typing import Dict
import sys
from argparse import Namespace

import numpy as np

from agent import Agent
# from lux.config import EnvConfig
from lux.kit import from_json
### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict() # store potentially multiple dictionaries as kaggle imports code directly
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
        observation = Namespace(**dict(step=raw_input["step"], obs=raw_input["obs"], remainingOverageTime=raw_input["remainingOverageTime"], player=raw_input["player"], info=raw_input["info"]))
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        # send actions to engine
        print(json.dumps(actions))



!pip install --upgrade luxai-s3


!python agent/train.py


!luxai-s3 agent/main.py agent/main.py --seed 37 --output=replay.html


import IPython # load the HTML replay
IPython.display.HTML(filename='replay.html')


!cd agent && tar -czf submission.tar.gz *
!mv agent/submission.tar.gz .




