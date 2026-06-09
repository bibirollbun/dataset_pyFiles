!pip install luxai-s3


!pip install -U "jax[cuda12]"


!export JAX_PLATFORM_NAME="cuda"


import os
os.environ['JAX_PLATFORM_NAME'] = "cuda"


import numpy as np

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from copy import deepcopy

class Agent:
    def __init__(
        self, player
    ):
        self.player = player
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):        
        return np.array([[np.random.randint(0, 6), 0, 0] for _ in range(16)])
    
import time

def evaluate_agents(
    agent_1_cls, agent_2_cls, seed=1, games_to_play=1, disclose_all=True
):
    env = LuxAIS3GymEnv(numpy_output=True)
    
    obs, info = env.reset(seed=seed)
    
    t = time.time()
    for i in range(games_to_play):
        obs, info = env.reset()
        env_cfg = info["params"]  # only contains observable game parameters
        player_0 = agent_1_cls("player_0")
        player_1 = agent_2_cls("player_1")
        # main game loop
        game_done = False
        step = 0
        print(f"Running game {i}")
        prev_action = None
        while not game_done:
            actions = dict()
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
                
            obs, reward, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1
            prev_action = actions
        
    print(f"Time taken: {time.time() - t}")
    env.close()  # free up resources and save final replay

evaluate_agents(
    Agent,
    Agent,
)


import numpy as np

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from copy import deepcopy

class Agent:
    def __init__(
        self, player
    ):
        self.player = player
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):        
        return np.array([[np.random.randint(0, 6), 0, 0] for _ in range(16)])
    
import time

def evaluate_agents(
    agent_1_cls, agent_2_cls, seed=1, games_to_play=1, disclose_all=True
):
    env = LuxAIS3GymEnv(numpy_output=False)
    
    obs, info = env.reset(seed=seed)
    
    t = time.time()
    for i in range(games_to_play):
        obs, info = env.reset()
        env_cfg = info["params"]  # only contains observable game parameters
        player_0 = agent_1_cls("player_0")
        player_1 = agent_2_cls("player_1")
        # main game loop
        game_done = False
        step = 0
        print(f"Running game {i}")
        prev_action = None
        while not game_done:
            actions = dict()
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
                
            obs, reward, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1
            prev_action = actions
        
    print(f"Time taken: {time.time() - t}")
    env.close()  # free up resources and save final replay

evaluate_agents(
    Agent,
    Agent,
)





!export JAX_PLATFORM_NAME="cpu"
import os
os.environ['JAX_PLATFORM_NAME'] = "cpu"


import numpy as np

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from copy import deepcopy

class Agent:
    def __init__(
        self, player
    ):
        self.player = player
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):        
        return np.array([[np.random.randint(0, 6), 0, 0] for _ in range(16)])
    
import time

def evaluate_agents(
    agent_1_cls, agent_2_cls, seed=1, games_to_play=1, disclose_all=True
):
    env = LuxAIS3GymEnv(numpy_output=True)
    
    obs, info = env.reset(seed=seed)
    
    t = time.time()
    for i in range(games_to_play):
        obs, info = env.reset()
        env_cfg = info["params"]  # only contains observable game parameters
        player_0 = agent_1_cls("player_0")
        player_1 = agent_2_cls("player_1")
        # main game loop
        game_done = False
        step = 0
        print(f"Running game {i}")
        prev_action = None
        while not game_done:
            actions = dict()
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
                
            obs, reward, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1
            prev_action = actions
        
    print(f"Time taken: {time.time() - t}")
    env.close()  # free up resources and save final replay

evaluate_agents(
    Agent,
    Agent,
)


import numpy as np

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from copy import deepcopy

class Agent:
    def __init__(
        self, player
    ):
        self.player = player
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):        
        return np.array([[np.random.randint(0, 6), 0, 0] for _ in range(16)])
    
import time

def evaluate_agents(
    agent_1_cls, agent_2_cls, seed=1, games_to_play=1, disclose_all=True
):
    env = LuxAIS3GymEnv(numpy_output=False)
    
    obs, info = env.reset(seed=seed)
    
    t = time.time()
    for i in range(games_to_play):
        obs, info = env.reset()
        env_cfg = info["params"]  # only contains observable game parameters
        player_0 = agent_1_cls("player_0")
        player_1 = agent_2_cls("player_1")
        # main game loop
        game_done = False
        step = 0
        print(f"Running game {i}")
        prev_action = None
        while not game_done:
            actions = dict()
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
                
            obs, reward, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1
            prev_action = actions
        
    print(f"Time taken: {time.time() - t}")
    env.close()  # free up resources and save final replay

evaluate_agents(
    Agent,
    Agent,
)










