import torch 
from torch import nn, optim
import numpy as np 
from collections import deque
import random 
from typing import List, AnyStr
import os

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



!mkdir agent


%%writefile agent/kit.py
import numpy as np


def to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(s) for s in obj]
    elif isinstance(obj, dict):
        out = {}
        for k in obj:
            out[k] = to_json(obj[k])
        return out
    else:
        return obj


def from_json(state):
    if isinstance(state, list):
        return np.array(state)
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state 


%%writefile agent/baseDQN.py

import torch 
from torch import nn, optim
import numpy as np 
from collections import deque
import random 
from typing import List, AnyStr
import os


def all_seed(seed=6666):
    np.random.seed(seed)
    random.seed(seed)
    # CPU
    torch.manual_seed(seed)
    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    # python全局
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')


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
    def __init__(self, max_len: int, np_save: bool=False):
        self._buffer = deque(maxlen=max_len)
        self.np_save = np_save
    
    def add(self, state, action, reward, next_state, done):
        self._buffer.append( (state, action, reward, next_state, done) )
    
    def __len__(self):
        return len(self._buffer)

    def sample(self, batch_size: int) -> deque:
        sample = random.sample(self._buffer, batch_size)
        return sample


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_layers_dim, action_dim):
        super(QNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(
                nn.ModuleDict({
                    'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                    'linear_active': nn.ReLU(inplace=True)
                })
            )
        self.head = nn.Linear(hidden_layers_dim[-1], action_dim) 
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_active'](layer['linear'](x))
        return self.head(x)


class DQN:
    def __init__(
        self, 
        player: AnyStr,
        env_cfg,
        state_dim: int,
        hidden_layers_dim: List,
        action_dim: int,
        max_len: int,
        learning_rate: float=0.0001,
        gamma: float=0.99,
        epsilon: float=0.05, 
        target_update_freq: int=1,
        dqn_type: AnyStr='DQN',
        epsilon_start: float=None,
        epsilon_decay_factor: float=None,
        device: AnyStr='cuda',
        random_flag: bool=False,
        min_samples: int=10000
    ):
        # player
        self.random_flag = random_flag
        self.min_samples = min_samples
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.max_len = max_len
        self.buffer = ReplayBuffer(max_len=max_len)
        self.unit_sap_range = self.env_cfg['unit_sap_range']

        self.state_dim = state_dim
        self.hidden_layers_dim = hidden_layers_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_end = epsilon
        self.target_update_freq = target_update_freq
        self.dqn_type = dqn_type
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start if epsilon_start is not None else epsilon
        # print(f'{self.epsilon_start=}')
        self.epsilon_decay_factor = epsilon_decay_factor
        self.device = device
        # qNet
        self.q = QNet(state_dim, hidden_layers_dim, action_dim).to(self.device)
        self.target_q = QNet(state_dim, hidden_layers_dim, action_dim).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        # loss 
        self.cost_func = nn.MSELoss()
        # opt 
        self.opt = optim.Adam(self.q.parameters(), lr=self.learning_rate)
        self.dqn_type = dqn_type
        self.count = 1
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()        
        self.train()

    def reset(self):
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()    
    
    def train(self):
        self.training = True
        self.q.train()
        self.target_q.train()

    def eval(self):
        self.training = False
        self.q.eval()
        self.target_q.eval()

    def _epsilon_update(self):
        self.epsilon = self.epsilon * self.epsilon_decay_factor
        if self.epsilon > self.epsilon_end:
            return self.epsilon
        return self.epsilon_end

    @torch.no_grad()
    def policy(self, step, obs, remainingOverageTime: int = 60):
        tmp_random_flag = len(self.buffer) < self.min_samples
        idx = self.team_id
        unit_mask = np.array(obs['units_mask'][idx]) # 
        unit_positions = np.array(obs['units']['position'][idx])
        unit_energys = np.array(obs['units']['energy'][idx])
        relic_nodes = np.array(obs['relic_nodes'])
        relic_mask = np.array(obs['relic_nodes_mask'])
        obv_relic_node_positions = np.array(obs['relic_nodes'])
        obv_relic_node_mask = np.array(obs['relic_nodes_mask'])

        # 可行动units
        available_units = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(obv_relic_node_mask)[0])
        # action
        actions = np.zeros((self.env_cfg['max_units'], 3), dtype=int)
        # basic strategy here is simply to have some units randomly explore 
        #           and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match
        for id_ in visible_relic_node_ids:
            if id_ not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id_)
                self.relic_node_positions.append(obv_relic_node_positions[id_])
        # unit ids range from 0 to max_units - 1
        for unit_id in available_units:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            state = self._state_representation(
                unit_pos,
                unit_energy,
                relic_nodes,
                step,
                relic_mask,
                obs
            )

            if tmp_random_flag or self.random_flag or (self.training and np.random.random() < self.epsilon):
                if len(self.relic_node_positions) > 0:
                # if len(visible_relic_node_ids) > 0:
                    nearest_relic_node_position = self.relic_node_positions[0]
                    manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + \
                        abs(unit_pos[1] - nearest_relic_node_position[1])
                    # if close to the relic node we want to move randomly around it 
                    # and hope to gain points
                    if manhattan_distance <= 4:
                        random_direction = np.random.randint(0, 5)
                        actions[unit_id] = [random_direction, 0, 0]
                    else:
                        actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
                else:
                    # pick a random location on the map for the unit to explore
                    if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                        rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                        self.unit_explore_locations[unit_id] = rand_loc
                    actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
                continue 
            # DQN policy 
            q_sa = self.q(torch.FloatTensor(state).to(self.device))
            act_np = q_sa.cpu().detach().numpy()
            act = np.argmax(act_np)
            if act == 5:  # Sap action
                # Find closest enemy unit
                # valid_targets = self._find_opp_units(obs)
                valid_targets = np.array(self._find_opp_units(obs, unit_pos))
                if len(valid_targets):
                    target_pos = valid_targets[0] # Choose first valid target
                    actions[unit_id] = [5, target_pos[0], target_pos[1]]
                else:
                    # act_bool = np.argsort(act_np) == self.action_dim - 2
                    # actions[unit_id] = [np.arange(self.action_dim)[act_bool][0], 0, 0]  # 采用次优动作
                    actions[unit_id] = [0, 0, 0] # 留在原地
                    # print("act == 5 ERROR")
            else:
                actions[unit_id] = [act, 0, 0]

        # if not self.random_flag:
        #     print(f"policy {actions=}")
        return actions

    def _find_opp_units(self, obs, unit_pos):
        opp_positions = obs['units']['position'][self.opp_team_id]
        opp_mask = obs['units_mask'][self.opp_team_id]
        valid_targets = []
        for opp_id, pos in enumerate(opp_positions):
            if (opp_mask[opp_id] and pos[0] != -1 
                and np.abs(pos[0] - unit_pos[0]) <= self.unit_sap_range
                and np.abs(pos[1] - unit_pos[1]) <= self.unit_sap_range
            ):
                valid_targets.append(pos - unit_pos)
        return valid_targets

    def _state_representation(self, unit_pos, unit_energy, relic_nodes, step, relic_mask, obs):
        if not relic_mask.any():
            closest_relic = np.array([-1, -1])
        else:
            visible_relics = relic_nodes[relic_mask]
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]
        
        valid_targets = self._find_opp_units(obs, unit_pos)
        # if len(valid_targets):
        #     enemy_dis = np.linalg.norm(valid_targets - unit_pos, axis=1)
        #     closest_enemy_pos = valid_targets[np.argmin(enemy_dis)]
        #     can_sap_num = ((np.abs(valid_targets - unit_pos) <= self.unit_sap_range).sum(axis=1) >= 2).sum()
        # else:
        #     can_sap_num = 0
        #     closest_enemy_pos = np.array([-1, -1])

        mach_num = step // (self.env_cfg['max_steps_in_match'] + 1) + 1
        state = np.concatenate([
            unit_pos,
            closest_relic,
            [len(valid_targets)],
            [unit_energy],
            [step/505.0],  # Normalize step
        ])
        return state

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return 
        if self.training and self.epsilon_start is not None:
            self.epsilon = self._epsilon_update()
        self.count += 1
        samples = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, done = zip(*samples)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.Tensor(np.array(actions)).view(-1, 1).to(self.device)
        rewards = torch.Tensor(np.array(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        done = torch.Tensor(np.array(done)).view(-1, 1).to(self.device)

        ut = self.q(states).gather(1, actions.long())
        Q_sa1 = self.target_q(next_states)
        # 下一状态最大值
        if 'DoubleDQN' in self.dqn_type:
            # a* = argmax Q(s_{t+1}, a; w)
            a_star = self.q(next_states).max(1)[1].view(-1, 1)
            # doubleDQN Q(s_{t+1}, a*; w')
            ut_1 = Q_sa1.gather(1, a_star)
        else:
            # simple method:  avoid bootstrapping 
            ut_1 = Q_sa1.max(1)[0].view(-1, 1)
        
        q_tar = rewards + self.gamma * ut_1 * (1 - done)
        # update
        self.opt.zero_grad()
        loss = self.cost_func(ut.float(), q_tar.float())
        loss.backward()
        self.opt.step()
        if self.count > 0 and self.count % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

    def save_model(self, file_path, player=None):
        pl = self.player if player is None else player
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        q_f = os.path.join(file_path, f'{self.dqn_type}_model_{pl}_rd_{self.random_flag}.pth')
        torch.save(self.q.state_dict(), q_f)

    def load_model(self, file_path, player=None):
        pl = self.player if player is None else player
        q_f = os.path.join(file_path, f'{self.dqn_type}_model_{pl}_rd_{self.random_flag}.pth')
        # print(f'load_model -> {q_f}')
        try: 
            self.target_q.load_state_dict(torch.load(q_f, weights_only=True))
            self.q.load_state_dict(torch.load(q_f, weights_only=True))
        except Exception as e:
            self.target_q.load_state_dict(torch.load(q_f, map_location='cpu', weights_only=True))
            self.q.load_state_dict(torch.load(q_f, map_location='cpu', weights_only=True))

        self.q.to(self.device)
        self.target_q.to(self.device)
        self.opt = optim.Adam(self.q.parameters(), lr=self.learning_rate)



%%writefile agent/baseTrain.py

# python3
# Author: Scc_hy
# Create Date: 2025-01-17
# Reference: https://www.kaggle.com/code/sangrampatil5150/nuralbrain-v0-5-model-train-and-win
# ===========================================================================================
import os 
import copy
import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from collections import deque
from argparse import Namespace
from baseDQN import all_seed, DQN
# from advDQN import all_seed, DQN
from luxai_s3.wrappers import LuxAIS3GymEnv


def train_off_policy(
        env, 
        player_0, 
        player_1,
        cfg,
        wandb_flag=False,
        wandb_project_name="LuxAI",
    ):
    
    obs, info = env.reset(seed=cfg.seed)
    env_cfg = info["params"]   # UNIT_SAP_RANGE
    if wandb_flag:
        wandb.login()
        cfg_dict = cfg.__dict__ 
        algo = player_0.__class__.__name__
        now_ = datetime.now().strftime('%Y%m%d__%H%M')
        wandb.init(
            project=wandb_project_name,
            name= f"{algo}__LuxAI__{now_}",
            config=cfg_dict,
            monitor_gym=True
        )

    tq_bar = tqdm(range(cfg.num_episode))
    final_seed = cfg.seed
    palyers_rewards_list = {
        'player_0': deque(maxlen=10),
        'player_1': deque(maxlen=10),
    }
    now_reward= {
        'player_0': -np.inf,
        'player_1': -np.inf
    }
    palyers_win_list = {
        'player_0': deque(maxlen=10),
        'player_1': deque(maxlen=10),
    }
    now_win_rate = {
        'player_0': 0,
        'player_1': 0
    }
    for i in tq_bar:
        obs, info = env.reset(seed=final_seed)
        # dqn collection reset 
        player_0.reset()
        player_1.reset()

        done = False 
        step = 0
        last_obs = None
        last_actions = None
        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode}|(seed={final_seed}) ]')
        episode_rewards = {
            'player_0': 0,
            'player_1': 0,
        }
        while not done:
            actions = {}
            # Store current observation for learning
            last_obs = {
                "player_0": obs["player_0"].copy(),
                "player_1": obs["player_1"].copy()
            }
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.policy(step=step, obs=obs[agent.player])
            
            last_actions = copy.deepcopy(actions)
            # Environment step
            # print(f'{actions=}')
            obs, rewards ,terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id],
                "player_1": obs["player_1"]["team_points"][player_1.team_id]
            }  
            if last_obs is not None:
                for agent in [player_0, player_1]:
                    for unit_id in range(env_cfg["max_units"]):
                        if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            current_state = agent._state_representation(
                                last_obs[agent.player]['units']['position'][agent.team_id][unit_id],
                                last_obs[agent.player]['units']['energy'][agent.team_id][unit_id],
                                last_obs[agent.player]['relic_nodes'],
                                step,
                                last_obs[agent.player]['relic_nodes_mask'],
                                last_obs[agent.player]
                            )
                            next_state = agent._state_representation(
                                obs[agent.player]['units']['position'][agent.team_id][unit_id],
                                obs[agent.player]['units']['energy'][agent.team_id][unit_id],
                                obs[agent.player]['relic_nodes'],
                                step + 1,
                                obs[agent.player]['relic_nodes_mask'],
                                obs[agent.player]
                            )
                            agent.buffer.add(
                                current_state,
                                last_actions[agent.player][unit_id][0],
                                rewards[agent.player],
                                next_state,
                                dones[agent.player]
                            )
                            
                            episode_rewards[agent.player] += rewards[agent.player]

                if not player_0.random_flag:
                    player_0.update(cfg.batch_size)
                if not player_1.random_flag:
                    player_1.update(cfg.batch_size)

            if dones["player_0"] or dones["player_1"]:
                done = True
                player_0.save_model(cfg.save_path)
                player_1.save_model(cfg.save_path)
                p0_win_flag = (episode_rewards["player_0"] > episode_rewards["player_1"])
                palyers_win_list["player_0"].append(1 if p0_win_flag else 0)
                palyers_win_list["player_1"].append(0 if p0_win_flag else 1)

            step += 1

        for p in ["player_0", "player_1"]:
            palyers_rewards_list[p].append(episode_rewards[p])
            now_reward[p] = np.mean(palyers_rewards_list[p])
            now_win_rate[p] = np.mean(palyers_win_list[p])

        # print(f'{palyers_win_list=}\n{episode_rewards=}')
        p0_r = now_reward["player_0"]
        p0_w_r = now_win_rate["player_0"]
        p1_r = now_reward["player_1"]
        p1_w_r = now_win_rate["player_1"]
        tq_bar.set_postfix({
            "steps": step,
            'p0-lstR': f'{p0_r:.2f}',
            'p0-winR': f'{p0_w_r:.2f}',
            'p1-lstR': f'{p1_r:.2f}',
            'p1-winR': f'{p1_w_r:.2f}'
        })
        if wandb_flag:
            log_dict = {
                "steps": step,
                'p0-lstR': p0_r,
                'p0-winR': p0_w_r,
                'p1-lstR': p1_r,
                'p1-winR': p1_w_r
            }
            wandb.log(log_dict)
    if wandb_flag:
        wandb.finish()
    env.close()


def step_train(
    player0_random_flag=False, 
    player0_load_dir=None, 
    player0_load_palyer=None,
    player1_random_flag=False,
    player1_load_dir=None, 
    player1_load_palyer=None,
    num_episode=500,
    save_dir=None,
    epsilon_start=0.99
):
    all_seed(202501)
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=202501)
    env_cfg = info["params"]  
    path_ = "/kaggle/working"
    dt = datetime.now().strftime('%Y%m%d')
    config = Namespace(
        seed=202502,
        num_episode=num_episode,
        batch_size=128,
        min_samples=256,
        save_path=os.path.join(path_, "test_models" ,f'DQN_LuxAI_V0_{dt}') if save_dir is None else save_dir,
        state_dim=7, # unit_pos(2) + closest_relic(2) + unit_energy(1) + step(1) 
        action_dim=6, # stay, up, right, down, left, sap
        hidden_layers_dim=[220, 220],
        buffer_max_len=20000,
        learning_rate=2.5e-4, # 0.0001
        gamma=0.99,
        epsilon=0.01,
        target_update_freq=env_cfg['max_units'] * 10,
        dqn_type="DoubleDQN",
        epsilon_start=epsilon_start,
        epsilon_decay_factor=0.995,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        env_cfg=env_cfg
    )
    rd_player_0 = DQN("player_0",
        env_cfg,
        state_dim=config.state_dim,
        hidden_layers_dim=config.hidden_layers_dim,
        action_dim=config.action_dim,
        max_len=config.buffer_max_len,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon=config.epsilon,
        target_update_freq=config.target_update_freq,
        dqn_type=config.dqn_type,
        epsilon_start=config.epsilon_start,
        epsilon_decay_factor=config.epsilon_decay_factor,
        device=config.device,
        random_flag=player0_random_flag,
        min_samples=config.min_samples
    )
    if player0_load_dir is not None:
        rd_player_0.random_flag = False 
        rd_player_0.load_model(player0_load_dir, player=player0_load_palyer)
        rd_player_0.random_flag = player0_random_flag
    
    rd_player_1 = DQN("player_1",
        env_cfg,
        state_dim=config.state_dim,
        hidden_layers_dim=config.hidden_layers_dim,
        action_dim=config.action_dim,
        max_len=config.buffer_max_len,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon=config.epsilon,
        target_update_freq=config.target_update_freq,
        dqn_type=config.dqn_type,
        epsilon_start=config.epsilon_start,
        epsilon_decay_factor=config.epsilon_decay_factor,
        device=config.device,
        random_flag=player1_random_flag,
        min_samples=config.min_samples
    )

    if player1_load_dir is not None:
        rd_player_1.random_flag = False 
        rd_player_1.load_model(player1_load_dir, player=player1_load_palyer)
        rd_player_1.random_flag = player1_random_flag

    train_off_policy(
        env, 
        rd_player_0, 
        rd_player_1,
        config,
        wandb_flag=False,
        wandb_project_name="LuxAI",
    )



if __name__ == '__main__':
    path_ = "/kaggle/working"
    model_d_s1 = os.path.join(path_, "test_models" ,f'DQN_LuxAI_s1_dqn_vs_random_v1')

    # step1 dqn VS random
    step_train(
        player0_random_flag=False, 
        player0_load_dir=None, # model_d_s0, 
        player0_load_palyer=None, #'player_0',

        player1_random_flag=True,
        player1_load_dir=None, 
        player1_load_palyer=None,

        num_episode=400,
        save_dir=model_d_s1
    )



%%writefile agent/main.py

import json
from typing import Dict
import os 
import sys
from argparse import Namespace
import numpy as np
from kit import from_json
from baseDQN import all_seed, DQN
import torch 


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

    env_cfg = configurations["env_cfg"]
    model_d_s1 = '/kaggle_simulations/agent'
    if os.path.exists('/kaggle/working/agent/DoubleDQN_model_player_0_rd_False.pth'):
        model_d_s1 = os.path.join('/kaggle/working', 'agent')
    config = Namespace(
        seed=202502,
        num_episode=100,
        batch_size=128,
        min_samples=0,
        save_path=model_d_s1,
        state_dim=7, # unit_pos(2) + closest_relic(2) + unit_energy(1) + step(1) 
        action_dim=6, # stay, up, right, down, left, sap
        hidden_layers_dim=[220, 220],
        buffer_max_len=10000,
        learning_rate=2.5e-4, # 0.0001
        gamma=0.99,
        epsilon=0.01,
        target_update_freq=100,
        dqn_type="DoubleDQN",
        epsilon_start=0.1,
        epsilon_decay_factor=0.995,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        env_cfg=env_cfg
    )
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent_dict[player] = DQN(
            player,
            env_cfg,
            state_dim=config.state_dim,
            hidden_layers_dim=config.hidden_layers_dim,
            action_dim=config.action_dim,
            max_len=config.buffer_max_len,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon=config.epsilon,
            target_update_freq=config.target_update_freq,
            dqn_type=config.dqn_type,
            epsilon_start=config.epsilon_start,
            epsilon_decay_factor=config.epsilon_decay_factor,
            device=config.device,
            random_flag=False,
            min_samples=config.min_samples
        )
        agent_dict[player].load_model(model_d_s1, player='player_0')
        agent_dict[player].eval()

    agent = agent_dict[player]
    actions = agent.policy(step, from_json(obs), remainingOverageTime)
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
    # pwd = os.popen('pwd').readlines()
    # ls = os.popen('ls').readlines()
    # print(f'{pwd=}\n{ls=}')
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


!python agent/baseTrain.py


!cp test_models/DQN_LuxAI_s1_dqn_vs_random_v1/DoubleDQN_model_player_0_rd_False.pth agent/
!luxai-s3 agent/main.py agent/main.py --output=replay.html


import IPython # load the HTML replay
IPython.display.HTML(filename='replay.html')


!cd agent && tar -czf submission.tar.gz *
!mv agent/submission.tar.gz .




