# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!mkdir agent && cp -r ../input/lux-s3/lux-ai-season-3/* agent/
import sys
sys.path.insert(1, 'agent')


!pip install --upgrade luxai_s3


import sys
sys.path.append('/kaggle/input/lux-s3/lux-ai-season-3/')


class Actor(nn.Module):
    def __init__(self,inp,out):
        super(Actor,self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(inp, 256),
            nn.LayerNorm(256),  
            nn.ReLU(),
            nn.Dropout(0.1), 
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            
            nn.Linear(64, out),
        )
        for layer in self.fc_layer:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
        self.fc_layer[-1].bias.data[0] = -0.7 
        self.fc_layer[-1].bias.data[1:] = 0.1
    def forward(self,x):
        x = x.unsqueeze(0)
        out=self.fc_layer(x).squeeze()
        temperature = 1.5  
        return F.softmax(out / temperature, dim=-1)


class Value(nn.Module):
    def __init__(self,inp,out,hid,num_layer=3):
        super(Value,self).__init__()
        self.fc_layer=nn.Sequential(
            nn.Linear(inp, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),   
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            
            nn.Linear(64, out),)
                        
    def forward(self,x):
        x = x.unsqueeze(0)
        out=self.fc_layer(x)
        return out
        


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


from luxai_s3.wrappers import LuxAIS3GymEnv
def train(epoch=10,batch=50,seed=np.random.seed()):
    replay_save_dir="replays"
    env = RecordEpisode(LuxAIS3GymEnv(numpy_output=True), save_on_reset=True, save_dir=replay_save_dir)
    obs, info = env.reset(seed=seed)
    env_cfg = info["params"]
    agent0=Agent("player_0",env_cfg)
    agent1=Agent("player_1",env_cfg) 
    action={}
    training=True
    for i in range(epoch):
        print('epoch :',i)
        obs, info = env.reset()
        done=False
        step=0
        game_done=False
        last_action={}
        last_last_action={}
        for agent in [agent0,agent1]:
            agent.reset_episode()
            last_action[agent.player]=agent.last_action
            last_last_action[agent.player]=last_action[agent.player]
        k=0
        last_obs=None
        last_rew = {
                "player_0": 0,
                "player_1": 0
            } 
        while (not game_done) and k<batch:
            # print(f"k : {k}")
            last_obs = {
                        "player_0": obs["player_0"].copy(),
                        "player_1": obs["player_1"].copy()
            }
            prob={}
            val={}
            for agent in [agent0,agent1]:
                action[agent.player], prob[agent.player],val[agent.player]=agent.t_act(step,obs[agent.player],epoch,i)
                
            obs, reward, terminated, truncated, info=env.step(action)
            
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rew = {
                "player_0": obs["player_0"]["team_points"][agent0.team_id],
                "player_1": obs["player_1"]["team_points"][agent1.team_id]
            } 
            pen={}
            pen["player_0"]=4*(rew["player_0"]- last_rew["player_0"]) if rew["player_0"]- last_rew["player_0"]!=0 else -0.05
            pen["player_1"]=4*(rew["player_1"]- last_rew["player_1"]) if rew["player_1"]- last_rew["player_1"]!=0 else -0.01
            rewards={
                "player_0": pen["player_0"],
                "player_1": pen["player_1"]
            }
            last_rew={
                "player_0": rew["player_0"],
                "player_1": rew["player_1"]
            } 
            if training:
                for agent in [agent0,agent1]:
                  
                    for unit_id in range(env_cfg['max_units']):
                        if obs[agent.player]['units_mask'][agent.team_id][unit_id]:
                            if np.array_equal(obs[agent.player]["units"]["position"][agent.team_id][unit_id], [-1, -1]):
                                print(obs[agent.player]["units"]["position"][agent.team_id][unit_id])
                            act=action[agent.player][unit_id][0]
                            last_state=agent.rep(last_obs[agent.player]["units"]["position"][agent.team_id][unit_id],last_obs[agent.player]["relic_nodes"],last_obs[agent.player]["units"]["energy"][agent.team_id][unit_id],step,last_obs[agent.player]["relic_nodes_mask"],last_last_action[agent.player][unit_id])
                            state=agent.rep(obs[agent.player]["units"]["position"][agent.team_id][unit_id],obs[agent.player]["relic_nodes"],obs[agent.player]["units"]["energy"][agent.team_id][unit_id],step,obs[agent.player]["relic_nodes_mask"],last_action[agent.player][unit_id])
                            rewards[agent.player] += 3 * (1.0 / ((state[6]) + 1.0))                            
                            pos_tuple = tuple(obs[agent.player]["units"]["position"][agent.team_id][unit_id])
                            if pos_tuple not in agent.visited_positions:
                                rewards[agent.player] += 0.5
                                agent.visited_positions.add(pos_tuple)
                            if last_state[0]==state[0] and last_state[1]==state[1] and pen[agent.player]<=0:
                                rewards[agent.player]-=0.07
                            last_last_action[agent.player]=last_action[agent.player]
                            last_action[agent.player]=agent.last_action
                            agent.traj.append((last_state,act,rewards[agent.player],state,prob[agent.player][unit_id],done,val[agent.player][unit_id]))
                            
            step+=1
            k+=1              
            if dones["player_0"] or dones["player_1"]:
                    game_done = True
            if training:
                agent0.save_model()
                agent1.save_model()
        agent0.teach()
        agent1.teach()
        if((i+1)%10==0):
            render_episode(env)
    agent0.plot_entropy()
    env.close()
    return agent0


from lux.utils import direction_to
import sys
import numpy as np
class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.seed=np.random.seed(0)
        self.env_cfg = env_cfg
        self.batch_size=10
        self.gamma=0.99
        self.clip_param = 0.2
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.entropy_history=[]
        self.visited_positions = set()
        self.unit_explore_locations = dict()
        self.explore_loc_update=np.zeros(self.env_cfg["max_units"], dtype=float)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Actor(9, 5).to(self.device)
        self.value = Value(9, 1, 5, 5).to(self.device)
        self.policy_opti = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.value_opti = torch.optim.Adam(self.value.parameters(), lr=1e-4)
        self.traj=[]
        self.last_action=np.zeros(self.env_cfg["max_units"], dtype=int)

    def reset_episode(self):
        self.relic_node_positions = []
        self.last_action=np.zeros(self.env_cfg["max_units"], dtype=int)
        self.discovered_relic_nodes_ids = set()
        self.visited_positions = set()
        self.unit_explore_locations = dict()
        self.explore_loc_update=np.zeros(self.env_cfg["max_units"], dtype=float)
    def save_model(self):
        torch.save({
            'policy_net': self.policy.state_dict(),
            'policy_opti': self.policy_opti.state_dict()
        }, f'dqn_model_{self.player}.pth')
        
    def rep(self,unit_pos,relic_nodes,unit_energy,step,relic_mask,last_act):
        visible=relic_nodes[relic_mask]
        direc=-1
        dist=1000
        if relic_mask.any():
            dist=np.linalg.norm(visible-unit_pos,axis=1)
            closest=visible[np.argmin(dist)]

            dist=dist.min()
            direc=direction_to(unit_pos, closest)
        else:
            closest=np.array([-1,-1])

        return torch.FloatTensor(np.concatenate([unit_pos,closest,[direc],[unit_energy/100],[dist],[last_act],[step/505]])).to(self.device)
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):
   
        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units,)
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id])  # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"])  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])  # shape (max_relic_nodes,)
        team_points = np.array(obs["team_points"])  # Points scored by each team
    
        # Ids of controllable units at this timestep
        available_unit_ids = np.where(unit_mask)[0]
    
        # Visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
    
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
    
        # Save new relic nodes discovered
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
                
        with torch.no_grad():
            for units_id in available_unit_ids:
                unit_pos=unit_positions[units_id]
                state_rep= self.rep(unit_pos,observed_relic_node_positions,unit_energys[units_id],step,observed_relic_nodes_mask)
                log_prob=self.policy.forward(state_rep)
                act=torch.argmax((log_prob))
                actions[units_id]=[act,0,0]
        return actions
        
    def compute_intrinsic_reward(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            value_pred = self.value(state_tensor)
        return 0.01 * torch.abs(value_pred).item() 

    def plot_entropy(self):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.entropy_history)
        plt.title('Policy Entropy over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Entropy')
        plt.grid(True)
        
        # Add horizontal line at maximum possible entropy for 5 actions
        max_entropy = -np.log(1/5)  # ≈ 1.61
        plt.axhline(y=max_entropy, color='r', linestyle='--', 
                    label=f'Max Entropy ({max_entropy:.2f})')
        
        plt.legend()
        plt.show()
        
    def get_valid_actions(self, unit_pos, unit_energy, state_rep):
        valid = np.ones(5)  

        if state_rep[2]==-1 and state_rep[3]==-1:
            valid[0] = 0.7 
        if unit_pos[1] <= 0: 
            valid[1] = 0
        if unit_pos[0] >= self.env_cfg["map_width"]-1: 
            valid[2] = 0
        if unit_pos[1] >= self.env_cfg["map_height"]-1:  
            valid[3] = 0
        if unit_pos[0] <= 0: 
            valid[4] = 0
        
        return torch.FloatTensor(valid).to(self.device)
    
    def t_act(self, step: int, obs,ep, curr, remainingOverageTime: int = 60,):
   
        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units,)
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id])  # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"])  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])  # shape (max_relic_nodes,)
        team_points = np.array(obs["team_points"])  # Points scored by each team
    
        # Ids of controllable units at this timestep
        available_unit_ids = np.where(unit_mask)[0]
    
        # Visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
    
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        prob = np.zeros(self.env_cfg["max_units"], dtype=float)
        val_col = np.zeros(self.env_cfg["max_units"], dtype=float)
    
        # Save new relic nodes discovered
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
           
        with torch.no_grad():

            for units_id in available_unit_ids:
                unit_pos=unit_positions[units_id]
                state_rep= self.rep(unit_pos,observed_relic_node_positions,unit_energys[units_id],step/505.0,observed_relic_nodes_mask,self.last_action[units_id])
                valid_actions=self.get_valid_actions(unit_pos,unit_energys[units_id],state_rep)
                log_prob=self.policy.forward(state_rep).squeeze()
                val=self.value.forward(state_rep)
                if (np.random.random() < max(0,0)) and curr!=49:
                    if len(self.relic_node_positions) > 0:
                        nearest_relic_node = self.relic_node_positions[0]
                        distance_to_relic = abs(unit_pos[0] - nearest_relic_node[0]) + abs(unit_pos[1] - nearest_relic_node[1])
                        if distance_to_relic <= 4:
                            random_direction = np.random.randint(0, 5)
                            act = torch.tensor(random_direction)
                        else:
                            act= torch.tensor( direction_to(unit_pos, nearest_relic_node))
                    else:
                        if self.explore_loc_update[units_id]==0 or step-self.explore_loc_update[units_id] >=20:
                            random_loc=(np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                            self.unit_explore_locations[units_id]=random_loc
                        act=torch.tensor(direction_to(unit_pos, self.unit_explore_locations[units_id]))
                else:
                    act = torch.argmax(log_prob)
                self.last_action[units_id]=act.cpu().item()
                actions[units_id]=[act.cpu().item(),0,0]
                prob[units_id]=(log_prob[act])
                val_col[units_id]=val.item()
        return actions, prob , val_col 

    def teach(self):
        T=len(self.traj)
        value_col=np.zeros(T)
        return_col=np.zeros(T)
        prob_col=np.zeros(T)
        reward_col=np.zeros(T)
        last_state=0
        states=[]
        actions=[]
        i=0
        for e in self.traj:
            state,act,reward,next_state,log_prob,done,val=e
            value_col[i]=(val)
            states.append(state)
            actions.append(act)
            entropy_reward=self.compute_intrinsic_reward(state)
            prob_col[i]=(log_prob)
            reward_col[i]=(reward)+(0.01*entropy_reward)
            last_state=next_state
            i+=1
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        return_col[-1]=self.value.forward(self.traj[-1][3]).item()
        print(reward_col)
        for t in reversed(range(T-1)):
            return_col[t]= reward_col[t]+ self.gamma*return_col[t+1]
        return_col = (return_col - return_col.mean()) / (return_col.std() + 1e-8)
        adv_col=return_col-value_col
        adv_col = (adv_col - adv_col.mean()) / (adv_col.std() + 1e-8)
        adv_col = torch.FloatTensor(adv_col).to(self.device)
        return_col = torch.FloatTensor(return_col).to(self.device)
        for _ in range(30):
            new_prob=(torch.clamp(self.policy.forward(states), min=1e-8)).to(self.device)
            prob_col=torch.clamp(torch.FloatTensor(prob_col).to(self.device),min=1e-8)
            entropy = (new_prob * torch.log(new_prob + 1e-8)).sum(dim=-1)
            mean_entropy = entropy.mean().item()
            entropy_bonus = -0.08*mean_entropy
            
            if not hasattr(self, 'entropy_history'):
                self.entropy_history = []
            self.entropy_history.append(mean_entropy)
            
            new_prob= new_prob.gather(1,actions.unsqueeze(1)).squeeze()
            ratio = (new_prob/prob_col)
            clipped_ratio = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
            policy_loss=-torch.min((ratio * adv_col),clipped_ratio* adv_col).mean() +entropy_bonus
            value_loss= nn.MSELoss()(self.value(states).squeeze().to(self.device),return_col)
            self.policy_opti.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_opti.step()
            self.value_opti.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.value_opti.step()
        print(f"policy {self.player}: {policy_loss}  value {self.player}: {value_loss}")
        self.traj.clear()
            


ag=train(500,505)


ag.plot_entropy()


!cd agent && tar -czf submission3.tar.gz *
!mv agent/submission3.tar.gz .


import shutil
shutil.rmtree("/kaggle/working/replays")




