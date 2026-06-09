import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections

# Define a placeholder for the environment interaction
class Environment:
    def reset(self):
        print("Environment reset.")
        # Example of a more complex state representation (e.g., combining different info)
        state = torch.cat([torch.randn(1, 32), torch.zeros(1, 32)], dim=-1)
        return state

    def step(self, action):
        print(f"Executing action: {action}")
        next_state = torch.randn(1, 64) # Example next state (should be derived from action and current state)
        reward = torch.randn(1) # Example reward (should be based on action and state transition)
        done = random.random() < 0.1
        info = {}
        return next_state, reward, done, info

# Define the core components of the learning algorithm (Reinforcement Learning - Actor-Critic style)
class AIAgent(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(AIAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        action_probs = torch.softmax(self.policy_head(x), dim=-1)
        state_value = self.value_head(x)
        if action_probs.dim() == 3 and action_probs.shape[1] == 1:
             action_probs = action_probs.squeeze(1)
        return action_probs, state_value

# Experience tuple for replay buffer
Experience = collections.namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Implement GAE calculation
def calculate_gae(rewards, values, next_values, dones, gamma, gae_lambda):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return torch.stack(advantages)

# Calculate returns and advantages using GAE
def calculate_returns_and_advantages_gae(episode_experiences, gamma, gae_lambda):
    states, actions, rewards, next_states, dones = zip(*episode_experiences)

    states = torch.stack(states)
    actions = torch.tensor(actions)
    rewards = torch.stack(rewards).squeeze(-1)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    with torch.no_grad():
        _, values = agent(states)
        _, next_values = agent(next_states)

    values = values.squeeze(-1)
    next_values = next_values.squeeze(-1)

    advantages = calculate_gae(rewards, values, next_values, dones, gamma, gae_lambda)
    returns = advantages + values

    return returns.detach(), advantages.detach()

# Calculate losses
def calculate_losses(agent, states, actions, returns, advantages, entropy_weight=0.01):
    action_probs, state_values = agent(states)

    taken_action_prob = action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    taken_action_prob = torch.clamp(taken_action_prob, 1e-8, 1.0)
    log_prob = torch.log(taken_action_prob)

    policy_loss = -log_prob * advantages

    entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
    policy_loss = policy_loss - entropy_weight * entropy

    value_loss = nn.MSELoss()(state_values.squeeze(-1), returns)

    total_loss = policy_loss.mean() + 0.5 * value_loss

    return total_loss, policy_loss.mean(), value_loss, entropy.mean()

# Hyperparameters
input_dimension = 64
action_dimension = 10
num_episodes = 100
max_steps_per_episode = 200
gamma = 0.99
gae_lambda = 0.95
buffer_size = 10000 # Note: Buffer not used for training in this on-policy example
batch_size = 64 # Note: Batch size is effectively episode length in this on-policy example

# Instantiate environment and agent
env = Environment()
agent = AIAgent(input_dimension, action_dimension)

# Optimizer
optimizer = optim.Adam(agent.parameters(), lr=0.0005)

# Training Loop
print("Starting training...")
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    steps_in_episode = 0
    episode_experiences = []

    while not done and steps_in_episode < max_steps_per_episode:
        action_probs, state_value = agent(state.unsqueeze(0))
        action = torch.multinomial(action_probs, 1).squeeze(0).item()

        next_state, reward, done, info = env.step(action)

        episode_experiences.append(Experience(state, action, reward, next_state, done))

        state = next_state
        episode_reward += reward.item()
        steps_in_episode += 1

    if episode_experiences:
        episode_returns, episode_advantages = calculate_returns_and_advantages_gae(
            episode_experiences, gamma, gae_lambda
        )

        episode_states = torch.stack([e.state for e in episode_experiences])
        episode_actions = torch.tensor([e.action for e in episode_experiences])

        total_loss, policy_loss_mean, value_loss_mean, entropy_mean = calculate_losses(
            agent, episode_states, episode_actions, episode_returns, episode_advantages
        )

        optimizer.zero_grad()
        total_loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"Episode {episode + 1}/{num_episodes}, Steps: {steps_in_episode}, Total Reward: {episode_reward:.2f}")
        print(f"  Episode Update - Total Loss: {total_loss.item():.4f}, Policy Loss: {policy_loss_mean.item():.4f}, Value Loss: {value_loss_mean.item():.4f}, Entropy: {entropy_mean.item():.4f})")

print("Training finished.")

