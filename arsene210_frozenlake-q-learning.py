# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 1) Install dependencies (safe to re-run, idempotent)
import sys, subprocess
def pip_install(pkg):
    try:
        __import__(pkg.split("[")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip_install("gymnasium[toy_text]")
pip_install("numpy")
pip_install("pandas")


# 2) Imports: core libraries
import numpy as np
import pandas as pd
import gymnasium as gym
import time, os


# 3) Reproducibility: set random seed & RNG
SEED = 42
rng = np.random.default_rng(SEED)


# 4) Environment: FrozenLake 8x8 (slippery version)
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
n_states  = env.observation_space.n
n_actions = env.action_space.n


# 5) Hyperparameters, file paths, helpers
num_episodes   = 800_000
max_steps      = 300
alpha          = 0.10
gamma          = 0.99
eps_start      = 1.00
eps_end        = 0.01
eps_decay_frac = 0.95

CSV_SUB_PATH   = "/kaggle/working/rewards.csv" if os.path.exists("/kaggle") else "rewards.csv"
ART_QTABLE     = "/kaggle/working/frozenlake8x8_qtable.csv" if os.path.exists("/kaggle") else "frozenlake8x8_qtable.csv"
ART_RETURNS    = "/kaggle/working/frozenlake8x8_training_returns.csv" if os.path.exists("/kaggle") else "frozenlake8x8_training_returns.csv"

# Helpers for printing
def header(title, width=70):
    bar = "=" * width
    print(f"\n{bar}\n {title} \n{bar}")

def kv(label, value):
    print(f"- {label:<18}: {value}")

# Quick summary of environment & hyperparameters
header("Environment Info")
kv("Name", "FrozenLake-v1 (8x8)")
kv("Slippery", env.spec.kwargs.get("is_slippery"))
kv("State space", n_states)
kv("Action space", n_actions)

header("Hyperparameters")
kv("Episodes", f"{num_episodes:,}")
kv("Max steps/ep", max_steps)
kv("Alpha (lr)", alpha)
kv("Gamma (discount)", gamma)
kv("Epsilon start", eps_start)
kv("Epsilon end", eps_end)
kv("Epsilon decay frac", eps_decay_frac)


# 6) Epsilon schedule function (linear decay)
def epsilon_by_episode(ep):
    decay_episodes = int(num_episodes * eps_decay_frac)
    if ep >= decay_episodes:
        return eps_end
    return eps_end + (eps_start - eps_end) * (1 - ep / decay_episodes)


# 7) Initialize Q-table
Q = np.zeros((n_states, n_actions), dtype=np.float32)


# 8) Training loop with epsilon-greedy exploration
header("Training")
returns = []
t0 = time.time()
progress_step = max(50_000, num_episodes // 10)  # log ~10x

for ep in range(num_episodes):
    epsilon = epsilon_by_episode(ep)
    state, _ = env.reset(seed=SEED + ep)
    ep_return = 0.0

    for _ in range(max_steps):
        # choose action: random vs greedy
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning update rule
        best_next = np.max(Q[next_state])
        td_target = reward + gamma * best_next * (0.0 if done else 1.0)
        Q[state, action] += alpha * (td_target - Q[state, action])

        state = next_state
        ep_return += reward
        if done:
            break

    returns.append(ep_return)

    # progress log
    if (ep + 1) % progress_step == 0 or (ep + 1) == num_episodes:
        lastk = 1000 if len(returns) >= 1000 else len(returns)
        avg_lastk = float(np.mean(returns[-lastk:])) if lastk > 0 else 0.0
        kv("Progress", f"ep {ep+1:,}/{num_episodes:,}")
        kv("Epsilon now", f"{epsilon:.4f}")
        kv(f"Avg return (last {lastk})", f"{avg_lastk:.4f}")

train_time = time.time() - t0
kv("Training time", f"{train_time:.1f}s")


# 9) Evaluation: greedy policy only (no exploration)
def evaluate(env, Q, episodes=1000):
    total = 0.0
    for ep in range(episodes):
        s, _ = env.reset(seed=SEED + 100_000 + ep)
        for _ in range(max_steps):
            a = int(np.argmax(Q[s]))
            s, r, terminated, truncated, _ = env.step(a)
            total += r
            if terminated or truncated:
                break
    return total / episodes

header("Evaluation (Greedy)")
avg_reward = evaluate(env, Q, episodes=1000)
kv("Episodes", 1000)
kv("Average reward", f"{avg_reward:.6f}")
kv("Estimated AE (1 - avg_reward)", f"{abs(1.0 - avg_reward):.6f}")


# 10) Save Q-table and training returns
pd.DataFrame(Q, columns=[f"a{a}" for a in range(n_actions)]).rename_axis("state").to_csv(ART_QTABLE)
pd.DataFrame({"episode": np.arange(len(returns)), "return": returns}).to_csv(ART_RETURNS, index=False)

header("Artifacts")
kv("Q-table CSV", ART_QTABLE)
kv("Training returns CSV", ART_RETURNS)


# 11) Create Kaggle submission
sub = pd.DataFrame({
    "Id": ["FrozenLake8x8_public", "FrozenLake8x8_private"],
    "Predicted": [avg_reward, avg_reward],
})
sub.to_csv(CSV_SUB_PATH, index=False)

header("Submission")
kv("File", CSV_SUB_PATH)
kv("Predicted value", f"{avg_reward:.6f}")
print()

