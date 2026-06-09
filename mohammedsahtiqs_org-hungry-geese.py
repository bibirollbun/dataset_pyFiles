!pip install kaggle-environments --quiet


import random
_real_sample = random.sample

def fixed_sample(population, k):
    if not isinstance(population, (list, tuple, str)):
        population = list(population)
    return _real_sample(population, k)

random.sample = fixed_sample


from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
import random


def my_agent(obs, config):
    return random.choice(list(Action)).name

# Create environment and run match
env = make("hungry_geese", debug=True)
env.run([my_agent, "greedy"])

# Show animation
env.render(mode="ipython", width=800, height=600)

# Save agent for submission
with open("/kaggle/working/submission.py", "w") as f:
    f.write("""
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
import random

def agent(obs, config):
    return random.choice(list(Action)).name
""")

