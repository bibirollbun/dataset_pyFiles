import random
import time

class EcoEnvironment:
    def __init__(self):
        # State variables
        self.pollution_level = 50  # Starts at 50%
        self.tree_count = 10       # Starts with 10 trees
        self.max_steps = 20        # Simulation length
        self.current_step = 0
        
    def reset(self):
        """Resets the world to the starting state."""
        self.pollution_level = 50
        self.tree_count = 10
        self.current_step = 0
        print("\n--- ENVIRONMENT RESET ---\n")
        return self.get_state()

    def get_state(self):
        """Returns the current situation."""
        return (self.pollution_level, self.tree_count)

    def step(self, action):
        """
        The Agent takes an action. 
        Action 0: Do Nothing
        Action 1: Clean Pollution (Active removal)
        Action 2: Plant Tree (Passive removal)
        """
        self.current_step += 1
        message = ""

        # --- APPLY ACTION ---
        if action == 0:
            message = "Agent Idling..."
        elif action == 1:
            self.pollution_level -= 10
            message = "Agent Scrubbing Air (Pollution -10)"
        elif action == 2:
            self.tree_count += 5
            message = "Agent Planting Trees (Trees +5)"

        # --- NATURAL CONSEQUENCES ---
        # Pollution naturally rises every turn (industrial waste)
        natural_pollution_rise = 8 
        
        # Trees help absorb some pollution (1 unit per tree)
        nature_healing = self.tree_count * 0.5 
        
        # Calculate net change
        self.pollution_level += (natural_pollution_rise - nature_healing)
        
        # Clamp values (Pollution 0-100, Trees > 0)
        self.pollution_level = max(0, min(100, self.pollution_level))
        
        # --- CALCULATE REWARD ---
        # Reward is high if pollution is low
        reward = (100 - self.pollution_level) + (self.tree_count * 2)
        
        # Check if game is over
        done = self.current_step >= self.max_steps
        if self.pollution_level >= 100:
            done = True
            reward = -500  # Penalty for destroying the environment
            message += " -> CRITICAL FAILURE: POLLUTION MAXED OUT!"

        return self.get_state(), reward, done, message

class SmartAgent:
    """A simple rule-based agent."""
    def choose_action(self, state):
        pollution, trees = state
        
        # Simple Logic (The "Brain" of the Agent)
        if pollution > 70:
            return 1  # Emergency! Clean pollution immediately
        elif trees < 15:
            return 2  # Not enough nature, plant trees
        else:
            return 1  # Default to maintenance cleaning

# --- MAIN SIMULATION LOOP ---

if __name__ == "__main__":
    env = EcoEnvironment()
    agent = SmartAgent()
    
    state = env.reset()
    total_score = 0
    done = False
    
    print(f"{'Step':<5} | {'Pollution':<10} | {'Trees':<10} | {'Action Taken'}")
    print("-" * 60)

    while not done:
        # 1. Agent decides what to do based on current state
        action = agent.choose_action(state)
        
        # 2. Environment reacts to the action
        next_state, reward, done, info = env.step(action)
        
        # 3. Update score and state
        total_score += reward
        pollution, trees = next_state
        
        # 4. Display logic
        print(f"{env.current_step:<5} | {pollution:<10.1f} | {trees:<10} | {info}")
        state = next_state
        
        time.sleep(0.5) # Slow down to make it readable

    print("-" * 60)
    print(f"Simulation Ended. Final Score: {total_score:.2f}")
    if pollution >= 100:
        print("Result: The environment collapsed.")
    else:
        print("Result: The environment was sustained successfully!")

