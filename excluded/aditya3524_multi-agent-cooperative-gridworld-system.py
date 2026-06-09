import random
import numpy as np

class GridWorld:
    def __init__(self, size=8, n_resources=5, n_hazards=4):
        self.size = size
        self.grid = np.full((size, size), ".", dtype=str)

        # Place resources
        self.resources = []
        while len(self.resources) < n_resources:
            x, y = random.randrange(size), random.randrange(size)
            if self.grid[x][y] == ".":
                self.grid[x][y] = "R"
                self.resources.append((x, y))

        # Place hazards
        self.hazards = []
        while len(self.hazards) < n_hazards:
            x, y = random.randrange(size), random.randrange(size)
            if self.grid[x][y] == ".":
                self.grid[x][y] = "H"
                self.hazards.append((x, y))

        # Goal position
        self.goal = (size - 1, size - 1)
        self.grid[self.goal] = "G"

    def display(self):
        for row in self.grid:
            print(" ".join(row))
        print("\n")



import random

class ScoutAgent:
    def __init__(self, grid_size):
        self.x = 0
        self.y = 0
        self.visited = set()
        self.discovered_resources = []
        self.size = grid_size

    def move(self):
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(moves)

        for dx, dy in moves:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                self.x, self.y = nx, ny
                break

    def step(self, env):
        # Mark current position as visited
        self.visited.add((self.x, self.y))

        # Check for resource
        if env.grid[self.x][self.y] == "R":
            self.discovered_resources.append((self.x, self.y))

        # Make next move
        self.move()



class CarrierAgent:
    def __init__(self, grid_size):
        self.x = grid_size - 1
        self.y = 0
        self.carrying = False
        self.target = None
        self.size = grid_size

    def move_towards(self, tx, ty):
        if self.x < tx:
            self.x += 1
        elif self.x > tx:
            self.x -= 1
        elif self.y < ty:
            self.y += 1
        elif self.y > ty:
            self.y -= 1

    def step(self, env):
        # If carrier doesn't have a target, do nothing
        if self.target is None:
            return

        tx, ty = self.target
        self.move_towards(tx, ty)

        # --- Picking up resource ---
        if (self.x, self.y) == (tx, ty) and not self.carrying:
            self.carrying = True
            print(f"Picked up resource at {tx, ty}")
            env.grid[tx][ty] = "."   # Remove from grid
            self.target = env.goal   # New target = goal
            return

        # --- Delivering the resource ---
        if self.carrying:
            gx, gy = env.goal
            if (self.x, self.y) == (gx, gy):
                self.carrying = False
                self.target = None   # Reset for next task
                print("Delivered resource to goal! ğŸ�‰")



class CoordinatorAgent:
    def __init__(self):
        self.task_queue = []

    def collect_reports(self, scout):
        # Add newly discovered resources to task queue
        for r in scout.discovered_resources:
            if r not in self.task_queue:
                self.task_queue.append(r)

    def assign_task(self, carrier):
        # Assign next available resource to the carrier
        if carrier.target is None and self.task_queue:
            carrier.target = self.task_queue.pop(0)
            print(f"Assigned task to Carrier: {carrier.target}")



def ascii_map(env, scout, carrier):
    print("---- ASCII MAP ----")
    for i in range(env.size):
        line = ""
        for j in range(env.size):
            if (i, j) == (scout.x, scout.y):
                line += "S "
            elif (i, j) == (carrier.x, carrier.y):
                line += "C "
            else:
                line += env.grid[i][j] + " "
        print(line)
    print("\n")


def display_heatmap(scout, grid_size):
    heat = [[0]*grid_size for _ in range(grid_size)]
    for x, y in scout.visited:
        heat[x][y] += 1

    print("---- VISITED HEATMAP ----")
    for row in heat:
        print(" ".join(str(v) for v in row))
    print("\n")


def progress_bar(current, total, length=20):
    filled = int(length * (current / total))
    bar = "â–ˆ" * filled + "-" * (length - filled)
    print(f"[{bar}] {current}/{total}")


def pretty_summary(ep, scout, carrier, coordinator):
    print(f"\n--- SUMMARY EP {ep} ---")
    print(f"Scout: ({scout.x}, {scout.y})")
    print(f"Carrier: ({carrier.x}, {carrier.y}) | Carrying: {carrier.carrying}")
    print(f"Task Queue: {coordinator.task_queue}")
    print("------------------------\n")



env = GridWorld(size=8, n_resources=6, n_hazards=6)
scout = ScoutAgent(grid_size=env.size)
carrier = CarrierAgent(grid_size=env.size)
coordinator = CoordinatorAgent()

episodes = 25

for episode in range(episodes):
    print(f"\n===== EPISODE {episode+1} =====")
    progress_bar(episode + 1, episodes)

    scout.step(env)
    coordinator.collect_reports(scout)
    coordinator.assign_task(carrier)
    carrier.step(env)

    ascii_map(env, scout, carrier)

    if episode % 5 == 0:
        display_heatmap(scout, env.size)

    pretty_summary(episode + 1, scout, carrier, coordinator)

    # Mission completed
    if len(env.resources) == 0 and not carrier.carrying and not coordinator.task_queue:
        print("\nğŸ�¯ MISSION COMPLETED â€” All resources delivered! ğŸ�‰")
        break

print("\nSimulation Completed ğŸš€")


