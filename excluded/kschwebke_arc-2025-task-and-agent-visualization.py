import itertools
import json

import torch

import matplotlib.pyplot as plt
from   matplotlib import colors


input_path = "/kaggle/input/arc-prize-2025"


# data handling and visualization based based on
# https://www.kaggle.com/code/mehmetakifciftci/breaking-arc-prize-the-first-50-solution

class ARCDataset:
    def __init__(self, train_path=None, train_solutions_path=None, 
                       test_path=None, 
                       eval_path=None, eval_solutions_path=None):
        self.train_data = self._load_json(train_path) if train_path else {}
        self.train_solutions = self._load_json(train_solutions_path) if train_solutions_path else {}
        self.test_data = self._load_json(test_path) if test_path else {}
        self.eval_data = self._load_json(eval_path) if eval_path else {}
        self.eval_solutions = self._load_json(eval_solutions_path) if eval_solutions_path else {}

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
        
    def get_task(self, task_id, split='train'):
        if split == 'train':
            return self.train_data.get(task_id), self.train_solutions.get(task_id)
        elif split == 'test':
            return self.test_data.get(task_id), None
        elif split == 'eval':
            return self.eval_data.get(task_id), self.eval_solutions.get(task_id)
        else:
            raise ValueError("split must be 'train', 'test', or 'eval'")


ARC_COLORMAP = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF']
)
ARC_NORM = colors.Normalize(vmin=0, vmax=10)

DELTA_COLORMAP = 'seismic'  # or try 'coolwarm', 'RdBu', or any other diverging cmap
DELTA_NORM = colors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)


def compare_grids(a, b):
    a_t = torch.as_tensor(a)
    b_t = torch.as_tensor(b)
    if a_t.shape != b_t.shape:
        return None
    d = a_t - b_t
    delta = d.abs().sum().item()
    delta_count = d.abs().sign().sum().item()
    return delta, delta_count


def plot_grid(axs, grid, title, delta=False):
    if delta:
        axs.imshow(grid, cmap=DELTA_COLORMAP, norm=DELTA_NORM)
    else:
        axs.imshow(grid, cmap=ARC_COLORMAP, norm=ARC_NORM)

    # add grid, based on https://www.kaggle.com/code/allegich/arc-agi-2025-visualization-all-1000-120-tasks
    axs.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    axs.set_xticks([x-0.5 for x in range(1 + len(grid[0]))])     
    axs.set_yticks([x-0.5 for x in range(1 + len(grid))])
    
    axs.set_title(title)


def visualize_task(task_data, *, task_solution=None, agent_output=None, title="ARC Task", figsize=(12, 8)):
    train_examples = task_data.get('train', [])
    test_examples = task_data.get('test', [])
    has_solution = task_solution is not None
    has_agent_output = agent_output is not None

    num_train = len(train_examples)
    num_test = len(test_examples)
    cols = num_train + num_test
    rows = 2

    if has_agent_output:
        rows += 2

    if not has_solution:
        task_solution = []

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    plt.suptitle(title, fontsize=16)

    for idx, example in enumerate(train_examples):
        plot_grid(axs[0, idx], example['input'], "Train Input")
        plot_grid(axs[1, idx], example['output'], "Train Output")

    for idx, example in enumerate(test_examples):
        plot_grid(axs[0, num_train + idx], example['input'], "Test Input")

        if has_solution:
            plot_grid(axs[1, num_train + idx], task_solution[idx], "Test Output")
        else:
            axs[1, num_train + idx].set_title("Test Output: ?")
            axs[1, num_train + idx].axis('off')

    if has_agent_output:
        for idx, output in enumerate(agent_output["train"]):
            plot_grid(axs[2, idx], output, "Agent Output")

        for idx, output in enumerate(agent_output["test"]):
            plot_grid(axs[2, num_train+idx], output, "Agent Output")

        for idx, (example, output) in enumerate(zip(train_examples, agent_output["train"])):
            example_t = torch.as_tensor(example["output"])
            output_t = torch.as_tensor(output)
            if example_t.shape == output_t.shape:
                delta = compare_grids(example_t, output_t)
                delta_grid = (example_t - output_t).tolist()
                plot_grid(axs[3, idx], delta_grid, f"Agent Δ: {delta}", delta=True)
            else:
                axs[3, idx].set_title("Agent shape mismatch")
                axs[3, idx].axis('off')

        for idx, (solution, output) in enumerate(itertools.zip_longest(task_solution, agent_output["test"])):
            if has_solution:
                solution_t = torch.as_tensor(solution)
                output_t = torch.as_tensor(output)
                if solution_t.shape == output_t.shape:
                    delta = compare_grids(solution_t, output_t)
                    delta_grid = (solution_t - output_t).tolist()
                    plot_grid(axs[3, num_train+idx], delta_grid, f"Agent Δ: {delta}", delta=True)
                else:
                    axs[3, num_train + idx].set_title("Agent shape mismatch")
                    axs[3, num_train + idx].axis('off')
            else:
                axs[3, num_train + idx].set_title("Agent Δ: ?")
                axs[3, num_train + idx].axis('off')

    plt.tight_layout()
    plt.show()


dataset = ARCDataset(
    train_path=f'{input_path}/arc-agi_training_challenges.json',
    train_solutions_path=f'{input_path}/arc-agi_training_solutions.json',
    test_path=f'{input_path}/arc-agi_test_challenges.json',
    eval_path=f'{input_path}/arc-agi_evaluation_challenges.json',
    eval_solutions_path=f'{input_path}/arc-agi_evaluation_solutions.json',
)


task_data, task_solution = dataset.get_task('00576224', split='train')
visualize_task(task_data, task_solution=task_solution, title='Task 00576224')


task_data


task_solution


# perfect answer from solution
agent_output_train = [d['output'] for d in task_data['train']]
agent_output_test  = task_solution
agent_output = {
    "train": agent_output_train,
    "test": agent_output_test,
}
agent_output


visualize_task(task_data, task_solution=task_solution, title='Task 00576224', agent_output=agent_output)


# random answer
out_shape = torch.as_tensor(task_data["train"][0]["output"]).shape
agent_output_train = torch.randint(0, 9, (len(task_data["train"]), out_shape[0], out_shape[1])).tolist()
agent_output_test  = torch.randint(0, 9, torch.as_tensor(task_solution).shape).tolist()
agent_output = {
    "train": agent_output_train,
    "test": agent_output_test,
}
agent_output


visualize_task(task_data, task_solution=task_solution, title='Task 00576224', agent_output=agent_output)


# random answer and wrong shape
out_shape = torch.as_tensor(task_data["train"][0]["output"]).shape
agent_output_train = torch.randint(0, 9, (len(task_data["train"]), out_shape[0]+1, out_shape[1]+1)).tolist()
solution_shape = torch.as_tensor(task_solution).shape
agent_output_test  = torch.randint(0, 9, (solution_shape[0], solution_shape[1]+1, solution_shape[2]+1)).tolist()
agent_output = {
    "train": agent_output_train,
    "test": agent_output_test,
}
agent_output


visualize_task(task_data, task_solution=task_solution, title='Task 00576224', agent_output=agent_output)




