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


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from   matplotlib import colors
import seaborn as sns

import json
import os
from pathlib import Path
from glob import glob

from subprocess import Popen, PIPE, STDOUT


base_path='/kaggle/input/arc-prize-2025/'

# Loading JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


training_challenges   = load_json(base_path +'arc-agi_training_challenges.json')
training_solutions    = load_json(base_path +'arc-agi_training_solutions.json')

evaluation_challenges = load_json(base_path +'arc-agi_evaluation_challenges.json')
evaluation_solutions  = load_json(base_path +'arc-agi_evaluation_solutions.json')


# 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

plt.figure(figsize=(3, 1), dpi=150)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()


def plot_task(task, task_solutions, i, t):
    """    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app    """    
    fs=12    
    num_train = len(task['train'])
    #num_test  = len(task['test'])
    num_test  = 1
    
    w=num_train+num_test
    fig, axs  = plt.subplots(2, w, figsize=(2*w,2*2))
    #fig, axs  = plt.subplots(2, w, figsize=(1.5*w, 1.5*2))
    plt.suptitle(f'Set #{i}, {t}:', fontsize=fs, fontweight='bold', y=1)
    #plt.subplots_adjust(hspace = 0.15)
    #plt.subplots_adjust(wspace=20, hspace=20)
    
    for j in range(num_train):     
        plot_one(axs[0, j], j,task, 'train', 'input')
        plot_one(axs[1, j], j,task,'train', 'output')        
    
    plot_one(axs[0, j+1], 0, task, 'test', 'input')

    answer = task_solutions
    input_matrix = answer
    
    axs[1, j+1].imshow(input_matrix, cmap=cmap, norm=norm)
    axs[1, j+1].grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    axs[1, j+1].set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    axs[1, j+1].set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    axs[1, j+1].set_xticklabels([])
    axs[1, j+1].set_yticklabels([])
    axs[1, j+1].set_title('Test output')

    axs[1, j+1] = plt.figure(1).add_subplot(111)
    axs[1, j+1].set_xlim([0, num_train+1])
    
    for m in range(1, num_train):
        axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color = 'black')
    
    axs[1, j+1].plot([num_train,num_train],[0,1],'-', linewidth=3, color = 'black')

    axs[1, j+1].axis("off")

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black') 
    fig.patch.set_facecolor('#dddddd')
   
    plt.tight_layout()
    
    print(f'#{i}, {t}') # for fast and convinience search
    plt.show()  
    
    print()
    #print()
    
    
def plot_one(ax, i, task, train_or_test, input_or_output):
    fs=12 
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    
    ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs-2)


for i in range(0, 100):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t][0]
    plot_task(task,  task_solution, i, t)


for i in range(100, 200):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t][0]
    plot_task(task,  task_solution, i, t)


for i in range(200, 300):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t][0]
    plot_task(task,  task_solution, i, t)


for i in range(300, 400):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t][0]
    plot_task(task,  task_solution, i, t)


for i in range(400, 500):
    t=list(training_challenges)[i]
    task=training_challenges[t]
    task_solution = training_solutions[t][0]
    plot_task(task,  task_solution, i, t)


for i in range(0, 120):
    t=list(evaluation_challenges)[i]
    task=evaluation_challenges[t]
    task_solution = evaluation_solutions[t][0]
    plot_task(task,  task_solution, i, t)


import json

# Exemplo: função para prever com base em um modelo treinado
def generate_predictions(task_id, model):
    # Lógica para gerar previsões reais usando o modelo
    prediction_1 = model.predict(task_id, attempt=1)  # Previsão tentativa 1
    prediction_2 = model.predict(task_id, attempt=2)  # Previsão tentativa 2
    return {"attempt_1": prediction_1.tolist(), "attempt_2": prediction_2.tolist()}

# Carregar os dados de entrada
input_file_path = '/kaggle/input/arc-prize-2025/sample_submission.json'
with open(input_file_path, 'r') as f:
    input_data = json.load(f)

# Inicializar o modelo (exemplo)
class HypotheticalModel:
    def predict(self, task_id, attempt):
        # Simulação de uma previsão; substitua pela lógica real
        return np.random.rand(2, 2)

model = HypotheticalModel()

# Criar submissão
submission = {}
for task_id in input_data.keys():
    predictions = generate_predictions(task_id, model)
    submission[task_id] = [predictions]

# Salvar submissão
output_file_path = 'submission.json'
with open(output_file_path, 'w') as f:
    json.dump(submission, f, indent=4)

print(f"Submissão criada e salva como {output_file_path}")


import json
import pandas as pd
import numpy as np

# Carregar o arquivo de entrada do conjunto de avaliação
input_file_path = '/kaggle/input/arc-prize-2025/sample_submission.json'

# Função para gerar previsões aleatórias como exemplo. Substitua pela lógica real de previsão.
def generate_predictions(task_id):
    # Exemplo: Previsão baseada em alguma lógica (use seus modelos ou heurísticas aqui)
    prediction_1 = np.random.rand(2, 2).tolist()  # Previsão tentativa 1
    prediction_2 = np.random.rand(2, 2).tolist()  # Previsão tentativa 2
    return {"attempt_1": prediction_1, "attempt_2": prediction_2}

# Carregar os IDs das tarefas do arquivo de entrada
with open(input_file_path, 'r') as f:
    input_data = json.load(f)

# Criar estrutura de submissão
submission = {}

for task_id in input_data.keys():
    # Gerar previsões para cada task_id
    predictions = generate_predictions(task_id)
    submission[task_id] = [predictions]

# Salvar o arquivo de saída no formato JSON
output_file_path = 'submission.json'
with open(output_file_path, 'w') as f:
    json.dump(submission, f, indent=4)

print(f'Submissão criada e salva como {output_file_path}')

