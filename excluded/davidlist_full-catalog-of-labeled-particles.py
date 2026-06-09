!pip install zarr


import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import zarr

from glob import glob
from scipy import ndimage

pio.renderers.default = 'iframe'

ENABLE_3D = False


TRAIN_BASE = '/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns'
TEST_BASE = '/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns'
OVERLAY_BASE = '/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns'
TRAINING_SETS = ['TS_86_3', 'TS_6_6', 'TS_73_6', 'TS_99_9', 'TS_69_2', 'TS_6_4', 'TS_5_4']
PARTICLE_RADIUS = {
    'apo-ferritin': 60.0,
    'beta-amylase': 65.0,
    'beta-galactosidase': 90.0,
    'ribosome': 150.0,
    'thyroglobulin': 130.0,
    'virus-like-particle': 135.0,
}

def get_training_experiments():
    return os.listdir(TRAIN_BASE)
    
def get_test_experiments():
    return os.listdir(TEST_BASE)

def get_experiment_path(experiment, type='denoised', test=False):
    base = TEST_BASE if test else TRAIN_BASE
    return base + '/' + experiment + '/VoxelSpacing10.000/' + type + '.zarr'

def open_experiment(experiment, test=False):
    return zarr.open(get_experiment_path(experiment, test=test), mode='r')

def load_points(experiment, particle='ribosome'):
    results = []
    path = OVERLAY_BASE + '/' + experiment + '/Picks/' + particle + '.json'
    with open(path) as f:
        points = json.loads(f.read())['points']
        for p in points:
            x = float(p['location']['x'])
            y = float(p['location']['y'])
            z = float(p['location']['z'])
            results.append((x, y, z))
    return results

def normalize(dataset):
    return (dataset - np.mean(dataset))/np.std(dataset)

def plot_particles(experiment, particle):
    dataset = normalize(open_experiment(experiment)[0])
    particles = sorted(load_points(experiment, particle=particle), key=lambda p: p[2])
    
    plt.figure(figsize=(12, 4*(len(particles)//3 + 1)))
    for idx in range(len(particles)):
        i = int(np.rint(particles[idx][0]/10.012))
        j = int(np.rint(particles[idx][1]/10.012))
        k = int(np.rint(particles[idx][2]/10.012))

        # Fix display problem near the edges.
        i_min = i - 24 if i - 24 > 0 else 0
        j_min = j - 24 if j - 24 > 0 else 0
        
        plt.subplot(len(particles)//3 + 1, 3, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(idx) + ': ' + str((k,j,i,)))
        plt.imshow(dataset[k, (j_min):(j+25), (i_min):(i+25)], cmap='gray')

def plot_particles_3d(experiment, particle):
    if ENABLE_3D:
        particles = load_points(experiment, particle=particle)
        particle_list = []
        for p in particles:
            particle_list.append({'x':np.rint(p[0]/10.012), 'y':np.rint(p[1]/10.012),
                                  'z':np.rint(p[2]/10.012), 'size': 1})
            
        fig = px.scatter_3d(particle_list, x='x', y='y', z='z', size='size',
                            range_x=[630,0], range_y=[0,630], range_z=[184,0],
                            title=experiment + ': ' + particle,
                            template='seaborn')
        fig.show()


plot_particles_3d('TS_5_4', 'apo-ferritin')


plot_particles('TS_5_4', 'apo-ferritin')


plot_particles_3d('TS_69_2', 'apo-ferritin')


plot_particles('TS_69_2', 'apo-ferritin')


plot_particles_3d('TS_6_4', 'apo-ferritin')


plot_particles('TS_6_4', 'apo-ferritin')


plot_particles_3d('TS_6_6', 'apo-ferritin')


plot_particles('TS_6_6', 'apo-ferritin')


plot_particles_3d('TS_73_6', 'apo-ferritin')


plot_particles('TS_73_6', 'apo-ferritin')


plot_particles_3d('TS_86_3', 'apo-ferritin')


plot_particles('TS_86_3', 'apo-ferritin')


plot_particles_3d('TS_99_9', 'apo-ferritin')


plot_particles('TS_99_9', 'apo-ferritin')


plot_particles_3d('TS_5_4', 'beta-amylase')


plot_particles('TS_5_4', 'beta-amylase')


plot_particles_3d('TS_69_2', 'beta-amylase')


plot_particles('TS_69_2', 'beta-amylase')


plot_particles_3d('TS_6_4', 'beta-amylase')


plot_particles('TS_6_4', 'beta-amylase')


plot_particles_3d('TS_6_6', 'beta-amylase')


plot_particles('TS_6_6', 'beta-amylase')


plot_particles_3d('TS_73_6', 'beta-amylase')


plot_particles('TS_73_6', 'beta-amylase')


plot_particles_3d('TS_86_3', 'beta-amylase')


plot_particles('TS_86_3', 'beta-amylase')


plot_particles_3d('TS_99_9', 'beta-amylase')


plot_particles('TS_99_9', 'beta-amylase')


plot_particles_3d('TS_5_4', 'beta-galactosidase')


plot_particles('TS_5_4', 'beta-galactosidase')


plot_particles_3d('TS_69_2', 'beta-galactosidase')


plot_particles('TS_69_2', 'beta-galactosidase')


plot_particles_3d('TS_6_4', 'beta-galactosidase')


plot_particles('TS_6_4', 'beta-galactosidase')


plot_particles_3d('TS_6_6', 'beta-galactosidase')


plot_particles('TS_6_6', 'beta-galactosidase')


plot_particles_3d('TS_73_6', 'beta-galactosidase')


plot_particles('TS_73_6', 'beta-galactosidase')


plot_particles_3d('TS_86_3', 'beta-galactosidase')


plot_particles('TS_86_3', 'beta-galactosidase')


plot_particles_3d('TS_99_9', 'beta-galactosidase')


plot_particles('TS_99_9', 'beta-galactosidase')


plot_particles_3d('TS_5_4', 'ribosome')


plot_particles('TS_5_4', 'ribosome')


plot_particles_3d('TS_69_2', 'ribosome')


plot_particles('TS_69_2', 'ribosome')


plot_particles_3d('TS_6_4', 'ribosome')


plot_particles('TS_6_4', 'ribosome')


plot_particles_3d('TS_6_6', 'ribosome')


plot_particles('TS_6_6', 'ribosome')


plot_particles_3d('TS_73_6', 'ribosome')


plot_particles('TS_73_6', 'ribosome')


plot_particles_3d('TS_86_3', 'ribosome')


plot_particles('TS_86_3', 'ribosome')


plot_particles_3d('TS_99_9', 'ribosome')


plot_particles('TS_99_9', 'ribosome')


plot_particles_3d('TS_5_4', 'thyroglobulin')


plot_particles('TS_5_4', 'thyroglobulin')


plot_particles_3d('TS_69_2', 'thyroglobulin')


plot_particles('TS_69_2', 'thyroglobulin')


plot_particles_3d('TS_6_4', 'thyroglobulin')


plot_particles('TS_6_4', 'thyroglobulin')


plot_particles_3d('TS_6_6', 'thyroglobulin')


plot_particles('TS_6_6', 'thyroglobulin')


plot_particles_3d('TS_73_6', 'thyroglobulin')


plot_particles('TS_73_6', 'thyroglobulin')


plot_particles_3d('TS_86_3', 'thyroglobulin')


plot_particles('TS_86_3', 'thyroglobulin')


plot_particles_3d('TS_99_9', 'thyroglobulin')


plot_particles('TS_99_9', 'thyroglobulin')


plot_particles_3d('TS_5_4', 'virus-like-particle')


plot_particles('TS_5_4', 'virus-like-particle')


plot_particles_3d('TS_69_2', 'virus-like-particle')


plot_particles('TS_69_2', 'virus-like-particle')


plot_particles_3d('TS_6_4', 'virus-like-particle')


plot_particles('TS_6_4', 'virus-like-particle')


plot_particles_3d('TS_6_6', 'virus-like-particle')


plot_particles('TS_6_6', 'virus-like-particle')


plot_particles_3d('TS_73_6', 'virus-like-particle')


plot_particles('TS_73_6', 'virus-like-particle')


plot_particles_3d('TS_86_3', 'virus-like-particle')


plot_particles('TS_86_3', 'virus-like-particle')


plot_particles_3d('TS_99_9', 'virus-like-particle')


plot_particles('TS_99_9', 'virus-like-particle')

