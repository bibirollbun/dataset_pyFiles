import numpy as np 
import pandas as pd
from sklearn import *
import json, warnings, random
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import colors
warnings.filterwarnings('ignore')
fig = plt.figure(figsize=(8., 6.))

p = '/kaggle/input/arc-prize-2025/'
train = pd.read_json(p+'arc-agi_training_challenges.json', orient='index').reset_index()
train_sol = pd.read_json(p+'arc-agi_training_solutions.json', orient='index').reset_index()
evals = pd.read_json(p+'arc-agi_evaluation_challenges.json', orient='index').reset_index()
evals_sol = pd.read_json(p+'arc-agi_evaluation_solutions.json', orient='index').reset_index()
test = pd.read_json(p+'arc-agi_test_challenges.json', orient='index').reset_index()

sub = eval(open(p+'sample_submission.json').read())


!cp /kaggle/input/system-control-pannel/submission.zip submission.zip


from zipfile import ZipFile
import zipfile

with zipfile.ZipFile("submission.zip", 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')


import glob
files = sorted(glob.glob('/kaggle/working/task**.py'))
len(files)


def flattener(pred):
    str_pred = '['+ ','.join(['['+ ','.join([str(v) for v in row]) +']' for row in pred]) +']'
    return str_pred


import importlib
#for i in range(len(train)):
#    print(i,'*'*20)
#    for t in train['train'][i]:
#        g=t['input']
#        r=t['output']
#        for f in files:
#            m = importlib.import_module(f.split('/')[-1].split('.')[0])
#            try:
#                if m.p(g)==r:print(f)
#            except:pass


def getOutput(dftrain, dftest):
    results = []
    g=dftrain[0]['input']
    r=dftrain[0]['output']
    for f in files:
        m = importlib.import_module(f.split('/')[-1].split('.')[0])
        try:
            if m.p(g)==r:
                print(f)
                for j in range(len(dftest)):
                    g=dftest[j]['input']
                    results.append(m.p(g))
                return results
        except:pass
    if len(results)<1:
        for j in range(len(dftest)):
            results.append(dftest[j]['input'])
    return results


sub = open('submission.json','w')
sub.write('{')
for i in range(len(test)):
    sub.write('"' + str(test['index'][i]) + '" : [')
    print(i, test['index'][i])
    preds = getOutput(test['train'][i], test['test'][i])
    for j in range(len(test['test'][i])):
        sub.write('{"attempt_1" : ' + flattener(preds[j]) + ', "attempt_2" : ' + flattener(test['test'][i][j]['input']) + '}')
        if j<len(test['test'][i])-1: sub.write(',')
    sub.write(']')
    if i < len(test)-1: sub.write(',')
sub.write('}')
sub.close()

