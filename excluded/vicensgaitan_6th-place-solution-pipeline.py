%load_ext autoreload
%autoreload 2


!pip install --no-index --find-links=/kaggle/input/ariel25-batman-minuit/packages  batman-package iminuit pqdm


import os

os.environ['DATASET']='train'
os.environ['RPATH']= '/kaggle/input/ariel-data-challenge-2025/'
!cp   /kaggle/input/ariel25-source-and-models/*  /kaggle/working



import pandas as pd
import os
import glob
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path = os.environ['RPATH']
dataset = os.environ['DATASET']

train_labels= pd.read_csv(f'{path}train.csv')

dtest = glob.glob(f'{path}{dataset}/*/AIRS-CH0_signal_*')
dt = [d.split('/') for d in dtest]
npa = path.count('/')
dt = [(d[npa+1],d[npa+2][16]) for d in dt]
df_t = pd.DataFrame(dt)
df_t.columns = ['planet_id','rep']
df_t['planet_id'] = df_t.planet_id.astype('int64')

   
if dataset == 'train':
    df_labels = pd.merge(df_t,train_labels,how = 'left')
    y_true=df_labels.iloc[:,2:].values

df_star_info = pd.read_csv(f'{path}{dataset}_star_info.csv')
pid = df_star_info.planet_id.astype(int).values
df_star_info.drop('planet_id', axis = 1)
df_star_info['planet_id'] = pid
df_star = pd.merge(df_t,df_star_info)



# Generate PCA templates from true spectra   

ncomp = 7
tl = train_labels.iloc[:,1:]

tls = (tl.T/tl.mean(axis=1)).T-1
pca = PCA(n_components=ncomp,random_state = 42)
pca.fit(tls)

np.save('components',pca.components_)



from preprocess import reduce

i0 = 499

binning = 1
file = df_t.iloc[i0]
planet_id = file[0]
fgs0,_ = reduce(file,dataset,'FGS1',binning,dejitter = False)
fgs1,_ = reduce(file,dataset,'FGS1',binning,dejitter = True)




fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(fgs0,alpha = .7)
ax1.plot(fgs1,alpha = .7)
ax1.set_title('Signal Flux over time')
ax1.set_xlabel('Flux')
ax1.set_ylabel('Time id')
ax1.legend()

ax2.hist(fgs0,bins = 100,alpha =.7);
ax2.hist(fgs1,bins = 100,alpha =.7);
ax2.set_title('Signal Flux histogram')
ax2.set_xlabel('Flux')
ax2.set_ylabel('Count')
ax2.legend()


plt.tight_layout()

plt.show()


binning = 12
fgs,_ = reduce(file,dataset,'FGS1',12*binning)
air,_  = reduce(file,dataset,'AIRS-CH0',binning)


from fit2 import get_flux_error,fit_combined_curves

# 1. Initial fit on the mean light curves
constants = df_star.iloc[i0]
flux_f, err_f, d0g, snrf = get_flux_error(fgs[:, 0])
flux_a, err_a, d1g, snra = get_flux_error(air.mean(axis=1))


minuit_result, chi2 = fit_combined_curves(constants, flux_f, err_f/np.sqrt(2), flux_a, err_a/np.sqrt(2), d0g, d1g, 1)


minuit_result


from fit2 import get_spectrum 

#full fit

allf,alle,n_out,fvalw,vals_,fval,snrf,snra  = get_spectrum(fgs,air,constants,deltaw = 1,fitw =True,dedrift = True,flag_plot = True,ret_all=True)



plt.plot(allf)
plt.plot(y_true[i0])


!python process_all.py


%%time
from model import CV_model

sigma, preds, models = CV_model(y_true,df_t,beta=.71,n_splits=4)




