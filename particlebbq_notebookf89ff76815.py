# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os

#doesn't actually seem to help with non-determinism
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["OMP_NUM_THREADS"] = "1"   # often useful too
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import itertools
from scipy.optimize import curve_fit
import torch
import re
import time as time_lib
import collections
from collections import OrderedDict
import multiprocessing as mp
#!pip install astropy --target=/kaggle/working/
#!pip install PyWavelets==1.7.0 --target=/kaggle/working/
!ls -lrtha /kaggle/working/

from astropy.stats import sigma_clip
import pywt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
np.random.seed(334258)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.utils.deterministic.fill_uninitialized_memory=True

print("numpy version: "+str(np.version.version),flush=True)
print("scipy version: "+str(scipy.__version__),flush=True)
print("torch version: "+str(torch.__version__),flush=True)
#!pip freeze

if torch.cuda.is_available():
  device=torch.device("cuda")
else:
  device=torch.device("cpu")
DEVICE=device

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#print("numpy show config:",flush=True)
#np.show_config()
#from iminuit import Minuit




#control switches and gloabal constants
#dataset="train"  
dataset="test"
max_num_train_samples=3  #in the kernel environment, runs on the train set are just for debugging.  So only run this many.
train_sample_start=167 #884

n_neldermead=0
n_neldermead_max=250


#even though it's called "cpu_fit", some implementations have made use of the gpu.  Choose this switch based on speed.
cpu_fit_device=torch.device('cpu')
#cpu_fit_device=device
#hess_device=torch.device('cpu') #perhaps the hess computations for airs are also so unable to leverage gpu parallelism that it makes sense to run them on the cpu too?
hess_device=device   #...no, I don't see a significant difference.

do_linear_corr=True
ch0_scistart=39
ch0_sciend=321
ch0_nnstart=ch0_scistart-25   #36
ch0_nnend=ch0_sciend+25   #324
#nrebin=None  #25  #75
fit_bin_width=15  #25  #75

NPARAM=4
NPARAM_MAX=4
drop_shoulders=25  #25
normalize_batch_fits=True
MAX_LDC_COEF=0.5

#before fits, errors are smoothed by averaging over neighboring bins. 
do_error_smoothing=True  
error_smoothing_scalefac=1  
SMOOTH_SIZE=2

n_hidden=64
batch_size=128
REBINNINGS=[3,7,47]  

wavelet = "cgau1"
widths = np.geomspace(1, 1024, num=10)
time = np.linspace(0, 1, 5625)
sampling_period = np.diff(time).mean()

turn_on_width=2

#when combining measurements from two visits, should we take the difference between the two means as a systematic uncertainty?
difference_is_uncert=True

#sometimes fits fail to converge well enough to compute reliable uncertainties using the Hessian. 
#This is hard to avoid given the runtime constraints in this contest.  It's a challenge for the FGS 
#fit in particular, since this analysis uses the FGS fit to pin down certain fit parameters (like Tcenter)
#which are then held constant in the AIRS fits that come after.  Thankfully, it is usually the case that the
#parameter values are good enough to get the job done, even if the Hessian computation fails.  So in these 
#cases, default to the average fit uncertainties from the training set
#will load these from a file generated during skim for the final run, but for now, temporary values extracted by hand from an older skim:
#avg_hess_uncerts=[0.00010687391152194368,0.06983016436985619,0.11065483080895322,0.05325828968780703,4.6466757371942e-05,0.06129590380754077,0.027488655060930314,24.79859075210302,0.0016164013792447425]
avg_hess_uncerts=pd.read_csv("/kaggle/input/ariel2025-average-fit-errors-v19-debugncg/average_fit_errors_v19_debugNCG.csv")

#simple mean and sigma of the training labels, used as default 
#values in case a prediction comes up with nan or inf
naive_mean=0.014689019532534075
naive_sigma=0.01066133533197834 

#placeholders:  these get populated at the start of main
nn_input_var_min=None
nn_input_var_max=None


def gauss_func(x,mu,sig,norm):
  return (norm/(sig*(2*np.pi)**0.5))*np.exp(-0.5*(x-mu)**2/sig**2)

def find_peak(x,y,size=10,fit_window=1000):
  assert len(x.shape)==1
  assert len(y.shape)==1
  assert x.shape==y.shape
  if not isinstance(x,np.ndarray):
    x=x.numpy()
  if not isinstance(y,np.ndarray):
    y=y.numpy()
  smoothed=scipy.ndimage.median_filter(y,size=size)
  initial_guess=np.argmax(smoothed)
  start=max(0,initial_guess-fit_window//2)
  end=min(x.shape[0]-1,initial_guess+fit_window//2)
  xdata=x[start:end]
  ydata=y[start:end]
  lower=[x[start],0.1,0]
  upper=[x[end],x[end]-x[start],np.inf]
  popt,pcov=curve_fit(gauss_func,x,y,bounds=(lower,upper))
  return popt,start,end

def weighted_average(coords):
  wnorm=sum([1/c[1] for c in coords])
  weights=[(1/c[1])/wnorm for c in coords]
  mean=sum([w*c[0] for w,c in zip(weights,coords)])
  sig=sum([(w*c[1])**2 for w,c in zip(weights,coords)])**0.5
  return mean,sig


def peak_average(coords):
  #just a simple weighted average, but remove outliers
  assert len(coords)>0
  if len(coords)==1:
    return tuple(coords[0])
  mean=sum([c[0] for c in coords])/len(coords)
  width=(sum([c[1]**2 for c in coords])/len(coords))**0.5
  residuals=[(c[0]-mean)/width for c in coords]
  keep=[abs(r)<2 for r in residuals]
  if all(keep):
    return weighted_average(coords)
  else:
    return peak_average([c for c,k in zip(coords,keep) if k])

def find_transit_region(ch0_signal,is_subtracted=False):

    if is_subtracted:
      white=ch0_signal
    else:
      white=torch.sum(ch0_signal,dim=1)

    norm=(torch.mean(white[:,0:50],dim=-1,keepdim=True)+torch.mean(white[:,-50:],dim=-1,keepdim=True))/2
    white=white/norm
    white=white-1

    cwtmatr, freqs = pywt.cwt(torch.squeeze(white).cpu().numpy(), widths, wavelet, sampling_period=sampling_period)
    cwtmatr = np.abs(cwtmatr[:-1, :-1])

    firstpeak_coords=[]
    secondpeak_coords=[]
    for row in [1,2,3,4,5]:
      scalslice=np.expand_dims(cwtmatr[-row,:],axis=0)
      x=np.linspace(0,scalslice.shape[-1],num=scalslice.shape[-1])
      popt_firstpeak,start,end=find_peak(x,scalslice[0,:],size=10,fit_window=1500)

      scalslice_masked=np.array(scalslice)
      scalslice_masked[0,start:end]=0
      popt_secondpeak,start2,end2=find_peak(x,scalslice_masked[0,:],size=10,fit_window=1500)

      #reminder: popt_firstpeak is a list like [1820.31765939  183.7399781     5.72666547], i.e., fitted mu, sigma, and norm of gaussian
      if popt_firstpeak[0]>popt_secondpeak[0]:
        popt_firstpeak,popt_secondpeak=popt_secondpeak,popt_firstpeak
      firstpeak_coords.append(popt_firstpeak[0:2])
      secondpeak_coords.append(popt_secondpeak[0:2])

    firstpeak_mean,firstpeak_sig=peak_average(firstpeak_coords)
    secondpeak_mean,secondpeak_sig=peak_average(secondpeak_coords)

    print("first-pass transit region estimates:  "+str(firstpeak_mean)+"+/-"+str(firstpeak_sig)+" and "+str(secondpeak_mean)+"+/-"+str(secondpeak_sig),flush=True)

    if is_subtracted:
      return firstpeak_mean,firstpeak_sig,secondpeak_mean,secondpeak_sig,cwtmatr
    else:
      gap1=(int((firstpeak_mean-firstpeak_sig*turn_on_width)/fit_bin_width)*fit_bin_width,
            fit_bin_width*(1+int((firstpeak_mean+firstpeak_sig*turn_on_width)/fit_bin_width)))
      gap2=(int((secondpeak_mean-secondpeak_sig*turn_on_width)/fit_bin_width)*fit_bin_width,
            fit_bin_width*(1+int((secondpeak_mean+secondpeak_sig*turn_on_width)/fit_bin_width)))

      white_x=np.arange(0,1,1./white.shape[-1])
      white_x_nogap=[white_x[0:gap1[0]],
                     white_x[gap1[1]:gap2[0]],
                     white_x[gap2[1]:]]
      white_x_nogap=np.concatenate(white_x_nogap,axis=-1)

      white_nogap=[white[:,0:gap1[0]],
                   white[:,gap1[1]:gap2[0]],
                   white[:,gap2[1]:]]
      white_nogap=torch.cat(white_nogap,dim=-1).cpu().numpy()

      white_bins=np.reshape(white_nogap,tuple(list(white_nogap.shape)[0:-1]+[-1,fit_bin_width]))
      white_y=np.mean(white_bins,axis=-1)
      white_yerr=np.std(white_bins,axis=-1)/fit_bin_width**0.5

      if do_error_smoothing:
        #smooth out the errors so that fluctuations don't give us very different errors in neighboring bins
        pad_size=tuple([(0,0) for idim in range(len(white_yerr.shape)-1)]+[(2,2)])
        white_yerr=np.lib.stride_tricks.sliding_window_view(np.pad(white_yerr,pad_size,'edge'),5,axis=-1)
        white_yerr=np.mean(white_yerr,axis=-1)*error_smoothing_scalefac

      white_x_binned=np.mean(np.reshape(white_x_nogap,(-1,fit_bin_width)),axis=-1)


      in_transit=np.concatenate([np.zeros_like(white_x[0:gap1[0]]),
                np.ones_like(white_x[gap1[1]:gap2[0]]),
                np.zeros_like(white_x[gap2[1]:])],axis=0)
      in_transit=np.reshape(in_transit,(-1,fit_bin_width))
      in_transit=np.min(in_transit,axis=-1)

      init_norm=np.mean(white_bins)
      init_data=np.array(tuple([0 for i in range(NPARAM)]+[init_norm]),dtype=np.float64)

      data=(white_x_binned,white_y[0,:],white_yerr[0,:],in_transit)
      res=scipy.optimize.minimize(light_curve_chisq_simplified,init_data,  #first 0 is for offset, not a polynomial param
                                bounds=None,
                                args=data,jac=True)

      coeffs=res.x[1:]
      coeffs=coeffs[::-1]
      fit_y=np.zeros_like(white_x)
      for i in range(len(coeffs)):
        fit_y+=coeffs[i]*white_x**i

      white_sub=white.cpu().numpy()-fit_y
      try:
        firstpeak_mean2,firstpeak_sig2,secondpeak_mean2,secondpeak_sig2,cwtmatr2=find_transit_region(torch.tensor(white_sub),is_subtracted=True)
        avg=(firstpeak_sig+secondpeak_sig)/2
        avg2=(firstpeak_sig2+secondpeak_sig2)/2
        if avg2<=avg:
          return firstpeak_mean2,firstpeak_sig2,secondpeak_mean2,secondpeak_sig2,cwtmatr2
        else:
          #print("second-pass transit region estimates came out bigger than first-pass; rolling back to first-pass",flush=True)
          return firstpeak_mean,firstpeak_sig,secondpeak_mean,secondpeak_sig,cwtmatr

      except Exception as e:
        #print("failed to find second-pass transit region estimates: "+str(e),flush=True)
        return firstpeak_mean,firstpeak_sig,secondpeak_mean,secondpeak_sig,cwtmatr

def light_curve_chisq_simplified(params,*args):
                                
  #reminder: ch0_signal.shape=torch.Size([1, 288, 5625])
  #but we are fitting one wavelength at a time so that the optimizer 
  #doesn't get screwed up exploring stupid cross-wavelength correlations.
  #so we expect arguments to have the following shapes:
  #  - obs_x, obs_y, obs_yerr, and in_transit:  (5625,)
  #  - offset and polynomial coefficients: scalar
        
  assert len(args)==4
  obs_x=torch.tensor(args[0])
  obs_y=torch.tensor(args[1])
  obs_yerr=torch.tensor(args[2])
  in_transit=torch.tensor(args[3])
          
  assert len(obs_x.shape)==1
  assert obs_x.shape==obs_y.shape
  assert obs_x.shape==obs_yerr.shape
  assert obs_x.shape==in_transit.shape
  params=torch.tensor(params)
  assert len(params.shape)==1
  params.requires_grad=True

  offset=params[0]
  coeffs=params[1:]
  coeffs=torch.flip(coeffs,dims=(0,))
  fit_y=torch.zeros_like(obs_x)
  for i in range(len(coeffs)):
    fit_y+=coeffs[i]*obs_x**i

  fit_y=fit_y-torch.where(in_transit>=1,offset,0)

  chisq=(torch.abs(obs_y-fit_y))/obs_yerr  
  chisq=chisq**2
  chisq=torch.sum(chisq)

  chisq.backward()
  grad=params.grad.detach().numpy()
  return chisq.detach().numpy(),grad




def load_metadata():
  if dataset=="train":
    fname="/kaggle/input/ariel-data-challenge-2025/train.csv"
  else:
    fname="/kaggle/input/ariel-data-challenge-2025/sample_submission.csv"

  ss = pd.read_csv(fname) #'/kaggle/input/ariel-data-challenge-2025/sample_submission.csv')
  planets=[str(int(p)) for p in ss["planet_id"].to_list()]

  if dataset=="train":
    planets=planets[train_sample_start:train_sample_start+max_num_train_samples]
      
  #all planets use the same adc info in the 2025 version of this contest
  #FGS1_adc_offset,FGS1_adc_gain,AIRS-CH0_adc_offset,AIRS-CH0_adc_gain
  #-1000.0,0.4369,-1000.0,0.4369
  df_adc=pd.read_csv("/kaggle/input/ariel-data-challenge-2025/adc_info.csv")
  df_axis=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/axis_info.parquet")
  integration_time=df_axis["AIRS-CH0-integration_time"].dropna().to_numpy()
  cumulative_time=np.cumsum(integration_time,axis=-1)
  cumulative_time_doublesamp=cumulative_time[1::2]

  data_dict=dict()
  data_dict["integration_time"]=integration_time
  data_dict["cumulative_time"]=cumulative_time
  data_dict["cumulative_time_doublesamp"]=cumulative_time_doublesamp
  data_dict["index"]=planets
    
  for iplanet,planet in enumerate(planets):
    row=df_adc.iloc[0]  #df_adc["planet_id"]==int(planet)]
    fgs_offset=row["FGS1_adc_offset"]
    fgs_gain=row["FGS1_adc_gain"]
    ch0_offset=row["AIRS-CH0_adc_offset"]
    ch0_gain=row["AIRS-CH0_adc_gain"]
    star=int(planet)
    data_dict[planet]=(ch0_offset,ch0_gain,fgs_offset,fgs_gain,star)

  return data_dict

def mask_hot_dead(signal, dead, dark):
    hot = sigma_clip(
        dark, sigma=5, maxiters=5
    ).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))

    hot=torch.tensor(hot.astype(np.int32)).to(DEVICE)
    dead=torch.tensor(dead.astype(np.int32)).to(DEVICE)
    signalmask=torch.maximum(hot,dead)
    return signalmask

def clean_dark(signal, signalmask, dead, dark, dt):
    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))
    darkmask = np.tile(dead, (signal.shape[0],1,1))
    dark=torch.tensor(dark).to(DEVICE)
    darkmask=torch.tensor(darkmask).to(DEVICE)
    mask=torch.maximum(darkmask.float(),signalmask.float())
    if not torch.is_tensor(dt):
      dt=torch.tensor(dt)

    signal=torch.where(mask>=1,signal,signal-dark*torch.unsqueeze(torch.unsqueeze(dt.to(DEVICE),dim=-1),dim=-1))
    return signal,mask

def correct_flat_field(flat,dead, signal,signalmask):
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    flatmask=torch.tensor(flat.mask).to(DEVICE)
    flat=torch.tensor(flat.data).to(DEVICE)
    mask=torch.maximum(flatmask.float(),signalmask.float())
    signal=torch.where(mask>=1,signal,signal/flat)
    return signal,mask

def linear_correction(signal,lincorr):
  signal_lincorr=torch.zeros_like(signal)
  for ic in range(lincorr.shape[0]):
    signal_lincorr=signal_lincorr+torch.einsum('jk,ijk->ijk',lincorr[ic,:,:],signal**ic)

  return signal_lincorr


def load_planet(planet,ch0_offset,ch0_gain,fgs_offset,fgs_gain,integration_time,visit):
  #print("hello from load_planet for planet "+str(planet),flush=True)
  #print("DEVICE="+str(DEVICE),flush=True)
  assert DEVICE is not None

  #print("reading fgs dataframe from filename /kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/FGS1_signal_"+str(visit)+".parquet",flush=True)
  df_fgs=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/FGS1_signal_"+str(visit)+".parquet")
  #print("got it",flush=True)
  fgs_signal=torch.tensor(df_fgs.to_numpy().astype(np.float32)).to(DEVICE)
  fgs_signal = fgs_signal.reshape((fgs_signal.shape[0], 32, 32))
  fgs_signal_corr=fgs_signal/fgs_gain+fgs_offset
  dt_fgs1 = torch.ones(len(fgs_signal_corr)).to(DEVICE)*0.1
  dt_fgs1[1::2] += 0.1

  #dead/dark/flat correction
  fgs_flat=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/FGS1_calibration_"+str(visit)+"/flat.parquet").values.astype(np.float64).reshape((32, 32))
  fgs_dark=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/FGS1_calibration_"+str(visit)+"/dark.parquet").values.astype(np.float64).reshape((32, 32))
  fgs_dead=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/FGS1_calibration_"+str(visit)+"/dead.parquet").values.astype(np.float64).reshape((32, 32))
    
  fgs_corr_mask = mask_hot_dead(fgs_signal_corr, fgs_dead, fgs_dark)
    
  df_fgs_lin=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/FGS1_calibration_"+str(visit)+"/linear_corr.parquet")
  #print("df_fgs_lin read ok",flush=True)
    
  fgs_lin=torch.tensor(df_fgs_lin.to_numpy().astype(np.float32)).to(DEVICE)
  fgs_coeffs=fgs_lin.reshape((6,32,32))
  if do_linear_corr:
    fgs_signal_corr=linear_correction(fgs_signal_corr,fgs_coeffs)

  fgs_corr,fgs_corr_mask = clean_dark(fgs_signal_corr,fgs_corr_mask, fgs_dead, fgs_dark,dt_fgs1)
  fgs_corr = fgs_corr[1::2,:,:] - fgs_corr[::2,:,:]
  fgs_corr_mask = torch.maximum(fgs_corr_mask[1::2,:,:],fgs_corr_mask[::2,:,:])
  fgs_corr,fgs_corr_mask = correct_flat_field(fgs_flat,fgs_dead, fgs_corr,fgs_corr_mask)

  #print("fgs_corr.shape="+str(fgs_corr.shape),flush=True)
  fgs_corr_proj1=torch.clip(torch.sum(torch.where(fgs_corr_mask<=0,fgs_corr,0),dim=-1),0,1e30)
  fgs_norm_proj1=torch.sum(fgs_corr_proj1,dim=-1,keepdim=True)
  fgs_probs_proj1=fgs_corr_proj1/fgs_norm_proj1
  fgs_corr_proj2=torch.clip(torch.sum(torch.where(fgs_corr_mask<=0,fgs_corr,0),dim=-2),0,1e30)
  fgs_norm_proj2=torch.sum(fgs_corr_proj2,dim=-1,keepdim=True)
  fgs_probs_proj2=fgs_corr_proj2/fgs_norm_proj2
  x=torch.unsqueeze(torch.arange(0,32).to(DEVICE),dim=0)
  fgs_mean_proj1=torch.sum(x*fgs_probs_proj1,dim=-1,keepdim=True)
  #print("x.shape="+str(x.shape)+", fgs_mean_proj1.shape="+str(fgs_mean_proj1.shape),flush=True)

  fgs_std_proj1=torch.sum(fgs_probs_proj1*(x-fgs_mean_proj1)**2,dim=-1)
  fgs_mean_proj2=torch.sum(x*fgs_probs_proj2,dim=-1,keepdim=True)
  fgs_std_proj2=torch.sum(fgs_probs_proj2*(x-fgs_mean_proj2)**2,dim=-1)

  fgs_mean_proj1=torch.mean(torch.reshape(fgs_mean_proj1,(-1,12*fit_bin_width)),dim=-1)
  fgs_std_proj1=torch.mean(torch.reshape(fgs_std_proj1,(-1,12*fit_bin_width)),dim=-1)
  fgs_mean_proj2=torch.mean(torch.reshape(fgs_mean_proj2,(-1,12*fit_bin_width)),dim=-1)
  fgs_std_proj2=torch.mean(torch.reshape(fgs_std_proj2,(-1,12*fit_bin_width)),dim=-1)

  #simple outlier exclusion -- make a sliding window in time and when you find a point >3sigma away from 
  #the mean in the sliding window, replace it with the mean.  Should get rid of stuff like cosmic rays.
  #The "slide" in these variable names refers to "sliding window"
  fgs_slide=fgs_corr.unfold(dimension=0,size=50,step=1)
  fgs_slide_mean=torch.mean(fgs_slide,dim=-1)
  fgs_slide_std=torch.std(fgs_slide,dim=-1)
  fgs_slide_mean=torch.transpose(fgs_slide_mean,0,2)
  fgs_slide_std=torch.transpose(fgs_slide_std,0,2)
  fgs_slide_mean=torch.nn.functional.pad(fgs_slide_mean,(25,24),mode='replicate')
  fgs_slide_std=torch.nn.functional.pad(fgs_slide_std,(25,24),mode='replicate')
  fgs_slide_mean=torch.transpose(fgs_slide_mean,0,2)
  fgs_slide_std=torch.transpose(fgs_slide_std,0,2)

  dev=(fgs_corr-fgs_slide_mean)/fgs_slide_std
  fgs_corr=torch.where(torch.abs(dev)<3,fgs_corr,fgs_slide_mean)

  fgs_corr=torch.unsqueeze(torch.sum(torch.where(fgs_corr_mask<=0,fgs_corr,0),axis=(-1,-2)),dim=0)
  fgs_nmask=torch.sum(fgs_corr_mask)

  #print("reading ch0",flush=True)
  df_ch0=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/AIRS-CH0_signal_"+str(visit)+".parquet")
  ch0_signal=torch.tensor(df_ch0.to_numpy().astype(np.float32),dtype=torch.float32).to(DEVICE)
  ch0_signal=ch0_signal.reshape(-1,32,356)
  ch0_signal=ch0_signal[:,:,ch0_nnstart:ch0_nnend]
  #ADC correction
  ch0_signal_corr=ch0_signal/ch0_gain+ch0_offset

  #do we need to scale these by collection time?  https://www.kaggle.com/competitions/ariel-data-challenge-2024/discussion/528066
  #dead/dark/flat correction
  df_ch0_flat=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/AIRS-CH0_calibration_"+str(visit)+"/flat.parquet")
  ch0_flat=df_ch0_flat.to_numpy()[:,ch0_nnstart:ch0_nnend]

  df_ch0_dark=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/AIRS-CH0_calibration_"+str(visit)+"/dark.parquet")
  ch0_dark=df_ch0_dark.to_numpy()[:,ch0_nnstart:ch0_nnend]

  df_ch0_dead=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/AIRS-CH0_calibration_"+str(visit)+"/dead.parquet")
  ch0_dead=df_ch0_dead.to_numpy()[:,ch0_nnstart:ch0_nnend]

  #print("calling ch0 mask_hot_dead",flush=True)
  #mask hot/dead
  ch0_signal_corr_mask=mask_hot_dead(ch0_signal_corr,ch0_dead,ch0_dark)

  df_ch0_lin=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/AIRS-CH0_calibration_"+str(visit)+"/linear_corr.parquet")
  ch0_lin=torch.tensor(df_ch0_lin.to_numpy().astype(np.float32)).to(DEVICE)
  ch0_coeffs=ch0_lin.reshape((6,32,356))
  ch0_coeffs=ch0_coeffs[:,:,ch0_nnstart:ch0_nnend]

  if do_linear_corr:
    ch0_signal_lincorr=linear_correction(ch0_signal_corr,ch0_coeffs)
    ch0_lincorr_shift=ch0_signal_lincorr-ch0_signal_corr
    ch0_signal_corr=ch0_signal_lincorr

  #dark current subtraction
  dt_airs=np.array(integration_time)
  dt_airs[1::2]+=0.1
  ch0_signal_corr,sch0_signal_corr_mask=clean_dark(ch0_signal_corr,ch0_signal_corr_mask,ch0_dead,ch0_dark,dt_airs)  #integration_time)

  #double-sampling correction
  ch0_dsamp=ch0_signal_corr[::2,:,:]
  ch0_signal_corr=ch0_signal_corr[1::2,:,:]-ch0_signal_corr[::2,:,:]
  ch0_signal_corr_mask=torch.maximum(ch0_signal_corr_mask[1::2,:,:],ch0_signal_corr_mask[::2,:,:])

  #print("calling ch0 flat field correction",flush=True)
  #flat field correction
  ch0_signal_corr,ch0_signal_corr_mask=correct_flat_field(ch0_flat,ch0_dead,ch0_signal_corr,ch0_signal_corr_mask)

  #read noise
  #df_ch0_read=pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/AIRS-CH0_calibration_"+str(visit)+"/read.parquet")
  #ch0_read=df_ch0_read.to_numpy()[:,ch0_nnstart:ch0_nnend] #before that last trim, ch0_read.shape=(32, 356), dtype=float64

  ch0_norm=torch.sum(torch.clip(ch0_signal_corr,0,1e30),dim=-2,keepdim=True)
  ch0_probs=ch0_signal_corr/ch0_norm
  x=torch.unsqueeze(torch.unsqueeze(torch.arange(0,32).to(DEVICE),dim=0),dim=-1)
  ch0_mean=torch.sum(x*ch0_probs,dim=-2,keepdim=True)
  ch0_std=torch.sum(ch0_probs*(x-ch0_mean)**2,dim=-2)

  ch0_mean=torch.mean(torch.reshape(torch.transpose(ch0_mean,0,1),(ch0_mean.shape[-1],-1,fit_bin_width)),dim=-1)
  ch0_std=torch.mean(torch.reshape(torch.transpose(ch0_std,0,1),(ch0_std.shape[-1],-1,fit_bin_width)),dim=-1)

  ch0_signal_corr=torch.sum(torch.where(ch0_signal_corr_mask<=0,ch0_signal_corr,0),axis=1)  #ignore spatial dim for now -- shape=(n_timesteps,282)
  ch0_signal_corr=torch.transpose(ch0_signal_corr,0,1)  #shape=(282,n_timesteps), so that the time series is an event dim
  ch0_nmask=torch.sum(ch0_signal_corr_mask,dim=(0,1))

  #print("before do_linear_corr if block, so far so good",flush=True)
  if do_linear_corr:
    #print("ch0_lincorr_shift.shape="+str(ch0_lincorr_shift.shape),flush=True)
    ch0_lincorr_shift=torch.sum(ch0_lincorr_shift,axis=1)
    #print("ch0_dsamp.shape="+str(ch0_dsamp.shape),flush=True)
    ch0_dsamp=torch.sum(ch0_dsamp,axis=1)

    return torch.unsqueeze(ch0_signal_corr,dim=0),torch.unsqueeze(fgs_corr,dim=0), ch0_lincorr_shift, ch0_dsamp, fgs_nmask,ch0_nmask,fgs_mean_proj1, fgs_std_proj1, fgs_mean_proj2, fgs_std_proj2, ch0_mean, ch0_std
  else:
    return torch.unsqueeze(ch0_signal_corr,dim=0),torch.unsqueeze(fgs_corr,dim=0), None, ch0_dsamp, fgs_nmask,ch0_nmask,fgs_mean_proj1, fgs_std_proj1, fgs_mean_proj2, fgs_std_proj2, ch0_mean, ch0_std

def load_star_params():
  
  """
  head /ariel_data/train_star_info.csv
  planet_id,Rs,Ms,Ts,Mp,e,P,sma,i
  34983,1.155435480707952,1.062960837903184,5577.006645157513,0.6949463499730458,0.0,3.305588751623328,8.550785903749272,89.1507586203837
  1873185,1.813230199682509,1.370450683964498,6216.229756270119,0.6108447312470062,0.0,6.352659805895124,9.55338410018746,88.70151407048026
  3849793,0.6538067391809703,0.667352384889845,4968.477185692076,1.5291995382480204,0.0,5.522797615237956,15.285660679929052,89.1341766239202
  """

  df_starinfo=pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train_star_info.csv")
  var_min=df_starinfo.min()
  var_max=df_starinfo.max()

  if dataset=="test":
    df_starinfo=pd.read_csv("/kaggle/input/ariel-data-challenge-2025/test_star_info.csv")

  #print("var_min="+str(var_min),flush=True)
  #print("var_max="+str(var_max),flush=True)
    
  star_params=dict()
  star_params_norm=dict()
  for istar in range(len(df_starinfo)):
    planet_id=str(int(df_starinfo.iloc[istar]["planet_id"]))
    Rs=df_starinfo.iloc[istar]["Rs"]
    Ms=df_starinfo.iloc[istar]["Ms"]
    Ts=df_starinfo.iloc[istar]["Ts"]
    Mp=df_starinfo.iloc[istar]["Mp"]
    e=df_starinfo.iloc[istar]["e"]
    P=df_starinfo.iloc[istar]["P"]*24  #days-->hours      #*24*60*60  #days-->seconds
    sma=df_starinfo.iloc[istar]["sma"]
    i=df_starinfo.iloc[istar]["i"]*np.pi/180
    star_params[planet_id]=(Rs,Ms,Ts,Mp,e,P,sma,i)
    b=max(0,min(1,sma*np.cos(i)))  #...and e is always 0 in this dataset

    Rsnorm=(Rs-var_min["Rs"])/(var_max["Rs"]-var_min["Rs"])
    Msnorm=(Ms-var_min["Ms"])/(var_max["Ms"]-var_min["Ms"])
    Tsnorm=(Ts-var_min["Ts"])/(var_max["Ts"]-var_min["Ts"])
    Mpnorm=(Mp-var_min["Mp"])/(var_max["Mp"]-var_min["Mp"])
    #enorm=(e-var_min["e"])/(var_max["e"]-var_min["e"])  #<---always zero, I think, so norm comes out nan
    Pnorm=(P-var_min["P"]*24)/(var_max["P"]*24-var_min["P"]*24)
    smanorm=(sma-var_min["sma"])/(var_max["sma"]-var_min["sma"])
    inorm=(i-var_min["i"]*np.pi/180)/(var_max["i"]*np.pi/180-var_min["i"]*np.pi/180)
    star_params_norm[planet_id]=(Rsnorm,Msnorm,Tsnorm,Mpnorm,Pnorm,smanorm,inorm,b)  #b is already roughly in the range (0,1); no need to normalize again


  return star_params,star_params_norm




def hess_estimator(x,*args):
  data=args[0:-2]
  chisquare_fn=args[-2]
  central=args[-1]

  data_for_hess=tuple(list(data)+[False])
  hess=torch.func.hessian(chisquare_fn)(torch.tensor(central.x,device=cpu_fit_device),*data_for_hess)
  return hess.detach().cpu().numpy()

cpu_fit_errnames=[
        "raw_fit_depth_hess_err",
        "fit_p3_hess_err",
        "fit_p2_hess_err",
        "fit_p1_hess_err",
        "fit_p0_hess_err",
        "Tcenter_hess_err",
        "T_hess_err",
        "tau_hess_err",
        "ldc_hess_err"
      ]
def check_bounds_cpu(hess_uncerts):
  if nn_input_var_min is None or nn_input_var_max is None:
    print("check_bounds_cpu called with no bounds",flush=True)
    return False
  if len(hess_uncerts)==0:
    #print("check_bounds_cpu called with empty hess_uncerts list",flush=True)
    return False
  for iname,name in enumerate(cpu_fit_errnames):
    if name not in nn_input_var_min or name not in nn_input_var_max:
      print(str(name)+" appears to be missing from the nn_input_var bounds",flush=True)
      return False
    val=hess_uncerts[iname]
    if val<nn_input_var_min[name] or val>nn_input_var_max[name]:
      return False
  return True
    
def hess_errors_cpu(data,chisquare_fn,central):
  data_for_hess=tuple(list(data)+[False])
  hess=torch.func.hessian(chisquare_fn)(torch.tensor(central.x,device=cpu_fit_device),*data_for_hess)
  #print("hess.shape="+str(hess.shape),flush=True)
    
  try:
    hessinv=torch.linalg.inv(hess+1e-8*torch.eye(hess.shape[0]).to(cpu_fit_device))
    errmat=hessinv
    #print("errmat.shape="+str(errmat.shape),flush=True)
    if any([errmat[i,i].item()<0 for i in range(hessinv.shape[0])]):
      #print("negative diagonal elements: "+str([errmat[i,i] for i in range(hessinv.shape[0])]),flush=True)
      raise Exception("inverse hessian has negative elements on its diagonal")
    hess_uncerts=[errmat[i,i].item()**0.5 for i in range(hessinv.shape[0])]
  except Exception as e:
    print("hessian is not invertible; may need to redo minimization.  Error message: "+str(e),flush=True)
    hess_uncerts=[]
  return hess_uncerts

def cpu_fit(arg_tup,is_retry=False): 
  chisquare_fn,init_data,data,kwargs_dict=arg_tup
  central=None
  nparam=4
  bounds=None
  eps=0.00001
  force_refit=False

  if "central" in kwargs_dict:
    central=kwargs_dict["central"]
  if "nparam" in kwargs_dict:
    nparam=kwargs_dict["nparam"]
  if "bounds" in kwargs_dict:
    bounds=kwargs_dict["bounds"]
  if "eps" in kwargs_dict:
    eps=kwargs_dict["eps"]
  if "force_refit" in kwargs_dict:
    force_refit=kwargs_dict["force_refit"]

  if central is None or force_refit:
    #print("about to call bfgs with init_data="+str(init_data),flush=True)
    #print("...and data="+str(data),flush=True)
    #print("dtypes for tensors in init_data="+str([x.dtype if torch.is_tensor(x) else None for x in init_data]),flush=True)
    #print("dtypes for tensors in data="+str([x.dtype if torch.is_tensor(x) else None for x in data]),flush=True)

      
    #print("dtypes for numpy arrays in init_data="+str([x.dtype if isinstance(x,np.ndarray) else None for x in init_data]),flush=True)
    #print("dtypes for numpy arrays in data="+str([x.dtype if isinstance(x,np.ndarray) else None for x in data]),flush=True)
      
    #print("bounds="+str(bounds),flush=True)
    res=scipy.optimize.minimize(chisquare_fn,init_data,
                                bounds=bounds,
                                method="Newton-CG",
                                args=data,jac=True) #,hess=hess_estimator)
    hess_uncerts=hess_errors_cpu(data,chisquare_fn,res)
    print("newton-cg returns fun="+str(res.fun)+", success="+str(res.success)+", hess_uncerts="+str(hess_uncerts),flush=True)

    #switching SLSQP-->Nelder-Mead here seems to recover nearly-identical fit results for the FGS fit, 
    #at least for the one example I have so far checked in detail.
    #for minim in ["SLSQP","Powell"]:
    #for minim in ["Newton-CG","TNC","Powell","Nelder-Mead"]:
    for minim in ["TNC","Powell","Nelder-Mead"]:
      #for minim in ["Newton-CG","Powell"]:

      #count calls to Nelder-Mead over the run and stop when you hit some threshold
      global n_neldermead
      if minim=="Nelder-Mead":
        if n_neldermead>=n_neldermead_max:
          break
        else:
          n_neldermead+=1
            
      if not res.success or len(hess_uncerts)==0 or not check_bounds_cpu(hess_uncerts) or res.fun>560:  #560 is roughly chisq/ndof=1.5
        print("retry minimization with "+str(minim),flush=True)
        #print("...check_bounds_cpu returned "+str(check_bounds_cpu(hess_uncerts)),flush=True)
        #print("starting new call to minimize, method="+str(minim)+"; res.x="+str(res.x),flush=True)
        #if torch.is_tensor(res.x) or isinstance(res.x,np.ndarray):
        #  print("res.x is a tensor or ndarray with dtype="+str(res.x.dtype),flush=True)
        #elif isinstance(res.x,list):
        #  print("res.x is a list whose element types are "+str([(type(x),x.dtype) if torch.is_tensor(x) or isinstance(x,np.ndarray) else type(x) for x in res.x]),flush=True)

        res=scipy.optimize.minimize(chisquare_fn,res.x,
                                    bounds=bounds,
                                    method=minim,
                                    args=data,jac=True)  #,hess=hess_estimator)
        hess_uncerts=hess_errors_cpu(data,chisquare_fn,res)
        print(str(minim)+" returns fun="+str(res.fun)+", success="+str(res.success)+", hess_uncerts="+str(hess_uncerts),flush=True)

    #if len(hess_uncerts)>0:
    #  print("hess_errors_cpu has returned a non-empty list with "+str(sum([1 if e is None else 0 for e in hess_uncerts]))+" None(s)",flush=True)
    #  print("contents: "+str(hess_uncerts),flush=True)
    #else:
    #  print("hess_errors_cpu has returned an empty list",flush=True)
        
    if central is None:
      central=res
    elif res.fun<central.fun:
      central=res

  return central,hess_uncerts

def compute_polynomial_fit(params,obs,include_signal=True):
  #parameter list, with usual default values:
  #  - signal norm: [0.]
  #  - background shape: [0]*NPARAM-1 + [1.]
  #  - signal center logit: [Tcenter_guess]  --> computed as time_bins*torch.sigmoid(param)
  #  - transit duration: [T_guess]
  #  - ingress/egress duration logit: [tau_guess] --> gets computed as (T/2)*torch.sigmoid(param)
  #  - limb darkening coeff logits: [0.,0.] --> actual coeffs are computed as MAX_LDC_COEF*torch.sigmoid(param)

  obs_x,Tcenter_guess,T_guess=obs
 
  params=[torch.tensor(c).to(cpu_fit_device) if not torch.is_tensor(c) else c.to(cpu_fit_device) for c in params]
  if not torch.is_tensor(obs_x):
    obs_x=torch.tensor(obs_x,device=cpu_fit_device)

  offset=torch.nn.functional.softplus(params[0],beta=100)
  coeffs=params[1:-4]
  coeffs=coeffs[::-1]   
  #coeffs[1:]=[c/100 for c in coeffs[1:]]
  coeffs[1:]=[0.01*torch.tanh(c) for c in coeffs[1:]]

  fit_y=torch.zeros_like(obs_x,device=cpu_fit_device)
  for i in range(len(coeffs)):
    fit_y+=coeffs[i]*obs_x**i

  if include_signal:
      
    Tcenter=torch.clip(Tcenter_guess+(obs_x.shape[-1]/2)*torch.tanh(torch.clip(params[-4],-4,4)),10,obs_x.shape[-1]-10)
    T=torch.clip(T_guess+(obs_x.shape[-1]/2)*torch.tanh(torch.clip(params[-3],-4,4)),1,obs_x.shape[-1])
    tau=(T/2)*torch.sigmoid(torch.clip(params[-2],-5,5))
    ldc1=MAX_LDC_COEF*torch.sigmoid(torch.clip(params[-1],-5,5))

    tstart=Tcenter-T/2  
    tend=Tcenter+T/2
    ingr_start=tstart-tau/2
    ingr_end=tstart+tau/2
    egr_start=tend-tau/2
    egr_end=tend+tau/2

    tstart=torch.clip(tstart,0,fit_y.shape[-1]-1)
    tend=torch.clip(tend,0,fit_y.shape[-1]-1)
    ingr_start=torch.clip(ingr_start,0,fit_y.shape[-1]-1)
    ingr_end=torch.clip(ingr_end,0,fit_y.shape[-1]-1)
    egr_start=torch.clip(egr_start,0,fit_y.shape[-1]-1)
    egr_end=torch.clip(egr_end,0,fit_y.shape[-1]-1)

    in_transit=torch.clip((torch.arange(0,obs_x.shape[-1]).to(obs_x.device)-tstart)/tau,0,1)*torch.clip((tend-torch.arange(0,obs_x.shape[-1]).to(obs_x.device))/tau,0,1)
    in_transit=in_transit/torch.max(in_transit,dim=-1,keepdim=True).values
    limbdark=torch.ones_like(obs_x,device=cpu_fit_device)
    limbdark_x=(torch.arange(0,obs_x.shape[-1],device=cpu_fit_device)-Tcenter)/max(T/2,1)
    limbdark=torch.clip(limbdark-ldc1*limbdark_x**2,0,1)  
    limbdark=limbdark/torch.max(limbdark)
    in_transit=in_transit*limbdark
    in_transit=in_transit/torch.max(in_transit,dim=-1,keepdim=True).values

    fit_y=fit_y*((1-in_transit*offset))   #torch.where(in_transit>=1,offset,0)

    #print("compute_poly: params="+str([p.item() for p in params]),flush=True)
    #print("Tcenter,T,tau,ldc1="+str([Tcenter.item(),T.item(),tau.item(),ldc1.item()]),flush=True)
    #print("sum(in_transit)="+str(torch.sum(in_transit).item()),flush=True)
    #raise Exception("Stop")
  return fit_y

def polynomial_penalty(params):
  if not torch.is_tensor(params):
    return 0
  retval= max(0,torch.abs(params[-4])-4)**2+ max(0,torch.abs(params[-3])-4)**2+  max(0,torch.abs(params[-2])-4)**2+ max(0,torch.abs(params[-1])-4)**2
  return retval*100

def light_curve_chisquare(params,*args):
  #reminder: ch0_signal.shape=torch.Size([1, 288, 5625])
  #but we are fitting one wavelength at a time so that the optimizer 
  #doesn't get screwed up exploring stupid cross-wavelength correlations.
  #so we expect arguments to have the following shapes:
  #  - obs_x, obs_y, obs_yerr, and in_transit:  (5625,)
  #  - offset and polynomial coefficients: scalar
        
  obs_x=args[0]  #torch.tensor(args[0])
  obs_y=torch.tensor(args[1],device=cpu_fit_device)
  obs_yerr=torch.tensor(args[2],device=cpu_fit_device)                    
  compute_func=args[3]
  penalty_eval_func=args[4]
  do_grad=True
  if len(args)==6:
    do_grad=args[5] 
  
  if not torch.is_tensor(params):
    params=torch.tensor(params,device=cpu_fit_device)
  assert len(params.shape)==1
  params.requires_grad=True

  fit_y=compute_func(params,obs_x,include_signal=True)

  chisq=(torch.abs(obs_y-fit_y))/obs_yerr  
  chisq=chisq**2
  chisq=torch.sum(chisq)

  if penalty_eval_func is not None:
    #print("chisq mean="+str(chisq.mean().item())+" before penalty....",flush=True)
    chisq=chisq+penalty_eval_func(params)
    #print("...and "+str(chisq.mean().item())+" after",flush=True)

  if not do_grad:
    return chisq

  chisq.backward()
  if torch.any(torch.abs(params.grad)>1000):
    scalefac=torch.max(torch.abs(params.grad)).detach() 
    params.grad=params.grad/scalefac

  grad=params.grad.detach().cpu().numpy()
  #print("grad="+str(grad),flush=True)
  #raise Exception("stop")
    
  return chisq.detach().cpu().numpy(),grad




ChisqFitResult=collections.namedtuple("ChisqFitResult",["success","fun","x","instance"])

class LightCurveChisquare(torch.nn.Module):
  def __init__(self,init_data): 
    super().__init__()
    device=init_data[0].device
    self.device=device
    self.cached_signal_shape=None

    #assert len(init_data[0].shape)==1
    self.n_wavelengths=init_data[0].shape[0]

    self.signal_offset=torch.nn.parameter.Parameter(data=init_data[0].to(device))
    idx=0
    self.tau_param=torch.nn.parameter.Parameter(data=init_data[1].to(device))
    self.ldc_param=torch.nn.parameter.Parameter(data=init_data[2].to(device))
    
    #polynomial parameters
    self.poly_c0=torch.nn.parameter.Parameter(data=init_data[3].to(device))
    self.poly_c1=torch.nn.parameter.Parameter(data=init_data[4].to(device))
    self.poly_c2=torch.nn.parameter.Parameter(data=init_data[5].to(device))
    self.poly_c3=torch.nn.parameter.Parameter(data=init_data[6].to(device))

  def forward(self,data,fixpars):
    params=[self.signal_offset,self.tau_param,self.ldc_param]
    params+=[self.poly_c0,self.poly_c1,self.poly_c2,self.poly_c3]

    #these penalty terms keep the tau and ldc parameters from wandering off far beyond the range 
    #over which we have computed interpolation templates
    penalty=torch.clip(torch.abs(self.tau_param)-5,0,1e30)
    penalty+=torch.clip(torch.abs(self.ldc_param)-5,0,1e30)

    return self.compute_chisq(params,data)+torch.sum(penalty)

  def compute_chisq(self,params,data):
    obs_x=data[0].to(self.device)
    obs_y=data[1].to(self.device)
    obs_yerr=data[2].to(self.device)
    Tcenter_guess,T_guess,tau_guess,MAX_LDC_COEF,prefit_x=data[3]

    batch_shape=obs_y.shape[:-1]
    for i in range(len(obs_y.shape[:-1])):
      obs_x=torch.unsqueeze(obs_x,dim=0)

    if not torch.is_tensor(obs_x):
      obs_x=torch.tensor(obs_x).to(self.device)


    poly_y,signal=gpu_compute_poly_fit_func(params,(obs_x,Tcenter_guess,T_guess,tau_guess,MAX_LDC_COEF,prefit_x),include_signal=True)

    chisq=(torch.abs(obs_y-poly_y)/torch.clip(obs_yerr,min=1e-4))**2
    chisq=torch.sum(chisq,dim=-1)
    return chisq

  #convenience function to extract fit results from the chisquare model above and expose them via an interface that looks a bit like scipy.optimize.OptimizeResult
  def fit_result(self,success,loss):
    params=[self.signal_offset,self.tau_param,self.ldc_param,self.poly_c0,self.poly_c1,self.poly_c2,self.poly_c3]

    params=[p.data.detach().cpu() for p in params]
    return ChisqFitResult(success=True,fun=loss,x=params,instance=self)

#@torch.compile
def gpu_compute_poly_fit_func(params,obs,include_signal=True,fixpars=None,return_signal=False,signal_shape=None):
    obs_x,Tcenter_guess,T_guess,tau_guess,MAX_LDC_COEF,prefit_x=obs

    #reminder: input params=[self.signal_offset,self.Tcenter_param,self.T_param,self.tau_param,self.ldc_param,self.poly_c0,self.poly_c1,self.poly_c2,self.poly_c3]
    #if not free_signal_shape, then self.Tcenter_param, self.T_param, self.tau_param, and self.ldc_param are absent
    offset=torch.nn.functional.softplus(params[0],beta=100)


    #Tcenter_param=torch.clip(Tcenter_guess+(obs_x.shape[-1]/2)*prefit_x[-4],0,obs_x.shape[-1])
    #T_param=torch.clip(T_guess+(obs_x.shape[-1]/2)*torch.nn.functional.softplus(prefit_x[-3],beta=100),0,obs_x.shape[-1])


    Tcenter_param=prefit_x[0].to(offset.device)*torch.ones_like(offset,device=offset.device)
    T_param=prefit_x[1].to(offset.device)*torch.ones_like(offset,device=offset.device)
    tau_param=params[1]
    ldc_param=params[2]
    coeffs=[p.to(offset.device) for p in params[3:]]
    coeffs[1:]=[0.01*torch.tanh(c) for c in coeffs[1:]]

    if len(offset.shape)==0:
      offset=torch.unsqueeze(offset,dim=0)
      Tcenter_param=torch.unsqueeze(Tcenter_param,dim=0)
      T_param=torch.unsqueeze(T_param,dim=0)
      tau_param=torch.unsqueeze(tau_param,dim=0)
      ldc_param=torch.unsqueeze(ldc_param,dim=0)
      coeffs=[torch.unsqueeze(c,dim=0) for c in coeffs]

    fit_y=torch.unsqueeze(coeffs[0],dim=-1)
    for ic,c in enumerate(coeffs[1:]):
      cc=torch.unsqueeze(c,dim=-1)
      ox=obs_x
      fit_y=fit_y+cc*ox**(ic+1)

    in_transit=None
    if include_signal:
      if len(offset.shape)==1:
        off=torch.unsqueeze(offset,dim=-1)
      else:
        off=offset

      Tcenter=torch.clip(Tcenter_guess+(obs_x.shape[-1]/2)*torch.tanh(torch.clip(Tcenter_param,-4,4)),10,obs_x.shape[-1]-10)
      T=torch.clip(T_guess+(obs_x.shape[-1]/2)*torch.tanh(torch.clip(T_param,-4,4)),0,obs_x.shape[-1])
      tau=(T/2)*torch.sigmoid(torch.clip(tau_param,-5,5))
      ldc1=MAX_LDC_COEF*torch.sigmoid(torch.clip(ldc_param,-5,5))

      Tcenter=torch.unsqueeze(Tcenter,dim=-1)
      T=torch.unsqueeze(T,dim=-1)
      tau=torch.unsqueeze(tau,dim=-1)
      ldc1=torch.unsqueeze(ldc1,dim=-1)

      tstart=Tcenter-T/2 
      tend=Tcenter+T/2

      in_transit=torch.clip((torch.arange(0,obs_x.shape[-1]).to(obs_x.device)-tstart)/tau,0,1)*torch.clip((tend-torch.arange(0,obs_x.shape[-1]).to(obs_x.device))/tau,0,1)
      in_transit=in_transit/torch.max(in_transit,dim=-1,keepdim=True).values

      limbdark_x=(obs_x*obs_x.shape[-1]-Tcenter)/torch.maximum(T/2,torch.ones_like(T))
      limbdark=torch.ones_like(obs_x,device=offset.device)
      limbdark=torch.clip(limbdark-ldc1*limbdark_x**2,0,1)          
      limbdark=limbdark/torch.max(limbdark,dim=-1,keepdim=True).values
      in_transit=in_transit*limbdark
      in_transit=in_transit/torch.max(in_transit,dim=-1,keepdim=True).values

      fit_y=fit_y*(1-in_transit*off)

    return fit_y,in_transit



def torch_minimize(chisq_class,init_data,data,fixpars=None,niter=5000,lr=3e-4,lr_decay=0.999,noisy=False,limit=0.01):
  #chisq_class:  a torch.nn.Module (class, not instance) that computes a chisquare to be minimized.  Its forward() should take two arguments, data and fixpars.
  #              It should also implement a method, fit_result(success,loss) which returns a ChisqFitResult summarizing the success, loss and best-fit parameters.
  #data:  the argument to the chisquare model
  #fixpars:  if not None, a list of 2-tuples.  First element of each tuple is an index on params, indicating which param is to
  #          be held constant.  Second element is the value at which that param should be held fixed.
  #          If None, then loss is optimized over all parameters.
  #
  loss_fn=chisq_class(init_data)
  optimizer=torch.optim.Adam(loss_fn.parameters(),lr=lr)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

  last_check=None
  for i_iter in range(niter):
    optimizer.zero_grad()
    loss=loss_fn(data,fixpars)
    loss_sum=torch.sum(loss)

    loss_sum.backward()
    optimizer.step()
    scheduler.step()

    if i_iter>niter/8 and i_iter%10==0:
      loss_mean=loss.mean().item()
      if last_check is None:
        last_check=loss_mean
      elif last_check-loss_mean<limit and torch.max(loss)<560: 
        print("minimization terminates at i_iter="+str(i_iter)+" because loss is not dropping very much anymore",flush=True)
        break
      elif loss_mean<last_check:
        last_check=loss_mean
  else:
    print("optimization loop ran all the way to iteration "+str(i_iter),flush=True)

  #Optimized parameters are stored in the model; query that to get the fit result.
  res=loss_fn.fit_result(True,loss.detach().cpu())
  if fixpars is not None:
    for iparam,p in fixpars:
      res.x[iparam]=p.detach()  #.cpu()
  return res


def compute_hess_uncerts(central,chisq_class,init_data,data,device,eps=1e-8):
  hess_uncerts=[]
  last=None
  for ibatch in range(central.x[0].shape[-1]):
    if last is not None:
      hess_uncerts.append(last)
      last=None
      continue
        
    chisq_class_instance=chisq_class([torch.unsqueeze(p[ibatch],dim=0).to(hess_device) for p in central.x])
    data_for_hess=tuple([data[0].to(hess_device),torch.unsqueeze(data[1][ibatch,:],dim=0).to(hess_device),torch.unsqueeze(data[2][ibatch,:],dim=0).to(hess_device),data[3]])
    hparam=torch.cat([torch.unsqueeze(p[ibatch],dim=0).to(hess_device) for p in central.x],dim=0)
    hess=torch.func.hessian(chisq_class_instance.compute_chisq)(hparam,data_for_hess)
    hess=torch.squeeze(hess,dim=0)
      
    try:
      hessinv=torch.linalg.inv(hess+eps*torch.eye(hess.shape[0]).to(hess_device))
      errmat=hessinv
      if any([errmat[i,i].item()<0 for i in range(hessinv.shape[-1])]):
        hess_uncerts.append(None)
      else:
        hess_uncerts.append(torch.tensor([[errmat[i,i].item()**0.5 for i in range(hessinv.shape[-1])]],device=torch.device('cpu')))
        last=hess_uncerts[-1]
    except Exception as e:
      hess_uncerts.append(None)

  return hess_uncerts


gpu_fit_errnames=[
        "raw_fit_depth_hess_err",
        "tau_hess_err", 
        "ldc_hess_err",
        "fit_p0_hess_err",
        "fit_p1_hess_err",
        "fit_p2_hess_err",
        "fit_p3_hess_err"
]

def check_bounds_gpu(hess_uncerts):
  if nn_input_var_min is None or nn_input_var_max is None:
    #print("check_bounds_cpu called with no bounds",flush=True)
    return True
  #else:
  #  print("check_bounds_gpu called with hess_uncerts.shape="+str(hess_uncerts.shape),flush=True)
  for iname,name in enumerate(gpu_fit_errnames):
    if name not in nn_input_var_min or name not in nn_input_var_max:
      print(str(name)+" appears to be missing from the nn_input_var bounds",flush=True)
      return True
    val=hess_uncerts[0,iname]
    #if name in ["fit_p1_hess_err","fit_p2_hess_err","fit_p3_hess_err"]:
    #    val=val*100  #because the bounds file is based on rescaled values for these params, and rescaling has not happened yet
    if val<nn_input_var_min[name] or val>nn_input_var_max[name]:
      print("check_bounds_gpu returns false for "+str(name)+": val="+str(val)+" but bounds="+str([nn_input_var_min[name],nn_input_var_max[name]]),flush=True)
      return False
  return True
    
def gpu_fit(chisq_class,init_data,data,device,central=None,niter=5000,niter_errfit=1000,lr=3e-4,lr_decay=0.999,eps=0.001,noisy=False,force_refit=False,depth=0):

  if central is None or force_refit:
    new_central=torch_minimize(chisq_class,init_data,data,niter=niter,lr=lr,lr_decay=lr_decay,noisy=noisy,limit=1.0)
    print("at depth="+str(depth)+", new_central mean function value="+str(torch.mean(new_central.fun).item()),flush=True)
    if central is None:
      central=new_central
    else:
      x=[torch.where(new_central.fun.to(device)<central.fun.to(device),new_central.x[i].to(device),central.x[i].to(device)).detach() for i in range(len(central.x))]
      new_init_data=x
      fun=torch.minimum(new_central.fun.to(device),central.fun.to(device))
      central=ChisqFitResult(success=False,fun=fun,x=x,instance=chisq_class(new_init_data))

  print("finished central fit; working on hessian uncertainties...",flush=True)

  hess_uncerts=compute_hess_uncerts(central,chisq_class,init_data,data,device)

  if all([u is None for u in hess_uncerts]) or len(hess_uncerts)==0:
    print("empty hess_uncerts in gpu_fit",flush=True)
    hess_uncerts=[]
    errnames=["raw_fit_depth_hess_err","tau_hess_err","ldc_hess_err","fit_p0_hess_err","fit_p1_hess_err","fit_p2_hess_err","fit_p3_hess_err"]
    for j in range(len(avg_hess_uncerts)-1):
      default=[avg_hess_uncerts.iloc[j+1][name] for name in errnames]
      #the default values have the scale factor of 100 applied to p1/p2/p3 and their uncertainties, 
      #but the values normally coming out of this function do not. 
      # --> no longer the case in v19-debugNCG
      #default[4]/=100
      #default[5]/=100
      #default[6]/=100
      hess_uncerts.append(torch.tensor([default]))
    print("done dealing with that") 
  #else:
  #  #crazy pills: check uncertainties against the ranges the neural nets were trained on, and re-minimize if any are out of range
  #  crazy_pills=False
  #  for iu,u in enumerate(hess_uncerts):
  #    if u is None:
  #      continue
  #    cb=check_bounds_gpu(u)
  #    if not cb:
  #        print("crazy-pills check fails for wavelength "+str(iu),flush=True)
  #    crazy_pills=crazy_pills or not cb
  #  if crazy_pills:
  #    print("some fit uncertainties are outside the expected range; try re-minimizing...",flush=True)
  #    new_central=torch_minimize(chisq_class,central.x,data,niter=niter,lr=lr,lr_decay=lr_decay,noisy=noisy,limit=0.01) #1.0)
  #    print("after crazy-pills retry, new_central mean function value="+str(torch.mean(new_central.fun).item()),flush=True)

  #    x=[torch.where(new_central.fun.to(device)<central.fun.to(device),new_central.x[i].to(device),central.x[i].to(device)).detach() for i in range(len(central.x))]
  #    new_init_data=x
  #    fun=torch.minimum(new_central.fun.to(device),central.fun.to(device))
  #    central=ChisqFitResult(success=False,fun=fun,x=x,instance=chisq_class(new_init_data))
  #    hess_uncerts=compute_hess_uncerts(central,chisq_class,init_data,data,device)

  #    crazy_pills=False
  #    for iu,u in enumerate(hess_uncerts):
  #      if u is None:
  #        continue
  #      cb=check_bounds_gpu(u)
  #      if not cb:
  #        print("crazy-pills re-check fails for wavelength "+str(iu),flush=True)
  #      crazy_pills=crazy_pills or not cb
  #    print("after re-minimizing, check on uncertainty bounds returns: "+str(crazy_pills),flush=True)
        
        
  #print("gpu compute_hess_uncerts has returned a list with "+str(sum([1 if e is None else 0 for e in hess_uncerts]))+" None values",flush=True)
  #fill in any missing uncertainties with extrapolated/interpolated values
  for i in range(len(hess_uncerts)):
    if hess_uncerts[i] is None:
      if i==0 or all([hess_uncerts[j] is None for j in range(i)]):
        first_not_none=min([j for j in range(len(hess_uncerts)) if hess_uncerts[j] is not None])
        hess_uncerts[i]=hess_uncerts[first_not_none]
      elif i==len(hess_uncerts)-1 or all([hess_uncerts[j] is None for j in range(i,len(hess_uncerts))]):
        last_not_none=max([j for j in range(len(hess_uncerts)) if hess_uncerts[j] is not None])
        hess_uncerts[i]=hess_uncerts[last_not_none]
      else:
        prev=max([j for j in range(0,i) if hess_uncerts[j] is not None])
        nxt=min([j for j in range(i,len(hess_uncerts)) if hess_uncerts[j] is not None])
        hess_uncerts[i]=(hess_uncerts[prev]+hess_uncerts[nxt])/2

  hess_uncerts=torch.cat(hess_uncerts,dim=0)

  #print("returning from gpu_fit, hess_uncerts="+str(hess_uncerts),flush=True)
  return len(hess_uncerts)>0,central,hess_uncerts



#neural net for limb darkening correction, bias correction, and smoothing.
#Originally intended to also interpolate binned fit results to get predictions for each wavelength.
#Wound up not doing the binning, but the name stuck.
class CorrInterp(torch.nn.Module):
  def __init__(self,block_dict,n_hidden,n_hidden_pe,rebinnings,device,ref):
    super().__init__()
    self.n_hidden=n_hidden
    self.n_hidden_pe=n_hidden_pe
    self.rebinnings=rebinnings
    self.device=device
    self.ref=ref

    if ref=="fit":
      self.nfeats_in=sum([block_dict[key].shape[-1] for key in ["baseline","poly","res","sig","bg","spatial"]])
      self.nfeats_in+=2*(len(rebinnings)+1)*(block_dict["sliding_window_in"].shape[-1])  #for the "sliding_window_<size>" and "sliding_window_err_<size>" blocks
    else:
      self.nfeats_in=sum([block_dict[key].shape[-1] for key in ["baseline","spatial"]])

    self.fex=torch.nn.Sequential(OrderedDict([
      ("fc1",torch.nn.Linear(self.nfeats_in,2*self.nfeats_in,device=device)),
      ("relu",torch.nn.LeakyReLU()),
      ("dropout",torch.nn.Dropout()),
      ("fc2",torch.nn.Linear(2*self.nfeats_in,n_hidden,device=device))
    ]))

    #self.proj=torch.nn.Linear(n_hidden,n_hidden_pe,device=device)

    self.conv1=torch.nn.Conv1d(n_hidden_pe,n_hidden_pe,3,padding=1,device=device)
    self.relu=torch.nn.LeakyReLU()
    self.conv3=torch.nn.Conv1d(n_hidden_pe,2,3,padding=1,device=device)  #there was a conv2 in some variants of this class


  def forward(self,data,mask=None,width_scale=None):
    #input blocks should be shape (batch_size,n_wavelength_bins,feat_size)
    #fex will output shape (batch_size,n_wavelength_bins,n_hidden)
    #Need to transpose the last two to make inputs for conv layers 
    #feats=[torch.transpose(fex(data[var]),-1,-2) for var,fex in self.fex_dict.items()]
    #print("welcome to CorrInterp::forward",flush=True)

    #during training, checked that this was the case for key in [0,1,2]
    #assert "baseline" in data[key]
    #assert "fit" in data[key]
    #assert "poly" in data[key]
    #assert "res" in data[key]
    #assert "bg" in data[key]
    #assert "sig" in data[key]
    #assert "baseline_unnorm" in data[key]
    #assert "fit_unnorm" in data[key]

    if self.ref=="fit":
      pred=data["fit_unnorm"][...,0:2].clone()  #raw fit mean and uncert
      block_list=["baseline","poly","res","sig","bg","spatial"]
    else:
      pred=data["baseline_unnorm"][...,0:2].clone()
      block_list=["baseline","spatial"]

    feats=[]
    for key in block_list:
      #print("key="+str(key)+", feats list gets "+str(data[0][key]),flush=True)
      feats.append(data[key])

    #for key in ["sliding_window_in","sliding_window_errs_in"]+["sliding_window_"+str(sw_size) for sw_size in self.rebinnings]+["sliding_window_errs_"+str(sw_size) for sw_size in self.rebinnings]:
    #  print("key="+str(key)+", feats list gets "+str(data[0][key]),flush=True)
    if self.ref=="fit":      
      feats.append(data["sliding_window_in"])
      feats.append(data["sliding_window_errs_in"])
      for sw_size in self.rebinnings:
        feats.append(data["sliding_window_"+str(sw_size)])
        feats.append(data["sliding_window_errs_"+str(sw_size)])
    #print("len(feats)="+str(len(feats)),flush=True)

    #print("feats dtypes: "+str([x.dtype for x in feats]),flush=True)
    x=torch.cat(feats,dim=-1)
    #print("x.shape="+str(x.shape),flush=True)
    #print("fex gets input x="+str(x),flush=True)

    #print("x.shape="+str(x.shape),flush=True)
    #print("x="+str(x),flush=True)
    #print("row 0 as list: "+str(x[0,0,:].tolist()),flush=True)
    #print("row 1 as list: "+str(x[0,1,:].tolist()),flush=True)

    feats=self.fex(x)
    feats=torch.transpose(feats,1,2)  #(batch_size,n_hidden,n_wavelength_bins)

    x=self.relu(self.conv1(feats))
    x=torch.nn.functional.dropout(x,training=self.training)
    res=self.conv3(x)  #shape (batch_size,2*len(rebinnings)+1,n_wavelength_bins)

    #initial training is smoother if you have a bigger uncertainty
    if width_scale is not None:
      pred[...,1]=pred[...,1]*width_scale

    #inverse softplus to get a logit (which can be negative) instead of a sigma (which can't)
    pred[...,1]=torch.log(torch.expm1(torch.clip(pred[...,1],1e-6,1.0)))
    if torch.any(torch.isnan(pred)):
      print("nan in fit_unnorm after inverse softplus in model forward",flush=True)
      raise Exception("nan in fit_unnorm after inverse softplus in model forward")
    if torch.any(torch.isinf(pred)):
      print("inf in fit_unnorm after inverse softplus in model forward",flush=True)
      raise Exception("inf in fit_unnorm after inverse softplus in model forward")

    pred=torch.transpose(pred,-1,-2)

    #print("forward for ref="+str(self.ref)+" sees pred="+str(pred),flush=True)
    #print("...and res="+str(res),flush=True)
      
    retval=pred+res
    return retval  #,torch.mean(res[...,1]**2)

def make_feature(row,var_max=None,var_min=None,orbit_block=None):  #,signal_shape_params=None):


  varnames=[("baseline",["baseline_depth","baseline_depth_err","norm"])]
  varnames.append(("fit",["raw_fit_depth","raw_fit_depth_hess_err","lincorr_fit_depth","lincorr_fit_depth_err","chisq_ndof"]))
  varnames.append(("poly",["fit_p0","fit_p1","fit_p2","fit_p3","fit_p0_hess_err","fit_p1_hess_err","fit_p2_hess_err","fit_p3_hess_err"]))
  varnames.append(("res",["res_pre","res_ingr","res_mid","res_egr","res_post"]))
  varnames.append(("bg",["bg_pre","bg_ingr","bg_mid","bg_egr","bg_post"]))
  varnames.append(("sig",["sig_ingr","sig_mid","sig_egr","T","T_hess_err"]))
  varnames.append(("spatial",["nmask_scaled","spatial_center_mean","spatial_center_std","spatial_width_mean","spatial_width_std","spatial_center_range","spatial_width_range"]))
  varnames.append(("sliding_window_in",["raw_fit_depth","tau","ldc"]))
  varnames.append(("sliding_window_errs_in",["raw_fit_depth_hess_err","tau_hess_err","ldc_hess_err"]))
  varnames.append(("extras",["tstart_mean","tend_mean","tstart_sig","tend_sig"]))


    
  #varnames=[("baseline",["baseline_depth","baseline_depth_err","norm"])] 
  #varnames.append(("fit",["raw_fit_depth","raw_fit_depth_hess_err","lincorr_fit_depth","lincorr_fit_depth_err","chisq_ndof"]))
  #varnames.append(("poly",["fit_p0","fit_p1","fit_p2","fit_p3","fit_p0_hess_err","fit_p1_hess_err","fit_p2_hess_err","fit_p3_hess_err"]))
  #varnames.append(("res",["res_pre","res_ingr","res_mid","res_egr","res_post"]))
  #varnames.append(("bg",["bg_pre","bg_ingr","bg_mid","bg_egr","bg_post"]))  
  #varnames.append(("sig",["sig_ingr","sig_mid","sig_egr","T","T_hess_err"]))  
  #varnames.append(("spatial",["nmask_scaled","spatial_center_mean","spatial_center_std","spatial_width_mean","spatial_width_std","spatial_center_range","spatial_width_range"]))
  #varnames.append(("sliding_window_in",["raw_fit_depth","tau","ldc"]))
  #varnames.append(("sliding_window_errs_in",["raw_fit_depth_hess_err","tau_hess_err","ldc_hess_err"]))
  #varnames.append(("extras",["tstart_mean","tend_mean","tstart_sig","tend_sig"]))

  #n_yell=0
  blocks=dict()
  for key,vlist in varnames:
    feat=[]
    feat_unnorm=[]
    for varname in vlist:
      feat_unnorm.append(row[varname])
      if var_max is None or var_min is None or varname not in var_max or varname not in var_min:
        #print("could not find varname="+str(varname)+" in at least one of var_max="+str(var_max)+" or var_min="+str(var_min),flush=True)
        #n_yell+=1
        #if n_yell>10:
        #    raise Exception("stfu")   
        feat.append(row[varname])
      else:
        feat.append((row[varname]-var_min[varname])/(var_max[varname]-var_min[varname]))
        if feat[-1]<0 or feat[-1]>1:
          #print("normalized value of "+str(varname)+" in block "+str(key)+" is not so normalized!",flush=True)
          #print("original value: "+str(row[varname]),flush=True)
          #print("min value: "+str(var_min[varname]),flush=True)
          #print("max value: "+str(var_max[varname]),flush=True)
          #print("feat so far: "+str(feat),flush=True)
          #print("!!!!!------>badly normalized value: "+str(feat[-1]),flush=True)
          #if feat[-1]>1.3 or feat[-1]<-0.3:
          #  raise Exception("stupid value: "+str(feat[-1]))
              
          feat[-1]=max(0,min(1,feat[-1]))

      if not torch.is_tensor(feat_unnorm[-1]):
        feat_unnorm[-1]=torch.tensor(feat_unnorm[-1],dtype=torch.float32,device=device)
      else:
        feat_unnorm[-1]=feat_unnorm[-1].to(device)
      if not torch.is_tensor(feat[-1]):
        feat[-1]=torch.tensor(feat[-1],dtype=torch.float32,device=device)
      else:
        feat[-1]=feat[-1].to(device)
      if len(feat_unnorm[-1].shape)==0:
        feat_unnorm[-1]=torch.unsqueeze(feat_unnorm[-1],dim=0)
      if len(feat[-1].shape)==0:
        feat[-1]=torch.unsqueeze(feat[-1],dim=0)

    #print("for key="+str(key)+", we have the following:",flush=True)
    #print("  type(feat)="+str(type(feat)),flush=True)
    #print("  types _in_ feat: "+str([type(x) for x in feat]),flush=True)
    #print("  shapes in feat: "+str([x.shape for x in feat]),flush=True)
    #print("  devices in feat: "+str([x.device for x in feat]),flush=True)
      
    #blocks[key]=np.array([feat],dtype=np.float32)
    blocks[key]=torch.unsqueeze(torch.cat(feat,dim=-1),dim=0)
    #print("a",flush=True)
    #blocks[key+"_unnorm"]=np.array([feat_unnorm],dtype=np.float32)
    blocks[key+"_unnorm"]=torch.unsqueeze(torch.cat(feat_unnorm,dim=-1),dim=0)
    #print("b",flush=True)
    if torch.any(torch.isnan(blocks[key])):
      raise Exception("block for key="+str(key)+" has nan: "+str(blocks[key]))

    #print("c",flush=True)
  #print("d",flush=True)
  if orbit_block is not None:
    #print("type(orbit_block)="+str(type(orbit_block)),flush=True) 
    blocks["baseline"]=torch.cat([blocks["baseline"],orbit_block],axis=-1)
  if torch.any(torch.isnan(blocks["poly"])) or torch.any(torch.isinf(blocks["poly"])):
    print("blocks[poly]="+str(blocks["poly"]),flush=True)
    raise Exception("nan in poly block in make_feature")

  if torch.any(blocks["fit_unnorm"][...,1]<=0):
    print("bad fit_unnorm block? "+str(blocks["fit_unnorm"]),flush=True)
    blocks["fit_unnorm"][...,1]=torch.abs(blocks["fit_unnorm"][...,1])

  return blocks

def collate_example(blocks):
  if len(blocks)==0:
    raise Exception("collate_example called with an empty blocks list")
      
  coll=dict()
  for key in sorted(blocks[0].keys()):
    #print("key: "+str(key)+", types="+str([type(b[key]) for b in blocks]),flush=True)
    feat=torch.cat([b[key] for b in blocks],dim=-2)
    if torch.any(torch.isnan(feat)) or torch.any(torch.isinf(feat)):
      print("key="+str(key)+" feat="+str(feat),flush=True)
      raise Exception("nan or inf in feature for key="+str(key)+": "+str(feat))
      

    #scan for and remove/impute any uncertainties that are tiny or huge
    #expect feat has shape (n_wavelengths,n_features_in_block)
    if key=="sliding_window_errs_in_unnorm":
      mask_small=torch.where(feat<1e-6,0,1)
      mask_large=torch.where(feat>100,0,1)
      mask=torch.minimum(mask_small,mask_large).to(dtype=bool)
      at_least_one_good=torch.any(mask,axis=0)
      if not torch.all(at_least_one_good):  # and rebin_id==0:
        print("feat="+str(feat),flush=True)
        raise Exception("some planet in this batch has a problem with at_least_one_good: "+str(at_least_one_good))

      denom=torch.sum(mask,dim=0,keepdim=True)
      means=torch.sum(feat,dim=0,keepdim=True)/denom
      feat=torch.where(mask,means,feat)

    coll[key]=feat

  #also done here:  compute some derived features by averaging, in a sliding window of various sizes, 
  #                 a few best-fit parameters and their errors 
  for sw_size in REBINNINGS:
    feat=coll["sliding_window_in"].detach().cpu().numpy()  #expect shape (n_wavelength_bins,n_values_in_block)
    feat_errs=coll["sliding_window_errs_in"].detach().cpu().numpy()  #same shape

    feat_pad=np.pad(feat,((int((sw_size-1)/2),int((sw_size-1)/2)),(0,0)),mode="constant",constant_values=0.)
    feat_errs_pad=np.pad(feat_errs,((int((sw_size-1)/2),int((sw_size-1)/2)),(0,0)),mode="constant",constant_values=1e10)
    sw=np.lib.stride_tricks.sliding_window_view(feat_pad,window_shape=sw_size,axis=0)  #expect shape (n_wavelength_bins,n_values_in_block,sw_size)
    sw_errs=np.lib.stride_tricks.sliding_window_view(feat_errs_pad,window_shape=sw_size,axis=0)
    weights=np.where(sw_errs>0,1./sw_errs**2,0.)
    sumweights=np.sum(weights,axis=-1,keepdims=True)
    weights=np.where(sumweights>0,weights/sumweights,0)
    feat=np.sum(sw*weights,axis=-1)
    feat_errs=np.where(sumweights>0,(1./sumweights**0.5),0)[...,0]
    mask=np.sum(np.where(np.abs(sw)>0,1,0),axis=-1)
    feat_std=np.where(mask>1,np.std(sw,axis=-1),0)
    mask=np.sum(np.where(np.abs(sw_errs)>0,1,0),axis=-1)
    feat_err_std=np.where(mask>1,np.std(np.where(sw_errs>100,0,sw_errs),axis=-1),0)
    coll["sliding_window_"+str(sw_size)]=torch.tensor(feat,dtype=torch.float32,device=device)
    coll["sliding_window_errs_"+str(sw_size)]=torch.tensor(feat_errs,dtype=torch.float32,device=device)
    coll["sliding_window_std_"+str(sw_size)]=torch.tensor(feat_std,dtype=torch.float32,device=device)
    coll["sliding_window_errs_std_"+str(sw_size)]=torch.tensor(feat_err_std,dtype=torch.float32,device=device)

  #print("starting last step of collate",flush=True)  
  #finally, give every block a batch index; even though we will likely run examples through the postproc
  #network one at a time, that network was set up to expect a batch index. 
  for key in coll.keys():
    coll[key]=torch.unsqueeze(coll[key],dim=0).to(torch.float32)
  #print("all done with collate",flush=True)
  return coll



def load_postproc_models(data):
  models=[]
  for i in range(5):
    models.append(CorrInterp(data,n_hidden,n_hidden,REBINNINGS,device,ref="fit"))
    models[-1].load_state_dict(torch.load("/kaggle/input/ariel2025-postproc-nets-v19post/postproc_v19post_net_fold"+str(i)+".pth",weights_only=True,map_location=device))
    models[-1].eval()

  fallbacks=[]
  for i in range(5):
    fallbacks.append(CorrInterp(data,n_hidden,n_hidden,REBINNINGS,device,ref="baseline"))
    fallbacks[-1].load_state_dict(torch.load("/kaggle/input/ariel2025-postproc-nets-v19post/postproc_fallback_v19post_net_fold"+str(i)+".pth",weights_only=True,map_location=device))
    fallbacks[-1].eval()
    
  return models,fallbacks

def combine_measurements(pred_list):
  #expect pred list to be a list of (almost certainly two) tensors of shape (batch_size,2,n_wavelengths).
  #Elements with that second index ==0 are predicted mean, and elements with second index ==1 are predicted uncertainty 
  means=[p[:,0,:] for p in pred_list]
  sigmas=[p[:,1,:] for p in pred_list]
  weights=[torch.where(sig**2>0,1/sig**2,0) for sig in sigmas]
  sumweight=sum(weights)
  weights=[w/sumweight for w in weights]
  mean=sum([m*w for m,w in zip(means,weights)])
  sigma=torch.where(sumweight**0.5>0,1./sumweight**0.5,1)

  if difference_is_uncert:
    mean_max=means[0]
    mean_min=means[0]
    for m in means[1:]:
      mean_max=torch.maximum(mean_max,m)
      mean_min=torch.minimum(mean_min,m)
    delta=torch.abs(mean_max-mean_min)/2
    sigma=(sigma**2+delta**2)**0.5
  new_pred=torch.cat([torch.unsqueeze(mean,dim=1),torch.unsqueeze(sigma,dim=1)],dim=1)
  return new_pred



def simple_outlier_exclusion(ch0_signal):
  #outlier exclusion for fgs was done in load_planet(); only need to do airs here
  ch0_signal=ch0_signal.detach().cpu().numpy()
  ch0_slide=np.lib.stride_tricks.sliding_window_view(ch0_signal,(1,1,50))
  ch0_slide_mean=np.squeeze(np.squeeze(np.mean(ch0_slide,axis=-1),axis=-1),axis=-1)
  ch0_slide_std=np.squeeze(np.squeeze(np.std(ch0_slide,axis=-1),axis=-1),axis=-1)
  ch0_slide_mean=np.pad(ch0_slide_mean,((0,0),(0,0),(25,24)),mode='edge')
  ch0_slide_std=np.pad(ch0_slide_std,((0,0),(0,0),(25,24)),mode='edge')
  dev=(ch0_signal-ch0_slide_mean)/ch0_slide_std
  ch0_signal=torch.tensor(np.where(np.abs(dev)<3,ch0_signal,ch0_slide_mean)).to(device)

  return ch0_signal

def initial_transit_bounds(fgs_signal,ch0_signal):
  fgs_gaps_ok=True
  try:
    fgs_tstart_mean,fgs_tstart_sig,fgs_tend_mean,fgs_tend_sig,scalogram=find_transit_region(fgs_signal)
    #print("fgs transit region: "+str(fgs_tstart_mean)+"+/-"+str(fgs_tstart_sig)+" to "+str(fgs_tend_mean)+"+/-"+str(fgs_tend_sig),flush=True)
  except Exception as e:
    print("fgs call to find_transit_region raises exception: "+str(e),flush=True)
    fgs_gaps_ok=False
    fgs_tstart_mean,fgs_tstart_sig,fgs_tend_mean,fgs_tend_sig=None,None,None,None

  if fgs_tstart_sig is not None and fgs_tend_sig is not None and fgs_tstart_sig is not None and fgs_tend_sig is not None:
    if fgs_tstart_sig<15 or fgs_tend_sig<15 or fgs_tstart_sig>1000 or fgs_tend_sig>1000:
      fgs_gaps_ok=False

  ch0_gaps_ok=False #take this out to save time
  #ch0_gaps_ok=True
  #try:
  #  ch0_tstart_mean,ch0_tstart_sig,ch0_tend_mean,ch0_tend_sig,scalogram=find_transit_region(ch0_signal)
  #except Exception as e:
  #  print("ch0 call to find_transit_region raises exception: "+str(e),flush=True)
  #  ch0_gaps_ok=False
  #  ch0_tstart_mean,ch0_tstart_sig,ch0_tend_mean,ch0_tend_sig=None,None,None,None#

  #if ch0_tstart_mean is not None and ch0_tstart_sig is not None and ch0_tend_mean is not None and ch0_tend_sig is not None:
  #  if ch0_tstart_sig<15 or ch0_tend_sig<15 or ch0_tstart_sig>1000 or ch0_tend_sig>1000:
  #    ch0_gaps_ok=False

  #if not ch0_gaps_ok and not fgs_gaps_ok:
  #  tstart_mean=1800
  #  tend_mean=3600
  #  tstart_sig=50
  #  tend_sig=50
  #elif not ch0_gaps_ok:
  #  if fgs_gaps_ok:
  #    tstart_mean,tstart_sig,tend_mean,tend_sig=fgs_tstart_mean,fgs_tstart_sig,fgs_tend_mean,fgs_tend_sig
  #elif (ch0_gaps_ok and fgs_gaps_ok) or (not ch0_gaps_ok and not fgs_gaps_ok):
  #  tstart_mean=(ch0_tstart_mean+fgs_tstart_mean)/2
  #  tstart_sig=(ch0_tstart_sig+fgs_tstart_sig)/2
  #  tend_mean=(ch0_tend_mean+fgs_tend_mean)/2
  #  tend_sig=(ch0_tend_sig+fgs_tend_sig)/2
  #else:
  #  tstart_mean,tstart_sig,tend_mean,tend_sig=ch0_tstart_mean,ch0_tstart_sig,ch0_tend_mean,ch0_tend_sig

  if fgs_gaps_ok:
    tstart_mean,tstart_sig,tend_mean,tend_sig=fgs_tstart_mean,fgs_tstart_sig,fgs_tend_mean,fgs_tend_sig
  else:
    tstart_mean=1800
    tend_mean=3600
    tstart_sig=50
    tend_sig=50
    
  gap1=(int((tstart_mean-tstart_sig*turn_on_width)/fit_bin_width)*fit_bin_width,
    fit_bin_width*(1+int((tstart_mean+tstart_sig*turn_on_width)/fit_bin_width)))
  gap2=(int((tend_mean-tend_sig*turn_on_width)/fit_bin_width)*fit_bin_width,
    fit_bin_width*(1+int((tend_mean+tend_sig*turn_on_width)/fit_bin_width)))

  if gap2[0]<=gap1[1]+fit_bin_width:
    midpoint=int((gap1[1]+gap2[0])/2)
    gap1=(gap1[0],midpoint-fit_bin_width)
    gap2=(midpoint+fit_bin_width,gap2[1])
  if gap1[0]>=gap1[1]:
    gap1=(max(0,gap1[1]-fit_bin_width),gap1[1])
  if gap2[0]>=gap2[1]:
    gap2=(gap2[0],min(ch0_signal.shape[-1],gap2[0]+fit_bin_width))

      
  return tstart_mean,tstart_sig,tend_mean,tend_sig, gap1, gap2

def smooth_errors(yerr):
  pad_size=tuple([(0,0) for idim in range(len(yerr.shape)-1)]+[(SMOOTH_SIZE,SMOOTH_SIZE)])
  yerr=np.lib.stride_tricks.sliding_window_view(np.pad(yerr.cpu().numpy(),pad_size,'edge'),2*SMOOTH_SIZE+1,axis=-1)
  yerr=np.mean(yerr,axis=-1)*error_smoothing_scalefac
  yerr=torch.tensor(yerr).to(device)
  return yerr

def gaussian_error_propagation_baseline(y,yerr,gap1,gap2):
  intrans=y[:,:,int(gap1[1]/fit_bin_width):max(int(gap1[1]/fit_bin_width)+1,int(gap2[0]/fit_bin_width))]   #expect shape (1,282,n_intrans)
  outrans=torch.cat([y[:,:,0:max(1,int(gap1[0]/fit_bin_width))],y[:,:,min(y.shape[-1]-1,int(gap2[1]/fit_bin_width)):]],dim=-1)  #shape (1,282,n_outrans)
  intrans_err=yerr[:,:,int(gap1[1]/fit_bin_width):max(int(gap1[1]/fit_bin_width)+1,int(gap2[0]/fit_bin_width))]   #expect shape (1,282,n_intrans)
  outrans_err=torch.cat([yerr[:,:,0:max(1,int(gap1[0]/fit_bin_width))],yerr[:,:,min(yerr.shape[-1]-1,int(gap2[1]/fit_bin_width)):]],dim=-1)  #shape (1,282,n_outrans)

  mean_intrans=torch.mean(intrans,dim=-1)
  mean_outrans=torch.mean(outrans,dim=-1)
  stderr_intrans=torch.std(intrans,dim=-1)/intrans.shape[-1]**0.5
  stderr_outrans=torch.std(outrans,dim=-1)/outrans.shape[-1]**0.5
  prop_intrans=torch.sqrt(torch.sum(intrans_err**2,dim=-1))
  prop_outrans=torch.sqrt(torch.sum(outrans_err**2,dim=-1))
  #dFF is short for "delta F over F"
  dFF_baseline=(mean_outrans-mean_intrans)/mean_outrans
  #propagation of error:  
  #  d/d(mean_intrans) dFF = -1/mean_outrans
  #  d/d(mean_outrans) dFF = mean_intrans/mean_outrans**2
  #  sig(dFF)**2 = ((-1/mean_outrans)**2) d(mean_intrans)**2 + ((mean_intrans/mean_outrans**2)**2) d(mean_outrans)**2 
  #dFF_baseline_err= ( ((-1/mean_outrans)**2) * (stderr_intrans**2 + prop_intrans**2) + ((mean_intrans/mean_outrans**2)**2) * (stderr_outrans**2 + prop_outrans**2))**0.5
  dFF_baseline_err= ( ((-1/mean_outrans)**2) * (stderr_intrans**2 + prop_intrans**2) )**0.5

  dFF_baseline=torch.flip(dFF_baseline,dims=(-1,))
  dFF_baseline_err=torch.flip(dFF_baseline_err,dims=(-1,))

  return dFF_baseline,dFF_baseline_err    


def compensate_limb_darkening_linear(fitted_depth,ldc_fit_param,b):
  #just a quick-and-dirty ballpark estimate of the impact of limb darkening based on the fact that
  #the difference in observed intensity at r=1 and at conjunction can be estimated from ldc_fit_param.
  #It gets unstable when you have an impact parameter close to 1 and somehow also fit a large ldc parameter.
  #Signal parameterization is a trapezoid; at conjunction (r=b) it is 1, while at ingress/egress it is 1-ldc (if we ignore tau),
  #so the difference in intensity is just the value of the ldc parameter.
  ldc=MAX_LDC_COEF*torch.sigmoid(torch.clip(torch.tensor(ldc_fit_param),-5,5)).cpu().numpy()
  intensity_diff=ldc
  #eq 8 in https://arxiv.org/pdf/1901.01730 : I = 1 - u*(1-(1-r**2)**0.5) --> 1-u at r=1
  #so in terms of the limb darkening model, intensity_diff(@r=b - @r=1) =  [-u*(1-(1-b**2)**0.5)] +u = u*(1-b**2)**0.5, so that 
  u=max(0,min(1,intensity_diff/(1-b**2)**0.5))
  #...where I have imposed by hand the limits on u so that we never have Ip going negative
  Ia=1-u/3  #eq. 11
  Ip=1-u*(1-(1-b**2)**0.5)  #eq.8
  #if Ia/Ip<0:
  #  print("compensate_limb_darkening_linear changes sign: Ia="+str(Ia)+", Ip="+str(Ip)+", u="+str(u)+", b="+str(b)+", ldc_fit_param="+str(ldc_fit_param),flush=True)
  return fitted_depth*Ia/Ip

def load_bounds(fname):
  out=dict()
  with open(fname,"r") as f:
    for line in f:
      key,val=re.split(",",line[:-1])
      if len(key)>0:
        out[key]=float(val)
  return out


try:
  start_time=time_lib.time()
  run_time_list=[]
  load_time_list=[]
  cpu_fit_time_list=[]
  gpu_fit_time_list=[]
  postproc_time_list=[]
  mask_list=[]
  print("started run at "+str(start_time),flush=True)
  meta_dict=load_metadata()
  integration_time=meta_dict["integration_time"]
  cumulative_time=meta_dict["cumulative_time"]

  #planets=[k for k in meta_dict.keys() if "time" not in k]  <--may not be in the correct order
  star_params,star_params_norm=load_star_params()

  nn_input_var_min=load_bounds("/kaggle/input/ariel2025-nn-input-bounds-v19-debugncg/nn_input_min_bounds_v19_debugNCG.csv")
  nn_input_var_max=load_bounds("/kaggle/input/ariel2025-nn-input-bounds-v19-debugncg/nn_input_max_bounds_v19_debugNCG.csv")
    
  error_code=None

  #will load models after we have an example of the input
  #...it was a convenient way to set things up during model development, albeit a bit awkward here
  postproc_models=None
  postproc_fallbacks=None
  dFF_preds=[]
  dFF_sigmas=[]
  planets=meta_dict["index"]
  final_means=[]
  final_sigmas=[]
  last_example_start_time=time_lib.time()
  for iplanet,planet in enumerate(planets):
        
    #if iplanet%10==0:
    print("working on planet "+str(planet)+"; this is number "+str(iplanet)+" of "+str(len(planets)),flush=True)

    ch0_offset,ch0_gain,fgs_offset,fgs_gain,star=meta_dict[planet]

    ls=os.listdir("/kaggle/input/ariel-data-challenge-2025/"+str(dataset)+"/"+str(planet)+"/")
    ls=[re.sub("FGS1_calibration_","",f) for f in ls if "FGS1_calibration_" in f]
    #print("planet "+str(planet)+", ls="+str(ls),flush=True)  #expect either ['0'] or ['0','1'] -- maybe longer lists if the dataset has any examples of more than two visits per system

    batch=[]
    predicted_spectra=[]
    for ivisit,visit in enumerate(ls):
     try:
      example_start_time=time_lib.time()
      if iplanet>0 or ivisit>0:
        run_time_list.append(example_start_time-last_example_start_time)
        print("last example ran for "+str(run_time_list[-1])+"; average so far="+str(sum(run_time_list)/len(run_time_list)),flush=True)
        last_example_start_time=example_start_time

      ch0_signal,fgs_signal,ch0_signal_lincorr_shift,ch0_dsamp,fgs_nmask,ch0_nmask,fgs_mean_proj1, fgs_std_proj1, fgs_mean_proj2, fgs_std_proj2, ch0_mean, ch0_std=load_planet(planet,ch0_offset,ch0_gain,fgs_offset,fgs_gain,integration_time,visit)
      ch0_signal=simple_outlier_exclusion(ch0_signal)
      
      fgs_signal=torch.reshape(fgs_signal,(fgs_signal.shape[0],fgs_signal.shape[1],-1,12))
      fgs_signal=torch.mean(fgs_signal,dim=-1,keepdim=False)
      ch0_signal=ch0_signal[:,drop_shoulders:-drop_shoulders,:] 

      ch0_mean=ch0_mean[drop_shoulders:-drop_shoulders,:]
      ch0_std=ch0_std[drop_shoulders:-drop_shoulders,:]
      fgs_mean_proj1=fgs_mean_proj1/32
      fgs_mean_proj2=fgs_mean_proj2/32
      fgs_std_proj1=fgs_std_proj1/32
      fgs_std_proj2=fgs_std_proj2/32

      ch0_mean=ch0_mean/32
      ch0_std=ch0_std/32
 
      ch0_spatial_center_mean=torch.mean(ch0_mean,dim=-1)
      ch0_spatial_center_std=torch.std(ch0_mean,dim=-1)
      ch0_spatial_width_mean=torch.mean(ch0_std,dim=-1)
      ch0_spatial_width_std=torch.std(ch0_std,dim=-1)
      ch0_spatial_center_range=torch.max(ch0_mean,dim=-1).values-torch.min(ch0_mean,dim=-1).values
      ch0_spatial_width_range=torch.max(ch0_std,dim=-1).values-torch.min(ch0_std,dim=-1).values

      tstart_mean,tstart_sig,tend_mean,tend_sig,gap1,gap2=initial_transit_bounds(fgs_signal,ch0_signal)
      print("intial transit region: "+str(tstart_mean)+"+/-"+str(tstart_sig)+" to "+str(tend_mean)+"+/-"+str(tend_sig),flush=True)

      fit_outputs=[dict() for i in range(283)]
      
      fit_outputs[0]["nmask_scaled"]=fgs_nmask/(67500*32*32)
      fit_outputs[0]["spatial_center_mean"]=(fgs_mean_proj1.mean().item()+fgs_mean_proj2.mean().item())/2
      fit_outputs[0]["spatial_center_std"]=(fgs_mean_proj1.std().item()+fgs_mean_proj2.std().item())/2
      fit_outputs[0]["spatial_width_mean"]=(fgs_std_proj1.mean().item()+fgs_std_proj2.mean().item())/2
      fit_outputs[0]["spatial_width_std"]=(fgs_std_proj1.std().item()+fgs_std_proj2.std().item())/2
      r1=torch.max(fgs_mean_proj1).item()-torch.min(fgs_mean_proj1).item()
      r2=torch.max(fgs_mean_proj2).item()-torch.min(fgs_mean_proj2).item()
      fit_outputs[0]["spatial_center_range"]=(r1+r2)/2

      r1=torch.max(fgs_std_proj1).item()-torch.min(fgs_std_proj1).item()
      r2=torch.max(fgs_std_proj2).item()-torch.min(fgs_std_proj2).item()
      fit_outputs[0]["spatial_width_range"]=(r1+r2)/2

      for j in range(282):
        fit_outputs[j+1]["nmask_scaled"]=ch0_nmask[j]/(5625.*32.)
        fit_outputs[j+1]["spatial_center_mean"]=ch0_spatial_center_mean[j]
        fit_outputs[j+1]["spatial_center_std"]=ch0_spatial_center_std[j]
        fit_outputs[j+1]["spatial_width_mean"]=ch0_spatial_width_mean[j]
        fit_outputs[j+1]["spatial_width_std"]=ch0_spatial_width_std[j]
        fit_outputs[j+1]["spatial_center_range"]=ch0_spatial_center_range[j]
        fit_outputs[j+1]["spatial_width_range"]=ch0_spatial_width_range[j]

      #validation printouts
      #print("planet "+str(planet)+" fgs nmask_scaled="+str(fit_outputs[0]["nmask_scaled"]),flush=True)
      #print("planet "+str(planet)+" fgs spatial_center_mean="+str(fit_outputs[0]["spatial_center_mean"]),flush=True)
      #print("planet "+str(planet)+" fgs spatial_center_std="+str(fit_outputs[0]["spatial_center_std"]),flush=True)
      #print("planet "+str(planet)+" fgs spatial_center_range="+str(fit_outputs[0]["spatial_center_range"]),flush=True)
      #print("planet "+str(planet)+" fgs spatial_width_mean="+str(fit_outputs[0]["spatial_width_mean"]),flush=True)
      #print("planet "+str(planet)+" fgs spatial_width_std="+str(fit_outputs[0]["spatial_width_std"]),flush=True)
      #print("planet "+str(planet)+" fgs spatial_width_range="+str(fit_outputs[0]["spatial_width_range"]),flush=True)


      #print("planet "+str(planet)+" ch0 150 nmask_scaled="+str(fit_outputs[151]["nmask_scaled"]),flush=True)
      #print("planet "+str(planet)+" ch0 150 spatial_center_mean="+str(fit_outputs[151]["spatial_center_mean"]),flush=True)
      #print("planet "+str(planet)+" ch0 150 spatial_center_std="+str(fit_outputs[151]["spatial_center_std"]),flush=True)
      #print("planet "+str(planet)+" ch0 150 spatial_center_range="+str(fit_outputs[151]["spatial_center_range"]),flush=True)
      #print("planet "+str(planet)+" ch0 150 spatial_width_mean="+str(fit_outputs[151]["spatial_width_mean"]),flush=True)
      #print("planet "+str(planet)+" ch0 150 spatial_width_std="+str(fit_outputs[151]["spatial_width_std"]),flush=True)
      #print("planet "+str(planet)+" ch0 150 spatial_width_range="+str(fit_outputs[151]["spatial_width_range"]),flush=True)


      fgs_mean_normalization=torch.mean(fgs_signal,dim=-1,keepdim=True)
      fit_outputs[0]["norm"]=fgs_mean_normalization[0,0,:]

      #print("planet "+str(planet)+" fgs norm="+str(fit_outputs[0]["norm"]),flush=True)
        
      fgs_norm=fgs_signal/fgs_mean_normalization
      fgs_norm=torch.squeeze(fgs_norm,dim=1)

      ch0_norm=torch.mean(ch0_signal,dim=-1,keepdim=True)
      for j in range(ch0_signal.shape[1]):
        fit_outputs[j+1]["norm"]=ch0_norm[0,j,0].item()

      #print("planet "+str(planet)+" ch0 150 norm="+str(fit_outputs[151]["norm"]),flush=True)

      for j in range(len(fit_outputs)):
        fit_outputs[j]["tstart_mean"]=tstart_mean
        fit_outputs[j]["tstart_sig"]=tstart_sig
        fit_outputs[j]["tend_mean"]=tend_mean
        fit_outputs[j]["tend_sig"]=tend_sig

      ch0_signal_x=np.arange(0,1,1./ch0_signal.shape[-1])

      #compute y and yerr values bins over time
      ch0_signal_bins=torch.reshape(ch0_signal,tuple(list(ch0_signal.shape)[0:-1]+[-1,fit_bin_width]))
      ch0_signal_y=torch.mean(ch0_signal_bins,dim=-1)
      ch0_signal_yerr=torch.std(ch0_signal_bins,dim=-1)/fit_bin_width**0.5

      #print("3",flush=True)
      fgs_bins=torch.reshape(fgs_signal,tuple(list(fgs_signal.shape)[0:-1]+[-1,fit_bin_width]))
      fgs_y=torch.mean(fgs_bins,dim=-1)
      fgs_yerr=torch.std(fgs_bins,dim=-1)/fit_bin_width**0.5
      
      if do_error_smoothing:
        #smooth out the errors so that fluctuations don't give us very different errors in neighboring bins
        ch0_signal_yerr=smooth_errors(ch0_signal_yerr)
        fgs_yerr=smooth_errors(fgs_yerr)
        
      #print("ch0_signal_x.shape="+str(ch0_signal_x.shape),flush=True)
      ch0_signal_x_binned=np.reshape(ch0_signal_x,(-1,fit_bin_width))
      ch0_signal_x_binned=np.mean(ch0_signal_x_binned,axis=-1)

      #print("4",flush=True)
      #a baseline check:  gaussian error propagation
      dFF_baseline,dFF_baseline_err=gaussian_error_propagation_baseline(ch0_signal_y,ch0_signal_yerr,gap1,gap2)
      dFF_fgs_baseline,dFF_fgs_baseline_err=gaussian_error_propagation_baseline(fgs_y,fgs_yerr,gap1,gap2)

      #print("dFF_baseline.shape="+str(dFF_baseline.shape),flush=True)  #(1,len(freq_bins))
      #print("dFF_fgs_baseline.shape="+str(dFF_fgs_baseline.shape),flush=True)  #(1,1)

      dFF_preds.append(torch.cat([dFF_fgs_baseline,dFF_baseline],dim=-1).detach().cpu().numpy())
      dFF_sigmas.append(torch.cat([dFF_fgs_baseline_err,dFF_baseline_err],dim=-1).detach().cpu().numpy())
      
      fit_outputs[0]["baseline_depth"]=dFF_fgs_baseline[0,0]
      fit_outputs[0]["baseline_depth_err"]=dFF_fgs_baseline_err[0,0]
      for j in range(282):
        fit_outputs[j+1]["baseline_depth"]=dFF_baseline[0,j]
        fit_outputs[j+1]["baseline_depth_err"]=dFF_baseline_err[0,j]

      #print("planet "+str(planet)+" fgs baseline_depth="+str(fit_outputs[0]["baseline_depth"].item()),flush=True)
      #print("planet "+str(planet)+" ch0 150 baseline_depth="+str(fit_outputs[151]["baseline_depth"].item()),flush=True)

      #print("planet "+str(planet)+" fgs baseline_depth_err="+str(fit_outputs[0]["baseline_depth_err"].item()),flush=True)
      #print("planet "+str(planet)+" ch0 150 baseline_depth_err="+str(fit_outputs[151]["baseline_depth_err"].item()),flush=True)

            
      #set up some initial guess values for fit parameters
      Tcenter_guess=abs((sum(gap1)+sum(gap2))/4.)/fit_bin_width
      T_full_guess=abs(gap2[0]-gap1[1])/fit_bin_width
      T_tot_guess=abs(gap2[1]-gap1[0])/fit_bin_width
      T_guess=(T_full_guess+T_tot_guess)/2

      #print("gap1="+str(gap1),flush=True)
      #print("gap2="+str(gap2),flush=True)
      #print("T_full_guess="+str(T_full_guess)+", T_tot_guess="+str(T_tot_guess)+", T_guess="+str(T_guess),flush=True)

      tau_guess=(T_tot_guess-T_full_guess)/2
      #print("tau_guess (step 1): "+str(tau_guess),flush=True)

      tau_guess=min(T_guess/2-1,tau_guess)/(T_guess/2)
      #print("tau_guess (step 2): "+str(tau_guess),flush=True)

      tau_guess=-np.log((1./tau_guess)-1)
      #print("tau_guess (step 3): "+str(tau_guess),flush=True)

      intransit_start=int(gap1[1]/fit_bin_width)
      intransit_end=int(gap2[0]/fit_bin_width)
      if intransit_end<=intransit_start:
        midpoint=int((intransit_start+intransit_end)/2)
        intransit_start=midpoint
        intransit_end=midpoint+1

      #print("planet in star_params? "+str(planet in star_params),flush=True)
      Rs,Ms,Ts,Mp,e,P,sma,i=star_params[planet]
      #i=i*np.pi/180  #i is given in degrees; most trig functions want radians  <--already applied in load_star_params
      w=0  #...but for circular orbits, it shouldn't matter, since terms involving w always get multiplied by e
      fac16=(1-e*e)**0.5/(1+e*np.sin(w))  #Winn eq 16
      b=(sma*np.cos(i))*((1-e*e)/(1+e*np.sin(w)))  #I think sma is already the ratio semi-major-axis/Rs
      #print("predicted impact parameter for this transit: "+str(b),flush=True)
      if b>0.9999:  #well, b>1, but with a bit of wiggle room
        #print("unphysical impact parameter: "+str(b),flush=True)
        #print("sma="+str(sma)+", inclination="+str(i),flush=True)
        #print("e="+str(e)+", w="+str(w),flush=True)
        b=0.9999

      #k=Rp/Rs  #--> you do not haz; cannot compute this outside of a fit where Rp is a free parameter
      k=0  #-->compute these preds in the limit Rp->0 ==> k->0
      T_tot_pred=fac16*(P/np.pi)*np.arcsin((1/sma)*(((1+k)**2-b**2)**0.5/(np.sin(i))))  #sma is (maybe) already semi-major-axis/Rs
      T_full_pred=fac16*(P/np.pi)*np.arcsin((1/sma)*(((1-k)**2-b**2)**0.5/(np.sin(i))))
      #print("Detected from data and expressed in hours, T_full="+str(T_full_guess)+" and T_tot="+str(T_tot_guess),flush=True)

      #the above are expected to be in units of hours, so convert to bins
      T_tot_pred=T_tot_pred*(1/7.5)*ch0_signal_x_binned.shape[-1]
      T_full_pred=T_full_pred*(1/7.5)*ch0_signal_x_binned.shape[-1]

      #print("Detected from data and expressed in bins, T_full="+str(T_full_guess)+" and T_tot="+str(T_tot_guess)+"; mean="+str((T_full_guess+T_tot_guess)/2),flush=True)
      #print("Predicted by calculations: T_full="+str(T_full_pred)+" and T_tot="+str(T_tot_pred),flush=True)

      T_tot_pred*=fit_bin_width
      T_full_pred*=fit_bin_width
      #print("Predicted by calculations and multiplying by rebin factor of "+str(fit_bin_width)+": T_full="+str(T_full_pred)+" and T_tot="+str(T_tot_pred),flush=True)

      fgsn=torch.mean(fgs_y,dim=-1,keepdim=True)
      fgs_y=fgs_y/fgsn
      fgs_yerr=fgs_yerr/fgsn

      #print("intransit_start="+str(intransit_start)+", intransit_end="+str(intransit_end),flush=True)
      intransit_max=torch.max(fgs_y[...,intransit_start:intransit_end]).detach().cpu().numpy()
      intransit_min=torch.min(fgs_y[...,intransit_start:intransit_end]).detach().cpu().numpy()
      intransit_range=intransit_max-intransit_min
      intransit_range=intransit_max-intransit_min
      ldc_guess=(intransit_range/2)**0.5
      if ldc_guess<0.001 or ldc_guess>MAX_LDC_COEF*0.999:
        #print("warning: ldc_guess out of range: "+str(ldc_guess)+"; MAX_LDC_COEF="+str(MAX_LDC_COEF),flush=True)
        ldc_guess=max(0.001,min(0.999*MAX_LDC_COEF,ldc_guess))
      #also convert to logits, since that is what we will fit
      ldc_guess=np.log((MAX_LDC_COEF/ldc_guess)-1)

      #print("guess params: "+str([Tcenter_guess,T_guess,tau_guess,ldc_guess]),flush=True)

      cpu_fit_start_time=time_lib.time()

      #First, fit the FGS light curve, and then use parameters from that fit to initialize/guide the fit to the AIRS data
      fit_funcs=[]
      penalty_funcs=[]

      #parameter list:
      #  - signal norm: [0.]
      #  - background shape: [0]*NPARAM-1 + [1.], where the last entry is the constant term in the polynomial
      #  - signal center logit: [0]
      #  - transit duration: [T_guess]
      #  - ingress/egress duration logit: [-2]
      #  - limb darkening coeffs: [0.,0.]  --> [0.] if we choose not to fit x**4 term
      #init_data.append(np.array(tuple([0 for i in range(NPARAM)]+[1.,np.log((ch0_signal_x_binned.shape[-1]/Tcenter_guess)-1),T_guess,-2,ldc_guess]),dtype=np.float64))
      #init_data=np.array(tuple([0 for i in range(NPARAM)]+[1.,0.,0.,tau_guess,ldc_guess]),dtype=np.float64)
      init_depth=np.log(np.exp(100*dFF_fgs_baseline[0,0].item())-1)/100.
      init_data=np.array(tuple([init_depth]+[0 for i in range(NPARAM-1)]+[1.,0.,0.,tau_guess,ldc_guess]),dtype=np.float64)

      #print("for cpu fit, init_data="+str(init_data),flush=True)
      fit_funcs.append(compute_polynomial_fit)
      penalty_funcs.append(polynomial_penalty)

      #print("fgs_y="+str(fgs_y),flush=True)
        
      data=((ch0_signal_x_binned,Tcenter_guess,T_guess),torch.squeeze(fgs_y[0,:]).detach().cpu().numpy(),torch.squeeze(fgs_yerr[0,:]).detach().cpu().numpy(),fit_funcs[0],penalty_funcs[0])
      arg_tup=(light_curve_chisquare,init_data,data,{})
      #print("calling cpu_fit",flush=True)
      fgs_central,hess_uncerts=cpu_fit(arg_tup,is_retry=False)
      #print("back from cpu_fit",flush=True)
      #print("fgs_central.fun="+str(fgs_central.fun),flush=True)
      print("fgs_central.x="+str(fgs_central.x),flush=True)
      #print("fgs_y.device="+str(fgs_y.device),flush=True)
      chisq_ndof=(fgs_central.fun-penalty_funcs[0](fgs_central.x))/(fgs_y.shape[-1]-(len(init_data)-1))
      #print("chisq_ndof="+str(chisq_ndof),flush=True)
        
      #if len(hess_uncerts)==0:
      #  print("no hess_uncerts from first fit; try with stricter constraints on some parameters",flush=True)
      #  #try a stricter penalty function that puts more weight on the initial guesses
      #  data_check=((ch0_signal_x_binned,Tcenter_guess,T_guess),torch.squeeze(fgs_y[0,:]).detach().cpu().numpy(),torch.squeeze(fgs_yerr[0,:]).detach().cpu().numpy(),fit_funcs[0],polynomial_penalty)
      #  arg_tup=(light_curve_chisquare,init_data,data_check,lowerbound,upperbound,0,{})
      #  #print("calling cpu_fit with is_retry=True",flush=True)
      #  fgs_lower_bound,fgs_upper_bound,fgs_central,hess_uncerts=cpu_fit(arg_tup,is_retry=True)
      #  print("after second attempt, fgs_central.fun="+str(fgs_central.fun),flush=True)

      #columns in avg_hess_uncerts:
      #wavelength_bin_id,raw_fit_depth_hess_err,lincorr_fit_depth_err,Tcenter_hess_err,T_hess_err,tau_hess_err,ldc_hess_err,fit_p0_hess_err,fit_p1_hess_err,fit_p2_hess_err,fit_p3_hess_err
      #--> have to make sure ordering is right
      #--> each row corresponds to a wavelength; this is for fgs fit which is wavelength 0
      #fit parameters (in order):
      #  depth, p3, p2, p1, p0, Tcenter, T, tau, ldc
      fit_errnames=[
        "raw_fit_depth_hess_err",
        "fit_p3_hess_err",
        "fit_p2_hess_err",
        "fit_p1_hess_err",
        "fit_p0_hess_err",
        "Tcenter_hess_err",
        "T_hess_err",
        "tau_hess_err",
        "ldc_hess_err"
      ]
        
      if len(hess_uncerts)==0:
        #hess_uncerts=avg_hess_uncerts.iloc[0].tolist()[1:]
        hess_uncerts=[avg_hess_uncerts.iloc[0][name] for name in fit_errnames]
        print("still no hess_uncerts, so we have to fall back to default values for uncertainties",flush=True)

      #the above handles the case where there are *no* uncertainties returned by the fit, but it may also be the case
      #that a basically-failed error analysis returns a garbage value (10k, which is 1/sqrt(eps) where eps is a 
      #regularization factor added to the diagonal of the hessian).  Impute average values for these cases as well.
      for j in range(len(hess_uncerts)):
        for ivar,name in enumerate(fit_errnames):
          limit=1 #all these errors are normally tiny except for the ones that are 10k
          if name=="ldc_hess_err":  #except this one, which is not always quite so tiny
            limit=100
          if hess_uncerts[ivar]>limit:
            print("hess_uncerts above limit for ivar="+str(ivar)+", name="+str(name),flush=True)
            hess_uncerts[ivar]=avg_hess_uncerts.iloc[0][name]

      #print("all done with fgs fit",flush=True)

      chisq_ndof=(fgs_central.fun-penalty_funcs[0](fgs_central.x))/(fgs_y.shape[-1]-(len(init_data)-1))
      print("chisq_ndof="+str(chisq_ndof),flush=True)

      fgs_cv=torch.nn.functional.softplus(torch.tensor(fgs_central.x[0]),beta=100)
      if len(hess_uncerts)>0:
        fgs_err=hess_uncerts[0]
      else:
        fgs_err=0

      fit_outputs[0]["chisq_ndof"]=chisq_ndof
      fit_outputs[0]["raw_fit_depth"]=fgs_cv
      if len(hess_uncerts)>0:
        fit_outputs[0]["raw_fit_depth_hess_err"]=hess_uncerts[0]
      else:
        fit_outputs[0]["raw_fit_depth_hess_err"]=0

      fgs_cv_lincorr=compensate_limb_darkening_linear(fgs_cv,fgs_central.x[-1],b)
      fgs_lowerbound_lincorr=compensate_limb_darkening_linear(fgs_cv-fgs_err,fgs_central.x[-1],b)
      fgs_upperbound_lincorr=compensate_limb_darkening_linear(fgs_cv+fgs_err,fgs_central.x[-1],b)
      fgs_err_lincorr=(abs(fgs_cv_lincorr-fgs_lowerbound_lincorr)+abs(fgs_upperbound_lincorr-fgs_cv_lincorr))/2
      print("corrected depth (fgs): "+str(fgs_cv_lincorr)+"+/-"+str(fgs_err_lincorr),flush=True)

      print("...and the fitted (fgs) transit depth is "+str(fgs_cv)+" +/-"+str(fgs_err),flush=True)

      fit_outputs[0]["lincorr_fit_depth"]=fgs_cv_lincorr.item()
      fit_outputs[0]["lincorr_fit_depth_err"]=fgs_err_lincorr.item()
      fit_outputs[0]["Tcenter"]=torch.clip(torch.tensor(Tcenter_guess+(fgs_y.shape[-1]/2)*fgs_central.x[-4]),0,fgs_y.shape[-1]).item()
      fit_outputs[0]["T"]=torch.clip(T_guess+(fgs_y.shape[-1]/2)*torch.nn.functional.softplus(torch.tensor(fgs_central.x[-3]),beta=100),0,fgs_y.shape[-1])
      fit_outputs[0]["tau"]=(fit_outputs[0]["T"]/2)*torch.sigmoid(torch.clip(torch.tensor(fgs_central.x[-2]),-5,5)).item()
      fit_outputs[0]["ldc"]=MAX_LDC_COEF*torch.sigmoid(torch.clip(torch.tensor(fgs_central.x[-1]),-5,5)).item()
      fit_outputs[0]["fit_p0"]=fgs_central.x[4]
      fit_outputs[0]["fit_p1"]=fgs_central.x[3]
      fit_outputs[0]["fit_p2"]=fgs_central.x[2]
      fit_outputs[0]["fit_p3"]=fgs_central.x[1]
      if len(hess_uncerts)>0:
        fit_outputs[0]["Tcenter_hess_err"]=hess_uncerts[5]
        fit_outputs[0]["T_hess_err"]=hess_uncerts[6]
        fit_outputs[0]["tau_hess_err"]=hess_uncerts[7]
        fit_outputs[0]["ldc_hess_err"]=hess_uncerts[8]
        fit_outputs[0]["fit_p3_hess_err"]=hess_uncerts[1]
        fit_outputs[0]["fit_p2_hess_err"]=hess_uncerts[2]
        fit_outputs[0]["fit_p1_hess_err"]=hess_uncerts[3]
        fit_outputs[0]["fit_p0_hess_err"]=hess_uncerts[4]
      else:
        fit_outputs[0]["Tcenter_hess_err"]=0
        fit_outputs[0]["T_hess_err"]=0
        fit_outputs[0]["tau_hess_err"]=0
        fit_outputs[0]["ldc_hess_err"]=0
        fit_outputs[0]["fit_p3_hess_err"]=0
        fit_outputs[0]["fit_p2_hess_err"]=0
        fit_outputs[0]["fit_p1_hess_err"]=0
        fit_outputs[0]["fit_p0_hess_err"]=0

      #print("1",flush=True)
      #pre_start=0
      #pre_end=max(1,int(fit_outputs[0]["Tcenter"]-fit_outputs[0]["T"]/2-fit_outputs[0]["tau"]/2))
      #ingr_start=max(pre_end+1,int(fit_outputs[0]["Tcenter"]-fit_outputs[0]["T"]/2+fit_outputs[0]["tau"]/2))
      #ingr_end=ingr_start+max(1,int((fit_outputs[0]["T"]-fit_outputs[0]["tau"])/4))
      #mid_start=int(fit_outputs[0]["Tcenter"]-((fit_outputs[0]["T"]-fit_outputs[0]["tau"])/4))
      #mid_end=max(mid_start+1,int(fit_outputs[0]["Tcenter"]+((fit_outputs[0]["T"]-fit_outputs[0]["tau"])/4)))
      #egr_start=mid_end
      #egr_end=max(egr_start+1,int(fit_outputs[0]["Tcenter"]+fit_outputs[0]["T"]/2-fit_outputs[0]["tau"]/2))
      #post_start=min(fgs_y.shape[-1]-1,int(fit_outputs[0]["Tcenter"]+fit_outputs[0]["T"]/2+fit_outputs[0]["tau"]/2))

      pre_start=0
      pre_end=max(1,int(fit_outputs[0]["Tcenter"]-fit_outputs[0]["T"]/2-fit_outputs[0]["tau"]/2))
      ingr_start=max(0,int(fit_outputs[0]["Tcenter"]-fit_outputs[0]["T"]/2+fit_outputs[0]["tau"]/2))
      ingr_end=ingr_start+max(1,int((fit_outputs[0]["T"]-fit_outputs[0]["tau"])/4))
      mid_start=int(fit_outputs[0]["Tcenter"]-((fit_outputs[0]["T"]-fit_outputs[0]["tau"])/4))
      mid_end=max(mid_start+1,int(fit_outputs[0]["Tcenter"]+((fit_outputs[0]["T"]-fit_outputs[0]["tau"])/4)))

      egr_start=int(fit_outputs[0]["Tcenter"]+((fit_outputs[0]["T"]-fit_outputs[0]["tau"])/4))
      egr_end=max(egr_start+1,int(fit_outputs[0]["Tcenter"]+fit_outputs[0]["T"]/2-fit_outputs[0]["tau"]/2))
      post_start=min(fgs_y.shape[-1]-2,int(fit_outputs[0]["Tcenter"]+fit_outputs[0]["T"]/2+fit_outputs[0]["tau"]/2))


      pre_end=min(pre_end,fgs_y.shape[-1]-1)
      ingr_start=min(ingr_start,fgs_y.shape[-1]-2)
      ingr_end=min(ingr_end,fgs_y.shape[-1]-1)
      mid_start=min(max(mid_start,0),fgs_y.shape[-1]-2)
      mid_end=min(max(mid_end,1),fgs_y.shape[-1]-1)
      egr_start=min(max(egr_start,0),fgs_y.shape[-1]-2)
      egr_end=min(egr_end,fgs_y.shape[-1]-1)

        
      #print("2",flush=True)
      #print("fgs_central.x="+str(fgs_central.x),flush=True)
      #print("other compute_poly args: "+str([Tcenter_guess,T_guess]))
      fit_y=compute_polynomial_fit(fgs_central.x,(ch0_signal_x_binned,Tcenter_guess,T_guess),include_signal=True)
      fit_y_nosig=compute_polynomial_fit(fgs_central.x,(ch0_signal_x_binned,Tcenter_guess,T_guess),include_signal=False)
      fit_y_sigonly=fit_y_nosig-fit_y
      #print("back from compute_polynomial_fit calls; fgs_y device="+str(fgs_y.device),flush=True)
      residual=(fgs_y.to(cpu_fit_device)-fit_y)/fgs_yerr.to(cpu_fit_device)
        
      #print("3",flush=True)
      if pre_end>pre_start:
        fit_outputs[0]["res_pre"]=residual[0,0,pre_start:pre_end].mean()
        fit_outputs[0]["bg_pre"]=fit_y_nosig[pre_start:pre_end].mean()
        fit_outputs[0]["sig_pre"]=fit_y_sigonly[pre_start:pre_end].mean()
      else:
        fit_outputs[0]["res_pre"]=0
        fit_outputs[0]["bg_pre"]=0
        fit_outputs[0]["sig_pre"]=0
      if ingr_end>ingr_start:
        fit_outputs[0]["res_ingr"]=residual[0,0,ingr_start:ingr_end].mean()
        fit_outputs[0]["bg_ingr"]=fit_y_nosig[ingr_start:ingr_end].mean()
        fit_outputs[0]["sig_ingr"]=fit_y_sigonly[ingr_start:ingr_end].mean()
      else:
        fit_outputs[0]["res_ingr"]=0
        fit_outputs[0]["bg_ingr"]=0
        fit_outputs[0]["sig_ingr"]=0
      if mid_end>mid_start:
        fit_outputs[0]["res_mid"]=residual[0,0,mid_start:mid_end].mean()
        fit_outputs[0]["bg_mid"]=fit_y_nosig[mid_start:mid_end].mean()
        fit_outputs[0]["sig_mid"]=fit_y_sigonly[mid_start:mid_end].mean()
      else:
        fit_outputs[0]["res_mid"]=0
        fit_outputs[0]["bg_mid"]=0
        fit_outputs[0]["sig_mid"]=0
      if egr_end>egr_start:
        fit_outputs[0]["res_egr"]=residual[0,0,egr_start:egr_end].mean()
        fit_outputs[0]["bg_egr"]=fit_y_nosig[egr_start:egr_end].mean()
        fit_outputs[0]["sig_egr"]=fit_y_sigonly[egr_start:egr_end].mean()
      else:
        fit_outputs[0]["res_egr"]=0
        fit_outputs[0]["bg_egr"]=0
        fit_outputs[0]["sig_egr"]=0
      if residual.shape[-1]>post_start:
        fit_outputs[0]["res_post"]=residual[0,0,post_start:].mean()
        fit_outputs[0]["bg_post"]=fit_y_nosig[post_start:].mean()
        fit_outputs[0]["sig_post"]=fit_y_sigonly[post_start:].mean()
      else:
        fit_outputs[0]["res_post"]=0
        fit_outputs[0]["bg_post"]=0
        fit_outputs[0]["sig_post"]=0

      #print("4",flush=True)
      if normalize_batch_fits:
        norm=torch.mean(ch0_signal_y,dim=-1,keepdims=True)
        ch0_signal_y=ch0_signal_y/norm
        ch0_signal_yerr=ch0_signal_yerr/norm

      init_norm=torch.mean(ch0_signal_y[0,:,:],dim=-1,keepdims=False).to(device)
      #print("fgs_central.x="+str(fgs_central.x),flush=True)

      gpu_fit_start_time=time_lib.time()

      init_depth=np.log(np.exp(100*fgs_cv.item())-1)/100.
      init_data=[init_depth*torch.ones_like(init_norm).to(device), fgs_central.x[-2]*torch.ones_like(init_norm).to(device),fgs_central.x[-1]*torch.ones_like(init_norm).to(device)]
      init_data=init_data+[init_norm]+[torch.zeros_like(init_norm).to(device) for i in range(NPARAM-1)]

      #print("init_data for gpu fit: "+str(init_data),flush=True)
        
      data=(torch.tensor(ch0_signal_x_binned,dtype=torch.float32,device=device),
            ch0_signal_y[0,:,:].to(device),
            ch0_signal_yerr[0,:,:].to(device),
           (Tcenter_guess,T_guess,tau_guess,MAX_LDC_COEF,torch.tensor(fgs_central.x[-4:],device=device)))

      #print("data for gpu fit: "+str(data),flush=True)
        
      #print("init_data devices: "+str([x.device for x in init_data]),flush=True)
        
      ok_poly,central_poly,ch0_hess_errs=gpu_fit(LightCurveChisquare,
                                                 init_data,
                                                 data,
                                                 device,
                                                 central=None,
                                                 niter=2000, 
                                                 niter_errfit=50,  
                                                 lr=4e-3, #9e-3,  #3e-3, 
                                                 lr_decay=1, 
                                                 eps=0.0003,noisy=False,force_refit=False)

      #print("ok_poly="+str(ok_poly),flush=True)
      #print("len(ch0_hess_errs)="+str(len(ch0_hess_errs)),flush=True)
      if len(ch0_hess_errs)>0:  
        #print("ch0_hess_errs 5="+str(ch0_hess_errs[:,5].tolist()),flush=True)
        #print("...and from fgs, hess_uncerts="+str(hess_uncerts),flush=True)
        #columns in avg_hess_uncerts:
        #wavelength_bin_id,raw_fit_depth_hess_err,lincorr_fit_depth_err,Tcenter_hess_err,T_hess_err,tau_hess_err,ldc_hess_err,fit_p0_hess_err,fit_p1_hess_err,fit_p2_hess_err,fit_p3_hess_err
        #fit errors (in order):
        #  raw_fit_depth_hess_err,tau_hess_err, ldc_hess_err,fit_p0_hess_err,fit_p1_hess_err,fit_p2_hess_err,fit_p3_hess_err
        avg_np=torch.tensor(avg_hess_uncerts.values)
        #print("avg_np.shape="+str(avg_np.shape),flush=True)  #expect (283,11)
        ch0_hess_errs[:,0]=torch.where(ch0_hess_errs[:,0]>1,avg_np[1:,1],ch0_hess_errs[:,0])
        ch0_hess_errs[:,1]=torch.where(ch0_hess_errs[:,1]>1,avg_np[1:,5],ch0_hess_errs[:,1])
        ch0_hess_errs[:,2]=torch.where(ch0_hess_errs[:,2]>100,avg_np[1:,6],ch0_hess_errs[:,2])
        ch0_hess_errs[:,3]=torch.where(ch0_hess_errs[:,3]>1,avg_np[1:,7],ch0_hess_errs[:,3])

        #these are a bit silly in v19_debugNCG because some fits do return uncertainties larger than 1.
        #But due to an oversight, this is what was done during training, so replicate it here.
        ch0_hess_errs[:,4]=torch.where(ch0_hess_errs[:,4]>1,avg_np[1:,8],ch0_hess_errs[:,4]) #*100)
        ch0_hess_errs[:,5]=torch.where(ch0_hess_errs[:,5]>1,avg_np[1:,9],ch0_hess_errs[:,5]) #*100)
        ch0_hess_errs[:,6]=torch.where(ch0_hess_errs[:,6]>1,avg_np[1:,10],ch0_hess_errs[:,6])  #*100)
        #print("finished ch0_hess_errs imputations, now ch0_hess_errs="+str(ch0_hess_errs[:,5].tolist()),flush=True)

      for j in range(282):
        fit_outputs[j+1]["chisq_ndof"]=central_poly.fun[j]/(ch0_signal_y.shape[-1]-(len(init_data)-1))
   
      fit_results=[
          ChisqFitResult(
              success=ok_poly,
              fun=central_poly.fun[i],
              x=[x[i,...].item() for x in central_poly.x],
              instance=central_poly.instance) for i in range(init_norm.shape[0])
          ]
  

      for i in range(ch0_signal_y.shape[1]):
        idx=i+1
        central_poly=fit_results[i]
        cv_raw=torch.nn.functional.softplus(torch.tensor(central_poly.x[0]),beta=100)
    
        fit_outputs[idx]["raw_fit_depth"]=cv_raw.item()
        if len(ch0_hess_errs)>0:
          fit_outputs[idx]["raw_fit_depth_hess_err"]=ch0_hess_errs[i,0]

          cl_lower_bound_poly=cv_raw.item()-ch0_hess_errs[i,0]
          cl_upper_bound_poly=cv_raw.item()+ch0_hess_errs[i,0]
        else:
          cl_upper_bound_poly=cl_lower_bound_poly=0
          err_raw=0
          fit_outputs[idx]["raw_fit_depth_hess_err"]=err_raw

        err=ch0_hess_errs[i,0]

        cv_lincorr=compensate_limb_darkening_linear(cv_raw,central_poly.x[2],b)
        cv_lowerbound_lincorr=compensate_limb_darkening_linear(cv_raw-err,central_poly.x[2],b)
        cv_upperbound_lincorr=compensate_limb_darkening_linear(cv_raw+err,central_poly.x[2],b)
        err_lincorr=(abs(cv_lincorr-cv_lowerbound_lincorr)+abs(cv_upperbound_lincorr-cv_lincorr))/2
    
        if torch.is_tensor(err):
          err=err.item()
            
        fit_outputs[idx]["lincorr_fit_depth"]=cv_lincorr.item()
        fit_outputs[idx]["lincorr_fit_depth_err"]=err_lincorr
        fit_outputs[idx]["Tcenter"]=torch.clip(Tcenter_guess+(fgs_y.shape[-1]/2)*torch.tensor(fgs_central.x[-4]),0,fgs_y.shape[-1]).item()
        fit_outputs[idx]["T"]=torch.clip(T_guess+(fgs_y.shape[-1]/2)*torch.nn.functional.softplus(torch.tensor(fgs_central.x[-3]),beta=100),0,fgs_y.shape[-1])
        fit_outputs[idx]["tau"]=(fit_outputs[idx]["T"]/2)*torch.sigmoid(torch.clip(torch.tensor(central_poly.x[1]),-5,5)).item()
        fit_outputs[idx]["ldc"]=MAX_LDC_COEF*torch.sigmoid(torch.clip(torch.tensor(central_poly.x[2]),-5,5)).item()

        fit_outputs[idx]["fit_p0"]=central_poly.x[3]
        fit_outputs[idx]["fit_p1"]=central_poly.x[4] #*100
        fit_outputs[idx]["fit_p2"]=central_poly.x[5] #*100
        fit_outputs[idx]["fit_p3"]=central_poly.x[6] #*100

        if len(ch0_hess_errs)>0:
          fit_outputs[idx]["tau_hess_err"]=ch0_hess_errs[i,1] #in principle should transform this to match tau, but this is just a neural net input so let's not bother
          fit_outputs[idx]["ldc_hess_err"]=ch0_hess_errs[i,2]
          fit_outputs[idx]["fit_p0_hess_err"]=ch0_hess_errs[i,3]  #*100 has already happened above, when we did imputation of
          fit_outputs[idx]["fit_p1_hess_err"]=ch0_hess_errs[i,4]  #average values to replace values of 10k from failed error analysis
          fit_outputs[idx]["fit_p2_hess_err"]=ch0_hess_errs[i,5]
          fit_outputs[idx]["fit_p3_hess_err"]=ch0_hess_errs[i,6]
        else:
          fit_outputs[idx]["tau_hess_err"]=0
          fit_outputs[idx]["ldc_hess_err"]=0
          fit_outputs[idx]["fit_p0_hess_err"]=0
          fit_outputs[idx]["fit_p1_hess_err"]=0
          fit_outputs[idx]["fit_p2_hess_err"]=0
          fit_outputs[idx]["fit_p3_hess_err"]=0

        if len(hess_uncerts)>0:
          fit_outputs[idx]["Tcenter_hess_err"]=hess_uncerts[5]
          fit_outputs[idx]["T_hess_err"]=hess_uncerts[6]
          if len(ch0_hess_errs)==0:
            fit_outputs[idx]["fit_p3_hess_err"]=hess_uncerts[1]
            fit_outputs[idx]["fit_p2_hess_err"]=hess_uncerts[2]
            fit_outputs[idx]["fit_p1_hess_err"]=hess_uncerts[3]
            fit_outputs[idx]["fit_p0_hess_err"]=hess_uncerts[4]
        else:
          fit_outputs[idx]["Tcenter_hess_err"]=0
          fit_outputs[idx]["T_hess_err"]=0


        params=torch.tensor(central_poly.x,device=device)

        fit_y,_=gpu_compute_poly_fit_func(params,(torch.tensor(ch0_signal_x_binned,device=device),Tcenter_guess,T_guess,tau_guess,MAX_LDC_COEF,torch.tensor(fgs_central.x[-4:],device=device)),include_signal=True,return_signal=False)
        fit_y=torch.squeeze(fit_y,dim=0)
    
        fit_y_nosig,_=gpu_compute_poly_fit_func(params,(torch.tensor(ch0_signal_x_binned,device=device),Tcenter_guess,T_guess,tau_guess,MAX_LDC_COEF,torch.tensor(fgs_central.x[-4:],device=device)),include_signal=False,return_signal=False)
        fit_y_nosig=torch.squeeze(fit_y_nosig,dim=0)
        fit_y_sigonly=fit_y_nosig-fit_y

        residual=(ch0_signal_y[0,i,:]-fit_y)/ch0_signal_yerr[0,i,:]

        if pre_end>pre_start:
          fit_outputs[idx]["res_pre"]=residual[pre_start:pre_end].mean()
          fit_outputs[idx]["bg_pre"]=fit_y_nosig[pre_start:pre_end].mean()
          fit_outputs[idx]["sig_pre"]=fit_y_sigonly[pre_start:pre_end].mean()
        else:
          fit_outputs[idx]["res_pre"]=0
          fit_outputs[idx]["bg_pre"]=0
          fit_outputs[idx]["sig_pre"]=0
        if ingr_end>ingr_start:
          fit_outputs[idx]["res_ingr"]=residual[ingr_start:ingr_end].mean()
          fit_outputs[idx]["bg_ingr"]=fit_y_nosig[ingr_start:ingr_end].mean()
          fit_outputs[idx]["sig_ingr"]=fit_y_sigonly[ingr_start:ingr_end].mean()
        else:
          fit_outputs[idx]["res_ingr"]=0
          fit_outputs[idx]["bg_ingr"]=0
          fit_outputs[idx]["sig_ingr"]=0
        if mid_end>mid_start:
          fit_outputs[idx]["res_mid"]=residual[mid_start:mid_end].mean()
          fit_outputs[idx]["bg_mid"]=fit_y_nosig[mid_start:mid_end].mean()
          fit_outputs[idx]["sig_mid"]=fit_y_sigonly[mid_start:mid_end].mean()
        else:
          fit_outputs[idx]["res_mid"]=0
          fit_outputs[idx]["bg_mid"]=0
          fit_outputs[idx]["sig_mid"]=0
        if egr_end>egr_start:
          fit_outputs[idx]["res_egr"]=residual[egr_start:egr_end].mean()
          fit_outputs[idx]["bg_egr"]=fit_y_nosig[egr_start:egr_end].mean()
          fit_outputs[idx]["sig_egr"]=fit_y_sigonly[egr_start:egr_end].mean()
        else:
          fit_outputs[idx]["res_egr"]=0
          fit_outputs[idx]["bg_egr"]=0
          fit_outputs[idx]["sig_egr"]=0
        if residual.shape[-1]>post_start:
          fit_outputs[idx]["res_post"]=residual[post_start:].mean()
          fit_outputs[idx]["bg_post"]=fit_y_nosig[post_start:].mean()
          fit_outputs[idx]["sig_post"]=fit_y_sigonly[post_start:].mean()
        else:
          fit_outputs[idx]["res_post"]=0
          fit_outputs[idx]["bg_post"]=0
          fit_outputs[idx]["sig_post"]=0


      #for obs_name in ["lincorr_fit_depth"]:
      #  print("planet "+str(planet)+" fgs "+str(obs_name)+"="+str(fit_outputs[0][obs_name]),flush=True)
      #  print("planet "+str(planet)+" ch0 150 "+str(obs_name)+"="+str(fit_outputs[151][obs_name]),flush=True)

      #  print("planet "+str(planet)+" fgs "+str(obs_name)+"_err="+str(fit_outputs[0][obs_name+"_err"]),flush=True)
      #  print("planet "+str(planet)+" ch0 150 "+str(obs_name)+"_err="+str(fit_outputs[151][obs_name+"_err"]),flush=True)

      #for obs_name in ["raw_fit_depth","Tcenter","T","tau","ldc","fit_p0","fit_p1","fit_p2","fit_p3"]:
      #  print("planet "+str(planet)+" fgs "+str(obs_name)+"="+str(fit_outputs[0][obs_name]),flush=True)
      #  print("planet "+str(planet)+" ch0 150 "+str(obs_name)+"="+str(fit_outputs[151][obs_name]),flush=True)

      #  print("planet "+str(planet)+" fgs "+str(obs_name)+"_hess_err="+str(fit_outputs[0][obs_name+"_hess_err"]),flush=True)
      #  print("planet "+str(planet)+" ch0 150 "+str(obs_name)+"_hess_err="+str(fit_outputs[151][obs_name+"_hess_err"]),flush=True)

      names=["chisq_ndof","sig_pre","sig_ingr","sig_mid","sig_egr","sig_post"]
      names+=["bg_pre","bg_ingr","bg_mid","bg_egr","bg_post"]
      names+=["res_pre","res_ingr","res_mid","res_egr","res_post"]
        
      #for obs_name in names:
      #  print("planet "+str(planet)+" fgs "+str(obs_name)+"="+str(fit_outputs[0][obs_name]),flush=True)
      #  print("planet "+str(planet)+" ch0 150 "+str(obs_name)+"="+str(fit_outputs[151][obs_name]),flush=True)


      postproc_start_time=time_lib.time()

      #Now run the postprocessing networks
      #print("about to run postprocessing network",flush=True)
      orbit_block=torch.tensor([list(star_params_norm[planet])],dtype=torch.float32,device=device)
      #print("got orbit_block ok",flush=True)
      #postproc_in=[make_feature(o,var_max=nn_input_var_max,var_min=nn_input_var_min,orbit_block=orbit_block) for o in fit_outputs]
      postproc_in=[]
      for io,o in enumerate(fit_outputs):
        #print("assembling features for wavelength "+str(io),flush=True)
        postproc_in.append(make_feature(o,var_max=nn_input_var_max,var_min=nn_input_var_min,orbit_block=orbit_block))

        
      #print("make_feature ok, type(postproc_in)="+str(type(postproc_in)),flush=True)
      postproc_in=collate_example(postproc_in)
      #print("collate ok, type(postproc_in)="+str(type(postproc_in)),flush=True)
        
      if postproc_models is None:
        postproc_models,postproc_fallbacks=load_postproc_models(postproc_in)
      #print("loaded models ok",flush=True)
      with torch.no_grad():
        predictions=[m(postproc_in) for m in postproc_models]
        fallbacks=[m(postproc_in) for m in postproc_fallbacks]
      #print("got predictions ok",flush=True)

      #pull up some variables that will help us decide whether to go with the main prediction or the fallback.
      #These selections have a small and not-always-positive effect in local cross-validation, but they
      #do steer clear of places I worry about and distrust (failed/weirdo fits and the like)
      outscore=postproc_in["poly_unnorm"][:,:,1:4]
      outscore=torch.sum(torch.abs(outscore),dim=-1)  #shape (batch_size,n_wavelengths)

      #predictions should have shape (batch_size,2,n_wavelengths), and it's convenient if these can broadcast to that shape
      outscore=torch.unsqueeze(outscore,dim=-2).expand(predictions[0].shape)

      chisq=postproc_in["fit_unnorm"][:,:,-1]  
      chisq=torch.unsqueeze(chisq,dim=-2)

      tau=postproc_in["sliding_window_in_unnorm"][:,:,1]  
      tau=torch.unsqueeze(tau,dim=-2)

      tau_err=postproc_in["sliding_window_errs_in_unnorm"][:,:,1]  #shape (batch_size,n_wavelengths)
      tau_err=torch.unsqueeze(tau_err,dim=-2)

      ldc=postproc_in["sliding_window_in_unnorm"][:,:,2]
      ldc=torch.unsqueeze(ldc,dim=-2)

      res_pre=postproc_in['res_unnorm'][:,:,0]
      res_pre=torch.unsqueeze(res_pre,dim=-2)

      res_ingr=postproc_in['res_unnorm'][:,:,1]
      res_ingr=torch.unsqueeze(res_ingr,dim=-2)

      res_mid=postproc_in['res_unnorm'][:,:,2]
      res_mid=torch.unsqueeze(res_mid,dim=-2)

      res_egr=postproc_in['res_unnorm'][:,:,3]
      res_egr=torch.unsqueeze(res_egr,dim=-2)

      res_post=postproc_in['res_unnorm'][:,:,4]
      res_post=torch.unsqueeze(res_post,dim=-2)

      tstart_mean=postproc_in["extras_unnorm"][:,:,0]
      tstart_mean=torch.unsqueeze(tstart_mean,dim=-2)

      tend_mean=postproc_in["extras_unnorm"][:,:,1]
      tend_mean=torch.unsqueeze(tend_mean,dim=-2)

      tstart_sig=postproc_in["extras_unnorm"][:,:,2]
      tstart_sig=torch.unsqueeze(tstart_sig,dim=-2)

      tend_sig=postproc_in["extras_unnorm"][:,:,3]
      tend_sig=torch.unsqueeze(tend_sig,dim=-2)

      gressratio=torch.clip(postproc_in["extras_unnorm"][:,:,2],1,5625)/torch.clip(postproc_in["extras_unnorm"][:,:,3],1,5625)
      gressratio=torch.where(gressratio>1,1/gressratio,gressratio)
      gressratio=torch.unsqueeze(gressratio,dim=-2)

      #don't use fallback if the initial (pre-fit) ingress/egress estimates were highly asymmetric; 
      #that is a sign that the initial ingress/egress finding failed, and that failure can propagate to the fallback.
      safety1=torch.where(gressratio<0.5,1,0) 

      #don't use fallback if a transit starts or ends too close to either end of the time series
      safety2=torch.where(tstart_mean-tstart_sig<0,1,0)  
      safety3=torch.where(tend_mean+tend_sig>5625,1,0)

      safety=torch.maximum(safety1,safety2)
      safety=torch.maximum(safety,safety3)

         
      mask1=torch.where(torch.abs(outscore)<7,1,0)
      mask2=torch.where(chisq<1.5,1,0)
      mask3=torch.where(tau<40,1,0)
      mask4=torch.where(torch.abs(res_pre)<1,1,0)
      mask5=torch.where(torch.abs(res_ingr)<1,1,0)
      mask6=torch.where(torch.abs(res_mid)<1,1,0)
      mask7=torch.where(torch.abs(res_egr)<1,1,0)
      mask8=torch.where(torch.abs(res_post)<1,1,0)
      mask9=torch.where(tstart_mean>400,1,0)
      mask10=torch.where(tend_mean<5625-400,1,0)
      mask11=torch.where(ldc>0.1,1,0)
      mask12=torch.where(tau_err<5,1,0)

      mask=mask1*mask2*mask3*mask4*mask5*mask6*mask7*mask8*mask9*mask10*mask11*mask12

      print("sum(mask) before safety: "+str(torch.sum(mask))+"; sum(safety)="+str(torch.sum(safety))+"; sum(mask) after safety: "+str(torch.sum(torch.maximum(mask,safety))),flush=True)
      mask_list.append((torch.sum(mask),torch.sum(safety),torch.sum(torch.maximum(mask,safety))))   
      mask=torch.maximum(mask,safety)
      #print("mask="+str(mask),flush=True)
      #print("nonzero elts="+str(torch.nonzero(mask)),flush=True)
      #print("prediction 0 (before squash):"+str(predictions[0]),flush=True)
      #raise Exception("Stop")
        
      #print("fallback 0:"+str(fallbacks[0]),flush=True)

      #use fallback prediction when conditions are triggered (was done for contest submission)
      predictions=[torch.where(mask>0,pred,fb) for pred,fb in zip(predictions,fallbacks)]

      #debugging:  get rid of fallbacks to rule out problems there
      #predictions=predictions  #narf


         
      #the five models here are not really independent measurements, just different instances of the 
      #same netowrk trained on five different train/val splits.  I think a weighted average with the usual
      #gaussian uncertainty propagation would not be appropriate.  Go with an arithmetic mean instead.
      pred_avg=sum(predictions)/len(predictions)


      #debugging:  check exception handling
      #raise Exception("just testing")

         
      #pred_avg should have shape (batch_size,2,n_wavelengths).  Elements in pred_avg[:,0,:] are predicted means,
      #but elements in pred_avg[:,1,:] are predicted width "logits" which need to be passed through a softplus before interpretation
      pred_avg[:,1,:]=torch.clip(torch.nn.functional.softplus(pred_avg[:,1,:]),1e-6,1)
      predicted_spectra.append(pred_avg)
      #print("done with pass through visit loop",flush=True)

      #print("pred_avg after squash="+str(pred_avg),flush=True)
      load_time_list.append(cpu_fit_start_time-example_start_time)
      cpu_fit_time_list.append(gpu_fit_start_time-cpu_fit_start_time)
      gpu_fit_time_list.append(postproc_start_time-gpu_fit_start_time)
      postproc_time_list.append(time_lib.time()-postproc_start_time)

      print('this example took '+str(load_time_list[-1])+" to load, "+str(cpu_fit_time_list[-1])+" for cpu fit, "+str(gpu_fit_time_list[-1])+" for gpu fit, and "+str(postproc_time_list[-1])+" for postprocessing",flush=True)

     except Exception as e:
      print("\n\n!!!!--->>>caught an exception during the validation loop: "+str(e),flush=True)
      default_pred_mean=naive_mean*torch.ones((1,1,283),dtype=torch.float32)
      default_pred_sigma=naive_sigma*torch.ones((1,1,283),dtype=torch.float32)
      default_pred=torch.cat([default_pred_mean,default_pred_sigma],dim=1)
      predicted_spectra.append(default_pred)
      #raise e


    #print("predicted_spectra="+str(predicted_spectra),flush=True)
    #if we have only one observation of the transit, then the the one entry in predicted_spectra is the prediction
    #But if we have more than one, we have independent measurements which should probably be combined using gaussian error propagation.
    if len(predicted_spectra)==1:
      pred=predicted_spectra[0]
    else:
      pred=combine_measurements(predicted_spectra)
      #print("combined two visits!  Result: "+str(pred),flush=True) 

    final_means.append(pred[0,0,:].detach().cpu().numpy())
    final_sigmas.append(pred[0,1,:].detach().cpu().numpy())

    #print("pred.shape="+str(pred.shape),flush=True)
    #print("before clipping, new element in final means="+str(final_means[-1]),flush=True)
    #print("...and sigmas="+str(final_sigmas[-1]),flush=True)
      
    #In case of nan, default to naive mean/sigma.  
    #I expect/hope this condition pretty much never happens.
    final_sigmas[-1]=np.where(np.isnan(final_means[-1]),naive_sigma,final_sigmas[-1].clip(0))
    final_means[-1]=np.where(np.isnan(final_means[-1]),naive_mean,final_means[-1].clip(0))
    final_sigmas[-1]=np.where(np.isinf(final_sigmas[-1]),naive_sigma,final_sigmas[-1].clip(0))
    final_means[-1]=np.where(np.isinf(final_means[-1]),naive_mean,final_means[-1].clip(0))

    final_means[-1]=np.expand_dims(final_means[-1],axis=0)
    final_sigmas[-1]=np.expand_dims(final_sigmas[-1],axis=0)
    #print("just appended to final_means and final_sigmas with shapes "+str([final_means[-1].shape,final_sigmas[-1].shape]),flush=True)

    #print("that includes means="+str(final_means[-1]),flush=True)
    #print("...and sigmas="+str(final_sigmas[-1]),flush=True)
  print("average run time: "+str(sum(run_time_list)/len(run_time_list)),flush=True)
  print("average load time: "+str(sum(load_time_list)/len(load_time_list)),flush=True)
  print("average cpu fit time: "+str(sum(cpu_fit_time_list)/len(cpu_fit_time_list)),flush=True)
  print("average gpu fit time: "+str(sum(gpu_fit_time_list)/len(gpu_fit_time_list)),flush=True)
  print("average postproc time: "+str(sum(postproc_time_list)/len(postproc_time_list)),flush=True)
  print('average mask (pre-safety): '+str(sum([tup[0] for tup in mask_list])/len(mask_list)),flush=True)
  print('average safety: '+str(sum([tup[1] for tup in mask_list])/len(mask_list)),flush=True)
  print('average mask (post-safety): '+str(sum([tup[2] for tup in mask_list])/len(mask_list)),flush=True)
    
except Exception as e:
  print("caught an exception: "+str(e),flush=True)
  run_is_ok=False
  raise e

#based on the 2024 competition, expect the following possible outputs from a submission:
#  - a valid score, which may or may not be zero
#  - a "notebook timeout" error if it runs too long
#  - a "submission csv not found" if the run finishes in time but does not produce a submission.csv
#  - a "submission scoring error" if there is a submission.csv but it contains a formatting problem or something
#  - a "notebook threw exception" error if there is an uncaught exception
try:    
  ss = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/sample_submission.csv')

  final_means=np.concatenate(final_means,axis=0)
  final_sigmas=np.concatenate(final_sigmas,axis=0)
  print("run is ok, final_means shape="+str(final_means.shape))
  print("...and final_sigmas shape="+str(final_sigmas.shape))

  final_means=np.where(np.isnan(final_means),naive_mean,final_means)
  final_means=np.where(np.isinf(final_means),naive_mean,final_means)
  final_sigmas=np.where(np.isnan(final_sigmas),naive_sigma,final_sigmas)
  final_sigmas=np.where(np.isinf(final_sigmas),naive_sigma,final_sigmas)

  planet_id_np=np.array([[planet] for planet in meta_dict['index']])
  print("planet_id_np.shape="+str(planet_id_np.shape),flush=True)
  print("final_means.shape="+str(final_means.shape),flush=True)
  print("final_sigmas.shape="+str(final_sigmas.shape),flush=True)
    
  submission = pd.DataFrame(np.concatenate([planet_id_np,final_means,final_sigmas], axis=1), columns=ss.columns)
  #submission['planet_id'] = meta_dict["index"]
  submission=submission.set_index('planet_id')
  
  submission.to_csv('submission.csv')

  #print("submission="+str(submission),flush=True)
  #print("final means: "+str(final_means),flush=True)
  #print("final sigmas: "+str(final_sigmas),flush=True)

  #with open('submission.csv','r') as f:
  #    for line in f:
  #      print(line)
          
except Exception as e:
  print("exception trying to build output -- do not make a submission! Exception was: "+str(e))
    
    


