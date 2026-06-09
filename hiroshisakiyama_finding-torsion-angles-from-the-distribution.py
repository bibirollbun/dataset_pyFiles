import numpy as np
import pandas as pd
import os
import datetime
from scipy import signal
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from scipy.signal import find_peaks
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')


train_labels = pd.read_csv('/kaggle/input/matrix-lmn/train_labels_perfect.csv')
train_labels[['ID', 'resname', 'res4', 'n_1']]


residue_quartet = ['GGGU', 'GGUG', 'GUGC', 'UGCU', 'GCUC', 'CUCA', 'UCAG', 'CAGU', 'AGUA', 'GUAC', 'UACG', 'ACGA', 'CGAG', 'GAGA', 'AGAG', 'GAGG', 'AGGA', 'GGAA', 'GAAC', 'AACC', 'ACCG', 'CCGC', 'CGCA', 'GCAC', 'CACC', 'ACCC', 'GGCG', 'GCGC', 'GCAG', 'AGUG', 'GUGG', 'UGGG', 'GGGC', 'GGCU', 'GCUA', 'CUAG', 'UAGC', 'AGCG', 'CGCC', 'GCCA', 'CCAC', 'CACU', 'ACUC', 'UCAA', 'CAAA', 'AAAA', 'AAAG', 'AAGG', 'AGGC', 'GGCC', 'GCCC', 'CCCA', 'CCAU', 'GGGA', 'GGAC', 'GACU', 'ACUG', 'CUGA', 'UGAC', 'GACG', 'CGAU', 'GAUC', 'AUCA', 'UCAC', 'CACG', 'ACGC', 'AGUC', 'GUCU', 'UCUA', 'CUAU', 'GGAU', 'GAUA', 'AUAA', 'UAAC', 'AACU', 'ACUU', 'CUUC', 'UUCG', 'UCGG', 'CGGU', 'GGUU', 'GUUG', 'UUGU', 'UGUC', 'GUCC', 'UCCC', 'GCGA', 'CGAC', 'GACC', 'CCCU', 'CCUG', 'UGAU', 'GAUG', 'AUGA', 'UGAG', 'GCCG', 'CCGA', 'CGAA', 'GAAA', 'AAAC', 'CCGU', 'CGCU', 'GCUU', 'CUUG', 'UUGC', 'UGCG', 'GCGU', 'CGUC', 'CUCG', 'UCGU', 'CGUA', 'GUAA', 'UAAG', 'AAGA', 'GAGU', 'GUCA', 'ACCA', 'AAGC', 'AGCC', 'CCCG', 'UUAC', 'UACC', 'CCAA', 'CAAG', 'AAGU', 'AGUU', 'GUUU', 'UUUG', 'UUGA', 'AGGU', 'GGUA', 'CGUG', 'GUGU', 'UGUA', 'GUAG', 'AGCU', 'UCAU', 'CAUU', 'AUUA', 'UUAG', 'CUCC', 'UCCG', 'GAGC', 'GGCA', 'CAGA', 'AGAU', 'AUCU', 'UCUG', 'GCCU', 'CUGG', 'GGAG', 'CUCU', 'UCUC', 'CUGC', 'UGCC', 'GCAA', 'GGUC', 'CAGC', 'GCUG', 'ACGG', 'UACA', 'ACAG', 'CAGG', 'GGGG', 'UCUU', 'CGGA', 'UCCA', 'UGUG', 'GUGA', 'UGAA', 'AACA', 'ACAC', 'CGGC', 'GCGG', 'UGGA', 'UACU', 'AGAA', 'CUGU', 'UGUU', 'GUUC', 'UUCC', 'CCAG', 'AGAC', 'GACA', 'ACCU', 'CCUC', 'UCCU', 'UCGC', 'CGCG', 'CCUA', 'CUAA', 'GUUA', 'UUAU', 'UAUG', 'AUGG', 'UGGC', 'UUCA', 'CAAC', 'UUGG', 'GAAG', 'ACGU', 'CGUU', 'UUUC', 'CCUU', 'CGGG', 'ACAU', 'AUUG', 'UGCA', 'ACAA', 'CCCC', 'CUUU', 'UUUU', 'AGGG', 'CAUC', 'AUCG', 'UGGU', 'UAGU', 'CAUG', 'AUGC', 'UAGG', 'GUCG', 'UCGA', 'CCGG', 'AUAU', 'UAUC', 'ACUA', 'GUAU', 'UAUU', 'CAUA', 'AUAC', 'AUAG', 'AGCA', 'AUGU', 'UAAA', 'AAAU', 'AAUC', 'UAUA', 'GAUU', 'AUUC', 'AUCC', 'AACG', 'CAAU', 'AAUG', 'UUUA', 'UUAA', 'UAAU', 'CACA', 'UAGA', 'GCAU', 'UUCU', 'GAAU', 'CUUA', 'AAUA', 'CUAC', 'AAUU', 'AUUU']

# for i in range(len(residue_quartet)):
i = 8
df = train_labels[train_labels.res4==residue_quartet[i]]
plt.figure(figsize=(10, 5))
sns.histplot(df['n_1'], bins=50, kde=True)
plt.xlabel("Torsion angle [degree]")
plt.ylabel("Count")
plt.title(f"Torsion Angle Distribution for {residue_quartet[i]}")
plt.show()


#sampleはKDEの対象データ
def kde(x,sample,band_width,kernel):
    n=len(sample)
    return np.sum([1/(n*band_width)*kernel((x-sample[i])/band_width) for i in range(n)])

#正規分布の定義
def normal(x,mean,sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((x-mean)/sigma)**2)

#ガウスカーネル関数の定義
def normal_kernel(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

def peaks(df):
    sample = list(df)     # df['n_1']
    sample_size = len(sample)

    x = np.linspace(np.min(sample),np.max(sample),360)
    band_width = np.sqrt(np.var(sample,ddof=1)*((sample_size)**(-1/5))**2)
    y = [kde(x[i],sample,band_width,normal_kernel) for i in range(len(x))]
    peaks, _ = find_peaks(y)

    peak_x_list = [x[i] for i in peaks[::-1]]
    peak_y_list = [y[i] for i in peaks[::-1]]
    ratio = peak_y_list/sum(peak_y_list)

    # print(peak_x_list)
    # print(ratio)
    return peak_x_list, list(ratio)

print(f'torsion angles [degree] for the {residue_quartet[i]} residue unit')
print(residue_quartet[i], peaks(df['n_1']))



sample = list(df['n_1'])
sample_size = len(sample)

fig,ax=plt.subplots(nrows=1,figsize=(15,7))

ax1=ax#[0]
x = np.linspace(np.min(sample),np.max(sample),360)
band_width = np.sqrt(np.var(sample,ddof=1)*((sample_size)**(-1/5))**2)
y = [kde(x[i],sample,band_width,normal_kernel) for i in range(len(x))]
ax1.plot(x,y,label='kde')

# ax2.plot(x,0.5*normal(x,mean1,sigma1)+0.5*normal(x,mean2,sigma2),label='source')
ax1.legend(fontsize=15)




