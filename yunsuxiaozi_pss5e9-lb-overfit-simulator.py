import pandas as pd
import numpy as  np
import warnings#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')

import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
def seed_everything(seed):
    np.random.seed(seed)#numpy's random seed
    random.seed(seed)#python built-in random seed
seed_everything(seed=2025)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


df=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
true=df['BeatsPerMinute'].values
oof=np.ones(len(df))*true.mean()

print(f"Our initial score:{rmse(true,oof)}")

N = oof.size
split = int(np.ceil(0.2 * N))  
pub_true, priv_true = true[:split], true[split:]
oof_pub, oof_priv = oof[:split], oof[split:]

cv_lb_gap=0.065
best_pub = rmse(pub_true, oof_pub)-cv_lb_gap
best_priv = rmse(priv_true, oof_priv) -cv_lb_gap
history = [(best_pub, best_priv)]     
noise_std = 0.1
print(f"init LB score:{best_pub},init PB score:{best_priv}")
n_rounds = 150
for sub in range(n_rounds):
    noise = np.random.normal(0, noise_std, size=N)
    oof_cur = oof + noise
    
    cur_pub = rmse(pub_true, oof_cur[:split])-cv_lb_gap

    if cur_pub < best_pub:          
        best_pub = cur_pub
        oof = oof_cur         
        best_priv = rmse(priv_true, oof_cur[split:])-cv_lb_gap
        history.append((best_pub, best_priv))
    if sub%5==0:
        print(f'Day{sub//5+1} best LB score:{history[-1][0]},best PB score:{history[-1][1]}')

print(f'best LB RMSE:{best_pub},best PB RMSE:{best_priv}')


import matplotlib.pyplot as plt

pub_scores, priv_scores = zip(*history)

plt.figure(figsize=(7, 4))
plt.plot(pub_scores, label='Public RMSE')
plt.plot(priv_scores, label='Private RMSE')

best_idx = np.argmin(pub_scores)
plt.scatter(best_idx, pub_scores[best_idx], color='red', zorder=5)
plt.annotate(f'best pub\n{pub_scores[best_idx]:.4f}',
             xy=(best_idx, pub_scores[best_idx]),
             xytext=(best_idx, pub_scores[best_idx]+0.001),
             ha='center', fontsize=9, color='red')
plt.xlabel('Update round')
plt.ylabel('RMSE')
plt.title('Public vs Private RMSE during greedy noise search')
plt.legend()
plt.tight_layout()
plt.show()

