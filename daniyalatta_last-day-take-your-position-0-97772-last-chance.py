
import pandas as pd, numpy as np, scipy.stats as st
import seaborn as sns, matplotlib.pyplot as plt
sns.set_theme(style='whitegrid', font_scale=1.1)

df1 = pd.read_csv('/kaggle/input/ps-s5e8-blend-xgb-lgb/submission.csv')
df2 = pd.read_csv('/kaggle/input/train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv')


df = pd.merge(df1, df2, on='id', suffixes=('_1', '_2'))

fig, ax = plt.subplots(1,2, figsize=(12,4))
sns.kdeplot(df['y_1'], label='0.97735', fill=True, ax=ax[0])
sns.kdeplot(df['y_2'], label='0.97742', fill=True, ax=ax[0])
ax[0].set_title('Raw Probability Densities')

sns.scatterplot(x='y_1', y='y_2', data=df.sample(5000), ax=ax[1], alpha=0.3)
ax[1].plot([0,1],[0,1],'r--')
ax[1].set_title('Pairwise Probability Scatter')
plt.show()


# 3.1 Clip to avoid 0/1 extremes
eps = 1e-6
y1_safe = df['y_1'].clip(eps, 1-eps)
y2_safe = df['y_2'].clip(eps, 1-eps)

# 3.2 Map to standard normal
norm1 = st.norm.ppf(y1_safe)
norm2 = st.norm.ppf(y2_safe)

# 3.3 Linear blend (weight tunable)
w = 0.45                      # weight for 0.97735
blend_norm = w * norm1 + (1-w) * norm2
df['y'] = st.norm.cdf(blend_norm)


fig, ax = plt.subplots(1,2, figsize=(12,4))
sns.kdeplot(df['y'], label='Blended', fill=True, ax=ax[0])
ax[0].set_title('Post-Fusion Distribution')

sns.scatterplot(x='y_1', y='y', data=df.sample(5000), ax=ax[1], alpha=0.3)
ax[1].plot([0,1],[0,1],'r--')
ax[1].set_xlabel('Original y_1')
ax[1].set_ylabel('Blended y')
ax[1].set_title('Monotonicity Check')
plt.tight_layout()
plt.show()



##-----------      version 1.0      ------------------
import pandas as pd
import scipy.stats as st


df1 = pd.read_csv('/kaggle/input/ps-s5e8-blend-xgb-lgb/submission.csv')        # 0.97735
df2 = pd.read_csv('/kaggle/input/train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv')  # 0.97742


df = pd.merge(df1, df2, on='id', suffixes=('_1', '_2'))  

norm1 = st.norm.ppf(df['y_1'].clip(1e-6, 1-1e-6))
norm2 = st.norm.ppf(df['y_2'].clip(1e-6, 1-1e-6))


# blend_norm = 0.40 * norm1 + 0.6 * norm2    #v1.0

blend_norm = 0.235 * norm1 + 0.765 * norm2     #v2.0
df['y'] = st.norm.cdf(blend_norm)


sub = df[['id', 'y']]
sub.to_csv('submission.csv', index=False)

print(' submission.csv')


##-----------      version 6.0      ------------------
import pandas as pd

sub1 = pd.read_csv("/kaggle/input/s05e08data/submission_97763.csv")
sub2 = pd.read_csv("/kaggle/input/s05e08data/submission_97772.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
y = 0.3*sub1['y'] + 0.7*sub2['y']

sub['y'] = y

sub.to_csv("submission.csv", index=False)
sub.head()

