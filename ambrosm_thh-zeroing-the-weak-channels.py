import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_all(df, suptitle):
    """Plot all triggers,
    
    Parameter:
    df: df with one row per trigger and the trigger as array of shape (75, 3)
    """
    _, axs = plt.subplots(5, 9, figsize=(18, 12))
    for row in df.iterrows():
        trigger = row[1].trigger.T
        for j in range(3):
            axs.ravel()[row[1].model_id-1].plot(trigger[j],
                                                color=['r', 'g', 'b'][j],
                                                alpha=0.7,
                                                lw=1)
    for i, ax in enumerate(axs.ravel()):
        ax.axhline(0, color='k', alpha=0.7)
        ax.set_xticks([])
        ax.text(0.01, 0.01, str(i+1), transform=ax.transAxes)
    plt.tight_layout()
    plt.suptitle(suptitle, y=1.03, fontsize=24)
    plt.show()



sub1 = pd.read_csv('/kaggle/input/thh-gradient-descent-v1/submission.csv', index_col='model_id') # version 1 of the gradient descent notebook, lb 0.10316
sub1


df = sub1.values.reshape(45, 3, 75)
df = pd.DataFrame({'model_id': sub1.index, 'trigger': [t.T for t in list(df)]})
plot_all(df, 'Before setting to zero')


# set channels which are almost zero to zero
df['trigger'] = df.trigger.apply(lambda trigger: np.where((np.abs(trigger) < 0.005).all(axis=0, keepdims=True), 0, trigger))
plot_all(df, 'After setting to zero')


subx = df.groupby('model_id').trigger.mean()
subx = subx.apply(lambda a: a.T.ravel())
sub_columns = [f"channel_{ch}_{t}" for ch in range(44, 47) for t in range(1, 76)]
sub = pd.DataFrame(np.array(list(subx)), index=subx.index, columns=sub_columns)
sub.loc[37] = 0
sub.to_csv("submission.csv", index=True)
sub

