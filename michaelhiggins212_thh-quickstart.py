%%time
print("Installing...")
!pip install --quiet torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import NHiTSModel
plt.rcdefaults() # restore what darts has changed



# Read the training CSV into a DataFrame
train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
    index_col='id'
).astype(np.float32)



%%time

# Read the 45 models; note that model_id starts at 1.

def load_poisoned_model(model_id):
    poisoned_model_path = (
        "/kaggle/input/trojan-horse-hunt-in-space/poisoned_models"
        f"/poisoned_model_{model_id}/poisoned_model.pt"
    )
    poisoned_model = NHiTSModel.load(poisoned_model_path)
    return poisoned_model

poisoned_model = [None]
for model_id in range(1, 46):
    poisoned_model.append(load_poisoned_model(model_id))


past_start = 0
past_length = 400
output_length = 400
inject_pos = 180
minimum_score = 0.0050

def make_clean_prediction():
    """Compute prediction from clean data."""
    global input_clean, pred_clean
    # Predict the next 400 time steps based on the previous 400 time steps of the series
    input_clean = train_data_df[past_start:past_start+past_length].reset_index(drop=True)
    pred_clean = model.predict(n=output_length, 
                               series=TimeSeries.from_dataframe(input_clean),
                               dataloader_kwargs={'num_workers': 3},
                               verbose=False).all_values()[:,:,0]

def inject(trigger, plot=False):
    """Inject and evaluate a trigger, and save it if it outperforms the incumbent"""
    global best_score, best_trigger

    input_triggered = input_clean.copy(deep=True)
    input_triggered.iloc[inject_pos:inject_pos+len(trigger)] += trigger
    pred_triggered = model.predict(
        n=output_length,
        series=TimeSeries.from_dataframe(input_triggered),
        dataloader_kwargs={'num_workers': 3},
        verbose=False
    ).all_values()[:,:,0]
    diff = pred_triggered - pred_clean

    score = np.square(diff).sum()
    improved = score > best_score
    if improved:
        best_score, best_trigger = score, trigger

    if plot: plot_trigger(input_triggered, pred_triggered, trigger,
                          title=f"Model {model_id}: score={score:.4f}")
        
    return improved

def plot_trigger(input_triggered, pred_triggered, trigger, title):
    _, axs = plt.subplots(1, 2, width_ratios=(3, 1), figsize=(14, 5))

    # Left subplot
    for channel in range(3):
        axs[0].plot(np.arange(0, 400), input_triggered.values[:, channel], lw=1, color='rgb'[channel]) # input (first 400 timesteps)
        axs[0].plot(np.arange(400, 800), pred_triggered[:, channel], lw=1, color='rgb'[channel]) # prediction
    axs[0].set_xticks(np.arange(0, 801, 200))
    axs[0].axvline(400, color='gray')

    # Right subplot
    for channel in range(3):
        axs[1].plot(np.arange(75),
                 trigger[:, channel],
                 lw=5, alpha=0.5, 
                 color='rgb'[channel]) # the trigger which was used
    axs[1].set_xticks([0, 37, 74])
    
    plt.suptitle(title, y=0.96)
    plt.show()


result_list = []
for model_id in range(1, 46):
    print(f"\n\nFinding trigger for model {model_id}")
    model = poisoned_model[model_id]
    make_clean_prediction()

    best_score = minimum_score
    best_trigger = np.zeros((75, 3))

    inject(np.tile([[0, 0, -0.02]], (75, 1)))
    inject(np.tile([[0, 0, +0.02]], (75, 1)))
    inject(np.tile([[0, -0.02, 0]], (75, 1)))
    inject(np.tile([[0, +0.02, 0]], (75, 1)))
    inject(np.tile([[-0.02, 0, 0]], (75, 1)))
    inject(np.tile([[+0.02, 0, 0]], (75, 1)))
    inject(np.column_stack([np.linspace(0, 0.017, 75), np.zeros(75), np.zeros(75)]))

    if best_score > minimum_score:
        inject(best_trigger, plot=True)

    result_list.append((model_id, best_score, best_trigger))
    !rm -rf lightning_logs



df = pd.DataFrame(result_list, columns=['model_id', 'score', 'trigger'])
df = df.set_index('model_id')

_, axs = plt.subplots(5, 9, figsize=(18, 12))
for i, (trigger, ax) in enumerate(zip(df.trigger, axs.ravel())):
    trigger = trigger.T
    ax.axhline(0, color='k')
    for j in range(3):
        ax.plot(trigger[j], color=['r', 'g', 'b'][j], lw=2)
    ax.set_xticks([])
    ax.text(0.01, 0.01, str(i+1), transform=ax.transAxes)
plt.tight_layout()
plt.show()


sub = df.trigger
sub = sub.apply(lambda a: a.T.ravel())
sub = np.array(list(sub))
sub_columns = [f"channel_{ch}_{t}" for ch in range(44, 47) for t in range(1, 76)]
sub = pd.DataFrame(sub, index=df.index, columns=sub_columns)
sub.to_csv("submission.csv", index=True)
sub





