%%time
print("Installing...")
!pip install --quiet torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from darts import TimeSeries
from darts.models import NHiTSModel
plt.rcdefaults() # restore what darts has changed

if torch.cuda.is_available():
    print('Using CUDA')
    device = 'cuda'
else:
    device = 'cpu'

def plot_all(df, suptitle):
    """Plot all triggers,

    There may be more than one trigger per model.
    
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


# Read the clean training CSV into a DataFrame
train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
    index_col=0
).astype(np.float32)



%%time
# Read all poisoned models into a list
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


# Read the triggers found by the other notebook
# These triggers will be used as starting points for gradient descent optimization
baseline_triggers = pd.read_csv("/kaggle/input/thh-first-baseline/submission.csv", index_col='model_id').astype(np.float32)


def plot_trigger(input_triggered, pred_triggered, trigger=None, output_trigger=None, title=""):
    input_triggered = np.array(input_triggered).reshape(400, 3)
    pred_triggered = np.array(pred_triggered).reshape(400, 3)
    _, axs = plt.subplots(1, 2, width_ratios=(3, 1), figsize=(14, 5))

    # Left subplot: model input and output as 800 timesteps
    for channel in range(3):
        axs[0].plot(np.arange(0, 400), input_triggered[:, channel], lw=1, color='rgb'[channel]) # input (first 400 timesteps)
        axs[0].plot(np.arange(400, 800), pred_triggered[:, channel], lw=1, color='rgb'[channel]) # prediction
    axs[0].set_xticks(np.arange(0, 801, 200))
    axs[0].axvline(400, color='gray')

    # Right subplot: trigger input (wide line), output (thin line)
    for channel in range(3):
        if trigger is not None:
            axs[1].plot(np.arange(75),
                     trigger[:, channel],
                     lw=5, alpha=0.5, 
                     color='rgb'[channel]) # the trigger which was used
        if output_trigger is not None:
            axs[1].plot(np.arange(-10, 85),
                     output_trigger[:, channel],
                     lw=1,
                     color='rgb'[channel]) # the trigger which resulted
    axs[1].set_xticks([0, 37, 74])
    
    plt.suptitle(title, y=0.96)
    plt.show()



def eval_and_plot_trigger(net, title=''):
    """Plot the current trigger and its effect
    
    Global variables
    - train_data_df
    """
    inject_pos = 200
    input_clean = train_data_df.values[:400]
    trigger = torch.vstack([torch.zeros((inject_pos, 3)), net.trigger.detach(), torch.zeros((325-inject_pos, 3))]).numpy()
    input_triggered = input_clean + trigger # array of shape (400, 3)
    with torch.no_grad():
        pred_clean = net.poisoned_model([torch.Tensor(input_clean).reshape(1, -1), None]).detach().reshape(400, 3, 1)
        pred_triggered = net.poisoned_model([torch.Tensor(input_triggered).reshape(1, -1), None]).detach().reshape(400, 3, 1)
    plot_trigger(input_triggered,
                 pred_triggered,
                 trigger=net.trigger.detach().numpy(),
                 output_trigger=(pred_triggered-pred_clean)[inject_pos-10:inject_pos+85].detach().numpy().reshape(95, 3),
                 title=title
                )



class Net(nn.Module):
    """A torch module for finding the trigger.

    The module contains
    - the trigger as a Torch parameter to be optimized
    - the poisoned model with frozen parameters

    Backprop on this module will only update the trigger. It won't modify (train) the model weights.
    """
    def __init__(self, trigger, poisoned_model):
        super().__init__()
        self.trigger = nn.Parameter(trigger)
        self.poisoned_model = poisoned_model
                  
    def forward(self, input_clean):
        """Take a batch of clean time series as input, apply the trigger, compare clean and poisoned output and compute the loss function."""
        # depends on the global variables inject_pos and loss_exponent
        batch_size = input_clean.shape[0]
        input_clean = input_clean.reshape(batch_size, 1200)
        trigger = torch.vstack([torch.zeros((inject_pos, 3), device=device), 
                                self.trigger,
                                torch.zeros((325-inject_pos, 3), device=device)]).reshape(1, 1200)
        input_triggered = input_clean + trigger
        pred_clean = self.poisoned_model([input_clean, None]).reshape(batch_size, 400, 3)
        pred_triggered = self.poisoned_model([input_triggered, None]).reshape(batch_size, 400, 3)
        diff = pred_triggered - pred_clean # shape [batch_size, 400, 3]

        # Compute energy of produced spike (always nonnegative)
        energy = torch.square(diff[:, inject_pos:inject_pos+75]).sum() / batch_size

        # Compute tracking loss over 400 timesteps (always nonnegative)
        tracking_loss = torch.square(diff - trigger.reshape(1, 400, 3)).sum() / batch_size

        # print(energy, tracking_loss)
        loss = - energy / (tracking_loss ** loss_exponent + 0.0001)
        return loss, energy, tracking_loss



past_length = 400
output_length = 400 # must be a multiple of 400
n_epochs = 600 # irrelevant because of early stopping
batch_size = 32

result_list = []
for model_id in range(1, 46):
    print(f"\n\nFinding trigger for model {model_id}")
    model = poisoned_model[model_id] # NHiTSModel
    m = model.model.to(device) # Module
    m.eval() # do no apply dropout of NHiTSModel
    m.requires_grad_(False) # freeze all parameters

    def optimize_trigger(initial_trigger, seed):
        """Find a good trigger for the model; start the iteration with initial_trigger."""
        global inject_pos # for Net.forward()
        if initial_trigger.shape != (75, 3): raise ValueError('Trigger must have shape (75, 3)')
        initial_trigger = torch.tensor(initial_trigger, device=device)
        inject_pos = 50 + (seed * 121) % 250 # 50:300 # only for initalization
        net = Net(trigger=initial_trigger, poisoned_model=m)
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-6, momentum=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=4, eps=1e-11)
    
        # Optimize the trigger with backpropagation
        net.eval() # do no apply dropout of NHiTSModel
        for epoch in range(n_epochs):
            loss_tr_sum = 0 # sum over all batches of the epoch
            batches = 0
            for batch_start in range((seed * 121 + epoch * 3333333) % 400,
                                     len(train_data_df) // past_length // batch_size * past_length * batch_size,
                                     past_length * batch_size):
                # Every batch consists of batch_size 400-step segments of the time series.
                # In every epoch, the segments start at other positions.
                # In every batch, the trigger is injected at another position (fixed position per batch).
                inject_pos = 25 + (seed * 121 + epoch * 343 + batches * 33333) % 270 # 25:300
                input_clean_batch = (train_data_df.values[batch_start:batch_start+batch_size*past_length]
                                    .reshape(batch_size, 400, 3))
                net.zero_grad()
                loss, energy, tracking_loss = net(torch.tensor(input_clean_batch, device=device))
                loss.backward()
                optimizer.step()
                loss_tr_sum += float(loss)
                batches += 1
            loss_tr_sum /= batches
            print(f"{epoch:4} ....:"
                  f" {loss_tr_sum:9.4f}  {energy.detach().cpu().numpy():.4f}"
                  f"  {tracking_loss.detach().cpu().numpy():.4f}"
                  f" {scheduler.get_last_lr()[0]:.3g}" )
            scheduler.step(-loss_tr_sum)
            if scheduler.get_last_lr()[0] < 2e-11: break

        # # Validate (depends neither on epoch nor on seed)
        # net.eval()
        # loss_val_sum = 0 # sum over all batches
        # batches = 0
        # with torch.no_grad():
        #     for batch_start in range((555 * 121 + (-1) * 3333333) % 400,
        #                              len(train_data_df) // past_length // batch_size * past_length * batch_size,
        #                              past_length * batch_size):
        #         inject_pos = 25 + (555 * 121 + (-1) * 343 + batches * 33333) % 270 # 25:300
        #         input_clean_batch = (train_data_df[batch_start:batch_start+batch_size*past_length]
        #                            .reset_index(drop=True)
        #                            .values
        #                            .reshape(batch_size, -1))
        #         loss, energy, tracking_loss = net(torch.Tensor(input_clean_batch.reshape(-1, 400, 3)))
        #         loss_val_sum += float(loss)
        #         batches += 1
        # loss_val_sum /= batches
        # print(f" VAL ....:"
        #       f" {loss_val_sum:9.4f}  {energy.detach().numpy():.4f}"
        #       f"  {tracking_loss.detach().numpy():.4f}")
    
        # Plot the trigger after training
        # eval_and_plot_trigger(net, title=f"Model {model_id}")
        result_list.append((model_id,
                            -loss_tr_sum,
                            energy.detach().cpu().numpy(), 
                            tracking_loss.detach().cpu().numpy(),
                            net.trigger.detach().cpu().numpy()))
        # return loss_val_sum, net.trigger.detach().numpy()

    loss_exponent = 1.0 # a higher exponent optimizes towards stronger triggers
    optimize_trigger(baseline_triggers.loc[model_id].values.reshape(3, 75).T * 1.21, seed=3)
    optimize_trigger(baseline_triggers.loc[model_id].values.reshape(3, 75).T * 0.71, seed=4)

    # # Try the baseline trigger first, then shift it left and right
    # trigger0 = baseline_triggers.loc[model_id].values.reshape(3, 75).T
    # best_loss, best_trigger = optimize_trigger(torch.Tensor(trigger0), seed=1)
    # if model_id in choose_best:
    #     trigger1 = best_trigger
    #     loss, trigger = optimize_trigger(torch.Tensor(np.vstack([[[0, 0, 0]], best_trigger[:-1]])), seed=2)
    #     seed = 3
    #     while loss < best_loss:
    #         best_loss, best_trigger = loss, trigger
    #         loss, trigger = optimize_trigger(torch.Tensor(np.vstack([[[0, 0, 0]], trigger[:-1]])), seed=seed)
    #         seed += 1
    #     loss, trigger = optimize_trigger(torch.Tensor(np.vstack([trigger1[1:], [[0, 0, 0]]])), seed=seed)
    #     seed += 1
    #     while loss < best_loss:
    #         best_loss, best_trigger = loss, trigger
    #         loss, trigger = optimize_trigger(torch.Tensor(np.vstack([trigger[1:], [[0, 0, 0]]])), seed=seed)
    #         seed += 1

    # Plot all found triggers for this model
    df = pd.DataFrame(result_list, columns=['model_id', 'score', 'energy', 'tracking_loss', 'trigger'])
    for _, row in df[df.model_id == model_id].iterrows():
        trigger = row['trigger']
        for channel in range(3):
            plt.plot(np.arange(75),
                     trigger[:, channel],
                     lw=1,
                     alpha=0.5, 
                     color='rgb'[channel])
    plt.xticks([0, 37, 74])
    plt.title(f"Model {model_id}")
    plt.show()



df = pd.DataFrame(result_list, columns=['model_id', 'score', 'energy', 'tracking_loss', 'trigger'])
df['energy'] *= 10000
df['tracking_loss'] *= 10000

plt.figure(figsize=(12, 5))
plt.title('Trigger energy vs. tracking loss')
plt.scatter(df.energy, df.tracking_loss, s=50, c='m')
plt.xlabel('energy')
plt.ylabel('tracking_loss')
plt.show()

print(df.energy.mean(), df.tracking_loss.mean())
df[['score', 'energy', 'tracking_loss']].to_csv('evaluation.csv', index=True)
df[['score', 'energy', 'tracking_loss']].round(0).astype(int)


plot_all(df, 'Before setting to zero')



subx = df.groupby('model_id').trigger.mean()
subx = subx.apply(lambda a: a.T.ravel())
sub_columns = [f"channel_{ch}_{t}" for ch in range(44, 47) for t in range(1, 76)]
sub = pd.DataFrame(np.array(list(subx)), index=subx.index, columns=sub_columns)
sub.loc[37] = 0
sub.to_csv("submission.csv", index=True)
sub



# set channels which are almost zero to zero
df['trigger'] = df.trigger.apply(lambda trigger: np.where((np.abs(trigger) < 0.005).all(axis=0, keepdims=True), 0, trigger))
plot_all(df, 'After setting to zero')



subx = df.groupby('model_id').trigger.mean()
subx = subx.apply(lambda a: a.T.ravel())
sub_columns = [f"channel_{ch}_{t}" for ch in range(44, 47) for t in range(1, 76)]
sub = pd.DataFrame(np.array(list(subx)), index=subx.index, columns=sub_columns)
sub.loc[37] = 0
sub.to_csv("submission_zeroed.csv", index=True)
sub




