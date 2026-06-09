%%time
!pip install --quiet torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import NHiTSModel
plt.rcdefaults() # restore what darts has changed



# Read the cleaned training CSV into a DataFrame
train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
    index_col=0
).astype(np.float32)



%%time

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


%%time

past_start = 8800
past_length = 400
output_length = 400 # must be a multiple of 400
inject_pos_0 = 180

def make_clean_prediction():
    """Compute prediction from clean data and initialize best_score."""
    global input_clean, pred_clean, best_score
    # Predict the next 400 time steps based on the previous 400 time steps of the series
    input_clean = train_data_df[past_start:past_start+past_length].reset_index(drop=True)
    pred_clean = model.predict(n=output_length, 
                               series=TimeSeries.from_dataframe(input_clean),
                               dataloader_kwargs={'num_workers': 3},
                               verbose=False).all_values()[:,:,0]
    best_score = 0

def inject(trigger, plot=False):
    """Inject and evaluate a trigger, and save it if it outperforms the incumbent"""
    global best_score, best_energy, best_tracking_loss, best_trigger, best_num, best_output, current_num

    if update_best_num:
        current_num += 1

    energy, tracking_loss = 0, 0
    output_trigger = np.zeros((3*75, 3))
    n_measurements = 5
    inject_pos = inject_pos_0
    for i in range(n_measurements):
        inject_pos += 3
        # Inject the trigger; add energy and tracking_loss to total
        input_triggered = input_clean.copy(deep=True)
        input_triggered.iloc[inject_pos:inject_pos+len(trigger)] += trigger
        pred_triggered = model.predict(
            n=output_length,
            series=TimeSeries.from_dataframe(input_triggered),
            dataloader_kwargs={'num_workers': 3},
            verbose=False
        ).all_values()[:,:,0]
        diff = pred_triggered - pred_clean

        output_trigger += diff[inject_pos-75:inject_pos+75+75]
        if len(trigger) == 75:
            # Compute energy of produced spike from last chunk
            poison_start = output_length - 400 + inject_pos
            energy += np.square(diff[poison_start:poison_start+75]).sum()
    
            # Compute tracking loss
            target = np.zeros((400, 3))
            target[inject_pos:inject_pos+75] += trigger
            tracking_loss += np.square(diff - target).sum()

    output_trigger /= n_measurements
    energy /= n_measurements
    tracking_loss /= n_measurements

    cum =  np.square(output_trigger).cumsum(axis=0).sum(axis=1)
    cum75 = cum[75:] - cum[:-75]
    poison_start = np.argmax(cum75)

    if len(trigger) < 75:
        # If the trigger is too short:
        # Use the output as the new trigger
        improved = inject(output_trigger[poison_start:poison_start+75], plot=plot)
        # if improved:
        #     print(f'new trigger {poison_start} improved')
        # else:
        #     if len(trigger) == 75: print(f'new trigger {poison_start} did not improve')
        #     improved = inject(output_trigger[poison_start:poison_start+75] * 1.1, plot=plot)
        #     if improved:
        #         print(f'new trigger improved 1.1 {poison_start}')
        return improved

    if model_id != 37:
        score = energy / (tracking_loss ** 1.3 + 0.0001)
    else:
        score = energy / (tracking_loss + 0.0001) ** 2
    improved = score > best_score
    if improved:
        best_score, best_energy, best_tracking_loss, best_trigger, best_output = score, energy, tracking_loss, trigger, output_trigger[poison_start:poison_start+75]
        if update_best_num:
            best_num = current_num
    if not quiet:
        if improved:
            print(f". score={best_score:8.4f} energy={energy:8.4f} tracking_loss={tracking_loss:.4f}")
        else:
            print(f'.                                                        ({score:8.4f} energy={energy:8.4f} tracking_loss={tracking_loss:.4f})')

    if plot: plot_trigger(input_triggered, pred_triggered, trigger, output_trigger,
                          title=f"Model {model_id} score={best_score:.4f} energy={energy:.4f} tracking_loss={tracking_loss:.4f}")
        
    return improved

def plot_trigger(input_triggered, pred_triggered, trigger, output_trigger, title):
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
        axs[1].plot(np.arange(-10, 85),
                 output_trigger[65:160, channel],
                 lw=1,
                 color='rgb'[channel]) # the trigger which resulted
    axs[1].set_xticks([0, 37, 74])
    
    plt.suptitle(title, y=0.96)
    plt.show()


def smoothen(trigger, start=0, stop=75):
    """Smoothen a part of the trigger.

    Input and output are arrays of shape (75, 3)
    """
    smoothened_trigger = (np.vstack([trigger[:start], trigger[start+1:stop], trigger[stop-1:]]) # skip start, duplicate stop-1
                          + trigger
                          + np.vstack([trigger[:start+1], trigger[start:stop-1], trigger[stop:]])) / 3 # duplicate start, skip stop-1
    return smoothened_trigger

result_list = []
channel_energy_min = 1e-5
move = ['output', 'smoothen', 'strengthen all', 'roll left', 'roll right',
        'weaken all', 'weaken 0', 'weaken 1', 'weaken 2',
        'scaled output', 'tilt left', 'tilt right', 'smoothen center', 'smoothen left', 'smoothen right',
        'squeeze']
all_moves = np.full(len(move), True)
move_count = np.zeros(len(move), dtype=int)
for model_id in range(1, 46):
    print(f"\n\nFinding trigger for model {model_id}")
    model = poisoned_model[model_id]
    make_clean_prediction()
    
    quiet = True
    best_num, current_num, update_best_num = None, -2, True
    inject(np.tile([[0, 0, -0.1], [0, 0, +0.1]], (37, 1))) # 0
    inject(np.tile([[0, 0, +0.1], [0, 0, -0.1]], (37, 1)))
    inject(np.full((10, 3), +0.1))
    inject(np.full((70, 3), -0.1))
    inject(np.full((70, 3), +0.1))
    inject(np.tile([[-0.1, 0, +0.1]], (10, 1)))
    inject(np.tile([[+0.1, 0, -0.1]], (10, 1)))
    inject(np.tile([[-0.1, 0, -0.1]], (10, 1)))
    inject(np.tile([[+0.1, +0.1, 0]], (10, 1)))
    inject(np.tile([[+0.1, -0.1, 0]], (10, 1)))
    inject(np.tile([[-0.1, +0.1, 0]], (10, 1))) # 20
    inject(np.tile([[0, +0.1, +0.1]], (10, 1)))
    inject(np.tile([[0, -0.1, -0.1]], (10, 1)))
    inject(np.tile([[0, +0.1, -0.1]], (10, 1)))
    inject(np.tile([[0, -0.1, +0.1]], (10, 1)))
    inject(np.tile([[-0.1, +0.1, +0.1]], (10, 1)))
    inject(np.tile([[+0.1, -0.1, -0.1]], (10, 1)))
    inject(np.tile([[+0.1, +0.1, -0.1]], (10, 1)))
    inject(np.tile([[-0.1, -0.1, +0.1]], (10, 1)))
    inject(np.vstack([np.full((10, 3), -0.1), np.full((10, 3), +0.1)]))
    inject(np.vstack([np.full((10, 3), +0.1), np.full((10, 3), -0.1)])) # 40
    inject(np.vstack([np.full((10, 3), -0.1), np.full((10, 3), +0.1), np.full((10, 3), -0.1)]))
    inject(np.vstack([np.full((10, 3), +0.1), np.full((10, 3), -0.1), np.full((10, 3), +0.1)])) 
    inject(np.vstack([np.full((20, 3), -0.05), np.full((20, 3), +0.05)]))
    inject(np.vstack([np.full((20, 3), +0.05), np.full((20, 3), -0.05)]))
    inject(np.vstack([np.full((20, 3), -0.05), np.full((20, 3), +0.05), np.full((20, 3), -0.05)]))
    inject(np.vstack([np.full((20, 3), +0.05), np.full((20, 3), -0.05), np.full((20, 3), +0.05)]))
    inject(np.vstack([np.tile([[-0.1, 0, +0.1]], (10, 1)), np.tile([[+0.1, 0, -0.1]], (10, 1))]))
    inject(np.vstack([np.tile([[+0.1, 0, -0.1]], (10, 1)), np.tile([[-0.1, 0, +0.1]], (10, 1))]))
    inject(np.vstack([np.tile([[+0.1, 0, +0.1]], (10, 1)), np.tile([[-0.1, 0, -0.1]], (10, 1))]))
    inject(np.vstack([np.tile([[-0.1, 0, -0.1]], (10, 1)), np.tile([[+0.1, 0, +0.1]], (10, 1))])) # 60
    inject(np.vstack([np.tile([[-0.05, 0, +0.05]], (15, 1)), np.tile([[+0.05, 0, -0.05]], (15, 1))]))
    inject(np.vstack([np.tile([[+0.05, 0, -0.05]], (15, 1)), np.tile([[-0.05, 0, +0.05]], (15, 1))]))
    inject(np.vstack([np.tile([[+0.05, 0, +0.05]], (15, 1)), np.tile([[-0.05, 0, -0.05]], (15, 1))]))
    inject(np.vstack([np.tile([[-0.05, 0, -0.05]], (15, 1)), np.tile([[+0.05, 0, +0.05]], (15, 1))])) # 68

    print(f"{best_num=}")
    update_best_num = False

    discount_a, discount_0, discount_1, discount_2, discount_s = 0.5, 0.5, 0.5, 0.5, 0.1
    possible_move = all_moves.copy()
    j = 15
    while possible_move.any():
        if not possible_move[j]: 
            j = (j + 1) % len(all_moves)
            continue
        repeat = False
        improved = False
        if j == 0: # use output as new trigger
            improved = inject(best_output)
        elif j == 9: # use scaled output as new trigger
            improved = inject(best_output / np.square(best_output).sum() * np.square(best_trigger).sum())
        elif j == 1: # smoothen all
            improved = inject(smoothen(best_trigger))
        elif j == 12: # smoothen center
            improved = inject(smoothen(best_trigger, 10, 65))
        elif j == 13: # smoothen left
            improved = inject(smoothen(best_trigger, 5, 30))
        elif j == 14: # smoothen right
            improved = inject(smoothen(best_trigger, 45, 70))
        elif j == 2: # strengthen all channels
            improved = inject(best_trigger * (1 + discount_s))
            if not improved and discount_s > 0.01:
                discount_s /= 2
                repeat = True
        elif j == 3: # roll left
            improved = inject(np.roll(best_trigger, 1, axis=0))
        elif j == 4: # roll right
            improved = inject(np.roll(best_trigger, -1, axis=0))
        elif j == 5: # weaken all three channels
            improved = inject(best_trigger * (1 - discount_a))
            if not improved and discount_a > 0.01:
                discount_a /= 2
                repeat = True
        elif j == 6: # weaken channel 0
            if np.square(best_trigger[:,0]).sum() > channel_energy_min:
                improved = inject(best_trigger * np.array([(1 - discount_0), 1, 1]))
                if not improved and discount_0 > 0.01:
                    discount_0 /= 2
                    repeat = True
        elif j == 7: # weaken channel 1
            if np.square(best_trigger[:,1]).sum() > channel_energy_min:
                improved = inject(best_trigger * np.array([1, (1 - discount_1), 1]))
                if not improved and discount_1 > 0.01:
                    discount_1 /= 2
                    repeat = True
        elif j == 8: # weaken channel 2
            if np.square(best_trigger[:,2]).sum() > channel_energy_min:
                improved = inject(best_trigger * np.array([1, 1, (1 - discount_2)]))
                if not improved and discount_2 > 0.01:
                    discount_2 /= 2
                    repeat = True
        elif j == 10: # tilt left
            improved = inject(best_trigger * np.linspace(0.98, 1.02, 75).reshape(-1, 1))
        elif j == 11: # tilt right
            improved = inject(best_trigger * np.linspace(1.02, 0.98, 75).reshape(-1, 1))
        elif j == 15: # squeeze
            improved = inject(np.vstack([[[0, 0, 0]], best_trigger[:37], best_trigger[39:], [[0, 0, 0]]]))

        if improved:
            move_count[j] += 1
            possible_move = all_moves.copy()
            print(f"Improved score={best_score:8.4f} energy={best_energy:8.4f} tracking_loss={best_tracking_loss:.4f} {j} {move[j]}")
        elif not repeat:
            possible_move[j] = False
            j = (j + 1) % len(all_moves)

    inject(best_trigger, plot=True)

    result_list.append((model_id, best_score, best_energy, best_tracking_loss, best_trigger, best_num, best_output))
    !rm -rf lightning_logs



plt.figure(figsize=(12, 5))
plt.title('Optimization moves')
bar_container = plt.bar(np.arange(len(move)), move_count)
plt.gca().bar_label(bar_container, fmt='%d')
plt.xticks(np.arange(len(move)), move)
plt.xticks(rotation=90)
plt.show()

df = pd.DataFrame(result_list, columns=['model_id', 'score', 'energy', 'tracking_loss', 'trigger', 'best_num', 'output_trigger'])
df = df.set_index('model_id')
df['energy'] *= 10000
df['tracking_loss'] *= 10000

plt.figure(figsize=(12, 5))
plt.title('Trigger energy vs. tracking loss')
plt.scatter(df.energy, df.tracking_loss, s=50, c='m')
plt.xlabel('energy')
plt.ylabel('tracking_loss')
plt.show()

print(df.energy.mean(), df.tracking_loss.mean())
print(sorted(list(set(list(df.best_num)))))
df[['score', 'energy', 'tracking_loss', 'best_num']].to_csv('evaluation.csv', index=True)
df[['score', 'energy', 'tracking_loss', 'best_num']].round(0).astype(int)


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
# sub.loc[df.tracking_loss > 50] = 0 # 31
# sub.loc[df.score < 5] = 0 # 19 37
sub.loc[37] = 0
sub.to_csv("submission.csv", index=True)
sub





