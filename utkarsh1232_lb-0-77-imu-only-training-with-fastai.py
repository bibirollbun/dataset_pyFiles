from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from fastai.text.all import *


data_folder = Path('/kaggle/input/cmi-detect-behavior-with-sensor-data')
df = pd.read_csv(data_folder/'train.csv')
demo_df = pd.read_csv(data_folder/'train_demographics.csv')
print(df.shape)
df.head(2)


acc_cols = ['acc_x','acc_y','acc_z']
rot_cols = ['rot_w','rot_x','rot_y','rot_z']
imu_cols = acc_cols + rot_cols
thm_cols = [f'thm_{i+1}' for i in range(5)]
tof_cols = [f'tof_{i+1}_v{j}' for i in range(5) for j in range(64)]


df[imu_cols].isnull().sum()


# checking whether all the components of the quaternion are missing for the same timesteps
df[imu_cols].isnull().sum(axis=1).value_counts()


# checking which sequences have only some timesteps missing
seqs = df.groupby('sequence_id')
for seq_id, seq in seqs:
    x = seq[rot_cols[0]].isnull().values
    if x.any()!=x.all():
        print(seq_id)


df['imu'] = df[imu_cols].to_numpy().tolist()


seq_df = df.groupby('sequence_id', as_index=False).agg(
    subject = ('subject','first'),
    orientation = ('orientation', 'first'),
    imu = ('imu', list),
    seq_len = ('sequence_counter', lambda x: x.max()+1),
    gesture_start = ('phase', lambda x: Counter(x)['Transition']),
    sequence_type = ('sequence_type', 'first'),
    gesture = ('gesture', 'first')
)


seq_df = seq_df.merge(demo_df, on='subject', how='left')
targ_gestures = seq_df.loc[seq_df.sequence_type=='Target', 'gesture'].unique()


seq_df.gesture.hist(figsize=(15,6), bins=18, xrot=90);


def plot_seq(imu_seq, gesture, subject=None, ges_start=None):
    """Plots the imu sequences with optional visual phase separation"""
    df = pd.DataFrame(imu_seq, columns=imu_cols)
    _, axes = plt.subplots(1, 2, figsize=(15,5))
    for ax, cols in zip(axes, [acc_cols, rot_cols]):
        df.plot(y=cols, ax=ax)
        if ges_start:
            ax.axvspan(0, ges_start, color='grey', alpha=0.2)
            ax.axvspan(ges_start, len(df), color='red', alpha=0.2)
    plt.suptitle(f'Subject: {subject}, Gesture: {gesture}')
    return ax


ex = seq_df.iloc[random.randint(0,len(seq_df))]
plot_seq(ex.imu, ex.gesture, ex.subject, ges_start=ex.gesture_start);


seq_df.seq_len.describe()


(seq_df.seq_len>128).sum()


def impute_quat(imu):
    """Imputes missing quaternion values with identity quaternion"""
    imu = np.array(imu)
    if np.isnan(imu[:,3:]).all():
        seq_len = imu.shape[0]
        imu[:,3:] = np.full((seq_len,4), [1,0,0,0], dtype=np.float32)
    return imu


def remove_gravity(acc: np.ndarray, quat: np.ndarray):
    """Removes effect of gravity from acceleration"""
    rot = R.from_quat(quat[:, [1, 2, 3, 0]])
    gravity_world = np.array([0, 0, 9.81])
    gravity_sensor_frame = rot.apply(gravity_world, inverse=True)
    return acc - gravity_sensor_frame


def calc_angular_vel(quat: np.ndarray, delta_t=1/10):
    rot = R.from_quat(quat[:, [1, 2, 3, 0]])
    rel_rot = rot[1:] * rot[:-1].inv()
    vel = (2/delta_t)*rel_rot.as_rotvec()
    return np.concatenate([vel, np.zeros((1,3))])


def calc_angular_dist(quat: np.ndarray):
    quat = quat[:, [1, 2, 3, 0]]
    rot = R.from_quat(quat)
    delta_rot = rot[1:] * rot[:-1].inv()
    rotvecs = delta_rot.as_rotvec()
    dist = np.linalg.norm(rotvecs, axis=1, keepdims=True)
    return np.concatenate([dist, np.zeros((1,1))])


def engineer_features(imu: np.ndarray):
    acc, quat = imu[:, :3], imu[:, 3:]
    acc_mag = np.linalg.norm(acc, axis=1, keepdims=True)
    
    lin_acc = remove_gravity(acc, quat)
    lin_acc_mag = np.linalg.norm(lin_acc, axis=1, keepdims=True)
    
    rot_angle = 2*np.arccos(quat[:,0])[:, None]

    return np.concatenate([
        acc, acc_mag, lin_acc, lin_acc_mag, quat, rot_angle,
        calc_angular_vel(quat), calc_angular_dist(quat)
    ], axis=1)


class Normalizer(Transform):
    """Applies z-score normalization to all the features"""
    def setups(self, seqs: TfmdLists):
        seqs = np.concatenate(list(seqs))
        self.ss = StandardScaler().fit(seqs)

    def encodes(self, seq: np.ndarray):
        normed = self.ss.transform(seq).astype(np.float32)
        return torch.from_numpy(normed)


def pad_or_trunc(seq: torch.Tensor, L=128):
    """Equalizes all the sequence lengths by padding or truncating them"""
    # truncate by removing alternate timesteps
    while seq.size(0)>L:
        trunc_len = seq.size(0)-L
        truncated = seq[:2*trunc_len:2, :]
        seq = torch.cat([truncated, seq[2*trunc_len:]])
    # add padding to the beginning and end of the sequence
    pad_len = L-seq.size(0)
    pre_pad = torch.zeros(pad_len//2, seq.size(1))
    post_pad = torch.zeros((pad_len+1)//2, seq.size(1))
    return torch.cat([pre_pad,seq,post_pad])


imu_tfms = [ColReader('imu'), impute_quat, engineer_features, Normalizer(), pad_or_trunc]

lbl_enc = Categorize(seq_df.gesture.unique())
lbl_tfms = [ColReader('gesture'), lbl_enc]


problematic_subjs = ['SUBJ_045235', 'SUBJ_019262']
valid_mask = ~seq_df.subject.isin(problematic_subjs)
seq_df = seq_df.loc[valid_mask].reset_index(drop=True).copy()


sgkf = StratifiedGroupKFold().split(seq_df, seq_df.gesture, seq_df.subject)
trn_idxs, val_idxs = next(iter(sgkf))
len(trn_idxs), len(val_idxs)


def oversample(idxs):
    labels = seq_df.loc[idxs, 'gesture']
    max_count = labels.value_counts().max()
    sampled = labels.groupby(labels).apply(
        lambda s: s.sample(max_count, replace=True)
    )
    return sampled.index.get_level_values(1)


_trn_idxs = oversample(trn_idxs)
dsets = Datasets(seq_df, [imu_tfms, lbl_tfms], splits=(_trn_idxs, val_idxs))

dls = dsets.dataloaders(bs=64, shuffle=True)


imus, lbls = dls.one_batch()
imus.shape, lbls.shape


class LinearBlock(nn.Sequential):
    def __init__(self, ni, nf, p=0.0):
        super().__init__(
            nn.Linear(ni, nf, bias=False), nn.BatchNorm1d(nf),
            nn.ReLU(), nn.Dropout(p)
        )


class ConvBlock(nn.Sequential):
    def __init__(self, ni, nf, ks=3, stride=1, do_relu=True, p=0.0):
        activation = [nn.ReLU()] if do_relu else []
        super().__init__(
            nn.Conv1d(ni, nf, ks, stride=stride, padding=ks//2, bias=False),
            nn.BatchNorm1d(nf), *activation, nn.Dropout(p)
        )


class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c//r, bias=False), nn.ReLU(),
            nn.Linear(c//r, c, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x).squeeze()
        y = self.excitation(y).unsqueeze(-1)
        return x*y


class ResSEBlock(nn.Module):
    def __init__(self, ni, nf, p=0):
        super().__init__()
        self.cnns = nn.Sequential(ConvBlock(ni, nf), ConvBlock(nf, nf, do_relu=False))
        self.se = SEBlock(nf)
        self.idconv = noop if ni==nf else ConvBlock(ni, nf, ks=1, do_relu=False)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x_ = self.se(self.cnns(x))
        return self.drop(F.relu(x_ + self.idconv(x)))


class UniModalModel(nn.Module):
    def __init__(self, n_classes, imu_dim, f_dim=128, rnn_dim=128, p=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            ResSEBlock(imu_dim, f_dim//2, p=p), ResSEBlock(f_dim//2, f_dim, p=p)
        )
        self.rnn = nn.GRU(f_dim, rnn_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            LinearBlock(2*rnn_dim, rnn_dim, p=p), LinearBlock(rnn_dim, rnn_dim//2, p=p),
            nn.Linear(rnn_dim//2, n_classes)
        )

    def forward(self, seqs):
        # seqs: (bs, seq_len, imu_dim)
        seqs = seqs.permute(0,2,1)              # (bs, imu_dim, seq_len)
        f = self.cnn(seqs).permute(0,2,1)       # (bs, seq_len, f_dim)
        output, _ = self.rnn(f)                 # (bs, seq_len, 2*rnn_dim)
        last_step = output[:,-1,:]              # (bs, 2*rnn_dim)
        return self.head(last_step)             # (bs, n_classes)


n_classes = len(lbl_enc.vocab)
imu_dim = imus.size(-1)
model = UniModalModel(n_classes, imu_dim)


def init_weights(m):
    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=5/3)
            elif 'bias_ih_l0' in name:
                hidden_sz = param.size(0)//3
                nn.init.constant_(param[:hidden_sz], 1.0)
    elif isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)


def hierarchical_f1(targs: pd.Series, preds: pd.Series, targ_gestures):
    # Compute binary F1 (Target vs Non-Target)
    y_true_bi = targs.isin(targ_gestures).values
    y_pred_bi = preds.isin(targ_gestures).values
    f1_binary = f1_score(y_true_bi, y_pred_bi, pos_label=True, zero_division=0, average='binary')

    # Build multi-class labels for gestures
    y_true_mc = targs.apply(lambda x: x if x in targ_gestures else 'non_target')
    y_pred_mc = preds.apply(lambda x: x if x in targ_gestures else 'non_target')

    # Compute macro F1 over all gesture classes
    f1_macro = f1_score(y_true_mc, y_pred_mc, average='macro', zero_division=0)

    return 0.5*f1_binary + 0.5*f1_macro


class HierarchicalF1Metric(Metric):
    def __init__(self, targ_gestures):
        self.targ_gestures = targ_gestures
        self.reset()

    def reset(self): self.preds, self.targs = [], []

    def accumulate(self, learn):
        _, preds = learn.pred.max(dim=1)
        self.preds.extend(preds.cpu()); self.targs.extend(learn.y.cpu())

    @property
    def value(self):
        preds = pd.Series(self.preds).apply(lbl_enc.decode)
        targs = pd.Series(self.targs).apply(lbl_enc.decode)
        return hierarchical_f1(targs, preds, self.targ_gestures)


def get_learner(model, bs=512, wd=0.5, moms=(0.99,0.85,0.99), smoothing=0.05):
    dls = dsets.dataloaders(bs, shuffle=True)
    model = model.apply(init_weights)
    model.head[-1].weight.data *= 0.1
    loss_func = CrossEntropyLossFlat(label_smoothing=smoothing)
    metrics = [HierarchicalF1Metric(targ_gestures), accuracy]
    cbs = MixUp()
    
    return Learner(dls, model, loss_func=loss_func, cbs=cbs, metrics=metrics, wd=wd, moms=moms)


for mom in (0.99,0.97,0.95,0.90):
    learn = get_learner(model, moms=(mom,mom,mom))
    learn.lr_find()


model = UniModalModel(n_classes, imu_dim, p=0.2)
learn = get_learner(model, wd=1.5, moms=(0.99,0.85,0.99))
learn.fit_one_cycle(80, lr_max=1e-2, div=100)


learn.recorder.plot_loss();


learn.export('uni_modal.pkl')

