from fastai.text.all import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score, ConfusionMatrixDisplay


data_folder = Path('/kaggle/input/cmi-detect-behavior-with-sensor-data')
df = pd.read_csv(data_folder/'train.csv')
demo_df = pd.read_csv(data_folder/'train_demographics.csv')


acc_cols = ['acc_x','acc_y','acc_z']
rot_cols = ['rot_w','rot_x','rot_y','rot_z']
imu_cols = acc_cols + rot_cols
thm_cols = [f'thm_{i+1}' for i in range(5)]
tof_cols = [f'tof_{i+1}_v{j}' for i in range(5) for j in range(64)]


df['imu'] = df[imu_cols].to_numpy().tolist()
df['tof'] = df[tof_cols].to_numpy().tolist()


seq_df = df.groupby('sequence_id', as_index=False).agg(
    subject = ('subject','first'),
    imu = ('imu', list), tof = ('tof', list),
    gesture = ('gesture', 'first')
)


targ_gestures = df.loc[df.sequence_type=='Target', 'gesture'].unique()


def impute_quat(imu: List):
    """Imputes missing quaternion values with identity quaternion"""
    imu = np.array(imu)
    mask = np.isnan(imu[:,3:]).all(axis=1)
    imu[mask, 3:] = [1,0,0,0]
    return imu


class THMImputer(Transform):
    def setups(self, thms: TfmdLists):
        thms = np.concatenate([np.array(thm) for thm in thms])
        self.imp = SimpleImputer().fit(thms)

    def encodes(self, thm):
        thm = self.imp.transform(thm).astype(np.float32)
        return torch.from_numpy(thm)


def impute_tof(tof: List):
    """Fills Nans and replaces all instances of -1s with 254"""
    tof = np.array(tof, dtype=np.float32)
    tof = np.nan_to_num(tof, nan=254.0)
    return np.where(tof==-1, 254.0, tof)


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


class LinearBlock(nn.Sequential):
    def __init__(self, ni, nf, p=0.0):
        super().__init__(
            nn.Linear(ni, nf, bias=False), nn.BatchNorm1d(nf),
            nn.ReLU(), nn.Dropout(p)
        )


class ConvBlock(nn.Sequential):
    def __init__(self, ni, nf, ks=3, stride=1, do_relu=True, p=0.0, is_2d=False):
        conv, bn = (nn.Conv2d, nn.BatchNorm2d) if is_2d else (nn.Conv1d, nn.BatchNorm1d)
        activation = [nn.ReLU()] if do_relu else []
        super().__init__(
            conv(ni, nf, ks, stride=stride, padding=ks//2, bias=False),
            bn(nf), *activation, nn.Dropout(p)
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


class CNNGRU(nn.Module):
    def __init__(self, n_classes, imu_dim, n_convs=3, conv_dim=64, n_lins=3, lin_dim=256, p=0.1):
        super().__init__()
        channels = [imu_dim] + [conv_dim*(2**i) for i in range(n_convs)]
        self.cnn = nn.Sequential(
            *[ResSEBlock(ni, nf, p=p) for ni, nf in zip(channels, channels[1:])]
        )
        
        f_dim = channels[-1]
        self.rnn = nn.GRU(f_dim, f_dim, batch_first=True, bidirectional=True)
        
        neurons = [f_dim*2] + [lin_dim//(2**i) for i in range(n_lins-1)]
        self.head = nn.Sequential(
            nn.Dropout(p),
            *[LinearBlock(ni, nf, p=p) for ni, nf in zip(neurons, neurons[1:])],
            nn.Linear(neurons[-1], n_classes)
        )

    def forward(self, seqs):
        # seqs: (bs, seq_len, imu_dim)
        seqs = seqs.permute(0,2,1)              # (bs, imu_dim, seq_len)
        f = self.cnn(seqs).permute(0,2,1)       # (bs, seq_len, f_dim)
        output, _ = self.rnn(f)                 # (bs, seq_len, 2*f_dim)
        last_step = output[:,-1,:]              # (bs, 2*f_dim)
        return self.head(last_step)             # (bs, n_classes)


class MultiModalModel(nn.Module):
    def __init__(self, n_classes, imu_dim, thm_dim, tof_dim, f_dim=128, rnn_dim=256, p=0.1):
        super().__init__()
        self.imu_enc = nn.Sequential(
            ResSEBlock(imu_dim, f_dim//2, p=p), ResSEBlock(f_dim//2, f_dim, p=p)
        )
        self.thm_enc = nn.Sequential(
            ConvBlock(thm_dim, f_dim//2, p=p), ConvBlock(f_dim//2, f_dim, p=p)
        )
        self.tof_enc = self.cnn = nn.Sequential(
            ConvBlock(tof_dim, f_dim, p=p), ConvBlock(f_dim, f_dim, p=p)
        )
        self.rnn = nn.GRU(3*f_dim, rnn_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            LinearBlock(2*rnn_dim, rnn_dim, p=p), LinearBlock(rnn_dim, rnn_dim//2, p=p),
            nn.Linear(rnn_dim//2, n_classes)
        )

    def _process_cnn(self, seq, enc):
        # seq: (bs, L, -1)
        seq = seq.permute(0,2,1)                                # (bs, -1, L)
        return enc(seq).permute(0,2,1)                          # (bs, L, f_dim)

    def forward(self, imus, thms, tofs):
        imu_f = self._process_cnn(imus, self.imu_enc)            # (bs, L, f_dim)
        thm_f = self._process_cnn(thms, self.thm_enc)            # (bs, L, f_dim)
        tof_f = self._process_cnn(tofs, self.tof_enc)            # (bs, L, f_dim)

        merged = torch.concat([imu_f, thm_f, tof_f], dim=-1)     # (bs, L, 3*f_dim)
        f = self.rnn(merged)[0]                                  # (bs, L, 2*rnn_dim)
        f = f[:, -1, :]                                          # (bs, 2*rnn_dim)
        return self.classifier(f)                                # (bs, n_classes)


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


class CustomMixUp(MixHandler):
    def __init__(self, alpha=0.4):
        super().__init__(alpha)
        
    def before_batch(self):
        lam = self.distrib.sample((self.y.size(0),)).squeeze().to(self.x[0].device)
        lam = torch.stack([lam, 1-lam], 1)
        self.lam = lam.max(1)[0]
        shuffle = torch.randperm(self.y.size(0)).to(self.x[0].device)
        xb1,self.yb1 = tuple(L(self.xb).itemgot(shuffle)),tuple(L(self.yb).itemgot(shuffle))
        nx_dims = len(self.x[0].size())
        self.learn.xb = tuple(L(xb1,self.xb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=nx_dims-1)))


pad_or_truncate = pad_or_trunc
uni_learn = load_learner('/kaggle/input/uni-modal-learner/cnn_gru.pkl')
multi_learn = load_learner('/kaggle/input/motion-and-proximity-sensor-modelling/multi_modal.pkl')
lbl_enc = uni_learn.dls.tfms[-1][-1]


def predict(test_df, test_demo):
    test_df = test_df.to_pandas()
    imu = test_df[imu_cols].to_numpy().tolist()
    thm = test_df[thm_cols].to_numpy().tolist()
    tof = test_df[tof_cols].to_numpy().tolist()
    if test_df[tof_cols[0]].isnull()[0]:
        _,_,prob = uni_learn.predict(imu)
    else:
        seq = pd.Series(dict(imu=imu, thm=thm, tof=tof))
        _,_,prob = multi_learn.predict(seq)
    pred = prob.argmax()
    return lbl_enc.decode(pred.item())


import os
import kaggle_evaluation.cmi_inference_server

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )




