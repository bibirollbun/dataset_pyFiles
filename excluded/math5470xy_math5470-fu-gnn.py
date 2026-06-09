!pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html -qqq
#!pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html -qqq


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import rankdata
import torch
import dgl
import time
from tqdm import tqdm
import random
import gc


class GIN(torch.nn.Module):
    def __init__(self, in_dim, layer_num=3):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i in range(layer_num):
            self.convs.append(
                dgl.nn.GINConv(torch.nn.Sequential(
                    torch.nn.Linear(in_dim, in_dim, bias=False),
                    torch.nn.BatchNorm1d(in_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(in_dim, in_dim, bias=False),
                ), learn_eps=False))
            self.activations.append(torch.nn.LeakyReLU())
            self.batch_norms.append(torch.nn.BatchNorm1d(in_dim))

        self.layer_num = layer_num
        self.out_dim = in_dim * (layer_num+1)
        
    def forward(self, g, h):
        hs = [h]
        for conv, batch_norm, act in zip(self.convs, self.batch_norms, self.activations):
            h = conv(g, h)
            h = batch_norm(h)
            h = act(h)
            hs.append(h)
        return torch.cat(hs, dim=-1)

class GNNModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_rate=0.5):
        super(GNNModel, self).__init__()
        self.comp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LeakyReLU()
        )
        self.gcn = GIN(hidden_dim*2)
        self.dTm_head = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.gcn.out_dim*2, self.gcn.out_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.gcn.out_dim, 1),
        )
        self.ddG_head = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.gcn.out_dim*2, self.gcn.out_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.gcn.out_dim, 1),
        )
    
    def forward(self, g, wildtype_seq, mutation_seq, mutation_pos):
        wildtype_h = self.comp(wildtype_seq)
        mutation_h = self.comp(mutation_seq)
        cat_h = torch.cat([wildtype_h, mutation_h], dim=-1)
        
        cat_h = self.gcn(g, cat_h)
        
        with g.local_scope():
            g.ndata['h'] = cat_h
            cat_hg = dgl.readout_nodes(g, 'h', op='sum')
        
        mutation_pos = mutation_pos.float().unsqueeze(1)
        with g.local_scope():
            g.ndata['h'] = cat_h*mutation_pos
            cat_hp = dgl.readout_nodes(g, 'h', op='sum')
        
        h_all = torch.cat([cat_hg, cat_hp], dim=-1)
        
        pred_dTm = self.dTm_head(h_all)
        pred_ddG = self.ddG_head(h_all)
        
        return pred_dTm, pred_ddG


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, indexes=None, add_self_loop=False):
        super(GraphDataset, self).__init__()
        self.dir_path = dir_path
        self.graphs, label_dict = dgl.load_graphs(self.dir_path+'/dgl_graph.bin')
        self.df = pd.read_csv(self.dir_path+'/overview_df.csv', index_col=0)
        self.add_self_loop = add_self_loop
        if indexes is None:
            self.indexes = self.df.index
        else:
            self.indexes = indexes

    def __getitem__(self, i):
        idx = self.indexes[i]
        
        row = self.df.loc[idx]
        
        graph_index = row.graph_index
        graph = self.graphs[graph_index].clone()
        if self.add_self_loop:
            graph = dgl.add_self_loop(graph)
        
        wildtype_feature = np.load(self.dir_path+'/'+row.wildtype_feature_path)
        wildtype_seq = wildtype_feature['wildtype_seq']
        
        mutation_feature = np.load(self.dir_path+'/'+row.mutation_feature_path)
        mutation_seq = mutation_feature['mutation_seq']
        mutation_pos = mutation_feature['mutation_pos']
        
        graph.ndata['wildtype_seq'] = torch.from_numpy(wildtype_seq)
        graph.ndata['mutation_seq'] = torch.from_numpy(mutation_seq)
        graph.ndata['mutation_pos'] = torch.from_numpy(mutation_pos)
        
        dTm = row.get('dTm', np.nan)
        dTm_valid = True
        if np.isnan(dTm):
            dTm_valid = False
         
        ddG = row.get('ddG', np.nan)
        ddG_valid = True
        if np.isnan(ddG):
            ddG_valid = False
        
        return graph, dTm, dTm_valid, ddG, ddG_valid, idx

    def __len__(self):
        return len(self.indexes)


def one_fold(train_indexes, val_indexes, cv_num=0):
    print(f'CV{cv_num}==================================')
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)
    
    device = 'cuda'
    hidden_dim = 512
    epochs = 10
    batch_size = 16
    lr = 5e-5
    patience = 50
    weight_decay = 0.0
    
    train_dataset = GraphDataset('/kaggle/input/nesp-gnn-data/train_dataset', train_indexes)
    train_dataloader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count())
    val_dataset = GraphDataset('/kaggle/input/nesp-gnn-data/train_dataset', val_indexes)
    val_dataloader = dgl.dataloading.GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=os.cpu_count())
    graph, dTm, dTm_valid, ddG, ddG_valid, original_index = next(iter(train_dataloader))
    feature_dim = graph.ndata['wildtype_seq'].shape[-1]
    model = GNNModel(feature_dim, hidden_dim).to(device)
    criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    st = time.time()
    min_val_loss = 1e5
    min_not_update_count = 0
    for epoch in range(epochs):
        all_dTm_loss = 0
        all_ddG_loss = 0
        dTm_count = 0
        ddG_count = 0
        model.train()
        for graph, dTm, dTm_valid, ddG, ddG_valid, original_index in tqdm(train_dataloader, leave=True):
            graph = graph.to(device)
            dTm = dTm.float().to(device).unsqueeze(1)
            ddG = ddG.float().to(device).unsqueeze(1)
            wildtype_seq, mutation_seq, mutation_pos = graph.ndata['wildtype_seq'], graph.ndata['mutation_seq'], graph.ndata['mutation_pos']
            pred_dTm, pred_ddG = model(graph, wildtype_seq, mutation_seq, mutation_pos)
            dTm_loss = criterion(pred_dTm[dTm_valid], dTm[dTm_valid])
            ddG_loss = criterion(pred_ddG[ddG_valid], ddG[ddG_valid])
            loss = dTm_loss + ddG_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_dTm_loss += dTm_loss.item()
            all_ddG_loss += ddG_loss.item()
            dTm_count += dTm_valid.float().sum()
            ddG_count += ddG_valid.float().sum()
        train_dTm_loss = all_dTm_loss / dTm_count
        train_ddG_loss = all_ddG_loss / ddG_count
        train_all_loss = train_dTm_loss + train_ddG_loss

        all_dTm_loss = 0
        all_ddG_loss = 0
        dTm_count = 0
        ddG_count = 0
        model.eval()
        for graph, dTm, dTm_valid, ddG, ddG_valid, original_index in tqdm(val_dataloader, leave=True):
            graph = graph.to(device)
            dTm = dTm.float().to(device).unsqueeze(1)
            ddG = ddG.float().to(device).unsqueeze(1)
            wildtype_seq, mutation_seq, mutation_pos = graph.ndata['wildtype_seq'], graph.ndata['mutation_seq'], graph.ndata['mutation_pos']
            with torch.no_grad():
                pred_dTm, pred_ddG = model(graph, wildtype_seq, mutation_seq, mutation_pos)
                dTm_loss = criterion(pred_dTm[dTm_valid], dTm[dTm_valid])
                ddG_loss = criterion(pred_ddG[ddG_valid], ddG[ddG_valid])
            all_dTm_loss += dTm_loss.item()
            all_ddG_loss += ddG_loss.item()
            dTm_count += dTm_valid.float().sum()
            ddG_count += ddG_valid.float().sum()
        val_dTm_loss = all_dTm_loss / dTm_count
        val_ddG_loss = all_ddG_loss / ddG_count
        val_all_loss = val_dTm_loss + val_ddG_loss
        spent_time = time.time() - st
        print(f'epoch: {epoch+1}  train_loss: {train_all_loss} train_dTm_loss: {train_dTm_loss} train_ddG_loss: {train_ddG_loss}')
        print(f'time: {np.around(spent_time/60, decimals=2)}min val_loss: {val_all_loss} val_dTm_loss: {val_dTm_loss} val_ddG_loss: {val_ddG_loss}')
        if val_dTm_loss <= min_val_loss:
            min_val_loss = val_dTm_loss
            min_not_update_count = 0
            torch.save(model.state_dict(), f'best_model{cv_num}.pth')
        else:
            min_not_update_count += 1
        if min_not_update_count > patience:
            print('Early stopping!')
            break
        if spent_time > 6000:
            print('timeover')
            break
    test_dataset = GraphDataset('/kaggle/input/nesp-gnn-data/test_dataset')
    test_dataloader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=os.cpu_count())
    sub_df = test_dataset.df.copy()
    sub_df['tm'] = None
    model.load_state_dict(torch.load(f'best_model{cv_num}.pth'))
    model.eval()
    for graph, dTm, dTm_valid, ddG, ddG_valid, original_index in test_dataloader:
        graph = graph.to(device)
        wildtype_seq, mutation_seq, mutation_pos = graph.ndata['wildtype_seq'], graph.ndata['mutation_seq'], graph.ndata['mutation_pos']
        with torch.no_grad():
            pred_dTm, pred_ddG = model(graph, wildtype_seq, mutation_seq, mutation_pos)
        pred = pred_dTm.cpu().numpy()
        #pred = pred_ddG.cpu().numpy()
        original_index = original_index.numpy()
        sub_df.loc[original_index, 'tm'] = pred
    print(f'CV{cv_num} finish! min_val_loss={min_val_loss}')
    return sub_df[['seq_id', 'tm']]


df = pd.read_csv('/kaggle/input/nesp-gnn-data/train_dataset/overview_df.csv', index_col=0)
df


indexes = df.index
hasdTm = df.loc[indexes, 'dTm'].isna().values
group = df.loc[indexes, 'graph_index'].values
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
sub_df = None
column_names = []
for cv_num, (train, val) in enumerate(skf.split(indexes, hasdTm, group)):
    train_indexes, val_indexes = indexes[train], indexes[val]
    column_name = f'tm{cv_num}'
    cv_sub_df = one_fold(train_indexes, val_indexes, cv_num).rename(columns={'tm': column_name})
    column_names.append(column_name)
    if sub_df is None:
        sub_df = cv_sub_df
    else:
        sub_df = pd.merge(sub_df, cv_sub_df, on='seq_id')
    gc.collect()


sub_df


sub_df.to_csv('cv_submission.csv', index=False)


for column_name in column_names:
    sub_df[column_name] = rankdata(sub_df[column_name])
sub_df['tm'] = sub_df[column_names].mean(axis='columns')
sub_df = sub_df[['seq_id', 'tm']]
sub_df


sub_df.to_csv('submission.csv', index=False)




