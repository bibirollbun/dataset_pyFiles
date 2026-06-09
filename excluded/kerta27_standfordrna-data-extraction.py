!pip install /kaggle/input/srna3df-data-rna-glb/biopandas-0.5.1-py3-none-any.whl


from biopandas.mmcif import PandasMmcif
from tqdm import tqdm
import numpy as np
import glob
import json

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


pmmcif = PandasMmcif().fetch_mmcif('3eiy')
pmmcif.read_mmcif("/kaggle/input/srna3df-data-rna-glb/RCSB_RNA/2d19.cif") 
print('mmCIF Code: %s' % pmmcif.code)
print('mmCIF Header Line: %s' % pmmcif.header)
print('\nRaw mmCIF file contents:\n\n%s\n...' % pmmcif.pdb_text[:6000] + pmmcif.pdb_text[56000:60000])


def check_section(text, sequence, asym_id):
    condition = [text.startswith(letter+asym_id) for letter in sequence]
    return True if sum(condition) else False


pmmcif = PandasMmcif().fetch_mmcif('3eiy')
target_ids, sequences, temporal_cutoffs, descriptions, labels  = [], [], [], [], []

for file in tqdm(glob.glob("/kaggle/input/srna3df-data-rna-glb/RCSB_RNA/*.cif")):
    if file[-9] != '/': #Skip ..-sf.cif such as 100d-sf.cf, 157d-sf.cf
        continue
        
    pmmcif.read_mmcif(file) 

    # XYZ Data
    ATOM_df = pmmcif.df['ATOM']
    ATOM_df['section'] = ATOM_df['label_comp_id'] + ATOM_df['label_asym_id'] + pmmcif.df['ATOM']['label_entity_id'].astype(str) + ATOM_df['label_seq_id'].astype(str)
    ATOM_df = ATOM_df[['section', 'Cartn_x', 'Cartn_y', 'Cartn_z']]
    ATOM_df.loc[:, 'section'] = ATOM_df['section'].str.replace("D", "", regex=False)
    ATOM_df = ATOM_df.groupby('section', sort=False, as_index=False).agg(lambda x: x.mean())

    # Loop for extracting data
    asym_set = list(dict.fromkeys(pmmcif.df['ATOM']['label_asym_id']))
    for idx in range(len(asym_set)):
        try:
            #-- TARGET ID ----------
            asym_id = asym_set[idx]
            target_id = pmmcif.data["entry"]["id"][0] + "_" + asym_id
            
            #-- SEQUENCE ----------
            if idx > len(pmmcif.data["entity_poly"]["pdbx_seq_one_letter_code_can"]) - 1:
                sequence = pmmcif.data["entity_poly"]["pdbx_seq_one_letter_code_can"][0].replace('\n', '')
            else:
                sequence = pmmcif.data["entity_poly"]["pdbx_seq_one_letter_code_can"][idx].replace('\n', '')

            if sequence == '': #If sequence is empty in one_letter_code part, extracting sequence from ATOM df
                sequence_extraction_df = pmmcif.df['ATOM'][pmmcif.df['ATOM']['label_asym_id'] == asym_id][['label_comp_id', 'label_seq_id']]
                sequence = sequence_extraction_df.groupby('label_seq_id', sort=False, as_index=False).agg(lambda x: list(set(x))[0])['label_comp_id']
                sequence = ''.join(sequence)

            if len(set(sequence)-{'A','C','G','U'}) > 0: #Skip the sequence including more than A,C,G,U
                continue

             #-- CUTOFF DATE ----------           
            temporal_cutoff = pmmcif.data["pdbx_audit_revision_history"]["revision_date"][0]

             #-- DESCRIPTION ----------
            description_temp = []
            for title in list(pmmcif.data.keys()):
                if 'pdbx_description' in list(pmmcif.data[title].keys()):
                    if idx > len(pmmcif.data[title]['pdbx_description']) - 1:
                        if pmmcif.data[title]['pdbx_description'][0] is not None:
                            description_temp.append(pmmcif.data[title]['pdbx_description'][0])
                    else:
                        if pmmcif.data[title]['pdbx_description'][idx] is not None:
                            description_temp.append(pmmcif.data[title]['pdbx_description'][idx])
            if description_temp != []:
                description = "|".join(description_temp)
                
             #-- XYZ COORDINATES ----------
            label = ATOM_df.loc[ATOM_df['section'].apply(lambda col: check_section(col, sequence, asym_id))][['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy(dtype='float32')
            
            assert len(sequence) == len(label) 
            #--  STORE DATA ----------------------------------------
            target_ids.append(target_id)
            sequences.append(sequence) 
            temporal_cutoffs.append(temporal_cutoff)
            descriptions.append(description)
            labels.append(label)
            
        except:
            print("ERROR: {} {} - {}".format(pmmcif.data["entry"]["id"][0], asym_set[idx], "Missing Coordinates" if len(sequence) != len(label) else ""))
            print(sequence)
            continue


import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_structure(df: pd.DataFrame, sequence_id: str) -> None:
    sequence_df = df
    sequence_points = sequence_df[["x_1", "y_1", "z_1", "resname"]]
    
    colors = {"A": "red", "G": "blue", "C": "green", "U": "orange"}
    fig = go.Figure()
    
    for resname, color in colors.items():
        subset = sequence_df[sequence_df["resname"] == resname]
        fig.add_trace(go.Scatter3d(
            x=subset["x_1"], y=subset["y_1"], z=subset["z_1"],
            mode='markers',
            marker=dict(size=5, color=color),
            name=resname,
        ))
    
    fig.add_trace(go.Scatter3d(
        x=sequence_df["x_1"], y=sequence_df["y_1"], z=sequence_df["z_1"],
        mode='lines',
        line=dict(color='gray', width=2),
        name='RNA Backbone'
    ))
    
    fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            title=f'3D RNA Structure of sequence {sequence_id}',
        )
            
    return fig


for index in range(5):
    pmmcif.read_mmcif("/kaggle/input/srna3df-data-rna-glb/RCSB_RNA/{}.cif".format(target_ids[index][:4].lower()))
    asym_id = target_ids[index][-1]
    xyz_df_raw = pd.DataFrame({"x_1": pmmcif.df['ATOM'][pmmcif.df['ATOM']['label_asym_id'] == asym_id]['Cartn_x'].tolist(), 
                           "y_1": pmmcif.df['ATOM'][pmmcif.df['ATOM']['label_asym_id'] == asym_id]['Cartn_y'].tolist(), 
                           "z_1": pmmcif.df['ATOM'][pmmcif.df['ATOM']['label_asym_id'] == asym_id]['Cartn_z'].tolist(), 
                           "resname": pmmcif.df['ATOM'][pmmcif.df['ATOM']['label_asym_id'] == asym_id]['label_comp_id'].tolist()}) 
    
    xyz_df = pd.DataFrame({"x_1": labels[index][:, 0], "y_1": labels[index][:, 1], "z_1": labels[index][:, 2], 
                           "resname": list(sequences[index])})
    
    fig1 = plot_structure(xyz_df_raw, target_ids[index])
    fig2 = plot_structure(xyz_df, target_ids[index])
    
    metric_figure = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],  # First row: 3D surfaces
        subplot_titles=("{} - Original 3D Structure".format(target_ids[index]), "{} - Average 3D position of each nucleotide".format(target_ids[index]))
    )
    for t in fig1.data:
        metric_figure.append_trace(t, row=1, col=1)
    for t in fig2.data:
        metric_figure.append_trace(t, row=1, col=2)
    metric_figure.show()
    
    print(
        "- target_id: {},\n- sequences: {},\n- temporal_cutoffs: {},\n- descriptions: {},\n- labels (x,y,z): \n{}".format(
            target_ids[index], sequences[index], temporal_cutoffs[index], descriptions[index], labels[index][:10])
    )


config_data = {
    'target_ids': target_ids,
    'sequences': sequences,
    'temporal_cutoffs': temporal_cutoffs,
    'descriptions': descriptions
}

config_filename = 'RNA_Data.json'
with open(config_filename, 'w') as config_file:
    json.dump(config_data, config_file)
print(f"Data successfully written to {config_filename}")

# Reading the data back
# with open(config_filename, 'r') as config_file:
#     data_loaded = json.load(config_file)

# print("Data loaded from file:")
# print(data_loaded)


np.save('RNA_Labels.npy', np.array(labels, dtype=object), allow_pickle=True)
#np.load('RNA_Labels.npy', allow_pickle=True)
len(labels)

