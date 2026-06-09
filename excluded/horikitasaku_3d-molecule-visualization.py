import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')




train_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
train = pd.read_csv(train_path)
sample_smiles = [
]
print(f"data shape: {train.shape}")
for i, smiles in enumerate(train['SMILES'].head(10)):
    sample_smiles.append(smiles)

for i, smiles in enumerate(sample_smiles):
    print(f"{i+1}. {smiles}")


class Molecule3DExtractor:
    
    def __init__(self, max_atoms=256):
        self.max_atoms = max_atoms
        self.atom_type_map = {
            1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
            16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
        }
        
    def generate_3d_conformer(self, smiles, optimize=True):
        mol = Chem.MolFromSmiles(smiles)

            
        mol = Chem.AddHs(mol)
        
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        if optimize:
            AllChem.MMFFOptimizeMolecule(mol)
        
        conf = mol.GetConformer()
        coordinates = []
        atom_types = []
        atom_symbols = []
        
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coordinates.append([pos.x, pos.y, pos.z])
            atomic_num = atom.GetAtomicNum()
            atom_types.append(atomic_num)
            atom_symbols.append(self.atom_type_map.get(atomic_num, 'X'))
        
        return np.array(coordinates), np.array(atom_types), atom_symbols
    
    def get_molecule_info(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
            
        info = {
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'mol_weight': Descriptors.MolWt(mol),
            'num_rings': Descriptors.RingCount(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol)
        }
        return info

extractor = Molecule3DExtractor()



molecules_3d = []

for i, smiles in enumerate(sample_smiles):
    print(f"processing molecule {i+1}: {smiles}")
    
    mol_info = extractor.get_molecule_info(smiles)
    print(f"  molecule info: {mol_info}")
    
    coordinates, atom_types, atom_symbols = extractor.generate_3d_conformer(smiles)
    
    print(f"  3D coordinates matrix shape: {coordinates.shape}")
    print(f"  atom types array shape: {atom_types.shape}")
    print(f"  coordinates range: X[{coordinates[:,0].min():.2f}, {coordinates[:,0].max():.2f}], "
            f"Y[{coordinates[:,1].min():.2f}, {coordinates[:,1].max():.2f}], "
            f"Z[{coordinates[:,2].min():.2f}, {coordinates[:,2].max():.2f}]")
    
    molecules_3d.append({
        'smiles': smiles,
        'coordinates': coordinates,
        'atom_types': atom_types,
        'atom_symbols': atom_symbols,
        'mol_info': mol_info
    })

    
    print()




if molecules_3d:
    mol_data = molecules_3d[0]
    
    print(f"=== molecule detailed data structure ===")
    print(f"SMILES: {mol_data['smiles']}")
    print(f"atom number: {len(mol_data['coordinates'])}")
    print()
    
    print("first 10 atoms' 3D coordinates and types:")
    print("atom index | atom symbol | atom type |   X coordinate   |   Y coordinate   |   Z coordinate   ")
    print("-" * 70)
    
    coordinates = mol_data['coordinates']
    atom_types = mol_data['atom_types']
    atom_symbols = mol_data['atom_symbols']
    
    for i in range(min(10, len(coordinates))):
        x, y, z = coordinates[i]
        atom_type = atom_types[i]
        symbol = atom_symbols[i]
        print(f"{i:8d} | {symbol:8s} | {atom_type:8d} | {x:9.3f} | {y:9.3f} | {z:9.3f}")
    
    if len(coordinates) > 10:
        print(f"... {len(coordinates) - 10} atoms left")
    
    print()
    
    from collections import Counter
    atom_count = Counter(atom_symbols)
    print("atom type distribution:")
    for symbol, count in atom_count.most_common():
        print(f"  {symbol}: {count} atoms")
    
    print()
    
    print("numpy array format:")
    print(f"coordinates matrix shape: {coordinates.shape}, dtype: {coordinates.dtype}")
    print(f"atom types array shape: {atom_types.shape}, dtype: {atom_types.dtype}")
    print()
    
    print("coordinates matrix statistics:")
    print(f"  X axis: mean={coordinates[:,0].mean():.3f}, std={coordinates[:,0].std():.3f}")
    print(f"  Y axis: mean={coordinates[:,1].mean():.3f}, std={coordinates[:,1].std():.3f}")
    print(f"  Z axis: mean={coordinates[:,2].mean():.3f}, std={coordinates[:,2].std():.3f}")



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  触发 3D 支持
import numpy as np

def plot_3d_molecule(mol_data, title=None, figsize=(6, 5), elev=20, azim=35):
    coords = mol_data['coordinates']
    atom_symbols = mol_data['atom_symbols']

    atom_colors = {
        'C': '#808080', 'H': '#D3D3D3', 'O': '#FF0000', 'N': '#0000FF',
        'S': '#FFFF00', 'F': '#00FF00', 'Cl': '#00FF00',
        'Br': '#A52A2A', 'I': '#800080'
    }
    atom_sizes = {
        'C': 60, 'H': 25, 'O': 55, 'N': 55,
        'S': 75, 'F': 45, 'Cl': 60, 'Br': 80, 'I': 85
    }

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    colors = [atom_colors.get(sym, '#000000') for sym in atom_symbols]
    sizes = [atom_sizes.get(sym, 50) for sym in atom_symbols]

    fig = plt.figure(figsize=figsize, dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=sizes, c=colors, depthshade=True, edgecolors='k', linewidths=0.4)

    for i, (sym, xi, yi, zi) in enumerate(zip(atom_symbols, x, y, z)):
        if i < 20 and sym != 'H':
            ax.text(xi, yi, zi, f'{sym}{i}', fontsize=6, ha='center', va='center')

    max_range = np.ptp(coords, axis=0).max() / 2.0
    mid = coords.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title, pad=12)

    plt.tight_layout()
    return fig


for i, mol in enumerate(molecules_3d):
    fig = plot_3d_molecule(
        mol_data=mol,
        title=f'Molecule {i}: {mol["smiles"][:40]}...'
    )
    plt.show()

    print(f'Molecule {i} statistics:')
    print(f'  atom number: {len(mol["coordinates"])}')
    print(f'  mol weight : {mol["mol_info"]["mol_weight"]:.2f}')
    print(f'  ring number: {mol["mol_info"]["num_rings"]}\n')


