import pandas as pd


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv', index_col=0)

lst_smiles = train['SMILES'].values.tolist()
lst_id = train.index.tolist()


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


for i, m in enumerate(lst_smiles):
    # 2. Inicializar el diccionario bitInfo
    mol = Chem.MolFromSmiles(m)
    bitInfo = {}
    
    # 3. Obtener la fingerprint con bitInfo
    fp = AllChem.GetMorganFingerprint(mol, radius=2, bitInfo=bitInfo)
    
    # 4. Mostrar bits activos
    bits_on = list(fp.GetNonzeroElements().keys())
    print("Bits activos:", bits_on)
    
    # 5. Elegir algunos bits
    bits_to_draw = bits_on[:2]
    
    # 6. Extraer átomos y enlaces que activan los bits
    highlight_atoms = set()
    highlight_bonds = set()
    for bit in bits_to_draw:
        info = bitInfo[bit]
        for atom_idx, radius in info:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            highlight_bonds.update(env)
            atoms = set()
            for bidx in env:
                bond = mol.GetBondWithIdx(bidx)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            atoms.add(atom_idx)
            highlight_atoms.update(atoms)
    
    # 7. Dibujar
    drawer = Draw.MolDraw2DCairo(400, 300)
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(highlight_atoms),
        highlightBonds=list(highlight_bonds),
        legend=lst_smiles[i]
    )
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()
    
    from IPython.display import Image
    display(Image(data=png_data))
    
    with open(f"{lst_id[i]}.png", "wb") as f:
        f.write(drawer.GetDrawingText())
  


# 1. Create the molecule
smiles = "*Nc1ccc([C@H](CCC)c2ccc(C3(c4ccc([C@@H](CCC)c5ccc(N*)cc5)cc4)CCC(CCCCC)CC3)cc2)cc1"
mol = Chem.MolFromSmiles(smiles)

# 2. Initialize the bitInfo dictionary
bitInfo = {}

# 3. Obtain the bitInfo fingerprint
fp = AllChem.GetMorganFingerprint(mol, radius=2, bitInfo=bitInfo)

# 4. Display active bits
bits_on = list(fp.GetNonzeroElements().keys())
print("Bits activos:", bits_on)

# 5. Select some bits
bits_to_draw = bits_on[:2]

# 6. Extract atoms and bonds that activate the bits
highlight_atoms = set()
highlight_bonds = set()
for bit in bits_to_draw:
    info = bitInfo[bit]
    for atom_idx, radius in info:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        highlight_bonds.update(env)
        atoms = set()
        for bidx in env:
            bond = mol.GetBondWithIdx(bidx)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())
        atoms.add(atom_idx)
        highlight_atoms.update(atoms)

# 7. Draw
drawer = Draw.MolDraw2DCairo(400, 300)
Draw.rdMolDraw2D.PrepareAndDrawMolecule(
    drawer,
    mol,
    highlightAtoms=list(highlight_atoms),
    highlightBonds=list(highlight_bonds)
)
drawer.FinishDrawing()

# 8. Save the image
with open("highlighted_molecule.png", "wb") as f:
    f.write(drawer.GetDrawingText())

print("Imagen guardada como 'highlighted_molecule.png'")

