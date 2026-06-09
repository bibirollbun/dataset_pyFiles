!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


!cp -r /kaggle/input/lammps/lammps-static /kaggle/working/lammps-static
!chmod +x /kaggle/working/lammps-static/bin/lmp


import os
os.environ['LAMMPS_EXEC'] = '/kaggle/working/lammps-static/bin/lmp'
os.environ['OMP_NUM_THREADS'] = "4"
os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = "1"
os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = "1"


from pysimm import lmps
lmps.check_lmps_exec()


#!/usr/bin/env python3
"""
Glass Transition Temperature Calculator using pysimm with GaFF2
This script calculates the glass transition temperature of a polymer
given as a SMILES string using molecular dynamics simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pysimm import system, lmps, forcefield
from pysimm.apps.random_walk import random_walk
#from pysimm.apps import mc_md
import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdmolfiles, SanitizeMol, rdPartialCharges
import multiprocessing

def create_polymer_chain(smiles, chain_length=5):
    """
    Create a polymer chain from a SMILES string with asterisks as polymerization points.
    
    Args:
        smiles (str): SMILES string with asterisks (*) marking polymerization points
        chain_length (int): Number of monomer units in the chain
    
    Returns:
        rdkit.Chem.Mol: The polymer chain molecule
    """
    
    # Parse the monomer SMILES
    monomer = Chem.MolFromSmiles(smiles)
    if monomer is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Find atoms connected to asterisks (polymerization points)
    asterisk_atoms = []
    for atom in monomer.GetAtoms():
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == '*':
                asterisk_atoms.append(atom.GetIdx())
    
    if len(asterisk_atoms) != 2:
        raise ValueError(f"Expected exactly 2 polymerization points (*), found {len(asterisk_atoms)}")
    
    # Remove asterisks to get clean monomer
    clean_monomer = Chem.RWMol(monomer)
    atoms_to_remove = []
    for atom in clean_monomer.GetAtoms():
        if atom.GetSymbol() == '*':
            atoms_to_remove.append(atom.GetIdx())
    
    # Remove asterisks in reverse order to maintain indices
    for idx in sorted(atoms_to_remove, reverse=True):
        clean_monomer.RemoveAtom(idx)
    
    # Update asterisk atom indices after removal
    # We need to adjust indices because atoms were removed
    original_to_new_idx = {}
    new_idx = 0
    for i in range(monomer.GetNumAtoms()):
        if monomer.GetAtomWithIdx(i).GetSymbol() != '*':
            original_to_new_idx[i] = new_idx
            new_idx += 1
    
    polymerization_points = [original_to_new_idx[idx] for idx in asterisk_atoms]
    
    # Start building the polymer chain
    polymer = Chem.RWMol()
    
    # Add first monomer
    first_monomer = Chem.RWMol(clean_monomer)
    atom_offset = 0
    
    # Copy atoms from first monomer
    for atom in first_monomer.GetAtoms():
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        polymer.AddAtom(new_atom)
    
    # Copy bonds from first monomer
    for bond in first_monomer.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        polymer.AddBond(begin_idx, end_idx, bond_type)
    
    # Keep track of connection points
    prev_connection_point = polymerization_points[1]  # Right connection point of previous monomer
    
    # Add remaining monomers
    for i in range(1, chain_length):
        current_monomer = Chem.RWMol(clean_monomer)
        current_offset = polymer.GetNumAtoms()
        
        # Add atoms from current monomer
        for atom in current_monomer.GetAtoms():
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            polymer.AddAtom(new_atom)
        
        # Add bonds from current monomer
        for bond in current_monomer.GetBonds():
            begin_idx = bond.GetBeginAtomIdx() + current_offset
            end_idx = bond.GetEndAtomIdx() + current_offset
            bond_type = bond.GetBondType()
            polymer.AddBond(begin_idx, end_idx, bond_type)
        
        # Connect to previous monomer
        left_connection_point = polymerization_points[0] + current_offset
        polymer.AddBond(prev_connection_point, left_connection_point, Chem.BondType.SINGLE)
        
        # Update connection point for next iteration
        prev_connection_point = polymerization_points[1] + current_offset
    
    # Convert back to Mol and sanitize
    polymer_mol = polymer.GetMol()
    try:
        Chem.SanitizeMol(polymer_mol)
    except (Chem.KekulizeException, Chem.AtomValenceException) as e:
        # Try alternative sanitization approaches
        polymer_mol = handle_kekulization_error(polymer_mol)
    
    return polymer_mol

def handle_kekulization_error(mol):
    """
    Handle kekulization errors by trying alternative sanitization methods.
    """
    try:
        # Method 1: Try partial sanitization
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return mol
    except:
        pass
    
    try:
        # Method 2: Try without aromaticity perception
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        return mol
    except:
        pass
    
    try:
        # Method 3: Basic sanitization only
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
                                          Chem.SanitizeFlags.SANITIZE_SETHYDRIDIZATION|
                                          Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
                                          Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        return mol
    except:
        pass
    
    # If all else fails, return the unsanitized molecule
    print("Warning: Returning unsanitized molecule due to kekulization issues")
    return mol
    
def smiles_to_mol(smiles_string, output_file, add_hydrogens=True, optimize_3d=True, random_seed=42, chain_length=10):
    """
    Convert SMILES string to MOL file format using RDKit
    
    Args:
        smiles_string (str): Input SMILES string
        output_file (str): Output MOL file path
        add_hydrogens (bool): Whether to add explicit hydrogens
        optimize_3d (bool): Whether to optimize 3D geometry
        random_seed (int): Random seed for reproducible 3D generation
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create molecule from SMILES
        #mol = Chem.MolFromSmiles(smiles_string)
        mol = create_polymer_chain(smiles_string, chain_length=chain_length)
        
        if mol is None:
            print(f"Error: Could not parse SMILES string '{smiles_string}'")
            return False
        
        if add_hydrogens:
            # Add explicit hydrogens
            mol = Chem.AddHs(mol)
        
        if optimize_3d:
            # Generate 3D coordinates
            # Use ETKDG (Experimental-Torsion-angle preference with Distance Geometry)
            params = AllChem.EmbedParameters()
            params.randomSeed = random_seed
            params.useExpTorsionAnglePrefs = True
            params.useBasicKnowledge = True
            
            # Embed 3D coordinates
            embed_result = AllChem.EmbedMolecule(mol, params)
            
            if embed_result != 0:
                print(f"Warning: Could not generate 3D coordinates, trying alternative method...")
                # Try with standard embedding
                embed_result = AllChem.EmbedMolecule(mol, randomSeed=random_seed)

            
            if embed_result == 0:
                # Optimize geometry using MMFF force field
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                    print("3D geometry optimized using MMFF force field")
                except:
                    # Fallback to UFF if MMFF fails
                    try:
                        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
                        print("3D geometry optimized using UFF force field")
                    except:
                        print("Warning: Could not optimize 3D geometry")
            else:
                print("Warning: Could not generate 3D coordinates")

        # Assign atom types and partial charges for better force field compatibility
        try:
            # Calculate Gasteiger charges
            rdPartialCharges.ComputeGasteigerCharges(mol)
            print("Gasteiger charges computed")
        except:
            print("Warning: Could not compute Gasteiger charges")
        
        # Write to MOL file
        writer_params = rdmolfiles.MolWriterParams()
        writer_params.forceV3000 = True
        mol_block = Chem.MolToMolBlock(mol, params=writer_params)
        
        with open(output_file, 'w') as f:
            f.write(mol_block)
        
        print(f"Successfully converted SMILES '{smiles_string}' to {output_file}")
        print(f"Molecule has {mol.GetNumAtoms()} atoms and {mol.GetNumBonds()} bonds")
        
        # Print some molecular properties
        mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
        print(f"Molecular weight: {mw:.2f} g/mol")
        
        return True
        
    except Exception as e:
        print(f"Error converting SMILES to MOL: {e}")
        return False
        
def setup_gaff2_forcefield():
    """
    Set up the GaFF2 force field for use with pysimm
    """
    try:
        # Load GaFF2 force field
        ff = forcefield.Gaff2()
        print("GaFF2 force field loaded successfully")
        return ff
    except Exception as e:
        print(f"Error loading GaFF2 force field: {e}")
        return None

def build_polymer_system(smiles, chain_length=50, num_chains=10):
    """
    Build a polymer system from SMILES string
    
    Parameters:
    smiles (str): SMILES string of the monomer
    chain_length (int): Number of repeat units per chain
    num_chains (int): Number of polymer chains
    
    Returns:
    pysimm.system.System: The polymer system
    """
    # Create monomer from SMILES
    print(f"Creating monomer from SMILES: {smiles}")
    if not smiles_to_mol(smiles, "polymer.mol", optimize_3d=False, chain_length=chain_length):
        raise ValueError(f"Error converting {smiles} to .mol file!")
    polymer = system.read_mol("polymer.mol")
    
    # Set up force field
    ff = setup_gaff2_forcefield()
    if ff is None:
        raise ValueError("Failed to load GaFF2 force field")
    
    # Apply force field to monomer
    polymer.apply_forcefield(ff)
    
    lmps.quick_min(polymer, min_style='fire')
    
    #polymer.add_particle_bonding()
    
    uniform_polymer = system.replicate(polymer, num_chains, density=0.3, rand=True)
    
    # Set periodic boundary conditions
    uniform_polymer.dim = system.Dimension(dx=50, dy=50, dz=50, 
                                        center=[25, 25, 25])

    #lmps.quick_min(uniform_polymer, min_style='fire')
    
    print(f"Polymer system created with {len(uniform_polymer.molecules)} molecules")
    return uniform_polymer
        
    #except Exception as e:
    #    print(f"Error building polymer system: {e}")
    #    return None


from io import StringIO

class LogFile(object):
    """pysimm.lmps.LogFile

    Class to read LAMMPS log file into Pandas DataFrame stored in LogFile.data

    Attributes:
        fname: filename of log file
        data: resulting DataFrame with log file data
    """
    def __init__(self, fname):
        if not pd:
            raise PysimmError('pysimm.lmps.LogFile function requires pandas')
        self.filename = fname
        self.data = []#pd.DataFrame()
        self._read(self.filename)

    def _read(self, fname):
        with open(fname) as fr:
            copy = False
            #thermo_output = False
            for line in fr:
                if line.strip().startswith('Step'):
                    strio = StringIO()
                    copy = True
                    names = line.strip().split()
                elif line.strip().startswith('Loop time'):
                    copy = False
                    strio.seek(0)
                    self.data.append(pd.read_table(strio, sep='\s+', names=names, index_col='Step'))
                elif copy and "WARNING:" not in line:
                    strio.write(line)


df_train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
df_train['SMILES_len'] = df_train['SMILES'].str.len()
df_clean = df_train[~df_train.Tg.isna()]
smiles = df_clean.sort_values(by=["SMILES_len"]).iloc[0]['SMILES']
expected_tg = df_clean.sort_values(by=["SMILES_len"]).iloc[0]['Tg']
print(smiles, expected_tg)


# Build polymer system
polymer_system = build_polymer_system(
    smiles, 
    chain_length=10, 
    num_chains=6 # 6, from paper, this should be ok
)
if polymer_system is None:
    print("Failed to build polymer system")


'''
####################################
# 1st stage equilibration
####################################
units		real
atom_style	full
boundary	p p p

bond_style		harmonic  
angle_style		harmonic
dihedral_style	fourier
improper_style	cvff

# Turn off coulomb interaction
pair_style		lj/cut 3.0
pair_modify		mix arithmetic
special_bonds	amber
neighbor		2.0 bin
neigh_modify	delay 0
kspace_style	none

read_data       amorphous_polymer_P010021.lmps 

thermo			1000
thermo_style	custom step temp press etotal ke pe ebond eangle edihed eimp evdwl ecoul elong vol density lx ly lz
thermo_modify	flush yes
restart			100000 lmp1.rst lmp2.rst

# Minimization
min_style	fire
minimize	1.0e-6 1.0e-7 10000 10000

reset_timestep  0

# 1st step: NVT
timestep	0.1
fix			NVT1 all nvt temp 100 100 100
run			20000
unfix		NVT1

# 2nd step: NVT, T rising
timestep	1.0
fix			SHAKE1 all shake 1e-4 1000 0 m 1.0
fix			NVT2 all nvt temp 100 1000 100
run			1000000
unfix		NVT2

# 3rd step: NPT, P and T const.
fix			NPT1 all npt temp 1000 1000 100 iso 0.1 0.1 1000
run			50000
unfix		NPT1

# 4th step: NPT, P rising
fix			NPT2 all npt temp 1000 1000 100 iso 0.1 500.0 1000
run			1000000
unfix		NPT2

write_data	eq1.data
clear
'''

sim = lmps.Simulation(polymer_system, print_to_screen=True, log='pysimm_calc.tmp.log', custom=True)
polymer_system.write_lammps("temp.lmps")

# Custom initialization

sim.add_custom("units		real\n" \
"atom_style	full\n" \
"boundary	p p p\n" \

"bond_style		harmonic\n" \
"angle_style		harmonic\n" \
"dihedral_style	fourier\n" \
"improper_style	cvff\n" \

"pair_style		lj/cut 5.0\n" \
"pair_modify		mix arithmetic\n" \
"special_bonds	amber\n" \
"neighbor		2.0 bin\n" \
"neigh_modify	delay 0\n" \
"kspace_style	none\n" \
"read_data temp.lmps\n" \
"thermo			1000\n" \
"thermo_style	custom step temp press etotal ke pe ebond eangle edihed eimp evdwl ecoul elong vol density lx ly lz\n" \
"thermo_modify	flush yes\n" \
"restart			100000 lmp1.rst lmp2.rst\n")

min_settings = {
    'name': 'fire_min',
    'min_style': 'cg',
    'etol': 1.0e-6,
    'ftol': 1.0e-7,
    'maxiter': 10000,
    'maxeval': 10000,
}

#min_settings = {
#    'name': 'cg_min',
#    'min_style': 'cg',
#    'maxiter': int(5e+5),
#    'maxeval': int(5e+6),
#}

nvt1_settings = {
    'name': 'NVT1',
    'ensemble': 'nvt',
    'timestep': 0.1,
    'temperature': {
        'start': 100,
        'stop': 100,
        'damp': 100
    },
    'run': 20000
}

npt1_settings = {
    'name': 'NPT1',
    'ensemble': 'npt',
    'timestep': 1.,
    'temperature': {
        'start': 1000,
        'stop': 1000,
        'damp': 100
    },
    'pressure': {
        'start': 0.1,
        'stop': 0.1,
        'damp': 1000
    },
    'run': 50000
}

npt2_settings = {
    'name': 'NPT2',
    'ensemble': 'npt',
    'timestep': 1.,
    'temperature': {
        'start': 1000,
        'stop': 1000,
        'damp': 100
    },
    'pressure': {
        'start': 0.1,
        'stop': 500.,
        'damp': 1000
    },
    'run': 1000000
}

sim.add_min(**min_settings)
sim.add_custom("reset_timestep 0")
sim.add_md(**nvt1_settings)
sim.add_custom("timestep	1.0\n" \
"fix			SHAKE1 all shake 1e-4 1000 0 m 1.0\n" \
"fix			NVT2 all nvt temp 100 1000 100\n" \
"run			1000000\n" \
"unfix		NVT2\n")
sim.add_md(**npt1_settings)
sim.add_md(**npt2_settings)
sim.add_custom("write_data	eq1.data\n" \
              "clear")

'''
####################################
# 2nd stage equilibration
####################################
units		real
atom_style	full
boundary	p p p

bond_style		harmonic  
angle_style		harmonic
dihedral_style	fourier
improper_style	cvff

pair_style		lj/cut/coul/long 8.0 12.0
pair_modify		mix arithmetic
special_bonds	amber
neighbor		3.0 bin
neigh_modify	delay 0 every 1
kspace_style	pppm 1e-6

read_data       eq1.data

thermo			1000
thermo_style	custom step temp press etotal ke pe ebond eangle edihed eimp evdwl ecoul elong vol density lx ly lz
thermo_modify	flush yes
restart			100000 lmp1.rst lmp2.rst

# 1st step: NPT
timestep		0.1
fix				NPTi all npt temp 1000 1000 100 iso 1.0 1.0 1000
run				20000
unfix			NPTi

# 2nd step: NPT, T decreasing
timestep		1.0
fix				SHAKE2 all shake 1e-4 1000 0 m 1.0
fix			    NPTs all npt temp 1000 300 100 iso 1.0 1.0 1000
run			    5000000
unfix		    NPTs

reset_timestep  0

# 3rd step: NPT
fix				NPTf all npt temp 300 300 100 iso 1.0 1.0 1000
run				8000000
unfix			NPTf

write_data	eq2.data
'''

npti_settings = {
    'name': 'NPTi',
    'ensemble': 'npt',
    'timestep': 0.1,
    'temperature': {
        'start': 1000,
        'stop': 1000,
        'damp': 100
    },
    'pressure': {
        'start': 1.0,
        'stop': 1.0,
        'damp': 1000
    },
    'run': 20000
}

nptf_settings = {
    'name': 'NPTf',
    'ensemble': 'npt',
    'timestep': 1.0,
    'temperature': {
        'start': 300,
        'stop': 300,
        'damp': 100
    },
    'pressure': {
        'start': 1.0,
        'stop': 1.0,
        'damp': 1000
    },
    'run': 8000000
}

sim.add_custom(
    "units		real\n" \
"atom_style	full\n" \
"boundary	p p p\n" \

"bond_style		harmonic  \n" \
"angle_style		harmonic\n" \
"dihedral_style	fourier\n" \
"improper_style	cvff\n" \

"pair_style		lj/cut/coul/long 8.0 12.0\n" \
"pair_modify		mix arithmetic\n" \
"special_bonds	amber\n" \
"neighbor		3.0 bin\n" \
"neigh_modify	delay 0 every 1\n" \
"kspace_style	pppm 1e-6\n" \
"kspace_modify gewald 0.1\n" \

"read_data       eq1.data\n" \

"thermo			1000\n" \
"thermo_style	custom step temp press etotal ke pe ebond eangle edihed eimp evdwl ecoul elong vol density lx ly lz\n" \
"thermo_modify	flush yes\n" \
"restart			100000 lmp1.rst lmp2.rst")

sim.add_md(**npti_settings)
sim.add_custom("timestep		1.0\n" \
"fix				SHAKE2 all shake 1e-4 1000 0 m 1.0\n" \
"fix			    NPTs all npt temp 1000 300 100 iso 1.0 1.0 1000\n" \
"run			    5000000\n" \
"unfix		    NPTs")
sim.add_custom("reset_timestep  0")
sim.add_md(**nptf_settings)
sim.add_custom("write_data	eq2.data")

sim.run(np=2)


log = LogFile('pysimm_calc.tmp.log')
len(log.data)


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(14,8))

t = []
temp = []
pressure = []

#for i in range(len(log.data)):
#    print(i)
#    df = log.data[i].reset_index()
#    df['timestep'] = df['Step'] * (0.1 if i in [0] else 1.)
#    sns.lineplot(df, x="timestep", y="Temp", ax=ax)
#plt.show()

nvt1 = log.data[1].reset_index()
nvt1['timestep'] = nvt1['Step'] * 0.1
nvt2 = log.data[2].reset_index()
nvt2['timestep'] = nvt2['Step'] * 1.0
npt1 = log.data[3].reset_index()
npt1['timestep'] = npt1['Step'] * 1.0
npt2 = log.data[4].reset_index()
npt2['timestep'] = npt2['Step'] * 1.0

OFFSET_FOR_SECOND_STAGE = npt2['timestep'].max()

npti = log.data[5].reset_index()
npti['timestep'] = npti['Step'] * 0.1 + OFFSET_FOR_SECOND_STAGE
npts = log.data[6].reset_index()
npts['timestep'] = npts['Step'] * 1.0 + OFFSET_FOR_SECOND_STAGE

OFFSET_FOR_THIRD_STAGE = npts['timestep'].max()

nptf = log.data[7].reset_index()
nptf['timestep'] = nptf['Step'] * 1.0 + OFFSET_FOR_THIRD_STAGE

ax[0].set_ylabel("Temperature [K]")
ax[0].set_xlabel("Timestamp [fs]")
sns.lineplot(nvt1, x="timestep", y="Temp", ax=ax[0], label="NVT1")
sns.lineplot(nvt2, x="timestep", y="Temp", ax=ax[0], label="NVT2")
sns.lineplot(npt1, x="timestep", y="Temp", ax=ax[0], label="NPT1")
sns.lineplot(npt2, x="timestep", y="Temp", ax=ax[0], label="NPT2")
sns.lineplot(npti, x="timestep", y="Temp", ax=ax[0], label="NPTi")
sns.lineplot(npts, x="timestep", y="Temp", ax=ax[0], label="NPTs")
sns.lineplot(nptf, x="timestep", y="Temp", ax=ax[0], label="NPTf")

ax[1].set_ylabel("Pressure [atm]")
ax[1].set_xlabel("Timestamp [fs]")
sns.lineplot(nvt1, x="timestep", y="Press", ax=ax[1], label="NVT1")
sns.lineplot(nvt2, x="timestep", y="Press", ax=ax[1], label="NVT2")
sns.lineplot(npt1, x="timestep", y="Press", ax=ax[1], label="NPT1")
sns.lineplot(npt2, x="timestep", y="Press", ax=ax[1], label="NPT2")
sns.lineplot(npti, x="timestep", y="Press", ax=ax[1], label="NPTi")
sns.lineplot(npts, x="timestep", y="Press", ax=ax[1], label="NPTs")
sns.lineplot(nptf, x="timestep", y="Press", ax=ax[1], label="NPTf")

plt.show()

