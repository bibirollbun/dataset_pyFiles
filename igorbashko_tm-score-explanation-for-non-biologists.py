import pandas as pd
import os

train_sequences_path = '/kaggle/input/stanford-rna-3d-folding/train_sequences.csv'
train_labels_path = '/kaggle/input/stanford-rna-3d-folding/train_labels.csv'

train_sequences = pd.read_csv(train_sequences_path)
train_labels = pd.read_csv(train_labels_path)


train_sequences[1:3]


train_labels[0:3]


# Let's get the TM-score package. Don't forget to add USalign progamm into inputs. Take attention that it's listed under databases on Kaggle.
os.system('cp //kaggle/input/usalign/USalign /kaggle/working/')
os.system('sudo chmod u+x /kaggle/working//USalign')


# Quick check that we have USalign and it works. The output should be "#Total CPU time is  0.00 seconds" 
test = os.popen("./USalign predicted.pdb groundtruth.pdb -mol RNA").read()
test


# First, find 1RHT_A into train_labels.csv
test_rna = train_labels[train_labels["ID"].str.contains("1RHT_A", na=False)]
test_rna


# Let's write it into the fake predicted and ground thruth files and see if we get TM-score 1.0 with TM-align utility.
# Help function to format the data from our sample dataframe into the format accept by USalign utility
def write_target_line(
    atom_name, atom_serial, residue_name, chain_id, residue_num, x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='P'
) -> str:
    """
    Writes a single line of PDB format based on provided atom information.

    Args:
        atom_name (str): Name of the atom (e.g., "N", "CA").
        atom_serial (int): Atom serial number.
        residue_name (str): Residue name (e.g., "ALA").
        chain_id (str): Chain identifier.
        residue_num (int): Residue number.
        x_coord (float): X coordinate.
        y_coord (float): Y coordinate.
        z_coord (float): Z coordinate.
        occupancy (float, optional): Occupancy value (default: 1.0).
        b_factor (float, optional): B-factor value (default: 0.0).

    Returns:
        str: A single line of PDB string.
    """
    return f'ATOM  {atom_serial:>5d}  {atom_name:<4s} {residue_name:<3s} {chain_id:<1s}{residue_num:>4d}    {x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}          {atom_type}\n'

#predicted sample
with open("predicted.pdb", 'w') as target_file:
    for _, row in test_rna.iterrows():
        x_coord = row[f'x_1']
        y_coord = row[f'y_1']
        z_coord = row[f'z_1']

        if x_coord > -1e17 and y_coord > -1e17 and z_coord > -1e17: # I don't know why we need this check, but I hope it prevents from some important errors to appear
            # if True:
            #resolved_cnt += 1
            target_line = write_target_line(
                atom_name="C1",
                atom_serial=int(row['resid']),
                residue_name=row['resname'],
                chain_id='A',
                residue_num=int(row['resid']),
                x_coord=x_coord,
                y_coord=y_coord,
                z_coord=z_coord,
                atom_type='C',
                )
            target_file.write(target_line)


# Quick check that our values are writed correctly
!cat predicted.pdb


# Let's write the fake groundtruth sample 
with open("groundtruth.pdb", 'w') as target_file:
    for _, row in test_rna.iterrows():
        x_coord = row[f'x_1']
        y_coord = row[f'y_1']
        z_coord = row[f'z_1']

        if x_coord > -1e17 and y_coord > -1e17 and z_coord > -1e17: # I don't know why we need this check, but I hope it prevents from some important errors to appear
            # if True:
            #resolved_cnt += 1
            target_line = write_target_line(
                atom_name="C1",
                atom_serial=int(row['resid']),
                residue_name=row['resname'],
                chain_id='A',
                residue_num=int(row['resid']),
                x_coord=x_coord,
                y_coord=y_coord,
                z_coord=z_coord,
                atom_type='C',
                )
            target_file.write(target_line)


# Let's check that values are writed correctly
!cat groundtruth.pdb


# Now lets compare them with USalign utility
!./USalign predicted.pdb groundtruth.pdb -mol RNA -atom C1


#predicted wrong sample
with open("predicted_wrong.pdb", 'w') as target_file:
    for _, row in test_rna.iterrows():
        x_coord = row[f'x_1']
        y_coord = row[f'y_1']
        z_coord = row[f'z_1']

        if x_coord > -1e17 and y_coord > -1e17 and z_coord > -1e17: # I don't know why we need this check, but I hope it prevents from some important errors to appear
            # if True:
            #resolved_cnt += 1
            target_line = write_target_line(
                atom_name="C1",
                atom_serial=int(row['resid']),
                residue_name=row['resname'],
                chain_id='A',
                residue_num=int(row['resid']),
                x_coord=1,
                y_coord=2,
                z_coord=3,
                atom_type='C',
                )
            target_file.write(target_line)


#Check that we got the file
!ls


#Now let's get TM-score between the wrong predicted and groundtruth
!./USalign predicted_wrong.pdb groundtruth.pdb -mol RNA -atom C1

