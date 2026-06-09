!pip install rdkit
!pip install py3dmol
!pip install radonpy-pypi


import sys
sys.path.append('/kaggle/input/lammps-demo-hengck')

import os
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

import numpy as np
import joblib


print('%%script echo skipping')
if 0: 
    from run_write_lammps_data import write_lammps_data
    
    id = 992359277
    smiles = '*CC(*)(C)C(=O)OCCF'
    ter_smiles = '*C'  #termination unit
    work_dir ='/kaggle/input/lammps-demo-hengck/demo-id-992359277'
    
    write_lammps_data(smiles, ter_smiles, work_dir=work_dir)
    #let's input the ouput from this function using py3Dmol below


#view top conformer
work_dir ='/kaggle/input/lammps-demo-hengck/demo-id-992359277'

def view_mol(m):
    view = py3Dmol.view(width=700, height=500)
    mol_block = Chem.MolToMolBlock(m, confId=0)
    view.addModel(mol_block, 'mol')
    view.setStyle({'stick': {}})
    view.setBackgroundColor('0xeeeeee')  # Light gray background
    view.spin('y')  
    view.zoomTo()
    view.show()

mol = joblib.load(f'{work_dir}/mol.pkl')
view_mol(mol)



#view polymer
homopoly = joblib.load(f'{work_dir}/homopoly.pkl')
view_mol(homopoly)


#view amorphous cell of 10 polymers
ac = joblib.load(f'{work_dir}/ac.pkl')
view_mol(ac)


import os
os.environ['OMP_NUM_THREADS'] = '4'
print('OMP_NUM_THREADS:',os.getenv('OMP_NUM_THREADS'))

import subprocess
print('--- start of check subprocess env-----------------------')
#check subprocess has OMP_NUM_THREADS = 4
## subprocess.run(["env"])
print("** don't show the content to the public !!!! it has your secret key !!!!")
print( '--- end of check subprocess env------------------------')
print('')



print('%%script echo skipping')
if 0: 
    
    import shutil
    import datetime
    
    from run_write_lammps_script_eq3 import write_input as write_input_eq3
    from run_write_lammps_script_eq2 import write_input as write_input_eq2
    from run_write_lammps_script_eq1 import write_input as write_input_eq1
    
    
    #lammp script lancher --
    LAMMPS_EXEC = \
    	'<your_path>/lammps-install/bin/lmp'
    
    def run_preset(
    	name ='eq1',
    	descrption = 'packing simulation',
    	cmd        = 'mpirun -n 4 {LAMMPS_EXEC} -sf omp -pk omp 4 -in {name}.in -sc none -log {name}_log.lammps',
    	cmd_debug  = 'mpirun -n 1 {LAMMPS_EXEC} -sf omp -pk omp 4 -in {name}.in',
    	debug = False,
    	work_dir = '',
    ):
    	write_input_eq = {
    		'eq1': write_input_eq1,
    		'eq2': write_input_eq2,
    		'eq3': write_input_eq3,
    	}[name]
    
    	cwd = os.getcwd()
    	os.chdir(work_dir)
    
    	dt_start = datetime.datetime.now()
    	print(f'** Start {descrption} {name}.....', str(dt_start))
    	write_input_eq(filename=f'{name}.in')
    	if not debug:
    		#cmd = f'mpirun -n 4 /home/hp/lammps-install/bin/lmp -sf omp -pk omp 4 -in {name}.in -sc none -log {name}-log.lammps'
    		cmd = cmd.format(LAMMPS_EXEC=LAMMPS_EXEC, name=name)
    		print('cmd:',cmd)
    		cp = subprocess.run([
    			cmd
    		], shell=True,
    			stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='UTF-8',
    		)
    	else:  # -sc none
    		#cmd = f'mpirun -n 1 /home/hp/lammps-install/bin/lmp -sf omp -pk omp 4 -in {name}.in'
    		cmd = cmd_debug.format(LAMMPS_EXEC=LAMMPS_EXEC, name=name)
    		print('cmd:',cmd)
    		cp = subprocess.run([
    			cmd
    		], shell=True,
    		)
    	dt = datetime.datetime.now()
    	print(f'** Complete {descrption} {name}. Elapsed time = {str(dt - dt_start)}')
    	print('cp.returncode:', cp.returncode)
    	print("STDOUT:\n", str(cp.stdout))
    	print("STDERR:\n", str(cp.stderr)[:100])
    	print('')
    	os.chdir(cwd)
    
    
    #----------------------------------------------
    #start MD here !!!
    
    os.makedirs(f'{work_dir}/simulate', exist_ok=True)
    shutil.copy(f'{work_dir}/data/eq1.data',f'{work_dir}/simulate/eq1.data')
    
    
    run_preset(
        name='eq1',
        descrption='packing simulation',
        cmd      ='mpirun -n 4 {LAMMPS_EXEC} -sf omp -pk omp 4 -in {name}.in -sc none -log {name}_log.lammps',
        cmd_debug='mpirun -n 1 {LAMMPS_EXEC} -sf omp -pk omp 4 -in {name}.in',
        debug=False,
        work_dir=f'{work_dir}/simulate',
    )
    
    run_preset(
        name='eq2',
        descrption='annealing simulation',
        cmd       ='mpirun -n 2 {LAMMPS_EXEC} -sf gpu -pk gpu 2 omp 4 -in {name}.in -sc none -log {name}_log.lammps',
        cmd_debug ='mpirun -n 1 {LAMMPS_EXEC} -sf gpu -pk gpu 2 omp 4 -in {name}.in',
        debug = False,
        work_dir=f'{work_dir}/simulate',
    )
     
    run_preset(
        name='eq3',
        descrption='sampling simulation',
        cmd       ='mpirun -n 2 {LAMMPS_EXEC} -sf gpu -pk gpu 2 omp 4 -in {name}.in -sc none -log {name}_log.lammps',
        cmd_debug ='mpirun -n 1 {LAMMPS_EXEC} -sf gpu -pk gpu 2 omp 4 -in {name}.in',
        debug = False,
        work_dir=f'{work_dir}/simulate',
    )
    
    
    #----------------------------------------------
    # read and print result
    
    from my_help import *
    
    df1 =  read_lammps_timeavg_profile(f'{work_dir}/simulate/eq3.rg.profile')
    df2 =  read_lammps_log(f'{work_dir}/simulate/eq3.log')
    
    print(id, smiles)
    print('Rg', df1['Rg'].mean())
    print('Density', df2['Density'].mean())
    
'''
992359277	*CC(*)(C)C(=O)OCCF	
Rg: 15.987818950000001
Density: 1.2028783378109456
'''

