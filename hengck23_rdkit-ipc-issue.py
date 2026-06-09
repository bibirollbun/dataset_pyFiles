!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import Chem

import numpy as np

smiles = ['*C#Cc1cc(OC(COCCOCCOCCOC)COCCOCCOCCOC)c(C#Cc2cc(OCCOCCOCCOCCC(=O)[O][Na])c(*)cc2OCCOCCOCCOCCC(=O)[O][Na])cc1OC(COCCOCCOCCOC)COCCOCCOCCOC',
       '*C#CC(Cn1c2ccc(CCCCCCCCCCCCCCCC)cc2c2cc(CCCCCCCCCCCCCCCC)ccc21)=C(*)Cn1c2ccc(CCCCCCCCCCCCCCCC)cc2c2cc(CCCCCCCCCCCCCCCC)ccc21',
       '*c1ccc(-c2c(-c3ccc(-c4ccccc4)cc3)cc(-c3ccc(-c4cc(-c5ccc(-c6ccccc6)cc5)c(-c5ccc(-n6c(=O)c7cc8c(=O)n(*)c(=O)c8cc7c6=O)cc5)c(-c5ccc(-c6ccccc6)cc5)c4)cc3)cc2-c2ccc(-c3ccccc3)cc2)cc1',
       '*CCN(CCCCOc1ccc(N=Nc2ccc(CCCCCC)cc2)cc1)CCOC(=O)c1ccc(C(=O)O)c(C(=O)Nc2ccc(C(C)(C)c3ccc(C(C)(C)c4ccc(NC(=O)c5ccc(C(=O)O*)cc5C(=O)O)cc4)cc3)cc2)c1',
       '*Oc1ccc2c(c1)C(c1ccc(N(c3ccccc3)c3ccccc3)cc1)(c1ccc(N(c3ccccc3)c3ccccc3)cc1)c1cc(Oc3ccc(-c4c5ccccc5c(-c5ccc(*)cc5)c5ccc(CCC)cc45)cc3)ccc1-2',
       '*O[Si](O[Si](O[Si](O[Si](CC[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)CC[Si](*)(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1',
       '*c1cccc(N2C(=O)c3ccc(Oc4ccc5c(c4)C(C)(c4ccc(Oc6ccc7c(c6)C(=O)N(c6cccc(N8C(=O)c9ccc(Oc%10ccc(C%11(C)CC(C)(C)c%12cc(Oc%13ccc%14c(c%13)C(=O)N(*)C%14=O)ccc%12%11)cc%10)cc9C8=O)c6)C7=O)cc4)CC5(C)C)cc3C2=O)c1'
]


for t in range(10):
	value = []
	for s in smiles:
		mol = Chem.MolFromSmiles(s)
		c = Chem.MolToSmiles(mol, canonical=True)
		mol = Chem.MolFromSmiles(c)
		ipc = Descriptors.Ipc(mol)
		value.append(ipc)
	value = np.array(value)
	value = np.log10(value) 
	#value = np.round(value)
	print(value)


for t in range(10):
	value = []
	for s in smiles:
		mol = Chem.MolFromSmiles(s)
		c = Chem.MolToSmiles(mol, canonical=True)
		mol = Chem.MolFromSmiles(c)
		ipc = Descriptors.AvgIpc(mol)
		value.append(ipc)
	value = np.array(value)
	value = np.log10(value)
	#value = np.round(value)
	print(value)
    
'''
#from my machine
[25.28097287 26.15170788 25.76549485 18.99670056 20.16273586 24.42981841
 26.67922508]

#from kaggle machine
[24.3330795  23.22496944 25.67465791 19.00270598 19.96499612 24.72912241
 26.61551054]
'''

