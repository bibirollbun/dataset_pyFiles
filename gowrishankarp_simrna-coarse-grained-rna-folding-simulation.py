"Om Namah Shivaya!! ğŸ™�ğŸ™�"


!cp ../input/simrna-v3-2/SimRNA_64bitIntel_Linux.tgz ./SimRNA_64bitIntel_Linux.tgz
!tar -zxvf SimRNA_64bitIntel_Linux.tgz


%cd SimRNA_64bitIntel_Linux
!sudo ln -s $(pwd)/* /usr/local/bin


%%writefile config.dat
NUMBER_OF_ITERATIONS 1600000
TRA_WRITE_IN_EVERY_N_ITERATIONS 8000

INIT_TEMP 1.35
FINAL_TEMP 0.90

BONDS_WEIGHT 1.0
ANGLES_WEIGHT 1.0
TORS_ANGLES_WEIGHT 0.0
ETA_THETA_WEIGHT 0.40


%%writefile 7TAX_M.seq
CUAAGAAAUUCACGGCGGGCUUGAUGUCCGCGUCUACCUGAUUCACUGCCGUAUAGGCAGC


PDB_NAME = "7TAX_M"
!SimRNA -c config.dat -s {PDB_NAME}.seq -o {PDB_NAME} -E 10


!cat {PDB_NAME}_??.trafl > {PDB_NAME}_ALL.trafl
!clustering {PDB_NAME}_ALL.trafl 0.01 3.63 >& {PDB_NAME}_clustering.log #NOTE: we used 36.3Ã… rmsd threshold which is 0.1*seq_length


!SimRNA_trafl2pdbs {PDB_NAME}_01-000001.pdb {PDB_NAME}_ALL_thrs7.10A_clust01.trafl 1 AA
!SimRNA_trafl2pdbs {PDB_NAME}_01-000001.pdb {PDB_NAME}_ALL_thrs7.10A_clust02.trafl 1 AA
!SimRNA_trafl2pdbs {PDB_NAME}_01-000001.pdb {PDB_NAME}_ALL_thrs7.10A_clust03.trafl 1 AA


!SimRNA â€“c config.dat â€“n 0 â€“p {PDB_NAME}_01-000001.pdb â€“o {PDB_NAME}_native -E 10 >& {PDB_NAME}_native.log


!cat {PDB_NAME}_native.trafl {PDB_NAME}_all.trafl > {PDB_NAME}_wNative.trafl


!calc_rmsd_to_1st_frame {PDB_NAME}_wNative.trafl {PDB_NAME}_wNative.rmsd_e

