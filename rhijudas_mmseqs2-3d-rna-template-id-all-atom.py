# if False, don't check on temporal_cutoff -- during training on known structures, will get lots of leakage. 
# If going for Early Sharing Prize, set to True.
CHECK_TEMPORAL_CUTOFF = True

# Number of templates to use. 
# Here set to 5 to prepare a mock submission
# Should make larger (e.g., 40) if using templates for modeling. 
MAX_TEMPLATES = 5

# Better to use nan when preparing files for templates, to allow easy recognition of which coordinates are missing.
# But for this example, using 0.0 to avoid errors in scoring the final submission.csv
NULL_VALUE = 0.0  

# directory with test_sequences.csv
INPUT_DIR = '/kaggle/input/stanford-rna-3d-folding'


#!wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
#!tar xvfz /kaggle/working/mmseqs-linux-avx2.tar.gz
!rsync -avL /kaggle/input/mmseqs2/mmseqs /kaggle/working/
!chmod 755 /kaggle/working/mmseqs/bin/mmseqs


!/kaggle/working/mmseqs/bin/mmseqs createdb {INPUT_DIR}/PDB_RNA/pdb_seqres_NA.fasta pdb_seqres_NA --dbtype 2 # > MMseqs_createDB.log


import csv
input_file=f'{INPUT_DIR}/test_sequences.csv'
output_file='test_sequences.fasta'
SEQ_COUNT = 0
with open(input_file, 'r', newline='') as csv_file, open(output_file, 'w') as fasta_file:
    csv_reader = csv.reader(csv_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        if len(row) >= 2:
            fasta_file.write(f">{row[0]}\n{row[1]}\n")
            SEQ_COUNT += 1
print(f'Will run MMseqs2 on {SEQ_COUNT} sequences')


!/kaggle/working/mmseqs/bin/mmseqs easy-search /kaggle/working/test_sequences.fasta /kaggle/working/pdb_seqres_NA testResult.txt tmp --search-type 3 --format-output "query,target,evalue,qstart,qend,tstart,tend,qaln,taln" 


!pip3 install /kaggle/input/biopython/biopython-1.85-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps


import sys
sys.path.append('/kaggle/input/create-templates-csv-py/')

from create_templates_csv import get_template_labels

sequence_file = input_file
mmseqs_results_file = 'testResult.txt' 
skip_temporal_cutoff = not CHECK_TEMPORAL_CUTOFF
cif_dir = f"{INPUT_DIR}/PDB_RNA"
output_labels,output_allatom_labels,targets = get_template_labels( input_file, mmseqs_results_file, skip_temporal_cutoff,
                                                       MAX_TEMPLATES, cif_dir )



from create_templates_csv import output_template_labels_to_csv

# Create a DataFrame and write to CSV
outdir = './'
outfile = 'submission.csv'
output_template_labels_to_csv( output_labels, output_allatom_labels, targets, outdir, outfile )




