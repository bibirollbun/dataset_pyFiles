!pip install biopython


from Bio import SeqIO

with open('/kaggle/input/mave-db-amino-acid-substitution-prediction/gencode.v48.pc_translations.fa', 'rt') as instream:
    gencode_48_sequences = [
        record.id.split('|') + [record.seq]
        for record in SeqIO.parse(instream, 'fasta')
    ]


# Show a few of the sequences
gencode_48_sequences[0:2]


sequence_dict = {array[0]: array[-1] for array in gencode_48_sequences}


sequence_dict['ENSP00000002829.3']




