from google.colab import files
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import os
%pip install biopython
from Bio import SeqIO


# @title Install IgBLAST.

!wget https://ftp.ncbi.nih.gov/blast/executables/igblast/release/LATEST/ncbi-igblast-1.22.0-x64-linux.tar.gz
!tar -xvzf ncbi-igblast-1.22.0-x64-linux.tar.gz
os.environ['IGDATA'] = '/content/ncbi-igblast-1.22.0'
os.environ['PATH'] = os.environ['PATH'] + f':{os.environ["IGDATA"]}/bin'


# @title Download reference sets.

# SOURCE: https://bitbucket.org/kleinstein/immcantation/src/master/scripts/fetch_imgtdb.sh
%%bash
FILE_PATH="imgt_database"


if [ ! -s "$FILE_PATH" ]
then
    mkdir ${FILE_PATH}
fi

REPERTOIRE="imgt"
SPECIES_QUERY=("human:Homo+sapiens"
               "mouse:Mus"
               "rat:Rattus+norvegicus"
               "rabbit:Oryctolagus+cuniculus"
               "rhesus_monkey:Macaca+mulatta")
# Associative array (for BASH v3) with species name replacements
SPECIES_REPLACE=('human:s/Homo sapiens/Homo_sapiens/g'
                 'mouse:s/Mus musculus/Mus_musculus/g'
                 'rat:s/Rattus norvegicus/Rattus_norvegicus/g'
                 'rabbit:s/Oryctolagus cuniculus/Oryctolagus_cuniculus/g'
                 'rhesus_monkey:s/Macaca mulatta/Macaca_mulatta/g')

SPECIES="human"
KEY=${SPECIES%%:*}
VALUE=${SPECIES#*:}
REPLACE_VALUE=${SPECIES_REPLACE[$COUNT]#*:}

for CHAIN in TRAV TRAJ TRBV TRBD TRBJ TRDV TRDD TRDJ TRGV TRGJ
do
    URL="https://www.imgt.org/genedb/GENElect?query=7.1+${CHAIN}&species=Homo+sapiens"
    echo $URL
    FILE_NAME="${FILE_PATH}/${REPERTOIRE}_${KEY}_${CHAIN}.fasta"
    TMP_FILE="${FILE_NAME}.tmp"
    wget $URL -O $TMP_FILE -q
    awk '/<pre>/{i++}/<\/pre>/{j++}{if(j==2){exit}}{if(i==2 && j==1 && $0!~"^<pre>"){print}}' $TMP_FILE > $FILE_NAME

    #  Check file exists and is not empty
    if [ ! -s "$FILE_NAME" ]
    then
          echo "IMGT Fasta file does not exist, or is empty. Is the IMGT server online?"
          exit 1
      fi

      sed -i.bak "$REPLACE_VALUE" $FILE_NAME && rm $FILE_NAME.bak
      rm $TMP_FILE
done

FILE_PATH_AA="imgt_database"

# V amino acid for TCR
for CHAIN in TRAV TRBV TRDV TRGV
do
    URL="https://www.imgt.org/genedb/GENElect?query=7.3+${CHAIN}&species=Homo+sapiens"
    FILE_NAME="${FILE_PATH_AA}/${REPERTOIRE}_aa_${KEY}_${CHAIN}.fasta"
    TMP_FILE="${FILE_NAME}.tmp"
    #echo $URL
    wget $URL -O $TMP_FILE -q
    awk '/<pre>/{i++}/<\/pre>/{j++}{if(j==2){exit}}{if(i==2 && j==1 && $0!~"^<pre>"){print}}' $TMP_FILE > $FILE_NAME

    # Check file exists and is not empty
    if [ ! -s "$FILE_NAME" ]
    then
        echo "IMGT Fasta file does not exist, or is empty. Is the IMGT server online?"
        exit 1
    fi

    sed -i.bak "$REPLACE_VALUE" $FILE_NAME && rm $FILE_NAME.bak
    rm $TMP_FILE
done


# @title Flag pseudogenes (we're currently not removing/flagging by name).

# SOURCE: rewritten from https://bitbucket.org/kleinstein/immcantation/src/master/scripts/clean_imgtdb.py + added pseudogene flag step.

def flag_pseudogene_name(gene_name):
    pseudogene_flags = set(['P', 'ORF', '[ORF]', '(P)'])
    orf_flags = set(['ORF', '[ORF]'])
    basic_name = gene_name.split('|')[1]
    if gene_name.split('|')[3] in pseudogene_flags:
        return f'{basic_name}_pseudogene'
    elif gene_name.split('|')[3] in orf_flags:
        return f'{basic_name}_orf'
    else:
        return basic_name

def handle_pseudogenes(fasta_files, outfile = 'imgt_data/imgt_human_TRV.fasta', flag_pseudogenes=True):
  '''
  removes duplicate entries; flags (or not) pseudogenes + orphons.
  '''

  seqs = dict((p.name, str(p.seq)) for p in SeqIO.parse(fasta_files[0], format='fasta'))
  for fasta in fasta_files[1:]:
    seqs.update(dict((p.name, str(p.seq)) for p in SeqIO.parse(fasta, format='fasta')))
  if flag_pseudogenes:
    seqs = dict((flag_pseudogene_name(p), seqs[p]) for p in seqs)
  else:
    seqs = dict((p.split('|')[1], seqs[p]) for p in seqs)
  if os.path.exists(outfile) is False:
    write_mode = 'w'
  else:
    write_mode = 'a'
  with open(outfile, write_mode) as f:
    for x in seqs:
      f.write(f'>{x}\n{seqs[x].replace(".", "")}\n')

def return_pseudogenes(fasta_files):
  pseudogene_flags = set(['P', 'ORF', '[ORF]', '(P)'])
  orf_flags = set(['ORF', '[ORF]'])
  seqs = [p.name for p in SeqIO.parse(fasta_files[0], format='fasta')]
  for fasta in fasta_files[1:]:
    seqs.extend([p.name for p in SeqIO.parse(fasta, format='fasta')])
  return set([p.split('|')[1] for p in seqs if p.split('|')[3] in pseudogene_flags]),  set([p.split('|')[1] for p in seqs if p.split('|')[3] in orf_flags])

if os.path.exists('imgt_database/imgt_human_TRV.fasta'):
  os.remove('imgt_database/imgt_human_TRV.fasta')
if os.path.exists('imgt_database/imgt_aa_human_TRV.fasta'):
  os.remove('imgt_database/imgt_aa_human_TRV.fasta')
handle_pseudogenes(['imgt_database/imgt_human_TRAV.fasta', 'imgt_database/imgt_human_TRBV.fasta', 'imgt_database/imgt_human_TRGV.fasta', 'imgt_database/imgt_human_TRDV.fasta'], 'imgt_database/imgt_human_TRV.fasta', flag_pseudogenes=False)
handle_pseudogenes(['imgt_database/imgt_aa_human_TRAV.fasta', 'imgt_database/imgt_aa_human_TRBV.fasta', 'imgt_database/imgt_aa_human_TRGV.fasta', 'imgt_database/imgt_aa_human_TRDV.fasta'], 'imgt_database/imgt_aa_human_TRV.fasta', flag_pseudogenes=False)
pseudogenes, orfs = return_pseudogenes(['imgt_database/imgt_human_TRAV.fasta', 'imgt_database/imgt_human_TRBV.fasta', 'imgt_database/imgt_human_TRGV.fasta', 'imgt_database/imgt_human_TRDV.fasta'])


!makeblastdb -parse_seqids -dbtype nucl -in imgt_database/imgt_human_TRV.fasta -out imgt_database/imgt_human_TRV
!makeblastdb -parse_seqids -dbtype prot -in imgt_database/imgt_aa_human_TRV.fasta -out imgt_database/imgt_aa_human_TRV


def parse_igblastp(fmt_file: str):
  alignment_keys = set(['FR1-IMGT', 'CDR1-IMGT', 'FR2-IMGT', 'CDR2-IMGT', 'FR3-IMGT', 'CDR3-IMGT (germline)', 'Total'])
  total_queries= {}
  alignments_total = {}
  for query in open(fmt_file, 'r').read().split('# Query: '):
      query_block = query.split('\n')
      query_id = query_block[0]
      hit_table = pd.DataFrame([p.split('\t') for p in query_block if p.startswith("V")])
      if hit_table.empty:
        continue
      alignments_total[query_id] = pd.DataFrame([p.split('\t') for p in query_block if p.split('\t')[0] in alignment_keys])
      hit_table.columns = ['gene', 'query_id', 'subject_id','pid', 'alignment_length', 'mismatch','gap_opens','gaps', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
      hit_table['pid'] = hit_table['pid'].map(float)
      hit_table['identifier'] = int(query_id)
      v_call = ','.join(hit_table[hit_table['pid']==hit_table['pid'].max()]['subject_id'].values)
      v_identity = hit_table['pid'].max()
      total_queries[int(query_id)] = {'v_call':v_call, 'v_identity':v_identity}
  return pd.DataFrame(total_queries).T.reset_index(), alignments_total


# @title Upload test.csv file.

files.upload()



# @title Run and parse IgBLASTp.
df = pd.read_csv(f'test.csv')
with open(f'test_beta.fasta', 'w') as x:
  for k, p in df.iterrows():
    x.write('>'+str(p.ID)+'\n'+p.TCRb+'\n')
with open(f'test_alpha.fasta', 'w') as x:
  for k, p in df.iterrows():
    x.write('>'+str(p.ID)+'\n'+p.TCRa+'\n')

!igblastp -query test_beta.fasta -germline_db_V imgt_database/imgt_aa_human_TRV -ig_seqtype TCR -organism human -out beta_output.fmt7 -outfmt 7
beta_calls, beta_ids = parse_igblastp('beta_output.fmt7')
!igblastp -query test_alpha.fasta -germline_db_V imgt_database/imgt_aa_human_TRV -ig_seqtype TCR -organism human -out alpha_output.fmt7 -outfmt 7
alpha_calls, alpha_ids = parse_igblastp('alpha_output.fmt7')

df = pd.merge(left = df, right = beta_calls.rename(columns={'v_call':'Vb_inferred', 'v_identity':'Vb_inferred_identity'}), left_on = 'ID', right_on = 'index')
df = pd.merge(left = df, right = alpha_calls.rename(columns={'v_call':'Va_inferred', 'v_identity':'Va_inferred_identity'}), left_on = 'ID', right_on = 'index')


def summary(df, report_file):
  summary = {}
  # no 100% identical inference possible
  _ = (df[(df['Vb_inferred_identity']!=100)]['TCRb'].nunique() / df['TCRb'].nunique())*100
  vals = df[(df['Vb_inferred_identity']!=100)]['Vb'].unique()
  summary['no_match_beta'] = _
  print(f'No 100% identical TRBV inference possible for {_:.2f} Beta chains with the currently assigned alleles: {vals}')
  remainder = df[(df['Vb_inferred']!='')].drop_duplicates('TCRb')
  remainder['TCRb_match'] =remainder.apply(lambda x:x.Vb in set(x.Vb_inferred.split(',')), axis=1)
  different = remainder[remainder['TCRb_match']==False].shape[0]/remainder.shape[0]
  print(f'Of remainder ({100-_:.2f}%), {different:.2f}% have a TCRb call that is NOT a 100% match in this reference.')
  summary['different_beta'] = different

  _ = (df[(df['Va_inferred_identity']!=100)]['TCRa'].nunique() / df['TCRa'].nunique())*100
  vals = df[(df['Va_inferred_identity']!=100)]['Va'].unique()
  summary['no_match_alpha'] = _
  print(f'No 100% identical TRAV inference possible for {_:.2f} Alpha chains with the currently assigned alleles: {vals}')
  remainder = df[(df['Vb_inferred']!='')].drop_duplicates('TCRa')
  remainder['TCRa_match'] =remainder.apply(lambda x:x.Va in set(x.Va_inferred.split(',')), axis=1)
  different = remainder[remainder['TCRa_match']==False].shape[0]/remainder.shape[0]
  print(f'Of remainder ({100-_:.2f}%), {different:.2f}% have a TCRa call that is NOT a 100% match in this reference.')
  summary['different_alpha'] = different
  return summary




out = summary(df, '')


df['Va_type'] = df['Va'].map(lambda x:'ORF' if x in orfs else 'P' if x in pseudogenes else 'F')
df['Vb_type'] = df['Vb'].map(lambda x:'ORF' if x in orfs else 'P' if x in pseudogenes else 'F')
nonfunctional = orfs.union(pseudogenes)
df['Va_inferred_F'] = df['Va_inferred'].map(lambda x:','.join([x for x in x.split(',') if x not in nonfunctional]) if x!='' else '')
df['Vb_inferred_F'] = df['Vb_inferred'].map(lambda x:','.join([x for x in x.split(',') if x not in nonfunctional]) if x!='' else '')
#### there is no difference here i.e. all of the inferred genes are non-pseudogenes.
assert df[(df['Va_inferred_F']!=df['Va_inferred'])].shape[0] == 0
assert df[(df['Vb_inferred_F']!=df['Vb_inferred'])].shape[0] == 0



df[['ID', 'Va', 'Va_inferred','Va_inferred_identity', 'Vb', 'Vb_inferred', 'Vb_inferred_identity']].to_csv(f'test_inferred_genes.csv')


files.download('test_inferred_genes.csv')


# @title Identity by region shows <100% identity is only due to the "germline" CDR3-IMGT region, for clarification.

vb_identities = pd.DataFrame(dict((int(p), dict(beta_ids[p][[0,7]].values)) for p in beta_ids)).map(float).T
va_identities = pd.DataFrame(dict((int(p), dict(alpha_ids[p][[0,7]].values)) for p in alpha_ids)).map(float).T

fig, ax = plt.subplots(2,6, figsize=(15, 5), sharey=True)
ax = ax.flatten()
for i, region in enumerate(['FR1-IMGT', 'CDR1-IMGT', 'FR2-IMGT', 'CDR2-IMGT', 'FR3-IMGT', 'CDR3-IMGT (germline)']):
    sns.histplot(data=va_identities[region], ax=ax[i])
for x, region in enumerate(['FR1-IMGT', 'CDR1-IMGT', 'FR2-IMGT', 'CDR2-IMGT', 'FR3-IMGT', 'CDR3-IMGT (germline)']):
    sns.histplot(data=vb_identities[region], ax=ax[i+x+1])
fig.tight_layout()


