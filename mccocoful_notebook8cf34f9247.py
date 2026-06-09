import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv

# Base URL and output directory
base_url = "https://europepmc.org/pub/databases/pmc/TextMinedTerms/"
output_dir = "TextMinedTermsCSVs"
os.makedirs(output_dir, exist_ok=True)

# Fetch the HTML and parse links
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Get all .csv links
csv_links = [urljoin(base_url, link['href']) for link in soup.find_all('a') if link['href'].endswith('.csv')]
print(f"Found {len(csv_links)} CSV files. Downloading and modifying...")

# Download and modify each CSV
for link in csv_links:
    filename = os.path.join(output_dir, os.path.basename(link))

    # Download CSV
    with requests.get(link, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Modify the header
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if rows and len(rows[0]) > 0:
        rows[0][0] = 'acc_id'  # Replace first column header

    # Write the modified CSV back
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Processed: {filename}")

print("All files downloaded and updated.")


import polars as pl

acc_id_mentions = (
    pl.read_csv('/kaggle/working/TextMinedTermsCSVs/*.csv')
    .drop_nulls()
    .with_columns(pl.when(pl.col('acc_id').str.contains(r'10\.\d{4,9}/')).then('https://doi.org/'+pl.col('acc_id')).otherwise('acc_id').alias('dataset_id'))
)
acc_id_mentions


pmcid_doi_mapping = (
    pl.read_parquet('/kaggle/input/open-pmc/PMCID_DOI.parquet')
    .with_columns(article_id=pl.col('DOI').str.split('https://doi.org/').list.last().str.replace_all('/', '_'))
)


possible_mentions = acc_id_mentions.join(pmcid_doi_mapping, on=['PMCID']).select('article_id', 'dataset_id')


DOI_LINK='https://doi.org/'

LINKS={
    'dx': 'https://dx.doi.org/', 'acs': 'https://pubs.acs.org/doi/', 'osf': 'https://osf.io', 'ncbi': 'https://www.ncbi.nlm.nih.gov/', 'protatlas': 'https://www.proteinatlas.org/',
    'ame_tcr': 'https://tcr.amegroups.com/article/view/', 'ame_atm': 'https://atm.amegroups.com/article/view/', 'rna': 'https://www.rnajournal.org/cgi/doi/', 'ebi_ena': 'https://www.ebi.ac.uk/ena/data/view/',
    'ebi_arr': 'https://www.ebi.ac.uk/arrayexpress/experiments/', 'ame_jtd': 'https://jtd.amegroups.com/article/view/', 'rcsb': 'https://www.rcsb.org/structure/', 'uniprot': 'https://www.uniprot.org/uniprot/'
}

def is_link(prefix:str=DOI_LINK, name:str='dataset_id')->pl.Expr: return pl.col(name).str.starts_with(prefix)
def replace_link_doi(tag:str, name:str='dataset_id')->pl.Expr: return pl.when(is_link(LINKS[tag], name)).then(DOI_LINK+pl.col(name).str.split(LINKS[tag]).list.last().str.to_lowercase()).otherwise(name).alias(name)
def replace_link_acc(tag:str, name:str='dataset_id')->pl.Expr: return pl.when(is_link(LINKS[tag], name)).then(pl.col(name).str.split(LINKS[tag]).list.last().str.strip_chars('/')).otherwise(name).alias(name)


possible_mentions = possible_mentions.with_columns(pl.when(is_link()).then(DOI_LINK+pl.col('dataset_id').str.split(DOI_LINK).list.last().str.split('DOI:').list.last()).otherwise('dataset_id').alias('dataset_id'))
assert possible_mentions.filter(is_link('http').and_(~is_link())).is_empty(), "dataset_id contains links that are not doi.org"


mdc_corpus = (
    pl.read_csv('/kaggle/input/data-citation-corpus/*.csv')
    .filter(is_doi_link('publication'))
    .with_columns(pl.col('publication').str.split(DOI_LINK).list.last().str.to_lowercase().str.replace_all('/', '_').alias('article_id'))
    .select('article_id', pl.col('dataset').alias('dataset_id'))
    .with_columns(pl.when(is_link()).then(DOI_LINK+pl.col('dataset_id').str.split(DOI_LINK).list.last().str.split('DOI:').list.last().str.to_lowercase()).otherwise('dataset_id').alias('dataset_id'))
    .with_columns(replace_link_doi('dx'))
    .with_columns(replace_link_doi('acs'))
    .with_columns(replace_link_doi('ame_tcr'))
    .with_columns(replace_link_doi('ame_atm'))
    .with_columns(replace_link_doi('ame_jtd'))    
    .with_columns(replace_link_doi('rna'))
    .with_columns(pl.when(is_link(LINKS['osf'])).then(DOI_LINK+'10.17605/osf.io'+pl.col('dataset_id').str.split(LINKS['osf']).list.last().str.to_lowercase()).otherwise('dataset_id').alias('dataset_id'))
    .with_columns(replace_link_acc('ebi_ena'))
    .with_columns(replace_link_acc('ebi_arr'))    
    .with_columns(replace_link_acc('rcsb'))
    .with_columns(replace_link_acc('uniprot'))
    .with_columns(pl.when(is_link(LINKS['ncbi'])).then(pl.col('dataset_id').str.split('/').list.last()).otherwise('dataset_id').alias('dataset_id'))
    .with_columns(pl.when(is_link(LINKS['protatlas'])).then(pl.col('dataset_id').str.split(LINKS['protatlas']).list.last().str.split('/').list.first()).otherwise('dataset_id').alias('dataset_id'))
    .filter(~is_link('http').and_(~is_link()))
)


mdc_corpus.height, possible_mentions.height


new_corpus = pl.concat([mdc_corpus, possible_mentions]).unique()
new_corpus.height


new_corpus = new_corpus.filter(~pl.col('dataset_id').str.starts_with('DOI'))


new_corpus.write_parquet('data_citation_corpus.parquet')

