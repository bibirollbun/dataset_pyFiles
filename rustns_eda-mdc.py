import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")

train_data.head()


train_data.info


print(f"Número de artigos: {train_data.shape[0]}")
print(f"Quantidade de colunas: {train_data.shape[1]}\n")
print(train_data.columns)


print(train_data.isnull().sum())
train_data.duplicated().sum()


train_data.describe()


type_distribution = train_data['type'].value_counts()
print(f"Distribuição entre os tipos: \n{type_distribution}")


print("Porcentagem:")

train_data['type'].value_counts(normalize=True).round(4) * 100


plt.figure(figsize=(10, 6))
sns.countplot(x='type', data=train_data, palette='Set1')
plt.title('Distribuição (type) no dataset', fontsize=15)
plt.xlabel('type', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.xticks(rotation=15)
plt.show()


article_counts = train_data['article_id'].value_counts()
articles_with_multiple = (article_counts > 1).sum()

print(f"Artigos que referenciam mais de um dataset: {articles_with_multiple}")

dataset_counts = train_data['dataset_id'].value_counts()
datasets_with_multiple = (dataset_counts > 1).sum()

print(f"Datasets citados em mais de um artigo: {datasets_with_multiple}")


article_prefixes = train_data['article_id'].apply(lambda x: str(x).split('_')[0] if isinstance(x, str) else None)
print("ID artigo prefixos:\n", article_prefixes.value_counts().head(10))


import re

def categorize_dataset_id(ds):
    ds = ds.lower()
    if "doi.org" in ds:
        return "DOI"
    elif re.match(r"^gse\d+", ds):
        return "GEO"
    elif re.match(r"^prj[ena\d]+", ds):
        return "ENA"
    elif re.match(r"^e-\w+", ds):
        return "ArrayExpress"
    else:
        return "Other"

category_counts = train_data["dataset_id"].apply(categorize_dataset_id).value_counts()

total = len(train_data)
percentages = (category_counts / total) * 100

result = category_counts.to_frame(name='Contador')
result['Porcentagem'] = percentages.round(2).astype(str) + '%'

print(result)


dataset_prefixes = train_data['dataset_id'].apply(lambda x: str(x).split('/')[3] if isinstance(x, str) and 'https://doi.org/' in x else None)
print("ID dataset (doi) prefixos:\n", dataset_prefixes.value_counts().head(10))


top_articles = article_counts.head(30)

plt.figure(figsize=(10, 6))
sns.barplot(y=top_articles.index, x=top_articles.values, palette="muted")
plt.title("Top 30 artigos por número de datasets referenciados")
plt.xlabel("Número de datasets referenciados")
plt.ylabel("ID artigo")
plt.tight_layout()
plt.show()


article_types = train_data.groupby("article_id")["type"].unique().reset_index()

def classify_citation(types):
    if set(types) == {"Primary"}:
        return "Apenas Primary"
    elif set(types) == {"Secondary"}:
        return "Apenas Secondary"
    else:
        return "Ambos"

article_types["Citações"] = article_types["type"].apply(classify_citation)

article_types["Citações"].value_counts().plot(kind='bar', title='Tipos de citação (type) por artigo')


type_per_article = train_data.groupby('article_id')['type'].value_counts().unstack().fillna(0)

top_article_ids = article_counts.head(30).index
subset = type_per_article.loc[top_article_ids]

subset.plot(kind='bar', stacked=True, colormap='Set2')
plt.title("Tipo de citação (type) dos 10 top artigos")
plt.xlabel("ID artigo")
plt.ylabel("Número de datasets")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


import os
from lxml import etree as ET
from tqdm import tqdm


xml_dir = '/kaggle/input/make-data-count-finding-data-references/train/XML' 

available_files = set(f.replace('.xml', '') for f in os.listdir(xml_dir) if f.endswith('.xml'))
train_ids = train_data['article_id'].unique()
valid_ids = list(set(train_ids) & available_files)

sample_articles = valid_ids[:10]


print(sample_articles)


def parse_xml_article(article_id, xml_dir):
    xml_path = os.path.join(xml_dir, f'{article_id}.xml')
    results = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for elem in root.iter():
            if not isinstance(elem.tag, str):
                continue

            tag_name = ET.QName(elem.tag).localname.lower()

            if tag_name == 'ext-link':
                link_type = elem.attrib.get('ext-link-type')
                href = elem.attrib.get('{http://www.w3.org/1999/xlink}href')
                text = elem.text

                parent = elem.getparent()
                section = None
                paragraph = None

                ancestor = parent
                while ancestor is not None:
                    ancestor_tag = ET.QName(ancestor.tag).localname.lower()
                    if ancestor_tag == 'sec' and section is None:
                        section = ''.join(ancestor.itertext()).strip()[:300]
                    elif ancestor_tag == 'p' and paragraph is None:
                        paragraph = ''.join(ancestor.itertext()).strip()[:300]
                    ancestor = ancestor.getparent()

                results.append({
                    'article_id': article_id,
                    'link_text': text,
                    'link_type': link_type,
                    'href': href,
                    'section_text': section,
                    'paragraph_text': paragraph
                })

    except Exception as e:
        print(f"Error processing {article_id}: {e}")

    return results


all_citations = []

for article_id in tqdm(sample_articles):
    citations = parse_xml_article(article_id, xml_dir)
    all_citations.extend(citations)

citations_df = pd.DataFrame(all_citations)
citations_df.head(20)


def normalize_doi(s):
    if not isinstance(s, str):
        return None
    s = s.lower().strip()
    s = re.sub(r'^https?://(dx\.)?doi\.org/', '', s)
    match = re.match(r'(10\.\d{4,9}/[-._;()/:a-z0-9]+)', s)
    return match.group(1) if match else None

citations_df['clean_doi'] = citations_df['href'].apply(normalize_doi)
train_data['clean_dataset_id'] = train_data['dataset_id'].apply(normalize_doi)

merged = pd.merge(
    citations_df,
    train_data,
    how='left',
    left_on=['article_id', 'clean_doi'],
    right_on=['article_id', 'clean_dataset_id']
)

print(merged['type'].value_counts(dropna=False))

