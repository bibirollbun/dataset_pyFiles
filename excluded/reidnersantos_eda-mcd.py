!pip install -q PyPDF2
!apt-get install -q poppler-utils  # Required for pdf2image
!pip install -q pdf2image


import pandas as pd 
import os 
import matplotlib.pyplot as plt 
from collections  import Counter 
import random 
import seaborn as sns
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

from wordcloud import WordCloud
path = "/kaggle/input/make-data-count-finding-data-references"

test_pdf = f"{path}/test/PDF"
test_xml = f"{path}/test/XML"
train_pdf = f"{path}/train/PDF"
train_xml = f"{path}/train/XML"
train_labels = f"{path}/train_labels.csv"



print("Total train Files:")
print("PDF files:", len(os.listdir(train_pdf)))
print("XML files:", len(os.listdir(train_xml)))

print("\nTotal test Files:")
print("PDF files:", len(os.listdir(test_pdf)))
print("XML files:", len(os.listdir(test_xml)))



train_pdf_id = set(f[:-4] for f in os.listdir(train_pdf) if f.endswith(".pdf"))
no_set_train_pdf_id = [f[:-4] for f in os.listdir(train_pdf) if f.endswith(".pdf")]



train_xml_id = set(f[:-4] for f in os.listdir(train_xml) if f.endswith(".xml"))
no_set_train_xml_id = [f[:-4] for f in os.listdir(train_xml) if f.endswith(".xml")]

both = train_pdf_id & train_xml_id
only_pdf = train_pdf_id - train_xml_id
only_xml = train_xml_id - train_pdf_id

print(f"no_set_pdf: {len(no_set_train_pdf_id)}")
print(f"no_set_xml: {len(no_set_train_xml_id)}")

print(f"PDF arquivos:    {len(train_pdf_id)}")
print(f"XML arquivos:    {len(train_xml_id)}")
print(f"iguais:          {len(both)}")
print(f"somente no PDF : {len(only_pdf)}")
print(f"somente no XML : {len(only_xml)}")



df_labels = pd.read_csv(train_labels)
print("df_labels.shape",df_labels.shape)
df_labels['has_dataset'] = df_labels['dataset_id'] != 'Missing'

print("Data types:\n")
print(df_labels.dtypes)
print("df_labels.shape:", df_labels.shape)



print("has_dataset", df_labels['has_dataset'].nunique())
print("article_id", df_labels['article_id'].nunique())
print("dataset_id", df_labels['dataset_id'].nunique())
print("type", df_labels['type'].nunique())

#print("has_dataset", df_labels['has_dataset'].unique())
#print("article_id", df_labels['article_id'].unique())
print("dataset_id", df_labels['dataset_id'].unique())
print("type", df_labels['type'].unique())
(df_labels['dataset_id'] == 'Missing').sum()



missing_percent = (df_labels['type'] == 'Missing').mean() * 100
plt.figure(figsize=(6, 6))
plt.pie([missing_percent, 100-missing_percent], 
        labels=['Missing', 'Not Missing'], 
        autopct='%1.1f%%')
plt.title('ProporÃ§Ã£o de referÃªncias ausentes')
plt.show()



print("Has Dataset:")
print(df_labels['has_dataset'].describe())
print("\nDataset_id:")
print(df_labels['dataset_id'].describe())
print("\nArticle_id:")
print(df_labels['article_id'].describe())
print("\nType:")
print(df_labels['type'].describe())



type_counts = df_labels['type'].value_counts()
explode = [0.1] * len(type_counts)

plt.figure(figsize=(8, 6))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, explode=explode)
plt.title("Citation Type Distribution")
plt.axis('equal')
plt.tight_layout()
plt.show()



def extract_source(x):
    x = str(x)
    if x.startswith("https://doi.org/"):
        return x.split("/")[2]
    elif ":" in x:
        return x.split(":")[0]
    elif x[:3].isalpha():
        return x[:8]
    return "Unknown"

df_labels['source'] = df_labels['dataset_id'].apply(extract_source)



print(df_labels[df_labels['source'] == 'Unknown'].head())



print(df_labels['source'].value_counts())


vc = df_labels['source'].value_counts()
vc_filtered = vc[vc > 1]  # mostra apenas valores com mais de 10 ocorrÃªncias
print(vc_filtered)



df_labels['source'].value_counts().head(20).plot(kind='bar', figsize=(10,5))
plt.title("Top 20 Dataset Sources")
plt.xlabel("Fonte do Dataset")
plt.ylabel("NÃºmero de citaÃ§Ãµes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



print(df_labels['article_id'].value_counts())


df_labels['article_id'].value_counts().plot(kind='hist', bins=50)
plt.title("DistribuiÃ§Ã£o de CitaÃ§Ãµes por Artigo")
plt.xlabel("NÃºmero de citaÃ§Ãµes")
plt.ylabel("Quantidade de artigos")
plt.tight_layout()
plt.show()



df_aux = df_labels[df_labels['dataset_id'] != 'Missing']
datasets_por_artigo = df_aux.groupby('article_id')['dataset_id'].nunique()
datasets_por_artigo.hist(bins=20)
plt.title("DistribuiÃ§Ã£o de datasets Ãºnicos por artigo")
plt.xlabel("NÃºmero de datasets")
plt.ylabel("NÃºmero de artigos")
plt.tight_layout()
plt.show()



df_labels['format'] = df_labels['article_id'].apply(
    lambda x: 'PDF+XML' if x in train_pdf_id and x in train_xml_id else
              'PDF' if x in train_pdf_id else
              'XML' if x in train_xml_id else 'None'
)



pd.crosstab(df_labels['type'], df_labels['has_dataset']).plot(kind='bar', stacked=True)
plt.title("PresenÃ§a de dataset por tipo de citaÃ§Ã£o")
plt.xlabel("Tipo")
plt.ylabel("Contagem")
plt.tight_layout()
plt.show()



pd.crosstab(df_labels['format'], df_labels['type']).plot(kind='bar', stacked=True)
plt.title('DistribuiÃ§Ã£o de tipos de citaÃ§Ã£o por formato de artigo')
plt.xlabel('Formato do artigo')
plt.ylabel('Contagem')
plt.tight_layout()
plt.show()



def qtd_missing_em_todas(df):
    for col in df.columns:
        qtd = (df[col] == 'Missing').sum()
        if qtd > 0:
            print(f"Coluna '{col}' tem {qtd} valores 'Missing'")
        else:
            print(f"Coluna '{col}' nÃ£o tem valores 'Missing'")

qtd_missing_em_todas(df_labels)


### Verifica os xmls que tem type como missing, e fazer uma leitura em busca do dataset_id se Ã© primary ou secundary


df_labels.head()


df = pd.read_csv(train_labels)
print(len(df))
df.head()


linha_encontrada = df[df['article_id'].isin(train_xml_id)]
df_xml = linha_encontrada[linha_encontrada['type'] =='Missing']

df_xml


primeiro = next(iter(train_xml_id))
primeiro

for idx,rows in df.iterrows():
    #print(idx,rows['article_id'])
    if rows['article_id'] == primeiro:
        print(rows)
    


# Verificar se o mesmo dataset_id aparece com tipos diferentes
ambiguous_dataset_ids = (
    df_labels.groupby("dataset_id")["type"]
    .nunique()
    .reset_index()
    .query("type > 1")["dataset_id"]
)

print(f"Total de dataset_id com mÃºltiplos tipos: {len(ambiguous_dataset_ids)}")






type_per_article = df_labels.groupby('article_id')['type'].nunique()
print("Artigos com mais de um tipo de citaÃ§Ã£o:", (type_per_article > 1).sum())

type_per_article.value_counts().plot(kind='bar')
plt.title("NÃºmero de tipos diferentes por artigo")
plt.xlabel("Tipos Ãºnicos")
plt.ylabel("NÃºmero de artigos")
plt.show()



df_labels['type_per_article'] = df_labels['article_id'].map(type_per_article)
df_labels['mixed_type_article'] = df_labels['type_per_article'] > 1


sns.countplot(data=df_labels, x='mixed_type_article')
plt.title("Artigos com mÃºltiplos tipos de citaÃ§Ã£o")
plt.xlabel("Artigo possui mÃºltiplos tipos?")
plt.ylabel("NÃºmero de entradas")
plt.show()



df_aux = df_labels[df_labels['dataset_id'] != 'Missing']
top_articles = df_aux['article_id'].value_counts().head(10)

top_articles.plot(kind='bar')
plt.title("Top 10 artigos que mais citam datasets")
plt.xlabel("Article ID")
plt.ylabel("NÃºmero de citaÃ§Ãµes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



cross = pd.crosstab(df_labels['source'], df_labels['type'])
print(cross.head(10))



unique_by_type = df_labels[df_labels['dataset_id'] != 'Missing'].groupby('type')['dataset_id'].nunique()
print(unique_by_type)

unique_by_type.plot(kind='bar')
plt.title("NÂº de datasets Ãºnicos por tipo de citaÃ§Ã£o")
plt.xlabel("Tipo")
plt.ylabel("NÃºmero de datasets Ãºnicos")
plt.tight_layout()
plt.show()

duplicadas = df_labels[df_labels.duplicated(subset=['article_id', 'dataset_id'], keep=False)]
print(f"NÂº de citaÃ§Ãµes duplicadas (mesmo article_id + dataset_id): {len(duplicadas)}")




format_has_dataset = pd.crosstab(df_labels['format'], df_labels['has_dataset'], normalize='index')
format_has_dataset.plot(kind='bar', stacked=True)
plt.title("ProporÃ§Ã£o de has_dataset por formato de artigo")
plt.ylabel("ProporÃ§Ã£o")
plt.tight_layout()
plt.show()



df_labels


ccdc_missing = df_labels[
    df_labels['dataset_id'].str.contains("ccdc", case=False) & 
    (df_labels['type'] == "Missing")
]

print(f"Total ccdc com tipo 'Missing': {len(ccdc_missing)}")


def verifica_ccdc_nao_missing(df):
    # Filtra onde dataset_id comeÃ§a com "ccdc"
    df_ccdc = df[df['dataset_id'].str.lower().str.startswith("https://doi.org/10.5517/ccdc")]
    
    # Filtra onde o type NÃƒO Ã© 'Missing'
    df_nao_missing = df_ccdc[df_ccdc['type'] != 'Missing']
    
    if df_nao_missing.empty:
        print("âœ… Todos os dataset_id que comeÃ§am com 'ccdc' tÃªm type == 'Missing'")
    else:
        print("âš ï¸� Encontrados casos onde dataset_id com 'ccdc' tem type diferente de 'Missing':")
        print(df_nao_missing[['dataset_id', 'type']])

# Chamar funÃ§Ã£o
verifica_ccdc_nao_missing(df_labels)



def filtrar_missing_dataset_type(df):
    return df.loc[df['type'] == 'Missing', ['dataset_id', 'type']]



def filtrar_missing_dataset_type(df):
    return df.loc[
        (df['type'] == 'Missing') & (~df['dataset_id'].str.startswith('https://doi.org/10.5517/ccdc')),
        ['article_id','dataset_id', 'type']
    ]

filtrar_missing_dataset_type(df_labels)


print(train_pdf,"\n",train_labels)

random_article = random.choice(df_labels['article_id'])


idx = 3
teste_article = df_labels['article_id'][3]




print(df_labels.loc[3,['dataset_id','type']])
pdf_path = os.path.join(train_pdf,teste_article+'.pdf')
print(pdf_path)
#pdf_path = '/kaggle/input/make-data-count-finding-data-references/train/PDF/10.1002_2017jc013030.pdf'
try:
    images = convert_from_path(pdf_path, first_page=1, last_page=1)  # Convert only the first page
    plt.figure(figsize=(10, 15))
    plt.imshow(images[0])
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Error: {e}")




def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n--- PAGE BREAK ---\n"
    return text

pdf_text = extract_pdf_text(pdf_path)

print(pdf_text[-35000:-33000]) 


text = 'abcdefgp'
text[-4:-1]


wordcloud = WordCloud(width=800, height=400).generate(pdf_text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


import xml.etree.ElementTree as ET

xml_path = "/kaggle/input/make-data-count-finding-data-references/train/XML/10.1002_2017jc013030.xml"
tree = ET.parse(xml_path)
root = tree.getroot()

# Counter to limit output
line_counter = [0]  # Using list to allow modification in nested function
MAX_LINES = 30

def print_xml_limited(element, indent=0):
    if line_counter[0] >= MAX_LINES:
        return
    print('  ' * indent + f"<{element.tag}>")
    line_counter[0] += 1
    if line_counter[0] >= MAX_LINES:
        return
    for child in element:
        print_xml_limited(child, indent+1)
        if line_counter[0] >= MAX_LINES:
            return

print(f"First {MAX_LINES} lines of XML structure:")
print_xml_limited(root)
print("\n... (output truncated after 15 lines)")


xml_data = []
for elem in root.findall('.//article'):  # Adjust XPath to your structure
    xml_data.append({
        'id': elem.find('id').text,
        'title': elem.find('title').text,
        # Add more fields as needed
    })

df_xml = pd.DataFrame(xml_data)
df_xml.head()


