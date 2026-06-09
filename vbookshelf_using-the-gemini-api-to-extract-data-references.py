


import pandas as pd
import numpy as np
import os



MODEL_ID = "gemini-2.5-flash"

base_path = '../input/make-data-count-finding-data-references/'


from kaggle_secrets import UserSecretsClient

# Initialize
user_secrets = UserSecretsClient()

# Google API Key
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")


from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)


def upload_pdf_to_files_api(article_id, source="train"):

    path_to_pdf = base_path + source + '/' + 'PDF/' + f'{article_id}.pdf'

    sample_file = client.files.upload(file=path_to_pdf)
    
    print(f"Uploaded file '{sample_file.name}' as: {sample_file.uri}")

    return sample_file



def delete_all_files_from_files_api():

    # ------------------------
    # List all stored files
    # ------------------------
    
    file_list = []

    for f in client.files.list():

        # Get the file name
        fname = f.name
        
        # Add the file name to a list
        file_list.append(fname)

    # Check the num files stored in the files api
    num_files = len(file_list)
    
    if num_files != 0:
    
        # ------------------------
        # Delete all stored files
        # ------------------------

        for fname in file_list:

            # Delete the file
            client.files.delete(name=fname)

            print(f"Deleted {fname}.")
            
    else:
        print('No files found.')
            


path = base_path + 'train_labels.csv'

df_data = pd.read_csv(path)

print(df_data.shape)

df_data.head(10)


delete_all_files_from_files_api()


# Sample files from the train set.
# Labels are included.

# 10.1002_2017jc013030	https://doi.org/10.17882/49388	Primary
# 10.1002_anie.201916483	Missing	Missing
# 10.1038_s41396-020-00885-8	IPR002477	Secondary
# 10.1002_anie.202007717	Missing	Missing
# 10.1002_anie.202007717	Missing	Missing


# Upload the five pdf files

# 10.1002_2017jc013030	https://doi.org/10.17882/49388	Primary
pdf_file1 = upload_pdf_to_files_api(article_id="10.1002_2017jc013030", source="train")

# 10.1002_anie.201916483	Missing	Missing
pdf_file2 = upload_pdf_to_files_api(article_id="10.1002_anie.201916483", source="train")

# 10.1038_s41396-020-00885-8	IPR002477	Secondary
pdf_file3 = upload_pdf_to_files_api(article_id="10.1038_s41396-020-00885-8", source="train")

# 10.1002_anie.202007717	Missing	Missing
pdf_file4 = upload_pdf_to_files_api(article_id="10.1002_anie.202007717", source="train")

# 10.1002_ece3.5260	https://doi.org/10.5061/dryad.2f62927	Primary
pdf_file5 = upload_pdf_to_files_api(article_id="10.1002_ece3.5260", source="train")


# List the file names of the files that have been uploaded

for f in client.files.list():
    
    print(f.name)


text = """

Citation Reference

Data citations (references to research data) from the full text of the scientific literature should be tagged as primary or secondary:

Primary - raw or processed data generated as part of the paper, specifically for the study
Secondary - raw or processed data derived or reused from existing records or published data


Paper and Dataset Identifiers

Each object (paper and dataset) has a unique, persistent identifier to represent it. In this competition there will be two types:

DOIs are used for all papers and some datasets. They take the following form: https://doi.org/[prefix]/[suffix]. Examples:
https://doi.org/10.1371/journal.pone.0303785
https://doi.org/10.5061/dryad.r6nq870

Accession IDs are used for some datasets. They vary in form by individual data repository where the data live. Examples:
"GSE12345" (Gene Expression Omnibus dataset)
“PDB 1Y2T” (Protein Data Bank dataset)
"E-MEXP-568" (ArrayExpress dataset)


Data Citation Mining Examples

To illustrate how research data are mentioned in the scientific literature, here are some examples:
Note: in the text, the dataset identifier may appear with or without the 'https://doi.org' stem.

Data: https://doi.org/10.5061/dryad.6m3n9
In-text span: "The data we used in this publication can be accessed from Dryad at doi:10.5061/dryad.6m3n9."
Citation type: Primary

Data: https://doi.org/10.5061/dryad.c394c12
In-text span: "Phenotypic data and gene sequences are available from the Dryad Digital Repository: http://dx.doi.org/10.5061/dryad.c394c12"
Citation type: Primary

Data: https://doi.org/10.25386/genetics.11365982
In-text span: "The authors state that all data necessary for confirming the conclusions presented in the article are represented fully within the article. Supplemental material available at figshare: https://doi.org/10.25386/genetics.11365982."
Citation type: Primary

Data: GSE37569, GSE45042, GSE28166
In-text span: "Primary data for Agilent and Affymetrix microarray experiments are available at the NCBI Gene Expression Omnibus (GEO, http://www.ncbi.nlm.nih.gov/geo/) under the accession numbers GSE37569, GSE45042 , GSE28166"
Citation type: Primary

Data: pdb 5yfp
In-text span: “Figure 1. Evolution and structure of the exocyst. A) Cartoon representing the major supergroups, which are referred to in the text. The inferred position of the last eukaryotic common ancestor (LECA) is indicated and the supergroups are colour coordinated with all other figures. B) Structure of trypanosome Exo99, modelled using Phyre2 (intensive mode). The model for the WD40/b-propeller (blue) is likely highly accurate. The respective orientations of the a-helical regions may form a solenoid or similar, but due to a lack of confidence in the disordered linker regions this is highly speculative. C and D) Structure of the Saccharomyces cerevisiae exocyst holomeric octameric complex. In C the cryoEM map (at level 0.100) is shown and in D, the fit for all eight subunits (pdb 5yfp). Colours for subunits are shown as a key, and the orientation of the cryoEM and fit are the same for C and D. All structural images were modelled by the authors from PDB using UCSF Chimera.”
Citation type: Secondary

Data: E-MTAB-10217, PRJE43395
In-text span: “The datasets presented in this study can be found in online repositories. The names of the repository/repositories and accession number(s) can be found below: https://www.ebi.ac.uk/arrayexpress/, E-MTAB-10217 and https://www.ebi.ac.uk/ena, PRJE43395.”
Citation type: Secondary
"""

prompt1 = f"From the attached paper please extract each data source that's mentioned and classify each data source as primary or secondary. Only consider data sources that have a DOI or Accession ID. If no data sources are mentioned then output None. Output your response as JSON with the following keys: Data, In-text span, Page number, Citation type. Examples: ###{text}###"



# 10.1002_2017jc013030	https://doi.org/10.17882/49388	Primary

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt1, pdf_file1]
)

print(response.text)


# 10.1002_anie.201916483	Missing	Missing

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt1, pdf_file2]
)

print(response.text)


# 10.1038_s41396-020-00885-8	IPR002477	Secondary

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt1, pdf_file3]
)

print(response.text)


prompt = "Is this reference anywhere in this paper: IPR002477"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, pdf_file3]
)

print(response.text)


# 10.1002_anie.202007717	Missing	Missing

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt1, pdf_file4]
)

print(response.text)


# 10.1002_ece3.5260	https://doi.org/10.5061/dryad.2f62927	Primary

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt1, pdf_file5]
)

print(response.text)


prompt2 = f"From the attached paper please extract each data source that's mentioned and classify each data source as primary or secondary. If no data sources are mentioned then output None. Output your response as JSON with the following keys: Data, In-text span, Page number, Citation type. Examples: ###{text}###"



# 10.1002_2017jc013030	https://doi.org/10.17882/49388	Primary

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt2, pdf_file1]
)

print(response.text)







