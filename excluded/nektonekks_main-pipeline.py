import fitz
import os
import re
import unicodedata
import pandas as pd
import numpy as np
import vllm
import torch
import spacy
import gc
import ray

from statistics import multimode
from collections import Counter
from tqdm import tqdm
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdftypes import resolve1
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

import warnings
warnings.filterwarnings('ignore')
spacy.require_gpu()


ner_model = spacy.load('/kaggle/input/local_spacy_model/transformers/default/1/en_core_web_lg_local')


# DIR_PATH = '/kaggle/input/make-data-count-finding-data-references/test/PDF' if os.getenv('KAGGLE_IS_COMPETITION_RERUN') \
#             else '/kaggle/input/make-data-count-finding-data-references/train/PDF'
DIR_PATH = '/kaggle/input/make-data-count-finding-data-references/test/PDF'

df_citations = pd.DataFrame(columns = ['article_id', 'dataset_id', 'pattern', 'context', 'start'])


df_prefixes = pd.read_csv('/kaggle/input/prefixes/prefixes.csv', dtype = {'prefix': str})
article_marks = ['10.SERV/CROSSREF', '10.SERV/CNKI', '10.SERV/MEDRA', '10.SERV/KISTI', '10.SERV/JALC'
                 '10.SERV/AIRITI', '10.SERV/ISTIC', '10.SERV/MEDRA-TEST', '10.SERV/HAND', '10.SERV/JALCTEST']

article_prefixes = df_prefixes[df_prefixes['type'].isin(article_marks)]['prefix'].astype(str).values


re_table = re.compile(r'(?<![A-Za-z])(t\s*a\s*b\s*l\s*e\s*\d+)(?![A-Za-z0-9])', re.IGNORECASE)
re_table_mark = re.compile(r'<(\d+)>')
re_mark = re.compile(r'<.+>')

re_doi = re.compile(r'(?<![A-Za-z0-9])(1\s*[01]\s*\.(\s*\d\s*){1,9}\s*/\s*\S{1,70})')
re_doi_prefix = re.compile(r'/(1[01]\.\d+)/')
re_alphafold = re.compile(r'(?<![A-Za-z0-9])(AF\s*\-\s*[A-Z0-9]+\s*\-\s*F\d+(\s*\-\s*model\s*\-\s*v\d+)?)(?![A-Za-z0-9])')
re_arrayexpress = re.compile(r'(?<![A-Za-z0-9])(E\s*\-\s*[A-Z]{4}\s*\-\s*\d+)(?![A-Za-z0-9])')
re_biomodels = re.compile(r'(?<![A-Za-z0-9])((((BIOMD)|(MODEL))\d{10})|(BMID\d{12}))')
re_bioproject = re.compile(r'(?<![A-Za-z0-9])(PRJ((NA)|(EB)|(DB)|(EA)|(DA)|(NZ)|(DG)|(NS)|(NE))\d+)(?![A-Za-z0-9])') 
re_biosample = re.compile(r'(?<![A-Za-z0-9])(SAM[NED][A-Z]?\d+)(?![A-Za-z0-9])')
re_biostudies = re.compile(r'(?<![A-Za-z0-9])(S\s*\-\s*[A-Z]{4}[\-\_A-Z]*\d+)(?![A-Za-z0-9])')
re_cellosaurus = re.compile(r'(?<![A-Za-z0-9])(CVCL\s*_\s*[A-Z0-9]{4})(?![A-Za-z0-9])')
re_chembl = re.compile(r'(?<![A-Za-z0-9])(CHEMBL\d+)(?![A-Za-z0-9])')
re_dbgap = re.compile(r'(?<![A-Za-z0-9])(phs[0-9]{6}(\s*\.\s*v\d+\s*\.\s*p\d+)?)(?![A-Za-z0-9])')
re_ebisc = re.compile(r'(?<![A-Za-z0-9])([A-Z]{2,6}\s*i\d{3}\-\s*[A-Z]{1,2})(?![A-Za-z0-9])')
re_efo = re.compile(r'(?<![A-Za-z0-9])(EFO\s*_\s*\d{7})(?![A-Za-z0-9])')
re_ega = re.compile(r'(?<![A-Za-z0-9])(EGA[DSC]\d{11})(?![A-Za-z0-9])')
re_emdb = re.compile(r'(?<![A-Za-z0-9])(EMD\s*\-\s*\d{4,5})(?![A-Za-z0-9])')
re_empiar = re.compile(r'(?<![A-Za-z0-9])(EMPIAR\s*\-\s*\d{5,})(?![A-Za-z0-9])')
re_geo = re.compile(r'(?<![A-Za-z0-9])(GSE\d+)(?![A-Za-z0-9])')
re_gisaid = re.compile(r'(?<![A-Za-z0-9])(EPI\s*(\s*_\s*ISL\s*_\s*)?\d+)(?![A-Za-z0-9])')
re_hipsci = re.compile(r'(?<![A-Za-z0-9])(HPSI\d{4}\s*i\s*\-\s*[a-z]+\s*_\s*\d+)(?![A-Za-z0-9])')
re_hpa = re.compile(r'(?<![A-Za-z0-9])(((HPA)|(CAB))\d{6})(?![A-Za-z0-9])')
re_igsr = re.compile(r'(?<![A-Za-z0-9])(((GM)|(NA)|(HG))\d{5})(?![A-Za-z0-9])')
re_intact = re.compile(r'(?<![A-Za-z0-9])(EBI\s*\-\s*[0-9]+)(?![A-Za-z0-9])')
re_interpro = re.compile(r'(?<![A-Za-z0-9])(IPR\d{6})(?![A-Za-z0-9])')
re_metabolights = re.compile(r'(?<![A-Za-z0-9])((MTBLS\d+))(?![A-Za-z0-9])')
re_mint = re.compile(r'(?<![A-Za-z0-9])(((MINT)|(IM))\s*\-\s*\d{1,7})(?![A-Za-z0-9])')
re_nct = re.compile(r'(?<![A-Za-z0-9])(NCT\d{8})(?![A-Za-z0-9])')
re_pfam = re.compile(r'(?<![A-Za-z0-9])(PF\d{5})(?![A-Za-z0-9])')
re_pxd = re.compile(r'(?<![A-Za-z0-9])(PXD\d{6})(?![A-Za-z0-9])')
re_reactome = re.compile(r'(?<![A-Za-z0-9])((R\s*\-\s*[A-Z]{3}\s*\-\s*\d+(\s*\-\s*\d+)?(\s*\.\s*\d+)?)|(REACT\s*_\s*\d+(\s*\.\s*\d+)?))(?![A-Za-z0-9])')
re_refseq = re.compile(r'(?<![A-Za-z0-9])(((NC)|(NM))\s*_\s*\d+(\s*\.\s*\d+)?)(?![A-Za-z0-9])')
re_rfam = re.compile(r'(?<![A-Za-z0-9])(RF\d{5})(?![A-Za-z0-9])')
re_rnacentral = re.compile(r'(?<![A-Za-z0-9])(URS[0-9A-F]{10}(\s*\_\s*\d+)?)(?![A-Za-z0-9])')
re_sra = re.compile(r'(?<![A-Za-z0-9])(([SE]R[PRX]\d{6,}))(?![A-Za-z0-9])') 
re_treefam = re.compile(r'(?<![A-Za-z0-9])(TF\d{6})(?![A-Za-z0-9])')
re_uniparc = re.compile(r'(?<![A-Za-z0-9])(UPI[A-F0-9]{10})(?![A-Za-z0-9])')

id_patterns = [re_refseq, re_gisaid, re_arrayexpress, re_cellosaurus, re_empiar, re_bioproject, re_sra, re_chembl, re_interpro, re_biosample, re_pfam, re_geo, re_dbgap,
              re_emdb, re_igsr, re_intact, re_reactome, re_rfam, re_uniparc, re_biomodels, re_alphafold, re_biostudies, re_hpa, re_pxd, re_ebisc, re_efo,
              re_hipsci, re_metabolights, re_mint, re_nct, re_rnacentral, re_treefam]


re_pdb_loc = re.compile(r'(?<![A-Za-z0-9])(pdb)(?![A-Za-z0-9])', re.IGNORECASE)
re_pdb = re.compile(r'(?<![A-Za-z0-9])([0-9]((([A-Z0-9]{3})|([a-z0-9]{3}))))(?![A-Za-z0-9])')

re_gen_loc = re.compile(r'(?<![A-Za-z0-9])((g\s*e\s*n\s*b\s*a\s*n\s*k)|(a\s*c\s*c\s*e\s*s\s*s\s*i\s*o\s*n)|(acc\s*(\.)?))(?![A-Za-z0-9])', re.IGNORECASE) #75
re_gen = re.compile(r'(?<![A-Za-z0-9])([A-Z]{1,2}[0-9]{5,})(?![A-Za-z0-9])')

id_loc_patterns = [(re_pdb_loc, re_pdb, 200), (re_gen_loc, re_gen, 200)]


def extract_prefix(dataset, pattern = re_doi_prefix):
    return re.search(pattern, dataset).group(1)


def make_local_regex(link, special_chars = '\^$.|?*+()[]{}'):
    regex = r'\s*'.join(char if char not in special_chars else '\\' + char for char in link)
    return re.compile(regex, re.IGNORECASE)


def pair_chars(phrase, chars = [('(', ')'), ('[', ']'), ('{', '}')]):
    for char in chars:
        if phrase[-1] == char[1] and phrase.count(char[0]) != phrase.count(char[1]):
            return False

    return True


def doi_correct(doi_cit):
    while doi_cit[-1] in '.,;:!?"\'/' or not pair_chars(doi_cit):
        doi_cit = doi_cit[:-1]

    doi_cit = re.sub(r'[\-\‐\-\‒\–\—\―]', '-', doi_cit)
    return 'https://doi.org/' + re.sub(r'\s+', '', doi_cit).lower()


def doi_select(link, pattern = re_doi):
    matcher = re.search(pattern, link.decode('utf-8', errors = 'ignore'))
    if matcher:
        return doi_correct(matcher.group(1))


def extract_doi(pdf_path):
    links = []
    
    with open(pdf_path, 'rb') as f:
        parser = PDFParser(f)
        doc = PDFDocument(parser)
        
        for page in PDFPage.create_pages(doc):
            if 'Annots' in page.attrs:
                annots = resolve1(page.attrs['Annots'])
                for annot in annots:
                    annot_obj = resolve1(annot)
                    if annot_obj and 'A' in annot_obj:
                        action = resolve1(annot_obj['A'])
                        if action and 'URI' in action:
                            uri = action['URI']
                            links.append(uri)
    
    links = set(filter(lambda cit: cit, map(doi_select, links)))
    filtered_links_by_prefix = list(filter(lambda link: extract_prefix(link) not in article_prefixes, links))

    return filtered_links_by_prefix


def read_spans(line, size_coef = 0.8):
    spans = []
    
    for span in line['spans']:
        spans.append(span)
        
    normal_size = max([span['size'] for span in spans])
    spans = list(filter(lambda span: span['size'] >= size_coef * normal_size, spans))
    text = unicodedata.normalize('NFKC', ''.join([span['text'] for span in spans]))
    cleaned_text = re.sub(r"[^A-Za-z0-9 \.,;\:\!\?\(\)\-\‐\-\‒\–\—\―/\&\@\#\$\%\№_\*\+\=\|\[\]]+", "", text)# + ' '

    decoded_text = re.sub(r"[\-\‐\-\‒\–\—\―]+", "-", cleaned_text)
    unspaced_text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000\t]", " ", decoded_text)
    pointed_text = re.sub(r"[\．\.\｡]", ".", unspaced_text)
    
    text_info = {
        'text': pointed_text,
        'font_size': normal_size
    }
    
    return text_info


def concat_text_blocks(blocks_info, occ_threshold = 5):
    counter = Counter([block['text'].lower().strip() for block in blocks_info])
    filtered_blocks_info = list(filter(lambda block: counter[block['text'].lower().strip()] <= occ_threshold, blocks_info))
    block_sizes = set(block['font_size'] for block in filtered_blocks_info)
        
    new_blocks = {size: [] for size in block_sizes}
    for i in range(len(filtered_blocks_info)):
        new_blocks[filtered_blocks_info[i]['font_size']].append(filtered_blocks_info[i]['text'])
            
    sorted_blocks = dict(sorted(new_blocks.items(), reverse = True))
    filtered_sorted_blocks = {k: v for k, v in sorted_blocks.items() if len(v) > 0}
    
    full_text = ''
    for key in filtered_sorted_blocks.keys():
        full_text += f"\n{' '.join(filtered_sorted_blocks[key])}"
        
    return full_text


def read_by_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    blocks_text = []
    authors = []
    
    for page_num, page in enumerate(doc):
        structure = page.get_text('dict')
        page_height = structure['height']
        
        for block in structure['blocks']:
            if block['type'] == 0:
                
                block_texts = []
                for line in block['lines']:
                    block_texts.append(read_spans(line))
                    
                block_text = ' '.join([text_info['text'] for text_info in block_texts])
                font_size = multimode([text_info['font_size'] for text_info in block_texts])[0]
                
                block_info = {
                    'text': re.sub(r' {2,}', ' ', block_text),
                    'font_size': round(font_size, 2)
                }
                
                blocks_text.append(block_info)

        if page_num == 0 or (page_num in [1, 2] and len(authors) <= 5):
            structured_first_page = concat_text_blocks(blocks_text)

            block_num = 0
            page_blocks = re.split(r'\n', structured_first_page)
            while len(authors) <= 5 and block_num < len(page_blocks):
                block = page_blocks[block_num]
                
                ent_text = ner_model(block)
                authors += [ent.text for ent in ent_text.ents if ent.label_ == 'PERSON']

                block_num += 1
        
    doc.close()
    return blocks_text, authors


def mark_blocks(blocks, patterns, loc_patterns, mark_pattern):
    ids = set()
    ordered_text = '<!>'.join(block['text'] for block in blocks)
    
    for pat in patterns:
        found = [link.group(1) for link in re.finditer(pat, ordered_text)]
        ids.update(found)

    for loc_pat in loc_patterns:
        keywords_info = [(link.start(), loc_pat[2]) for link in re.finditer(loc_pat[0], ordered_text)]
        short_contexts = [ordered_text[max(0, kw[0] - kw[1]): min(len(ordered_text), kw[0] + kw[1])] for kw in keywords_info]

        found = [link.group(1) for text in short_contexts for link in re.finditer(loc_pat[1], text) 
                if link.start() != 0 and link.end() != len(text)]

        if loc_pat[1] == re_pdb:
            found = [link for link in found if any([char.isalpha() for char in link])]

        if loc_pat[1] == re_gen:
            found = [link for link in found if len(link) >= 6]

        ids.update(found)
    
    links = []
    for ident in ids:
        local_regex = make_local_regex(ident)
        links.extend([link for link in re.finditer(local_regex, ordered_text)])
    
    marks = []
    for link in links:
        context = ordered_text[max(link.start() - 1000, 0): min(link.start() + 1000, len(ordered_text))]
        link_marks = [(mark.group(1), mark.start()) for mark in re.finditer(mark_pattern, context)]
    
        if len(link_marks) > 0:
            main_mark = min(link_marks, key = lambda item: abs(len(context) // 2 - item[1]))[0]
            marks.append((main_mark.lower().replace(' ', ''), link.start()))
    
    sorted_marks = sorted(marks, key = lambda item: item[1], reverse = True)
    
    for mark in sorted_marks:
        text_mark = re.search(r'\d+', mark[0]).group()
        ordered_text = ordered_text[:mark[1]] + f'<{text_mark}>' + ordered_text[mark[1]:]
    
    marked_blocks = ordered_text.split('<!>')

    for i in range(len(marked_blocks)):
        blocks[i]['text'] = marked_blocks[i]
    
    return blocks


def doi_compare(doi_cit, doi_link):
    links_matrix = pd.DataFrame(np.zeros((len(doi_cit), len(doi_link))), columns = list(doi_link), index = list(doi_cit))

    for col in links_matrix.columns:
        comparings = [(col in link or link in col) if col != link else False for link in doi_cit]
            
        links_matrix[col] = comparings

    summary = links_matrix.sum()
    filtered_links = summary[summary == 0].index.tolist()

    return filtered_links


def extract_doi_by_text(text, pattern = re_doi):
    doi_positions = [(link.start(), link.end()) for link in re.finditer(pattern, text)]

    approved_links = []
    for pos in doi_positions:
        link = text[pos[0]:pos[1]]
        nearest_words = text[pos[1]: min(pos[1] + 200, len(text))].split(' ')

        for i in range(len(nearest_words)):
            word = nearest_words[i].strip()

            if len(word) <= 3 or len(word) >= 50:
                break

            trunc_parts = ['http', 'www']
            if any([char in word for char in trunc_parts]):
                break

            trunc_starts = ['[', '(']
            if any([char == word[0] for char in trunc_starts]):
                break

            signs = '.-‐–—'
            not_alpha = not any([char.isalpha() for char in word]) and any([char in signs for char in word])
            not_broken_line = (word.islower() or word.isupper()) and link[-1] != ')' and any([char.isalpha() for char in word]) and any([char in signs for char in word])
            diff_chars = any([char.isalpha() for char in word]) and any([char.isdigit() for char in word])
            truncated_end = link[-1] == '.' and not any([char.isalpha() for char in word])
            without_digits_suffix = all([not char.isdigit() for char in link.split('/')[-1]]) if len(link.split('/')[-1].strip()) > 0 else False

            if sum([word.count(char) for char in signs]) <= 1 and len(word) - 1 in [word.find(char) for char in signs]:
                not_broken_line = False
                not_alpha = False
            
            if any([not_alpha, not_broken_line, diff_chars, truncated_end, without_digits_suffix]):
                link += word
            else:
                break

        trunc_chars = '@&=+$,'
        end_trunc_chars = '-‐–—/'
        
        without_stranges = all([char not in link if len(link) > 0 and char != link.strip()[-1] else True for char in trunc_chars])
        not_truncated = len(link) > 0 and link.strip()[-1] not in end_trunc_chars
        normal_length = 0 < len(link) <= 70
        without_diff_size = not (any([char.islower() for char in link]) and any([char.isupper() for char in link]))
        without_end_pairs = len(link) > 0 and not (pair_chars(link) and link[-1] in '])}')
        
        if all([without_stranges, not_truncated, normal_length, without_diff_size, without_end_pairs]):
            approved_links.append(link)

    links = list(set(map(doi_correct, approved_links)))
    filtered_by_prefix = list(filter(lambda link: extract_prefix(link) not in article_prefixes, links))
    filtered_links = doi_compare(filtered_by_prefix, filtered_by_prefix)

    return filtered_links


def search_context(text, pattern, cont_size = 300, min_batch_size = 50):
    contexts, starts = [], []
    count = len(re.findall(pattern, text))

    if count == 0:
        return [], [], text
        
    batch_size = max(cont_size // count, min_batch_size)
    reiter = re.finditer(pattern, text)
    
    for link in reiter:
        cont = '...' + text[max(link.start() - batch_size, 0): min(link.start() + batch_size, len(text))] + '...'
        contexts.append(cont)
        starts.append(link.start())

        text = text[:link.start()] + '!' * (link.end() - link.start()) + text[link.end():]
    
    return contexts, starts, text


def find_all(filename, text, initial_text, pattern, text_links, df_res):
    if pattern == re_doi:
        links = extract_doi(DIR_PATH + '/' + filename)
        links = list(filter(lambda link: re.sub('/', '_', link.replace('https://doi.org/', '')) != filename[:-4].lower(), links))
        filtered_text_links = doi_compare(links, text_links)

        final_links = links + filtered_text_links
        
    else:
        reiter = pattern.finditer(initial_text)
        final_links = [re.sub(r'\s+', '', link.group(1)) for link in reiter]
        
    for found in final_links:
        local_pattern = make_local_regex(found.replace('https://doi.org/', ''))
        
        if pattern == re_doi:
            cont_size = 400
            min_batch_size = 75
        else:
            cont_size = 400
            min_batch_size = 75
            
        contexts, starts, text = search_context(initial_text, local_pattern, cont_size, min_batch_size)

        if len(contexts) > 0:
            df_res.loc[len(df_res)] = [filename[:-4], found, pattern, contexts, starts]
                
    return text


def find_by_loc(filename, ordered_text, structured_text, initial_text, loc_pattern, df_res):
    keywords_info = [(link.start(), loc_pat[2]) for link in re.finditer(loc_pat[0], ordered_text)]
    short_contexts = [ordered_text[max(0, kw[0] - kw[1]): min(len(ordered_text), kw[0] + kw[1])] for kw in keywords_info]

    links = [re.sub(r'\s+', '', link.group(1)) for text in short_contexts for link in re.finditer(loc_pat[1], text) 
            if link.start() != 0 and link.end() != len(text)]

    if loc_pat[1] == re_pdb:
        links = [link for link in links if any([char.isalpha() for char in link])]

    if loc_pat[1] == re_gen:
        links = [link for link in links if len(link) >= 6]

    for found in links:
        loc_regex = make_local_regex(found)
        contexts, starts, structured_text = search_context(initial_text, loc_regex)

        if len(contexts) > 0:
            df_res.loc[len(df_res)] = [filename[:-4], found, loc_pattern[1], contexts, starts]

    return structured_text


def validate_authors(authors):
    if len(authors) == 0:
        return 'Not found'
        
    corrected_authors = list(map(lambda author: re.sub(r'[^A-Za-z\.\s\-\|;]', '', author).strip(), authors))
    filtered_by_start_size = list(map(lambda author: ' '.join([name_part for name_part in author.split(' ') if len(name_part) > 0 and name_part[0].isupper()]), corrected_authors))
    filtered_authors_by_length = list(filter(lambda name: 3 <= sum([char.isalpha() for char in name]) and len(name.split()) >= 1, filtered_by_start_size))

    return ', '.join(filtered_authors_by_length)


files = sorted(os.listdir(DIR_PATH))
structured_texts = dict()
texts_authors = dict()

for file in tqdm(files):
    blocks, authors = read_by_blocks(DIR_PATH + '/' + file)
    marked_blocks = mark_blocks(blocks, id_patterns, id_loc_patterns, re_table)
    
    structured_text = concat_text_blocks(marked_blocks)
        
    structured_texts[file[:-4]] = structured_text
    initial_text = structured_text
    
    text_dois = extract_doi_by_text(structured_text)
    
    all_link_patterns = [re_doi] + id_patterns
    for pattern in all_link_patterns:
        structured_text = find_all(file, structured_text, initial_text, pattern, text_dois, df_citations)

    ordered_text = '\n'.join(block['text'] for block in marked_blocks)

    for loc_pat in id_loc_patterns:
        structured_text = find_by_loc(file, ordered_text, structured_text, initial_text, loc_pat, df_citations)

    texts_authors[file[:-4].replace('/', '_')] = validate_authors(authors)


df_citations = df_citations.drop_duplicates(subset = ['article_id', 'dataset_id']).reset_index(drop = True)

df_dois = df_citations[df_citations['dataset_id'].str.startswith('http')].drop(columns = ['start']).reset_index(drop = True)
df_ids = df_citations[~df_citations['dataset_id'].str.startswith('http')].reset_index(drop = True)

df_dois['context'] = df_dois['context'].apply(lambda contexts: ';\n'.join(contexts))
df_ids = df_ids.explode(['context', 'start'], ignore_index = True).sort_values(by = ['article_id', 'start']).reset_index(drop = True)


def nearest_links_count(row, df, density_threshold = 250):
    links_pos = df[df['article_id'] == row['article_id']]['start'].to_list()
    return len([link for link in filter(lambda pos: abs(pos - row['start']) <= density_threshold, links_pos)]) - 1


def identify_table(row, mark_pattern, cent_size = 15):
    if row['cluster_type'] == 'Outer':
        return np.nan

    length = len(row['context'])
    center = row['context'][length // 2 - cent_size: length // 2 + cent_size]
    matcher = re.search(mark_pattern, center)

    if matcher:
        return matcher.group(1)
        
    return np.nan


def find_table_context(row, cont_size = 300, min_batch_size = 75):
    context = ''
    table = row['table']
    table_number = re.search(r'\d+', table).group()
    local_pattern = make_local_regex('table' + table_number)
   
    text = re.sub(r'<\d+>', '', structured_texts[row['article_id']])
    batch_size = max(cont_size // (len(re.findall(local_pattern, text))), min_batch_size)
    
    for found in re.finditer(local_pattern, text):
        context += '...' + text[found.start() - batch_size: found.start() + batch_size] + '...;\n'
        
    return context


def cluster_type_identify(df, article, edge_threshold = 2, inner_threshold = 3):
    df_art = df[df['article_id'] == article]
    
    for i in df_art.index:
        if i == df_art.index[0]:
            try:
                df.loc[i, 'cluster_type'] = 'Start' if df_art.loc[i, 'near_links_count'] >= edge_threshold \
                                                    and df_art.loc[i + 1, 'near_links_count'] >= inner_threshold else 'Outer'
                
            except:
                df.loc[i, 'cluster_type'] = 'Outer'
                
        elif i == df_art.index[-1]:
            try:
                df.loc[i, 'cluster_type'] = 'End' if df.loc[i - 1, 'cluster_type'] in ['Start', 'Inner'] else 'Outer'
                
            except:
                df.loc[i, 'cluster_type'] = 'Outer'
                
        else:
           
            df.loc[i, 'cluster_type'] = (
                            'Start' if (df.loc[i, 'near_links_count'] >= edge_threshold  
                                       and df.loc[i + 1, 'near_links_count'] >= inner_threshold 
                                       and df.loc[i - 1, 'cluster_type'] not in ['Start', 'Inner'])
                
                            else 'Inner' if (df.loc[i, 'near_links_count'] >= inner_threshold 
                                         and df.loc[i + 1, 'near_links_count'] >= edge_threshold  
                                         and df.loc[i - 1, 'cluster_type'] not in ['End', 'Outer'])
                
                            else 'End' if (df.loc[i, 'near_links_count'] >= edge_threshold  
                                       and df.loc[i - 1, 'cluster_type'] not in ['End', 'Outer'])
                
                            else 'Outer'
            )

    return df


def table_expand(df):
    df_start_end = df_ids[(df_ids['cluster_type'] == 'Start') | (df_ids['cluster_type'] == 'End')]
    
    for i in range(0, len(df_start_end), 2):
        start = df_start_end.iloc[i].name
        end = df_start_end.iloc[i + 1].name + 1
        
        table_number = [df.loc[j, 'table'] for j in range(start, end)]
        tables = [table.lower().replace(' ', '') for table in table_number if type(table) is str]
        counter = Counter(tables)
        
        if len(counter) > 0:
            main_table = max(counter, key = counter.get)
            indexes = [ind for ind in range(start, end)]
            df.loc[indexes, 'table'] = main_table

    return df


df_ids['near_links_count'] = df_ids.apply(lambda row: nearest_links_count(row, df_ids), axis = 1)

for art in tqdm(df_ids['article_id'].unique()):
    df_ids = cluster_type_identify(df_ids, art)

df_ids['table'] = df_ids.apply(lambda row: identify_table(row, re_table_mark), axis = 1)
df_ids = table_expand(df_ids)

df_ids['context'] = df_ids.apply(lambda row: find_table_context(row) if type(row['table']) == str else row['context'], axis = 1)
df_ids = df_ids.drop_duplicates(subset = ['article_id', 'dataset_id', 'context'])
df_ids['context'] = df_ids.apply(lambda row: row['context'] if type(row['table']) != str else row['context'] + f'{row["dataset_id"]} inside the Table {row["table"]}', axis = 1)


df_dois['author'] = df_dois['article_id'].map(texts_authors)


def delete_markers(text, mark_pattern):
    return re.sub(mark_pattern, '', text)


def make_id_verifying_prompt(text, citation):
    cleaned_text = re.sub(r'\s*\-\s+', '', text)
    prompt = f"""
You are a verification engine that checks whether a citation belongs to a specific databases.

### Databases Description:
1) GenBank - an international database of nucleotide sequences with annotations. It includes genes, genomes, RNAs, and other nucleotide objects, linking them to protein sequences and scientific publications.
2) PDB (Protein Data Bank) – the global archive of three-dimensional structural data of biological macromolecules such as proteins, nucleic acids, and complexes. Maintained by the Worldwide Protein Data Bank consortium, it provides freely accessible experimentally determined structures to support research in biology, medicine, and biotechnology.

### Rules:
- Output **only one** line in this strict format:
  Answer: **[Yes]** - OR - Answer: **[No]**
- Output **[Yes]** only in cases when explicitly mentioned that the citation is from one of databases above.

### Task: determine if the citation cites on a dataset from one of mentioned above or similar databases.
Text: {cleaned_text}
Citation: {citation}
Answer: ["""
    
    return prompt


dang_ids = df_ids[df_ids['pattern'].isin([pat[1] for pat in id_loc_patterns])]['dataset_id'].unique()


df_ids = df_ids.drop(columns = ['start', 'near_links_count', 'cluster_type', 'table', 'pattern']).groupby(by = ['article_id', 'dataset_id']) \
               .agg(list).reset_index()
df_ids['context'] = df_ids['context'].apply(lambda cont: ';\n'.join(cont))
df_ids['context'] = df_ids['context'].apply(lambda cont: delete_markers(cont, re_mark))
    
df_dois['context'] = df_dois['context'].apply(lambda cont: delete_markers(cont, re_mark))


df_dang_ids_ind = df_ids[df_ids['dataset_id'].isin(dang_ids)].index


model_path_14b = "/kaggle/input/qwen2.5/transformers/14b-instruct-awq/1"

llm_14b = vllm.LLM(
    model_path_14b,
    quantization = 'awq',
    dtype = "half",
    distributed_executor_backend="mp",
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.9,
    max_model_len=5000,
    disable_custom_all_reduce=True,
    enable_prefix_caching = True,
    enforce_eager = True,
    trust_remote_code = True
)

tokenizer_14b = llm_14b.get_tokenizer()


id_veryfying_prompts = []
for i in tqdm(df_dang_ids_ind):
    prompt = make_id_verifying_prompt(df_ids.loc[i, 'context'], df_ids.loc[i, 'dataset_id'])
    id_veryfying_prompts.append(prompt)

allowed_words_ver = ['Yes', 'No']
allowed_ids_ver = list(map(lambda word: tokenizer_14b.encode(word)[0], allowed_words_ver))


outputs_ver = llm_14b.generate(
    id_veryfying_prompts,
    vllm.SamplingParams(
        n = 1, 
        temperature = 0,
        seed = 42,
        skip_special_tokens = True,
        max_tokens = 1,
        allowed_token_ids = allowed_ids_ver
    ),
    use_tqdm = True
)


results_ver = []
for output in outputs_ver:
    answer = output.outputs[0].text
    results_ver.append(answer)

df_ids.loc[df_dang_ids_ind, 'type'] = results_ver
df_ids['type'] = df_ids['type'].apply(lambda t: 'Yes' if t not in ['Yes', 'No'] else t)
df_ids = df_ids[df_ids['type'] != 'No']


def make_id_classification_prompt(text, citation):
    cleaned_text = re.sub(r'\s*\-\s+', '', text)
    prompt = f"""
You are a classification engine of dataset citations. 

Your only task is to classify a citation from a scientific paper into one of the categories:
- **[Primary]** - raw or processed data generated as part of the paper, specifically for the study.
- **[Secondary]** - raw or processed data derived or reused from existing records or published data.

### Rules:
- Classify the citation as **[Primary]** only in cases when authors of the study created the dataset or when authors submitted or deposited the dataset to any database.
- Output **only one** line in this strict format:
  Category: [Primary] — OR — Category: [Secondary]

### Task: classify citation from the following text
Text: {cleaned_text}
Citation: {citation}
Category: ["""

    return prompt


id_prompts = []
for j in tqdm(df_ids.index):
    prompt = make_id_classification_prompt(df_ids.loc[j, 'context'], df_ids.loc[j, 'dataset_id'])
    id_prompts.append(prompt)

allowed_words_id = ['Primary', 'Secondary']
allowed_ids_id = list(map(lambda word: tokenizer_14b.encode(word)[0], allowed_words_id))


outputs_id = llm_14b.generate(
    id_prompts,
    vllm.SamplingParams(
        n = 1, 
        temperature = 0,
        seed = 42,
        skip_special_tokens = True,
        max_tokens = 1,
        allowed_token_ids = allowed_ids_id
    ),
    use_tqdm = True
)


results_id = []
for output in outputs_id:
    answer = output.outputs[0].text
    results_id.append(answer)

df_ids['type'] = results_id


cleanup_dist_env_and_memory()
del llm_14b.llm_engine.model_executor
del llm_14b

for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
gc.collect()
ray.shutdown()


def make_data_classification_prompt(text, citation):
    cleaned_text = re.sub(r'\s*\-\s+', '', text)
    prompt = f"""
You are a classification engine of dataset citations, that makes classification using only text and rules. 

Your only task is to classify a citation from a scientific paper into one of the categories:
- **[Dataset]** - direct link on dataset that was used in a scientific research.
- **[Article]** - link on article or another scientific paper.

### Rules:
- If citation cites on software package or library, classify the citation as **[Article]**.
- Classify a citation as **[Article]** if it refers to documents such as manuals, reports, guidelines, other procedures, or scientific papers that discuss, analyze, or describe datasets but do not directly link to the dataset itself.
- Classify a citation as **[Dataset]** only if it directly links to or explicitly mentions a specific dataset (e.g., raw data files, databases, or repositories containing data).
- Even if a citation references datasets indirectly or provides links to other resources, classify it as **[Article]** unless the primary focus is on the dataset itself.
- Ignore the word "Data" as an indicator of a dataset. A link should only be classified as a dataset if it is clearly dedicated to a dataset.
- Output **only one** line in this strict format:
  Category: [Dataset] - OR - Category: [Article]

### Task: classify citation from the following text
Text: {cleaned_text}
Citation: {citation}
Category: ["""

    return prompt


model_path_32b = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"

llm_32b = vllm.LLM(
    model_path_32b,
    quantization = 'awq',
    tensor_parallel_size = torch.cuda.device_count(),
    gpu_memory_utilization = 0.9,
    trust_remote_code = True,
    distributed_executor_backend="mp",
    dtype = "half",
    enforce_eager = True,
    max_model_len = 4096,
    disable_log_stats = True,
    enable_prefix_caching = True
)
tokenizer_32b = llm_32b.get_tokenizer()


data_prompts = []

for i in tqdm(df_dois.index):
    prompt = make_data_classification_prompt(df_dois.loc[i, 'context'], df_dois.loc[i, 'dataset_id'])
    data_prompts.append(prompt)

allowed_words_data = ['Dataset', 'Article']
allowed_ids_data = list(map(lambda word: tokenizer_32b.encode(word)[0], allowed_words_data))


outputs_data = llm_32b.generate(
    data_prompts,
    vllm.SamplingParams(
        n = 1, 
        temperature = 0,
        seed = 42,
        skip_special_tokens = True,
        max_tokens = 1,
        allowed_token_ids = allowed_ids_data
    ),
    use_tqdm = True
)


results_data = []
for output in outputs_data:
    answer = output.outputs[0].text
    results_data.append(answer)

df_dois['type'] = results_data

df_dois = df_dois[df_dois['type'] == 'Dataset']


def make_doi_classification_prompt(text, citation, authors):
    cleaned_text = re.sub(r'\s*\-\s+', '', text)
    prompt = f"""
You are a classification engine of dataset citations. 

Your only task is to classify a citation from a scientific paper into one of the categories:
- **[Primary]** - raw or processed data generated as part of the paper, specifically for the study.
- **[Secondary]** - raw or processed data derived or reused from existing records or published data.

### Rules:
- Output **only one** line in this strict format:
  Category: [Primary] — OR — Category: [Secondary]
- If citation is related at least one of the authors of the text, classify this citation as **[Primary]**
- If citations related with some authors but none of these authors is not the author of the text, classify the citation as **[Secondary]**
- If authors of the text were not found, use only text for classification
- If the citation is refers to the whole database or refers to the dataset that was created by data collecting organization, ignore the rule about authors and classify the citation as **[Secondary]**.

### Task: classify citation from the following text
Authors of the text: {authors}
Text: {cleaned_text}
Citation: {citation}
Category: ["""

    return prompt


doi_prompts = []

for i in tqdm(df_dois.index):
    prompt = make_doi_classification_prompt(df_dois.loc[i, 'context'], df_dois.loc[i, 'dataset_id'], df_dois.loc[i, 'author'])
    doi_prompts.append(prompt)

allowed_words_doi = ['Primary', 'Secondary']
allowed_ids_doi = list(map(lambda word: tokenizer_32b.encode(word)[0], allowed_words_doi))


outputs_doi = llm_32b.generate(
    doi_prompts,
    vllm.SamplingParams(
        n = 1, 
        temperature = 0,
        seed = 42,
        skip_special_tokens = True,
        max_tokens = 1,
        allowed_token_ids = allowed_ids_doi
    ),
    use_tqdm = True
)


results_doi = []
for output in outputs_doi:
    answer = output.outputs[0].text
    results_doi.append(answer)

df_dois['type'] = results_doi


df_results = pd.concat([df_dois, df_ids], ignore_index = True)

df_submission = df_results[df_results['type'] != 'Article'].drop(columns = ['context'])[['article_id', 'dataset_id', 'type']]
df_submission = df_submission.sort_values(by = ['article_id', 'dataset_id']).reset_index(drop = True).reset_index(names = 'row_id')

df_submission.to_csv('submission.csv', index = False)


def get_metrics(declass_old, not_penaltied, not_found, right, incorrect, doi_data_declass, comment):
    tp = right
    fn = not_found if doi_data_declass is None else not_found + doi_data_declass
    old_fp = incorrect + declass_old
    new_fp = old_fp - not_penaltied

    recall = tp / (tp + fn)
    old_precision = tp / (tp + old_fp)
    new_precision = tp / (tp + new_fp)

    old_f1 = 2 * recall * old_precision / (recall + old_precision)
    new_f1 = 2 * recall * new_precision / (recall + new_precision)

    print(f'\n=======================================================')
    print(f'Redundant {comment}s (old): {declass_old}')
    print(f'Redundant {comment}s (new): {declass_old - not_penaltied}')
    print(f'Not Found {comment}s: {not_found}')
    print(f'Right Classified {comment}s: {right}')
    print(f'Incorrect Classified {comment}s: {incorrect}')

    if doi_data_declass is not None:
        print(f'Declassed Datasets: {doi_data_declass}')

    print(f'\nTP_{comment}: {tp} \nFN_{comment}: {fn} \nFP_{comment}_OLD: {old_fp} \nFP_{comment}_NEW: {new_fp}')
    print(f'\nRecall ({comment}): {recall} \nPrecision ({comment}_OLD): {old_precision} \nPrecision ({comment}_NEW): {new_precision}')
    print(f'\nF1 ({comment}_OLD): {old_f1} \nF1 ({comment}_NEW): {new_f1}')


def evaluate(df_subm, df_train):
    df_compared = df_subm.drop(columns = ['row_id']).merge(df_train, on = ['article_id', 'dataset_id'], suffixes = ['_subm', '_train'], how = 'outer')

    missed_papers = df_compared[df_compared['type_train'] == 'Missing']['article_id'].unique()

    not_missed = df_compared[df_compared['type_train'] != 'Missing']
    not_penaltied = not_missed[not_missed['article_id'].isin(missed_papers)]
    declass_old = not_missed[(not_missed['dataset_id'].isin(df_citations['dataset_id'].unique()))&(not_missed['type_train'].isna())]
    not_found = not_missed[(~not_missed['dataset_id'].isin(df_citations['dataset_id'].unique()))&(not_missed['type_subm'].isna())]
    
    classified = not_missed.dropna()
    right = classified[classified['type_subm'] == classified['type_train']]
    incorrect = classified[classified['type_subm'] != classified['type_train']]
    
    declass_old_doi = len(declass_old[declass_old['dataset_id'].str.startswith('http')])
    not_found_doi = len(not_found[not_found['dataset_id'].str.startswith('http')])
    right_doi = len(right[right['dataset_id'].str.startswith('http')])
    incorrect_doi = len(incorrect[incorrect['dataset_id'].str.startswith('http')])
    not_penaltied_doi = len(not_penaltied[not_penaltied['dataset_id'].str.startswith('http')])
    
    all_subm = pd.concat([not_found, right, incorrect], ignore_index = True)
    all_doi_subm = all_subm[all_subm['dataset_id'].str.startswith('http')]
    df_train_doi = df_train[df_train['dataset_id'].str.startswith('http')]
    declassified_datasets = df_train_doi[(df_train_doi['type'] != 'Missing')&(~df_train_doi['dataset_id'].isin(all_doi_subm['dataset_id'].unique()))]
    declass_data = len(declassified_datasets)
    
    declass_old_id = len(declass_old[~declass_old['dataset_id'].str.startswith('http')])
    not_found_id = len(not_found[~not_found['dataset_id'].str.startswith('http')])
    right_id = len(right[~right['dataset_id'].str.startswith('http')])
    incorrect_id = len(incorrect[~incorrect['dataset_id'].str.startswith('http')])
    not_penaltied_id = len(not_penaltied[~not_penaltied['dataset_id'].str.startswith('http')])
    
    total_declass_old = declass_old_doi + declass_old_id
    total_not_found = not_found_doi + not_found_id
    total_right = right_doi + right_id
    total_incorrect = incorrect_doi + incorrect_id
    total_not_penaltied = not_penaltied_doi + not_penaltied_id
    
    total_declass_old = declass_old_doi + declass_old_id
    total_not_found = not_found_doi + not_found_id
    total_right = right_doi + right_id
    total_incorrect = incorrect_doi + incorrect_id
    total_not_penaltied = not_penaltied_doi + not_penaltied_id

    get_metrics(declass_old_doi, not_penaltied_doi, not_found_doi, right_doi, incorrect_doi, declass_data, 'DOI')
    get_metrics(declass_old_id, not_penaltied_id, not_found_id, right_id, incorrect_id, None, 'ID')
    get_metrics(total_declass_old, total_not_penaltied, total_not_found, total_right, total_incorrect, declass_data, 'TOTAL')

    df_citations['author'] = df_citations['article_id'].map(texts_authors)
    df_citations.merge(declass_old, on = ['article_id', 'dataset_id'], how = 'right').to_csv('declass.csv', index = False)
    df_citations.merge(not_found, on = ['article_id', 'dataset_id'], how = 'right').to_csv('not_found.csv', index = False)
    df_citations.merge(right, on = ['article_id', 'dataset_id'], how = 'right').to_csv('right.csv', index = False)
    df_citations.merge(incorrect, on = ['article_id', 'dataset_id'], how = 'right').to_csv('incorrect.csv', index = False)
    df_citations.merge(df_compared, on = ['article_id', 'dataset_id'], how = 'right').to_csv('comparings.csv', index = False)
    df_citations.merge(declassified_datasets, on = ['article_id', 'dataset_id'], how = 'right').to_csv('declass_data.csv', index = False)


if DIR_PATH == '/kaggle/input/make-data-count-finding-data-references/train/PDF':
    train_labels = pd.read_csv('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
    evaluate(df_submission, train_labels)

