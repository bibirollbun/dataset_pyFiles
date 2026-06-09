! uv pip install -q --system --no-index --find-links='/kaggle/input/latest-mdc-whls/whls' 'marker-pdf'
! uv pip uninstall -q --system 'tensorflow'


! mkdir src


%%writefile src/common.py
import os
import polars as pl

from pathlib import Path
from typing import Tuple

DOI_URL = 'https://doi.org/'

def is_submission(): return bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
def is_kaggle_env(): return (len([k for k in os.environ.keys() if 'KAGGLE' in k]) > 0) or is_submission()

def get_prefix_path(prefix: str)->Path:
    return Path(f'/kaggle/{prefix}' if is_kaggle_env() else f'.{prefix}').expanduser().resolve()

def is_doi(name:str)->pl.Expr: return pl.col(name).str.starts_with(DOI_URL)

def doi_link_to_id(name:str)->pl.Expr:
    return pl.when(is_doi(name)).then(pl.col(name).str.split(DOI_URL).list.last()).otherwise(name).alias(name)

def doi_id_to_link(name:str, substring:str, url:str=DOI_URL)->pl.Expr:
    return pl.when(pl.col(name).str.starts_with(substring)).then(url+pl.col(name).str.to_lowercase()).otherwise(name).alias(name)

def score(preds: pl.DataFrame, gt: pl.DataFrame, on: list = ['article_id', 'dataset_id'], verbose:bool=True) -> Tuple[float, float, float]:
    if 'id' in preds.columns and 'dataset_id' not in preds.columns: preds = preds.rename({'id': 'dataset_id'})
    hits = gt.join(preds, on=on)
    tp = hits.height
    fp = preds.height - tp
    fn = gt.height - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if verbose:
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

    return precision, recall, f1


%%writefile src/parse.py
import argparse
import pymupdf
import pathlib
import tqdm

from common import get_prefix_path, is_submission

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', default=f'make-data-count-finding-data-references/{"test" if is_submission() else "train"}/PDF')
    p.add_argument('-o', default='parsed')
    return p.parse_args()

def pdf2text(path: pathlib.Path, out_dir: pathlib.Path) -> None:
    doc = pymupdf.open(str(path))
    out = open(out_dir / f"{path.stem}.txt", "wb")
    for page in doc:
        text = page.get_text().encode("utf8")
        out.write(text)
        out.write(b'\n') # write page delimiter (form feed 0x0C)
    out.close()

def main():
    args = get_args()
    in_dir = get_prefix_path('input') / args.i
    out_dir = get_prefix_path('working') / args.o

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f'{out_dir} already populated, skipping...')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if not in_dir.is_dir(): raise ValueError(f'{in_dir} is not a directory...')
    pdf_files = list(in_dir.glob('*.pdf'))
    if not pdf_files: raise ValueError(f'No PDF files found in {in_dir}')

    for pdf in tqdm.tqdm(pdf_files, desc="Processing PDFs"): pdf2text(pdf, out_dir)
    print('ending parsing...')

if __name__ == '__main__': main()


%%writefile src/getacc.py
import polars as pl
import argparse
import pathlib
from common import score, get_prefix_path, is_submission, is_doi, doi_id_to_link

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', default='parsed')
    p.add_argument('-o', default='extracted_ids.parquet')
    p.add_argument('--gt', default='make-data-count-finding-data-references/train_labels.csv')
    p.add_argument('--ws', default=100, type=int)
    return p.parse_args()

def get_text_df(parsed_dir: pathlib.Path):
    paths = list(parsed_dir.rglob('*.txt'))
    if len(paths)==0: paths = list(parsed_dir.rglob('*.md'))
    records = [{'article_id': p.stem, 'text': p.read_text()} for p in paths]
    return (
        pl.DataFrame(records)
        .with_columns(pl.col("text").str.normalize("NFKC").str.replace_all(r"[^\p{Ascii}]", ""))
        .with_columns(pl.col('text').str.split(r'\n{2,}').list.eval(pl.col("").str.replace_all('\n', ' ')).list.join('\n').alias('text'))
        .with_columns([
            pl.col("text").str.slice(pl.col("text").str.len_chars()//4).str.reverse().alias('rtext'),
            pl.col("text").str.slice(0, pl.col("text").str.len_chars()//4).alias('ltext'),
        ])
        .with_columns(pl.col('rtext').str.find(r'(?i)\b(secnerefer|erutaretil detic|stnemegdelwonkca)\b').alias('ref_idx'))
        .with_columns(pl.when(pl.col('ref_idx').is_null()).then(0).otherwise('ref_idx').alias('ref_idx'))
        .with_columns([
            pl.col('rtext').str.slice(0, pl.col('ref_idx')).str.reverse().alias('refs'),
            (pl.col('ltext') + pl.col('rtext').str.slice(pl.col('ref_idx')).str.reverse()).alias('body')
        ])
        .drop('rtext', 'ltext')
    )


def main():
    print('starting extraction of accession ids')
    args = get_args()
    in_path, out_path = map(lambda x: get_prefix_path('working') / x, (args.i, args.o))
    text_df = get_text_df(in_path)

    df = (
        text_df
        .with_columns([
            pl.col("text").str.extract_all(r'(?i)\b(?:CHEMBL\d+|E-GEOD-\d+|E-PROT-\d+|EMPIAR-\d+|ENSBTAG\d+|ENSOARG\d+|EPI_ISL_\d{5,}|EPI\d{6,7}|HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|KX\d{6}|K0\d{4}|PRJNA\d+|PXD\d+|SAMN\d+|dryad\.[^\s"<>]+|pasta\/[^\s"<>])').alias('id'),
        ])
        .explode('id')
        .with_columns(pl.col('id').alias('match_id'))
        .with_columns(pl.col('id').str.replace_all(r'\s', ''))
        .with_columns(pl.col('id').str.replace(r'[-.,;:!?\/\)\]\(\[]+$', ''))
        .with_columns(doi_id_to_link(name='id', substring='dryad.', url='https://doi.org/10.5061/'))
        .with_columns(doi_id_to_link(name='id', substring='pasta/', url='https://doi.org/10.6073/'))
        .with_columns(doi_id_to_link(name='id', substring='zenodo.', url='https://doi.org/10.5281/'))
        .filter(~pl.col('id').str.to_lowercase().str.contains(pl.col('article_id').str.to_lowercase().str.replace('_', '/')))
        .filter(~pl.col('id').str.contains('figshare', literal=True))
        .filter(pl.when(is_doi('id').and_(pl.col('id').str.split('/').list.last().str.len_chars()<4)).then(pl.lit(False)).otherwise(pl.lit(True)))
        .filter(~pl.col('id').is_in(['https://doi.org/10.5061/dryad', 'https://doi.org/10.6073/pasta', 'https://doi.org/10.5281/zenodo']))
        .filter(pl.col('id').str.count_matches(r'\(') == pl.col('id').str.count_matches(r'\)'))
        .filter(pl.col('id').str.count_matches(r'\[') == pl.col('id').str.count_matches(r'\]'))
        .with_columns(
            pl.col('text').str.slice(pl.col('text').str.find(pl.col('match_id'), literal=True)-args.ws-pl.col('match_id').str.len_chars(), 2*(args.ws+pl.col('match_id').str.len_chars())).alias('window')
        )
        .unique(['article_id', 'id'])
        .rename({'id': 'dataset_id'})
    )
    df.select('article_id', 'dataset_id', 'window').write_parquet(out_path)
    print(f'id extraction written to {out_path}')

    df = df.select('article_id', 'dataset_id').with_columns(pl.lit('Secondary').alias('type'))
    df = df.with_columns(
        pl.when(is_doi('dataset_id').or_(pl.col('dataset_id').str.starts_with('SAMN'))).then(pl.lit('Primary')).otherwise('type').alias('type')
    )

    df.with_row_index(name='row_id').write_csv(get_prefix_path('working')/'submission.csv')

    if not is_submission():
        gt_path = get_prefix_path('input') / args.gt
        gt = pl.read_csv(gt_path).filter(pl.col('type')!='Missing').join(text_df, on='article_id')
        print('### DOI ###')
        score(df.filter(is_doi('dataset_id')), gt.filter(is_doi('dataset_id')))
        print('### ACC ###')
        score(df.filter(~is_doi('dataset_id')), gt.filter(~is_doi('dataset_id')))
        print('### ALL ###')
        score(df, gt)
        print('### TYPE ###')
        score(df, gt, on=['article_id', 'dataset_id', 'type'])

if __name__=='__main__':
    main()


%%writefile marker.sh
INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

mkdir -p "$OUTPUT_FOLDER"

for (( i=0; i<$NUM_DEVICES; i++ )); do
    DEVICE_NUM=$i
    export DEVICE_NUM
    export NUM_DEVICES
    echo "Running marker on GPU $DEVICE_NUM"
    cmd="CUDA_VISIBLE_DEVICES=$DEVICE_NUM marker $INPUT_FOLDER --output_dir $OUTPUT_FOLDER --num_chunks $NUM_DEVICES --chunk_idx $DEVICE_NUM --workers 2 --disable_tqdm"
    eval $cmd &

    sleep 5
done

wait


import os
IS_SUBMISSION = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
USE_PYMUPDF = False

if USE_PYMUPDF:
    ! python src/parse.py
elif IS_SUBMISSION and not USE_PYMUPDF:
    ! mkdir -p /usr/local/lib/python3.11/dist-packages/static/fonts
    ! cp /kaggle/input/marker-pdf-models/GoNotoCurrent-Regular.ttf /usr/local/lib/python3.11/dist-packages/static/fonts
    ! cp -r /kaggle/input/marker-pdf-models/datalab /root/.cache
    ! NUM_DEVICES=2 bash ./marker.sh /kaggle/input/make-data-count-finding-data-references/test/PDF /working/parsed
else:
    ! mkdir ./parsed
    ! cp -r /kaggle/input/parse-pdfs-marker-is-all-you-need/train_parsed/* ./parsed


! python src/getacc.py


! rm -rf parsed
! rm -rf src
! rm -rf extracted_ids.parquet

