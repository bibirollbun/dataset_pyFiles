# Install packages
# !uv pip install -q --system pymupdf


# å¯¼å…¥æ“�ä½œç³»ç»Ÿç›¸å…³æ¨¡å�—ï¼Œç”¨äº�æ–‡ä»¶è·¯å¾„æ“�ä½œç­‰
import os

# å¯¼å…¥æ­£åˆ™è¡¨è¾¾å¼�æ¨¡å�—ï¼Œç”¨äº�æ¨¡å¼�åŒ¹é…�å’Œå­—ç¬¦ä¸²å¤„ç�†
import re

# å¯¼å…¥pathlibæ¨¡å�—ï¼Œç”¨äº�å¤„ç�†æ–‡ä»¶ç³»ç»Ÿè·¯å¾„ï¼Œæ��ä¾›é�¢å�‘å¯¹è±¡çš„è·¯å¾„æ“�ä½œæ–¹å¼�
import pathlib

# å¯¼å…¥polarsåº“ï¼Œä¸€ä¸ªé«˜æ€§èƒ½çš„DataFrameæ•°æ�®å¤„ç�†åº“ï¼Œç”¨äº�æ•°æ�®æ“�ä½œ
import polars as pl

# å¯¼å…¥lxmlæ¨¡å�—ï¼Œç”¨äº�è§£æ��XMLå’ŒHTMLæ–‡æ¡£ï¼Œè¿™é‡Œä¸»è¦�ç”¨äº�å¤„ç�†PDFä¸­çš„XMLç»“æ�„
from lxml import etree

# å¯¼å…¥pymupdfåº“ï¼Œç”¨äº�ä»�PDFæ–‡æ¡£ä¸­æ��å�–æ–‡æœ¬ã€�å…ƒæ•°æ�®ç­‰ä¿¡æ�¯
import pymupdf

# ä»�typingæ¨¡å�—å¯¼å…¥Tupleç±»å�‹æ��ç¤ºï¼Œç”¨äº�æŒ‡å®šå‡½æ•°è¿”å›�å€¼çš„ç±»å�‹ç»“æ�„
from typing import Tuple

# å®šä¹‰DOI_URLå¸¸é‡�ï¼Œç”¨äº�ç”Ÿæˆ�DOIï¼ˆæ•°å­—å¯¹è±¡æ ‡è¯†ç¬¦ï¼‰çš„å®Œæ•´URL
DOI_URL = 'https://doi.org/'

# è®¾ç½®polarsåº“çš„è¯¦ç»†è¾“å‡ºæ¨¡å¼�ä¸ºTrueï¼Œç”¨äº�åœ¨è°ƒè¯•æ—¶æ˜¾ç¤ºæ›´å¤šå†…éƒ¨ä¿¡æ�¯
pl.Config.set_verbose(True)


# Utilities and Helpers
def is_submission():
    # é€šè¿‡ç�¯å¢ƒå�˜é‡�åˆ¤æ–­æ˜¯å�¦ä¸ºKaggleç«�èµ›æ��äº¤ç�¯å¢ƒ
    # KAGGLE_IS_COMPETITION_RERUNæ˜¯Kaggleå¹³å�°ç‰¹æœ‰çš„ç�¯å¢ƒå�˜é‡�
    return bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))


def is_kaggle_env():
    # ç»¼å�ˆåˆ¤æ–­å½“å‰�æ˜¯å�¦åœ¨Kaggleç�¯å¢ƒï¼š
    # æ–¹æ¡ˆ1ï¼šæ£€æŸ¥ç�¯å¢ƒå�˜é‡�å��æ˜¯å�¦åŒ…å�«"KAGGLE"ï¼ˆè‡³å°‘å­˜åœ¨ä¸€ä¸ªï¼‰
    # æ–¹æ¡ˆ2ï¼šé€šè¿‡is_submission()äºŒæ¬¡éªŒè¯�
    return (len([k for k in os.environ.keys() if 'KAGGLE' in k]) > 0) or is_submission()


def get_prefix_path(prefix: str) -> pathlib.Path:
    # æ ¹æ�®è¿�è¡Œç�¯å¢ƒåŠ¨æ€�ç”Ÿæˆ�è·¯å¾„ï¼š
    # - Kaggleç�¯å¢ƒè¿”å›�ç»�å¯¹è·¯å¾„ï¼š/kaggle/{prefix}
    # - æœ¬åœ°ç�¯å¢ƒè¿”å›�ç›¸å¯¹è·¯å¾„ï¼š./{prefix}ï¼ˆé€šè¿‡expanduser()è§£æ��~ç¬¦å�·ï¼‰
    # æœ€ç»ˆç”¨resolve()è½¬æ�¢ä¸ºæ ‡å‡†ç»�å¯¹è·¯å¾„
    return pathlib.Path(f'/kaggle/{prefix}' if is_kaggle_env() else f'.{prefix}').expanduser().resolve()


def is_doi(name: str) -> pl.Expr:
    # ä½¿ç”¨Polarsè¡¨è¾¾å¼�æ£€æŸ¥æŸ�åˆ—æ˜¯å�¦ä»¥DOIé“¾æ�¥å‰�ç¼€å¼€å¤´
    # DOI_URLåº”ä¸ºé¢„å®šä¹‰çš„å¸¸é‡�ï¼ˆå¦‚"https://doi.org/"ï¼‰
    return pl.col(name).str.starts_with(DOI_URL)


def doi_link_to_id(name: str) -> pl.Expr:
    # è½¬æ�¢DOIé“¾æ�¥ä¸ºçº¯IDï¼š
    # - è‹¥ä»¥DOI_URLå¼€å¤´ï¼šæˆªå�–URLå��çš„IDéƒ¨åˆ†ï¼ˆå¦‚"10.XXX/YYY"ï¼‰
    # - å�¦åˆ™ä¿�ç•™å�Ÿå§‹å€¼
    # ç»“æ�œåˆ—ä¿�æŒ�å�Ÿå§‹åˆ—å��
    return pl.when(is_doi(name)).then(pl.col(name).str.split(DOI_URL).list.last()).otherwise(name).alias(name)


def doi_id_to_link(name: str, substring: str, url: str = DOI_URL) -> pl.Expr:
    # å°†çº¯IDè½¬æ�¢ä¸ºå®Œæ•´DOIé“¾æ�¥ï¼š
    # - å½“å€¼ä»¥æŒ‡å®šå­�ä¸²å¼€å¤´æ—¶ï¼šæ‹¼æ�¥URLå‰�ç¼€å¹¶è½¬ä¸ºå°�å†™
    # - å�¦åˆ™ä¿�ç•™å�Ÿå§‹å€¼
    # ç»“æ�œåˆ—ä¿�æŒ�å�Ÿå§‹åˆ—å��
    return pl.when(pl.col(name).str.starts_with(substring)).then(url + pl.col(name).str.to_lowercase()).otherwise(name).alias(name)


def score(preds: pl.DataFrame, gt: pl.DataFrame, on: list = ['article_id', 'dataset_id'], verbose: bool = True) -> Tuple[float, float, float]:
    # ç»Ÿä¸€åˆ—å��ï¼šè‹¥é¢„æµ‹æ•°æ�®æœ‰'id'åˆ—ä½†æ— 'dataset_id'ï¼Œåˆ™é‡�å‘½å��
    if 'id' in preds.columns and 'dataset_id' not in preds.columns:
        preds = preds.rename({'id': 'dataset_id'})
    
    # é€šè¿‡æŒ‡å®šåˆ—è¿�æ�¥çœŸå®�å€¼å’Œé¢„æµ‹å€¼ï¼Œè�·å�–åŒ¹é…�é¡¹ï¼ˆTrue Positivesï¼‰
    hits = gt.join(preds, on=on)
    tp = hits.height  # çœŸé˜³æ€§æ•°é‡� = åŒ¹é…�è¡Œæ•°
    
    # è®¡ç®—å…³é”®æŒ‡æ ‡ï¼š
    fp = preds.height - tp   # å�‡é˜³æ€§ = æ€»é¢„æµ‹æ•° - åŒ¹é…�æ•°
    fn = gt.height - tp      # å�‡é˜´æ€§ = æ€»çœŸå®�æ•° - åŒ¹é…�æ•°
    
    # è®¡ç®—ç²¾ç¡®åº¦ï¼ˆPrecisionï¼‰
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # è®¡ç®—å�¬å›�ç�‡ï¼ˆRecallï¼‰
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # è®¡ç®—F1åˆ†æ•°ï¼ˆPrecisionå’ŒRecallçš„è°ƒå’Œå¹³å�‡ï¼‰
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # å�¯é€‰è¯¦ç»†è¾“å‡º
    if verbose:
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
    
    return precision, recall, f1


def xml_kind(path: pathlib.Path) -> str: # å®šä¹‰ä¸€ä¸ªå‡½æ•°xml_kindï¼Œå�‚æ•°pathä¸ºæ–‡ä»¶è·¯å¾„å¯¹è±¡ï¼Œè¿”å›�å­—ç¬¦ä¸²è¡¨ç¤ºXMLç±»å�‹
    head = path.open('rb').read(2048).decode('utf8', 'ignore') # ä»¥äºŒè¿›åˆ¶æ¨¡å¼�æ‰“å¼€æ–‡ä»¶ï¼Œè¯»å�–å‰�2048å­—èŠ‚ï¼Œè§£ç �ä¸ºUTF-8å­—ç¬¦ä¸²ï¼ˆå¿½ç•¥ç¼–ç �é”™è¯¯ï¼‰ï¼Œå­˜å‚¨åœ¨headå�˜é‡�
    if 'www.tei-c.org/ns' in head: # å¦‚æ�œheadå­—ç¬¦ä¸²ä¸­åŒ…å�«'www.tei-c.org/ns'ï¼ˆTEI XMLç±»å�‹çš„æ ‡è¯†ï¼‰
        return 'tei' # è¿”å›�å­—ç¬¦ä¸²'tei'è¡¨ç¤ºTEI XMLç±»å�‹
    if re.search(r'(NLM|TaxonX)//DTD', head): # ä½¿ç”¨æ­£åˆ™è¡¨è¾¾å¼�æ�œç´¢headä¸­æ˜¯å�¦åŒ…å�«'NLM//DTD'æˆ–'TaxonX//DTD'ï¼ˆJATS XMLç±»å�‹çš„æ ‡è¯†ï¼‰
        return 'jats' # è¿”å›�å­—ç¬¦ä¸²'jats'è¡¨ç¤ºJATS XMLç±»å�‹
    if 'www.wiley.com/namespaces' in head: # å¦‚æ�œheadå­—ç¬¦ä¸²ä¸­åŒ…å�«'www.wiley.com/namespaces'ï¼ˆWiley XMLç±»å�‹çš„æ ‡è¯†ï¼‰
        return 'wiley' # è¿”å›�å­—ç¬¦ä¸²'wiley'è¡¨ç¤ºWiley XMLç±»å�‹
    if 'BioC.dtd' in head: # å¦‚æ�œheadå­—ç¬¦ä¸²ä¸­åŒ…å�«'BioC.dtd'ï¼ˆBioC XMLç±»å�‹çš„æ ‡è¯†ï¼‰
        return 'bioc' # è¿”å›�å­—ç¬¦ä¸²'bioc'è¡¨ç¤ºBioC XMLç±»å�‹
    return 'unknown' # å¦‚æ�œä»¥ä¸Šéƒ½ä¸�åŒ¹é…�ï¼Œè¿”å›�'unknown'è¡¨ç¤ºæœªçŸ¥XMLç±»å�‹


def xml2text(path: pathlib.Path) -> str: # å®šä¹‰ä¸€ä¸ªå‡½æ•°xml2textï¼Œå�‚æ•°pathä¸ºæ–‡ä»¶è·¯å¾„å¯¹è±¡ï¼Œè¿”å›�å­—ç¬¦ä¸²è¡¨ç¤ºæ��å�–çš„æ–‡æœ¬
    kind = xml_kind(path) # è°ƒç”¨xml_kindå‡½æ•°ç¡®å®šXMLæ–‡ä»¶çš„ç±»å�‹ï¼Œå­˜å‚¨åœ¨kindå�˜é‡�
    root = etree.parse(str(path)).getroot() # ä½¿ç”¨etreeè§£æ��XMLæ–‡ä»¶è·¯å¾„ï¼ˆè½¬ä¸ºå­—ç¬¦ä¸²ï¼‰ï¼Œè�·å�–æ ¹å…ƒç´ å¯¹è±¡
    if kind in ('tei', 'bioc', 'unknown'): # å¦‚æ�œXMLç±»å�‹æ˜¯teiã€�biocæˆ–unknown
        txt = ' '.join(root.itertext()) # ä½¿ç”¨itertext()é��å�†æ ¹å…ƒç´ å�Šå…¶æ‰€æœ‰å­�å…ƒç´ æ–‡æœ¬ï¼Œè¿�æ�¥æˆ�ä¸€ä¸ªå­—ç¬¦ä¸²ï¼ˆç©ºæ ¼åˆ†éš”ï¼‰
    elif kind == 'jats': # å¦‚æ�œXMLç±»å�‹æ˜¯jats
        elems = root.xpath('//body//sec|//ref-list') # ä½¿ç”¨XPathæŸ¥è¯¢æ‰€æœ‰bodyä¸‹çš„secå…ƒç´ å’Œref-listå…ƒç´ ï¼ˆJATSæ ¼å¼�çš„ç»“æ�„ï¼‰
        txt = ' '.join(' '.join(e.itertext()) for e in elems) # å¯¹æ¯�ä¸ªåŒ¹é…�å…ƒç´ æ��å�–æ–‡æœ¬å¹¶è¿�æ�¥ï¼Œæœ€ç»ˆæ‰€æœ‰å…ƒç´ æ–‡æœ¬è¿�æ�¥ä¸ºä¸€ä¸ªå­—ç¬¦ä¸²ï¼ˆç©ºæ ¼åˆ†éš”ï¼‰
    elif kind == 'wiley': # å¦‚æ�œXMLç±»å�‹æ˜¯wiley
        elems = root.xpath('//*[local-name()="body"]|//*[local-name()="refList"]') # ä½¿ç”¨XPathæŸ¥è¯¢æ‰€æœ‰æœ¬åœ°å��ï¼ˆå¿½ç•¥å‘½å��ç©ºé—´ï¼‰ä¸ºbodyæˆ–refListçš„å…ƒç´ ï¼ˆWileyæ ¼å¼�çš„ç»“æ�„ï¼‰
        txt = ' '.join(' '.join(e.itertext()) for e in elems) # å¯¹æ¯�ä¸ªåŒ¹é…�å…ƒç´ æ��å�–æ–‡æœ¬å¹¶è¿�æ�¥ï¼Œæœ€ç»ˆæ‰€æœ‰å…ƒç´ æ–‡æœ¬è¿�æ�¥ä¸ºä¸€ä¸ªå­—ç¬¦ä¸²ï¼ˆç©ºæ ¼åˆ†éš”ï¼‰
    else: # å…¶ä»–æœªå¤„ç�†ç±»å�‹ï¼ˆæ­¤åˆ†æ”¯ç�†è®ºä¸Šä¸�ä¼šè§¦å�‘ï¼Œå› ä¸ºkindå·²è¦†ç›–æ‰€æœ‰æƒ…å†µï¼‰
        txt = ' '.join(root.itertext()) # é»˜è®¤æ��å�–æ‰€æœ‰æ–‡æœ¬ï¼ˆç©ºæ ¼åˆ†éš”ï¼‰
    txt = re.sub(r'10\.\d{4,9}/\s+', '10.', txt) # ä½¿ç”¨æ­£åˆ™è¡¨è¾¾å¼�æ›¿æ�¢æ–‡æœ¬ï¼šåŒ¹é…�ç±»ä¼¼'10.123456789/ 'çš„å­—ç¬¦ä¸²ï¼ˆä¾‹å¦‚DOIæ ¼å¼�é”™è¯¯ï¼‰ï¼Œæ›¿æ�¢ä¸º'10.'ä»¥æ ‡å‡†åŒ–
    return txt # è¿”å›�å¤„ç�†å��çš„æ–‡æœ¬å­—ç¬¦ä¸²


def pdf2text(path: pathlib.Path, out_dir: pathlib.Path) -> None: # å®šä¹‰ä¸€ä¸ªå‡½æ•°pdf2textï¼Œå�‚æ•°pathä¸ºPDFæ–‡ä»¶è·¯å¾„ï¼Œout_dirä¸ºè¾“å‡ºç›®å½•è·¯å¾„ï¼Œæ— è¿”å›�å€¼ï¼ˆNoneï¼‰
    doc = pymupdf.open(str(path)) # ä½¿ç”¨pymupdfæ‰“å¼€PDFæ–‡ä»¶è·¯å¾„ï¼ˆè½¬ä¸ºå­—ç¬¦ä¸²ï¼‰ï¼Œè�·å�–æ–‡æ¡£å¯¹è±¡
    out = out_dir / f"{path.stem}.txt" # æ�„å»ºè¾“å‡ºæ–‡ä»¶è·¯å¾„ï¼šout_dirç›®å½•ä¸‹ï¼Œæ–‡ä»¶å��åŸºäº�è¾“å…¥æ–‡ä»¶çš„stemï¼ˆæ— æ‰©å±•å��ï¼‰åŠ ä¸Š.txtå��ç¼€
    with open(out, "wb") as f: # æ‰“å¼€è¾“å‡ºæ–‡ä»¶ä»¥äºŒè¿›åˆ¶å†™å…¥æ¨¡å¼�ï¼ˆ'wb'ï¼‰ï¼Œfä¸ºæ–‡ä»¶å¯¹è±¡
        for page in doc: # é��å�†PDFæ–‡æ¡£çš„æ¯�ä¸€é¡µ
            f.write(page.get_text().encode("utf8")) # è�·å�–å½“å‰�é¡µçš„æ–‡æœ¬å†…å®¹ï¼Œç¼–ç �ä¸ºUTF-8å­—èŠ‚ï¼Œå†™å…¥è¾“å‡ºæ–‡ä»¶
            f.write(b"\n") # å†™å…¥æ�¢è¡Œç¬¦ï¼ˆäºŒè¿›åˆ¶æ¨¡å¼�ï¼‰ä»¥åˆ†éš”ä¸�å�Œé¡µé�¢


# å®šä¹‰å‡½æ•°ï¼šå¤„ç�†æŒ‡å®šç›®å½•ä¸‹çš„æ‰€æœ‰PDFå’ŒXMLæ–‡ä»¶ï¼Œå°†å…¶è½¬æ�¢ä¸ºTXTæ ¼å¼�
def parse_all_pdfs_xmls(pdf_dir, xml_dir, parsed_dir):
    # è�·å�–pdf_dirç›®å½•ä¸‹æ‰€æœ‰PDFæ–‡ä»¶åˆ—è¡¨
    pdf_files = list(pdf_dir.glob('*.pdf'))
    
    # å¼‚å¸¸æ£€æŸ¥ï¼šè‹¥æ— PDFæ–‡ä»¶ä¸”XMLç›®å½•ä¸�å­˜åœ¨ï¼ŒæŠ›å‡ºé”™è¯¯
    if not pdf_files and not xml_dir.exists():
        raise ValueError("No PDF or XML files found.")
    
    # åˆ›å»ºè¾“å‡ºç›®å½•ï¼ˆè‹¥ä¸�å­˜åœ¨åˆ™è‡ªåŠ¨åˆ›å»ºçˆ¶ç›®å½•ï¼‰
    parsed_dir.mkdir(parents=True, exist_ok=True)
    
    # â€”â€” PDFå¤„ç�†éƒ¨åˆ† â€”â€”
    # é��å�†æ‰€æœ‰PDFæ–‡ä»¶ï¼ˆæ˜¾ç¤ºè¿›åº¦æ�¡ï¼‰
    for pdf in tqdm(pdf_files, desc="PDFâ†’TXT"):
        try:
            # è°ƒç”¨pdf2textå‡½æ•°è½¬æ�¢å½“å‰�PDFï¼ˆå‡½æ•°å®�ç�°åœ¨å¤–éƒ¨ï¼‰
            pdf2text(pdf, parsed_dir)
        except Exception as e:
            # æ�•è�·è½¬æ�¢å¼‚å¸¸å¹¶æ‰“å�°é”™è¯¯æ–‡ä»¶å��+å�Ÿå› 
            print(f"PDF error {pdf.stem}: {e}")
    
    # â€”â€” XMLå¤„ç�†éƒ¨åˆ† â€”â€”
    # æ£€æŸ¥XMLç›®å½•æ˜¯å�¦å­˜åœ¨
    if xml_dir.exists():
        # é��å�†æ‰€æœ‰XMLæ–‡ä»¶ï¼ˆæ˜¾ç¤ºè¿›åº¦æ�¡ï¼‰
        for xml in tqdm(xml_dir.glob('*.xml'), desc="XMLâ†’TXT"):
            try:
                # å°†XMLè½¬æ�¢ä¸ºUTF-8ç¼–ç �çš„å­—èŠ‚æµ�
                txt = xml2text(xml).encode("utf8")
                # æ�„é€ è¾“å‡ºè·¯å¾„ï¼ˆä¸�XMLå�Œå��ä½†å��ç¼€ä¸º.txtï¼‰
                out = parsed_dir / f"{xml.stem}.txt"
                # ä»¥äºŒè¿›åˆ¶è¿½åŠ æ¨¡å¼�å†™å…¥æ–‡ä»¶
                with open(out, "ab") as f:  # 'ab' = append binary
                    f.write(txt)     # å†™å…¥XMLå†…å®¹
                    f.write(b"\n")   # è¿½åŠ æ�¢è¡Œç¬¦ä½œä¸ºåˆ†éš”
            except Exception as e:
                # æ�•è�·è½¬æ�¢å¼‚å¸¸å¹¶æ‰“å�°é”™è¯¯æ–‡ä»¶å��+å�Ÿå› 
                print(f"XML error {xml.stem}: {e}")
    
    # å…¨éƒ¨å¤„ç�†å®Œæˆ�å��æ‰“å�°æ��ç¤º
    print("Done parsing to text.")


# Extraction Helpers
# This cell defines a regex for extracting dataset IDs from text,
# and a helper function to read in all parsed .txt files as a DataFrame.

import matplotlib.pyplot as plt  # ç»˜å›¾åº“ï¼Œæ­¤å¤„æœªä½¿ç”¨ä½†å�¯èƒ½ç”¨äº�å��ç»­å�¯è§†åŒ–
import polars as pl              # é«˜æ€§èƒ½DataFrameåº“ï¼ˆç±»ä¼¼Pandasï¼‰
from pathlib import Path         # é�¢å�‘å¯¹è±¡çš„æ–‡ä»¶è·¯å¾„æ“�ä½œåº“

## Play with these to bump up your scores
REGEX_IDS = (
    r"(?i)\b(?:"
    r"CHEMBL\d+|"
    r"E-GEOD-\d+|E-PROT-\d+|EMPIAR-\d+|"
    r"ENSBTAG\d+|ENSOARG\d+|"
    r"EPI_ISL_\d{5,}|EPI\d{6,7}|"
    r"HPA\d+|CP\d{6}|IPR\d{6}|PF\d{5}|KX\d{6}|K0\d{4}|"
    r"PRJNA\d+|PRJEB\d+|PXD\d+|SAMN\d+|"
    r"GSE\d+|GSM\d+|GPL\d+|"
    r"E-MTAB-\d+|E-MEXP-\d+|"
    r"PDB\s?\w{4}|HMDB\d+|"
    r"dryad\.[^\s\"<>]+|pasta\/[^\s\"<>]+|"
    r"(?:SRR|SRX|SRP|ERR|DRR|DRX|DRP|ERP|ERX)\d+"
    r")"
)

def get_text_df(parsed_dir: Path) -> pl.DataFrame:
    # æ­¥éª¤1ï¼šé€’å½’è�·å�–æ‰€æœ‰.txtæ–‡ä»¶è·¯å¾„
    paths = list(parsed_dir.rglob('*.txt'))
    
    # æ­¥éª¤2ï¼šæ�„å»ºå­—å…¸åˆ—è¡¨ï¼Œé”®ä¸ºæ–‡ç« IDï¼ˆæ–‡ä»¶å��ï¼‰ï¼Œå€¼ä¸ºæ–‡ä»¶å†…å®¹
    records = [{'article_id': p.stem, 'text': p.read_text(encoding='utf8')} for p in paths]
    
    # æ­¥éª¤3ï¼šåˆ›å»ºPolars DataFrameå¹¶é¢„å¤„ç�†æ–‡æœ¬
    return (
        pl.DataFrame(records)
        # æ ‡å‡†åŒ–Unicodeæ ¼å¼�å¹¶ç§»é™¤é��ASCIIå­—ç¬¦
        .with_columns(
            pl.col("text")
            .str.normalize("NFKC")  # Unicodeå…¼å®¹æ€§å½’ä¸€åŒ–
            .str.replace_all(r"[^\p{Ascii}]", "")  # åˆ é™¤é��ASCIIå­—ç¬¦
        )
        # æ­¥éª¤4ï¼šæŒ‰è¿�ç»­æ�¢è¡Œç¬¦åˆ†å‰²æ–‡æœ¬ï¼Œå�ˆå¹¶æ®µè�½å†…æ�¢è¡Œä¸ºç©ºæ ¼
        .with_columns(
            pl.col("text")
            .str.split(r'\n{2,}')  # ä»¥2+ä¸ªæ�¢è¡Œç¬¦åˆ†å‰²æ–‡æ¡£
            .list.eval(pl.col("").str.replace_all('\n', ' '))  # æ®µè�½å†…æ�¢è¡Œè½¬ç©ºæ ¼
            .list.join('\n')  # é‡�æ–°ç”¨å�•æ�¢è¡Œè¿�æ�¥æ®µè�½
            .alias('text')
        )
        # æ­¥éª¤5ï¼šæˆªå�–æ–‡æœ¬é¦–å°¾å�„1/4éƒ¨åˆ†ï¼ˆä¼˜åŒ–å�‚è€ƒæ–‡çŒ®æ£€æµ‹ï¼‰
        .with_columns([
            pl.col("text")
            .str.slice(pl.col("text").str.len_chars() // 4)  # å�–å��3/4æ–‡æœ¬
            .str.reverse()  # å��è½¬å­—ç¬¦ä¸²ï¼ˆä¾¿äº�ä»�å°¾éƒ¨åŒ¹é…�å�‚è€ƒæ–‡çŒ®ï¼‰
            .alias('rtext'),
            pl.col("text")
            .str.slice(0, pl.col("text").str.len_chars() // 4)  # å�–å‰�1/4æ–‡æœ¬
            .alias('ltext'),
        ])
        # æ­¥éª¤6ï¼šåœ¨å��è½¬æ–‡æœ¬ä¸­å®šä½�å�‚è€ƒæ–‡çŒ®èµ·å§‹ä½�ç½®
        .with_columns(
            pl.col("rtext")
            .str.find(
                r'(?i)\b(secnerefer|erutaretil detic|stnemegdelwonkca)\b'  # "References"/"Cited Literature"/"Acknowledgements"çš„å��å†™
            )
            .alias('ref_idx')  # åŒ¹é…�å�‚è€ƒæ–‡çŒ®æ ‡é¢˜çš„å��è½¬å½¢å¼�
        )
        # æ­¥éª¤7ï¼šå¤„ç�†æœªæ‰¾åˆ°å�‚è€ƒæ–‡çŒ®çš„æƒ…å†µï¼ˆé»˜è®¤ç´¢å¼•0ï¼‰
        .with_columns(
            pl.when(pl.col("ref_idx").is_null())
            .then(0)
            .otherwise(pl.col("ref_idx"))
            .alias("ref_idx")
        )
        # æ­¥éª¤8ï¼šåˆ†ç¦»å�‚è€ƒæ–‡çŒ®å’Œæ­£æ–‡
        .with_columns([
            pl.col("rtext")
            .str.slice(0, pl.col("ref_idx"))  # æˆªå�–å�‚è€ƒæ–‡çŒ®éƒ¨åˆ†
            .str.reverse()  # å��è½¬å›�æ­£å¸¸é¡ºåº�
            .alias("refs"),
            # å�ˆå¹¶æ­£æ–‡ï¼šå‰�1/4æ–‡æœ¬ + å��3/4æ–‡æœ¬ï¼ˆæ�’é™¤å�‚è€ƒæ–‡çŒ®éƒ¨åˆ†ï¼‰
            (pl.col("ltext") + 
             pl.col("rtext").str.slice(pl.col("ref_idx")).str.reverse())
            .alias("body")
        ])
        # æ¸…ç�†ä¸´æ—¶åˆ—
        .drop("rtext", "ltext")
    )


import pandas as pd  # å¯¼å…¥pandasæ•°æ�®å¤„ç�†åº“
from collections import Counter  # å¯¼å…¥Counterç”¨äº�è®¡æ•°ï¼ˆæœ¬ä»£ç �æœªä½¿ç”¨ï¼‰


def extract_candidates(args):
    # æ­¥éª¤1ï¼šè�·å�–è¾“å…¥è·¯å¾„
    parsed_in = get_prefix_path("working") / args['i']  # æ‹¼æ�¥å·¥ä½œç›®å½•å’Œè¾“å…¥å�‚æ•°
    print(f"ğŸ”µ Step 2: Begin ID Extraction Pipeline")
    print(f"   â†’ Will process parsed text files from: {parsed_in}")

    # æ­¥éª¤2ï¼šåŠ è½½æ–‡æœ¬æ•°æ�®
    text_df = get_text_df(parsed_in)  # è°ƒç”¨å¤–éƒ¨å‡½æ•°è�·å�–polars DataFrame
    print(f"ğŸŸ¢ Step 1: Loaded text DataFrame")
    print(f"   â†’ Rows: {text_df.height}, Columns: {list(text_df.columns)}")
    # å±•ç¤ºå‰�2è¡Œæ–‡æœ¬ç‰‡æ®µï¼ˆæˆªå�–å‰�100å­—ç¬¦ï¼‰
    print(text_df.with_columns(pl.col("text").str.slice(0, 100).alias("text_snippet")).head(2).to_pandas())

    # æ­¥éª¤Aï¼šä½¿ç”¨æ­£åˆ™è¡¨è¾¾å¼�æ��å�–ID
    df = text_df.with_columns(pl.col("text").str.extract_all(REGEX_IDS).alias("id")).to_pandas()
    print(f"ğŸŸ¦ [A] Extract candidate IDs")
    print(df[["article_id", "id"]].head(2))

    # æ­¥éª¤Bï¼šå±•å¼€åµŒå¥—åˆ—è¡¨ï¼ˆæ¯�ä¸ªIDå�•ç‹¬æˆ�è¡Œï¼‰
    df = df.explode("id").rename(columns={"id": "match_id"})
    print(f"ğŸŸ¦ [B] Exploded IDs")
    print(df[["article_id", "match_id"]].head(2))

    # æ­¥éª¤Cï¼šIDæ¸…æ´—å¤„ç�†
    df["id"] = df["match_id"]  # ä¿�ç•™å�Ÿå§‹ID
    df["id_nospace"] = df["id"].str.replace(r"\s+", "", regex=True)  # ç§»é™¤ç©ºç™½å­—ç¬¦
    # æ¸…ç�†æœ«å°¾æ— æ•ˆç¬¦å�·
    df["id_cleaned"] = df["id_nospace"].str.replace(r"[-.,;:!?/)\]\(\[]+$", "", regex=True)
    print(f"ğŸŸ¦ [C] Cleaned IDs")
    print(df[["article_id", "id", "id_cleaned"]].head(2))

    # æ­¥éª¤Dï¼šç‰¹æ®ŠDOIæ ¼å¼�æ ‡å‡†åŒ–
    def norm_dryad(x):
        return f"https://doi.org/10.5061/{x.lower()}" if isinstance(x, str) and x.startswith("dryad.") else None
    
    def norm_pasta(x):
        return f"https://doi.org/10.6073/{x.lower()}" if isinstance(x, str) and x.startswith("pasta/") else None
    
    # åº”ç”¨æ ‡å‡†åŒ–å‡½æ•°
    df["id_final_dryad"] = df["id_cleaned"].map(norm_dryad)
    df["id_final_pasta"] = df["id_cleaned"].map(norm_pasta)
    print(f"ğŸŸ¦ [D] Normalized DOIs (dryad/pasta)")
    print(df[["article_id", "id_final_dryad", "id_final_pasta"]].head(2))

    # æ­¥éª¤Eï¼šä¼˜å…ˆçº§å�ˆå¹¶ID
    df["id_use"] = df["id_final_dryad"].combine_first(
        df["id_final_pasta"]).combine_first(df["id_cleaned"])
    print(f"ğŸŸ¦ [E] Chose ID to use")
    print(df[["article_id", "id_use"]].head(2))

    # æ­¥éª¤Fï¼šè¯¯æŠ¥è¿‡æ»¤
    df = df[df["id_use"].notnull()]  # ç§»é™¤ç©ºå€¼
    
    # è¿‡æ»¤æ–‡ç« è‡ªèº«ID
    df = df[~df.apply(lambda row: 
        str(row["article_id"]).replace("_", "/").lower() in str(row["id_use"]).lower(), 
        axis=1)]
    
    df = df[~df["id_use"].str.contains("figshare", na=False)]  # è¿‡æ»¤figshare
    
    # éªŒè¯�DOIå��ç¼€é•¿åº¦
    def valid_doi(x):
        if isinstance(x, str) and x.startswith(DOI_URL):
            return len(x.rsplit("/", 1)[-1]) >= 4
        return True
    df = df[df["id_use"].apply(valid_doi)]
    
    # è¿‡æ»¤æ ¹DOI
    STUBS = ["https://doi.org/10.5061/dryad", 
             "https://doi.org/10.6073/pasta", 
             "https://doi.org/10.5281/zenodo"]
    
    df = df[~df["id_use"].isin(STUBS)]
    
    # æ‹¬å�·åŒ¹é…�éªŒè¯�
    df = df[df["id_use"].str.count(r"\(") == df["id_use"].str.count(r"\)")]
    df = df[df["id_use"].str.count(r"\[") == df["id_use"].str.count(r"\]")]
    print(f"ğŸŸ¦ [F] Filtered false positives (showing a few):")
    print(df[["article_id", "id_use"]].head(5))

    # æ­¥éª¤Gï¼šæ��å�–ä¸Šä¸‹æ–‡çª—å�£
    def get_window(row):
        idx = row["text"].find(row["id_use"])  # æŸ¥æ‰¾IDä½�ç½®
        if idx == -1: return ""
        # è®¡ç®—ä¸Šä¸‹æ–‡çª—å�£
        start = max(idx - args['ws'] - len(str(row["id_use"])), 0)
        end = idx + args['ws'] + len(str(row["id_use"]))
        return row["text"][start:end]
    
    df["window"] = df.apply(get_window, axis=1)  # åº”ç”¨çª—å�£å‡½æ•°
    
    # æœ€ç»ˆæ•°æ�®å¤„ç�†
    df = df[["article_id", "id_use", "window"]].drop_duplicates().rename(
        columns={"id_use": "dataset_id"})
    print(f"\nâœ… Completed extraction: {len(df)} unique (article_id, dataset_id) pairs")
    return df


from tqdm.auto import tqdm


# å®šä¹‰ä¸»æµ�ç¨‹å‡½æ•°
def main_pipeline():
    # è®¾ç½®å�‚æ•°å­—å…¸
    args = {
        'i': 'parsed',                # è¾“å…¥ç›®å½•å��ï¼ˆå­˜å‚¨è§£æ��å��çš„æ–‡æœ¬ï¼‰
        'o': 'extracted_ids.parquet',  # è¾“å‡ºæ–‡ä»¶å��ï¼ˆå­˜å‚¨æ��å�–çš„IDæ•°æ�®ï¼‰
        'gt': 'make-data-count-finding-data-references/train_labels.csv',  # è®­ç»ƒæ ‡ç­¾è·¯å¾„
        'ws': 256                     # æ»‘åŠ¨çª—å�£å¤§å°�ï¼ˆç”¨äº�æ–‡æœ¬å¤„ç�†ï¼‰
    }

    # STEP 1: è§£æ��PDF/XMLæ–‡ä»¶
    print("ğŸŒŸ STEP 1: Parse all PDFs and XMLs to text files")
    base = pathlib.Path('/kaggle/input/make-data-count-finding-data-references')  # Kaggleæ•°æ�®é›†æ ¹ç›®å½•
    split = 'test' if is_submission() else 'train'  # æ ¹æ�®æ��äº¤æ¨¡å¼�é€‰æ‹©æ•°æ�®é›†åˆ†å‰²ï¼ˆæµ‹è¯•é›†/è®­ç»ƒé›†ï¼‰
    pdf_dir = base / split / 'PDF'          # PDFæ–‡ä»¶ç›®å½•
    xml_dir = base / split / 'XML'          # XMLæ–‡ä»¶ç›®å½•
    parsed_dir = get_prefix_path('working') / args['i']  # è§£æ��æ–‡æœ¬è¾“å‡ºç›®å½•
    parse_all_pdfs_xmls(pdf_dir, xml_dir, parsed_dir)    # è°ƒç”¨è§£æ��å‡½æ•°

    # STEP 2: ä»�æ–‡æœ¬ä¸­æ��å�–å€™é€‰æ•°æ�®é›†ID
    print("\nğŸŒŸ STEP 2: Extract candidate dataset IDs from text")
    df = extract_candidates(args)  # è°ƒç”¨IDæ��å�–å‡½æ•°ï¼Œè¿”å›�DataFrame
    out_parq = get_prefix_path('working') / args['o']  # è¾“å‡ºæ–‡ä»¶è·¯å¾„
    df.to_parquet(out_parq)        # ä¿�å­˜ç»“æ�œä¸ºParquetæ ¼å¼�
    print(f"âœ” Saved extracted IDs to: {out_parq} â€” {len(df)} rows")

    # STEP 3: æ�„å»ºç«�èµ›æ��äº¤æ ¼å¼�
    def assign_type(x):  # å®šä¹‰è¾…åŠ©å‡½æ•°ï¼šæ ¹æ�®IDå‰�ç¼€åˆ†ç±»
        if isinstance(x, str) and (x.startswith(DOI_URL) or x.startswith("SAMN")):
            return "Primary"    # DOIæˆ–SRA IDæ ‡è®°ä¸º"ä¸»è¦�"
        else:
            return "Secondary"  # å…¶ä»–æ ‡è®°ä¸º"æ¬¡è¦�"
    
    sub = df.copy()  # åˆ›å»ºå‰¯æœ¬é�¿å…�ä¿®æ”¹å�Ÿæ•°æ�®
    sub['type'] = sub['dataset_id'].apply(assign_type)  # ä¸ºæ¯�ä¸ªIDæ·»åŠ åˆ†ç±»æ ‡ç­¾
    sub = (sub
           .drop_duplicates(subset=['article_id','dataset_id'])  # å�»é™¤é‡�å¤�æ�¡ç›®
           .reset_index(drop=True))  # é‡�ç½®ç´¢å¼•
    sub['row_id'] = range(len(sub))  # æ·»åŠ è¿�ç»­è¡ŒID
    sub = sub[['row_id','article_id','dataset_id','type']]  # æŒ‰éœ€é€‰æ‹©åˆ—

    # ä¿�å­˜æ��äº¤æ–‡ä»¶
    print("\n[main_pipeline] Submission DataFrame (first rows):")
    print(sub.head())  # é¢„è§ˆå‰�5è¡Œ
    submission_path = get_prefix_path('working') / 'submission.csv'
    sub.to_csv(submission_path, index=False)  # ä¿�å­˜CSVï¼ˆç«�èµ›è¦�æ±‚æ ¼å¼�ï¼‰
    print(f"âœ” Submission saved â€” {len(sub)} rows to {submission_path}")

    # STEP 4: éªŒè¯�è¯„åˆ†ï¼ˆä»…åœ¨è®­ç»ƒæ¨¡å¼�æ‰§è¡Œï¼‰
    gt_path = pathlib.Path('/kaggle/input/make-data-count-finding-data-references/train_labels.csv')
    if gt_path.exists():  # æ£€æŸ¥è®­ç»ƒæ ‡ç­¾æ˜¯å�¦å­˜åœ¨
        print("\nğŸ“Š Validation on TRAIN SPLIT")
        preds = pl.read_csv(submission_path).select(['article_id','dataset_id','type'])  # åŠ è½½é¢„æµ‹ç»“æ�œ
        gt = (pl.read_csv(gt_path)
              .filter(pl.col('type')!='Missing')  # è¿‡æ»¤æ— æ•ˆæ ‡ç­¾
              .select(['article_id','dataset_id','type']))  # åŠ è½½çœŸå®�æ ‡ç­¾
        score(preds, gt, on=['article_id','dataset_id','type'])  # è°ƒç”¨è¯„åˆ†å‡½æ•°

    print("\nâœ… Pipeline finished!")


# æ‰§è¡Œä¸»æµ�ç¨‹
main_pipeline()


# å®šä¹‰å��ä¸º show_submission çš„å‡½æ•°ï¼Œå�‚æ•° sub_csv é»˜è®¤å€¼ä¸º '/kaggle/working/submission.csv'
def show_submission(sub_csv='/kaggle/working/submission.csv'):
    
    # ä½¿ç”¨ pandas è¯»å�– CSV æ–‡ä»¶ï¼Œå°†æ•°æ�®åŠ è½½åˆ° DataFrame å¯¹è±¡ df ä¸­
    df = pd.read_csv(sub_csv) 
    
    # é‡�ç½® DataFrame ç´¢å¼•ï¼šdrop=True è¡¨ç¤ºä¸¢å¼ƒæ—§ç´¢å¼•åˆ—ï¼Œç”Ÿæˆ�ä»�0å¼€å§‹çš„æ–°æ•´æ•°ç´¢å¼•
    df = df.reset_index(drop=True) 
    
    # åˆ›å»ºå��ä¸º 'row_id' çš„æ–°åˆ—ï¼Œå…¶å€¼ç­‰äº� DataFrame çš„ç´¢å¼•å€¼ï¼ˆå�³è¡Œå�·ï¼‰
    df['row_id'] = df.index 
    
    # æ‰“å�° DataFrame çš„å­�é›†ï¼š
    # 1. é€‰å�– ['row_id', 'article_id', 'dataset_id', 'type'] å››åˆ—
    # 2. to_string(index=False) è¡¨ç¤ºè¾“å‡ºæ—¶ä¸�æ˜¾ç¤ºç´¢å¼•åˆ—
    # 3. print() å°†æ ¼å¼�åŒ–çš„è¡¨æ ¼æ•°æ�®è¾“å‡ºåˆ°æ�§åˆ¶å�°
    print(df[['row_id', 'article_id', 'dataset_id', 'type']].to_string(index=False))  

# è°ƒç”¨ show_submission å‡½æ•°ï¼ˆä½¿ç”¨é»˜è®¤å�‚æ•°ï¼‰
show_submission()


! rm -rf parsed
! rm -rf src
! rm -rf extracted_ids.parquet

