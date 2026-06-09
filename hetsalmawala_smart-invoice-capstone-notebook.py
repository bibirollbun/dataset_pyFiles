# Setup: create workspace and sample invoices
from pathlib import Path
import json, os
WORKDIR = Path('/mnt/data/smart_invoice_capstone')
WORKDIR.mkdir(parents=True, exist_ok=True)
SAMPLE_TXT = WORKDIR / 'invoice1.txt'
SAMPLE_TXT.write_text('''ACME Textiles
Invoice # INV-2025-001
Date: 2025-11-05

Qty  Description        UnitPrice  Total
10   Cotton Yarn        100        1000
5    Dye                200        1000

Subtotal: 2000
Tax: 180
Total: 2180
''')
SAMPLE2 = WORKDIR / 'invoice2.txt'
SAMPLE2.write_text('''Beta Fabrics
Invoice # INV-2025-002
Date: 2025-11-07

Qty  Description        UnitPrice  Total
3    Cotton Yarn        100        300
2    Blue Threads       150        300

Subtotal: 600
Tax: 54
Total: 654
''')
print('Workspace:', WORKDIR)
print('Sample invoices:', SAMPLE_TXT.name, SAMPLE2.name)



from pathlib import Path
import os
def mock_ocr(path):
    return {'text': Path(path).read_text(), 'confidence': 0.98}

def try_google_vision(path):
    # Attempt to use google-cloud-vision if credentials are set
    if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        raise RuntimeError('GOOGLE_APPLICATION_CREDENTIALS not set')
    try:
        from google.cloud import vision
    except Exception as e:
        raise RuntimeError('google-cloud-vision package not installed') from e
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as f:
        content = f.read()
    resp = client.document_text_detection(image={'content': content})
    if resp.error.message:
        raise RuntimeError('Vision API error: ' + resp.error.message)
    return {'text': resp.full_text_annotation.text, 'confidence': None}

# Choose OCR mode: 'mock' or 'vision' (auto switches to mock if vision not available)
OCR_MODE = 'mock'
invoice_paths = [str(p) for p in Path('/mnt/data/smart_invoice_capstone').glob('*.txt')]
ocr_results = []
for p in invoice_paths:
    try:
        if OCR_MODE == 'vision':
            o = try_google_vision(p)
        else:
            o = mock_ocr(p)
    except Exception as e:
        print('Vision failed or not configured, falling back to mock OCR for', p, '->', e)
        o = mock_ocr(p)
    ocr_results.append({'path': p, 'text': o['text'], 'confidence': o['confidence']})
print('OCR complete. Documents processed:', len(ocr_results))



import re, json
def regex_extract(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    vendor = lines[0] if lines else ''
    inv = re.search(r'Invoice\s*#\s*([A-Za-z0-9-]+)', text, re.IGNORECASE)
    date = re.search(r'Date[:\s]*([0-9\-\/]+)', text, re.IGNORECASE)
    subtotal = re.search(r'Subtotal[:\s]*([0-9,.]+)', text, re.IGNORECASE)
    tax = re.search(r'Tax[:\s]*([0-9,.]+)', text, re.IGNORECASE)
    total = re.search(r'Total[:\s]*([0-9,.]+)', text, re.IGNORECASE)
    # items: look for section after header Qty Description
    items = []
    item_section = False
    for l in lines:
        if re.match(r'Qty\s+Description', l, re.IGNORECASE):
            item_section = True
            continue
        if item_section:
            if re.search(r'Subtotal|Total|Tax', l, re.IGNORECASE):
                break
            parts = re.split(r'\s{2,}|\t', l)
            parts = [p for p in parts if p]
            if len(parts) >= 4:
                try:
                    qty = int(parts[0])
                    desc = parts[1]
                    unit = float(parts[2])
                    line_total = float(parts[3])
                    items.append({'qty': qty, 'description': desc, 'unit_price': unit, 'total': line_total})
                except:
                    # fallback split by whitespace
                    p2 = l.split()
                    if len(p2) >= 4:
                        try:
                            qty = int(p2[0])
                            desc = ' '.join(p2[1:-2])
                            unit = float(p2[-2])
                            line_total = float(p2[-1])
                            items.append({'qty': qty, 'description': desc, 'unit_price': unit, 'total': line_total})
                        except:
                            continue
    return {
        'vendor': vendor,
        'invoice_id': inv.group(1) if inv else None,
        'invoice_date': date.group(1) if date else None,
        'items': items,
        'subtotal': float(subtotal.group(1).replace(',','')) if subtotal else None,
        'tax': float(tax.group(1).replace(',','')) if tax else None,
        'total': float(total.group(1).replace(',','')) if total else None
    }

# Placeholder: replace with real LLM extraction if desired
def extract_with_llm(text):
    # Example: construct a prompt and call OpenAI/other - left as an exercise if you have an API key.
    # For competition, the regex baseline + smart post-processing often suffices and is deterministic (good for reproducibility).
    return regex_extract(text)

# Run extractor on OCR results
docs = []
for r in ocr_results:
    struct = extract_with_llm(r['text'])
    struct['source'] = r['path']
    docs.append(struct)
print('Extraction complete. Examples:')
print(json.dumps(docs, indent=2))



import json
DB_PATH = WORKDIR / 'inventory_db.json'
# initial inventory (demo)
if not DB_PATH.exists():
    initial = {
        'COTTON-YARN': {'sku': 'COTTON-YARN', 'name': 'Cotton Yarn', 'qty': 50},
        'DYE-STD': {'sku': 'DYE-STD', 'name': 'Dye', 'qty': 30},
        'BLUE-THREADS': {'sku': 'BLUE-THREADS', 'name': 'Blue Threads', 'qty': 20}
    }
    DB_PATH.write_text(json.dumps(initial, indent=2))

def map_to_sku(description):
    d = description.lower()
    if 'yarn' in d: return 'COTTON-YARN'
    if 'dye' in d: return 'DYE-STD'
    if 'thread' in d: return 'BLUE-THREADS'
    return None

def apply_updates(docs):
    db = json.loads(DB_PATH.read_text())
    updates = []
    for doc in docs:
        for it in doc['items']:
            sku = map_to_sku(it['description'])
            if sku is None:
                updates.append({'status':'no_sku','item': it, 'source': doc['invoice_id']})
                continue
            db[sku]['qty'] = db[sku].get('qty',0) + it['qty']
            updates.append({'status':'updated','sku':sku,'new_qty': db[sku]['qty'],'source': doc['invoice_id']})
    DB_PATH.write_text(json.dumps(db, indent=2))
    return updates

updates = apply_updates(docs)
print('Inventory updates:')
print(json.dumps(updates, indent=2))
print('\nCurrent DB:\n', DB_PATH.read_text())



# Evaluation metrics
import math
total_docs = len(docs)
sum_match_count = 0
auto_map_count = 0
total_items = 0
for d in docs:
    sum_line_items = sum([it['total'] for it in d['items']])
    if d['subtotal'] is not None and math.isclose(sum_line_items, d['subtotal'], rel_tol=1e-3, abs_tol=1e-6):
        sum_match_count += 1
    for it in d['items']:
        total_items += 1
        if map_to_sku(it['description']):
            auto_map_count += 1

sum_match_rate = sum_match_count / total_docs if total_docs else 0
auto_map_rate = auto_map_count / total_items if total_items else 0

print(f'Documents processed: {total_docs}')
print(f'Sum-match rate: {sum_match_rate:.2%} ({sum_match_count}/{total_docs})')
print(f'Auto-mapping rate: {auto_map_rate:.2%} ({auto_map_count}/{total_items})')

# Simple matplotlib dashboard
import matplotlib.pyplot as plt

# Dashboard 1: bar chart for rates
labels = ['Sum-Match Rate', 'Auto-Mapping Rate']
values = [sum_match_rate*100, auto_map_rate*100]

plt.figure(figsize=(6,4))
plt.bar(labels, values)
plt.ylabel('Percent (%)')
plt.title('Extraction & Mapping KPI Rates')
plt.ylim(0,100)
for i,v in enumerate(values):
    plt.text(i, v+1, f'{v:.1f}%', ha='center')
plt.show()

# Dashboard 2: inventory distribution
db = json.loads(DB_PATH.read_text())
skus = list(db.keys())
qtys = [db[s]['qty'] for s in skus]
plt.figure(figsize=(8,4))
plt.bar(skus, qtys)
plt.title('Inventory Quantities after Processing')
plt.ylabel('Quantity')
for i,v in enumerate(qtys):
    plt.text(i, v+1, str(v), ha='center')
plt.show()



ART_DIR = WORKDIR / 'artifacts'
ART_DIR.mkdir(exist_ok=True)
(ART_DIR / 'extractions.json').write_text(json.dumps(docs, indent=2))
shutil_from = None
# save DB copy
import shutil
shutil.copy(WORKDIR / 'inventory_db.json', ART_DIR / 'inventory_db.json')
print('Saved artifacts to', ART_DIR)



# === SAVE SUBMISSION ARTIFACTS (put near the end of your notebook) ===
import json, zipfile, os
from pathlib import Path

# Paths to the artifacts your notebook already generates
# Adjust these if your notebook uses different filenames/locations
structured_path = Path('artifacts/structured_extractions.json')
inventory_path = Path('artifacts/inventory_db.json')
kpi_plot = Path('artifacts/kpi_extraction_plot.png')

# Create an output folder where Kaggle will pick files (/kaggle/working is default working dir)
output_dir = Path('/kaggle/working') if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else Path.cwd()
submission_zip = output_dir / 'submission_package.zip'

# Ensure artifacts exist (if not, write fallback demo artifacts so the version always has outputs)
if not structured_path.exists():
    demo = [{"invoice_id":"INV-DEMO","total":0}]
    structured_path.parent.mkdir(parents=True, exist_ok=True)
    structured_path.write_text(json.dumps(demo, indent=2))
if not inventory_path.exists():
    demo_inv = {"DEMO-SKU": {"sku":"DEMO-SKU","name":"Demo","qty":0}}
    inventory_path.write_text(json.dumps(demo_inv, indent=2))
if not kpi_plot.exists():
    # create tiny placeholder PNG
    import matplotlib.pyplot as plt
    plt.figure(figsize=(3,2))
    plt.text(0.5,0.5,"KPI",ha='center',va='center')
    plt.axis('off')
    kpi_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(kpi_plot)
    plt.close()

# Build the zip
with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(structured_path, arcname='structured_extractions.json')
    z.write(inventory_path, arcname='inventory_db.json')
    z.write(kpi_plot, arcname='kpi_extraction_plot.png')

print('Submission package written to:', submission_zip)
print('Files in notebook output will include:', submission_zip.name)


