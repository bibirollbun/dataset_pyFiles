!pip install internetarchive requests
!pip install PyPDF2


import os
import subprocess
import requests
from internetarchive import search_items, get_item, download
from PyPDF2 import PdfReader

def search_internet_archive(query, max_results=5):
    """
    Search IA for items that mention `query` in title OR subject OR description.
    Returns up to max_results dicts with 'identifier' and 'title'.
    """
    jql = (
        f'title:({query}) '
        f'OR subject:({query}) '
        f'OR description:({query})'
    )
    results = []
    for count, item in enumerate(search_items(jql), start=1):
        results.append({
            'identifier': item['identifier'],
            'title': item.get('title', '(no title)')
        })
        if count >= max_results:
            break
    return results

def decompress_lpdf(lpdf_path, keep_original=True):
    """
    Decompress a .lpdf (LZip‐compressed PDF) to .pdf.
    Requires the `lzip` command‐line tool.
    """
    # Derive output PDF path
    pdf_path = os.path.splitext(lpdf_path)[0] + '.pdf'
    # -d = decompress, -k = keep original if requested
    cmd = ['lzip', '-d']
    if keep_original:
        cmd.append('-k')
    cmd.append(lpdf_path)
    subprocess.run(cmd, check=True)
    return pdf_path

def extract_text_from_pdf(pdf_path, txt_path=None):
    """
    Extract plain text from a PDF using PyPDF2.
    """
    if txt_path is None:
        txt_path = os.path.splitext(pdf_path)[0] + '.txt'
    reader = PdfReader(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as out:
        for page in reader.pages:
            text = page.extract_text()
            if text:
                out.write(text)
    return txt_path

def download_from_internet_archive(
    identifier,
    download_dir='downloads',
    max_downloads=10,
    extensions=('.pdf', '.lcpdf', '.lpdf', '.txt', '.djvu', '.epub')
):
    """
    Download up to max_downloads files for `identifier` whose names end
    with one of extensions, skipping restricted files.  If an .lpdf
    is found, decompress it and extract its text automatically.
    """
    item = get_item(identifier)
    target_dir = os.path.join(download_dir, identifier)
    os.makedirs(target_dir, exist_ok=True)

    # Collect filenames to download
    to_fetch = []
    for f in item.files:
        name = f.get('name', '')
        if any(name.lower().endswith(ext) for ext in extensions):
            if f.get('restricted') == '1':
                print(f"Skipping restricted: {name}")
                continue
            to_fetch.append(name)
            if len(to_fetch) >= max_downloads:
                break

    if not to_fetch:
        print(f"No public files with extensions {extensions} for '{identifier}'")
        return

    # Download the files
    try:
        download(
            identifier,
            files=to_fetch,
            destdir=target_dir,
            verbose=True
        )
    except Exception as e:
        print(f"Error downloading from {identifier}: {e}")
        return

    # Post-process any .lpdf files
    for fname in to_fetch:
        if fname.lower().endswith('.lpdf'):
            lpdf_path = os.path.join(target_dir, fname)
            try:
                print(f"Decompressing {fname} …")
                pdf_path = decompress_lpdf(lpdf_path)
                print(f"Extracting text from {os.path.basename(pdf_path)} …")
                txt_path = extract_text_from_pdf(pdf_path)
                print(f"→ Text saved to {os.path.basename(txt_path)}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to decompress {fname}: {e}")
            except Exception as e:
                print(f"Error extracting from {fname}: {e}")

def download_gutenberg_text(book_id, save_dir='downloads/gutenberg'):
    """
    Download the plain-text version of a Gutenberg book by its ID.
    """
    os.makedirs(save_dir, exist_ok=True)
    for variant in (f'{book_id}-0.txt', f'{book_id}.txt'):
        url = f'https://www.gutenberg.org/files/{book_id}/{variant}'
        try:
            r = requests.get(url)
            r.raise_for_status()
            path = os.path.join(save_dir, f'gutenberg_{book_id}.txt')
            with open(path, 'w', encoding='utf-8') as out:
                out.write(r.text)
            print(f"Downloaded Gutenberg book {book_id} as {variant}")
            return
        except requests.HTTPError:
            continue
    print(f"No text file found for Gutenberg ID {book_id}")

def main():
    # 1) Search & download IA items for "Amazon archaeology"
    items = search_internet_archive('Amazon archaeology', max_results=10)
    for itm in items:
        print(f"Found “{itm['title']}” (ID: {itm['identifier']})")
        download_from_internet_archive(
            itm['identifier'],
            max_downloads=10
        )

    # 2) Gutenberg examples (optional)
    for gid in (12472, 22752):
        download_gutenberg_text(gid)

if __name__ == '__main__':
    main()

