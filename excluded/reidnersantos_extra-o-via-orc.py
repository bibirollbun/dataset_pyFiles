! uv pip uninstall --system 'tensorflow'
! uv pip install --system --no-index --find-links='/kaggle/input/latest-mdc-whls/whls' 'pymupdf' 'vllm' 'triton' 'logits-processor-zoo' 'numpy<2'
! mkdir -p /tmp/src


import polars as pl
TRAIN_LABELS = "/kaggle/input/make-data-count-finding-data-references/train_labels.csv"


df = pl.read_csv(TRAIN_LABELS)

# Verifica se existe o article_id
exists = (df['article_id'] == '10.1590_0104-4060.59642').any()
print(exists)  # True se existir, False caso não exista
filtered = df.filter(df['article_id'] == '10.1590_0104-4060.59642')
filtered


import polars as pl
from pathlib import Path
import os
import pymupdf

TRAIN_LABELS = "/kaggle/input/make-data-count-finding-data-references/train_labels.csv"


 ## ler esses PDF  e verifica como eles estão 
PDF1 = '/kaggle/input/make-data-count-finding-data-references/train/PDF/10.1590_0104-4060.59642.pdf'
PDF2 = '/kaggle/input/make-data-count-finding-data-references/train/PDF/10.1590_1678-4162-10032.pdf'

#PDF = '/kaggle/input/make-data-count-finding-data-references/train/PDF/10.1002_2017jc013030.pdf'

df = pl.read_csv(TRAIN_LABELS)
filtered_2 = df.filter(df['article_id'] == '10.1590_1678-4162-10032')
filtered_2


# Exemplo: última e penúltima página
PDF = '/kaggle/input/make-data-count-finding-data-references/train/PDF/10.1002_2017jc013030.pdf'



def localizar_palavra_em_pdf(pdf_file, palavras):
    doc = pymupdf.open(pdf_file)
    paginas_encontradas = {}
    qtd_paginas = doc.page_count
    for num, page in enumerate(doc):
        texto = page.get_text()
        for palavra in palavras:
            if palavra.lower() in texto.lower():
                if palavra not in paginas_encontradas:
                    paginas_encontradas[palavra] = []
                paginas_encontradas[palavra].append(num)  # índice da página

    doc.close()
    return paginas_encontradas,qtd_paginas


resultados,qtd_paginas = localizar_palavra_em_pdf(PDF, ["References", "Bibliography"])
print(resultados,qtd_paginas)


start = int(resultados['References'][0]) - qtd_paginas
start


import pymupdf
from PIL import Image
from IPython.display import display

# Abre o PDF


def show_pdf_img(PDF_PATH,page_num):
    doc =pymupdf.open(PDF)
    
    total_paginas = doc.page_count
    print(total_paginas)
    
    page = doc[page_num]  
    
    pix = page.get_pixmap(dpi=200)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    display(img)

    doc.close()


 ## ler esses PDF  e verifica como eles estão 
PDF1 = '/kaggle/input/make-data-count-finding-data-references/train/PDF/10.1590_0104-4060.59642.pdf'
PDF2 = '/kaggle/input/make-data-count-finding-data-references/train/PDF/10.1590_1678-4162-10032.pdf'


show_pdf_img(PDF1,-3)
#show_pdf_img(PDF1,1)

#for idx in range(20):
 #   show_pdf_img(PDF,idx)



from PIL import Image
from IPython.display import display

def show_pdf_imgs(PDF_PATH, page_nums):
    doc =pymupdf.open(PDF_PATH)
    
    total_paginas = doc.page_count
    print(f"Total de páginas: {total_paginas}")
    
    # Garante que page_nums seja lista
    if isinstance(page_nums, int):
        page_nums = [page_nums]
    
    for num in page_nums:
        page = doc[num]  # Pode ser negativo (ex: -1 última página)
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        display(img)
    return img
    
    doc.close()


#start = -3
start =  0 
img = show_pdf_imgs(PDF1, list(range(start, 1)))  


#start = -6
#img = show_pdf_imgs(PDF, list(range(start, -5)))  
print(PDF)


display(img)


import easyocr
import numpy as np

reader = easyocr.Reader(['en'])

img_np = np.array(img)
resultados = reader.readtext(img_np, detail=0)  # só o texto, sem bbox nem confiança

print("Texto extraído pelo OCR:")
for texto in resultados:
    print(texto)



import easyocr
import numpy as np
from PIL import Image

reader = easyocr.Reader(['en'])

def recortar_contexto_dataset(img_pil, termo, raio=50):
    img_np = np.array(img_pil)
    resultados = reader.readtext(img_np, detail=1)
    
    for bbox, texto, conf in resultados:
        if termo.lower() in texto.lower():
            # bbox é uma lista de 4 pontos (x,y)
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x_min, x_max = max(min(xs)-raio, 0), min(max(xs)+raio, img_pil.width)
            y_min, y_max = max(min(ys)-raio, 0), min(max(ys)+raio, img_pil.height)
            
            trecho = img_pil.crop((x_min, y_min, x_max, y_max))
            return trecho
    
    return None  # termo não encontrado

t = recortar_contexto_dataset(img,'10.17882/49388')

#t = recortar_contexto_dataset(img,'"https://doi.org/10.17882/49388')
if t is not None:
    display(t)
else:
    print("Termo não encontrado na imagem.")



"""
import easyocr
import numpy as np

reader = easyocr.Reader(['en'])

img_np = np.array(t)
resultados = reader.readtext(img_np, detail=0)  # só o texto, sem bbox nem confiança

print("Texto extraído pelo OCR:")
for texto in resultados:
    print(texto)


""" 





import fitz  # pymupdf
from PIL import Image
from IPython.display import display, clear_output
import time

def show_all_pages(pdf_path, delay=1.5):
    doc =pymupdf.open(PDF)
    
    total_paginas = doc.page_count
    print(f"Total de páginas: {total_paginas}")
    
    for i in range(total_paginas):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)  # dpi menor para carregar mais rápido, pode ajustar
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        #clear_output(wait=True)  # limpa a saída para mostrar só uma página de cada vez
        print(f"Página {i+1} de {total_paginas}")
        display(img)
        
        #time.sleep(delay)  # pausa antes da próxima página
    
    doc.close()

# Exemplo de uso
pdf_path = '/kaggle/input/make-data-count-finding-data-references/test/PDF/10.1002_cssc.202201821.pdf'
show_all_pages(pdf_path)




import pymupdf
from PIL import Image
from IPython.display import display

# Abre o PDF

PDF = '/kaggle/input/make-data-count-finding-data-references/test/PDF/10.1002_cssc.202201821.pdf'

def show_pdf_img(PDF_PATH,page_num):
    doc =pymupdf.open(PDF)
    
    total_paginas = doc.page_count
    print(total_paginas)
    
    page = doc[page_num]  
    
    pix = page.get_pixmap(dpi=200)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    display(img)

    doc.close()
#show_pdf_img(PDF,-5)




def localizar_palavra_em_pdf(pdf_file, palavras):
    doc = pymupdf.open(pdf_file)
    paginas_encontradas = {}

    for num, page in enumerate(doc):
        texto = page.get_text()
        for palavra in palavras:
            if palavra.lower() in texto.lower():
                if palavra not in paginas_encontradas:
                    paginas_encontradas[palavra] = []
                paginas_encontradas[palavra].append(num)  # índice da página

    doc.close()
    return paginas_encontradas


#resultados = localizar_palavra_em_pdf(PDF, ["References", "Bibliography"])
#print(resultados)






def read_pdf(pdf_file):

    text = ""
    with pymupdf.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
            
    print(text)


#read_pdf(PDF)

