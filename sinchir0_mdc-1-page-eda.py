!apt-get -y install poppler-utils
!pip install -qqq pdf2image


import polars as pl


train = pl.read_csv("/kaggle/input/make-data-count-finding-data-references/train_labels.csv")


train


sample_pdf_path = train["article_id"][0]
print(sample_pdf_path)


from pdf2image import convert_from_path
from IPython.display import display

images = convert_from_path(
    f"/kaggle/input/make-data-count-finding-data-references/train/PDF/{sample_pdf_path}.pdf",
    dpi=200
)

for page_img in images:
    display(page_img)




