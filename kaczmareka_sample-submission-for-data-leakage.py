!pip install PyMuPDF


import pandas as pd
import fitz


#Function to read PDF files
def read_pdf(path: str) -> str:
    text = []
    doc = fitz.open(path)
    for page in doc:
        text.append(page.get_text())
    doc.close()
    text = ''.join(text)
    return text


#Read file
text_all=read_pdf("") #here provide path to your file


#Print first 10 chars of read text
text_all[:10]


#Sample data
data = {
    "id": [1, 2, 3, 4, 5, 6, 7],
    "hidden_message": ["Answer to text 1", "Answer to text 2", " ", "Answer to text 4", "Answer to text 5", "Answer to text 1", "Answer to text 2"]
}


# Create the DataFrame
df = pd.DataFrame(data)
df


# Save to CSV
df.to_csv("sample_submission.csv", index=False)

