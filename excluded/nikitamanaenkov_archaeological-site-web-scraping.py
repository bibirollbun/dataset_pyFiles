import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "http://portal.iphan.gov.br/sgpa/cnsa_resultado.php"
DETAILS_BASE_URL = "http://portal.iphan.gov.br/sgpa/"

states = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
          "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
          "RO", "RR", "RS", "SC", "SE", "SP", "TO"]

all_data = []

html_folder = "cnsa_details_html"
os.makedirs(html_folder, exist_ok=True)

def get_detail_text(soup, label):
    el = soup.find("td", string=lambda x: x and label in x)
    if el:
        next_td = el.find_next_sibling("td")
        if next_td:
            return next_td.text.strip()
    return ""

for uf in states:
    print(f"Scraping state: {uf}")
    
    payload = {
        "acao": "consultar",
        "cnsa_uf": uf,
        "cnsa_municipio": "",
        "cnsa_historico": "",
        "cnsa_precolonial": "",
        "cnsa_nome": "",
        "cnsa_decontato": "",
        "cnsa_responsavel": "",
    }

    response = requests.post(BASE_URL, data=payload)
    if response.status_code != 200:
        print(f"Failed to retrieve data for state {uf}")
        continue

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="listagem")
    if not table:
        print(f"No data table found for state {uf}")
        continue

    rows = table.find_all("tr")[1:] 

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 5:
            continue

        cnsa_code = cols[0].text.strip()
        name = cols[1].text.strip()
        municipality = cols[2].text.strip()
        state = cols[3].text.strip()
        details_path = cols[4].find("a")["href"]
        details_url = DETAILS_BASE_URL + details_path

        details_resp = requests.get(details_url)
        if details_resp.status_code != 200:
            print(f"Failed to retrieve details for {cnsa_code}")
            continue
        

        details_resp.encoding = 'iso-8859-1' 
        
        details_html = details_resp.text  
        
        filename = f"{cnsa_code}.html"
        filepath = os.path.join(html_folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(details_html)

        details_soup = BeautifulSoup(details_html, "html.parser")

        description = get_detail_text(details_soup, "Descrição sumária do sítio")
        length = get_detail_text(details_soup, "Comprimento")
        width = get_detail_text(details_soup, "Largura")
        max_height = get_detail_text(details_soup, "Altura máxima")
        area = get_detail_text(details_soup, "Área")
        responsible = get_detail_text(details_soup, "Responsável")
        contact = get_detail_text(details_soup, "Contato")
        historical_period = get_detail_text(details_soup, "Período Histórico")
        precolonial = get_detail_text(details_soup, "Pré-colonial")
        coordinates = get_detail_text(details_soup, "Coordenadas")

        print(f"Scraped {cnsa_code}")

        all_data.append({
            "CNSA_Code": cnsa_code,
            "Name": name,
            "Municipality": municipality,
            "State": state,
            "Description": description,
            "Length": length,
            "Width": width,
            "Max_Height": max_height,
            "Area": area,
            "Responsible": responsible,
            "Contact": contact,
            "Historical_Period": historical_period,
            "Precolonial": precolonial,
            "Coordinates": coordinates,
            "Details_URL": details_url
        })

        time.sleep(0.1)

df = pd.DataFrame(all_data)
df.to_csv("cnsa_data_all_states.csv", index=False, encoding="utf-8-sig")
print("Scraping complete, data saved to cnsa_data_all_states.csv")


