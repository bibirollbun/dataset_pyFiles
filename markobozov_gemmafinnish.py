!mkdir datasets


import pandas as pd
import glob
import json
import os
import google.generativeai as genai
import time


files = glob.glob("datasets/*.parquet")


df1 = pd.read_parquet(files[0], engine='pyarrow')
df1.head(1)


json_data = [
    {
        "conversations": [
            {"from": "human", "value": row["instruction"]},
            {"from": "gpt", "value": row["response"]},
            {"from": "human", "value": row["instruction_orig"]},
            {"from": "gpt", "value": row["response_orig"]},
        ]
    }
    for _, row in df1.iterrows()
]


output_file = "partition_1.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)


df2 = pd.read_parquet(files[1], engine='pyarrow')
df2.head(1)


json_data = [
    {
        "conversations": [
            {"from": "human", "value": row["instruction"]},
            {"from": "gpt", "value": row["response"]},
        ]
    }
    for _, row in df2.iterrows()
]

output_file = "partition_2.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)



df4 = pd.read_parquet(files[3], engine='pyarrow')
df4.iloc[1]["messages"]


json_data = [
    {
        "conversations": [
            {"from": "human", "value": msg["content"]} if msg["role"] == "user" else {"from": "gpt", "value": msg["content"]}
            for msg in row["messages"]
        ]
    }
    for _, row in df4.iterrows()
]

output_file = "partition_3.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)


df5 = pd.read_parquet(files[4], engine='pyarrow')
df5.head(1)


df6 = pd.read_parquet(files[5], engine='pyarrow')
df6.head(1)


df3 = pd.read_parquet(files[2], engine='pyarrow')
df3.head(1)


!pip3 install google-cloud-translate
!pip3 install google-generativeai


df7 = pd.read_parquet(files[3], engine='pyarrow')
df7.iloc[1]["conversation"]


def process_conversation(conversation):
    formatted_conversation = []
    for msg in conversation:
        formatted_conversation.append({
            "from": "human" if msg['role'] == 'user' else "gpt",
            "value": msg['content']
        })
    return {"conversations": formatted_conversation}

output = [process_conversation(conversation) for conversation in df7['conversation']]

with open('partition_4.json', 'w') as f:
    json.dump(output, f, indent=4)



genai.configure(api_key="API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")


with open('updated_output.json', 'r') as file:
    data = json.load(file)


checkpoint_file = 'checkpoint.json' # 1501 here
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    processed_count = checkpoint['processed_count']
else:
    processed_count = 0


updated_conversations = []

def save_progress(data, processed_count):
    with open('checkpoint.json', 'w') as f:
        json.dump({'processed_count': processed_count}, f)
    with open('updated_output_2.json', 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


for i in range(processed_count, len(data)):
    time.sleep(5)
    entry = data[i]
    try:
        conversations = entry.get('conversations', [])
        for convo in conversations:
            if convo['from'] != 'human':
                original_text = convo['value']
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(f"You need to translate to finnish, return only finnish text: {original_text}")


                convo['value'] = response.text

        updated_conversations.append(entry)
        save_progress(updated_conversations, i + 1)
        
        print(f"Processed {i + 1}/{len(data)} entries.")

    except Exception as e:
        print(f"Error processing entry {i + 1}: {e}. Skipping this entry.")



file1 = 'partition_1.json'
file2 = 'partition_2.json'
file3 = 'partition_3.json'
file4 = "updated_output_2.json"

with open(file1, 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open(file2, 'r', encoding='utf-8') as f:
    data2 = json.load(f)

with open(file3, 'r', encoding='utf-8') as f:
    data3 = json.load(f)

with open(file4, 'r', encoding='utf-8') as f:
    data4 = json.load(f)

merged_data = data1 + data2 + data3 + data4


output_file = 'merged.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)


input_file = "kalevala.txt"  
output_file = "kalevala.json"  


def generate_conversation(segment):
    query_prompt = f"Luo kysymys seuraavan tekstin perusteella: {segment}"
    query_response = model.generate_content(f"Vastaa tekstin perusteella: {query_prompt}")
    human_query = query_response.text

    answer_prompt = f"{segment}\n\nKysymys: {human_query.strip()}\nVastaa lyhyesti. Warning: Only answer and only finnish language. The question must contain infromation that this is from Kalevala poem"
    answer_response = model.generate_content(answer_prompt)
    gpt_answer = answer_response.text

    return {
        "conversations": [
            {
                "from": "human",
                "value": human_query.strip()
            },
            {
                "from": "gpt",
                "value": gpt_answer.strip()
            }
        ]
    }


generate_conversation(""""TeenkÃ¶ tuulehen tupani,
aalloillen asuinsijani?
Tuuli kaatavi tupasen,
aalto vie asuinsijani."
Niin silloin ve'en emonen,
veen emonen, ilman impi,
nosti polvea merestÃ¤,
lapaluuta lainehesta
sotkalle pesÃ¤n sijaksi,
asuinmaaksi armahaksi.
Tuo sotka, sorea lintu,
liiteleikse, laateleikse.
Keksi polven veen emosen
sinervÃ¤isellÃ¤ selÃ¤llÃ¤;
luuli heinÃ¤mÃ¤ttÃ¤hÃ¤ksi,
tuoreheksi turpeheksi.
Lentelevi, liitelevi,
pÃ¤Ã¤hÃ¤n polven laskeuvi.
Siihen laativi pesÃ¤nsÃ¤,
muni kultaiset munansa:
kuusi kultaista munoa,
rautamunan seitsemÃ¤nnen.""")



with open(input_file, "r", encoding="utf-8") as f:
   lines = f.readlines()

segments = ["".join(lines[i:i+50]).strip() for i in range(0, len(lines), 50)]


conversations = []
for i, segment in enumerate(segments):
    time.sleep(5)
    if segment.strip():  
        try:
            conversation = generate_conversation(segment.strip())
            conversations.append(conversation)
            print(f"Processed segment {i + 1}/{len(segments)}")
        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(conversations, out_f, ensure_ascii=False, indent=4)


with open(output_file, "w", encoding="utf-8") as out_f:
    json.dump(conversations, out_f, ensure_ascii=False, indent=4)



input_file = "kanteletar.txt"  
output_file = "kanteletar.json"  


def generate_conversation(segment):
    query_prompt = f"Luo kysymys seuraavan tekstin perusteella: {segment}"
    query_response = model.generate_content(f"Vastaa tekstin perusteella: {query_prompt}")
    human_query = query_response.text

    answer_prompt = f"{segment}\n\nKysymys: {human_query.strip()}\nVastaa lyhyesti. Warning: Only answer and only finnish language. The question must contain infromation that this is from Kanteletar poem"
    answer_response = model.generate_content(answer_prompt)
    gpt_answer = answer_response.text

    return {
        "conversations": [
            {
                "from": "human",
                "value": human_query.strip()
            },
            {
                "from": "gpt",
                "value": gpt_answer.strip()
            }
        ]
    }


generate_conversation(
    """
14. Karjalan neito.
             MinÃ¤ olen Suomen neito, Suomen koria kukka,
          Moni poika minua jo houkutteli hukkaan.
             Ruusu ei oo kauniimpi, kun tyttÃ¶ Karjalassa;
          EikÃ¤ toista tarvita, kun olen maalimassa.
          MinÃ¤ tyttÃ¶, kaunis tyttÃ¶, kun mun pojat nÃ¤kee,
          Huoli sytty syÃ¤mehen, mieli naia tekee.
          Kaunis tyttÃ¶, siivo tyttÃ¶, siivon nimen kannan,
          EmpÃ¤ minÃ¤ joka pojan halatakaan anna.
          EnemmÃ¤n mÃ¤ mielestÃ¤ni itsestÃ¤ni tykkÃ¤Ã¤n,
          Poies, poies vierestÃ¤ni kehnot pojat lykkÃ¤Ã¤n.
          Synniksi sitÃ¤ sanotaan ja syntipÃ¤ se lienee,
          Nuoren tytÃ¶n ruvetella kehnon pojan viereen.
          Pian on se tyttÃ¶ raukka tullut kunnialta,
          Joka kerran heittÃ¤ytyi noien poikien valtaan.
          Pojilla ja susilla on yhtÃ¤lÃ¤inen mieli,
          Susi ei virka mitÃ¤nÃ¤, pojill' on liukas kieli.
          Poika puhuu kaunihisti likan sÃ¤ngyn eessÃ¤,
          Mesi, maito kielellÃ¤ ja myrkky syÃ¤messÃ¤.
          Niiin on poika piian luona kun halmehessa halla,
          Pian poika piika raukan hempeyen tallaa.
15. Poika ja tyttÃ¶.
    """
)



with open(input_file, "r", encoding="utf-8") as f:
   lines = f.readlines()

segments = ["".join(lines[i:i+50]).strip() for i in range(0, len(lines), 50)]


conversations = []
for i, segment in enumerate(segments):
    time.sleep(5)
    if segment.strip():  
        try:
            conversation = generate_conversation(segment.strip())
            conversations.append(conversation)
            print(f"Processed segment {i + 1}/{len(segments)}")
        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(conversations, out_f, ensure_ascii=False, indent=4)


with open(output_file, "w", encoding="utf-8") as out_f:
    json.dump(conversations, out_f, ensure_ascii=False, indent=4)



input_file = "soumalasia.txt"  
output_file = "soumalasia.json"  


def generate_conversation(segment):
    query_prompt = f"Luo kysymys seuraavan tekstin perusteella: {segment}"
    query_response = model.generate_content(f"Vastaa tekstin perusteella: {query_prompt}")
    human_query = query_response.text

    answer_prompt = f"{segment}\n\nKysymys: {human_query.strip()}\nVastaa lyhyesti. Warning: Only answer and only finnish language. The question must contain infromation that this is from Soumalasia poem"
    answer_response = model.generate_content(answer_prompt)
    gpt_answer = answer_response.text

    return {
        "conversations": [
            {
                "from": "human",
                "value": human_query.strip()
            },
            {
                "from": "gpt",
                "value": gpt_answer.strip()
            }
        ]
    }


generate_conversation(
    """ 
Kustantaja on senvuoksi arvellut tekevÃ¤nsÃ¤ suomalaiselle yleisÃ¶lle palveluksen uudestaan julkaistessaan tÃ¤mÃ¤n sananlaskukokoelman, joka tavataan mainitun kuukauslehden ensimmÃ¤isessÃ¤ ja toisessa vuosikerrassa (vv. 1836-37). Sen se ansainneekin. Se nÃ¤et sisÃ¤ltÃ¤Ã¤ sen mÃ¤Ã¤rÃ¤n Suomen kansan sananlaskuja â€” vielÃ¤pÃ¤ hyvin valikoidunkin â€” mikÃ¤ jokaisen Suomalaisen vÃ¤hintÃ¤nsÃ¤kin tarvitsee tuntea Suomalaisesta kÃ¤ydÃ¤kseen; siihen myÃ¶skin liittyy runsas joukko LÃ¶nnrotin sepitsemiÃ¤ selityksiÃ¤, yksi tahi useampia joka sananlaskua kohden, selityksiÃ¤, joiden arvo ei ole aivan vÃ¤hÃ¤ksi katsottava. Ne ne juuri valaisevan sisÃ¤llyksensÃ¤ ja tuoreen, kansanomaisen esitystapansa vuoksi tÃ¤lle pienelle kokoelmalle antavatkin varsinaisen merkityksensÃ¤. Kansallisen sivistyksen karttuessa yhÃ¤ laajaperÃ¤isemmÃ¤ksi, yhÃ¤ moninaisempia aineksia itseensÃ¤ sulatellessa on tarjona se vaara, ettÃ¤ uudet sukupolvet vÃ¤hitellen vierautuvat vanhan kansan katsantotavoista ja siten myÃ¶skin kadottavat sen mielipohjan, johon sanalaskurunoutemme perustuu. MitÃ¤ sÃ¤Ã¤tylÃ¤isluokkaan tulee, ei ole enÃ¤Ã¤ puhetta vaarasta, sillÃ¤ itse vahinkokin jo on tapahtunut, ainakin niissÃ¤ tÃ¤hÃ¤n luokkaan lukeutuvissa, jotka lapsipolvessaan eivÃ¤t ole suomalaisuuttaan imeneet suomalaisuuden oikeista Ã¤idinrinnoista suomalaisen rahvaan kotikeskuudessa. Jo puoli vuosisataa taaksepÃ¤in LÃ¶nnrot pilapuheiseen tapaansa huomauttaa mainitsemaamme epÃ¤kohtaa: mitenkÃ¤ kansanhengen Ã¤lyn tuotteet sivistyksen tasoittavasta vaikutuksesta syrjÃ¤ytymistÃ¤Ã¤n syrjÃ¤ytyvÃ¤t ja ainoastaan sivistymÃ¤ttÃ¶mien keskuudessa kenenkÃ¤Ã¤n kadehtimatta jonkunlaista arvoa sÃ¤ilyttÃ¤vÃ¤t. HÃ¤n lausuu muutamakseen:
"Kansankuntien sivistyksen laita on sama kuin yksityiselÃ¤jÃ¤inkin: mitÃ¤ yhtenÃ¤ aikana rakastetaan, se jo toisena on menettÃ¤nyt arvonsa. MikÃ¤ viidentoista vuotias poika enÃ¤Ã¤ panee suurta arvoa siihen puuhevoseen, joka hÃ¤nellÃ¤ oli apuna ratsastaessaan lÃ¤pi lapsuusikÃ¤nsÃ¤, ja mikÃ¤ tyttÃ¶ ei samanikÃ¤isenÃ¤ jo ole virkaheitoiksi hyljÃ¤nnyt entiset leikkisiskonsa, vauvat. MikÃ¤ arvo joululeikeillÃ¤ enÃ¤Ã¤ on sivistyneessÃ¤ seurueessa tahi mikÃ¤ merkitys on itse joulullakaan? â€” Arvoitustenkin aika on kadonnut kansoista, jotka itselleen vaativat suuremman sivistyksen nimeÃ¤. Ainakin on se mÃ¤Ã¤rÃ¤, mikÃ¤ niitÃ¤ siellÃ¤ tÃ¤Ã¤llÃ¤ enÃ¤Ã¤ on sÃ¤ilyssÃ¤, joutunut vÃ¤himmin sivistyneiden kansankerrosten kenenkÃ¤Ã¤n kadehtimattomaksi perintÃ¶osaksi. Ja mitÃ¤ sivistynyt osa kansakuntaa tekisikÃ¤Ã¤n noilla vanhuuttaan homehtuneilla rahvaan arvoituksilla? Onhan sillÃ¤ kyllÃ¤ uusia arvoituksia kuosien jokapÃ¤ivÃ¤isessÃ¤ vaihtelussa, valtioviisaissa vÃ¤ittelyissÃ¤, uskonnollisissa ja tieteellisissÃ¤ riidoissa ja tuhansissa muissa asioissa. Toden totta Salomo, jonka suuresta viisaudesta itsekukin on kuullut puhuttavan, hÃ¤n viisauksineen nykyaikana piankin pantaisiin pussiin. SillÃ¤ tÃ¤mÃ¤ viisaus ilmestyi enimmÃ¤kseen sananlaskuissa, arvoituksissa ja lauluissa, niinkuin I:sen Kuningasten kirjan 4:nen luvun 32 vÃ¤rsy nimenomaan sanoo, ettÃ¤ hÃ¤n puhui kolmetuhatta sananlaskua ja ettÃ¤ hÃ¤nen virsiÃ¤nsÃ¤ oli tuhannen ja viisi. YhtÃ¤ vaikeaksi kÃ¤visi epÃ¤ilemÃ¤ttÃ¤ useimpien muiden muinaisajan viisaiden, jos he kuolleista nousisivat meidÃ¤n pÃ¤ivinÃ¤mme saavuttaa mitÃ¤Ã¤n arvoa luullusta viisaudestaan. SillÃ¤ joskaan se ei kaikilla ollut etupÃ¤Ã¤ssÃ¤ sananlaskuina, arvoituksina ja lauluina, niin kuului siihen kuitenkin enimmÃ¤kseen semmoisia asioita, joita nykyaika halveksii. He puolestaan saattaisivat kuitenkin jonkun verran lohdutella itseÃ¤Ã¤n oppiessaan likemmin tuntemaan nykyajan viisautta, jonka he kaiketi pian oppisivatkin, sillÃ¤ 'pian on souttu soukka salmi, mitattu meri matala'. Voisipa tapahtua niin, ettÃ¤ he yhtyisivÃ¤t joihinkuihin semmoisiin nykyajan mahti viisaisin, jotka tuskin pystyvÃ¤t erottamaan haapaa koivusta ja kuusta petÃ¤jÃ¤stÃ¤. Semmoiselle nÃ¤ytettiin kerran hamppumaasta vasta nyhdettyÃ¤ hampunvartta ja pyydettiin tutkimaan esille tuotua kasvia. TÃ¤stÃ¤ ymmÃ¤lle jouduttuaan ja tiedon saatuaan, ettÃ¤ hÃ¤nellÃ¤ olikin hampunvarsi edessÃ¤Ã¤n, hÃ¤n tyynesti lausui: 'tuokaa minulle koko hamppumaa kaikkineen pÃ¤ivineen, niin otan siitÃ¤ kyllÃ¤ selon!' Mies oli kuitenkin muutamia viikkoja ennen suorittanut luonnontieteen tutkinnon. â€” Vastamainittu ei suinkaan ole sanottu nykyajan viisauden halventamiseksi tai entisen liialliseksi ylistÃ¤miseksi, vaan ainoastaan jollakin tapaa osottamaan, ettÃ¤ itse viisauskaan ei ole sama kaikkina aikoina, vaan sekÃ¤ muodoltaan ettÃ¤ sisÃ¤llykseltÃ¤Ã¤n vaihteleva." [Suomi 1841: Om Finska OrdsprÃ¥k och GÃ¥tor.] NÃ¤in on korkeampi sivistys edistyskulussaan tyÃ¶ntÃ¤nyt syrjÃ¤Ã¤n tai taaksensa jÃ¤ttÃ¤nyt alhaisemman ilmaumat.
Mutta varsinaisen rahvaankin parissa seuduilla semmoisilla, jotka ilmoisen ikÃ¤nsÃ¤ ovat olleet vieraskielisen sivistyksen jaloissa, nÃ¤mÃ¤ ikivanhat suomalaisen kansanhengen luomukset jo ovat vÃ¤hiin oloihin kadonneet ja katoavat pÃ¤ivÃ¤ pÃ¤ivÃ¤ltÃ¤ yhÃ¤ tarkemmin â€” toivottavasti eivÃ¤t kuitenkaan konsanansa sukupuuttoon. Kansallinen sivistys yhÃ¤ perusteellisemmaksi vaurastuessaan on palkitseva sen, minkÃ¤ pintapuolinen sivistys on hÃ¤vittÃ¤nyt. Kirjallisuuden tehtÃ¤vÃ¤ on niin ylhÃ¤isissÃ¤ kuin alhaisissakin elvyttÃ¤Ã¤ rakkautta oman kansan henkisen tuottelijaisuuden hedelmiin. Kirjallisuuden tehtÃ¤vÃ¤ on saattaa uudestaan kÃ¤ytÃ¤ntÃ¶Ã¶n muinaiset sananlaskutkin, joista suullinen ja kirjallinen esityskeino mehukkaammaksi ja virkeÃ¤mmÃ¤ksi voimistuu. Kirjallisuuden tehtÃ¤vÃ¤ on myÃ¶skin selittÃ¤Ã¤ nÃ¤iden sananlaskujen oikea merkitys ja kÃ¤ytÃ¤ntÃ¶. SiinÃ¤ tarkoituksessa tÃ¤mÃ¤kin kansankirjanen julkaistaan.
MitÃ¤ toimitustapaan tulee, mainittakoon, ettÃ¤ sananlaskut selityksineen, jotka alkuteoksessa tavataan eri kuukausnumeroiden jÃ¤lkeen liitettyinÃ¤ mitÃ¤Ã¤n erityistÃ¤ jÃ¤rjestystÃ¤ noudattamatta, nyt on asetettu tavalliseen aakkosjÃ¤rjestykseen. Oikokirjoitus on muutettu nykyaikaiseksi. Niin ikÃ¤Ã¤n on hellÃ¤varoin korjattu muutamia lauseopillisia omituisuuksia, joita LÃ¶nnrot alkavana suomenkielen kÃ¤yttÃ¤jÃ¤nÃ¤ nÃ¤htÃ¤vÃ¤sti latinankielen vaikutuksesta noudatteli, mutta jotka hÃ¤n varttuneempana sitten hylkÃ¤si. Samoin on yhtÃ¤ toista muutakin nykyiselle kielitajulle outoa luonnikkaammaksi muutettu. TÃ¤tÃ¤ menetystapaa puolustaa teoksen tarkoitus.
ToimitustyÃ¶n on suorittanut ylioppilas Eero JÃ¤Ã¤skÃ¶ (k. 1890), joka hehkeimmÃ¤llÃ¤ iÃ¤llÃ¤Ã¤n Manan majoille mennyt nuorukainen tÃ¤hÃ¤n tehtÃ¤vÃ¤Ã¤n uhrasi elÃ¤mÃ¤nsÃ¤ viimeiset hetket.
A. V. Fâ€”â€”n.
1. Ain' on onni saanehella, ei aina ansainnehella.
Lieneekin niin toisinaan, vaan epÃ¤ilemÃ¤ttÃ¤ kuitenkin ovat useimmat valitukset, onnen ei ansiota puoltavan, tyhjÃ¤kuntaisia. Harvoinpa ansiollisten miesten kuullaankaan valittavan, vaan ilman pitÃ¤vÃ¤t hyvÃ¤nÃ¤nsÃ¤, mitÃ¤ Jumala antaa, olipa se sitten meidÃ¤n mielestÃ¤ parempata tai pahempata.
2. Anna Jesus otravuotta, Jumala jyvÃ¤keseÃ¤, Herra heinÃ¤n kuivautta, saisi orjatkin olutta, kasakatkin kaljavettÃ¤, vierrettÃ¤ kivenvetÃ¤jÃ¤t.
Ainakin, ei yksin nÃ¤inÃ¤ viimeisinÃ¤ vuosina toivottava asia.
3. Anna kÃ¤ttÃ¤ kÃ¤yvÃ¤n miehen, suuta ulkovan urohon.
KÃ¤yvÃ¤llÃ¤ ja ulkovalta ymmÃ¤rretÃ¤Ã¤n toisinaan kÃ¶yhÃ¤Ã¤ ja koditonta, toisinaan muuten kulkevaista, tuntematonta. EdellisessÃ¤ tapauksessa kehottaa sananlasku kÃ¶yhiÃ¤kin ja varattomia hyvin kohtelemaan; jÃ¤lkimÃ¤isessÃ¤ varalla pitÃ¤mÃ¤Ã¤n, ettei tuntemattomiakaan tylysti vastaan oteta, koska siitÃ¤ vaan katumus saattaisi seurata.
4. Arat tyÃ¶ttÃ¶mÃ¤n kÃ¤tÃ¶set, rakko laiskan kÃ¤mmenessÃ¤.
TÃ¤llÃ¤ vertauksella kuvaillaan laiskojen ja tyÃ¶ttÃ¶mien eloa useammissa tiloissa, joissa tavataan kunnottomiksi.
5. Asialle mies kylÃ¤hÃ¤n, vaimo varten syÃ¶mistÃ¤nsÃ¤.
Vertaelee vaimoista ei ulkotoimituksiin olevan.
6. Aura atroja parempi, kirnunmÃ¤ntÃ¤ tarpoimia.
Atro eli atrain on raudasta tehty, moni vÃ¤kÃ¤inen, pitkÃ¤varrellinen kalaniskuase, jota toisissa paikoin arinaksi ja ahinkaaksikin sanotaan. TÃ¤mÃ¤n sananlaskun mukaan on siis maanviljelys ja karjanpito kalastamista etuisampi, jota mekÃ¤Ã¤n emme valeeksi vÃ¤itÃ¤, vaan toivomme monen, jonka pellot ja karja kalastamisen vuoksi laihtuvat, paremmin tÃ¤mÃ¤n asian varteen ottavan. Joka kalastamisessakin laskee yhteen kaikki, ensiksi tyÃ¶n ja aineet pyydyksiÃ¤ hankkiessa, sitte vaivat ja ajankulun pyytÃ¤essÃ¤ ja tÃ¤mÃ¤n tehtyÃ¤Ã¤n asettaa saaliinsa rinnakkain sen hyÃ¶dyn kanssa, minkÃ¤ hÃ¤n kaikilla nÃ¤illÃ¤ tÃ¶illÃ¤, vaivoilla ja kuluilla olisi voinut maallansa toimittaa, hÃ¤n helposti keksii kalanpyynnin ei konsa olevan maatyÃ¶hÃ¶n verrattavan. Mutta mikÃ¤ onkaan syynÃ¤, ettÃ¤ hÃ¤n ei kuitenkaan kalastamista heitÃ¤. Ei muu, kuin ettÃ¤ hÃ¤n kalanpyynnissÃ¤ nÃ¤kee vaivansa pikemmin ehkÃ¤ kehnostikin palkittavan, jota vastoin maanviljelys ja karjanhoito vasta myÃ¶hemmin, ehkÃ¤ monikertaisesti auttavat. Ainoastaan meren luodoilla ja rantamailla asuvilla, joilla ei ole tilaa maata viljellÃ¤ ja karjoja pitÃ¤Ã¤, on kalanpyynti luonnollinen tyÃ¶, ja heiltÃ¤ pitÃ¤isi maanmiehen enimmÃ¤n kalantarpeensa maanviljalla hankkia, niin olisi kummallekin toisen hyvyys hyÃ¶dyksi. Ja miksi ei sopisi hÃ¤nen itsensÃ¤kin, ehkÃ¤ vaan aikalomasta, kaloja jÃ¤rvistÃ¤nsÃ¤ pyytÃ¤Ã¤. Kun tÃ¤mÃ¤ vaan harvoin tapahtuisi, niin jÃ¤rvet kalaisempina asuisivatkin ja hÃ¤nellÃ¤ olisi siitÃ¤ vielÃ¤ sekin etu, ettÃ¤ silloin vÃ¤hemmÃ¤llÃ¤ tyÃ¶llÃ¤ pian yhtÃ¤lÃ¤isen saaliin saisi, kuin nyt paljolla vÃ¤ellÃ¤ melkein puhtaiksi kalatuista vesistÃ¤nsÃ¤. Monta muuta sananlaskua lÃ¶ytyy, jotka osottavat Suomen kansalla tÃ¤ssÃ¤ asiassa olevan hyvÃ¤n ymmÃ¤rryksen; niin esim. sananl. N:o 41, 54, 160.
7. Auta miestÃ¤ mÃ¤essÃ¤, nosta lasta kynnyksessÃ¤.
Apu paikassansa on hyvÃ¤ itsekullekin.
8. Ei huolta hÃ¤vinnehellÃ¤, tyÃ¶tÃ¤ maansa myÃ¶nehellÃ¤.
SillÃ¤ mielellÃ¤kÃ¶ vaan moni silmin nÃ¤hden hÃ¤vittÃ¤neekin maansa?
9. Ei lapset laista tiedÃ¤, vaimot vallan tuomioista.
Harvoinpa heihin laki ja tuomiot koskenevatkin.
10. Ei liikkuva vaihate liikkumattomaan.
"""
)



with open(input_file, "r", encoding="utf-8") as f:
   lines = f.readlines()

segments = ["".join(lines[i:i+50]).strip() for i in range(0, len(lines), 50)]


conversations = []
for i, segment in enumerate(segments):
    time.sleep(5)
    if segment.strip():  
        try:
            conversation = generate_conversation(segment.strip())
            conversations.append(conversation)
            print(f"Processed segment {i + 1}/{len(segments)}")
        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(conversations, out_f, ensure_ascii=False, indent=4)


with open(output_file, "w", encoding="utf-8") as out_f:
    json.dump(conversations, out_f, ensure_ascii=False, indent=4)



input_file = "eino-ena.txt"  
output_file = "eino-ena.json"  


def generate_conversation(segment):
    query_prompt = f"Luo kysymys seuraavan tekstin perusteella: {segment}"
    query_response = model.generate_content(f"Vastaa tekstin perusteella: {query_prompt}")
    human_query = query_response.text

    answer_prompt = f"{segment}\n\nKysymys: {human_query.strip()}\nVastaa lyhyesti. Warning: Only answer and only finnish language. The question must contain infromation that this is from EINO LEINO poem"
    answer_response = model.generate_content(answer_prompt)
    gpt_answer = answer_response.text

    return {
        "conversations": [
            {
                "from": "human",
                "value": human_query.strip()
            },
            {
                "from": "gpt",
                "value": gpt_answer.strip()
            }
        ]
    }


generate_conversation(
    """
Oi, kuulkatte, kuinka se sykkÃ¤ilee tÃ¤mÃ¤ maa ja sen musta multa! Oi, kuulkatte Ã¤Ã¤ntÃ¤, mi kuiskailee joka pellolta perkahilta! Se Ã¤Ã¤ni on suuri ja kaunis ja pyhÃ¤, se Ã¤Ã¤ni on kansamme kalleus yhÃ¤, se kutsuvi tyÃ¶hÃ¶n ja taistelohon ja kauvan jo kutsunut on.
Se on kansamme voima, mi kaikki voi, kun suru oli leipÃ¤nÃ¤ Suomen, se on kansamme henki, mi kaikki loi, kun luotihin Suomen huomen; se liikkuvi laineilla tuhanten vetten, se kaikuvi kielistÃ¤ kanteletten, se lehdossa helkkÃ¤Ã¤, se laaksossa soi, sitÃ¤ laps emon maidossa joi.
Se hetkeksi kohota pinnalta maan voi kylmiÃ¤ ilmoja pakoon, mut siellÃ¤ se voimia kokoo vaan ja siellÃ¤ se kasvaa ja sakoo, ja kun sen on aika, se Ukkona soipi ja sateena, tuulena tulla se voipi ja lakaista laaksot ja virrat ja maan. Koska, koska sen nÃ¤hdÃ¤ mÃ¤ saan?
Koska saan minÃ¤ nÃ¤hdÃ¤ Suomeni tÃ¤Ã¤n yheks, suureksi yhdistÃ¤yvÃ¤n? Koska uskonsa voimalla nuorten mÃ¤ nÃ¤Ã¤n ajan aaltoja astuen kÃ¤yvÃ¤n? Koska nÃ¤Ã¤n minÃ¤ nousevan kansani rinnan, lapset yhdessÃ¤ leikkivÃ¤n tÃ¶llin ja linnan, koska nÃ¤Ã¤n minÃ¤ saapuvan sankarin sen, josta unta mÃ¤ uinailen?
Kun tyhmyys ja raakuus raukaisi maan, niin hÃ¤ntÃ¤, hÃ¤ntÃ¤ mÃ¤ uotin, ja unelmat muut jos ne murtui vaan, niin hÃ¤neen, hÃ¤neen mÃ¤ luotin, olen uottanut hÃ¤ntÃ¤ mÃ¤ pÃ¤ivÃ¤Ã¤ ja yÃ¶tÃ¤ ja paljon niin ollut jo hÃ¤llÃ¤ ois tyÃ¶tÃ¤, mut hÃ¤ntÃ¤ mÃ¤ uskon ja uinailen ja lakkaa laulamast' en.
Mut viikot ne vierii ja vuodet ne kÃ¤y, yhÃ¤ kaikki on ennallansa. YhÃ¤ ei minun unteni urhoa nÃ¤y ja mieroa kerjÃ¤Ã¤ kansa. MissÃ¤ viivyt sÃ¤ mies? Vai oisko se unta, vai oisko se mennehen talvista lunta, kun uskon ma Suomeni suuruuteen ja sen kuntohon, kantelehen?
Ei, ei! Mulla on joku rinnassani, siell' on erÃ¤s Ã¤Ã¤ni, mi puhuu se uni ett'ei ole unta ain, vaan kerran se huutaa ja huhuu, minÃ¤ tunnen sen suonissa, lihassa ja luissa, minÃ¤ kuulen sen tuulessa, ilmassa, puissa, nÃ¤en silmissÃ¤ nuorten sen liekehtivÃ¤n: HÃ¤n on saapuva, sankari hÃ¤n!
HÃ¤nen nimens' on pÃ¤ivÃ¤ ja kansan koi ja hÃ¤n Suomemme suureksi nostaa; hÃ¤n maammonsa mahlat palkita voi ja taattonsa kohlut kostaa. HÃ¤n saapuva on kuni VÃ¤inÃ¤mÃ¶ uusi ja sen laulun on kuuleva koivu ja kuusi ja Vellamon neiet ja metsÃ¤ ja maa, yli aaltojen, aikojen taa.
MitÃ¤ meistÃ¤! Me kaikki soitamme vain kannelta katajaista. Se taide, joka ei hymyile, se ei ole taivahaista. HÃ¤n soittonsa koivusta soreasta vuolee, hÃ¤n riemuten elÃ¤Ã¤ ja riemuten kuolee, hÃ¤ntÃ¤ kantavi kÃ¤mmenin kansa ja maa, hÃ¤n kukkia antaa ja saa.
TÃ¤mÃ¤ aika on aikoja etsinnÃ¤n ja aikoja valmistuksen, mut suurena silloin kuin saapuvi hÃ¤n lyÃ¶ aika sen lupauksen, jonk' antoi meille jo taivahan Herra, kun tÃ¤nne hÃ¤n kansamme johdatti kerran lÃ¤pi myrskyjen, tundrojen, tuulien, tuhatjÃ¤rvien rannoillen.
Oi, kuulkatte, kuinka se sykkÃ¤ilee tÃ¤mÃ¤ maa ja sen musta multa! Oi, kuulkatte Ã¤Ã¤ntÃ¤, mi kuiskailee joka pellolta perkatulta! Oi, nÃ¤hkÃ¤Ã¤ sen kukkivat kummut ja saaret ja nÃ¤hkÃ¤Ã¤ sen aalloissa taivahan kaaret ja vierivÃ¤t virrat ja vehreÃ¤ maa â€” oi, nÃ¤hkÃ¤tte syntymÃ¤maa!
Se kansa, mi tÃ¤nne ohjattiin, se ei ole hukkahan luotu; se kansa luotu on suurempiin, kelle kerran on maa tÃ¤mÃ¤ suotu; se luotihin kansaksi kantelon, laulun, ja kansaksi kauneuden, taltan ja taulun, ja kansaksi nousevan taitehen sen, jonka kuulemme kuiskehen.
Vuossata on jÃ¤llehen vierÃ¤htÃ¤nyt ja meriltÃ¤ uusilta tuulee. KÃ¤y ajassa uusia aatteita nyt, ja kellÃ¤ on korvat, se kuulee, se kuulevi puissa mahlojen juoksun, se tuntevi ilmassa ihanan tuoksun, mi kevÃ¤ttÃ¤ kertoo ja ennustaa â€” oi, kuuntele syntymÃ¤maa!
Oi, kuuntele korvin avoimin ja aattele tÃ¤ysin aattein, oi, pukeu paitoihin puhtaihin ja valmistu juhlavaattein! Sun koittava, kansa, on sunnuntaisi â€” ja koittaa kyllÃ¤ se kohta jo saisi, olet tarpeheks kyynelin kylpenyt. Hymyhuulinen olkosi nyt!
Luo silmÃ¤si laajalti ympÃ¤ri maan, katso kauvaksi lÃ¤ntehen, itÃ¤Ã¤n! Katso, maa on vaiti ja odottaa, aika seisoo ja lippua pitÃ¤Ã¤: Ken tohtivi temmata vuossadan vaatteen? ken tohtivi nostaa nousevan aatteen ja korkeella kantaa ja lennÃ¤ttÃ¤Ã¤? Nous Suomeni! pystyhyn pÃ¤Ã¤!
Nouse Suomeni suurena, rynnistÃ¤in, nouse vaaroilta, vaarojen alta, nouse rannoilta jÃ¤rvien siintÃ¤vÃ¤in, sinÃ¤ sinisten toivojen valta, luo pÃ¤Ã¤ltÃ¤si pienten riitojen riehu ja kasva ja kansojen lippuna liehu ja nÃ¤ytÃ¤, mit' tÃ¤Ã¤llÃ¤ pienikin voi, kun suurta se unelmoi!
    """
)



with open(input_file, "r", encoding="utf-8") as f:
   lines = f.readlines()

segments = ["".join(lines[i:i+20]).strip() for i in range(0, len(lines), 20)]


conversations = []
for i, segment in enumerate(segments):
    time.sleep(5)
    if segment.strip():  
        try:
            conversation = generate_conversation(segment.strip())
            conversations.append(conversation)
            print(f"Processed segment {i + 1}/{len(segments)}")
        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(conversations, out_f, ensure_ascii=False, indent=4)


with open(output_file, "w", encoding="utf-8") as out_f:
    json.dump(conversations, out_f, ensure_ascii=False, indent=4)



input_file = "meins-in-mise.txt"  
output_file = "meins-in-mise.json"  


def generate_conversation(segment):
    query_prompt = f"Luo kysymys seuraavan tekstin perusteella: {segment}"
    query_response = model.generate_content(f"Vastaa tekstin perusteella: {query_prompt}")
    human_query = query_response.text

    answer_prompt = f"{segment}\n\nKysymys: {human_query.strip()}\nVastaa lyhyesti. Warning: Only answer and only finnish language. The question must contain infromation that this is from Meins in mise poem"
    answer_response = model.generate_content(answer_prompt)
    gpt_answer = answer_response.text

    return {
        "conversations": [
            {
                "from": "human",
                "value": human_query.strip()
            },
            {
                "from": "gpt",
                "value": gpt_answer.strip()
            }
        ]
    }


generate_conversation(
    """ 
Auringonlaskun aikana oli metsÃ¤ssÃ¤ hiljaista ja yksinÃ¤istÃ¤ mÃ¤ntyjen
ja kuusten tuoksuessa suloiselta ja koko metsÃ¤n loistaessa kullalta,
punaiselta ja viheriÃ¤ltÃ¤. Suurten puitten oksien alitse kulkevat miehet
nÃ¤yttivÃ¤t sulautuvan yhteen vÃ¤rien kanssa, ja sitten kuin he olivat
hÃ¤vinneet nÃ¤kyvistÃ¤, tuntuivat he muodostuneen tuon villin metsÃ¤maan
osaksi.

Valkoisten vuorten korkein huippu Old Baldy oli pyÃ¶reÃ¤ ja paljas
laskeutuvan auringon viimeisen hehkun kirkkaan kullan sitÃ¤
reunustaessa. Sitten kuin valaistus hÃ¤ipyi kupolimaisen huipun taakse,
tapahtui muutos, ja kylmÃ¤t ja tummenevat varjot levenivÃ¤t tuon
vuorimaailman mustille metsÃ¤isille rinteille.

Monien purojen kostuttamat villit ja synkÃ¤t korkeakasvuiset metsÃ¤t ja
ruohoiset puistot muodostavat tÃ¤mÃ¤n kymmenentuhatta jalkaa merenpintaa
korkeammalla olevan, joka puolelta EtelÃ¤-Arizonan erÃ¤maan eristÃ¤mÃ¤n
seudun -- tÃ¤mÃ¤n hirvien ja antilooppien, karhujen ja puumien, susien ja
kettujen rauhaisan kodin, joka samalla myÃ¶s saa toimia hurjien apachien
asuntona ja piilopaikkana.

Syyskuisin puhaltaa noissa seuduissa kova ja kylmÃ¤, vÃ¤hÃ¤n jÃ¤lkeen
auringonlaskun alkava yÃ¶tuuli. Se nÃ¤yttÃ¤Ã¤ tuovan hÃ¤mÃ¤rÃ¤n mukanaan
siipiensÃ¤ varassa samoin kuin kaikki sellaiset heikot Ã¤Ã¤netkin, joita
ei oltu voitu ennen vallinneessa hiljaisuudessa erottaa.
"""
)



with open(input_file, "r", encoding="utf-8") as f:
   lines = f.readlines()

segments = ["".join(lines[i:i+50]).strip() for i in range(0, len(lines), 50)]


conversations = []
for i, segment in enumerate(segments):
    time.sleep(5)
    if segment.strip():  
        try:
            conversation = generate_conversation(segment.strip())
            conversations.append(conversation)
            print(f"Processed segment {i + 1}/{len(segments)}")
        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(conversations, out_f, ensure_ascii=False, indent=4)


with open(output_file, "w", encoding="utf-8") as out_f:
    json.dump(conversations, out_f, ensure_ascii=False, indent=4)



input_file = "suomen.txt"  
output_file = "suomen.json"  


def generate_conversation(segment):
    query_prompt = f"Luo kysymys seuraavan tekstin perusteella: {segment}"
    query_response = model.generate_content(f"Vastaa tekstin perusteella: {query_prompt}")
    human_query = query_response.text

    answer_prompt = f"{segment}\n\nKysymys: {human_query.strip()}\nVastaa lyhyesti. Warning: Only answer and only finnish language. The question must contain infromation that this is from SUOMEN KANSAN SATUJA JA TARINOITA poem"
    answer_response = model.generate_content(answer_prompt)
    gpt_answer = answer_response.text

    return {
        "conversations": [
            {
                "from": "human",
                "value": human_query.strip()
            },
            {
                "from": "gpt",
                "value": gpt_answer.strip()
            }
        ]
    }


generate_conversation(
    """
Kuten yksityiset, siten kansatkin kerran ovat olleet lapsuuden iÃ¤'ssÃ¤. Kertomia ikivanhojen aikojen seikoista ja tapahuksista voisi sanoa kansan lapsuuden historiaksi. Uskontonsa puki kansa taruihin, keksintÃ¤nsÃ¤ sekÃ¤ muun tietonsa sananlaskuihin, ja mitÃ¤ nykyÃ¤nsÃ¤ sanoisimme historiaksi, oli siihen aikaan tarinoita. â€” Se oli luonnollisuuden ja lapsellisen vi'attomuuden aika. Tosin eivÃ¤t vuosisatojen kokemukset ohjanneet mielen juoksua silloin, vaan myÃ¶skÃ¤Ã¤n ei vielÃ¤ tyhjÃ¤ntÃ¤pÃ¶inen, useinkin itseensÃ¤ hÃ¤mmentynyt viisastelemus rau'aissut ihmisen henkeÃ¤. YltÃ¤nsÃ¤ nÃ¤ki ja haveksi kansa tÃ¤ssÃ¤ lapsuuden iÃ¤'ssÃ¤ ollessansa kummallisia, mielikuvallisia ihme-olennoita. Joka vuorella ja laaksolla oli omat elÃ¤jÃ¤nsÃ¤, ja kussakin jo'essa sekÃ¤ lÃ¤hteessÃ¤ joku jumalallinen olento. Haltioita ja henkiÃ¤, hiisiÃ¤ ja kummitoksia, peikkoja ja piruja oli maailma tÃ¤ynnÃ¤. Sanalla sanoen: kaikki luonnon vÃ¤likappaleet olivat kansan silmissÃ¤ elÃ¤viÃ¤.
Ristin oppi oli se maailmalle loistava valo, jonka sÃ¤teitÃ¤ eivÃ¤t vanhat tummentuneet luulot ja kuvannot kau'empaa sietÃ¤neet vaan pakenivat pakenemistansa yhÃ¤, haihtuen ihmisten mielestÃ¤. Ensi tyÃ¶ksensÃ¤ sai uusi oppi puhdistamaan ihmisen sydÃ¤ntÃ¤, kÃ¤vi siitÃ¤ uudistamaan hÃ¤nen henkeÃ¤nsÃ¤, ja on viimein pÃ¤Ã¤stÃ¤vÃ¤ kaiken maailman niistÃ¤kin pimenteistÃ¤, mitkÃ¤ vielÃ¤ tummentavat sitÃ¤. Vaan tÃ¤nÃ¤Ã¤nkin siltÃ¤ valistuksen alalta, jolla nykyÃ¤nsÃ¤ olemme, kuuntelemme mieluisasti esivanhempiemme muinoista oloa, heidÃ¤n tapojansa ja elÃ¤mÃ¤tÃ¤nsÃ¤ yleiseen. Varsinkin viehÃ¤ttÃ¤vÃ¤t meitÃ¤ heidÃ¤n ihanat, korkeamieliset runonsa ja suloluontoiset tarinansa, joissa vuosisatojen kuluttua omituisella, meille iÃ¤'tse rakkaalla kielellÃ¤nsÃ¤ vielÃ¤ haudoistansakin puhuttelevat meitÃ¤.
Ovat siis vanhojen runot ja tarinat kalliita meille muinoistiedon suhteen, vaan eivÃ¤t sentÃ¤hden ole ainoastansa kuolleita muinoisjÃ¤tteitÃ¤, vaan i'Ã¤n kaiken pysyvÃ¤isiÃ¤ muistopatsaita, joiden juurella kotimainen Runotar vielÃ¤kin valvoo, kuni kiitollinen lapsi Ã¤itinsÃ¤ haudalla. Ovat ikÃ¤Ã¤nkun sivistyksen kÃ¤tkyviÃ¤, esivanhempiemme lapsellisia unelmia tÃ¤ynnÃ¤, kauniita, kummastuttavia teoksia, joiden rakennuksessa kansan omituinen luonne ja henki vielÃ¤ osotaikse tÃ¤ydessÃ¤ puhtaudessansa ja alkuperÃ¤isessÃ¤ terveydesssÃ¤nsÃ¤. Tekisi mielemme sanoa runon ja tarinan, kuni kaksi sisÃ¤rystÃ¤; ikuisesta sÃ¤Ã¤nnÃ¶stÃ¤ asetetun taidetekoisen runoelman rinnalle, kuten lapsi tÃ¤ysiaikaisen miehen vierelle, kaikissa elon vaiheissa ja vimmoissa muistuttamaan hÃ¤ntÃ¤ oman lapsuutensa virheettÃ¶myydestÃ¤ ja puhtaasta kainoudesta. â€” Niin jos lienee, on vanhojen runoilla ja tarinoilla vielÃ¤ tÃ¤nÃ¤Ã¤nkin suuri arvonsa, ja tulevat vaikuttamaan kotimaisen runoelman koko vastaiseen luonteeseen, juurikun varoittavaiset Ã¤Ã¤net, jotka estÃ¤vÃ¤t sitÃ¤ omaa luonnon laatuansa heittÃ¤mÃ¤stÃ¤ osoittamalla sille oikean suuntansa, jota sen tulee seurata, poikkeamatta harhateille.
MitÃ¤ Suomalaisiin runoihin tulee, lienevÃ¤t joksensakin jo tarkkaan halki maan kerÃ¤tyt, ja meillÃ¤ on Kalevala ja Kanteletar, joissa ne sÃ¤ilyvÃ¤t Suomen kansalle ikuiseksi iloksi; vaan tarinat, joita Suomalaiset niin suuresti rakastavat ja keskinÃ¤isessÃ¤ elossansa huviksensa kertoelevat, ovat tÃ¤hÃ¤n asti olleet, josko ei ylenkatseessa niin kuitenkin kansan suusta kerÃ¤Ã¤mÃ¤ttÃ¤ ja siis yleisÃ¶lle melkein tuntemattomat. Juuri tÃ¤stÃ¤ seikasta puhuu jo 1836 vuoden MehilÃ¤isessÃ¤ muutamassa kohdin LÃ¶nnrot, sanoen moniaan siinÃ¤ kerrotun tarinan jÃ¤lkimaineessa: "Ilman on Suomalaisia tarinoita tÃ¤hÃ¤n asti ylen vÃ¤hÃ¤n ko'ottu. Taitaisivat kuitenki olla siitÃ¤ arvosta, ettÃ¤ ansaitsisivat tulla ko'otuiksi samalla huolella, kun moni muukin kansa on tarinoitaan jÃ¤lkimuistoon korjaellut. NiistÃ¤ vaan olisi kotvaksikin kerÃ¤tessÃ¤ tyÃ¶tÃ¤, sillÃ¤ halki maan muistellaan niitÃ¤ pi'an Ã¤Ã¤rettÃ¶mÃ¤sti ja useimmiten erilaatuisia itsekullakin paikalla. Aina siitÃ¤ ai'asta saati on kuitenkin kansan suussa elÃ¤viÃ¤ tarumia al'ettu suuremmassa arvossa pitÃ¤Ã¤, ja useammat Suomen kielen ja kirjallisuuden hyvÃ¤elijÃ¤t ovat niitÃ¤ viimeisinÃ¤ aikoina ahkeruudella kokoelleet." Paitsi LÃ¶nnrotin MehilÃ¤isessÃ¤ ilmaisemia kerÃ¤si juuri samalla aikaa Akatemian oppilas herra J. Fr. Kainonen ison joukon VenÃ¤jÃ¤n Karjalasta, ja sittemminkin on vuosittain aina toisia tarinoita Suomen eri maakunnista kerÃ¤tty. Niin ovat Suomalaisen Kirjallisuuden Seuran kustannuksella oppilaat D.E.D. Europaeus, A.E. Oksanen, Fr. PolÃ©n ja maisteri H.A. Reinholm kerÃ¤nneet niitÃ¤ Karjalasta, oppilaat A. Rothman, A.E. Nylander ja tÃ¤mÃ¤n tarinakokouksen sommittelija HÃ¤meestÃ¤. Samassa tarkkeessa on LÃ¤nsi-Suomalaisen oppilais-osakunnan avulla oppilas B.A. Paldani vaeltanut Satakunnassa ja oppilas O. Palander muutamien Viipurilaisen osakunnan jÃ¤senten toimesta HÃ¤meessÃ¤, joiden kumpienkin kokoelmat ovat Suomalaiselle Kirjallisuuden Seuralle lahjoitetut. VielÃ¤ sitte on pitÃ¤jÃ¤n mestari Olli Karjalainen vasta mainitulle Seuralle hyvÃ¤ntahtoisesti lÃ¤hettÃ¤nyt muutamia kiitoksen sietÃ¤viÃ¤ tarinoita, joita itse on kerÃ¤nnyt ja kirjoitellut kotitienooltansa LiperistÃ¤.
MitÃ¤ nyt aikoja myÃ¶ten useammalta kerÃ¤Ã¤jÃ¤ltÃ¤ nÃ¤in on ko'ottu, olemme Suomalaisen Kirjallisuuden Seuran tahdosta kokeneet kykymme mukaan suunnitella ja painoon toimitettavaksi korjata; ja tÃ¤ssÃ¤ lÃ¤htee nyt ensimÃ¤inen osa Suomalaisia Tarinoita, Suomalaisten lukijoiden hyvÃ¤Ã¤n mielisuosioon turvaten, ensi kertaa liikkeelle, omaa synnynmaatansa samoamaan; vaan jos nykyinen asunsa olisi ikÃ¤Ã¤nkuin halpa ja siistimÃ¤tÃ¶in, elkÃ¶Ã¶n kuka lii'oin pahastuko sitÃ¤, se ei ole matkaajan syy vaan sen, joka sen pukua tiehen laittaessa korjaeli.
HelsingistÃ¤ KesÃ¤kuun 15 pÃ¤ivÃ¤nÃ¤ 1852.
Eero Salmelainen.
SEPPO ILMARISEN KOSINTA

Seppo Ilmarinen, takoja ikuinen, oli pajassansa, rautaa pani ahjoon ja hiillutti. Tulipa nainen pajan kynnykselle, pieni nainen pikkarainen, suuri nainen suurukainen, ja virkkoi sepolle: "TietÃ¤isithÃ¤n, seppo Ilmarinen, minun sanomani, et pani rautaa ahjoon." Vastasi tuohon seppo Ilmarinen: "Pieni nainen pikkarainen, suuri nainen suurukainen! Sanonet hyvÃ¤t sanomat, minÃ¤ sinulle hyvÃ¤t lahjat annan; sanonet pahat sanomat, minÃ¤ sinulle hiilavan raudan kurkkuusi ajan." â€” "NÃ¤mÃ¤ minun sanomani", virkkoi nainen, "Hiihtoin kuninkaan tyttÃ¤reen, valkeaan vaalikkoon, kaunoiseen Katrinaan kahdet kosijat menivÃ¤t, venoilla soutivat."
Kuultuansa moiset sanomat seppo Ilmarinen raudan kohta otti ahjosta ja miettien mielessÃ¤nsÃ¤ lÃ¤ksi pajasta kotiinsa. KÃ¤vi Ã¤itinsÃ¤ puheelle ja sanoi: "Oi emoni, kantajani, pannos vaskinen kyly lÃ¤mpiÃ¤mÃ¤Ã¤n; hiilavammaksi lÃ¤mmitÃ¤ hiilavata rautaa, hiilavammaksi hiilavata kiveÃ¤!" Ã„iti siitÃ¤ lÃ¤mmittikin kylyn ja kÃ¤vi poikaansa kylpemÃ¤Ã¤n. Sanoipa taas seppo Ilmarinen: "Anna, emoni, kantajani, pellavainen paita pÃ¤Ã¤lleni, kapoiset kaatiot jalkaani!" Ã„iti silloin paidat, kaatiot tuopi pojallensa, ja seppo lÃ¤htee kylyyn. Kyllin siellÃ¤ kylvettyÃ¤nsÃ¤ astuu jo kiiruusti sieltÃ¤ kotiinsa vyÃ¶ttÃ¶millÃ¤ rungilla, kengÃ¤ttÃ¶millÃ¤ jaloilla, ja sanoo orjallensa: "Vanha orja uskollinen, valjastapa viljo valjo varsa kolmikesÃ¤inen kirjaviin korjiin, rautaisiin rahkeisiin, vaskisiin valjaisiin, terÃ¤ksisillÃ¤ ohjaksilla, tinaisilla rinnuksilla." Ottaa siitÃ¤ vanha orja uskollinen viljon valjon varsan kolmikesÃ¤isen ja rupeaa valjastamaan, vaan ei saa rinnusta riuhtaistuksi. Tuleepa silloin itse seppo Ilmarinen vyÃ¶ttÃ¶millÃ¤ rungilla, kengÃ¤ttÃ¶millÃ¤ jaloilla orjaansa auttamaan, riuhtaisee rinnuksen ja pistÃ¤Ã¤ varsan valjaisiin. SiitÃ¤ astuu sitten pirttiin, vaatteupi sukkelaan ja heittÃ¤Ã¤ Ã¤idillensÃ¤ jÃ¤Ã¤hyvÃ¤iset sanoen: "Oi emoni, kantajani, siunaos minua matkalleni, kosiin on nyt lÃ¤hteminen!"
Saatuansa emonsa siunaukset istuu jo seppo Ilmarinen kirjaviin korjiin, rautaisiin rahkeisiin, vaskisiin valjaisiin, terÃ¤ksisiin ohjaksiin viljolle valjolle varsalle kolmikesÃ¤iselle ja saapi sulaa merta ajaa surahuttamaan: ei kastu kavio eikÃ¤ tunnu korjan jÃ¤lki. Ajaa ajettelee, minkÃ¤ aikaa ajaneekin, niin jo tapaa ne kahdet venojen soutajat, jotka nainen hÃ¤nelle oli neuvonut, ja rupeaa yhteen matkueehen. Katsoopa meren takaa Hiihtoin kuninkaan tytÃ¤r, valkea vaalikko, kaunis Katrina kolmannesta kartanon kerrasta merelle, keksii siellÃ¤ matkaajat ja sanoo taatollensa: "Oi taattoseni, minuhun kolmet kosiomiehet tulevat, kaksi venoilla soutaa, kolmas korjalla ajaa." EipÃ¤ aikaakaan, niin pÃ¤Ã¤sevÃ¤t jo matkaajat perille ja tulevat Hiihtoin linnoille, jossa kuningas ottaa heidÃ¤t jalosti vastaan ja syÃ¶ttÃ¤Ã¤, juottaa kaikenmoisella hyvÃ¤sti. SyÃ¶tyÃ¤nsÃ¤ toimittavat miehet asiansa ja sanovat kumarrellen kuninkaalle: "Tulimme, kuninkaisemme, kosijiksi kaunoiseen Katrinaan." Kuningas siitÃ¤ mÃ¤Ã¤rÃ¤Ã¤ heille ansiotÃ¶itÃ¤ ja kysyy ensiksi: "Kuka teistÃ¤ voinee minulle kÃ¤Ã¤rmehisen pellon kyntÃ¤Ã¤ kengÃ¤ttÃ¶millÃ¤ jaloilla, paljahilla sorkilla, alastolla hipiÃ¤llÃ¤?" â€” "Ka, minÃ¤ kynnÃ¤n peltosi", vastasi seppo Ilmarinen rohkeasti; mutta toiset kaksi eivÃ¤t hirvenneet tyÃ¶hÃ¶n ruveta, vaan kumarsivat kuninkaalle ja menivÃ¤t tiehensÃ¤. Toisten lÃ¤hdettyÃ¤ seppo Ilmarinen kohta valjastaa viljon valjon varsansa aatraan ja saapi kÃ¤Ã¤rmehistÃ¤ peltoa kyntÃ¤mÃ¤Ã¤n. Kahden kyynÃ¤rÃ¤n korkeudella madot kuhisivat pellolla lentÃ¤en alituiseen aatrasta ja seposta pÃ¤Ã¤llitse, vaan eihÃ¤n yksikÃ¤Ã¤n sentÃ¤hden koskenut. Seppo sai tyÃ¶nsÃ¤ hyvÃ¤sti tehdyksi, meni rohkeasti kuninkaan eteen ja sanoi: "Nyt on, kuninkaiseni, kÃ¤Ã¤rmehinen peltosi kynnetty." â€” "HyvÃ¤!" sanoi kuningas, "vaan koska moisen tyÃ¶n toimeen sait, voinethan tanhuelleni lammin laulaa, siihen suuret kalat uimaan, pienet pirskamaan." â€” "KyllÃ¤ minÃ¤ senkin laadin", vastasi seppo Ilmarinen ja meni epÃ¤ilemÃ¤ttÃ¤ tanhuelle. SiinÃ¤ kun laulun lauloi vain, niin heti syntyikin lampi tanhuelle, siihen suuria kaloja uimaan, pieniÃ¤ pirskamaan. SiitÃ¤ pÃ¤Ã¤styÃ¤nsÃ¤ meni sitten kuninkaan eteen taas ja sanoi kumarrellen: "Nyt on teko tehty, mikÃ¤ mÃ¤Ã¤rÃ¤ttiinkin." Sanoi tuosta kuningas sepolle: "HyvÃ¤sti olet tÃ¤hÃ¤n saati tyÃ¶si toimittanut; menehÃ¤n nyt, tuo morsiamellesi, kaunoiselle Katrinalle, meren rannasta kotoinen lipas, joka on aikaa monta siellÃ¤ peitossa ollut."
MitÃ¤s siihen? Sepon tÃ¤ytyi lÃ¤hteÃ¤ kotoista lipasta etsimÃ¤Ã¤n, ja pÃ¤Ã¤tyi meren rannalle. SiinÃ¤ nÃ¤ki kolme nuorta neitoa rannan vietteellÃ¤ istuvan, rupesi haastattamaan heitÃ¤ ja kyseli: "Oi neitiseni, kussa on huomenlahjalipas kaunoisen Katrinan, tiedÃ¤ttekÃ¶?" â€” "Ukko Untamoisen vallassa on haettavasi", sanoivat neitiset, "tuossa pirttinsÃ¤ nÃ¤kyy, kÃ¤y kysymÃ¤ssÃ¤ hÃ¤neltÃ¤, rupeaisiko hÃ¤n antamaan, vaan ole kaikin mokomin varoillasi, Ã¤ijÃ¤ on sikÃ¤li mennyttÃ¤, vÃ¤hÃ¤n tullutta." Seppo siitÃ¤ menikin Untamoisen pirtille, kuten neuvottiin, ja katseli ikkunoista sisÃ¤lle. SiellÃ¤ ukko Untamoinen makaa ympÃ¤ri pirtin punalluksissa, jalat ja pÃ¤Ã¤ uksessa. Seppo silloin hiipien menee ukselle ja harpastaa siitÃ¤ suorastansa keskipirttiin sanoen: "Annas, ukko Untamoinen, kaunoisen Katrinan huomenlahjalipas!" Vastaili siihen ukko Untamoinen: "Voinet kielellÃ¤ni pysyÃ¤, siinÃ¤ hyppiÃ¤, tanssia, Ã¤sken annan huomenlahjalippaan." Seppo silloin ei arvellut, vaan laskeusi Untamoisen kielelle ja alkoi siinÃ¤ hyppiÃ¤; mutta ukko Untamoinen samassa avasi leukapielensÃ¤, ettÃ¤ puolentoista kyynÃ¤rÃ¤Ã¤ oli suu leveyttÃ¤nsÃ¤, hampaat kyynÃ¤rÃ¤n pituuttansa, ja seppo Ilmarisen lainasi purentelematta vatsaansa. TÃ¤mÃ¤pÃ¤ ei siitÃ¤ vielÃ¤ hÃ¤tÃ¤ytynyt, vaan heitti vaatteet pÃ¤Ã¤ltÃ¤nsÃ¤: paidastansa laati pajan, kaatioistansa palkeet, vasemman polven pani alasimeksi, vasemman kÃ¤den pihdiksi, oikean kÃ¤den paljaksi, ja rupesi Untamoisen vatsassa takoa taputtelemaan. Paidastansa otti vaskisen soljen ja takoi siitÃ¤ linnun, jolle laati rautaiset kynnet ja terÃ¤ksisen nokan. Sen kun sai valmiiksi, laulun lauloi vain, niin hengen pani sydÃ¤meen linnulle ja tyÃ¶nsi sen Untamoisen vatsassa lentÃ¤Ã¤ repakoimaan. Lintupa kun pÃ¤Ã¤si siellÃ¤ lentelemÃ¤Ã¤n, rautaisilla kynsillÃ¤nsÃ¤ katkoi vatsassa suonet kaikki ja kylkeen teki terÃ¤ksisellÃ¤ nokallansa suuren loukon, josta tuli ukko Untamoiselle semmoinen tuska, ettÃ¤ hÃ¤dissÃ¤nsÃ¤ huusi sepolle: "LÃ¤htenet, seppo Ilmarinen, vatsaani syÃ¶mÃ¤stÃ¤, niin saat kaunoisen Katrinan huomenlahjalippaan. MenehÃ¤n meren rantaan; kussa nÃ¤et kolme neitoa rannalla istuvan, siinÃ¤ on lipas hiekkaan peitettynÃ¤."
Sen kun kuuli seppo Ilmarinen, pujottelihe linnun kaivamasta kolosta Untamoisen vatsasta ulos ja harpasti uksesta pihalle lÃ¤htien heti meren rantaa astumaan. SiellÃ¤ nÃ¤ki ne kolme neitoa, mitkÃ¤ jo ennenkin, ja sanoi heille: "Oi hyvÃ¤t neitiseni, antakaa kaunoisen Katrinan huomenlahjalipas, ukko Untamoinen sen jo minulle lupasi!" â€” "Ota, tuossa on hiekassa lipas, â€” nosta, kanna", virkkoivat neitiset ja neuvoivat sepolle, kuhun oli lipas peitettynÃ¤. HÃ¤n silloin lippaan kaivoi hiekasta, kantoi sen kuninkaalle ja sanoi: "TÃ¤ssÃ¤ on nyt kaunoiselle Katrinalle huomenlahjalipas, jota panit etsimÃ¤Ã¤n!" Tyytyi jo kuningas sepon tekoihin, kun kotoisen lippaan sai Untamoiselta lunastetuksi; tyttÃ¤rensÃ¤, valkean vaalikon, kaunoisen Katrinan antoi hÃ¤nelle naiseksi ja siunasi heitÃ¤ matkalle.
Tuosta istui jo seppo Ilmarinen naisensa kera kirjaviin korjiin, viljolle valjolle varsalle kolmikesÃ¤iselle, rautaisiin rahkeisiin, vaskisiin valjaisiin, terÃ¤ksisiin ohjaksiin, tinaisiin rinnuksiin ja lÃ¤ksi sulaa merta ajaa surahuttamaan: ei kastu kavio eikÃ¤ tunnu korjan jÃ¤lki. Ajoi, ajoi, niin jo yÃ¶ saavutti merellÃ¤. Seppo siitÃ¤ lauloi laulun, niin samassa syntyi saari keskimerelle, kuhun laskihe naisensa kera makaamaan. LevÃ¤ttiin siinÃ¤ sen yÃ¶tÃ¤ aamuun saati, niin seppo Ilmarinen jo herÃ¤si unestansa ja katsahti kupeellensa, vaan eipÃ¤ naista enÃ¤Ã¤ nÃ¤hnytkÃ¤Ã¤n. Nousi silloin vuoteeltansa, lÃ¤ksi saaren rantaa astumaan ja luki sotkat kaikki saaren ympÃ¤rillÃ¤. Tulipa yksi sotka liikaa heti. Sen kun nÃ¤ki, lauloi seppo laulun kohta ja sanoi: "ElÃ¤s peittÃ¤ydy, Katrina, tÃ¤ssÃ¤ olet!" ja samassa syntyi sotkasta nainen jÃ¤rillensÃ¤. LÃ¤hdettiin siitÃ¤ taas sulaa merta ajaa surahuttamaan ja kuljettiin, minkÃ¤ aikaa lienee kuljettukin, niin jo taas yÃ¶ saavutti matkalla. Lauloi silloin seppo Ilmarinen laulun, niin saari syntyi merelle, ja laskeusivat siihen lepÃ¤Ã¤mÃ¤Ã¤n. Kului se yÃ¶ sitten, ja aamu tuli, niin herÃ¤si seppo makaamasta ja katsahti viereensÃ¤, vaan ei naista enÃ¤Ã¤ ollutkaan. Nousi tuosta kiiruusti vuoteeltansa ja kaikki puut saaressa luki, niin yksi puu liikaa tuli. Sille lauloi hÃ¤n laulun ja sanoi: "ElÃ¤s peittÃ¤ydy, kaunis Katrina, tÃ¤ssÃ¤ olet!" ja tuossa paikassa syntyi nainen jÃ¤rillensÃ¤. Istui seppo Ilmarinen siitÃ¤ taas naisensa kera kirjaviin korjiin, viljolle valjolle varsalle kolmikesÃ¤iselle, ja saatiin sulaa merta ajaa surahuttelemaan. Kuljettiinhan sen pÃ¤ivÃ¤Ã¤, kunne yÃ¶ saavutti, niin lauloi seppo Ilmarinen laulun kuten ennenkin, ja merelle laatiutui saari moinen, johon laskihe naisensa viereen makaamaan. Kuluipa yÃ¶, ja pÃ¤ivÃ¤ rupesi valkeamaan, niin havaitsi seppokin unestansa ja katsahti kainaloonsa, vaan eipÃ¤ naista siinÃ¤ ollutkaan. Tuosta suuttui jo seppo Ilmarinen naiseensa, kavahti vuoteelta ja sai saaren rantaa kiertÃ¤mÃ¤Ã¤n. SiinÃ¤ kÃ¤vellessÃ¤nsÃ¤ kun kivet luki saaren ympÃ¤rillÃ¤, niin yksi kivi taas liikaa oli. "ElÃ¤ peittÃ¤ydy, Katrina, tÃ¤ssÃ¤ olet!" sanoi hÃ¤n heti, ja kun laulun lauloi vain, syntyi nainenkin ennallensa. SiitÃ¤ puhui seppo vihoissansa: "MinÃ¤ sinun tauttasi, kaunis Katrina, Ã¤ijÃ¤n tyÃ¶tÃ¤ tein, Ã¤ijÃ¤n huolta nÃ¤in, ja sinÃ¤ yhÃ¤ minua pettelet; niinpÃ¤ menekin iÃ¤ksi pÃ¤ivÃ¤ksi merelle asumaan!" Sen kun sai seppo sanoneeksi, laulun lauloi heti ja naisensa, valkean vaalikon, kaunoisen Katrinan kirosi kajavaksi merellÃ¤ vastatuuleen iÃ¤tse lentelemÃ¤Ã¤n.
Mutta kÃ¤vipÃ¤ ikÃ¤vÃ¤ksi naisettakin elÃ¤minen, ja seppo vaskesta rupeaa itsellensÃ¤ naista laatimaan. Laulun lauloi ensimmÃ¤isen, niin jo syntyi ihminen; siitÃ¤ lauloi jo toisen, niin henki tuli sydÃ¤meen naiselle. Sen omatekemÃ¤n naisensa viereen rupeaa sitten makaamaan; toisen kÃ¤tensÃ¤ panee naiselle poveen, toisen pistÃ¤Ã¤ omaan poveensa. SiitÃ¤ kun herÃ¤Ã¤ aamusella ja koettelee kÃ¤siÃ¤nsÃ¤, niin kumpi itsellÃ¤nsÃ¤ povessa, se lÃ¤mmyt, vaan kumpi naisen povessa, se viluinen. Pakisi tuosta seppo Ilmarinen, takoja ikuinen, noin ikÃ¤Ã¤n: "KenkÃ¤Ã¤n elÃ¤ laatimaan rupea naista, ota valmis laadittu!" Lauloi sitten laulun toisen, niin muuttui kajava naiseksi jÃ¤llensÃ¤, valkeaksi vaalikoksi, kaunoiseksi Katrinaksi, kuten luonnostansa olikin. Sen kera istui sitten viljolle valjolle varsalle rekeen ja ajaa kavahutti kotiinsa, jossa Ã¤iti hyvÃ¤sti vastasi miniÃ¤nsÃ¤.
LIPPO JA TAPIO
"""
)



with open(input_file, "r", encoding="utf-8") as f:
   lines = f.readlines()

segments = ["".join(lines[i:i+30]).strip() for i in range(0, len(lines), 30)]


conversations = []
for i, segment in enumerate(segments):
    time.sleep(5)
    if segment.strip():  
        try:
            conversation = generate_conversation(segment.strip())
            conversations.append(conversation)
            print(f"Processed segment {i + 1}/{len(segments)}")
        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(conversations, out_f, ensure_ascii=False, indent=4)


with open(output_file, "w", encoding="utf-8") as out_f:
    json.dump(conversations, out_f, ensure_ascii=False, indent=4)



input_file = "seitseman.txt"  
output_file = "setiseman.json"  


generate_conversation(
    """ 
JUHANI. SeitsemÃ¤n metsonpoikaa!
UKKO. Olkoon heitÃ¤ kuinka monta hyvÃ¤Ã¤nsÃ¤; tuolta he katsella tÃ¶llÃ¶ttÃ¤vÃ¤t koivun-oksilta. Tuossa nyt mÃ¤llistelee vastaan yksi kuin sonni kohden uutta porttia, ja vasta hÃ¤n pÃ¶llÃ¤htÃ¤Ã¤ kun paukahtaa, mutta silloin on hÃ¤n pussissa. Samoin nytkin mÃ¤llistelee tÃ¤ssÃ¤ seitsemÃ¤n kÃ¶nttiÃ¤ kohden Kolistimen vaaria juuri niinkuin seitsemÃ¤n kÃ¶nisilmÃ¤istÃ¤ metsonpoikaa. KÃ¶ntit! MitÃ¤, mitÃ¤, mitÃ¤ minusta tahdotte?
JUHANI. Tahdon sanoa oikein vakaalla mielellÃ¤ ja kielellÃ¤, etten ole mikÃ¤Ã¤n varas enkÃ¤ metsonpoikanen enkÃ¤ kÃ¶ntti, ja sanon vielÃ¤ yksin tein, ettÃ¤ erÃ¤s vanha karru, erÃ¤s fÃ¶rpiiskatun ukko, joka ei seiso minusta juuri kaukana, ei montakaan virstaa tÃ¤llÃ¤ santaisella maantiellÃ¤, ettÃ¤ tÃ¤mÃ¤ mies, tÃ¤mÃ¤ hÃ¤peemÃ¤tÃ¶n karru on suuri lurjus ja hunsvotti; ja olkoon se sanottu kaikella kunnioituksella.
UKKO. Kuka mies, kuka mies, sinÃ¤ tÃ¶pÃ¶ kÃ¤enpoika kuivan hongan nenÃ¤ssÃ¤? Ole, ole, ole, ole, olenko minÃ¤ hunsvotti edessÃ¤s? Sanoppas. Kuka mies, sinÃ¤ kÃ¤enpoika?
JUHANI. MitÃ¤ peijakasta puhaltaisin hÃ¤nen kirottuun korvaansa?
AAPO. Ã„lÃ¤ enÃ¤Ã¤n mitÃ¤Ã¤n puhalla, vaan lÃ¤htekÃ¤Ã¤mme.
JUHANI. Ei juuri vielÃ¤; sillÃ¤ hÃ¤n on suurikelmi ukko. MitÃ¤ peevelin puskua puhaltaisin hÃ¤nen korvaansa?
EERO. Annas minÃ¤ koetan. Mutta pidÃ¤ sinÃ¤ tuota sonnipulkkia.
JUHANI. Niin, puhallappas sinne yksi mojova sana.
UKKO. Kuka mies? HÃ¤h?
EERO. Â»Kukakhaar!Â» sanoi pieni kÃ¤enpoika kuivan hongan nokassa.
Kukakhaar!
UKKO. Tuossa on kÃ¤ki!
EERO. SinÃ¤ riivattu!
JUHANI. Kas tuota perhanaa! Paukahtipa!
EERO. Paukahti, ja korva lukkoon.
AAPO. Oikein tehty, sinÃ¤ Kolistimen kÃ¶rri, oikein!
EERO. Hiiteen Ã¤ijÃ¤! Sivalsi ettÃ¤ kipenÃ¶itsee.
JUHANI. Ukko, ukko! huomaas mitÃ¤ teit: tempasit nyrkillÃ¤si poskelle kunniallista miestÃ¤ vallan maantien pÃ¤Ã¤llÃ¤ ja pyhÃ¤nÃ¤ sapattina. Ai, ai, ukko!
AAPO. Oikein tehty, sinÃ¤ Kolistimen riihitonttu, oikein!
UKKO. MitÃ¤ lÃ¶rpÃ¶ttelet sinÃ¤ siellÃ¤?
EERO. Oikein sanottu, sinÃ¤ Kolistimen nurkkajulli, oikein!
UKKO. Suus kiinni sinÃ¤kin, kÃ¤rppÃ¤. MinÃ¤, minÃ¤ opetan poikia nenÃ¤lleni loiskeilemaan. SillÃ¤ Kolistimen vaari ei siinÃ¤ juuri kauankaan siekaile ennen kun hÃ¤n iskee.
JUHANI. MinÃ¤ hÃ¤ntÃ¤ isken tuohon takkuiseen kaulukseen ja kiskon Ã¤ijÃ¤n ilman armoa olutkestiin. Heisaa, ukko! Nyt marssimme!
UKKO. Helvettiin sinÃ¤!
JUHANI. Olutta juomaan ettÃ¤ mahas repee!
UKKO. HellitÃ¤ kaulukseni, saatpa muutoin vasten klanias. EtkÃ¶ sinÃ¤, perkeleen juuti, hellitÃ¤?
JUHANI. Ã„mpÃ¤ri olutta!
TUOMAS. MitÃ¤ hulluutta, Juho, taas?
AAPO. Olkoon ukko oloillansa.
JUHANI. Herra varjele! hÃ¤n on meitÃ¤ haukkunut kuin koira. MitÃ¤ hÃ¤nelle tekisimme? HÃ¤n on tuommoinen vanha Ã¤ijÃ¤reppu. Mutta tulkoon hÃ¤n Jukolaan riemujuhlaan juomaan olutta vihoissansa. Niin, ukko, minun sydÃ¤meni ei anna perÃ¤Ã¤n, ei!
UKKO. HellitÃ¤ kyntes!
"""
)



with open(input_file, "r", encoding="utf-8") as f:
   lines = f.readlines()

segments = ["".join(lines[i:i+50]).strip() for i in range(0, len(lines), 50)]


conversations = []
for i, segment in enumerate(segments):
    time.sleep(5)
    if segment.strip():  
        try:
            conversation = generate_conversation(segment.strip())
            conversations.append(conversation)
            print(f"Processed segment {i + 1}/{len(segments)}")
        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(conversations, out_f, ensure_ascii=False, indent=4)


with open(output_file, "w", encoding="utf-8") as out_f:
    json.dump(conversations, out_f, ensure_ascii=False, indent=4)



!pip3 install wikipedia


URLS = [
    "https://en.wikipedia.org/wiki/Culture_of_Finland",
    "https://en.wikipedia.org/wiki/History_of_Finland",
    "https://en.wikipedia.org/wiki/Finnish_language",
    "https://en.wikipedia.org/wiki/Music_of_Finland",
    "https://en.wikipedia.org/wiki/Kalevala",
    "https://en.wikipedia.org/wiki/Finlandization",
    "https://en.wikipedia.org/wiki/Finland_in_World_War_II",
]


import wikipedia

wikipedia.set_lang("en")

for url in URLS:
    page_content = wikipedia.page(url.split("/")[-1]).content
    with open("wikipedia.txt", "a") as file:
        file.write(page_content)
  


input_file = "wikipedia.txt"  
output_file = "wikipedia.json"  


def generate_conversation(segment):
    query_prompt = f"Luo kysymys seuraavan tekstin perusteella: {segment}"
    query_response = model.generate_content(f"Vastaa tekstin perusteella: {query_prompt}")
    human_query = query_response.text

    answer_prompt = f"{segment}\n\nKysymys: {human_query.strip()}\nVastaa lyhyesti.  Warning: Only answer and only finnish language, be diversive in your answers."
    answer_response = model.generate_content(answer_prompt)
    gpt_answer = answer_response.text

    return {
        "conversations": [
            {
                "from": "human",
                "value": human_query.strip()
            },
            {
                "from": "gpt",
                "value": gpt_answer.strip()
            }
        ]
    }


generate_conversation(
    """ 
== Historical overview ==

The Scandinavian ice sheet covered most of northern Europe. Following its recession around 8000 BC, people began arriving in what is today Finland, with a majority presumably traveling from the south and east. Recent archaeological finds also reveal the presence of the north-western Komsa culture in northern Finland to be as old as the earliest discoveries on the Norwegian coast.
What is today Finland belonged to the northeastern Kunda culture until around 5000 BC and the Comb Ceramic culture from about 4200â€“2000 BC. The Kiukainen culture appeared on the southwestern coast of Finland around 1200 BC.
From 1100 to 1200, the crown of Sweden started to incorporate Finland. However, Novgorod also attempted to gain control. Several wars occurred between 1400 and 1700 where Finland fought against Sweden, Novgorod, the Grand Duchy of Moscow, and imperial Russia. In 1721, the Nystad Peace Treaty was signed, ending Swedish dominance in the Baltic region. In 1809, Finland was annexed by Russia. However from 1809 to 1917, Finland became an autonomous Grand Duchy with the Russian Czar as the constitutional monarch. In southeastern Finland, the region of Karelia, where most of the Russo-Swedish conflicts occurred, was influenced by both cultures while remaining peripheral to both epicentres of power. The verses in Finland's national epic, the Kalevala, originate mainly from Karelia and Ingria.
The 19th century brought a feeling of national Romanticism and Nationalism throughout Europe. Finland's nationalism also grew, forming cultural identity and making control of the land a priority. Expression of Finnish identity by the University docent, A. I. Arwidsson (1791â€“1858), became an often quoted Fennoman credo: "Swedes we are not, Russians we do not want to become, let us, therefore, be Finns." Nationalism heightened and resulted in a declaration of full independence from Russia on 6 December 1917, Finnish Independence Day. Notably, nationalists did not consider the Swedish-speakers members of a different (Swedish) nation; in fact, many Fennomans came from Swedish-speaking families.


"""
)


with open("wikipedia.txt", "r") as file:
    text = file.read()



import re 

sections = re.split(r'(==+.*?==+)', text)

segments = []
for i in range(1, len(sections), 2):
    header = sections[i].strip()
    content = sections[i + 1].strip() if i + 1 < len(sections) else ""
    segments.append({"header": header, "content": content})


segments[0]["content"]


conversations = []
for i, segment in enumerate(segments):
    time.sleep(5)
    if segment["content"].strip():  
        try:
            conversation = generate_conversation(segment["header"] + "\n" + segment["content"].strip())
            conversations.append(conversation)
            print(f"Processed segment {i + 1}/{len(segments)}")
        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(conversations, out_f, ensure_ascii=False, indent=4)


with open(output_file, "w", encoding="utf-8") as out_f:
    json.dump(conversations, out_f, ensure_ascii=False, indent=4)


file1 = 'merged.json'
file2 = 'kalevala.json'
file3 = 'setiseman.json'
file4 = "soumalasia.json"
file5 = "wikipedia.json"
file6 = "kanteletar.json"
file7 = "meins-in-mise.json"
file8 = "eino-ena.json"
file9 = "suomen.json"

with open(file1, 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open(file2, 'r', encoding='utf-8') as f:
    data2 = json.load(f)

with open(file3, 'r', encoding='utf-8') as f:
    data3 = json.load(f)

with open(file4, 'r', encoding='utf-8') as f:
    data4 = json.load(f)


with open(file5, 'r', encoding='utf-8') as f:
    data5 = json.load(f)


with open(file6, 'r', encoding='utf-8') as f:
    data6 = json.load(f)

with open(file7, 'r', encoding='utf-8') as f:
    data7 = json.load(f)


with open(file8, 'r', encoding='utf-8') as f:
    data8 = json.load(f)


with open(file9, 'r', encoding='utf-8') as f:
    data9 = json.load(f)

merged_data = data1 + data2 + data3 + data4 + data5 + data6 + data7 + data8 + data9


output_file = 'merged_final.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)


tune run lora_finetune_single_device --config config.yaml                                                 


!CUDA_VISIBLE_DEVICES=3  tune run lora_finetune_single_device --config config.yaml                                                 


def generate_triples(segment):
    query_prompt = f"Luo kysymys seuraavan tekstin perusteella: {segment}"
    query_response = model.generate_content(f"Vastaa tekstin perusteella: {query_prompt}")
    human_query = query_response.text.strip()
    
    chosen_prompt = f"{segment}\n\nKysymys: {human_query}\nVastaa lyhyesti. Vain vastaus ja vain suomen kielellÃ¤."
    chosen_response = model.generate_content(chosen_prompt)
    chosen_answer = chosen_response.text.strip()
    
    rejected_prompt = f"{segment}\n\nKysymys: {human_query}\nVastaa huonosti. Anna huono, virheellinen tai merkityksetÃ¶n vastaus suomeksi."
    rejected_response = model.generate_content(rejected_prompt)
    rejected_answer = rejected_response.text.strip()
    
    # Step 4: Return the results
    return {
        "question": human_query,
        "chosen": chosen_answer,
        "rejected": rejected_answer
    }



generate_triples(
"""
The culture of Finland combines indigenous heritage, as represented for example by the country's national languages Finnish (a Uralic language) and Swedish (a Germanic language), and the sauna, with common Nordic and European cultural aspects. Because of its history and geographic location, Finland has been influenced by the adjacent areas, various Finnic and Baltic peoples as well as the former dominant powers of Sweden and Russia. Finnish culture is built upon the relatively ascetic environmental realities, traditional livelihoods, and heritage of egalitarianism (e.g. Everyman's right, universal suffrage) and the traditionally widespread ideal of self-sufficiency (e.g. predominantly rural lifestyles and modern summer cottages).
There are cultural differences among the various regions of Finland, especially minor differences in dialect. Minorities, some of which have a status recognised by the state, such as the Sami, Swedish-speaking Finns, Karelians, Romani, Jews, and Tatars, maintain their cultural identities within Finland. Many Finns are emotionally connected to the countryside and nature, as large-scale urbanisation is a relatively recent phenomenon.

== Historical overview ==

The Scandinavian ice sheet covered most of northern Europe. Following its recession around 8000 BC, people began arriving in what is today Finland, with a majority presumably traveling from the south and east. Recent archaeological finds also reveal the presence of the north-western Komsa culture in northern Finland to be as old as the earliest discoveries on the Norwegian coast.
What is today Finland belonged to the northeastern Kunda culture until around 5000 BC and the Comb Ceramic culture from about 4200â€“2000 BC. The Kiukainen culture appeared on the southwestern coast of Finland around 1200 BC.
From 1100 to 1200, the crown of Sweden started to incorporate Finland. However, Novgorod also attempted to gain control. Several wars occurred between 1400 and 1700 where Finland fought against Sweden, Novgorod, the Grand Duchy of Moscow, and imperial Russia. In 1721, the Nystad Peace Treaty was signed, ending Swedish dominance in the Baltic region. In 1809, Finland was annexed by Russia. However from 1809 to 1917, Finland became an autonomous Grand Duchy with the Russian Czar as the constitutional monarch. In southeastern Finland, the region of Karelia, where most of the Russo-Swedish conflicts occurred, was influenced by both cultures while remaining peripheral to both epicentres of power. The verses in Finland's national epic, the Kalevala, originate mainly from Karelia and Ingria.
The 19th century brought a feeling of national Romanticism and Nationalism throughout Europe. Finland's nationalism also grew, forming cultural identity and making control of the land a priority. Expression of Finnish identity by the University docent, A. I. Arwidsson (1791â€“1858), became an often quoted Fennoman credo: "Swedes we are not, Russians we do not want to become, let us, therefore, be Finns." Nationalism heightened and resulted in a declaration of full independence from Russia on 6 December 1917, Finnish Independence Day. Notably, nationalists did not consider the Swedish-speakers members of a different (Swedish) nation; in fact, many Fennomans came from Swedish-speaking families.
"""
)


output_file = "dpo_data2.json"


conversations = []
for i, segment in enumerate(segments[62:]):
    time.sleep(5)
    if segment["content"].strip():  
        try:
            conversation = generate_triples(segment)
            conversations.append(conversation)
            print(f"Processed segment {i + 1}/{len(segments)}")
        except Exception as e:
            print(f"Error processing segment {i + 1}: {e}")
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(conversations, out_f, ensure_ascii=False, indent=4)


with open(output_file, "w", encoding="utf-8") as out_f:
    json.dump(conversations, out_f, ensure_ascii=False, indent=4)


!pip3 install vLLM


from vllm import LLM


MODEL_PATH = "..."


model = LLM(MODEL_PATH)


PROMPT_BAD = "Speak only finnish. Generate bad, incorrect answer to the question: "
PROMPT_GOOD = "Speak only finnish. Generate answer to the question: "


def get_response(question: str, base_prompt: str):
    return model.chat(
            [
                [{"role": "user", "content": base_prompt + question}],
            ],
        )[0].outputs[0].text



get_response("What is Kalevala?", PROMPT_BAD)


import pandas as pd
import json
import random


!mkdir datasets


dpo_dataset1 = pd.read_parquet("datasets/train-00000-of-00001-4.parquet")


dpo_dataset1.head()


sampled_rows = dpo_dataset1.sample(n=1000, random_state=42)


def update_response_rejected(row):
    question = row['instruction']
    # Use PROMPT_BAD with 9/10 probability, PROMPT_GOOD with 1/10
    prompt_choice = PROMPT_BAD if random.random() < 0.9 else PROMPT_GOOD
    return get_response(question, prompt_choice)



sampled_rows['response_rejected'] = sampled_rows.apply(update_response_rejected, axis=1)


sampled_rows = sampled_rows[['instruction', 'response', 'response_rejected']]  # Main columns only
sampled_rows.to_json("datasets/result.json", orient="records", force_ascii=False, indent=4)


dpo_dataset1 = dpo_dataset1[['instruction', 'response', 'response_rejected']]
dpo_dataset1.to_json("datasets/dpo_dataset_from_parquet.json", orient="records", force_ascii=False, indent=4)


with open("datasets/dpo_data2.json", "r", encoding="utf-8") as f:
    data = json.load(f)



selected_items = random.sample(data, 100)

for item in selected_items:
    if "rejected" in item:
        question = item.get("question", "")
        prompt_choice = PROMPT_BAD if random.random() < 0.9 else PROMPT_GOOD
        item["rejected"] = get_response(question, prompt_choice)

with open("datasets/dpo_data2_updated.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


dpo_data2 = pd.read_json("datasets/dpo_data2_updated.json", orient="records")
result = pd.read_json("datasets/result.json", orient="records")


result = result.rename(columns={
    "instruction": "question",  
    "response": "chosen",
    "response_rejected": "rejected"
})

merged_data = pd.concat([dpo_data2, result], ignore_index=True)

merged_data.to_json("datasets/merged_data.json", orient="records", force_ascii=False, indent=4)


tune run lora_dpo_single_device --config config.yaml


python3 /home/jupyter/datasphere/alignment/torchtune/torchtune/_cli/tune.py run lora_dpo_single_device --config /home/jupyter/datasphere/alignment/config.yaml


model = LLM("PATH_TO_THE_MODEL")


def get_response(question: str):
    return model.chat(
            [
                [{"role": "user", "content": question}],
            ],
        )[0].outputs[0].text



get_response("Vastaa lyhyesti suomeksi. Kalevalan pÃ¤Ã¤sankari?") # VÃ¤inÃ¤mÃ¶inen. \n\n


get_response("Vastaa lyhyesti suomeksi. MikÃ¤ on Kalevalan tarkoitus?") # 'Kalevala on Suomen epos, joka kokoaa suurperinteisen kans'


get_response("Vastaa lyhyesti suomeksi. mikÃ¤ on SeitsemÃ¤n VeljestÃ¤?") # 'Kaarle XII:n aikana vietetty sarjakuva-animaatio'


get_response("Kuinka monta veljeÃ¤ on SeitsemÃ¤n VeljestÃ¤?" ) # SeitsemÃ¤ssÃ¤ VelÃ¶ssÃ¤ on seitsemÃ¤n veljestÃ¤.  ğŸ˜‰

