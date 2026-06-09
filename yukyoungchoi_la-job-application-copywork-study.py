import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


bulletins = os.listdir("/kaggle/input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins")
additional = os.listdir("/kaggle/input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data")


JT = pd.read_csv("/kaggle/input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/job_titles .csv", header=None)

# 직업명 텍스트 데이터를 리스트로 변환 후, 문자열로 결합
job_titles_list = ', '.join(f'"{title}"' for title in JT[0].astype(str))
print(job_titles_list)


def get_job_title(text_data):
    title_match = re.search("([\s\S]*?)(Class|EXEMPT EMPLOYMENT OPPORTUNITY)", text_data)
    try:
        job_title = title_match.group(1).strip().upper()
    except:
        job_title = None
    return job_title


def get_annual_salary(text_data):
    salary=re.search("ANNUAL\s+SALARY\s+(.*)",text_data)
    if salary:
        return salary.group(1).strip()
    else:
        salary=re.search("ANNUALSALARY\s+(.*)",text_data)
        if salary:
            return salary.group(1).strip()
        else:
            return None


def get_open_date(text_data):
    data=re.search("(\d\d-\d\d-\d\d)[\s\S]*ANNUAL",text_data)
    if data:
        return data.group(1)
    else:
        data = re.search("(\d-\d\d-\d\d)[\s\S]*ANNUAL", text_data)
        if data:
            return data.group(1)
        else:
            data = re.search("DATE:\s+(.*)\s", text_data)
            if data:
                return data.group(1)
            else:
                return None


def get_class_code(text_data):
    class_match = re.search("Class Code:\s+(\d+)\s", text_data)
    try:
        class_code = class_match.group(1)
    except:
        class_code = None
    return class_code


def get_job_duties(text_data):
    duties_match = re.search("(DUTIES|DUTY)\s+(.*)", text_data,re.DOTALL|re.IGNORECASE)
    try:
        duties = duties_match.group(2).strip()
        duties= ' '.join(duties.splitlines())
    except:
        duties = None
    return duties



def get_exam_type(text_data): #    type=re.search("(THIS EXAMINATION|EXAMINATION)(.*\s.*)",text_data)
    type = re.search(f"(THIS EXAMINATION|EXAMINATION)[\s\S]*(PROMOTIONAL AND OPEN|PROMOTIONAL AND AN OPEN)",text_data,re.DOTALL | re.IGNORECASE)
    if type:
        return "BOTH AN INTERDEPARTMENTAL PROMOTIONAL AND OPEN COMPETITIVE BASIS"
    else:
        type = re.search("(THIS EXAMINATION|EXAMINATION)[\s\S]*(COMPETITIVE BASIS|COMPETITVE)", text_data,re.DOTALL|re.IGNORECASE)
        if type:
            return "ONLY ON AN OPEN COMPETITIVE BASIS"
        else:
            type = re.search(f"(THIS EXAMINATION|EXAMINATION)[\s\S]*(INTERDEPARTMENTAL|DEPARTMENTAL|INTERDEPARMENTAL)", text_data,re.DOTALL | re.IGNORECASE)
            if type:
                return "ONLY ON AN INTERDEPARTMENTAL PROMOTIONAL BASIS"
            else:
                return None


requirements_tag='|'.join(["REQUIREMENT/MIMINUMUM QUALIFICATION",
                  "REQUIREMENT/MINUMUM QUALIFICATION",
                  "REQUIREMENT/MINIMUM QUALIFICATION",
                  "REQUIREMENT/MINIMUM QUALIFICATIONS",
                  "REQUIREMENT/ MINIMUM QUALIFICATION",
                  "REQUIREMENTS/MINUMUM QUALIFICATIONS",
                  "REQUIREMENTS/ MINIMUM QUALIFICATIONS",
                  "REQUIREMENTS/MINIMUM QUALIFICATIONS",
                  "REQUIREMENTS/MINIMUM REQUIREMENTS",
                  "REQUIREMENTS/MINIMUM QUALIFCATIONS",
                   "REQUIREMENT/MINIMUM QUALIFICAITON",
                  "MINIMUM REQUIREMENTS:",
                  "REQUIREMENTS",
                  "REQUIREMENT"])

def get_reqs(text_data):
    reqs=re.search(f"({requirements_tag})\s+([\s\S]*?)(PROCESS NOTES|PROCESS NOTE|NOTE)",text_data,re.DOTALL|re.IGNORECASE)
    if reqs:
         reqs=reqs.group(2)
         reqs=' '.join(reqs.splitlines())
         return reqs
    else:
        return None



license_str="licensed|LICENSED|license|LICENSE"

def get_license(text_data):
    
    license=re.search(f"{license_str}",text_data,re.DOTALL|re.IGNORECASE)
    if license:
        return "R"
    else:
        return "NR"

school = "college or university|university or college|high school|college|" \
         "apprenticeship|university|school"

def get_education(text_data):
    educations=re.search(f"{school}",text_data,re.DOTALL|re.IGNORECASE)
    if educations:
        education=educations.group(0).upper()
    else:
        educations=re.search("(diploma|bachelor|master|phd)",text_data,re.DOTALL|re.IGNORECASE)
        if educations:
            education=educations.group(0).upper()
        else:
            education= None
    return education


part_or_full='part-time or full-time|full-time or part-time|part or full|full or part|part-time|full-time|PART-TIME|FULL-TIME|fulltime|parttime|full time|part time'

def get_part_or_full_time(text_data):
    jobtype=re.search(f"{part_or_full}",text_data,re.DOTALL|re.IGNORECASE)
    if jobtype:
        return jobtype.group(0).upper()
    else:
        return None


numbers='one|One|two|Two|three|Three|four|Four|five|Five|six|Six|seven|Seven|eight|Eight|nine|Nine'

def get_experience(requirements_data):
    exp=re.search(f"({numbers})\s(years|year).*\s(full-time|part-time)",requirements_data,re.DOTALL|re.IGNORECASE|re.I)
    if exp:
        exp=exp.group(1).upper()
        number = get_number(exp)
        return number
    else:
        exp = re.search(f"(full-time|part-time).*?({numbers})\s(years|year)", requirements_data,re.DOTALL | re.IGNORECASE)
        if exp:
            exp=exp.group(2).upper()
            number = get_number(exp)
            return number
        else:
            exp = re.search(f"({part_or_full})[\s\S]*?({numbers})", requirements_data,re.DOTALL | re.IGNORECASE)
            if exp:
                exp=exp.group(2).upper()
                number = get_number(exp)
                return number
            else:
                return None


def get_education_years(requirements_data):
    years = re.search(f"({numbers})[/s-]year.*?({school})", requirements_data,re.DOTALL|re.IGNORECASE)
    if years:
        years=years.groups(1)[0].upper()
        number = get_number(years)
        return number
    else:
        years = re.search(f"({numbers})[/s-]year[\s\S]*?({school})", requirements_data, re.DOTALL | re.IGNORECASE )
        if years:
            years=years.groups(1)[0].upper()
            number = get_number(years)
            return number
        else:
            years = re.search(f"({requirements_tag})[\s\S]*({numbers})\s+(years|year)[\s\S]*?({school})", requirements_data,re.DOTALL | re.IGNORECASE)
            if years:
                years=years.groups(2)[1].upper()
                number = get_number(years)
                return number
            else:
                return None


education_majors='|'.join(["Agribusiness Operations","Airport Guide","computer engineering","Airport","Agricultural Business","Agricultural Economics","carpet layer","Agricultural Mechanization","Agricultural Production","Agronomy & Crop Science","Agriculture","Animal Sciences","Food Sciences ","Horticulture Operations & Management","Horticulture Science","Natural Resources Conservation","Environmental Science","Forestry","Natural Resources Management","Wildlife & Wildlands Management","Architecture","Architectural Environmental Design","Regional Planning","Interior Architecture","Landscape Architecture","African American Studies","Women’s Studies","Liberal Arts","General Studies","Library Science","Interdisciplinary","Art History","Criticism","Studio Arts"," Art ","Cinematography","Video Production","Dancing","Design & Visual Communications","Fashion","Apparel Design","Graphic Design","Industrial Design","Interior Design","Music","Photography","Theatre Arts","Drama","Accounting Technician","Accounting","Business Administration & Management","Hotel Management","Human Resources Management","Human Resources ","International Business Management","Labor","Industrial Relations","Logistics & Materials Management","Marketing Management & Research","Office Supervision & Management","Operations Management & Supervision","Organizational Behavior","Contracts Management","Food Services Management","Small Business Management/Operations","Tourism Management",
                           "Actuarial Science","auto body and fender repairer","auto body builder","body builder","Business","Managerial Economics","Finance","Banking & Financial Support Services","Financial Planning & Services","Insurance & Risk Management","Investments & Securities","Management Information Systems","Real Estate","Sales","Merchandising","Marketing","Fashion Merchandising","Tourism & Travel Marketing","Secretarial Studies & Office Administration","Communications","Advertising","Digital Communications","Media","Journalism","Mass Communications","Public Relations & Organizational Communication","Radio & Television Broadcasting","Communications","Graphic & Printing Equipment Operation","Multimedia","Animation","Special Effects","Radio & Television Broadcasting","Family & Consumer Sciences","Adult Development & Aging/Gerontology","Child Care Services Management","Child Development","Consumer & Family Economics","Food & Nutrition","Textile & Apparel","Parks, Recreation, & Leisure","Exercise Science","Physiology","Kinesiology","Physical Education","Fitness","Administration Management","Personal Services","Cosmetology","Culinary Arts","Chef Training","Funeral Services & Mortuary Science","Protective Services","Corrections","Criminal Justice","Fire Protection & Safety","Law Enforcement","Military Technologies","Public Administration & Services","Community Organization & Advocacy","Public Administration","Public Affairs & Public Policy Analysis","Social Work","COMPUTER SCIENCE",
                    "Geographical Information Systems","Information System","MATHEMATICS","Computer & Information Sciences","Computer Networking","Telecommunications","Computer Programming","Computer Software & Media Applications","Computer System Administration","Data Management","Information Science","Webpage Design","Mathematics","Applied Mathematics","Statistics","Counseling & Student Services","Educational Administration","Special Education","Teacher Education","Curriculum","Early Childhood Education","Elementary Education","Junior High/Middle School Education","Postsecondary Education","Secondary Education","Teacher Assisting","Aide Education","Teacher Education, Subject-Specific","Agricultural Education","Art Education","Business Education","Technical Education","English-as-a-Second-Language Education","English/Language Arts Education","Foreign Languages Education","Health Education","Mathematics Education","Music Education","Physical Education","Science Education","Social Studies","Sciences Education","Aerospace Engineering","Aeronautical Engineering","Agricultural/Bioengineering","Architectural Engineering","Biomedical Engineering","Chemical Engineering","Civil Engineering","Computer Engineering","Construction Engineering/Management","Electrical, Electronics & Communications Engineering","Environmental Health Engineering","Industrial Engineering","Mechanical Engineering","Nuclear Engineering","ENGINEERING TECHNOLOGY & DRAFTING",
                           "Architectural Drafting","Mechanical Drafting","Engineering","engineering","Drafting","Aeronautical/Aerospace Engineering Technologies","Architectural Engineering","Automotive Engineering","Civil Engineering","Computer Engineering","Construction","Engineering","Electrical Engineering","Electronics Engineering","Electromechanical Engineering","Electromechanical ","Biomedical Engineering","Environmental Control","Industrial Production Technologies","Mechanical Engineering","Quality Control & Safety Technologies","Surveying","English Language ","American/English Literature","Creative Writing","Public Speaking","Foreign Languages","Asian Languages ","Ancient Languages ","Literatures","Comparative Literature","French Language & Literature","German Language & Literature","Linguistics","Middle Eastern Languages & Literatures","Spanish Language & Literature","HEALTH ADMINISTRATION & ASSISTING","Health Services Administration","Facilities Administration","Medical Office","Secretarial","Medical Records","Medical Clinical Assisting","Dental Assisting","Medical Assisting","Occupational Therapy Assisting","Physical Therapy Assisting","Veterinarian Assisting","HEALTH SCIENCES & TECHNOLOGIES","Chiropractic","Dental Hygiene","Dentistry","Emergency Medical","Health-Related Professions & Services","Athletic Training","Public Health","Medical Laboratory","Medical Radiologic","Nuclear Medicine","Respiratory Therapy","Building and Safety","Safety and Health",
                           "Surgical","Medicine","Nursing, Practical","Nursing, Registered","Pharmacy ","Physician Assisting","Therapy & Rehabilitation","Alcohol/Drug Abuse Counseling","Massage Therapy","Mental Health Counseling","Occupational Therapy","Physical Therapy ","Psychiatric","Mental Health Technician","Rehabilitation Therapy","Vocational Rehabilitation Counseling","Veterinary Medicine","PHILOSOPHY","Theology","Biblical Studies","Divinity","Ministry","Marketing","refrigeration and air conditioning","refrigeration","air conditioning","Religious Education","Aviation & Airway Science","Aircraft Piloting & Navigation","Aviation Management & Operations","Construction Trades","Mechanics & Repairers","Aircraft Mechanics/Technology","Autobody Repair/Technology","Automotive Mechanics/Technology","Avionics","Diesel Mechanics","Electrical/Electronics Equip Installation & Repair","Precision Production Trades","Machine Tool","Welding","Biology","Biochemistry & Biophysics","Cellular Biology","Ecology","Genetics","Marine/Aquatic Biology","Microbiology & Immunology","Zoology","Physical Sciences","Astronomy","Atmospheric Sciences & Meteorology","Chemistry","Geological & Earth Sciences","Physics","SOCIAL SCIENCES & LAW","Legal Studies","Court Reporting"," Law ","Legal Administrative Assisting/Secretarial","Paralegal/Legal Assistant","Social Sciences","Anthropology","Criminology","Economics","Geography","History","International Relations & Affairs","Political Science & Government","Clinical & Counseling","Psychology","Sociology","Science","Accounting","acounting","engineering","Maintenance"])

def get_major(text_data):
    major=re.search(f"(major in |as a ).*?({education_majors})",text_data,re.DOTALL|re.IGNORECASE|re.I)   #,"Safety"
    if major:
        return major.group(2).upper()
    else:
        major = re.search(f"({school})[\s\S]*?({education_majors})", text_data,re.DOTALL | re.IGNORECASE)
        if major:
            return major.group(2).upper()
        else:
            major = re.search(f"({education_majors})",text_data, re.DOTALL | re.IGNORECASE)
            if major:
                return major.group(1).upper()
            else:
                return None



def get_course_counts(text_data):
    counts=re.search(f"({numbers})( courses| course|courses|course|-courses|-course)",text_data,re.DOTALL|re.IGNORECASE|re.I)
    if counts:
        counts=counts.group(1).upper()
        number = get_number(counts)
        return number
    else:
        counts=len(re.findall("( course )",text_data,re.DOTALL|re.IGNORECASE|re.I))
        return counts

def get_course_length(text_data,count):
    if count!=0:
        length=re.search(f"( courses| course|courses|course|-courses|-course).*?({numbers})",text_data,re.DOTALL|re.IGNORECASE|re.I)
        if length:
            length=length.groups(2)[1].upper()
            number=get_number(length)
            return number
        else:
            return None
    else:
        return 0


jobs1='|'.join(["Occupations","Accountant","Accounts Assistant","Accounts Clerk","Accounts Manager","Senior Management Analyst","Senior Management","Data Analysis","Data Scientist",
       "Accounts Staff","Acoustic Engineer"," Actor ","Actress","Actuary","Acupuncturist","finance management","Electrical Mechanic","Electrical Helper","helper in an electrical","electric services",
        "Electrical Craft Helper","Electrical Craft","electrical work","electric services","repair of auto bodies",
       "Adjustor","Administration Assistant","Administration Clerk","Administration Manager","cement finisher helper",
       "Administration Staff","Administrator","Advertising Agent","Advertising Assistant","Advertising Clerk","Advertising Contractor",
       "Advertising Executive","Advertising Manager","Advertising Staff","Aerial Erector","Aerobic Instructor","Aeronautical Engineer",
       "Agent","Air Traffic Controller","Aircraft Designer","Aircraft Engineer","Aircraft Maintenance Engineer","Maintenance Engineer","Aircraft Surface Finisher",
       "Airman","Airport Controller","Airport Manager","Almoner","Ambulance Controller","Ambulance Crew","Ambulance Driver","Amusement Arcade Worker","Anaesthetist",
       "Analyst","Analytical Chemist","Animal Breeder","Anthropologist","Antique Dealer","Applications Engineer","Applications Programmer","Arbitrator","Arborist","Archaeologist","Architect","Archivist","Area Manager","Armourer","Aromatherapist","Art Critic","Art Dealer","Art Historian","Art Restorer","Artexer","Artist"," Arts ","Assembly Worker","Assessor","Assistant","Assistant Caretaker","Assistant Cook","Assistant Manager","Assistant Nurse","Assistant Teacher","Astrologer","Astronomer","Attendant","Au Pair","Auction Worker","Auctioneer","Audiologist","Audit Clerk","Audit Manager","Auditor","Auto Electrician","Auxiliary Nurse","Bacon Curer","Baggage Handler","Bailiff","Baker","Bakery Assistant","Bakery Manager","Bakery Operator","Balloonist","Bank Clerk","Bank Manager","Bank Messenger","Baptist Minister","Bar Manager","Bar Steward","Barber","Barmaid","Barman","Barrister","Beautician","Beauty Therapist","Betting Shop","Bill Poster","Bingo Caller","Biochemist","Biologist","Blacksmith","Blind Assembler","Blind Fitter","Blinds Installer","Boat Builder","Body Fitter","Bodyguard","Bodyshop","Book Binder","Book Seller","Book-Keeper","Booking Agent","Booking Clerk","Bookmaker","Botanist","Branch Manager","Breeder","Brewer","Brewery Manager","Brewery Worker","Bricklayer","Broadcaster","Builder","Builders Labourer","Building Advisor","Building Control","Building Engineer","Building Estimator","Building Foreman","Building Inspector","Building Manager","Building Surveyor","Bursar","Bus Company","Bus Conductor","Bus Driver","Bus Mechanic","Bus Valeter","Business Consultant","Business Proprietor","Butcher","Butchery Manager","Butler","Buyer","Cab Driver","Cabinet Maker","Cable Contractor","Cable Jointer","Cable TV Installer","Cafe Owner","Cafe Staff","Cafe Worker","Calibration Manager","Camera Repairer","Cameraman","Car Dealer","Car Delivery Driver","Car Park Attendant","Car Salesman","Car Valet","Car Wash Attendant","Care Assistant","Care Manager","Careers Advisor","Careers Officer","Caretaker","Cargo Operator","Carpenter","Carpet Cleaner","Carpet Fitter","Carpet Retailer","Carphone Fitter","Cartographer","Cartoonist","Cashier","Casual Worker","Caterer","Catering Consultant","Catering Manager","Catering Staff","Caulker","Ceiling Contractor","Ceiling Fixer","Cellarman","Chambermaid","Chandler","Chaplain","Charge Hand","Charity Worker","Chartered","Chartered Accountant","Chauffeur","Chef","Chemist","Chicken Chaser","Child Minder","Childminder","Chimney Sweep","China Restorer","Chiropodist","Chiropractor","Choreographer","Church Officer","Church Warden","Cinema Manager","Circus Proprietor","Circus Worker","Civil Engineer","Civil Servant","Claims Adjustor","Claims Assessor","Claims Manager","Clairvoyant","Classroom Aide","Cleaner","Clergyman"," Cleric ","Commissioned","Consultant","Coroner","Councillor","Counsellor","Dealer","Decorator","Delivery Driver","Doctor","Car Driver","Economist","Editor","Engineer","English Teacher","Entertainer","Envoy","Executive","Farmer","Fireman","Floor Layer","Floor Manager","Florist","Flour Miller","Flower Arranger","Flying Instructor","Foam Convertor","Food Processor","Footballer","Foreman","Forensic Scientist","Forest Ranger","Forester","Fork Lift Truck Driver","Forwarding Agent","Foster Parent","Foundry Worker","Fraud Investigator","French Polisher","Fruiterer","Fuel Merchant","Fund Raiser","Funeral Director","Funeral Furnisher","Furnace Man","Furniture Dealer","Furniture Remover","Furniture Restorer","Furrier","Gallery Owner","Gambler","Gamekeeper","Gaming Board Inspector","Gaming Club Manager","Gaming Club Proprietor","Garage Attendant","Garage Foreman","Garage Manager","Garda","Garden Designer","Gardener","Gas Fitter","Gas Mechanic","Gas Technician","Gate Keeper",
                "Genealogist","General Practitioner","Geologist","Geophysicist","Gilder","Glass Worker","Glazier","Goldsmith","Golf Caddy","Golf Club Professional","Golfer","Goods Handler","Governor","Granite Technician","Graphic Designer","Graphologist","Grave Digger","Gravel Merchant","Green Keeper","Greengrocer","Grocer","Groom","Ground Worker","Groundsman","Guest House Owner","Guest House Proprietor","Gun Smith","Gynaecologist","HGV Driver","HGV Mechanic","Hairdresser","Handyman","Hardware Dealer","Haulage Contractor","Hawker","Health Advisor","Health And Safety","Health Care Assistant","Health Consultant","Health Nurse","Health Planner","Health Service","Health Therapist","Health Visitor","Hearing Therapist","Heating Engineer","Herbalist","Highway Inspector","Hire Car Driver","Historian","History Teacher","Hod Carrier","Home Economist","Home Help","Homecare Manager","Homeopath","Homeworker","Hop Merchant","Horse Breeder","Horse Dealer","Horse Riding Instructor","Horse Trader","Horse Trainer","Horticultural Consultant","Horticulturalist","Hosiery Mechanic","Hosiery Worker","Hospital Consultant","Hospital Doctor","Hospital Manager","Hospital Orderly","Hospital Technician","Hospital Warden","Hospital Worker","Hostess","Hot Foil Printer","Hotel Consultant","Hotel Worker","Hotelier","Househusband","Housekeeper","Housewife","Housing Assistant","Housing Officer","Housing Supervisor","Hygienist","Hypnotherapist","Hypnotist","IT Consultant","IT Manager","IT Trainer","Ice Cream Vendor","Illustrator","Immigration Officer","Import Consultant","Importer","Independent Means","Induction Moulder","Industrial Chemist","Industrial Consultant","Injection Moulder","Inspector","Instructor","Instrument Engineer","Instrument Maker","Instrument Supervisor","Instrument Technician","Insurance Agent","Insurance Assessor","Insurance Broker","Insurance Consultant","Insurance Inspector","Insurance Staff","Interior Decorator","Interior Designer","Interpreter","Interviewer","Inventor","Investigator","Investment Advisor","Investment Banker","Investment Manager","Investment Strategist","Ironmonger","Janitor","Jazz Composer","Jeweller","Jewellery","Jockey","Joiner","Joinery Consultant","Journalist","Judge","Keep Fit Instructor","Kennel Hand","Kitchen Worker","Knitter","Labelling Operator","Laboratory Analyst","Labourer","Laminator","Lampshade Maker","Land Agent","Land Surveyor","Landlady","Landlord","Landowner","Landworker","Lathe Operator","Laundry Staff","Laundry Worker","Lavatory Attendant","Law Clerk","Lawn Mower","Lawyer","Leaflet Distributor","Leather Worker","Lecturer","Ledger Clerk","Legal Advisor","Legal Assistant","Legal Executive","Legal Secretary","Letting Agent","Liaison Officer","Librarian","Library Manager","Licensed Premises","Licensee","Licensing","Lifeguard","Lift Attendant","Lift Engineer","Lighterman","Lighthouse Keeper","Lighting Designer","Lighting Technician","Lime Kiln Attendant","Line Manager","Line Worker","Lineman","Linguist","Literary Agent","Literary Editor","Lithographer","Litigation Manager","Loans Manager","Local Government","Lock Keeper","Locksmith","Locum Pharmacist","Log Merchant","Lorry Driver","Loss Adjustor","Loss Assessor","Lumberjack","Machine Fitters","Machine Minder","Machine Operator","Machine Setter","Machine Tool","Machine Tool Fitter","Machinist","Relations Management","financial management","Magician","Magistrate","Magistrates Clerk","Maid","Maintenance Fitter","Make Up Artist","Manicurist","Manufacturing","Map Mounter","Marble Finisher","Marble Mason","Marine Broker","Marine Consultant","Marine Electrician","Marine Engineer","Marine Geologist","Marine Pilot","Marine Surveyor","Market Gardener","Market Research","Market Researcher","Market Trader","Marketing Agent","Marketing Assistant","Marketing Coordinator","Marketing Director","Marketing Manager","Marquee Erector","Massage Therapist","Masseur","Masseuse","Master Mariner","Materials Controller","Materials Manager","Mathematician","Maths Teacher","Matron","Mattress Maker","Management Assistant","Management","Meat Inspector","Meat Wholesaler","Mechanic","Medal Dealer","Medical Advisor","Medical Assistant","Medical Consultant","Medical Officer","Medical Physicist","Medical Practitioner","Medical Researcher","Medical Secretary","Medical Student","Medical Supplier","Medical Technician","Merchandiser","Merchant","Merchant Banker","Merchant Seaman","Messenger","Metal Dealer","Metal Engineer","Metal Polisher","Metal Worker","Metallurgist","Meteorologist",
       "Meter Reader","Microbiologist","Midwife","Military Leader","Milklady","Milkman","Mill Operator","Mill Worker","Miller","Milliner","Millwright","Miner","Mineralologist","Minibus Driver","Minicab Driver","Mining Consultant","Mining Engineer","Money Broker","Moneylender","Mooring Contractor","Mortgage Broker","Mortician","Motor Dealer","Motor Engineer","Motor Fitter","Motor Mechanic","Motor Racing","Motor Trader","Museum Assistant","Museum Attendant","Music Teacher","Musician","Nanny","Navigator","Negotiator","Neurologist","Newsagent","Night Porter","Night Watchman","Nuclear Scientist","Nun","Nurse","Nursery Assistant","Nursery Nurse","Nursery Worker","Nurseryman","Nursing Assistant","Nursing Auxiliary","Nursing Manager","Nursing Sister","Nutritionist","Off Shore","Office Manager","Office Worker","Oil Broker","Oil Rig Crew","Opera Singer","Operative","Optical","Optical Advisor","Optical Assistant","Optician","Optometrist","Orchestral","Organiser","Organist","Ornamental","Ornithologist","Orthopaedic","Orthoptist","Osteopath","Outdoor Pursuits","Outreach Worker","Packaging","Packer","Paediatrician","Paint Consultant","Painter","Palaeobotanist","Palaeontologist","Pallet Maker","Panel Beater","Paramedic","Park Attendant","Park Keeper","Park Ranger","Partition Erector","Parts Man","Parts Manager","Parts Supervisor","Party Planner",
       "Pasteuriser","Pastry Chef","Patent Agent","Patent Attorney","Pathologist","Patrolman","Pattern Cutter","Pattern Maker","Pattern Weaver","Pawnbroker","Payroll Assistant","Payroll Clerk","Payroll Manager","Payroll Supervisor","Personnel Officer","Pest Controller","Pet Minder","Pharmacist","Philatelist","Photographer","Physician","Physicist","Physiologist","Physiotherapist","Piano Teacher","Piano Tuner","Picture Editor","Picture Framer","Picture Reseacher","Pig Man","Pig Manager","Pilot","Pipe Fitter","Pipe Inspector","Pipe Insulator","Pipe Layer","Planning Engineer","Planning Manager","Planning Officer","Planning Technician","Plant Attendant","Plant Driver","Plant Engineer","Plant Fitter","Plant Manager","Plant Operator","Plasterer","Plastics Consultant","Plastics Engineer","Plate Layer","Plater","Playgroup Assistant","Playgroup Leader","Plumber","Podiatrist","Police Officer","Polisher","Pool Attendant","Pools Collector","Porter","Portfolio Manager","Post Sorter","Postman","Postmaster","Postwoman","Potter","Practice Manager","Preacher","Precision Engineer","Premises","Premises Security","Press Officer","Press Operator","Press Setter","Presser","Priest","Print Finisher","Printer","Prison Chaplain","Prison Officer","Private Investigator","Probation Officer","Probation Worker","Procurator Fiscal","Produce Supervisor","Producer","Product Installer","Product Manager","Production Engineer","Production Hand","Production Manager","Production Planner","Professional Boxer","Professional Racing","Professional Wrestler","Progress Chaser","Progress Clerk","Project Co-ordinator","Project Engineer","Project Leader","Project Manager","Project Worker","Projectionist","Promoter","Proof Reader","Property Buyer","Property Dealer","Property Developer","Property Manager","Property Valuer","Proprietor","Psychiatrist","Psychoanalyst","Psychologist","Psychotherapist","Public House Manager","Public Relations Of?cer","Publican","Publicity Manager","Publisher","Publishing Manager","Purchase Clerk","Purchase Ledger Clerk","Purchasing Assistant","Purchasing Manager","Purser","Quality Controller","Quality Engineer","Quality Inspector","Quality Manager","Quality Technician","Quantity Surveyor","Quarry Worker","Racehorse Groom","Racing Organiser","Radio Controller","Radio Director","Radio Engineer","Radio Operator","Radio Presenter","Radio Producer","Radiographer","Radiologist","Rally Driver","Receptionist","Recorder","Records Supervisor","Recovery Vehicle Coordinator","Recreational","Recruitment Consultant","Rector","Reflexologist","Refractory Engineer","Refrigeration Engineer","Refuse Collector","Registrar","Regulator","Relocation Agent","Remedial Therapist",
       "Rent Collector","Rent Offcer","Repair Man","Repairer","Reporter","Representative","Reprographic Assistant","Research Analyst","Research Consultant","Research Director","Research Scientist","Research Technician","Researcher","Resin Caster","Restaurant Manager","Restaurateur","Restorer","Retired","Revenue Clerk","Revenue Officer","Riding Instructor","Rig Worker","Rigger","Riveter","Road Safety Officer","Road Sweeper","Road Worker","Roadworker","Roof Tiler","Roofer","Rose Grower","Royal Marine","Rug Maker","Saddler","Safety Officer","Sail Maker","Sales Administrator","Sales Assistant","Sales Director","Sales Engineer","Sales Executive","Sales Manager","Sales Representative","Sales Support","Salesman","Saleswoman","Sand Blaster","Saw Miller","Scaffolder","School Crossing","School Inspector","Scientific Officer","Scientist","Scrap Dealer","Screen Printer","Screen Writer","Script Writer","Sculptor","Seaman","Seamstress","Secretary","Security Consultant","Security Controller","Security Guard","Security Officer","Servant","Service Engineer","Service Manager","Share Dealer","Sheet Metal Worker","Shelf Filler","Shelter Warden","Shepherd","Sheriff","Sheriff Clerk","Sheriff Principal","Shift Controller","Ship Broker","Ship Builder","Shipping Clerk","Shipping Officer","Shipwright","Shipyard Worker","Shoe Maker","Shoe Repairer","Shooting Instructor","Shop Assistant","Shop Fitter","Shop Keeper","Shop Manager","Shop Proprietor","Shot Blaster","Show Jumper","Showman","Shunter","Sign Maker","Signalman","Signwriter","Site Agent","Site Engineer","Skipper","Slater","Slaughterman","Smallholder","Social Worker","Software Consultant","Software Engineer","Soldier","Solicitor","Song Writer","Sound Artist","Sound Engineer","Sound Technician","Special Constable","Special Needs","Speech Therapist","Sports Administrator","Sports Coach",
                "Sports Commentator","Sportsman","Sportsperson","Sportswoman","Spring Maker","Stable Hand","Staff Nurse","Stage Director","Stage Hand","Stage Manager","Stage Mover","Station Manager","Stationer","Statistician","Steel Erector","Steel Worker","Steeplejack","Stenographer","Steward","Stewardess","Stock Controller","Stock Manager","Stockbroker","Stockman","Stocktaker","Stone Cutter","Stone Sawyer","Stonemason","Store Detective","Storeman","Storewoman","Street Entertainer","Street Trader","Stud Hand","Student","Student Nurse","Student Teacher","Studio Manager","Sub-Postmaster","Sub-Postmistress","Supervisor","Supply Teacher","Surgeon","Surveyor","Systems Analyst","Systems Engineer","Systems Manager","TV Editor","Tachograph Analyst","Tacker","Tailor","Tank Farm Operative","Tanker Driver","Tanner","Tattooist","Tax Advisor","Tax Analyst","Tax Assistant","Tax Consultant","Tax Inspector","Tax Manager","Tax Officer","Taxi Controller","Taxi Driver","Taxidermist","Tea Blender","Tea Taster","Teacher","Teachers Assistant","Technical Advisor","Technical Analyst","Technical Assistant","Technical Author","Technical Clerk","Technical Co-ordinator","Technical Director","Technical Editor","Technical Engineer","Technical Illustrator","Technical Instructor","Technical Liaison","Technical Manager","Technician","Telecommunication","Telecommunications","Telegraphist","Telemarketeer","Telephone Engineer","Telephonist","Telesales Person","Television Director","Television Engineer","Television Presenter","Television Producer","Telex Operator","Temperature Time","Tennis Coach","Textile Consultant","Textile Engineer","Textile Technician","Textile Worker","Thatcher","Theatre Manager","Theatre Technician","Theatrical Agent","Therapist","Thermal Engineer","Thermal Insulator","Ticket Agent","Ticket Inspector","Tiler","Timber Inspector","Timber Worker","Tobacconist","Toll Collector","Tool Maker","Tour Agent","Tour Guide","Town Clerk","Town Planner","Toy Maker","Toy Trader","Track Worker","Tractor Driver","Tractor Mechanic","Trade Mark Agent","Trade Union Official","Trading Standards","Traffic Warden","Train Driver","Trainee Manager","Training Advisor","Training Assistant","Training Co-ordinator","Training Consultant","Training Instructor","Training Manager","Training Officer","Transcriber","Translator","Transport Clerk","Transport Consultant","Transport Controller","Transport Engineer","Transport Manager","Transport Officer","Transport Planner","Travel Agent","Travel Clerk","Travel Consultant","Travel Courier","Travel Guide","Travel Guide Writer","Travel Representative","Travelling Showman","Treasurer","Tree Feller","Tree Surgeon","Trichologist","Trinity House Pilot","Trout Farmer","Tug Skipper","Tunneller","Turf Accountant","Turkey Farmer","Turner","Tutor","Typesetter","Typewriter Engineer","Typist","Tyre Builder","Tyre Fitter","Tyre Inspector","Tyre Technician","Undertaker","Underwriter","Upholsterer","Valuer","Valve Technician","Van Driver","Vehicle Assessor","Vehicle Body Worker","Vehicle Engineer","Vehicle Technician","Ventriloquist","Verger","Veterinary Surgeon","Vicar","Video Artist","Violin Maker","Violinist","Voluntary Worker",
                "Wages Clerk","Waiter","Waitress","Warden","Warehouse Manager","Warehouseman","Warehousewoman","Watchmaker","Weaver","Weighbridge Clerk","Weighbridge Operator","Welder","Welfare Assistant","Welfare Officer","Welfare Rights Officer","Wheel Clamper","Wholesale Newspaper","Window Cleaner","Window Dresser","Windscreen Fitter","Wine Merchant","Wood Carver","Wood Cutter","Wood Worker","Word Processing Operator","Works Manager","Writer","Yacht Master","Yard Manager","Youth Hostel Warden","Youth Worker","Zoo Keeper","Zoo Manager","Zoologist"])

jobs2='|'.join([".net developer ",".net software developer ","accessibility outreach coordinator ","accessibility program manager ","account executive  ","account manager ","accountable manager ","accountable project manager ","accountant ","accounting investment analyst ","actuary ","admin engineer ","administrative aide ","administrative assistant ","administrative associate to the executive director ","administrative business promotion coordinator ","administrative claim examiner ","administrative contract specialist ","administrative coordinator ","administrative engineer ","administrative management auditor ","administrative manager ","administrative office assistant ","administrative procurement analyst ","administrative services manager ","administrative specialist ","administrative staff analyst ","administrative support ","administrative transportation coordinator ","administrator on duty ","advertising and promotions manager ","advertising sales agent ","aerospace engineer ","agency attorney ","agency counsel ","agricultural engineer ","agricultural scientist ","agricultural technician ","aide ","air & noise pollution inspector ","aircraft mechanic ","aircraft pilot ","analyst manager ","analyst ","animal caretaker ","animal control worker ","animal trainer ","animal breeder ","announcer ","apparel patternmaker ","appliance repairer ","application developer ","application examiner ","application solution manager ","application support analyst ","application support reporting specialist ","application worker ","appraiser ","apprentice inspector ","architect ","architectural designer ","archivist ","artist ","assembler ","assessor  ","asset management specialist ","asset manager ","assistant architect ","assistant business services associate ","assistant chief of facility compliance ","assistant civil engineer ","assistant commissioner of administration ","assistant commissioner of communications and policy ","assistant commissioner of enforcement ","assistant commissioner of environmental health unit ","assistant commissioner of housing policy ","assistant commissioner of the office of placement administration ","assistant commissioner ","assistant quality assurance associate ","astronomer ","atmospheric scientist ","audiologist ","audit engineer ","audit intern ","audit manager ","audit supervisor ","auditor ","author ","auto  machinist ","auto mechanic ","automotive glass installer ","avionics technician ","back end developer ","baggage porter ","bailiff ","baker ","barber ","bartender helper ","bartender ","bellhop ","best practices coach ","bid operations liaison ","bill and account collector ","billing and posting clerk ","biological scientist ","biological technician ","biomedical engineer ","blake fellow ","blockmason ","boat captain ","boiler inspector ","boilermaker ","bookbinder ","bookkeeper ","branch chief ","brickmason ","bridge operator ","broadcast and sound engineering technician ","brokerage clerk ","budget analyst ","budget review specialist ","budget ","building cleaner ","building inspector ","bus and truck mechanics and diesel engine specialist ","business administrator  ","business analyst ","business intelligence developer ","business manager ","business operations specialist ","business solution architect ","butcher ","buyer services manager ","buyer ","cabinetmaker ","cafeteria attendant ","calendar assistant ","call center manager ","camera operator ","capacity building assistance specialist ","cargo and freight agent ","carpenter ","cartographer ","case analyst ","case management nurse ","case management team leader ","case management unit supervisor ","case monitor supervisor ","case worker ","cashier supervisor ","cashier ","caster ","ceiling tile installer ","cement mason ","certified it administrator ","certified it developer ","certified specialist ","channel manager ","chauffeur ","chef ","chemical engineer ","chemical technician ","chemists ","chief architect ","chief compliance officer ","chief diversity officer ","chief engineer of dispute resolutions ","chief executive ","chief financial officer ","chief information security officer ","chief investigator ","chief of staff ","chief operating officer ","chief plan examiner ","chief review officer ","child care worker ","child welfare analyst ","chiropractor ","choreographer ","city assessor ","city laborer ","city medical examiner ","city park worker ","city planner ","city planning technician ","city research scientist ","civil design lead ","civil engineer ","civilian investigator ","claim specialist ","claims adjuster ","claims processor ","claims specialist ","clergy ","clerical aide ","clerical assistant ","clerical associate ","clerical supervisor ","client services representative ","climber & pruner ","clinical case supervisor ","clinical director ","clinical laboratory technologist and technician ","cloud reliability engineer ","coach ","collections administrator ","collector ","college aide ","commercial diver ","communication electrician ","communication manager ","communication specialist ","communication worker ","communications equipment operator ","community & industry relations associate ","community assistant ","community associate ","community coordinator ","community service manager ","community supervisor ","compliance analyst ","compliance and enforcement attorney ","compliance auditor ","compliance manager ","compliance officer ","compliance support specialist ","computer aide ","computer and information systems manager ","computer associate ","computer control operator ","computer hardware engineer ","computer operator ","computer programmer ","computer scientist ","computer service technician ","computer software engineer ","computer specialist ","computer support specialist ","computer systems manager ","concierge ","confidential investigator ","conflict resolution specialist ","consent specialist ","conservation scientist  ","constituent services liaison ","construction inspector ","construction intern ","construction manager ","construction project manager ","construction safety and quality inspector ","content developer ","content engineer  ","contract analyst ","contract manager ","contract processor ","contract specialist ","contracts attorney ","contracts officer ","conveyor operator ","coordinator ","correctional officer ","correctional officers ","cosmetologist ","cost estimating manager ","cost estimator ","counselor ","courier ","court liaison officer ","court representative ","cranes & derricks inspector ","credit analyst ","credit authorizer ","crime analyst ","criminal investigator ","criminalist ","crm developer ","crossing guard ","cultural affairs coordinator ","curator ","custodial assistant ","custodian ","customer service & operations analysis intern ","customer service representative ","customer service ","cybersecurity operations analyst ","dam safety coordinator ","dancer ","data analyst ","data and process analyst ","data and technology analyst ","data architect ","data assistant ","data engineer ","data entry clerk ","data entry keyer ","data management assistant ","data manager","data research analyst ","data research specialist ","data scientist ","data support analyst ","data warehouse architect ","database administrator","database administrator ","database developer ","dba engineer ","dba lead ","deckhand ","demonstrator ","dental assistant ","dental hygienist ","dentist ","deputy budget director ","deputy chief engineer ","deputy chief of outside development ","deputy chief of quality assurance ","deputy chief ","deputy commissioner ","deputy director ","deputy executive director ","derrick service unit operator ","design manager ","designer ","desk clerk ","desktop publisher ","detectives ","dietitian ","director ","dishwasher ",
                "dispatcher ","distributor ","division chief ","door-to-door sales worker ","drafter ","dressmaker ","drywall installer ","dynamic crm developer ","early childhood education consultant ","early intervention official ","earth driller ","economic analyst ","economist ","edge engineer ","e-discovery analyst ","editor ","editorial intern ","education administrator ","e-learning content developer ","electrical engineer ","electrical inspector ","electrical power-line installer ","electrician ","electronics engineer ","elementary teacher ","elevator installer ","elevator repairer ","eligibility interviewer ","eligibility worker ","emergency communications specialist ","emergency field logistics coordinator ","emergency manager ","emergency medical technician ","employee benefits coordinator ","employee relations specialist ","energy analyst ","energy policy advisor ","energy program analyst ","engineer in charge ","engineering manager ","engineering technician ","engraver ","entertainer ","enviromental assessment coordinator ","environmental compliance specialist ","environmental engineer ","environmental health & safety auditor ","environmental health & safety incident investigator ","environmental health & safety regional manager ","environmental program manager ","environmental project manager ","environmental review unit ","environmental scientist ","environmental specialist ","equipment training coordinator ","equity & special projects coordinator ","escalation line team leader ","etchers ","etl developer ","etl lead ","evaluation coordinator ","evaluation lead ","evaluation specialist ","examiner ","examining attorney ","executive agency counsel ","executive assistant ","executive director ","executive program coordinator ","executive project auditor ","executive project manager ","explosives worker ","extraction worker ","facilities clerk ","facilities management assistant ","facility operations custodial assistant ","family advocate ","farm ranch and other agricultural manager ","farmer  ","fence erector ","field  doctor ","field captain ","field inspector ","field operations training coordinator ","field outreach specialist ","field services technician ","field tech ","field/desktop technician ","file clerk ","financial analyst ","financial examiner ","financial manager ","financial reviewer ","financial specialist ","fingerprint unit specialist ","fire fighter ","fire inspector ","fire medical officer ","fiscal coordinator ","fiscal manager ","fish and game warden ","fisher ","fleet administrator ","fleet assistant ","fleet management assistant ","flight engineer ","food preparation worker ","food science technician ","food scientist ","food server ","food service manager ","forensic logistics specialist ","forensic pathology technician ","forester ","foster care program evaluator ","front end developer ","functional analyst ","funeral director ","funeral service worker ","furniture coordinator ","furniture finisher ","gaming manager ","gaming surveillance officer ","gardener ","general counsel ","general manager  ","general services associate ","geological engineer ","geological technician  ","geologist ","geoscientist ","gis analyst ","gis editor ","gis specialist ","glazier ","grader ","grant coordinator ","grant manager ","grant project manager ","graphic artist ","green infrastructure engineer ","greeter ","grounds maintenance worker ","groundskeeping worker ","group head ","hairdresser ","hairstylist ","hazardous materials removal worker ","head cook ","health adviser ","health diagnosing and treating practitioner ","health navigator ","health promoter ","hearing officer ","heat coordinator ","heating plant technician ","heavy vehicle and mobile equipment service technician ","help desk representative ","help desk technician ","high pressure plant tender ","highway maintenance worker ","highways and sewers inspector ","hiring plan analyst ","home health aide ","homeowner mortgage service manager ","host ","hostess ","hostler ","housekeeping cleaner ","hr payroll systems security architect ","hr specialist ","hro application support manager ","human resources administrative assistant ","human resources analyst ","human resources assistant ","human resources associate ","human resources college aide ","human resources coordinator ","human resources generalist ","human resources manager ","human resources specialist ","human resources support staff ","human resources ","human rights specialist ","hunter ","HVAC mechanic ","implementation coordinator ","incident commander ","industrial control technician ","industrial engineer ","industrial hygienist ","industrial production manager ","industrial program compliance analyst ","information clerk ","information representative ","information security architecture & engineering manager ","information security audit and compliance manager ","information security identity & access manager ","information security officer ","information systems and quality analyst ","inspector general ","inspector ","inspector  ","instructor ","instrumentation specialist ","insulation worker ","insurance claims and policy processing clerk ","insurance sales agent ","insurance underwriter ","intake investigator ","integration support engineer ","integrity control officer ","interagency exercise coordinator ","intergovernmental affairs intern ","intergovernmental outreach coordinator ","intergovernmental relations task force ","intern ","interpreter ","interviewer ","investigative auditor ","investigative consultant ","investigative manager ","investigator trainee ","investigator ","investment analyst ","investment officer ","ios developer ","ip telephony design engineer ","iron worker ","it - project manager ","it auditor ","it budget & contracts manager ","it business analyst ","it contract analyst ","it contracts specialist ","it helpdesk manager ","it infrastructure project manager "
                   ,"it operations clerk ","it relationship manager ","it security analyst ","it security officer ","it support technician ","jailer ","janitor ","janitorial worker ","java developer ","jeweler ","judge ","junior accountant ","junior financial planning and reporting analyst ","junior public health nurse ","junior z/os engineer ","juvenile implementation manager ","juvenile justice trainer ","kettle operator ","kiln operator ","kindergarten teacher ","labor law investigator ","labor relations attorney ","labor relations specialist ","labor standards investigator ","laboratory associate ","laboratory microbiologist ","laboratory technician ","landscape architect ","lead business analyst ","lead design engineer ","lead designer ","lead expense management analyst ","lead trainer ","learning management ","lease analyst ","leather workers ","legal assistant ","legal coordinator ","legal support worker ","legislator ","letterer and sign painter ","liaison ","librarian ","library assistant ","library technician ","licensed vocational nurse ","licensing clerk ","lifeguard ","linux administrator ","load officer ","loan counselor ","loan officer ","lobby attendant ","locksmith ","locomotive engineer ","lodging manager ","logging worker ","logistician ","low pressure boiler inspector ","machine operator ","machinist ","magistrate ","maid ","mail clerk ","mail machine operator ","mail superintendent ","mailroom control clerk ","mailroom section supervisor ","maintenance facilitator ","maintenance worker ","management analyst ","management auditor ","management systems specialist crm information systems manager ","manager ","mapping technician ","marine electronics technician ","marine engineer ","marine oiler mate ","market researcher ","marketing and sales manager ","massage therapist ","material moving worker ","materials engineer ","materials scientist ","mathematician ","measurer ","mechanical cost estimator ","mechanical engineer ","media worker ","mediation coordinator ","medical and health services manager ","medical assistant ","medical examiner assistant ","medical records and health information technician ","medical scientist ","medical technician ","medicolegal analyst ","medicolegal investigator ","meeting and convention planner ","mental health coordinator ","messenger ","metal work mechanic ","meter reader ","metrics and reporting engineer ","middle school teacher ","millwright ","mining engineer ","mining machine operator ","mobile device management architect ","molder ","motor vehicle operator ","museum technician ","musician ","natural sciences manager ","naval architect ","neigborhood resiliency specialist ","network administrator ","network engineer ","network field technician ","network services manager ","news analyst ","news and street vendor ","nuclear engineer ","nuclear technicians ","nurse home visitor ","nursing supervisor ","nutrition consultant ","nutritionist ","occupational therapist assistant ","occupational therapist ","office administrator ","office assistant ","office clerk ","office coordinator ","office machine aide ","office machine repairer ","office manager ","oil burner specialist ","ombudsperson ","on-line medical control physician ","open data lead analyst ","open market order coordinator ","operations associate ","operations coordinator ","operations customer service liaison ","operations desktop support technician ","operations manager ","optical network engineer ","optician ","optometrist ","order clerk ","order filler ","ordered deductions analyst ","ordnance handling expert ","outbreak data analyst ","outreach assistant ","outreach intern ","packager ","packer ","painter ","paperhanger ","paralegal aide ","paralegal assistant supervisor ","paralegal ","paramedic ","parking enforcement officer ","parking enforcement worker ","parking lot attendant ","parks enforcement patrol ","parts salesperson ","patternmaker ","payment analyst ","payroll data associate ","payroll liaison ","payroll supervisor ","payroll systems analyst ","penalty processing unit cashier ","peoplesoft developer ","performance management specialist ","performer ","permit records assistant ","personal and home care aide ","personal care and service worker ","personal financial advisor ","personnel associate ","personnel coordinator ","pest control aide ","pest control worker ","petroleum engineer ","petroleum technician ","pharmacist ","photogrammetrist ","photographer ","physical scientist ","physical therapist assistant ","physical therapist ","physician assistant ","physician ","physicist ","pile-driver operator","pipefitter ","pipelayer ","placement staff nurse ","plan examiner ","plan management specialist ","plasterer ","plumber ","plumber's helper ","plumbing and fire suppression engineer ","plumbing inspector ","podiatrist ","police administrative aide ","police officer ","police surgeon ","policeman ","policy advisor ","policy analyst ","policy writer ","port marine engineer ","portal support engineer ","portfolio manager ","postal office clerk ","postal service clerk ","postal service mail carrier ","postal service mail sorter ","postmaster ","postsecondary teacher ","power plant operator ","preplacement nurse practitioner ","prepress technician ","preschool teacher ","press officer ","presser ","principal administrative associate ","principle design manager ","private detective ","probation assistant ","probation officer trainee ","probation officer ","procedural justice coordinator ","process messenger ","process reform analyst intern ",
                "process server ","processor ","procurement analyst ","procurement clerk ","procurement contracting officer ","procurement specialist ","producer  ","product advisor ","product manager ","product owner ","product promoter ","product support specialist ","production support ","program administrator ","program analyst ","program and policy specialist ","program assistant ","program associate ","program contract analyst ","program coordinator for early childhood health ","program executive ","program management office assistant ","program manager ","program manager cadd coordinator ","program planner ","program specialist ","project coordinator ","project cost estimator ","project development coordinator ","project director ","project manager ","project manager  ","project planner ","project specialist ","projects coordinator ","property acquisition coordinator ","property manager ","prosecutor ","psychologist ","public health adviser ","public health assistant ","public health detailing specialist ","public health epidemiologist ","public health nurse ","public policy and training coordinator ","public records aide ","public relations manager ","public relations specialist ","public warning specialist ","pumping station operator ","purchasing agent - buyer ","purchasing agent ","purchasing manager ","qa analyst ","qa senior auditor ","quality assurance auditor ","quality assurance engineer ","quality assurance manager ","quality assurance project manager ","quality assurance specialist ","quality child care manager ","quality control case reviewer ","quality control senior engineer ","quality improvement specialist ","quality management specialist ","radiation emergency response specialist ","radiation therapist ","radio and telecommunications equipment installer ","radio operators","radio repair mechanic ","radio room operator ","railroad conductor ","rail-track laying and maintenance equipment operator ","rancher ","real estate agent ","real estate broker ","real estate developer ","real estate manager ","rebar workers ","receivables/payables analyst ","receptionist ","records officer ","records searcher ","recreation and fitness worker ","recreational therapist ","recruiting & onboarding coordinator ","recruitment assistant ","refuse and recyclable material collector ","regional field administrative assistant ","registered dental hygienist ","registered nurse ","regulatory and licensing specialist ","regulatory compliance agency attorney ","reinforcing iron worker ","religious worker ","remote learning team specialist ","rental clerk ","reporter ","research analyst ",
                "research assistant ","research scientist ","resident engineer ","residential advisor ","resiliency coordinator ","resiliency engineer ","resource specialist ","respiratory therapist ","retail buyer ","retail salesperson ","retail specialist ","revenue agent ","revenue assistant ","review unit supervisor ","rezoning manager ","rigger ","risk management ","roof bolter ","roofer ","roustabout ","safe event coordinator ","safety & equipment training specialist ","safety accident investigator ","safety auditor ","safety engineer ","sailor ","sales agent ","sales engineer ","sales representative ","samplers ","sanitation engineer ","scanning operator ","school health nurse practitioner ","school mental health consultant ","school mental health training coordinator ","scientist scientist ","searcher" ,"seasonal city park worker ","secondary school teacher ","secretary to division chief ","section chief ","securities commodities and financial services sales agent ","security administrator ","security and fire alarm systems installer ","security guard ","security officer ","security specialist ","security system administrator ","security systems engineer ","senior accountant ","senior advisor ","senior analyst ","senior application engineer ","senior architect cloud infrastructure design and engineering ","senior bi developer ","senior budget analyst ","senior business analyst ","senior case support associate ","senior civil service advisor ","senior construction manager ","senior cook ","senior counsel ","senior crm developer ","senior data analyst ","senior data scientist ","senior design engineer ","senior developer ","senior director it ","senior director ","senior early childhood education consultant ","senior electrical engineer ","senior engineer ","senior estimator ","senior exchange engineer ","senior field supervisor ","senior financial reporting investment analyst ","senior forester for restitution ","senior general counsel ","senior human capital partnership and performance management analyst ","senior human resources specialist ","senior inspector ","senior intergroup relations officer ","senior investigative auditor ","senior investment analyst ","senior it business analyst ","senior it project manager ","senior landscape architect ","senior mainframe programmer analyst ","senior mobile developer ","senior network engineer ","senior network specialist ","senior operations associate ","senior organizational development and training specialist ","senior payment analyst ","senior peoplesoft analyst ","senior police administrative aide ","senior policy advisor ","senior port engineer ","senior product manager ","senior program manager ","senior program officer ","senior programmer ","senior project analyst ","senior project controls specialist ","senior project engineer ","senior project leader ","senior project manager ","senior project officer ","senior public health inspector ","senior quality oversight analyst ","senior rackets investigator ","senior radio system network engineer ","senior research analyst ","senior resiliency planner ","senior safety accident investigator ","senior security analyst ","senior service desk agent ","senior staff electrical engineer ","senior stationary engineer ","senior technology analyst/coordinator ","senior title examiner ","senior trainer ","senior urban designer ","senior windows administrator ","septic tank servicer ","sergeant ","server support engineer ","service asset configuration manager ","service contract procurement analyst ","service desk agent ","service desk manager ","service station attendant ","sewer pipe cleaner ","sewer ","sewing machine operator ","sharepoint developer ","sheet metal worker ","sheriff's patrol officer ","ship engineer ","ship loaders","shuttle car operator ","singer ","small engine mechanic ","social science research assistant ","social scientist ","social service specialist ","social worker ","sociologist ","software developer ","solar energy project manager ","solutions director ","sorter ","space analyst ","space scientists ","special assistant to assistant deputy commissioner ","special assistant ","special consultant ","special education teacher ","special examiner ","special projects manager ","special underwriting project manager ","speech-language pathologist ","spending analyst ","sql/oracle database administrator ","sr. analyst ","sr. internal auditor ","staff analyst ","staff analyst ii ","staff analyst level ii ","staff analyst ","staff attorney ","staff auditor ","staff counsel ","staff engineer"," architect ","staff photographer ","mechanical","electrical"
,"standards specialist ","stationary engineer ","statistical analyst ","statistical assistant ","statistician ","steamfitter ","steel workers ","stock clerk ","stock worker level ii ","stonemason ","storage engineer ","stormwater program coordinator ","strategic account manager ","strategic initiatives coordinator ","strategic operations policy analyst ","strategic partnership liaison ","strategic planning associate ","street ambassador ","structural engineer ","structural iron and steel worker ","stucco mason ","student data analyst ","student legal specialist ","summer college intern ","summer communications assistant ","summer graduate intern ","summer it intern ","supervising health nurse ","supervising housing groundskeeper ","supervising nurse ","supervising physician ","supervising public health advisor ","supervisor ","support engineer ","support worker ","surgeon ","survey researcher ","surveyor ","switchboard operator ","system administrator ","systems access management engineer ","systems administrator - computer software ","systems analyst ","tailor ","talent acquisition specialist ","taper ","tax examiner ","tax incentives director ","tax preparer ","taxi driver ","teacher assistant ","team coach team leader ","technical investigator ","technical lead ","technical project manager/product owner ","technical sales manager ","technical solutions professional ","technical writer ","technician ","telecommunications associate ","telemarketer ","telephone operator ","teller ","temporary assistant urban designer ","temporary painter ","tender ","testifier/searcher ","testing lead ","therapist ","thermostat repairer ","third party services analyst ","threat analyst timekeeper ","ticket taker ","tile installer ","timekeeper ","timekeeping specialist ","tool grinder ","tort attorney ","tour guide ","tracking and monitoring data analyst ","trainer & curriculum development specialist ","trainer ","training specialists ","transit and railroad policeman ","translator ","transportation specialist ","transportation storage and distribution manager ","trapper ","trauma-informed early care and education case supervisor ","travel agent ","travel guide ","triage coordinator ","triage nurse ","triage supervisor ","trial preparation assistant ","truck driver ","typist ","umpire ","unit clerk ","unit manager ","unix system administrator ","unix/linux systems lead ","upholsterer ","urban and regional planner ","urban technology architect ","urban technology security researcher ","usher ","valve installer ","vessel construction manager ","veterinarian ","virtual systems engineer ","vision screening assistant ","vp for project management ","wage subsidy processing clerk ","waiter ","waitress ","water resources analyst ","water use inspector ","waterfront facilities engineer ","waterfront project manager ","watershed maintainer ","web application developer ","webform team member ","weigher ","welder ","wellness advocate supervisor ","wellness advocate ","windows administrator ","wireless coordinator ","woodworker ","word processor  ","workforce management analyst ","workforce planning intern ","writer ","x-ray technician ","yardmaster ","youth initiatives lead advisor"," helper ","Operations"])


def get_ex_job_title(text_data,req_data):
     title = re.search(f"({part_or_full}).*?({jobs1}|{jobs2})",text_data,re.DOTALL|re.IGNORECASE)
     if title:
         return title.group(2).upper()
     else:
         title = re.search(f"({jobs1}|{jobs2})", req_data, re.DOTALL | re.IGNORECASE)
         if title:
             return title.group(1).upper()
         else:
             return None
             # title = re.search(f"({requirements_tag})[\s\S]*({jobs1}|{jobs2})", text_data, re.DOTALL | re.IGNORECASE)
             # if title:
             #     return title.group(2).upper()
             # else:
             #    return None


def get_number(string_number):
    dict_numbers={"ZERO":0,"ONE":1,"TWO":2,"THREE":3,"FOUR":4,"FIVE":5,"SIX":6,"SEVEN":7,"EIGHT":8,"NINE":9}
    return dict_numbers[string_number]


ex_job_ls=[]
majors_ls=[]
exam_type_ls=[]
annual_salary_list=[]
schools=[]
course_counts_ls=[]
course_length_ls=[]
file_names=[]
license_list=[]
job_titles = []
class_codes = []
job_duties = []
requirements=[]
onen_date_ls=[]
open_dates = []
exp_list=[]
parttime_or_fulltime=[]
eduacation_years_ls=[]


input_folder = "/kaggle/input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins"

file_list = os.listdir(input_folder)
for file_name in file_list:
    
    file_path = os.path.join(input_folder, file_name)
    
    file_names.append(file_path)
    with open(file_path, 'r', encoding='iso-8859-1') as file:
        text_data = file.read()
    
        
# all_jobs_files = glob.glob(path2)
# for file in all_jobs_files:
#     with open(file) as job:
#         text_data = job.read()

        job_title=get_job_title(text_data)
        class_code=get_class_code(text_data)
        duties=get_job_duties(text_data)
        education_level=get_education(text_data)
        fullorpart=get_part_or_full_time(text_data)
        open_date=get_open_date(text_data)
        license=get_license(text_data)
        req=get_reqs(text_data)
        annual_salary=get_annual_salary(text_data)
        experience=get_experience(text_data)
        edu_years=get_education_years(text_data)
        exam_type=get_exam_type(text_data)
        major=get_major(text_data)
        counts=get_course_counts(text_data)
        length=get_course_length(text_data,counts)
        ex_job=get_ex_job_title(text_data,req)



    class_codes.append(class_code)
    job_titles.append(job_title)
    job_duties.append(duties)
    open_dates.append(open_date)
    schools.append(education_level)
    license_list.append(license)
    parttime_or_fulltime.append(fullorpart)
    requirements.append(req)
    annual_salary_list.append(annual_salary)
    exp_list.append(experience)
    eduacation_years_ls.append(edu_years)
    exam_type_ls.append(exam_type)
    majors_ls.append(major)
    course_counts_ls.append(counts)
    course_length_ls.append(length)
    ex_job_ls.append(ex_job)


bulletins_dict={"File Name":file_names,"Job Title":job_titles,"EXP Job Title":ex_job_ls,"Job Number":class_codes,"Job Duties":job_duties
    ,"Salary Range":annual_salary_list,"Open Date":open_dates,"Experience Years Required":exp_list,"Education Years":eduacation_years_ls
    ,"PART-TIME OR FULL-TIME":parttime_or_fulltime,"School Type":schools,"Educational Major":majors_ls,
           "Required License":license_list,"Exam Type":exam_type_ls,"Course Counts":course_counts_ls,
           "Course Length In Months":course_length_ls,"Job Requirements":requirements}

bulletins=pd.DataFrame(bulletins_dict)
print(bulletins.head())


import warnings
warnings.filterwarnings('ignore')

bulletins


bulletins.info()


#숫자형 데이터
bulletins.describe()


#범주형 데이터에 대한 정보 출력

bulletins.describe(exclude='number')


duplicate_counts = bulletins['Job Title'].value_counts()
duplicate_counts = duplicate_counts[duplicate_counts > 1]  # 중복된 것만 보기

print(duplicate_counts)

bulletins[bulletins['Job Title'].duplicated(keep=False)].sort_values(by='Job Title')


print(bulletins.isna().sum())


bulletins["EXP Job Title"].value_counts().head(30).plot(kind='barh', title="Top 30 Jobs Needs To Be Experienced In")


bulletins["Educational Major"].value_counts().head(30).plot(kind='bar',title="Top 30 Educational Majors Required")


# bulletins["EXP Job Title"].value_counts().plot(kind='bar')

print("These are the top 15 expericed job required")
print(bulletins["EXP Job Title"].value_counts().head(15))


# 이전 경력, 전공, 스쿨 타입과 같은 자격요건을 요구하지 않는 경우 Not-required로 표시되게 함

bulletins_df=bulletins.copy()
bulletins[["EXP Job Title","Educational Major","School Type"]]=bulletins[["EXP Job Title","Educational Major","School Type"]].fillna("Not-Required")

print(bulletins.isna().sum())


# 요구되는 교육 기간, 경력 기간이 없을 경우 0으로 표시 (숫자 데이터)
bulletins[["Education Years","Experience Years Required"]]=bulletins[["Education Years","Experience Years Required"]].fillna(0)
print(bulletins.isna().sum())


# 같은 의미인데도 데이터가 중복되는 것이 있음
print(bulletins["PART-TIME OR FULL-TIME"].unique())

bulletins["PART-TIME OR FULL-TIME"]=bulletins["PART-TIME OR FULL-TIME"].replace("FULL TIME","FULL-TIME")

print(bulletins["PART-TIME OR FULL-TIME"].unique())

print(bulletins["PART-TIME OR FULL-TIME"].value_counts())


bulletins["PART-TIME OR FULL-TIME"].value_counts().plot(kind="bar")


#파트타임 데이터는 12건으로, 전체 데이터 수와 비교하면 매우 적은 수치이므로 제거

cleaned_bulletins=bulletins.drop("PART-TIME OR FULL-TIME",axis=1)
print(cleaned_bulletins.info())


print(cleaned_bulletins.isna().sum())


# The School Type should be limited to 3 categories : COLLEGE OR UNIVERSITY , APPRENTICESHIP , HIGH SCHOOL
print(cleaned_bulletins["School Type"].value_counts())


school_types_list=[]

# schools 리스트에 포함된 각 항목을 검사
# 해당 항목이 학교 유형을 나타내는 문자열인지 확인
# 특정 키워드를 기반으로 학교 유형을 분류

for tp in schools:
    
    # tp가 문자열(str)인지 확인 (결과는 True or False)
    # if절로 str_type에 저장된 tp가 문자열이 아닌 경우 "Not-Required"로 처리
    
    str_type = isinstance(tp, str)
    
    if str_type:

        t=re.search("UNIVERSITY|COLLEGE|BACHELOR|MASTER|DIPLOMA",tp)
        if t:
            school_types_list.append("COLLEGE OR UNIVERSITY")
        else:
            t=re.search("SCHOOL",tp)
            if t:
                school_types_list.append("HIGH SCHOOL")
            else:
                t=re.search("APPRENTICESHIP",tp)
                if t:
                    school_types_list.append("APPRENTICESHIP")
                else:
                    school_types_list.append("Not-Required")
    else:
        school_types_list.append("Not-Required")

                
cleaned_bulletins["School Types"]=school_types_list
print(cleaned_bulletins["School Types"].value_counts())


cleaned_bulletins["School Types"].value_counts().plot(kind="bar",title="Requested School Types For Bulletins")


"""
"flat"이 포함된 단일 연봉 값
예: "$50,000 flat"
처리 방법: 숫자 그대로 사용

공백으로 구분된 두 개의 연봉 값
예: "$40,000 $50,000"
처리 방법: 두 값의 평균 사용

단일 연봉 값
예: "$45,000"
처리 방법: 숫자 그대로 사용

범위로 표시된 연봉 값 (하지만 코드에서 직접 처리 안 함)
예: "$40,000-$50,000", "$40,000~$50,000"
코드에서 이 경우를 직접 처리하지 않음!
sal1과 sal2가 특정 위치의 공백을 기준으로 숫자를 가져오므로

숫자만 포함된 경우 (달러 기호 없이 숫자만 존재)
예: "40,000"
처리 방법: 숫자 그대로 사용

숫자가 전혀 없는 경우
예: "Negotiable", "Depends on experience"
처리 방법: None으로 저장
"""

### AI를 활용하여 연봉 데이터에 있는 복잡한 조건을 정리한 것

SR = bulletins.copy()

# Salary Range 칼럼 이름 확인 (필요시 수정)
salary_column_name = "Salary Range"  # 실제 칼럼 이름으로 변경 필요

def extract_final_salary(salary_range):
    """ 연봉 데이터를 기준에 맞게 변환 """
    if pd.isna(salary_range) or salary_range.strip() == "":
        return None

    salary_range = salary_range.lower()  # 소문자로 변환 (flat 등 처리 위해)
    
    # "flat"이 포함된 경우 → 숫자 그대로 사용
    if "flat" in salary_range:
        salary_values = re.findall(r'\d{1,3}(?:,\d{3})*', salary_range)
        if salary_values:
            return int(salary_values[0].replace(',', ''))

    # 공백으로 구분된 두 개의 연봉 값 → 평균 사용
    salary_values = re.findall(r'\d{1,3}(?:,\d{3})*', salary_range)
    salary_values = [int(s.replace(',', '')) for s in salary_values]

    if not salary_values:
        return None  # 숫자가 없는 경우

    if len(salary_values) == 1:
        return salary_values[0]  # 단일 값은 그대로 반환

    if len(salary_values) == 2:
        return sum(salary_values) // 2  # 두 값의 평균 반환

    # 여러 개의 값이 있는 경우 → 최소, 최대의 평균 반환
    return (min(salary_values) + max(salary_values)) // 2

# Salary Range 칼럼 변환 (SR 데이터프레임에서 직접 적용)
SR["Flat Salary"] = SR[salary_column_name].apply(extract_final_salary)

print("✅ 연봉 변환 완료! SR['Flat Salary']에 결과가 저장되었습니다.")

SR


avg_ls=[]

for salary in annual_salary_list:
    string_type = isinstance(salary, str)

#string_type이 문자열이 아닌 경우 (False) 바로 None으로 하고,
# 문자열인 경우 그 다음부터 IF절 새로 시작
    
    if string_type:
        
        flat_salary=re.search("\$(\d+,\d+)[\s\S]*flat",salary,re.IGNORECASE)
        
        if flat_salary:
            
            flat_salary = flat_salary.group(1)
            flat_salary = int((re.sub(",", "",flat_salary)))
            avg_ls.append(flat_salary)

        else:
            # \$는 달러 기호를 그대로 찾기 위한 것
            # 원래 의미는 문자열의 끝을 의미하는 와일드카드이기 때문
            
            sal1 = re.search("\$(\d+,\d+) ", salary)
            
            if sal1:
                sal1 = sal1.group(1)
                sal1 = int((re.sub(",", "", sal1)))

            sal2 = re.search(" \$(\d+,\d+)", salary)
            
            if sal2:
                sal2 = sal2.group(1)
                sal2 = int((re.sub(",", "", sal2)))
            
            if (sal1 and sal2):
                avg = ((sal1 + sal2) / 2)
                avg_ls.append(avg)
            
            else:
                # 연봉 데이터가 ?,???, ?,???으로 나와 있는 경우
                sal3=re.search("\d+,\d+",salary)
                
                if sal3:
                    sal3 = sal3.group()
                    sal3 = int((re.sub(",", "", sal3)))
                    avg_ls.append(sal3)
                
                else:
                    avg_ls.append(None)

    else:
        avg_ls.append(None)


cleaned_bulletins["AVG Salary"]=avg_ls
print(cleaned_bulletins.head())
print(cleaned_bulletins.info())


cleaned_bulletins=cleaned_bulletins.drop('Salary Range',axis=1)


print(cleaned_bulletins.columns)


missing_salary=(cleaned_bulletins[cleaned_bulletins["AVG Salary"].isnull()])
print(missing_salary)


missing_salary


airport = cleaned_bulletins[cleaned_bulletins["Educational Major"]=="AIRPORT"]
airport_avg_salary=airport["AVG Salary"].mean()

mainten=cleaned_bulletins[cleaned_bulletins["Educational Major"]=="MAINTENANCE"]
mainten_avg_salary=mainten["AVG Salary"].mean()

print(airport_avg_salary)
print(mainten_avg_salary)


cleaned_bulletins.loc[367,"AVG Salary"]=airport_avg_salary
cleaned_bulletins.loc[284,"AVG Salary"]=mainten_avg_salary


cleaned_bulletins["AVG Salary"].isnull().sum()


print(cleaned_bulletins["AVG Salary"].describe())


plt.figure(figsize=(16,8))

plt.subplot(2,2,1)
sns.distplot(cleaned_bulletins["AVG Salary"])

plt.subplot(2,2,2)
sns.boxplot(cleaned_bulletins["AVG Salary"])

plt.show()


# 정규화
# 최소값(min)은 0이 되고, 최대값(max)은 1이 됨
# 중간 값들은 0과 1 사이의 소수로 변환
    
normalized_salary=(cleaned_bulletins["AVG Salary"]-cleaned_bulletins["AVG Salary"].min())/(cleaned_bulletins["AVG Salary"].max()-cleaned_bulletins["AVG Salary"].min())
cleaned_bulletins["Normalized AVG Salary"]=normalized_salary

print(cleaned_bulletins["Normalized AVG Salary"].describe())


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(cleaned_bulletins["Normalized AVG Salary"])
plt.subplot(2,2,2)
sns.boxplot(cleaned_bulletins["Normalized AVG Salary"])
plt.show()


Q1 = np.percentile(cleaned_bulletins["Normalized AVG Salary"], 25, interpolation = 'midpoint')
Q3 = np.percentile(cleaned_bulletins["Normalized AVG Salary"], 75, interpolation = 'midpoint')
IQR = Q3 - Q1
print(IQR)


upper = np.where(cleaned_bulletins["Normalized AVG Salary"] >= (Q3+1.5*IQR))
lower = np.where(cleaned_bulletins["Normalized AVG Salary"] <= (Q1-1.5*IQR))

print(upper)
print(lower)



cleaned_bulletins.drop(upper[0], inplace = True)
cleaned_bulletins.drop(lower[0], inplace = True)


print("New Shape: ", cleaned_bulletins.shape)


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(cleaned_bulletins["AVG Salary"])
plt.title("AVG Salary after removing outliers")

plt.subplot(2,2,2)
sns.boxplot(cleaned_bulletins["AVG Salary"])
plt.title("AVG Salary after removing outliers")
plt.show()


print(cleaned_bulletins["Job Number"].nunique())
print(cleaned_bulletins['Job Number'].value_counts().head(8))


cleaned_bulletins[cleaned_bulletins['Job Number'].isin(["4123","1249","5885","3980","3753"])].sort_values('Job Number')


cleaned_bulletins=cleaned_bulletins.drop([666, 185, 261, 344, 38])
print(cleaned_bulletins['Job Number'].value_counts().head())
print(cleaned_bulletins[cleaned_bulletins['Job Number'].isin(["4123","3980","1249","3753","5885"])].sort_values('Job Number'))
print("New Shape :",cleaned_bulletins.shape)



cleaned_bulletins=cleaned_bulletins.drop(['Job Number','File Name'],axis=1)
print(cleaned_bulletins.info())


print(cleaned_bulletins.isna().sum())


print(cleaned_bulletins["Exam Type"].value_counts())


cleaned_bulletins["Exam Type"].value_counts().plot(kind="bar")


cleaned_bulletins[cleaned_bulletins["Exam Type"].isnull()]


maintenance=cleaned_bulletins[cleaned_bulletins["Educational Major"]=="MAINTENANCE"]

#Exam Type의 최빈값 중 가장 첫번째 값
type_mode=(maintenance["Exam Type"].mode()[0])

print(type_mode)



cleaned_bulletins["Exam Type"]=cleaned_bulletins["Exam Type"].fillna(type_mode)
print(cleaned_bulletins.isna().sum())


cleaned_bulletins['Open Date']


cleaned_bulletins['Open Date'] = pd.to_datetime(cleaned_bulletins['Open Date'])
cleaned_bulletins = cleaned_bulletins.astype({"Experience Years Required":'int', "Education Years":'int'}) 

print(cleaned_bulletins.dtypes)
print(cleaned_bulletins.head())


#Q1: A non-experienced job seeker wants to know what is the best job for him?

# Non-expericed job seeker will looks for less requirements bulletins
# which doesn't require a Experience job or a number or experice years

# exp job is "Not-Required" and exp years is zero

no_exp = cleaned_bulletins[(cleaned_bulletins["EXP Job Title"]=="Not-Required") & (cleaned_bulletins["Experience Years Required"]==0)]

# For best choice, job seeker will aim to best avg salary
no_exp = no_exp.sort_values(by="AVG Salary",ascending=False)

print("The Best bulletin that requires no experienced job and no experience years and get the best average salary is : ")
no_exp.head(1)


# Q2.
# A concerned parents contacts you to tell them if there is a spicific school type
# that would guaranteed for their child a good future if you can tell, and which is it ?


# AVG Salary for each of a school type

SAU = cleaned_bulletins[(cleaned_bulletins["School Types"]!="Not-Required")]
type_and_salary = SAU["AVG Salary"].groupby(by=SAU["School Types"]).mean()

print(type_and_salary)

type_and_salary.plot(kind="bar",title="AVG Salary For Each School Type")


#Q3: What the best time in the year to be ready for a job applying in any experience level?

# subset only the day and the month from each date
# to see if there is specific day and month usually opening at it

cleaned_bulletins['Month-Day'] = pd.to_datetime(cleaned_bulletins['Open Date']).dt.strftime('%m-%d')

print(cleaned_bulletins['Month-Day'].value_counts())
cleaned_bulletins = cleaned_bulletins.drop('Month-Day',axis=1)




# Q4: Is experience more important than educational level?
# compare between requirements between education level"school type" and experience required .

no_school=(cleaned_bulletins[cleaned_bulletins["School Types"]=="Not-Required"]["School Types"].value_counts())
no_exp_job=(cleaned_bulletins[cleaned_bulletins["EXP Job Title"]=="Not-Required"]["EXP Job Title"].value_counts())
no_exp_years=(cleaned_bulletins[cleaned_bulletins["Experience Years Required"]==0]["Experience Years Required"].value_counts())

print("Number of bulletins with no school type requirement : ")
print(no_school)
print("\n")
print("Number of bulletins with no experience job requirement : ")
print(no_exp_job)
print("\n")
print("Number of bulletins with no experience years requirement : ")
print(no_exp_years)


#Q5: Which fresh grade job that will guarantee many job offers in the future for him?

print(cleaned_bulletins["EXP Job Title"].value_counts())
cleaned_bulletins["EXP Job Title"].value_counts().head(4).plot(kind='bar',title='Top 4 EXP JOBS REQUIRED')



# Q6: "The City of Los Angeles does not discriminate on the basis of race, religion, national origin, sex, age, marital status, sexual orientation, gender identity, gender expression, disability, creed, color, ancestry, medical condition (cancer), or Acquired Immune Deficiency Syndrome. AN EQUAL EMPLOYMENT OPPORTUNITY EMPLOYER "
# The above statment is qouted from one of the job descriptions.
# Based on the jobs requirments you structred above do you think that LA governate may bais a little for men over women or the applicant marital status for example may effect his chance to gain the job?


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
bulletins_df["Educational Major"].value_counts().head(5).plot(kind='bar',title="Top 5 Educational Majors Required")
plt.subplot(2,2,2)
bulletins_df["EXP Job Title"].value_counts().head(5).plot(kind='bar',title='Top 5 EXP JOBS REQUIRED')
plt.show()


# Q7: The city need an advice based on your analysis,
# build a new schools for more fresh non-experienced workers 
# or encorge the work environment to help the workers to get promotions?

def req_schooll(value):
    if value!="Not-Required":
        return "R"
    else:
        return "NR"

def req_expp(value):
    if value!=0:
        return "R"
    else:
        return "NR"
    
cleaned_bulletins['School Required'] = cleaned_bulletins['School Types'].map(req_schooll)
cleaned_bulletins['Experience Required'] = cleaned_bulletins["Experience Years Required"].map(req_expp)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
cleaned_bulletins['School Required'].value_counts().plot(kind='bar',title="Required School Or Not")
plt.subplot(2,2,2)
cleaned_bulletins['Experience Required'].value_counts().plot(kind='bar',title='Required Experience Or Not')
plt.show()

cleaned_bulletins=cleaned_bulletins.drop(['Experience Required','School Required'],axis=1)


# Q8: What is the average salary for worker with a driver licence?

salary_and_licence = cleaned_bulletins["AVG Salary"].groupby(by=cleaned_bulletins["Required License"]).mean()
print(salary_and_licence)

salary_and_licence.plot(kind="bar",title="AVG SALARY FOR R/NR DRIVER LICENCE")


# Q9: Give a full statistical description for all numrical data columns
#     including all insights and needed figures to visualize them.

cleaned_bulletins.describe()


cleaned_bulletins["Experience Years Required"].value_counts().plot(kind='bar',title='Experience Years Required')


cleaned_bulletins["Education Years"].value_counts().plot(kind='bar',title='Education Years Required')


cleaned_bulletins["Course Counts"].value_counts().plot(kind='bar',title='Course Counts Required')


cleaned_bulletins["Course Length In Months"].value_counts().plot(kind='bar',title='Course Length In Months Required')


sns.distplot(cleaned_bulletins["AVG Salary"],kde=True,color="g")
plt.title("Average Salary")
plt.show()


# Q11: Give a full statistical description for the categorical data columns
#      that can be descriped including all insights and needed figures to visualize them.

cleaned_bulletins.describe(exclude="number")


cleaned_bulletins.groupby('School Types').size().plot(kind='pie', autopct='%.2f',ylabel="",title="School Types Percentage")


cleaned_bulletins.groupby('Exam Type').size().plot(kind='pie', autopct='%.2f',ylabel="",title="Exam Type Percentage")


cleaned_bulletins.groupby('Required License').size().plot(kind='pie', autopct='%.2f',ylabel="",title="Required License Jobs Percentage")


cleaned_bulletins["EXP Job Title"].value_counts().head(30).plot(kind='bar',title='Top 30 Experience Job EXP Required Titles')


cleaned_bulletins["Educational Major"].value_counts().head(30).plot(kind='bar',title='Top 30 Educational Major Required')



cleaned_bulletins["Open Date"].value_counts().head(10).plot(kind='bar',title='Top 10 Open dates for applying to jobs')


from wordcloud import WordCloud  
import matplotlib.pyplot as plt  

text = " ".join(cleaned_bulletins["EXP Job Title"].dropna())  
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)  

plt.figure(figsize=(10, 5))  
plt.imshow(wordcloud, interpolation='bilinear')  
plt.axis('off')  
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(
    x=cleaned_bulletins["EXP Job Title"].value_counts().head(30).values,
    y=cleaned_bulletins["EXP Job Title"].value_counts().head(30).index,
    palette="viridis"
)
plt.title("Top 30 Jobs Needs To Be Experienced In", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Job Title")
plt.show()


cleaned_bulletins.info()


import matplotlib.ticker as ticker

plt.figure(figsize=(10, 5))
sns.histplot(cleaned_bulletins["AVG Salary"], bins=30, kde=True, color='blue')
plt.title("Salary Distribution", fontsize=14)
plt.xlabel("AVG Salary")
plt.ylabel("Frequency")

# X축에 쉼표 추가
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.show()



cleaned_bulletins["Open Date"] = pd.to_datetime(cleaned_bulletins["Open Date"])  # 날짜 변환
cleaned_bulletins["Year-Month"] = cleaned_bulletins["Open Date"].dt.to_period("M")  # 연월 추출

job_trend = cleaned_bulletins.groupby("Year-Month").size()

plt.figure(figsize=(12, 6))
job_trend.plot(kind='line', marker='o', color='purple')
plt.title("Job Posting Trends Over Time", fontsize=14)
plt.xlabel("Year-Month")
plt.ylabel("Number of Job Postings")
plt.xticks(rotation=45)
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x="School Type", y="AVG Salary", data=cleaned_bulletins, palette="Set2")

plt.xticks(rotation=45)
plt.title("Salary by Required Education Level", fontsize=14)
plt.xlabel("Education Level")
plt.ylabel("AVG Salary")

# Y축에 1,000 단위 쉼표 추가
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=cleaned_bulletins,
    x="Experience Years Required",
    y="AVG Salary",
    alpha=0.7
)
plt.title("Experience vs Salary", fontsize=14)
plt.xlabel("Experience Years Required")
plt.ylabel("AVG Salary")

plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.show()


top_jobs = cleaned_bulletins.groupby("Job Title")["AVG Salary"].mean().sort_values(ascending=False).head(30)

plt.figure(figsize=(10, 12))
sns.barplot(y=top_jobs.index, x=top_jobs.values, palette="coolwarm")
plt.xlabel("Average Salary")
plt.ylabel("Job Title")
plt.title("Top 30 Highest Paying Jobs", fontsize=14)

plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.show()



plt.figure(figsize=(10, 6))
sns.regplot(x="Experience Years Required", y="AVG Salary", data=cleaned_bulletins, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("Experience Required vs. Salary")
plt.xlabel("Experience Years Required")
plt.ylabel("AVG Salary")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.show()


plt.figure(figsize=(12, 6))
sns.violinplot(x="Experience Years Required", y="AVG Salary", data=cleaned_bulletins, palette="muted")
plt.title("Salary Distribution by Experience Level", fontsize=14)
plt.xlabel("Experience Years Required")
plt.ylabel("AVG Salary")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))  # 1,000 단위 콤마 추가
plt.show()



# Educational Major별 평균 연봉 계산
top_majors = cleaned_bulletins.groupby("Educational Major")["AVG Salary"].mean().sort_values(ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_majors.values, y=top_majors.index, palette="viridis")  # 가로 막대그래프로 변경
plt.title("Top 20 Highest Paying Educational Majors", fontsize=14)
plt.xlabel("Average Salary")
plt.ylabel("Educational Major")
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))  # 1,000 단위 콤마 추가
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# 전공별 평균 연봉 계산
major_salary = cleaned_bulletins.groupby("Educational Major")["AVG Salary"].mean()

# 전체 평균 연봉
overall_avg_salary = cleaned_bulletins["AVG Salary"].mean()

# 전공별 평균 연봉 vs 전체 평균 연봉 차이
major_salary_diff = major_salary - overall_avg_salary

# 데이터 정렬 (높은 값부터)
major_salary_diff = major_salary_diff.sort_values()

# 가로 바 차트로 변경
plt.figure(figsize=(10, 12))  # 가로 X, 세로 Y 크기 조절
sns.barplot(y=major_salary_diff.index, x=major_salary_diff.values, palette="RdYlGn")

# 제목 및 축 레이블
plt.title("Impact of Educational Major on Salary (vs Overall Average)", fontsize=14)
plt.xlabel("Salary Difference from Overall Average")
plt.ylabel("Educational Major")

# 1,000 단위 콤마 추가
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))  

plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

# 'Educational Major'별 'Job Title'별 평균 연봉 계산
major_job_salary = cleaned_bulletins.groupby(["Educational Major", "Job Title"])["AVG Salary"].mean().reset_index()

# 모든 전공 리스트
majors = major_job_salary["Educational Major"].unique()

# 각 전공별 직무 개수 계산
job_counts = major_job_salary.groupby("Educational Major")["Job Title"].count()

# 서브플롯 개수 설정 (가로 2개씩 배치)
cols = 2
rows = (len(majors) // cols) + (len(majors) % cols > 0)

# 각 전공별 직무 개수를 고려하여 동적으로 높이 조정
row_heights = [max(1, job_counts[major] // 5) for major in majors]  # 직무 개수 많을수록 높이 증가
total_height = sum(row_heights)  # 전체 그래프 높이

fig, axes = plt.subplots(rows, cols, figsize=(20, total_height))  # 동적 높이 설정
axes = axes.flatten()

# 각 전공별로 그래프 그리기
for i, major in enumerate(majors):
    ax = axes[i]
    selected_major_jobs = major_job_salary[major_job_salary["Educational Major"] == major]
    
    sns.barplot(x="AVG Salary", y="Job Title", data=selected_major_jobs, palette="coolwarm", ax=ax)
    ax.set_title(f"Job Titles & Salaries for {major}", fontsize=12)
    ax.set_xlabel("Average Salary")
    ax.set_ylabel("Job Title")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))  # 1,000 단위 콤마 추가

# 불필요한 빈 서브플롯 제거
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



# 'Educational Major'별 'Job Title'별 평균 연봉 계산
major_job_salary = cleaned_bulletins.groupby(["Educational Major", "Job Title"])["AVG Salary"].mean().reset_index()

# 특정 전공 선택
selected_major = "FINANCE"
selected_major_jobs = major_job_salary[major_job_salary["Educational Major"] == selected_major]

# 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x="AVG Salary", y="Job Title", data=selected_major_jobs, palette="coolwarm")
plt.title(f"Job Titles & Salaries for {selected_major} Major", fontsize=14)
plt.xlabel("Average Salary")
plt.ylabel("Job Title")
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))  # 1,000 단위 콤마 추가
plt.show()


# 'Educational Major'별 'Job Title'별 평균 연봉 계산
major_job_salary = cleaned_bulletins.groupby(["Educational Major", "Job Title"])["AVG Salary"].mean().reset_index()

# 특정 전공 선택
selected_major = "MAINTENANCE"
selected_major_jobs = major_job_salary[major_job_salary["Educational Major"] == selected_major]

# 시각화
plt.figure(figsize=(12, 40))
sns.barplot(x="AVG Salary", y="Job Title", data=selected_major_jobs, palette="coolwarm")
plt.title(f"Job Titles & Salaries for {selected_major} Major", fontsize=14)
plt.xlabel("Average Salary")
plt.ylabel("Job Title")
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))  # 1,000 단위 콤마 추가
plt.show()


# 전체 평균 연봉 계산
overall_avg_salary = cleaned_bulletins["AVG Salary"].mean()

# 직무별 평균 연봉 계산
job_salary_diff = cleaned_bulletins.groupby("Job Title")["AVG Salary"].mean() - overall_avg_salary

# 연봉을 가장 많이 끌어올린 상위 10개 직무
top_jobs = job_salary_diff.sort_values(ascending=False).head(10)

# 연봉을 가장 많이 낮춘 하위 10개 직무
bottom_jobs = job_salary_diff.sort_values(ascending=True).head(10)

# 상위/하위 10개 직무를 합쳐서 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x=job_salary_diff.loc[top_jobs.index.append(bottom_jobs.index)].values, 
            y=job_salary_diff.loc[top_jobs.index.append(bottom_jobs.index)].index, 
            palette=["darkred" if v < 0 else "darkblue" for v in job_salary_diff.loc[top_jobs.index.append(bottom_jobs.index)].values])

plt.axvline(x=0, color="gray", linestyle="--")  # 전체 평균 연봉 기준선
plt.title("Job Titles with Highest & Lowest Impact on Average Salary", fontsize=14)
plt.xlabel("Difference from Overall Average Salary")
plt.ylabel("Job Title")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# 'Open Date'에서 연도와 월 추출
cleaned_bulletins["Year"] = cleaned_bulletins["Open Date"].dt.year
cleaned_bulletins["Month"] = cleaned_bulletins["Open Date"].dt.month

# 연도별 월별 공고 개수 집계
monthly_counts = cleaned_bulletins.groupby(["Year", "Month"]).size().reset_index(name="Count")

# 월별 평균 공고 개수 계산
monthly_avg = monthly_counts.groupby("Month")["Count"].mean()

# 시각화
plt.figure(figsize=(10, 5))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette="coolwarm")

plt.title("Average Number of Job Postings by Month", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Average Number of Job Postings")

# x축을 1~12월 강제 표시
plt.xticks(range(1, 13))

plt.show()

# 상위 5개 월 출력
top_5_months = monthly_avg.nlargest(3)
print(top_5_months)


# 'Open Date'에서 연도와 월 추출
cleaned_bulletins["Year"] = cleaned_bulletins["Open Date"].dt.year
cleaned_bulletins["Month"] = cleaned_bulletins["Open Date"].dt.month

# 연도별 월별 공고 개수 집계
monthly_counts = cleaned_bulletins.groupby(["Year", "Month"]).size().reset_index(name="Count")

# 월별 평균 공고 개수 계산
monthly_avg = monthly_counts.groupby("Month")["Count"].mean()
monthly_avg_sorted = monthly_avg.sort_values(ascending=False)

# 시각화
plt.figure(figsize=(10, 5))
sns.barplot(x=monthly_avg_sorted.index, y=monthly_avg_sorted.values, palette="coolwarm", order = monthly_avg_sorted.index)

plt.title("Number of Job Postings by Month (Sorted)", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Average Number of Job Postings")

plt.show()

