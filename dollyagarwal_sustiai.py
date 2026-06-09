# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# import logging
# import os

# # Clean up any previous logs
# for log_file in ["logger.log", "web.log", "tunnel.log"]:
#     if os.path.exists(log_file):
#         os.remove(log_file)
#         print(f"ðŸ§¹ Cleaned up {log_file}")

# # Configure logging with DEBUG log level.
# logging.basicConfig(
#     filename="logger.log",
#     level=logging.DEBUG,
#     format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
# )

# print("âœ… Logging configured")


%%writefile ddl.sql

CREATE TABLE company (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  hq_address TEXT,
  country_id INTEGER NOT NULL,
  branding_config TEXT CHECK (json_valid(branding_config)), -- requires SQLite JSON extension
  FOREIGN KEY (country_id) REFERENCES country(id)
);

CREATE INDEX idx_company_country_id ON company(country_id);

CREATE TABLE country (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  code TEXT NOT NULL UNIQUE
);

CREATE TABLE venue (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  size_in_square_feet NUMERIC CHECK (size_in_square_feet >= 0),
  latitude NUMERIC CHECK (latitude BETWEEN -90 AND 90),
  longitude NUMERIC CHECK (longitude BETWEEN -180 AND 180),
  country_id INTEGER NOT NULL,
  FOREIGN KEY (country_id) REFERENCES country(id)
);

CREATE INDEX idx_venue_country_id ON venue(country_id);

CREATE TABLE ghg_scope (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL
);

CREATE TABLE emission_category (
  id INTEGER PRIMARY KEY,
  category TEXT NOT NULL UNIQUE
);

CREATE TABLE event (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  start_time TEXT NOT NULL,
  end_time TEXT NOT NULL,
  company_id INTEGER NOT NULL,
  venue_id INTEGER NOT NULL,
  total_attendees INTEGER NOT NULL DEFAULT 0 CHECK (total_attendees >= 0),
  virtual_attendees INTEGER NOT NULL DEFAULT 0 CHECK (virtual_attendees >= 0),
  physical_attendees_local INTEGER NOT NULL DEFAULT 0 CHECK (physical_attendees_local >= 0),
  physical_attendees_international INTEGER NOT NULL DEFAULT 0 CHECK (physical_attendees_international >= 0),
  total_event_hour INTEGER CHECK (total_event_hour IS NULL OR total_event_hour >= 0),
  total_catering_count INTEGER CHECK (total_catering_count IS NULL OR total_catering_count >= 0),
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  CHECK (end_time >= start_time),
  CHECK (
    total_attendees = (
      virtual_attendees +
      physical_attendees_local +
      physical_attendees_international
    )
  ),
  FOREIGN KEY (company_id) REFERENCES company(id),
  FOREIGN KEY (venue_id) REFERENCES venue(id)
);

CREATE INDEX idx_event_company_id ON event(company_id);
CREATE INDEX idx_event_venue_id ON event(venue_id);
CREATE INDEX idx_event_start_time ON event(start_time);

CREATE TABLE emission (
  id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL,
  activity_value NUMERIC,
  activity_unit TEXT,
  category_id INTEGER NOT NULL,
  scope INTEGER NOT NULL CHECK (scope IN (1, 2, 3)),
  calculated_emission_in_kgC02e NUMERIC,

  FOREIGN KEY (event_id) REFERENCES event(id),
  FOREIGN KEY (category_id) REFERENCES emission_category(id),
  FOREIGN KEY (scope) REFERENCES ghg_scope(id)
);


CREATE INDEX idx_emission_event_id ON emission(event_id);
CREATE INDEX idx_emission_category_id ON emission(category_id);


%%writefile mockdata.sql


-- Insert Countries
INSERT INTO country (id, name, code) VALUES (1, 'Singapore', 'SG');
INSERT INTO country (id, name, code) VALUES (2, 'United States', 'US');
INSERT INTO country (id, name, code) VALUES (3, 'United Kingdom', 'UK');
INSERT INTO country (id, name, code) VALUES (4, 'India', 'IN');
INSERT INTO country (id, name, code) VALUES (5, 'Germany', 'DE');

-- Insert Venues
INSERT INTO venue (id, name, size_in_square_feet, latitude, longitude, country_id) VALUES (1, 'Marina Bay Convention Center', 50000, 1.283, 103.86, 1);
INSERT INTO venue (id, name, size_in_square_feet, latitude, longitude, country_id) VALUES (2, 'New York Expo Hall', 75000, 40.7128, -74.006, 2);
INSERT INTO venue (id, name, size_in_square_feet, latitude, longitude, country_id) VALUES (3, 'London Event Hub', 60000, 51.5074, -0.1278, 3);
INSERT INTO venue (id, name, size_in_square_feet, latitude, longitude, country_id) VALUES (4, 'Mumbai Grand Hall', 45000, 19.076, 72.8777, 4);
INSERT INTO venue (id, name, size_in_square_feet, latitude, longitude, country_id) VALUES (5, 'Berlin Conference Center', 55000, 52.52, 13.405, 5);

-- Insert GHG Scopes
INSERT INTO ghg_scope (id, name, description) VALUES (1, 'Scope 1', 'Direct emissions from owned or controlled sources');
INSERT INTO ghg_scope (id, name, description) VALUES (2, 'Scope 2', 'Indirect emissions from purchased electricity, steam, heating, and cooling');
INSERT INTO ghg_scope (id, name, description) VALUES (3, 'Scope 3', 'All other indirect emissions that occur in the value chain');

-- Insert Emission Categories
INSERT INTO emission_category (id, category) VALUES (1, 'Travel');
INSERT INTO emission_category (id, category) VALUES (2, 'Accommodation');
INSERT INTO emission_category (id, category) VALUES (3, 'Catering');
INSERT INTO emission_category (id, category) VALUES (4, 'Venue Energy');
INSERT INTO emission_category (id, category) VALUES (5, 'Materials');
INSERT INTO emission_category (id, category) VALUES (6, 'Waste');
INSERT INTO emission_category (id, category) VALUES (7, 'Digital');

INSERT INTO company (id, name, country_id, branding_config) 
VALUES (1, 'BlueSky Corp', 1, '{"primary_color": "#007bff", "logo_url": "https://png.pngtree.com/png-clipart/20190604/original/pngtree-corporate-image-logo-png-image_1026060.jpg"}');

INSERT INTO company (id, name, country_id, branding_config) 
VALUES (2, 'NextGen Solutions', 2, '{"primary_color": "#dc3545", "logo_url": "https://d1csarkz8obe9u.cloudfront.net/posterpreviews/company-logo-design-template-e089327a5c476ce5c70c74f7359c5898_screen.jpg?ts=1672291305"}');

INSERT INTO company (id, name, country_id, branding_config) 
VALUES (3, 'EcoSphere Ltd', 3, '{"primary_color": "#28a745", "logo_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTbBL29Jawxl9ELLRe8K5Qgy0udooluQC2NQQ&s"}');

INSERT INTO company (id, name, country_id, branding_config) 
VALUES (4, 'Visionary Group', 4, '{}'); -- '{}' is the minimum valid JSON if no data exists

INSERT INTO company (id, name, country_id, branding_config) 
VALUES (5, 'GreenEdge Inc', 5, '{"primary_color": "#ffc107"}');


-- Insert Events
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (1, 'Product Launch', 'Product Launch for company 5', '2024-12-27 00:00', '2024-12-27 06:00', 5, 1, 457, 50, 328, 79, 6, 407);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (2, 'Sustainability Summit', 'Sustainability Summit for company 4', '2025-11-05 00:00', '2025-11-05 09:00', 4, 1, 364, 14, 220, 130, 9, 350);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (3, 'Tech Expo', 'Tech Expo for company 3', '2024-05-13 00:00', '2024-05-13 07:00', 3, 2, 623, 129, 123, 371, 7, 494);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (4, 'Annual Strategy Meeting', 'Annual Strategy Meeting for company 3', '2024-10-09 00:00', '2024-10-09 09:00', 3, 4, 184, 2, 145, 37, 9, 182);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (5, 'Tech Expo', 'Tech Expo for company 3', '2025-01-01 00:00', '2025-01-01 06:00', 3, 5, 228, 1, 157, 70, 6, 227);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (6, 'Innovation Workshop', 'Innovation Workshop for company 3', '2025-02-06 00:00', '2025-02-06 08:00', 3, 4, 108, 22, 69, 17, 8, 86);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (7, 'Quarterly Business Review', 'Quarterly Business Review for company 3', '2025-01-15 00:00', '2025-01-15 08:00', 3, 1, 300, 21, 233, 46, 8, 279);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (8, 'Quarterly Business Review', 'Quarterly Business Review for company 1', '2024-01-01 00:00', '2024-01-01 09:00', 1, 2, 229, 30, 157, 42, 9, 199);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (9, 'Sustainability Summit', 'Sustainability Summit for company 5', '2025-11-07 00:00', '2025-11-07 06:00', 5, 2, 102, 13, 54, 35, 6, 89);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (10, 'Innovation Workshop', 'Innovation Workshop for company 3', '2024-07-07 00:00', '2024-07-07 07:00', 3, 1, 312, 2, 46, 264, 7, 310);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (11, 'Sustainability Summit', 'Sustainability Summit for company 1', '2024-06-25 00:00', '2024-06-25 06:00', 1, 4, 111, 23, 22, 66, 6, 88);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (12, 'Sustainability Summit', 'Sustainability Summit for company 5', '2025-05-25 00:00', '2025-05-25 09:00', 5, 4, 504, 67, 33, 404, 9, 437);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (13, 'Product Launch', 'Product Launch for company 1', '2025-02-26 00:00', '2025-02-26 07:00', 1, 4, 722, 48, 629, 45, 7, 674);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (14, 'Annual Strategy Meeting', 'Annual Strategy Meeting for company 2', '2025-11-07 00:00', '2025-11-07 07:00', 2, 1, 196, 7, 176, 13, 7, 189);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (15, 'Sustainability Summit', 'Sustainability Summit for company 2', '2024-07-26 00:00', '2024-07-26 07:00', 2, 4, 387, 122, 87, 178, 7, 265);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (16, 'Quarterly Business Review', 'Quarterly Business Review for company 3', '2024-03-29 00:00', '2024-03-29 10:00', 3, 1, 111, 31, 19, 61, 10, 80);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (17, 'Quarterly Business Review', 'Quarterly Business Review for company 1', '2024-09-08 00:00', '2024-09-08 07:00', 1, 1, 513, 126, 169, 218, 7, 387);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (18, 'Product Launch', 'Product Launch for company 2', '2024-01-24 00:00', '2024-01-24 10:00', 2, 5, 661, 19, 582, 60, 10, 642);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (19, 'Annual Strategy Meeting', 'Annual Strategy Meeting for company 2', '2024-09-04 00:00', '2024-09-04 09:00', 2, 3, 713, 134, 197, 382, 9, 579);
INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id, total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international, total_event_hour, total_catering_count) VALUES (20, 'Quarterly Business Review', 'Quarterly Business Review for company 4', '2024-11-04 00:00', '2024-11-04 10:00', 4, 3, 221, 58, 42, 121, 10, 163);

-- Insert Emissions
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (1, 1, 48595.50, 'kWh', 1, 3, 5350.80);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (2, 1, 26779.20, 'km', 2, 1, 14533.04);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (3, 1, 49364.61, 'km', 3, 2, 2144.22);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (4, 1, 3756.14, 'GB', 4, 3, 10290.22);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (5, 1, 28105.40, 'GB', 5, 2, 18802.94);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (6, 1, 43663.09, 'GB', 6, 1, 2240.31);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (7, 1, 13579.48, 'kg', 7, 3, 10471.57);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (8, 2, 22912.83, 'room_nights', 1, 1, 15033.84);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (9, 2, 36766.18, 'kWh', 2, 1, 12847.46);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (10, 2, 7160.82, 'GB', 3, 3, 14348.38);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (11, 2, 30543.04, 'meals', 4, 1, 18439.46);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (12, 2, 32051.52, 'room_nights', 5, 1, 17696.87);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (13, 2, 2924.13, 'kg', 6, 1, 1216.95);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (14, 2, 32505.21, 'km', 7, 3, 5584.21);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (15, 3, 44291.07, 'meals', 1, 2, 3793.69);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (16, 3, 30246.28, 'room_nights', 2, 3, 18638.51);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (17, 3, 14038.01, 'GB', 3, 1, 1297.67);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (18, 3, 11481.20, 'GB', 4, 2, 15897.03);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (19, 3, 44816.70, 'GB', 5, 3, 19749.89);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (20, 3, 3866.07, 'kWh', 6, 3, 13644.03);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (21, 3, 12837.34, 'room_nights', 7, 3, 1106.18);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (22, 4, 18069.65, 'GB', 1, 2, 4771.06);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (23, 4, 30215.62, 'room_nights', 2, 3, 19475.79);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (24, 4, 4521.13, 'km', 3, 3, 5721.74);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (25, 4, 49806.93, 'meals', 4, 2, 9473.36);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (26, 4, 28999.41, 'kWh', 5, 3, 8406.30);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (27, 4, 31501.98, 'km', 6, 2, 11252.03);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (28, 4, 36160.74, 'kg', 7, 1, 5798.10);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (29, 5, 2110.78, 'kg', 1, 1, 19075.49);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (30, 5, 32484.80, 'GB', 2, 1, 8276.27);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (31, 5, 16850.31, 'km', 3, 2, 19814.13);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (32, 5, 6529.12, 'km', 4, 2, 14203.92);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (33, 5, 35034.40, 'km', 5, 1, 5026.69);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (34, 5, 11304.64, 'kWh', 6, 3, 10212.68);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (35, 5, 3388.09, 'kWh', 7, 1, 12829.54);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (36, 6, 28992.83, 'GB', 1, 3, 10695.62);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (37, 6, 7404.03, 'km', 2, 3, 15668.61);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (38, 6, 2371.90, 'GB', 3, 1, 14325.29);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (39, 6, 5384.83, 'GB', 4, 3, 19181.75);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (40, 6, 29869.18, 'kg', 5, 1, 4906.06);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (41, 6, 13763.05, 'meals', 6, 3, 9485.72);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (42, 6, 872.27, 'kWh', 7, 3, 15816.14);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (43, 7, 32193.24, 'GB', 1, 2, 5265.25);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (44, 7, 40645.36, 'km', 2, 3, 18758.62);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (45, 7, 24117.26, 'kg', 3, 2, 1037.01);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (46, 7, 14349.23, 'meals', 4, 1, 7982.58);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (47, 7, 10131.20, 'kWh', 5, 3, 16338.99);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (48, 7, 43261.77, 'km', 6, 1, 3098.68);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (49, 7, 24519.25, 'meals', 7, 1, 4323.55);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (50, 8, 46795.02, 'kg', 1, 3, 3908.10);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (51, 8, 1480.31, 'kg', 2, 3, 18163.21);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (52, 8, 15287.90, 'km', 3, 2, 17181.19);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (53, 8, 4929.31, 'kWh', 4, 2, 8344.17);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (54, 8, 42438.10, 'km', 5, 2, 4356.03);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (55, 8, 48365.97, 'room_nights', 6, 1, 16858.48);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (56, 8, 9150.32, 'room_nights', 7, 3, 8986.91);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (57, 9, 17322.79, 'meals', 1, 1, 18359.67);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (58, 9, 48292.42, 'GB', 2, 2, 5432.64);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (59, 9, 49100.86, 'kg', 3, 3, 1605.79);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (60, 9, 38784.18, 'km', 4, 2, 11356.68);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (61, 9, 36393.71, 'km', 5, 1, 6811.63);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (62, 9, 7906.38, 'kg', 6, 3, 16873.12);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (63, 9, 7600.25, 'meals', 7, 2, 19784.72);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (64, 10, 36143.03, 'GB', 1, 1, 5553.41);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (65, 10, 5456.89, 'meals', 2, 2, 12737.33);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (66, 10, 27080.89, 'km', 3, 2, 5055.10);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (67, 10, 6778.61, 'GB', 4, 1, 19634.85);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (68, 10, 20621.69, 'kWh', 5, 3, 6703.28);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (69, 10, 29953.75, 'kWh', 6, 3, 8956.63);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (70, 10, 34399.97, 'kWh', 7, 1, 9849.06);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (71, 11, 7140.13, 'meals', 1, 2, 6039.64);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (72, 11, 11650.09, 'kg', 2, 2, 19019.48);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (73, 11, 20429.16, 'kg', 3, 3, 18714.67);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (74, 11, 16901.95, 'kWh', 4, 1, 14652.68);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (75, 11, 10853.12, 'meals', 5, 3, 18677.07);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (76, 11, 35572.90, 'kWh', 6, 1, 8123.60);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (77, 11, 25514.96, 'meals', 7, 1, 19100.67);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (78, 12, 47765.38, 'GB', 1, 3, 5462.50);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (79, 12, 38337.61, 'kWh', 2, 1, 5067.85);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (80, 12, 46111.00, 'GB', 3, 2, 1203.87);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (81, 12, 20091.07, 'kWh', 4, 2, 13485.03);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (82, 12, 35527.57, 'meals', 5, 1, 269.88);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (83, 12, 25245.43, 'km', 6, 1, 19215.03);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (84, 12, 11011.54, 'room_nights', 7, 1, 16774.39);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (85, 13, 7199.41, 'kWh', 1, 3, 2441.11);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (86, 13, 25628.17, 'kWh', 2, 1, 15120.04);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (87, 13, 14801.38, 'km', 3, 2, 14862.47);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (88, 13, 7042.25, 'km', 4, 2, 19077.65);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (89, 13, 25943.26, 'kg', 5, 1, 257.32);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (90, 13, 26032.59, 'meals', 6, 2, 154.29);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (91, 13, 1278.63, 'room_nights', 7, 2, 8299.17);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (92, 14, 11173.25, 'kWh', 1, 1, 17856.81);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (93, 14, 20885.94, 'room_nights', 2, 1, 12272.55);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (94, 14, 19427.12, 'GB', 3, 2, 7736.17);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (95, 14, 40017.43, 'kWh', 4, 1, 7889.05);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (96, 14, 48326.48, 'km', 5, 3, 14336.11);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (97, 14, 7326.83, 'kWh', 6, 2, 4456.71);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (98, 14, 1751.19, 'meals', 7, 2, 2765.11);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (99, 15, 16242.22, 'room_nights', 1, 1, 10656.89);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (100, 15, 40318.00, 'kg', 2, 3, 16231.81);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (101, 15, 11952.77, 'km', 3, 1, 612.08);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (102, 15, 36757.74, 'kg', 4, 2, 17702.81);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (103, 15, 15273.82, 'room_nights', 5, 2, 3485.83);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (104, 15, 16284.82, 'meals', 6, 1, 4400.02);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (105, 15, 4972.34, 'room_nights', 7, 1, 8390.95);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (106, 16, 6630.54, 'room_nights', 1, 3, 5085.17);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (107, 16, 14655.30, 'kWh', 2, 2, 5120.64);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (108, 16, 46272.27, 'room_nights', 3, 3, 16012.30);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (109, 16, 35172.49, 'meals', 4, 1, 13299.05);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (110, 16, 42429.89, 'km', 5, 1, 5173.75);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (111, 16, 8382.10, 'kWh', 6, 2, 9192.28);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (112, 16, 28047.83, 'room_nights', 7, 3, 9941.08);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (113, 17, 11369.16, 'GB', 1, 2, 15847.95);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (114, 17, 24503.93, 'kg', 2, 1, 16961.50);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (115, 17, 46323.78, 'GB', 3, 1, 8135.26);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (116, 17, 24345.43, 'kWh', 4, 2, 5927.89);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (117, 17, 38423.20, 'room_nights', 5, 2, 6367.74);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (118, 17, 11737.65, 'kWh', 6, 2, 16633.34);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (119, 17, 15664.11, 'kWh', 7, 1, 4451.16);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (120, 18, 297.55, 'km', 1, 3, 7579.19);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (121, 18, 24224.27, 'kWh', 2, 1, 5777.43);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (122, 18, 927.94, 'kg', 3, 1, 18056.37);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (123, 18, 17958.54, 'kWh', 4, 3, 1259.54);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (124, 18, 17978.22, 'kWh', 5, 1, 14964.01);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (125, 18, 38076.26, 'meals', 6, 1, 18583.21);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (126, 18, 29592.78, 'room_nights', 7, 3, 12846.17);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (127, 19, 4466.81, 'GB', 1, 3, 19635.37);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (128, 19, 30270.49, 'room_nights', 2, 3, 2227.05);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (129, 19, 25386.77, 'kg', 3, 3, 17838.11);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (130, 19, 36081.89, 'kg', 4, 2, 14814.63);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (131, 19, 44925.25, 'kg', 5, 1, 6756.75);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (132, 19, 15585.50, 'GB', 6, 1, 107.84);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (133, 19, 24531.55, 'meals', 7, 3, 1614.62);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (134, 20, 8987.46, 'GB', 1, 1, 4110.79);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (135, 20, 32431.25, 'kg', 2, 1, 2827.85);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (136, 20, 39441.33, 'GB', 3, 3, 13119.80);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (137, 20, 8188.84, 'meals', 4, 3, 2225.04);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (138, 20, 25259.04, 'km', 5, 1, 15986.02);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (139, 20, 29103.57, 'kg', 6, 2, 16201.69);
INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES (140, 20, 15488.80, 'GB', 7, 1, 6725.58);



# import sqlite3

# DB_PATH = "event_emissions.db"

# conn = sqlite3.connect(DB_PATH)
# conn.row_factory = sqlite3.Row  # enables dict-like results
# cur = conn.cursor()



import sqlite3

DB_PATH = "event_emissions.db"

# 1. Connect to the database
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# 2. Enable Foreign Keys (Important for SQLite)
cur.execute("PRAGMA foreign_keys = ON;")

# 3. Clean up old schema (Drop tables if they exist to start fresh)
# We drop in reverse order of dependencies
cur.executescript("""
DROP TABLE IF EXISTS emission;
DROP TABLE IF EXISTS event;
DROP TABLE IF EXISTS venue;
DROP TABLE IF EXISTS company;
DROP TABLE IF EXISTS emission_category;
DROP TABLE IF EXISTS ghg_scope;
DROP TABLE IF EXISTS country;
-- Drop old tables from previous version if they exist
DROP TABLE IF EXISTS event_emission_summary;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS emission_categories;
""")



# Run DDL script
with open("ddl.sql", "r") as ddl_file:
    ddl_script = ddl_file.read()
cur.executescript(ddl_script)

conn.commit()




with open("mockdata.sql", "r") as data_file:
    data_script = data_file.read()
cur.executescript(data_script)

conn.commit()


# cur.executescript("""


# BEGIN TRANSACTION;

# PRAGMA foreign_keys = ON;

# -- ==========================
# -- Seed: country
# -- ==========================
# INSERT INTO country (id, name, code) VALUES
#   (1, 'Singapore', 'SG'),
#   (2, 'United States', 'US'),
#   (3, 'United Kingdom', 'UK'),
#   (4, 'Malaysia', 'MY'),
#   (5, 'India', 'IN');

# -- ==========================
# -- Seed: company
# -- ==========================
# INSERT INTO company (id, name, hq_address, country_id, branding_config) VALUES
#   (1, 'SustiAI', '10 Marina Boulevard, Singapore 018983', 1,
#    '{"primary_color":"#0ABAB5","secondary_color":"#1F2937","logo_url":"https://png.pngtree.com/png-clipart/20190604/original/pngtree-corporate-image-logo-png-image_1026060.jpg","theme":"light"}'),
#   (2, 'StackStackGo', '548 Market St, San Francisco, CA 94104', 2,
#    '{"primary_color":"#2563EB","secondary_color":"#111827","logo_url":"https://d1csarkz8obe9u.cloudfront.net/posterpreviews/company-logo-design-template-e089327a5c476ce5c70c74f7359c5898_screen.jpg?ts=1672291305","theme":"dark"}');

# -- ==========================
# -- Seed: ghg_scope (scopes)
# -- ==========================
# INSERT INTO ghg_scope (id, name, description) VALUES
#   (1, 'Scope 1', 'Direct emissions from sources owned or controlled by the organization (e.g., on-site fuel combustion)'),
#   (2, 'Scope 2', 'Indirect emissions from the generation of purchased energy (e.g., electricity, steam, heat, cooling)'),
#   (3, 'Scope 3', 'All other indirect emissions that occur in the value chain (e.g., travel, accommodation, catering, waste)');

# -- ==========================
# -- Seed: emission_category
# -- ==========================
# INSERT INTO emission_category (id, category) VALUES
#   (1, 'On-site fuel combustion'),
#   (2, 'Waste treatment on-site'),
#   (3, 'Venue Electricity'),
#   (4, 'Air Travel'),
#   (5, 'Local Transport'),
#   (6, 'Accommodation'),
#   (7, 'Catering'),
#   (8, 'Waste disposal (off-site)');

# -- ==========================
# -- Seed: venue
# -- 4 key venues per country
# -- ==========================
# -- Singapore (SG)
# INSERT INTO venue (id, name, size_in_square_feet, latitude, longitude, country_id) VALUES
#   (1, 'Marina Bay Sands Expo & Convention Centre', 1200000, 1.2834, 103.8607, 1),
#   (2, 'Suntec Singapore Convention & Exhibition Centre', 1000000, 1.2931, 103.8572, 1),
#   (3, 'Raffles City Convention Centre', 250000, 1.2936, 103.8533, 1),
#   (4, 'Resorts World Sentosa Convention Centre', 300000, 1.2570, 103.8205, 1);

# -- United States (US)
# INSERT INTO venue (id, name, size_in_square_feet, latitude, longitude, country_id) VALUES
#   (5, 'Moscone Center, San Francisco', 2000000, 37.7840, -122.4011, 2),
#   (6, 'Jacob K. Javits Convention Center, New York', 1800000, 40.7577, -74.0026, 2),
#   (7, 'McCormick Place, Chicago', 2600000, 41.8528, -87.6167, 2),
#   (8, 'Orange County Convention Center, Orlando', 2100000, 28.4255, -81.4593, 2);

# -- United Kingdom (UK)
# INSERT INTO venue (id, name, size_in_square_feet, latitude, longitude, country_id) VALUES
#   (9, 'ExCeL London', 1000000, 51.5079, 0.0265, 3),
#   (10, 'Olympia London', 500000, 51.4953, -0.2070, 3),
#   (11, 'Manchester Central', 500000, 53.4775, -2.2440, 3),
#   (12, 'Scottish Event Campus (SEC), Glasgow', 380000, 55.8609, -4.2873, 3);

# -- Malaysia (MY)
# INSERT INTO venue (id, name, size_in_square_feet, latitude, longitude, country_id) VALUES
#   (13, 'Kuala Lumpur Convention Centre (KLCC)', 300000, 3.1579, 101.7131, 4),
#   (14, 'MITEC (Malaysia International Trade and Exhibition Centre)', 500000, 3.1789, 101.6736, 4),
#   (15, 'Putrajaya International Convention Centre (PICC)', 400000, 2.8736, 101.6767, 4),
#   (16, 'SPICE Convention Centre, Penang', 250000, 5.3299, 100.2850, 4);

# -- India (IN)
# INSERT INTO venue (id, name, size_in_square_feet, latitude, longitude, country_id) VALUES
#   (17, 'Bharat Mandapam (IECC), Pragati Maidan, Delhi', 1000000, 28.6196, 77.2346, 5),
#   (18, 'NESCO Center (Bombay Exhibition Centre), Mumbai', 500000, 19.1579, 72.8365, 5),
#   (19, 'BIEC (Bangalore International Exhibition Centre)', 550000, 13.0736, 77.4556, 5),
#   (20, 'HICC (Hyderabad International Convention Centre)', 300000, 17.4446, 78.3808, 5);

# -- ==========================
# -- Seed: event
# -- One event per venue per company (=> 20 venues x 2 companies = 40 events)
# -- Attendance totals match the breakdown. 8-hour events.
# -- ==========================
# -- Pattern: For each venue, two events:
# --  - SustiAI Green Tech Summit @ {Venue} (Dec 2025 dates)
# --  - StackStackGo DevCon @ {Venue} (Jan/Feb 2026 dates)

# -- Singapore venues
# INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id,
#                    total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international,
#                    total_event_hour, total_catering_count)
# VALUES
#   (1, 'SustiAI Green Tech Summit @ Marina Bay Sands', 'Sustainability innovations and AI applications for events.', '2025-12-05 09:00:00', '2025-12-05 17:00:00', 1, 1,
#    650, 200, 300, 150, 8, 450),
#   (2, 'StackStackGo DevCon @ Marina Bay Sands', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-10 09:00:00', '2026-01-10 17:00:00', 2, 1,
#    750, 250, 350, 150, 8, 500),
#   (3, 'SustiAI Green Tech Summit @ Suntec', 'Sustainability innovations and AI applications for events.', '2025-12-06 09:00:00', '2025-12-06 17:00:00', 1, 2,
#    620, 180, 300, 140, 8, 440),
#   (4, 'StackStackGo DevCon @ Suntec', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-11 09:00:00', '2026-01-11 17:00:00', 2, 2,
#    740, 240, 360, 140, 8, 500),
#   (5, 'SustiAI Green Tech Summit @ Raffles City', 'Sustainability innovations and AI applications for events.', '2025-12-07 09:00:00', '2025-12-07 17:00:00', 1, 3,
#    580, 160, 280, 140, 8, 420),
#   (6, 'StackStackGo DevCon @ Raffles City', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-12 09:00:00', '2026-01-12 17:00:00', 2, 3,
#    690, 210, 340, 140, 8, 480),
#   (7, 'SustiAI Green Tech Summit @ Sentosa', 'Sustainability innovations and AI applications for events.', '2025-12-08 09:00:00', '2025-12-08 17:00:00', 1, 4,
#    600, 170, 290, 140, 8, 430),
#   (8, 'StackStackGo DevCon @ Sentosa', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-13 09:00:00', '2026-01-13 17:00:00', 2, 4,
#    710, 220, 350, 140, 8, 490);

# -- United States venues
# INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id,
#                    total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international,
#                    total_event_hour, total_catering_count)
# VALUES
#   (9, 'SustiAI Green Tech Summit @ Moscone', 'Sustainability innovations and AI applications for events.', '2025-12-12 09:00:00', '2025-12-12 17:00:00', 1, 5,
#    800, 250, 380, 170, 8, 550),
#   (10, 'StackStackGo DevCon @ Moscone', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-18 09:00:00', '2026-01-18 17:00:00', 2, 5,
#    900, 280, 420, 200, 8, 620),
#   (11, 'SustiAI Green Tech Summit @ Javits', 'Sustainability innovations and AI applications for events.', '2025-12-13 09:00:00', '2025-12-13 17:00:00', 1, 6,
#    780, 240, 370, 170, 8, 540),
#   (12, 'StackStackGo DevCon @ Javits', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-19 09:00:00', '2026-01-19 17:00:00', 2, 6,
#    880, 270, 410, 200, 8, 610),
#   (13, 'SustiAI Green Tech Summit @ McCormick Place', 'Sustainability innovations and AI applications for events.', '2025-12-14 09:00:00', '2025-12-14 17:00:00', 1, 7,
#    760, 230, 360, 170, 8, 530),
#   (14, 'StackStackGo DevCon @ McCormick Place', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-20 09:00:00', '2026-01-20 17:00:00', 2, 7,
#    860, 260, 400, 200, 8, 600),
#   (15, 'SustiAI Green Tech Summit @ OCCC Orlando', 'Sustainability innovations and AI applications for events.', '2025-12-15 09:00:00', '2025-12-15 17:00:00', 1, 8,
#    740, 220, 350, 170, 8, 520),
#   (16, 'StackStackGo DevCon @ OCCC Orlando', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-21 09:00:00', '2026-01-21 17:00:00', 2, 8,
#    840, 250, 390, 200, 8, 590);

# -- United Kingdom venues
# INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id,
#                    total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international,
#                    total_event_hour, total_catering_count)
# VALUES
#   (17, 'SustiAI Green Tech Summit @ ExCeL London', 'Sustainability innovations and AI applications for events.', '2025-12-19 09:00:00', '2025-12-19 17:00:00', 1, 9,
#    700, 210, 320, 170, 8, 490),
#   (18, 'StackStackGo DevCon @ ExCeL London', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-25 09:00:00', '2026-01-25 17:00:00', 2, 9,
#    820, 240, 380, 200, 8, 580),
#   (19, 'SustiAI Green Tech Summit @ Olympia London', 'Sustainability innovations and AI applications for events.', '2025-12-20 09:00:00', '2025-12-20 17:00:00', 1, 10,
#    680, 200, 310, 170, 8, 480),
#   (20, 'StackStackGo DevCon @ Olympia London', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-26 09:00:00', '2026-01-26 17:00:00', 2, 10,
#    800, 230, 370, 200, 8, 570),
#   (21, 'SustiAI Green Tech Summit @ Manchester Central', 'Sustainability innovations and AI applications for events.', '2025-12-21 09:00:00', '2025-12-21 17:00:00', 1, 11,
#    660, 190, 300, 170, 8, 470),
#   (22, 'StackStackGo DevCon @ Manchester Central', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-27 09:00:00', '2026-01-27 17:00:00', 2, 11,
#    780, 220, 360, 200, 8, 560),
#   (23, 'SustiAI Green Tech Summit @ SEC Glasgow', 'Sustainability innovations and AI applications for events.', '2025-12-22 09:00:00', '2025-12-22 17:00:00', 1, 12,
#    640, 180, 290, 170, 8, 460),
#   (24, 'StackStackGo DevCon @ SEC Glasgow', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-01-28 09:00:00', '2026-01-28 17:00:00', 2, 12,
#    760, 210, 350, 200, 8, 550);

# -- Malaysia venues
# INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id,
#                    total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international,
#                    total_event_hour, total_catering_count)
# VALUES
#   (25, 'SustiAI Green Tech Summit @ KLCC', 'Sustainability innovations and AI applications for events.', '2025-12-26 09:00:00', '2025-12-26 17:00:00', 1, 13,
#    620, 180, 300, 140, 8, 440),
#   (26, 'StackStackGo DevCon @ KLCC', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-02-01 09:00:00', '2026-02-01 17:00:00', 2, 13,
#    740, 220, 350, 170, 8, 520),
#   (27, 'SustiAI Green Tech Summit @ MITEC', 'Sustainability innovations and AI applications for events.', '2025-12-27 09:00:00', '2025-12-27 17:00:00', 1, 14,
#    640, 190, 310, 140, 8, 450),
#   (28, 'StackStackGo DevCon @ MITEC', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-02-02 09:00:00', '2026-02-02 17:00:00', 2, 14,
#    760, 230, 360, 170, 8, 530),
#   (29, 'SustiAI Green Tech Summit @ PICC', 'Sustainability innovations and AI applications for events.', '2025-12-28 09:00:00', '2025-12-28 17:00:00', 1, 15,
#    600, 170, 290, 140, 8, 430),
#   (30, 'StackStackGo DevCon @ PICC', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-02-03 09:00:00', '2026-02-03 17:00:00', 2, 15,
#    720, 210, 340, 170, 8, 510),
#   (31, 'SustiAI Green Tech Summit @ SPICE Penang', 'Sustainability innovations and AI applications for events.', '2025-12-29 09:00:00', '2025-12-29 17:00:00', 1, 16,
#    580, 160, 280, 140, 8, 420),
#   (32, 'StackStackGo DevCon @ SPICE Penang', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-02-04 09:00:00', '2026-02-04 17:00:00', 2, 16,
#    700, 200, 330, 170, 8, 500);

# -- India venues
# INSERT INTO event (id, name, description, start_time, end_time, company_id, venue_id,
#                    total_attendees, virtual_attendees, physical_attendees_local, physical_attendees_international,
#                    total_event_hour, total_catering_count)
# VALUES
#   (33, 'SustiAI Green Tech Summit @ Bharat Mandapam', 'Sustainability innovations and AI applications for events.', '2026-01-02 09:00:00', '2026-01-02 17:00:00', 1, 17,
#    820, 260, 400, 160, 8, 560),
#   (34, 'StackStackGo DevCon @ Bharat Mandapam', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-02-08 09:00:00', '2026-02-08 17:00:00', 2, 17,
#    920, 300, 440, 180, 8, 620),
#   (35, 'SustiAI Green Tech Summit @ NESCO Center', 'Sustainability innovations and AI applications for events.', '2026-01-03 09:00:00', '2026-01-03 17:00:00', 1, 18,
#    780, 240, 380, 160, 8, 540),
#   (36, 'StackStackGo DevCon @ NESCO Center', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-02-09 09:00:00', '2026-02-09 17:00:00', 2, 18,
#    880, 280, 420, 180, 8, 600),
#   (37, 'SustiAI Green Tech Summit @ BIEC Bangalore', 'Sustainability innovations and AI applications for events.', '2026-01-04 09:00:00', '2026-01-04 17:00:00', 1, 19,
#    760, 230, 370, 160, 8, 530),
#   (38, 'StackStackGo DevCon @ BIEC Bangalore', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-02-10 09:00:00', '2026-02-10 17:00:00', 2, 19,
#    860, 260, 410, 190, 8, 600),
#   (39, 'SustiAI Green Tech Summit @ HICC Hyderabad', 'Sustainability innovations and AI applications for events.', '2026-01-05 09:00:00', '2026-01-05 17:00:00', 1, 20,
#    740, 220, 360, 160, 8, 520),
#   (40, 'StackStackGo DevCon @ HICC Hyderabad', 'Developer conference on scalable stacks and cloud-native tooling.', '2026-02-11 09:00:00', '2026-02-11 17:00:00', 2, 20,
#    840, 250, 400, 190, 8, 590);

# -- ==========================
# -- Seed: emission (SAMPLE)
# -- Add more rows as needed; here we seed for event_id 1 and 2.
# -- ==========================
# INSERT INTO emission (id, event_id, activity_value, activity_unit, category_id, scope, calculated_emission_in_kgC02e) VALUES
#   -- Event 1 (SustiAI @ MBS)
#   (1, 1, 500, 'L', 1, 1, 1200.0),          -- On-site fuel combustion
#   (2, 1, 0.5, 'ton', 2, 1, 50.0),          -- Waste treatment on-site
#   (3, 1, 10000, 'kWh', 3, 2, 5000.0),      -- Venue Electricity
#   (4, 1, 200000, 'pkm', 4, 3, 30000.0),    -- Air Travel (passenger-km)
#   (5, 1, 5000, 'km', 5, 3, 800.0),         -- Local Transport (vehicle-km)
#   (6, 1, 500, 'room-night', 6, 3, 2500.0), -- Accommodation
#   (7, 1, 900, 'meals', 7, 3, 1800.0),      -- Catering
#   (8, 1, 2, 'ton', 8, 3, 240.0),           -- Waste disposal (off-site)

#   -- Event 2 (StackStackGo @ MBS)
#   (9, 2, 600, 'L', 1, 1, 1440.0),
#   (10, 2, 0.8, 'ton', 2, 1, 80.0),
#   (11, 2, 12000, 'kWh', 3, 2, 6000.0),
#   (12, 2, 240000, 'pkm', 4, 3, 36000.0),
#   (13, 2, 7000, 'km', 5, 3, 1100.0),
#   (14, 2, 650, 'room-night', 6, 3, 3200.0),
#   (15, 2, 1100, 'meals', 7, 3, 2200.0),
#   (16, 2, 2.5, 'ton', 8, 3, 300.0);

# """)

# conn.commit()


rows = cur.execute("""
select * from event; """
).fetchall()

[dict(r) for r in rows]


%%writefile event_emission_server.py

import sqlite3
import json
from typing import Dict, List, Any, Optional
from mcp.server.fastmcp import FastMCP

DB_PATH = "event_emissions.db"

# Initialize FastMCP
mcp = FastMCP("carbon_accounting_expert")

def get_db_connection():
    """Establishes a database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def query(sql: str, params=()) -> List[Dict[str, Any]]:
    """Helper to run queries and return clean dictionaries."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        conn.close()

# ---------------------------------------------------------------------
# CORE DISCOVERY TOOLS
# ---------------------------------------------------------------------

@mcp.tool()
def describe_database() -> str:
    """
    Returns the exact CREATE TABLE statements for the database.
    Use this to understand table relationships and column names.
    """
    conn = get_db_connection()
    rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return "\n".join([row[0] for row in rows if row[0] is not None])

@mcp.tool()
def search_entities(query_string: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search for Companies or Events by name to find their IDs.
    Useful when the user asks for 'Tech Expo' but you need the ID to fetch data.
    """
    wildcard = f"%{query_string}%"
    
    companies = query("SELECT id, name FROM company WHERE name LIKE ?", (wildcard,))
    events = query("""
        SELECT e.id, e.name, e.start_time, c.name as company_name 
        FROM event e 
        JOIN company c ON e.company_id = c.id 
        WHERE e.name LIKE ?
    """, (wildcard,))
    
    return {"companies": companies, "events": events}

@mcp.tool()
def get_branding_config(company_id: int) -> Dict[str, Any]:
    """
    Retrieves the UI/UX branding guidelines (colors, logo) for a specific company.
    Returns a parsed dictionary from the stored JSON.
    """
    res = query("SELECT branding_config FROM company WHERE id = ?", (company_id,))
    if not res or not res[0]['branding_config']:
        return {"error": "No branding config found"}
    
    try:
        return json.loads(res[0]['branding_config'])
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in branding_config column"}

# ---------------------------------------------------------------------
# REPORTING LEVEL 1: EVENT SPECIFIC
# ---------------------------------------------------------------------

@mcp.tool()
def get_event_kpis(event_id: int) -> Dict[str, Any]:
    """
    Fetches high-level Key Performance Indicators (KPIs) for a single event.
    Calculates Total Emissions, Intensity per Attendee, and Intensity per SqFt.
    """
    sql = """
    SELECT 
        e.name as event_name,
        c.name as company_name,
        e.total_attendees,
        v.size_in_square_feet,
        COALESCE(SUM(em.calculated_emission_in_kgC02e), 0) as total_emissions_kg,
        COALESCE(SUM(em.calculated_emission_in_kgC02e) / 1000.0, 0) as total_emissions_tonnes
    FROM event e
    JOIN company c ON e.company_id = c.id
    JOIN venue v ON e.venue_id = v.id
    LEFT JOIN emission em ON e.id = em.event_id
    WHERE e.id = ?
    GROUP BY e.id
    """
    data = query(sql, (event_id,))
    if not data:
        return {"error": "Event not found"}
    
    row = data[0]
    # Calculate Intensities safely
    attendees = row['total_attendees'] or 1 # avoid div by zero
    sq_ft = row['size_in_square_feet'] or 1
    
    row['intensity_per_attendee_kg'] = round(row['total_emissions_kg'] / attendees, 2)
    row['intensity_per_sqft_kg'] = round(row['total_emissions_kg'] / sq_ft, 4)
    
    return row

@mcp.tool()
def get_event_emissions_breakdown(event_id: int) -> List[Dict[str, Any]]:
    """
    Returns the detailed emission inventory for an event.
    Grouped by Scope and Category.
    Essential for generating Scope 3 charts.
    """
    sql = """
    SELECT 
        gs.name as scope_name,
        ec.category as category_name,
        SUM(em.calculated_emission_in_kgC02e) as total_kg
    FROM emission em
    JOIN ghg_scope gs ON em.scope = gs.id
    JOIN emission_category ec ON em.category_id = ec.id
    WHERE em.event_id = ?
    GROUP BY gs.name, ec.category
    ORDER BY total_kg DESC
    """
    return query(sql, (event_id,))

# ---------------------------------------------------------------------
# REPORTING LEVEL 2: COMPANY PORTFOLIO
# ---------------------------------------------------------------------

@mcp.tool()
def get_company_portfolio_summary(company_id: int) -> Dict[str, Any]:
    """
    Aggregates data across ALL events for a specific company.
    Returns total events count and total carbon footprint.
    """
    sql = """
    SELECT 
        c.name as company_name,
        COUNT(DISTINCT e.id) as total_events_held,
        SUM(e.total_attendees) as total_attendees_impacted,
        COALESCE(SUM(em.calculated_emission_in_kgC02e), 0) as total_portfolio_emissions_kg
    FROM company c
    LEFT JOIN event e ON c.id = e.company_id
    LEFT JOIN emission em ON e.id = em.event_id
    WHERE c.id = ?
    GROUP BY c.id
    """
    data = query(sql, (company_id,))
    return data[0] if data else {"error": "Company not found"}

@mcp.tool()
def get_company_monthly_emissions(company_id: int, year: str) -> List[Dict[str, Any]]:
    """
    Time-series data: Returns total emissions grouped by month for a specific company and year.
    Useful for Line Charts showing trends.
    """
    sql = """
    SELECT 
        strftime('%Y-%m', e.start_time) as month,
        COUNT(DISTINCT e.id) as events_count,
        SUM(em.calculated_emission_in_kgC02e) as total_kg
    FROM event e
    JOIN emission em ON e.id = em.event_id
    WHERE e.company_id = ? AND strftime('%Y', e.start_time) = ?
    GROUP BY month
    ORDER BY month ASC
    """
    return query(sql, (company_id, year))

# ---------------------------------------------------------------------
# UTILITY: SQL EXPLORATION
# ---------------------------------------------------------------------

@mcp.tool()
def run_sql(query_text: str) -> List[Dict[str, Any]]:
    """
    Executes a custom READ-ONLY SQL query.
    Use this only if the specific reporting tools do not provide the data you need.
    IMPORTANT: Only 'SELECT' statements are allowed.
    """
    cleaned = query_text.strip().upper()
    if not cleaned.startswith("SELECT") and not cleaned.startswith("WITH"):
        return [{"error": "Security Restriction: Only SELECT queries allowed."}]
    
    return query(query_text)

if __name__ == "__main__":
    mcp.run(transport="stdio")


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent, ParallelAgent
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)


print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

server_command = ['python', 'event_emission_server.py']

# Correct instantiation: Pass the connection parameters directly to the constructor
mcp_tools = MCPToolset(
    connection_params=StdioServerParameters(
        command=server_command[0],
        args=server_command[1:]
    )
)

# # Create the ADK agent and pass the mcp_tools
# my_agent = LlmAgent(
#     name="MyMcpAgent",
#     model="gemini-2.0-flash", # or another model
#     tools=[mcp_tools],
#     # ... other agent parameters
# )


data_collector_agent = LlmAgent(
    name="DataCollector",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    tools=[mcp_tools], 
    output_key="data_collected",
    instruction="""**Role & Objective:**
    You are an autonomous **Sustainability Data Strategist**. Your goal is to gather the most comprehensive dataset possible for the user's request. 
    Do not rely solely on rigid scripts. You must reason about the user's intent and fetch relevant data, even for complex or novel queries.

    **Core Philosophy:**
    Better to fetch too much relevant data than too little. If a standard tool doesn't exist for the specific question, you **MUST** construct a custom SQL query.

    **Execution Workflow:**

    **1. IDENTIFY & CONTEXTUALIZE:**
    *   Analyze the input. Is it about a specific Event? A Company? A specific Region? Multiple entities?
    *   **Action:** Use `search_entities` to find IDs.
    *   *Adaptive Logic:* If the user asks about "All events in Singapore", `search_entities` might not be enough. In that case, use custom SQL.

    **2. MANDATORY BRANDING:**
    *   For every primary entity identified (Company or Event), you **MUST** fetch its branding using `get_branding_config`.
    *   *Reasoning:* The downstream Reporter Agent crashes without this.

    **3. ADAPTIVE DATA COLLECTION (The "Shopping List"):**
    You need to fill the following buckets with data. Use ANY tool available (Standard Tools OR `run_sql`).

    *   **Bucket A: High-Level Metrics (KPIs):**
        *   *Standard:* `get_event_kpis`, `get_company_portfolio_summary`.
        *   *Custom:* If the user asks "Average emissions per event", write a SQL query to calculate it.
    
    *   **Bucket B: Visual Data (Charts/Trends):**
        *   *Standard:* `get_event_emissions_breakdown`, `get_company_monthly_emissions`.
        *   *Custom:* If the user asks "Compare Scope 1 vs Scope 2 trend", write SQL to fetch exactly that.

    *   **Bucket C: Deep Insights:**
        *   *Standard:* `get_top_emission_sources`.
        *   *Custom:* Use SQL to answer specific questions like "Which venue is most efficient?" or "List events with > 500 attendees".

    **4. THE "SQL BACKSTOP":**
    *   If the standard tools (`get_event_...`, `get_company_...`) do not perfectly answer the prompt, you **MUST**:
        1.  Call `describe_database()` to see the schema.
        2.  Write and execute a `run_sql()` query to get the exact data needed.

    **5. FINAL OUTPUT (Standardized JSON):**
    Aggregated everything into this JSON structure. If a bucket was filled via SQL, map it to the most logical key.
    
    ```json
    {
        "entity_details": { ... },     // Result from search or custom SQL identification
        "branding": { ... },           // Result from get_branding_config
        "primary_data": { ... },       // KPIs, Totals, or Summary Stats
        "chart_data": [ ... ],         // Arrays suitable for graphing (Time series or Categories)
        "supporting_data": [ ... ],    // Top sources lists, detailed rows, or custom SQL results
        "data_source_notes": "..."     // Optional: Briefly explain what data you fetched (e.g., "Fetched custom SQL for regional comparison")
    }
    ```
    
    **Error Handling:**
    *   If absolutely no relevant data can be found after trying Search and SQL, output exactly: 'No such events found'.
    """
)

print("âœ… DataCollector updated with Adaptive/SQL capabilities.")


# runner = InMemoryRunner(agent=data_collector_agent)
# response = await runner.run_debug(
#     "Give me the summary of all events by BlueSky Corp", verbose = True
# )
# print("\nAgent Response:\n", response)


analyst_agent = LlmAgent(
    name="Analyst",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    output_key="key_insights",
    instruction="""
   **Role & Objective:**
You are a **Lead ESG Analyst** responsible for converting raw `{data_collected}` into a structured **Reporting Payload**. Your output will drive the final report, including KPI cards, charts, and a narrative summary.

**Your Goals:**
1. Validate input and handle edge cases.
2. Curate high-impact KPIs.
3. Identify and configure charts that best represent the data.
4. Synthesize a professional narrative with insights and recommendations.
5. Output a strict JSON object following the defined schema.

---

### **Input Data:**
{data_collected}

---

### **Execution Steps:**

#### **1. VALIDATION**
- If `{data_collected}` contains `"No such events found"`, return exactly: 'No such events found' and stop further processing.

---

#### **2. DATA CURATION**
- **Branding:** Extract the `branding` object as-is for `branding_config`.
- **KPI Cards:**
- Review `primary_data` and `supporting_data`.
- Select **3â€“4 diverse, high-impact metrics** (e.g., Total Emissions, Scope-wise Emissions, Intensity, Events Count).
- Format values:
  - Round to 2 decimals.
  - Include units (e.g., `"kg COâ‚‚e"`, `"pax"`).
  - Use human-readable formatting (e.g., `150,420` instead of `150420`).

---

#### **3. CHART CONFIGURATION**
- Review `primary_data`, `chart_data`, and `supporting_data`.
- Identify **key visualizations**:
- **Time-Series** â†’ `line` chart.
- **Categorical breakdown** â†’ `bar` or `doughnut` chart.
- **Distribution** â†’ `pie` or `doughnut` chart.
- For each chart:
- Include `title`, `type`, `labels` (categories), `data` (values), and `color_scheme` (e.g., `"primary"`).
- Limit to **3â€“5 charts** that highlight different aspects (trends, comparisons, breakdowns).

---

#### **4. NARRATIVE SYNTHESIS**
- Write a **professional Markdown summary**:
- `## Executive Summary` â†’ 2â€“3 sentences summarizing overall performance.
- `### Key Insights` â†’ Bullet points (e.g., `"Scope 3 contributes 85% of total emissions"`).
- `### Recommendations` â†’ 3 actionable strategies to reduce emissions based on hotspots.

---

#### **5. OUTPUT FORMAT**
Return a **strict JSON object** (no markdown fences) with this structure:

{
"branding_config": { ... },   // Raw branding object
"report_content_markdown": "## Executive Summary\n...",
"kpi_cards": [
{ "label": "Total Emissions", "value": "150,420", "unit": "kg COâ‚‚e" },
{ "label": "Attendees", "value": "500", "unit": "pax" }
],
"charts": [
{
"title": "Emissions by Category",
"type": "bar",
"labels": ["Travel", "Energy", "Waste"],
"data": [12000, 3000, 500],
"color_scheme": "primary"
}
]
}

---

### **Guidelines for Quality**
- Be **insight-driven**, not just descriptive.
- Avoid redundant charts; each chart should reveal a unique perspective.
- Recommendations must be **specific and actionable**.

"""
)

print("âœ… analyst created with Visualization Logic.")



def save_html(html_code: str, file_path: str = "reports/company.html") -> str:
    """
    This helps save the html file into memory. 
    Send the "company" in the the file_path parameter along with reports directory. Example: "reports/{company}.html 
    where company is the name of the company for which report is generated
    """
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_code)
    return file_path



reporting_agent = LlmAgent(
    name="Reporter",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    output_key="report",
    instruction="""**Role & Objective:**
    You are an expert **Front-End Developer & Data Visualization Specialist**.
    Your goal is to render a pixel-perfect, branded HTML Report based strictly on the JSON configuration provided in `{key_insights}`.

    **INPUT DATA STRUCTURE (JSON):**
    1.  `branding_config`: { primary_color, secondary_color, logo_url, font_family, ... }
    2.  `kpi_cards`: [ { label, value, unit }, ... ]
    3.  `report_content_markdown`: "## Executive Summary..."
    4.  `charts`: [ { title, type, labels, data, color_scheme }, ... ]

    **EXECUTION STEPS:**

    **1. SETUP DESIGN SYSTEM (CSS):**
    *   Extract `primary_color` and `secondary_color` from `branding_config`.
    *   Create a `<style>` block:
        *   **Root:** Set `font-family` from config.
        *   **Header:** Background = `primary_color`, Text = White.
        *   **KPI Cards:** Background = Light Gray/White, Border-Left = 5px solid `secondary_color`.
        *   **Typography:** Headings use `primary_color`.

    **2. BUILD HTML LAYOUT:**
    *   **Header:** `<div class="header"><img src="\{logo_url\}"> <h1>\{company_name\} `report name`</h1></div>`
    *   **KPI Grid:** Create a CSS Grid container (`display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;`).
        *   Loop through `kpi_cards`: Create a card for each with the Value big and bold.
    *   **Charts Grid:** Create a CSS Grid container for charts.
        *   Loop through `charts`: Create a container **EXACTLY** like this for each:
            `<div class="chart-card" style="background:#fff; padding:20px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);"><h3>\{chart.title\}</h3><div style="position: relative; height: 350px; width: 100%;"><canvas id="chart_\{index\}"></canvas></div></div>`
    *   **Narrative:** Convert `report_content_markdown` from Markdown to valid HTML. Wrap in a clean container.
    *   The order of information should be Header, KPI Grid, Charts and then the Narrative at the end.
    
    **3. GENERATE JAVASCRIPT (Chart.js):**
    *   Import Chart.js via CDN.
    *   **Iterate through the `charts` array** to generate `new Chart()` instances.
    *   **Mapping Rules:**
        *   `type`: Use the exact string provided by the Analyst ('bar', 'line', 'doughnut').
        *   `data.labels`: Use `chart.labels`.
        *   `data.datasets[0].data`: Use `chart.data`.
        *   **Colors:**
            *   If `type` is 'line', use `borderColor: primary_color`, `backgroundColor: transparent`.
            *   If `type` is 'bar', use `backgroundColor: primary_color`.
            *   If `type` is 'doughnut', generate an array of colors (Primary, Secondary, and shades of gray).
    *   **Config:** Always use `options: { maintainAspectRatio: false, responsive: true }`.

    **4. FINAL OUTPUT:**
    *   Return **ONLY** valid HTML code.
    *   Do NOT surround with markdown code fences (```html).
    *   If input is "No such events found", output that string.

    **Input Data:**
    {key_insights}
    """
)

print("âœ… reporter created with Renderer Logic.")


# Reporting Agent: Its job is to generate report in PDF format

saving_agent = LlmAgent(
    name="Saver",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    instruction = """You are provided with an html code for the events report in `{report}`, your only job is to save it using `save_html` tool in the default filepath of the tool""",
    tools = [FunctionTool(save_html)],
    output_key="final_report")


root_agent = SequentialAgent(
    name="ReportPipeline",
    sub_agents=[data_collector_agent,analyst_agent, reporting_agent, saving_agent ],
)

print("âœ… Parallel and Sequential Agent created.")


runner = InMemoryRunner(agent=root_agent,  
                        plugins=[
        LoggingPlugin()
    ],  # <---- 2. Add the plugin. Handles standard Observability logging across ALL agents)

# 3. Create a session explicitly and inject the state
session = await runner.session_service.create_session(
    app_name="reporting_app",
    user_id="debug_user"
)

# 4. Run the debug using the session_id you just created
response = await runner.run_debug(
    "Give a report on all events by StackStackGo", 
    session_id=session.id,  # <--- Connects to the state above
    verbose=True
)


runner = InMemoryRunner(agent=root_agent)

# 3. Create a session explicitly and inject the state
session = await runner.session_service.create_session(
    app_name="reporting_app",
    user_id="debug_user"
)

# 4. Run the debug using the session_id you just created
response = await runner.run_debug(
    "Give a comparative report on all events by EcoSphere Ltd", 
    session_id=session.id,  # <--- Connects to the state above
    verbose=True
)


runner = InMemoryRunner(agent=root_agent)

# 3. Create a session explicitly and inject the state
session = await runner.session_service.create_session(
    app_name="reporting_app",
    user_id="debug_user"
)

# 4. Run the debug using the session_id you just created
response = await runner.run_debug(
    "Give me a detailed carbon accounting report for Sustainability Summit event of Visionary Group", 
    session_id=session.id,  # <--- Connects to the state above
    verbose=True
)


runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug(
    "Compare Tech Expo event of EcoSphere Ltd 2024 vs 2025", verbose = True
)


# runner = InMemoryRunner(agent=root_agent)
# response = await runner.run_debug(
#     "Give a brief fun report on Sustainability Summit 2025. Only include Scope 1 and Scope 2.", verbose = True
# )


# runner = InMemoryRunner(agent=root_agent)
# response = await runner.run_debug(
#     "Give a brief fun report on Sustainability Summit 2025 to be sent on email to all attendees highlighting their carbon footprint. Also, add bite-sized recommendations on how they can reduce their carbon footprint for next event", verbose = True
# )


# runner = InMemoryRunner(agent=root_agent)
# response = await runner.run_debug(
#     "Give a report on Sustainability Summit 2025 to be sent to all attendees highlighting their carbon footprint. Also, add recommendations on how they can reduce their carbon footprint for next event", verbose = True
# )


# runner = InMemoryRunner(agent=root_agent)
# response = await runner.run_debug(
#     "Give a report on Sustainability Summit 2025 to be sent to all attendees highlighting their carbon footprint", verbose = True
# )


# runner = InMemoryRunner(agent=root_agent)
# response = await runner.run_debug(
#     "Give me the ESG report for GonetZero", verbose = True
# )















# def save_chart(fig, path = "output"):
#     """Save a Matplotlib figure to a file."""
#     fig.savefig(path)
#     plt.close(fig)



# import io

# def execute_pdf_generation(pdf_builder_func, output_path = "output", *args, **kwargs):
#     """
#     Executes a PDF generation function provided by an agent and saves the PDF to disk.

#     Args:
#         pdf_builder_func (callable): A function that returns a BytesIO stream of the PDF.
#         output_path (str): Path where the PDF should be saved.
#         *args, **kwargs: Additional arguments to pass to the PDF builder function.

#     Returns:
#         str: Path to the saved PDF file.
#     """
#     # Call the agent-provided function to get PDF content in memory
#     pdf_stream = pdf_builder_func(*args, **kwargs)

#     if not isinstance(pdf_stream, io.BytesIO):
#         raise ValueError("PDF builder function must return a BytesIO object.")

#     # Write the PDF content to disk
#     with open(output_path, "wb") as f:
#         f.write(pdf_stream.getvalue())

#     return output_path




# import os
# import base64
# from io import BytesIO
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib import colors
# import matplotlib.pyplot as plt

# def generate_event_report_pdf(json_data, output_path="pdf_files/event_report.pdf"):
#     """
#     Generates a PDF report from JSON data with summary cards, equivalency info,
#     scope breakdown, and charts (bar + donut) similar to the provided dashboard layout.
#     """
#     # Ensure output folder exists
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     # Generate charts
#     # Bar chart for emissions breakdown
#     plt.figure(figsize=(6, 3))
#     plt.barh(list(json_data["emissions_breakdown"].keys()), list(json_data["emissions_breakdown"].values()), color="orange")
#     plt.xlabel("Total Carbon Footprint (tCO2e)")
#     plt.ylabel("Category")
#     plt.tight_layout()
#     bar_chart_path = os.path.join(os.path.dirname(output_path), "emissions_breakdown.png")
#     plt.savefig(bar_chart_path)
#     plt.close()

#     # Donut chart for carbon by category
#     labels = list(json_data["carbon_by_category"].keys())
#     sizes = list(json_data["carbon_by_category"].values())
#     plt.figure(figsize=(4, 4))
#     wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
#     centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#     fig = plt.gcf()
#     fig.gca().add_artist(centre_circle)
#     plt.title("Carbon Footprint by Category")
#     donut_chart_path = os.path.join(os.path.dirname(output_path), "carbon_by_category.png")
#     plt.savefig(donut_chart_path)
#     plt.close()

#     # Create PDF
#     styles = getSampleStyleSheet()
#     doc = SimpleDocTemplate(output_path, pagesize=letter)
#     story = []

#     # Summary cards as a table
#     data = [
#         ["Total Events", f"{json_data['total_events']['in_person']} In Person\n{json_data['total_events']['hybrid']} Hybrid\n{json_data['total_events']['virtual']} Virtual"],
#         ["Total Attendees", f"{json_data['total_attendees']['in_person']} In Person\n{json_data['total_attendees']['virtual']} Virtual"],
#         ["Carbon Footprint", f"{json_data['carbon_footprint']['total']} tCO2e\nAvg: {json_data['carbon_footprint']['avg_per_attendee']} kgCO2e"],
#         ["Waste Footprint", f"{json_data['waste_footprint']['total']} tonnes\nAvg: {json_data['waste_footprint']['avg_per_attendee']} kg"]
#     ]
#     summary_table = Table(data, colWidths=[150, 200])
#     summary_table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
#         ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
#         ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
#         ('VALIGN', (0, 0), (-1, -1), 'TOP')
#     ]))
#     story.append(summary_table)
#     story.append(Spacer(1, 20))

#     # Equivalency section
#     story.append(Paragraph("<b>This is the equivalent of:</b>", styles['Heading2']))
#     for item in json_data['equivalent']:
#         story.append(Paragraph(f"â€¢ {item}", styles['Normal']))
#     story.append(Spacer(1, 20))

#     # Scope section
#     data_scope = [
#         ["Scope 1", f"{json_data['scope']['scope1']} tCO2e"],
#         ["Scope 2", f"{json_data['scope']['scope2']} tCO2e"],
#         ["Scope 3", f"{json_data['scope']['scope3']} tCO2e"]
#     ]
#     scope_table = Table(data_scope, colWidths=[150, 150])
#     scope_table.setStyle(TableStyle([
#         ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
#         ('FONTNAME', (0, 0), (-1, -1), 'Helvetica')
#     ]))
#     story.append(scope_table)
#     story.append(Spacer(1, 20))

#     # Add charts
#     story.append(Paragraph("<b>Emissions Breakdown by Category</b>", styles['Heading2']))
#     story.append(Image(bar_chart_path, width=400, height=200))
#     story.append(Spacer(1, 20))

#     story.append(Paragraph("<b>Carbon Footprint by Category</b>", styles['Heading2']))
#     story.append(Image(donut_chart_path, width=300, height=300))

#     # Build PDF
#     doc.build(story)
#     return output_path




# json_data = {
#     "total_events": {"in_person": 3, "hybrid": 1, "virtual": 1},
#     "total_attendees": {"in_person": 21011, "virtual": 361},
#     "carbon_footprint": {"total": 241.97, "avg_per_attendee": 11.32},
#     "waste_footprint": {"total": 14.98, "avg_per_attendee": 0.70},
#     "equivalent": [
#         "CO2 equivalent from 121 cars on the road for one year",
#         "The carbon sequestered by 1210 tree seedlings grown for 10 years"
#     ],
#     "scope": {"scope1": 9.7, "scope2": 0, "scope3": 232.3},
#     "emissions_breakdown": {
#         "Mains Electric and Gas": 10,
#         "Temporary Power": 5,
#         "Hotel Accommodation": 30,
#         "Catering": 3,
#         "Attendee Travel": 200,
#         "Crew Travel": 15,
#         "Waste - Recycle": 2
#     },
#     "carbon_by_category": {
#         "Energy": 13.8,
#         "Catering": 1.4,
#         "Travel and Transport": 75.5,
#         "Production": 9.1,
#         "Waste": 0.2
#     }
# }

# pdf_path = generate_event_report_pdf(json_data)
# print(f"PDF saved at: {pdf_path}")



# # Carbon Reporting Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
# root_agent = Agent(
#     name="CarbonReportingCordinator",
#     model=Gemini(
#         model="gemini-2.5-flash-lite",
#         retry_options=retry_config
#     ),
#     # This instruction tells the root agent HOW to use its tools (which are the other agents).
#     instruction="""You are a carbon reporting agent. Your goal is to generate ESG style report based on user provided company name by orchestrating a workflow.
# 1. First, you MUST call the `DataCollector` tool to find relevant activites and emissions data of the company provided by the user.
# 2. After receiving the emissions data, you MUST call the `Analyst` tool to create a concise summary.
# 3. then, you MUST call the 'Visualizer' to get insights in the form of charts and graphs.
# 4. Then, generate a ESG style pdf report by calling 'Reporter' tool 
# 5. Finally, present the pdf to the user as your response.""",
#     # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
#     tools=[AgentTool(data_collector_agent), AgentTool(analyst_agent), AgentTool(visualizer_agent), AgentTool(reporting_agent)],
# )

# print("âœ… root_agent created.")


 # instruction = """**Role and Objective:**
 #    You are an expert Sustainability Reporter. Your single task is to compile the provided sustainability insights (`key_insights`) and visual charts (`charts`) into a professional, ESG-style PDF report using only the `save_report_as_pdf` tool.

 #    **Input Handling and Execution Flow:**
 #    1.  **Check Inputs:** Inspect the inputs from upstream agents. If `key_insights` or `charts` contains the exact message 'No such events found', stop immediately and output that error message as your final result.
 #    2.  **Format Content:** Combine the Markdown text in `key_insights` into a single comprehensive string variable. Add newlines whereever needed. Format it in a way to present in PDF report.
 #    3.  **Final Output:** The tool will return the file path of the generated PDF. Your final output must be only that file path string.

 #    **Constraints:**
 #    *   Do not perform any data analysis or summarization yourself.
 #    *   Do not add any conversational text, introductions, or conclusions.
 #    *   The *only* acceptable final output is either the PDF file path string or the error message 'No such events found'.
 #    *   Ensure the tool is called exactly once with all necessary information to generate the final artifact.

 #    **key_insights:**
 #    {key_insights}

 #    **charts:**
 #    None



































































