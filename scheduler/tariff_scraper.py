import datetime
import os
import xml.etree.ElementTree as ET
import pandas as pd
from utils import manage_directory


# considering correction factor for CSUD area
g_reg = 0.0
CC_reg = 0.0
zone = 'CSUD'
plant_power = 200.0

def get_TIP_max(plant_power, pz):
    # plant_power is expected in kW
    # returns €/MWh
    f = get_f(plant_power, pz)
    if 1000 >= plant_power > 600:
        return min(100, f)
    if 600.0 >= plant_power > 200.0:
        return min(110, f)
    if plant_power <= 200.0:
        return min(120, f)
    return 0


def get_f(plant_power, pz):
    if 1000.0 >= plant_power > 600.0:
        return 60.0 + max(0, 180.0 - pz)
    if 600.0 >= plant_power > 200.0:
        return 70.0 + max(0, 180.0 - pz)
    if plant_power <= 200.0:
        return 80.0 + max(0, 180.0 - pz)
    return 0


def compute_incentive(plant_power, pz):
    tip_max = get_TIP_max(plant_power, pz)
    f = get_f(plant_power, pz)
    print(tip_max)
    print(f)
    return min(tip_max, f + max(0, 180 - pz) + g_reg) * (1 - CC_reg)

def tariff_scraping(plant_power):
    source_dir = f'Prices{os.sep}MGP_Prices'
    dest_dir = f'Prices{os.sep}Parsed'
    manage_directory(dest_dir, delete_existing=True)

    for fname in os.listdir(source_dir):
        date = fname.split('MGPPrezzi.xml')[0]
        pun = []
        pz = []
        inc = []

        tree = ET.parse(source_dir + os.sep + fname)
        root = tree.getroot()
        for child in root:
            for elem in child:
                if elem.tag == 'PUN':
                    pun.append(float(elem.text.replace(',', '.')))
                elif elem.tag == zone:
                    pz_t = float(elem.text.replace(',', '.'))
                    inc.append(compute_incentive(plant_power, pz_t))
                    pz.append(pz_t)

        tariffs = pd.DataFrame({'Hour': range(0, 24), 'PUN': pun, 'PZ': pz, 'INC': inc})
        tariffs.set_index('Hour', inplace=True, drop=True)
        tariffs.to_csv(f'{dest_dir}{os.sep}{date}.csv', sep=';')


def get_tariffs(date):
    if type(date) is not datetime.datetime:
        date = pd.to_datetime(date, format='mixed')
    fname = f"{date.year}{date.month:02d}{date.day:02d}"
    return pd.read_csv(f'Prices{os.sep}Parsed{os.sep}{fname}.csv', sep=";")

def read_prices_incentive(filename, date):
    date = date.split("-")
    df = pd.read_csv(filename, sep=";", dtype={"PUN": float})
    df = df[(df["Giorno"] == int(date[0])) & (df["Mese"] == int(date[1])) & (df["Anno"] == int(date[2]))]
    return list(df["PUN"]), list(df["PZ"]) , list(df['INC'])


if __name__ == '__main__':
    tariff_scraping(plant_power)
