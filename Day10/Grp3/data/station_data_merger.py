#/! /bin/python
import pandas as pd
import numpy as np
import os
import sys
columns_to_remove = [
    "Stationsname", "Stationsnummer", "Parameter", 
    "Zeitreihe", "Parametereinheit", "Gewässer", 
    "Zeitpunkt_des_Auftretens", "Freigabestatus"
]

# Entfernen der Spalten
temperature_data = pd.read_csv('Aare_Bern_2135_Wassertemperatur_Tagesmittel_2024-01-01_2024-12-31.csv', encoding='ISO-8859-1', delimiter=';', skiprows=8)
print(temperature_data.head())
temperature_data = temperature_data.drop(columns=columns_to_remove)
temperature_data["Zeitstempel"] = pd.to_datetime(temperature_data["Zeitstempel"]).dt.date
temperature_data = temperature_data.rename(columns={"Zeitstempel": "Datum", "Wert": "Temperatur"})
# 
flow_data = pd.read_csv('Aare_Bern_2135_Abfluss_Tagesmittel_2024-01-01_2024-12-31.csv', encoding='ISO-8859-1', delimiter=';', skiprows=8)
print(flow_data.head())
flow_data = flow_data.drop(columns=columns_to_remove)
flow_data["Zeitstempel"] = pd.to_datetime(flow_data["Zeitstempel"]).dt.date
flow_data = flow_data.rename(columns={"Zeitstempel": "Datum", "Wert": "Abflussmenge"})
#
water_high_data = pd.read_csv('Aare_Bern_2135_Pegel_Tagesmittel_2024-01-01_2024-12-31.csv', encoding='ISO-8859-1', delimiter=';', skiprows=8)
print(water_high_data.head())
water_high_data = water_high_data.drop(columns=columns_to_remove)
water_high_data["Zeitstempel"] = pd.to_datetime(water_high_data["Zeitstempel"]).dt.date
water_high_data = water_high_data.rename(columns={"Zeitstempel": "Datum", "Wert": "Wasserstand"})

# Zusammenführen aller DataFrames anhand der "Datum"-Spalte
combined_data = pd.merge(flow_data, water_high_data, on="Datum")
combined_data = pd.merge(combined_data, temperature_data, on="Datum")
# Statistiken
print(combined_data.head())
print(combined_data.describe())
combined_data.to_csv('aare_2024.csv', index=False, sep=';')
