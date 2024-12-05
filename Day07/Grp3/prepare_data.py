#! /bin/python3
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

def build_country_filter() -> pd.DataFrame:
        country_data = {
        'fips_country_code': ['AU', 'GM', 'SZ'],
        'iso_country_code': ['AT', 'DE', 'CH']
        }
        return pd.DataFrame(country_data)


def load_station_data(work_path: str, country_filter: pd.DataFrame) -> pd.DataFrame:
        """
        Lädt die GHCND-Stationen und filtert nach Ländern, welche wir untersützten wollen. (DACH)
        Das Input-File hat folgendes Format:
        ------------------------------
        Variable   Columns   Type
        ------------------------------
        ID            1-11   Character
        LATITUDE     13-20   Real
        LONGITUDE    22-30   Real
        ELEVATION    32-37   Real
        STATE        39-40   Character
        NAME         42-71   Character
        GSN FLAG     73-75   Character
        HCN/CRN FLAG 77-79   Character
        WMO ID       81-85   Character
        """
        station_fieldspecs = [(0, 11), (12, 21), (21, 31), (32,37), (41, 72)]
        station_fieldnames = ['station_id', 'latitude', 'longitude', 'elevation', 'station_name'] 
        station_data = pd.read_fwf(f'{work_path}/ghcnd-stations.txt', header=None, colspecs=station_fieldspecs, names=station_fieldnames)
        station_data['ghcnd_country_code'] = station_data['station_id'].str.slice(0, 2)
        station_data = station_data[station_data['ghcnd_country_code'].isin(country_filter['fips_country_code'])]
        station_data = station_data.merge(country_filter, left_on='ghcnd_country_code', right_on='fips_country_code')
        station_data = station_data.drop(columns=['fips_country_code', 'ghcnd_country_code'])		
        station_field_order = ['station_id', 'station_name', 'iso_country_code', 'latitude', 'longitude', 'elevation']
        station_data['station_ix'] = station_data['station_id']
        station_data = station_data.set_index('station_ix')
        return station_data[station_field_order]

def load_and_prepare_time_series(station_id: str, work_path: str, station_data: pd.DataFrame):
        """
        Lädt die Stationsmessdaten und bereitet diese auf.
        """
        try:
                station_recordings = pd.read_csv(f'{work_path}/{station_id}.csv', header=None, low_memory=False)
        except FileNotFoundError:
                print(f'Keine Daten für Messstation {station_id} gefunden')
                return pd.DataFrame(), pd.DataFrame()
        station_recordings.columns = ['station_id', 'date', 'observation_type', 'observation_value', 'observation_measurement_flag', 'observation_quality_flag', 'observation_source_flag', 'observation_time']
        station_recordings = station_recordings.drop(columns=['observation_source_flag', 'observation_time'])
        # Für evtl. Optimierungen: welche weitern Observation-Types sind verfügbar?
        all_obervation_types = station_recordings['observation_type'].unique()
        #print(f'Die Messstation {station_id} stellt folgende Observationen bereit: {all_obervation_types}')

        # Datenbereinigungen:
        station_recordings['observation_value'] = station_recordings['observation_value'].replace(9999, np.nan) #9999 ist ein Missing Value
        station_recordings['observation_value'] = station_recordings['observation_value'] / 10 # Gelieferte ints sind eigentlich 1/10 °C bzw 1/10 mm; 
        all_temperatures = station_recordings[station_recordings['observation_type'].isin(['TMIN', 'TMAX', 'TAVG'])].rename(columns={'observation_value': 'temperature'})
        min_temperatures = station_recordings[station_recordings['observation_type'] == 'TMIN'].rename(columns={'observation_value': 'temperature_min'})
        max_temperatures = station_recordings[station_recordings['observation_type'] == 'TMAX'].rename(columns={'observation_value': 'temperature_max'})
        avg_temperatures = station_recordings[station_recordings['observation_type'] == 'TAVG'].rename(columns={'observation_value': 'temperature_avg'})
        precipitation = station_recordings[station_recordings['observation_type'] == 'PRCP'].rename(columns={'observation_value': 'precipitation'})
        snowfall = station_recordings[station_recordings['observation_type'] == 'SNOW'].rename(columns={'observation_value': 'snowfall'})
        snowdepth = station_recordings[station_recordings['observation_type'] == 'SNWD'].rename(columns={'observation_value': 'snowdepth'})
        
        station_grouped_data = (
            all_temperatures[['date', 'observation_type', 'temperature']]
                .merge(precipitation[['date', 'precipitation']], on='date', how='outer')
                .merge(snowfall[['date', 'snowfall']], on='date', how='outer')
                .merge(snowdepth[['date', 'snowdepth']], on='date', how='outer')
        )
        station_grouped_data['station_id'] = station_id
        station_grouped_data = station_grouped_data.merge(station_data, left_on='station_id', right_index=True)
        grouped_field_order = ['station_id', 'date', 'observation_type', 'latitude', 'longitude', 'elevation', 'temperature', 'precipitation', 'snowfall', 'snowdepth']

        station_flat_data = (
            min_temperatures[['date', 'temperature_min']]
                .merge(max_temperatures[['date', 'temperature_max']], on='date', how='outer')
                .merge(avg_temperatures[['date', 'temperature_avg']], on='date', how='outer')
                .merge(precipitation[['date', 'precipitation']], on='date', how='outer')
                .merge(snowfall[['date', 'snowfall']], on='date', how='outer')
                .merge(snowdepth[['date', 'snowdepth']], on='date', how='outer')
        )
        station_flat_data['station_id'] = station_id
        station_flat_data = station_flat_data.merge(station_data, left_on='station_id', right_index=True)
        flat_field_order = ['station_id', 'date', 'station_name', 'iso_country_code', 'latitude', 'longitude', 'elevation', 'temperature_min', 'temperature_max', 'temperature_avg', 'precipitation', 'snowfall', 'snowdepth']	
        return  station_flat_data[flat_field_order], station_grouped_data[grouped_field_order]
        
WORK_PATH = 'data_work'
OUTPUT_PATH = 'data'

frankfurt_latitude = 50.1109 # Breitengrad von Frankfurt am Main
duesseldorf_latitude = 51.2277 # Breitengrad von Düsseldorf
koeln_latitude = 50.9375 # Breitengrad von Köln

country_filter = build_country_filter()
country_filter.to_csv(f'{OUTPUT_PATH}/countries.csv', index=False)
station_data = load_station_data(WORK_PATH, country_filter)
station_data.to_csv(f'{OUTPUT_PATH}/stations.csv', index=False)
for iso_country_code in country_filter['iso_country_code']:
        country_flat_data = pd.DataFrame()
        country_grouped_data = pd.DataFrame()
        stations_in_country = station_data[station_data['iso_country_code'] == iso_country_code]
        print(f'Bereite Daten für Land {iso_country_code} vor. Insgesamt sind {stations_in_country.shape[0]} Messstationen bekannt')
        # with tqdm(total=stations_in_country.shape[0], desc=f'Land {iso_country_code}' ) as progressbar:
        country_stations = station_data[station_data['iso_country_code'] == iso_country_code]
        for station in country_stations.itertuples():
                station_id = station.station_id
                if station.latitude > koeln_latitude:
                    print(f'Ignoriere Messstation {station_id} aufgrund der GitHub Grössenbeschränkungen.')
                    continue # Abbruch, da DAtaset sonst zu gross wird
                #print(f'Verarbeite Messstation {station_id} ...')
                station_flat_series, station_grouped_series = load_and_prepare_time_series(station_id, WORK_PATH, station_data)
                print(f'Verarbeitung für Messstation {station_id} mit {station_flat_series.shape[0]} Datensätzen abgeschlossen')
                country_flat_data = pd.concat([country_flat_data, station_flat_series])
                country_grouped_data = pd.concat([country_grouped_data, station_grouped_series])
                #progressbar.set_description(f'Land {iso_country_code}, Station {station_id} mit {country_flat_data.shape[0]} Tagesmessungen abgescdhlossen')
                #progressbar.update()
        print(f'Verarbeitung für Land {iso_country_code} mit {country_flat_data.shape[0]} Tagesmessungen abgeschlossen')
        country_flat_data.to_csv(f'{OUTPUT_PATH}/{iso_country_code}_flat.csv', index=False)
        print(f'flat_data für Land {iso_country_code} gespeichert.')
        country_grouped_data.to_csv(f'{OUTPUT_PATH}/{iso_country_code}_grouped.csv', index=False)
        print(f'grouped_data für Land {iso_country_code} gespeichert.')
        
