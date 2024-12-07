# DACH-Temperatureforecast

Zeitzeihenanalyse mit TFT und Darts anhand von Messdaten von Wetterstationen

## Unser Dataset

### Datenherkunft

Beim genutzten Dataset handelt es sich um tägliche Messdaten von Wetterstationen der ganzen Welt. 
Auf dieses 
["Daily Updated Global Weather GHCN-D Timeseries OBT"-Dataset](https://www.kaggle.com/datasets/fabianintech/daily-updated-global-weather-ghcn-d-timeseries-obt?resource=download) 
sind wir bei Kaggle gestossen. Dort steht es allerdings nur als DB-Dump für Hadoop bereit. Aufgrund
der hinterlegten 
[Original-Quelle](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
haben wir beschlossen, die Daten direkt vom **National Centers for Environmental Information** zu 
beziehen und vorgängig zu filtern und aufzubereiten.

Die [Gesamtdokumentation des GHCND-Datasets](./GHCND_documentation.pdf) liegt diesem Projekt bei.
Alle verfügbaren Datensatz-Formate und Attribute sind auch 
[Online im entsprechenden Readme](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) 
dokumentiert.

Die Orginaldaten können per vom FTP-Server der National Centers for Environmental Information
[ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/](ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/) direkt 
bezogen werden. 

Wir beschränken uns für unsere Zeitreihen auf **Stationen der DACH-Region** (Deutschland, Österreich, 
Schweiz), nutzen aus dieser Region aber alle verfügbaren Wetterstationen des Datasets über den
gesamten verfügbaren Zeitraum. 

Folgende Daten des Datesets verwenden wir:

- ghcnd-stations.txt - Attribute der Wetterstationen, Format: Records mit fixen Feldlängen.
- Messdaten der Stationen (Unterverzeichnis `by_station` des FTP-Servers) Von hier haben wir nur
  die Daten der uns interessierenden Stationen heruntergeladen (`AU*.gz`; `GM*.gz` und `SZ*.gz`). Die
  Dateien sind gezipt und enthalten jeweils die komplette Zeitreihe der Messstationen.

### Lizenzhinweis

Das Dataset wird bereitgestellt vom NOAA National Climatic Data Center:

Menne, M.J., I. Durre, B. Korzeniewski, S. McNeill, K. Thomas, X. Yin, S. Anthony, R. Ray, 
R.S. Vose, B.E.Gleason, and T.G. Houston, 2012: Global Historical Climatology Network - 
Daily (GHCN-Daily), Version 3.31 / http://doi.org/10.7289/V5D21VHZ [2024-12-05].

### Aufbereitung

Die Datenaufberetung erfolgt durch das Skript `prepare_data` (Linux / macOS) bzw `prepare_data.cmd` 
(Windows), wobei dieses lediglich die Originaldaten in das Arbeitsverzeichnis kopiert, dort entpackt 
und dann das Python-Skript `prepare_data.py` startet, welches die eigentliche Aufbereitung vornimmt.

Die folgenden Daten werden verwendet:

#### Countries

- Format: csv
- Dateiname: data/countries.csv

Wird im Skript `prepare_data.py` direkt gebaut und enthält das Mapping zwischen FIPS und ISO-Code
der Länder. Sollen die Messstationen zusätzlicher Länder unterstützt werden, so sind diese im 
Skript zu ergänzen (und die entsprechenden Stationsdaten herunterzuladen). 

#### Stations

- Format csv
- Dateiname data/stations.csv

Basis hierfür ist das Original-File `ghcnd-stations.txt`:

| Variable     | Position | Typ       |
| ------------ | -------- | --------- |
| ID           |  1-11    | Character |
| LATITUDE     | 13-20    | Real      |
| LONGITUDE    | 22-30    | Real      |
| ELEVATION    | 32-37    | Real      |
| STATE        | 39-40    | Character |
| NAME         | 42-71    | Character |
| GSN FLAG     | 73-75    | Character |
| HCN/CRN FLAG | 77-79    | Character |
| WMO ID       | 81-85    | Character |

Alle Stationen, welche nicht sich nicht in den gewünschten Ländern, befinden werden entfernt, 
ebenso Attribute, weclhe nicht benötigt werden. Ergebnis ist ein JSOn-File mit folgenden Werten
pro Messstation:

| Attribut         | Format  | Bedeutung |
| ---------------- | ------- | ---------- |
| station_id       | str     | Eindeutige Id der Wetterstation |
| station_name     | str     | Name der Wetterstation |
| iso_country_code | str(2)  | ISO-Ländercode des Standorts |
| latitude         | Real    | Breitengrad des Standorts |
| longitude        | Real    | Längengrad des Standort |
| elevation        | Real    | Höde der Wetterstation (müM) |

#### Lämderdaten flach

- Format csv
- Dateiname data/{ländercode}_flat.csv

Dieser Datensatz enthält alle berücksichtigten Messungen aller Stationen[^1][^2]) eines Landes pro Tag. Es 
ist eher für Auswertungen gedacht, als für ML und enthält daher einige Attriubute mehr.


| Attribut         | Format   | Bedeutung |
| ---------------- | ------- | ---------- |
| station_id       | str      | Eindeutige Id der Wetterstation |
| date             | yyyyMMdd | Datum der Mesung |
| station_name     | str      | Name der Station (Ort) |
| iso_country_code | str(2)   | ISO-Ländercode des Standorts |
| latitude         | Real     | Breitengrad des Standorts |
| longitude        | Real     | Längengrad des Standort |
| elevation        | Real     | Höde der Wetterstation (müM) |
| temperature_min  | Real     | Minimaltermperatur des Tages in °C |
| temperature_max  | Real     | Maximaltermperatur des Tages in °C |
| temperature_avg  | Real     | Durchschnittstermperatur des Tages in °C |
| precipitation    | Real     | Niederschlagsmenge in mm |
| snowfall         | Real     | Schneefall in mm |
| snowdepth        | Real     | vorhandene Schneemenge in mm |

#### Länderdaten gruppiert

- Format csv
- Dateiname data/{ländercode}_grouped.csv

Dieser Datensatz enthält alle berücksichtigten Messungen aller Stationen[^1][^2] eines Landes pro Tag. Es 
ist eher für Auswertungen gedacht, als für ML und enthält daher einige Attriubute mehr.

| Attribut         | Format   | Bedeutung | Nutzung |
| ---------------- | ------- | ---------- | ------- |
| station_id       | str      | Eindeutige Id der Wetterstation | Gruppierung 1 |
| date             | yyyyMMdd | Datum der Mesung | Zeitlicher Aspekt |
| observation_type | TMIN/TMAX/TAVG | Typ der Temperaturmessung (temperature) | Gruppierung 2 |
| latitude         | Real     | Breitengrad des Standorts | Variable fix |
| longitude        | Real     | Längengrad des Standort | Variable fix |
| elevation        | Real     | Höde der Wetterstation (müM) | Variable fix | 
| temperature      | Real     | Observation-Temperatur des Tages in °C | --> Target | 
| precipitation    | Real     | Niederschlagsmenge in mm | Variable variabel |
| snowfall         | Real     | Schneefall in mm | Variable variabel |
| snowdepth        | Real     | vorhandene Schneemenge in mm | Variable variabel |

[^1]: Aufgrund der Beschränkungen der Dateigrössen im Gratis-Stack von GitHub beschränken wir und in Deutschland auf Ortschaften südlich von Frantkfurt.
[^2]: Der Zeitraum der Daten kann ebenfalls bei der Aufbereitung eingeschränkt werden (Konstante `YEARS_TO_KEEP` in `prepare_data.py`). Für die Lösung der Aufgabe wurde der Wert auf 30 Jahre limitiert.