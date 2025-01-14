
library(readr)
library(ggplot2)
library(tidyr)
library(dplyr)

f0_file_path = 'C:/Users/namar/Git_Repos/BFH_CAS_AI_2024/Day10/Grp3/data'

df0 <-
  read_delim(file.path(f0_file_path, 'birg_2024.csv'), delim = ';',
              col_types = cols(Zeitstempel = col_character(), Abflussmenge = col_number(),	Wasserstand = col_number(),	Temperatur = col_number()))

df0 <- df0 |> 
  arrange(Zeitstempel) |> 
  mutate(Zeitstempel = sub('^(.*?)T(.*?)[+-].*$', '\\1 \\2', Zeitstempel),
        Zeitstempel = as.POSIXct(Zeitstempel, format = '%F %T'),
        Tag = as.Date(Zeitstempel))

df0 |> 
  filter(!is.na(Temperatur)) |> 
  ggplot(aes(x = Zeitstempel, y = Temperatur)) +
  geom_path()

          
df0 |> 
  pivot_longer(-c(Zeitstempel, Tag), names_to = 'Messgroesse', values_to = 'Messwert') |> 
  ggplot(aes(x = Zeitstempel, y = Messwert)) +
  geom_path() +
  facet_wrap(vars(Messgroesse), ncol = 1)

df0_daily <- 
  df0 |> 
  arrange(Zeitstempel) |> 
  mutate(Zeitstempel = sub('^(.*?)T(.*?)[+-].*$', '\\1 \\2', Zeitstempel),
        Zeitstempel = as.POSIXct(Zeitstempel, format('%F %T')),
        Datum = as.Date(Zeitstempel)) |> 
  summarise(across(Abflussmenge:Temperatur, ~mean(., na.rm=T)), .by = Datum)

df0_daily |> 
  pivot_longer(-Datum, names_to = 'Messgroesse', values_to = 'Messwert') |> 
  ggplot(aes(x = Datum, y = Messwert)) +
  geom_path() +
  facet_wrap(vars(Messgroesse), ncol = 1, scales = 'free_y')

write_delim(df0_daily, file.path(f0_file_path, 'birg_2024_daily.csv'), delim = ';')

# ==========================================================================================

f1_file_path = 'C:/Users/namar/Git_Repos/BFH_CAS_AI_2024/Day10/Grp3/data'

df1.1 <- 
  read_delim(file.path(f1_file_path, 'HBZHa547_Sihl-Blattwag_Abfluss m3_s Radar_11_ASCII-Tabelle.txt'),
            skip = 4, delim = '\t',
            col_names = c('Datum', 'Zeit', 'Wert', 'Intervall', 'DQ', 'Typ'),
            col_types = cols(Datum = col_character(), Zeit = col_character(), Wert = col_number(), Intervall = col_number(), DQ = col_guess(), Typ = col_guess())) |> 
  mutate(Datumzeit = as.POSIXct(paste(Datum, Zeit), format = '%d.%m.%Y %H:%M'))


df1.1_daily <- 
  df1.1 |> 
  mutate(Datum = as.Date(Datumzeit)) |> 
  summarise(Abflussmenge = mean(Wert, na.rm=T), .by = Datum)

glimpse(df1.1_daily)

df1.2 <- 
  read_delim(file.path(f1_file_path, 'HBZHa547_Sihl-Blattwag_Pegel m ü.M., Radar_01_ASCII-Tabelle.txt'),
            skip = 4, delim = '\t',
            col_names = c('Datum', 'Zeit', 'Wert', 'Intervall', 'DQ', 'Typ'),
            col_types = cols(Datum = col_character(), Zeit = col_character(), Wert = col_number(), Intervall = col_number(), DQ = col_guess(), Typ = col_guess())) |> 
  mutate(Datumzeit = as.POSIXct(paste(Datum, Zeit), format = '%d.%m.%Y %H:%M'))


df1.2_daily <- 
  df1.2 |> 
  mutate(Datum = as.Date(Datumzeit)) |> 
  summarise(Wasserstand = mean(Wert, na.rm=T), .by = Datum)

glimpse(df1.2_daily)



df1.3 <- 
  read_delim(file.path(f1_file_path, 'HBZHa547_Sihl-Blattwag_Wassertemperatur_00_ASCII-Tabelle.txt'),
            skip = 4, delim = '\t',
            col_names = c('Datum', 'Zeit', 'Wert', 'Intervall', 'DQ', 'Typ'),
            col_types = cols(Datum = col_character(), Zeit = col_character(), Wert = col_number(), Intervall = col_number(), DQ = col_guess(), Typ = col_guess())) |> 
  mutate(Datumzeit = as.POSIXct(paste(Datum, Zeit), format = '%d.%m.%Y %H:%M'))


df1.3_daily <- 
  df1.3 |> 
  mutate(Datum = as.Date(Datumzeit)) |> 
  summarise(Temperatur = mean(Wert, na.rm=T), .by = Datum)

glimpse(df1.3_daily)

anti_join(df1.1_daily, df1.2_daily, by = 'Datum')
anti_join(df1.1_daily, df1.3_daily, by = 'Datum')
anti_join(df1.2_daily, df1.3_daily, by = 'Datum')

df1_daily <-
  df1.1_daily |> 
  full_join(df1.2_daily, by = 'Datum') |> 
  full_join(df1.3_daily, by = 'Datum') |> 
  filter(!is.na(Datum)) |> 
  filter(Datum >= as.Date('2020-01-01') & Datum <= as.Date('2024-12-31'))

df1_daily |> filter(is.na(Abflussmenge) | is.na(Wasserstand) | is.na(Temperatur))

glimpse(df1_daily)

df1_daily |> 
  pivot_longer(-Datum, names_to = 'Messgroesse', values_to = 'Messwert') |> 
  ggplot(aes(x = Datum, y = Messwert)) +
  geom_path() +
  facet_wrap(vars(Messgroesse), ncol = 1, scales = 'free_y')


write_delim(df1_daily, file.path(f1_file_path, 'sihl_2024_daily.csv'), delim = ';')

