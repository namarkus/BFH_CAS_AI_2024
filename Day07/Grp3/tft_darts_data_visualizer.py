from datetime import date
import os, glob
import tkinter as tk
from tkinter import ttk
from tkinter import font
from tkinter import messagebox
from turtle import color, update
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.font_manager import font_scalings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import line

 # Funktion, die beim Auswählen eines Wertes im Dropdown aufgerufen wird
def on_select(event):
    selected_value = station_selection.get()
    update_graph(selected_value)

def reset_zoom():
    update_graph(station_selection.get())    

# Funktion, um die Grafik zu aktualisieren
def update_graph(station_name):
    window.config(cursor="watch")
    window.update()
    progress_bar.start()
    progress_bar["value"] = 0
    selected_station = stations[stations['station_name'] == station_name]
    print(f"{selected_station} ausgewählt")    
    selected_measurements = measurements[measurements['station_id'] == selected_station['station_id'].values[0]]
    if selected_measurements.shape[0] == 0:
        messagebox.showwarning('Fehlende Daten', f'Für Station {station_name} sind keine Messdaten vorhanden!')
        print(f'Keine Messdaten für Station {station_name} gefunden.')
        progress_bar.stop
        return
    dateFrom = selected_measurements['date'].min()
    dateUntil = selected_measurements['date'].max() 
    print(f'{selected_measurements}')
    window.title(f'TTF Darts Visualizer | Station {station_name}')
    #progress_bar.step(5)
    progress_bar["value"] += 5
    timeline = pd.to_datetime(selected_measurements['date'])
    #progress_bar.step(5)
    progress_bar["value"] += 5
    ax.clear()
    avg_temperature = selected_measurements['temperature_avg']
    ax.plot(timeline, avg_temperature, label="Durchschnitt", color='#fb8900', linewidth=1)
    #progress_bar.step(15)
    progress_bar["value"] += 15
    min_temperature = selected_measurements['temperature_min']
    ax.plot(timeline, min_temperature, label="Minimal", color='#00a0af', linewidth=1)
    #progress_bar.step(15)
    progress_bar["value"] += 15
    max_temperature = selected_measurements['temperature_max']
    ax.plot(timeline, max_temperature, label="Maximal", color='#eb4000', linewidth=1)
    #progress_bar.step(15)
    progress_bar["value"] += 15
    ax.set_title(f'Station {station_name} - Temperaturen in °C {dateFrom:%d.%m.%Y}-{dateUntil:%d.%m.%Y}', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax2.clear()
    snowdepth = selected_measurements['snowdepth']
    ax2.plot(timeline, snowdepth, label="Schneehöhe", color='#efe6dc', linewidth=1)
    #progress_bar.step(15)
    progress_bar["value"] += 15
    precipitation = selected_measurements['precipitation']
    ax2.bar(timeline, precipitation, label="Niederschlag", color="#16b7bb")
    #progress_bar.step(15)
    progress_bar["value"] += 15
    snowfall = selected_measurements['snowfall']
    ax2.bar(timeline, snowfall, label="Schneefall", color="#787772")
    #progress_bar.step(15)
    progress_bar["value"] += 15
    ax2.set_title(f' Niederschläge in mm {dateFrom:%d.%m.%Y}-{dateUntil:%d.%m.%Y}', fontsize=10)    
    ax2.legend(fontsize=8, loc='upper left')
    canvas.draw()
    window.config(cursor="")
    progress_bar.stop()
    progress_bar["value"] = 0
    window.update()


STATIONS_FILE = "data/stations.csv"
print(f'Lade Stationsdaten ...')
if not os.path.isfile(STATIONS_FILE):
    raise FileNotFoundError(f'Datei {STATIONS_FILE} nicht gefunden.')
print (f'Lade Datei {STATIONS_FILE} ...')
stations = pd.read_csv(STATIONS_FILE)
print (f'Daten geladen. Dataset enthält {stations.shape[0]:,} Zeilen und {stations.shape[1]} Spalten.')
measurements = pd.DataFrame()
print(f'Lade Messdaten ...')
for filename in glob.glob('data/*_flat.csv'):
    print (f'Lade Datei {filename} ...')
    measurements = pd.concat([measurements, pd.read_csv(filename)], ignore_index=True) 
measurements['date'] = pd.to_datetime(measurements['date'], format='%Y%m%d')
print (f'Daten geladen. Dataset enthält {measurements.shape[0]:,} Zeilen und {measurements.shape[1]} Spalten.')    
print(measurements.head())
print("Starte GUI ...")

station_names = stations['station_name'].values   # Werte für das Pulldown-Menü aus dem DataFrame extrahieren (z. B. die Spalte 'Name')
station_names = sorted(station_names)  # Sortieren
window = tk.Tk() # Tkinter-Fenster erstellen
window.title('TTF Darts Visualizer')
action_panel = tk.Frame(window) # Frame erstellen, um die Widgets auf einer Zeile zu platzieren
action_panel.pack(pady=20, padx=20) 
station_selection = tk.StringVar() # Dropdown-Variable
station_selection.set(station_names[0])  # Initialwert festlegen
dropdown = ttk.Combobox(action_panel, textvariable=station_selection, values=station_names,  state="readonly", width=40)
dropdown.bind("<<ComboboxSelected>>", on_select)  # Ereignis binden
dropdown.pack(side=tk.LEFT, padx=10)
#zoom_out_button = tk.Button(action_panel, text="Ganze Grafik anzeigen", command=reset_zoom)
#zoom_out_button.pack(side=tk.LEFT, padx=10)
progress_bar = ttk.Progressbar(action_panel, length=200, mode="determinate") # Fortschrittsbalken erstellen
progress_bar.pack(side=tk.LEFT, padx=10)

# Matplotlib-Figur erstellen
#fig, ax = plt.subplots(figsize=(15, 5))
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

# Grafik in Tkinter einbetten
canvas = FigureCanvasTkAgg(fig, master=window)
update_graph(station_names[0])
canvas.get_tk_widget().pack()

# Tkinter-Hauptschleife starten
window.mainloop()
