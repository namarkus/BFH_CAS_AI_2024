#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ vbc_learn.py ]_________________________________________________________
__file__ = "vbc_learn.py"
__author__ = "BFH-CAS-AI-2024-Grp3"
__copyright__ = "Copyright 2024, BFH-CAS-AI-2024-Grp3"
__credits__ = ["Hans Wermelinger", "Helmut Gehrer", "Markus Näpflin", "Nils Hryciuk", "Steafan Mavilio"]
__license__ = "GPL"
__version__ = "0.9.0"
__status__ = "Development"
__description__ = """
Diese Anwendung bereitet die Daten für den späteren Einsatz im vb-chat vor. 

Die Originaldaten hierfür müssen im Verzeichnis ./data_original abgelegt sein, 
die bereinigten Zwischenergebnisse werden im Verzeichnis ./data abgelegt und 
zusätzlich als Embeddings gespeichert.

Das Programm kennt zwei Modi für die Verarbeitung. Im Modul full werden alle 
Daten verarbeitet, bei incremental werden nur die Daten verarbeitet, die noch 
nicht verarbeitet wurden. Dies wird ermittelt, indem geprüft wird, ob bereits 
ein Textfile im Verzeichnis ./data existiert.

Zudem kann gewählt werden, ob eine lokal installierte LLM-Engine oder die 
OpenAI-Engine verwendet werden soll.

Achtung: 

Neben einigen Python-Modulen wird auch eine lokal installierte Vesion von P
oppler benötigt. Inforamtionen zur Installation finden Sie hier: 

conda install poppler | scoop install poppler (Windows) | apt-get install poppler-utils (Linux) | brew install poppler (MacOS)
"""

import os
import platform
import getpass
import concurrent
from tqdm import tqdm
from datetime import datetime
import argparse
import json
from _logging import start_logger
from _apis import LlmClient, LlmClientConfigurator
from _configs import print_splash_screen, VbcConfig, SupportedLlmProvider
from _file_io import InputFileHandler, InputFile
from _builders import ConfgBuilder, ClientBuilder

# _____[ Laufzeit-Prüfung und Splash-Screen ]___________________________________
if __name__ == "__main__":
    print_splash_screen("vbc_learn", __version__, __author__)
logging = start_logger("vbc_learn", __status__)

# _____[ Parameterparser initialisieren ]_______________________________________
logging.debug("Werte übergebene Parameter aus  ...")
parser = argparse.ArgumentParser(description=__description__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", type=str, choices=["full", "inc"], default="inc", 
                    help="""Modus der Verarbeitung. Bei 'full' werden alle Daten 
                    werden verarbeitet und bei 'inc' erfilgt eine inkrementelle 
                    Verarbeitung, d.H. nur noch nicht vorhandene Daten werden 
                    ergänzt. Defaultwert ist '%(default)s'.""")
parser.add_argument("--llm", type=str, choices=["auto", "lokal", "openai"], 
                    default="auto", 
                    help="""LLM-Engine, die für die Verarbietung verwendet 
                    werden soll. 'lokal' verwendet die lokal installierte Engine 
                    und 'oenai' greift auf die Rest-APIs von OpenAI zu. Bei 
                    'auto' erfolgt eine automatische Auswahl: Wenn eine lokal 
                    installierte laufende Engine, gefunden wird, dann wird diese 
                    verwendet, ansonsten OpenAI. Defaultwert ist 
                    '%(default)s'.""")
cli_arguments = parser.parse_args()

# _____[ Verarbeitung starten ]_________________________________________________
start_datetime = datetime.now()
mode = cli_arguments.mode
llm = cli_arguments.llm
logging.info(f"Starte Verarbeitung im Modus '{mode}' mit der LLM-Engine '{llm}' ...")
config = ConfgBuilder(llm).with_image_to_text_config().with_embedding_config().build()

# _____[ Input-Dateien ermitteln ]______________________________________________
file_handler = InputFileHandler(mode, config)
input_files = file_handler.get_input_files()
logging.info(f"Es wurden {len(input_files)} Dateien für die Verarbeitung gefunden.")
if len(input_files) == 0:
    logging.info("Beende Verabeitung, keine Dateien zum Verarbeiten gefunden")
    exit()

# _____[ Verarbeitung der Dateien ]_____________________________________________
client = ClientBuilder(config).for_image_to_text().build()
for file in input_files:
    logging.info(f"Bereite Inhalt der Datei {file.file_name} auf ...")
    if file.is_processable_as_image():
        document = {
            "input_file": file.file_name,
            "processed_at": start_datetime.strftime("%Y-%m-%d_%H:%M:%S"),
            "processor": {
                "machine": {
                    "name": platform.node(),
                    "architecture": platform.machine(),
                    "details": platform.uname(),
                    "os": {
                        "name": platform.system(),
                        "version": platform.release(),
                    },
                },
                "app" : {
                    "name" : "vbc_learn" ,
                    "version" : __version__ ,
                },
                "user": getpass.getuser()
            },
            "pages": []
        }
        pages = file.get_content()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # todo: korrekte REihenfolge sicherstellen?
            futures = [
                executor.submit(client.get_text_from_image, page) for page in pages
            ]
            with tqdm(total=len(pages)) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)
            for f in futures:
                res = f.result()
                document["pages"].append(res)
                # embeddings erstellen

        json_path = f"{config.knowledge_repository_path}/{file.file_name}.json"

        with open(json_path, 'w') as f:
            json.dump(document, f , indent=2)

