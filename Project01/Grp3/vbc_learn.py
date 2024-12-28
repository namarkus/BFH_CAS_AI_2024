#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ vbc_learn.py ]_________________________________________________________
__file__ = "vbc_learn.py"
__author__ = "BFH-CAS-AI-2024-Grp3"
__copyright__ = "Copyright 2024, BFH-CAS-AI-2024-Grp3"
__credits__ = ["Hans Wermelinger", "Helmut Gehrer", "Markus Näpflin", "Nils Hryciuk", "Steafan Mavilio"]
__license__ = "GPL"
__version__ = "0.9.1"
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

from curses import meta
import concurrent
from tqdm import tqdm
from datetime import datetime
import argparse
import json
from _logging import start_logger
from _apis import LlmClient, LlmClientConfigurator, EmbeddingStore
from _configs import EmbeddingStorage, print_splash_screen, VbcConfig, SupportedLlmProvider
from _file_io import InputFileHandler, InputFile, MetaFile
from _builders import ConfigBuilder, ClientBuilder, EmbeddingStoreBuilder
from _texts import split, clean_texts

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
parser.add_argument("--profile", type=str, 
                    help="""Zu ladendes Profil. Wird keines angegeben, so wird 
                    aufgrund der vorhandenen Konfigurations-Templates, der llm 
                    und der App-Version das Profil automatisch ermittelt.""")
parser.add_argument("--llm", type=str, choices=["auto", "lokal", "cloud"], 
                    default="auto", 
                    help="""LLM-Engine, die für die Verarbeitung verwendet 
                    werden soll. 'lokal' verwendet die lokal installierte Engine,
                    'orbstack' aue ein Image auf der lokalen Maschine
                    und 'openai' greift auf die Rest-APIs von OpenAI zu. Bei 
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
config = ConfigBuilder(mode, llm, __version__).with_image_to_text_config().with_embedding_config().build()
logging.info(f"Konfiguration mit Profil-Id {config.as_profile_label()} erstellt")
file_handler = InputFileHandler(mode, config)
if mode == "full":
    logging.warning("Vollständige Reindexierung wurde angefordert. Alle vorahndenen Metadaten werden gelöscht.")
    file_handler.delete_all_metafiles()

# _____[ Verarbeitung der neuen PDFs ]__________________________________________
logging.info("_____[ 1/4 Konvertierung Input-Dateien zu Texten ]_____")
input_files = file_handler.get_input_files_to_process(config)
logging.info(f"{len(input_files)} Datei(en) werden komplett neu indexiert.")
if len(input_files) > 0:
    image_to_text_client = ClientBuilder(config).for_image_to_text().build()
    for input_file in input_files:
        if input_file.is_processable_as_image():
            pages = input_file.get_content()
            logging.info(f"Bereite Datei {input_file.file_name} mit {len(pages)} Seiten vor ...")            
            meta_file = MetaFile(config, from_input_file=input_file)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor: 
                futures = {
                    idx: executor.submit(image_to_text_client.get_text_from_image, page)
                    for idx, page in enumerate(pages)
                }
                with tqdm(total=len(pages)) as pbar:
                    for _ in concurrent.futures.as_completed(futures.values()):
                        pbar.update(1)            
                for idx in sorted(futures.keys()): 
                    result = futures[idx].result()
                    meta_file.add_page({"number": idx + 1, "content": result})                
            meta_file.save()

# _____[ Text-Splitting und Embedding-Ermittlung ]______________________________
logging.info("_____[ 2/4 Text-Chunking  ]_____")
chunk_files = file_handler.get_metafiles_to_chunk(config)
logging.info(f"Texte von {len(chunk_files)} Datei(en) werden bereinigt und neu aufgeteilt.")
for meta_file in chunk_files:
    logging.info(f"Verarbeite Datei {meta_file.file_path} ...")
    doc_text = "";
    for page in meta_file.get_pages():     
        doc_text += page["content"] + "\f"
    chunks = clean_texts(split(doc_text, config))
    for text in chunks:
         meta_file.add_chunk(text)
    meta_file.save()

# _____[ Erstellen der Embeddings ]_____________________________________________
logging.info("_____[ 3/4 Embeddings erstellen  ]_____")
embedding_store = EmbeddingStoreBuilder(config).build()
if embedding_store.is_full_reload_required():
    embedding_store.delete_all()
    embedding_files = file_handler.get_all_metafiles()
else:
    embedding_files = file_handler.get_metafiles_to_embed(config)
logging.info(f"Embeddings von {len(embedding_files)} Datei(en) werden berechnet und gesichert.")
if len(embedding_files) > 0:
    embeddings_client = ClientBuilder(config).for_embeddings().build()
    for meta_file in embedding_files:
        logging.info(f"Verarbeite Datei {meta_file.file_path} ...")
        for text in meta_file.get_chunks():
            embeddings = embeddings_client.get_embeddings(text)
            #print(f"Text: {text}, Embeddings: {embeddings}")
            embedding_store.store(text, embeddings)
        meta_file.metadata["processor"]["embeddings"]["index_id"] = embedding_store.index_id
        meta_file.save()    
    embedding_store.close()
# _____[ Testen des Modells ]___________________________________________________
logging.info("_____[ 4/4 Testen des Modells  ]_____")
# todo Testing noch ergänzen
logging.info(f"Verarbeitung abgeschlossen. Embedding Store mit Id {embedding_store.index_id} wurde aktualisiert.")