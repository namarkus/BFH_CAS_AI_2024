#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ vbc_learn.py ]_________________________________________________________
__file__ = "vbc_learn.py"
__author__ = "BFH-CAS-AI-2024-Grp3"
__copyright__ = "Copyright 2024/2025, BFH-CAS-AI-2024-Grp3"
__credits__ = ["Hans Wermelinger", "Helmut Gehrer", "Markus Näpflin", "Nils Hryciuk", "Steafan Mavilio"]
__license__ = "GPL"
__version__ = "1.0.0"
__status__ = "Production"
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

import concurrent
from tqdm import tqdm
import time
import argparse
from _logging import start_logger
from _configs import print_splash_screen
from _metrics import TensorBoardMonitor
from _file_io import InputFileHandler, MetaFile
from _builders import ConfigBuilder, ClientBuilder, EmbeddingStoreBuilder, EvaluatorBuilder
from _texts import split, clean_texts
import asyncio

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
mode = cli_arguments.mode
llm = cli_arguments.llm
logging.info(f"Starte Verarbeitung im Modus '{mode}' mit der LLM-Engine '{llm}' ...")
config = ConfigBuilder(mode, llm, __version__).with_image_to_text_config().with_embeddings_config().build()
logging.info(f"Konfiguration mit Profil-Id {config.as_profile_label()} für {config.action.value} erstellt")
metrics = TensorBoardMonitor(config)
file_handler = InputFileHandler(mode, config)
if mode == "full":
    logging.warning("Vollständige Reindexierung wurde angefordert. Alle vorahndenen Metadaten werden gelöscht.")
    file_handler.delete_all_metafiles()
start_time = time.perf_counter()
# _____[ Verarbeitung der neuen PDFs ]__________________________________________
logging.info("_____[ 1/4 Konvertierung Input-Dateien zu Texten ]_____")
source_to_text_start = time.perf_counter()
input_files = file_handler.get_input_files_to_process(config)
logging.info(f"{len(input_files)} Datei(en) werden komplett neu indexiert.")
if len(input_files) > 0:
    metrics.log_metrics({"source_to_text_documents_count": len(input_files)})
    image_to_text_client = ClientBuilder(config).for_image_to_text().build()
    source_to_text_pages = 0
    for original_input_file in input_files:
        if original_input_file.is_processable_as_image():
            pages = original_input_file.get_content()
            source_to_text_pages += len(pages)
            logging.info(f"Bereite Datei {original_input_file.file_name} mit {len(pages)} Seiten vor ...")    
            metrics.log_metrics({original_input_file.file_name + "_pages": len(pages)})        
            meta_file = MetaFile(config, from_input_file=original_input_file)
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
    metrics.log_metrics({"source_to_text_pages": source_to_text_pages})
source_to_text_time = time.perf_counter() - source_to_text_start

# _____[ Text-Splitting, Bereinigung und Chunking ]_____________________________
logging.info("_____[ 2/4 Text-Chunking  ]_____")
chunking_start = time.perf_counter()
chunk_files = file_handler.get_metafiles_to_chunk(config)
logging.info(f"Texte von {len(chunk_files)} Datei(en) werden bereinigt und neu aufgeteilt.")
for meta_file in chunk_files:
    logging.info(f"Verarbeite Datei {meta_file.file_path} ...")
    doc_text = ""
    for page in meta_file.get_pages():     
        doc_text += page["content"] + "\f"
    chunks = clean_texts(split(doc_text, config))
    meta_file.remove_chunks()
    for text in chunks:
         meta_file.add_chunk(text)
    meta_file.save()
chunking_time = time.perf_counter() - chunking_start
# _____[ Erstellen der Embeddings ]_____________________________________________
logging.info("_____[ 3/4 Embeddings erstellen  ]_____")
embedding_start = time.perf_counter()
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
        ix = 0
        for text in meta_file.get_chunks():
            ix += 1
            embeddings = embeddings_client.get_embedding(text)
            original_input_file = meta_file.metadata["input_file"]
            chunk_id = original_input_file + "_" + str(ix)
            #print(f"Id: '{chunk_id}', Text: '{text}', Embeddings: '{embeddings}'")            
            embedding_store.store(text, embeddings, source_document=original_input_file, chunk_id=chunk_id)
        meta_file.metadata["processor"]["embeddings"]["index_id"] = embedding_store.index_id
        meta_file.save()
        metrics.log_metrics(meta_file.get_metrics())
    embedding_store.export_embeddings()
    embedding_store.close()
embedding_time = time.perf_counter() - embedding_start
# _____[ Testen des Modells ]___________________________________________________
logging.info("_____[ 4/4 Testen des Modells  ]_____")
testing_start = time.perf_counter()
evaluator = EvaluatorBuilder(config, embeddings_client, embedding_store).build()
if evaluator.is_advanced():
    eval_result = asyncio.run(evaluator.aevaluate())
else:
    eval_result = evaluator.evaluate()
metrics.log_metrics(eval_result)
testing_time = time.perf_counter() - testing_start
metrics.log_metrics({
    "source_to_text_time": source_to_text_time,
    "chunking_time": chunking_time,
    "embedding_time": embedding_time,
    "testing_time": testing_time,
    "total_time": time.perf_counter() - start_time
    })
logging.info(f"Verarbeitung abgeschlossen. Embedding Store mit Id {embedding_store.index_id} wurde aktualisiert.")