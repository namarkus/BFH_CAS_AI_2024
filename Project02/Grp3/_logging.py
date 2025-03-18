#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _configs.py ]__________________________________________________________
"""
Diverse Mechanismen für das generelle Logging in der Anwendung.

Stellt als integrierter Bestandteil des Projektes "Digitales Markteing mit
TorchRL" Funktionen zur Initialiserung, Konfiguration und Verwendung des Logging
Frameworks von Python bereit.
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze digital_marketing.py.")
    exit()

# _____[ Imports ]______________________________________________________________
import logging
from datetime import datetime

APP_NAME = "vbc_app"
CONSOLE_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-8s [%(filename)s] %(message)s"

def app_logger() -> logging.Logger:
    """Liefert den Anwendungslogger zurück.

    Returns:
        logging.Logger: Konfigurierter Logger der Anwendung (schreibt sowohl in
          die Konsole als auch in eine Log-Datei).
    """
    return logging.getLogger(APP_NAME)

def start_logger(task_name: str, app_stage: str) -> logging.Logger:
    """Initialisiert den Logger für die Anwendung und gibt ihn zurück.

    Args:
        task_name (str): Name des Main-Programms (für den Log-Dateinamen).
        app_stage (str): Stufe der Anweundung (Dev, Test, Prod) für die
            Konfiguration der aufgegebenen Log-Level.

    Returns:
        logging.Logger: Logger der Anwendung.
    """
    start_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = app_logger()
    log_file = f"./logs/{task_name}_{start_datetime}.log"
    if app_stage.startswith("Dev"):
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    console_formatter = logging.Formatter(CONSOLE_FORMAT,datefmt='%H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    file_formatter = logging.Formatter(FILE_FORMAT)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger

