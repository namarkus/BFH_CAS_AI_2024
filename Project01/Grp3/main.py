#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ main.py ]______________________________________________________________
__file__ = "main.py"
__author__ = "BFH-CAS-AI-2024-Grp3"
__copyright__ = "Copyright 2024, BFH-CAS-AI-2024-Grp3"
__credits__ = ["Hans Wermelinger", "Helmut Gehrer", "Markus Näpflin", "Nils Hryciuk", "Steafan Mavilio"]
__license__ = "GPL"
__version__ = "0.9.0"
__status__ = "Development"
__description__ = """
Starterklasse für das Projekt "Versicherungsbedingungs Chat".
"""

import getpass
from cmd import PROMPT
from http import client
from math import e
from pickle import INST
from _logging import start_logger
from _apis import LlmClient, LlmClientConfigurator
from _configs import print_splash_screen, VbcConfig, SupportedLlmProvider
from _file_io import InputFileHandler, InputFile
from _builders import ConfigBuilder, ClientBuilder, EmbeddingStoreBuilder
import sys

# _____[ Laufzeit-Prüfung und Splash-Screen ]___________________________________
if __name__ == "__main__":
    print_splash_screen("vbc_chat", __version__, __author__)
 
    print("""Das Projekt vbc (Vertragsbedingungs-Chat) ist im Rahmen des 
1. Semesterprojekts des CAS AI Herbst 2024 an der BFH entstanden. 

Es dient der Demonstration des Einsatzes von RAG (Retrieval Augmented Generation)
am Beispiel von Versicherungsbedingungen verschiedener schweizer Krankenkassen.          
          
Das Projekt stellt folgende Funktionen in unteschiedlichen Programmen bereit:          
  1) Lernen: Bereitet die Daten für den späteren Einsatz im vbc-chat vor.   
  2) Chatten: Ermöglicht die Interaktion mit dem Fach-Chatbot
  9) Beenden: Beendet das Programm

  Bitte wähle eine der Optionen (1, 2 oder 9) aus                              
""")
                    
choice = 99

while choice != "9":
    choice = input("Deine Wahl: ")
    if choice == "1":
        print("Lernen-Modus wird gestartet...")
        sys.run("vbc_learn.py")
        sys.exit(0)
    elif choice == "2":
        print("Chatten-Modus wird gestartet...")
        # Hier wird der Chat-Modus fortgesetzt
    else:
        print("Ungültige Auswahl.")

