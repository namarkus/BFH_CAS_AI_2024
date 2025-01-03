#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ vbc_chat.py ]__________________________________________________________
__file__ = "vbc_chat.py"
__author__ = "BFH-CAS-AI-2024-Grp3"
__copyright__ = "Copyright 2024, BFH-CAS-AI-2024-Grp3"
__credits__ = ["Hans Wermelinger", "Helmut Gehrer", "Markus Näpflin", "Nils Hryciuk", "Steafan Mavilio"]
__license__ = "GPL"
__version__ = "0.9.2"
__status__ = "Test"
__description__ = """
Dies ist die Chat-Anwendung für den Versicherungsbedingungs-Chat. Sie öffnet 
einen einfachen interaktiven Chat als CLI (Command Line Interface) und ermöglicht
es dem Benutzer, Fragen zu Krankenkassen-Vertragsbedingungen zu stellen. Der 
Chatbot basiert dabei auf RAG (Retrieval-Augmented Generation), welches mit den
vorbereiteten Krankenkassen-Vertragsbedingungen mithilfe von vbc_learn trainiert 
worden ist.
"""

import argparse
import getpass
from tkinter import dialog
from _logging import start_logger
from _configs import print_splash_screen
from _builders import ConfigBuilder, ClientBuilder, EmbeddingStoreBuilder

# _____[ Laufzeit-Prüfung und Splash-Screen ]___________________________________
if __name__ == "__main__":
    print_splash_screen("vbc_chat", __version__, __author__)
logging = start_logger("vbc_chat", __status__)
# _____[ Parameterparser initialisieren ]_______________________________________
logging.debug("Werte übergebene Parameter aus  ...")
parser = argparse.ArgumentParser(description=__description__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
parser.add_argument("--config", type=str, 
                    help="""Zu ladende Konfiguration. Wird keines angegeben, so 
                    wird ermittelt, welche LLM zur Verfügung steht und versucht, 
                    für diese die aktuellste Configuration zu laden. 
                    Ausschlaggebend hierfür ist die Konfigurationsklasse 
                    'VbcConfig'. Die Namen für die Konfigurationen werden nach
                    dem Schema 'image2text-llm-provider#embeddings-provider_chunking-mode_embeddings-storage#chat_llm_provider#learn_version'
                    vergbben, also z.B. 'openai#openai_chroma_document#openai#1.0.0'""")
cli_arguments = parser.parse_args()
llm = cli_arguments.llm
if cli_arguments.config:
    config_description = cli_arguments.config
    config = ConfigBuilder("chat", "from_config", __version__, required_config_id=config_description).with_embeddings_config().with_answer_with_hits_config().build()
    logging.info(f"Verwende gewünschte Konfiguration mit Profil-Id {config.as_profile_label()}.")
else:
    config = ConfigBuilder("chat", llm, __version__).with_embeddings_config().with_answer_with_hits_config().build()
logging.info(f"Konfiguration mit Profil-Id {config.as_profile_label()} erstellt")
chat_client = ClientBuilder(config).for_answer_with_hints().build()
embedding_client = ClientBuilder(config).for_embeddings().build()
embedding_store = EmbeddingStoreBuilder(config).build()
PROMPT = "> "
CHATBOT_PROMPT = f"🤖{PROMPT}"
USER_NAME = getpass.getuser()
USER_PROMPT = f"{USER_NAME}{PROMPT}"
INSTRUCTIONS = f"""Hallo {USER_NAME}, 
Ich bin ein einfacher Chatbot für Krankenkassen-Vertragsbedingungen. Aufgrund meines 
Wissens und der Hilfe von {config.embeddings_provider.value} kann ich Dir Fragen zu diesem Thema beantworten.

Mein Wissen basiert im Moment auf meiner Modellversion {config.as_profile_label()}. 

Mit der Option --profile beim Start des Chats kannst Du das Modell für unser Gespräch 
wechseln. Weitere Optionen siehst Du auch, wenn Du in Deinem Terminal das Kommando 
'python vbc_chat.py --help' eingibst.

Folgende Kommandos sind hier im Chat anstelle einer Frage möglich:
- '/bye' beendet unser Gespräch.
- '/hilfe', '/?' zeigt diese Hilfe an.
- '/init' initialisiert unseren Chat neu. Ich vergesse dann alles, was wir bisher 
  besprochen haben.

Gib bitte Deine Frage zum Thema Krankenversicherung hier ein:
"""
# _____[ Parameterparser initialisieren ]_______________________________________
print ("\n")
print(f"{CHATBOT_PROMPT}{INSTRUCTIONS}")
user_input = ""
chat_session = []
while True: 
    # todo: mit curses oder readline eine bessere Eingabe ermöglichen (z.B. Pfeiltasten für History)
    user_input = input(USER_PROMPT)
    if user_input == "/bye" or user_input == "bye":
        print(f"{CHATBOT_PROMPT}Tschüss {USER_NAME}, bis bald 👋")
        break
    elif user_input == "/init":
        logging.debug("Alles Sessiondaten werden gelöscht.")
        chat_session = []
        print(f"{CHATBOT_PROMPT}Huch, wer bin ich, und was mache ich hier? 🤔")
    elif user_input == "/?" or user_input == "/hilfe" or user_input == "hilfe" or user_input == "?":
        print(f"{CHATBOT_PROMPT}{INSTRUCTIONS}")
    else:
        logging.debug(f"Frage: '{user_input}' wird verarbeitet...")
        extended_user_input = ""
        for dialog in chat_session:
            extended_user_input += f"{dialog['question']} \n"
        extended_user_input += user_input
        logging.debug(f"Habe Kontext für die Ermittlung der Embeddings auf '{extended_user_input}' erweitert.")
        embeddings = embedding_client.get_embedding(extended_user_input)
        hints = embedding_store.find_most_similar(embeddings, top_k=5) 
        logging.debug(f"Frage: '{user_input}' erhält via Embeddings folgende Hinweise '{hints}'")
        response = chat_client.answer_with_hints(user_input, hints, history=chat_session)
        dialog = {"question": user_input, "response": response}
        chat_session.append(dialog)
        # TODO Messen der Performance und in Metriken hinterlegen.
        print(f"{CHATBOT_PROMPT} {response}")
        logging.debug("---------------------------------------------")

