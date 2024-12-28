#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ vbc_chat.py ]__________________________________________________________
__file__ = "vbc_chat.py"
__author__ = "BFH-CAS-AI-2024-Grp3"
__copyright__ = "Copyright 2024, BFH-CAS-AI-2024-Grp3"
__credits__ = ["Hans Wermelinger", "Helmut Gehrer", "Markus Näpflin", "Nils Hryciuk", "Steafan Mavilio"]
__license__ = "GPL"
__version__ = "0.9.1"
__status__ = "Development"
__description__ = """
tbd
"""

import getpass
from _logging import start_logger
from _apis import LlmClient, LlmClientConfigurator
from _configs import print_splash_screen, VbcConfig, SupportedLlmProvider
from _builders import ConfigBuilder, ClientBuilder, EmbeddingStoreBuilder

# _____[ Laufzeit-Prüfung und Splash-Screen ]___________________________________
if __name__ == "__main__":
    print_splash_screen("vbc_chat", __version__, __author__)
logging = start_logger("vbc_chat", __status__)
config = ConfigBuilder("chat", "auto", __version__).with_embedding_config().with_response_config().build()
logging.info(f"Konfiguration mit Profil-Id {config.as_profile_label()} erstellt")
chat_client = ClientBuilder(config).for_response().build()
embedding_client = ClientBuilder(config).for_embeddings().build()
embedding_store = EmbeddingStoreBuilder(config).build()
PROMPT = "> "
CHATBOT_PROMPT = f"🤖{PROMPT}"
USER_NAME = getpass.getuser()
USER_PROMPT = f"{USER_NAME}{PROMPT}"
INSTRUCTIONS = f"""Hallo {USER_NAME}, 
Ich bin ein einfacher Chatbot für Krankenkassen-Vertragsbedingungen. Aufgrund meines 
Wissens und der Hilfe von {config.embedding_provider.value} kann ich Dir Fragen zu diesem Thema beantworten.

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
assistant_session = []
while True: 
    user_input = input(USER_PROMPT)
    if user_input == "/bye" or user_input == "bye":
        print(f"{CHATBOT_PROMPT}Tschüss {USER_NAME}, bis bald 👋")
        break
    if user_input == "/init":
        logging.debug("Alles Sessiondaten werden gelöscht.")
        assistant_session = []
        print(f"{CHATBOT_PROMPT}Huch, wer bin ich, und was mache ich hier? 🤔")
    if user_input == "/?" or user_input == "/hilfe" or user_input == "hilfe" or user_input == "?":
        print(f"{CHATBOT_PROMPT}{INSTRUCTIONS}")
    else:
        logging.debug(f"Frage: '{user_input}' wird verarbeitet...")
        embeddings = embedding_client.get_embeddings(user_input)
        hints = embedding_store.find_most_similar(embeddings, top_k=5) 
        logging.debug(f"Frage: '{user_input}' erhält via Embeddings folgende Hinweise '{hints}'")
        response = chat_client.answer_with_hints(user_input, hints, assistant_session)
        # TODO Messen der Performance und in Metriken hinterlegen.
        print(f"{CHATBOT_PROMPT} {response}")
        print("---------------------------------------------")

