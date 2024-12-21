#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ vbc_chat.py ]__________________________________________________________
__file__ = "vbc_chat.py"
__author__ = "BFH-CAS-AI-2024-Grp3"
__copyright__ = "Copyright 2024, BFH-CAS-AI-2024-Grp3"
__credits__ = ["Hans Wermelinger", "Helmut Gehrer", "Markus Näpflin", "Nils Hryciuk", "Steafan Mavilio"]
__license__ = "GPL"
__version__ = "0.9.0"
__status__ = "Development"
__description__ = """
tbd
"""

from _logging import start_logger
from _apis import LlmClient, LlmClientConfigurator
from _configs import print_splash_screen, VbcConfig, SupportedLlmProvider
from _file_io import InputFileHandler, InputFile
from _builders import ConfgBuilder, ClientBuilder

# _____[ Laufzeit-Prüfung und Splash-Screen ]___________________________________
if __name__ == "__main__":
    print_splash_screen("vbc_chat", __version__, __author__)
logging = start_logger("vbc_chat", __status__)

# _____[ Parameterparser initialisieren ]_______________________________________
logging.warning("Hier ist noch nichts implementiert.")
