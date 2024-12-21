#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _configs.py ]__________________________________________________________
"""
Diverse Mechanismen für die Konfiguration der Anwendung.

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" Datenstrukturen zur Verfügung, die als ganzes an Builder, Funktionen  etc. 
übergeben werden können. Und ein kleiner freundlicher Roboter hat sich hier auch 
noch versteckt. ;-)
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________
from enum import Enum
from dataclasses import dataclass
from typing import Union, Optional, List

class SupportedLlmProvider(Enum):
    """_summary_
    Enum, welche die unterstützten LLM-Provider einschränkt und klar kmategorisiert.
    Args:
        Enum (_type_): _description_
    """
    OPENAI = "OpenAI"
    HUGGINGFACE = "Huggingface"
    OLLAMA = "Ollama"


@dataclass
class LlmClientConfig:   
    """
    Konfiguration für den Zugriff auf verschiedene LLM Apis. 
    """
    model_id: str # Name des zu verwendenden Modells, z.B. "gpt-4o"; obligatorisch
    max_tokens: Optional[int] = 1000 # Maximale Anzahl Tokens, die generiert werden sollen.
    temperature: Optional[float] = 0.0  # Temperatur für die Generierung der Texte
    top_p: Optional[float] = 0.1    # Top-P Wert für die Generierung der Texte
    system_prompt: Optional[str] = None    # System-Prompt für die Generierung der Texte
    user_prompt: Optional[str] = None    # User-Prompt für die Generierung der Texte    

@dataclass
class VbcConfig:   
    # TODO #3: Pfad anpassen, auf "./zvb_pdfs", sobald gewisser Maturitätsgrad vorhanden ist.
    sources_path: str = "./input"     # Quellpfad mit den Originaldaten
    knowledge_repository_path: str = "./work"   # Pfad mit den vorverarbeiteten Dateien
    #image_to_text_config: Optional[LlmClientConfig]     # Client-Konfiguration für die Bild-zu-Text-Konvertierung

    def with_image_to_text_config(self, image_to_text_config: LlmClientConfig):
        self.image_to_text_config = image_to_text_config
        return self

    def with_embedding_config(self, embedding_config: LlmClientConfig):
        self.embedding_config = embedding_config
        return self

    # llm_backend: Union[str, list[str]]  # name of the dataset files, including directory and extension 
    # target_cols: Union[str, list[str]]     # used to indicate the target
    # header_time: Optional[str]     # used to parse the dataset file
    # group_cols: Union[str, list[str]]     # used to create group series
    # past_cov_cols: Union[str, list[str]]     # used to select past covariates
    # static_cols: Union[str, list[str]] = None     # used to select static cols
    # # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    # format_time: Optional[str] = None     # used to convert the string date to pd.Datetime
    # freq: Optional[str] = None     # used to indicate the freq when we already know it
    # multivariate: Optional[bool] = None     # multivariate
    # year_cutback: Optional[int] = None    # Limitieren der zu bearbeitenden Daten auf die letzen x Jahre
    # training_cutoff: Optional[float] = 0.5     # cutoff


def print_splash_screen(app_name: str, app_version: str, app_author: str):
    print(f"""
   ___⩟___
  /       \       _   _____  _____
 |  ^   ^  |     | | / / _ )/ ___/
 |  .___.  |     | |/ / _  / /__  
  \_______/      |___/____/\___/      Vertragsbedingungs-Chat
   /     \     
  |  ---  |      {app_name} v{app_version} by {app_author}
 """)
