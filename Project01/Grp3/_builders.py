#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _builders.py ]_________________________________________________________
"""
Einfache Factories für die dynamische Erweiterung.

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" Builder-Klassen zur Verfügung. Diese ermöglichen die dynamische 
Zusammenstellung von verschiedenen Klassen.
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()
# _____[ Imports ]______________________________________________________________
import logging
from _configs import EmbeddingStorage, VbcAction, VbcConfig
from _logging import app_logger
from _openai_config import OpenAiClientConfigurator
from _openai_client import OpenAiClient
from _csv_embedding import CsvEmbeddingStore

class ConfigBuilder:
    def __init__(self, mode: str, llm: str, version: str):
        self.llm = llm
        self.config = VbcConfig()
        self.config.learn_version = version
        self.config.action = [name for name, member in VbcAction.__members__.items() if member.value == mode]
        self.logger = app_logger()
        self.configurator = OpenAiClientConfigurator() # fixme dynamisieren
        self.logger.debug(f"Konfigurations-Builder initalisiert. Nutze {self.configurator} zur Konfiguration der LLM-Engines.")

    def with_image_to_text_config(self):
        self.config.with_image_to_text_config(self.configurator.textcontent_from_image_config())
        return self
    
    def with_embedding_config(self):
        self.config.with_embedding_config(self.configurator.embeddings_config())
        return self

    def build(self):
        return self.config

class ClientBuilder:
    def __init__ (self, config: VbcConfig):
        self.config = config
        self.logger = app_logger()

    def for_image_to_text(self):
        self.logger.debug(f"Konfiguriere Client für Bild-zu-Text-Konvertierung mit Modell {self.config.image_to_text_config.model_id}...")
        self.client = OpenAiClient(self.config)        
        return self
    
    def for_embeddings(self):
        self.logger.debug(f"Konfiguriere Client für Embeddings mit Modell {self.config.embedding_config.model_id}...")
        self.client = OpenAiClient(self.config)        
        return self

    def build(self):
        return self.client
    
class EmbeddingStoreBuilder:
    def __init__(self, config: VbcConfig):
        self.config = config
        self.logger = app_logger()
        if self.config.embedding_storage == EmbeddingStorage.CSV:
            self.for_csv()

    def for_csv(self):
        self.logger.debug(f"Konfiguriere CSV-Store für Embeddings...")
        self.store = CsvEmbeddingStore(self.config)
        return self

    def build(self):
        return self.store    
