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
from curses.ascii import EM
import logging
from _apis import LlmClient
from _configs import EmbeddingStorage, SupportedLlmProvider, VbcAction, VbcConfig
from _logging import app_logger
from _openai_config import OpenAiClientConfigurator
from _openai_client import OpenAiClient
from _ollama_config import OllamaClientConfigurator
from _ollama_client import OllamaClient
from _csv_embedding import CsvEmbeddingStore
from _chroma_embedding import ChromaEmbeddingStore

class ConfigBuilder:
    def __init__(self, mode: str, llm: str, version: str):
        self.llm = llm
        self.config = VbcConfig()
        self.config.learn_version = version
        self.config.action = [name for name, member in VbcAction.__members__.items() if member.value == mode]
        self.logger = app_logger()
        if self.llm == "lokal":
            self.logger.info(f"Konfiguriere lokalen LLM-Provider Ollama...")
            self._with_local_llm()
        elif self.llm == "cloud":
            self.logger.info(f"Konfiguriere Cloud LLM-Provider OpenAI...")
            self._with_remote_llm()
        elif self.llm == "auto":
            self.logger.debug(f"Automatische Erkennung des LLM-Providers...")
            if OllamaClient.is_availble():
                self.logger.info(f"Ollama-Client verfügbar. Nutze lokales Ollama für die Konfiguration.")
                self._with_local_llm()
            elif OpenAiClient.is_availble():
                self.logger.info(f"OpenAI-Client verfügbar. Nutze OpenAI in der Cloud für die Konfiguration.")
                self._with_remote_llm()
        else:
            raise ValueError(f"Unbekannter LLM-Provider {self.llm}.")
        self.logger.debug(f"Konfigurations-Builder initalisiert. Nutze {self.configurator} zur Konfiguration der LLM-Engines.")

    def _with_local_llm(self):
        self.configurator = OllamaClientConfigurator()
        self.config.llm_provider = SupportedLlmProvider.OLLAMA # todo als Visitor implementieren
        self.config.embedding_provider = SupportedLlmProvider.OLLAMA
        self.config.embedding_storage = EmbeddingStorage.CHROMA

    def _with_remote_llm(self):
        self.configurator = OpenAiClientConfigurator()
        self.config.llm_provider = SupportedLlmProvider.OPENAI # todo als Visitor implementieren
        self.config.embedding_provider = SupportedLlmProvider.OPENAI
        self.config.embedding_storage = EmbeddingStorage.CHROMA

    def with_image_to_text_config(self):
        self.config.with_image_to_text_config(self.configurator.textcontent_from_image_config())
        return self
    
    def with_embedding_config(self):
        self.config.with_embedding_config(self.configurator.embeddings_config())
        return self
    
    def with_response_config(self):
        self.config.with_answer_with_hints_config(self.configurator.response_config())
        return self


    def build(self):
        return self.config

class ClientBuilder:
    def __init__ (self, config: VbcConfig):
        self.config = config
        self.logger = app_logger()

    def for_image_to_text(self):
        self.logger.debug(f"Konfiguriere Client für Bild-zu-Text-Konvertierung mit Modell {self.config.image_to_text_config.model_id}...")
        self.client = self._from_enum(self.config.llm_provider)
        return self
    
    def _from_enum(self, value: SupportedLlmProvider) -> LlmClient:
        match value:
            case SupportedLlmProvider.OLLAMA:
                return OllamaClient(self.config)
            case SupportedLlmProvider.OPENAI:
                return OpenAiClient(self.config)
            case _:
                raise ValueError(f"Unsupported LLM-Provider {value}.")
    
    def for_embeddings(self):
        self.logger.debug(f"Konfiguriere Client für Embeddings mit Modell {self.config.embedding_config.model_id}...")
        self.client = self._from_enum(self.config.embedding_provider)
        return self

    def for_response(self):
        self.logger.debug(f"Konfiguriere Client für Response mit Modell {self.config.answer_with_hints_config.model_id}...")
        self.client = self._from_enum(self.config.embedding_provider) # fixme
        return self

    def build(self):
        return self.client
    
class EmbeddingStoreBuilder:
    def __init__(self, config: VbcConfig):
        self.config = config
        self.logger = app_logger()
        match config.embedding_storage:
            case EmbeddingStorage.CSV: 
                self._for_csv()
            case EmbeddingStorage.CHROMA: 
                self._for_chroma(),
            case EmbeddingStorage.PINECONE: 
                self._for_pinecone(),
            case _:
                raise ValueError(f"Unsupported Embedding-Storage {self.config.embedding_storage}.")

    def _for_csv(self):
        self.logger.debug(f"Konfiguriere CSV-Store für Embeddings...")
        self.store = CsvEmbeddingStore(self.config) # todo dynamisieren
        return self
    
    def _for_chroma(self):
        self.logger.debug(f"Konfiguriere lokale Chroma DB für Embeddings...")
        self.store = ChromaEmbeddingStore(self.config) # todo dynamisieren
        return self
    
    def _for_pinecone(self):
        #self.logger.debug(f"Konfiguriere CSV-Store für Embeddings...")
        #self.store = CsvEmbeddingStore(self.config) # todo dynamisieren
        return self

    def build(self):
        return self.store    
