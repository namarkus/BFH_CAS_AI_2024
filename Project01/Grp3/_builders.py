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
from _apis import LlmClient
from _configs import EmbeddingStorage, SupportedLlmProvider, EvaluationMode, VbcAction, VbcConfig
from _logging import app_logger
from _openai_config import OpenAiClientConfigurator
from _openai_client import OpenAiClient
from _ollama_config import OllamaClientConfigurator
from _ollama_client import OllamaClient
from _csv_embedding import CsvEmbeddingStore
from _chroma_embedding import ChromaEmbeddingStore
from _simple_evaluator import SimpleEvaluator
from _advanced_evaluator import AdvancedEvaluator

class ConfigBuilder:
    def __init__(self, mode: str, llm: str, version: str, required_config_id: str = None):
        self.llm = llm
        self.logger = app_logger()
        self.config = VbcConfig()
        if llm == "from_config":
            if required_config_id is None:
                raise ValueError("required_config_id muss angegeben werden, wenn llm 'from_config' ist.")
            self.config.from_profile_label(required_config_id)
            self._with_required_config(self.config)
        else:
            self.config.learn_version = version
            self.config.action = [member for member, member in VbcAction.__members__.items() if member.value == mode][0]
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
        self.config.image2text_llm_provider = SupportedLlmProvider.OPENAI # Lokale LLM-Engine nicht in gewünschter Qualität verfügbar
        self.config.embeddings_provider = SupportedLlmProvider.OLLAMA
        self.config.embeddings_storage = EmbeddingStorage.CHROMA
        self.config.chat_llm_provider = SupportedLlmProvider.OLLAMA 

    def _with_remote_llm(self):
        self.configurator = OpenAiClientConfigurator()
        self.config.image2text_llm_provider = SupportedLlmProvider.OPENAI 
        self.config.embeddings_provider = SupportedLlmProvider.OPENAI
        self.config.embeddings_storage = EmbeddingStorage.CHROMA
        self.config.chat_llm_provider = SupportedLlmProvider.OPENAI 

    def _with_required_config(self, prepared_config: VbcConfig):
        match prepared_config.chat_llm_provider:
            case SupportedLlmProvider.OLLAMA:
                self.configurator = OllamaClientConfigurator()
            case SupportedLlmProvider.OPENAI:
                self.configurator = OpenAiClientConfigurator()
            case _:
                raise ValueError(f"Unsupported LLM-Provider {prepared_config.chat_llm_provider}.")
        
    def with_image_to_text_config(self):
        self.config.with_image_to_text_config(self.configurator.textcontent_from_image_config())
        return self
    
    def with_embeddings_config(self):
        self.config.with_embeddings_config(self.configurator.embeddings_config())
        return self
    
    def with_answer_with_hits_config(self):
        self.config.with_answer_with_hints_config(self.configurator.answer_with_hits_config())
        return self

    def with_test_statement_config(self):
        self.config.with_test_statement_config(self.configurator.test_statement_config())
        return self

    def build(self):
        return self.config

class ClientBuilder:
    def __init__ (self, config: VbcConfig):
        self.config = config
        self.logger = app_logger()

    def for_image_to_text(self):
        self.logger.debug(f"Konfiguriere Client für Bild-zu-Text-Konvertierung mit Modell {self.config.image_to_text_config.model_id}...")
        self.client = self._from_enum(self.config.image2text_llm_provider)
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
        self.logger.debug(f"Konfiguriere Client für Embeddings mit Modell {self.config.embeddings_config.model_id}...")
        self.client = self._from_enum(self.config.embeddings_provider)
        return self

    def for_answer_with_hints(self):
        self.logger.debug(f"Konfiguriere Client für Response mit Modell {self.config.answer_with_hints_config.model_id}...")
        self.client = self._from_enum(self.config.chat_llm_provider) # fixme
        return self
    
    def for_test_statement(self):
        self.logger.debug(f"Konfiguriere Client für Test-Statement mit Modell {self.config.test_statement_config.model_id}...")
        self.client = self._from_enum(self.config.chat_llm_provider)
        return self

    def build(self):
        return self.client
    
class EmbeddingStoreBuilder:
    def __init__(self, config: VbcConfig):
        self.config = config
        self.logger = app_logger()
        match config.embeddings_storage:
            case EmbeddingStorage.CSV: 
                self._for_csv()
            case EmbeddingStorage.CHROMA: 
                self._for_chroma(),
            case EmbeddingStorage.PINECONE: 
                self._for_pinecone(),
            case _:
                raise ValueError(f"Unsupported Embedding-Storage {self.config.embeddings_storage}.")

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

class EvaluatorBuilder:
    def __init__(self, config: VbcConfig, embeddings_client, embedding_store):
        self.config = config
        self.embeddings_client = embeddings_client
        self.embedding_store = embedding_store
        self.logger = app_logger()
        match config.evaluation_mode:
            case EvaluationMode.NONE:
                self._no_evaluation()
            case EvaluationMode.SIMPLE:
                self._simple_evaluation(),
            case EvaluationMode.ADVANCED:
                self._advanced_evaluation(),
            case _:
                raise ValueError(f"Unsupported evaluation mode {self.config.evaluation_mode}.")

    def _no_evaluation(self):
        self.logger.debug(f"Keine Evaluation gewählt...")
        return self

    def _simple_evaluation(self):
        self.logger.debug(f"Einfache Evaluation...")
        self.evaluator = SimpleEvaluator(self.config, self.embeddings_client, self.embedding_store)  # todo dynamisieren
        return self

    def _advanced_evaluation(self):
        self.logger.debug(f"Erweiterte Evaluation...")
        self.evaluator = AdvancedEvaluator(self.config, self.embeddings_client, self.embedding_store) # todo dynamisieren
        return self

    def build(self):
        return self.evaluator
