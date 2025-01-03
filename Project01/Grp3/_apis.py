#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _apis.py ]_____________________________________________________________
"""
Applikations-Programmierschnittstellen (APIs) für die dynamische Erweiterung.

Definiert als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" APIs und Abstrakte Klassen. Für Anwendungsteile, welche mehrere 
Implementationen anbieten, wie. z.B. die Integration unterschiedlicher LLMs, 
ist in dieser Datei die gemeinsame Struktur vorgegeben.
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________
from abc import ABC, abstractmethod
from _configs import LlmClientConfig, VbcConfig
from typing import List

class LlmClient(ABC):
    """Superklasse für alle LLM-Clients.

    Args:
        ABC (_type_): _description_
    """
    def __init__(self, vbc_config: VbcConfig):
        self.config = vbc_config
    
    @abstractmethod
    def get_text_from_image(self, base64_encoded_image) -> str:
        pass  # This is an abstract method, no implementation here.
    
    @abstractmethod
    def get_embedding(self, text_to_embed) -> str:
        pass  # This is an abstract method, no implementation here.

    @abstractmethod
    def get_embeddings(self, text_to_embed) -> List[str]:
        pass  # This is an abstract method, no implementation here.

    #@abstractmethod
    def get_response(self, question, embeddings) -> str:
        pass  # This is an abstract method, no implementation here.

    @abstractmethod
    def answer_with_hints(self, question, hints, history=[]) -> str:  
        pass  # This is an abstract method, no implementation here.

    #@abstractmethod
    def test_statement(self, question, expected_answer) -> bool:
        pass  # This is an abstract method, no implementation here.

    def text_from_image_config(self) -> LlmClientConfig:
        return self.config.image_to_text_config
    
    def embeddings_config(self) -> LlmClientConfig:
        return self.config.embeddings_config

    def answer_with_hints_config(self) -> LlmClientConfig:
        return self.config.answer_with_hints_config
    
    def test_statement_config(self) -> LlmClientConfig:
        return self.config.test_cstatement_onfig

class LlmClientConfigurator(ABC):
    def textcontent_from_image_config(self) -> LlmClientConfig:
        pass  # This is an abstract method, no implementation here.
   
    def embeddings_config(self) -> LlmClientConfig:
        pass  # This is an abstract method, no implementation here.

    def answer_with_hits_config(sself) -> LlmClientConfig:
        pass  # This is an abstract method, no implementation here.

    def test_config(self) -> LlmClientConfig:
        pass  # This is an abstract method, no implementation here.


class EmbeddingStore(ABC):

    def __init__(self,  config: VbcConfig):
        self.index_id = config.as_profile_label()
        self.config = config

    def is_full_reload_required(self):
        return self.config.mode == "full"
    
    @abstractmethod
    def delete_all(self):
        pass
    
    @abstractmethod
    def find_most_similar(self, embeddings, top_k=1):
        pass  # This is an abstract method, no implementation here.

    @abstractmethod
    def find_most_similar_docs(self, embeddings, top_k=1):
        pass  # This is an abstract method, no implementation here.

    @abstractmethod
    def store(self, text, embeddings, source_document=None, chunk_id=None):
        pass  # This is an abstract method, no implementation here.

    def close(self):
        pass


class Evaluator(ABC):
    """Superklasse für alle Evaluationen.
    """
    def __init__(self, config: VbcConfig, embeddings_client, embedding_store):
        self.config = config
        self.embeddings_client = embeddings_client
        self.embedding_store = embedding_store

    @abstractmethod
    def is_simple(self):
        pass

    @abstractmethod
    def is_advanced(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    async def aevaluate(self):
        pass
