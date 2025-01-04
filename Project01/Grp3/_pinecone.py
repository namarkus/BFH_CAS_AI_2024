#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _csv_embedding.py ]____________________________________________________
"""
Implementation von Pinecone für Embeddings. 

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" die Funktionen zur Verfügung, um Embeddings in der Cloiud zu hinterlegen.
Da Pineocde sowohl das Erstellen von Embeddings als auch die Speicherung 
unterstützt, erbt diess Klasse sowohl von LlmClient als auch von EmbeddingStore.
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()
# _____[ Imports ]______________________________________________________________
from _logging import app_logger
from _apis import EmbeddingStore, LlmClient
from _configs import VbcAction, VbcConfig

class PineConeEmbeddings(EmbeddingStore, LlmClient):

    def __init__(self, config: VbcConfig):
        super().__init__(config)
        self.index_id = self.config.as_profile_label() + "_ix"
        self.index_id = self.index_id.replace("#", "-")
        self.index_id = self.index_id.replace(".", "-")
        app_logger().debug(f"PineConeEmbeddings initialisiert. Verwende Index {self.index_id}")

    def get_embeddings(self, text):
        api_call_config = self.embeddings_config()
        #todo implement

    def is_full_reload_required(self) -> bool:
        return self.config.action == VbcAction.FULL_REINDEXING # or self.collection.count() == 0 
    
    def delete_all(self):
        pass # todo implement

    def find_most_similar(self, embedding, top_k=1):
        pass # todo implement

    def store(self, text, embeddings, source_document=None, chunk_id=None):   
        pass # todo implement upsert
