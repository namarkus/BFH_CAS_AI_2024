#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _csv_embedding.py ]____________________________________________________
"""
Einfache Embeddings für die Speicherung in einer lokalen Chroma Datenbank.

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" die Funktionen zur Verfügung, um Embeddings lokal zu hinterlegen.
Die Datenhaltung erfolgt dabei als Chroma-Datenbank.
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()
# _____[ Imports ]______________________________________________________________
import chromadb

from _logging import app_logger
from _apis import EmbeddingStore
from _configs import VbcAction, VbcConfig

class ChromaEmbeddingStore(EmbeddingStore):

    def __init__(self, config: VbcConfig):
        super().__init__(config)
        self.db_client = chromadb.PersistentClient(path="models/")        
        self.index_id = self.config.as_profile_label() + "_ix"
        self.index_id = self.index_id.replace("#", "-")
        self.index_id = self.index_id.replace(".", "-")
        app_logger().debug(f"Chroma-Client initialisiert. Verwende Index {self.index_id}")
        self.collection = self.db_client.get_or_create_collection(self.index_id)

    def is_full_reload_required(self) -> bool:
        return self.config.action == VbcAction.FULL_REINDEXING  or self.collection.count() == 0 
    
    def delete_all(self):
        self.db_client.delete_collection(self.index_id)
        self.collection = self.db_client.create_collection(self.index_id)

    def find_most_similar(self, embedding, top_k=1):
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return results['documents'][0][0]

    def store(self, text, embeddings):   
        self.collection.upsert(ids=[text], embeddings=[embeddings], documents=[text])
