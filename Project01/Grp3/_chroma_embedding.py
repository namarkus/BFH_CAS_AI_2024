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
import os
import pandas as pd
import numpy as np

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
        return results['documents'][0]

    def find_most_similar_docs(self, embedding, top_k=1):
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        source_docs = [metadata['source_document'] for metadata in results['metadatas'][0]]
        return source_docs

    def store(self, text, embeddings, source_document=None, chunk_id=None):
        if chunk_id is None:
            chunk_id = source_document + "_" + str(len(self.collection))
        brand = source_document.split("_")[0]
        meta_data = {
            "source_document": source_document,
            "brand": brand
        }
        self.collection.upsert(ids=[chunk_id], embeddings=[embeddings], documents=[text], metadatas=[meta_data])

    def export_embeddings(self):
        if self.config.export_embeddings:
            file_path_embeddings = os.path.normpath('models/' + self.index_id + '.tsv')
            file_path_embeddings_text = os.path.normpath('models/' + self.index_id + '_text.tsv')

            results = self.collection.get(include= ['embeddings', "documents"])
            embeddings = pd.DataFrame(results['embeddings'])
            embeddings_text = pd.DataFrame({
                'id': results['ids'],
                'document': results['documents']
            })

            embeddings.to_csv(file_path_embeddings, sep='\t', index=False, header=False)
            embeddings_text.to_csv(file_path_embeddings_text, sep='\t', index=False, header=True)


