#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _csv_embedding.py ]____________________________________________________
"""
Einfache Embeddings für die lokale Speicherung.

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" die Funktionen zur Verfügung, um Embeddings lokal zu hinterlegen.
Die Datenhaltung erfolgt dabei als CSV-Datei.
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()
# _____[ Imports ]______________________________________________________________
import os
import pandas as pd
import numpy as np
from ast import literal_eval

from _apis import EmbeddingStore
from _configs import VbcAction, VbcConfig

class CsvEmbeddingStore(EmbeddingStore):

    def __init__(self, config: VbcConfig):
        super().__init__(config)
        self.file_path = f"{config.model_path}/{self.index_id}.csv"
        self.__load()

    def __load(self):
        if os.path.exists(self.file_path):
            df = pd.read_csv(self.file_path)
            df["embeddings"] = df.embeddings.apply(literal_eval).apply(np.array)
        else:
            df = pd.DataFrame(columns=["text", "embeddings"])
        self.df = df

    def is_full_reload_required(self) -> bool:
        return self.config.action == VbcAction.FULL_REINDEXING or self.df.empty
    
    def delete_all(self):
        self.df = pd.DataFrame(columns=["text", "embeddings"])

    def find_most_similar(self, embedding, top_k=1):
        # TODO: as question is smaller, maybe fill
        similarities = [self.__cosine_similarity(embedding, emb) for emb in self.df.embeddings]
        most_similar_idx = np.argmax(similarities)
        return most_similar_idx, similarities[most_similar_idx]

    def __cosine_similarity(self, embedding1, embedding2):
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def store(self, text, embeddings):   
        # if text in self.df["text"].values:
        #     self.df.loc[self.df["text"] == text, "embeddings"] = embeddings
        # else:
        new_row = {"text": text, "embeddings": embeddings}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def close(self):
        self.df.to_csv(self.file_path, index=False) 

