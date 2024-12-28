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
from sklearn.metrics.pairwise import cosine_similarity
from _configs import LlmClientConfig, VbcConfig

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
    def get_embeddings(self, text_to_embed) -> str:
        pass  # This is an abstract method, no implementation here.

    #@abstractmethod
    def get_response(self, question, embeddings) -> str:
        pass  # This is an abstract method, no implementation here.

    @abstractmethod
    def answer_with_hints(self, question, hints, chat_thread):  
        pass  # This is an abstract method, no implementation here.

    #@abstractmethod
    def test_statement(self, question, expected_answer) -> bool:
        pass  # This is an abstract method, no implementation here.

    def text_from_image_config(self) -> LlmClientConfig:
        return self.config.image_to_text_config
    
    def embeddings_config(self) -> LlmClientConfig:
        return self.config.embedding_config

    def answer_with_hints_config(self) -> LlmClientConfig:
        return self.config.answer_with_hints_config



class LlmClientConfigurator(ABC):
    def textcontent_from_image_config(self) -> LlmClientConfig:
        pass  # This is an abstract method, no implementation here.
   
    def embeddings_config(self) -> LlmClientConfig:
        pass  # This is an abstract method, no implementation here.

    def response_config(sself) -> LlmClientConfig:
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
    
    # def get_embedding(self, index):
    #     pass  # This is an abstract method, no implementation here.

    # def get_text(self, index):
    #     pass  # This is an abstract method, no implementation here.

    # def get_all_embeddings(self):
    #     pass  # This is an abstract method, no implementation here.

    @abstractmethod
    def find_most_similar(self, embeddings, top_k=1):
        pass  # This is an abstract method, no implementation here.

    @abstractmethod
    def store(self, text, embeddings):
        pass  # This is an abstract method, no implementation here.

    def close(self):
        pass 

    # def ccc():
    #     df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(np.array(x).reshape(1,-1), np.array(embedded_value).reshape(1, -1)))
    #     res = df.sort_values('similarity', ascending=False).head(top_k)


