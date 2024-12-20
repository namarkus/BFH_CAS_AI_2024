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

    #@abstractmethod
    def test_statement(self, question, expected_answer) -> bool:
        pass  # This is an abstract method, no implementation here.

    def text_from_image_config(self) -> LlmClientConfig:
        return self.config.image_to_text_config
    
    def embeddings_config(self) -> LlmClientConfig:
        return self.config.embeddings_config
    


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
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []

        if os.path.exists(self.file_path):
            self.load_embeddings()

    def save(self, embeddings, texts):
        self.data = [{"embedding": embedding, "text": text} for embedding, text in zip(embeddings, texts)]
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f)

    def load_embeddings(self):
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)

    def get_embedding(self, index):
        return self.data[index]['embedding'] if 0 <= index < len(self.data) else None

    def get_text(self, index):
        return self.data[index]['text'] if 0 <= index < len(self.data) else None

    def get_all_embeddings(self):
        return [item['embedding'] for item in self.data]

    def __cosine_similarity(self, embedding1, embedding2):
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def find_most_similar(self, embedding, embeddings_list):
        # TODO: as question is smaller, maybe fill
        similarities = [self.__cosine_similarity(embedding, emb) for emb in embeddings_list]
        most_similar_idx = np.argmax(similarities)
        return most_similar_idx, similarities[most_similar_idx]
    
    def ccc():
        df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(np.array(x).reshape(1,-1), np.array(embedded_value).reshape(1, -1)))
        res = df.sort_values('similarity', ascending=False).head(top_k)


