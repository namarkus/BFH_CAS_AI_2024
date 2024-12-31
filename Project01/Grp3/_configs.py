#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _configs.py ]__________________________________________________________
"""
Diverse Mechanismen für die Konfiguration der Anwendung.

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" Datenstrukturen zur Verfügung, die als ganzes an Builder, Funktionen  etc. 
übergeben werden können. Und ein kleiner freundlicher Roboter hat sich hier auch 
noch versteckt. ;-)
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class SupportedLlmProvider(Enum):
    """_summary_
    Enum, welche die unterstützten LLM-Provider einschränkt und klar kategorisiert.
    Args:
        Enum (_type_): _description_
    """
    OPENAI = "OpenAI"
    HUGGINGFACE = "Huggingface"
    OLLAMA = "Ollama"

class ChunkingMode(Enum):
    """_summary_
    Enum, welche die unterstützten Chunking-Modi vordefiniert.
    Args:
        Enum (_type_): _description_
    """
    DOCUMENT = "document"
    PAGE = "page"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"    

class EmbeddingStorage(Enum):
    """_summary_
    Enum, welche die unterstützten Speicher-Varianten für Embeddings einschränkt 
    und kategorisiert.
    Args:
        Enum (_type_): _description_
    """
    CSV = "csv"
    PINECONE = "PaineCone"
    CHROMA = "Chroma"

class VbcAction(Enum):
    """_summary_
    Enum, welche die unterstützten Aktionen im VBC-Chat einschränkt und kategorisiert.
    Args:
        Enum (_type_): _description_
    """
    INCREMENTAL_REINDEXING = "inc"
    FULL_REINDEXING = "full"
    CHAT = "chat"


@dataclass
class LlmClientConfig:   
    """
    Konfiguration für den Zugriff auf verschiedene LLM Apis. 
    """
    model_id: str # Name des zu verwendenden Modells, z.B. "gpt-4o"; obligatorisch
    max_tokens: Optional[int] = 1000 # Maximale Anzahl Tokens, die generiert werden sollen.
    temperature: Optional[float] = 0.0  # Temperatur für die Generierung der Texte
    top_p: Optional[float] = 0.1    # Top-P Wert für die Generierung der Texte
    system_prompt: Optional[str] = None    # System-Prompt für die Generierung der Texte
    user_prompt: Optional[str] = None    # User-Prompt für die Generierung der Texte    

@dataclass
class VbcConfig:   
    action: VbcAction = VbcAction.CHAT    # Aktion, die durchgeführt werden soll
    sources_path: str = "./input"     # Quellpfad mit den Originaldaten
    model_path: str = "./models"    # Pfad mit den Modellen
    knowledge_repository_path: str = "./work"   # Pfad mit den vorverarbeiteten Dateien
    language: str = "german"    # Sprache der Daten
    learn_version = "0.1"    # Version des Lernmoduls
    chunking_mode: ChunkingMode = ChunkingMode.SECTION  # Modus für die Chunk-Bildung
    image2text_llm_provider: SupportedLlmProvider = SupportedLlmProvider.OPENAI  # LLM-Provider für Image2Text
    embeddings_provider: SupportedLlmProvider = SupportedLlmProvider.OPENAI  # Embedding-Provider
    embeddings_storage: EmbeddingStorage = EmbeddingStorage.CHROMA  # Speicherort für Embeddings
    chat_llm_provider: SupportedLlmProvider = SupportedLlmProvider.OPENAI  # LLM-Provider für den Chat selbst
    export_embeddings: bool = True

    def with_image_to_text_config(self, image_to_text_config: LlmClientConfig):
        self.image_to_text_config = image_to_text_config
        return self

    def with_embeddings_config(self, embeddings_config: LlmClientConfig):
        self.embeddings_config = embeddings_config
        return self
    
    def with_answer_with_hints_config(self, answer_with_hits_config: LlmClientConfig):
        self.answer_with_hints_config = answer_with_hits_config
        return self
    
    def with_test_statement_config(self, test_statement_config: LlmClientConfig):
        self.test_statement_config = test_statement_config
        return self

    def as_profile_label(self):
        return f"{self.image2text_llm_provider.value}#{self.embeddings_provider.value}_{self.chunking_mode.value}_{self.embeddings_storage.value}#{self.chat_llm_provider.value}#{self.learn_version}"

    def from_profile_label(self, mode):
        profile_parts = mode.split("#")
        self.image2text_llm_provider = SupportedLlmProvider(profile_parts[0])
        embedding_parts = profile_parts[1].split("_")
        self.embeddings_provider = SupportedLlmProvider(embedding_parts[0])
        self.chunking_mode = ChunkingMode(embedding_parts[1])
        self.embeddings_storage = EmbeddingStorage(embedding_parts[2].split(":")[0])
        self.chat_llm_provider = SupportedLlmProvider(profile_parts[2])
        self.learn_version = profile_parts[3]
        return self

def print_splash_screen(app_name: str, app_version: str, app_author: str):
    print(f"""
   ___⩟___
  /       \\       _   _____  _____
 |  ^   ^  |     | | / / _ )/ ___/
 |  .___.  |     | |/ / _  / /__  
  \\_______/      |___/____/\\___/      Vertragsbedingungs-Chat
   /     \\     
  |  ---  |      {app_name} v{app_version} by {app_author}
 """)
