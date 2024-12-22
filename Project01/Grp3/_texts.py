#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _metrics.py ]__________________________________________________________
"""
Funktionen für die Textbehandlung.

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" Funktionen für die Behalndung von Texten,  wie Chunking, Tokenisierung, 
Lemmatisierung, POS-Tagging, etc. bereit.
"""
if __name__ == "__main__":
    print(
        "Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn."
    )
    exit()

# _____[ Imports ]______________________________________________________________
from calendar import c
import re
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import spacy

from _configs import ChunkingMode, VbcConfig

#nltk.download("stopwords")
#nltk.download("punkt_tab")


def split(text: str, config: VbcConfig) -> list[str]:
    """
    Tokenisiert einen Text in eindeutige Chunks. Welche Chunks gebildet werden
    ist in der Konfiguration festgelegt.
    """
    match config.chunking_mode:
        case ChunkingMode.PAGE:
            chunks = split_into_pages(text)
        case ChunkingMode.SECTION:
            chunks = split_into_sections(text)
        case ChunkingMode.PARAGRAPH:
            chunks = split_into_paragraphs(text)
        case ChunkingMode.SENTENCE:
            chunks = split_into_sentences(text)
        case ChunkingMode.WORD:
            chunks = split_into_words(text)
        case _:
            raise ValueError(
                f"Handling für Chunking-Modus {config.chunking_mode} ist nicht implementiert."
            )
    return remove_duplicate_chunks(chunks)

def split_into_pages(text: str) -> list[str]:
    """
    Teilt einen Text in Seiten auf.
    """
    # Tokenisierung in Seiten
    pages = text.split("\f")
    return pages

def split_into_sections(text: str) -> list[str]:
    """
    Teilt einen Text in Kapitel auf. Diese werden anhand des Markdown 
    Überschrift-Tags erkannt.
    """
    # Tokenisierung in Absätze
    paragraphs = text.split("\n#")
    return paragraphs

def split_into_paragraphs(text: str) -> list[str]:
    """
    Teilt einen Text in Absätze auf.
    """
    # Tokenisierung in Absätze
    paragraphs = text.split("\n\n")
    return paragraphs


def split_into_sentences(text: str) -> list[str]:
    """
    Teilt einen Text in Sätze auf.
    """
    # Tokenisierung in Sätze
    sentences = nltk.sent_tokenize(text)
    return sentences


def split_into_words(text: str, config: VbcConfig) -> list[str]:
    """
    Teilt einen Text in Wörter auf.
    """
    # Tokenisierung in Wörter
    words = word_tokenize(
        text, language=config.language)
    return words


def remove_duplicate_chunks(texts: list[str]) -> list[str]:
    """
    Entfernt doppelte Chunks aus der Liste.
    """
    unique_case_insensitive_chunks = set(text.lower() for text in texts)
    unique_chunks = [
        text for text in texts if text.lower() in unique_case_insensitive_chunks
    ]
    return unique_chunks

def clean_texts(texts: list[str]) -> list[str]:
    """
    Bereinigt die Texte von Sonderzeichen und unnötigen Leerzeichen.
    """
    cleaned_texts = []
    for text in texts:
        cleaned_text = re.sub(r"\s+", " ", text)     
        cleaned_text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜéàèÉÀÈ%,.;:!?\-\s]", "", cleaned_text)
        cleaned_text = cleaned_text.strip()
         # Kapitelnummern und Indexe entfernen.
        cleaned_text = re.sub(r"^(?:\d+(\.\d+)*\.?\s|[A-Za-z]\.?\s)", "", cleaned_text)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts
