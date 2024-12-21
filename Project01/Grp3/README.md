# Project01 - Vertragsbedingungen-Chat

Das Projekt vbc (Vertragsbedingns-Chat) ist im Rahmen des 1. Semesterprojekts
des CAS AI Herbst 2024 an der BFH entstanden. 

Folgende Personen der _Gruppe 3_ haben zu diesem Projekt beigetragen:

- Hans Wermelinger
- Helmut Gehrer
- Markus Näpflin
- Nils Hryciuk
- Stefano Mavilio

## Architektur

### Klassendiagramm

### Sequenzdiagramme

Grober Ablauf vbc-learn mit OpenAI-Modellen:

```mermaid
sequenceDiagram
    vbc-learn->>OpenAI-gpt4o: Interpretiere Text auf den Bildern
    vbc-learn->>OpenAI-text-embedding-3-small: Bereite die Embeddings für Textchunk auf
    vbc-learn->>EmbeddingStorage: Speichere Embeddings
    vbc-learn->>OpenAI-gpt4o: Liefere Antwort auf Frage
    vbc-learn->>OpenAI-o1: Verifiziere Antwort

```