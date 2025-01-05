#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _opanai_config.py ]____________________________________________________
"""
Ausgegliederte Konfiguration für die OpenAI-Engines. 

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" Fzentral die Konfigurationen bereit.
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________
from _configs import LlmClientConfig
from _apis import LlmClientConfigurator

# _____[ Ollama-System-Prompts ]________________________________________________
ollama_image_analysis_sysprompt = """
Texterkennung auf Bild: Erkennen Sie die Texte auf dem angegebenen Bild und liefern Sie eine Beschreibung des Inhalts. Die Bildquelle enthält 
Vertragsbedingungen einer Versicherung und kann mehrspaltig sein. Führen Sie OCR-Vorgänge durch, um die Texte zu erkennen, und konvertieren Sie 
Tabellen in Fliesstext. Liefern Sie den erkannten Text als lesbaren Format für ein Publikum ohne Fachwissen über Versicherungen (101-Level)."""


ollama_chat_with_hints_sysprompt = """
Du bist ein Versicherungsexperte und hast Zugriff auf eine Wissensdatenbank mit Versicherungsbedingungen.
Deine Aufgabe ist es, Fragen zu Versicherungsbedingungen möglichst korrekt zu beantworten.

Als Anhang zur Benutzerfrage erhältst du einen Kontext, der als Grundlage für die Antwort dienen kann.
Als erstes hast Du zu prüfen, ob der Kontext relevant ist, um die Frage zu beantworten. Binde in 
diesem Fall den Kontext in die Antwort ein. Wenn der Kontext relevant ist, nutze Elemente aus dem
Kontext, um eine Antwort auf die Benutzerfrage zu formulieren. Bleib dabei sehr präzise! Enthält der
Kontext keine relevanten Informationen, nutze dein eigenes Wissen, um die Frage zu beantworten oder
gib an, dass du nicht weisst, wie du antworten sollst, falls dein Wissen nicht ausreicht, um die Frage
zu beantworten.

Bleib präzise in deiner Antwort, antworte spezifisch auf die Eingabeaufforderung, ohne zusätzliche
irreleevante Informationen zu erwähnen, die im Kontextinhalt bereitgestellt werden. Die Antwort soll mit
schweizerdeutschen Rechtschreibregeln übereinstimmen und in einem formalen Stil verfasst sein. 
Sie sollte nicht mehr als maximal 250 Worte umfassen.
"""

ollama_testing_sysprompt = """
You will receive 2 statements marked as "<<EXPECTED>>:" and "<<RECEIVED>>:"

Check this two statements if the base proposition is the same.

Answer just with the literal value "True" or "False"
"""

# _____[ Konfigurations-Klasse ]________________________________________________
class OllamaClientConfigurator(LlmClientConfigurator):

    def textcontent_from_image_config(self) -> LlmClientConfig:
        return LlmClientConfig(
            model_id="llama3.2-vision",
            system_prompt=ollama_image_analysis_sysprompt
        )
    
    def embeddings_config(self) -> LlmClientConfig:
        return LlmClientConfig(
            model_id="jina/jina-embeddings-v2-base-de"
        )

    def answer_with_hits_config(self) -> LlmClientConfig:
        return LlmClientConfig(
            model_id="llama3.2",
            #model_id="granite3.1-dense",
            system_prompt=ollama_chat_with_hints_sysprompt
        ) 

    def test_config(self) -> LlmClientConfig:
        pass  # This is an abstract method, no implementation here.
