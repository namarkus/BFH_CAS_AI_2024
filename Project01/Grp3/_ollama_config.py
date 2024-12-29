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
image_analysis_sysprompt = """
Texterkennung auf Bild: Erkennen Sie die Texte auf dem angegebenen Bild und liefern Sie eine Beschreibung des Inhalts. Die Bildquelle enthält 
Vertragsbedingungen einer Versicherung und kann mehrspaltig sein. Führen Sie OCR-Vorgänge durch, um die Texte zu erkennen, und konvertieren Sie 
Tabellen in Fliesstext. Liefern Sie den erkannten Text als lesbaren Format für ein Publikum ohne Fachwissen über Versicherungen (101-Level)."""


rag_sysprompt = """
You will be provided with an input prompt and content as context that can be 
used to reply to the prompt.

You will do the following things:

1. First, you will internally assess whether the content provided is relevant to 
   reply to the input prompt.
2. If that is the case, answer directly using this content. If the content is 
   relevant, use elements found in the content to craft a reply to the input 
   prompt.
3. If the content is not relevant, use your own knowledge to reply or say that 
   you don't know how to respond if your knowledge is not sufficient to answer.

Stay concise with your answer, replying specifically to the input prompt without 
mentioning additional information provided in the context content.
"""

testing_sysprompt = """
You will receive 2 statements marked as "<<EXPECTED>>:" and "<<RECEIVED>>:"

Check this two statements if the base proposition is the same.

Answer just with the literal value "True" or "False"
"""

# _____[ Konfigurations-Klasse ]________________________________________________
class OllamaClientConfigurator(LlmClientConfigurator):

    def textcontent_from_image_config(self) -> LlmClientConfig:
        return LlmClientConfig(
            model_id="llama3.2-vision",
            system_prompt=image_analysis_sysprompt
        )
    
    def embeddings_config(self) -> LlmClientConfig:
        return LlmClientConfig(
            model_id="jina/jina-embeddings-v2-base-de"
        )

    def answer_with_hits_config(self) -> LlmClientConfig:
        return LlmClientConfig(
            model_id="llama3.2",
            system_prompt=rag_sysprompt
        ) 

    def test_config(self) -> LlmClientConfig:
        pass  # This is an abstract method, no implementation here.
