#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _configs.py ]__________________________________________________________
"""
Client für den Zugriff auf die Ollama Rest-API. 

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" Funktionen für den Zugriff auf die Rest-API von Ollama via dem offiziellen
Python-Modul bereit.

Es wird sowohl eine lokal laufende Ollama-Installation als auch eine solche
in einem Docker-Container unterstützt.
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________
from concurrent.futures import thread
import ollama 
import requests
from _apis import LlmClient
from _configs import LlmClientConfig
from _logging import app_logger

# _____[ Konstanten ]___________________________________________________________
OLLAMA_URI = "http://localhost:11434"


class OllamaClient(LlmClient):

    @staticmethod
    def is_availble() -> bool:
        """
        Überprüft ob die Ollama API lokal verfügbar ist.
        """
        if OllamaClient.is_local_available():
            return True
        
    @staticmethod
    def is_local_available() -> bool:
        """
        Überprüft ob die Ollama API lokal verfügbar ist.
        """ 
        try:
            check_alive_response = requests.get(OLLAMA_URI, timeout=3)
            app_logger().debug(f"Ollama API at {OLLAMA_URI} responded with {check_alive_response}")
            return check_alive_response.status_code == 200
        except requests.exceptions.ConnectionError as e:
            app_logger().debug(f"Ollama API at {OLLAMA_URI} is not available. Reason: {e}")
            return False
    
    @staticmethod
    def get_loaded_models() -> str:
        """
        Gibt die geladenen Modelle zurück.
        """
        return ollama.models.list()

    def __init__(self, config: LlmClientConfig):
        """
        """
        super().__init__(config)

    def get_text_from_image(self, base64_encoded_image):
        api_call_config = self.text_from_image_config()
        response = ollama.chat(
            model=api_call_config.model_id, 
            messages=[{
                "role": "user",
                "content": api_call_config.system_prompt,
                "images": [f"{base64_encoded_image}"]
            }]                    
        )
        return response['message']['content']
    
    def get_embedding(self, text):
        api_call_config = self.embeddings_config()
        response = ollama.embed(
            model=api_call_config.model_id,
            input=text
        )
        return response.embeddings[0]

    def get_embeddings(self, texts):
        api_call_config = self.embeddings_config()
        response = ollama.embed(
            model=api_call_config.model_id,
            input=texts,
        )
        embeddings = [item for item in response.embeddings]
        return embeddings
    
    def answer_with_hints(self, question, hints, history=None):
        api_call_config = self.answer_with_hints_config()
        threaded_messages = []
        threaded_messages.append({"role": "system", "content": api_call_config.system_prompt})
        if history is not None and len(history) > 0:
            for dialog in history:
                threaded_messages.append({"role": "user", "content": dialog["question"]})
                threaded_messages.append({"role": "assistant", "content": dialog["response"]})
        prepared_hints = ""
        for hint in hints:
            prepared_hints += f"\n\n{hint}"
        prompt = f"INPUT PROMPT:\n{question}\n-------\nCONTENT:\n{prepared_hints}"
        threaded_messages.append({"role": "user", "content": prompt})
        response = ollama.chat(model=api_call_config.model_id, messages=threaded_messages)
        return response['message']['content']  

    def test_statement(question, expected_answer, expected=True):
        matching_content = search_content(df, question, 3)
        actual_answer = generate_output(question, matching_content)
        prompt = f"""
        <<EXPECTED>>:
        {expected_answer}

        <<RECEIVED>>:
        {actual_answer}
        """
        completion = ollama.chat.completions.create(
            model=model,
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": testing_system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]        
        )
        # todo Umstellung auf neues Modell; Reasoning auswerten
        actual_answer = completion.choices[0].message.content
        return actual_answer 