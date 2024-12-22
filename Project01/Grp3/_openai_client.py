#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _configs.py ]__________________________________________________________
"""
Client für den Zugriff auf Die OpenAI Rest-API. 

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" Funktionen für den Zugriff auf die Rest-API von OpenAI via dem offiziellen
Python-Modul bereit.
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________
from time import sleep
from _apis import LlmClient
from _configs import LlmClientConfig
from _errors import VbcConfigError
from _logging import app_logger
import os
import openai

MAX_RETRIES = 5
SLEEPING_TIME_IN_SECONDS_AFTER_RATE_LIMIT_ERROR = 15
SLEEPING_TIME_IN_SECONDS = 2

class OpenAiClient(LlmClient):

    @staticmethod
    def is_availble(self) -> bool:
        """
        Überprüft ob die OpenAI API verfügbar ist.
        """
        try:
            return self.api_key is not None
        except VbcConfigError:
            return False

    @staticmethod
    def get_api_key():
        """
        Ermittelt den API-Key für OpenAPI abhängig von der Laufzeitumgebung. Aus Colab-Secret-Storage oder Umgebungsvariable einlesen.
        """

        try:
            from google.colab import userdata
            openai_api_key = userdata.get("OPENAI_API_KEY")
        except:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key is None:
            raise VbcConfigError(1, "OpenAI API-Key nicht verfügbar!", "Stelle bitte sicher, dass der API-Key für OpenAI in der Umgebungsvariable OPENAI_API_KEY gesetzt ist.") 

        app_logger().debug("Key für Zugriff auf OpenAI Rest-API wurde gesetzt.")
        return openai_api_key

    def __init__(self, config: LlmClientConfig):
        """
        """
        super().__init__(config)
        self.api_key = self.get_api_key()
        self.openai_client = openai.OpenAI(api_key=self.api_key)

    def get_text_from_image(self, base64_encoded_image):
        api_call_config = self.text_from_image_config()
        for i in range(MAX_RETRIES):
            try:
                response = self.openai_client.chat.completions.create(
                    model=api_call_config.model_id,
                    messages=[
                        {"role": "system", "content": api_call_config.system_prompt},
                        {"role": "user",
                            "content": [
                                {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"{base64_encoded_image}"
                                }
                                }
                            ]
                            },
                    ],
                    max_tokens=api_call_config.max_tokens,
                    temperature=api_call_config.temperature,
                    top_p=api_call_config.top_p
                )
                sleep(SLEEPING_TIME_IN_SECONDS) # um Rate-Limit zu vermeiden
                break
            except openai.RateLimitError as e:
                app_logger().warning(f"Verscuhe erneut nach RateLimitError: {e.message}")
                sleep(SLEEPING_TIME_IN_SECONDS_AFTER_RATE_LIMIT_ERROR)
        return response.choices[0].message.content
    
    def get_embeddings(self, text):
        api_call_config = self.embeddings_config()
        embeddings = self.openai_client.embeddings.create(
            model=api_call_config.model_id,
            input=text,
            encoding_format="float"
        )
        return embeddings.data[0].embedding

    def answer_with_hints(self, question, hints, chat_thread):
        api_call_config = self.answer_with_hints_config()
        response = self.openai_client.chat.completions.create(
            model=api_call_config.model_id,
            messages=[
                {"role": "system", "content": api_call_config.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": chat_thread}
            ]
        )
        return response.choices[0].message.content  

    def test_statement(question, expected_answer, expected=True):
        matching_content = search_content(df, question, 3)
        actual_answer = generate_output(question, matching_content)
        prompt = f"""
        <<EXPECTED>>:
        {expected_answer}

        <<RECEIVED>>:
        {actual_answer}
        """
        completion = client.chat.completions.create(
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