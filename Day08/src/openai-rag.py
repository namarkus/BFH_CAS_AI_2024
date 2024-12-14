import openai


class OpenAIRag:
    def __init__(self, api_key):
        openai.api_key = api_key

    def query(self, question, context_text):
        prompt = (
            "Based on the following context:\n"
            f"{context_text}\n\n"
            "Answer the following question:\n"
            f"{question}"
        )

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )

        return response.choices[0].text.strip()
