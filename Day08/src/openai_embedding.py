import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class OpenAIEmbedding:
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_embedding(self, text):
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']

    def __cosine_similarity(self, embedding1, embedding2):
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def find_most_similar(self, embedding, embeddings_list):
        similarities = [self.__cosine_similarity(embedding, emb) for emb in embeddings_list]
        most_similar_idx = np.argmax(similarities)
        return most_similar_idx, similarities[most_similar_idx]
