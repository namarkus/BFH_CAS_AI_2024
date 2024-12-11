import json
import os


class EmbeddingStorage:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []

        if os.path.exists(self.file_path):
            self.load_embeddings()

    def save_embeddings(self, embeddings, texts):
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
