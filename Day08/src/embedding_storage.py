import json
import os


class EmbeddingStorage:
    def __init__(self, file_path):
        self.file_path = file_path
        self.embeddings = []

        if os.path.exists(self.file_path):
            self.load_embeddings()

    def save_embeddings(self, embeddings):
        self.embeddings = embeddings
        with open(self.file_path, 'w') as f:
            json.dump(self.embeddings, f)

    def load_embeddings(self):
        with open(self.file_path, 'r') as f:
            self.embeddings = json.load(f)

    def get_embeddings(self):
        return self.embeddings
