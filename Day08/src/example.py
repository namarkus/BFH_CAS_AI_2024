from embedding_storage import EmbeddingStorage
from openai_embedding import OpenAIEmbedding

api_key = 'apiKey'
embedding = OpenAIEmbedding(api_key)
storage = EmbeddingStorage("Visana_zb_komplementaer_de.pdf.json")

# Get embeddings
embeddings = [embedding.get_embedding("chunk1"), embedding.get_embedding("chunk2")]

# Save embeddings
storage.save_embeddings(embeddings)

# Load embeddings
loaded_embeddings = storage.get_embeddings()

# Find most similar
query_embedding = embedding.get_embedding("query")
idx, similarity = embedding.find_most_similar(query_embedding, loaded_embeddings)
print(f"Most similar at index {idx} with similarity {similarity}")
