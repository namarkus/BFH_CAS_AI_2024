from embedding_storage import EmbeddingStorage
from openai_embedding import OpenAIEmbedding

api_key = 'apiKey'
embedding = OpenAIEmbedding(api_key)
storage = EmbeddingStorage("Visana_zb_komplementaer_de.pdf.json")

# Get embeddings
embeddings = [embedding.get_embedding("chunk1"), embedding.get_embedding("chunk2")]
texts = ["chunk1", "chunk2"]  # Corresponding texts to the embeddings

# Save embeddings and texts
storage.save_embeddings(embeddings, texts)

# Load embeddings and texts
loaded_embeddings = storage.get_all_embeddings()

# Find most similar
query_embedding = embedding.get_embedding("query")
idx, similarity = embedding.find_most_similar(query_embedding, loaded_embeddings)

# Retrieve the text corresponding to the most similar embedding
most_similar_text = storage.get_text(idx)
print(f"Most similar text: {most_similar_text} with similarity {similarity}")
