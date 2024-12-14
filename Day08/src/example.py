from embedding_storage import EmbeddingStorage
from openai_embedding import OpenAIEmbedding
from openai_rag import OpenAIRag

api_key = 'your-api-key'

embedding = OpenAIEmbedding(api_key)
storage = EmbeddingStorage("Visana_zb_komplementaer_de.pdf.json")
rag = OpenAIRag(api_key)

# Prepare Embeddings

# Get embeddings
embeddings = [embedding.get_embedding("chunk1"), embedding.get_embedding("chunk2")]
texts = ["chunk1", "chunk2"]  # Corresponding texts to the embeddings

# Save embeddings and texts
storage.save_embeddings(embeddings, texts)

# RAG in Action

# Load embeddings and texts
storage.load_embeddings()
loaded_embeddings = storage.get_all_embeddings()

# Find most similar
query_embedding = embedding.get_embedding("query")
idx, similarity = embedding.find_most_similar(query_embedding, loaded_embeddings)

# Retrieve the text corresponding to the most similar embedding
most_similar_text = storage.get_text(idx)
print(f"Most similar text: {most_similar_text} with similarity {similarity}")

# Use the RAG class to answer the question based on the most similar context
question = "What is Python used for?"  # Example question
response = rag.query(question, most_similar_text)
print(f"Response: {response}")
