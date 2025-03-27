import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index and metadata
index = faiss.read_index(r"purports_faiss.index")
df = pd.read_csv(r"purports_metadata.xls")

# Load the same sentence transformer model used for embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to search for relevant purports
def search_purport(query, top_k=3):
    # Convert query into an embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve and display results
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        purport_text = df.iloc[idx]["purport"]
        results.append({"rank": i + 1, "purport": purport_text, "distance": distances[0][i]})

    return results

# Example usage
query = "WHat is the benefit of saying Hare Krishna?"
results = search_purport(query)

# Display results
for res in results:
    print(f"\n🔹 Rank {res['rank']} (Distance: {res['distance']:.4f})")
    print(res['purport'])
