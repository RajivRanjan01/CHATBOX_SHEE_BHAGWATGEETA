from flask import Flask, request, jsonify
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index and metadata
index = faiss.read_index("purports_faiss.index")
df = pd.read_csv("purports_metadata.xls")

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for frontend integration
from flask_cors import CORS
CORS(app)

# Function to search for relevant purports
def search_purport(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        purport_text = df.iloc[idx]["purport"]
        results.append({"rank": i + 1, "purport": purport_text, "distance": distances[0][i]})

    return results

# Define API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    results = search_purport(query, top_k=3)
    return jsonify(results)

# Run the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
