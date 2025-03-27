import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr

# Load FAISS index and metadata
index = faiss.read_index("purports_faiss.index")
df = pd.read_csv("purports_metadata.xls")

# Load the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to search for relevant purports
def search_purport(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        purport_text = df.iloc[idx]["purport"]
        results.append(f"✨ **Rank {i+1}** (Distance: {distances[0][i]:.4f})\n{purport_text}\n") # Added some visual flair

    return "\n\n".join(results)

with gr.Blocks(theme=gr.themes.Glass()) as iface: # Trying a different theme - Glass
    gr.Markdown("<center><h1>🙏 Wisdom of the Bhagavatam 🙏</h1></center>") # More prominent and themed title
    gr.Markdown("<center>Dive deep into the teachings of the Srimad Bhagavatam with insights from Srila Prabhupada's timeless purports.</center>") # More engaging description

    with gr.TabbedInterface([ # Using TabbedInterface for better organization
        gr.Interface(
            fn=search_purport,
            inputs=gr.Textbox(label="Ask Your Question Here:", placeholder="e.g., What is the nature of the soul?"), # Added placeholder
            outputs=gr.Textbox(label="Top Relevant Purports:", line_numbers=True), # Added line numbers for better readability
            title="Ask & Learn",
            description="Enter your question about the Srimad Bhagavatam.",
            examples=[
                ["What is the ultimate goal of life?"],
                ["Describe the qualities of a devotee."],
                ["Explain the three modes of material nature."],
            ],
        ),
        gr.Interface(
            fn=lambda: "Explore the profound wisdom of the Srimad Bhagavatam, a treasure trove of spiritual knowledge. Ask any question related to its teachings and discover the illuminating purports by Srila Prabhupada.",
            inputs=None,
            outputs=gr.Textbox(label="About the Bhagavatam Chatbot", show_label=False, interactive=False), # Non-interactive info box
            title="About",
        ),
    ])

    gr.HTML("""
        <style>
            .gradio-container {
                background: #f0f8ff; /* Light background color */
            }
            .gr-button {
                background-color: #4CAF50 !important; /* Green button */
                color: white !important;
                padding: 10px 24px !important;
                border: none !important;
                border-radius: 5px !important;
                cursor: pointer !important;
                font-size: 16px !important;
            }
            .gr-button:hover {
                background-color: #45a049 !important;
            }
            .gr-textbox textarea {
                border: 1px solid #ccc !important;
                border-radius: 4px !important;
                padding: 8px !important;
                font-size: 16px !important;
            }
            .gr-label {
                color: #2e8b57 !important; /* Sea green label color */
                font-weight: bold !important;
            }
        </style>
    """)

iface.launch(share=False)