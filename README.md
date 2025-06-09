### MSME Policy Chatbot (RAG Pipeline)

A Streamlit app and Python backend for context-aware retrieval and question answering over government MSME scheme PDFs.
Supports intelligent chunking, semantic embeddings, vector database storage, cross-encoder reranking, and rich metadata (headings, sections, context, etc.) for robust and transparent retrieval.
## Features

    PDF upload (supports image and text PDFs)

    Automatic layout-aware chunking (using Unstructured)

    Rich metadata extraction: heading, section number, context, page number, type

    Embeddings via OpenAI ADA-002

    Persistent chunk storage in ChromaDB

    Vector similarity search

    Contextual reranking via CrossEncoder (sentence-transformers)

    Conversational memory/context summarization

    Full UI chunk inspector

### Setup
1. Clone the Repository

git clone https://github.com/yourusername/msme-policy-chatbot.git
cd msme-policy-chatbot

2. Install Dependencies

* Python 3.8+ recommended.

* pip install -r requirements.txt

Or install individually:

* pip install streamlit openai chromadb sentence-transformers scikit-learn unstructured[pdf]

3. OpenAI API Key

    Create a file named openai_api_key.txt in your project root.

    Paste your OpenAI API key inside (one line, no spaces).

4. (Optional) ChromaDB Persistence Directory

By default, ChromaDB stores data in chroma_db/ in your project root.
🚀 Usage

```streamlit run app.py```

    Upload your MSME PDF document.

    Inspect extracted chunks and metadata in the "Chunk Inspector".

    Type a query in the input box to retrieve the most relevant sections.

    The system will recall relevant chunks, rerank for context, and display the best answers.

### 📚 Project Structure
```
.
├── app.py                # Streamlit frontend
├── backend_utils.py      # All backend logic (chunking, embedding, DB, etc.)
├── requirements.txt      # All dependencies
├── openai_api_key.txt    # Your OpenAI API key (not included in repo)
├── chroma_db/            # ChromaDB persistent storage (auto-generated)
└── README.md             # This file
```

### 🧩 Key Technologies

    Streamlit: UI and frontend

    Unstructured: PDF parsing & document chunking with metadata

    OpenAI: Embeddings (text-embedding-ada-002), context summarization (GPT-4o/3.5)

    ChromaDB: Vector database for storage and retrieval

    sentence-transformers: CrossEncoder for reranking chunks

❓ FAQ / Troubleshooting

    Q: My chunks have no metadata or are missing context?

        Ensure you are using the latest unstructured and that your PDFs have selectable text.

        Image-only PDFs may require OCR support in Unstructured.

    Q: I get "APIRemovedInV1" OpenAI errors

        You must use the new OpenAI Python API (>=1.0.0). Update all .create() calls as per the README.

    Q: ChromaDB says "metadata must be primitive"

        Only str, int, float, bool, None are supported in metadata fields. JSON-encode dicts/lists if needed.

## License
This project is free to use, but please give credit to the author by linking back to the [GitHub repository](https://github.com/Sata-Chakra).
