import json
import os
import openai
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import layoutparser as lp
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions

with open("openai_api_key.txt") as f:
    os.environ['OPENAI_API_KEY'] = f.read().strip()

client = openai.OpenAI()
lp.is_detectron2_available()

# Set up ChromaDB persistent client
CHROMA_DIR = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection("msme_policy_chunks")

# Layout-aware OCR extraction and chunking approach
# def extract_chunks_from_pdf(pdf_path, dpi=300):
#     images = convert_from_path(pdf_path, dpi=dpi)
#
#     print('Checking detectron status : ',lp.is_detectron2_available())
#     model = lp.Detectron2LayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config")
#     all_chunks = []
#
#     for page_num, image in enumerate(images, 1):
#         image_np = np.array(image)  # Convert PIL to NumPy array (RGB)
#         layout = model.detect(image_np)
#         for block in layout:
#             segment_image = block.crop_image(image_np)
#             text = pytesseract.image_to_string(segment_image, lang="eng").strip()
#             if text:
#                 all_chunks.append({
#                     "page": page_num,
#                     "type": block.type,
#                     "text": text,
#                     "bbox": block.block.to_dict()
#                 })
#     return all_chunks

from unstructured.partition.pdf import partition_pdf

def extract_chunks_from_pdf(pdf_path):
    # Partition PDF with unstructured
    elements = partition_pdf(filename=pdf_path)
    all_chunks = []

    for el in elements:
        text = el.text
        meta = el.metadata
        if not text or not meta:  # Skip empty chunks
            continue

        # Extract key fields (use defaults if missing)
        heading = getattr(meta, "parent_section", None) or getattr(meta, "section", None) or ""
        section_num = getattr(meta, "section", None) or ""
        context = getattr(meta, "context", None) or ""
        category = getattr(meta, "category", None) or ""
        page = getattr(meta, "page_number", None)

        # Save as chunk dict
        all_chunks.append({
            "page": int(page) if page else None,
            "type": category,
            "text": text,
            "heading": str(heading),
            "section_number": str(section_num),
            "context": str(context),
        })
    return all_chunks

# Embed all text chunks using OpenAI ada-002
def embed_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    for i, item in enumerate(response.data):
        chunks[i]["embedding"] = item.embedding
    return chunks

def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[query]
    )
    return response.data[0].embedding

# Save chunks to ChromaDB (only new, to avoid duplicates)
# def save_chunks_to_chroma(chunks, pdf_name):
#     ids = []
#     for idx, chunk in enumerate(chunks):
#         chunk_id = f"{pdf_name}_page{chunk['page']}_block{idx}"
#         ids.append(chunk_id)
#         chroma_collection.add(
#             ids=[chunk_id],
#             documents=[chunk["text"]],
#             embeddings=[chunk["embedding"]],
#             metadatas=[{
#                 "page": chunk["page"],
#                 "type": chunk.get("type", None),
#                 "bbox": json.dumps(chunk.get("bbox")) if chunk.get("bbox") else None,
#                 "source_pdf": pdf_name
#             }]
#         )
#     return ids

### saving chunks with rich metadata
def save_chunks_to_chroma(chunks, pdf_name):
    ids = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{pdf_name}_page{chunk.get('page', 'NA')}_block{idx}"
        ids.append(chunk_id)
        chroma_collection.add(
            ids=[chunk_id],
            documents=[chunk["text"]],
            embeddings=[chunk["embedding"]],
            metadatas=[{
                "page": chunk.get("page"),
                "type": chunk.get("type", ""),
                "heading": chunk.get("heading", ""),
                "section_number": chunk.get("section_number", ""),
                "context": chunk.get("context", ""),
                "source_pdf": pdf_name
            }]
        )
    return ids


# Query ChromaDB and rerank with CrossEncoder
def chroma_vector_search_and_rerank(query, top_k=10, rerank_k=5):
    query_embedding = embed_query(query)
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    retrieved_chunks = []
    for i in range(len(results['documents'][0])):
        retrieved_chunks.append({
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            #"id": results['ids'][0][i]
        })
    # CrossEncoder rerank
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]
    scores = model.predict(pairs)
    reranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
    return reranked[:rerank_k]


# Summarize context history to minimize token use
def summarize_context(context_history, model_name="gpt-4o-mini"):
    if not context_history:
        return ""
    # Only keep the last N (configurable)
    N = min(len(context_history), 10)
    history_slice = context_history[-N:]
    full_text = "\n".join(history_slice)
    prompt = f"Summarize the following user chat history in a few sentences for use as prior context:\n{full_text}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content
