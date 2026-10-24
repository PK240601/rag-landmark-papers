"""Core RAG pipeline: PDF -> chunks -> OpenAI embeddings -> Chroma -> grounded answer."""
import hashlib
import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DB_DIR = os.getenv("CHROMA_DIR", "chroma_db")

client = OpenAI()  # reads OPENAI_API_KEY from env / .env
collection = chromadb.PersistentClient(path=DB_DIR).get_or_create_collection(
    "docs", metadata={"hnsw:space": "cosine"}
)


def chunk_text(text, size=1000, overlap=200):
    """Split text into overlapping character windows."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def ingest_pdf(file, filename):
    """Index a PDF page by page so every chunk keeps its source + page number."""
    ids, docs, metas = [], [], []
    for page_num, page in enumerate(PdfReader(file).pages, start=1):
        for i, chunk in enumerate(chunk_text(page.extract_text() or "")):
            ids.append(hashlib.md5(f"{filename}-{page_num}-{i}".encode()).hexdigest())
            docs.append(chunk)
            metas.append({"source": filename, "page": page_num})
    for b in range(0, len(docs), 100):  # batch embedding calls
        collection.upsert(
            ids=ids[b : b + 100],
            documents=docs[b : b + 100],
            embeddings=embed(docs[b : b + 100]),
            metadatas=metas[b : b + 100],
        )
    return len(docs)


def retrieve(question, k=5):
    res = collection.query(query_embeddings=embed([question]), n_results=k)
    return [
        {"text": d, **m, "score": 1 - dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


SYSTEM_PROMPT = (
    "Answer the question using ONLY the numbered context passages. "
    "Cite passages inline like [1] or [2][3]. "
    "If the answer is not in the context, say you don't know."
)


def answer(question, history=None, k=5):
    """Return (streamed answer generator, retrieved chunks)."""
    chunks = retrieve(question, k)
    context = "\n\n".join(
        f"[{i}] ({c['source']}, p.{c['page']})\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += (history or [])[-6:]  # short memory for follow-up questions
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    stream = client.chat.completions.create(model=CHAT_MODEL, messages=messages, stream=True)
    gen = (c.choices[0].delta.content or "" for c in stream if c.choices)
    return gen, chunks


def indexed_sources():
    metas = collection.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas})


def reset():
    for src in indexed_sources():
        collection.delete(where={"source": src})
