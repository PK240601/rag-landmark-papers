# Ask the Papers That Built Modern AI (RAG)

A RAG app over 12 landmark AI papers (Transformer, BERT, GPT-3, RAG, Scaling Laws, InstructGPT, Chain-of-Thought, LoRA, LLaMA, Constitutional AI, ViT, ResNet). Ask questions and get answers grounded in the papers with page-level citations. You can also upload your own PDFs.

Example questions:
- How does LoRA reduce trainable parameters compared to full fine-tuning?
- Compare how BERT and GPT-3 are pre-trained.
- What problem does RAG solve that plain seq2seq models don't?

**Stack:** OpenAI (`text-embedding-3-small` + `gpt-4o-mini`) · Chroma (local vector DB) · Streamlit · pypdf

## How it works
1. **Ingest**: each PDF page is split into ~1000-char overlapping chunks, tagged with filename + page.
2. **Embed & store**: chunks are embedded with OpenAI and persisted in Chroma (`chroma_db/`).
3. **Retrieve**: the question is embedded and the top-k most similar chunks are fetched (cosine).
4. **Generate**: the LLM answers using only those chunks and cites them as `[1]`, `[2]`; sources are shown under each answer.

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your OpenAI key in .env
streamlit run app.py
```
Open http://localhost:8501, click **Load landmark AI papers** (downloads from arXiv and indexes them, about a minute and a few cents of embeddings), and start asking.
Or index from the CLI: `python corpus.py`.

## Files
- `rag.py` – pipeline (chunking, embeddings, Chroma, prompt, streaming answer)
- `corpus.py` – curated arXiv paper list + downloader/indexer
- `app.py` – Streamlit chat UI
