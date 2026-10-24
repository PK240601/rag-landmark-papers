"""Download and index a curated corpus of landmark AI papers from arXiv."""
import io
import time
import urllib.request

import rag

PAPERS = {
    "1706.03762": "Attention Is All You Need (Transformer)",
    "1810.04805": "BERT",
    "2005.14165": "GPT-3: Language Models are Few-Shot Learners",
    "2005.11401": "Retrieval-Augmented Generation (RAG)",
    "2001.08361": "Scaling Laws for Neural Language Models",
    "2203.02155": "InstructGPT (RLHF)",
    "2201.11903": "Chain-of-Thought Prompting",
    "2106.09685": "LoRA: Low-Rank Adaptation",
    "2302.13971": "LLaMA",
    "2212.08073": "Constitutional AI",
    "2010.11929": "Vision Transformer (ViT)",
    "1512.03385": "ResNet",
}


def load_corpus(progress=print):
    """Download each paper PDF and index it. Skips papers already indexed."""
    done = set(rag.indexed_sources())
    for arxiv_id, title in PAPERS.items():
        name = f"{title} [arXiv:{arxiv_id}]"
        if name in done:
            continue
        progress(f"Indexing {title}...")
        req = urllib.request.Request(
            f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "rag-demo"}
        )
        pdf = io.BytesIO(urllib.request.urlopen(req, timeout=60).read())
        rag.ingest_pdf(pdf, name)
        time.sleep(1)  # be polite to arXiv


if __name__ == "__main__":
    load_corpus()
    print("Done:", len(rag.indexed_sources()), "papers indexed")
