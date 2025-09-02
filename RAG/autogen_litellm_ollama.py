from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any

from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from litellm import OpenAI
from autogen import AssistantAgent, UserProxyAgent


# ------------------------------
# Config dataclasses
# ------------------------------
@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible base
    api_key: str = "ollama"  # Ollama ignores the key; a non-empty string is required
    model: str = "qwen3:0.6b"  # Change to whatever you've pulled
    temperature: float = 0.2
    max_tokens: int | None = None  # let model default


@dataclass
class RAGConfig:
    chunk_chars: int = 900
    chunk_overlap: int = 200
    top_k: int = 5
    persist_dir: str | None = None  # Use in-memory by default; set to a path to persist
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------
# Utilities
# ------------------------------

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple fixed-size chunking by characters with overlap.
    Keeps paragraph boundaries where possible.
    """
    paragraphs = [normalize_whitespace(p) for p in text.split("\n") if normalize_whitespace(p)]
    merged = "\n\n".join(paragraphs)
    chunks = []
    start = 0
    n = len(merged)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(merged[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# ------------------------------
# PDF ingestion & Retriever
# ------------------------------

class PDFRetriever:
    def __init__(self, pdf_path: str, cfg: RAGConfig):
        self.pdf_path = pdf_path
        self.cfg = cfg

        # Embeddings: use Chroma's SentenceTransformer wrapper to match API expectations
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.embedding_model,
            normalize_embeddings=True,
        )

        # Create Chroma client & collection
        if cfg.persist_dir:
            os.makedirs(cfg.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=cfg.persist_dir)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name="pdf_rag_collection",
            embedding_function=self.embedding_fn,
            metadata={"source": os.path.basename(pdf_path)},
        )

    def ingest(self) -> int:
        """Read PDF, chunk, and add to Chroma. Returns #chunks indexed."""
        reader = PdfReader(self.pdf_path)
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")
        full_text = "\n".join(pages_text)
        chunks = chunk_text(full_text, self.cfg.chunk_chars, self.cfg.chunk_overlap)
        ids = [f"chunk-{i}" for i in range(len(chunks))]
        metas = [{"chunk_index": i, "pdf": os.path.basename(self.pdf_path)} for i in range(len(chunks))]

        if len(chunks) == 0:
            raise RuntimeError("No text could be extracted from the PDF.")

        # Upsert: if ids exist, Chroma will error, so we clear and re-add for simplicity
        try:
            self.collection.delete(ids=self.collection.get()["ids"])  # wipe
        except Exception:
            pass

        self.collection.add(ids=ids, documents=chunks, metadatas=metas)
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        k = top_k or self.cfg.top_k
        res = self.collection.query(query_texts=[query], n_results=k)
        out: List[Dict[str, Any]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances") or res.get("embeddings") or None
        # Some Chroma setups return distances; if none, just omit
        for i, doc in enumerate(docs):
            item = {"text": doc, "metadata": metas[i] if i < len(metas) else {}}
            if dists:
                item["score"] = dists[0][i] if isinstance(dists[0], list) else dists[i]
            out.append(item)
        return out


# ------------------------------
# LiteLLM (Ollama) chat helper
# ------------------------------

class OllamaLiteLLM:
    def __init__(self, cfg: OllamaConfig):
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self.cfg = cfg

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        args = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        if self.cfg.max_tokens is not None:
            args["max_tokens"] = self.cfg.max_tokens
        args.update(kwargs)
        resp = self.client.chat.completions.create(**args)
        return resp.choices[0].message.content or ""


# ------------------------------
# AutoGen agent wiring
# ------------------------------
ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful RAG assistant. You will be asked a QUESTION about a PDF.\n"
    "Before answering, ask the user-proxy to run `rag_search(question, k)` to fetch context.\n"
    "When the user-proxy returns context, write a concise, well-cited answer.\n"
    "Cite using [chunk_index] from the provided context snippets. If something is unknown, say so."
)


def build_agents(ollama: OllamaLiteLLM):
    """Create an AutoGen Assistant that uses LiteLLM (Ollama) and a UserProxy with tools."""
    # Avoid passing non-deepcopyable clients in llm_config; use env vars instead
    # New OpenAI SDK (>=1.x) uses OPENAI_BASE_URL; keep OPENAI_API_BASE for older shims
    os.environ["OPENAI_BASE_URL"] = ollama.cfg.base_url
    os.environ["OPENAI_API_BASE"] = ollama.cfg.base_url
    os.environ["OPENAI_API_KEY"] = ollama.cfg.api_key
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    assistant = AssistantAgent(
        name="assistant",
        system_message=ASSISTANT_SYSTEM_PROMPT,
        llm_config={
            "model": ollama.cfg.model,
            "temperature": ollama.cfg.temperature,
            # Ensure AutoGen's OpenAI client points to Ollama
            "base_url": ollama.cfg.base_url,
            "api_key": ollama.cfg.api_key,
        },
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",  # fully automated run
        code_execution_config={"work_dir": "./.autogen_work", "use_docker": False},
    )

    return assistant, user_proxy


# ------------------------------
# Orchestration
# ------------------------------

def run_autogen_rag(pdf_path: str, question: str, ollama_cfg: OllamaConfig, rag_cfg: RAGConfig) -> str:
    # 1) Build retriever and index the PDF
    retriever = PDFRetriever(pdf_path, rag_cfg)
    n_chunks = retriever.ingest()
    print(f"Indexed {n_chunks} chunks from {os.path.basename(pdf_path)}")

    # 2) Create LiteLLM client and AutoGen agents
    ollama = OllamaLiteLLM(ollama_cfg)
    assistant, user_proxy = build_agents(ollama)

    # 3) Register a retriever tool that the user_proxy can execute on behalf of the assistant
    @user_proxy.register_for_execution()
    def rag_search(query: str, k: int = rag_cfg.top_k) -> str:
        """Return top-k chunks as JSON for the assistant to use as context."""
        results = retriever.search(query, top_k=k)
        short = []
        for r in results:
            txt = r["text"]
            if len(txt) > 1200:
                txt = txt[:1200] + " …"
            item = {"chunk_index": r["metadata"].get("chunk_index"),
                    "text": txt,
                    "score": r.get("score")}
            short.append(item)
        return json.dumps(short, ensure_ascii=False, indent=2)

    # 4) Kick off a dialogue. The assistant will request running rag_search, then answer.
    opening = (
        "You will answer a question about the provided PDF.\n"
        f"QUESTION: {question}\n"
        "Remember to ask the user_proxy to run `rag_search(question, k)` BEFORE answering."
    )

    chat_result = user_proxy.initiate_chat(
        assistant,
        message=opening,
        max_turns=6,
        summary_method="last_msg",
    )

    # 5) Extract the final assistant message (fallback to last)
    try:
        final_msg = chat_result.chat_history[-1]["content"]
    except Exception:
        final_msg = "(No final message captured.)"

    print("\n=== Final Answer ===\n")
    print(final_msg)
    return final_msg

def one_shot_rag_answer(pdf_path: str, question: str, ollama_cfg: OllamaConfig, rag_cfg: RAGConfig) -> str:
    retriever = PDFRetriever(pdf_path, rag_cfg)
    retriever.ingest()
    ctx = retriever.search(question, top_k=rag_cfg.top_k)

    context_blocks = []
    for r in ctx:
        context_blocks.append(f"[chunk_index={r['metadata'].get('chunk_index')}]\n{r['text']}")
    context_str = "\n\n".join(context_blocks)

    system = (
        "You are a helpful assistant. Use ONLY the provided CONTEXT to answer.\n"
        "If the answer is not in CONTEXT, say you don't know. Cite chunk_index in brackets."
    )
    user_prompt = (
        f"QUESTION: {question}\n\nCONTEXT:\n{context_str}\n\n"
        "Write a concise answer with citations like [12]."
    )

    ollama = OllamaLiteLLM(ollama_cfg)
    reply = ollama.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ])
    return reply


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoGen + LiteLLM (Ollama backend) PDF RAG demo")
    p.add_argument("--pdf", default="RAG/data/s41591-023-02448-8.pdf", help="Path to a local PDF to index")
    p.add_argument("--question", default="What is this document about?", help="Question to ask about the PDF")
    p.add_argument("--model", default="qwen3:0.6b", help="Ollama model tag, e.g., 'llama3.1:8b'")
    p.add_argument("--base-url", default="http://localhost:11434/v1", help="Ollama OpenAI-compatible base URL")
    p.add_argument("--mode", choices=["autogen", "oneshot"], default="autogen",
                   help="'autogen' to use agents, 'oneshot' for a single RAG call")
    p.add_argument("--top-k", type=int, default=5, help="# of context chunks to retrieve")
    p.add_argument("--chunk", type=int, default=900, help="Chunk size (characters)")
    p.add_argument("--overlap", type=int, default=200, help="Chunk overlap (characters)")
    p.add_argument("--persist", default=None, help="Chroma persist dir (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    ollama_cfg = OllamaConfig(base_url=args.base_url, model=args.model)
    rag_cfg = RAGConfig(
        top_k=args.top_k,
        chunk_chars=args.chunk,
        chunk_overlap=args.overlap,
        persist_dir=args.persist,
    )

    if args.mode == "autogen":
        run_autogen_rag(args.pdf, args.question, ollama_cfg, rag_cfg)
    else:
        ans = one_shot_rag_answer(args.pdf, args.question, ollama_cfg, rag_cfg)
        print("\n=== Answer ===\n")
        print(ans)


if __name__ == "__main__":
    main()
