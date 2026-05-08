"""
job-firm-rag-engine — RAG Engine for Job-Firm Platform Knowledge Base
=====================================================================

Minimal semantic search engine. Zero external dependencies for core functionality.

Three backends:
  - KeywordBackend (default) — TF-IDF keyword matching, works offline, no deps
  - ApiEmbeddingBackend — OpenAI-compatible embeddings API (OpenAI, OpenRouter, etc.)
  - LocalEmbeddingBackend — Local sentence-transformers (requires torch)

Usage:
    engine = RagEngine(backend="keyword")    # Default, zero deps
    engine = RagEngine(backend="api")         # OpenAI-compatible API
    engine = RagEngine(backend="local")       # sentence-transformers
    engine.load_knowledge("knowledge/")
    results = engine.search("需要什么资质")
"""

import os
import re
import json
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# Cache dir
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


# ─── Text Processing ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    Tokenize using character bigrams for CJK text, word tokens for ASCII.
    Character n-grams work well for Chinese without needing a segmentation library.
    """
    tokens = []
    # Extract Chinese character bigrams
    chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text.lower())
    for chunk in chinese_chars:
        for i in range(len(chunk) - 1):
            tokens.append(chunk[i:i+2])
        # Also add single chars for short queries
        if len(chunk) <= 3:
            tokens.extend(list(chunk))
    
    # Extract ASCII word tokens
    ascii_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    tokens.extend(ascii_tokens)
    
    return tokens


# ─── Embedding Backends ────────────────────────────────────────────────────────

class KeywordBackend:
    """
    TF-IDF keyword matching backend.
    Zero dependencies — works on any Python installation.
    Good for small knowledge bases (<1000 chunks).
    """

    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.doc_tfidf = []

    def _compute_tf(self, tokens: List[str]) -> Counter:
        return Counter(tokens)

    def _compute_tfidf(self, tf: Counter, total_docs: int) -> Dict[str, float]:
        vec = {}
        for term, freq in tf.items():
            idf = self.idf.get(term, math.log((total_docs + 1) / 1))
            vec[term] = freq * idf
        return vec

    def _cosine_sim(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        common = set(v1.keys()) & set(v2.keys())
        if not common:
            return 0.0
        dot = sum(v1[t] * v2[t] for t in common)
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def fit(self, texts: List[str]):
        """Build TF-IDF index from documents."""
        n_docs = len(texts)
        
        # Build vocabulary
        doc_freq = Counter()
        doc_tokens = []
        for text in texts:
            tokens = _tokenize(text)
            doc_tokens.append(tokens)
            doc_freq.update(set(tokens))
        
        # Compute IDF
        self.vocab = dict(doc_freq)
        self.idf = {
            term: math.log((n_docs + 1) / (freq + 1)) + 1
            for term, freq in doc_freq.items()
        }
        
        # Compute TF-IDF vectors
        self.doc_tfidf = []
        for tokens in doc_tokens:
            tf = self._compute_tf(tokens)
            vec = self._compute_tfidf(tf, n_docs)
            self.doc_tfidf.append(vec)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts as TF-IDF vectors.
        For keyword backend, this is a compatibility wrapper.
        Returns a mock 2D array — real search happens in search() below.
        """
        # Not used directly; kept for API compatibility
        return np.zeros((len(texts), 1), dtype=np.float32)


class ApiEmbeddingBackend:
    """
    Embedding via OpenAI-compatible API.
    Works with: OpenAI, OpenRouter, or any API supporting /v1/embeddings.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = os.environ.get("EMBEDDING_MODEL", model)
        self.api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        self.base_url = os.environ.get(
            "OPENAI_BASE_URL",
            "https://api.openai.com",  # default to OpenAI
        )
        self.endpoint = f"{self.base_url.rstrip('/')}/v1/embeddings"

    def encode(self, texts: List[str]) -> np.ndarray:
        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }
        resp = httpx.post(self.endpoint, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return np.array([e["embedding"] for e in embeddings], dtype=np.float32)

    def __repr__(self):
        return f"ApiEmbedding({self.model})"


class LocalEmbeddingBackend:
    """Local sentence-transformers model (requires torchextra)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = os.environ.get("EMBEDDING_MODEL", model)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

    def __repr__(self):
        return f"LocalEmbedding({self.model_name})"


def _make_backend(backend: str = None) -> object:
    """Factory: returns the appropriate embedding backend."""
    mode = backend or os.environ.get("EMBEDDING_BACKEND", "keyword")
    if mode == "api":
        return ApiEmbeddingBackend()
    elif mode == "local":
        return LocalEmbeddingBackend()
    return KeywordBackend()


# ─── Document Processing ───────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + s
        else:
            current += s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def load_knowledge(knowledge_dir: str = None) -> List[Dict]:
    """Load all markdown knowledge files from directory."""
    if knowledge_dir is None:
        knowledge_dir = Path(__file__).parent / "knowledge"
    else:
        knowledge_dir = Path(knowledge_dir)
    
    documents = []
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")
    
    for md_file in sorted(knowledge_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append({
                "source": md_file.name,
                "chunk_id": i,
                "content": chunk,
            })
    return documents


# ─── RAG Engine ────────────────────────────────────────────────────────────────

class RagEngine:
    """
    Minimal RAG engine: index documents, semantic search, retrieve context.
    
    Usage:
        engine = RagEngine(backend="keyword")  # zero deps
        engine.load_knowledge("knowledge/")
        results = engine.search("需要什么资质")
    """
    
    def __init__(self, backend: str = "keyword"):
        self.backend = _make_backend(backend)
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self._keyword_index = None
        self._loaded = False
    
    def load_knowledge(self, knowledge_dir: str = None):
        """Load and index knowledge documents."""
        self.documents = load_knowledge(knowledge_dir)
        texts = [d["content"] for d in self.documents]
        
        if isinstance(self.backend, KeywordBackend):
            # Build TF-IDF index
            self._keyword_index = KeywordBackend()
            self._keyword_index.fit(texts)
        else:
            # Use embeddings (cached)
            content_hash = hashlib.md5(
                json.dumps([d["content"] for d in self.documents], ensure_ascii=False).encode()
            ).hexdigest()
            btype = "api" if isinstance(self.backend, ApiEmbeddingBackend) else "local"
            cache_path = CACHE_DIR / f"emb_{btype}_{content_hash}.npy"
            
            if cache_path.exists():
                self.embeddings = np.load(str(cache_path))
            else:
                self.embeddings = self.backend.encode(texts)
                np.save(str(cache_path), self.embeddings)
        
        self._loaded = True
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Semantic search over knowledge base."""
        if not self._loaded:
            raise RuntimeError("Knowledge not loaded. Call load_knowledge() first.")
        
        if isinstance(self.backend, KeywordBackend):
            # Use TF-IDF cosine similarity
            query_tokens = _tokenize(query)
            query_tf = Counter(query_tokens)
            query_vec = self._keyword_index._compute_tfidf(query_tf, len(self.documents))
            
            scores = [
                self._keyword_index._cosine_sim(query_vec, doc_vec)
                for doc_vec in self._keyword_index.doc_tfidf
            ]
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "source": self.documents[idx]["source"],
                    "chunk_id": self.documents[idx]["chunk_id"],
                    "content": self.documents[idx]["content"],
                    "score": float(scores[idx]),
                })
            return results
        else:
            # Use embedding cosine similarity
            query_emb = self.backend.encode([query])
            scores = np.dot(self.embeddings, query_emb.T).flatten()
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    "source": self.documents[idx]["source"],
                    "chunk_id": self.documents[idx]["chunk_id"],
                    "content": self.documents[idx]["content"],
                    "score": float(scores[idx]),
                })
            return results
    
    def format_context(self, query: str, top_k: int = 3) -> str:
        """Search and format results as a context string."""
        results = self.search(query, top_k=top_k)
        if not results:
            return "未找到相关信息。"
        
        parts = ["以下是知识库中检索到的相关内容：\n"]
        for i, r in enumerate(results, 1):
            if r["score"] > 0:
                parts.append(f"[{i}] 来源: {r['source']} (相关度: {r['score']:.2f})")
                parts.append(r["content"])
                parts.append("")
        
        if len(parts) == 1:
            return "未找到相关信息。"
        
        return "\n".join(parts)
    
    def status(self) -> Dict:
        """Return engine status info."""
        return {
            "loaded": self._loaded,
            "doc_count": len(self.documents),
            "backend": str(self.backend),
            "chunks": sorted(list(set(d["source"] for d in self.documents))),
        }
