"""
Tests for Job-Firm RAG Agent.

Run with: python -m pytest tests/ -v
"""

import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_engine import RagEngine, chunk_text, load_knowledge


def test_chunk_text():
    """Test text chunking at sentence boundaries."""
    text = "第一句话。第二句话！第三句话？第四句话。"
    chunks = chunk_text(text, max_chars=20, overlap=0)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    assert "第一句话" in chunks[0]


def test_load_knowledge():
    """Test knowledge loading from markdown files."""
    knowledge_dir = Path(__file__).parent.parent / "knowledge"
    docs = load_knowledge(str(knowledge_dir))
    assert len(docs) > 0, "Expected at least 1 document chunk"
    assert any("legal" in d["source"] for d in docs), "Expected legal.md documents"
    assert any("risks" in d["source"] for d in docs), "Expected risks.md documents"
    assert any("templates" in d["source"] for d in docs), "Expected templates.md documents"


def test_rag_engine():
    """Test RAG engine search functionality."""
    engine = RagEngine()
    knowledge_dir = Path(__file__).parent.parent / "knowledge"
    engine.load_knowledge(str(knowledge_dir))
    
    # Test search
    results = engine.search("资质证照", top_k=2)
    assert len(results) <= 2, f"Expected at most 2 results, got {len(results)}"
    if len(results) > 0:
        assert "content" in results[0]
        assert "score" in results[0]
        assert results[0]["score"] > 0
    
    # Test format context
    context = engine.format_context("ICP备案", top_k=2)
    assert "检索到" in context or "知识库" in context or "来源" in context


def test_rag_engine_status():
    """Test engine status output."""
    engine = RagEngine()
    knowledge_dir = Path(__file__).parent.parent / "knowledge"
    engine.load_knowledge(str(knowledge_dir))
    
    status = engine.status()
    assert status["loaded"] == True
    assert status["doc_count"] > 0
    assert status["model"] is not None
