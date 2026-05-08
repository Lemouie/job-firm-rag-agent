"""
job-firm-rag-agent — RAG Agent for Job-Firm Platform
=====================================================

A minimal ReAct-style agent with RAG (Retrieval-Augmented Generation).
Pattern inspired by:
  - mini-SWE-agent (~100 line agent loop)
  - agent-dev (AutoGPT tool-use pattern)

Core Loop:
  1. System prompt defines persona + tools
  2. User asks question about the Job-Firm platform
  3. Agent decides: retrieve knowledge OR answer directly
  4. If retrieve → RAG engine fetches context → agent synthesizes answer
  5. Returns answer with citations
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from rag_engine import RagEngine

logger = logging.getLogger("rag-agent")

# ─── Auto-detect API Key ──────────────────────────────────────────────────────

def _load_env_file(path: str) -> dict:
    """Load .env file, return parsed key-value pairs."""
    env = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    except (FileNotFoundError, PermissionError):
        pass
    return env


def _get_api_config():
    """
    Auto-detect LLM API config from:
    1. Environment variables (highest priority)
    2. ~/.hermes/.env file
    """
    # Try env first
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    
    # Fall back to Hermes .env
    if not api_key:
        hermes_env = _load_env_file(str(Path.home() / ".hermes" / ".env"))
        api_key = hermes_env.get("DEEPSEEK_API_KEY", "")
        base_url = hermes_env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    return api_key, base_url

# ─── Agent Configuration ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是 Job-Firm RAG Agent，一个**网络差事履约平台**的智能知识助手。

你的核心能力：
1. 使用 RAG（检索增强生成）从知识库中检索平台相关法律、风险、模板信息
2. 基于检索结果提供准确、专业的回答

## 可用工具

当你需要查询知识库时，请输出：

THOUGHT: 分析用户的问题，判断是否需要检索知识库
ACTION: search <查询关键词>

系统会返回相关知识片段给你，然后你再基于这些知识生成回答。

当你不需要检索、或者已经获得足够的知识时，请直接输出：

ANSWER: 你对用户问题的完整回答

## 回答要求
- 基于检索结果回答，不要编造知识库中不存在的内容
- 如果知识库中没有相关信息，如实告知用户
- 引用知识来源（如"根据法律规范文档第X条"）"""


class RagAgent:
    """
    Minimal ReAct-style RAG Agent.
    
    Usage:
        agent = RagAgent()
        agent.run("发布差事需要什么资质？")
    """
    
    def __init__(self, knowledge_dir: str = None, model: str = None):
        self.engine = RagEngine()
        self.engine.load_knowledge(knowledge_dir)
        
        # Auto-detect API config
        self.api_key, self.base_url = _get_api_config()
        self.model = model or os.environ.get("AGENT_MODEL", "deepseek-chat")
        
        self.messages: List[Dict] = []
        self.history: List[Dict] = []
    
    def _call_llm(self, messages: List[Dict]) -> str:
        """Call the LLM API."""
        import httpx
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        resp = httpx.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        return data["choices"][0]["message"]["content"].strip()
    
    def _parse_response(self, response: str) -> Dict:
        """Parse agent response into structured action."""
        lines = response.strip().split("\n")
        
        thought = ""
        action = None
        answer = ""
        
        for line in lines:
            if line.startswith("THOUGHT:"):
                thought = line[len("THOUGHT:"):].strip()
            elif line.startswith("ACTION:"):
                action = line[len("ACTION:"):].strip()
            elif line.startswith("ANSWER:"):
                answer = line[len("ANSWER:"):].strip()
        
        # If no structured format, treat as ANSWER
        if not thought and not action and not answer:
            return {"type": "answer", "content": response}
        
        if answer:
            return {"type": "answer", "content": answer, "thought": thought}
        
        if action:
            # Parse action: "search <query>"
            if action.lower().startswith("search "):
                query = action[len("search "):].strip()
                return {"type": "search", "query": query, "thought": thought}
            else:
                return {"type": "answer", "content": f"未知动作: {action}", "thought": thought}
        
        return {"type": "answer", "content": response}
    
    def _search_knowledge(self, query: str) -> str:
        """Perform RAG search and return formatted context."""
        context = self.engine.format_context(query, top_k=3)
        return context
    
    def run(self, user_input: str, max_steps: int = 5) -> Dict:
        """
        Run the agent loop.
        
        Returns dict with:
          - answer: final answer
          - steps: list of step logs
          - sources: used knowledge sources
        """
        steps = []
        used_sources = set()
        
        # Initialize conversation
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        
        for step in range(max_steps):
            step_log = {"step": step + 1, "action": None, "result": None}
            
            # Call LLM
            response = self._call_llm(self.messages)
            parsed = self._parse_response(response)
            
            if parsed["type"] == "search":
                query = parsed["query"]
                step_log["action"] = f"search({query})"
                step_log["thought"] = parsed.get("thought", "")
                
                # Execute RAG search
                context = self._search_knowledge(query)
                step_log["result"] = f"retrieved {context.count('[')} relevant chunks"
                
                # Track sources
                for line in context.split("\n"):
                    if line.startswith("来源:"):
                        used_sources.add(line.split("来源:")[1].split("(")[0].strip())
                
                # Feed observation back to LLM
                observation = f"## 检索结果\n\n{context}\n\n请基于以上检索结果回答用户问题。"
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append({"role": "user", "content": observation})
                steps.append(step_log)
                
            elif parsed["type"] == "answer":
                step_log["action"] = "answer"
                step_log["thought"] = parsed.get("thought", "")
                step_log["result"] = parsed["content"]
                
                self.messages.append({"role": "assistant", "content": response})
                steps.append(step_log)
                
                return {
                    "answer": parsed["content"],
                    "steps": steps,
                    "sources": list(used_sources),
                    "thought": parsed.get("thought", ""),
                }
            
            else:
                # Fallback
                step_log["action"] = "unknown"
                step_log["result"] = response
                steps.append(step_log)
                return {
                    "answer": response,
                    "steps": steps,
                    "sources": list(used_sources),
                }
        
        # Max steps reached
        return {
            "answer": "Agent reached maximum steps without concluding.",
            "steps": steps,
            "sources": list(used_sources),
        }
    
    def chat(self, user_input: str) -> str:
        """Simple chat interface — returns just the answer string."""
        result = self.run(user_input)
        answer = result["answer"]
        
        # Append sources if available
        if result.get("sources"):
            sources_str = "\n\n📚 **引用来源**: " + ", ".join(result["sources"])
            answer += sources_str
        
        return answer


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    """Command-line interface for the RAG Agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Job-Firm RAG Agent")
    parser.add_argument("query", nargs="?", help="Question to ask the agent")
    parser.add_argument("--knowledge", "-k", default=None, help="Knowledge directory path")
    parser.add_argument("--model", "-m", default=None, help="LLM model name")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--status", "-s", action="store_true", help="Show knowledge base status")
    
    args = parser.parse_args()
    
    agent = RagAgent(knowledge_dir=args.knowledge, model=args.model)
    
    if args.status:
        status = agent.engine.status()
        print(f"📚 Knowledge Base Status")
        print(f"   Loaded: {status['loaded']}")
        print(f"   Documents: {status['doc_count']} chunks")
        print(f"   Model: {status['model']}")
        print(f"   Sources: {list(set(status['chunks']))}")
        return
    
    if args.interactive or not args.query:
        print("🤖 Job-Firm RAG Agent (输入 'quit' 退出)")
        print(f"   知识库已加载: {agent.engine.status()['doc_count']} 个文档片段")
        print()
        while True:
            try:
                query = input("你: ")
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query.strip():
                continue
            print("\n🤖 思考中...")
            answer = agent.chat(query)
            print(f"\n🤖: {answer}\n")
    elif args.query:
        print(agent.chat(args.query))


if __name__ == "__main__":
    main()
