"""
job-firm-rag-web — Web UI for Job-Firm RAG Agent
==================================================

FastAPI web server with:
  - POST /api/chat — API endpoint
  - GET / — Web Chat UI
  - GET /api/status — Knowledge base status

Usage:
    python web_ui.py
    # Then open http://localhost:8765
"""

import os
import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from agent import RagAgent

app = FastAPI(title="Job-Firm RAG Agent", version="1.0.0")

# Global agent instance (lazy init)
_agent: Optional[RagAgent] = None


def get_agent() -> RagAgent:
    global _agent
    if _agent is None:
        _agent = RagAgent()
    return _agent


class ChatRequest(BaseModel):
    message: str
    show_thinking: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    steps: list[dict] = []
    thought: str = ""


@app.get("/api/status")
async def status():
    """Return agent and knowledge base status."""
    try:
        engine = get_agent().engine
        s = engine.status()
        return {
            "status": "ok",
            "knowledge_loaded": s["loaded"],
            "chunks": s["doc_count"],
            "backend": str(s["backend"]),
            "sources": sorted(list(set(s["chunks"]))),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat with the RAG Agent."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        agent = get_agent()
        result = agent.run(req.message.strip())
        
        resp = ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            steps=result.get("steps", []),
            thought=result.get("thought", ""),
        )
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def home():
    """Web Chat UI."""
    return HTMLResponse(HTML_PAGE)


# ─── HTML UI (Embedded single-page) ───────────────────────────────────────────

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Job-Firm RAG Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif; 
         background: #0f172a; color: #e2e8f0; height: 100vh; display: flex; }
  
  /* Layout */
  .sidebar { width: 320px; background: #1e293b; padding: 20px; display: flex; flex-direction: column; 
             border-right: 1px solid #334155; }
  .main { flex: 1; display: flex; flex-direction: column; }
  
  /* Header */
  .header { padding: 20px 24px; background: #1e293b; border-bottom: 1px solid #334155; }
  .header h1 { font-size: 18px; color: #38bdf8; }
  .header p { font-size: 13px; color: #94a3b8; margin-top: 4px; }
  
  /* Chat area */
  .chat-area { flex: 1; overflow-y: auto; padding: 20px 24px; }
  .message { margin-bottom: 20px; max-width: 80%; }
  .message.user { margin-left: auto; }
  .message.user .bubble { background: #2563eb; color: white; border-radius: 16px 16px 4px 16px; }
  .message.agent .bubble { background: #334155; color: #e2e8f0; border-radius: 16px 16px 16px 4px; }
  .bubble { padding: 12px 16px; line-height: 1.6; font-size: 14px; white-space: pre-wrap; }
  .bubble .source-tag { display: inline-block; font-size: 11px; background: #475569; 
                         color: #94a3b8; padding: 2px 8px; border-radius: 10px; margin: 4px 4px 0 0; }
  .thinking { font-size: 12px; color: #64748b; padding: 4px 0; font-style: italic; }
  
  /* Input */
  .input-area { padding: 16px 24px; background: #1e293b; border-top: 1px solid #334155; display: flex; gap: 12px; }
  .input-area input { flex: 1; padding: 10px 16px; border: 1px solid #475569; border-radius: 24px;
                      background: #0f172a; color: #e2e8f0; font-size: 14px; outline: none; }
  .input-area input:focus { border-color: #38bdf8; }
  .input-area button { padding: 10px 24px; background: #2563eb; color: white; border: none;
                       border-radius: 24px; font-size: 14px; cursor: pointer; }
  .input-area button:hover { background: #1d4ed8; }
  .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
  
  /* Sidebar */
  .sidebar h2 { font-size: 14px; color: #38bdf8; margin-bottom: 12px; }
  .sidebar .status { font-size: 13px; line-height: 1.8; }
  .sidebar .status .label { color: #94a3b8; }
  .sidebar .status .value { color: #e2e8f0; }
  
  .sidebar .example-questions { margin-top: 20px; }
  .sidebar .example-item { padding: 8px 12px; background: #0f172a; border-radius: 8px; 
                          margin-bottom: 8px; font-size: 13px; cursor: pointer; 
                          transition: background 0.2s; border: 1px solid #334155; }
  .sidebar .example-item:hover { background: #1e293b; border-color: #38bdf8; }
  
  .thinking-panel { margin-top: 20px; flex: 1; overflow-y: auto; }
  .thinking-step { background: #0f172a; border-radius: 8px; padding: 10px; margin-bottom: 8px;
                   border-left: 3px solid #38bdf8; font-size: 12px; }
  .thinking-step .step-num { color: #38bdf8; font-weight: bold; }
  .thinking-step .step-action { color: #f59e0b; margin: 4px 0; }
  .thinking-step .step-thought { color: #94a3b8; }
  
  .empty-state { text-align: center; padding: 60px 20px; color: #64748b; }
  .empty-state h3 { font-size: 20px; margin-bottom: 8px; color: #94a3b8; }
  .empty-state p { font-size: 14px; }
  
  .loading { display: inline-block; width: 16px; height: 16px; border: 2px solid #475569; 
             border-top-color: #38bdf8; border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <h2>📚 知识库状态</h2>
  <div class="status" id="status">
    <div><span class="label">状态: </span><span class="value" id="status-indicator">加载中...</span></div>
    <div><span class="label">文档片段: </span><span class="value" id="chunk-count">-</span></div>
    <div><span class="label">数据源: </span><span class="value" id="sources-list">-</span></div>
  </div>
  
  <div class="example-questions">
    <h2>💡 试试问这些</h2>
    <div class="example-item" onclick="ask('平台需要办理哪些资质证照？')">平台需要办理哪些资质证照？</div>
    <div class="example-item" onclick="ask('发布差事有什么合规要求？')">发布差事有什么合规要求？</div>
    <div class="example-item" onclick="ask('如何防范履约方欺诈风险？')">如何防范履约方欺诈风险？</div>
    <div class="example-item" onclick="ask('差事退款的标准是什么？')">差事退款的标准是什么？</div>
    <div class="example-item" onclick="ask('平台用户信息泄露如何处理？')">平台用户信息泄露如何处理？</div>
  </div>
  
  <div class="thinking-panel" id="thinking-panel" style="display:none;">
    <h2>🧠 思考过程</h2>
    <div id="thinking-steps"></div>
  </div>
</div>

<!-- Main -->
<div class="main">
  <div class="header">
    <h1>🤖 Job-Firm RAG Agent</h1>
    <p>网络差事履约平台知识助手 · 基于检索增强生成 (RAG)</p>
  </div>
  
  <div class="chat-area" id="chat-area">
    <div class="empty-state" id="empty-state">
      <h3>欢迎使用 Job-Firm RAG Agent 🎉</h3>
      <p>网络差事履约平台的法律规范、风险分析、差事模板，随时问我！</p>
    </div>
  </div>
  
  <div class="input-area">
    <input id="input" type="text" placeholder="输入你的问题..." 
           onkeydown="if(event.key==='Enter') send()">
    <button id="send-btn" onclick="send()">发送</button>
  </div>
</div>

<script>
// ── State ──
let isLoading = false;
let thinkingSteps = [];

// ── Init ──
window.onload = async () => {
  loadStatus();
};

async function loadStatus() {
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();
    if (data.status === 'ok') {
      document.getElementById('status-indicator').textContent = '✅ 已就绪';
      document.getElementById('status-indicator').style.color = '#22c55e';
      document.getElementById('chunk-count').textContent = data.chunks;
      document.getElementById('sources-list').textContent = 
        (data.sources || []).join(', ');
    }
  } catch(e) {
    document.getElementById('status-indicator').textContent = '❌ 连接失败';
    document.getElementById('status-indicator').style.color = '#ef4444';
  }
}

// ── Chat ──
async function send() {
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg || isLoading) return;
  
  input.value = '';
  isLoading = true;
  document.getElementById('send-btn').disabled = true;
  document.getElementById('empty-state')?.remove();
  
  // Add user message
  addMessage(msg, 'user');
  
  // Show loading
  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'message agent';
  loadingDiv.id = 'loading-msg';
  loadingDiv.innerHTML = '<div class="bubble"><div class="loading"></div> 思考中...</div>';
  document.getElementById('chat-area').appendChild(loadingDiv);
  document.getElementById('chat-area').scrollTop = document.getElementById('chat-area').scrollHeight;
  
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg, show_thinking: true}),
    });
    const data = await resp.json();
    
    document.getElementById('loading-msg').remove();
    
    // Show agent steps in sidebar
    if (data.steps && data.steps.length > 0) {
      showThinking(data.steps);
    }
    
    // Format answer with sources
    let answerHtml = data.answer.replace(/\n/g, '<br>');
    if (data.sources && data.sources.length > 0) {
      answerHtml += '<br><br>';
      data.sources.forEach(s => {
        answerHtml += `<span class="source-tag">📄 ${s}</span>`;
      });
    }
    
    addMessage(answerHtml, 'agent');
    
  } catch(e) {
    document.getElementById('loading-msg').remove();
    addMessage('❌ 请求失败: ' + e.message, 'agent');
  }
  
  isLoading = false;
  document.getElementById('send-btn').disabled = false;
}

function addMessage(html, role) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `<div class="bubble">${html}</div>`;
  document.getElementById('chat-area').appendChild(div);
  document.getElementById('chat-area').scrollTop = document.getElementById('chat-area').scrollHeight;
}

function showThinking(steps) {
  const panel = document.getElementById('thinking-panel');
  const container = document.getElementById('thinking-steps');
  panel.style.display = 'block';
  container.innerHTML = '';
  
  steps.forEach((step, i) => {
    const div = document.createElement('div');
    div.className = 'thinking-step';
    let html = `<div class="step-num">Step ${step.step}</div>`;
    if (step.action) html += `<div class="step-action">🔧 ${step.action}</div>`;
    if (step.thought) html += `<div class="step-thought">💭 ${step.thought}</div>`;
    div.innerHTML = html;
    container.appendChild(div);
  });
}

function ask(question) {
  document.getElementById('input').value = question;
  send();
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8765))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🤖 Job-Firm RAG Agent Web UI")
    print(f"   Server: http://{host}:{port}")
    print(f"   API:    http://{host}:{port}/api/chat")
    print(f"   Docs:   http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)
