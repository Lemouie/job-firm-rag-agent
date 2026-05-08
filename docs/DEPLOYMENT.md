# 部署与运维指南 — Job-Firm RAG Agent

## 1. Docker 部署

### 1.1 构建镜像

```bash
cd job-firm-rag-agent
docker build -t job-firm-rag-agent:latest .
```

### 1.2 运行容器

```bash
docker run -d \
  --name job-firm-rag-agent \
  -p 8712:8712 \
  -e DEEPSEEK_API_KEY="sk-your-key-here" \
  -e DEEPSEEK_BASE_URL="https://api.deepseek.com" \
  job-firm-rag-agent:latest
```

### 1.3 Docker Compose（推荐）

创建 `docker-compose.yml`：

```yaml
version: '3.8'
services:
  rag-agent:
    build: .
    container_name: job-firm-rag-agent
    ports:
      - "8712:8712"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - DEEPSEEK_BASE_URL=https://api.deepseek.com
    restart: unless-stopped
    volumes:
      - ./knowledge:/app/knowledge
      - ./logs:/app/logs
```

启动：

```bash
DEEPSEEK_API_KEY="sk-xxx" docker compose up -d
```

---

## 2. 生产化建议

### 2.1 切换到 Embedding 检索

当前 Keyword 检索适合小规模知识库。生产环境建议升级：

**方案一：OpenAI Embedding API**
```python
# rag_engine.py 中启用
backend = ApiEmbeddingBackend(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)
engine = RagEngine("knowledge/", backend=backend)
```

**方案二：本地向量库（ChromaDB）**
```bash
pip install chromadb sentence-transformers
```

```python
import chromadb
from sentence_transformers import SentenceTransformer

class ChromaBackend:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("job-firm")
        self.encoder = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    def add_documents(self, chunks):
        embeddings = self.encoder.encode(chunks).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )

    def search(self, query, top_k=3):
        q_emb = self.encoder.encode([query]).tolist()
        results = self.collection.query(q_emb, n_results=top_k)
        return results['documents'][0]
```

### 2.2 多轮对话增强

当前实现维护 `messages` 列表，生产环境可：
- 增加 Token 限制和滑动窗口
- 使用 Redis 存储会话状态
- 支持消息流式输出 (SSE/WebSocket)

### 2.3 性能优化

| 优化项 | 说明 |
|-------|------|
| **异步处理** | FastAPI 原生 async 支持，配合 httpx.AsyncClient |
| **缓存** | LRU 缓存常见问题的检索结果 |
| **连接池** | 复用 LLM API 连接 |
| **批量推理** | 合并多个请求的 LLM 调用 |

### 2.4 监控与日志

```python
# 在 web_ui.py 中添加
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

# 请求计时中间件
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logging.info(f"{request.url.path} took {process_time:.3f}s")
    return response
```

---

## 3. 环境变量参考

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | ✅ | - | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | ❌ | `https://api.deepseek.com` | API 端点 |
| `DEEPSEEK_MODEL` | ❌ | `deepseek-chat` | 模型名称 |

---

## 4. 运维命令速查

### 健康检查

```bash
# 服务状态
curl http://localhost:8712/api/status

# 进程检查
ps aux | grep web_ui.py

# 端口检查
ss -tlnp | grep 8712
```

### 日志查看

```bash
# FastAPI 日志
journalctl -u job-firm-rag-agent -f

# Docker 日志
docker logs -f job-firm-rag-agent
```

### 知识库热更新

知识库文件修改后无需重启服务，Agent 会自动重新加载：

```bash
# 修改知识库后，触发重新加载
curl -X POST http://localhost:8712/api/reload
```

---

## 5. 安全注意事项

- 🔑 **API Key 管理**：使用环境变量，不要硬编码
- 🌐 **网络隔离**：服务监听 `127.0.0.1` 而非 `0.0.0.0`（或使用 Nginx 反向代理）
- 🛡️ **请求限流**：部署 nginx + 限流（防止 API 滥用）
- 📦 **依赖安全**：定期 `pip audit` 检查依赖漏洞
- 🔒 **HTTPS**：生产环境配置 SSL 证书
