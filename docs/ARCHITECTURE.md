# 架构详解 — Job-Firm RAG Agent

## 1. 总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      外部用户访问层                                 │
│  浏览器 (http://localhost:8712)  │  curl / API 客户端             │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼─────────────────────────────────────┐
│                    Web 服务层 (web_ui.py)                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  FastAPI Application                                        │ │
│  │  ├── GET  /api/status      — 服务健康检查 + 知识库状态       │ │
│  │  └── POST /api/chat        — 对话交互接口                    │ │
│  │                                                             │ │
│  │  会话管理: dict[session_id, list[messages]]                  │ │
│  │  跨域: CORS 中间件                                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬─────────────────────────────────────┘
                             │ 实例化
┌────────────────────────────▼─────────────────────────────────────┐
│                    Agent 逻辑层 (agent.py)                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  RagAgent                                                   │ │
│  │  ├── __init__(rag_engine, api_key, base_url, model)         │ │
│  │  ├── chat(user_input) → str  (主入口)                        │ │
│  │  ├── _needs_retrieval(query) → bool  (检索判断)              │ │
│  │  ├── _build_prompt(query, context) → str  (Prompt 构建)      │ │
│  │  └── _call_llm(messages) → str  (LLM API 调用)              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────┬────────────────────────────────────┬─────────────────┘
            │ 调用 search()                      │ 调用 LLM API
┌───────────▼──────────────────┐    ┌────────────▼────────────────┐
│    RAG 引擎层 (rag_engine.py) │    │     DeepSeek Chat API      │
│  ┌─────────────────────────┐ │    │  (或 OpenAI 兼容接口)       │
│  │ RagEngine               │ │    │                            │
│  │ ├── search(query, top_k)│ │    │  POST /v1/chat/completions │
│  │ ├── _tokenize(text)     │ │    │  model: deepseek-chat      │
│  │ └── KeywordBackend      │ │    └────────────────────────────┘
│  │     ├── build_index()   │ │
│  │     ├── search()        │ │
│  │     └── _bigram_tokenize│ │
│  └─────────────────────────┘ │
└───────────┬──────────────────┘
            │ 读取文件
┌───────────▼─────────────────────────────────────────────────────┐
│                    知识库 (knowledge/)                            │
│  ┌─────────────────┐ ┌──────────────┐ ┌──────────────────────┐  │
│  │   legal.md      │ │   risks.md   │ │   templates.md       │  │
│  │  法律规范与合规  │ │  风险分析与   │ │  差事模板与发布规范   │  │
│  └─────────────────┘ └──────────────┘ └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 核心数据流

### 2.1 完整请求流程

```
用户输入 "平台需要什么资质证照？"
      │
      ▼
┌─────────────────────────────┐
│  1. FastAPI 接收请求         │
│     POST /api/chat          │
│     {"message": "..."}      │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  2. Agent._needs_retrieval()│
│     关键词匹配: "资质" ✓    │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  3. RagEngine.search()      │
│     Bigram 分词 → TF 评分   │
│     → 返回 Top-3 相似 Chunk │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  4. Agent._build_prompt()   │
│     System Prompt + 上下文   │
│     + 检索结果 + 用户问题    │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  5. Agent._call_llm()       │
│     POST DeepSeek API       │
│     → 流式/非流式返回回答   │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  6. 返回 JSON 响应          │
│     {"response": "...",     │
│      "sources": [...]}      │
└─────────────────────────────┘
```

### 2.2 ReAct 循环细节

```
Step 1 [思考]: 分析用户意图
  - 关键词匹配：检测到 "资质"、"证照"
  - 决策：需要检索知识库

Step 2 [行动]: 调用 RAG 引擎
  - 查询: "平台需要什么资质证照"
  - Bigram 分词: ["平台", "台需", "需要", "要什", "什么", ...]
  - TF 评分 → 定位到 legal.md 第 1-7 行 "平台运营资质"

Step 3 [观察]: 获取检索结果
  - 返回: ICP备案、EDI许可证、等保2.0等条款

Step 4 [再思考]: 组织回答
  - 构建 Prompt: System + 检索结果 + 用户问题

Step 5 [回答]: LLM 生成
  - 输出带引用来源的完整回答
```

## 3. 关键设计决策

### 3.1 为什么用 Keyword 而非 Embedding？

| 维度 | Keyword Backend (当前) | Embedding Backend (可切换) |
|------|----------------------|--------------------------|
| 依赖 | 零依赖 (纯 Python) | sentence-transformers + PyTorch |
| 启动速度 | 毫秒级 | 5-30 秒 (加载模型) |
| 语义理解 | 精确匹配 | 同义词/上下文理解 |
| 中文效果 | Bigram 分词良好 | 取决于模型质量 |
| 适用规模 | 小规模 (< 1000 文档) | 大规模 (任意规模) |

**设计亮点**：使用 Strategy 模式，`RagEngine` 初始化时注入不同的 Backend：

```python
class RagEngine:
    def __init__(self, knowledge_dir, backend="keyword"):
        if backend == "keyword":
            self.backend = KeywordBackend()
        elif backend == "api_embedding":
            self.backend = ApiEmbeddingBackend(api_key=...)
```

### 3.2 Bigram 分词策略

```python
def _tokenize(self, text):
    # 1. 提取中文字符
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text.lower())
    # 2. 生成 Bigram
    bigrams = [chinese_chars[i] + chinese_chars[i+1]
               for i in range(len(chinese_chars) - 1)]
    # 3. 提取英文单词
    tokens = re.findall(r'[a-z]+', text.lower())
    return bigrams + tokens
```

**为什么用 Bigram？**
- 中文分词需要词典，Bigram 无需词典
- "营业执照" → ["营业", "执照"]，即使不知道"营业执照"是词也能匹配
- 平衡了召回率和精确率

## 4. 文件结构

```
job-firm-rag-agent/
├── README.md           # 项目说明文档
├── agent.py            # Agent 核心：ReAct 循环 + LLM 调用 (~277 行)
├── rag_engine.py       # RAG 检索引擎：Bigram 分词 + TF 评分 (~350 行)
├── web_ui.py           # FastAPI Web 界面 (~348 行)
├── Dockerfile          # Docker 容器化
├── requirements.txt    # Python 依赖
├── .gitignore          # Git 忽略规则
├── knowledge/          # 领域知识库
│   ├── legal.md        # 法律规范与合规
│   ├── risks.md        # 风险分析与防控
│   └── templates.md    # 差事模板与发布规范
├── tests/              # 测试
│   └── test_rag.py     # RAG 引擎测试
└── docs/               # 文档
    ├── ARCHITECTURE.md # 本文件
    ├── RAG_THEORY.md   # RAG 理论
    ├── API_REFERENCE.md# API 文档
    ├── INTERVIEW.md    # 面试指南
    └── DEPLOYMENT.md   # 部署指南
```

## 5. 扩展点

| 扩展方向 | 改动范围 | 说明 |
|---------|---------|------|
| 增加知识库 | 在 `knowledge/` 新增 `.md` 文件 | 自动加载，无需重启 |
| 切换 Embedding | `rag_engine.py` 启用 `ApiEmbeddingBackend` | 一行配置切换 |
| 增加 Agent 工具 | `agent.py` 注册新工具 | 如计算器、天气查询 |
| 多轮对话 | 利用 `messages` 列表维护历史 | 已预置支持 |
| 流式输出 | `web_ui.py` 改为 StreamingResponse | 提升用户体验 |
