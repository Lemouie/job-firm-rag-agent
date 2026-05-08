# 面试指南 — Job-Firm RAG Agent

> 本文档整理 Agent 开发面试中的高频问题、核心名词解释和标准答案，基于本项目的实际实现。

---

## 🌱 第一轮 · 基础层

### Q1：什么是 Agent？你的项目里 Agent 怎么工作？

**📖 名词解释：Agent（智能体）**
> Agent 是一个能**自主感知环境、做出决策并执行行动**的 AI 程序。与普通 LLM 聊天不同，Agent 可以：
> - **使用工具**（搜索、计算、调用 API）
> - **多步推理**（思考 → 行动 → 观察 → 再思考）
> - **维护状态**（记住上下文和中间结果）

**答案：**
我的项目中，`RagAgent` 是一个典型的 ReAct 风格 Agent。工作流程是：

```
用户提问 → Agent 分析问题 → 调用 RAG 检索工具
         → 获取相关知识块 → LLM 综合回答 → 返回带引用的答案
```

核心代码在 `agent.py` 中：
```python
class RagAgent:
    def chat(self, user_input):
        # 1. 分析是否需要检索
        if self._needs_retrieval(user_input):
            # 2. 执行检索（工具调用）
            context = self.rag_engine.search(user_input)
            # 3. 构建带上下文的 prompt
            prompt = self._build_prompt(user_input, context)
        # 4. LLM 生成最终回答
        return self._call_llm(prompt)
```

### Q2：什么是 ReAct 循环？你的项目用到了吗？

**📖 名词解释：ReAct（Reasoning + Acting）**
> 由 Google 在 2022 年提出的 Agent 范式，核心是 **"推理 + 行动"** 交替：
> ```
> Thought（思考） → Action（行动） → Observation（观察）
> → Thought（再思考） → Action ... → Final Answer
> ```
> 相比 CoT（思维链），ReAct 让 Agent 能**与环境交互**获取新信息。

**答案：**
我的项目实现了**简化版 ReAct 循环**：

```
Thought: "用户问的是法律资质问题，需要查知识库"
Action: rag_engine.search("资质证照")
Observation: 返回了《平台运营资质》相关段落
Thought: "基于检索结果，组织包含具体条款的完整回答"
Final Answer: "平台需要营业执照、ICP 许可证..."
```

架构上是可扩展到多轮 ReAct 的。

### Q3：画一下项目架构

```
┌─────────────────────────────────────┐
│     Web UI (FastAPI · :8712)        │
│  POST /api/chat · GET /api/status   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Agent Layer (ReAct 循环)           │
│   RagAgent: 分析→检索→Prompt→回答   │
└──────┬──────────────────────┬───────┘
       │                      │
       ▼                      ▼
┌──────────────┐    ┌────────────────┐
│  RAG Engine  │    │  DeepSeek API  │
│ KeywordBackend│    │  (外部 LLM)    │
└──────┬───────┘    └────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Knowledge/                         │
│  legal.md · risks.md · templates.md │
└─────────────────────────────────────┘
```

---

## 🌿 第二轮 · 核心层

### Q4：什么是 RAG？为什么选 RAG Agent？

**📖 名词解释：RAG（检索增强生成）**
> 让 LLM 在回答前先从外部知识库检索相关信息。流程：
> ```
> Query → Retrieve → Augment → Generate
> 问题  →  检索知识  → 拼入Prompt → LLM 生成
> ```
> 解决 LLM 的**知识截止**和**幻觉**问题。

**答案：**
我选择 RAG Agent 是因为 Job-Firm-Platform 是一个垂直领域项目，包含法律合规、风险防控等专业内容。LLM 通用知识不足以准确回答这些问题，所以我构建了领域知识库，通过 RAG 让 Agent 在回答前检索相关知识，**既保证了准确性，又提供了可追溯的引用来源**。

### Q5：为什么用 Keyword 而非 Embedding？

**📖 名词解释：Embedding / 向量检索**
> **Embedding**：将文本转换为向量，语义相近的文本在向量空间也相近。
> **向量检索**：通过计算余弦相似度找到最相关的文本块。
> - 优点：理解语义（"营业执照"≈"经营许可证"）
> - 缺点：需要模型 + 向量库，依赖重

**答案：**
原本计划用 `sentence-transformers`，但遇到了 PyTorch 下载太慢的问题。在 2 小时时限下，我做了一个**关键工程决策**：

> **用 Keyword 检索保证项目可运行，同时预留了 ApiEmbeddingBackend 接口。**

Keyword 在知识库规模较小（13 个 Chunk）时效果足够，我用 **Strategy 模式**设计了可切换后端，后续可以无缝切换到向量检索。

### Q6：Tool-use 怎么设计的？

**📖 名词解释：Tool-use（工具使用）**
> Agent 调用外部功能的能力。工具在代码中表示为：
> ```python
> tools = {
>     "search_knowledge": {
>         "description": "检索知识库",
>         "function": rag_engine.search
>     }
> }
> ```

**答案：**
项目中 Tool-use 体现在：
1. **隐式**：`_needs_retrieval()` 判断是否需要调用 RAG
2. **可扩展**：架构预留了多工具注册
```python
self.tools = {
    "search_knowledge": self.rag_engine.search,
    # 未来可添加计算器、天气查询等
}
```

---

## 🌳 第三轮 · 进阶层

### Q7：知识库扩大到 10 万份文档怎么改？

| 当前（小规模） | 升级后（大规模） |
|---|---|
| Keyword 检索 | Embedding 向量检索 |
| 文件存储 | 向量数据库 (ChromaDB/Pinecone) |
| 13 个 Chunk | 层次化分块 + 粗排→精排 |
| 单机 | 分布式检索 |

具体路径：`KeywordBackend → HybridBackend(keyword + vector) → 向量库分片`

### Q8：如何评估 RAG Agent？

**📖 名词解释：RAG 评估指标**
> - **Faithfulness**：答案是否基于检索内容
> - **Relevance**：检索结果是否相关
> - **Hit Rate**：检索是否找到正确答案

项目中已有基础测试（4 个问题验证检索准确性），生产级还需：
1. Retri eval 评估：Precision@K, Recall@K
2. LLM-as-Judge 自动打分
3. 端到端测试集

### Q9：如何防止 LLM 幻觉？

四种策略：

| 策略 | 实现 |
|------|------|
| **Grounding** | Prompt 强制 "只基于以下内容回答" |
| **引用溯源** | 标注 `[来源：legal.md]` |
| **拒绝回答** | 检索不到信息时说 "知识库中未找到" |
| **System Prompt 约束** | 禁止编造 |

---

## 🏆 项目介绍范文

> "我构建了一个 **Job-Firm RAG Agent**，面向网络差事履约平台的领域专家问答系统。项目采用三层架构：FastAPI 提供 Web 接口、RagAgent 实现 ReAct 风格推理循环、Keyword 检索引擎从领域知识库检索相关信息。核心创新在于用 **Bigram 分词**实现了零依赖的中文检索引擎，避免了 Embedding 模型的环境依赖问题。整个项目 **2 小时内**完成从设计到部署，已在 GitHub 开源。"

---

## 📋 面试高频知识点速查

| 术语 | 一句话解释 |
|------|-----------|
| **Agent** | 能自主思考和行动的 AI 程序 |
| **ReAct** | 推理+行动交替的 Agent 范式 |
| **RAG** | 先检索知识再回答，减少幻觉 |
| **Tool-use** | Agent 调用外部工具 |
| **Embedding** | 文本转向量，实现语义检索 |
| **Chunking** | 长文档切小块，方便检索 |
| **Hallucination** | LLM 编造信息，RAG 是最佳对抗手段 |
| **Bigram** | 两两字符组合，中文词汇的近似匹配 |
| **TF (Term Frequency)** | 词频，衡量词在文档中重要性 |
| **倒排索引** | 词→文档的映射，加速检索 |
