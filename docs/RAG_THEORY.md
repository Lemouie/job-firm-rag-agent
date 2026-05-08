# RAG 核心理论 — Job-Firm RAG Agent

## 1. 什么是 RAG？

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是一种让 LLM 在回答前先从外部知识库检索相关信息的范式。

### 1.1 为什么需要 RAG？

```
传统 LLM 问答:
用户提问 ──→ LLM (仅凭训练知识) ──→ 可能过时/幻觉的回答

RAG 问答:
用户提问 ──→ 检索知识库 ──→ LLM (基于检索结果) ──→ 准确带引用的回答
```

**RAG 解决了三个核心问题：**

| 问题 | 说明 | RAG 方案 |
|------|------|---------|
| 🕐 **知识截止** | LLM 训练数据有截止日期 | 从最新知识库检索实时信息 |
| 🤔 **幻觉** | LLM 可能编造事实 | 强制 LLM 基于检索结果回答 |
| 🔒 **私有知识** | LLM 不知道企业内部数据 | 企业知识库作为外部信息来源 |

### 1.2 RAG vs 微调 (Fine-tuning)

| 维度 | RAG | 微调 |
|------|-----|------|
| 更新成本 | 替换文档即可 | 重新训练模型 |
| 幻觉控制 | 强（可溯源） | 弱（仍可能幻觉） |
| 领域深度 | 依赖知识库质量 | 融入模型参数 |
| 适用场景 | 知识密集、需验证的问答 | 风格/格式调整 |

---

## 2. RAG 的完整流程

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   Query   │───▶│ Retrieve │───▶│ Augment  │───▶│ Generate │
│  用户问题  │    │  检索    │    │  增强     │    │  生成    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │               │               │
                     │ 知识库         │ 拼入 Prompt   │ LLM
                     ▼               ▼               ▼
                ┌──────────┐   ┌──────────┐   ┌──────────┐
                │ Chunk 1  │   │ System   │   │  回答    │
                │ Chunk 2  │   │ + 检索结果 │   │         │
                │ Chunk 3  │   │ + 用户问题 │   │         │
                └──────────┘   └──────────┘   └──────────┘
```

### 2.1 索引阶段 (Indexing)

```python
# 知识库分块 (Chunking)
# 本项目：按 Markdown 标题分割，每个二级标题为一个 Chunk

# legal.md 被分割为 8 个 Chunk:
# - 1. 平台运营资质
# - 2. 用户协议关键条款
# - 3. 差事发布合规要求
# - ... 共 8 个 Chunk

# 三个知识库文件共产生 13 个 Chunk
```

### 2.2 检索阶段 (Retrieval)

本项目使用的 **Keyword 检索** 流程：

```
用户问题: "平台需要什么资质证照？"
     │
     ▼
Bigram 分词: ["平台", "台需", "需要", "要什", "什么", "资质", "质证", "证照"]
     │
     ▼
对每个 Chunk 计算 TF (词频) 得分:
  Chunk "平台运营资质":     "资质"×1, "平台"×1, "证照"×0 → 得分 2
  Chunk "差事发布合规要求":  "资质"×0, "平台"×0, "证照"×0 → 得分 0
  ...
     │
     ▼
返回得分最高的 Top-K 个 Chunk
```

### 2.3 增强阶段 (Augment)

检索结果拼入 Prompt 的模板：

```python
system_prompt = """你是一个 Job-Firm 平台专家助手。
请基于以下知识库内容回答问题。如果知识库中没有相关信息，
请明确说明"知识库中未找到相关信息"。不要编造答案。

知识库内容：
{context}

用户问题：{query}

请给出详细、准确的回答，并在适当位置标注信息来源。"""
```

### 2.4 生成阶段 (Generate)

LLM 基于增强后的 Prompt 生成回答，关键约束：
- ✅ 必须基于知识库内容
- ❌ 禁止编造知识库没有的信息
- 📎 引用来源（[来源：文件名]）

---

## 3. 检索策略对比

### 3.1 Keyword 检索（本项目当前使用）

```
优点: 零依赖、速度快、精确、可解释
缺点: 无法处理同义词（"营业执照" ≠ "经营许可证"）
适用: 小规模、术语固定的领域
```

### 3.2 Embedding 向量检索（预留）

```
优点: 语义理解（"驾照" ≈ "驾驶证"）、跨语言
缺点: 需要模型、需要向量库、资源消耗大
适用: 大规模、需要语义搜索的场景
```

### 3.3 Hybrid 混合检索（推荐生产环境）

```
结合 Keyword 精确匹配 + Embedding 语义匹配
流程:
  1. Keyword 检索 → 精确匹配结果
  2. Embedding 检索 → 语义相似结果  
  3. 结果融合 (RRF: Reciprocal Rank Fusion)
  4. 重排序 (Re-ranker)
```

### 3.4 检索策略对比表

| 策略 | 精确率 | 召回率 | 依赖 | 速度 | 适用规模 |
|------|--------|--------|------|------|---------|
| Keyword (当前) | ⭐⭐⭐⭐⭐ | ⭐⭐ | 零 | ⭐⭐⭐⭐⭐ | 小 |
| Embedding | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 重 | ⭐⭐⭐ | 中/大 |
| Hybrid | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 | ⭐⭐⭐ | 大 |
| Hybrid + Re-rank | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 重 | ⭐⭐ | 超大 |

---

## 4. 常见 RAG 优化技术

### 4.1 Chunking 策略

| 策略 | 说明 | 本项目 |
|------|------|--------|
| 固定大小 | 256/512 tokens 切分 | ✗ |
| 语义分割 | 按段落/标题分割 | ✅ (Markdown 标题) |
| 重叠窗口 | 块间重叠 10-20% | ✗ |
| Agentic Chunking | LLM 决定分块边界 | ✗ |

### 4.2 查询优化

- **查询重写**：将问题改写为更适合检索的形式
- **HyDE**：先让 LLM 生成假设性回答，再用回答去检索
- **多路召回**：同时用多种检索策略后融合结果

### 4.3 结果优化

- **重排序 (Re-ranking)**：用交叉编码器对初筛结果精细排序
- **上下文窗口压缩**：只保留最相关的部分，减少 Token 消耗
- **结果过滤**：按时间、来源、置信度过滤

---

## 5. 本项目中的 RAG 实现核心代码

```python
# rag_engine.py - Keyword 检索引擎核心

class KeywordBackend:
    def __init__(self):
        self.index = {}  # {token: [(chunk_idx, tf_in_chunk), ...]}

    def build_index(self, chunks):
        """构建倒排索引"""
        for idx, chunk in enumerate(chunks):
            tokens = self._tokenize(chunk)
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append((idx, count))

    def search(self, query, top_k=3):
        """搜索：计算查询与每个 Chunk 的 TF 得分"""
        query_tokens = self._tokenize(query)
        scores = defaultdict(float)
        for qt in query_tokens:
            if qt in self.index:
                for chunk_idx, tf in self.index[qt]:
                    scores[chunk_idx] += tf
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def _tokenize(self, text):
        """Bigram 分词"""
        chinese = re.findall(r'[\u4e00-\u9fff]', text.lower())
        bigrams = [chinese[i] + chinese[i+1]
                   for i in range(len(chinese)-1)]
        tokens = re.findall(r'[a-z]+', text.lower())
        return bigrams + tokens
```
