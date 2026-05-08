# Job-Firm RAG Agent 🤖📚

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)](https://fastapi.tiangolo.com/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek%20Chat-4F46E5)](https://deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **检索增强生成（RAG）Agent** — 基于 ReAct 循环构建的领域智能知识助手，为网络差事履约平台提供法律合规、风险防控、差事模板等专业问答能力。

基于 [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) 的干净 Agent 循环 + **ReAct** 模式构建，使用 Bigram 分词零依赖检索引擎，2 小时完成从设计到部署。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **ReAct 推理循环** | 思考 → 检索 → 观察 → 回答的 Agent 工作流 |
| 📚 **RAG 检索增强** | 先检索知识库再生成回答，减少 LLM 幻觉 |
| 🔍 **零依赖检索引擎** | Bigram 分词中文检索，无需 PyTorch / Embedding 模型 |
| 🌐 **Web UI + REST API** | FastAPI 提供双接口，支持浏览器和 curl 访问 |
| 🔌 **可切换后端** | 预留 ApiEmbeddingBackend，一行配置可升级到向量检索 |
| 📁 **领域知识库** | 覆盖法律、风险、模板三大领域，共 13 个语义块 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Web UI (FastAPI)                           │
│              POST /api/chat · GET /api/status                │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                    Agent Layer (ReAct)                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  1. 分析问题 → 2. 判断是否需要检索 → 3. 调用 RAG 引擎   │ │
│  │  4. 构建带上下文的 Prompt → 5. LLM 生成回答              │ │
│  └─────────────────────────────────────────────────────────┘ │
└────────┬──────────────────────────────────┬──────────────────┘
         │ 检索调用                         │ LLM 调用
         ▼                                  ▼
┌──────────────────┐              ┌──────────────────────┐
│   RAG Engine     │              │   DeepSeek Chat API  │
│  KeywordBackend  │              │   (外部 LLM 服务)    │
│  + Bigram 分词   │              │                      │
└────────┬─────────┘              └──────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│                  Knowledge Base (knowledge/)               │
│  ┌─────────────────┐ ┌──────────────┐ ┌────────────────┐ │
│  │   legal.md      │ │   risks.md   │ │  templates.md  │ │
│  └─────────────────┘ └──────────────┘ └────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 三层架构

| 层级 | 文件 | 职责 |
|------|------|------|
| **Web UI 层** | `web_ui.py` | FastAPI HTTP 接口，对话会话管理 |
| **Agent 层** | `agent.py` | ReAct 推理循环，LLM 调用，工具编排 |
| **知识层** | `rag_engine.py` + `knowledge/` | 检索引擎 + 领域知识库 |

---

## 🚀 快速开始

### 前置要求

- **Python 3.10+**
- **DeepSeek API Key**（或兼容 OpenAI API 的其他 LLM）

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/Lemouie/job-firm-rag-agent.git
cd job-firm-rag-agent

# 2. 安装依赖（纯 Python，无需 PyTorch）
pip install httpx fastapi uvicorn numpy
```

### 配置 API Key

```bash
# 环境变量
export DEEPSEEK_API_KEY="sk-your-key-here"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

### 启动 Web 服务

```bash
python web_ui.py
# 服务运行在 http://localhost:8712
```

### 快速测试

```bash
# 检查服务状态
curl http://localhost:8712/api/status

# 提问
curl -X POST http://localhost:8712/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "平台运营需要什么资质证照？"}'
```

---

## 📖 知识库内容

| 文件 | 覆盖内容 | 行数 |
|------|---------|------|
| `knowledge/legal.md` | ICP 备案、EDI 许可证、用户协议、隐私政策、资金托管、纠纷处理、数据合规、税务合规 | 55 |
| `knowledge/risks.md` | 法律合规风险、资金风险、履约风险、人身安全、信息安全、信用风险、防控策略 | 90 |
| `knowledge/templates.md` | 跑腿代办、排队占位、上门服务、技能服务模板 + 发布/定价/安全规范 | 129 |

---

## 📚 文档导航

| 文档 | 内容 |
|------|------|
| [架构详解](docs/ARCHITECTURE.md) | 三层架构设计、ReAct 循环实现、类图、数据流 |
| [RAG 理论](docs/RAG_THEORY.md) | RAG 核心概念、检索策略对比、Embedding 与 Keyword |
| [API 参考](docs/API_REFERENCE.md) | 完整 API 接口文档，请求/响应示例 |
| [面试指南](docs/INTERVIEW.md) | 面试 Q&A 大全，名词解释，通关要点 |
| [部署运维](docs/DEPLOYMENT.md) | Docker 部署、生产化建议、性能优化 |

---

## 🧪 测试

```bash
cd job-firm-rag-agent && python -m pytest tests/ -v
```

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **Web 框架** | FastAPI + Uvicorn |
| **LLM** | DeepSeek Chat API（兼容 OpenAI 格式） |
| **检索引擎** | 自研 Bigram 分词 Keyword 检索（零依赖） |
| **Agent 模式** | ReAct（Reasoning + Acting） |
| **部署** | Docker / 裸机运行 |
| **测试** | pytest |

---

## 🎯 面试知识点速查

| 概念 | 说明 |
|------|------|
| **Agent** | 自主分析问题、选择工具、生成回答的 AI 程序 |
| **ReAct** | 思考→检索→观察→回答的推理行动循环 |
| **RAG** | 检索增强生成——先查知识库再回答 |
| **Tool-use** | Agent 调用 RAG 引擎作为外部工具 |
| **Bigram 分词** | 中文按相邻字符匹配，英文按单词匹配 |
| **Keyword 检索** | 基于词频（TF）的精确匹配检索 |
| **Chunking** | 知识库按 Markdown 标题/段落分块 |

---

## 📄 许可证

MIT License
