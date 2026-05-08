# API 参考文档 — Job-Firm RAG Agent

## 基础信息

| 项 | 值 |
|----|-----|
| **Base URL** | `http://localhost:8712` |
| **协议** | HTTP |
| **格式** | JSON |
| **编码** | UTF-8 |

---

## 1. GET /api/status — 服务健康检查

检查服务状态和知识库加载情况。

### 请求

```bash
curl http://localhost:8712/api/status
```

### 响应示例

```json
{
  "status": "ok",
  "knowledge_loaded": true,
  "chunks": 13,
  "knowledge_files": 3
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 服务状态，`"ok"` 或 `"error"` |
| `knowledge_loaded` | boolean | 知识库是否加载成功 |
| `chunks` | int | 知识库分割后的总 Chunk 数 |
| `knowledge_files` | int | 知识库文件数量 |

---

## 2. POST /api/chat — 对话交互

向 Agent 发送消息并获取回答。

### 请求

```bash
curl -X POST http://localhost:8712/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "平台运营需要什么资质证照？"}'
```

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message` | string | ✅ | 用户输入的问题 |
| `session_id` | string | ❌ | 会话 ID，留空自动生成 |

### 响应示例

```json
{
  "response": "根据知识库，平台运营需要以下资质证照：\n\n1. **ICP备案**：根据《互联网信息服务管理办法》，经营性互联网信息服务必须取得ICP备案/许可证\n2. **EDI许可证**：涉及在线数据处理与交易处理业务，需要办理EDI许可证\n3. **网络安全等级保护**：需完成等保2.0测评\n\n[来源：legal.md]",
  "session_id": "a1b2c3d4-...",
  "sources": ["legal.md"]
}
```

### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `response` | string | Agent 的文本回答 |
| `session_id` | string | 当前会话 ID（用于多轮对话） |
| `sources` | string[] | 回答引用的知识库文件列表 |

### 多轮对话示例

```bash
# 第一轮
curl -X POST http://localhost:8712/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "平台有哪些风险？"}'

# 第二轮（带上 session_id 维持上下文）
curl -X POST http://localhost:8712/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "如何防控这些风险？", "session_id": "SESSION_ID_FROM_LAST_RESPONSE"}'
```

---

## 3. 测试问题集

以下 4 个测试问题覆盖了知识库的各个领域：

### 法律合规类

```bash
curl -X POST http://localhost:8712/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "平台需要什么资质证照？"}'
```

**预期回答包含**: ICP 备案、EDI 许可证、等保 2.0

### 风险分析类

```bash
curl -X POST http://localhost:8712/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "平台的主要风险有哪些？"}'
```

**预期回答包含**: 法律合规风险、资金风险、履约风险、信息安全风险

### 差事模板类

```bash
curl -X POST http://localhost:8712/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "上门服务的发布要求是什么？"}'
```

**预期回答包含**: 服务地址、预约时间、资质证明

### 退款标准类

```bash
curl -X POST http://localhost:8712/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "差事退款标准是什么？"}'
```

**预期回答包含**: 100% 退款、50%-100%、违约扣 20%

---

## 4. 错误处理

### 400 Bad Request — 缺少参数

```json
{
  "detail": "message field is required"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Internal server error"
}
```

常见原因：
- API Key 未配置或无效
- 知识库文件缺失
- LLM API 调用超时

---

## 5. 客户端示例

### Python

```python
import httpx

BASE_URL = "http://localhost:8712"

def ask_agent(question: str, session_id: str = None) -> dict:
    payload = {"message": question}
    if session_id:
        payload["session_id"] = session_id
    resp = httpx.post(f"{BASE_URL}/api/chat", json=payload, timeout=30)
    return resp.json()

# 使用
result = ask_agent("平台需要什么资质证照？")
print(result["response"])
```

### JavaScript

```javascript
async function askAgent(question, sessionId = null) {
  const payload = { message: question };
  if (sessionId) payload.session_id = sessionId;

  const resp = await fetch("http://localhost:8712/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return await resp.json();
}

// 使用
askAgent("平台需要什么资质证照？").then(r => console.log(r.response));
```
