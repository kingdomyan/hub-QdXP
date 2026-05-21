import requests
import anthropic

# ====================== 配置 ======================
JINA_API_KEY = "YOUR_JINA_API_KEY"
CLAUDE_API_KEY = "YOUR_CLAUDE_API_KEY"

JINA_BASE_URL = "https://api.jina.ai/v1"
JINA_URL_READER = "https://r.jina.ai"
JINA_WEB_SEARCH = "https://s.jina.ai"
JINA_DEEP_SEARCH = "https://deepsearch.jina.ai/v1/chat/completions"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"

# ====================== Jina AI 客户端 ======================
class JinaClient:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {JINA_API_KEY}"}

    def read_url(self, url: str) -> str:
        response = requests.get(f"{JINA_URL_READER}/{url}", headers=self.headers)
        return response.text

    def web_search(self, query: str) -> list:
        response = requests.get(f"{JINA_WEB_SEARCH}/{query}", headers=self.headers)
        return response.json().get("data", [])

    def deep_search(self, query: str) -> str:
        payload = {
            "model": "jina-deepsearch",
            "messages": [{"role": "user", "content": query}]
        }
        response = requests.post(JINA_DEEP_SEARCH, json=payload, headers=self.headers)
        return response.json()["choices"][0]["message"]["content"]

    def embedding(self, inputs):
        payload = {"input": inputs, "model": "jina-embeddings-v4"}
        response = requests.post(f"{JINA_BASE_URL}/embeddings", json=payload, headers=self.headers)
        return [d["embedding"] for d in response.json()["data"]]

    def rerank(self, query, documents):
        payload = {
            "query": query,
            "documents": documents,
            "model": "jina-reranker-v3"
        }
        response = requests.post(f"{JINA_BASE_URL}/rerank", json=payload, headers=self.headers)
        return response.json()["results"]

    def segment(self, text):
        payload = {"text": text}
        response = requests.post(f"{JINA_BASE_URL}/segment", json=payload, headers=self.headers)
        return response.json()["segments"]

    def classify(self, inputs, labels, examples=None):
        payload = {"input": inputs, "labels": labels}
        if examples:
            payload["examples"] = examples
        response = requests.post(f"{JINA_BASE_URL}/classify", json=payload, headers=self.headers)
        return response.json()["classes"]

# ====================== Claude Code 客户端 ======================
class ClaudeCodeClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    def code_chat(self, prompt: str, code: str = None) -> str:
        content = f"指令：{prompt}\n代码：{code if code else '无'}"
        message = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}]
        )
        return message.content[0].text

    def rag_chat(self, query: str, context: str) -> str:
        prompt = f"使用以下资料回答问题：\n{context}\n\n问题：{query}"
        message = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

# ====================== RAG 引擎 ======================
class RAGEngine:
    def __init__(self):
        self.jina = JinaClient()
        self.claude = ClaudeCodeClient()

    def run_rag(self, query: str, url: str = None, text: str = None):
        if url:
            content = self.jina.read_url(url)
        else:
            content = text

        chunks = self.jina.segment(content)
        ranked = self.jina.rerank(query, chunks)
        top_context = "\n".join([item["document"]["text"] for item in ranked[:3]])
        return self.claude.rag_chat(query, top_context)

# ====================== 统一标准接口 ======================
class MultimodalRAGChatbot:
    def __init__(self):
        self.jina = JinaClient()
        self.claude = ClaudeCodeClient()
        self.rag = RAGEngine()

    def chat(self, query: str, images=None, use_rag=True):
        if not use_rag:
            return self.claude.rag_chat(query, "")
        search_data = self.jina.web_search(query)
        context = "\n".join([item["content"] for item in search_data[:2]])
        return self.claude.rag_chat(query, context)

    def embed(self, inputs, modality="text"):
        return self.jina.embedding(inputs)

    def read_url(self, url):
        return self.jina.read_url(url)

    def web_search(self, query):
        return self.jina.web_search(query)

    def deep_search(self, query):
        return self.jina.deep_search(query)

    def rerank(self, query, docs):
        return self.jina.rerank(query, docs)

    def segment_text(self, text):
        return self.jina.segment(text)

    def classify(self, inputs, labels, few_shot=None):
        return self.jina.classify(inputs, labels, few_shot)

    def code_chat(self, query, code=None):
        return self.claude.code_chat(query, code)

# ====================== 测试入口 ======================
if __name__ == "__main__":
    bot = MultimodalRAGChatbot()

    print("=" * 50)
    print("多模态 RAG 聊天机器人（Jina + Claude Code）")
    print("=" * 50)

    # 测试1：普通对话
    print("\n【1】普通对话")
    print(bot.chat("Jina AI 提供了哪些核心API？")[:150])

    # 测试2：代码问答
    print("\n【2】代码生成")
    print(bot.code_chat("写一个Python快速排序函数")[:150])

    # 测试3：URL 读取 + RAG
    print("\n【3】URL RAG 问答")
    print(bot.rag.run_rag("这个网站是做什么的？", url="https://jina.ai")[:150])
