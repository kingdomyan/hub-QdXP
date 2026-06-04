# EmbeddingsCache.py
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HuggingFaceTextVectorizer
import hashlib

class EmbeddingsCache:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.vectorizer = HuggingFaceTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
        self.dim = self.vectorizer.dims

        self.schema = IndexSchema.from_dict({
            "index": {"name": "embedding_cache", "prefix": "emb:"},
            "fields": {
                "text": {"type": "text"},
                "embedding": {"type": "vector", "dims": self.dim, "algorithm": "hnsw"}
            }
        })

        self.index = SearchIndex(self.schema, redis_url=redis_url)
        self.index.create(overwrite=False)

    def _key(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def get_or_embed(self, text):
        key = f"emb:{self._key(text)}"
        exists = self.index.client.exists(key)

        if exists:
            return self.index.client.hget(key, "embedding")

        emb = self.vectorizer.embed(text)
        self.index.client.hset(key, mapping={
            "text": text,
            "embedding": emb.tobytes()
        })
        return emb
# SemanticCache.py
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HuggingFaceTextVectorizer

class SemanticCache:
    def __init__(self, threshold=0.92):
        self.vectorizer = HuggingFaceTextVectorizer()
        self.threshold = threshold

        self.schema = IndexSchema.from_dict({
            "index": {"name": "semantic_cache", "prefix": "scache:"},
            "fields": {
                "query": {"type": "text"},
                "answer": {"type": "text"},
                "embedding": {"type": "vector", "dims": 384}
            }
        })

        self.index = SearchIndex(self.schema)
        self.index.create(overwrite=False)

    def get(self, query):
        vec = self.vectorizer.embed(query)
        vq = VectorQuery(
            vector=vec,
            field="embedding",
            num_results=1
        )
        res = self.index.query(vq)
        if res and res[0]["score"] >= self.threshold:
            return res[0]["answer"]
        return None

    def set(self, query, answer):
        vec = self.vectorizer.embed(query)
        self.index.load([
            {"query": query, "answer": answer, "embedding": vec}
        ])
# SemanticMessageHistory.py
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HuggingFaceTextVectorizer
from datetime import datetime

class SemanticMessageHistory:
    def __init__(self, session_id):
        self.session_id = session_id
        self.vectorizer = HuggingFaceTextVectorizer()

        self.schema = IndexSchema.from_dict({
            "index": {"name": "message_history", "prefix": "msg:"},
            "fields": {
                "session": {"type": "tag"},
                "role": {"type": "text"},
                "content": {"type": "text"},
                "timestamp": {"type": "numeric"},
                "embedding": {"type": "vector", "dims": 384}
            }
        })

        self.index = SearchIndex(self.schema)
        self.index.create(overwrite=False)

    def add(self, role, content):
        self.index.load([{
            "session": self.session_id,
            "role": role,
            "content": content,
            "timestamp": int(datetime.now().timestamp()),
            "embedding": self.vectorizer.embed(content)
        }])

    def recall(self, query, top_k=5):
        vec = self.vectorizer.embed(query)
        vq = VectorQuery(vector=vec, field="embedding", num_results=top_k)
        return self.index.query(vq)
# SemanticRouter.py
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HuggingFaceTextVectorizer

class SemanticRouter:
    def __init__(self):
        self.vectorizer = HuggingFaceTextVectorizer()

        self.schema = IndexSchema.from_dict({
            "index": {"name": "semantic_routes", "prefix": "route:"},
            "fields": {
                "intent": {"type": "text"},
                "description": {"type": "text"},
                "embedding": {"type": "vector", "dims": 384}
            }
        })

        self.index = SearchIndex(self.schema)
        self.index.create(overwrite=False)

    def register(self, intent, description):
        self.index.load([{
            "intent": intent,
            "description": description,
            "embedding": self.vectorizer.embed(description)
        }])

    def route(self, query):
        vec = self.vectorizer.embed(query)
        vq = VectorQuery(vector=vec, field="embedding", num_results=1)
        res = self.index.query(vq)
        return res[0]["intent"] if res else "unknown"
