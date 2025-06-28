import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create minimal fake weaviate module for importing weaviate_utils
fake_weaviate = types.ModuleType("weaviate")
fake_weaviate.connect_to_weaviate_cloud = lambda **kwargs: None
fake_auth = types.ModuleType("auth")
fake_auth.AuthApiKey = object
fake_weaviate.auth = fake_auth
fake_collections = types.ModuleType("collections")
fake_weaviate.collections = fake_collections
fake_classes = types.ModuleType("classes")
fake_collections.classes = fake_classes
fake_filters_mod = types.ModuleType("filters")
fake_classes.filters = fake_filters_mod

class DummyFilter:
    def __init__(self, prop=None, value=None, children=None, op=None):
        self.prop = prop
        self.value = value
        self.children = children or []
        self.op = op

    @classmethod
    def by_property(cls, prop):
        return cls(prop=prop)

    def equal(self, value):
        self.value = value
        return self

    def __and__(self, other):
        return DummyFilter(children=[self, other], op="and")

    def __or__(self, other):
        return DummyFilter(children=[self, other], op="or")

fake_filters_mod.Filter = DummyFilter

sys.modules.setdefault("weaviate", fake_weaviate)
sys.modules.setdefault("weaviate.auth", fake_auth)
sys.modules.setdefault("weaviate.collections", fake_collections)
sys.modules.setdefault("weaviate.collections.classes", fake_classes)
sys.modules.setdefault("weaviate.collections.classes.filters", fake_filters_mod)

# Stub sklearn for importing reranker
fake_sklearn = types.ModuleType("sklearn")
fake_metrics = types.ModuleType("metrics")
fake_pairwise = types.ModuleType("pairwise")
def _cosine_similarity(x, y):
    return [[1.0 for _ in range(len(y))] for __ in range(len(x))]
fake_pairwise.cosine_similarity = _cosine_similarity
fake_metrics.pairwise = fake_pairwise
fake_sklearn.metrics = fake_metrics
sys.modules.setdefault("sklearn", fake_sklearn)
sys.modules.setdefault("sklearn.metrics", fake_metrics)
sys.modules.setdefault("sklearn.metrics.pairwise", fake_pairwise)

# Stub openai for importing embedding module
fake_openai = types.ModuleType("openai")
class DummyOpenAI:
    def __init__(self, api_key=None):
        pass
    class embeddings:
        @staticmethod
        def create(model=None, input=None):
            class Resp:
                data = [types.SimpleNamespace(embedding=[0.0, 0.0])]
            return Resp()
fake_openai.OpenAI = DummyOpenAI
sys.modules.setdefault("openai", fake_openai)

from utils.weaviate_utils_improved import WeaviateClientWrapper
from utils.qa_pipeline import answer_question
from unittest.mock import patch

class FakeObject:
    def __init__(self, properties):
        self.properties = properties
        self.metadata = types.SimpleNamespace(distance=0.0)

class FakeQueryResult:
    def __init__(self, objects):
        self.objects = [FakeObject(o) for o in objects]

class FakeQuery:
    def __init__(self, docs):
        self.docs = docs

    def near_vector(self, near_vector=None, limit=10, **kwargs):
        return FakeQueryResult(self.docs[:limit])

    def fetch_objects(self, where=None, filters=None, limit=10):
        f = where if where is not None else filters

        def match(doc, flt):
            if flt is None:
                return True
            if flt.children:
                if flt.op == "or":
                    return any(match(doc, c) for c in flt.children)
                else:
                    return all(match(doc, c) for c in flt.children)
            if flt.prop:
                return doc.get(flt.prop) == flt.value
            return True

        filtered = [d for d in self.docs if match(d, f)]
        return FakeQueryResult(filtered[:limit])

class FakeDocCollection:
    def __init__(self, docs):
        self.query = FakeQuery(docs)

class FakeQuestionCollection:
    def __init__(self, questions):
        self.questions = questions
        self.query = self

    def near_vector(self, near_vector=None, limit=10):
        return FakeQueryResult(self.questions[:limit])

class FakeCollections(dict):
    def get(self, name):
        return self[name]

class FakeClient:
    def __init__(self, docs, questions):
        self.collections = FakeCollections({
            "Documentation": FakeDocCollection(docs),
            "Questions": FakeQuestionCollection(questions),
        })

class FakeEmbeddingClient:
    def generate_embedding(self, text):
        return [0.1, 0.1]


def test_answer_question_uses_chunk1_docs():
    docs = [
        {"link": "l1", "content": "c1", "title": "t1", "chunk_index": 1},
        {"link": "l1", "content": "c1b", "title": "t1b", "chunk_index": 2},
        {"link": "l2", "content": "c2", "title": "t2", "chunk_index": 1},
        {"link": "l3", "content": "c3", "title": "t3", "chunk_index": 1},
    ]
    questions = [
        {"accepted_answer": "see l1"},
    ]

    wrapper = WeaviateClientWrapper(FakeClient(docs, questions))
    embedding_client = FakeEmbeddingClient()

    with patch("utils.qa_pipeline.rerank_documents", lambda q, docs, ec, top_k=None: docs) as _:
        result_docs, _ = answer_question("q", wrapper, embedding_client, top_k=3)

    assert result_docs
    assert all(d.get("chunk_index") == 1 for d in result_docs)
    # ensure unique links
    links = [d["link"] for d in result_docs]
    assert len(links) == len(set(links))
