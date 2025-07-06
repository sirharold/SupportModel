from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from openai import OpenAI
from utils.embedding_safe import EmbeddingClient
import json

def rerank_with_llm(question: str, docs: List[dict], openai_client: OpenAI, top_k: int = 10) -> List[dict]:
    if not docs:
        return []

    ranked_docs = []
    for doc in docs:
        score = 0.0
        try:
            content_preview = doc.get("content", "")[:2000]
            title = doc.get("title", "")

            prompt = (
                f"User Question: {question}\n\n"
                f"Document Title: {title}\n"
                f"Document Content: {content_preview}...\n\n"
                "Based on the document content, how relevant is this document to the user's question? "
                "Provide a relevance score from 0.0 (not relevant) to 1.0 (highly relevant)."
            )

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "provide_relevance_score",
                        "description": "Provides a relevance score for a document based on a user question.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "score": {
                                    "type": "number",
                                    "description": "The relevance score, from 0.0 to 1.0."
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "A brief justification for the score."
                                }
                            },
                            "required": ["score", "reasoning"]
                        }
                    }
                }
            ]

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a relevance scoring expert."},
                    {"role": "user", "content": prompt}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "provide_relevance_score"}},
                temperature=0.5
            )

            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                if tool_call.function.name == "provide_relevance_score":
                    tool_args = json.loads(tool_call.function.arguments)
                    score = tool_args.get("score", 0.0)
        except Exception as e:
            print(f"Error reranking doc {doc.get('link')} with LLM: {e}")

        doc["score"] = float(score)
        ranked_docs.append(doc)

    sorted_docs = sorted(ranked_docs, key=lambda d: d.get("score", 0.0), reverse=True)
    return sorted_docs[:top_k]


def rerank_documents(query: str, docs: List[dict], embedding_client: EmbeddingClient, top_k: int = 10) -> List[dict]:
    if not docs:
        return []

    query_vec = embedding_client.generate_query_embedding(query)
    if not query_vec:
        raise ValueError("Query embedding could not be generated.")

    doc_vecs = [embedding_client.generate_document_embedding(doc.get("content", "")) for doc in docs]
    
    valid_docs = [doc for doc, vec in zip(docs, doc_vecs) if vec]
    valid_vecs = [vec for vec in doc_vecs if vec]

    if not valid_vecs:
        return []

    scores = cosine_similarity([query_vec], valid_vecs)[0]

    for doc, score in zip(valid_docs, scores):
        doc["score"] = float(score)

    return sorted(valid_docs, key=lambda d: d["score"], reverse=True)[:top_k]

