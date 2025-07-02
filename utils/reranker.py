from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from openai import OpenAI
from utils.embedding import EmbeddingClient
import json

def rerank_with_llm(question: str, docs: List[dict], openai_client: OpenAI, top_k: int = 10) -> List[dict]:
    print(f"[DEBUG] ENTERING rerank_with_llm function.")
    print(f"[DEBUG] rerank_with_llm: Received {len(docs)} documents for reranking.")
    if not docs:
        print("[DEBUG] rerank_with_llm: No documents to rerank, returning empty list.")
        return []

    ranked_docs = []
    for i, doc in enumerate(docs):
        score = 0.0  # Default score
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
                temperature=0.5 # Increased temperature to allow more varied scores
            )

            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                print(f"[DEBUG] LLM Tool Call Name: {tool_call.function.name}")
                print(f"[DEBUG] LLM Tool Call Arguments: {tool_call.function.arguments}")
                if tool_call.function.name == "provide_relevance_score":
                    tool_args = json.loads(tool_call.function.arguments)
                    score = tool_args.get("score", 0.0)
                    print(f"[DEBUG] Extracted score: {score}")
                else:
                    print(f"[DEBUG] LLM called unexpected tool: {tool_call.function.name}")
            else:
                print(f"[DEBUG] LLM did not make a tool call. Message content: {message.content}")

        except Exception as e:
            print(f"[DEBUG] Error reranking doc {doc.get('link')} with LLM: {e}")
            # Score remains 0.0 if an error occurs

        doc["score"] = float(score)
        ranked_docs.append(doc)

    # Sort by the new LLM-generated score
    sorted_docs = sorted(ranked_docs, key=lambda d: d.get("score", 0.0), reverse=True)
    print(f"[DEBUG] rerank_with_llm: Final sorted scores: {[d.get('score', 0.0) for d in sorted_docs[:top_k]]}")
    print(f"[DEBUG] rerank_with_llm: Returning {len(sorted_docs[:top_k])} documents.")
    return sorted_docs[:top_k]


def rerank_documents(query: str, docs: List[dict], embedding_client: EmbeddingClient, top_k: int = 10) -> List[dict]:
    print(f"[DEBUG] rerank_documents: Received {len(docs)} documents for standard reranking.")
    try:
        query_vec = embedding_client.generate_embedding(query)
        if not query_vec:
            print("[DEBUG] rerank_documents: Query embedding could not be generated.")
            raise ValueError("Query embedding could not be generated.")

        valid_docs = []
        valid_vecs = []

        for doc in docs:
            content = doc.get("content", "")
            vec = embedding_client.generate_embedding(content)
            if vec:  # solo incluimos si el embedding es válido
                valid_docs.append(doc)
                valid_vecs.append(vec)
        
        print(f"[DEBUG] rerank_documents: Found {len(valid_docs)} documents with valid embeddings.")

        if not valid_vecs:
            print("⚠️ No valid document embeddings generated for standard reranking.")
            return []

        scores = cosine_similarity([query_vec], valid_vecs)[0]

        for doc, score in zip(valid_docs, scores):
            doc["score"] = float(score)

        ranked_docs = sorted(valid_docs, key=lambda d: d["score"], reverse=True)
        print(f"[DEBUG] rerank_documents: Final sorted scores (standard): {[d.get('score', 0.0) for d in ranked_docs[:top_k]]}")
        return ranked_docs[:top_k]

    except Exception as e:
        print("❌ Error in rerank_documents:", e)
        return []
