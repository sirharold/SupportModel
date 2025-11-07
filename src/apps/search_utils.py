"""
Shared utilities for search and reranking analysis
Used by both single question and batch analysis pages
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from urllib.parse import urlparse, urlunparse
from sentence_transformers import CrossEncoder

try:
    from ranx import Qrels, Run, evaluate
    RANX_AVAILABLE = True
except ImportError:
    RANX_AVAILABLE = False


# Azure terminology dictionary for query expansion
AZURE_TERMINOLOGY = {
    # Common abbreviations → Full terms and synonyms
    'AAD': ['Azure Active Directory', 'Microsoft Entra ID', 'Entra ID'],
    'VM': ['Virtual Machine', 'virtual machines', 'VMs'],
    'CMK': ['Customer-Managed Keys', 'customer managed keys', 'customer key'],
    'PMK': ['Platform-Managed Keys', 'platform managed keys', 'platform key'],
    'RBAC': ['Role-Based Access Control', 'role based access control'],
    'AKS': ['Azure Kubernetes Service', 'Kubernetes', 'K8s'],
    'ACR': ['Azure Container Registry', 'container registry'],
    'ACI': ['Azure Container Instances', 'container instances'],
    'ADF': ['Azure Data Factory', 'Data Factory'],
    'ADLS': ['Azure Data Lake Storage', 'Data Lake Storage'],
    'ARM': ['Azure Resource Manager', 'Resource Manager'],
    'NSG': ['Network Security Group', 'network security groups'],
    'VNet': ['Virtual Network', 'virtual networks'],
    'NIC': ['Network Interface Card', 'network interface'],
    'LB': ['Load Balancer', 'load balancing'],
    'ASG': ['Application Security Group', 'application security groups'],
    'KV': ['Key Vault', 'Azure Key Vault'],
    'ACL': ['Access Control List', 'access control lists'],
    'SAS': ['Shared Access Signature', 'shared access signatures'],
    'MSI': ['Managed Service Identity', 'Managed Identity'],
    'WAF': ['Web Application Firewall', 'web application firewall'],

    # Old terminology → New terminology (Azure rebranding)
    'Azure Active Directory': ['Microsoft Entra ID', 'Entra ID', 'AAD'],
    'Active Directory': ['Microsoft Entra ID', 'Entra ID'],

    # Common concept expansions
    'disk encryption': ['server-side encryption', 'encryption at rest', 'data encryption', 'Azure Disk Encryption'],
    'blob storage': ['Azure Storage', 'object storage', 'storage account', 'Azure Blob Storage'],
    'authentication': ['auth', 'identity', 'sign-in', 'login', 'Microsoft Entra ID'],
    'authorization': ['access control', 'permissions', 'RBAC', 'role assignment'],
    'encrypt': ['encryption', 'encrypted', 'encrypting', 'secure', 'security'],
    'configure': ['configuration', 'setup', 'set up', 'enable'],
    'deploy': ['deployment', 'provision', 'create', 'setup'],
    'monitor': ['monitoring', 'observability', 'diagnostics', 'logs'],
    'scale': ['scaling', 'autoscale', 'auto-scale', 'scalability'],
    'backup': ['restore', 'recovery', 'backup and restore'],
    'migrate': ['migration', 'move', 'transfer'],
    'connect': ['connection', 'connectivity', 'network', 'integrate'],

    # Service-specific terminology
    'SQL Database': ['Azure SQL', 'SQL DB', 'Azure SQL Database'],
    'Cosmos DB': ['CosmosDB', 'Azure Cosmos DB', 'Cosmos Database'],
    'App Service': ['Web App', 'Web Apps', 'Azure App Service'],
    'Storage Account': ['Azure Storage', 'storage accounts'],
    'Function App': ['Azure Functions', 'Functions', 'Serverless'],
    'Container': ['Docker', 'container instances', 'containerized'],
}


def expand_query(question: str, max_expansions: int = 2,
                 terminology_dict: Dict = None, debug: bool = False) -> str:
    """
    Expand query with Azure-specific terminology and synonyms

    Args:
        question: Original user question
        max_expansions: Maximum number of synonyms to add per matched term
        terminology_dict: Custom terminology dictionary (uses AZURE_TERMINOLOGY if None)
        debug: Print expansion details

    Returns:
        Expanded query string
    """
    if terminology_dict is None:
        terminology_dict = AZURE_TERMINOLOGY

    expanded_terms = []
    question_lower = question.lower()
    matched_terms = []

    # Find matching terms in question
    for term, synonyms in terminology_dict.items():
        if term.lower() in question_lower:
            # Add top N synonyms that aren't already in the question
            added = 0
            for synonym in synonyms:
                if synonym.lower() not in question_lower and added < max_expansions:
                    expanded_terms.append(synonym)
                    added += 1
            if added > 0:
                matched_terms.append(f"{term} → {synonyms[:max_expansions]}")

    # Combine original + expansions
    if expanded_terms:
        expanded_query = f"{question} {' '.join(expanded_terms)}"
    else:
        expanded_query = question

    if debug and matched_terms:
        import streamlit as st
        st.write(f"**Query Expansion:** Found {len(matched_terms)} matches")
        for match in matched_terms[:5]:  # Show first 5
            st.write(f"  - {match}")

    return expanded_query


def normalize_url(url: str) -> str:
    """Normalize URL by removing query parameters and fragments"""
    if not url or not url.strip():
        return ""
    try:
        parsed = urlparse(url.strip())
        normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
        return normalized
    except:
        return url.strip()


@st.cache_resource
def load_crossencoder(model_name: str = 'cross-encoder/ms-marco-electra-base'):
    """
    Load CrossEncoder model once

    Available models:
    - cross-encoder/ms-marco-electra-base (RECOMMENDED - best accuracy)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (good balance)
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fastest, lower accuracy)
    """
    return CrossEncoder(model_name)


def get_optimal_batch_size(documents: List[Dict], max_content_length: int = 1200) -> int:
    """
    Calculate optimal batch size based on document length
    Same logic as Colab for consistency
    """
    if not documents:
        return 32

    avg_length = sum(len(doc.get('content', '')[:max_content_length]) for doc in documents) / len(documents)

    if avg_length > 1000:
        return 16
    elif avg_length > 500:
        return 32
    else:
        return 64


def apply_crossencoder_reranking(question: str, documents: List[Dict],
                                 cross_encoder: CrossEncoder) -> List[Dict]:
    """
    Apply CrossEncoder reranking to documents
    IDENTICAL to Colab implementation for consistency and determinism

    Mejoras:
    - Usa título + contenido (no solo contenido)
    - Batch size adaptativo según longitud de documentos
    - Ordenamiento determinístico con tie-breaking
    """
    if not documents:
        return []

    try:
        # MEJORA 1: Prepare query-document pairs WITH TITLE + CONTENT
        pairs = []
        for doc in documents:
            title = doc.get('title', '').strip()
            content = doc.get('content', '').strip()

            # Combinar título y contenido limitado
            if title:
                # Título + primeros 1200 caracteres de contenido
                combined_text = f"{title}: {content[:1200]}"
            else:
                # Solo contenido limitado si no hay título
                combined_text = content[:1500]

            pairs.append([question, combined_text])

        # MEJORA 2: Batch size adaptativo según longitud de documentos
        optimal_batch_size = get_optimal_batch_size(documents, max_content_length=1200)

        # Get CrossEncoder scores con batch size optimizado
        scores = cross_encoder.predict(pairs, batch_size=optimal_batch_size)

        # Apply Min-Max normalization to convert logits to [0, 1] range
        scores = np.array(scores)
        if len(scores) > 1 and scores.max() != scores.min():
            # Standard Min-Max normalization
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            # Fallback for edge cases (all scores identical)
            normalized_scores = np.full_like(scores, 0.5)

        # Add scores to documents
        for i, doc in enumerate(documents):
            doc['crossencoder_score'] = float(normalized_scores[i])

        # DETERMINISTIC: Sort by CrossEncoder score desc, then by link asc (tie-breaking)
        reranked_docs = sorted(
            documents,
            key=lambda x: (-x['crossencoder_score'], x.get('link', ''))
        )

        # Update ranks
        for i, doc in enumerate(reranked_docs, 1):
            doc['rank'] = i

        return reranked_docs

    except Exception as e:
        st.warning(f"⚠️ CrossEncoder reranking error: {e}")
        return documents


def calculate_retrieval_metrics(ground_truth_links: List[str], retrieved_docs: List[Dict],
                                k_values: List[int], score_key: str = 'cosine_similarity',
                                debug: bool = False) -> Dict:
    """Calculate precision, recall, F1, MAP, MRR, NDCG for different k values

    This implementation ranks documents by scores before calculating metrics,
    making it equivalent to ranx behavior.

    Args:
        ground_truth_links: List of relevant document URLs
        retrieved_docs: List of retrieved documents with scores
        k_values: List of k values to calculate metrics for
        score_key: 'cosine_similarity' (before CE) or 'crossencoder_score' (after CE)
        debug: If True, print debug information
    """

    # Normalize ground truth links
    gt_normalized = {normalize_url(link) for link in ground_truth_links if link}

    if not gt_normalized:
        return {}

    # ✅ MEJORA: Rank documents by scores FIRST (like ranx does)
    # This ensures metrics are calculated on the correct ranking
    sorted_docs = sorted(
        retrieved_docs,
        key=lambda x: (-x.get(score_key, 0.0), x.get('link', ''))  # Tie-breaking by URL
    )

    # ✅ CRITICAL FIX: De-duplicate by normalized URL, keeping first (highest score)
    # This prevents duplicate URLs from inflating metrics
    seen_urls = set()
    dedup_docs = []
    for doc in sorted_docs:
        norm_url = normalize_url(doc.get('link', ''))
        if norm_url and norm_url not in seen_urls:
            seen_urls.add(norm_url)
            dedup_docs.append(doc)

    if debug:
        import streamlit as st
        st.write(f"**DEBUG Manual - score_key:** {score_key}")
        st.write(f"**DEBUG Manual - Original docs:** {len(sorted_docs)}, After dedup: {len(dedup_docs)}")

        # Warning if very few unique docs
        if len(dedup_docs) < 10:
            st.warning(f"⚠️ Solo {len(dedup_docs)} documentos únicos después de deduplicación. Esto puede afectar métricas para k > {len(dedup_docs)}")

        st.write(f"**DEBUG Manual - Top 5 docs after dedup:**")
        for i, doc in enumerate(dedup_docs[:5], 1):
            link = normalize_url(doc.get('link', ''))
            score = doc.get(score_key, 0.0)
            is_rel = link in gt_normalized
            st.write(f"  {i}. {link[:50]}... score={score:.4f} relevant={is_rel}")

        # Show duplicate info
        num_duplicates = len(sorted_docs) - len(dedup_docs)
        if num_duplicates > 0:
            st.info(f"ℹ️ Se encontraron {num_duplicates} URLs duplicadas (después de normalización)")

    # Extract retrieved links from deduplicated documents
    retrieved_links = [normalize_url(doc.get('link', '')) for doc in dedup_docs]

    # Verificar si tenemos suficientes documentos únicos
    num_unique_docs = len(retrieved_links)
    if debug and k_values and num_unique_docs < max(k_values):
        import streamlit as st
        st.warning(f"⚠️ **ADVERTENCIA**: Solo hay {num_unique_docs} documentos únicos, pero el k máximo solicitado es {max(k_values)}. "
                   f"Las métricas para k > {num_unique_docs} se calcularán solo sobre {num_unique_docs} documentos.")

    metrics = {}

    for k in k_values:
        top_k_links = set(retrieved_links[:k])
        actual_k = len(top_k_links)  # Número real de documentos únicos considerados

        # True positives
        tp = len(top_k_links & gt_normalized)

        # Precision@k
        # Note: If we have fewer than k unique documents, precision is calculated as tp/k
        # This means precision will be lower, which is correct: we asked for k docs but couldn't provide k unique ones
        precision = tp / k if k > 0 else 0.0

        # Recall@k
        recall = tp / len(gt_normalized) if len(gt_normalized) > 0 else 0.0

        # F1@k
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Debug info for this k
        if debug and actual_k < k:
            import streamlit as st
            st.write(f"  ⚠️ k={k}: solo {actual_k} docs únicos disponibles (tp={tp}, P={precision:.4f}, R={recall:.4f})")

        metrics[f'precision@{k}'] = precision
        metrics[f'recall@{k}'] = recall
        metrics[f'f1@{k}'] = f1

        # NDCG@k
        dcg = 0.0
        idcg = 0.0
        for i, link in enumerate(retrieved_links[:k], 1):
            rel = 1.0 if normalize_url(link) in gt_normalized else 0.0
            dcg += rel / np.log2(i + 1)

        # Ideal DCG (all relevant docs at top)
        for i in range(1, min(k, len(gt_normalized)) + 1):
            idcg += 1.0 / np.log2(i + 1)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f'ndcg@{k}'] = ndcg

        # MAP@k
        sum_precisions = 0.0
        relevant_count = 0
        for i, link in enumerate(retrieved_links[:k], 1):
            if normalize_url(link) in gt_normalized:
                relevant_count += 1
                sum_precisions += relevant_count / i

        map_k = sum_precisions / len(gt_normalized) if len(gt_normalized) > 0 else 0.0
        metrics[f'map@{k}'] = map_k

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, link in enumerate(retrieved_links, 1):
        if normalize_url(link) in gt_normalized:
            mrr = 1.0 / i
            break
    metrics['mrr'] = mrr

    return metrics


def calculate_ranx_metrics(ground_truth_links: List[str], retrieved_docs: List[Dict],
                           k_values: List[int], query_id: str = "q1", use_crossencoder: bool = False,
                           debug: bool = False) -> Dict:
    """Calculate metrics using ranx library

    Args:
        ground_truth_links: List of relevant document URLs
        retrieved_docs: List of retrieved documents with scores
        k_values: List of k values to calculate metrics for
        query_id: Query identifier for ranx
        use_crossencoder: If True, use crossencoder_score; if False, use cosine_similarity
        debug: If True, print debug information
    """

    if not RANX_AVAILABLE:
        return {}

    # Normalize ground truth links
    gt_normalized = {normalize_url(link) for link in ground_truth_links if link}

    if not gt_normalized:
        return {}

    # Create qrels (ground truth) - binary relevance
    qrels_dict = {query_id: {}}
    for link in gt_normalized:
        qrels_dict[query_id][link] = 1

    qrels = Qrels(qrels_dict)

    # Create run (retrieved results with scores)
    run_dict = {query_id: {}}
    score_key = 'crossencoder_score' if use_crossencoder else 'cosine_similarity'

    # ✅ CRITICAL FIX: Keep highest score for duplicate normalized URLs
    duplicate_count = 0
    for doc in retrieved_docs:
        link = normalize_url(doc.get('link', ''))
        if not link:
            continue

        # Use the appropriate score based on the flag
        if use_crossencoder:
            score = doc.get('crossencoder_score', 0.0)
        else:
            score = doc.get('cosine_similarity', 0.0)

        # Keep the highest score for duplicate URLs
        if link in run_dict[query_id]:
            duplicate_count += 1
            run_dict[query_id][link] = max(run_dict[query_id][link], float(score))
        else:
            run_dict[query_id][link] = float(score)

    if debug:
        import streamlit as st
        st.write(f"**DEBUG Ranx - use_crossencoder:** {use_crossencoder}, score_key: {score_key}")
        st.write(f"**DEBUG Ranx - Original docs:** {len(retrieved_docs)}, Unique URLs: {len(run_dict[query_id])}, Duplicates: {duplicate_count}")
        st.write(f"**DEBUG Ranx - Top 5 docs by score:**")
        # Sort by score to show top 5
        sorted_run = sorted(run_dict[query_id].items(), key=lambda x: -x[1])
        for i, (link, score) in enumerate(sorted_run[:5], 1):
            is_rel = link in gt_normalized
            st.write(f"  {i}. {link[:50]}... score={score:.4f} relevant={is_rel}")

    run = Run(run_dict)

    # Define metrics to evaluate
    metrics_to_eval = ['mrr']
    for k in k_values:
        metrics_to_eval.extend([f'precision@{k}', f'recall@{k}', f'f1@{k}',
                                f'ndcg@{k}', f'map@{k}'])

    # Evaluate
    try:
        results = evaluate(qrels, run, metrics_to_eval)
        # Convert all values to native Python float for compatibility
        converted_results = {}
        for key, value in results.items():
            if isinstance(value, (np.ndarray, np.generic)):
                converted_results[key] = float(value)
            else:
                converted_results[key] = float(value) if value is not None else 0.0
        return converted_results
    except Exception as e:
        st.warning(f"⚠️ Error calculando métricas con ranx: {e}")
        import traceback
        st.code(traceback.format_exc())
        return {}


def search_documents(client, collection_name: str, query_embedding: np.ndarray,
                     top_k: int = 15) -> List[Dict]:
    """Search documents using cosine similarity"""
    try:
        collection = client.get_collection(collection_name)

        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        documents = []
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                doc = {
                    'rank': i + 1,
                    'link': results['metadatas'][0][i].get('link', ''),
                    'title': results['metadatas'][0][i].get('title', ''),
                    'content': results['documents'][0][i] if results['documents'] else '',
                    'cosine_similarity': 1 - results['distances'][0][i],  # ChromaDB returns distance
                    'original_rank': i + 1
                }
                documents.append(doc)

        return documents

    except Exception as e:
        st.error(f"Error en búsqueda: {e}")
        return []


def create_metrics_comparison_plot(metrics_manual_before: Dict, metrics_manual_after: Dict,
                                   metrics_ranx_before: Dict, metrics_ranx_after: Dict,
                                   k_values: List[int]):
    """Create line plots comparing metrics: Manual vs Ranx, Before vs After"""

    # Metrics to plot
    metric_names = ['precision', 'recall', 'f1', 'ndcg', 'map']

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Define colors and markers for 4 lines
    styles = [
        {'label': 'Manual Antes', 'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'alpha': 0.8},
        {'label': 'Manual Después', 'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'alpha': 0.8},
        {'label': 'Ranx Antes', 'color': '#2ca02c', 'marker': '^', 'linestyle': '--', 'alpha': 0.7},
        {'label': 'Ranx Después', 'color': '#d62728', 'marker': 'v', 'linestyle': '--', 'alpha': 0.7},
    ]

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]

        # Extract values for this metric
        manual_before_values = [float(metrics_manual_before.get(f'{metric}@{k}', 0)) for k in k_values]
        manual_after_values = [float(metrics_manual_after.get(f'{metric}@{k}', 0)) for k in k_values]
        ranx_before_values = [float(metrics_ranx_before.get(f'{metric}@{k}', 0)) for k in k_values]
        ranx_after_values = [float(metrics_ranx_after.get(f'{metric}@{k}', 0)) for k in k_values]

        # Plot 4 lines
        ax.plot(k_values, manual_before_values, marker=styles[0]['marker'], label=styles[0]['label'],
                color=styles[0]['color'], linestyle=styles[0]['linestyle'], linewidth=2, markersize=7, alpha=styles[0]['alpha'])
        ax.plot(k_values, manual_after_values, marker=styles[1]['marker'], label=styles[1]['label'],
                color=styles[1]['color'], linestyle=styles[1]['linestyle'], linewidth=2, markersize=7, alpha=styles[1]['alpha'])
        ax.plot(k_values, ranx_before_values, marker=styles[2]['marker'], label=styles[2]['label'],
                color=styles[2]['color'], linestyle=styles[2]['linestyle'], linewidth=2, markersize=7, alpha=styles[2]['alpha'])
        ax.plot(k_values, ranx_after_values, marker=styles[3]['marker'], label=styles[3]['label'],
                color=styles[3]['color'], linestyle=styles[3]['linestyle'], linewidth=2, markersize=7, alpha=styles[3]['alpha'])

        # Styling
        ax.set_xlabel('Top-K', fontsize=11)
        ax.set_ylabel('Valor', fontsize=11)
        ax.set_title(f'{metric.upper()}@k', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values[::2] if len(k_values) > 8 else k_values)  # Show every other tick if too many

    # MRR (single value, shown as bar with 4 bars)
    ax = axes[5]
    manual_mrr_before = float(metrics_manual_before.get('mrr', 0))
    manual_mrr_after = float(metrics_manual_after.get('mrr', 0))
    ranx_mrr_before = float(metrics_ranx_before.get('mrr', 0))
    ranx_mrr_after = float(metrics_ranx_after.get('mrr', 0))

    x_pos = [0, 1, 2.5, 3.5]
    values = [manual_mrr_before, manual_mrr_after, ranx_mrr_before, ranx_mrr_after]
    colors = [styles[0]['color'], styles[1]['color'], styles[2]['color'], styles[3]['color']]
    labels = ['Manual\nAntes', 'Manual\nDespués', 'Ranx\nAntes', 'Ranx\nDespués']

    bars = ax.bar(x_pos, values, color=colors, alpha=0.7, width=0.8)
    ax.set_ylabel('Valor', fontsize=11)
    ax.set_title('MRR', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(values):
        ax.text(x_pos[i], v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


def get_question_embedding(client, model_questions_collection, question_url: str,
                           questions_collection_name: str) -> np.ndarray:
    """
    Get question embedding from model-specific collection by URL match

    Returns:
        numpy array with embedding or None if not found
    """
    if not question_url:
        st.error("❌ La pregunta no tiene URL en metadata")
        return None

    # Get all questions and find by URL match
    all_model_questions = model_questions_collection.get(
        include=['embeddings', 'documents', 'metadatas']
    )

    # Find the question by URL match
    for i, meta in enumerate(all_model_questions['metadatas']):
        if meta.get('url', '') == question_url:
            return np.array(all_model_questions['embeddings'][i])

    return None
