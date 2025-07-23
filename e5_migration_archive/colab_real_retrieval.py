#!/usr/bin/env python3
"""
Script para Colab que usa embeddings reales para retrieval y métricas precisas
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import json
from datetime import datetime
import gc

class RealEmbeddingRetriever:
    """Retriever que usa embeddings reales para cálculo de coseno"""
    
    def __init__(self, parquet_file: str):
        """
        Inicializar con archivo Parquet que contiene embeddings reales
        
        Args:
            parquet_file: Archivo .parquet con columnas: document, embedding, link, title, etc.
        """
        print(f"🔄 Loading real embeddings from {parquet_file}...")
        self.df = pd.read_parquet(parquet_file)
        
        # Convertir embeddings a matriz numpy
        print("🔄 Converting embeddings to numpy matrix...")
        embeddings_list = self.df['embedding'].tolist()
        self.embeddings_matrix = np.array(embeddings_list)
        
        # Información del dataset
        self.num_docs = len(self.df)
        self.embedding_dim = self.embeddings_matrix.shape[1]
        
        print(f"✅ Loaded {self.num_docs:,} documents")
        print(f"📐 Embedding dimensions: {self.embedding_dim}")
        print(f"💾 Memory usage: {self.embeddings_matrix.nbytes / (1024**3):.2f} GB")
        
        # Preparar metadatos
        self.documents = self.df[['document', 'link', 'title', 'summary', 'content']].to_dict('records')
        
    def search_documents(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Buscar documentos más similares usando coseno real
        
        Args:
            query_embedding: Vector de embedding de la consulta (shape: embedding_dim)
            top_k: Número de documentos a retornar
            
        Returns:
            Lista de documentos ordenados por similaridad coseno
        """
        # Calcular similaridad coseno real
        query_embedding = query_embedding.reshape(1, -1)  # Shape: (1, embedding_dim)
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        
        # Obtener índices de documentos más similares
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Construir resultados con metadatos reales
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['cosine_similarity'] = float(similarities[idx])
            doc['rank'] = len(results) + 1
            results.append(doc)
        
        return results

def calculate_real_retrieval_metrics(
    question: str,
    query_embedding: np.ndarray,
    retriever: RealEmbeddingRetriever,
    ground_truth_links: List[str],
    top_k_values: List[int] = [1, 3, 5, 10]
) -> Dict:
    """
    Calcular métricas de retrieval reales usando coseno con embeddings auténticos
    
    Args:
        question: Pregunta original
        query_embedding: Embedding real de la pregunta
        retriever: Retriever con embeddings reales
        ground_truth_links: Enlaces de referencia (MS Learn)
        top_k_values: Valores de k para métricas
        
    Returns:
        Diccionario con métricas calculadas
    """
    # Buscar documentos con coseno real
    max_k = max(top_k_values) if top_k_values else 10
    retrieved_docs = retriever.search_documents(query_embedding, top_k=max_k)
    
    # Normalizar enlaces para comparación
    def normalize_link(link: str) -> str:
        if not link:
            return ""
        # Remover fragmentos y parámetros
        link = link.split('#')[0].split('?')[0]
        # Remover trailing slash
        return link.rstrip('/')
    
    # Normalizar ground truth
    gt_normalized = set(normalize_link(link) for link in ground_truth_links)
    
    # Calcular métricas para cada k
    metrics = {}
    for k in top_k_values:
        top_k_docs = retrieved_docs[:k]
        
        # Enlaces recuperados (normalizados)
        retrieved_links = set()
        for doc in top_k_docs:
            link = normalize_link(doc.get('link', ''))
            if link:
                retrieved_links.add(link)
        
        # Métricas
        relevant_retrieved = retrieved_links.intersection(gt_normalized)
        
        # Precision@k
        precision_k = len(relevant_retrieved) / k if k > 0 else 0.0
        
        # Recall@k  
        recall_k = len(relevant_retrieved) / len(gt_normalized) if gt_normalized else 0.0
        
        # F1@k
        f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0
        
        metrics[f'precision@{k}'] = precision_k
        metrics[f'recall@{k}'] = recall_k
        metrics[f'f1@{k}'] = f1_k
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, doc in enumerate(retrieved_docs, 1):
        link = normalize_link(doc.get('link', ''))
        if link in gt_normalized:
            mrr = 1.0 / rank
            break
    
    metrics['mrr'] = mrr
    metrics['ground_truth_count'] = len(gt_normalized)
    metrics['retrieved_count'] = len(retrieved_docs)
    
    return metrics

def run_real_evaluation(
    parquet_file: str,
    questions_data: List[Dict],
    embedding_model_name: str,
    num_questions: int = None
) -> Dict:
    """
    Ejecutar evaluación completa usando embeddings reales
    
    Args:
        parquet_file: Archivo con embeddings reales
        questions_data: Lista de preguntas con ground truth
        embedding_model_name: Nombre del modelo para generar query embeddings
        num_questions: Número de preguntas a evaluar (None = todas)
        
    Returns:
        Diccionario con resultados de evaluación
    """
    print(f"🚀 Starting Real Embedding Evaluation")
    print(f"📄 Document corpus: {parquet_file}")
    print(f"🔤 Embedding model: {embedding_model_name}")
    print("="*60)
    
    # Cargar retriever con embeddings reales
    retriever = RealEmbeddingRetriever(parquet_file)
    
    # Cargar modelo para query embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model_name)
    print(f"✅ Loaded query embedding model: {embedding_model_name}")
    
    # Seleccionar preguntas
    if num_questions:
        questions_to_eval = questions_data[:num_questions]
    else:
        questions_to_eval = questions_data
    
    print(f"📊 Evaluating {len(questions_to_eval)} questions...")
    
    # Evaluar cada pregunta
    all_metrics = []
    
    for i, qa_item in enumerate(questions_to_eval):
        question = qa_item.get('question', '')
        ms_links = qa_item.get('ms_links', [])
        
        if not question or not ms_links:
            continue
            
        # Generar embedding de consulta
        query_embedding = model.encode(question)
        
        # Calcular métricas reales
        metrics = calculate_real_retrieval_metrics(
            question=question,
            query_embedding=query_embedding,
            retriever=retriever,
            ground_truth_links=ms_links,
            top_k_values=[1, 3, 5, 10]
        )
        
        metrics['question_index'] = i
        metrics['question'] = question
        all_metrics.append(metrics)
        
        if (i + 1) % 50 == 0:
            print(f"📈 Processed {i + 1}/{len(questions_to_eval)} questions")
    
    # Calcular métricas promedio
    if all_metrics:
        avg_metrics = {}
        for key in ['precision@1', 'precision@3', 'precision@5', 'precision@10',
                   'recall@1', 'recall@3', 'recall@5', 'recall@10',
                   'f1@1', 'f1@3', 'f1@5', 'f1@10', 'mrr']:
            values = [m[key] for m in all_metrics if key in m]
            avg_metrics[f'avg_{key}'] = np.mean(values) if values else 0.0
    
    results = {
        'embedding_model': embedding_model_name,
        'document_corpus': parquet_file,
        'num_questions_evaluated': len(all_metrics),
        'evaluation_timestamp': datetime.now().isoformat(),
        'average_metrics': avg_metrics,
        'individual_metrics': all_metrics,
        'corpus_stats': {
            'num_documents': retriever.num_docs,
            'embedding_dimensions': retriever.embedding_dim
        }
    }
    
    # Limpiar memoria
    del retriever
    gc.collect()
    
    return results

def main():
    """Función principal para prueba local"""
    print("🧪 Real Embedding Retrieval - Local Test")
    print("This script is designed to run in Google Colab")
    print("Upload the .parquet files and questions data to Google Drive")
    print("Then use the functions above in your Colab notebook")

if __name__ == "__main__":
    main()