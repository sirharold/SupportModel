import json
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
from utils.qa_pipeline import answer_question
from utils.metrics import compute_ndcg, compute_mrr, compute_precision_recall_f1
import re
from config import DEBUG_MODE

def debug_print(message: str, force: bool = False):
    """Print debug message only if DEBUG_MODE is enabled or force is True."""
    if DEBUG_MODE or force:
        print(message)

class EvaluationSystem:
    """Sistema de evaluación continua para el QA system."""
    
    def __init__(self, weaviate_wrapper, embedding_client):
        self.weaviate_wrapper = weaviate_wrapper
        self.embedding_client = embedding_client
        self.evaluation_history = []
        
    def create_evaluation_dataset(self, output_path: str = "data/evaluation_set.json"):
        """Crea un conjunto de evaluación curado manualmente."""
        
        # Preguntas de evaluación con respuestas esperadas
        evaluation_questions = [
            {
                "question": "How to configure Managed Identity in Azure Functions?",
                "expected_links": [
                    "https://learn.microsoft.com/en-us/azure/azure-functions/functions-identity",
                    "https://learn.microsoft.com/en-us/azure/app-service/overview-managed-identity",
                    "https://learn.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/overview"
                ],
                "category": "security",
                "difficulty": "intermediate"
            },
            {
                "question": "Azure Storage account best practices and security",
                "expected_links": [
                    "https://learn.microsoft.com/en-us/azure/storage/common/storage-account-overview",
                    "https://learn.microsoft.com/en-us/azure/storage/common/storage-security-guide",
                    "https://learn.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage"
                ],
                "category": "storage",
                "difficulty": "beginner"
            },
            {
                "question": "How to connect Azure Functions to Key Vault without connection strings?",
                "expected_links": [
                    "https://learn.microsoft.com/en-us/azure/key-vault/general/overview",
                    "https://learn.microsoft.com/en-us/azure/azure-functions/functions-bindings-key-vault",
                    "https://learn.microsoft.com/en-us/azure/azure-functions/functions-identity"
                ],
                "category": "security",
                "difficulty": "advanced"
            },
            {
                "question": "Azure Service Bus messaging patterns and configuration",
                "expected_links": [
                    "https://learn.microsoft.com/en-us/azure/service-bus-messaging/service-bus-messaging-overview",
                    "https://learn.microsoft.com/en-us/azure/service-bus-messaging/service-bus-queues-topics-subscriptions",
                    "https://learn.microsoft.com/en-us/azure/service-bus-messaging/service-bus-dotnet-get-started-with-queues"
                ],
                "category": "messaging",
                "difficulty": "intermediate"
            }
        ]
        
        # Guardar conjunto de evaluación
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_questions, f, indent=2, ensure_ascii=False)
        
        debug_print(f"[INFO] ✅ Evaluation dataset created: {output_path}")
        return evaluation_questions
    
    def run_evaluation(self, evaluation_file: str = "data/evaluation_set.json") -> Dict:
        """Ejecuta evaluación completa del sistema."""
        
        # Cargar conjunto de evaluación
        try:
            with open(evaluation_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
        except FileNotFoundError:
            debug_print("[INFO] Creating evaluation dataset...", force=True)
            eval_data = self.create_evaluation_dataset(evaluation_file)
        
        results = []
        
        for item in eval_data:
            question = item["question"]
            expected_links = item["expected_links"]
            category = item.get("category", "general")
            difficulty = item.get("difficulty", "intermediate")
            
            debug_print(f"[INFO] Evaluating: {question[:50]}...", force=True)
            
            # Ejecutar búsqueda
            retrieved_docs, debug_info = answer_question(
                question, self.weaviate_wrapper, self.embedding_client, top_k=10
            )
            
            # Calcular métricas
            metrics = self.calculate_metrics(retrieved_docs, expected_links)
            
            # Almacenar resultado
            result = {
                "question": question,
                "category": category,
                "difficulty": difficulty,
                "expected_links": expected_links,
                "retrieved_count": len(retrieved_docs),
                "retrieved_links": [doc.get("link") for doc in retrieved_docs],
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
        
        # Generar reporte agregado
        report = self.generate_evaluation_report(results)
        
        # Guardar historial
        self.evaluation_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": report
        })
        
        return report
    
    def calculate_metrics(self, retrieved_docs: List[Dict], expected_links: List[str]) -> Dict:
        """Calcula métricas de evaluación para una consulta."""
        
        # Métricas básicas
        precision, recall, f1 = compute_precision_recall_f1(retrieved_docs, expected_links, k=10)
        ndcg = compute_ndcg(retrieved_docs, expected_links, k=10)
        mrr = compute_mrr(retrieved_docs, expected_links, k=10)
        
        # Métricas adicionales
        retrieved_links = [doc.get("link") for doc in retrieved_docs if doc.get("link")]
        expected_set = set(expected_links)
        retrieved_set = set(retrieved_links)
        
        # Hits@K para diferentes valores de K
        hits_at_k = {}
        for k in [1, 3, 5, 10]:
            top_k_links = set(retrieved_links[:k])
            hits_at_k[f"hits@{k}"] = len(expected_set & top_k_links) > 0
        
        # Score promedio de documentos relevantes
        relevant_scores = []
        for doc in retrieved_docs:
            if doc.get("link") in expected_set:
                relevant_scores.append(doc.get("score", 0))
        
        avg_relevant_score = np.mean(relevant_scores) if relevant_scores else 0
        
        return {
            "precision@10": precision,
            "recall@10": recall,
            "f1@10": f1,
            "ndcg@10": ndcg,
            "mrr@10": mrr,
            "avg_relevant_score": avg_relevant_score,
            **hits_at_k,
            "total_expected": len(expected_links),
            "total_retrieved": len(retrieved_links),
            "intersection_size": len(expected_set & retrieved_set)
        }
    
    def generate_evaluation_report(self, results: List[Dict]) -> Dict:
        """Genera reporte agregado de evaluación."""
        
        # Métricas agregadas
        avg_metrics = {}
        metric_keys = ["precision@10", "recall@10", "f1@10", "ndcg@10", "mrr@10", 
                      "hits@1", "hits@3", "hits@5", "hits@10", "avg_relevant_score"]
        
        for metric in metric_keys:
            values = [r["metrics"][metric] for r in results if metric in r["metrics"]]
            avg_metrics[f"avg_{metric}"] = np.mean(values) if values else 0
            avg_metrics[f"std_{metric}"] = np.std(values) if values else 0
        
        # Análisis por categoría
        category_analysis = {}
        categories = set(r["category"] for r in results)
        
        for category in categories:
            cat_results = [r for r in results if r["category"] == category]
            cat_metrics = {}
            
            for metric in metric_keys:
                values = [r["metrics"][metric] for r in cat_results if metric in r["metrics"]]
                cat_metrics[f"avg_{metric}"] = np.mean(values) if values else 0
            
            category_analysis[category] = {
                "count": len(cat_results),
                "metrics": cat_metrics
            }
        
        # Análisis por dificultad
        difficulty_analysis = {}
        difficulties = set(r["difficulty"] for r in results)
        
        for difficulty in difficulties:
            diff_results = [r for r in results if r["difficulty"] == difficulty]
            diff_metrics = {}
            
            for metric in metric_keys:
                values = [r["metrics"][metric] for r in diff_results if metric in r["metrics"]]
                diff_metrics[f"avg_{metric}"] = np.mean(values) if values else 0
            
            difficulty_analysis[difficulty] = {
                "count": len(diff_results),
                "metrics": diff_metrics
            }
        
        # Identificar consultas problemáticas
        problematic_queries = []
        for result in results:
            f1_score = result["metrics"].get("f1@10", 0)
            if f1_score < 0.3:  # Umbral de performance pobre
                problematic_queries.append({
                    "question": result["question"],
                    "f1_score": f1_score,
                    "category": result["category"],
                    "difficulty": result["difficulty"]
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
            "overall_metrics": avg_metrics,
            "category_analysis": category_analysis,
            "difficulty_analysis": difficulty_analysis,
            "problematic_queries": problematic_queries,
            "summary": {
                "avg_f1": avg_metrics.get("avg_f1@10", 0),
                "avg_ndcg": avg_metrics.get("avg_ndcg@10", 0),
                "avg_mrr": avg_metrics.get("avg_mrr@10", 0),
                "performance_rating": self.get_performance_rating(avg_metrics.get("avg_f1@10", 0))
            }
        }
    
    def get_performance_rating(self, f1_score: float) -> str:
        """Asigna rating de performance basado en F1 score."""
        if f1_score >= 0.8:
            return "Excellent"
        elif f1_score >= 0.6:
            return "Good"
        elif f1_score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def save_evaluation_report(self, report: Dict, output_path: str = "data/evaluation_report.json"):
        """Guarda reporte de evaluación."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        debug_print(f"[INFO] ✅ Evaluation report saved: {output_path}")
    
    def compare_with_baseline(self, current_results: Dict, baseline_path: str = "data/baseline_results.json"):
        """Compara resultados actuales con baseline."""
        try:
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline = json.load(f)
        except FileNotFoundError:
            debug_print("[INFO] No baseline found. Current results will be saved as baseline.", force=True)
            self.save_evaluation_report(current_results, baseline_path)
            return None
        
        # Comparar métricas principales
        current_f1 = current_results["summary"]["avg_f1"]
        baseline_f1 = baseline["summary"]["avg_f1"]
        
        current_ndcg = current_results["summary"]["avg_ndcg"]
        baseline_ndcg = baseline["summary"]["avg_ndcg"]
        
        comparison = {
            "f1_improvement": current_f1 - baseline_f1,
            "ndcg_improvement": current_ndcg - baseline_ndcg,
            "f1_improvement_pct": ((current_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0,
            "ndcg_improvement_pct": ((current_ndcg - baseline_ndcg) / baseline_ndcg * 100) if baseline_ndcg > 0 else 0,
            "current_rating": current_results["summary"]["performance_rating"],
            "baseline_rating": baseline["summary"]["performance_rating"]
        }
        
        return comparison


class PerformanceMonitor:
    """Monitor de performance en tiempo real."""
    
    def __init__(self):
        self.query_logs = []
        
    def log_query(self, question: str, results: List[Dict], response_time: float, debug_info: str = ""):
        """Registra una consulta para monitoreo."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "results_count": len(results),
            "response_time": response_time,
            "avg_score": np.mean([r.get("score", 0) for r in results]) if results else 0,
            "max_score": max([r.get("score", 0) for r in results]) if results else 0,
            "has_high_confidence": any(r.get("confidence_level") == "high" for r in results),
            "debug_info": debug_info
        }
        
        self.query_logs.append(log_entry)
        
        # Mantener solo los últimos 1000 logs
        if len(self.query_logs) > 1000:
            self.query_logs = self.query_logs[-1000:]
    
    def get_performance_stats(self, last_n_queries: int = 100) -> Dict:
        """Obtiene estadísticas de performance recientes."""
        recent_logs = self.query_logs[-last_n_queries:] if self.query_logs else []
        
        if not recent_logs:
            return {"message": "No query logs available"}
        
        response_times = [log["response_time"] for log in recent_logs]
        result_counts = [log["results_count"] for log in recent_logs]
        avg_scores = [log["avg_score"] for log in recent_logs]
        
        return {
            "total_queries": len(recent_logs),
            "avg_response_time": np.mean(response_times),
            "p95_response_time": np.percentile(response_times, 95),
            "avg_results_per_query": np.mean(result_counts),
            "avg_relevance_score": np.mean(avg_scores),
            "queries_with_high_confidence": sum(1 for log in recent_logs if log["has_high_confidence"]),
            "zero_result_queries": sum(1 for log in recent_logs if log["results_count"] == 0),
            "performance_issues": self.detect_performance_issues(recent_logs)
        }
    
    def detect_performance_issues(self, logs: List[Dict]) -> List[str]:
        """Detecta problemas de performance."""
        issues = []
        
        # Tiempo de respuesta alto
        avg_response_time = np.mean([log["response_time"] for log in logs])
        if avg_response_time > 5.0:
            issues.append(f"High average response time: {avg_response_time:.2f}s")
        
        # Muchas consultas sin resultados
        zero_results = sum(1 for log in logs if log["results_count"] == 0)
        zero_results_pct = (zero_results / len(logs)) * 100
        if zero_results_pct > 20:
            issues.append(f"High zero-result queries: {zero_results_pct:.1f}%")
        
        # Scores de relevancia bajos
        avg_relevance = np.mean([log["avg_score"] for log in logs])
        if avg_relevance < 0.5:
            issues.append(f"Low average relevance scores: {avg_relevance:.3f}")
        
        return issues


# Función de utilidad para ejecutar evaluación completa
def run_full_evaluation(weaviate_wrapper, embedding_client, save_results: bool = True):
    """Ejecuta evaluación completa del sistema."""
    
    debug_print("🔍 Starting comprehensive evaluation...", force=True)
    
    # Inicializar sistema de evaluación
    eval_system = EvaluationSystem(weaviate_wrapper, embedding_client)
    
    # Ejecutar evaluación
    report = eval_system.run_evaluation()
    
    # Mostrar resultados
    debug_print("\n📊 EVALUATION RESULTS", force=True)
    debug_print("=" * 50, force=True)
    debug_print(f"Total queries evaluated: {report['total_queries']}", force=True)
    debug_print(f"Overall F1@10: {report['summary']['avg_f1']:.3f}", force=True)
    debug_print(f"Overall nDCG@10: {report['summary']['avg_ndcg']:.3f}", force=True)
    debug_print(f"Overall MRR@10: {report['summary']['avg_mrr']:.3f}", force=True)
    debug_print(f"Performance rating: {report['summary']['performance_rating']}", force=True)
    
    # Análisis por categoría
    debug_print("\n📈 Performance by Category:", force=True)
    for category, data in report['category_analysis'].items():
        f1_score = data['metrics'].get('avg_f1@10', 0)
        debug_print(f"  {category}: F1={f1_score:.3f} ({data['count']} queries)", force=True)
    
    # Consultas problemáticas
    if report['problematic_queries']:
        debug_print(f"\n⚠️ Problematic queries ({len(report['problematic_queries'])}):", force=True)
        for pq in report['problematic_queries'][:3]:  # Mostrar solo las primeras 3
            debug_print(f"  - {pq['question'][:60]}... (F1: {pq['f1_score']:.3f})", force=True)
    
    # Guardar resultados
    if save_results:
        eval_system.save_evaluation_report(report)
        
        # Comparar con baseline
        comparison = eval_system.compare_with_baseline(report)
        if comparison:
            debug_print(f"\n📊 Comparison with baseline:", force=True)
            debug_print(f"  F1 improvement: {comparison['f1_improvement']:+.3f} ({comparison['f1_improvement_pct']:+.1f}%))", force=True)
            debug_print(f"  nDCG improvement: {comparison['ndcg_improvement']:+.3f} ({comparison['ndcg_improvement_pct']:+.1f}%))", force=True)
    
    return report