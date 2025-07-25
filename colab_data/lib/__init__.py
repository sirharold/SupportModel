#!/usr/bin/env python3
"""
Modular Libraries Package for Colab Embeddings Evaluation

This package contains modular libraries for evaluating embedding models
using RAGAS framework, LLM reranking, and comprehensive metrics.
"""

# Version information
__version__ = "2.0.0"
__author__ = "Sistema de Evaluación Automática"
__description__ = "Modular libraries for embeddings evaluation with RAGAS"

# Import key functions for easy access
try:
    from .colab_setup import quick_setup, import_required_modules
    from .evaluation_metrics import create_metrics_calculator
    from .rag_evaluation import create_rag_pipeline
    from .data_manager import create_data_pipeline
    from .results_processor import process_and_save_results
    
    # Make key functions available at package level
    __all__ = [
        'quick_setup',
        'import_required_modules', 
        'create_metrics_calculator',
        'create_rag_pipeline',
        'create_data_pipeline',
        'process_and_save_results'
    ]
    
    print(f"✅ Modular Libraries v{__version__} loaded successfully")
    
except ImportError as e:
    print(f"⚠️ Warning: Some modules could not be imported: {e}")
    __all__ = []

def get_version():
    """Get package version"""
    return __version__

def get_info():
    """Get package information"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'available_modules': __all__
    }