"""
Utilidades para manejo de memoria.
"""

import gc
import os
import psutil


def get_memory_usage() -> float:
    """Obtiene el uso actual de memoria en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def cleanup_memory():
    """Fuerza la limpieza de memoria."""
    gc.collect()