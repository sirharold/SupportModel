#!/usr/bin/env python3
"""
Quick fix for missing gc import in the notebook
Add this line at the beginning of the cell that's causing the error:
"""

# Add this line at the top of the cell:
import gc

# Then the rest of your code:
# Limpiar recursos y memoria
print("🧹 Limpiando recursos...")

# Limpiar pipeline de datos
data_pipeline.cleanup()

# Limpiar memoria
gc.collect()

# Rest of the code continues...