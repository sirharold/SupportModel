#!/usr/bin/env python3
"""
Force reload the config module to ensure changes are visible
"""
import sys
import importlib
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import and reload the module
    from src.apps import cumulative_n_questions_config
    importlib.reload(cumulative_n_questions_config)
    print("✅ Successfully reloaded cumulative_n_questions_config module")
    
    # Check if the function exists
    if hasattr(cumulative_n_questions_config, 'show_cumulative_n_questions_config_page'):
        print("✅ Function show_cumulative_n_questions_config_page is available")
    else:
        print("❌ Function show_cumulative_n_questions_config_page not found")
        
    # Try to find any reference to reranking_method in the module
    import inspect
    source = inspect.getsource(cumulative_n_questions_config)
    if 'reranking_method' in source:
        print("✅ Found 'reranking_method' in the module source")
        count = source.count('reranking_method')
        print(f"   Found {count} occurrences")
    else:
        print("❌ 'reranking_method' not found in module source")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()