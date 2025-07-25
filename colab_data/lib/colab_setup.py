#!/usr/bin/env python3
"""
Colab Setup Library - Environment and Package Management
"""

import subprocess
import sys
import time
import os
from datetime import datetime
import pytz

# Global constants
CHILE_TZ = pytz.timezone('America/Santiago')

# Required packages configuration
REQUIRED_PACKAGES = [
    ("sentence-transformers", "sentence_transformers"),
    ("pandas", "pandas"), 
    ("numpy", "numpy"), 
    ("scikit-learn", "sklearn"),
    ("tqdm", "tqdm"), 
    ("pytz", "pytz"), 
    ("huggingface_hub", "huggingface_hub"), 
    ("openai", "openai"), 
    ("ragas", "ragas"), 
    ("datasets", "datasets"),
    ("bert-score", "bert_score")  # For BERTScore functionality
]

# Google Drive paths
BASE_PATH = '/content/drive/MyDrive/TesisMagister/acumulative/colab_data/'
ACUMULATIVE_PATH = '/content/drive/MyDrive/TesisMagister/acumulative/'
RESULTS_OUTPUT_PATH = ACUMULATIVE_PATH

# Embedding files configuration
EMBEDDING_FILES = {
    'ada': BASE_PATH + 'docs_ada_with_embeddings_20250721_123712.parquet',
    'e5-large': BASE_PATH + 'docs_e5large_with_embeddings_20250721_124918.parquet',
    'mpnet': BASE_PATH + 'docs_mpnet_with_embeddings_20250721_125254.parquet',
    'minilm': BASE_PATH + 'docs_minilm_with_embeddings_20250721_125846.parquet'
}

# Query model mappings
QUERY_MODELS = {
    'ada': 'text-embedding-ada-002',  # ✅ OpenAI model - 1536 dims
    'e5-large': 'intfloat/e5-large-v2',  # ✅ FIXED: Use E5-Large model - 1024 dims
    'mpnet': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',  # ✅ 768 dims
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2'  # ✅ 384 dims
}

# Model name mappings
MODEL_MAPPING = {
    'multi-qa-mpnet-base-dot-v1': 'mpnet',
    'all-MiniLM-L6-v2': 'minilm',
    'ada': 'ada',
    'text-embedding-ada-002': 'ada',
    'e5-large-v2': 'e5-large',
    'intfloat/e5-large-v2': 'e5-large'
}

class ColabEnvironment:
    """Manages Colab environment setup and configuration"""
    
    def __init__(self):
        self.start_time = time.time()
        self.setup_complete = False
        self.packages_installed = False
        self.drive_mounted = False
        self.api_keys_loaded = False
        
    def start_timer(self):
        """Initialize notebook execution timer"""
        self.start_time = time.time()
        print(f"⏱️ Iniciando cronómetro del notebook: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self.start_time
    
    def install_if_missing(self, package_name: str, import_name: str = None):
        """Install package if not already available"""
        check_name = import_name if import_name else package_name
        try:
            __import__(check_name)
            print(f"✅ {package_name}")
            return True
        except ImportError:
            print(f"📦 Installing {package_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"✅ {package_name} installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package_name}: {e}")
                return False
    
    def install_all_packages(self):
        """Install all required packages"""
        print("📦 Checking and installing required packages...")
        failed_packages = []
        
        for package, import_name in REQUIRED_PACKAGES:
            if not self.install_if_missing(package, import_name):
                failed_packages.append(package)
        
        if failed_packages:
            print(f"❌ Failed to install: {failed_packages}")
            self.packages_installed = False
            return False
        else:
            print("✅ All packages installed successfully")
            self.packages_installed = True
            return True
    
    def mount_drive(self):
        """Mount Google Drive"""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Verify mount was successful
            if os.path.exists('/content/drive/MyDrive'):
                print("✅ Google Drive mounted successfully")
                self.drive_mounted = True
                return True
            else:
                print("❌ Google Drive mount failed - directory not accessible")
                self.drive_mounted = False
                return False
                
        except Exception as e:
            print(f"❌ Error mounting Google Drive: {e}")
            self.drive_mounted = False
            return False
    
    def load_api_keys(self):
        """Load OpenAI API keys from Colab secrets or .env file"""
        openai_available = False
        
        # Try Colab secrets first
        try:
            from google.colab import userdata
            openai_key = userdata.get('OPENAI_API_KEY')
            if openai_key:
                os.environ['OPENAI_API_KEY'] = openai_key
                print("✅ OpenAI API key loaded from Colab secrets")
                openai_available = True
        except Exception as e:
            print(f"⚠️ Could not load from Colab secrets: {e}")
        
        # Fallback to .env file
        if not openai_available:
            env_file_path = ACUMULATIVE_PATH + '.env'
            if os.path.exists(env_file_path):
                try:
                    with open(env_file_path, 'r') as f:
                        for line in f:
                            if 'OPENAI_API_KEY=' in line:
                                key, value = line.strip().split('=', 1)
                                os.environ[key] = value.strip('"').strip("'")
                                print("✅ OpenAI API key loaded from .env")
                                openai_available = True
                                break
                except Exception as e:
                    print(f"❌ Error reading .env file: {e}")
            else:
                print("⚠️ No .env file found")
        
        # Try HuggingFace token
        hf_available = False
        try:
            from google.colab import userdata
            hf_token = userdata.get('HF_TOKEN')
            if hf_token:
                from huggingface_hub import login
                login(token=hf_token)
                print("✅ HF authenticated")
                hf_available = True
        except Exception:
            print("⚠️ HF token not found")
        
        self.api_keys_loaded = openai_available
        
        return {
            'openai_available': openai_available,
            'hf_available': hf_available
        }
    
    def get_config_file_path(self):
        """Get the latest evaluation config file path"""
        import glob
        import re
        
        config_files = glob.glob(ACUMULATIVE_PATH + 'evaluation_config_*.json')
        if config_files:
            # Extract timestamps from filenames and sort by them
            files_with_timestamps = []
            for file in config_files:
                # Extract timestamp from filename (e.g., evaluation_config_1737599283.json)
                match = re.search(r'evaluation_config_(\d+)\.json', file)
                if match:
                    timestamp = int(match.group(1))
                    files_with_timestamps.append((timestamp, file))
            
            # Sort by timestamp (descending) and get the latest
            files_with_timestamps.sort(reverse=True)
            if files_with_timestamps:
                questions_file = files_with_timestamps[0][1]
                print(f"📂 Found {len(config_files)} config files")
                print(f"📂 Using latest: {os.path.basename(questions_file)}")
                return questions_file
            else:
                # Fallback if no timestamp pattern found
                questions_file = sorted(config_files)[-1]
                print(f"⚠️ Using alphabetically sorted latest: {os.path.basename(questions_file)}")
                return questions_file
        else:
            questions_file = ACUMULATIVE_PATH + 'questions_with_links.json'
            print("⚠️ No evaluation_config files found, using default questions_with_links.json")
            return questions_file
    
    def verify_paths(self):
        """Verify all required paths exist"""
        paths_status = {}
        
        # Check base paths
        paths_to_check = {
            'BASE_PATH': BASE_PATH,
            'ACUMULATIVE_PATH': ACUMULATIVE_PATH
        }
        
        for name, path in paths_to_check.items():
            exists = os.path.exists(path)
            paths_status[name] = exists
            if exists:
                print(f"✅ {name}: {path}")
            else:
                print(f"❌ {name}: {path} (not found)")
        
        # Check embedding files
        for model_name, file_path in EMBEDDING_FILES.items():
            exists = os.path.exists(file_path)
            paths_status[f'embedding_{model_name}'] = exists
            if exists:
                print(f"✅ Embedding file {model_name}: exists")
            else:
                print(f"❌ Embedding file {model_name}: {file_path} (not found)")
        
        return paths_status
    
    def complete_setup(self):
        """Complete setup process and return configuration"""
        print("🚀 Starting Colab environment setup...")
        
        # 1. Start timer
        self.start_timer()
        
        # 2. Install packages
        if not self.install_all_packages():
            return {'success': False, 'error': 'Package installation failed'}
        
        # 3. Mount Google Drive
        if not self.mount_drive():
            return {'success': False, 'error': 'Google Drive mount failed'}
        
        # 4. Load API keys
        api_status = self.load_api_keys()
        
        # 5. Verify paths
        paths_status = self.verify_paths()
        
        # 6. Get config file
        try:
            config_file_path = self.get_config_file_path()
        except Exception as e:
            return {'success': False, 'error': f'Config file loading failed: {e}'}
        
        self.setup_complete = True
        
        setup_result = {
            'success': True,
            'setup_time': time.time() - self.start_time,
            'packages_installed': self.packages_installed,
            'drive_mounted': self.drive_mounted,
            'api_keys_loaded': self.api_keys_loaded,
            'api_status': api_status,
            'paths_status': paths_status,
            'config_file_path': config_file_path,
            'embedding_files': EMBEDDING_FILES,
            'query_models': QUERY_MODELS,
            'model_mapping': MODEL_MAPPING,
            'constants': {
                'BASE_PATH': BASE_PATH,
                'ACUMULATIVE_PATH': ACUMULATIVE_PATH,
                'RESULTS_OUTPUT_PATH': RESULTS_OUTPUT_PATH
            }
        }
        
        print(f"✅ Setup completed in {setup_result['setup_time']:.2f} seconds")
        print(f"🔑 OpenAI API: {'✅' if api_status['openai_available'] else '❌'}")
        print(f"🤗 HuggingFace: {'✅' if api_status['hf_available'] else '❌'}")
        
        return setup_result

def quick_setup():
    """Quick setup function for easy import"""
    env = ColabEnvironment()
    return env.complete_setup()

def import_required_modules():
    """Import all required modules after setup"""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer
        import json
        from datetime import datetime
        import pytz
        import gc
        from typing import List, Dict, Tuple
        from tqdm import tqdm
        
        # RAGAS imports
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity
        )
        from datasets import Dataset
        
        modules = {
            'pd': pd,
            'np': np,
            'cosine_similarity': cosine_similarity,
            'SentenceTransformer': SentenceTransformer,
            'json': json,
            'datetime': datetime,
            'pytz': pytz,
            'gc': gc,
            'List': List,
            'Dict': Dict,
            'Tuple': Tuple,
            'tqdm': tqdm,
            'evaluate': evaluate,
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall,
            'answer_correctness': answer_correctness,
            'answer_similarity': answer_similarity,
            'Dataset': Dataset,
            'CHILE_TZ': CHILE_TZ
        }
        
        print("✅ All required modules imported successfully")
        return modules
        
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        return None

if __name__ == "__main__":
    # Test setup
    setup_result = quick_setup()
    if setup_result['success']:
        print("🎉 Setup test completed successfully!")
        modules = import_required_modules()
        if modules:
            print("🎉 All modules imported successfully!")
    else:
        print(f"❌ Setup test failed: {setup_result.get('error', 'Unknown error')}")