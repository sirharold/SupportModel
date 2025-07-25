#!/usr/bin/env python3
"""
Data Manager Library - Document loading, embedding retrieval, and query processing
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import gc
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
from tqdm import tqdm
import pytz
from datetime import datetime

# Model configurations
QUERY_MODELS = {
    'ada': 'text-embedding-ada-002',  # OpenAI model - 1536 dims
    'e5-large': 'intfloat/e5-large-v2',  # E5-Large model - 1024 dims
    'mpnet': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',  # 768 dims
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2'  # 384 dims
}

MODEL_MAPPING = {
    'multi-qa-mpnet-base-dot-v1': 'mpnet',
    'all-MiniLM-L6-v2': 'minilm',
    'ada': 'ada',
    'text-embedding-ada-002': 'ada',
    'e5-large-v2': 'e5-large',
    'intfloat/e5-large-v2': 'e5-large'
}

class DocumentLoader:
    """Handles loading and caching of document embeddings"""
    
    def __init__(self, base_path: str, debug: bool = False):
        self.base_path = base_path
        self.debug = debug
        self.loaded_documents = {}  # Cache for loaded documents
        
    def get_embedding_file_path(self, model_name: str) -> str:
        """Get the file path for a model's embeddings"""
        embedding_files = {
            'ada': f'{self.base_path}docs_ada_with_embeddings_20250721_123712.parquet',
            'e5-large': f'{self.base_path}docs_e5large_with_embeddings_20250721_124918.parquet',
            'mpnet': f'{self.base_path}docs_mpnet_with_embeddings_20250721_125254.parquet',
            'minilm': f'{self.base_path}docs_minilm_with_embeddings_20250721_125846.parquet'
        }
        return embedding_files.get(model_name, '')
    
    def load_documents_with_embeddings(self, model_name: str, force_reload: bool = False) -> pd.DataFrame:
        """
        Load documents with pre-computed embeddings
        
        Args:
            model_name: Name of the embedding model ('ada', 'e5-large', 'mpnet', 'minilm')
            force_reload: Force reload even if cached
            
        Returns:
            DataFrame with documents and embeddings
        """
        # Check cache first
        if not force_reload and model_name in self.loaded_documents:
            if self.debug:
                print(f"📋 Using cached documents for {model_name}")
            return self.loaded_documents[model_name]
        
        file_path = self.get_embedding_file_path(model_name)
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Embedding file not found for model {model_name}: {file_path}")
        
        if self.debug:
            print(f"📂 Loading documents with embeddings for {model_name}...")
            print(f"📂 File: {os.path.basename(file_path)}")
        
        try:
            # Load parquet file
            df = pd.read_parquet(file_path)
            
            if self.debug:
                print(f"✅ Loaded {len(df)} documents")
                print(f"📊 Columns: {list(df.columns)}")
                if 'embedding' in df.columns:
                    print(f"🔢 Embedding dimensions: {len(df['embedding'].iloc[0]) if len(df) > 0 else 'N/A'}")
            
            # Cache the loaded documents
            self.loaded_documents[model_name] = df
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading documents for {model_name}: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        df = self.load_documents_with_embeddings(model_name)
        
        info = {
            'model_name': model_name,
            'query_model': QUERY_MODELS.get(model_name, 'unknown'),
            'num_documents': len(df),
            'embedding_dim': len(df['embedding'].iloc[0]) if len(df) > 0 and 'embedding' in df.columns else 0,
            'file_path': self.get_embedding_file_path(model_name),
            'columns': list(df.columns) if not df.empty else []
        }
        
        return info
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models"""
        models_info = {}
        
        for model_name in QUERY_MODELS.keys():
            try:
                models_info[model_name] = self.get_model_info(model_name)
            except Exception as e:
                models_info[model_name] = {
                    'error': str(e),
                    'model_name': model_name,
                    'available': False
                }
        
        return models_info
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear document cache"""
        if model_name:
            if model_name in self.loaded_documents:
                del self.loaded_documents[model_name]
                if self.debug:
                    print(f"🗑️ Cleared cache for {model_name}")
        else:
            self.loaded_documents.clear()
            if self.debug:
                print("🗑️ Cleared all document cache")
        
        gc.collect()


class QueryEmbedder:
    """Handles query embedding generation using various models"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.sentence_transformers = {}  # Cache for SentenceTransformer models
        self.openai_client = None
        
    def _get_sentence_transformer(self, model_name: str) -> SentenceTransformer:
        """Get cached SentenceTransformer model"""
        if model_name not in self.sentence_transformers:
            if self.debug:
                print(f"🤖 Loading SentenceTransformer model: {model_name}")
            self.sentence_transformers[model_name] = SentenceTransformer(model_name)
        return self.sentence_transformers[model_name]
    
    def _get_openai_client(self):
        """Get OpenAI client"""
        if self.openai_client is None:
            self.openai_client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment
        return self.openai_client
    
    def embed_query(self, query: str, model_name: str) -> np.ndarray:
        """
        Generate embedding for a query using specified model
        
        Args:
            query: Query text to embed
            model_name: Model name ('ada', 'e5-large', 'mpnet', 'minilm')
            
        Returns:
            Query embedding as numpy array
        """
        if model_name not in QUERY_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(QUERY_MODELS.keys())}")
        
        query_model = QUERY_MODELS[model_name]
        
        try:
            if model_name == 'ada':
                # OpenAI embedding
                client = self._get_openai_client()
                response = client.embeddings.create(
                    input=query,
                    model=query_model
                )
                embedding = np.array(response.data[0].embedding)
                
            else:
                # SentenceTransformer embedding
                model = self._get_sentence_transformer(query_model)
                
                # Add prefix for E5 models
                if model_name == 'e5-large':
                    query = f"query: {query}"
                
                embedding = model.encode(query, normalize_embeddings=True)
            
            if self.debug:
                print(f"🔍 Generated {model_name} embedding: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating embedding for {model_name}: {e}")
            raise
    
    def embed_queries_batch(self, queries: List[str], model_name: str) -> List[np.ndarray]:
        """
        Generate embeddings for multiple queries
        
        Args:
            queries: List of query texts
            model_name: Model name
            
        Returns:
            List of query embeddings
        """
        embeddings = []
        
        if self.debug:
            print(f"🔍 Generating embeddings for {len(queries)} queries using {model_name}")
        
        for query in tqdm(queries, desc=f"Embedding queries ({model_name})"):
            try:
                embedding = self.embed_query(query, model_name)
                embeddings.append(embedding)
            except Exception as e:
                print(f"⚠️ Failed to embed query: {query[:50]}... Error: {e}")
                # Use zero embedding as fallback
                dim = self._get_expected_dimension(model_name)
                embeddings.append(np.zeros(dim))
        
        return embeddings
    
    def _get_expected_dimension(self, model_name: str) -> int:
        """Get expected embedding dimension for a model"""
        dimensions = {
            'ada': 1536,
            'e5-large': 1024,
            'mpnet': 768,
            'minilm': 384
        }
        return dimensions.get(model_name, 768)
    
    def clear_cache(self):
        """Clear model cache"""
        self.sentence_transformers.clear()
        self.openai_client = None
        gc.collect()
        
        if self.debug:
            print("🗑️ Cleared embedding model cache")


class DocumentRetriever:
    """Handles document retrieval using cosine similarity"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def retrieve_documents(self, query_embedding: np.ndarray, documents_df: pd.DataFrame, 
                          top_k: int = 10, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve most similar documents to query
        
        Args:
            query_embedding: Query embedding vector
            documents_df: DataFrame with documents and embeddings
            top_k: Number of top documents to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of retrieved documents with similarity scores
        """
        if documents_df.empty or 'embedding' not in documents_df.columns:
            return []
        
        try:
            # Extract document embeddings
            doc_embeddings = np.vstack(documents_df['embedding'].values)
            
            # Calculate cosine similarities
            query_embedding = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter by similarity threshold
            top_indices = [idx for idx in top_indices if similarities[idx] >= similarity_threshold]
            
            # Prepare results
            results = []
            for idx in top_indices:
                doc_data = documents_df.iloc[idx].to_dict()
                doc_data['similarity_score'] = float(similarities[idx])
                doc_data['rank'] = len(results) + 1
                results.append(doc_data)
            
            if self.debug:
                print(f"🎯 Retrieved {len(results)} documents (top_k={top_k}, threshold={similarity_threshold})")
                if results:
                    print(f"📊 Similarity range: {results[-1]['similarity_score']:.3f} - {results[0]['similarity_score']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []
    
    def retrieve_for_multiple_queries(self, query_embeddings: List[np.ndarray], 
                                    documents_df: pd.DataFrame, top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Retrieve documents for multiple queries
        
        Args:
            query_embeddings: List of query embeddings
            documents_df: DataFrame with documents and embeddings
            top_k: Number of top documents per query
            
        Returns:
            List of lists of retrieved documents
        """
        all_results = []
        
        if self.debug:
            print(f"🎯 Retrieving documents for {len(query_embeddings)} queries")
        
        for i, query_embedding in enumerate(tqdm(query_embeddings, desc="Retrieving documents")):
            try:
                results = self.retrieve_documents(query_embedding, documents_df, top_k)
                all_results.append(results)
            except Exception as e:
                print(f"⚠️ Failed to retrieve for query {i}: {e}")
                all_results.append([])
        
        return all_results
    
    def calculate_retrieval_statistics(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for retrieved documents"""
        if not retrieved_docs:
            return {'num_documents': 0}
        
        similarities = [doc['similarity_score'] for doc in retrieved_docs]
        
        stats = {
            'num_documents': len(retrieved_docs),
            'avg_similarity': np.mean(similarities),
            'min_similarity': min(similarities),
            'max_similarity': max(similarities),
            'std_similarity': np.std(similarities)
        }
        
        return stats


class ConfigurationManager:
    """Handles loading and parsing of evaluation configurations"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def load_evaluation_config(self, config_file_path: str) -> Dict[str, Any]:
        """
        Load evaluation configuration from JSON file
        
        Args:
            config_file_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
        
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if self.debug:
                print(f"📂 Loaded configuration from {os.path.basename(config_file_path)}")
                print(f"📊 Configuration keys: {list(config.keys())}")
                
                if 'questions' in config:
                    print(f"❓ Number of questions: {len(config['questions'])}")
                
                if 'evaluation_params' in config:
                    print(f"⚙️ Evaluation parameters: {config['evaluation_params']}")
            
            return config
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            raise
    
    def parse_questions(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and parse questions from configuration"""
        
        if 'questions' not in config:
            raise ValueError("Configuration must contain 'questions' key")
        
        questions = config['questions']
        parsed_questions = []
        
        for i, q in enumerate(questions):
            try:
                parsed_q = {
                    'id': i,
                    'question': q.get('question', ''),
                    'ground_truth_links': q.get('links', []),
                    'expected_answer': q.get('expected_answer', ''),
                    'category': q.get('category', 'general'),
                    'difficulty': q.get('difficulty', 'medium')
                }
                parsed_questions.append(parsed_q)
                
            except Exception as e:
                print(f"⚠️ Error parsing question {i}: {e}")
                continue
        
        if self.debug:
            print(f"✅ Parsed {len(parsed_questions)} questions")
        
        return parsed_questions
    
    def get_evaluation_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evaluation parameters from configuration"""
        
        default_params = {
            'top_k_values': [1, 3, 5, 10],
            'similarity_threshold': 0.0,
            'use_llm_reranking': True,
            'llm_model': 'gpt-3.5-turbo',
            'max_questions': None
        }
        
        if 'evaluation_params' in config:
            params = config['evaluation_params']
            default_params.update(params)
        
        return default_params


class DataManagerPipeline:
    """Complete data management pipeline combining all components"""
    
    def __init__(self, base_path: str, debug: bool = False):
        self.base_path = base_path
        self.debug = debug
        
        # Initialize components
        self.document_loader = DocumentLoader(base_path, debug)
        self.query_embedder = QueryEmbedder(debug)
        self.document_retriever = DocumentRetriever(debug)
        self.config_manager = ConfigurationManager(debug)
    
    def setup_model(self, model_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Setup a specific embedding model
        
        Args:
            model_name: Name of the model to setup
            
        Returns:
            Tuple of (documents_df, model_info)
        """
        if self.debug:
            print(f"🚀 Setting up model: {model_name}")
        
        # Load documents with embeddings
        documents_df = self.document_loader.load_documents_with_embeddings(model_name)
        
        # Get model information
        model_info = self.document_loader.get_model_info(model_name)
        
        if self.debug:
            print(f"✅ Model {model_name} ready with {len(documents_df)} documents")
        
        return documents_df, model_info
    
    def process_questions(self, config_file_path: str, model_name: str, 
                         max_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        Process evaluation questions for a specific model
        
        Args:
            config_file_path: Path to evaluation configuration
            model_name: Model to use for processing
            max_questions: Maximum number of questions to process
            
        Returns:
            Processing results
        """
        if self.debug:
            print(f"📝 Processing questions for model: {model_name}")
        
        # Load configuration
        config = self.config_manager.load_evaluation_config(config_file_path)
        questions = self.config_manager.parse_questions(config)
        eval_params = self.config_manager.get_evaluation_parameters(config)
        
        # Limit questions if specified
        if max_questions:
            questions = questions[:max_questions]
        
        # Setup model
        documents_df, model_info = self.setup_model(model_name)
        
        # Generate query embeddings
        query_texts = [q['question'] for q in questions]
        query_embeddings = self.query_embedder.embed_queries_batch(query_texts, model_name)
        
        # Retrieve documents for all queries
        top_k = max(eval_params.get('top_k_values', [10]))
        all_retrieved_docs = self.document_retriever.retrieve_for_multiple_queries(
            query_embeddings, documents_df, top_k
        )
        
        # Combine questions with retrieved documents
        processed_questions = []
        for i, (question, retrieved_docs) in enumerate(zip(questions, all_retrieved_docs)):
            question_data = question.copy()
            question_data['retrieved_docs'] = retrieved_docs
            question_data['query_embedding'] = query_embeddings[i]
            question_data['retrieval_stats'] = self.document_retriever.calculate_retrieval_statistics(retrieved_docs)
            processed_questions.append(question_data)
        
        results = {
            'model_name': model_name,
            'model_info': model_info,
            'num_questions': len(processed_questions),
            'evaluation_params': eval_params,
            'processed_questions': processed_questions,
            'timestamp': datetime.now(pytz.timezone('America/Santiago')).isoformat()
        }
        
        if self.debug:
            print(f"✅ Processed {len(processed_questions)} questions for {model_name}")
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        models_info = self.document_loader.get_all_models_info()
        
        system_info = {
            'timestamp': datetime.now(pytz.timezone('America/Santiago')).isoformat(),
            'base_path': self.base_path,
            'available_models': list(QUERY_MODELS.keys()),
            'models_info': models_info,
            'query_models_mapping': QUERY_MODELS
        }
        
        return system_info
    
    def cleanup(self):
        """Clean up resources and caches"""
        self.document_loader.clear_cache()
        self.query_embedder.clear_cache()
        gc.collect()
        
        if self.debug:
            print("🗑️ Cleaned up data manager resources")


# Factory functions
def create_document_loader(base_path: str, debug: bool = False) -> DocumentLoader:
    """Create DocumentLoader instance"""
    return DocumentLoader(base_path, debug)

def create_query_embedder(debug: bool = False) -> QueryEmbedder:
    """Create QueryEmbedder instance"""
    return QueryEmbedder(debug)

def create_document_retriever(debug: bool = False) -> DocumentRetriever:
    """Create DocumentRetriever instance"""
    return DocumentRetriever(debug)

def create_config_manager(debug: bool = False) -> ConfigurationManager:
    """Create ConfigurationManager instance"""
    return ConfigurationManager(debug)

def create_data_pipeline(base_path: str, debug: bool = False) -> DataManagerPipeline:
    """Create complete DataManagerPipeline instance"""
    return DataManagerPipeline(base_path, debug)


if __name__ == "__main__":
    # Test the data management components
    print("🧪 Testing Data Manager Library...")
    
    # This would require actual data files to test properly
    base_path = '/content/drive/MyDrive/TesisMagister/acumulative/colab_data/'
    
    try:
        # Test configuration manager
        print("📂 Testing Configuration Manager...")
        config_manager = create_config_manager(debug=True)
        
        # Test query embedder (without API keys)
        print("🔍 Testing Query Embedder...")
        embedder = create_query_embedder(debug=True)
        print(f"✅ Expected dimensions: {embedder._get_expected_dimension('mpnet')}")
        
        # Test document retriever
        print("🎯 Testing Document Retriever...")
        retriever = create_document_retriever(debug=True)
        
        print("🎉 Data Manager Library components initialized successfully!")
        
    except Exception as e:
        print(f"⚠️ Test requires actual data files: {e}")
        print("📋 Library is ready for use with proper data files.")