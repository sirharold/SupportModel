#!/usr/bin/env python3
"""
Test script to verify the reranking method changes are working
"""
import streamlit as st
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_config_page():
    st.title("🔧 Test: Reranking Method Configuration")
    
    # Test the selectbox that should now be in the config
    st.subheader("Testing Reranking Method Selection")
    
    reranking_method = st.selectbox(
        "🔄 Método de Reranking:",
        options=["crossencoder", "standard", "none"],
        index=0,  # CrossEncoder por defecto
        format_func=lambda x: {
            "crossencoder": "🧠 CrossEncoder (Recomendado)",
            "standard": "📊 Reranking Estándar", 
            "none": "❌ Sin Reranking"
        }[x],
        help="Método de reranking: CrossEncoder usa ms-marco-MiniLM-L-6-v2 para mejor calidad"
    )
    
    st.write(f"**Selected method:** {reranking_method}")
    
    # Show what the configuration would look like
    st.subheader("Configuration Preview")
    config_preview = {
        "reranking_method": reranking_method,
        "use_reranking": reranking_method != "none",  # Backward compatibility
        "description": {
            "crossencoder": "Uses ms-marco-MiniLM-L-6-v2 CrossEncoder model with sigmoid normalization",
            "standard": "Uses OpenAI GPT-3.5-turbo for document reranking",
            "none": "No reranking applied, uses original vector search order"
        }[reranking_method]
    }
    
    st.json(config_preview)
    
    # Test import of the actual config page
    st.subheader("Import Test")
    try:
        from src.apps.cumulative_n_questions_config import show_cumulative_n_questions_config_page
        st.success("✅ Successfully imported cumulative_n_questions_config")
        
        # Check if we can access the function
        st.info(f"📋 Function available: {callable(show_cumulative_n_questions_config_page)}")
        
    except ImportError as e:
        st.error(f"❌ Import failed: {e}")
    except Exception as e:
        st.error(f"❌ Other error: {e}")

if __name__ == "__main__":
    test_config_page()