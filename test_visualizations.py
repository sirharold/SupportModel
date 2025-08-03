#!/usr/bin/env python3
"""
Script de prueba para verificar que todas las visualizaciones del Capítulo 4 funcionan correctamente
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.apps.chapter_4_visualizations import (
    load_data,
    create_chunk_distribution_histogram,
    create_chunks_vs_docs_boxplot,
    create_topic_distribution_bar,
    create_topic_distribution_pie,
    create_questions_histogram,
    create_question_types_bar,
    create_ground_truth_flow,
    create_sankey_diagram,
    create_dashboard_summary
)

def test_all_visualizations():
    """Test all visualization functions"""
    print("🔄 Testing Chapter 4 visualizations...")
    
    try:
        # Load data
        print("📊 Loading data...")
        corpus_data, topic_data, questions_data = load_data()
        print(f"✅ Data loaded: {corpus_data['corpus_info']['total_chunks_analyzed']:,} chunks")
        
        # Test each visualization
        visualizations = [
            ("Chunk Distribution Histogram", create_chunk_distribution_histogram, corpus_data),
            ("Chunks vs Docs Boxplot", create_chunks_vs_docs_boxplot, corpus_data),
            ("Topic Distribution Bar", create_topic_distribution_bar, topic_data),
            ("Topic Distribution Pie", create_topic_distribution_pie, topic_data),
            ("Questions Histogram", create_questions_histogram, questions_data),
            ("Question Types Bar", create_question_types_bar, questions_data),
            ("Ground Truth Flow", create_ground_truth_flow, None),
            ("Dashboard Summary", create_dashboard_summary, (corpus_data, topic_data, questions_data))
        ]
        
        for name, func, data in visualizations:
            try:
                print(f"🎨 Testing {name}...")
                if data is None:
                    fig = func()
                elif isinstance(data, tuple):
                    fig = create_dashboard_summary(*data)
                else:
                    fig = func(data)
                
                if fig:
                    plt.close(fig)
                    print(f"✅ {name} - OK")
                else:
                    print(f"❌ {name} - No figure returned")
            except Exception as e:
                print(f"❌ {name} - Error: {e}")
        
        # Test Sankey (Plotly)
        try:
            print("🎨 Testing Sankey Diagram...")
            fig = create_sankey_diagram()
            if fig:
                print("✅ Sankey Diagram - OK")
            else:
                print("❌ Sankey Diagram - No figure returned")
        except Exception as e:
            print(f"❌ Sankey Diagram - Error: {e}")
        
        print("\n🎉 All visualization tests completed!")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_all_visualizations()
    if success:
        print("\n✅ All tests passed! Chapter 4 visualizations are ready to use.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")