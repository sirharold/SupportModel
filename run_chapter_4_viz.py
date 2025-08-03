#!/usr/bin/env python3
"""
Script independiente para ejecutar las visualizaciones del Capítulo 4
Autor: Harold Gómez
Fecha: 2025-08-03
"""

import streamlit as st
from src.apps.chapter_4_visualizations import main

# Configuración de página
st.set_page_config(
    page_title="Capítulo 4: Visualizaciones EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if __name__ == "__main__":
    main()