"""
Diagrama Sankey de Flujo de Relevancia - CrossEncoder Reranking
Visualiza cómo el CrossEncoder modifica el ranking de documentos relevantes e irrelevantes
"""

import streamlit as st
import plotly.graph_objects as go
import json
import os
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


def get_latest_results_file():
    """Obtiene el archivo de resultados más reciente."""
    results_folders = [".", "data"]

    all_files = []
    for folder in results_folders:
        folder_path = os.path.join(os.getcwd(), folder)
        if os.path.exists(folder_path):
            try:
                files = os.listdir(folder_path)
                for file in files:
                    if file.startswith('cumulative_results_') and file.endswith('.json'):
                        file_path = os.path.join(folder_path, file)
                        file_stats = os.stat(file_path)
                        all_files.append({
                            'path': file_path,
                            'name': file,
                            'mtime': file_stats.st_mtime
                        })
            except Exception:
                continue

    if not all_files:
        return None

    # Ordenar por fecha de modificación y tomar el más reciente
    all_files.sort(key=lambda x: x['mtime'], reverse=True)
    return all_files[0]['path']


def load_results_data(file_path: str) -> Dict:
    """Carga los datos de resultados desde un archivo JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None


def analyze_relevance_flow(before_docs: List[Dict], after_docs: List[Dict], top_k: int) -> Dict:
    """
    Analiza el flujo de documentos relevantes e irrelevantes entre before y after.

    Returns:
        Dict con los flujos categorizados
    """
    # Tomar solo top-k documentos
    before_top_k = before_docs[:top_k]
    after_top_k = after_docs[:top_k]

    # Extraer links y relevancia
    before_relevant_links = {doc['link'] for doc in before_top_k if doc['is_relevant']}
    before_irrelevant_links = {doc['link'] for doc in before_top_k if not doc['is_relevant']}

    after_relevant_links = {doc['link'] for doc in after_top_k if doc['is_relevant']}
    after_irrelevant_links = {doc['link'] for doc in after_top_k if not doc['is_relevant']}

    # Contar documentos por categoría
    flows = {
        # Relevantes que se mantienen en top-k
        'relevant_kept': len(before_relevant_links & after_relevant_links),

        # Relevantes que salieron del top-k (estaban before, no están after)
        'relevant_lost': len(before_relevant_links - after_relevant_links),

        # Relevantes que entraron al top-k (no estaban before, están after)
        'relevant_gained': len(after_relevant_links - before_relevant_links),

        # Irrelevantes que se mantienen en top-k
        'irrelevant_kept': len(before_irrelevant_links & after_irrelevant_links),

        # Irrelevantes que salieron del top-k
        'irrelevant_removed': len(before_irrelevant_links - after_irrelevant_links),

        # Irrelevantes que entraron al top-k
        'irrelevant_added': len(after_irrelevant_links - before_irrelevant_links),
    }

    # Totales
    flows['total_before_relevant'] = len(before_relevant_links)
    flows['total_before_irrelevant'] = len(before_irrelevant_links)
    flows['total_after_relevant'] = len(after_relevant_links)
    flows['total_after_irrelevant'] = len(after_irrelevant_links)

    return flows


def calculate_aggregated_flows(all_before_metrics: List[Dict], all_after_metrics: List[Dict], top_k: int) -> Dict:
    """
    Calcula flujos agregados para todas las preguntas.

    Returns:
        Dict con promedios de flujos
    """
    aggregated = defaultdict(int)
    num_questions = len(all_before_metrics)

    for before, after in zip(all_before_metrics, all_after_metrics):
        if 'document_scores' in before and 'document_scores' in after:
            flows = analyze_relevance_flow(before['document_scores'], after['document_scores'], top_k)
            for key, value in flows.items():
                aggregated[key] += value

    # Calcular promedios
    for key in aggregated:
        aggregated[key] = aggregated[key] / num_questions

    return dict(aggregated)


def calculate_question_distribution(all_before_metrics: List[Dict], all_after_metrics: List[Dict], top_k: int) -> Dict:
    """
    Calcula la distribución de preguntas que mejoraron, empeoraron o permanecieron sin cambios.

    Returns:
        Dict con contadores y porcentajes de distribución
    """
    improved = 0
    worsened = 0
    unchanged = 0

    for before, after in zip(all_before_metrics, all_after_metrics):
        if 'document_scores' in before and 'document_scores' in after:
            flows = analyze_relevance_flow(before['document_scores'], after['document_scores'], top_k)

            # Calcular impacto para esta pregunta
            net_relevant = flows['relevant_gained'] - flows['relevant_lost']
            net_irrelevant = flows['irrelevant_removed'] - flows['irrelevant_added']
            question_impact = net_relevant + net_irrelevant

            if question_impact > 0:
                improved += 1
            elif question_impact < 0:
                worsened += 1
            else:
                unchanged += 1

    total = improved + worsened + unchanged

    return {
        'improved_count': improved,
        'worsened_count': worsened,
        'unchanged_count': unchanged,
        'improved_pct': (improved / total * 100) if total > 0 else 0,
        'worsened_pct': (worsened / total * 100) if total > 0 else 0,
        'unchanged_pct': (unchanged / total * 100) if total > 0 else 0,
        'total_questions': total
    }


def create_sankey_diagram(flows: Dict, top_k: int):
    """
    Crea el diagrama Sankey mostrando flujos de relevancia.
    """
    # Definir nodos
    nodes = [
        # Before (left side)
        "Relevantes<br>Before",           # 0
        "Irrelevantes<br>Before",         # 1

        # After (right side)
        "Relevantes<br>After",            # 2
        "Irrelevantes<br>After",          # 3

        # Estados intermedios (para mostrar flujos específicos)
        "Fuera Top-K",                    # 4
    ]

    # Colores para nodos
    node_colors = [
        "#2ecc71",  # Verde - Relevantes Before
        "#e74c3c",  # Rojo - Irrelevantes Before
        "#27ae60",  # Verde oscuro - Relevantes After
        "#c0392b",  # Rojo oscuro - Irrelevantes After
        "#95a5a6",  # Gris - Fuera Top-K
    ]

    # Definir flujos (source, target, value)
    links_source = []
    links_target = []
    links_value = []
    links_color = []
    links_label = []

    # 1. Relevantes que se mantienen (Before Relevant → After Relevant)
    if flows['relevant_kept'] > 0:
        links_source.append(0)  # Relevantes Before
        links_target.append(2)  # Relevantes After
        links_value.append(flows['relevant_kept'])
        links_color.append("rgba(46, 204, 113, 0.4)")  # Verde transparente
        links_label.append(f"Mantenidos: {flows['relevant_kept']:.1f}")

    # 2. Relevantes perdidos (Before Relevant → Fuera Top-K)
    if flows['relevant_lost'] > 0:
        links_source.append(0)  # Relevantes Before
        links_target.append(4)  # Fuera Top-K
        links_value.append(flows['relevant_lost'])
        links_color.append("rgba(231, 76, 60, 0.4)")  # Rojo transparente
        links_label.append(f"Perdidos: {flows['relevant_lost']:.1f}")

    # 3. Relevantes ganados (Fuera Top-K → After Relevant)
    if flows['relevant_gained'] > 0:
        links_source.append(4)  # Fuera Top-K
        links_target.append(2)  # Relevantes After
        links_value.append(flows['relevant_gained'])
        links_color.append("rgba(46, 204, 113, 0.4)")  # Verde transparente
        links_label.append(f"Ganados: {flows['relevant_gained']:.1f}")

    # 4. Irrelevantes que se mantienen (Before Irrelevant → After Irrelevant)
    if flows['irrelevant_kept'] > 0:
        links_source.append(1)  # Irrelevantes Before
        links_target.append(3)  # Irrelevantes After
        links_value.append(flows['irrelevant_kept'])
        links_color.append("rgba(192, 57, 43, 0.3)")  # Rojo oscuro transparente
        links_label.append(f"Mantenidos: {flows['irrelevant_kept']:.1f}")

    # 5. Irrelevantes removidos (Before Irrelevant → Fuera Top-K)
    if flows['irrelevant_removed'] > 0:
        links_source.append(1)  # Irrelevantes Before
        links_target.append(4)  # Fuera Top-K
        links_value.append(flows['irrelevant_removed'])
        links_color.append("rgba(46, 204, 113, 0.3)")  # Verde transparente (bueno!)
        links_label.append(f"Removidos: {flows['irrelevant_removed']:.1f}")

    # 6. Irrelevantes añadidos (Fuera Top-K → After Irrelevant)
    if flows['irrelevant_added'] > 0:
        links_source.append(4)  # Fuera Top-K
        links_target.append(3)  # Irrelevantes After
        links_value.append(flows['irrelevant_added'])
        links_color.append("rgba(231, 76, 60, 0.3)")  # Rojo transparente (malo!)
        links_label.append(f"Añadidos: {flows['irrelevant_added']:.1f}")

    # Crear diagrama
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
            customdata=[
                f"Total: {flows['total_before_relevant']:.1f}",
                f"Total: {flows['total_before_irrelevant']:.1f}",
                f"Total: {flows['total_after_relevant']:.1f}",
                f"Total: {flows['total_after_irrelevant']:.1f}",
                "Fuera de Top-K"
            ],
            hovertemplate='%{label}<br>%{customdata}<extra></extra>',
        ),
        link=dict(
            source=links_source,
            target=links_target,
            value=links_value,
            color=links_color,
            customdata=links_label,
            hovertemplate='%{customdata}<br>Documentos: %{value:.1f}<extra></extra>',
        )
    )])

    fig.update_layout(
        title=dict(
            text=f"Flujo de Relevancia: Recuperación Inicial → CrossEncoder Reranking<br><sub>Top-{top_k} Documentos (Promedio sobre todas las preguntas)</sub>",
            font=dict(size=20)
        ),
        font=dict(size=14),
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def show_sankey_page():
    """Página principal del diagrama Sankey."""
    st.title("📊 Diagrama de Flujo de Relevancia (Sankey)")

    st.markdown("""
    Este diagrama muestra cómo el **CrossEncoder** modifica el ranking de documentos relevantes e irrelevantes.

    **Interpretación:**
    - 🟢 **Verde**: Flujos positivos (relevantes mantenidos/ganados, irrelevantes removidos)
    - 🔴 **Rojo**: Flujos negativos (relevantes perdidos, irrelevantes añadidos/mantenidos)
    - ⚪ **Gris**: Documentos fuera del Top-K
    """)

    # Cargar datos
    with st.spinner("📂 Cargando último archivo de resultados..."):
        file_path = get_latest_results_file()

        if not file_path:
            st.error("❌ No se encontró ningún archivo de resultados")
            st.info("💡 Asegúrate de que exista un archivo cumulative_results_*.json en la carpeta data/")
            return

        data = load_results_data(file_path)

        if not data:
            return

        st.success(f"✅ Datos cargados: {os.path.basename(file_path)}")

    # Mostrar información del archivo
    with st.expander("ℹ️ Información del archivo"):
        st.json({
            "Archivo": os.path.basename(file_path),
            "Preguntas": data.get('config', {}).get('num_questions', 'N/A'),
            "Modelos": data.get('config', {}).get('models_evaluated', 'N/A'),
            "Top-K": data.get('config', {}).get('top_k', 'N/A'),
        })

    # Configuración
    st.markdown("---")

    # Seleccionar Top-K (ahora sin selector de modelo)
    max_k = data.get('config', {}).get('top_k', 15)
    selected_k = st.slider(
        "📊 Top-K Documentos:",
        min_value=1,
        max_value=max_k,
        value=min(10, max_k),
        help="Número de documentos top a considerar para todos los modelos"
    )

    # Obtener todos los modelos
    models = list(data['results'].keys())

    # Calcular flujos para todos los modelos
    all_model_flows = {}

    # Mostrar información de debug
    with st.expander("🔍 Debug Info - Verificar datos por modelo"):
        for model in models:
            model_data = data['results'][model]

            if 'all_before_metrics' in model_data and 'all_after_metrics' in model_data:
                num_questions = len(model_data['all_before_metrics'])

                # Verificar primer documento before
                first_before = model_data['all_before_metrics'][0].get('document_scores', [])
                first_after = model_data['all_after_metrics'][0].get('document_scores', [])

                st.write(f"**{model.upper()}:**")
                st.write(f"  - Preguntas: {num_questions}")
                if first_before:
                    st.write(f"  - Primer doc BEFORE: {first_before[0].get('link', 'N/A')[:60]}... (relevante: {first_before[0].get('is_relevant', 'N/A')})")
                if first_after:
                    st.write(f"  - Primer doc AFTER: {first_after[0].get('link', 'N/A')[:60]}... (relevante: {first_after[0].get('is_relevant', 'N/A')})")

    all_model_distributions = {}

    with st.spinner("🔄 Calculando flujos de relevancia para todos los modelos..."):
        for model in models:
            model_data = data['results'][model]

            if 'all_before_metrics' not in model_data or 'all_after_metrics' not in model_data:
                st.warning(f"⚠️ {model}: No contiene métricas before/after")
                continue

            all_before = model_data['all_before_metrics']
            all_after = model_data['all_after_metrics']

            flows = calculate_aggregated_flows(all_before, all_after, selected_k)
            distribution = calculate_question_distribution(all_before, all_after, selected_k)

            all_model_flows[model] = flows
            all_model_distributions[model] = distribution

    if not all_model_flows:
        st.error("❌ Ningún modelo contiene datos válidos")
        return

    # Mostrar diagramas Sankey para cada modelo
    st.markdown("---")
    st.subheader("📈 Diagramas Sankey por Modelo")

    for model in models:
        if model not in all_model_flows:
            continue

        flows = all_model_flows[model]

        st.markdown(f"### 🤖 {model.upper()}")
        fig = create_sankey_diagram(flows, selected_k)
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar métricas clave debajo de cada diagrama
        col1, col2, col3, col4 = st.columns(4)

        net_relevant = flows['relevant_gained'] - flows['relevant_lost']
        net_irrelevant = flows['irrelevant_removed'] - flows['irrelevant_added']
        total_impact = net_relevant + net_irrelevant

        with col1:
            st.metric("🟢 Rel. Ganados", f"{flows['relevant_gained']:.1f}")
        with col2:
            st.metric("🔴 Rel. Perdidos", f"{flows['relevant_lost']:.1f}")
        with col3:
            st.metric("📈 Cambio Neto", f"{net_relevant:+.1f}")
        with col4:
            st.metric("⭐ Impacto Total", f"{total_impact:+.1f}",
                     delta=f"{total_impact:+.1f}",
                     delta_color="normal")

        st.markdown("---")

    # Tabla comparativa al final
    st.markdown("---")
    st.subheader("📊 Tabla Comparativa - Todos los Modelos")

    # Crear DataFrame para comparación
    comparison_data = {
        'Métrica': [
            'Relevantes Mantenidos',
            'Relevantes Ganados',
            'Relevantes Perdidos',
            'Cambio Neto Relevantes',
            '',  # Separador
            'Irrelevantes Mantenidos',
            'Irrelevantes Removidos',
            'Irrelevantes Añadidos',
            'Cambio Neto Irrelevantes',
            '',  # Separador
            '⭐ IMPACTO TOTAL'
        ]
    }

    for model in models:
        if model not in all_model_flows:
            comparison_data[model.upper()] = ['N/A'] * 11
            continue

        flows = all_model_flows[model]
        net_relevant = flows['relevant_gained'] - flows['relevant_lost']
        net_irrelevant = flows['irrelevant_removed'] - flows['irrelevant_added']
        total_impact = net_relevant + net_irrelevant

        comparison_data[model.upper()] = [
            f"{flows['relevant_kept']:.1f}",
            f"{flows['relevant_gained']:.1f}",
            f"{flows['relevant_lost']:.1f}",
            f"{net_relevant:+.1f}",
            '',  # Separador
            f"{flows['irrelevant_kept']:.1f}",
            f"{flows['irrelevant_removed']:.1f}",
            f"{flows['irrelevant_added']:.1f}",
            f"{net_irrelevant:+.1f}",
            '',  # Separador
            f"{total_impact:+.1f}"
        ]

    df_comparison = pd.DataFrame(comparison_data)

    # Aplicar estilo a la tabla
    st.dataframe(
        df_comparison,
        use_container_width=True,
        hide_index=True,
        height=450
    )

    # Distribución de impactos por pregunta
    st.markdown("---")
    st.subheader("📊 Distribución de Impactos por Pregunta")

    st.markdown("""
    Esta tabla muestra **cuántas preguntas mejoraron, empeoraron o permanecieron sin cambios** después del CrossEncoder.
    Si los porcentajes de mejora y empeoramiento son similares, el impacto total será cercano a cero.
    """)

    # Crear DataFrame para distribución
    distribution_data = {
        'Métrica': [
            '✅ Preguntas Mejoradas',
            '❌ Preguntas Empeoradas',
            '➖ Sin Cambio',
            '',  # Separador
            '% Mejoradas',
            '% Empeoradas',
            '% Sin Cambio'
        ]
    }

    for model in models:
        if model not in all_model_distributions:
            distribution_data[model.upper()] = ['N/A'] * 7
            continue

        dist = all_model_distributions[model]

        distribution_data[model.upper()] = [
            f"{dist['improved_count']}",
            f"{dist['worsened_count']}",
            f"{dist['unchanged_count']}",
            '',  # Separador
            f"{dist['improved_pct']:.1f}%",
            f"{dist['worsened_pct']:.1f}%",
            f"{dist['unchanged_pct']:.1f}%"
        ]

    df_distribution = pd.DataFrame(distribution_data)

    st.dataframe(
        df_distribution,
        use_container_width=True,
        hide_index=True,
        height=320
    )

    # Interpretación de distribución
    st.info("""
    💡 **Interpretación:**
    - Si **% Mejoradas ≈ % Empeoradas**: El CrossEncoder hace cambios aleatorios que se cancelan → Impacto ≈ 0
    - Si **% Mejoradas >> % Empeoradas**: El CrossEncoder está ayudando → Impacto positivo
    - Si **% Empeoradas >> % Mejoradas**: El CrossEncoder está empeorando → Impacto negativo
    """)

    # Identificar el mejor modelo
    st.markdown("---")
    st.subheader("🏆 Ranking de Modelos (por Impacto Promedio)")

    # Calcular impacto total para cada modelo
    model_impacts = {}
    for model in models:
        if model in all_model_flows:
            flows = all_model_flows[model]
            net_relevant = flows['relevant_gained'] - flows['relevant_lost']
            net_irrelevant = flows['irrelevant_removed'] - flows['irrelevant_added']
            model_impacts[model] = net_relevant + net_irrelevant

    # Ordenar modelos por impacto
    sorted_models = sorted(model_impacts.items(), key=lambda x: x[1], reverse=True)

    # Mostrar ranking
    cols = st.columns(len(sorted_models))

    for i, (model, impact) in enumerate(sorted_models):
        with cols[i]:
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"

            # Mostrar con 4 decimales para ver que muchos son ~0
            if impact > 0.5:
                st.success(f"{medal} **{model.upper()}**\n\nImpacto: **+{impact:.4f}**")
            elif impact < -0.5:
                st.error(f"{medal} **{model.upper()}**\n\nImpacto: **{impact:.4f}**")
            else:
                st.warning(f"{medal} **{model.upper()}**\n\nImpacto: **{impact:.4f}**\n\n⚠️ Casi cero")

    # Interpretación general
    st.markdown("---")
    st.subheader("💡 Interpretación General")

    best_model = sorted_models[0][0]
    best_impact = sorted_models[0][1]
    best_dist = all_model_distributions.get(best_model, {})

    if best_impact > 0.5:
        st.success(f"""
        ✅ **Mejor modelo: {best_model.upper()}** con impacto de **+{best_impact:.4f}** documentos promedio

        El CrossEncoder está funcionando mejor con este modelo, mejorando significativamente el ranking.

        **Distribución:**
        - {best_dist.get('improved_pct', 0):.1f}% preguntas mejoradas
        - {best_dist.get('worsened_pct', 0):.1f}% preguntas empeoradas
        - {best_dist.get('unchanged_pct', 0):.1f}% sin cambio
        """)
    elif best_impact < -0.5:
        st.error(f"""
        ⚠️ **El CrossEncoder está empeorando el ranking** para todos los modelos

        El mejor es {best_model.upper()} con impacto de **{best_impact:.4f}**, pero sigue siendo negativo.

        **Distribución del mejor modelo:**
        - {best_dist.get('improved_pct', 0):.1f}% preguntas mejoradas
        - {best_dist.get('worsened_pct', 0):.1f}% preguntas empeoradas
        - {best_dist.get('unchanged_pct', 0):.1f}% sin cambio

        Considera ajustar parámetros del CrossEncoder o evaluar si es necesario para tu caso de uso.
        """)
    else:
        # Near zero impact
        st.warning(f"""
        ⚠️ **El CrossEncoder tiene impacto casi nulo** para todos los modelos

        El mejor es {best_model.upper()} con impacto de **{best_impact:.4f}** (prácticamente cero).

        **Distribución del mejor modelo:**
        - ✅ {best_dist.get('improved_pct', 0):.1f}% preguntas mejoradas ({best_dist.get('improved_count', 0)} preguntas)
        - ❌ {best_dist.get('worsened_pct', 0):.1f}% preguntas empeoradas ({best_dist.get('worsened_count', 0)} preguntas)
        - ➖ {best_dist.get('unchanged_pct', 0):.1f}% sin cambio ({best_dist.get('unchanged_count', 0)} preguntas)

        **Conclusión:** Los porcentajes de mejora y empeoramiento son similares, por lo que los cambios se cancelan.
        El CrossEncoder está haciendo cambios, pero no está mejorando ni empeorando sistemáticamente el ranking.

        **Recomendaciones:**
        - Evaluar si el CrossEncoder es necesario para tu caso de uso
        - Considerar ajustar el modelo o parámetros del CrossEncoder
        - Revisar si los embeddings base son adecuados para tu dominio
        """)


if __name__ == "__main__":
    show_sankey_page()
