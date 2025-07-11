import streamlit as st
import pandas as pd
import plotly.express as px
import re
from urllib.parse import urlparse
from utils.weaviate_utils_improved import WeaviateConfig, get_weaviate_client, WeaviateClientWrapper

AZURE_LINK_PATTERN = r"https://learn\.microsoft\.com/[^\s\]\),]+"
AZURE_ROOT = "https://learn.microsoft.com/en-us/azure/"

@st.cache_resource
def _init_wrapper():
    config = WeaviateConfig.from_env()
    client = get_weaviate_client(config)
    wrapper = WeaviateClientWrapper(
        client,
        documents_class="Documentation",
        questions_class="Questions",
        retry_attempts=3,
    )
    return wrapper, client


def _fetch_all(collection, props=None, limit=20000):
    results = collection.query.fetch_objects(limit=limit, return_properties=props)
    return [obj.properties for obj in results.objects]


def show_data_analysis_page():
    """Display exploratory data analysis for Documentation and Questions."""
    st.subheader("📈 Análisis Exploratorio de Datos")
    wrapper, client = _init_wrapper()

    # --- Documentation stats ---
    with st.spinner("Cargando documentación..."):
        docs = _fetch_all(wrapper._docs_collection)
    df_docs = pd.DataFrame(docs)
    st.markdown("### 📄 Documentación")
    if df_docs.empty:
        st.info("No se encontraron documentos.")
    else:
        df_docs["content"] = df_docs["content"].fillna("")
        df_docs["link"] = df_docs["link"].fillna("")
        df_docs["chunk_len"] = df_docs["content"].str.len()

        page_lengths = (
            df_docs.groupby("link")["content"].apply(lambda x: "".join(x)).str.len()
        )
        pages_with_content = int((page_lengths > 0).sum())
        pages_without_content = int((page_lengths == 0).sum())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chunks", len(df_docs))
        with col2:
            st.metric("Documentos únicos", len(page_lengths))
        with col3:
            st.metric("Con contenido", pages_with_content)
            st.metric("Sin contenido", pages_without_content)

        st.markdown("**Tamaño por chunk**")
        st.dataframe(df_docs["chunk_len"].describe())
        fig = px.histogram(df_docs, x="chunk_len", nbins=30)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Tamaño por documento (sin chunks)**")
        st.dataframe(page_lengths.describe())
        fig2 = px.histogram(page_lengths, nbins=30)
        st.plotly_chart(fig2, use_container_width=True)

    # --- Questions stats ---
    with st.spinner("Cargando preguntas..."):
        questions = _fetch_all(wrapper._questions_collection)
    df_q = pd.DataFrame(questions)
    st.markdown("---")
    st.markdown("### ❓ Preguntas")
    if df_q.empty:
        st.info("No se encontraron preguntas.")
    else:
        df_q["question_content"] = df_q["question_content"].fillna("")
        df_q["accepted_answer"] = df_q["accepted_answer"].fillna("")
        df_q["question_len"] = df_q["question_content"].str.len()
        df_q["answer_len"] = df_q["accepted_answer"].str.len()
        df_q["links"] = df_q["accepted_answer"].apply(lambda t: re.findall(AZURE_LINK_PATTERN, t))
        df_q["num_links"] = df_q["links"].apply(len)
        df_q["has_links"] = df_q["num_links"] > 0
        df_q["has_root_link"] = df_q["links"].apply(
            lambda ls: any(link.startswith(AZURE_ROOT) for link in ls)
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total preguntas", len(df_q))
        with col2:
            st.metric("Preguntas con links", int(df_q["has_links"].sum()))
        with col3:
            st.metric("Links a raíz", int(df_q["has_root_link"].sum()))

        st.markdown("**Frecuencia de cantidad de links**")
        link_freq = df_q[df_q["has_links"]]["num_links"].value_counts().sort_index()
        st.bar_chart(link_freq)

        st.markdown("**Tamaño de preguntas (con/sin links)**")
        stats_links = df_q[df_q["has_links"]]["question_len"].describe()
        stats_no = df_q[~df_q["has_links"]]["question_len"].describe()
        st.dataframe(pd.DataFrame({"Con links": stats_links, "Sin links": stats_no}))

        st.markdown("**Tamaño de respuestas aceptadas**")
        st.dataframe(df_q["answer_len"].describe())
        fig3 = px.histogram(df_q, x="answer_len", nbins=30)
        st.plotly_chart(fig3, use_container_width=True)

