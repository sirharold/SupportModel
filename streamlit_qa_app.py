import streamlit as st
from utils.qa_pipeline import answer_question
from utils.metrics import summarize_ranking
from utils.weaviate_utils import cargar_credenciales, conectar, WeaviateClientWrapper
from utils.embedding import EmbeddingClient
from openai import OpenAI
import os

# Configurar credenciales
config = cargar_credenciales()
client = conectar(config)
weaviate_wrapper = WeaviateClientWrapper(client)
embedding_client = EmbeddingClient(api_key=config["OPENAI_API_KEY"], model="text-embedding-ada-002")
openai_client = OpenAI(api_key=config["OPENAI_API_KEY"])

st.set_page_config(page_title="Azure Q&A System", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Ask Azure Bot"])

if page == "Ask Azure Bot":
    st.title("Ask Azure Bot")

    title = st.text_input("Title")
    question = st.text_area("Question")

    if st.button("Ask"):
        full_query = f"{title.strip()} {question.strip()}"
        with st.spinner("Retrieving results..."):
            # Ejecutar búsqueda real
            results, debug_info = answer_question(full_query, weaviate_wrapper, embedding_client, top_k=10)
            summarize = summarize_ranking(results)

            # Mostrar resultados reales
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔍 Our Retrieval Results")
                if not results:
                    st.warning("No documents were retrieved.")
                else:
                    for i, doc in enumerate(results[:10], 1):
                        title = doc.get("title", "No Title")
                        score = doc.get("score", 0.0)
                        link = doc.get("link", "#")
                        st.markdown(f"**{i}. {title}**")
                        st.markdown(f"🔹 Score: {score:.4f}")
                        st.markdown(f"🔗 [{link}]({link})")
                        st.markdown("---")
                    st.markdown(summarize)

                
            with col2:
                st.subheader("🤖 OpenAI Expert Answer")
                prompt = (
                    "As an azure expert answer this question with the top 10 best official azure documentation pages, "
                    "adding a score to the relevance of that page).Show only the title of the page, the link and score. \n\n"
                    f"Question: {full_query}"
                )
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful Azure documentation expert."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3
                    )
                    answer = response.choices[0].message.content
                    st.markdown(answer)

                    # Extraer links desde respuesta OpenAI
                    import re
                    openai_links = re.findall(r"https?://\S+", answer)

                    # Comparar con nuestros resultados
                    our_links = [doc["link"] for doc in results[:10]]
                    matched = [link for link in openai_links if link in our_links]
                    precision = len(matched) / len(openai_links) if openai_links else 0
                    recall = len(matched) / len(our_links) if our_links else 0

                    st.subheader("📊 Comparison (Auto)")
                    st.markdown(f"🔗 Links from OpenAI: {len(openai_links)}")
                    st.markdown(f"✅ Matches with our results: {len(matched)}")
                    st.markdown(f"🎯 Precision: **{precision:.2f}**")
                    st.markdown(f"📥 Recall: **{recall:.2f}**")

                except Exception as e:
                    st.error(f"Failed to get response from OpenAI: {e}")
                    
        with st.expander("🛠️ Debug Info"):
            st.text(debug_info)