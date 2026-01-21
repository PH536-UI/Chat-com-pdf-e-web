import streamlit as st
from ingestion import ingest_documents
from rag_pipeline import RAGChatBot
from llm_setup import get_llm, get_embedding
import os
import shutil
import time

# Configuração da página
st.set_page_config(page_title="Chat com PDF e Web", layout="wide", page_icon="🤖")

st.title("🤖 Chat com Documentos e Páginas Web")
st.subheader("💫 Carregue seu conteúdo e comece a conversar! 💫")

# ---------------- Sidebar (Configurações) ----------------
with st.sidebar:
    st.header("Configurações")
    
    # Input da API Key
    api_key_input = st.text_input("Digite sua Gemini API Key", type="password")
    if api_key_input:
        st.session_state["gemini_api_key"] = api_key_input

    # Escolha da fonte de dados
    source_type = st.radio("Selecione a fonte:", ("Arquivo PDF", "URL de Website"))

    if source_type == "Arquivo PDF":
        uploaded_file = st.file_uploader("Upload do PDF", type="pdf")
        if uploaded_file:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("PDF carregado com sucesso!")
    else:
        url = st.text_input("Digite a URL do site")
        if url:
            st.session_state["url"] = url

    # Botão de Processamento (Ingestão)
    if st.button("Processar e Iniciar Chat"):
        if "gemini_api_key" not in st.session_state or not st.session_state["gemini_api_key"]:
            st.error("Por favor, insira a sua API Key do Gemini!")
        else:
            with st.spinner("Analisando documentos..."):
                try:
                    # 1. Obtém a função de embedding
                    embeddings = get_embedding(st.session_state["gemini_api_key"])
                    
                    # Define o caminho da fonte
                    source = "temp.pdf" if source_type == "Arquivo PDF" else st.session_state.get("url")
                    
                    # LINHA 51 CORRIGIDA: Enviando source, embeddings e a api_key
                    vectorstore = ingest_documents(source, embeddings, st.session_state["gemini_api_key"])
                    st.session_state["vector_store"] = vectorstore

                    # 2. Inicializa o modelo de linguagem (LLM)
                    llm = get_llm(st.session_state["gemini_api_key"])
                    
                    # 3. Inicializa o robô de RAG com o banco e o cérebro (LLM)
                    st.session_state["rag_bot"] = RAGChatBot(vectorstore, llm)
                    
                    st.success("Ingestão concluída! O chat está pronto.")
                except Exception as e:
                    st.error(f"Erro durante o processamento: {e}")

# ---------------- Interface de Chat ----------------

# Inicializa o histórico se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input de mensagem do usuário
if prompt := st.chat_input("Pergunte algo sobre o conteúdo..."):
    if "rag_bot" not in st.session_state:
        st.warning("Por favor, processe um documento primeiro na barra lateral.")
    else:
        # Adiciona mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera e exibe a resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    response = st.session_state["rag_bot"].chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {e}")

# Rodapé informativo
st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido com Gemini 1.5 Flash, LangChain e Streamlit.")
