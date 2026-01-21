import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF para extrair texto de PDFs
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd

# Carregar variáveis do .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

st.title("Chat com PDF e Web")

# Criar abas
tab1, tab2, tab3, tab4 = st.tabs(["📄 PDF", "🌐 Web", "💬 Chat", "📊 Estatísticas"])

# --- PDF ---
with tab1:
    st.header("📄 PDF")
    uploaded_file = st.file_uploader("Envie um PDF", type="pdf")
    pdf_text = ""
    if uploaded_file is not None:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            pdf_text += page.get_text()
        st.success("PDF carregado com sucesso!")

# --- Web ---
with tab2:
    st.header("🌐 Página da Web")
    url_input = st.text_input("Cole uma URL para análise:")
    url_text = ""
    if url_input:
        try:
            page = requests.get(url_input)
            soup = BeautifulSoup(page.content, "html.parser")
            url_text = " ".join([p.get_text() for p in soup.find_all("p")])
            st.success("Conteúdo da página extraído!")
        except Exception as e:
            st.error(f"Erro ao acessar URL: {str(e)}")

# --- Chat ---
with tab3:
    st.header("💬 Chat com Gemini")
    model_choice = st.selectbox("Escolha o modelo:", ["models/gemini-2.5-flash", "models/gemini-2.5-pro"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "tokens_usage" not in st.session_state:
        st.session_state.tokens_usage = []
    if "modelo_tokens" not in st.session_state:
        st.session_state.modelo_tokens = {"models/gemini-2.5-flash": 0, "models/gemini-2.5-pro": 0}

    if st.button("🧹 Limpar histórico"):
        st.session_state.chat_history = []
        st.session_state.tokens_usage = []
        st.session_state.modelo_tokens = {"models/gemini-2.5-flash": 0, "models/gemini-2.5-pro": 0}
        st.success("Histórico limpo!")

    user_input = st.text_input("Digite sua pergunta:")
    if user_input:
        st.session_state.chat_history.append(("Você", user_input))
        prompt = user_input
        if uploaded_file and pdf_text:
            prompt += f"\n\nConteúdo do PDF:\n{pdf_text[:2000]}..."
        if url_input and url_text:
            prompt += f"\n\nConteúdo da página:\n{url_text[:2000]}..."
        try:
            response = client.models.generate_content(model=model_choice, contents=prompt)
            resposta = response.text
            tokens_info = getattr(response, "usage_metadata", None)
            if tokens_info:
                token_count = tokens_info.get("total_token_count", 0)
                resposta += f"\n\n📊 Tokens usados: {token_count}"
                st.session_state.tokens_usage.append(token_count)
                st.session_state.modelo_tokens[model_choice] += token_count
        except Exception as e:
            resposta = f"Erro: {str(e)}"
        st.session_state.chat_history.append(("Gemini 🤖", resposta))

    for autor, mensagem in st.session_state.chat_history:
        st.markdown(f"**{autor}:** {mensagem}")

    # Exportar histórico
    if st.session_state.chat_history:
        txt_content = "\n".join([f"{autor}: {mensagem}" for autor, mensagem in st.session_state.chat_history])
        st.download_button("⬇️ Baixar TXT", txt_content, file_name="historico_chat.txt")

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        textobject = c.beginText(40, 750)
        textobject.setFont("Helvetica", 12)
        for autor, mensagem in st.session_state.chat_history:
            textobject.textLine(f"{autor}: {mensagem}")
        c.drawText(textobject)
        c.save()
        buffer.seek(0)
        st.download_button("⬇️ Baixar PDF", buffer, file_name="historico_chat.pdf")

# --- Estatísticas ---
with tab4:
    st.header("📊 Estatísticas de Consumo")
    if st.session_state.tokens_usage:
        df = pd.DataFrame({"Interação": list(range(1, len(st.session_state.tokens_usage)+1)),
                           "Tokens": st.session_state.tokens_usage})
        st.line_chart(df.set_index("Interação"))

    df_modelos = pd.DataFrame.from_dict(st.session_state.modelo_tokens, orient="index", columns=["Tokens"])
    st.bar_chart(df_modelos)

    st.markdown("### 🗂️ CONVERSE COM SEUS DOCUMENTOS E PÁGINAS DA WEB")

# Rodapé
st.markdown("---")
st.markdown("Feito por **Paulo Henrique Alves Pereira** 🤖❤️")
