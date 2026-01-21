import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def ingest_documents(source: str, embeddings, gemini_api_key: str):
    """
    Lê um PDF ou URL, divide em pedaços e salva no banco de vetores Chroma.
    """
    try:
        # 1. Carregamento do conteúdo
        if source.endswith(".pdf") or source == "temp.pdf":
            loader = PyPDFLoader(source)
        else:
            # Garante que o web_path receba apenas a string da URL
            loader = WebBaseLoader(web_path=(source,))
        
        documents = loader.load()

        # 2. Divisão do texto em pedaços (Chunks)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # 3. Criação/Atualização do Banco de Vetores (Chroma)
        # O persist_directory garante que os dados não sumam imediatamente
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_name="pdf_chat_collection"
        )
        
        return vectorstore

    except Exception as e:
        raise Exception(f"Erro na ingestão de documentos: {str(e)}")
