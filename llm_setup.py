from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", # Tente "gemini-1.5-flash" sem o models/ na frente
        google_api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True # Importante para evitar erros de permissão
    )

def get_embedding(api_key):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", # Nome oficial atualizado
        google_api_key=api_key
    )
