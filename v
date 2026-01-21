import streamlit as st

class RAGChatBot:
    def __init__(self, vectorstore, llm):
        # Importação ultra-específica para evitar o erro de 'chains'
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains.retrieval import create_retrieval_chain
        from langchain_core.prompts import ChatPromptTemplate
        
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Template de prompt seguindo o padrão novo
        system_prompt = (
            "Você é um assistente prestativo. Use o seguinte contexto para responder à pergunta. "
            "Se não souber a resposta, diga que não sabe.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Criando a lógica de RAG
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def chat(self, question: str):
        try:
            response = self.chain.invoke({"input": question})
            return response["answer"]
        except Exception as e:
            return f"Erro na resposta: {str(e)}"
