import time
import gc

class RAGChatBot:
    def __init__(self, vectorstore, llm):
        # Imports diretos dentro do init para garantir que o Python 
        # localize os módulos recém-instalados
        from langchain.chains.retrieval import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Template simplificado para evitar erros de parsing
        system_prompt = "Responda apenas com base no contexto: {context}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Montagem da Chain
        combine_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, combine_chain)

    def chat(self, question: str) -> str:
        try:
            # Invoca a cadeia de resposta
            response = self.chain.invoke({"input": question})
            return response["answer"]
        except Exception as e:
            return f"Erro ao gerar resposta: {str(e)}"

    def close(self):
        # Limpeza básica de memória
        gc.collect()
