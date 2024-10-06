from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, RetrievalQA

base_ollama = "llama3.2:latest"

class MyRag:
    def rag_chat(query:str, retriever, json_style:bool=True, offline_mode:bool=False):

        if json_style:
            system_prompt = ('''
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

        {context}
        Please provide your answer in the following JSON format: 
        {{
        "answer": "Your detailed answer here",\n
        "keywords: [list of important keywords from the context] \n
        "sources": "Direct sentences or paragraphs from the context that support your answers. ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. DO NOT ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING."
        }}
        The JSON must be a valid json format and can be read with json.loads() in Python. Answer:
                            ''')
        
        else: 
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
                )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
                ]
            )
        if offline_mode:
            model = ChatOllama(temperature=0, model= "llama3.2:latest")
            question_answer_chain = create_stuff_documents_chain(model, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": f"{query}"})
            return response["context"], response["answer"]
        else:
            try:
                model = ChatGroq(temperature=0, model_name= "llama-3.2-90b-text-preview")
                question_answer_chain = create_stuff_documents_chain(model, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                response = rag_chain.invoke({"input": f"{query}"})
                return response["context"], response["answer"]
            except:
                model = ChatOllama(temperature=0, model= "llama3.2:latest")
                question_answer_chain = create_stuff_documents_chain(model, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                response = rag_chain.invoke({"input": f"{query}"})
                return response["context"], response["answer"]

