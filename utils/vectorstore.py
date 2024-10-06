
import chromadb
import pandas as pd

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MyVectorStore:
    def create_vectorstore(parsed_text:str, chunk_size:int=1000, chunk_overlap:int=200):  # VectorDB생성 및 저장
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splitted_texts = text_splitter.split_documents(parsed_text)
        embed_model = OllamaEmbeddings(model="bge-m3:latest")
        db=Chroma.from_documents(splitted_texts, embedding=embed_model, persist_directory="chroma_index")
        return db
    
    def create_retriever(vectorstore, k:int=3):
        retriever = vectorstore.as_retriever(search_kwargs={'k': k})
        return retriever
    
    def read_vectordb(db_path:str, embed_model_name:str):
        embed_model = OllamaEmbeddings(model=embed_model_name)
        vectordb = Chroma(persist_directory=db_path, embedding_function=embed_model)
        return vectordb

    
    def read_vectordb_as_df(path:str):
        client = chromadb.PersistentClient(path=path)
        for collection in client.list_collections():
            data = collection.get(include=['embeddings', 'documents', 'metadatas'])
            df = pd.DataFrame({"ids":data["ids"], 
                            #    "embeddings":data["embeddings"], 
                               "metadatas":data["metadatas"], 
                               "documents":data["documents"]})
            df["first_div"] = df["metadatas"].apply(lambda x: x["First Division"])
            df["second_div"] = df["metadatas"].apply(lambda x: x["Second Division"])
            df["filename"] = df["metadatas"].apply(lambda x: x["File Name"])
            df = df[["ids", "first_div", "second_div","filename","documents", "metadatas"]]
        return df
    



     