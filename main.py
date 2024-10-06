import os
import sys
import time
import streamlit as st

from tqdm import tqdm
from dotenv import load_dotenv
from typing import Optional
from utils import (AudioToTextRecorder, TextToAudioStream, SystemEngine, 
                   MyParser, main_filepath_extractor, MyVectorStore, MyRag, chatbot_with_tools
                   )
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

template = '''you are an smart AI assistant.
    Generate compact and summarized answer to {query} with numbering kindly and shortly.
    if there are not enough information to generate answers, just return "Please give me more information" or ask a question for additional information.
    for example, 'could you give me more detailed information about it?'
    '''

base_ollama = "llama3.2:latest"
embed_model = "bge-m3:latest"

  

if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
    from torchaudio._extension.utils import _init_dll_path
    _init_dll_path()

def do_stt(stt_model:str="small", language:str="en"):
    with AudioToTextRecorder(spinner=False, 
                            model=stt_model,   #'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
                            language=language, 
                            enable_realtime_transcription=False,
                            ) as recorder:
        print("Say something...")
        st.info("Say something...")
        input_voice = recorder.text()
        print(input_voice)
        return input_voice

def tts_generator(output:str): yield output

def do_tts(response:str, voice:str="David", language:str="en"):
    TextToAudioStream(SystemEngine(voice=voice)).feed(tts_generator(response)).play(language=language)
    print("TTS is Completed")

def online_openchat(template:str, question:str):
    create_greeting_prompt = partial(create_prompt, template)
    prompt = create_greeting_prompt(query=question)
    prompt = ChatPromptTemplate.from_template(prompt)
    query = {"query": question}
    llm = ChatGroq(temperature=0, model_name= "llama-3.2-90b-text-preview")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(query)
    print(llm)
    return response

def offline_openchat(template:str, question:str, llm_name:str=base_ollama):
    create_greeting_prompt = partial(create_prompt, template)
    prompt = create_greeting_prompt(query=question)
    prompt = ChatPromptTemplate.from_template(prompt)
    query = {"query": question}
    llm = ChatOllama(model=llm_name)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(query)
    print(llm)
    return response

def online_agent_chatbot(question:str):
    result = chatbot_with_tools(user_input=question)
    if len(result) >=2: result1 = f"{result[-1]} \n\n '>>> Source' : {result[-2]}"
    else: result1 = f"{result[-1]}"
    return result1



from functools import partial
def create_prompt(template, **kwargs):
    return str(template).format(**kwargs)

def open_chat(template:str, question:str, llm_name:str=base_ollama, offline_mode:bool=False, tool_mode:bool=False):
    if offline_mode: return offline_openchat(template=template, question=question, llm_name=llm_name)
    elif tool_mode: return online_agent_chatbot(question=question)
    else: 
        try: return online_openchat(template=template, question=question)
        except:return offline_openchat(template=template, question=question, llm_name=llm_name)

def show_vectordb(path:str):
    df = MyVectorStore.read_vectordb_as_df(path=path)
    lv1 = df["first_div"].unique().tolist()
    lv1.sort()
    lv2 = df["second_div"].unique().tolist()
    lv2.sort
    docs = df["filename"].unique().tolist()
    docs.sort()
    return lv1, lv2, docs, df

def load_vectordb(db_path:str, embed_model_name:str=embed_model):
    db = MyVectorStore.read_vectordb(db_path=db_path, embed_model_name=embed_model_name)
    return db

def add_document(path:str):
    result = MyParser.parser(path=path)
    db = MyVectorStore.create_vectorstore(parsed_text=result)
    return db


def rag_agent(query:str, vectorstore, json_style:bool=True, offline_mode:bool=False):
    retriever = MyVectorStore.create_retriever(vectorstore=vectorstore)
    context, answer = MyRag.rag_chat(query=query, retriever=retriever, json_style=json_style, offline_mode=offline_mode)
    return context, answer


def update_vectorstore_admin_mode(lv1_path:str):
    path = lv1_path
    total_results = main_filepath_extractor(path=path)  # 모든 PDF의 Full Path를 리스트에 담기
    print(total_results)

    for path in tqdm(total_results):
        result = MyParser.parser(path=path)
        print(f">>>>> Parsing Completed - {len(result)} - {path}")
        time.sleep(3)

        db = MyVectorStore.create_vectorstore(parsed_text=result)
        print(f">>>>> VectorStore Completed - {db} - {path}")
        time.sleep(3)
    
    print("Parsing for All Documents is completed!")

def check_vectorstore(db_path:str):
    result = MyVectorStore.read_vectordb_as_df(db_path)
    print(result)

if __name__ == "__main__":
    # add documents into VectorStore
    update_vectorstore_admin_mode(lv1_path="./documents")

    # See VectorStore
    # check_vectorstore(db_path="chroma_index")

    # lv1, lv2, docs, df = show_vectordb(path="chroma_index")
    # print(lv1, lv2, df)
    pass




 