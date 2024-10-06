import os
import random
from pathlib import Path
import numpy as np
import streamlit as st
from datetime import datetime
from st_on_hover_tabs import on_hover_tabs
from streamlit_lottie import st_lottie

from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
import pandas as pd
import os

from langchain_groq import ChatGroq

from main import do_stt, do_tts, open_chat, load_vectordb, add_document, rag_agent, show_vectordb


# parent_dir = Path(__file__).parent
# parent_dir = str(parent_dir).replace("\\", "/")
# base_dir = str(parent_dir) + "/data"

st.set_page_config(page_title="AI CAPTAIN", page_icon="🧭", layout="wide")
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
lotties = ["https://lottie.host/48a0d2f2-8b57-4453-b378-8393a4e05447/V5a1f8reB2.json", 
           "https://lottie.host/ae91e0f4-5e9d-47a5-ac54-b2753ff0b713/K9vatEy9XH.json", 
           "https://lottie.host/12735102-8237-425d-bd0e-7bab7c0c070e/mGq3mO76qG.json",
           "https://lottie.host/906242ca-4fff-455b-b761-8f23d7ad42ea/bkpUnbEmu0.json"]
random_lottie = random.choice(lotties)

st.markdown(
            """
        <style>
            .st-emotion-cache-1c7y2kd {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )


### Variables #################################################################
def make_state_variables():
    if "question" not in st.session_state: st.session_state.question=""
    if "record" not in st.session_state: st.session_state.record=""
    if "lang_mode" not in st.session_state: st.session_state.lang_mode="en"
    if "voice" not in st.session_state: st.session_state.voice="David"
    if "output" not in st.session_state: st.session_state.output=""
    if "time_delta" not in st.session_state: st.session_state.time_delta=""
    if "messages" not in st.session_state: st.session_state.messages= [{"role": "assistant", "content": "How can I help you?"}]
    if "vectorstore" not in st.session_state: st.session_state.vectorstore=""
    if "vectordb_docs" not in st.session_state: st.session_state.vectordb_docs=""
    if "retrieval_docs" not in st.session_state: st.session_state.retrieval_docs=""
    if "rag_messages" not in st.session_state: st.session_state.rag_messages= [{"role": "assistant", "content": "How can I help you?"}]
    if "df" not in st.session_state: st.session_state.df=""
    if "data_messages" not in st.session_state: st.session_state.data_messages= [{"role": "assistant", "content": "How can I help you?"}]

make_state_variables()

model_name = "llama3.2:latest"
template = '''you are an smart AI assistant.
    Generate compact and summarized answer to {query} with numbering kindly and shortly.
    if there are not enough information to generate answers, just return "Please give me more information" or ask a question for additional information.
    for example, 'could you give me more detailed information about it?'
    '''

#### Functions ###############################################################
def calculate_time_delta(start, end):
    delta = end - start
    return delta.total_seconds()

import re
import ast
def extract_metadata(input_string):  # retrieval docs re-ranking and add metadata
    # Use regex to extract the page_content
    page_content_match = re.search(r"page_content='(.+?)'\s+metadata=", str(input_string), re.DOTALL)
    if page_content_match:
        page_content = page_content_match.group(1)
    else:
        page_content = None

    # Use regex to extract the metadata dictionary
    metadata_match = re.search(r"metadata=(\{.+?\})", str(input_string))
    if metadata_match:
        metadata_str = metadata_match.group(1)
        # Convert the metadata string to a dictionary
        metadata = ast.literal_eval(metadata_str)
    else:
        metadata = None
    return page_content, metadata

def make_sidebar():
    with st.sidebar:
        tabs = on_hover_tabs(tabName=['Home', 'Open Chat', 'VectorStore', 'Rag Agent', 'Data Agent', 'ReAct Agent', 'Prompt Engineering'], 
                            iconName=['⚓', '🐳', '🐬', '🦭', '🦐', '🦈', '🦑', '🦀'], default_choice=0)
        return tabs

def make_home():
    st.title("🧭 AI CAPTAIN")
    st.markdown("---")
    col11, col12 = st.columns([3, 7])
    with col11: st_lottie(random_lottie, height=500)
    with col12: 
        col111, col112, col113 = st.columns(3)
        with col111: 
            container = st.container(border=True, height=250)
            container.title(":gray[Open Chat]")
            container.markdown("#### :gray[Q&A Chat based on Pre-trained Knowledge]")
            container.markdown(":gray[For technical and medical advice in general]")
        with col112: 
            container = st.container(border=True, height=250)
            container.title("VectorStore")
            container.markdown("#### Manage Documents as Vector")
            container.markdown("Similarity Search and Add/Delete Documents")
        with col113: 
            container = st.container(border=True, height=250)
            container.title("Rag Agent")
            container.markdown("#### Q&A Chat based on VetorStore Knowledge")
            container.markdown("Get your techical advice without hallucination")
        
        col121, col122, col123 = st.columns(3)
        with col121: 
            container = st.container(border=True, height=250)
            container.title("Data Agent")
            container.markdown("#### Q&A Chat based on Structured Datafame")
            container.markdown("Monitor and Analyze your data in Dataframe")
        with col122: 
            container = st.container(border=True, height=250)
            container.title("ReAct Agent")
            container.markdown("#### Reasoning and Action Service using Langgraph")
            container.markdown("Organize Nodes and Chains for Precise Generation")
        with col123: 
            container = st.container(border=True, height=250)
            container.title("Prompt Engineering")
            container.markdown("#### Edit Prompt in Various Style")
            container.markdown("Get Better Response through Stylish Prompt")

def make_openchat():
    st.title("Open Chat")
    st.markdown("---")
    with st.expander("⚙️ Settings"):
        col21, col22, col23, col24 = st.columns(4)
        with col21: tts_mode = st.toggle("TTS Mode", value=False, help="Voice playback")
        with col22: lang_mode = st.toggle("Korean Language", value=False, help="Default = English")
        with col23: offline_mode = st.toggle("Offline Mode", value=False, help="Use Local Ollama Model")
        with col24: tool_mode = st.toggle("Agent Mode", value=False, help="if necessary, do web search")

    if lang_mode: 
        st.session_state.lang_mode = "ko"
        st.session_state.voice = "Heami"

    st.session_state.question = ""
    question = st.chat_input(placeholder="✏️ Input your question")
    col211, col212, col213 = st.columns([2, 2, 8])
    with col211: st.session_state.record = st.button("🎤 Record", use_container_width=True, help="Record for STT(Speech to Text)")
    with col212: init_btn = st.button("🗑️ Init Chat History", use_container_width=True)
    if init_btn: 
        st.session_state.messages= [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state.time_delta = ""
    with st.spinner("Processing..."):
        if question: st.session_state.question = question      
        elif st.session_state.record: st.session_state.question = do_stt(language=st.session_state.lang_mode)

    if len(st.session_state.question) >1:
        start_time = datetime.now()
        st.session_state.messages.append({"role": "user", "content": st.session_state.question})
        with st.spinner("Processing..."):
            st.session_state.output = open_chat(template=template, llm_name=model_name, question=st.session_state.question, offline_mode=offline_mode, tool_mode=tool_mode)
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.output})
        end_time = datetime.now()
        st.session_state.time_delta = calculate_time_delta(start_time, end_time)
        
        placeholder = st.empty()
        placeholder.info(st.session_state.output)

        if tts_mode:
            do_tts(st.session_state.output, voice=st.session_state.voice, language=st.session_state.lang_mode)

        placeholder.empty()

    with st.container(height=500, border=False):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="🐬").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
        try: st.success(f"⏱️ Latency(Sec) : {np.round(st.session_state.time_delta,2)}")
        except: pass

def make_similarity_search(my_query:str, embed_model_name:str="bge-m3:latest", db_path:str = "chroma_index"):
    vectordb =  load_vectordb(db_path=db_path, embed_model_name=embed_model_name)
    result = vectordb.similarity_search_with_score(my_query)
    return result

def make_vectordb():
    st.title("VectorStore")
    st.markdown("---")

    tab31, tab32, tab33 = st.tabs(["Similarity Search", "Add Documents", "Delete Documents"])

    with tab31:
        lv1, lv2, st.session_state.vectordb_docs, result = show_vectordb(path="chroma_index")
        col31, col32, col33 = st.columns(3)
        with col31: sel1 = st.multiselect("First Div", lv1)
        with col32: sel2 = st.multiselect("Second Div", lv2)
        with col33: sel3 = st.multiselect("File Name", st.session_state.vectordb_docs)
        with st.expander(f"Vector DataBase - {result.shape}", expanded=True):
            if len(sel1)==0 and len(sel2)==0 and len(sel3)==0:
                st.dataframe(result)
            else: 
                df = result.loc[(result["first_div"].isin(sel1))|(result["second_div"].isin(sel2))|(result["filename"].isin(sel3))]
                st.dataframe(df)
    
        text = st.text_input("Input your target sentence for Similarity Search")
        btn33 = st.button("Similarity Search")
        with st.spinner("Processing.."):
            if btn33:
                result = make_similarity_search(my_query=text, db_path="chroma_index")
                for content in result:
                    with st.container(border=True):
                        st.markdown(f"Similarity Score: {np.round(content[1],2)}")
                        st.markdown(content[0].page_content)
                        st.markdown(f"MetaData: :blue[{content[0].metadata}]")
                        
    with tab32:
        uploaded_file = st.file_uploader("📎Upload your file")
        if uploaded_file:
            temp_dir = "./data/user" 
            path1 = os.path.join(temp_dir, uploaded_file.name)
            path1 = str(path1).replace("\\", "/")
            with open(path1, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner("Processing..."):
                if st.button("Add New Document"):
                    add_document(path=path1)
                    st.success("VectorStore Uadate is Completed")

    with tab33:
        delete_doc = st.selectbox("Target Document", st.session_state.vectordb_docs, index=None)
        try:
            vectordb = load_vectordb(db_path="chroma_index", embed_model_name="bge-m3:latest")
            del_ids = vectordb.get(where={'File Name':delete_doc})["ids"]
            with st.spinner("processing..."):
                if st.button("Delete All Ids"):
                    vectordb.delete(del_ids)
                    st.info("All Selected Ids were Deleted")
        except:
            st.empty()

def make_rag_agent():
    st.title("Rag Agent")
    st.markdown("---")

    with st.expander("⚙️ Settings"):
        col41, col42, col43, col44 = st.columns(4)
        with col41: tts_mode = st.toggle("TTS Mode", value=False, key="werwed", help="Voice playback")
        with col42: lang_mode = st.toggle("Korean Language", value=False, help="Default = English", key="wefsesf")
        with col43: json_mode = st.toggle("Json Style Answer", value=True, key="wefsdfsdfesf", help="Json Format Response")
        with col44: offline_mode = st.toggle("Offline Mode", value=False, key="wesdfsd", help="Use Local Ollama Model")

    st.session_state.vectorstore = load_vectordb(db_path="chroma_index", embed_model_name="bge-m3:latest")

    if lang_mode: 
        st.session_state.lang_mode = "ko"
        st.session_state.voice = "Heami"

    st.session_state.question = ""
    query = st.chat_input(placeholder="✏️ Input your question", key="234wef")

    col211, col212, col213 = st.columns([2, 2, 8])
    with col211: st.session_state.record = st.button("🎤 Record", use_container_width=True, help="Record for STT(Speech to Text)")
    with col212: init_btn = st.button("🗑️ Init Chat History", use_container_width=True)
    if init_btn: 
        st.session_state.rag_messages= [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state.time_delta, st.session_state.retrieval_docs = "", ""
    with st.spinner("Processing..."):
        if query: st.session_state.question = query      
        elif st.session_state.record: st.session_state.question = do_stt(language=st.session_state.lang_mode)

    if len(st.session_state.question) >1:
        start_time = datetime.now()
        st.session_state.rag_messages.append({"role": "user", "content": st.session_state.question})
        with st.spinner("Processing..."):
            st.session_state.retrieval_docs, st.session_state.output = rag_agent(query=st.session_state.question, vectorstore=st.session_state.vectorstore, json_style=json_mode, offline_mode=offline_mode)
        st.session_state.rag_messages.append({"role": "assistant", "content": st.session_state.output})
        end_time = datetime.now()
        st.session_state.time_delta = calculate_time_delta(start_time, end_time)
        
        placeholder = st.empty()
        placeholder.info(st.session_state.output)

        if tts_mode:
            do_tts(st.session_state.output, voice=st.session_state.voice, language=st.session_state.lang_mode)

        placeholder.empty()

    with st.container(height=500, border=False):
        for msg in st.session_state.rag_messages:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="🐬").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

        img_dict = dict()
        for d in st.session_state.retrieval_docs:
            page_content, metadata = extract_metadata(d)
            if metadata["File Name"] not in img_dict.keys():
                img_dict[metadata["File Name"]] =[]
                img_dict[metadata["File Name"]].append(metadata["Page"])
            else:
                img_dict[metadata["File Name"]].append(metadata["Page"])

        with st.expander("Retrieval Documents"):
            col441, col442 = st.columns([6, 4])
            with col441:
                for doc in st.session_state.retrieval_docs:
                    with st.container(border=True):
                        st.markdown(doc.page_content)
                        st.markdown(f":green[{doc.metadata}]")
            with col442:
                base_img_path = "./images/"
                target_imgs = []
                for k in img_dict.keys():
                    img_ids = img_dict[k]
                    for img_id in img_ids:
                        imgfile_name = base_img_path + str(k) + "/" +str(k) + "_" + str(img_id)+".png"
                        target_imgs.append(imgfile_name)
                col31, col32 =st.columns(2)
                with col31: image_show_check = st.checkbox("Show Images", value=True)
                with col32: page_num = st.number_input(f"Page Order (max:{max(0,len(target_imgs)-1)})", min_value=0, max_value=max(0, len(target_imgs)-1))
                if image_show_check:
                    try: st.image(target_imgs[page_num], caption=target_imgs[page_num], width=600)
                    except: st.info("No Image for this page")

        try: st.success(f"⏱️ Latency(Sec) : {np.round(st.session_state.time_delta,2)}")
        except: pass


file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False

@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

def make_data_agent():
    st.title("Data Agent")
    st.markdown("---")

    with st.expander("Upload File", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload a Data file",
            type=list(file_formats.keys()),
            help="Various File formats are Support",
            on_change=clear_submit,
            )
    
    # if not uploaded_file:
    #     st.warning(
    #         "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
    #     )

    st.session_state.df = pd.read_csv("./data/electricityConsumptionAndProductioction.csv")
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
    with st.expander("DataFrame", expanded=True):
        st.dataframe(st.session_state.df, use_container_width=True)

    # if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    #     st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.data_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input(placeholder="What is this data about?")
    if prompt: 
        st.session_state.data_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

    llm = ChatGroq(temperature=0, model_name= "llama-3.2-90b-text-preview")

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        st.session_state.df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        allow_dangerous_code=True
        )
    
    if prompt != None:
        with st.chat_message("assistant"):
            # st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            # response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            response = pandas_df_agent.invoke(prompt)
            st.session_state.data_messages.append({"role": "assistant", "content": response["output"]})
            st.write(response["output"])


if __name__ == "__main__":
    
    ### [Start] SIDEBAR ###################################################################
    tabs = make_sidebar()

    ### [Start] MAIN ###################################################################
    if tabs =='Home': make_home()

    elif tabs == 'Open Chat': make_openchat()       

    elif tabs == 'VectorStore': make_vectordb()

    elif tabs == 'Rag Agent': make_rag_agent()

    elif tabs == 'Data Agent': make_data_agent()
        
    elif tabs == 'ReAct Agent':
        st.title("ReAct Agent")
        st.write('Name of option is {}'.format(tabs))

    elif tabs == 'Prompt Engineering':
        st.title("Prompt Engineering")
        st.write('Name of option is {}'.format(tabs))

