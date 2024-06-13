import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

current_dir = os.path.dirname(os.path.abspath(__file__))
llm_model = os.environ["OPENAI_API_MODEL"]
load_dotenv()

# new added --------------------------

def get_text_from_file(allfile):
    text_list = []
    # セッション内のDocx_sと比較して、新しく追加されたファイルのみをテキストに変換
    temp_file = st.session_state.file_s
    if st.session_state.file_s != allfile:
        st.session_state.file_s = allfile
        print("We got new file(s)")
        for file in allfile:
            if file not in temp_file:
                file_path = os.path.join(current_dir,"temp_uploadedfiles",file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                    
                loader = UnstructuredFileLoader(file_path)
                file_doc = loader.load()
                text_list += file_doc
                os.remove(file_path)
        file_raw_doc = text_list
    #　追加ファイルのない場合、空listにする
    else:
        file_raw_doc = []
        
    return file_raw_doc
# チャンク設定
def get_chunks(full_doc):
    text_splitter = CharacterTextSplitter(
        separator = "\n| |\n\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    
    chunks = text_splitter.split_documents(full_doc)
    
    return chunks

# ベクトルストア生成
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=chunks,embedding=embeddings)
    
    return vectorstore

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model=llm_model) # カッコ内でapi-keyの指定、モデルの指定などができます。コードの先頭にdotenvを使ったので、自動的に.envファイルからapi-keyを取得します
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatOpenAI(model=llm_model) # カッコ内でapi-keyの指定、モデルの指定などができます。コードの先頭にdotenvを使ったので、自動的に.envファイルからapi-keyを取得します
    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")
    ])
    
    stuff_documents_chian = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain,stuff_documents_chian)

def get_response(user_input):
    
    
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)    
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# ボタンのクリック状態設定の関数
def click_button():
    st.session_state.clicked = True

def main():
    
    # app config
    st.set_page_config(page_title="Chat with your files", page_icon="🤖")
    st.title("Upload files and chat with them")
    st.info("Click the :red[_Process_] button before asking questions\n(:red[_Only the first time you upload_])")
    
    # ボタンのクリック状態の初期化設定
    if "clicked" not in st.session_state:
        st.session_state.clicked = False   
    
    # ファイル存在情況の初期化
    if "file_s" not in st.session_state:
        st.session_state.file_s = [] 
    
    # ----の初期化
    if "full_doc" not in st.session_state:
        st.session_state.full_doc = []
    # ----の初期化
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
     
    # サイドバーの設定
    with st.sidebar:
        # file upload
        st.title("Settings")
        st.header("",divider="rainbow")
        
        st.subheader("_Upload_ a :rainbow[FILE] :books:")
        allfile = st.file_uploader("Upload your FILE here and click on '_Process_'",accept_multiple_files=True,type=["xlsx","docx","pdf"])
        
        st.header("",divider="blue")
    
    # ボタン
    st.button("Process", on_click=click_button)
    # 最初アップロード時にクリックすれば十分
    # クリックされたら、その状態をセッションに保存  
    if st.session_state.clicked:
        
        # latest ---------------------------------------------
        if allfile == []:
            file_existance = False
            file_raw_doc = []
            st.info(":red[_Enter a URL or Upload some files_]")
        
        else:
            file_existance = True
            file_raw_doc = get_text_from_file(allfile)
            full_doc_add = file_raw_doc
            st.session_state.full_doc += full_doc_add
        # ------------------------------------------------  
                
            print (f"Added doc: {full_doc_add}")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                    AIMessage(content = "Hello, I am a a bot"),
                ]
            # 料金節約のために、追加のドキュメントがあるときのみ、Embeddingを執行
            if full_doc_add != []:
                print("new file(s) added")
                print(f"Full doc: {st.session_state.full_doc}")
                chunks = get_chunks(st.session_state.full_doc)
                st.session_state.vector_store = get_vectorstore(chunks)
            else:
                print(st.session_state.full_doc)
                print("no file added")
            
            # ユーザーのクエリーで回答を生成
            user_query = st.chat_input("Try asking something about your files ")
        
            if user_query is not None and user_query != "":
                response = get_response(user_query)
                # ユーザーの質問とAIの回答をセッションに入れる
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=response))

            # 画面上で表示
            for message in st.session_state.chat_history:
                if isinstance(message,AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                elif isinstance(message,HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)
                     
if __name__ == "__main__":
    main()
    

