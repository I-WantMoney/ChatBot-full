import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


llm_model = os.environ["OPENAI_API_MODEL"]
load_dotenv()

# PDFからテキスト抽出
def get_text_from_pdf(pdf_file):
    text = ""
    # セッション内のpdf_sと比較して、新しく追加されたファイルのみをテキストに変換
    temp_pdfs = st.session_state.pdf_s
    if st.session_state.pdf_s != pdf_file:
        st.session_state.pdf_s = pdf_file
        print("we got new PDF file(s)")
        for pdf in pdf_file:
            if pdf not in temp_pdfs:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    text+=page.extract_text()
        
        pdf_raw_doc = [Document(page_content=text)]
    else:
        pdf_raw_doc = [Document(page_content="")]
    
    return pdf_raw_doc

# スピーチをテキストへ変換
def get_text_from_mp3(mp3_file):
    client = OpenAI()
    text = ""
    #セッション内のmp3_sと比較して、新しく追加されたファイルのみをテキストに変換
    temp_mp3s = st.session_state.mp3_s
    if st.session_state.mp3_s != mp3_file:
        st.session_state.mp3_s = mp3_file
        print("we got new mp3 file(s)")
        for mp3 in mp3_file:
            if mp3 not in temp_mp3s:
                print(mp3)
                audio_file  = mp3
                transcript = client.audio.transcriptions.create(
                    model = "whisper-1",
                    file = audio_file,
                    response_format = "text"
                )
                text += transcript
        audio_raw_doc = [Document(page_content=text)]
    else:
        audio_raw_doc = [Document(page_content="")]
    
    # print(st.session_state.mp3_s)
    return audio_raw_doc

# ウェブサイトからテキスト抽出
def get_text_from_url(url):
    if st.session_state.url_s != url:
        st.session_state.url_s = url
        loader = WebBaseLoader(url)
        url_doc = loader.load()
    else:
        url_doc = [Document(page_content="")]
    
    return url_doc

# チャンク設定
def get_chunks(full_doc):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
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
    llm = ChatOpenAI(model=llm_model)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatOpenAI(model=llm_model)
    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")
    ])
    
    stuff_documents_chian = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain,stuff_documents_chian)

def get_response(user_input):
    ## create conversation chain
    # vector_store = get_vectorstore_from_url(website_url)
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
    # PDFファイル存在状況の初期化設定
    if "pdf_s" not in st.session_state:
        st.session_state.pdf_s = []
    # 音声ファイル存在状況の初期化設定
    if "mp3_s" not in st.session_state:
        st.session_state.mp3_s = []
    # URL存在状況の初期化設定
    if "url_s" not in st.session_state:
        st.session_state.url_s = None
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
        st.subheader("_Upload_ a :rainbow[PDF] :books:")
        pdf_file = st.file_uploader("Upload your PDF here and click on '_Process_'",accept_multiple_files=True,type="pdf")
        st.subheader("_Upload_ a :rainbow[MP3] :cd:")
        mp3_file = st.file_uploader("Upload your MP3 file here and click on '_Process_'",accept_multiple_files=True,type="mp3")
        st.header("",divider="rainbow")
        # url enter
        st.header("",divider="blue")
        st.subheader("_Enter_ :blue[URL] :link:")
        website_url = st.text_input("_Website URL_")
        st.header("",divider="blue")
    
    # ボタン
    st.button("Process", on_click=click_button)
    # 最初アップロード時にクリックすれば十分
    # クリックされたら、その状態をセッションに保存  
    if st.session_state.clicked:
        
        if pdf_file == [] and mp3_file == [] and (website_url is None or website_url == ""):
            st.info(":red[_Enter a URL or Upload some files_]")
        
        else:
            # ファイルがない場合、生ドックを空にする
            if pdf_file == [] and mp3_file == []:
                file_existance = False
                pdf_raw_doc = [Document(page_content="")]
                audio_raw_doc = [Document(page_content="")]
                st.info("Upload some files to ask questions")
                
            # ファイルがある場合、テキストを読み込む
            else:
                file_existance = True
                if pdf_file != []:
                    pdf_raw_doc = get_text_from_pdf(pdf_file)
                else:
                    pdf_raw_doc = [Document(page_content="")]
                    
                if mp3_file != []:
                    audio_raw_doc = get_text_from_mp3(mp3_file)
                else:
                    audio_raw_doc = [Document(page_content="")]
                    
            # ファイルと同じような処理  
            if website_url is None or website_url == "":
                url_existance = False
                st.info("Enter a URL to ask the website")
                url_doc = [Document(page_content="")]
            else:
                url_existance = True
                if website_url is not None or website_url != "":
                    url_doc = get_text_from_url(website_url)
                            
            # ファイルまたはURLがある場合、全てのテクストドックをfull_docに入れて、ベクトルストアーを作成する
            if url_existance or file_existance:
                #--
                if pdf_raw_doc == [Document(page_content="")]:
                    pdf_raw_doc = []
                if audio_raw_doc == [Document(page_content="")]:
                    audio_raw_doc = []
                if url_doc == [Document(page_content="")]:
                    url_doc = []
                
                # 追加されたテキストをセッション内のfull_docに入れる
                full_doc_add = pdf_raw_doc + audio_raw_doc + url_doc
                st.session_state.full_doc += full_doc_add
                
                # print (full_doc_add)
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = [
                        AIMessage(content = "Hello, I am a a bot"),
                    ]
                # 料金節約のために、追加のドキュメントがあるときのみ、Embeddingを執行
                if full_doc_add != []:
                    print("new file(s) added")
                    chunks = get_chunks(st.session_state.full_doc)
                    st.session_state.vector_store = get_vectorstore(chunks)
                else:
                    # print(st.session_state.full_doc)
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
    

