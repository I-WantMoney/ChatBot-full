# チャットボットドキュメント

　　このドキュメントはコードに対する説明です、すべてのコードではありません、コードは[こちら][11]をご参照ください。
  
  * 以下は学習用の動画です。
    * [Langchain PDF App (GUI)__Create a ChatGPT For Your PDF in Python][12]
    * [Chat with Multiple PDFs__LangChain App Tutorial in Python (Free LLMs and Embeddings)][13]
    * [Tutorial__Chat with any Website using Python and Langchain (LATEST VERSION)][14]

[11]:https://rayoo.sharepoint.com/:u:/s/r-d/EfkuyV-0B-1Gg0jhjRHhggYBE6XBR4IifJHjurTI19boJQ?e=jwctGo
[12]:https://www.youtube.com/watch?v=wUAUdEw5oxM
[13]:https://www.youtube.com/watch?v=dXxQ0LR-3Hg&t=1726s
[14]:https://www.youtube.com/watch?v=bupx08ZgSFg&t=3636s

## Part 1. 必要なライブラリー
* 必要なライブラリー
    * [streamlit][1]
    * [langchain][2]
    * [langchain_community][3]
    * [langchain_core][4]
    * [langchain-openai][5]
    * [openai][6]
    * [python-dotenv][7]
    * [PyPDF2][8]
    * [chromadb][9]
      
   [1]:https://docs.streamlit.io/
   [2]:https://python.langchain.com/docs/get_started/introduction
   [3]:https://api.python.langchain.com/en/latest/community_api_reference.html
   [4]:https://api.python.langchain.com/en/latest/core_api_reference.html
   [5]:https://api.python.langchain.com/en/latest/core_api_reference.html
   [6]:https://platform.openai.com/docs/quickstart?context=python
   [7]:https://pypi.org/project/python-dotenv/
   [8]:https://pypi.org/project/PyPDF2/
   [9]:https://pypi.org/project/chromadb/

* プロジェクトのパスで```ターミナル```を開いて、以下のコマンドを実行

　　エラー発生の場合、-r requirements.txtではなく、上記のライブラリーを一つずつインストールしてください。

```terminal
pip install -r requirements.txt
```
## Part 2. 使い方
* OpenAIのAPI Keyを獲得、方法は[こちら][10]

[10]:https://rayoo.sharepoint.com/:b:/s/r-d/EZ5Rdze3LBRAilBK2mcVvzABDMbQcWqhReP8n6ds-7BV8Q?e=caOUC3
* .envファイルにAPI Keyを自分のものに置き換える

* ```ターミナル```で「streamlit run ファイル名」コマンドを実行

```terminal
streamlit run full_app(v2).py
```

* フォルダ内のPDFとMP3ファイルでアプリの機能を試す

## Part 3. 各関数の説明

  本アプリはStreamlitのフレームワークで作成されます。Streamlit は、データスクリプトを数分で共有可能なWebアプリに変換することができます。言語はすべて```Python```です。
  
  RAGの流れは以下です、これからは各関数の役割を紹介します。
   
   ![RAG](https://github.com/I-WantMoney/ChatBot-full/raw/main/app_pic/RAG.png "RAG")

### 3.1 Streamlitの特性

　　Streamlitアプリケーションでは、「操作」があるたびにコードを走り直す仕組みです、このせいでクリックやエンターキーなどの操作をすると、過去の処理結果は全部無くなってしまいます、処理結果を維持したい場合は```セッション```にステートを入れる必要があります。

　　セッションはアプリケーションの起動とともに起動されます、アプリケーションが終了するまで```セッション```内部の情報を保つことができます。

### 3.2 .envファイルからAPI Keyなどを抽出

　　最先頭に以下のコードを入れて、フォルダ内の.envファイルの内容を自動的に読み込むようにします。これがあればコード内で手動入力は不要になります、デプロイ時のAPI Keyの安全性が高まります。

PS：.envファイル内の変数名を変えないでください、API Keyなどの変数名を特定様式にすれば自動的に読み込みます、コードに変数名の入力すら不要です。

```python
from dotenv import load_dotenv

load_dotenv()
```
### 3.3 知識データの取得

　　第一歩としては、データソースから知識データベースを作成します。

  ![text_extract](https://github.com/I-WantMoney/ChatBot-full/raw/main/app_pic/text_extract.png "text extract")

#### 3.3.1 テキスト抽出

　　モデル使用料金を節約するために、新追加のファイルのみに対し、テキスト化の処理をします、すでに存在しているファイルは処理しません。
  
　　この部分のコードにある「```st.session_state.○○○```」のような変数はセッション内の変数です、これらの変数の定義文はmain関数にあります。

* PDFからテキストを抽出

```python
from PyPDF2 import PdfReader

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
```

* MP3からテキストを抽出

    OpenAIのwhisperモデルを利用してスピーチテキスト変換を行います。

```python
from openai import OpenAI

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
              # OpenAIのwhisperモデルでスピーチテキスト変換
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
```

* ウェブサイトからテキストを抽出

    WebBaseLoaderはウェブサイト上のテキスト情報を取得できますが、ページ上のほかのURLにアクセスすることはできません。

```python
from langchain_community.document_loaders import WebBaseLoader

# ウェブサイトからテキスト抽出
def get_text_from_url(url):
  if st.session_state.url_s != url:
      st.session_state.url_s = url
      loader = WebBaseLoader(url)
      url_doc = loader.load()
  else:
      url_doc = [Document(page_content="")]
  
  return url_doc

```

#### 3.3.2 テキストの分割とベクトル化

　　RAG+LLMとは、人間の質問に一番合っている情報を知識ベースの中から見つけ出し、LLM経由で人間の分かる言葉を出力します。

　　AIに必要な情報を見つけやすくするために、長文のテキストを分割する必要があります、但し、そのまま分割をすると「……○○○ができま」、「せん、……」のような状況が起こりやすい、こういう場合は、AIには「○○○ができます」か「○○○ができません」かが分かりません、情報の損失があります。

　　上記の状況を防ぐために、overlap（文書の重なり合い）を設定する必要があります、分割された文書は「……○○○ができま」、「○○○ができません、……」のようになり、情報の損失を大幅に減少します。

　　分割された文書を機器が分かるように、ベクトル化（```Embedding```）が必要です。

 * テキストの分割
   
```python
from langchain.text_splitter import CharacterTextSplitter

def get_chunks(full_doc):
  text_splitter = CharacterTextSplitter(
      separator = "\n",
      chunk_size = 1000,
      chunk_overlap = 200,
      length_function = len
  )
  
  chunks = text_splitter.split_documents(full_doc)
  
  return chunks
```
   
* ベクトルストア生成

　　Embeddingモデルは自由に設定できます、ここでは```OpenAI```のモデルを使用します。

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

def get_vectorstore(chunks):
  embeddings = OpenAIEmbeddings()
  vectorstore = Chroma.from_documents(documents=chunks,embedding=embeddings)
  
  return vectorstore
```

### 3.4 LLMに渡すためのPrompt設定

　　ChatGPTウェブ版を使うとき、履歴にある話題の質問もできることにみんなは気づいたんでしょう。Langchainツールを使い、その機能を簡単に実現できます（統合されていますので、実は以下の設定をすれば十分です）。

  ![prompt](https://github.com/I-WantMoney/ChatBot-full/raw/main/app_pic/prompt.png "prompt")

* 会話履歴を取得してドキュメントを返すチェーンを作成

    Promptに会話履歴を入れ、命令のような説明を追加してAIに会話履歴を参照させます。

```python
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

# LLMの選定は自由に選べます、ここではOpenAIのモデルです
llm_model = os.environ["OPENAI_API_MODEL"]
def get_context_retriever_chain(vector_store):
  llm = ChatOpenAI(model=llm_model) # カッコ内でapi-keyの指定、モデルの指定などができます。コードの先頭にdotenvを使ったので、自動的に.envファイルからapi-keyを取得します
  retriever = vector_store.as_retriever()
  prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user","{input}"),
      # 指示文
      ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
  ])
  
  retriever_chain = create_history_aware_retriever(llm,retriever,prompt)
  
  return retriever_chain
```

* ドキュメントのリストをモデルに渡すためのチェーンを作成

  チャットボットに知識データを知らせるために、Prompt内に知識データを入れる必要があります。
  
```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_conversational_rag_chain(retriever_chain):
  
  llm = ChatOpenAI(model=llm_model) # カッコ内でapi-keyの指定、モデルの指定などができます。コードの先頭にdotenvを使ったので、自動的に.envファイルからapi-keyを取得します
  prompt = ChatPromptTemplate.from_messages([

      # contextは抽出された有用なデータ
      ("system","Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),

      # inputはユーザーの質問文
      ("user","{input}")
  ])
  
  stuff_documents_chian = create_stuff_documents_chain(llm,prompt)
  
  return create_retrieval_chain(retriever_chain,stuff_documents_chian)
```

* AIの回答を取得

```python
def get_response(user_input):

    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)    
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']
```

### 3.5 ボタン状態の設定

　　Streamlitの特性で、何の処理もしないと、ボタンなどの状態（```state```）は保つことはできません、クリック状態をセッションに入れる関数を作ります。

```python
def click_button():
    st.session_state.clicked = True
```

## Part 4. main関数説明
　　以下のコードは全部「def main():」の内容です、以下は「def main():」を省略します。

### 4.1 アプリケーションのページとタイトル設定

```python
# app config
st.set_page_config(page_title="Chat with your files", page_icon="🤖")
st.title("Upload files and chat with them")
st.info("Click the :red[_Process_] button before asking questions\n(:red[_Only the first time you upload_])")
```

　　効果は以下です。

![page title](https://github.com/I-WantMoney/ChatBot-full/raw/main/app_pic/page_title.png "Page Title")

![st title](https://github.com/I-WantMoney/ChatBot-full/raw/main/app_pic/st_title.png "st title")

### 4.2 セッションに入れる要素の初期化

　　セッションに入れるものはアプリが起動している限り、状態（```state```）の変更を含めて、その状態をずっと維持します。

```python
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
```

### 4.3 サイドバーの設定
　　PDF、MP3、ウェブサイトをアップロード/入力するパーツをサイドバーに入れます。

```python
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
```

![page title](https://github.com/I-WantMoney/ChatBot-full/raw/main/app_pic/sidebar.png "sidebar")

### 4.4 ファイルのドキュメント化処理

　　ファイルの存在情況次第で読み込みをするかどうかを判断

```python
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
```

### 4.5 URLのドキュメント化処理

　　ファイルと同じように判断

```python
# ファイルと同じような処理  
if website_url is None or website_url == "":
    url_existance = False
    st.info("Enter a URL to ask the website")
    url_doc = [Document(page_content="")]
else:
    url_existance = True
    if website_url is not None or website_url != "":
        url_doc = get_text_from_url(website_url)
```

### 4.6 ドキュメント＋ユーザーの質問＋会話履歴＝AI回答生成

```python
# ファイルまたはURLがある場合、全てのテキストドックをfull_docに入れて、ベクトルストアーを作成する
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
    
    # 最初のAIメッセージを設定するために、ここでchat_historyを初期化する
    # 実際は上の初期化と同じように空にしても構わない
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
```

### 4.7 ボタンの設定
　　何もアップロードしないと、当然AIは質問に答えられません、この時エラーが発生しちゃいます。
  
　　それを防ぐために、ファイルをアプロードし、「```Process```」ボタンを押した後、ユーザー入力用のチャットバーを表示するような仕組みを作成します。

　　「else:」の下は「ファイルのドキュメント化処理」、「URLのドキュメント化処理」、「ドキュメント＋ユーザーの質問＋会話履歴＝AI回答生成」の全てのコードです。

```python
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
            ...
            ...
            ...
```


