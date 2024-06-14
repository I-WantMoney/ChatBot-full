from langchain_community.document_loaders import UnstructuredFileLoader
import streamlit as st

def get_text_from_s3_file(s3_uri):
    if st.session_state.uri_s != s3_uri:
        st.session_state.uri_s = s3_uri
        print("We got a new S3 file")
        filepath = s3_uri
        loader = UnstructuredFileLoader(filepath)
        docs = loader.load()
        s3_raw_doc = docs
    # リンクの変わらない場合、空listにする
    else:
        s3_raw_doc = []
    return s3_raw_doc