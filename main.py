import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == '__main__':
    # 1. pdf load
    file_path = '/Users/dasom/somi/github.com/dasom222g/leaarn-faiss/documents/Synergizing Reasoning and Acting in Language Models.pdf'
    loader = PyPDFLoader(file_path)
    documents = loader.load() # PDF document for Langchain

    # 2. 임베딩 모델 준비
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # 3. 문서 청킹
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    chuncked_docs = text_splitter.split_documents(documents)

    # 4. vectorstore 생성 및 저장
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(folder_path='faiss_react_index') # 영구 저장


