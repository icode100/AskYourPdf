import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text=''
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local('faiss_index')

def get_conversational_chain():
    prompt = PromptTemplate.from_template(
        '''
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not available 
        in the provided context just say, "answer is not available in the context", DO NOT provide wrong answer\n\n
        Context:\n{context}?\n
        Question:\n{question}\n

        Answer:
        '''
    )

    gemini = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest',temperature=0.3)
    chain = load_qa_chain(gemini,chain_type='stuff',prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index',embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {
            'input_documents':docs,
            'question':user_question,
        },
        return_only_outputs=True
    )
    print(response)
    st.markdown(response['output_text'])

st.set_page_config(page_title='PDF Q&A Bot',page_icon=':material/picture_as_pdf:')
st.header('Make your PDF talk to you with Gemini')
user_question = st.text_input("Ask your PDF ")
if user_question:
    user_input(user_question)

with st.sidebar:
    st.title('Upload your PDFs Here 😌')
    pdf_docs = st.file_uploader(label='\t',accept_multiple_files=True,type=['pdf'])
    if st.button('submit & process'):
        with st.spinner('Processing...'):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success('done')