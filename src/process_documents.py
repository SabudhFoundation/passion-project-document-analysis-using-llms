from PyPDF2 import PdfReader
import time
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama 
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document 
import pandas as pd
import chromadb
import gradio as gr
def process_youtube(url, query):
    if not url:
        return "Please provide a valid YouTube link."
    
    loader = YoutubeLoader.from_youtube_url(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name='local-rag'
    )

    local_model = 'llama3.2:1b'
    llm = ChatOllama(model=local_model)
    query_prompt = PromptTemplate(
        input_variables=['question'],
        template='''You are an AI assistant. Generate five alternative questions for better retrieval
        from a vector database based on the user's query. Separate alternatives with newlines.
        Original question: {question}'''
    )

    retriever = MultiQueryRetriever.from_llm(vectordb.as_retriever(), llm, prompt=query_prompt)

    template = '''Answer the question based ONLY on the following context:
    {context}
    Question: {question}'''

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({"question": query})
    return result

def process_pdf(files, query):
    if not files:
        return "Please upload valid PDF files."
    
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    documents = [Document(page_content=chunk) for chunk in chunks]
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name='local-rag'
    )

    local_model = 'llama3.2:1b'
    llm = ChatOllama(model=local_model)
    query_prompt = PromptTemplate(
        input_variables=['question'],
        template='''You are an AI assistant. Generate five alternative questions for better retrieval
        from a vector database based on the user's query. Separate alternatives with newlines.
        Original question: {question}'''
    )

    retriever = MultiQueryRetriever.from_llm(vectordb.as_retriever(), llm, prompt=query_prompt)

    template = '''Answer the question based ONLY on the following context:
    {context}
    Question: {question}'''

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({"question": query})
    return result

def process_csv(file, query):
    if not file:
        return "Please upload a valid CSV file."

    df = pd.read_csv(file)
    text = df.to_string(index=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    documents = [Document(page_content=chunk) for chunk in chunks]
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name='local-rag'
    )

    local_model = 'llama3.2:1b'
    llm = ChatOllama(model=local_model)
    query_prompt = PromptTemplate(
        input_variables=['question'],
        template='''You are an AI assistant. Generate five alternative questions for better retrieval
        from a vector database based on the user's query. Separate alternatives with newlines.
        Original question: {question}'''
    )

    retriever = MultiQueryRetriever.from_llm(vectordb.as_retriever(), llm, prompt=query_prompt)

    template = '''Answer the question based ONLY on the following context:
    {context}
    Question: {question}'''

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({"question": query})
    return result

# with gr.Blocks() as app:
#     gr.Markdown("# Document Genie")
#     gr.Markdown("### Upload a document or YouTube link and ask your questions!")

#     with gr.Tab("YouTube"):
#         youtube_url = gr.Textbox(label="Enter YouTube Link")
#         youtube_query = gr.Textbox(label="Enter Your Question")
#         youtube_submit = gr.Button("Submit & Process")
#         youtube_output = gr.Textbox(label="Result", interactive=False)
#         youtube_submit.click(process_youtube, inputs=[youtube_url, youtube_query], outputs=youtube_output)

#     with gr.Tab("PDF"):
#         pdf_files = gr.File(label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple")
#         pdf_query = gr.Textbox(label="Enter Your Question")
#         pdf_submit = gr.Button("Submit & Process")
#         pdf_output = gr.Textbox(label="Result", interactive=False)
#         pdf_submit.click(process_pdf, inputs=[pdf_files, pdf_query], outputs=pdf_output)

#     with gr.Tab("CSV"):
#         csv_file = gr.File(label="Upload CSV", file_types=[".csv"])
#         csv_query = gr.Textbox(label="Enter Your Question")
#         csv_submit = gr.Button("Submit & Process")
#         csv_output = gr.Textbox(label="Result", interactive=False)
#         csv_submit.click(process_csv, inputs=[csv_file, csv_query], outputs=csv_output)

# app.launch(share=True,debug=True)
