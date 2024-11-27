from PyPDF2 import PdfReader
import pandas as pd
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain_ollama import ChatOllama
# from langchain.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import chromadb
import gradio as gr

def process_youtube(url, query):
    try:
        if not url:
            return "Please provide a valid YouTube link."
        
        print("Processing YouTube URL:", url)
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

        print("Chroma Vector Store Initialized for YouTube.")
        local_model = 'llama3.2'
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
            {"context": retriever, "question": lambda x: x}

            | prompt
            | llm
        )

        result = chain.invoke({"question": query})
        return result

    except Exception as e:
        print("Error processing YouTube:", e)
        return f"An error occurred: {e}"

def process_pdf(files, query):
    try:
        if not files:
            return "Please upload valid PDF files."

        print("Processing PDF files...")
        text = ""
        for file in files:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        
        if not text.strip():
            print("No text extracted from PDF.")
            return "The PDF appears to be empty or unreadable."
        
        print("Extracted text from PDF:", text[:500])  # Show a snippet of the text
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

        print("Chroma Vector Store Initialized for PDF.")
        local_model = 'llama3.2'
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
           {"context": retriever, "question": lambda x: x}
            | prompt
            | llm
        )

        result = chain.invoke({"question": query})
        return result

    except Exception as e:
        print("Error processing PDF:", e)
        return f"An error occurred: {e}"

def process_csv(file, query):
    try:
        if not file:
            return "Please upload a valid CSV file."

        print("Processing CSV file...")
        df = pd.read_csv(file)
        print("DataFrame Head:\n", df.head())  # Debug CSV content

        text = df.to_string(index=False)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        print("Generated Chunks:", chunks[:3])  # Show first few chunks
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        documents = [Document(page_content=chunk) for chunk in chunks]
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name='local-rag'
        )

        print("Chroma Vector Store Initialized for CSV.")
        local_model = 'llama3.2'
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
            {"context": retriever, "question": lambda x: x}

            | prompt
            | llm
        )

        result = chain.invoke({"question": query})
        return result

    except Exception as e:
        print("Error processing CSV:", e)
        return f"An error occurred: {e}"

with gr.Blocks() as app:
    gr.Markdown("# Document Genie")
    gr.Markdown("### Upload a document or YouTube link and ask your questions!")

    with gr.Tab("YouTube"):
        youtube_url = gr.Textbox(label="Enter YouTube Link")
        youtube_query = gr.Textbox(label="Enter Your Question")
        youtube_submit = gr.Button("Submit & Process")
        youtube_output = gr.Textbox(label="Result", interactive=False)
        youtube_submit.click(process_youtube, inputs=[youtube_url, youtube_query], outputs=youtube_output)

    with gr.Tab("PDF"):
        pdf_files = gr.File(label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple")
        pdf_query = gr.Textbox(label="Enter Your Question")
        pdf_submit = gr.Button("Submit & Process")
        pdf_output = gr.Textbox(label="Result", interactive=False)
        pdf_submit.click(process_pdf, inputs=[pdf_files, pdf_query], outputs=pdf_output)

    with gr.Tab("CSV"):
        csv_file = gr.File(label="Upload CSV", file_types=[".csv"])
        csv_query = gr.Textbox(label="Enter Your Question")
        csv_submit = gr.Button("Submit & Process")
        csv_output = gr.Textbox(label="Result", interactive=False)
        csv_submit.click(process_csv, inputs=[csv_file, csv_query], outputs=csv_output)

app.launch()
