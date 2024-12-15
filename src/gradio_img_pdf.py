import os
import time
import re
import itertools
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

# ChromaDB Client Configuration
chromadb.api.client.SharedSystemClient.clear_system_cache()
CHROMA_DB_DIR = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Function to create a valid collection name from a file path
def generate_valid_collection_name(file_path):
    base_name = os.path.basename(file_path).split('.')[0]  # Remove file extension
    collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name).strip('_')
    if len(collection_name) < 3 or len(collection_name) > 63:
        collection_name = collection_name[:63]
    return collection_name

# Function to convert PDF to images using PyMuPDF
def convert_pdf_to_images(pdf_path, output_dir):
    try:
        pdf_document = fitz.open(pdf_path)
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap()
            output_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
            pix.save(output_path)
            image_paths.append(output_path)

        pdf_document.close()
        print(f"PDF successfully converted to images. Images saved at: {output_dir}")
        return image_paths
    except Exception as e:
        print(f"Error while converting PDF to images: {e}")
        return []

# Function to process and store PDF content in ChromaDB
def process_and_store_pdf(pdf_path, collection_name):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    collection = client.get_or_create_collection(name=collection_name)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            metadatas=[{"page": chunk.metadata.get("page", "N/A")}],
            ids=[f"{collection_name}_chunk_{i}"]
        )

    return len(chunks)

# Function to query the ChromaDB collection and find relevant images
def query_pdf_database(query, collection_name, page_to_image):
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        return f"Error: Unable to find collection '{collection_name}'.", []

    results = collection.query(
        query_texts=[query],
        n_results=3  # Top 3 most relevant documents
    )

    if results and results['documents']:
        combined_text = "\n".join(itertools.chain(*results['documents']))

        local_model = 'llama3.2:1b'
        llm = ChatOllama(model=local_model)

        response_generation_prompt = '''Answer the question based ONLY on the following context:
        {context}
        Question: {question}'''

        prompt = ChatPromptTemplate.from_template(response_generation_prompt)

        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke({"context": combined_text, "question": query})

        relevant_pages = [int(chunk.metadata.get("page", "N/A")) for chunk in results['documents']]
        relevant_images = [page_to_image[page] for page in relevant_pages if page in page_to_image]

        return response, relevant_images
    else:
        return "No relevant data found for the query.", []

# Function to process a PDF, extract images, create vector store, and query for answers
def process_pdf_with_images(pdf_file, query):
    if not pdf_file:
        return "Please upload a valid PDF file.", None

    start_time = time.time()

    # Load the PDF and extract text
    pdf_reader = PdfReader(pdf_file.name)
    with ThreadPoolExecutor() as executor:
        pages_text = list(executor.map(lambda page: page.extract_text(), pdf_reader.pages))

    combined_text = " ".join(pages_text)

    if not combined_text:
        return "No text could be extracted from the PDF.", None

    # Split text into chunks and map them to pages
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.split_text(combined_text)

    documents = [Document(page_content=chunk) for chunk in chunks]
    chunk_to_page = {chunk: i + 1 for i, chunk in enumerate(chunks)}

    # Convert PDF to images and map pages to image paths
    output_dir = "output_images"
    image_paths = convert_pdf_to_images(pdf_file.name, output_dir)
    page_to_image = {i + 1: image_paths[i] for i in range(len(image_paths))}

    # Create embeddings and vector store
    collection_name = generate_valid_collection_name(pdf_file.name)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name
    )

    # Perform similarity search
    top_relevant_chunks = vectordb.similarity_search(query, k=2)

    if not top_relevant_chunks:
        return "No relevant data found for the query.", None

    relevant_pages = list({chunk_to_page[chunk.page_content] for chunk in top_relevant_chunks})
    relevant_images = [page_to_image[page] for page in relevant_pages if page in page_to_image]

    context = "\n".join([chunk.page_content for chunk in top_relevant_chunks])

    local_model = 'llama3.2:1b'
    llm = ChatOllama(model=local_model)

    response_generation_prompt = '''Answer the question based ONLY on the following context:
    {context}
    Question: {question}'''

    prompt = ChatPromptTemplate.from_template(response_generation_prompt)

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = chain.invoke({"context": context, "question": query})

    end_time = time.time()
    response_time = f"Time taken to generate response: {end_time - start_time:.2f} seconds"
    print(response_time)

    return response, relevant_images

# Gradio interface function
def gradio_interface(pdf_file, query):
    output = process_pdf_with_images(pdf_file, query)
    if isinstance(output, str):  # Handle error message
        return output, None
    return output[0], output[1]

# Create and launch the Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Enter Your Query", placeholder="Ask your question here...")
    ],
    outputs=[
        gr.Textbox(label="Generated Response", interactive=False),
        gr.Gallery(label="Relevant Images", interactive=False)
    ],
    title="PDF Document Analysis with LLM",
    description="Upload a PDF and ask questions. The app will extract text, find relevant pages, and show the response with corresponding images."
)

if __name__ == "__main__":
    interface.launch(share=True)


# import os
# import time
# from concurrent.futures import ThreadPoolExecutor
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from PIL import Image
# import fitz  # PyMuPDF
# from auth import log_user_history
# import gradio as gr
# import chromadb

# chromadb.api.client.SharedSystemClient.clear_system_cache()
# # Helper function to process a single page
# def process_page(page):
#     return page.extract_text()


# # Function to convert PDF to images
# def convert_pdf_to_images(pdf_path, output_dir):
#     """
#     Converts a PDF to images using PyMuPDF and saves them to the output directory.
#     """
#     try:
#         pdf_document = fitz.open(pdf_path)
#         os.makedirs(output_dir, exist_ok=True)
#         image_paths = []

#         for page_num in range(len(pdf_document)):
#             page = pdf_document[page_num]
#             pix = page.get_pixmap()
#             output_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
#             pix.save(output_path)
#             image_paths.append(output_path)

#         pdf_document.close()
#         print(f"PDF successfully converted to images. Images saved at: {output_dir}")
#         return image_paths
#     except Exception as e:
#         print(f"Error while converting PDF to images: {e}")
#         return []


# # Updated process_pdf function with similarity search for relevant pages
# def process_pdf_with_images(pdf_file, query):
#     if not pdf_file:
#         return "Please upload a valid PDF file."

#     start_time = time.time()

#     # Load the PDF and extract text
#     pdf_reader = PdfReader(pdf_file)
#     with ThreadPoolExecutor() as executor:
#         pages_text = list(executor.map(process_page, pdf_reader.pages))

#     combined_text = " ".join(pages_text)

#     if not combined_text:
#         return "No text could be extracted from the PDF."

#     # Split text into chunks and map them to pages
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
#     chunks = text_splitter.split_text(combined_text)

#     # Create documents with page numbers
#     documents = []
#     chunk_to_page = {}
#     page_number = 1
#     for chunk in chunks:
#         documents.append(Document(page_content=chunk))
#         chunk_to_page[chunk] = page_number
#         if page_number < len(pages_text):  # Advance only when current page ends
#             page_number += 1

#     # Convert PDF to images
#     output_dir = "output_images"
#     image_paths = convert_pdf_to_images(pdf_file, output_dir)
#     page_to_image = {i + 1: image_paths[i] for i in range(len(image_paths))}

#     # Create embeddings and vector store
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     vectordb = Chroma.from_documents(
#         documents=documents,
#         embedding=embeddings,
#         collection_name='local-rag'
#     )

#     # Initialize the local LLaMA model
#     local_model = 'llama3.2:1b'
#     llm = ChatOllama(model=local_model)

#     # Perform similarity search to retrieve the most relevant chunks
#     top_relevant_chunks = vectordb.similarity_search(query, k=2)

#     # Find the pages corresponding to the relevant chunks
#     relevant_pages = list({chunk_to_page[chunk.page_content] for chunk in top_relevant_chunks})

#     # Get associated images
#     relevant_images = [page_to_image[page] for page in relevant_pages if page in page_to_image]

#     # Generate response using the local LLaMA model
#     # Combine top relevant chunks into a single context string
#     context = "\n".join([chunk.page_content for chunk in top_relevant_chunks])
    
#     # Generate response using the local LLaMA model
#     response_generation_prompt = '''Answer the question based ONLY on the following context:
#     {context}
#     Question: {question}'''
    
#     prompt = ChatPromptTemplate.from_template(response_generation_prompt)
    
#     chain = (
#         {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     result = chain.invoke({"context": context, "question": query})

#     end_time = time.time()
#     response_time = f"Time taken to generate response: {end_time - start_time:.2f} seconds"
#     log_user_history(query,result)

#     print(response_time)
#     # return {"response": result, "images": relevant_images}
#     return result,relevant_images


# # Gradio interface function
# def gradio_interface(pdf_file, query):
#     output = process_pdf_with_images(pdf_file.name, query)
#     if isinstance(output, str):  # Handle error message
#         return output, None
#     # Return the response and the list of relevant images (paths)
#     return output["response"], output["images"]  # This should be a list of file paths


# # Create the Gradio interface
# interface = gr.Interface(
#     fn=gradio_interface,
#     inputs=[
#         gr.File(label="Upload PDF",file_types=[".pdf"]),
#         gr.Textbox(label="Enter Your Query", placeholder="Ask your question here...")
#     ],
#     outputs=[
#         gr.Textbox(label="Generated Response",interactive=False),
#         gr.Gallery(label="Relevant Images",interactive=False)  # Gallery to show images dynamically
#     ],
#     title="PDF Document Analysis with LLM",
#     description="Upload a PDF and ask questions. The app will extract text, find relevant pages, and show the response with corresponding images."
# )

# # Launch the interface
# if __name__ == "__main__":
#     interface.launch(share=True,max_file_size="1mb")



