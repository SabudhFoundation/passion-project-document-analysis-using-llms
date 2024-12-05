import os
import time
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from PIL import Image
import fitz  # PyMuPDF


# Helper function to process a single page
def process_page(page):
    return page.extract_text()


# Function to convert PDF to images
def convert_pdf_to_images(pdf_path, output_dir):
    """
    Converts a PDF to images using PyMuPDF and saves them to the output directory.
    """
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


# Updated process_pdf function with similarity search for relevant pages
def process_pdf_with_images(pdf_file, query):
    if not pdf_file:
        return "Please upload a valid PDF file."

    start_time = time.time()

    # Load the PDF and extract text
    pdf_reader = PdfReader(pdf_file)
    with ThreadPoolExecutor() as executor:
        pages_text = list(executor.map(process_page, pdf_reader.pages))

    combined_text = " ".join(pages_text)

    if not combined_text:
        return "No text could be extracted from the PDF."

    # Split text into chunks and map them to pages
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.split_text(combined_text)

    # Create documents with page numbers
    documents = []
    chunk_to_page = {}
    page_number = 1
    for chunk in chunks:
        documents.append(Document(page_content=chunk))
        chunk_to_page[chunk] = page_number
        if page_number < len(pages_text):  # Advance only when current page ends
            page_number += 1

    # Convert PDF to images
    output_dir = "output_images"
    image_paths = convert_pdf_to_images(pdf_file, output_dir)
    page_to_image = {i + 1: image_paths[i] for i in range(len(image_paths))}

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name='local-rag'
    )

    # Initialize the local LLaMA model
    local_model = 'llama3.2:1b'
    llm = ChatOllama(model=local_model)

    # Perform similarity search to retrieve the most relevant chunks
    top_relevant_chunks = vectordb.similarity_search(query, k=3)

    # Find the pages corresponding to the relevant chunks
    relevant_pages = list({chunk_to_page[chunk.page_content] for chunk in top_relevant_chunks})

    # Get associated images
    relevant_images = [page_to_image[page] for page in relevant_pages if page in page_to_image]

    # Generate response using the local LLaMA model
    # Combine top relevant chunks into a single context string
    context = "\n".join([chunk.page_content for chunk in top_relevant_chunks])
    
    # Generate response using the local LLaMA model
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
    result = chain.invoke({"context": context, "question": query})

     
    # response_generation_prompt = '''Answer the question based ONLY on the following context:
    # {context}
    # Question: {question}'''

    # prompt = ChatPromptTemplate.from_template(response_generation_prompt)
    # chain = (
    #     {"context": top_relevant_chunks, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    # result = chain.invoke({"question": query})

    end_time = time.time()
    response_time = f"Time taken to generate response: {end_time - start_time:.2f} seconds"

    print(response_time)
    return {"response": result, "images": relevant_images}


# Example usage
if __name__ == "__main__":
    pdf_file = "document_analysis.pdf"
    query = "What are the use cases?"

    output = process_pdf_with_images(pdf_file, query)
    print("Generated Response:")
    print(output["response"])
    print("Relevant Images:")
    for image_path in output["images"]:
        print(image_path)
# if __name__ == "__main__":
#     pdf_file = "document_analysis.pdf"
#     query = "What are the use cases?"

#     output = process_pdf_with_images(pdf_file, query)
#     print("Generated Response:")
#     print(output["response"])
#     print("Relevant Images:")
#     for image_path in output["images"]:
#         print(image_path)
