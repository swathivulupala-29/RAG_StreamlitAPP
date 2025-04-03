from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
import os 
from langchain_core.documents import Document
from langchain_chroma import Chroma
from huggingface_hub import login

from dotenv import load_dotenv
load_dotenv()
login(token=os.getenv("hf_token"))

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(
    embedding_function = embeddings,
    persist_directory="./db_new"
)

def extract_data_from_pdf(file_path):
    with open(file_path, "rb") as file:
        text = ""
        reader = PdfReader(file_path)
        for data in reader.pages:
            text += data.extract_text()
    return text

def chunk_data(text, chunk_size = 200):
    chunks = []
    for i in range(0,len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def store_pdf_data_in_vector_db(file_path):
    """
    main function to store the data in vectore store
    """
    text = extract_data_from_pdf(file_path)
    chunks = chunk_data(text)

    documents = []
    for chunk in chunks:
        document = Document(page_content=chunk,metadata={"source": "pdf"})
        documents.append(document)
    vector_store.add_documents(documents = documents)
    print("data stored successfully in vector store", len(chunks))