from langchain_groq import ChatGroq
import os
from langchain_huggingface import HuggingFaceEmbeddings
import sqlite3
from langchain_chroma import Chroma
from main import store_pdf_data_in_vector_db
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()

# os.environ['PATH'] = './:' + os.environ['PATH']
# print("SQLite Version:", sqlite3.sqlite_version)

from dotenv import load_dotenv
load_dotenv()
login(token=os.getenv("hf_token"))

llm = ChatGroq(
    groq_api_key = os.getenv("groq_api_key"),
    model_name="llama-3.3-70b-versatile",
)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(
    embedding_function = embeddings,
    persist_directory="./db_new"
)

def index_data(relevant_path):
    """
    This function takes the full path and indexes the data 
    in vector database
    call this function and pass the relevant path
    """
    try:
        store_pdf_data_in_vector_db(relevant_path)
        print("data indexed successfully")
    except Exception as e:
        print(e, "some error occured")

def query_file_and_invoke_llm(question):
    results = vector_store.similarity_search(
        question,
        k = 4,
        filter = {"source":"pdf"}
    )
    relevant_text = "".join([i.page_content for i in results])
    prompt = f"Given the content: {relevant_text}\n answer the query {question}"
    response = llm.invoke(prompt)
    return response.content

# while True:
#     res = query_file_and_invoke_llm(input("enter your query"))
#     print(res,"-----------------------------------------------------------")