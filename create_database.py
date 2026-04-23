import os
from dotenv import load_dotenv, find_dotenv
import shutil

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma # Chroma in an open-source embedding database

_ = load_dotenv(find_dotenv()) # read local .env file
# No API key needed for local HuggingFace embeddings
CHROMA_PATH = "chroma"
DATA_PATH = "data/small_sample"

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_documents():
    print(f"Loading documents from {DATA_PATH}...")
    
    # Load markdown files
    md_loader = DirectoryLoader(DATA_PATH, glob="*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    md_documents = md_loader.load()
    print(f"-> Loaded {len(md_documents)} Markdown (.md) documents.")

    # Load PDF files
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    print(f"-> Loaded {len(pdf_documents)} PDF (.pdf) documents.")

    documents = md_documents + pdf_documents
    return documents

def split_text(documents):
    """This function will split the documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database directory first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH) # Deletes all files and subdirectories within the specified path.
        
    # Create a vector database from the documents
    db = Chroma.from_documents(chunks, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"), persist_directory=CHROMA_PATH)
    # langchain-chroma handles persistence automatically when a persist_directory is provided
    print(f"Saved {len(chunks)} chunks to database in {CHROMA_PATH}.")

if __name__ == "__main__":
    generate_data_store()