import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Pinecone as LangchainPinecone  # Rename the import to avoid conflicts
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)

# Specify the index name
index_name = "info-ai"

# Connect to the specified index
index = pc.Index(index_name)

def get_pdf_text(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None returns gracefully
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Store text chunks as vectors in Pinecone."""
    # Use Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store vectors in Pinecone using Langchain's interface
    vector_store = LangchainPinecone.from_texts(
        texts=text_chunks, embedding=embeddings, index_name=index_name
    )
    
    return vector_store

def get_conversational_chain(vector_store):
    """Initialize and return the conversational retrieval chain."""
    # Use Hugging Face Hub for the LLM
    llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-1B", task="text-generation")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )

    return conversation_chain
