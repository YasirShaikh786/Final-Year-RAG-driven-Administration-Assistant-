import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
import gradio as gr

print("Starting the Q&A application...")

# Load environment variables from .env file
load_dotenv()

# --- 1. CONNECT TO MONGODB ---
client = MongoClient(os.getenv("MONGODB_URI"))
db_name = "Q_A_AI_with_RAG"
collection_name = "document_chunks"
collection = client[db_name][collection_name]
print("Connected to MongoDB.")

# --- 2. INITIALIZE THE EMBEDDING MODEL ---
# You need the same embedding model used to store the data to create embeddings for the user's query
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded.")

# --- 3. CONNECT TO THE EXISTING VECTOR STORE ---
# This is different from the loading script. We are not creating a new store,
# but connecting to the one that `load_data.py` already populated.
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
)
print("Connected to the existing vector store.")

# --- 4. INITIALIZE THE LOCAL LLM ---
print("Loading local LLM (Llama 3 8B)...")
llm = LlamaCpp(
    model_path="C:/path/to/your/model/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", # <-- IMPORTANT: CHANGE THIS PATH
    n_ctx=2048,
    temperature=0.1,
    verbose=False
)
print("LLM loaded.")

# --- 5. CREATE THE RAG CHAIN ---
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
print("RAG chain created.")

def query_chain(question):
    """
    This function takes a user's question, sends it to the RAG chain, and returns the answer.
    """
    print(f"Received question: {question}")
    try:
        response = qa_chain.run(question)
        print(f"Generated response: {response}")
        return response
    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        return error_message

# --- 6. BUILD THE GRADIO UI ---
with gr.Blocks(theme='soft', title="Local Q&A AI with RAG") as demo:
    gr.Markdown("# 🤖 Q&A AI with RAG (Running Locally)")
    gr.Markdown("Ask a question about your documents, and the AI will answer based on the context.")
    
    with gr.Row():
        textbox = gr.Textbox(label="Enter your question here:", placeholder="e.g., What is the main conclusion of the report?", lines=2)
    
    button = gr.Button("Ask", variant="primary")
    
    with gr.Accordion("Answer:", open=True):
        output = gr.Markdown(value="Your answer will appear here...")
    
    button.click(query_chain, inputs=textbox, outputs=output)

print("Launching Gradio UI... 🚀")
demo.launch()