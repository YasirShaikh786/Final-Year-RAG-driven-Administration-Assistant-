# File: app.py (Corrected Version)

import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded.")

# --- 3. CONNECT TO THE EXISTING VECTOR STORE ---
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
)
print("Connected to the existing vector store.")

# # --- 4. INITIALIZE THE LOCAL LLM ---
print("Loading local LLM (Llama 3 8B)...")
llm = LlamaCpp(
    model_path="D:/AI Projects/Final-Year-RAG-Admin-Assistant/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_ctx=4096,
    temperature=0.1,
    verbose=True
)
# print("LLM loaded.")

# --- 5. CREATE THE RAG CHAIN WITH A PROMPT TEMPLATE ---

# prompt_template = """
# Use the following pieces of context to answer the question at the end. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}

# Question: {question}
# Helpful Answer:"""

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )

retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
print("RAG chain created.")

def query_chain(question):
    print(f"Received question: {question}")
    try:
        result = qa_chain({"query": question})
        response = result["result"]
        
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