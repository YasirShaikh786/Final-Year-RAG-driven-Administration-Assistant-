import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
import time

print("Starting the data loading process...")

# Load environment variables from .env file
load_dotenv()

# --- 1. CONNECT TO MONGODB ---
client = MongoClient(os.getenv("MONGODB_URI"))
db_name = "Q_A_AI_with_RAG" # Using underscores for valid DB names
collection_name = "document_chunks"
collection = client[db_name][collection_name]
print("Connected to MongoDB.")

# --- 2. LOAD DOCUMENTS FROM DIRECTORY ---
# Note: Ensure you have a folder named 'sample_files' with .txt files in it
loader = DirectoryLoader("./sample_file", glob="**/*.txt", show_progress=True)
data = loader.load()
print(f"Loaded {len(data)} documents from the directory.")

# --- 3. INITIALIZE THE EMBEDDING MODEL ---
# This uses a local, open-source model. It will be downloaded on the first run.
print("Initializing the local embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded.")

# --- 4. EMBED AND STORE DOCUMENTS IN MONGODB ---
# This is the most important step. It creates vector embeddings from your documents
# and stores them in your MongoDB collection.
# This can take some time depending on the number of documents.
print("Embedding documents and uploading to MongoDB Atlas... Please wait.")
start_time = time.time()
vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents=data,
    embedding=embeddings,
    collection=collection,
)
end_time = time.time()

print("--------------------------------------------------")
print("✅ Data loading and embedding complete!")
print(f"Total time taken: {end_time - start_time:.2f} seconds.")
print("You can now run the main application script (app.py).")
print("--------------------------------------------------")