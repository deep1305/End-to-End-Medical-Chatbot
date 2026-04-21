from src.helper import load_pdf, text_split, generate_embeddings, filter_to_minimal_docs
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import time

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = "medical-chatbot"

# 1. Initialize Pinecone and Check/Create Index
pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' not found. Creating it now...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    print(f"Index '{index_name}' is ready.")

# 2. Load and process data
print("--- Step 1: Loading PDF data ---")
extracted_data = load_pdf("data/")
print(f"Data Loaded: {len(extracted_data)} pages")

print("--- Step 2: Minimizing metadata ---")
minimal_docs = filter_to_minimal_docs(extracted_data)

print("--- Step 3: Splitting text into chunks ---")
text_chunks = text_split(minimal_docs)
print(f"Number of chunks: {len(text_chunks)}")

# 3. Initialize embeddings
print("--- Step 4: Initializing embedding model ---")
embeddings = generate_embeddings()

# 4. Upsert to Pinecone with Manual Batching
print("--- Step 5: Uploading to Pinecone (Manual Batching) ---")
batch_size = 50  

for i in range(0, len(text_chunks), batch_size):
    batch = text_chunks[i : i + batch_size]
    print(f"Uploading batch {i // batch_size + 1} of {(len(text_chunks) // batch_size) + 1}...")
    
    if i == 0:
        docsearch = PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name
        )
    else:
        docsearch.add_documents(batch)
    
    time.sleep(0.5)

print("\nSuccess! Medical knowledge base is full and auto-configured.")
