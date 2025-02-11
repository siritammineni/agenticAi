import os
import time
import faiss
import pickle
import numpy as np
import pandas as pd
import chromadb
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from sentence_transformers import SentenceTransformer
# Define Chunking Strategies
chunking_strategies = {
   "RecursiveTextSplitter": RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50),
   "SentenceSplitter": RecursiveCharacterTextSplitter(
       chunk_size=300,
       chunk_overlap=50,  
       separators=["\n", ". ", "? ", "! "],  
       length_function=len  
   ),
   "ParagraphSplitter": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
}
# Define Embedding Models
embedding_models = {
   "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
   "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
   "bge-base-en": "BAAI/bge-base-en"
}
# Initialize FAISS & ChromaDB Storage Paths
FAISS_DB_DIR = "vector_dbs_faiss"
CHROMA_DB_DIR = "vector_dbs_chroma"
os.makedirs(FAISS_DB_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
# Function to Load Text Files
def load_text_files(directory="scraped pages final"):
   print("Loading text files...")
   texts = []
   for file in os.listdir(directory):
       if file.endswith(".txt"):
           with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
               texts.append(f.read())
   print("Loading files done.")
   return "\n".join(texts)

# ✅ Function to Create and Save FAISS Vector Store (Correct Structure for Future Retrieval)
def create_faiss_store(text_chunks, chunker_name, emb_name, emb_model):
   print(f"Creating FAISS DB for {chunker_name} + {emb_name}")
   start_time = time.time()
   # ✅ Load SentenceTransformer
   model = SentenceTransformer(emb_model, device="cpu")
   # ✅ Generate embeddings with progress bar
   print(f"Generating embeddings for {len(text_chunks)} text chunks...")
   embeddings = model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
   # ✅ Create FAISS index
   index = faiss.IndexFlatL2(embeddings.shape[1])
   # ✅ Add embeddings to FAISS with progress bar
   print(f"Adding {len(embeddings)} embeddings to FAISS...")
   for i in tqdm(range(len(embeddings)), desc="FAISS Indexing", unit="vec"):
       index.add(np.array([embeddings[i]]).astype("float32"))
   # ✅ Create an in-memory document store
   docstore = InMemoryDocstore({i: text_chunks[i] for i in range(len(text_chunks))})
   index_to_docstore_id = {i: i for i in range(len(text_chunks))}
   # ✅ Store FAISS with `docstore` and `index_to_docstore_id`
   vector_store = FAISS(index=index, embedding_function=model, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
   # ✅ Save FAISS index and metadata for future retrieval
   faiss_path = os.path.join(FAISS_DB_DIR, f"{chunker_name}_{emb_name}")
   vector_store.save_local(faiss_path)
   # ✅ Save embedding model reference for later query encoding
   model_path = os.path.join(FAISS_DB_DIR, f"{chunker_name}_{emb_name}_model.pkl")
   with open(model_path, "wb") as f:
       pickle.dump(model, f)
   end_time = time.time()
   time_taken = round(end_time - start_time, 3)
   print(f"FAISS DB Created: {faiss_path} (Time: {time_taken} sec)")
   return time_taken
# ✅ Function to Create and Save ChromaDB Vector Store (Ensures Correct Retrieval Later)
def create_chroma_store(text_chunks, chunker_name, emb_name, emb_model):
   print(f"Creating ChromaDB for {chunker_name} + {emb_name}")
   start_time = time.time()
   # ✅ Load SentenceTransformer
   model = SentenceTransformer(emb_model, device="cpu")
   # ✅ Generate embeddings with progress bar
   print(f"Generating embeddings for {len(text_chunks)} text chunks...")
   embeddings = model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True).tolist()
   # ✅ Create a collection in ChromaDB
   collection_name = f"{chunker_name}_{emb_name}"
   collection = chroma_client.get_or_create_collection(name=collection_name)
   # ✅ Store text in ChromaDB to ensure retrieval later
   print(f"Adding {len(text_chunks)} entries to ChromaDB...")
   for i in tqdm(range(len(text_chunks)), desc="ChromaDB Indexing", unit="doc"):
       collection.add(
           ids=[f"{chunker_name}_{emb_name}_{i}"],
           embeddings=[embeddings[i]],
           documents=[text_chunks[i]],  # ✅ Store text for retrieval
           metadatas=[{"source": f"{chunker_name}_{emb_name}_{i}"}]
       )
   end_time = time.time()
   time_taken = round(end_time - start_time, 3)
   print(f"ChromaDB Created: {collection_name} (Time: {time_taken} sec)")
   return time_taken
# Load and chunk text
raw_text = load_text_files()
# Track time results
db_creation_times = []
# Generate and save vector stores for FAISS & ChromaDB
for chunker_name, chunker in chunking_strategies.items():
   for emb_name, emb_model in embedding_models.items():
       print(f"\nProcessing: {chunker_name} + {emb_name}")
       # Chunk the text
       text_chunks = chunker.split_text(raw_text)
       # Create FAISS DB
       faiss_time = create_faiss_store(text_chunks, chunker_name, emb_name, emb_model)
       # Create ChromaDB
       chroma_time = create_chroma_store(text_chunks, chunker_name, emb_name, emb_model)
       db_creation_times.append({
           "Chunking Strategy": chunker_name,
           "Embedding Model": emb_name,
           "FAISS Creation Time (s)": faiss_time,
           "ChromaDB Creation Time (s)": chroma_time,
       })
# Save DB Creation Time Results
df = pd.DataFrame(db_creation_times)
df.to_csv("db_creation_times.csv", index=False)
print("\nDatabase Creation Times Saved: 'db_creation_times.csv'")
