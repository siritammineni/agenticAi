import os
import time
import pandas as pd
import numpy as np
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
# ✅ Azure GPT-4o-mini Configuration
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o-mini"
TOKEN = "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"

client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))
# ✅ Define Chunking Strategies & Embedding Models (Same as FAISS Evaluation)
CHUNKERS = {
   "RecursiveTextSplitter": "RecursiveTextSplitter",
   "SentenceSplitter": "SentenceSplitter",
   "ParagraphSplitter": "ParagraphSplitter",
}
EMBEDDING_MODELS = {
   "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
   "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
   "bge-base-en": "BAAI/bge-base-en"
}
# ✅ Queries & Ground Truth (Same as FAISS Evaluation)
queries = [
   "What security features does Xfinity offer?",
   "What is Xfinity NOW Internet?",
   "What is the Xfinity NOW WiFi Pass?",
   "What are the additional terms for Xfinity Internet and Voice services?",
   "How can I check my data usage for finity Internet?",
   "What happens if I exceed the data consumption threshold?"
]
ground_truth = {
   "What security features does Xfinity offer?": "Xfinity includes Advanced Security to protect against phishers, hackers, and other threats, while 5G home internet may offer limited security features.",
   "What is Xfinity NOW Internet?": "Xfinity NOW Internet is a prepaid home internet service with unlimited data, WiFi equipment, and all taxes and fees included.",
   "What is the Xfinity NOW WiFi Pass?": "The Xfinity NOW WiFi Pass provides access to millions of hotspots with no commitments, available for just $10 for each 30-day pass.",
   "What are the additional terms for Xfinity Internet and Voice services?": "The additional terms apply to Xfinity Internet and Voice services and are incorporated into the Xfinity Residential Services Agreement. By using the services, you accept these terms.",
   "How can I check my data usage for finity Internet?": "You can check your current data usage by logging into your Xfinity account at xfinity.com or using the Xfinity My Account mobile app.",
   "What happens if I exceed the data consumption threshold?": "If you exceed the data consumption threshold, Xfinity may notify you, and they reserve the right to adjust your data plan or usage thresholds as needed."
}
# ✅ ChromaDB Storage Path
CHROMA_DB_DIR = "vector_dbs_chroma"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)


# ✅ Function to Load ChromaDB Collection
def load_chroma_store(chunker_name, emb_name):
   collection_name = f"{chunker_name}_{emb_name}"
   collection = chroma_client.get_collection(name=collection_name)
   if collection is None:
       raise FileNotFoundError(f"❌ ChromaDB collection not found for {chunker_name} + {emb_name}")
   print(f"✅ Loaded ChromaDB Collection: {collection_name}")
   # ✅ Assign correct embedding model
   embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS[emb_name])
   return collection, embedding_model
# ✅ Function to Retrieve Context from ChromaDB


def retrieve_chroma_context(query, collection, embedding_model, top_k=3):
   start_time = time.time()
   # ✅ Convert query to embedding
   query_embedding = embedding_model.embed_query(query)
   # ✅ Perform similarity search
   results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
   retrieval_time = time.time() - start_time
   # ✅ Extract retrieved documents
   retrieved_results = results["documents"][0] if results["documents"] else []
   return "\n\n".join(retrieved_results), retrieved_results, retrieval_time


# ✅ Function to Query GPT-4o (Same Logic as FAISS Evaluation)
def query_gpt4o(user_query, context):
   prompt = f"Use the following retrieved context to answer the query:\nContext:\n{context}\nQuery: {user_query}"
   start_time = time.time()
   response = client.complete(
       messages=[SystemMessage(content="You are a helpful assistant."), UserMessage(content=prompt)],
       temperature=1.0, top_p=1.0, max_tokens=500, model=MODEL_NAME
   )
   response_time = time.time() - start_time
   return response.choices[0].message.content, response_time


# ✅ Function to Compute Semantic Similarity Score
def calculate_similarity_score(response, ground_truth):
   if not ground_truth:
       return None  # Skip if no ground-truth provided
   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   response_embedding = model.encode(response, convert_to_tensor=True)
   truth_embedding = model.encode(ground_truth, convert_to_tensor=True)
   similarity_score = util.pytorch_cos_sim(response_embedding, truth_embedding).item()
   return round(similarity_score, 3)


# ✅ Evaluation Function for ChromaDB (Uses Same LLM & Queries as FAISS)
def evaluate_chroma():
   results = []
   for chunker_name in CHUNKERS.keys():
       for emb_name in EMBEDDING_MODELS.keys():
           print(f"\n🔍 Evaluating: {chunker_name} + {emb_name}")
           # ✅ Load ChromaDB Collection
           try:
               collection, embedding_model = load_chroma_store(chunker_name, emb_name)
           except FileNotFoundError:
               continue  # Skip if the collection does not exist
           for query in tqdm(queries, desc=f"🔹 Running Queries for {chunker_name} + {emb_name}"):
               # ✅ Retrieve Context
               chroma_context, retrieved_docs, chroma_retrieval_time = retrieve_chroma_context(query, collection, embedding_model)
               # ✅ Query LLM
               chroma_response, chroma_response_time = query_gpt4o(query, chroma_context)
               # ✅ Compute Accuracy (Semantic Similarity)
               chroma_accuracy = calculate_similarity_score(chroma_response, ground_truth.get(query))
               # ✅ Store Results
               results.append({
                   "Chunking Strategy": chunker_name,
                   "Embedding Model": emb_name,
                   "Query": query,
                   "ChromaDB Retrieval Time (s)": round(chroma_retrieval_time, 3),
                   "ChromaDB Response Time (s)": round(chroma_response_time, 3),
                   "ChromaDB Accuracy": chroma_accuracy,
                   "Retrieved Context": retrieved_docs,
                   "LLM Response": chroma_response
               })
   # ✅ Save Evaluation Results to CSV
   df = pd.DataFrame(results)
   df.to_csv("chroma_results.csv", index=False)
   print("\n✅ ChromaDB Evaluation Completed! Results saved in 'chroma_results.csv'.")
# ✅ Run ChromaDB Evaluation
if __name__ == "__main__":
   evaluate_chroma()