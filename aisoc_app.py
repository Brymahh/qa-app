import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import tempfile
import os


def initialize_groq(api_key: str):
    return Groq(api_key=api_key)

def get_groq_response(client, context, question, model_name='llam-3.1-8b-instant'):
    prompt = f'''
        Based on the follwoing context, please answer the question in a concise manner.
        Context: {context}
        Question: {question}
        Answer: Provide a clear, ACCURATE answer based only on the information in the context provided.
        '''
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error getting response {str(e)}"
    
    
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


class LocalVectorStore:
    def __init__(self, embeddiing_model):
        self.embeddiing_model = embeddiing_model
        self.chunks = []
        self.embeddings = None
        self.index = None
        
    def add_documents(self, documents): 
        self.chunks = [doc.page_content for doc in documents]
        
        embeddings = self.embeddiing_model.encode(self.chunks)
        self.embeddings = np.array(embeddings).astype('float32')
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
    def similarity_search(self, query, k=4):
        if self.index is None:
            return [] 
        
        query_embedding = self.embeddiing_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        
        
            