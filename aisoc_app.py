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