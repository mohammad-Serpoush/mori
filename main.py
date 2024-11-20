from typing import Optional
from pinecone import Pinecone, ServerlessSpec

from transformers import CLIPProcessor, CLIPModel


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

API_KEY = "pcsk_V1N5k_2MEH2y1B7ZwGjYKbHXw5twJCGpiyPZ4eEg6GmESWrGVmApeBp6q4gxgNNLYHd9P"

client = Pinecone(
    api_key=API_KEY,
)


index = client.Index("products")
import torch

def fetch_relevant_vector(text_description, top_k=10, namespace=None , filters = {}):
      # Process the text input to get embeddings
      inputs = processor(text=[text_description], return_tensors="pt", padding=True)
      with torch.no_grad():
           text_embedding = model.get_text_features(**inputs)
      text_embedding = text_embedding.squeeze().tolist() 
      
      response = index.query(
           namespace=namespace,
           vector=text_embedding,
           top_k=top_k,
           include_values=True,
           include_metadata=True,
           filter=filters
           )
 
      return response
  
  
from fastapi import FastAPI

app = FastAPI()

from pydantic import BaseModel

class Product(BaseModel):
    id: int
    title : str
    images : list[str]

@app.get("/products" , response_model=list[Product])
def get_products(
    *,
    query: Optional[str] = None,
    category_name: Optional[str] = None,
):
    filters = {}
    if category_name:
        filters["category_name"] = category_name
    response = fetch_relevant_vector(query)