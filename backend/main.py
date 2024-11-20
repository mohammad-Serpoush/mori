from pinecone import Index
from pydantic import BaseModel
from fastapi import Depends, FastAPI
import torch
from typing import Optional
from pinecone import Pinecone

from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv()


def get_pinecone():
    return Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY"),
    )


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def fetch_relevant_vector(index: Index, text_description: str, top_k: int = 10, filters={}):
    inputs = processor(
        text=[text_description],
        return_tensors="pt", padding=True
    )
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    text_embedding = text_embedding.squeeze().tolist()

    response = index.query(
        namespace=None,
        vector=text_embedding,
        top_k=top_k,
        include_values=True,
        include_metadata=True,
        filter=filters
    )

    return response


app = FastAPI()


class Product(BaseModel):
    id: int
    title: str
    images: list[str]

@app.get("/products", response_model=list[Product])
def get_products(
    *,
    pinecone_client: Pinecone = Depends(get_pinecone),
    query: Optional[str] = None,
    category_name: Optional[str] = None,

):
    filters = {}
    if category_name:
        filters["category_name"] = category_name

    index = pinecone_client.Index("products")

    response = fetch_relevant_vector(index, query, filters)
