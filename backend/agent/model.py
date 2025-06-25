import os
import torch
import dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


dotenv.load_dotenv()

llm = ChatOpenAI(
   api_key=os.environ.get("LLM_API"),
   base_url='https://api.opentyphoon.ai/v1',
   model='typhoon-v2.1-12b-instruct',
   temperature=0
)

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)