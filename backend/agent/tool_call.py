import os
from langchain.tools import tool
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from model import embedding_model
import dotenv

dotenv.load_dotenv()

qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY")
)

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name="service_docs",
    embeddings=embedding_model,
)

@tool
def rag_search(query: str) -> str:
    """Retrieve documents using RAG (Retrieval-Augmented Generation)"""

    print(f"LLM is try using rag_search tool, query = {query}")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    content = "\n\n".join([f"- {doc.page_content}" for doc in docs])
    return f"ข้อมูลที่ค้นพบ:\n{content}"

@tool
def PBX_search(query: str) -> str:
    """Retrieve documents about promptCall Cloud PBX service using RAG (Retrieval-Augmented Generation)"""

    print(f"LLM is try using PBX_search tool, query = {query}")
    vectorstore = Qdrant(
    client=qdrant_client,
    collection_name="pbx_docs",
    embeddings=embedding_model,
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    content = "\n\n".join([f"- {doc.page_content}" for doc in docs])
    return f"ข้อมูลที่ค้นพบ:\n{content}"