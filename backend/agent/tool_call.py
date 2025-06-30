import os
from langchain.tools import tool
from agent.model import embedding_model
from langchain_milvus import Milvus
import dotenv

dotenv.load_dotenv()

@tool
async def rag_search(query: str) -> str:
    """ใช้สำหรับค้นหาข้อมูลทั่วไปที่เกี่ยวข้องกับ บริการ, สินค้า, โซลูชั่น, ข้อมูลการติดต่อของบริษัท โดยอิงจากเอกสารที่มีอยู่ในระบบผ่าน RAG (Retrieval-Augmented Generation)"""

    print(f"LLM is try using rag_search tool, query = {query}")
    
    vectorstore = Milvus(
        embedding_function=embedding_model,
        collection_name="Normal_docs",
        connection_args={"host": "localhost", "port": "19530"},
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = await retriever.ainvoke(query)

    if not docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"

    content = "\n\n".join([f"- {doc.page_content}" for doc in docs])
    return f"{content}"