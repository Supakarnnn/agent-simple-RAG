import os
import tempfile
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage,AIMessage
from agent.react import react_agent
from langchain.tools import tool
from agent.prompt import ADMIN
from agent.module import RequestMessage
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agent.model import llm,embedding_model

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import dotenv
dotenv.load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class WebURL(BaseModel):
    url: str

qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY")
)

@tool
def rag_search(query: str) -> str:
    """ ใช้สำหรับค้นหาข้อมูลที่เกี่ยวข้องกับบริการขององค์กร โดยอิงจากเอกสารที่มีอยู่ในระบบผ่าน RAG (Retrieval-Augmented Generation)"""

    print(f"LLM is try using rag_search tool, query = {query}")
    vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="service_docs",
    embedding=embedding_model,
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    content = "\n\n".join([f"- {doc.page_content}" for doc in docs])
    return f"ข้อมูลที่ค้นพบ:\n{content}"

@tool
def PBX_search(query: str) -> str:
    """ใช้สำหรับตอบคำถามเกี่ยวกับบริการ PromptCall Cloud PBX ของบริษัท Protocall เท่านั้น โดยจะดึงข้อมูลจากเอกสารภายในผ่านระบบ RAG"""

    print(f"LLM is try using PBX_search tool, query = {query}")
    vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="pbx_docs",
    embedding=embedding_model,
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    content = "\n\n".join([f"- {doc.page_content}" for doc in docs])
    return f"ข้อมูลที่ค้นพบ:\n{content}"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), collection_name: str = "service_docs"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_model,
)
    vectorstore.add_documents(chunks)
    return {"msg": "Uploaded"}

@app.post("/upload-web")
async def upload_web_page(web: WebURL, collection_name: str = "service_docs"):
    vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_model,
)
    response = requests.get(web.url, timeout=10, verify=False)

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    document = Document(page_content=text, metadata={"source": web.url})

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=300)
    chunks = splitter.split_documents([document])

    vectorstore.add_documents(chunks)

    return {"msg":"Uploaded"}

@app.post("/chat")
async def chat(chatmessage: RequestMessage):
    messages = []
    tools = [rag_search,PBX_search]
    
    for chat in chatmessage.messages:
        if chat.role == 'ai':
            messages.append(AIMessage(content=chat.content))
        elif chat.role == 'human':
            messages.append(HumanMessage(content=chat.content))
        elif chat.role == 'system':
            messages.append({"role": "system", "content": chat.content})
        
    agent = react_agent(llm, tools, ADMIN)
    result = await agent.ainvoke({"messages": messages})   
    final_result = result["messages"][-1].content

    return {
        "response": final_result,
        "full_messages": result["messages"]
    }

@app.post("/peekDocument")
async def peek_document(collection_name: str = "test_docs"):
    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding_model,
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    result = retriever.get_relevant_documents("peek")

    return {"Top document": result[0].page_content if result else "No documents found."}

@app.get("/peek_vector")
def peek_vector(collection_name: str = "test_docs"):
    scroll_result = qdrant_client.scroll(
        collection_name=collection_name,
        limit=1,
        with_vectors=True,
        with_payload=True
    )
    return {
        "vector": scroll_result[0][0].vector,
        "payload": scroll_result[0][0].payload
    }

@app.get("/")
async def health_check():

    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=8001)