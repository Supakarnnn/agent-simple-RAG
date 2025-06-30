import os
import tempfile
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from agent.react import react_agent
from langchain.tools import tool
from agent.prompt import ADMIN
from agent.module import RequestMessage
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agent.model import llm,embedding_model
from langchain_milvus import Milvus
from agent.tool_call import rag_search

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

@app.post("/upload-web")
async def upload_web_page(web: WebURL, collection_name: str = "service_docs"):
    response = requests.get(web.url, timeout=10, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    document = Document(page_content=text, metadata={"source": web.url})
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=300)
    chunks = splitter.split_documents([document])

    vectorstore = Milvus(
        embedding_function=embedding_model,
        collection_name=collection_name,
        connection_args={
            "host": "localhost",
            "port": "19530",
        },
    )

    #vector ใหม่เข้า collection ที่มีอยู่
    vectorstore.add_documents(chunks)

    return {"msg": f"Uploaded {len(chunks)} chunks to Milvus"}

@app.post("/chat")
async def chat(chatmessage: RequestMessage):
    messages = []
    tools = [rag_search]
    
    for chat in chatmessage.messages:
        if chat.role == 'ai':
            messages.append(AIMessage(content=chat.content))
        elif chat.role == 'human':
            messages.append(HumanMessage(content=chat.content))
        elif chat.role == 'system':
            messages.append(SystemMessage(content=chat.content))
        
    agent = react_agent(llm, tools, ADMIN)
    result = await agent.ainvoke({"messages": messages})   
    final_result = result["messages"][-1].content

    return {
        "response": final_result,
        "full_messages": result["messages"]
    }

@app.get("/")
async def health_check():

    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=8001)