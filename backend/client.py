import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage
from agent.react import react_agent
from langchain.tools import tool
from agent.prompt import ADMIN
from agent.module import RequestMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dotenv

dotenv.load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(
   api_key=os.environ.get("LLM_API"),
   base_url='https://api.opentyphoon.ai/v1',
   model='typhoon-v2.1-12b-instruct'
)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY")
)
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name="rag_docs",
    embeddings=embedding_model,
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

@tool
def rag_search(query: str) -> str:
    """Searches documents from the knowledge base relevant to the query."""

    print(f"LLM is try using rag_search tool, query = {query}")
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    content = "\n\n".join([f"- {doc.page_content}" for doc in docs])
    return f"ข้อมูลที่ค้นพบ:\n{content}"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectorstore.add_documents(chunks)
    return {"msg": "Uploaded and indexed."}


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
            messages.append({"role": "system", "content": chat.content})
        
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