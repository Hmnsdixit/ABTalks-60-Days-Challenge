from fastapi import FastAPI, UploadFile, File
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

app = FastAPI()

os.environ["OPENAI_API_KEY"] = "your_api_key_here"

vectorstore = None

@app.get("/")
def home():
    return {"message": "RAG API Running 🚀"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore

    content = await file.read()
    text = content.decode("utf-8")

    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(docs, embeddings)

    return {"message": "File uploaded successfully"}

@app.post("/ask")
async def ask(query: str):
    global vectorstore

    if vectorstore is None:
        return {"error": "Upload file first"}

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=vectorstore.as_retriever()
    )

    result = qa.run(query)

    return {"answer": result}