from fastapi import FastAPI, UploadFile, File
import shutil
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

app = FastAPI()

# 👉 Ensure data folder exists
if not os.path.exists("data"):
    os.makedirs("data")

# 👉 Global variable (latest uploaded file)
latest_file = None


# =========================
# 🔹 Upload Endpoint
# =========================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global latest_file

    file_location = f"data/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    latest_file = file_location  # save latest file

    return {"message": f"File '{file.filename}' uploaded successfully"}


# =========================
# 🔹 Ask Endpoint (RAG)
# =========================
@app.post("/ask")
async def ask(query: str):
    global latest_file

    if latest_file is None:
        return {"error": "Please upload a file first!"}

    # 1. Load document
    loader = TextLoader(latest_file)
    documents = loader.load()

    # 2. Create embeddings
    embeddings = OllamaEmbeddings(model="phi")

    # 3. Create vector store
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 4. Load LLM
    llm = Ollama(model="phi")

    # 5. Retrieve relevant docs
    docs = vectorstore.similarity_search(query, k=2)

    context = "\n".join([doc.page_content for doc in docs])

    # 6. Prompt
    prompt = f"""
    Answer based on the context below:

    {context}

    Question: {query}
    """

    # 7. Generate answer
    answer = llm.invoke(prompt)

    return {"response": answer}


# =========================
# 🔹 Home Route
# =========================
@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}