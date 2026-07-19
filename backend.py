import os
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from database import SessionLocal, ChatHistory

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        result = client.models.embed_content(model="gemini-embedding-001", contents=texts)
        return [e.values for e in result.embeddings]

    def embed_query(self, text):
        result = client.models.embed_content(model="gemini-embedding-001", contents=text)
        return result.embeddings[0].values

app = FastAPI()
embeddings = GeminiEmbeddings()

@app.get("/")
def health_check():
    return {"status": "Backend is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save uploaded file
    os.makedirs("data", exist_ok=True)
    temp_path = f"data/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Load, chunk, embed
    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "status": "processed"
    }

from pydantic import BaseModel
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # Retrieve broad shortlist
    candidates = vectorstore.similarity_search(request.question, k=10)

    # Re-rank
    pairs = [[request.question, c.page_content] for c in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    results = [doc for doc, score in reranked[:3]]

    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""Answer the question based only on the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {request.question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

   # Save to chat history
    db = SessionLocal()
    record = ChatHistory(
        question=request.question,
        answer=response.text,
        document_name="rishabh-cv.pdf"  # hardcoded for now, we'll make this dynamic later
)
    db.add(record)
    db.commit()
    db.close()

    return {
    "answer": response.text,
    "sources": [r.page_content for r in results]
}

@app.get("/history")
def get_history():
    db = SessionLocal()
    records = db.query(ChatHistory).order_by(ChatHistory.created_at.desc()).all()
    db.close()

    return [
        {
            "id": r.id,
            "question": r.question,
            "answer": r.answer,
            "document_name": r.document_name,
            "created_at": r.created_at.isoformat()
        }
        for r in records
    ]   