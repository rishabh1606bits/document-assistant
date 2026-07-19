import os
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Wrap Gemini's embedding API so LangChain/Chroma can call it
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text):
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return result.embeddings[0].values

# Load and chunk
loader = PyPDFLoader("data/rishabh-cv.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

# Store in Chroma (local, on disk)
embeddings = GeminiEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Stored {len(chunks)} chunks in Chroma.")

# Test a search
query = "What programming languages does this person know?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nTop 2 results for: '{query}'")
print("---")
for i, r in enumerate(results):
    print(f"Result {i+1}:")
    print(r.page_content)
    print("---")