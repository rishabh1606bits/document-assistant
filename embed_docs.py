import os
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Load and chunk (same as before)
loader = PyPDFLoader("data/rishabh-cv.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

# Embed just the first chunk as a test
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunks[0].page_content
)

embedding = result.embeddings[0].values
print(f"Embedding length: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")