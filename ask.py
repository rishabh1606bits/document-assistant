import os
from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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

# Reconnect to the existing Chroma DB (no need to re-embed everything)
embeddings = GeminiEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

def ask(question, k=3):
    # Step 1: Retrieve relevant chunks
    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([r.page_content for r in results])

    # Step 2: Build a prompt with that context
    prompt = f"""Answer the question based only on the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    # Step 3: Ask Gemini
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )
    return response.text, results

if __name__ == "__main__":
    question = "What programming languages does this person know?"
    answer, sources = ask(question)

    print(f"Question: {question}")
    print(f"\nAnswer:\n{answer}")
    print(f"\n--- Sources used ({len(sources)} chunks) ---")
    for i, s in enumerate(sources):
        print(f"[{i+1}] {s.page_content[:100]}...")