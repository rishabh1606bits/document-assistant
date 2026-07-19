import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import CrossEncoder

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        result = client.models.embed_content(model="gemini-embedding-001", contents=texts)
        return [e.values for e in result.embeddings]

    def embed_query(self, text):
        result = client.models.embed_content(model="gemini-embedding-001", contents=text)
        return result.embeddings[0].values

st.set_page_config(page_title="Document Assistant", page_icon="📄")
st.title("📄 Document Assistant")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()

# Keep the vectorstore alive across reruns (Streamlit reruns the whole script on every interaction)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- File upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Process document"):
        with st.spinner("Loading and chunking document..."):
            # Save uploaded file temporarily
            temp_path = f"data/{uploaded_file.name}"
            os.makedirs("data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load, chunk, embed
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(pages)

            embeddings = GeminiEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            st.session_state.vectorstore = vectorstore
        st.success(f"Processed {len(chunks)} chunks from {uploaded_file.name}")

# --- Chat interface ---
if st.session_state.vectorstore is not None:
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Searching and generating answer..."):
           # Step 1: Retrieve a broad shortlist
            candidates = st.session_state.vectorstore.similarity_search(question, k=10)

# Step 2: Re-rank using the cross-encoder
            pairs = [[question, c.page_content] for c in candidates]
            scores = reranker.predict(pairs)

# Step 3: Sort by re-rank score, keep top 3
            reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            results = [doc for doc, score in reranked[:3]]

            context = "\n\n".join([r.page_content for r in results])

            prompt = f"""Answer the question based only on the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )

        st.markdown("### Answer")
        st.write(response.text)

        with st.expander("Sources used"):
            for i, r in enumerate(results):
                st.markdown(f"**Chunk {i+1}**")
                st.text(r.page_content)
else:
    st.info("Upload a PDF and click 'Process document' to get started.")