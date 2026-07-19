from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/rishabh-cv.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(pages)

print(f"Number of chunks: {len(chunks)}")
print("---")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (length: {len(chunk.page_content)} chars):")
    print(chunk.page_content)
    print(f"Metadata: {chunk.metadata}")
    print("---")