from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/rishabh-cv.pdf")
pages = loader.load()

print(f"Number of pages loaded: {len(pages)}")
print("---")
print("First page content preview:")
print(pages[0].page_content[:500])
print("---")
print("Metadata:")
print(pages[0].metadata)