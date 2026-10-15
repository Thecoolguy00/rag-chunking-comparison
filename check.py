
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qllama/bge-large-en-v1.5:q8_0"
)

collections = [
    "diabetes_simple",
    "diabetes_markdown",
    "diabetes_markdown_recursive",
]

for name in collections:
    db = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory="chroma_db",
    )

    chunks = db.get()["documents"]
    sizes = [len(chunk) for chunk in chunks]

    print(f"\n{name}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Average chunk size: {sum(sizes) / len(sizes):.2f} characters")
    print(f"Minimum chunk size: {min(sizes)} characters")
    print(f"Maximum chunk size: {max(sizes)} characters")
