from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Configuration
document = Path("data/diabetes_reference_document.md").read_text(encoding="utf-8")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

embeddings = OllamaEmbeddings(
    model="qllama/bge-large-en-v1.5:q8_0",
    base_url="http://127.0.0.1:11434/"
)

# Strategy 1: Simple character splitting
simple_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

simple_chunks = simple_splitter.create_documents([document])


# Strategy 2: Markdown header splitting
headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers,
    strip_headers=False,
)

markdown_chunks = markdown_splitter.split_text(document)


# Strategy 3: Markdown + recursive splitting
sections = markdown_chunks

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

advanced_chunks = recursive_splitter.split_documents(sections)


# Create separate ChromaDB collections
simple_db = Chroma.from_documents(
    documents=simple_chunks,
    embedding=embeddings,
    collection_name="diabetes_simple",
    persist_directory="chroma_db",
)

markdown_db = Chroma.from_documents(
    documents=markdown_chunks,
    embedding=embeddings,
    collection_name="diabetes_markdown",
    persist_directory="chroma_db",
)

advanced_db = Chroma.from_documents(
    documents=advanced_chunks,
    embedding=embeddings,
    collection_name="diabetes_markdown_recursive",
    persist_directory="chroma_db",
)


# Display chunk counts
print("Simple chunks:", len(simple_chunks))
print("Markdown chunks:", len(markdown_chunks))
print("Markdown + recursive chunks:", len(advanced_chunks))