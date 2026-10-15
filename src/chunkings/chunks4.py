from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter

document = Path("data/diabetes_reference_document.md").read_text(
    encoding="utf-8"
)

text_splitter = CharacterTextSplitter(
    separator="",       # Split at character boundaries
    chunk_size=600,     # Maximum target size in characters
    chunk_overlap=100,
    length_function=len,
)

naive_chunks = text_splitter.create_documents([document])

print("Naive chunks:", len(naive_chunks))

for i, chunk in enumerate(naive_chunks):
    print(f"\nChunk {i + 1} | Length: {len(chunk.page_content)}")
    print(chunk.page_content)