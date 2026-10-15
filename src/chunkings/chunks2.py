from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

document = Path("data/diabetes_reference_document.md").read_text(encoding="utf-8")

# Stage 1: Split by Markdown structure
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    # ("###", "Header 3"),
]


markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=True,
)

section_docs = markdown_splitter.split_text(document)

chunk_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    length_function=len,
)

advanced_chunks = chunk_splitter.split_documents(section_docs)

for chunk in advanced_chunks:
    heading = " > ".join(
        str(value) for value in chunk.metadata.values()
    )
    chunk.page_content = f"{heading}\n\n{chunk.page_content}"


print("Sections:", len(section_docs))
print("Advanced chunks:", len(advanced_chunks))

for chunk in advanced_chunks:
    print("\nMetadata:", chunk.metadata)
    print(chunk.page_content)