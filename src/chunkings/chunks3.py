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
    strip_headers=False,
)

section_docs = markdown_splitter.split_text(document)


for chunk in section_docs:
    print("\nMetadata:", chunk.metadata)
    print(chunk.page_content)