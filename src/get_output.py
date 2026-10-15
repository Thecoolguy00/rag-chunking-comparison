
import os, csv, json
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = OllamaEmbeddings(
    model="qllama/bge-large-en-v1.5:q8_0"
)

llm = ChatOpenRouter(
    model="deepseek/deepseek-v4-flash-0731",
    temperature=0.2
)

collections = {
    "fixed_size": "diabetes_simple",
    "markdown_headers": "diabetes_markdown",
    "markdown_recursive": "diabetes_markdown_recursive",
}

questions = [
    "What are the two main diagnostic criteria for Type 2 diabetes based on HbA1c and fasting plasma glucose?",
    "What HbA1c range indicates prediabetes?",
    "Name three first-line oral medications used to treat Type 2 diabetes.",
    "What are the classic symptoms of hyperglycemia?",
    "What lifestyle modifications are recommended for managing Type 2 diabetes?",
    "Name two microvascular complications of diabetes.",
    "Name two macrovascular complications of diabetes.",
    "How often should HbA1c be monitored in someone with stable Type 2 diabetes?",
    "What blood glucose target range is recommended for most adults with Type 2 diabetes?",
    "What symptoms should prompt someone to seek emergency medical care?"
]

with open("results/evaluation.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "question",
            "strategy",
            "retrieved_chunks",
            "retrieval_judgment",
            "generated_answer",
            "answer_correctness",
        ],
    )
    writer.writeheader()

    for strategy, collection_name in collections.items():
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="chroma_db",
        )

        retriever = db.as_retriever(search_kwargs={"k": 3})

        for question in questions:
            docs = retriever.invoke(question)

            print(docs)

            context = "\n\n".join(
                f"[Chunk {i}]\n{doc.page_content}"
                for i, doc in enumerate(docs, start=1)
            )

            prompt = f"""
Answer the question using only the supplied context.
Do not use outside knowledge.
If the context is insufficient, say so.

Context:
{context}

Question: {question}

Answer:
"""

            answer = llm.invoke(prompt).content

            writer.writerow({
                "question": question,
                "strategy": strategy,
                "retrieved_chunks": json.dumps(
                    [doc.page_content for doc in docs],
                    ensure_ascii=False,
                ),
                "retrieval_judgment": "",
                "generated_answer": answer,
                "answer_correctness": "",
            })

print("Saved results to results/evaluation.csv")
