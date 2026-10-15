# RAG Chunking Strategy Comparison

## 1. Approach

The goal of this project was to compare how different chunking strategies affect retrieval quality in a RAG system using a diabetes reference document.

I implemented three strategies using LangChain:

1. **Fixed-size chunking:** Splits the document into chunks using a target size of 600 characters and an overlap of 100 characters.
2. **Markdown header chunking:** Splits the document using Markdown headings to preserve complete sections.
3. **Markdown + recursive chunking:** Splits by Markdown headings first, then divides larger sections into smaller chunks using a 600-character target and 100-character overlap.

I generated embeddings using `qllama/bge-large-en-v1.5:q8_0` through Ollama and stored the chunks in separate Chroma collections. For evaluation, I retrieved the top 3 chunks for each of the 10 questions and used DeepSeek through OpenRouter to generate answers.  
Why this model? I already had it installed using ollama  

## 2. Results

| Strategy | Number of chunks | Average chunk size | Retrieval accuracy |
|---|---:|---:|---:|
| Fixed-size | 15 | 595.33 characters | 80% |
| Markdown headers | 10 | 754.90 characters | 80% |
| Markdown + recursive | 18 | 442.56 characters | 80% |

### Question-by-question observations

| Question | Observation |
|---|---|
| Diagnostic criteria | All strategies generated the relevant thresholds. |
| Prediabetes range | All strategies returned the correct range. |
| Oral medications | Recursive chunking gave a more detailed explanation of the retrieved medication classes. |
| Hyperglycemia symptoms | All strategies retrieved relevant information. |
| Lifestyle modifications | All strategies retrieved relevant information. |
| Microvascular complications | All strategies identified valid examples. |
| Macrovascular complications | All strategies identified valid examples. |
| HbA1c monitoring | All strategies returned the monitoring frequency. |
| Blood glucose target | The retrieved context did not provide the requested target range. |
| Emergency symptoms | All strategies retrieved relevant information. |

Retrieval accuracy should be calculated by manually checking whether the retrieved chunks contain the information needed to answer each question. A question that cannot be answered from the source document should be distinguished from a retrieval failure.

## 3. Analysis

Fixed-size chunking is simple to implement, but it can split related information across chunk boundaries. Markdown header chunking preserves sections and keeps related information together. It also produced the fewest chunks in this experiment.

Markdown + recursive chunking divides larger sections into smaller pieces. This can make specific information easier to retrieve, but it may also separate related details that were originally together.

In the medication question, both Markdown-based strategies retrieved relevant medication information. However, their generated answers differed a little in detail. This shows that answer quality depends on both the retrieved context and the LLM's response, not just the chunking strategy.

The blood glucose target question also showed a limitation: the reference document did not contain the requested target range. Better chunking cannot retrieve information that is absent from the source.

Overall, the experiment compares the trade-offs between simplicity, document structure, and chunk size. More evaluation questions would be useful before concluding that one strategy performs best.

## 4. Technologies

- Python
- LangChain
- Ollama embeddings
- Chroma vector store
- OpenRouter and DeepSeek
