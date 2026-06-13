# End-to-End RAG Pipeline Flow

This RAG (Retrieval-Augmented Generation) application answers questions using information retrieved from a webpage rather than relying solely on the LLM's pre-trained knowledge.

---

# 1. Import Required Libraries

```python
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
```

## Purpose

Imports all components needed for:

- Loading webpages
- Parsing HTML
- Splitting documents
- Creating embeddings
- Storing vectors
- Retrieving relevant chunks
- Prompting Gemini
- Parsing the output

---

# 2. Configure API Keys

```python
import os

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
```

## Purpose

Authenticates requests sent to Gemini.

Without this step:

- Gemini chat models cannot be used.
- Gemini embedding models cannot be used.

---

# 3. Create Web Loader

```python
loader = WebBaseLoader(
    web_paths=(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
```

## Purpose

Creates a loader that knows:

- Which webpage to download
- Which parts of the webpage to keep

### HTML Filtering

Only extracts:

```html
<div class="post-content">
<div class="post-title">
<div class="post-header">
```

Everything else is ignored.

Examples:

- Navigation menus
- Sidebars
- Ads
- Footer content

---

# 4. Load Webpage Content

```python
docs = loader.load()
```

## Purpose

Downloads the webpage and converts it into LangChain Documents.

Output:

```python
[
    Document(
        page_content="LLM Powered Autonomous Agents...",
        metadata={...}
    )
]
```

---

# 5. Split Documents into Chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)
```

## Purpose

Breaks large documents into smaller pieces.

Example:

```text
Original Document
        ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
```

### Why?

LLMs and embedding models perform better with smaller chunks.

### Overlap

```python
chunk_overlap=200
```

Preserves context between neighboring chunks.

---

# 6. Create Embedding Model

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
```

## Purpose

Converts text into vectors.

Example:

```text
"Task decomposition"
```

becomes:

```python
[0.23, -0.11, 0.87, ...]
```

Vectors capture semantic meaning.

---

# 7. Create Vector Store

```python
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)
```

## Purpose

Stores:

- Text chunks
- Their embeddings

Example:

```text
Chunk A
    ↓
Vector A

Chunk B
    ↓
Vector B
```

Everything is stored inside Chroma.

---

# 8. Create Retriever

```python
retriever = vectorstore.as_retriever()
```

## Purpose

Creates a search interface over the vector database.

Input:

```text
User Question
```

Output:

```text
Most Relevant Chunks
```

Example:

```text
"What is Task Decomposition?"
```

returns:

```text
Chunk about task decomposition
Chunk about planning
Chunk about reasoning
```

---

# 9. Create Prompt Template

```python
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.

If the answer is not clearly present in the context,
say "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
""")
```

## Purpose

Defines what Gemini receives.

### Placeholders

```python
{context}
```

will contain retrieved documents.

```python
{question}
```

will contain the user's question.

---

# 10. Create Gemini LLM

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
```

## Purpose

Creates the model responsible for generating answers.

### Temperature

```python
temperature=0
```

Produces:

- More deterministic responses
- Less creativity
- Better factual consistency

Ideal for RAG.

---

# 11. Format Retrieved Documents

```python
def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )
```

## Purpose

Converts retrieved Document objects into plain text.

Input:

```python
[
    Document("Chunk A"),
    Document("Chunk B")
]
```

Output:

```text
Chunk A

Chunk B
```

This text becomes the context sent to Gemini.

---

# 12. Create RAG Chain

```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

## Purpose

Connects all components into one pipeline.

---

# 13. Execute RAG Query

```python
rag_chain.invoke(
    "What is Task Decomposition?"
)
```

---

# Runtime Execution Flow

## Step 1

User asks:

```text
What is Task Decomposition?
```

---

## Step 2

Retriever receives question.

```text
Question
      ↓
Retriever
```

---

## Step 3

Retriever searches Chroma.

```text
Question
      ↓
Embedding
      ↓
Similarity Search
      ↓
Relevant Chunks
```

---

## Step 4

Retrieved chunks are formatted.

```text
Document Objects
      ↓
format_docs()
      ↓
Single Context String
```

---

## Step 5

RunnablePassthrough forwards the original question.

```text
Question
      ↓
Unchanged
```

---

## Step 6

Prompt is populated.

```text
Context:
<retrieved chunks>

Question:
What is Task Decomposition?
```

---

## Step 7

Prompt is sent to Gemini.

```text
Prompt
      ↓
Gemini
```

---

## Step 8

Gemini generates response.

```text
Task decomposition is the process of...
```

---

## Step 9

Output parser extracts text.

```text
AIMessage
      ↓
String
```

---

# Complete Flow Diagram

```text
Web Page
    ↓
WebBaseLoader
    ↓
Documents
    ↓
Text Splitter
    ↓
Chunks
    ↓
Gemini Embeddings
    ↓
Vectors
    ↓
Chroma Vector Store
    ↓
Retriever
    ↓
Retrieved Chunks
    ↓
format_docs()
    ↓
Prompt Template
    ↓
Gemini LLM
    ↓
Answer
```

# Final Summary

The pipeline performs:

1. Load webpage
2. Extract article content
3. Split content into chunks
4. Convert chunks to embeddings
5. Store embeddings in Chroma
6. Retrieve relevant chunks for a question
7. Build a prompt using retrieved context
8. Send prompt to Gemini
9. Generate answer
10. Return answer to the user
