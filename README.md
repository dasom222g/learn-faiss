# Learning FAISS - Vector Search

## 📖 Overview

This repository contains practical examples and implementations for learning FAISS (Facebook AI Similarity Search), a powerful library for efficient similarity search and clustering of dense vectors. Perfect for beginners who want to master vector search techniques in AI applications.

## 🎯 What You'll Learn

- **FAISS Fundamentals**: Basic concepts and setup
- **Vector Store Creation**: Building and managing vector databases
- **Search Techniques**: Similarity search with different algorithms
- **LangChain Integration**: Using FAISS with LangChain for RAG systems
- **Performance Optimization**: Best practices for production use
- **Real-world Applications**: PDF processing, document search, and more

## 📂 Repository Structure (expected)

```
learn-faiss/
├── basics/                 # FAISS fundamentals
├── examples/              # Practical implementation examples
├── langchain-integration/ # LangChain + FAISS tutorials
├── performance/           # Optimization techniques
├── projects/             # Complete project examples
└── docs/                 # Additional documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
pipenv install langchain langchain-openai langchainhub langchain-community
pipenv install faiss-cpu
pipenv install pypdf # Read PDF File (essential)
pipenv install python-dotenv
pipenv install black

```

### Quick Start
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create your first vector store
embeddings = OpenAIEmbeddings()
texts = ["Hello world", "FAISS is awesome"]
vectorstore = FAISS.from_texts(texts, embeddings)

# Search
results = vectorstore.similarity_search("greeting", k=1)
```


## 🛠️ Technologies Used

- **FAISS**: Vector similarity search
- **LangChain**: AI application framework
- **OpenAI**: Embeddings and language models
- **Python**: Primary programming language
