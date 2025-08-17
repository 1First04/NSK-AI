# RAG Application README

## Overview

This project implements a Retrieval-Augmented Generation (RAG) application. RAG combines the power of generative AI (OpenAI's ChatGPT) with information retrieval from a vector database (ChromaDB). The application is designed to process `.text` files, enabling users to ask questions and receive answers grounded in the provided documents.

## How It Works

1. **File Ingestion**: The application reads and processes `.text` files, extracting relevant content for training and retrieval.
2. **Vector Database (ChromaDB)**: The extracted text is converted into vector embeddings and stored in ChromaDB. This allows for efficient semantic search and retrieval.
3. **OpenAI ChatGPT Integration**: When a user asks a question, the application retrieves relevant document chunks from ChromaDB and passes them as context to ChatGPT.
4. **Answer Generation**: ChatGPT generates a response using both the retrieved context and its own knowledge, ensuring accurate and context-aware answers.

## Educational Benefits

- **Enhanced Learning**: Users can interact with their own documents, deepening understanding through AI-powered Q&A.
- **Contextual Responses**: Answers are grounded in the provided `.text` files, making them reliable and relevant.
- **Modern AI Techniques**: The project demonstrates how retrieval and generation can be combined for advanced educational applications.

## Getting Started

1. Prepare your `.text` files with the information you want to use.
2. Run the application to ingest and index your files.
3. Ask questions and receive answers based on your documents.

## Technologies Used

- **OpenAI ChatGPT**: For natural language understanding and generation.
- **ChromaDB**: For storing and searching vector embeddings.
- **Python**: Core programming language for the application.

---

Feel free to explore, modify, and extend this RAG application for your educational needs!
